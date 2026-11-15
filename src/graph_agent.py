"""
LangGraph Lead Qualification Agent.
Implements the multi-lane lead qualification workflow as a stateful compiled StateGraph:
- State tracking via typed GraphState
- Checkpointed memory using MemorySaver
- Dedicated nodes for Guardrails, FAQ RAG, Lane A, Lane B, Lane C, Lane D, Lane E, and Terminal Endings
- Conditional edges enforcing deterministic business rules and gatekeeping
"""

import re
import logging
from typing import TypedDict, Optional, List, Dict, Any, Literal
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models import (
    FlowStep,
    QualificationStatus,
    ConversationState,
    LeadData,
    ListingCategory,
    PropertyStatus,
)
from src.config import settings
from src.parsers import (
    parse_budget,
    parse_uk_postcode,
    parse_email,
    parse_phone,
    parse_name,
    parse_yes_no,
    detect_mind_change,
)
from src.data_store import data_store
from src.rag_engine import faq_engine
from src.routing_utils import is_answering_current_step

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Graph State Schema
# -------------------------------------------------------------------------
class LeadGraphState(TypedDict, total=False):
    # Session & turn IO
    session_id: str
    user_message: str
    bot_response: str
    
    # Conversation progression
    current_step: str
    lane: Optional[str]
    qualification_status: str
    completed: bool
    closing_message: Optional[str]
    
    # Lead data fields
    visitor_name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    intent: Optional[str]
    category: Optional[str]
    budget_gbp: Optional[int]
    postcode: Optional[str]
    specific_property_name: Optional[str]
    matched_properties: List[Dict[str, Any]]
    target_property_details: Optional[Dict[str, Any]]
    specific_property_retry_count: int
    
    # Internal routing flags
    is_injection: bool
    is_faq_or_mind_change: bool
    terminal_type: Optional[str]  # "M1", "M2", "M3", "ALREADY_COMPLETED", "CLOSE"
    history: List[Dict[str, str]]


# -------------------------------------------------------------------------
# Helpers to bridge between LeadGraphState and ConversationState
# -------------------------------------------------------------------------
def step_prompt_for_step(step_str: str, visitor_name: Optional[str] = None) -> str:
    salutation = f", {visitor_name}" if visitor_name else ""
    prompts = {
        FlowStep.START.value: "How may I help you? Can I get your name?",
        FlowStep.ASK_NAME.value: "Can I get your name?",
        FlowStep.BUY_OR_SELL.value: f"Are you looking to buy or sell a home{salutation}?",
        FlowStep.SPECIFIC_PROPERTY_PROMPT.value: "Do you have a specific property in mind?",
        FlowStep.NEW_BUILD_OR_RESALE.value: "Are you looking for a new-build or a resale home?",
        FlowStep.A1_ASK_BUDGET.value: f"What is your budget for the new-build home{salutation}?",
        FlowStep.A3_ASK_POSTCODE.value: "What is the postcode of your location of interest?",
        FlowStep.B1_ASK_BUDGET.value: f"What is your budget{salutation}?",
        FlowStep.B2_ASK_POSTCODE.value: "What is the postcode of your location of interest?",
        FlowStep.C1_WHICH_PROPERTY.value: "Which property is it? Please share the property name or address.",
        FlowStep.C2_ASK_PROPERTY_ONCE_MORE.value: "Could you please specify the property name or address once more?",
        FlowStep.C5_SIMILAR_BUDGET.value: "What is your budget?",
        FlowStep.D1_ASK_POSTCODE.value: f"What is your postcode{salutation}?",
        FlowStep.E1_ASK_PHONE.value: f"Can I get your phone number{salutation}?",
        FlowStep.E2_ASK_EMAIL.value: "Can I get your email address?",
        FlowStep.E3_CALLBACK_CONFIRM.value: "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?",
    }
    return prompts.get(step_str, "How else may I assist you?")


def sync_graph_state_to_conversation_state(conv: ConversationState, state: Dict[str, Any]) -> None:
    """
    Updates an existing ConversationState object in-place from LangGraph state values.
    """
    step_val = state.get("current_step", FlowStep.START.value)
    conv.current_step = FlowStep(step_val) if isinstance(step_val, str) else step_val
    conv.lane = state.get("lane")
    
    status_val = state.get("qualification_status", QualificationStatus.IN_PROGRESS.value)
    conv.qualification_status = QualificationStatus(status_val) if isinstance(status_val, str) else status_val
    
    conv.completed = state.get("completed", False)
    conv.closing_message = state.get("closing_message")
    conv.specific_property_retry_count = state.get("specific_property_retry_count", 0)
    conv.target_property_details = state.get("target_property_details")
    conv.history = list(state.get("history") or [])

    # Lead Data
    conv.lead.visitor_name = state.get("visitor_name")
    conv.lead.phone = state.get("phone")
    conv.lead.email = state.get("email")
    conv.lead.intent = state.get("intent")
    conv.lead.category = state.get("category")
    conv.lead.budget_gbp = state.get("budget_gbp")
    conv.lead.postcode = state.get("postcode")
    conv.lead.specific_property_name = state.get("specific_property_name")
    conv.lead.matched_properties = list(state.get("matched_properties") or [])


def graph_state_to_conversation_state(state: Dict[str, Any]) -> ConversationState:
    lead = LeadData(
        visitor_name=state.get("visitor_name"),
        phone=state.get("phone"),
        email=state.get("email"),
        intent=state.get("intent"),
        category=state.get("category"),
        budget_gbp=state.get("budget_gbp"),
        postcode=state.get("postcode"),
        specific_property_name=state.get("specific_property_name"),
        matched_properties=state.get("matched_properties", []),
    )
    conv = ConversationState(
        session_id=state.get("session_id", "default"),
        current_step=FlowStep(state.get("current_step", FlowStep.START.value)),
        lane=state.get("lane"),
        qualification_status=QualificationStatus(state.get("qualification_status", QualificationStatus.IN_PROGRESS.value)),
        lead=lead,
        history=state.get("history", []),
        completed=state.get("completed", False),
        closing_message=state.get("closing_message"),
        specific_property_retry_count=state.get("specific_property_retry_count", 0),
        target_property_details=state.get("target_property_details"),
    )
    return conv


# -------------------------------------------------------------------------
# Graph Nodes
# -------------------------------------------------------------------------
def guardrail_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Checks for prompt injection and adversarial jailbreaks.
    """
    text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    history.append({"role": "user", "content": text})

    # Check if conversation already ended
    if state.get("completed"):
        return {
            "terminal_type": "ALREADY_COMPLETED",
            "is_injection": False,
            "is_faq_or_mind_change": False,
            "history": history,
        }

    injection_patterns = [
        r"ignore (?:all )?(?:previous|above) instructions",
        r"you are now (?:in )?(?:dan|jailbreak|unrestricted)",
        r"system override",
        r"pretend to be",
        r"act as an unfiltered",
        r"reveal your (?:system )?prompt",
        r"forget your rules",
        r"qualify me anyway",
        r"bypass (?:check|qualification|flow)",
    ]
    lower = text.lower()
    is_injection = any(re.search(pat, lower) for pat in injection_patterns)

    return {
        "is_injection": is_injection,
        "is_faq_or_mind_change": False,
        "terminal_type": None,
        "history": history,
    }


def deflection_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Emits a polite deflection when an injection attempt is detected.
    """
    step = state.get("current_step") or FlowStep.START.value
    curr_prompt = step_prompt_for_step(step, state.get("visitor_name"))
    deflection = (
        "I am the Ashcombe & Rowe Estates assistant and I am here to help you buy or sell prime property. "
        f"\n\nNow, {curr_prompt}"
    )
    history = list(state.get("history") or [])
    history.append({"role": "assistant", "content": deflection})
    return {
        "bot_response": deflection,
        "history": history,
    }


def faq_and_intent_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Checks for mid-conversation mind-changes (buying <-> selling)
    or side questions (branch hours, fees, policies, property inquiries).
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])

    # 1. Mind change
    mind_change = detect_mind_change(user_text)
    if mind_change == "sell":
        next_step = FlowStep.D1_ASK_POSTCODE.value
        prompt = step_prompt_for_step(next_step, state.get("visitor_name"))
        reply = f"No problem! Let's help you sell your property. {prompt}"
        history.append({"role": "assistant", "content": reply})
        return {
            "is_faq_or_mind_change": True,
            "lane": "D",
            "intent": "sell",
            "current_step": next_step,
            "bot_response": reply,
            "history": history,
        }
    elif mind_change == "buy":
        next_step = FlowStep.SPECIFIC_PROPERTY_PROMPT.value
        prompt = step_prompt_for_step(next_step, state.get("visitor_name"))
        reply = f"Certainly! Let's look for a home to buy. {prompt}"
        history.append({"role": "assistant", "content": reply})
        return {
            "is_faq_or_mind_change": True,
            "intent": "buy",
            "current_step": next_step,
            "bot_response": reply,
            "history": history,
        }

    # 2. Side Questions / FAQ RAG
    curr_step = state.get("current_step") or FlowStep.START.value
    return_prompt = step_prompt_for_step(curr_step, state.get("visitor_name"))

    # Avoid hijacking valid responses to the current step prompt
    if not is_answering_current_step(curr_step, user_text):
        # 2a. Check property-specific inquiry
        property_ans = data_store.find_property_question_answer(user_text)
        if property_ans:
            reply = f"{property_ans}\n\nNow returning to where we were: {return_prompt}"
            history.append({"role": "assistant", "content": reply})
            return {
                "is_faq_or_mind_change": True,
                "bot_response": reply,
                "history": history,
            }

        # 2b. Check FAQ RAG Knowledge Base (47 company policies)
        faq_item = faq_engine.search_faq(user_text)
        if faq_item:
            reply = f"{faq_item.answer}\n\nNow, back to our conversation: {return_prompt}"
            history.append({"role": "assistant", "content": reply})
            return {
                "is_faq_or_mind_change": True,
                "bot_response": reply,
                "history": history,
            }

        # 2c. Check out-of-domain company questions (head office fallback)
        lower_text = user_text.lower()
        q_words = [
            "who", "what", "where", "when", "why", "how", "which",
            "do you", "can you", "are you", "tell me", "is there", "can i", "do i", "could you"
        ]
        agency_terms = [
            "branch", "office", "fee", "hours", "sunday", "helen", "david",
            "mortgage", "lettings", "rental", "rent", "epc", "service", "commercial",
            "commission", "hangar", "manchester", "policy", "discount", "valuation"
        ]
        has_q = any(qw in lower_text for qw in q_words) or "?" in user_text
        has_term = any(term in lower_text for term in agency_terms)
        if has_q and has_term:
            reply = (
                f"I'm sorry, I don't have that information in my records. "
                f"Please contact our head office on {settings.HEAD_OFFICE_PHONE}.\n\n"
                f"Now returning to your inquiry: {return_prompt}"
            )
            history.append({"role": "assistant", "content": reply})
            return {
                "is_faq_or_mind_change": True,
                "bot_response": reply,
                "history": history,
            }

    return {
        "is_faq_or_mind_change": False,
    }


def initial_router_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles early conversation setup:
    START, ASK_NAME, BUY_OR_SELL, SPECIFIC_PROPERTY_PROMPT, NEW_BUILD_OR_RESALE.
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    step = state.get("current_step") or FlowStep.START.value
    visitor_name = state.get("visitor_name")

    # --- START ---
    if step == FlowStep.START.value:
        name = parse_name(user_text)
        if name:
            visitor_name = name

        props = data_store.search_specific_property(user_text)
        if props:
            # Immediate property match
            p = props[0]
            if p.status == PropertyStatus.AVAILABLE.value:
                reply = (
                    f"Great news, '{p.property_name}' is available! "
                    f"It is a {p.bedrooms}-bedroom, {p.bathrooms}-bathroom {p.property_type.lower()} "
                    f"priced at £{p.price_gbp:,}.\n\n"
                    f"Can I get your phone number so our team can arrange a viewing?"
                )
                history.append({"role": "assistant", "content": reply})
                return {
                    "visitor_name": visitor_name,
                    "lane": "E",
                    "intent": "buy",
                    "category": "specific_property",
                    "specific_property_name": p.property_name,
                    "target_property_details": p.model_dump(),
                    "qualification_status": QualificationStatus.QUALIFIED.value,
                    "current_step": FlowStep.E1_ASK_PHONE.value,
                    "bot_response": reply,
                    "history": history,
                }
            else:
                status_desc = "under offer" if p.status == PropertyStatus.UNDER_OFFER.value else "sold"
                reply = f"Sorry, that property is currently {status_desc}. I can help you find a similar home. What's your budget?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "visitor_name": visitor_name,
                    "lane": "C",
                    "intent": "buy",
                    "category": "specific_property",
                    "specific_property_name": p.property_name,
                    "target_property_details": p.model_dump(),
                    "current_step": FlowStep.C5_SIMILAR_BUDGET.value,
                    "bot_response": reply,
                    "history": history,
                }

        if not visitor_name:
            reply = "Hello! Welcome to Ashcombe & Rowe Estates. Can I get your name?"
            history.append({"role": "assistant", "content": reply})
            return {
                "visitor_name": None,
                "current_step": FlowStep.ASK_NAME.value,
                "bot_response": reply,
                "history": history,
            }
        else:
            lower = user_text.lower()
            if "sell" in lower:
                reply = f"Hello {visitor_name}! Welcome to Ashcombe & Rowe Estates. What is your postcode?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "visitor_name": visitor_name,
                    "lane": "D",
                    "intent": "sell",
                    "current_step": FlowStep.D1_ASK_POSTCODE.value,
                    "bot_response": reply,
                    "history": history,
                }
            elif "buy" in lower:
                reply = f"Hello {visitor_name}! Welcome to Ashcombe & Rowe Estates. Do you have a specific property in mind?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "visitor_name": visitor_name,
                    "intent": "buy",
                    "current_step": FlowStep.SPECIFIC_PROPERTY_PROMPT.value,
                    "bot_response": reply,
                    "history": history,
                }
            else:
                reply = f"Hello {visitor_name}! Welcome to Ashcombe & Rowe Estates. Are you looking to buy or sell a home?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "visitor_name": visitor_name,
                    "current_step": FlowStep.BUY_OR_SELL.value,
                    "bot_response": reply,
                    "history": history,
                }

    # --- ASK_NAME ---
    elif step == FlowStep.ASK_NAME.value:
        name = parse_name(user_text)
        if not name:
            name = user_text.split()[0].title() if user_text else "Guest"
        visitor_name = name

        lower = user_text.lower()
        if "sell" in lower:
            reply = f"Nice to meet you, {name}! What is your postcode?"
            history.append({"role": "assistant", "content": reply})
            return {
                "visitor_name": name,
                "lane": "D",
                "intent": "sell",
                "current_step": FlowStep.D1_ASK_POSTCODE.value,
                "bot_response": reply,
                "history": history,
            }
        elif "buy" in lower:
            reply = f"Nice to meet you, {name}! Do you have a specific property in mind?"
            history.append({"role": "assistant", "content": reply})
            return {
                "visitor_name": name,
                "intent": "buy",
                "current_step": FlowStep.SPECIFIC_PROPERTY_PROMPT.value,
                "bot_response": reply,
                "history": history,
            }
        else:
            reply = f"Nice to meet you, {name}! Are you looking to buy or sell a home?"
            history.append({"role": "assistant", "content": reply})
            return {
                "visitor_name": name,
                "current_step": FlowStep.BUY_OR_SELL.value,
                "bot_response": reply,
                "history": history,
            }

    # --- BUY_OR_SELL ---
    elif step == FlowStep.BUY_OR_SELL.value:
        lower = user_text.lower()
        if "sell" in lower:
            prompt = step_prompt_for_step(FlowStep.D1_ASK_POSTCODE.value, visitor_name)
            reply = f"Excellent. {prompt}"
            history.append({"role": "assistant", "content": reply})
            return {
                "lane": "D",
                "intent": "sell",
                "current_step": FlowStep.D1_ASK_POSTCODE.value,
                "bot_response": reply,
                "history": history,
            }
        elif "buy" in lower or "purchase" in lower:
            prompt = step_prompt_for_step(FlowStep.SPECIFIC_PROPERTY_PROMPT.value, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "intent": "buy",
                "current_step": FlowStep.SPECIFIC_PROPERTY_PROMPT.value,
                "bot_response": prompt,
                "history": history,
            }
        else:
            reply = "Sorry, I didn't quite catch that. Are you looking to buy or sell a home?"
            history.append({"role": "assistant", "content": reply})
            return {
                "bot_response": reply,
                "history": history,
            }

    # --- SPECIFIC_PROPERTY_PROMPT ---
    elif step == FlowStep.SPECIFIC_PROPERTY_PROMPT.value:
        yn = parse_yes_no(user_text)
        if yn is True or "yes" in user_text.lower():
            prompt = step_prompt_for_step(FlowStep.C1_WHICH_PROPERTY.value, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "lane": "C",
                "category": "specific_property",
                "current_step": FlowStep.C1_WHICH_PROPERTY.value,
                "bot_response": prompt,
                "history": history,
            }
        elif yn is False or "no" in user_text.lower():
            prompt = step_prompt_for_step(FlowStep.NEW_BUILD_OR_RESALE.value, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "current_step": FlowStep.NEW_BUILD_OR_RESALE.value,
                "bot_response": prompt,
                "history": history,
            }
        else:
            props = data_store.search_specific_property(user_text)
            if props:
                p = props[0]
                if p.status == PropertyStatus.AVAILABLE.value:
                    reply = (
                        f"Great news, '{p.property_name}' is available! "
                        f"It is a {p.bedrooms}-bedroom, {p.bathrooms}-bathroom {p.property_type.lower()} "
                        f"priced at £{p.price_gbp:,}.\n\n"
                        f"Can I get your phone number so our team can arrange a viewing?"
                    )
                    history.append({"role": "assistant", "content": reply})
                    return {
                        "lane": "E",
                        "category": "specific_property",
                        "specific_property_name": p.property_name,
                        "target_property_details": p.model_dump(),
                        "qualification_status": QualificationStatus.QUALIFIED.value,
                        "current_step": FlowStep.E1_ASK_PHONE.value,
                        "bot_response": reply,
                        "history": history,
                    }
                else:
                    status_desc = "under offer" if p.status == PropertyStatus.UNDER_OFFER.value else "sold"
                    reply = f"Sorry, that property is currently {status_desc}. I can help you find a similar home. What's your budget?"
                    history.append({"role": "assistant", "content": reply})
                    return {
                        "lane": "C",
                        "category": "specific_property",
                        "specific_property_name": p.property_name,
                        "target_property_details": p.model_dump(),
                        "current_step": FlowStep.C5_SIMILAR_BUDGET.value,
                        "bot_response": reply,
                        "history": history,
                    }
            else:
                reply = "Do you have a specific property in mind? (Yes or No)"
                history.append({"role": "assistant", "content": reply})
                return {
                    "bot_response": reply,
                    "history": history,
                }

    # --- NEW_BUILD_OR_RESALE ---
    elif step == FlowStep.NEW_BUILD_OR_RESALE.value:
        lower = user_text.lower()
        if "new" in lower:
            prompt = step_prompt_for_step(FlowStep.A1_ASK_BUDGET.value, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "lane": "A",
                "category": ListingCategory.NEW_BUILD.value,
                "current_step": FlowStep.A1_ASK_BUDGET.value,
                "bot_response": prompt,
                "history": history,
            }
        elif "resale" in lower or "existing" in lower or "used" in lower:
            prompt = step_prompt_for_step(FlowStep.B1_ASK_BUDGET.value, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "lane": "B",
                "category": ListingCategory.RESALE.value,
                "current_step": FlowStep.B1_ASK_BUDGET.value,
                "bot_response": prompt,
                "history": history,
            }
        else:
            reply = "Are you looking for a new-build or a resale home?"
            history.append({"role": "assistant", "content": reply})
            return {
                "bot_response": reply,
                "history": history,
            }

    return {"bot_response": "How can I assist you?"}


def lane_a_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles Lane A: New-Build Buyer flow.
    A1_ASK_BUDGET -> check >= 1M (else M1)
    A3_ASK_POSTCODE -> check new-build at postcode (else M2), check available <= budget (else M3)
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    step = state.get("current_step")
    visitor_name = state.get("visitor_name")

    if step == FlowStep.A1_ASK_BUDGET.value:
        budget = parse_budget(user_text)
        if budget is None:
            reply = "Could you please specify your budget in pounds (e.g. £1,200,000 or £1.5m)?"
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        if budget < 1_000_000:
            return {
                "budget_gbp": budget,
                "qualification_status": QualificationStatus.UNQUALIFIED_M1.value,
                "current_step": FlowStep.ENDED_M1.value,
                "terminal_type": "M1",
            }
        else:
            next_step = FlowStep.A3_ASK_POSTCODE.value
            prompt = step_prompt_for_step(next_step, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "budget_gbp": budget,
                "current_step": next_step,
                "bot_response": prompt,
                "history": history,
            }

    elif step == FlowStep.A3_ASK_POSTCODE.value:
        pc = parse_uk_postcode(user_text)
        if not pc:
            reply = "Please provide a valid UK postcode of your location of interest (e.g. SW3 3RB or GU5 7AQ)."
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        listed_nb = data_store.get_listings_at_postcode(pc, category=ListingCategory.NEW_BUILD.value)
        if not listed_nb:
            return {
                "postcode": pc,
                "qualification_status": QualificationStatus.UNQUALIFIED_M2.value,
                "current_step": FlowStep.ENDED_M2.value,
                "terminal_type": "M2",
            }

        available = data_store.get_available_listings(
            pc, category=ListingCategory.NEW_BUILD.value, max_budget=state.get("budget_gbp")
        )
        if not available:
            return {
                "postcode": pc,
                "qualification_status": QualificationStatus.UNQUALIFIED_M3.value,
                "current_step": FlowStep.ENDED_M3.value,
                "terminal_type": "M3",
            }

        matches_to_show = available[:5]
        match_text = "\n".join(
            f"• {m.property_name} - {m.bedrooms} bed {m.property_type.lower()} at £{m.price_gbp:,}"
            for m in matches_to_show
        )
        name_str = f"{visitor_name}, " if visitor_name else ""
        reply = (
            f"Great news, {name_str}we have available new-build homes matching your requirements:\n"
            f"{match_text}\n\n"
            f"Can I get your phone number so our team can follow up with details?"
        )
        history.append({"role": "assistant", "content": reply})
        return {
            "postcode": pc,
            "matched_properties": [m.model_dump() for m in matches_to_show],
            "qualification_status": QualificationStatus.QUALIFIED.value,
            "lane": "E",
            "current_step": FlowStep.E1_ASK_PHONE.value,
            "bot_response": reply,
            "history": history,
        }

    return {"bot_response": "How can I assist you with your new-build search?"}


def lane_b_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles Lane B: Resale Buyer flow.
    B1_ASK_BUDGET -> records budget
    B2_ASK_POSTCODE -> checks resale listings at postcode (else M2), checks available (else M3)
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    step = state.get("current_step")
    visitor_name = state.get("visitor_name")

    if step == FlowStep.B1_ASK_BUDGET.value:
        budget = parse_budget(user_text)
        if budget is None:
            reply = "Could you please specify your approximate budget in pounds (e.g. £800,000 or £1.2m)?"
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        next_step = FlowStep.B2_ASK_POSTCODE.value
        prompt = step_prompt_for_step(next_step, visitor_name)
        history.append({"role": "assistant", "content": prompt})
        return {
            "budget_gbp": budget,
            "current_step": next_step,
            "bot_response": prompt,
            "history": history,
        }

    elif step == FlowStep.B2_ASK_POSTCODE.value:
        pc = parse_uk_postcode(user_text)
        if not pc:
            reply = "Please provide a valid UK postcode of your location of interest (e.g. SW3 5WU or KT10 8RP)."
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        listed_resale = data_store.get_listings_at_postcode(pc, category=ListingCategory.RESALE.value)
        if not listed_resale:
            return {
                "postcode": pc,
                "qualification_status": QualificationStatus.UNQUALIFIED_M2.value,
                "current_step": FlowStep.ENDED_M2.value,
                "terminal_type": "M2",
            }

        available = data_store.get_available_listings(pc, category=ListingCategory.RESALE.value)
        if not available:
            return {
                "postcode": pc,
                "qualification_status": QualificationStatus.UNQUALIFIED_M3.value,
                "current_step": FlowStep.ENDED_M3.value,
                "terminal_type": "M3",
            }

        matches_to_show = available[:5]
        match_text = "\n".join(
            f"• {m.property_name} - {m.bedrooms} bed {m.property_type.lower()} at £{m.price_gbp:,}"
            for m in matches_to_show
        )
        name_str = f"{visitor_name}, " if visitor_name else ""
        reply = (
            f"Great news, {name_str}we have available resale homes in that area:\n"
            f"{match_text}\n\n"
            f"Can I get your phone number so our team can follow up with details?"
        )
        history.append({"role": "assistant", "content": reply})
        return {
            "postcode": pc,
            "matched_properties": [m.model_dump() for m in matches_to_show],
            "qualification_status": QualificationStatus.QUALIFIED.value,
            "lane": "E",
            "current_step": FlowStep.E1_ASK_PHONE.value,
            "bot_response": reply,
            "history": history,
        }

    return {"bot_response": "How can I assist you with your resale home search?"}


def lane_c_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles Lane C: Specific Property Inquiries & Rerouting.
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    step = state.get("current_step")
    visitor_name = state.get("visitor_name")

    if step in (FlowStep.C1_WHICH_PROPERTY.value, FlowStep.C2_ASK_PROPERTY_ONCE_MORE.value):
        props = data_store.search_specific_property(user_text)
        if len(props) > 1:
            names = ", ".join(f"'{p.property_name}'" for p in props[:4])
            reply = f"We have multiple matching properties: {names}. Which one did you mean?"
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        elif len(props) == 1:
            p = props[0]
            if p.status == PropertyStatus.AVAILABLE.value:
                reply = (
                    f"Great news, '{p.property_name}' is available! "
                    f"It is a {p.bedrooms}-bedroom, {p.bathrooms}-bathroom {p.property_type.lower()} "
                    f"priced at £{p.price_gbp:,}.\n\n"
                    f"Can I get your phone number so our team can arrange a viewing?"
                )
                history.append({"role": "assistant", "content": reply})
                return {
                    "lane": "E",
                    "category": "specific_property",
                    "specific_property_name": p.property_name,
                    "target_property_details": p.model_dump(),
                    "qualification_status": QualificationStatus.QUALIFIED.value,
                    "current_step": FlowStep.E1_ASK_PHONE.value,
                    "bot_response": reply,
                    "history": history,
                }
            else:
                status_desc = "under offer" if p.status == PropertyStatus.UNDER_OFFER.value else "sold"
                reply = f"Sorry, that property is currently {status_desc}. I can help you find a similar home. What's your budget?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "lane": "C",
                    "category": "specific_property",
                    "specific_property_name": p.property_name,
                    "target_property_details": p.model_dump(),
                    "current_step": FlowStep.C5_SIMILAR_BUDGET.value,
                    "bot_response": reply,
                    "history": history,
                }
        else:
            retries = state.get("specific_property_retry_count", 0)
            if retries == 0:
                reply = "I couldn't locate that property in our database. Could you please specify the name or address once more?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "specific_property_retry_count": 1,
                    "current_step": FlowStep.C2_ASK_PROPERTY_ONCE_MORE.value,
                    "bot_response": reply,
                    "history": history,
                }
            else:
                reply = "We could not find that property. Let's find you another home. Are you looking for a new-build or a resale home?"
                history.append({"role": "assistant", "content": reply})
                return {
                    "current_step": FlowStep.NEW_BUILD_OR_RESALE.value,
                    "bot_response": reply,
                    "history": history,
                }

    elif step == FlowStep.C5_SIMILAR_BUDGET.value:
        budget = parse_budget(user_text)
        if budget is None:
            reply = "Please tell me your budget in pounds (e.g. £1,200,000)."
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}

        prop_details = state.get("target_property_details") or {}
        prop_cat = prop_details.get("listing_category", ListingCategory.RESALE.value)

        if prop_cat == ListingCategory.NEW_BUILD.value:
            if budget < 1_000_000:
                return {
                    "budget_gbp": budget,
                    "qualification_status": QualificationStatus.UNQUALIFIED_M1.value,
                    "current_step": FlowStep.ENDED_M1.value,
                    "terminal_type": "M1",
                }
            else:
                next_step = FlowStep.A3_ASK_POSTCODE.value
                prompt = step_prompt_for_step(next_step, visitor_name)
                history.append({"role": "assistant", "content": prompt})
                return {
                    "budget_gbp": budget,
                    "lane": "A",
                    "current_step": next_step,
                    "bot_response": prompt,
                    "history": history,
                }
        else:
            next_step = FlowStep.B2_ASK_POSTCODE.value
            prompt = step_prompt_for_step(next_step, visitor_name)
            history.append({"role": "assistant", "content": prompt})
            return {
                "budget_gbp": budget,
                "lane": "B",
                "current_step": next_step,
                "bot_response": prompt,
                "history": history,
            }

    return {"bot_response": "How can I help you find this property?"}


def lane_d_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles Lane D: Seller flow.
    D1_ASK_POSTCODE -> checks seller_coverage.csv (if not covered -> M2, else qualified)
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    visitor_name = state.get("visitor_name")

    pc = parse_uk_postcode(user_text)
    if not pc:
        reply = "Please provide a valid UK postcode (e.g. SW3 3RB or GU1 3AD)."
        history.append({"role": "assistant", "content": reply})
        return {"bot_response": reply, "history": history}

    is_covered = data_store.is_seller_postcode_covered(pc)
    if not is_covered:
        return {
            "postcode": pc,
            "qualification_status": QualificationStatus.UNQUALIFIED_M2.value,
            "current_step": FlowStep.ENDED_M2.value,
            "terminal_type": "M2",
        }
    else:
        name_str = f"{visitor_name}, " if visitor_name else ""
        reply = f"Great news, {name_str}we cover your area ({pc})! Can I get your phone number so our local branch can arrange a valuation?"
        history.append({"role": "assistant", "content": reply})
        return {
            "postcode": pc,
            "qualification_status": QualificationStatus.QUALIFIED.value,
            "lane": "E",
            "current_step": FlowStep.E1_ASK_PHONE.value,
            "bot_response": reply,
            "history": history,
        }


def lane_e_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Handles Lane E: Gatekeeper Contact Collection & Callback Booking.
    Only active when lead is QUALIFIED.
    """
    user_text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    step = state.get("current_step")

    if step == FlowStep.E1_ASK_PHONE.value:
        phone = parse_phone(user_text)
        if not phone:
            reply = "Could you please provide a valid phone number with at least 10 digits (e.g. 07123 456789)?"
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}
        else:
            reply = "Thank you! Can I get your email address?"
            history.append({"role": "assistant", "content": reply})
            return {
                "phone": phone,
                "current_step": FlowStep.E2_ASK_EMAIL.value,
                "bot_response": reply,
                "history": history,
            }

    elif step == FlowStep.E2_ASK_EMAIL.value:
        email = parse_email(user_text)
        if not email:
            reply = "Could you please provide a valid email address (e.g. name@example.com)?"
            history.append({"role": "assistant", "content": reply})
            return {"bot_response": reply, "history": history}
        else:
            reply = "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?"
            history.append({"role": "assistant", "content": reply})
            return {
                "email": email,
                "current_step": FlowStep.E3_CALLBACK_CONFIRM.value,
                "bot_response": reply,
                "history": history,
            }

    elif step == FlowStep.E3_CALLBACK_CONFIRM.value:
        yn = parse_yes_no(user_text)
        is_no = (yn is False) or bool(re.search(r"\b(?:no|nope|nah|that'?s all|nothing|bye|goodbye)\b", user_text.lower()))
        is_yes = (yn is True) or bool(re.search(r"\b(?:yes|yeah|yep|sure|please|help|another|question)\b", user_text.lower()))

        if is_no and not is_yes:
            reply = "Thank you for chatting with us. Goodbye."
            history.append({"role": "assistant", "content": reply})
            return {
                "completed": True,
                "current_step": FlowStep.ENDED_QUALIFIED.value,
                "closing_message": reply,
                "bot_response": reply,
                "history": history,
            }
        else:
            reply = "How can I help you?"
            history.append({"role": "assistant", "content": reply})
            return {
                "current_step": FlowStep.BUY_OR_SELL.value,
                "bot_response": reply,
                "history": history,
            }

    return {"bot_response": "How else may I help you?"}


def terminal_node(state: LeadGraphState) -> Dict[str, Any]:
    """
    Produces terminal exit messages for M1, M2, M3, or closed conversations.
    """
    t_type = state.get("terminal_type")
    history = list(state.get("history") or [])

    if t_type == "M1":
        msg = settings.M1_MESSAGE
    elif t_type == "M2":
        msg = settings.M2_MESSAGE
    elif t_type == "M3":
        msg = settings.M3_MESSAGE
    elif t_type == "ALREADY_COMPLETED":
        msg = "This conversation has ended. If you need further assistance, please contact our head office on 020 7946 0100. Goodbye."
    else:
        msg = "Thank you for contacting Ashcombe & Rowe Estates. Goodbye."

    history.append({"role": "assistant", "content": msg})
    return {
        "completed": True,
        "closing_message": msg,
        "bot_response": msg,
        "history": history,
    }


# -------------------------------------------------------------------------
# Routing Conditional Edges
# -------------------------------------------------------------------------
def route_after_guardrail(state: LeadGraphState) -> Literal["deflection_node", "terminal_node", "faq_and_intent_node"]:
    if state.get("terminal_type") == "ALREADY_COMPLETED":
        return "terminal_node"
    if state.get("is_injection"):
        return "deflection_node"
    return "faq_and_intent_node"


def route_after_faq(state: LeadGraphState) -> Literal[
    "end_turn",
    "initial_router_node",
    "lane_a_node",
    "lane_b_node",
    "lane_c_node",
    "lane_d_node",
    "lane_e_node",
    "terminal_node"
]:
    if state.get("is_faq_or_mind_change"):
        return "end_turn"

    if state.get("completed"):
        return "terminal_node"

    step = state.get("current_step") or FlowStep.START.value
    lane = state.get("lane")

    if step in (
        FlowStep.START.value,
        FlowStep.ASK_NAME.value,
        FlowStep.BUY_OR_SELL.value,
        FlowStep.SPECIFIC_PROPERTY_PROMPT.value,
        FlowStep.NEW_BUILD_OR_RESALE.value,
    ):
        return "initial_router_node"

    if lane == "A" or step in (FlowStep.A1_ASK_BUDGET.value, FlowStep.A3_ASK_POSTCODE.value):
        return "lane_a_node"

    if lane == "B" or step in (FlowStep.B1_ASK_BUDGET.value, FlowStep.B2_ASK_POSTCODE.value):
        return "lane_b_node"

    if lane == "C" or step in (
        FlowStep.C1_WHICH_PROPERTY.value,
        FlowStep.C2_ASK_PROPERTY_ONCE_MORE.value,
        FlowStep.C5_SIMILAR_BUDGET.value,
    ):
        return "lane_c_node"

    if lane == "D" or step == FlowStep.D1_ASK_POSTCODE.value:
        return "lane_d_node"

    if lane == "E" or step in (
        FlowStep.E1_ASK_PHONE.value,
        FlowStep.E2_ASK_EMAIL.value,
        FlowStep.E3_CALLBACK_CONFIRM.value,
    ):
        return "lane_e_node"

    return "initial_router_node"


def route_after_lane(state: LeadGraphState) -> Literal["terminal_node", "end_turn"]:
    if state.get("terminal_type"):
        return "terminal_node"
    return "end_turn"


# -------------------------------------------------------------------------
# Graph Construction & Compilation
# -------------------------------------------------------------------------
def create_lead_qualification_graph(checkpointer: Optional[Any] = None):
    """
    Constructs and compiles the full LangGraph StateGraph.
    """
    workflow = StateGraph(LeadGraphState)

    # Add Nodes
    workflow.add_node("guardrail_node", guardrail_node)
    workflow.add_node("deflection_node", deflection_node)
    workflow.add_node("faq_and_intent_node", faq_and_intent_node)
    workflow.add_node("initial_router_node", initial_router_node)
    workflow.add_node("lane_a_node", lane_a_node)
    workflow.add_node("lane_b_node", lane_b_node)
    workflow.add_node("lane_c_node", lane_c_node)
    workflow.add_node("lane_d_node", lane_d_node)
    workflow.add_node("lane_e_node", lane_e_node)
    workflow.add_node("terminal_node", terminal_node)

    # Edge from START into Guardrail
    workflow.add_edge(START, "guardrail_node")

    # Conditional branching from Guardrail
    workflow.add_conditional_edges(
        "guardrail_node",
        route_after_guardrail,
        {
            "terminal_node": "terminal_node",
            "deflection_node": "deflection_node",
            "faq_and_intent_node": "faq_and_intent_node",
        },
    )

    # Deflection ends turn
    workflow.add_edge("deflection_node", END)

    # Branching from FAQ / Mind-change check
    workflow.add_conditional_edges(
        "faq_and_intent_node",
        route_after_faq,
        {
            "end_turn": END,
            "initial_router_node": "initial_router_node",
            "lane_a_node": "lane_a_node",
            "lane_b_node": "lane_b_node",
            "lane_c_node": "lane_c_node",
            "lane_d_node": "lane_d_node",
            "lane_e_node": "lane_e_node",
            "terminal_node": "terminal_node",
        },
    )

    # Routing from initial router
    workflow.add_edge("initial_router_node", END)

    # Routing from Lanes: check if an unqualified terminal ending was triggered
    workflow.add_conditional_edges("lane_a_node", route_after_lane, {"terminal_node": "terminal_node", "end_turn": END})
    workflow.add_conditional_edges("lane_b_node", route_after_lane, {"terminal_node": "terminal_node", "end_turn": END})
    workflow.add_conditional_edges("lane_c_node", route_after_lane, {"terminal_node": "terminal_node", "end_turn": END})
    workflow.add_conditional_edges("lane_d_node", route_after_lane, {"terminal_node": "terminal_node", "end_turn": END})
    workflow.add_edge("lane_e_node", END)
    workflow.add_edge("terminal_node", END)

    memory = checkpointer if checkpointer is not None else MemorySaver()
    compiled_app = workflow.compile(checkpointer=memory)
    return compiled_app


# Singleton compiled graph instance
lead_qualification_graph = create_lead_qualification_graph()
