"""
Deterministic Conversation State Machine and Lead Qualification Engine.
Strictly implements the conversation flow from flowchart.png across Lanes A, B, C, D, E.
Handles side questions, FAQ RAG retrieval, and flow resumption.
"""

import logging
import re
from typing import Tuple, Optional, Dict, Any, List

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


class LeadQualificationStateMachine:
    def __init__(self):
        self.data_store = data_store
        self.faq_engine = faq_engine

    def step_prompt_for(self, state: ConversationState) -> str:
        """Returns the natural prompt for the current step."""
        name = state.lead.visitor_name or ""
        salutation = f", {name}" if name else ""

        prompts = {
            FlowStep.START: "How may I help you? Can I get your name?",
            FlowStep.ASK_NAME: "Can I get your name?",
            FlowStep.BUY_OR_SELL: f"Are you looking to buy or sell a home{salutation}?",
            FlowStep.SPECIFIC_PROPERTY_PROMPT: "Do you have a specific property in mind?",
            FlowStep.NEW_BUILD_OR_RESALE: "Are you looking for a new-build or a resale home?",
            FlowStep.A1_ASK_BUDGET: f"What is your budget for the new-build home{salutation}?",
            FlowStep.A3_ASK_POSTCODE: "What is the postcode of your location of interest?",
            FlowStep.B1_ASK_BUDGET: f"What is your budget{salutation}?",
            FlowStep.B2_ASK_POSTCODE: "What is the postcode of your location of interest?",
            FlowStep.C1_WHICH_PROPERTY: "Which property is it? Please share the property name or address.",
            FlowStep.C2_ASK_PROPERTY_ONCE_MORE: "Could you please specify the property name or address once more?",
            FlowStep.C5_SIMILAR_BUDGET: "What is your budget?",
            FlowStep.D1_ASK_POSTCODE: f"What is your postcode{salutation}?",
            FlowStep.E1_ASK_PHONE: f"Can I get your phone number{salutation}?",
            FlowStep.E2_ASK_EMAIL: "Can I get your email address?",
            FlowStep.E3_CALLBACK_CONFIRM: "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?",
        }
        return prompts.get(state.current_step, "How else may I assist you?")

    def handle_side_question_or_faq(self, user_text: str, state: ConversationState) -> Optional[str]:
        """
        Checks if the input is a company, branch, or property question.
        Returns the answer + return prompt if so, else None.
        """
        # 1. Check for specific property question
        property_answer = self.data_store.find_property_question_answer(user_text)
        if property_answer:
            return_prompt = self.step_prompt_for(state)
            return f"{property_answer}\n\nNow returning to where we were: {return_prompt}"

        # 2. Check FAQ RAG
        faq_item = self.faq_engine.search_faq(user_text)
        if faq_item:
            return_prompt = self.step_prompt_for(state)
            return f"{faq_item.answer}\n\nNow, back to our conversation: {return_prompt}"

        # 3. Check if user is asking a clear company question not in FAQ (out of domain)
        lower_text = user_text.lower()
        question_words = ["who", "what", "where", "when", "why", "how", "do you", "can you", "are you", "tell me", "is there"]
        has_question = any(lower_text.startswith(qw) for qw in question_words) or "?" in user_text
        agency_terms = [
            "branch", "office", "fee", "hours", "sunday", "helen", "david",
            "mortgage", "lettings", "rental", "epc", "service", "commercial",
            "commission", "hangar", "manchester", "policy", "discount"
        ]

        if has_question and any(term in lower_text for term in agency_terms):
            return_prompt = self.step_prompt_for(state)
            return (
                f"I'm sorry, I don't have that information in my records. "
                f"Please contact our head office on {settings.HEAD_OFFICE_PHONE}.\n\n"
                f"Now returning to your inquiry: {return_prompt}"
            )

        return None

    def process_turn(self, state: ConversationState, user_input: str) -> str:
        """
        Executes one conversation turn, updating state deterministically.
        """
        user_text = user_input.strip()
        state.history.append({"role": "user", "content": user_text})

        # Check if conversation already ended
        if state.completed:
            msg = "This conversation has ended. If you need further assistance, please contact our head office on 020 7946 0100. Goodbye."
            state.history.append({"role": "assistant", "content": msg})
            return msg

        # Check for mid-conversation mind-change (e.g. switch from buying to selling)
        mind_change = detect_mind_change(user_text)
        if mind_change:
            if mind_change == "sell":
                state.lane = "D"
                state.lead.intent = "sell"
                state.current_step = FlowStep.D1_ASK_POSTCODE
                response = f"No problem! Let's help you sell your property. {self.step_prompt_for(state)}"
                state.history.append({"role": "assistant", "content": response})
                return response
            elif mind_change == "buy":
                state.lead.intent = "buy"
                state.current_step = FlowStep.SPECIFIC_PROPERTY_PROMPT
                response = f"Certainly! Let's look for a home to buy. {self.step_prompt_for(state)}"
                state.history.append({"role": "assistant", "content": response})
                return response

        # Check for side questions (FAQ / Property inquiry)
        if not is_answering_current_step(state.current_step, user_text):
            side_ans = self.handle_side_question_or_faq(user_text, state)
            if side_ans:
                state.history.append({"role": "assistant", "content": side_ans})
                return side_ans

        # Process standard flow according to current step
        step = state.current_step
        response = ""

        # --- STEP: START ---
        if step == FlowStep.START:
            # First turn: greet and ask name
            # Check if name or intent was given in first message
            name = parse_name(user_text)
            if name:
                state.lead.visitor_name = name

            # Check if specific property was named in first message
            props = self.data_store.search_specific_property(user_text)
            if props:
                state.lane = "C"
                state.lead.intent = "buy"
                state.lead.category = "specific_property"
                state.lead.specific_property_name = props[0].property_name
                state.target_property_details = props[0].model_dump()
                response = self._handle_lane_c_match(state, props[0])
                state.history.append({"role": "assistant", "content": response})
                return response

            if not state.lead.visitor_name:
                state.current_step = FlowStep.ASK_NAME
                response = "Hello! Welcome to Ashcombe & Rowe Estates. Can I get your name?"
            else:
                lower = user_text.lower()
                if "sell" in lower:
                    state.lane = "D"
                    state.lead.intent = "sell"
                    state.current_step = FlowStep.D1_ASK_POSTCODE
                    response = f"Hello {state.lead.visitor_name}! Welcome to Ashcombe & Rowe Estates. What is your postcode?"
                elif "buy" in lower:
                    state.lead.intent = "buy"
                    state.current_step = FlowStep.SPECIFIC_PROPERTY_PROMPT
                    response = f"Hello {state.lead.visitor_name}! Welcome to Ashcombe & Rowe Estates. Do you have a specific property in mind?"
                else:
                    state.current_step = FlowStep.BUY_OR_SELL
                    response = f"Hello {state.lead.visitor_name}! Welcome to Ashcombe & Rowe Estates. Are you looking to buy or sell a home?"

        # --- STEP: ASK_NAME ---
        elif step == FlowStep.ASK_NAME:
            name = parse_name(user_text)
            if not name:
                # If cannot extract clean name, use first word as friendly default
                name = user_text.split()[0].title() if user_text else "Guest"
            state.lead.visitor_name = name

            # Check if intent was also mentioned (e.g. "I'm David, looking to buy")
            lower = user_text.lower()
            if "sell" in lower:
                state.lane = "D"
                state.lead.intent = "sell"
                state.current_step = FlowStep.D1_ASK_POSTCODE
                response = f"Nice to meet you, {name}! What is your postcode?"
            elif "buy" in lower:
                state.lead.intent = "buy"
                state.current_step = FlowStep.SPECIFIC_PROPERTY_PROMPT
                response = f"Nice to meet you, {name}! Do you have a specific property in mind?"
            else:
                state.current_step = FlowStep.BUY_OR_SELL
                response = f"Nice to meet you, {name}! Are you looking to buy or sell a home?"

        # --- STEP: BUY_OR_SELL ---
        elif step == FlowStep.BUY_OR_SELL:
            lower = user_text.lower()
            if "sell" in lower:
                state.lane = "D"
                state.lead.intent = "sell"
                state.current_step = FlowStep.D1_ASK_POSTCODE
                response = f"Excellent. {self.step_prompt_for(state)}"
            elif "buy" in lower or "purchase" in lower:
                state.lead.intent = "buy"
                state.current_step = FlowStep.SPECIFIC_PROPERTY_PROMPT
                response = self.step_prompt_for(state)
            else:
                response = f"Sorry, I didn't quite catch that. Are you looking to buy or sell a home?"

        # --- STEP: SPECIFIC_PROPERTY_PROMPT ---
        elif step == FlowStep.SPECIFIC_PROPERTY_PROMPT:
            yn = parse_yes_no(user_text)
            if yn is True or "yes" in user_text.lower():
                state.lane = "C"
                state.lead.category = "specific_property"
                state.current_step = FlowStep.C1_WHICH_PROPERTY
                response = self.step_prompt_for(state)
            elif yn is False or "no" in user_text.lower():
                state.current_step = FlowStep.NEW_BUILD_OR_RESALE
                response = self.step_prompt_for(state)
            else:
                # User might have directly stated the property name here
                props = self.data_store.search_specific_property(user_text)
                if props:
                    state.lane = "C"
                    state.lead.category = "specific_property"
                    response = self._handle_lane_c_match(state, props[0])
                else:
                    response = "Do you have a specific property in mind? (Yes or No)"

        # --- STEP: NEW_BUILD_OR_RESALE ---
        elif step == FlowStep.NEW_BUILD_OR_RESALE:
            lower = user_text.lower()
            if "new" in lower:
                state.lane = "A"
                state.lead.category = ListingCategory.NEW_BUILD.value
                state.current_step = FlowStep.A1_ASK_BUDGET
                response = self.step_prompt_for(state)
            elif "resale" in lower or "existing" in lower or "used" in lower:
                state.lane = "B"
                state.lead.category = ListingCategory.RESALE.value
                state.current_step = FlowStep.B1_ASK_BUDGET
                response = self.step_prompt_for(state)
            else:
                response = "Are you looking for a new-build or a resale home?"

        # --- LANE A: NEW-BUILD BUYER ---
        elif step == FlowStep.A1_ASK_BUDGET:
            budget = parse_budget(user_text)
            if budget is None:
                response = "Could you please specify your budget in pounds (e.g. £1,200,000 or £1.5m)?"
            else:
                state.lead.budget_gbp = budget
                # A2 Decision: Budget >= £1,000,000?
                if budget < 1_000_000:
                    state.qualification_status = QualificationStatus.UNQUALIFIED_M1
                    state.completed = True
                    state.closing_message = settings.M1_MESSAGE
                    state.current_step = FlowStep.ENDED_M1
                    response = settings.M1_MESSAGE
                else:
                    state.current_step = FlowStep.A3_ASK_POSTCODE
                    response = self.step_prompt_for(state)

        elif step == FlowStep.A3_ASK_POSTCODE:
            pc = parse_uk_postcode(user_text)
            if not pc:
                response = "Please provide a valid UK postcode of your location of interest (e.g. SW3 3RB or GU5 7AQ)."
            else:
                state.lead.postcode = pc
                # A4 Decision: New-builds listed at this postcode?
                listed_nb = self.data_store.get_listings_at_postcode(pc, category=ListingCategory.NEW_BUILD.value)
                if not listed_nb:
                    state.qualification_status = QualificationStatus.UNQUALIFIED_M2
                    state.completed = True
                    state.closing_message = settings.M2_MESSAGE
                    state.current_step = FlowStep.ENDED_M2
                    response = settings.M2_MESSAGE
                else:
                    # A5 Decision: Any available within budget?
                    available_matches = self.data_store.get_available_listings(
                        pc, category=ListingCategory.NEW_BUILD.value, max_budget=state.lead.budget_gbp
                    )
                    if not available_matches:
                        state.qualification_status = QualificationStatus.UNQUALIFIED_M3
                        state.completed = True
                        state.closing_message = settings.M3_MESSAGE
                        state.current_step = FlowStep.ENDED_M3
                        response = settings.M3_MESSAGE
                    else:
                        # A6: Show matches (up to 5: name, beds, price)
                        matches_to_show = available_matches[:5]
                        state.lead.matched_properties = [m.model_dump() for m in matches_to_show]
                        state.qualification_status = QualificationStatus.QUALIFIED
                        state.lane = "E"
                        state.current_step = FlowStep.E1_ASK_PHONE
                        
                        match_text = "\n".join(
                            f"• {m.property_name} - {m.bedrooms} bed {m.property_type.lower()} at £{m.price_gbp:,}"
                            for m in matches_to_show
                        )
                        name_str = f"{state.lead.visitor_name}, " if state.lead.visitor_name else ""
                        response = (
                            f"Great news, {name_str}we have available new-build homes matching your requirements:\n"
                            f"{match_text}\n\n"
                            f"Can I get your phone number so our team can follow up with details?"
                        )

        # --- LANE B: RESALE BUYER ---
        elif step == FlowStep.B1_ASK_BUDGET:
            budget = parse_budget(user_text)
            if budget is None:
                response = "Could you please specify your approximate budget in pounds (e.g. £800,000 or £1.2m)?"
            else:
                # Recorded for sales team; no minimum and not used as filter
                state.lead.budget_gbp = budget
                state.current_step = FlowStep.B2_ASK_POSTCODE
                response = self.step_prompt_for(state)

        elif step == FlowStep.B2_ASK_POSTCODE:
            pc = parse_uk_postcode(user_text)
            if not pc:
                response = "Please provide a valid UK postcode of your location of interest (e.g. SW3 5WU or KT10 8RP)."
            else:
                state.lead.postcode = pc
                # B3 Decision: Resale homes listed at this postcode?
                listed_resale = self.data_store.get_listings_at_postcode(pc, category=ListingCategory.RESALE.value)
                if not listed_resale:
                    state.qualification_status = QualificationStatus.UNQUALIFIED_M2
                    state.completed = True
                    state.closing_message = settings.M2_MESSAGE
                    state.current_step = FlowStep.ENDED_M2
                    response = settings.M2_MESSAGE
                else:
                    # B4 Decision: Any of them available now? (no budget filter)
                    available_resale = self.data_store.get_available_listings(pc, category=ListingCategory.RESALE.value)
                    if not available_resale:
                        state.qualification_status = QualificationStatus.UNQUALIFIED_M3
                        state.completed = True
                        state.closing_message = settings.M3_MESSAGE
                        state.current_step = FlowStep.ENDED_M3
                        response = settings.M3_MESSAGE
                    else:
                        # B5: Show matches (up to 5: name, beds, price)
                        matches_to_show = available_resale[:5]
                        state.lead.matched_properties = [m.model_dump() for m in matches_to_show]
                        state.qualification_status = QualificationStatus.QUALIFIED
                        state.lane = "E"
                        state.current_step = FlowStep.E1_ASK_PHONE

                        match_text = "\n".join(
                            f"• {m.property_name} - {m.bedrooms} bed {m.property_type.lower()} at £{m.price_gbp:,}"
                            for m in matches_to_show
                        )
                        name_str = f"{state.lead.visitor_name}, " if state.lead.visitor_name else ""
                        response = (
                            f"Great news, {name_str}we have available resale homes in that area:\n"
                            f"{match_text}\n\n"
                            f"Can I get your phone number so our team can follow up with details?"
                        )

        # --- LANE C: SPECIFIC PROPERTY ---
        elif step in (FlowStep.C1_WHICH_PROPERTY, FlowStep.C2_ASK_PROPERTY_ONCE_MORE):
            props = self.data_store.search_specific_property(user_text)
            if len(props) > 1:
                # Multiple matches -> ask which one
                names = ", ".join(f"'{p.property_name}'" for p in props[:4])
                response = f"We have multiple matching properties: {names}. Which one did you mean?"
            elif len(props) == 1:
                response = self._handle_lane_c_match(state, props[0])
            else:
                # Not found
                if state.specific_property_retry_count == 0:
                    state.specific_property_retry_count += 1
                    state.current_step = FlowStep.C2_ASK_PROPERTY_ONCE_MORE
                    response = "I couldn't locate that property in our database. Could you please specify the name or address once more?"
                else:
                    # Still not found -> continue as general buyer
                    state.current_step = FlowStep.NEW_BUILD_OR_RESALE
                    response = "We could not find that property. Let's find you another home. Are you looking for a new-build or a resale home?"

        elif step == FlowStep.C5_SIMILAR_BUDGET:
            budget = parse_budget(user_text)
            if budget is None:
                response = "Please tell me your budget in pounds (e.g. £1,200,000)."
            else:
                state.lead.budget_gbp = budget
                # C6 Decision: Property type?
                prop_details = state.target_property_details or {}
                prop_cat = prop_details.get("listing_category", ListingCategory.RESALE.value)
                
                if prop_cat == ListingCategory.NEW_BUILD.value:
                    if budget < 1_000_000:
                        state.qualification_status = QualificationStatus.UNQUALIFIED_M1
                        state.completed = True
                        state.closing_message = settings.M1_MESSAGE
                        state.current_step = FlowStep.ENDED_M1
                        response = settings.M1_MESSAGE
                    else:
                        state.lane = "A"
                        state.current_step = FlowStep.A3_ASK_POSTCODE
                        response = self.step_prompt_for(state)
                else:
                    # Resale: continue at B2 (do not ask budget again)
                    state.lane = "B"
                    state.current_step = FlowStep.B2_ASK_POSTCODE
                    response = self.step_prompt_for(state)

        # --- LANE D: SELLER ---
        elif step == FlowStep.D1_ASK_POSTCODE:
            pc = parse_uk_postcode(user_text)
            if not pc:
                response = "Please provide a valid UK postcode (e.g. SW3 3RB or GU1 3AD)."
            else:
                state.lead.postcode = pc
                # D2 Decision: In seller coverage list?
                is_covered = self.data_store.is_seller_postcode_covered(pc)
                if not is_covered:
                    state.qualification_status = QualificationStatus.UNQUALIFIED_M2
                    state.completed = True
                    state.closing_message = settings.M2_MESSAGE
                    state.current_step = FlowStep.ENDED_M2
                    response = settings.M2_MESSAGE
                else:
                    state.qualification_status = QualificationStatus.QUALIFIED
                    state.lane = "E"
                    state.current_step = FlowStep.E1_ASK_PHONE
                    name_str = f"{state.lead.visitor_name}, " if state.lead.visitor_name else ""
                    response = f"Great news, {name_str}we cover your area ({pc})! Can I get your phone number so our local branch can arrange a valuation?"

        # --- LANE E: CONTACT DETAILS & CLOSE ---
        elif step == FlowStep.E1_ASK_PHONE:
            phone = parse_phone(user_text)
            if not phone:
                response = "Could you please provide a valid phone number with at least 10 digits (e.g. 07123 456789)?"
            else:
                state.lead.phone = phone
                state.current_step = FlowStep.E2_ASK_EMAIL
                response = "Thank you! Can I get your email address?"

        elif step == FlowStep.E2_ASK_EMAIL:
            email = parse_email(user_text)
            if not email:
                response = "Could you please provide a valid email address (e.g. name@example.com)?"
            else:
                state.lead.email = email
                state.current_step = FlowStep.E3_CALLBACK_CONFIRM
                response = "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?"

        elif step == FlowStep.E3_CALLBACK_CONFIRM:
            yn = parse_yes_no(user_text)
            is_no = (yn is False) or bool(re.search(r"\b(?:no|nope|nah|that'?s all|nothing|bye|goodbye)\b", user_text.lower()))
            is_yes = (yn is True) or bool(re.search(r"\b(?:yes|yeah|yep|sure|please|help|another|question)\b", user_text.lower()))

            if is_no and not is_yes:
                state.completed = True
                state.current_step = FlowStep.ENDED_QUALIFIED
                state.closing_message = "Thank you for chatting with us. Goodbye."
                response = "Thank you for chatting with us. Goodbye."
            else:
                # Yes -> Reassist ("How can I help you?"), retain details already collected
                state.current_step = FlowStep.BUY_OR_SELL
                response = "How can I help you?"

        else:
            response = "How can I assist you further?"

        state.history.append({"role": "assistant", "content": response})
        return response

    def _handle_lane_c_match(self, state: ConversationState, prop) -> str:
        state.target_property_details = prop.model_dump()
        state.lead.specific_property_name = prop.property_name
        
        # C3 Decision: Available?
        if prop.status == PropertyStatus.AVAILABLE.value:
            # C4: Share details -> E - Contact details
            state.qualification_status = QualificationStatus.QUALIFIED
            state.lane = "E"
            state.current_step = FlowStep.E1_ASK_PHONE
            name_str = f"{state.lead.visitor_name}, " if state.lead.visitor_name else ""
            return (
                f"Great news, {name_str}'{prop.property_name}' is available! "
                f"It is a {prop.bedrooms}-bedroom, {prop.bathrooms}-bathroom {prop.property_type.lower()} "
                f"priced at £{prop.price_gbp:,}.\n\n"
                f"Can I get your phone number so our team can arrange a viewing?"
            )
        else:
            # C5: Sold / under offer
            state.current_step = FlowStep.C5_SIMILAR_BUDGET
            status_desc = "under offer" if prop.status == PropertyStatus.UNDER_OFFER.value else "sold"
            return f"Sorry, that property is currently {status_desc}. I can help you find a similar home. What's your budget?"


# Global singleton instance
state_machine = LeadQualificationStateMachine()
