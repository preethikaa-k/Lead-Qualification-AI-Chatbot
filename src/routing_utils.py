"""
Routing and Intent Guardrails for Ashcombe & Rowe Estates.
Determines whether user input is directly answering the active flowchart step
or whether it represents an out-of-flow side question / FAQ inquiry.
"""

from typing import Union
from src.models import FlowStep
from src.parsers import (
    parse_name,
    parse_yes_no,
    parse_budget,
    parse_uk_postcode,
    parse_phone,
    parse_email,
)
from src.data_store import data_store


def is_answering_current_step(curr_step: Union[str, FlowStep], user_text: str) -> bool:
    """
    Returns True if user_text is a direct, valid answer to the prompt for curr_step.
    Returns False if user_text is a side question, FAQ, or non-matching input.
    """
    step_val = curr_step.value if isinstance(curr_step, FlowStep) else curr_step
    lower = user_text.lower().strip()

    is_explicit_q = "?" in user_text or any(
        lower.startswith(w + " ") or lower == w
        for w in [
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "can i",
            "can you",
            "do you",
            "is there",
            "are you",
            "tell me",
        ]
    )

    if step_val in (FlowStep.START.value, FlowStep.ASK_NAME.value):
        if is_explicit_q:
            return False
        return bool(parse_name(user_text))

    elif step_val == FlowStep.BUY_OR_SELL.value:
        if is_explicit_q:
            return False
        return "buy" in lower or "sell" in lower or "purchase" in lower

    elif step_val == FlowStep.SPECIFIC_PROPERTY_PROMPT.value:
        if is_explicit_q:
            return False
        yn = parse_yes_no(user_text)
        if yn is not None or "yes" in lower or "no" in lower:
            return True
        return bool(data_store.search_specific_property(user_text))

    elif step_val == FlowStep.NEW_BUILD_OR_RESALE.value:
        if is_explicit_q:
            return False
        return any(w in lower for w in ["new", "resale", "existing", "used"])

    elif step_val in (
        FlowStep.A1_ASK_BUDGET.value,
        FlowStep.B1_ASK_BUDGET.value,
        FlowStep.C5_SIMILAR_BUDGET.value,
    ):
        return parse_budget(user_text) is not None

    elif step_val in (
        FlowStep.A3_ASK_POSTCODE.value,
        FlowStep.B2_ASK_POSTCODE.value,
        FlowStep.D1_ASK_POSTCODE.value,
    ):
        return parse_uk_postcode(user_text) is not None

    elif step_val in (
        FlowStep.C1_WHICH_PROPERTY.value,
        FlowStep.C2_ASK_PROPERTY_ONCE_MORE.value,
    ):
        if is_explicit_q:
            return False
        return bool(data_store.search_specific_property(user_text))

    elif step_val == FlowStep.E1_ASK_PHONE.value:
        return parse_phone(user_text) is not None

    elif step_val == FlowStep.E2_ASK_EMAIL.value:
        return parse_email(user_text) is not None

    elif step_val == FlowStep.E3_CALLBACK_CONFIRM.value:
        if is_explicit_q:
            return False
        return (
            parse_yes_no(user_text) is not None
            or any(
                w in lower
                for w in [
                    "thank",
                    "thanks",
                    "bye",
                    "goodbye",
                    "nothing",
                    "all good",
                    "that's all",
                    "that's everything",
                ]
            )
        )

    return False
