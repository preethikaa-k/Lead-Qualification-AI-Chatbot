"""
Deterministic Input Parsers and Normalizers.
Handles natural language budgets, UK postcodes, phone numbers, emails, and names.
"""

import re
from typing import Optional, Tuple


def parse_budget(text: str) -> Optional[int]:
    """
    Parses natural language budget expressions into an integer in GBP.
    Handles:
      - '£1.5m', '1.5 million', '1.5m' -> 1,500,000
      - '950k', '£950k', '950 thousand' -> 950,000
      - '£1,200,000', '1,200,000', '1200000' -> 1,200,000
      - 'around £2m', 'my budget is 800k'
    """
    if not text:
        return None

    clean_text = text.lower().replace(",", "").strip()

    # Match millions: e.g. 1.5m, 1.5 million, 2m, 2 million, £1.5m
    m_match = re.search(r"£?\s*(\d+(?:\.\d+)?)\s*(?:m|million|millions|mil)\b", clean_text)
    if m_match:
        val = float(m_match.group(1))
        return int(val * 1_000_000)

    # Match thousands: e.g. 950k, 950 thousand, £850k
    k_match = re.search(r"£?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand|grand)\b", clean_text)
    if k_match:
        val = float(k_match.group(1))
        return int(val * 1_000)

    # Match raw numbers: e.g. £1500000, 1500000, £950000
    num_match = re.search(r"£?\s*(\d{5,9})\b", clean_text)
    if num_match:
        return int(num_match.group(1))

    return None


def parse_uk_postcode(text: str) -> Optional[str]:
    """
    Extracts and standardizes UK postcodes.
    Handles any spacing and case:
      - 'gu5 7aq' -> 'GU5 7AQ'
      - 'SW33RB' -> 'SW3 3RB'
      - 'Postcode is ox2 5py' -> 'OX2 5PY'
    UK postcode format regex:
      Area: 1-2 letters
      District: 1-2 digits or 1 digit + 1 letter
      Space
      Sector: 1 digit
      Unit: 2 letters
    """
    if not text:
        return None

    # Matches standard UK outward + inward postcode parts
    pattern = r"\b([A-Z]{1,2}[0-9][0-9A-Z]?)\s*([0-9][A-Z]{2})\b"
    match = re.search(pattern, text.upper().strip())
    if match:
        outward = match.group(1)
        inward = match.group(2)
        return f"{outward} {inward}"

    return None


def parse_email(text: str) -> Optional[str]:
    """
    Extracts a valid email address from text.
    """
    if not text:
        return None

    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text.strip())
    if match:
        return match.group(0).lower()
    return None


def parse_phone(text: str) -> Optional[str]:
    """
    Validates and extracts UK or international telephone numbers.
    Must contain at least 10 digits.
    """
    if not text:
        return None

    # Find candidate sequences of digits, spaces, hyphens, plus signs
    candidates = re.findall(r"(?:(?:\+|00)\d{1,3}[\s-]*)?(?:0\d{2,4}[\s-]*)?\d{3,4}[\s-]?\d{3,4}", text)
    for c in candidates:
        digits_only = re.sub(r"\D", "", c)
        if len(digits_only) >= 10:
            return c.strip()

    # Fallback: check all consecutive digits
    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) >= 10 and len(digits_only) <= 15:
        return text.strip()

    return None


def parse_name(text: str) -> Optional[str]:
    """
    Extracts the visitor's name from introductory phrases or bare inputs.
    e.g. 'I am Sarah Connor', "My name's David", 'Call me Alice', 'Sarah'
    """
    if not text:
        return None

    cleaned = text.strip()
    # Match patterns like: my name is X, i'm X, i am X, this is X, call me X
    pattern = r"(?:my name is|i am|i'm|call me|this is|it's)\s+([A-Za-z\s'-]{2,30})"
    match = re.search(pattern, cleaned, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        # Clean any trailing punctuation
        name = re.sub(r"[.,!?;:].*$", "", name).strip()
        stop_words = {"looking", "interested", "want", "here", "to"}
        name_words = set(re.findall(r"\b\w+\b", name.lower()))
        if name and not (stop_words & name_words):
            return name.title()

    # If it's a short 1-3 word phrase without verbs, assume it's the direct name
    words = cleaned.split()
    if 1 <= len(words) <= 3:
        # Check that none of the words are common conversation verbs/stops
        stop_words = {"hi", "hello", "hey", "yes", "no", "buy", "sell", "house", "flat", "please"}
        if not any(w.lower() in stop_words for w in words):
            clean_name = re.sub(r"[^A-Za-z\s'-]", "", cleaned).strip()
            if len(clean_name) >= 2:
                return clean_name.title()

    return None


def parse_yes_no(text: str) -> Optional[bool]:
    """
    Parses affirmative or negative answers.
    """
    clean = text.lower().strip()
    clean = re.sub(r"[^\w\s]", "", clean)
    words = set(clean.split())

    affirmatives = {"yes", "yeah", "yep", "yup", "sure", "definitely", "correct", "please", "indeed", "ok", "okay"}
    negatives = {"no", "nope", "nah", "not", "none", "nothing", "goodbye", "bye", "dont", "neither"}

    if words & affirmatives:
        return True
    if words & negatives:
        return False
    return None


def detect_mind_change(text: str) -> Optional[str]:
    """
    Detects if the visitor is changing their mind mid-flow.
    Returns 'buy', 'sell', or None.
    Handles phrases like:
      - 'Actually, I want to sell instead'
      - 'I changed my mind, I want to buy'
      - 'I'd rather sell my house'
      - 'Switch to selling'
      - 'I want to sell my property now'
    """
    clean = text.lower()
    
    # Explicit mind-change markers or verbs
    sell_patterns = [
        r"\b(?:actually|change(?:d)? my mind|instead|rather|switch to|prefer to)\b.*\b(?:sell|selling)\b",
        r"\b(?:want to|like to|decided to|switch to)\s+(?:sell|selling)\b",
        r"\bsell my (?:home|house|property|flat)\b",
    ]
    buy_patterns = [
        r"\b(?:actually|change(?:d)? my mind|instead|rather|switch to|prefer to)\b.*\b(?:buy|buying|purchase)\b",
        r"\b(?:want to|like to|decided to|switch to)\s+(?:buy|buying|purchase)\b",
        r"\bbuy a (?:home|house|property|flat)\b",
    ]

    for pat in sell_patterns:
        if re.search(pat, clean):
            return "sell"

    for pat in buy_patterns:
        if re.search(pat, clean):
            return "buy"

    # Fallback to keyword containment
    if any(k in clean for k in ("actually", "change my mind", "instead", "rather")):
        if "sell" in clean:
            return "sell"
        if "buy" in clean or "purchas" in clean:
            return "buy"

    return None
