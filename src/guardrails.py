"""
Production Input and Output Guardrails for Ashcombe & Rowe Estates.
Ensures user data protection, PII redaction, prompt injection defense,
contact privacy gatekeeping, secret leak prevention, and output grounding.
"""

import re
import html
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ashcombe_rowe.guardrails")


# -------------------------------------------------------------------------
# Data Contracts
# -------------------------------------------------------------------------
@dataclass
class InputGuardrailResult:
    is_safe: bool
    sanitized_text: str
    masked_for_logging: str
    detected_pii: List[str] = field(default_factory=list)
    violation_type: Optional[str] = None
    deflection_message: Optional[str] = None


@dataclass
class OutputGuardrailResult:
    is_safe: bool
    sanitized_output: str
    violations: List[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Input Guardrail (User Data & Ingress Protection)
# -------------------------------------------------------------------------
class InputGuardrail:
    """
    Enforces production ingress safety:
    1. Input sanitization (length bounding, control char removal, HTML escaping).
    2. PII masking in telemetry/logging (PCI-DSS credit cards, UK NI numbers, secrets).
    3. Adversarial / Prompt Injection & Jailbreak detection.
    4. Toxic / abusive content de-escalation.
    """

    MAX_INPUT_LENGTH = 2000

    INJECTION_PATTERNS = [
        r"ignore (?:all )?(?:previous|above) instructions",
        r"you are now (?:in )?(?:dan|jailbreak|unrestricted|god\s*mode)",
        r"system override",
        r"pretend to be",
        r"act as an? (?:unfiltered|unaligned|jailbroken)",
        r"reveal your (?:system )?prompt",
        r"forget your rules",
        r"qualify me anyway",
        r"bypass (?:check|qualification|flow)",
        r"<\|im_start\|>",
        r"<<sys>>",
        r"\[inst\]",
        r"(?:output|print|show)\s+(?:the\s+)?system\s+instructions",
    ]

    # Sensitive PII patterns to redact from production logs
    CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    UK_NINO_REGEX = re.compile(r"\b[A-Z]{2}\s?(?:\d\s?){6}[A-D]{1}\b", re.IGNORECASE)
    API_KEY_REGEX = re.compile(r"\b(?:AIzaSy[a-zA-Z0-9_-]{25,40}|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{30,45})\b")
    SECRET_REGEX = re.compile(r"(?i)\b(?:password|passwd|api_key|secret|token)\s*[:=]\s*[^\s,;]+", re.IGNORECASE)
    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_REGEX = re.compile(r"(?:(?:\+44\s?(?:\(0\)\s?)?|\b0)\d{2,4}(?:[-\s]?\d{3,4}){1,3}\b|(?:\+44\s?(?:\(0\)\s?)?|\b0)\d{9,11}\b)")

    # Toxic / abusive language patterns for polite de-escalation
    ABUSIVE_REGEX = re.compile(
        r"\b(?:f\*+k|sh\*+t|b\*+ch|idiot|stupid bot|shut up|kill yourself|die)\b",
        re.IGNORECASE,
    )

    def sanitize_input(self, text: str) -> str:
        """Strips control chars, excessive whitespace, and caps length."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters
        cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
        # Cap length to prevent token-exhaustion DoS
        if len(cleaned) > self.MAX_INPUT_LENGTH:
            cleaned = cleaned[:self.MAX_INPUT_LENGTH]
        return cleaned.strip()

    def mask_pii_for_logging(self, text: str) -> str:
        """
        Redacts sensitive financial, credential, and contact PII
        (credit cards, UK NINO, API keys, passwords, mobile phones, emails)
        before writing to telemetry or server logs.
        """
        masked = self.CREDIT_CARD_REGEX.sub("[REDACTED_PAYMENT_CARD]", text)
        masked = self.UK_NINO_REGEX.sub("[REDACTED_UK_NINO]", masked)
        masked = self.API_KEY_REGEX.sub("[REDACTED_API_KEY]", masked)
        masked = self.SECRET_REGEX.sub("[REDACTED_CREDENTIAL]", masked)
        masked = self.EMAIL_REGEX.sub("[REDACTED_EMAIL]", masked)
        masked = self.PHONE_REGEX.sub("[REDACTED_PHONE_NUMBER]", masked)
        return masked

    def scan_pii(self, text: str) -> List[str]:
        """Identifies specific types of personal data present in the user message."""
        found = []
        if self.PHONE_REGEX.search(text):
            found.append("PHONE_NUMBER")
        if self.EMAIL_REGEX.search(text):
            found.append("EMAIL_ADDRESS")
        if self.CREDIT_CARD_REGEX.search(text):
            found.append("PAYMENT_CARD")
        if self.UK_NINO_REGEX.search(text):
            found.append("UK_NINO")
        if self.API_KEY_REGEX.search(text) or self.SECRET_REGEX.search(text):
            found.append("SECRET_CREDENTIAL")
        return found

    def check_prompt_injection(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in self.INJECTION_PATTERNS)

    def check_abusive_content(self, text: str) -> bool:
        return bool(self.ABUSIVE_REGEX.search(text))

    def apply(self, user_text: str, current_prompt: Optional[str] = None) -> InputGuardrailResult:
        sanitized = self.sanitize_input(user_text)
        masked_log = self.mask_pii_for_logging(sanitized)
        detected_pii = self.scan_pii(sanitized)

        # 1. Prompt Injection Check
        if self.check_prompt_injection(sanitized):
            prompt_suffix = f"\n\nNow, {current_prompt}" if current_prompt else ""
            deflection = (
                "I am the Ashcombe & Rowe Estates assistant and I am here to help you buy or sell prime property."
                f"{prompt_suffix}"
            )
            return InputGuardrailResult(
                is_safe=False,
                sanitized_text=sanitized,
                masked_for_logging=masked_log,
                detected_pii=detected_pii,
                violation_type="PROMPT_INJECTION",
                deflection_message=deflection,
            )

        # 2. Abusive Language De-escalation
        if self.check_abusive_content(sanitized):
            prompt_suffix = f"\n\n{current_prompt}" if current_prompt else ""
            deflection = (
                "We kindly ask that conversation remain professional and courteous. "
                "I am here to assist with your property search or sale across London, Surrey, and Oxford."
                f"{prompt_suffix}"
            )
            return InputGuardrailResult(
                is_safe=False,
                sanitized_text=sanitized,
                masked_for_logging=masked_log,
                detected_pii=detected_pii,
                violation_type="ABUSIVE_CONTENT",
                deflection_message=deflection,
            )

        return InputGuardrailResult(
            is_safe=True,
            sanitized_text=sanitized,
            masked_for_logging=masked_log,
            detected_pii=detected_pii,
        )


# -------------------------------------------------------------------------
# Output Guardrail (Model Egress & Policy Enforcement)
# -------------------------------------------------------------------------
class OutputGuardrail:
    """
    Enforces production egress safety:
    1. Secret & internal system leak prevention (API keys, code snippets, stack traces).
    2. Contact Privacy Gatekeeper (never solicits phone/email before qualification).
    3. Ground truth verification (guarantees head office phone is 020 7946 0100).
    4. Cleans formatting artifacts (strips 'Bot:', 'Assistant:', markdown role tags).
    """

    OFFICE_PHONE = "020 7946 0100"
    SECRET_LEAK_PATTERNS = [
        r"\b(?:AIzaSy[a-zA-Z0-9_-]{25,40}|sk-[a-zA-Z0-9]{20,})\b",
        r"(?i)\b(?:GEMINI_API_KEY|LLM_API_KEY|DATABASE_URL)\s*=\s*[^\s,;]+",
        r"<\|im_start\|>|<\|im_end\|>",
        r"Traceback \(most recent call last\):",
    ]

    CONTACT_SOLICITATION_REGEX = re.compile(
        r"(?:can i get your (?:phone number|email address)|what is your (?:phone number|email address))",
        re.IGNORECASE,
    )

    ROLE_PREFIX_REGEX = re.compile(r"^(?:(?:Assistant|Bot|AI|System)\s*:\s*)", re.IGNORECASE)

    def check_secret_leak(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.SECRET_LEAK_PATTERNS)

    def redact_secrets(self, text: str) -> str:
        for pat in self.SECRET_LEAK_PATTERNS:
            text = re.sub(pat, "[PROTECTED_SYSTEM_PARAMETER]", text)
        return text

    def clean_formatting(self, text: str) -> str:
        # Strip leading Bot: or Assistant: prefixes if hallucinated by LLM
        lines = [self.ROLE_PREFIX_REGEX.sub("", line) for line in text.splitlines()]
        return "\n".join(lines).strip()

    def enforce_head_office_grounding(self, text: str) -> str:
        """
        Ensures that if an office phone number is mentioned, it is strictly the canonical 020 7946 0100.
        """
        # Replace hallucinated telephone patterns in closing messages with official phone
        hallucinated_phone_pat = re.compile(r"\b0[1-9]\d{2,4}\s?\d{3,4}\s?\d{3,4}\b")
        for match in hallucinated_phone_pat.finditer(text):
            val = match.group(0).replace(" ", "")
            if val != "02079460100" and ("office" in text.lower() or "reach" in text.lower()):
                text = text.replace(match.group(0), self.OFFICE_PHONE)
        return text

    def apply(self, output_text: str, state_context: Optional[Dict[str, Any]] = None) -> OutputGuardrailResult:
        if not output_text:
            return OutputGuardrailResult(is_safe=True, sanitized_output="")

        violations: List[str] = []
        sanitized = output_text

        # 1. Secret Leak Prevention
        if self.check_secret_leak(sanitized):
            violations.append("SECRET_LEAK_DETECTED")
            sanitized = self.redact_secrets(sanitized)

        # 2. Contact Details Gatekeeping Rule (Flowchart Compliance)
        # If the visitor is NOT qualified, the bot must NEVER solicit phone or email.
        if state_context:
            qual_status = state_context.get("qualification_status")
            current_step = state_context.get("current_step")
            allowed_contact_steps = {"E1_ASK_PHONE", "E2_ASK_EMAIL", "E3_CALLBACK_CONFIRM"}

            # If not in Lane E and status is not QUALIFIED
            if qual_status != "QUALIFIED" and current_step not in allowed_contact_steps:
                if self.CONTACT_SOLICITATION_REGEX.search(sanitized):
                    violations.append("UNQUALIFIED_CONTACT_SOLICITATION")
                    # Strip out unauthorized contact solicitation to protect user privacy
                    sanitized = self.CONTACT_SOLICITATION_REGEX.sub("", sanitized).strip()

        # 3. Head Office Grounding
        sanitized = self.enforce_head_office_grounding(sanitized)

        # 4. Clean formatting
        sanitized = self.clean_formatting(sanitized)

        return OutputGuardrailResult(
            is_safe=len(violations) == 0,
            sanitized_output=sanitized,
            violations=violations,
        )


# Global instances for reuse
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()
