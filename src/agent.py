"""
High-Level Lead Qualification Agent.
Manages conversation sessions, prompt injection guardrails, and orchestrates turns.
"""

import uuid
import re
from typing import Dict, Optional, List
from src.models import ConversationState, FlowStep, QualificationStatus, LeadData
from src.state_machine import state_machine
from src.llm_factory import llm_client


class LeadQualificationAgent:
    def __init__(self):
        self.state_machine = state_machine
        self.llm = llm_client
        self.sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationState:
        if not session_id or session_id not in self.sessions:
            new_id = session_id or str(uuid.uuid4())
            state = ConversationState(session_id=new_id)
            self.sessions[new_id] = state
            return state
        return self.sessions[session_id]

    def _check_prompt_injection(self, text: str) -> bool:
        """
        Detects adversarial attempts to hijack bot instructions or bypass lead qualification.
        """
        injection_patterns = [
            r"ignore (?:all )?(?:previous|above) instructions",
            r"you are now (?:in )?(?:dan|jailbreak|unrestricted)",
            r"system override",
            r"pretend to be",
            r"act as an unfiltered",
            r"reveal your (?:system )?prompt",
            r"forget your rules",
        ]
        lower = text.lower()
        return any(re.search(pat, lower) for pat in injection_patterns)

    def chat(self, session_id: str, user_message: str) -> str:
        """
        Processes a user message for the given session.
        Returns the agent's response.
        """
        state = self.get_or_create_session(session_id)

        # 1. Guardrail against prompt injection
        if self._check_prompt_injection(user_message):
            curr_prompt = self.state_machine.step_prompt_for(state)
            deflection = (
                "I am the Ashcombe & Rowe Estates assistant and I am here to help you buy or sell prime property. "
                f"\n\nNow, {curr_prompt}"
            )
            state.history.append({"role": "user", "content": user_message})
            state.history.append({"role": "assistant", "content": deflection})
            return deflection

        # 2. Process turn via deterministic state machine
        reply = self.state_machine.process_turn(state, user_message)
        return reply

    def get_session_state(self, session_id: str) -> Optional[ConversationState]:
        return self.sessions.get(session_id)

    def get_all_qualified_leads(self) -> List[LeadData]:
        return [
            s.lead
            for s in self.sessions.values()
            if s.qualification_status == QualificationStatus.QUALIFIED and s.lead.phone
        ]


# Global singleton instance
agent = LeadQualificationAgent()
