"""
High-Level Agentic Lead Qualification Orchestrator.
Powered by LangGraph:
1. Stateful compiled StateGraph with MemorySaver checkpointing
2. Multi-lane routing (Lanes A, B, C, D, E) and Gatekeeper validation
3. Unified LLM (Local Qwen 2.5 with Google Gemini Failover)
4. LangChain Tools for RAG, Property Lookup, and Coverage Checks
5. Prompt Injection & Adversarial Jailbreak Defenses
"""

import uuid
import re
import logging
from typing import Dict, Optional, List, Any

from src.models import ConversationState, FlowStep, QualificationStatus, LeadData
from src.state_machine import state_machine
from src.llm_factory import llm_client
from src.tools import ALL_ESTATE_TOOLS
from src.config import settings
from src.graph_agent import (
    lead_qualification_graph,
    graph_state_to_conversation_state,
    sync_graph_state_to_conversation_state,
    LeadGraphState,
)

logger = logging.getLogger(__name__)


class LeadQualificationAgent:
    def __init__(self, use_langgraph: bool = True):
        self.use_langgraph = use_langgraph
        self.graph = lead_qualification_graph
        self.state_machine = state_machine
        self.llm = llm_client
        self.tools = ALL_ESTATE_TOOLS
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
        Enforces strict compliance: 'Keep the bot on topic and resistant to attempts to make it ignore its rules.'
        """
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
        return any(re.search(pat, lower) for pat in injection_patterns)

    def chat(self, session_id: str, user_message: str) -> str:
        """
        Processes a user message for the given session.
        Orchestrated via LangGraph StateGraph with checkpointed memory.
        """
        state = self.get_or_create_session(session_id)

        if self.use_langgraph:
            config = {"configurable": {"thread_id": session_id}}
            result = self.graph.invoke(
                {"user_message": user_message, "session_id": session_id},
                config=config,
            )
            reply = result.get("bot_response", "")
            
            # Sync session state IN PLACE from LangGraph result
            sync_graph_state_to_conversation_state(state, result)
            return reply

        # Fallback to direct state machine if LangGraph disabled
        if self._check_prompt_injection(user_message):
            curr_prompt = self.state_machine.step_prompt_for(state)
            deflection = (
                "I am the Ashcombe & Rowe Estates assistant and I am here to help you buy or sell prime property. "
                f"\n\nNow, {curr_prompt}"
            )
            state.history.append({"role": "user", "content": user_message})
            state.history.append({"role": "assistant", "content": deflection})
            return deflection

        reply = self.state_machine.process_turn(state, user_message)
        return reply

    def get_session_state(self, session_id: str) -> Optional[ConversationState]:
        state = self.sessions.get(session_id)
        if state is None:
            state = self.get_or_create_session(session_id)
        if self.use_langgraph:
            config = {"configurable": {"thread_id": session_id}}
            snapshot = self.graph.get_state(config)
            if snapshot and snapshot.values:
                sync_graph_state_to_conversation_state(state, snapshot.values)
        return state

    def get_all_qualified_leads(self) -> List[LeadData]:
        """
        Returns all verified leads who completed qualification and provided contact info.
        """
        return [
            s.lead
            for s in self.sessions.values()
            if s.qualification_status == QualificationStatus.QUALIFIED and s.lead.phone
        ]

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """
        Returns operational diagnostics for monitoring and health endpoints.
        """
        return {
            "orchestrator": "LangGraph (StateGraph + MemorySaver)",
            "active_sessions": len(self.sessions),
            "qualified_leads_count": len(self.get_all_qualified_leads()),
            "llm_status": self.llm.get_provider_status(),
            "tools_registered": [t.name for t in self.tools],
        }


# Global singleton instance
agent = LeadQualificationAgent()

