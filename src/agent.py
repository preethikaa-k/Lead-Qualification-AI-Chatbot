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
from src.guardrails import input_guardrail, output_guardrail, InputGuardrail, OutputGuardrail
from src.memory import short_term_memory, ShortTermSessionManager
from src.graph_agent import (
    lead_qualification_graph,
    graph_state_to_conversation_state,
    sync_graph_state_to_conversation_state,
    LeadGraphState,
)

logger = logging.getLogger(__name__)


class LeadQualificationAgent:
    input_guardrail: InputGuardrail = input_guardrail
    output_guardrail: OutputGuardrail = output_guardrail
    memory: ShortTermSessionManager = short_term_memory

    def __init__(self, use_langgraph: bool = True):
        self.use_langgraph = use_langgraph
        self.graph = lead_qualification_graph
        self.state_machine = state_machine
        self.llm = llm_client
        self.tools = ALL_ESTATE_TOOLS
        self.sessions: Dict[str, ConversationState] = {}
        self.full_histories: Dict[str, List[Dict[str, str]]] = {}
        self.live_transcript_files: Dict[str, str] = {}
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.memory = short_term_memory

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationState:
        if not session_id:
            new_id = str(uuid.uuid4())
            state = ConversationState(session_id=new_id)
            self.sessions[new_id] = state
            self.memory.save_session(new_id, state)
            return state

        if session_id in self.sessions:
            return self.sessions[session_id]

        # Check Redis / distributed session store
        loaded = self.memory.load_session(session_id)
        if loaded:
            self.sessions[session_id] = loaded
            return loaded

        state = ConversationState(session_id=session_id)
        self.sessions[session_id] = state
        self.memory.save_session(session_id, state)
        return state

    def _check_prompt_injection(self, text: str) -> bool:
        """
        Detects adversarial attempts to hijack bot instructions or bypass lead qualification.
        Enforces strict compliance: 'Keep the bot on topic and resistant to attempts to make it ignore its rules.'
        """
        return self.input_guardrail.check_prompt_injection(text)

    def _save_live_transcript(self, session_id: str, state: ConversationState) -> None:
        """
        Automatically writes a full live conversation transcript to the transcripts/ folder
        in both human-readable text (.txt) and structured JSON (.json).
        Updates in real time as the client speaks and provides their name.
        """
        try:
            import json
            from pathlib import Path
            transcripts_dir = Path(__file__).resolve().parent.parent / "transcripts"
            transcripts_dir.mkdir(parents=True, exist_ok=True)

            history = self.full_histories.get(session_id, [])
            name = (state.lead.visitor_name or "visitor").strip().lower()
            name_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            sid_short = session_id[:8] if session_id else "live"
            new_base = f"transcript_live_{name_slug}_{sid_short}"

            old_base = self.live_transcript_files.get(session_id)
            if old_base and old_base != new_base:
                old_txt = transcripts_dir / f"{old_base}.txt"
                old_json = transcripts_dir / f"{old_base}.json"
                if old_txt.exists():
                    try:
                        old_txt.unlink()
                    except Exception:
                        pass
                if old_json.exists():
                    try:
                        old_json.unlink()
                    except Exception:
                        pass

            self.live_transcript_files[session_id] = new_base

            # 1. Write Text Transcript
            txt_path = transcripts_dir / f"{new_base}.txt"
            display_name = state.lead.visitor_name or "Anonymous Visitor"
            step_str = state.current_step.value if hasattr(state.current_step, "value") else str(state.current_step)
            status_str = state.qualification_status.value if hasattr(state.qualification_status, "value") else str(state.qualification_status)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Live Conversation: {display_name} ===\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Lane: {state.lane or 'Unassigned'} | Current Step: {step_str}\n")
                f.write(f"Qualification Status: {status_str}\n")
                f.write("-" * 50 + "\n\n")
                for turn in history:
                    speaker = "VISITOR" if turn["role"] == "user" else "BOT"
                    f.write(f"{speaker}: {turn['content']}\n\n")

            # 2. Write JSON Transcript
            json_path = transcripts_dir / f"{new_base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": session_id,
                    "visitor_name": state.lead.visitor_name,
                    "lane": state.lane,
                    "current_step": step_str,
                    "qualification_status": status_str,
                    "completed": state.completed,
                    "lead_data": state.lead.model_dump(),
                    "history": history,
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not auto-save live transcript: {e}")

    def chat(self, session_id: str, user_message: str) -> str:
        """
        Processes a user message for the given session.
        Orchestrated via LangGraph StateGraph with checkpointed memory,
        protected by production-grade Input and Output Guardrails.
        """
        state = self.get_or_create_session(session_id)

        # 1. Input Guardrail: Sanitization, Logging PII Redaction, Injection & Abuse Defense
        curr_prompt = self.state_machine.step_prompt_for(state)
        input_res = self.input_guardrail.apply(user_message, current_prompt=curr_prompt)
        logger.info(f"Session {session_id} ingress (sanitized & PII-masked): {input_res.masked_for_logging}")

        if not input_res.is_safe:
            deflection = input_res.deflection_message or (
                "I am the Ashcombe & Rowe Estates assistant and I am here to help you buy or sell prime property. "
                f"\n\nNow, {curr_prompt}"
            )
            state.history.append({"role": "user", "content": input_res.sanitized_text})
            state.history.append({"role": "assistant", "content": deflection})

            if session_id not in self.full_histories:
                self.full_histories[session_id] = []
            self.full_histories[session_id].append({"role": "user", "content": input_res.sanitized_text})
            self.full_histories[session_id].append({"role": "assistant", "content": deflection})
            self._save_live_transcript(session_id, state)
            return deflection

        sanitized_msg = input_res.sanitized_text

        # 2. Process via LangGraph (or fallback)
        if self.use_langgraph:
            config = {"configurable": {"thread_id": session_id}}
            result = self.graph.invoke(
                {"user_message": sanitized_msg, "session_id": session_id},
                config=config,
            )
            raw_reply = result.get("bot_response", "")
            
            # Sync session state IN PLACE from LangGraph result
            sync_graph_state_to_conversation_state(state, result)
            
            # 3. Output Guardrail: Secret redaction, Contact Privacy Gatekeeper, Grounding verification
            output_res = self.output_guardrail.apply(raw_reply, state_context=result)
            final_reply = output_res.sanitized_output

            # 4. Short-Term Memory Sliding Buffer & Redis Persistence
            self.memory.add_turn(session_id, "user", sanitized_msg)
            self.memory.add_turn(session_id, "assistant", final_reply)
            self.memory.save_session(session_id, state)

            # Record full turn history & auto-save live transcript
            if session_id not in self.full_histories:
                self.full_histories[session_id] = []
            self.full_histories[session_id].append({"role": "user", "content": sanitized_msg})
            self.full_histories[session_id].append({"role": "assistant", "content": final_reply})
            self._save_live_transcript(session_id, state)

            return final_reply

        # Fallback to direct state machine if LangGraph disabled
        raw_reply = self.state_machine.process_turn(state, sanitized_msg)
        state_ctx = {
            "qualification_status": state.qualification_status.value if hasattr(state.qualification_status, "value") else str(state.qualification_status),
            "current_step": state.current_step.value if hasattr(state.current_step, "value") else str(state.current_step),
        }
        output_res = self.output_guardrail.apply(raw_reply, state_context=state_ctx)
        final_reply = output_res.sanitized_output

        # 4. Short-Term Memory Sliding Buffer & Redis Persistence
        self.memory.add_turn(session_id, "user", sanitized_msg)
        self.memory.add_turn(session_id, "assistant", final_reply)
        self.memory.save_session(session_id, state)

        # Record full turn history & auto-save live transcript
        if session_id not in self.full_histories:
            self.full_histories[session_id] = []
        self.full_histories[session_id].append({"role": "user", "content": sanitized_msg})
        self.full_histories[session_id].append({"role": "assistant", "content": final_reply})
        self._save_live_transcript(session_id, state)

        return final_reply

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
            "guardrails_active": True,
            "guardrails_policies": [
                "InputSanitization (2000-char bound, control char removal, DoS defense)",
                "PIIRedaction (PCI-DSS payment cards, UK NINO, credential masking for logs)",
                "PromptInjectionDefense (Jailbreaks, role hijacking, instruction overrides)",
                "SecretLeakPrevention (API keys, env vars, stack trace redaction)",
                "ContactPrivacyGatekeeper (Zero pre-qualification contact harvesting)",
                "GroundingFirewall (Head office canonical 020 7946 0100 verification)",
            ],
            "short_term_memory": self.memory.get_status(),
        }


# Global singleton instance
agent = LeadQualificationAgent()

