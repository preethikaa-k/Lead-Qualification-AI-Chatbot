"""
Streamlit Web Frontend for Ashcombe & Rowe Estates Chatbot.
Provides an interactive chat interface, live lead status dashboard,
flow step tracker, transcript exporter, and Live Hybrid RAG Observability.
"""

import streamlit as st
import uuid
import json
import pandas as pd
from pathlib import Path

from src.agent import agent
from src.models import QualificationStatus, FlowStep
from src.config import settings
from src.rag_engine import faq_engine

st.set_page_config(
    page_title="Ashcombe & Rowe Estates - AI Assistant",
    page_icon="🏡",
    layout="wide",
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Auto-start conversation
    initial_bot_msg = agent.chat(st.session_state.session_id, "Hello")
    st.session_state.messages.append({"role": "assistant", "content": initial_bot_msg})

# Sidebar Dashboard
with st.sidebar:
    st.title("🏡 Lead Qualification")
    st.caption("Ashcombe & Rowe Estates · Commversion Assessment")
    
    st.markdown("### 🤖 Architecture & Telemetry")
    st.markdown("- **Orchestrator:** `LangGraph (StateGraph)`")
    st.markdown(f"- **Primary LLM:** `{settings.LLM_PROVIDER}` ({settings.LLM_MODEL})")
    st.markdown(f"- **Failover LLM:** `{settings.FALLBACK_LLM_PROVIDER}` ({settings.FALLBACK_LLM_MODEL})")
    st.markdown("- **RAG Engine:** `Chroma (Dense) + BM25 (Sparse) Hybrid`")
    st.markdown("- **Grounding Firewall:** `Strict Threshold 0.60 (Zero Hallucination)`")
    
    current_state = agent.get_session_state(st.session_state.session_id)
    
    if current_state:
        # Qualification Status Badge
        status = current_state.qualification_status.value
        if status == QualificationStatus.QUALIFIED.value:
            st.success(f"STATUS: QUALIFIED LEAD ✅")
        elif "UNQUALIFIED" in status:
            st.error(f"STATUS: {status} ❌")
        else:
            st.info(f"STATUS: IN PROGRESS ⏳")

        st.divider()
        st.subheader("📍 Conversation State")
        st.write(f"**Current Step:** `{current_state.current_step.value}`")
        st.write(f"**Active Lane:** `{current_state.lane or 'Start / Unassigned'}`")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

        st.divider()
        st.subheader("📋 Captured Lead Details")
        lead = current_state.lead
        st.write(f"**Name:** {lead.visitor_name or 'Not yet captured'}")
        st.write(f"**Intent:** {lead.intent or '-'}")
        st.write(f"**Category:** {lead.category or '-'}")
        st.write(f"**Budget:** {f'£{lead.budget_gbp:,}' if lead.budget_gbp else '-'}")
        st.write(f"**Postcode:** {lead.postcode or '-'}")
        st.write(f"**Phone:** {lead.phone or 'Locked (Qualified only)'}")
        st.write(f"**Email:** {lead.email or 'Locked (Qualified only)'}")

        if lead.matched_properties:
            st.divider()
            st.subheader(f"🏠 Matches Found ({len(lead.matched_properties)})")
            for prop in lead.matched_properties[:3]:
                st.markdown(f"- **{prop['property_name']}** (£{prop['price_gbp']:,})")

        st.divider()
        
        # ---------------------------------------------------------------------
        # Live Hybrid RAG & Grounding Observability Section
        # ---------------------------------------------------------------------
        with st.expander("📊 Live RAG Observability & Grounding", expanded=False):
            rag_metrics = faq_engine.get_observability_metrics()
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Queries", rag_metrics.get("total_queries", 0))
                st.metric("Grounded Hits", rag_metrics.get("grounded_hits", 0))
            with col_m2:
                st.metric("Avg Latency", f"{rag_metrics.get('avg_latency_ms', 0)} ms")
                st.metric("Fidelity Rate", rag_metrics.get("grounding_fidelity_rate", "100%"))

            st.caption(f"Vector Store: **{rag_metrics.get('vector_store_primary', 'Chroma')}** | Sparse: **BM25Okapi**")

            recent_logs = faq_engine.get_observability_audit_log(limit=5)
            if recent_logs:
                st.markdown("**Recent Query Grounding Audit:**")
                for item in recent_logs:
                    badge = "🟢 Grounded" if item.get("is_grounded") else "🔴 Deflected"
                    st.markdown(
                        f"- `{item['timestamp'][11:19]}` **{badge}** (Hybrid: `{item['hybrid_score']}`)\n"
                        f"  *Query:* \"{item['query'][:35]}...\"\n"
                        f"  *Source:* {item['selected_faq_id'] or 'None (Out of Domain)'}"
                    )

        st.divider()
        # Export Transcript
        transcript_json = json.dumps(current_state.history, indent=2)
        st.download_button(
            label="📥 Download Transcript (JSON)",
            data=transcript_json,
            file_name=f"transcript_{st.session_state.session_id[:8]}.json",
            mime="application/json",
        )

    if st.button("🔄 Reset Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        initial_msg = agent.chat(st.session_state.session_id, "Hello")
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()

# Main Chat View
st.header("Ashcombe & Rowe Estates")
st.caption("Prime Residential Property in London, Surrey & Oxford · Head Office: 020 7946 0100")

# Optional Developer Quick Test Shortcuts (collapsed by default)
with st.expander("🧪 Developer & Evaluator Shortcuts (Click to simulate pre-scripted test paths)", expanded=False):
    st.caption("These buttons simulate sample test personas (Sarah, David, Arthur) for evaluation purposes:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Simulate Sarah (New-Build £1.5M)"):
            st.session_state.messages.append({"role": "user", "content": "I'm Sarah and looking to buy a new-build with a £1.5M budget"})
            reply = agent.chat(st.session_state.session_id, "I'm Sarah and looking to buy a new-build with a £1.5M budget")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
    with col2:
        if st.button("Simulate David (Resale SW3 5WU)"):
            st.session_state.messages.append({"role": "user", "content": "Hi, I'm David, looking for a resale house"})
            reply = agent.chat(st.session_state.session_id, "Hi, I'm David, looking for a resale house")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
    with col3:
        if st.button("Simulate Arthur (Seller KT11 4GR)"):
            st.session_state.messages.append({"role": "user", "content": "Hello, I want to sell my house, my name is Arthur"})
            reply = agent.chat(st.session_state.session_id, "Hello, I want to sell my house, my name is Arthur")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Prominent Restart Banner when Conversation has Ended
if current_state and current_state.completed:
    st.divider()
    st.info(f"🏁 **This conversation has concluded ({current_state.qualification_status.value}).**")
    if st.button("✨ Start New Conversation", type="primary", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        initial_msg = agent.chat(st.session_state.session_id, "Hello")
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()

# Chat Input Box
user_query = st.chat_input("Type your message here..." if not (current_state and current_state.completed) else "Conversation ended. Click 'Start New Conversation' above to restart.")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    response = agent.chat(st.session_state.session_id, user_query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    st.rerun()
