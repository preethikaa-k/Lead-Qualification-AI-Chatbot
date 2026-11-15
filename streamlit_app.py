"""
Luxury Streamlit Web Application for Ashcombe & Rowe Estates.
Features:
- Bespoke luxury estate agency visual design (Navy & Champagne Gold)
- Interactive context-aware quick suggestion chips
- Live lead qualification state & pipeline progress tracker
- Matched property showcase cards with live pricing
- Real-time Hybrid RAG, Grounding, and Guardrail observability
- Full conversation transcript export (JSON & Formatted TXT)
"""

import streamlit as st
import uuid
import json
from pathlib import Path

from src.agent import agent
from src.models import QualificationStatus, FlowStep
from src.config import settings
from src.rag_engine import faq_engine
from src.memory import short_term_memory

# Page Configuration
st.set_page_config(
    page_title="Ashcombe & Rowe Estates — Prime Residential Advisory",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Bespoke Luxury CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography & Palette */
    :root {
        --primary-navy: #0f172a;
        --secondary-navy: #1e293b;
        --accent-gold: #c59b27;
        --accent-gold-light: #fef3c7;
        --surface-light: #f8fafc;
        --border-color: #e2e8f0;
        --text-dark: #0f172a;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Luxury Header */
    .luxury-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .luxury-header h1 {
        font-family: 'Cinzel', serif;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #f8fafc;
        margin: 0 0 6px 0;
    }
    .luxury-header p {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: #94a3b8;
        margin: 0 0 14px 0;
    }
    .luxury-header .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .luxury-header .badge-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(197, 155, 39, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        color: #f1f5f9;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Property Cards */
    .property-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #c59b27;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .property-card .prop-title {
        font-family: 'Cinzel', serif;
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
    }
    .property-card .prop-details {
        font-size: 13px;
        color: #475569;
        margin-top: 4px;
    }
    .property-card .prop-price {
        font-size: 14px;
        font-weight: 600;
        color: #b45309;
        margin-top: 4px;
    }

    /* Lead Status Card */
    .status-card {
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .status-qualified {
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        color: #065f46;
    }
    .status-unqualified {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }
    .status-progress {
        background: #eff6ff;
        border: 1px solid #93c5fd;
        color: #1e40af;
    }

    /* Quick Action Chips */
    .chip-container {
        margin: 12px 0 16px 0;
    }
    .chip-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session Initialization
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Trigger initial greeting
    initial_bot_msg = agent.chat(st.session_state.session_id, "Hello")
    st.session_state.messages.append({"role": "assistant", "content": initial_bot_msg})

current_state = agent.get_session_state(st.session_state.session_id)

# -----------------------------------------------------------------------------
# Sidebar: Lead Telemetry & Architecture Dashboard
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏰 Lead Qualification")
    st.caption("Ashcombe & Rowe Estates · Prime Advisory")

    if current_state:
        # Qualification Badge
        status_val = current_state.qualification_status.value
        if status_val == QualificationStatus.QUALIFIED.value:
            st.markdown('<div class="status-card status-qualified">STATUS: QUALIFIED LEAD ✓</div>', unsafe_allow_html=True)
        elif "UNQUALIFIED" in status_val:
            st.markdown(f'<div class="status-card status-unqualified">STATUS: {status_val} ✕</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-progress">STATUS: IN PROGRESS ⏳</div>', unsafe_allow_html=True)

        # Qualification Funnel Progress Tracker
        step_progress_map = {
            FlowStep.START.value: 10,
            FlowStep.ASK_NAME.value: 20,
            FlowStep.BUY_OR_SELL.value: 35,
            FlowStep.SPECIFIC_PROPERTY_PROMPT.value: 50,
            FlowStep.NEW_BUILD_OR_RESALE.value: 50,
            FlowStep.A1_ASK_BUDGET.value: 65,
            FlowStep.A3_ASK_POSTCODE.value: 80,
            FlowStep.B1_ASK_BUDGET.value: 65,
            FlowStep.B2_ASK_POSTCODE.value: 80,
            FlowStep.C1_WHICH_PROPERTY.value: 60,
            FlowStep.D1_ASK_POSTCODE.value: 65,
            FlowStep.E1_ASK_PHONE.value: 85,
            FlowStep.E2_ASK_EMAIL.value: 95,
            FlowStep.E3_CALLBACK_CONFIRM.value: 100,
            FlowStep.ENDED_QUALIFIED.value: 100,
            FlowStep.ENDED_M1.value: 100,
            FlowStep.ENDED_M2.value: 100,
            FlowStep.ENDED_M3.value: 100,
        }
        curr_step_name = current_state.current_step.value if hasattr(current_state.current_step, "value") else str(current_state.current_step)
        prog_pct = step_progress_map.get(curr_step_name, 25)
        st.progress(prog_pct / 100.0, text=f"Step Progress: {prog_pct}% ({curr_step_name})")

        st.divider()
        st.markdown("#### 📋 Captured Lead Profile")
        lead = current_state.lead
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown(f"**Name:**\n`{lead.visitor_name or 'Pending'}`")
            st.markdown(f"**Intent:**\n`{lead.intent or 'Pending'}`")
            st.markdown(f"**Budget:**\n`{f'£{lead.budget_gbp:,}' if lead.budget_gbp else 'Pending'}`")
        with col_l2:
            st.markdown(f"**Category:**\n`{lead.category or 'Pending'}`")
            st.markdown(f"**Postcode:**\n`{lead.postcode or 'Pending'}`")
            st.markdown(f"**Active Lane:**\n`{current_state.lane or 'Start'}`")

        # Contact Details Card (Strict Gatekeeper Protection)
        st.markdown("**Contact Information (Privacy Protected):**")
        phone_val = lead.phone if lead.phone else "🔒 Locked (Qualified only)"
        email_val = lead.email if lead.email else "🔒 Locked (Qualified only)"
        st.markdown(f"- Phone: `{phone_val}`")
        st.markdown(f"- Email: `{email_val}`")

        # Matched Properties
        if lead.matched_properties:
            st.divider()
            st.markdown(f"#### 🏛️ Matching Residences ({len(lead.matched_properties)})")
            for prop in lead.matched_properties[:3]:
                st.markdown(f"""
                <div class="property-card">
                    <div class="prop-title">{prop.get('property_name', 'Luxury Residence')}</div>
                    <div class="prop-details">{prop.get('bedrooms', 0)} Bed {prop.get('property_type', 'Property').title()} · {prop.get('town', 'Prime Location')} ({prop.get('postcode', '')})</div>
                    <div class="prop-price">£{prop.get('price_gbp', 0):,}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        # Telemetry & Observability Accordions
        mem_status = short_term_memory.get_status()
        with st.expander("🧠 Short-Term Memory (Local 5-Turn)", expanded=False):
            st.markdown(f"""
            - **Storage Mode:** `Local In-Memory Buffer`
            - **Sliding Window:** `Last {mem_status.get('window_size', 5)} Turns (Auto-Pruned)`
            - **Persistence Engine:** `Zero External Workflow (Local Memory)`
            """)
            recent_turns = short_term_memory.get_sliding_window(st.session_state.session_id, limit=5)
            if recent_turns:
                st.caption("Active In-Memory Dialogue Window:")
                for t in recent_turns:
                    spk = "👤 User" if t["role"] == "user" else "🤖 Bot"
                    st.text(f"{spk}: {t['content'][:45]}...")

        with st.expander("🛡️ Ingress & Egress Guardrails", expanded=False):
            st.markdown("""
            - 🔒 **PII Redaction**: Payment cards, UK NINOs, mobile phones, and emails masked in logs.
            - 🛡️ **Jailbreak Defense**: Directives like `DAN`, `override`, `ignore rules` blocked with Zero-LLM routing bypass.
            - 🚪 **Contact Gatekeeper**: Strict refusal to harvest contact details before qualification.
            - 🎯 **Canonical Phone**: Guarantees head office `020 7946 0100` on M1–M3 endings.
            """)

        with st.expander("📊 Hybrid RAG & Grounding Monitor", expanded=False):
            rag_metrics = faq_engine.get_observability_metrics()
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Queries", rag_metrics.get("total_queries", 0))
                st.metric("Grounded Hits", rag_metrics.get("grounded_hits", 0))
            with col_m2:
                st.metric("Avg Latency", f"{rag_metrics.get('avg_latency_ms', 0)} ms")
                st.metric("Fidelity Rate", rag_metrics.get("grounding_fidelity_rate", "100%"))
            st.caption("Vector Store: **Chroma DB** | Sparse: **BM25Okapi** | Fusion: **0.60**")

        st.divider()
        # Export Actions
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            transcript_json = json.dumps(current_state.history, indent=2)
            st.download_button(
                label="📥 JSON Log",
                data=transcript_json,
                file_name=f"transcript_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            txt_lines = []
            for h in current_state.history:
                spk = "Visitor" if h.get("role") == "user" else "Assistant"
                txt_lines.append(f"{spk}: {h.get('content', '')}\n")
            st.download_button(
                label="📄 TXT Log",
                data="\n".join(txt_lines),
                file_name=f"transcript_{st.session_state.session_id[:8]}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            short_term_memory.delete_session(st.session_state.session_id)
            initial_msg = agent.chat(st.session_state.session_id, "Hello")
            st.session_state.messages.append({"role": "assistant", "content": initial_msg})
            st.rerun()

# -----------------------------------------------------------------------------
# Main Chat View & Luxury Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="luxury-header">
    <h1>ASHCOMBE & ROWE ESTATES</h1>
    <p>Prime Residential Property Advisory & Lead Qualification Concierge</p>
    <div class="badges">
        <span class="badge-pill">📍 Mayfair HQ: 22 Halden Mews, W1K 2PX</span>
        <span class="badge-pill">📞 Head Office: 020 7946 0100</span>
        <span class="badge-pill">✉️ concierge@ashcombe-rowe.example</span>
        <span class="badge-pill">🛡️ Strict Privacy Protected</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Display Chat Messages
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        st.chat_message("assistant", avatar="🏰").write(msg["content"])

# -----------------------------------------------------------------------------
# Context-Aware Quick Action Suggestion Chips
# -----------------------------------------------------------------------------
def handle_quick_reply(text_to_send: str):
    st.session_state.messages.append({"role": "user", "content": text_to_send})
    with st.spinner("Consulting prime estate records..."):
        bot_reply = agent.chat(st.session_state.session_id, text_to_send)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()

is_concluded = bool(current_state and current_state.completed)

if not is_concluded and current_state:
    step_key = current_state.current_step.value if hasattr(current_state.current_step, "value") else str(current_state.current_step)
    
    st.markdown('<div class="chip-container"><div class="chip-label">Suggested Quick Responses (Click to send):</div></div>', unsafe_allow_html=True)
    
    if step_key in (FlowStep.START.value, FlowStep.ASK_NAME.value):
        q_cols = st.columns(4)
        if q_cols[0].button("👋 Sarah", use_container_width=True):
            handle_quick_reply("Sarah")
        if q_cols[1].button("👋 David", use_container_width=True):
            handle_quick_reply("David")
        if q_cols[2].button("👋 Arthur", use_container_width=True):
            handle_quick_reply("Arthur")
        if q_cols[3].button("❓ Chelsea branch hours?", use_container_width=True):
            handle_quick_reply("What are the Chelsea branch opening hours?")

    elif step_key == FlowStep.BUY_OR_SELL.value:
        q_cols = st.columns(3)
        if q_cols[0].button("🏡 I want to buy a home", use_container_width=True):
            handle_quick_reply("I want to buy a home")
        if q_cols[1].button("💼 I want to sell a property", use_container_width=True):
            handle_quick_reply("I want to sell my property")
        if q_cols[2].button("🏛️ Tell me about your company", use_container_width=True):
            handle_quick_reply("I want to know more about your real estate company")

    elif step_key == FlowStep.SPECIFIC_PROPERTY_PROMPT.value:
        q_cols = st.columns(2)
        if q_cols[0].button("❌ No, show me recommendations", use_container_width=True):
            handle_quick_reply("No")
        if q_cols[1].button("📍 Yes, I have a property in mind", use_container_width=True):
            handle_quick_reply("Yes")

    elif step_key == FlowStep.NEW_BUILD_OR_RESALE.value:
        q_cols = st.columns(2)
        if q_cols[0].button("✨ New-build residence", use_container_width=True):
            handle_quick_reply("New-build")
        if q_cols[1].button("🗝️ Resale property", use_container_width=True):
            handle_quick_reply("Resale")

    elif step_key in (FlowStep.A1_ASK_BUDGET.value, FlowStep.B1_ASK_BUDGET.value, FlowStep.C5_SIMILAR_BUDGET.value):
        q_cols = st.columns(4)
        if q_cols[0].button("💷 £1,500,000", use_container_width=True):
            handle_quick_reply("£1,500,000")
        if q_cols[1].button("💷 £2,000,000", use_container_width=True):
            handle_quick_reply("£2,000,000")
        if q_cols[2].button("💷 £3,500,000", use_container_width=True):
            handle_quick_reply("£3,500,000")
        if q_cols[3].button("💷 £750,000 (Resale)", use_container_width=True):
            handle_quick_reply("£750,000")

    elif step_key in (FlowStep.A3_ASK_POSTCODE.value, FlowStep.B2_ASK_POSTCODE.value, FlowStep.D1_ASK_POSTCODE.value):
        q_cols = st.columns(4)
        if q_cols[0].button("📍 SW3 3RB (Chelsea)", use_container_width=True):
            handle_quick_reply("SW3 3RB")
        if q_cols[1].button("📍 SW10 6AA (Chelsea)", use_container_width=True):
            handle_quick_reply("SW10 6AA")
        if q_cols[2].button("📍 KT11 4GR (Cobham)", use_container_width=True):
            handle_quick_reply("KT11 4GR")
        if q_cols[3].button("📍 OX1 2EA (Oxford)", use_container_width=True):
            handle_quick_reply("OX1 2EA")

    elif step_key == FlowStep.E1_ASK_PHONE.value:
        q_cols = st.columns(2)
        if q_cols[0].button("📱 07123 456789", use_container_width=True):
            handle_quick_reply("07123 456789")
        if q_cols[1].button("📱 +44 7987 654321", use_container_width=True):
            handle_quick_reply("+44 7987 654321")

    elif step_key == FlowStep.E2_ASK_EMAIL.value:
        q_cols = st.columns(2)
        if q_cols[0].button("✉️ sarah.client@example.co.uk", use_container_width=True):
            handle_quick_reply("sarah.client@example.co.uk")
        if q_cols[1].button("✉️ david.investor@example.com", use_container_width=True):
            handle_quick_reply("david.investor@example.com")

    elif step_key == FlowStep.E3_CALLBACK_CONFIRM.value:
        q_cols = st.columns(2)
        if q_cols[0].button("👍 No, thank you!", use_container_width=True):
            handle_quick_reply("No, thank you")
        if q_cols[1].button("❓ When is your Oxford branch open?", use_container_width=True):
            handle_quick_reply("When is the Oxford branch open?")

# -----------------------------------------------------------------------------
# Conversation Conclusion Banner
# -----------------------------------------------------------------------------
if is_concluded:
    st.divider()
    status_label = current_state.qualification_status.value
    st.info(f"✨ **This conversation has concluded ({status_label}).** Thank you for contacting Ashcombe & Rowe Estates.")
    if st.button("🌟 Start New Consultation", type="primary", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        short_term_memory.delete_session(st.session_state.session_id)
        with st.spinner("Initializing concierge session..."):
            initial_msg = agent.chat(st.session_state.session_id, "Hello")
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()

# -----------------------------------------------------------------------------
# Standard Chat Input Box
# -----------------------------------------------------------------------------
placeholder_text = "Type your response or question here..." if not is_concluded else "Consultation concluded. Click 'Start New Consultation' above to restart."
user_query = st.chat_input(placeholder_text, disabled=is_concluded)

if user_query and not is_concluded:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.spinner("Ashcombe & Rowe Concierge is consulting estate records..."):
        response = agent.chat(st.session_state.session_id, user_query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
