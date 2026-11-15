"""
Luxury Client Web Portal for Ashcombe & Rowe Estates.
Designed exclusively for high-net-worth clients, buyers, and sellers.
Features a distraction-free, elegant chat interface with useful agency information,
verified branch hours, prime locations, and matched residence showcases.
"""

import streamlit as st
import uuid
import json

from src.agent import agent
from src.models import QualificationStatus, FlowStep
from src.config import settings
from src.memory import short_term_memory

# Page Configuration
st.set_page_config(
    page_title="Ashcombe & Rowe Estates — Prime Residential Advisory",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Bespoke Luxury Styling (Navy & Champagne Gold)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --primary-navy: #0f172a;
        --secondary-navy: #1e293b;
        --accent-gold: #c59b27;
        --accent-gold-light: #fef3c7;
        --surface-light: #f8fafc;
        --border-color: #e2e8f0;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Luxury Header */
    .luxury-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    .luxury-header h1 {
        font-family: 'Cinzel', serif;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #f8fafc;
        margin: 0 0 6px 0;
    }
    .luxury-header p {
        font-family: 'Inter', sans-serif;
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0 0 12px 0;
    }
    .luxury-header .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
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

    /* Client Preferences Card */
    .client-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .client-card .item-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        font-weight: 600;
    }
    .client-card .item-value {
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 8px;
    }

    /* Property Card */
    .property-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #c59b27;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .property-card .prop-title {
        font-family: 'Cinzel', serif;
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
    }
    .property-card .prop-details {
        font-size: 12.5px;
        color: #475569;
        margin-top: 3px;
    }
    .property-card .prop-price {
        font-size: 13.5px;
        font-weight: 600;
        color: #b45309;
        margin-top: 3px;
    }

    /* Choice Buttons */
    .choice-container {
        margin: 8px 0 14px 0;
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
    initial_bot_msg = agent.chat(st.session_state.session_id, "Hello")
    st.session_state.messages.append({"role": "assistant", "content": initial_bot_msg})

current_state = agent.get_session_state(st.session_state.session_id)
lead = current_state.lead if current_state else None
is_concluded = bool(current_state and current_state.completed)

# -----------------------------------------------------------------------------
# Sidebar: Client Information & Estate Directory
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏰 Ashcombe & Rowe Estates")
    st.caption("Prime Residential Advisory · Founded 2004")

    # If the client has specified criteria, display their active search profile
    if lead and (lead.visitor_name or lead.intent or lead.budget_gbp or lead.postcode):
        st.markdown("#### 📋 Your Property Search")
        st.markdown('<div class="client-card">', unsafe_allow_html=True)
        if lead.visitor_name:
            st.markdown(f'<div class="item-label">Client Name</div><div class="item-value">{lead.visitor_name}</div>', unsafe_allow_html=True)
        if lead.intent:
            intent_label = "Buying a Residence" if lead.intent.lower() == "buy" else "Selling a Property"
            st.markdown(f'<div class="item-label">Consultation Focus</div><div class="item-value">{intent_label}</div>', unsafe_allow_html=True)
        if lead.category:
            cat_label = "New-Build Development" if "new" in lead.category.lower() else "Distinguished Resale"
            st.markdown(f'<div class="item-label">Collection</div><div class="item-value">{cat_label}</div>', unsafe_allow_html=True)
        if lead.budget_gbp:
            st.markdown(f'<div class="item-label">Stated Budget</div><div class="item-value">£{lead.budget_gbp:,}</div>', unsafe_allow_html=True)
        if lead.postcode:
            st.markdown(f'<div class="item-label">Target Location</div><div class="item-value">{lead.postcode}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display Matching Residences if any have been shortlisted
    if lead and lead.matched_properties:
        st.markdown(f"#### 🏛️ Recommended Homes ({len(lead.matched_properties)})")
        for prop in lead.matched_properties[:4]:
            st.markdown(f"""
            <div class="property-card">
                <div class="prop-title">{prop.get('property_name', 'Prime Residence')}</div>
                <div class="prop-details">{prop.get('bedrooms', 0)} Bed {prop.get('property_type', 'Property').title()} · {prop.get('town', 'Prime Location')} ({prop.get('postcode', '')})</div>
                <div class="prop-price">£{prop.get('price_gbp', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    # Agency Directory & Verified Branch Hours (From Official Agency Records)
    st.markdown("#### 📍 Prime Territory & Branches")
    st.markdown("""
    - **Prime London**: Mayfair, Chelsea, Kensington, Richmond, Hampstead
    - **Surrey Country**: Cobham, Guildford
    - **Oxfordshire**: Central Oxford & Headington
    """)

    with st.expander("🕒 Branch Opening Hours", expanded=False):
        st.markdown("""
        - **Head Office (Mayfair)**: Mon–Fri 9:00 am – 6:00 pm *(Closed Weekends)*
        - **Chelsea Branch**: Mon–Fri 9:00 am – 7:00 pm, Sat 10:00 am – 5:00 pm, **Sun 11:00 am – 3:00 pm** *(Sunday Open)*
        - **Hampstead Branch**: Mon–Fri 9:00 am – 6:30 pm, Sat 10:00 am – 4:00 pm
        - **Richmond Branch**: Mon–Fri 9:00 am – 6:00 pm, Sat 9:30 am – 4:30 pm
        - **Cobham Branch**: Mon–Fri 9:00 am – 6:00 pm, Sat 10:00 am – 4:00 pm
        - **Guildford Branch**: Mon–Fri 8:30 am – 5:30 pm, Sat 9:00 am – 4:00 pm
        - **Oxford Branch**: Mon–Fri 9:00 am – 5:30 pm, Sat 10:00 am – 3:00 pm
        *All branches closed on UK Bank Holidays.*
        """)

    with st.expander("📞 Direct Contact Information", expanded=False):
        st.markdown("""
        - **Head Office**: `020 7946 0100`
        - **General Inquiries**: `hello@ashcombe-rowe.example`
        - **Address**: `22 Halden Mews, Mayfair, London W1K 2PX`
        - **Response Commitment**: Within 24 hours by phone or email.
        """)

    st.divider()
    # User Action: Start Fresh
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        short_term_memory.delete_session(st.session_state.session_id)
        with st.spinner("Connecting with concierge..."):
            initial_msg = agent.chat(st.session_state.session_id, "Hello")
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()

# -----------------------------------------------------------------------------
# Main Chat View
# -----------------------------------------------------------------------------
st.markdown("""
<div class="luxury-header">
    <h1>ASHCOMBE & ROWE ESTATES</h1>
    <p>Luxury Residential Property Advisory · London, Surrey & Oxford</p>
    <div class="badges">
        <span class="badge-pill">📍 Mayfair HQ: 22 Halden Mews, W1K 2PX</span>
        <span class="badge-pill">📞 Direct Phone: 020 7946 0100</span>
        <span class="badge-pill">✉️ concierge@ashcombe-rowe.example</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display Dialogue Turns
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        st.chat_message("assistant", avatar="🏰").write(msg["content"])

# -----------------------------------------------------------------------------
# Optional One-Click Choice Chips for Binary Options (NO NAME SUGGESTIONS)
# -----------------------------------------------------------------------------
def send_client_reply(text_to_send: str):
    st.session_state.messages.append({"role": "user", "content": text_to_send})
    with st.spinner("Ashcombe & Rowe Concierge is consulting estate records..."):
        bot_reply = agent.chat(st.session_state.session_id, text_to_send)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()

if not is_concluded and current_state:
    step_key = current_state.current_step.value if hasattr(current_state.current_step, "value") else str(current_state.current_step)

    # Only show quick buttons for binary/choice questions (Never suggest personal names!)
    if step_key == FlowStep.BUY_OR_SELL.value:
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("🏡 I want to buy a home", use_container_width=True):
            send_client_reply("I want to buy a home")
        if col_c2.button("💼 I want to sell a property", use_container_width=True):
            send_client_reply("I want to sell my property")

    elif step_key == FlowStep.SPECIFIC_PROPERTY_PROMPT.value:
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("❌ No, show me recommendations", use_container_width=True):
            send_client_reply("No")
        if col_c2.button("📍 Yes, I have a specific home in mind", use_container_width=True):
            send_client_reply("Yes")

    elif step_key == FlowStep.NEW_BUILD_OR_RESALE.value:
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("✨ New-build residence", use_container_width=True):
            send_client_reply("New-build")
        if col_c2.button("🗝️ Resale residence", use_container_width=True):
            send_client_reply("Resale")

# -----------------------------------------------------------------------------
# Conversation Conclusion Notice
# -----------------------------------------------------------------------------
if is_concluded:
    st.divider()
    status_val = current_state.qualification_status.value
    if status_val == QualificationStatus.QUALIFIED.value:
        st.success("✨ **Thank you! Your inquiry has been forwarded to our senior property negotiators. A dedicated specialist will be in touch within 24 hours.**")
    else:
        st.info("✨ **Thank you for consulting Ashcombe & Rowe Estates. You can reach our head office directly on 020 7946 0100 for any further inquiries.**")

    if st.button("🌟 Start New Property Search", type="primary", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        short_term_memory.delete_session(st.session_state.session_id)
        with st.spinner("Connecting with concierge..."):
            initial_msg = agent.chat(st.session_state.session_id, "Hello")
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()

# -----------------------------------------------------------------------------
# Standard Natural Chat Input Box
# -----------------------------------------------------------------------------
input_placeholder = "Type your response or question here..." if not is_concluded else "Conversation concluded. Click 'Start New Property Search' above to begin anew."
user_query = st.chat_input(input_placeholder, disabled=is_concluded)

if user_query and not is_concluded:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.spinner("Ashcombe & Rowe Concierge is consulting estate records..."):
        response = agent.chat(st.session_state.session_id, user_query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
