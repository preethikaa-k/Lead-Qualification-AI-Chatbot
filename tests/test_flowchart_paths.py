"""
End-to-End Conversation Flow Tests.
Simulates realistic visitor dialog across Lanes A, B, C, D, and E.
"""

import pytest
import uuid
from src.agent import LeadQualificationAgent
from src.models import FlowStep, QualificationStatus


@pytest.fixture
def agent():
    return LeadQualificationAgent()


class TestConversationPaths:
    def test_full_lane_b_resale_journey(self, agent):
        session_id = str(uuid.uuid4())
        
        # 1. Greet and provide name
        r1 = agent.chat(session_id, "Hello, I am James")
        assert "James" in r1
        
        # 2. Buy or sell -> Buy
        r2 = agent.chat(session_id, "I want to buy a home")
        assert "specific property" in r2.lower()
        
        # 3. Specific property -> No
        r3 = agent.chat(session_id, "No")
        assert "new-build or a resale" in r3.lower()
        
        # 4. Resale
        r4 = agent.chat(session_id, "Resale")
        assert "budget" in r4.lower()
        
        # 5. Budget (no minimum for resale)
        r5 = agent.chat(session_id, "£600,000")
        assert "postcode" in r5.lower()
        
        # 6. Postcode (SW3 5WU has resale)
        r6 = agent.chat(session_id, "sw3 5wu")
        assert "phone number" in r6.lower()
        
        # 7. Contact Details: Phone
        r7 = agent.chat(session_id, "07987 654321")
        assert "email" in r7.lower()
        
        # 8. Contact Details: Email
        r8 = agent.chat(session_id, "james@example.com")
        assert "24 hours" in r8
        
        # 9. Callback confirm -> No
        r9 = agent.chat(session_id, "No, thank you")
        assert "Goodbye" in r9
        
        state = agent.get_session_state(session_id)
        assert state.completed is True
        assert state.qualification_status == QualificationStatus.QUALIFIED
        assert state.lead.phone == "07987 654321"
        assert state.lead.email == "james@example.com"

    def test_full_lane_c_available_property_journey(self, agent):
        session_id = str(uuid.uuid4())
        
        agent.chat(session_id, "Hi, my name is Karen")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "Yes")  # Has specific property
        
        # Ask which property -> provide available property name
        reply = agent.chat(session_id, "Apartment 1, Aldermoor Place")
        
        # Should confirm availability, price, and ask for phone
        assert "available" in reply.lower()
        assert "1,330,000" in reply
        assert "phone" in reply.lower()
        
        state = agent.get_session_state(session_id)
        assert state.current_step == FlowStep.E1_ASK_PHONE

    def test_full_lane_c_sold_property_rerouting(self, agent):
        session_id = str(uuid.uuid4())
        
        agent.chat(session_id, "Hi, I'm Liam")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "Yes")
        
        # Townhouse 2, Riverside Quarter is SOLD (new_build)
        reply = agent.chat(session_id, "Townhouse 2, Riverside Quarter")
        assert "sold" in reply.lower() or "under offer" in reply.lower()
        assert "budget" in reply.lower()
        
        # Liam gives budget below £1M for this new-build -> triggers M1
        m1_reply = agent.chat(session_id, "800k")
        state = agent.get_session_state(session_id)
        assert state.qualification_status == QualificationStatus.UNQUALIFIED_M1
        assert "under £1 million" in m1_reply

    def test_lane_e_reassist_loop(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hi, I am Mia")
        agent.chat(session_id, "Sell")
        agent.chat(session_id, "GU1 3AD")
        agent.chat(session_id, "07111 222333")
        agent.chat(session_id, "mia@example.com")
        
        # At E3 pre-closure, visitor says Yes to need more help
        reassist_reply = agent.chat(session_id, "Yes, I also have another question")
        assert "How can I help you?" in reassist_reply
        
        state = agent.get_session_state(session_id)
        # Verify previous contact details are preserved!
        assert state.lead.phone == "07111 222333"
        assert state.lead.email == "mia@example.com"
