"""
Unit and Integration tests for Lead Qualification Rules & Flowchart branches.
Verifies M1, M2, M3 closing triggers, qualification criteria, and contact detail gates.
"""

import pytest
import uuid
from src.agent import LeadQualificationAgent
from src.models import QualificationStatus, FlowStep
from src.config import settings


@pytest.fixture
def agent():
    return LeadQualificationAgent()


class TestQualificationRules:
    def test_lane_a_budget_below_1m_triggers_m1(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Alice")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "No")  # No specific property
        agent.chat(session_id, "New-build")
        
        # User enters budget below £1,000,000 (e.g. 950k)
        reply = agent.chat(session_id, "My budget is 950k")
        state = agent.get_session_state(session_id)

        assert state.qualification_status == QualificationStatus.UNQUALIFIED_M1
        assert state.completed is True
        assert "Sorry, we don't cater to new-build properties under £1 million" in reply
        assert settings.HEAD_OFFICE_PHONE in reply
        # Contact details MUST NOT be collected for unqualified leads
        assert state.lead.phone is None
        assert state.lead.email is None

    def test_lane_a_postcode_no_new_builds_triggers_m2(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Bob")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "No")
        agent.chat(session_id, "New-build")
        agent.chat(session_id, "£1.5m")  # Valid budget >= £1M
        
        # Postcode with no new-build listings (e.g. SW10 9RR only has resale in dataset)
        reply = agent.chat(session_id, "SW10 9RR")
        state = agent.get_session_state(session_id)

        assert state.qualification_status == QualificationStatus.UNQUALIFIED_M2
        assert state.completed is True
        assert "Sorry, we don't cater to the postcode you provided" in reply
        assert settings.HEAD_OFFICE_PHONE in reply
        assert state.lead.phone is None

    def test_lane_a_no_available_within_budget_triggers_m3(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Charlie")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "No")
        agent.chat(session_id, "New-build")
        # In SW3 3RB, lowest available price is £1,230,000. Budget of £1,100,000 >= £1M but < £1,230,000
        agent.chat(session_id, "£1,100,000")
        reply = agent.chat(session_id, "SW3 3RB")
        state = agent.get_session_state(session_id)

        assert state.qualification_status == QualificationStatus.UNQUALIFIED_M3
        assert state.completed is True
        assert "Sorry, we have no available homes matching your requirements" in reply
        assert settings.HEAD_OFFICE_PHONE in reply

    def test_lane_a_qualified_collects_contact_details(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "David")
        agent.chat(session_id, "Buy")
        agent.chat(session_id, "No")
        agent.chat(session_id, "New-build")
        agent.chat(session_id, "£2m")
        reply = agent.chat(session_id, "SW3 3RB")
        state = agent.get_session_state(session_id)

        # Should be qualified and at Lane E asking for phone
        assert state.qualification_status == QualificationStatus.QUALIFIED
        assert state.current_step == FlowStep.E1_ASK_PHONE
        assert len(state.lead.matched_properties) > 0

        # Now collect phone and email
        agent.chat(session_id, "07123 456789")
        assert state.lead.phone == "07123 456789"
        assert state.current_step == FlowStep.E2_ASK_EMAIL

        agent.chat(session_id, "david@example.com")
        assert state.lead.email == "david@example.com"
        assert state.current_step == FlowStep.E3_CALLBACK_CONFIRM

        # Confirm call-back pre-closure
        final_reply = agent.chat(session_id, "No, that's everything")
        assert state.completed is True
        assert "Thank you for chatting with us. Goodbye." in final_reply

    def test_lane_d_seller_covered_postcode(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Emma")
        agent.chat(session_id, "Sell")
        
        # Covered postcode from seller_coverage.csv
        reply = agent.chat(session_id, "GU1 3AD")
        state = agent.get_session_state(session_id)

        assert state.qualification_status == QualificationStatus.QUALIFIED
        assert state.current_step == FlowStep.E1_ASK_PHONE
        assert state.lane == "E"

    def test_lane_d_seller_uncovered_postcode_triggers_m2(self, agent):
        session_id = str(uuid.uuid4())
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Frank")
        agent.chat(session_id, "Sell")
        
        # Uncovered postcode
        reply = agent.chat(session_id, "E1 6AN")
        state = agent.get_session_state(session_id)

        assert state.qualification_status == QualificationStatus.UNQUALIFIED_M2
        assert state.completed is True
        assert "Sorry, we don't cater to the postcode you provided" in reply
        assert state.lead.phone is None
