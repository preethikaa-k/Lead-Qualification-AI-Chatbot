"""
Core Data Models for Ashcombe & Rowe Estates Lead Qualification Chatbot.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ListingCategory(str, Enum):
    NEW_BUILD = "new_build"
    RESALE = "resale"


class PropertyStatus(str, Enum):
    AVAILABLE = "available"
    UNDER_OFFER = "under_offer"
    SOLD = "sold"


class VisitorIntent(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class FlowStep(str, Enum):
    START = "START"
    ASK_NAME = "ASK_NAME"
    BUY_OR_SELL = "BUY_OR_SELL"
    SPECIFIC_PROPERTY_PROMPT = "SPECIFIC_PROPERTY_PROMPT"
    NEW_BUILD_OR_RESALE = "NEW_BUILD_OR_RESALE"
    
    # Lane A: New-build buyer
    A1_ASK_BUDGET = "A1_ASK_BUDGET"
    A3_ASK_POSTCODE = "A3_ASK_POSTCODE"
    
    # Lane B: Resale buyer
    B1_ASK_BUDGET = "B1_ASK_BUDGET"
    B2_ASK_POSTCODE = "B2_ASK_POSTCODE"
    
    # Lane C: Buyer with specific property
    C1_WHICH_PROPERTY = "C1_WHICH_PROPERTY"
    C2_ASK_PROPERTY_ONCE_MORE = "C2_ASK_PROPERTY_ONCE_MORE"
    C5_SIMILAR_BUDGET = "C5_SIMILAR_BUDGET"
    
    # Lane D: Seller
    D1_ASK_POSTCODE = "D1_ASK_POSTCODE"
    
    # Lane E: Contact details & close
    E1_ASK_PHONE = "E1_ASK_PHONE"
    E2_ASK_EMAIL = "E2_ASK_EMAIL"
    E3_CALLBACK_CONFIRM = "E3_CALLBACK_CONFIRM"
    
    # Terminal States
    ENDED_QUALIFIED = "ENDED_QUALIFIED"
    ENDED_M1 = "ENDED_M1"
    ENDED_M2 = "ENDED_M2"
    ENDED_M3 = "ENDED_M3"


class QualificationStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    QUALIFIED = "QUALIFIED"
    UNQUALIFIED_M1 = "UNQUALIFIED_M1"
    UNQUALIFIED_M2 = "UNQUALIFIED_M2"
    UNQUALIFIED_M3 = "UNQUALIFIED_M3"


class PropertyListing(BaseModel):
    property_id: str
    property_name: str
    address: str
    town: str
    postcode: str
    listing_category: str
    property_type: str
    bedrooms: int
    bathrooms: int
    price_gbp: int
    status: str
    description: str
    branch: str
    listed_date: str


class FAQItem(BaseModel):
    faq_id: str
    category: str
    question: str
    answer: str


class LeadData(BaseModel):
    visitor_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    intent: Optional[str] = None  # buy | sell
    category: Optional[str] = None  # new_build | resale | specific_property
    budget_gbp: Optional[int] = None
    postcode: Optional[str] = None
    specific_property_name: Optional[str] = None
    matched_properties: List[Dict[str, Any]] = Field(default_factory=list)


class ConversationState(BaseModel):
    session_id: str
    current_step: FlowStep = FlowStep.START
    lane: Optional[str] = None  # "A", "B", "C", "D", "E"
    qualification_status: QualificationStatus = QualificationStatus.IN_PROGRESS
    lead: LeadData = Field(default_factory=LeadData)
    history: List[Dict[str, str]] = Field(default_factory=list)
    pending_side_question_return_step: Optional[FlowStep] = None
    specific_property_retry_count: int = 0
    completed: bool = False
    closing_message: Optional[str] = None
    target_property_details: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    current_step: str
    qualification_status: str
    lane: Optional[str] = None
    completed: bool
    lead: LeadData
