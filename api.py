"""
Production FastAPI Server for Ashcombe & Rowe Estates Chatbot.
Provides RESTful APIs for chat, session state, qualified leads, and health checks.
Includes:
- API Key Security & RBAC for sensitive lead harvesting
- Production CORS configuration with environment-defined origins
- Real-time RAG Observability & Telemetry monitoring endpoints
- Comprehensive Request Validation and Error Handling
"""

import os
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

from src.models import ChatRequest, ChatResponse, ConversationState, LeadData
from src.agent import agent
from src.config import settings
from src.data_store import data_store
from src.rag_engine import faq_engine

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade AI Lead Qualification Chatbot API with Hybrid RAG Observability.",
    version="1.1.0",
)

# -----------------------------------------------------------------------------
# Security & Authentication for Sensitive Lead Endpoints
# -----------------------------------------------------------------------------
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


def verify_admin_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Validates API key for sensitive data access (/api/leads and APM telemetry).
    If ADMIN_API_KEY is not configured in .env, access is permitted in local dev mode.
    """
    if ADMIN_API_KEY and api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-KEY header. Unauthorized to view lead PII.",
        )
    return True


# -----------------------------------------------------------------------------
# Production CORS Configuration
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Health & Diagnostic Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "fallback_llm_provider": settings.FALLBACK_LLM_PROVIDER,
        "fallback_llm_model": settings.FALLBACK_LLM_MODEL,
        "rag_vector_store": faq_engine.vector_store_type,
        "sparse_retriever": "BM25Okapi",
        "properties_count": len(data_store.properties),
        "seller_coverage_count": len(data_store.seller_postcodes),
        "faq_count": len(faq_engine.faqs),
        "diagnostics": agent.get_system_diagnostics(),
    }


# -----------------------------------------------------------------------------
# Chat & Conversation Routing
# -----------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest) -> ChatResponse:
    clean_msg = request.message.strip()
    if not clean_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(clean_msg) > 4000:
        raise HTTPException(status_code=400, detail="Message exceeds maximum allowed length (4,000 chars).")

    bot_reply = agent.chat(request.session_id, clean_msg)
    state = agent.get_session_state(request.session_id)
    if not state:
        raise HTTPException(status_code=500, detail="Failed to retrieve session state.")

    return ChatResponse(
        session_id=state.session_id,
        message=bot_reply,
        current_step=state.current_step.value,
        qualification_status=state.qualification_status.value,
        lane=state.lane,
        completed=state.completed,
        lead=state.lead,
    )


@app.get("/api/session/{session_id}", response_model=ConversationState)
def get_session(session_id: str) -> ConversationState:
    state = agent.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    return state


# -----------------------------------------------------------------------------
# Secure Lead Management (Authenticated)
# -----------------------------------------------------------------------------
@app.get(
    "/api/leads",
    response_model=List[LeadData],
    dependencies=[Depends(verify_admin_key)],
    summary="Get Qualified Leads (Secured)",
    description="Returns all verified leads who completed qualification and provided phone/email. Protected by X-API-KEY.",
)
def get_qualified_leads() -> List[LeadData]:
    return agent.get_all_qualified_leads()


# -----------------------------------------------------------------------------
# RAG Observability & Grounding Telemetry Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/observability/rag", dependencies=[Depends(verify_admin_key)])
def get_rag_observability(limit: int = 50) -> Dict[str, Any]:
    """
    Returns live RAG metrics:
    - Grounding accuracy rate
    - Average and P95 latency (ms)
    - Total queries and out-of-domain rejection count
    - Detailed audit log of recent hybrid (Chroma + BM25) queries
    """
    return {
        "metrics": faq_engine.get_observability_metrics(),
        "audit_log": faq_engine.get_observability_audit_log(limit=limit),
    }


@app.get("/api/observability/metrics")
def get_telemetry_metrics() -> Dict[str, Any]:
    """Lightweight APM metrics endpoint for load balancers and Prometheus exporters."""
    return faq_engine.get_observability_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
