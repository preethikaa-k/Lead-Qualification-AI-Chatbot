"""
Production FastAPI Server for Ashcombe & Rowe Estates Chatbot.
Provides RESTful APIs for chat, session state, qualified leads, and health checks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from src.models import ChatRequest, ChatResponse, ConversationState, LeadData
from src.agent import agent
from src.config import settings
from src.data_store import data_store
from src.rag_engine import faq_engine

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade AI Lead Qualification Chatbot API following flowchart.png",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "properties_count": len(data_store.properties),
        "seller_coverage_count": len(data_store.seller_postcodes),
        "faq_count": len(faq_engine.faqs),
    }


@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    bot_reply = agent.chat(request.session_id, request.message)
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


@app.get("/api/leads", response_model=List[LeadData])
def get_qualified_leads() -> List[LeadData]:
    return agent.get_all_qualified_leads()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
