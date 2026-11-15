"""
Configuration and Environment Settings for Ashcombe & Rowe Estates Chatbot.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "Ashcombe & Rowe Estates - Lead Qualification Chatbot"
    HEAD_OFFICE_PHONE: str = "020 7946 0100"
    HEAD_OFFICE_EMAIL: str = "hello@ashcombe-rowe.example"
    HEAD_OFFICE_ADDRESS: str = "22 Halden Mews, Mayfair, London W1K 2PX"
    
    # LLM Configuration (Primary: Google Gemini Cloud, Fallback: Local Qwen via Ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # gemini | ollama | groq | openai
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    LLM_API_KEY: str = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or ""
    )
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    
    # Fallback LLM Settings (Automatic Failover: Primary Cloud Gemini -> Secondary Local Qwen)
    FALLBACK_LLM_PROVIDER: str = os.getenv("FALLBACK_LLM_PROVIDER", "ollama")  # ollama | gemini | groq | openai | none
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "qwen2.5:3b")
    FALLBACK_LLM_API_KEY: str = os.getenv("FALLBACK_LLM_API_KEY", "ollama")
    FALLBACK_LLM_BASE_URL: str = os.getenv("FALLBACK_LLM_BASE_URL", "http://localhost:11434")
    LLM_ENRICHMENT_ENABLED: bool = os.getenv("LLM_ENRICHMENT_ENABLED", "true").lower() in ("true", "1", "yes")
    
    # Embedding & Vector Store Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local | gemini | openai
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    RAG_VECTOR_STORE: str = os.getenv("RAG_VECTOR_STORE", "chroma")  # chroma | in_memory
    
    # Short-Term Memory & Local In-Memory Session Store (Default: Local, Last 5 Turns, Zero External Daemon)
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TTL_SECONDS: int = int(os.getenv("REDIS_TTL_SECONDS", "86400"))  # 24 hour TTL
    SHORT_TERM_MEMORY_WINDOW: int = int(os.getenv("SHORT_TERM_MEMORY_WINDOW", "5"))  # strict 5-turn sliding window
    
    # Data Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PROPERTIES_FILE: Path = DATA_DIR / "properties_for_buyers.csv"
    SELLER_COVERAGE_FILE: Path = DATA_DIR / "seller_coverage.csv"
    FAQ_FILE: Path = DATA_DIR / "faq.csv"
    
    # Standard Ending Messages (M1, M2, M3) as defined by business rules
    M1_MESSAGE: str = (
        "Sorry, we don't cater to new-build properties under £1 million. "
        "You can reach our head office on 020 7946 0100. Goodbye."
    )
    M2_MESSAGE: str = (
        "Sorry, we don't cater to the postcode you provided. "
        "You can reach our head office on 020 7946 0100. Goodbye."
    )
    M3_MESSAGE: str = (
        "Sorry, we have no available homes matching your requirements right now. "
        "You can reach our head office on 020 7946 0100. Goodbye."
    )


settings = Settings()
