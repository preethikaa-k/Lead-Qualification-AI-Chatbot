"""
Unified LLM Factory.
Provides a pluggable, provider-agnostic interface supporting:
- Ollama (Local CPU open-source: qwen2.5:3b, llama3.1:8b)
- Groq (Cloud open-source high-speed: llama-3.3-70b-versatile, llama-3.1-8b-instant)
- Google Gemini (gemini-2.0-flash, gemini-1.5-flash)
- OpenAI (gpt-4o-mini, gpt-4o)
Includes graceful error handling and fallback capabilities.
"""

import logging
from typing import Optional, List, Dict, Any
from src.config import settings

logger = logging.getLogger(__name__)


class UnifiedLLM:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.LLM_MODEL
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self._llm = None
        self._init_client()

    def _init_client(self):
        try:
            if self.provider == "groq":
                try:
                    from langchain_groq import ChatGroq
                    self._llm = ChatGroq(
                        model=self.model_name,
                        api_key=self.api_key,
                        temperature=0.2,
                    )
                except ImportError:
                    # Fallback to standard ChatOpenAI with Groq base_url
                    from langchain_openai import ChatOpenAI
                    self._llm = ChatOpenAI(
                        model=self.model_name,
                        api_key=self.api_key,
                        base_url="https://api.groq.com/openai/v1",
                        temperature=0.2,
                    )
            elif self.provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.2,
                )
            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.2,
                )
            elif self.provider == "ollama":
                try:
                    from langchain_community.chat_models import ChatOllama
                    self._llm = ChatOllama(
                        model=self.model_name,
                        base_url=self.base_url,
                        temperature=0.2,
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize ChatOllama: {e}")
                    self._llm = None
            else:
                logger.warning(f"Unknown provider '{self.provider}'. Running in rule-guided mode.")
                self._llm = None
        except Exception as e:
            logger.warning(f"Error initializing LLM client for provider '{self.provider}': {e}")
            self._llm = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates text completion. Gracefully handles network drops, rate limits,
        or offline status without crashing.
        """
        if self._llm is None:
            return ""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}:{self.model_name}): {e}")
            return ""


# Global singleton instance
llm_client = UnifiedLLM()
