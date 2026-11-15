"""
Tests for Multi-Provider LLM Failover and Tool Calling.
Verifies:
1. Primary LLM initialization (Ollama Qwen 2.5)
2. Fallback LLM initialization (Google Gemini 2.0 Flash)
3. Automatic failover execution when primary encounters connection failure / timeout
4. Safe graceful degradation when both providers are unavailable
5. LangChain Tool integration (FAQ RAG, Seller Coverage, Buyer Listings, Specific Property)
"""

import pytest
from unittest.mock import MagicMock, patch
from src.llm_factory import UnifiedLLM
from src.tools import (
    search_faq_tool,
    check_seller_postcode_coverage,
    lookup_buyer_listings,
    search_specific_property_tool,
)
from src.config import settings


class TestLLMFailoverAndTools:
    def test_llm_factory_initialization_status(self):
        llm = UnifiedLLM()
        status = llm.get_provider_status()
        
        assert "primary" in status
        assert "fallback" in status
        assert status["primary"]["provider"] == settings.LLM_PROVIDER
        assert status["fallback"]["provider"] == settings.FALLBACK_LLM_PROVIDER
        assert status["failover_ready"] is True

    def test_automatic_failover_when_primary_fails(self):
        """
        Simulates primary Ollama Qwen connection drop.
        Verifies that UnifiedLLM seamlessly calls secondary Gemini client.
        """
        llm = UnifiedLLM()
        
        # Mock primary client to throw ConnectionError (e.g. Ollama daemon down)
        mock_primary = MagicMock()
        mock_primary.invoke.side_effect = ConnectionError("Could not connect to Ollama daemon at localhost:11434")
        
        # Mock fallback client to succeed (Gemini)
        mock_fallback = MagicMock()
        mock_fallback_resp = MagicMock()
        mock_fallback_resp.content = "Gemini fallback response text"
        mock_fallback.invoke.return_value = mock_fallback_resp
        
        llm._primary_client = mock_primary
        llm._fallback_client = mock_fallback
        
        result = llm.generate("Test prompt")
        
        # Assert primary was attempted
        assert mock_primary.invoke.called
        # Assert fallback was triggered and succeeded
        assert mock_fallback.invoke.called
        assert result == "Gemini fallback response text"

    def test_graceful_degradation_when_both_fail(self):
        """
        Simulates catastrophic outage where both local Ollama and cloud Gemini fail.
        Verifies zero crash and safe empty string return.
        """
        llm = UnifiedLLM()
        
        mock_primary = MagicMock()
        mock_primary.invoke.side_effect = ConnectionError("Ollama offline")
        
        mock_fallback = MagicMock()
        mock_fallback.invoke.side_effect = TimeoutError("Gemini timed out")
        
        llm._primary_client = mock_primary
        llm._fallback_client = mock_fallback
        
        result = llm.generate("Test prompt")
        assert result == ""

    def test_search_faq_tool_valid(self):
        result = search_faq_tool.invoke({"question": "Are you open on Sundays in Chelsea?"})
        assert "Chelsea" in result or "Sunday" in result or "020 7946 0100" in result

    def test_search_faq_tool_out_of_domain(self):
        result = search_faq_tool.invoke({"question": "Do you sell spaceship rocket fuel?"})
        assert "020 7946 0100" in result

    def test_seller_postcode_tool(self):
        # KT11 4GR is in seller_coverage.csv
        covered_res = check_seller_postcode_coverage.invoke({"postcode": "KT11 4GR"})
        assert "COVERED" in covered_res

        # ZE1 0AA is not covered
        uncovered_res = check_seller_postcode_coverage.invoke({"postcode": "ZE1 0AA"})
        assert "NOT covered" in uncovered_res

    def test_buyer_listings_tool(self):
        # SW3 3RB has new-build listings
        res = lookup_buyer_listings.invoke({"postcode": "SW3 3RB", "category": "new_build", "max_budget": 2000000})
        assert "available" in res.lower()
        assert "SW3 3RB" in res

    def test_specific_property_tool(self):
        # Multi-match query
        multi_res = search_specific_property_tool.invoke({"property_name_or_address": "Aldermoor Place"})
        assert "Multiple matching properties" in multi_res

        # Single match query
        single_res = search_specific_property_tool.invoke({"property_name_or_address": "Apartment 1, Aldermoor Place"})
        assert "Apartment 1, Aldermoor Place" in single_res
        assert "£" in single_res
