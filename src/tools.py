"""
LangChain Tools for Ashcombe & Rowe Estates AI Lead Qualification Agent.
Provides tool definitions for:
1. FAQ Semantic RAG Lookup (faq.csv via VectorStore)
2. Property Buyer Listing Lookup (properties_for_buyers.csv)
3. Specific Property Name/Address Search (properties_for_buyers.csv)
4. Seller Coverage Postcode Check (seller_coverage.csv)
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from src.data_store import data_store
from src.rag_engine import faq_engine
from src.config import settings


@tool
def search_faq_tool(question: str) -> str:
    """
    Search the Ashcombe & Rowe Estates knowledge base for company background,
    branch addresses, opening hours, fees, policies, and buying/selling process.
    Use this whenever a visitor asks general or company questions.
    """
    faq_item = faq_engine.search_faq(question)
    if faq_item:
        return f"[FAQ Answer]: {faq_item.answer}"
    return (
        f"I'm sorry, I don't have that information in our records. "
        f"Please contact our head office on {settings.HEAD_OFFICE_PHONE}."
    )


@tool
def check_seller_postcode_coverage(postcode: str) -> str:
    """
    Check if a seller's UK postcode is covered by Ashcombe & Rowe Estates.
    Returns whether the agency accepts homes for sale in this area.
    """
    covered = data_store.is_seller_postcode_covered(postcode)
    if covered:
        return f"Postcode '{postcode}' is COVERED. Ashcombe & Rowe accepts property listings here."
    return f"Postcode '{postcode}' is NOT covered. We only serve designated luxury postcodes."


@tool
def lookup_buyer_listings(postcode: str, category: str, max_budget: Optional[int] = None) -> str:
    """
    Look up available buyer properties by UK postcode and category ('new_build' or 'resale').
    For new-builds, a max_budget can be provided to check availability within budget.
    """
    category_clean = category.strip().lower()
    all_at_pc = data_store.get_listings_at_postcode(postcode, category=category_clean)
    if not all_at_pc:
        return f"No {category_clean} listings exist at postcode '{postcode}'."

    available = data_store.get_available_listings(postcode, category=category_clean, max_budget=max_budget)
    if not available:
        return f"{category_clean} properties exist at '{postcode}', but none are currently available within budget."

    results = []
    for p in available[:5]:
        results.append(f"• {p.property_name} - {p.bedrooms} bed {p.property_type.lower()} at £{p.price_gbp:,}")

    return f"Found {len(available)} available {category_clean} listings at '{postcode}':\n" + "\n".join(results)


@tool
def search_specific_property_tool(property_name_or_address: str) -> str:
    """
    Look up a specific property by its name or street address in the agency's portfolio.
    Returns listing details, status ('available', 'under_offer', or 'sold'), and price.
    """
    matches = data_store.search_specific_property(property_name_or_address)
    if not matches:
        return f"No property matching '{property_name_or_address}' was found."

    if len(matches) > 1:
        names = ", ".join(f"'{p.property_name}'" for p in matches[:4])
        return f"Multiple matching properties found: {names}. Please specify which one."

    p = matches[0]
    return (
        f"Property: {p.property_name} ({p.address}, {p.town} {p.postcode})\n"
        f"Category: {p.listing_category}\n"
        f"Type: {p.property_type}, {p.bedrooms} beds, {p.bathrooms} baths\n"
        f"Price: £{p.price_gbp:,}\n"
        f"Status: {p.status}\n"
        f"Description: {p.description}"
    )


ALL_ESTATE_TOOLS = [
    search_faq_tool,
    check_seller_postcode_coverage,
    lookup_buyer_listings,
    search_specific_property_tool,
]
