"""
Unit tests for EstateDataStore.
Verifies property lookups, availability constraints, and seller coverage logic.
"""

import pytest
from src.data_store import data_store
from src.models import PropertyStatus, ListingCategory


class TestDataStore:
    def test_seller_postcode_coverage(self):
        # Known covered postcodes from seller_coverage.csv
        assert data_store.is_seller_postcode_covered("GU1 3AD") is True
        assert data_store.is_seller_postcode_covered("gu1 3ad") is True
        assert data_store.is_seller_postcode_covered("SW3 3RB") is True
        assert data_store.is_seller_postcode_covered("KT11 4GR") is True
        
        # Uncovered postcode
        assert data_store.is_seller_postcode_covered("E1 6AN") is False
        assert data_store.is_seller_postcode_covered("B1 1AA") is False

    def test_only_available_properties_returned(self):
        # SW3 3RB has available properties and 1 under_offer (AR-0005)
        all_props = data_store.get_listings_at_postcode("SW3 3RB", category=ListingCategory.NEW_BUILD.value)
        assert len(all_props) == 5

        available = data_store.get_available_listings("SW3 3RB", category=ListingCategory.NEW_BUILD.value)
        assert len(available) == 4
        # Verify no under_offer or sold properties in available list
        for p in available:
            assert p.status == PropertyStatus.AVAILABLE.value

    def test_budget_filter_for_new_builds(self):
        # SW3 3RB prices: 1,230,000; 1,330,000; 1,615,000; 2,075,000
        # Budget = 1,400,000 should return only 2 listings
        available = data_store.get_available_listings(
            "SW3 3RB", category=ListingCategory.NEW_BUILD.value, max_budget=1_400_000
        )
        assert len(available) == 2
        for p in available:
            assert p.price_gbp <= 1_400_000

    def test_resale_listings_lookup(self):
        # SW3 5WU has resale properties (Rosemount House, 89 Bramerton Street, etc.)
        resales = data_store.get_available_listings("SW3 5WU", category=ListingCategory.RESALE.value)
        assert len(resales) > 0
        for p in resales:
            assert p.listing_category == ListingCategory.RESALE.value
            assert p.status == PropertyStatus.AVAILABLE.value

    def test_specific_property_search(self):
        # Single exact match
        props = data_store.search_specific_property("Apartment 1, Aldermoor Place")
        assert len(props) == 1
        assert props[0].property_name == "Apartment 1, Aldermoor Place"
        assert props[0].status == PropertyStatus.AVAILABLE.value

        # Under offer property
        sold_props = data_store.search_specific_property("Townhouse 2, Riverside Quarter")
        assert len(sold_props) == 1
        assert sold_props[0].status == PropertyStatus.SOLD.value

        # Non-existent property
        none_props = data_store.search_specific_property("Buckingham Palace Flat 4")
        assert len(none_props) == 0
