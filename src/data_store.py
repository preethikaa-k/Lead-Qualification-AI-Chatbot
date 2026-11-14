"""
Structured Data Store for Ashcombe & Rowe Estates.
Deterministic querying for properties_for_buyers.csv and seller_coverage.csv.
Guarantees zero hallucinations and exact business rule enforcement.
"""

import csv
import re
from typing import List, Optional, Set, Dict, Any
from pathlib import Path
from difflib import SequenceMatcher

from src.models import PropertyListing, ListingCategory, PropertyStatus
from src.config import settings
from src.parsers import parse_uk_postcode


class EstateDataStore:
    def __init__(self, properties_file: Optional[Path] = None, seller_coverage_file: Optional[Path] = None):
        self.properties_file = properties_file or settings.PROPERTIES_FILE
        self.seller_coverage_file = seller_coverage_file or settings.SELLER_COVERAGE_FILE
        
        self.seller_postcodes: Set[str] = set()
        self.properties: List[PropertyListing] = []
        self.properties_by_postcode: Dict[str, List[PropertyListing]] = {}
        
        self._load_seller_coverage()
        self._load_properties()

    def _normalize_postcode(self, pc: str) -> str:
        parsed = parse_uk_postcode(pc)
        if parsed:
            return parsed
        return re.sub(r"\s+", " ", pc.strip().upper())

    def _load_seller_coverage(self):
        if not self.seller_coverage_file.exists():
            return
        with open(self.seller_coverage_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_pc = row.get("postcode", "")
                norm_pc = self._normalize_postcode(raw_pc)
                if norm_pc:
                    self.seller_postcodes.add(norm_pc)

    def _load_properties(self):
        if not self.properties_file.exists():
            return
        with open(self.properties_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prop = PropertyListing(
                    property_id=row["property_id"],
                    property_name=row["property_name"],
                    address=row["address"],
                    town=row["town"],
                    postcode=self._normalize_postcode(row["postcode"]),
                    listing_category=row["listing_category"].strip().lower(),
                    property_type=row["property_type"],
                    bedrooms=int(row["bedrooms"]),
                    bathrooms=int(row["bathrooms"]),
                    price_gbp=int(row["price_gbp"]),
                    status=row["status"].strip().lower(),
                    description=row["description"],
                    branch=row["branch"],
                    listed_date=row["listed_date"],
                )
                self.properties.append(prop)
                
                # Group by normalized postcode
                pc = prop.postcode
                if pc not in self.properties_by_postcode:
                    self.properties_by_postcode[pc] = []
                self.properties_by_postcode[pc].append(prop)

    # ---------------- Seller Rules (Lane D) ----------------
    def is_seller_postcode_covered(self, raw_postcode: str) -> bool:
        """
        Lane D2 check: Does the visitor's postcode appear in seller_coverage.csv?
        """
        norm = self._normalize_postcode(raw_postcode)
        return norm in self.seller_postcodes

    # ---------------- Buyer Rules (Lane A & Lane B) ----------------
    def get_listings_at_postcode(self, raw_postcode: str, category: Optional[str] = None) -> List[PropertyListing]:
        """
        Gets all listings (regardless of availability) at a postcode, optionally filtered by category.
        Used for:
          A4 check: New-builds listed at this postcode?
          B3 check: Resale homes listed at this postcode?
        """
        norm = self._normalize_postcode(raw_postcode)
        listings = self.properties_by_postcode.get(norm, [])
        if category:
            cat_clean = category.strip().lower()
            listings = [p for p in listings if p.listing_category == cat_clean]
        return listings

    def get_available_listings(
        self,
        raw_postcode: str,
        category: str,
        max_budget: Optional[int] = None
    ) -> List[PropertyListing]:
        """
        Checks for available listings.
        Business rule:
          Only listings with status 'available' are ever offered or shown.
          'under_offer' counts as not available.
        Used for:
          A5 check: Any available within budget? (status == 'available' and price_gbp <= budget)
          B4 check: Any resale available now? (status == 'available')
        """
        listings = self.get_listings_at_postcode(raw_postcode, category=category)
        
        # Filter strictly for available
        available = [p for p in listings if p.status == PropertyStatus.AVAILABLE.value]
        
        if max_budget is not None and category == ListingCategory.NEW_BUILD.value:
            available = [p for p in available if p.price_gbp <= max_budget]
            
        return available

    # ---------------- Specific Property Rules (Lane C) ----------------
    def search_specific_property(self, query: str) -> List[PropertyListing]:
        """
        Searches properties by name or address.
        Used for Lane C1 & C2.
        Returns:
          [] if not found
          [prop] if single match
          [prop1, prop2, ...] if multiple matches
        """
        if not query or len(query.strip()) < 3:
            return []

        q_clean = query.strip().lower()
        exact_matches = []
        fuzzy_matches = []

        for p in self.properties:
            p_name_lower = p.property_name.lower()
            p_addr_lower = p.address.lower()

            # Exact or substring match in name or address
            if q_clean in p_name_lower or q_clean in p_addr_lower:
                exact_matches.append(p)
                continue

            # Fuzzy match on property name
            ratio = SequenceMatcher(None, q_clean, p_name_lower).ratio()
            if ratio >= 0.70:
                fuzzy_matches.append((ratio, p))

        if exact_matches:
            return exact_matches

        # Sort fuzzy matches by similarity descending
        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in fuzzy_matches[:5]]

    def find_property_question_answer(self, query: str) -> Optional[str]:
        """
        If a user asks a question about a specific property anywhere in the flow,
        looks up the property in properties_for_buyers.csv and returns factual info.
        """
        props = self.search_specific_property(query)
        if not props:
            return None

        p = props[0]
        return (
            f"{p.property_name} ({p.address}, {p.town} {p.postcode}): "
            f"A {p.bedrooms}-bedroom, {p.bathrooms}-bathroom {p.property_type.lower()} listed at £{p.price_gbp:,}. "
            f"Current status: {p.status.replace('_', ' ')}. "
            f"Description: {p.description}"
        )


# Global singleton instance
data_store = EstateDataStore()
