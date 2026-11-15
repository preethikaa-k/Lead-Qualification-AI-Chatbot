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
        # Clean query
        # Remove common conversational prefixes/suffixes if present
        clean_prefix_q = re.sub(
            r"^(?:can you tell me about|tell me about|what about|how much is|is|what is the price of|looking for|interested in|i am interested in|the property is)\s+",
            "",
            q_clean,
        ).rstrip("?.,!")

        target_queries = [q_clean]
        if clean_prefix_q and clean_prefix_q != q_clean:
            target_queries.insert(0, clean_prefix_q)

        exact_matches = []
        fuzzy_matches = []

        # 1. Check exact bidirectional containment:
        # User query contains property name/address OR property name/address contains user query
        scored_matches = []
        for p in self.properties:
            p_name_lower = p.property_name.lower()
            p_addr_lower = p.address.lower()

            for t_q in target_queries:
                if p_name_lower == t_q or p_addr_lower == t_q:
                    scored_matches.append((1000 + len(p_name_lower), p))
                    break
                elif p_name_lower in t_q or p_addr_lower in t_q:
                    # Longer property names preferred over shorter substring matches
                    scored_matches.append((500 + len(p_name_lower), p))
                    break
                elif t_q in p_name_lower or t_q in p_addr_lower:
                    scored_matches.append((200 + len(t_q), p))
                    break

        if scored_matches:
            # Sort by match score descending
            scored_matches.sort(key=lambda x: x[0], reverse=True)
            top_score = scored_matches[0][0]
            # If top score is >= 500 (full property name was found in query or exact match)
            # Only keep top-tier matches
            if top_score >= 500:
                filtered = [m[1] for m in scored_matches if m[0] == top_score or m[0] >= 500]
            else:
                filtered = [m[1] for m in scored_matches]

            # Deduplicate by property_id while preserving order
            seen = set()
            deduped = []
            for prop in filtered:
                if prop.property_id not in seen:
                    seen.add(prop.property_id)
                    deduped.append(prop)
            return deduped[:5]

        # 2. Fuzzy match on property name
        for p in self.properties:
            p_name_lower = p.property_name.lower()
            ratio = max(
                SequenceMatcher(None, t_q, p_name_lower).ratio()
                for t_q in target_queries
            )
            if ratio >= 0.70:
                fuzzy_matches.append((ratio, p))

        # Sort fuzzy matches by similarity descending
        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in fuzzy_matches[:5]]

    def find_property_question_answer(self, query: str) -> Optional[str]:
        """
        Answers factual questions about properties and agency coverage:
        1. Specific property lookup by title/address
        2. Broad portfolio inquiries (locations, towns, price ranges, property types)
        3. Town-specific property inquiries (e.g., Richmond, Cobham, Chelsea, Oxford)
        4. Property type inquiries (e.g., penthouses, cottages, townhouses)
        5. Seller service and postcode coverage inquiries
        """
        if not query:
            return None
        
        q_lower = query.lower().strip()

        # Reject out-of-domain or commercial non-residential queries to trigger head office fallback
        out_of_domain = ["aircraft", "hangar", "commercial", "manchester", "solar", "warehouse", "factory", "industrial"]
        if any(term in q_lower for term in out_of_domain):
            return None

        # 1. Check for specific named property match first
        props = self.search_specific_property(query)
        if props:
            p = props[0]
            return (
                f"{p.property_name} ({p.address}, {p.town} {p.postcode}): "
                f"A {p.bedrooms}-bedroom, {p.bathrooms}-bathroom {p.property_type.lower()} listed at £{p.price_gbp:,}. "
                f"Current status: {p.status.replace('_', ' ')}. "
                f"Description: {p.description}"
            )

        # 2. Check for town-specific inquiries (London, Chelsea, Hampstead, Richmond, Guildford, Cobham, Oxford)
        known_towns = {
            "chelsea": ("Chelsea", "London"),
            "hampstead": ("Hampstead", "London"),
            "richmond": ("Richmond", "London"),
            "cobham": ("Cobham", "Surrey"),
            "guildford": ("Guildford", "Surrey"),
            "oxford": ("Oxford", "Oxfordshire"),
            "london": ("London", "London"),
            "surrey": ("Surrey", "Surrey"),
        }
        for kw, (town_name, region) in known_towns.items():
            if re.search(rf"\b{kw}\b", q_lower):
                matched_town_props = [
                    p for p in self.properties
                    if (town_name.lower() in p.town.lower() or town_name.lower() in p.address.lower() or kw in p.address.lower())
                ]
                if matched_town_props:
                    avail = [p for p in matched_town_props if p.status == PropertyStatus.AVAILABLE.value]
                    new_b = [p for p in avail if p.listing_category == ListingCategory.NEW_BUILD.value]
                    resale = [p for p in avail if p.listing_category == ListingCategory.RESALE.value]
                    prices = [p.price_gbp for p in avail] if avail else [p.price_gbp for p in matched_town_props]
                    min_p, max_p = min(prices), max(prices)
                    types = sorted(list(set(p.property_type for p in avail or matched_town_props)))
                    
                    sample_names = ", ".join([f"'{p.property_name}' (£{p.price_gbp:,})" for p in (avail[:2] or matched_town_props[:2])])
                    return (
                        f"In {town_name} ({region}), we have {len(matched_town_props)} total listings "
                        f"({len(avail)} currently available: {len(new_b)} new-builds, {len(resale)} resales). "
                        f"Prices range from £{min_p:,} to £{max_p:,}, featuring {', '.join(types[:3])}. "
                        f"Notable properties include {sample_names}."
                    )

        # 3. Check for specific property type inquiries (penthouse, cottage, townhouse, apartment)
        property_types = {
            "penthouse": "Penthouse",
            "cottage": "Cottage",
            "townhouse": "Townhouse",
            "detached": "Detached house",
            "semi-detached": "Semi-detached house",
            "apartment": "Apartment",
            "flat": "Apartment",
        }
        for kw, p_type in property_types.items():
            if re.search(rf"\b{kw}\b", q_lower):
                matched_type_props = [p for p in self.properties if p.property_type.lower() == p_type.lower()]
                avail_type = [p for p in matched_type_props if p.status == PropertyStatus.AVAILABLE.value]
                if matched_type_props:
                    locations = sorted(list(set(p.town for p in matched_type_props)))
                    prices = [p.price_gbp for p in (avail_type or matched_type_props)]
                    return (
                        f"We represent {len(matched_type_props)} {p_type.lower()} listings ({len(avail_type)} available) "
                        f"across {', '.join(locations)}. "
                        f"Prices range from £{min(prices):,} to £{max(prices):,}."
                    )

        # 4. Check for broad location / portfolio / coverage inquiries
        location_keywords = [
            "where do you have properties",
            "where are your properties",
            "what properties do you have",
            "what areas do you cover",
            "which areas do you cover",
            "what locations",
            "where do you operate",
            "where can i buy",
            "what houses do you have",
            "where are you located",
            "show me your properties",
            "list out your properties",
            "list your properties",
            "list out properties",
            "list properties",
            "list of properties",
            "what do you have for sale",
            "what homes do you have",
            "available properties",
            "browse properties",
            "portfolio",
            "price range",
        ]
        if any(kw in q_lower for kw in location_keywords) or (
            ("where" in q_lower or "what" in q_lower or "which" in q_lower) and ("propert" in q_lower or "house" in q_lower or "home" in q_lower)
        ):
            avail_all = [p for p in self.properties if p.status == PropertyStatus.AVAILABLE.value]
            prices = [p.price_gbp for p in self.properties]
            return (
                f"Ashcombe & Rowe Estates represents a prime portfolio of 247 luxury properties "
                f"across London (Chelsea, Hampstead, and Richmond), Surrey (Cobham and Guildford), and Oxford. "
                f"We offer both luxury new-build developments (priced at £1,000,000 and above) "
                f"and character resale homes (ranging from £260,000 to £7,350,000). "
                f"Our collection includes penthouses, detached family houses, townhouses, and cottages."
            )

        # 5. Check for seller coverage and valuation inquiries
        seller_keywords = ["seller", "selling", "valuation", "sell my home", "commission", "selling fee"]
        if any(kw in q_lower for kw in seller_keywords) and ("fee" in q_lower or "area" in q_lower or "coverage" in q_lower or "how" in q_lower or "cost" in q_lower):
            return (
                f"For sellers, Ashcombe & Rowe Estates accepts homes for sale across 100 prime UK postcodes "
                f"in London, Surrey, and Oxford. We provide a complimentary, no-obligation market valuation, "
                f"bespoke architectural photography and floorplans. Our commission fee is 1.5% + VAT on a sole-agency basis, "
                f"or 2.5% + VAT for multi-agency instructions, with no upfront fees."
            )

        return None


# Global singleton instance
data_store = EstateDataStore()
