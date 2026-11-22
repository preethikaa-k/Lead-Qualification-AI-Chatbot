"""
Unit tests for deterministic parsers (budgets, postcodes, emails, phones, names).
"""

import pytest
from src.parsers import (
    parse_budget,
    parse_uk_postcode,
    parse_email,
    parse_phone,
    parse_name,
    parse_yes_no,
    detect_mind_change,
)


class TestParsers:
    def test_parse_budget_millions(self):
        assert parse_budget("£1.5m") == 1_500_000
        assert parse_budget("1.5 million") == 1_500_000
        assert parse_budget("around 2m") == 2_000_000
        assert parse_budget("£1.2M") == 1_200_000
        assert parse_budget("1m") == 1_000_000

    def test_parse_budget_thousands(self):
        assert parse_budget("950k") == 950_000
        assert parse_budget("£850k") == 850_000
        assert parse_budget("500 thousand") == 500_000

    def test_parse_budget_raw_numbers(self):
        assert parse_budget("1,200,000") == 1_200_000
        assert parse_budget("£1,000,000") == 1_000_000
        assert parse_budget("750000") == 750_000

    def test_parse_budget_invalid(self):
        assert parse_budget("I don't know") is None
        assert parse_budget("") is None

    def test_parse_postcode_normalization(self):
        assert parse_uk_postcode("gu5 7aq") == "GU5 7AQ"
        assert parse_uk_postcode("GU57AQ") == "GU5 7AQ"
        assert parse_uk_postcode("my location is sw3 3rb please") == "SW3 3RB"
        assert parse_uk_postcode("kt11 4gr") == "KT11 4GR"
        assert parse_uk_postcode("ox2 5py") == "OX2 5PY"

    def test_parse_postcode_invalid(self):
        assert parse_uk_postcode("London") is None
        assert parse_uk_postcode("12345") is None

    def test_parse_email(self):
        assert parse_email("My email is john.doe@example.co.uk") == "john.doe@example.co.uk"
        assert parse_email("SARAH@EXAMPLE.COM") == "sarah@example.com"
        assert parse_email("invalid email without at") is None

    def test_parse_phone(self):
        assert parse_phone("07123 456789") is not None
        assert parse_phone("+44 7123 456789") is not None
        assert parse_phone("Call me on 020 7946 0100") is not None
        assert parse_phone("123") is None  # Too short

    def test_parse_name(self):
        assert parse_name("I'm Sarah Connor") == "Sarah Connor"
        assert parse_name("My name is David Rowe") == "David Rowe"
        assert parse_name("Call me Arthur") == "Arthur"
        assert parse_name("Helen") == "Helen"

    def test_parse_yes_no(self):
        assert parse_yes_no("yes") is True
        assert parse_yes_no("yeah definitely") is True
        assert parse_yes_no("no") is False
        assert parse_yes_no("nope that is all") is False

    def test_detect_mind_change(self):
        assert detect_mind_change("Actually, I want to sell instead") == "sell"
        assert detect_mind_change("I change my mind, I want to buy") == "buy"
        assert detect_mind_change("Just looking for Chelsea branch") is None
