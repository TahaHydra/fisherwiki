"""Scientific-name parsing: the cases that actually occur in GBIF/iNat data."""

from __future__ import annotations

import pytest

from fwdata.taxonomy.names import (
    canonical_form,
    is_binomial,
    normalize_name,
    parse_scientific_name,
)


class TestAuthorshipStripping:
    @pytest.mark.parametrize(
        "raw",
        [
            "Perca fluviatilis",
            "Perca fluviatilis Linnaeus, 1758",
            "Perca fluviatilis L., 1758",
            "PERCA FLUVIATILIS",
            "  perca   Fluviatilis  ",
            "Perca (Perca) fluviatilis",
        ],
    )
    def test_all_spellings_collapse_to_one_key(self, raw):
        assert canonical_form(raw) == "Perca fluviatilis"

    def test_parenthesised_authorship_is_captured_not_dropped_silently(self):
        p = parse_scientific_name("Salmo trutta (Linnaeus, 1758)")
        assert p.canonical == "Salmo trutta"
        assert "Linnaeus" in p.authorship

    def test_multi_author_with_ampersand(self):
        p = parse_scientific_name("Acipenser gueldenstaedtii Brandt & Ratzeburg, 1833")
        assert p.canonical == "Acipenser gueldenstaedtii"
        assert "Brandt" in p.authorship


class TestHybridsAndUncertainty:
    def test_hybrid_marker_x(self):
        p = parse_scientific_name("Salmo x trutta")
        assert p.hybrid and p.canonical == "Salmo trutta"

    def test_hybrid_multiplication_sign(self):
        p = parse_scientific_name("Salmo trutta × salar")
        assert p.hybrid

    def test_open_nomenclature_sp_reduces_to_genus(self):
        p = parse_scientific_name("Sebastes sp.")
        assert p.uncertain
        assert p.canonical == "Sebastes"
        assert p.rank == "genus"

    def test_cf_is_flagged_but_epithet_kept(self):
        p = parse_scientific_name("Sebastes cf. norvegicus")
        assert p.uncertain and p.canonical == "Sebastes norvegicus"

    def test_trade_code_is_dropped_not_mangled(self):
        # Regression: "Hypostomus sp. L001" used to yield "Hypostomus l"
        # because digit-stripping left a one-letter pseudo-epithet.
        p = parse_scientific_name("Hypostomus sp. L001")
        assert p.canonical == "Hypostomus"
        assert p.uncertain
        assert p.rank == "genus"

    def test_numeric_placeholder_epithet_dropped(self):
        assert canonical_form("Corydoras sp. C121") == "Corydoras"


class TestInfraspecific:
    @pytest.mark.parametrize(
        "raw,expected_rank",
        [
            ("Salvelinus alpinus subsp. erythrinus", "subspecies"),
            ("Coregonus lavaretus var. maraena", "variety"),
            ("Gasterosteus aculeatus aculeatus", "subspecies"),
        ],
    )
    def test_rank_inference(self, raw, expected_rank):
        assert parse_scientific_name(raw).rank == expected_rank

    def test_binomial_drops_infraspecific_part(self):
        p = parse_scientific_name("Salvelinus alpinus subsp. erythrinus")
        assert p.binomial == "Salvelinus alpinus"
        assert p.canonical == "Salvelinus alpinus erythrinus"


class TestUnicodeAndWhitespace:
    def test_nbsp_and_curly_quotes(self):
        assert normalize_name("Perca fluviatilis") == "Perca fluviatilis"

    def test_diacritics_in_authorship_do_not_break_parsing(self):
        assert canonical_form("Gobio gobio Bonaparte, 1846") == "Gobio gobio"


class TestDegenerateInput:
    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_input_is_empty_not_an_exception(self, raw):
        assert canonical_form(raw or "") == ""

    def test_genus_only(self):
        p = parse_scientific_name("Esox")
        assert p.rank == "genus" and p.canonical == "Esox"
        assert not is_binomial("Esox")

    def test_is_binomial(self):
        assert is_binomial("Esox lucius")
