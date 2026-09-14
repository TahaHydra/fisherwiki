"""The curated safety table must fail loudly rather than silently.

A typo in a family name means a venom warning never reaches a user, which is
the failure mode with the worst consequence and the lowest chance of being
noticed. So the loader validates strictly and the expander reports anything
that matched nothing.
"""

from __future__ import annotations

import pytest
import yaml

from fwdata.safety import SafetyDataError, VALID_RANKS, VALID_SEVERITIES, expand, load


class TestRealTableIsValid:
    def test_ships_valid(self):
        sources, warnings = load()
        assert len(sources) >= 1
        assert len(warnings) >= 10

    def test_every_warning_cites_a_declared_source(self):
        sources, warnings = load()
        ids = {s.source_id for s in sources}
        for w in warnings:
            assert w.source_id in ids, f"{w.taxon} cites unknown {w.source_id}"

    def test_severities_and_ranks_are_in_vocabulary(self):
        _, warnings = load()
        for w in warnings:
            assert w.severity in VALID_SEVERITIES
            assert w.rank in VALID_RANKS

    def test_every_warning_has_a_non_empty_summary(self):
        _, warnings = load()
        for w in warnings:
            assert w.summary.strip(), w.taxon

    def test_the_genuinely_dangerous_groups_are_covered(self):
        # Not an exhaustive list - a spot check that the table has not been
        # gutted. These are the groups most likely to seriously hurt an angler.
        _, warnings = load()
        covered = {w.taxon for w in warnings}
        for taxon in ("Synanceiidae", "Scorpaenidae", "Trachinidae", "Dasyatidae"):
            assert taxon in covered, f"{taxon} has no warning"

    def test_dangerous_groups_are_marked_danger(self):
        _, warnings = load()
        by_taxon = {w.taxon: w for w in warnings}
        for taxon in ("Synanceiidae", "Scorpaenidae", "Trachinidae", "Dasyatidae"):
            assert by_taxon[taxon].severity == "danger", taxon


class TestLoaderRejectsBadData:
    def _write(self, tmp_path, doc):
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return p

    def _ok_source(self):
        return {
            "source_id": "s1", "title": "T", "license": "CC0-1.0",
            "retrieved_on": "2026-01-01", "citation": "C",
        }

    def test_unknown_source_is_rejected(self, tmp_path):
        doc = {
            "sources": [self._ok_source()],
            "warnings": [{
                "rank": "family", "taxon": "X", "kind": "k",
                "severity": "danger", "summary": "s", "source_id": "nope",
            }],
        }
        with pytest.raises(SafetyDataError, match="unknown source"):
            load(self._write(tmp_path, doc))

    def test_invalid_severity_is_rejected(self, tmp_path):
        doc = {
            "sources": [self._ok_source()],
            "warnings": [{
                "rank": "family", "taxon": "X", "kind": "k",
                "severity": "VERY_BAD", "summary": "s", "source_id": "s1",
            }],
        }
        with pytest.raises(SafetyDataError, match="invalid severity"):
            load(self._write(tmp_path, doc))

    def test_invalid_rank_is_rejected(self, tmp_path):
        doc = {
            "sources": [self._ok_source()],
            "warnings": [{
                "rank": "subphylum", "taxon": "X", "kind": "k",
                "severity": "info", "summary": "s", "source_id": "s1",
            }],
        }
        with pytest.raises(SafetyDataError, match="invalid rank"):
            load(self._write(tmp_path, doc))

    def test_missing_summary_is_rejected(self, tmp_path):
        doc = {
            "sources": [self._ok_source()],
            "warnings": [{
                "rank": "family", "taxon": "X", "kind": "k",
                "severity": "info", "source_id": "s1",
            }],
        }
        with pytest.raises(SafetyDataError, match="summary"):
            load(self._write(tmp_path, doc))

    def test_duplicate_warning_is_rejected(self, tmp_path):
        w = {
            "rank": "family", "taxon": "X", "kind": "k",
            "severity": "info", "summary": "s", "source_id": "s1",
        }
        doc = {"sources": [self._ok_source()], "warnings": [w, dict(w)]}
        with pytest.raises(SafetyDataError, match="duplicate warning"):
            load(self._write(tmp_path, doc))

    def test_empty_table_is_rejected(self, tmp_path):
        doc = {"sources": [self._ok_source()], "warnings": []}
        with pytest.raises(SafetyDataError, match="no warnings"):
            load(self._write(tmp_path, doc))


class TestExpansion:
    def _taxa(self):
        return [
            {"fw_taxon_id": 1, "scientific_name": "Scorpaena porcus",
             "genus": "Scorpaena", "family": "Scorpaenidae",
             "order": "Scorpaeniformes", "class": "Actinopterygii"},
            {"fw_taxon_id": 2, "scientific_name": "Perca fluviatilis",
             "genus": "Perca", "family": "Percidae",
             "order": "Perciformes", "class": "Actinopterygii"},
            {"fw_taxon_id": 3, "scientific_name": "Dasyatis pastinaca",
             "genus": "Dasyatis", "family": "Dasyatidae",
             "order": "Myliobatiformes", "class": "Chondrichthyes"},
        ]

    def test_family_warning_reaches_its_species(self):
        _, warnings = load()
        rows, _ = expand(warnings, self._taxa(), log=lambda m: None)
        scorpion = [r for r in rows if r["fw_taxon_id"] == 1]
        assert any(r["kind"] == "venomous_spines" for r in scorpion)
        assert all(r["applies_to"] == "family" for r in scorpion
                   if r["kind"] == "venomous_spines")

    def test_a_species_with_no_applicable_warning_gets_none(self):
        # Absence must be absence, not a fabricated reassurance.
        _, warnings = load()
        rows, _ = expand(warnings, self._taxa(), log=lambda m: None)
        assert not [r for r in rows if r["fw_taxon_id"] == 2]

    def test_class_level_warning_reaches_a_ray(self):
        _, warnings = load()
        rows, _ = expand(warnings, self._taxa(), log=lambda m: None)
        ray = [r for r in rows if r["fw_taxon_id"] == 3]
        kinds = {r["kind"] for r in ray}
        assert "venomous_spines" in kinds     # Dasyatidae
        assert "handling" in kinds            # Chondrichthyes

    def test_every_generated_row_carries_a_source(self):
        _, warnings = load()
        rows, _ = expand(warnings, self._taxa(), log=lambda m: None)
        assert rows
        for r in rows:
            assert r["source_id"]

    def test_unmatched_warnings_are_reported_not_swallowed(self):
        # A family name that matches nothing is almost always a typo, and a typo
        # means a venom warning silently never appears.
        _, warnings = load()
        _, report = expand(warnings, self._taxa(), log=lambda m: None)
        assert "unmatched_warnings" in report
        # Most warnings will not match this three-species fixture.
        assert len(report["unmatched_warnings"]) > 0
