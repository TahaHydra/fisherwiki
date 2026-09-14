"""Regulation rule packs must fail loudly and must never infer.

Regulations are legally consequential and change frequently. A rule that cannot
be applied, or one attributed to a jurisdiction it did not come from, is worse
than no rule at all.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
import yaml

from fwdata.regulations import (
    VALID_RULE_KINDS,
    RegulationDataError,
    Rule,
    RuleSet,
    available_packs,
    load,
)


def _source():
    return {
        "source_id": "s1", "title": "Byelaws", "authority": "Agency",
        "url": "https://example.gov/byelaws", "retrieved_on": "2026-01-01",
        "licence": "OGL-3.0",
    }


def _pack(tmp_path, rules, **overrides):
    doc = {
        "jurisdiction": "GB-ENG", "version": 1, "retrieved_on": "2026-01-01",
        "sources": [_source()], "rules": rules,
    }
    doc.update(overrides)
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def _rule(**over):
    r = {
        "jurisdiction": "GB-ENG", "kind": "minimum_size",
        "taxon": "Esox lucius", "taxon_rank": "species",
        "value_num": 40.0, "unit": "cm",
        "text": "Minimum size 40 cm.", "valid_from": "2026-01-01",
        "source_id": "s1",
    }
    r.update(over)
    return r


class TestLoaderRejectsBadData:
    def test_loads_a_valid_pack(self, tmp_path):
        rs = load(_pack(tmp_path, [_rule()]))
        assert rs.jurisdiction == "GB-ENG"
        assert len(rs.rules) == 1
        assert rs.rules[0].value_num == 40.0

    def test_unknown_source_is_rejected(self, tmp_path):
        with pytest.raises(RegulationDataError, match="unknown source"):
            load(_pack(tmp_path, [_rule(source_id="nope")]))

    def test_unknown_kind_is_rejected(self, tmp_path):
        with pytest.raises(RegulationDataError, match="unknown rule kind"):
            load(_pack(tmp_path, [_rule(kind="vibes")]))

    def test_size_limit_without_a_number_is_rejected(self, tmp_path):
        # A size limit with no size cannot be applied, so it is not a rule.
        with pytest.raises(RegulationDataError, match="no value_num"):
            load(_pack(tmp_path, [_rule(value_num=None)]))

    def test_bag_limit_without_a_number_is_rejected(self, tmp_path):
        with pytest.raises(RegulationDataError, match="no value_num"):
            load(_pack(tmp_path, [_rule(kind="bag_limit", value_num=None)]))

    def test_rule_without_valid_from_is_rejected(self, tmp_path):
        r = _rule()
        del r["valid_from"]
        with pytest.raises(RegulationDataError, match="valid_from"):
            load(_pack(tmp_path, [r]))

    def test_pack_without_a_source_is_rejected(self, tmp_path):
        p = tmp_path / "j.yaml"
        p.write_text(yaml.safe_dump({
            "jurisdiction": "GB-ENG", "version": 1,
            "retrieved_on": "2026-01-01", "sources": [], "rules": [],
        }), encoding="utf-8")
        with pytest.raises(RegulationDataError, match="at least one source"):
            load(p)

    def test_source_without_a_url_is_rejected(self, tmp_path):
        s = _source()
        del s["url"]
        p = tmp_path / "j.yaml"
        p.write_text(yaml.safe_dump({
            "jurisdiction": "GB-ENG", "version": 1,
            "retrieved_on": "2026-01-01", "sources": [s], "rules": [],
        }), encoding="utf-8")
        with pytest.raises(RegulationDataError, match="url"):
            load(p)

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(RegulationDataError):
            load(tmp_path / "nope.yaml")


class TestValidity:
    def test_an_expired_rule_is_not_currently_valid(self):
        r = Rule("GB-ENG", "minimum_size", "Esox lucius", "species", 40.0, "cm",
                 "x", "2020-01-01", "2021-12-31", "s1")
        assert not r.is_currently_valid

    def test_a_future_rule_is_not_yet_valid(self):
        future = (date.today() + timedelta(days=400)).isoformat()
        r = Rule("GB-ENG", "minimum_size", "Esox lucius", "species", 40.0, "cm",
                 "x", future, None, "s1")
        assert not r.is_currently_valid

    def test_an_open_ended_current_rule_is_valid(self):
        r = Rule("GB-ENG", "minimum_size", "Esox lucius", "species", 40.0, "cm",
                 "x", "2020-01-01", None, "s1")
        assert r.is_currently_valid


class TestStaleness:
    def test_staleness_is_reported_in_days(self):
        rs = RuleSet("GB-ENG", 1, (date.today() - timedelta(days=42)).isoformat())
        assert rs.staleness() == 42

    def test_old_data_is_flagged_stale(self):
        rs = RuleSet("GB-ENG", 1, (date.today() - timedelta(days=400)).isoformat())
        assert rs.is_stale()

    def test_recent_data_is_not_stale(self):
        rs = RuleSet("GB-ENG", 1, (date.today() - timedelta(days=10)).isoformat())
        assert not rs.is_stale()

    def test_an_unparseable_date_is_treated_as_maximally_stale(self):
        # Failing safe: if we cannot tell how old it is, assume the worst.
        rs = RuleSet("GB-ENG", 1, "not-a-date")
        assert rs.is_stale()


class TestLookup:
    def _rs(self):
        return RuleSet("GB-ENG", 1, date.today().isoformat(), rules=[
            Rule("GB-ENG", "minimum_size", "Esox lucius", "species", 40.0, "cm",
                 "species rule", "2020-01-01", None, "s1"),
            Rule("GB-ENG", "bag_limit", "Esocidae", "family", 2.0, "count",
                 "family rule", "2020-01-01", None, "s1"),
            Rule("GB-ENG", "licence_required", None, "species", None, "none",
                 "everyone rule", "2020-01-01", None, "s1"),
        ])

    def test_most_specific_rule_comes_first(self):
        rules = self._rs().for_taxon("Esox lucius", family="Esocidae")
        assert [r.text for r in rules] == [
            "species rule", "family rule", "everyone rule"
        ]

    def test_general_rules_apply_to_any_species(self):
        rules = self._rs().for_taxon("Perca fluviatilis")
        assert [r.text for r in rules] == ["everyone rule"]

    def test_no_data_means_no_rules_not_a_neighbours_rules(self):
        # The critical property: an uncovered jurisdiction returns nothing.
        # Offering a neighbouring county's limits would be worse than silence.
        empty = RuleSet("US-FL", 1, date.today().isoformat(), rules=[])
        assert empty.for_taxon("Micropterus salmoides") == []

    def test_expired_rules_are_excluded_from_lookup(self):
        rs = RuleSet("GB-ENG", 1, date.today().isoformat(), rules=[
            Rule("GB-ENG", "minimum_size", "Esox lucius", "species", 40.0, "cm",
                 "expired", "2019-01-01", "2020-12-31", "s1"),
        ])
        assert rs.for_taxon("Esox lucius") == []


class TestShippedState:
    def test_no_jurisdiction_data_ships_yet(self):
        # Deliberate. See data/regulations/README.md - a rule pack nobody
        # maintains is worse than none, because the app would confidently show
        # a limit that changed last season.
        assert available_packs() == []

    def test_the_template_is_not_loaded_as_a_pack(self):
        # Files starting with '_' are templates, not data.
        assert all(not p.name.startswith("_") for p in available_packs())

    def test_rule_kinds_cover_the_common_cases(self):
        for kind in ("minimum_size", "bag_limit", "closed_season",
                     "licence_required", "protected"):
            assert kind in VALID_RULE_KINDS
