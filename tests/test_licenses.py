"""Licence normalisation must fail closed. These tests encode that contract."""

from __future__ import annotations

import pytest

from fwdata.licenses import (
    PRODUCTION,
    PRODUCTION_SA,
    RESEARCH_NC,
    License,
    attribution_string,
    get_policy,
    normalize,
)


class TestNormalizeFailsClosed:
    @pytest.mark.parametrize("raw", ["", None, "   ", "none", "null"])
    def test_missing_licence_is_all_rights_reserved_not_public_domain(self, raw):
        # This is the single most dangerous mis-parse: on iNaturalist an empty
        # licence column means ARR. Treating it as permissive would put
        # copyrighted photographs into a shippable corpus.
        assert normalize(raw) is License.ARR

    @pytest.mark.parametrize(
        "raw", ["weird-thing", "ask the photographer", "CC-BY-MAYBE", "????", "free"]
    )
    def test_unparseable_is_unknown(self, raw):
        assert normalize(raw) is License.UNKNOWN

    def test_unknown_is_never_admitted_by_any_policy(self):
        for pol in (PRODUCTION, PRODUCTION_SA, RESEARCH_NC):
            assert not pol.admits(License.UNKNOWN)
            assert not pol.admits(License.ARR)


class TestNormalizeKnownForms:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("CC0", License.CC0),
            ("cc0", License.CC0),
            ("http://creativecommons.org/publicdomain/zero/1.0/", License.CC0),
            ("cc-by", License.CC_BY),
            ("CC BY 2.0", License.CC_BY),
            ("https://creativecommons.org/licenses/by/4.0/", License.CC_BY),
            ("cc-by-sa", License.CC_BY_SA),
            ("http://creativecommons.org/licenses/by-sa/3.0/", License.CC_BY_SA),
            ("cc-by-nc", License.CC_BY_NC),
            ("cc-by-nc-sa", License.CC_BY_NC_SA),
            ("cc-by-nc-nd", License.CC_BY_NC_ND),
            ("cc-by-nd", License.CC_BY_ND),
            ("Public Domain Mark 1.0", License.PD),
            ("C", License.ARR),
        ],
    )
    def test_round_trip(self, raw, expected):
        assert normalize(raw) is expected

    def test_https_and_http_uris_agree(self):
        a = normalize("http://creativecommons.org/licenses/by-nc/4.0/")
        b = normalize("https://creativecommons.org/licenses/by-nc/4.0/")
        assert a is b is License.CC_BY_NC


class TestCapabilityPredicates:
    def test_noncommercial_licences_are_not_commercial(self):
        for lic in (
            License.CC_BY_NC,
            License.CC_BY_NC_SA,
            License.CC_BY_NC_ND,
            License.ARR,
            License.UNKNOWN,
        ):
            assert not lic.allows_commercial

    def test_no_derivatives_licences_never_enter_a_training_corpus(self):
        # ND media is excluded from *every* policy, including research, because
        # training is at minimum arguably a derivative use.
        for lic in (License.CC_BY_ND, License.CC_BY_NC_ND):
            assert not lic.allows_derivatives
            for pol in (PRODUCTION, PRODUCTION_SA, RESEARCH_NC):
                assert not pol.admits(lic)

    def test_public_domain_needs_no_attribution(self):
        assert not License.CC0.requires_attribution
        assert not License.PD.requires_attribution
        assert License.CC_BY.requires_attribution
        assert License.CC_BY_SA.requires_attribution

    def test_copyleft_detection(self):
        assert License.CC_BY_SA.is_copyleft
        assert License.CC_BY_NC_SA.is_copyleft
        assert not License.CC_BY.is_copyleft


class TestPolicies:
    def test_production_is_cc0_pd_ccby_only(self):
        assert PRODUCTION.allowed == frozenset(
            {License.CC0, License.PD, License.CC_BY}
        )
        assert PRODUCTION.commercial_safe

    def test_production_excludes_sharealike_by_default(self):
        # ShareAlike is opt-in: it may impose obligations on derived weights.
        assert not PRODUCTION.admits(License.CC_BY_SA)
        assert PRODUCTION_SA.admits(License.CC_BY_SA)

    def test_research_corpus_is_flagged_not_commercial_safe(self):
        assert RESEARCH_NC.admits(License.CC_BY_NC)
        assert not RESEARCH_NC.commercial_safe

    def test_every_policy_admits_only_commercially_usable_media_when_flagged(self):
        for pol in (PRODUCTION, PRODUCTION_SA):
            for lic in pol.allowed:
                assert lic.allows_commercial, f"{pol.name} admits non-commercial {lic}"

    def test_sql_in_list_is_quoted_and_sorted(self):
        assert PRODUCTION.sql_in_list() == "('CC-BY-4.0', 'CC0-1.0', 'PUBLIC-DOMAIN')"

    def test_unknown_policy_name_is_fatal(self):
        with pytest.raises(SystemExit):
            get_policy("does-not-exist")


class TestAttribution:
    def test_cc0_phrasing(self):
        s = attribution_string("Jane Doe", License.CC0, "inaturalist")
        assert "no rights reserved" in s and "Jane Doe" in s

    def test_ccby_phrasing_names_the_licence(self):
        s = attribution_string("Jane Doe", License.CC_BY, "inaturalist")
        assert "some rights reserved" in s and "CC-BY" in s

    def test_missing_creator_does_not_produce_empty_attribution(self):
        s = attribution_string(None, License.CC_BY, "gbif")
        assert "unknown" in s


class TestEnumIdempotence:
    """Regression: License subclasses str, which broke round-tripping.

    ``ImageProvenance`` normalised its licence field whenever it looked like a
    ``str``. Because ``License`` *is* a ``str`` subclass that test passed for
    real enum members, and ``str(License.CC_BY)`` yields ``'License.CC_BY'`` on
    Python 3.11+, not ``'CC-BY-4.0'``. Every correctly-licensed image was
    therefore re-normalised to UNKNOWN and refused by the store.
    """

    def test_license_member_is_a_str_instance(self):
        # The property that caused the bug; asserted so the fix keeps making
        # sense if someone later changes the base class.
        assert isinstance(License.CC_BY, str)

    @pytest.mark.parametrize("lic", list(License))
    def test_normalize_is_idempotent_on_enum_members(self, lic):
        assert normalize(lic) is lic

    @pytest.mark.parametrize("lic", list(License))
    def test_normalize_is_idempotent_on_enum_values(self, lic):
        # Round-tripping through .value must also be stable, since the
        # provenance store persists the value string and reads it back.
        if lic in (License.UNKNOWN, License.ARR):
            return
        assert normalize(lic.value) is lic
