"""Edge cases found auditing the V2 split system before initialising it.

Each class below pins a defect that was real, reproduced against the live
provenance store, and fixed. They are separated from ``test_splits_v2.py``
because that file states the design guarantees, while these state the specific
ways the first implementation failed to deliver them.

Every one of these is a leakage bug: the failure mode is the same photograph,
or the same photographer, appearing in both train and the sealed release
holdout, which no accuracy metric can reveal.
"""

from __future__ import annotations

import duckdb
import pytest

from fwdata.splits_v2 import DATASET_VERSION, V2SplitStore
from test_splits_v2 import (
    _add_rows,
    _assign,
    _assignments,
    _corpus,
    _insert,
    _make_store,
    _row,
)


class TestNearDuplicateSearchIsExact:
    """The banded index was not exact, and its own comment said why.

    It required one 16-bit band to match exactly. By pigeonhole that finds
    every pair only when the threshold is *below* the band count; the threshold
    is 6 and there are 4 bands. Pairs at distance 4-6 spread across all four
    bands were dropped silently. On the real store that was 451 of 2,960
    genuine same-taxon pairs - 15% of them.
    """

    def test_four_band_counterexample_is_found(self):
        from fwdata.dedupe import near_duplicate_pairs

        a = 0
        b = (1 << 0) | (1 << 16) | (1 << 32) | (1 << 48)
        assert bin(a ^ b).count("1") == 4
        assert not any(
            ((a >> (i * 16)) & 0xFFFF) == ((b >> (i * 16)) & 0xFFFF)
            for i in range(4)
        ), "fixture no longer exercises the no-shared-band case"

        found = near_duplicate_pairs(
            [("a", f"{a:016x}"), ("b", f"{b:016x}")], threshold=6
        )
        assert [(x, y) for x, y, _ in found] == [("a", "b")]

    @pytest.mark.parametrize("spread", ["across_bands", "adjacent_bits"])
    def test_every_distance_up_to_the_threshold_is_found(self, spread):
        from fwdata.dedupe import near_duplicate_pairs

        base = 0x0F1E2D3C4B5A6978
        for d in range(7):
            mask = 0
            for bit in range(d):
                # across_bands walks the four 16-bit bands round-robin, so the
                # differing bits are spread over every band - the shape the old
                # "one band must match exactly" index could not see. It must
                # stay inside 64 bits, hence the wrap.
                mask |= 1 << (
                    ((bit % 4) * 16 + bit // 4) if spread == "across_bands" else bit
                )
            other = base ^ mask
            found = near_duplicate_pairs(
                [("a", f"{base:016x}"), ("b", f"{other:016x}")], threshold=6
            )
            assert found, f"distance {d} ({spread}) was missed"
            assert found[0][2] == d

    def test_distance_above_the_threshold_is_excluded(self):
        from fwdata.dedupe import near_duplicate_pairs

        far = sum(1 << (bit * 8) for bit in range(7))
        assert bin(far).count("1") == 7
        assert near_duplicate_pairs(
            [("a", f"{0:016x}"), ("b", f"{far:016x}")], threshold=6
        ) == []

    @pytest.mark.parametrize("scale", ["brute_force", "indexed"])
    def test_matches_brute_force(self, scale):
        """Both code paths, including a degenerate hash repeated many times.

        The old implementation also skipped any bucket over ``max_bucket``
        outright, which is a false negative by construction - and degenerate
        all-zero hashes are exactly what fills such a bucket.
        """
        import random

        from fwdata.dedupe import BRUTE_FORCE_MAX, near_duplicate_pairs

        n = 400 if scale == "brute_force" else BRUTE_FORCE_MAX + 300
        rng = random.Random(n)
        items = [(f"k{i}", f"{rng.getrandbits(64):016x}") for i in range(n)]
        items += [(f"zero{i}", "0000000000000000") for i in range(40)]

        hashes = {k: int(v, 16) for k, v in items}
        keys = list(hashes)
        brute = {
            tuple(sorted((keys[i], keys[j])))
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
            if bin(hashes[keys[i]] ^ hashes[keys[j]]).count("1") <= 6
        }
        got = {tuple(sorted((a, b))) for a, b, _ in near_duplicate_pairs(items)}
        assert got == brute, (
            f"{len(brute - got)} missed, {len(got - brute)} spurious"
        )


class TestDuplicateShaKeepsEveryObservationLink:
    def test_a_repost_merges_both_observations_into_one_leak_group(
        self, tmp_path, monkeypatch
    ):
        """The same photograph posted twice must merge both sightings.

        Collapsing a duplicated sha with ``any_value(group_key)`` kept one
        observation and discarded the other. When the discarded observation
        holds a *different* image, that image's sighting never joins the
        component, so two sightings sharing identical pixels can land in
        different splits. 923 images on the real store carry more than one
        observation; 405 of those sit in observations holding other images.
        """
        rows = _corpus(n_classes=12)
        shared = _row(1000, "obs-A", "user-A")
        repost = _row(1000, "obs-B", "user-B", sha=shared["sha256"])
        other = _row(1000, "obs-B", "user-B")
        rows += [shared, repost, other]

        db = _make_store(tmp_path, monkeypatch, rows)
        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)
            groups = dict(store.con.execute(
                "SELECT sha256, group_id FROM split_group_members "
                "WHERE dataset_version = ?", [DATASET_VERSION],
            ).fetchall())
            assignment = _assignments(store)

        assert groups[shared["sha256"]] == groups[other["sha256"]], (
            "the second observation's other image is in a different leak group "
            "from the image it shares with the first"
        )
        assert assignment[shared["sha256"]] == assignment[other["sha256"]]

    def test_every_photographer_of_a_reposted_image_is_recorded(
        self, tmp_path, monkeypatch
    ):
        """Losing the second photographer means a later group of theirs will
        not inherit, and that photographer then spans two splits."""
        rows = _corpus(n_classes=12)
        shared = _row(1000, "obs-A", "user-A")
        repost = _row(1000, "obs-B", "user-B", sha=shared["sha256"])
        rows += [shared, repost]

        db = _make_store(tmp_path, monkeypatch, rows)
        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)
            observers = store.con.execute(
                """
                SELECT g.observer_key FROM split_groups g
                JOIN split_group_members m USING (dataset_version, group_id)
                WHERE m.dataset_version = ? AND m.sha256 = ?
                """,
                [DATASET_VERSION, shared["sha256"]],
            ).fetchone()[0]

        assert "user-A" in observers and "user-B" in observers, (
            f"only one photographer survived: {observers!r}"
        )


class TestObserverRuleDoesNotActivateTooLate:
    def test_a_photographer_never_spans_splits_as_a_class_grows(
        self, tmp_path, monkeypatch
    ):
        """A class assigned while thin, then grown past the old 8-photographer
        gate, must not leave a photographer straddling train and a holdout.

        The gate consulted the class's *current* photographer count, so
        everything assigned before the class grew got independent per-group
        hashes. Assignments are immutable, so switching the rule on afterwards
        can never repair them.
        """
        rows = [_row(1000, f"early-obs-{g}", f"photog-{g % 5}") for g in range(24)]
        db = _make_store(tmp_path, monkeypatch, rows)
        _assign(db, "batch0")

        _add_rows(db, [
            _row(1000, f"late-obs-{g}", f"photog-{5 + (g % 9)}") for g in range(36)
        ])
        _assign(db, "batch1")

        with V2SplitStore(db) as store:
            spanning = store.con.execute(
                """
                SELECT count(*) FROM (
                    SELECT taxon_id, observer_key FROM split_groups
                    WHERE dataset_version = ? AND observer_key IS NOT NULL
                    GROUP BY taxon_id, observer_key
                    HAVING count(DISTINCT split) > 1)
                """,
                [DATASET_VERSION],
            ).fetchone()[0]

        assert spanning == 0, (
            f"{spanning} photographer/class pairs span two splits - the rule "
            f"activated after they had already been assigned"
        )

    def test_inheritance_survives_a_restart(self, tmp_path, monkeypatch):
        """Inheritance must re-seed from persisted rows, not only from memory,
        or every new process starts each photographer over again."""
        rows = [_row(1000, f"obs-{g}", f"photog-{g % 6}") for g in range(24)]
        db = _make_store(tmp_path, monkeypatch, rows)
        _assign(db, "batch0")

        for rnd in range(4):
            _add_rows(db, [
                _row(1000, f"more-{rnd}-{g}", f"photog-{g % 6}") for g in range(6)
            ])
            _assign(db, f"batch{rnd + 1}")

        with V2SplitStore(db) as store:
            spanning = store.con.execute(
                """
                SELECT count(*) FROM (
                    SELECT taxon_id, observer_key FROM split_groups
                    WHERE dataset_version = ? AND observer_key IS NOT NULL
                    GROUP BY taxon_id, observer_key
                    HAVING count(DISTINCT split) > 1)
                """,
                [DATASET_VERSION],
            ).fetchone()[0]
        assert spanning == 0


class TestKeysAreSourceNamespaced:
    def test_two_sources_reusing_an_id_are_not_merged(self, tmp_path, monkeypatch):
        """``observer_key`` is a bare provider integer - 308,227 of 308,227 on
        the real store - and ``group_key`` a bare provider id. A second source
        reusing either value would otherwise be treated as the same
        photographer, or merge two species into a single leak group.
        """
        rows = _corpus(n_classes=12)
        db = _make_store(tmp_path, monkeypatch, rows)

        a = _row(1000, "12345", "777")
        a["candidate_id"] = "inaturalist:aaa"
        b = _row(1007, "12345", "777")
        b["candidate_id"] = "gbif:bbb"
        con = duckdb.connect(str(db))
        _insert(con, [a, b])
        con.close()

        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)
            groups = dict(store.con.execute(
                "SELECT sha256, group_id FROM split_group_members "
                "WHERE dataset_version = ?", [DATASET_VERSION],
            ).fetchall())
            observers = {
                r[0] for r in store.con.execute(
                    "SELECT DISTINCT observer_key FROM split_groups "
                    "WHERE dataset_version = ? AND observer_key IS NOT NULL",
                    [DATASET_VERSION],
                ).fetchall()
            }

        assert groups[a["sha256"]] != groups[b["sha256"]], (
            "two sources sharing an observation id were merged into one leak "
            "group - across different species"
        )
        assert any(o.startswith("inaturalist:") for o in observers)
        assert any(o.startswith("gbif:") for o in observers)


class TestSplitFractionsAreFrozen:
    """The ratio is a property of the corpus, not of a run.

    Assigning batch zero at one ratio and a later batch at another moves nothing
    already assigned - immutability still holds - but leaves the two halves of
    the corpus with different holdout proportions, which biases every
    measurement taken across them and is invisible afterwards.
    """

    def test_the_frozen_default_is_80_10_5_5(self):
        from fwdata.splits_v2 import DEFAULT_FRACTIONS

        assert DEFAULT_FRACTIONS == {
            "train": 0.80, "validation": 0.10, "dev_test": 0.05, "final_test": 0.05,
        }

    def test_the_default_sums_to_one(self):
        from fwdata.splits_v2 import DEFAULT_FRACTIONS, validate_fractions

        assert validate_fractions(DEFAULT_FRACTIONS) == DEFAULT_FRACTIONS

    @pytest.mark.parametrize("bad", [
        {"train": 0.8, "validation": 0.1, "dev_test": 0.05, "final_test": 0.10},
        {"train": 0.8, "validation": 0.1, "dev_test": 0.05, "final_test": 0.02},
        {"train": 0.9, "validation": 0.1, "dev_test": 0.05},
        {"train": 1.1, "validation": 0.0, "dev_test": 0.0, "final_test": -0.1},
    ])
    def test_fractions_that_do_not_sum_to_one_are_refused(self, bad):
        from fwdata.splits_v2 import validate_fractions

        with pytest.raises(ValueError):
            validate_fractions(bad)

    def test_unknown_split_names_are_refused(self):
        from fwdata.splits_v2 import validate_fractions

        with pytest.raises(ValueError, match="unknown"):
            validate_fractions({"train": 0.8, "validation": 0.1, "dev_test": 0.05,
                                "final_test": 0.05, "test": 0.0})

    def test_first_assign_records_the_ratio(self, tmp_path, monkeypatch):
        from fwdata.splits_v2 import DEFAULT_FRACTIONS

        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=10))
        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)
            assert store.stored_fractions() == DEFAULT_FRACTIONS

    def test_a_later_batch_cannot_change_the_ratio(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=10))
        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)

        _add_rows(db, [_row(1000, f"later-{g}", f"user-later-{g}") for g in range(8)])
        with V2SplitStore(db) as store:
            with pytest.raises(ValueError, match="frozen"):
                store.assign(
                    batch="batch1", log=lambda m: None,
                    fractions={"train": 0.70, "validation": 0.10,
                               "dev_test": 0.10, "final_test": 0.10},
                )

    def test_a_later_batch_with_the_frozen_ratio_proceeds(self, tmp_path, monkeypatch):
        from fwdata.splits_v2 import DEFAULT_FRACTIONS

        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=10))
        before = _assign(db, "batch0")
        _add_rows(db, [_row(1000, f"later-{g}", f"user-later-{g}") for g in range(8)])
        with V2SplitStore(db) as store:
            store.assign(batch="batch1", log=lambda m: None,
                         fractions=dict(DEFAULT_FRACTIONS))
            after = _assignments(store)
        assert all(after.get(k) == v for k, v in before.items())
        assert len(after) > len(before)

    def test_the_frozen_ratio_survives_reopening_the_database(
        self, tmp_path, monkeypatch
    ):
        from fwdata.splits_v2 import DEFAULT_FRACTIONS

        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=10))
        with V2SplitStore(db) as store:
            store.assign(batch="batch0", log=lambda m: None)
        with V2SplitStore(db) as store:
            assert store.stored_fractions() == DEFAULT_FRACTIONS

    def test_the_ratio_actually_shapes_the_split(self, tmp_path, monkeypatch):
        """80/10/5/5 must produce a visibly larger train share than the holdouts,
        so a silently-ignored config would be caught."""
        db = _make_store(tmp_path, monkeypatch,
                         _corpus(n_classes=40, groups_per_class=40,
                                 observers_per_class=40))
        with V2SplitStore(db) as store:
            stats = store.assign(batch="batch0", log=lambda m: None)
        images = stats.images_by_split
        total = sum(images.values())
        assert images["train"] / total > 0.70
        for holdout in ("dev_test", "final_test"):
            assert images[holdout] / total < 0.12
