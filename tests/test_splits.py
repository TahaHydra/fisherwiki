"""The leakage guard is the most load-bearing check in the pipeline.

It caught three distinct real bugs during development, and in every case nothing
in the accuracy metrics would have revealed the problem - a leaking split simply
looks good. So the guard itself needs tests, built on synthetic data where the
right answer is known.
"""

from __future__ import annotations

import hashlib

import pytest

from fwdata import config
from fwdata.splits import (
    CorpusBuilder,
    SplitStrategy,
    assign_split,
    stable_fraction,
)


# ---------------------------------------------------------------------------
# deterministic assignment
# ---------------------------------------------------------------------------
class TestStableAssignment:
    def test_same_key_always_gives_the_same_fraction(self):
        a = stable_fraction("observation-123", "salt")
        b = stable_fraction("observation-123", "salt")
        assert a == b

    def test_fraction_is_in_range(self):
        for i in range(500):
            f = stable_fraction(f"obs-{i}", "s")
            assert 0.0 <= f < 1.0

    def test_different_salts_give_different_assignments(self):
        keys = [f"obs-{i}" for i in range(300)]
        a = [assign_split(k, {"train": 0.8, "val": 0.1, "test": 0.1}, "salt-a")
             for k in keys]
        b = [assign_split(k, {"train": 0.8, "val": 0.1, "test": 0.1}, "salt-b")
             for k in keys]
        assert a != b

    def test_assignment_does_not_depend_on_python_hash_randomisation(self):
        # The whole reason for SHA-256 here: Python's str hash is salted per
        # process, which would reshuffle the split on every run and make
        # "never tune against test" impossible to honour.
        expected = hashlib.sha256(b"s\x00k").digest()
        assert stable_fraction("k", "s") == int.from_bytes(expected[:8], "big") / (1 << 64)

    def test_proportions_are_roughly_respected(self):
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(20000):
            counts[assign_split(f"obs-{i}", fractions, "s")] += 1
        assert 0.77 < counts["train"] / 20000 < 0.83
        assert 0.07 < counts["val"] / 20000 < 0.13
        assert 0.07 < counts["test"] / 20000 < 0.13


# ---------------------------------------------------------------------------
# the builder, against a synthetic provenance store
# ---------------------------------------------------------------------------
def _make_store(tmp_path, monkeypatch, rows):
    """Build a minimal provenance database containing exactly ``rows``.

    ``rows`` are dicts with candidate_id, sha256, taxon_id, name, group_key,
    observer_key.
    """
    paths = config.Paths(tmp_path).ensure()
    monkeypatch.setattr(config, "PATHS", paths)

    import duckdb

    db = tmp_path / "provenance.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE corpus_members (
            corpus VARCHAR, candidate_id VARCHAR, sha256 VARCHAR,
            class_id BIGINT, taxon_id BIGINT, split VARCHAR, weight DOUBLE,
            PRIMARY KEY (corpus, candidate_id)
        );
        CREATE TABLE prov_src (
            candidate_id VARCHAR, sha256 VARCHAR, species_taxon_id BIGINT,
            accepted_scientific_name VARCHAR, group_key VARCHAR,
            observer_key VARCHAR, latitude DOUBLE, longitude DOUBLE,
            cas_path VARCHAR, width BIGINT, height BIGINT, observed_on VARCHAR,
            license VARCHAR
        );
        """
    )
    con.executemany(
        "INSERT INTO prov_src VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (r["candidate_id"], r["sha256"], r["taxon_id"], r["name"],
             r["group_key"], r["observer_key"], r.get("lat"), r.get("lon"),
             f"{r['sha256'][:2]}/{r['sha256'][2:4]}/{r['sha256']}.jpg",
             500, 375, "2024-01-01", "CC-BY-4.0")
            for r in rows
        ],
    )
    con.execute("CREATE VIEW provenance AS SELECT * FROM prov_src")
    con.close()
    return db


def _rows(n_classes=5, groups_per_class=10, images_per_group=3,
          observers_per_class=5):
    rows = []
    k = 0
    for c in range(n_classes):
        for g in range(groups_per_class):
            for i in range(images_per_group):
                k += 1
                rows.append({
                    "candidate_id": f"src:{k}",
                    "sha256": f"{k:064x}",
                    "taxon_id": 1000 + c,
                    "name": f"Genus{c} species{c}",
                    "group_key": f"obs-{c}-{g}",
                    "observer_key": f"user-{c}-{g % observers_per_class}",
                })
    return rows


class TestBuilderProducesCleanSplits:
    def test_no_group_spans_two_splits(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _rows())
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.leakage_groups == 0
            assert stats.leakage_hashes == 0
        finally:
            b.close()

    def test_every_class_gets_val_and_test(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _rows(groups_per_class=20))
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.classes == 5
            assert stats.classes_missing_val == 0
            assert stats.classes_missing_test == 0
        finally:
            b.close()

    def test_identical_images_are_deduplicated(self, tmp_path, monkeypatch):
        # The same photograph under two candidate ids in two observations.
        # Without dedup these land in different groups and therefore possibly
        # different splits: byte-identical pixels in train and test.
        rows = _rows()
        dup = dict(rows[0])
        dup["candidate_id"] = "src:duplicate"
        dup["group_key"] = "obs-0-999"          # a different observation
        rows.append(dup)

        db = _make_store(tmp_path, monkeypatch, rows)
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.leakage_hashes == 0
            total = sum(stats.images.values())
            # 150 unique images, not 151.
            assert total == 150
        finally:
            b.close()

    def test_conflicting_labels_are_counted_not_silently_dropped(
        self, tmp_path, monkeypatch
    ):
        rows = _rows()
        conflict = dict(rows[0])
        conflict["candidate_id"] = "src:conflict"
        conflict["taxon_id"] = 9999            # same bytes, different species
        conflict["name"] = "Other species"
        conflict["group_key"] = "obs-9-9"
        rows.append(conflict)

        db = _make_store(tmp_path, monkeypatch, rows)
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.label_conflicts >= 1
        finally:
            b.close()

    def test_classes_below_the_evidence_bar_are_excluded(
        self, tmp_path, monkeypatch
    ):
        rows = _rows(n_classes=3)
        # A fourth class with only two images from one observation.
        for i in range(2):
            rows.append({
                "candidate_id": f"src:thin{i}", "sha256": f"{900 + i:064x}",
                "taxon_id": 2000, "name": "Thin species",
                "group_key": "obs-thin", "observer_key": "user-thin",
            })
        db = _make_store(tmp_path, monkeypatch, rows)
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.classes == 3
        finally:
            b.close()

    def test_rebuilding_gives_the_same_split(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _rows())
        b = CorpusBuilder(db)
        try:
            first = b.build("t", min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            second = b.build("t", min_images_per_class=5,
                             min_observations_per_class=3, log=lambda m: None)
            assert first.images == second.images
            assert first.groups == second.groups
        finally:
            b.close()

    def test_adding_images_does_not_move_existing_observations(
        self, tmp_path, monkeypatch
    ):
        """The property that makes 'never tune against test' enforceable."""
        import duckdb

        base = _rows(n_classes=3, groups_per_class=20)
        db = _make_store(tmp_path, monkeypatch, base)
        b = CorpusBuilder(db)
        try:
            b.build("t", min_images_per_class=5, min_observations_per_class=3,
                    log=lambda m: None)
            before = dict(b.con.execute(
                "SELECT candidate_id, split FROM corpus_members WHERE corpus='t'"
            ).fetchall())
        finally:
            b.close()

        # Add a brand-new observation to each class.
        con = duckdb.connect(str(db))
        extra = []
        for c in range(3):
            for i in range(3):
                k = 500000 + c * 10 + i
                extra.append((f"src:new{k}", f"{k:064x}", 1000 + c,
                              f"Genus{c} species{c}", f"obs-new-{c}",
                              f"user-new-{c}", None, None, "x/y/z.jpg",
                              500, 375, "2024-01-01", "CC-BY-4.0"))
        con.executemany("INSERT INTO prov_src VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        extra)
        con.close()

        b = CorpusBuilder(db)
        try:
            b.build("t", min_images_per_class=5, min_observations_per_class=3,
                    log=lambda m: None)
            after = dict(b.con.execute(
                "SELECT candidate_id, split FROM corpus_members WHERE corpus='t'"
            ).fetchall())
        finally:
            b.close()

        moved = [cid for cid, split in before.items() if after.get(cid) != split]
        assert moved == [], f"{len(moved)} observations changed split on rebuild"


class TestObserverStrategyFallsBackSafely:
    def test_observer_grouping_does_not_leak(self, tmp_path, monkeypatch):
        # Per-class stratification is unsound for observer groups (an observer
        # spans many species). The builder must fall back to a single global
        # assignment rather than producing 748 groups spanning splits, as an
        # earlier version did.
        db = _make_store(tmp_path, monkeypatch,
                         _rows(n_classes=6, groups_per_class=20,
                               observers_per_class=4))
        b = CorpusBuilder(db)
        try:
            stats = b.build("t", strategy=SplitStrategy.OBSERVER,
                            min_images_per_class=5,
                            min_observations_per_class=3, log=lambda m: None)
            assert stats.leakage_groups == 0
            assert stats.leakage_hashes == 0
        finally:
            b.close()
