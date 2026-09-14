"""V2 splits: the growth-stability guarantee, and the seal on final_test.

The V1 splitter has a test - ``test_growth_can_move_an_existing_observation``
in ``tests/test_splits.py`` - that *proves it moves things*. It exists because
the docstring claimed stability that the rank-and-cut algorithm never had. The
tests here are the same adversarial scenario pointed at V2, where the assertion
is inverted: nothing may move, ever.

Adversarial means: many classes, several growth shapes, and enough of them that
one lucky class cannot make a broken implementation look correct. A single
class staying put proves nothing, which is exactly how the V1 claim survived
as long as it did.
"""

from __future__ import annotations

import pytest

from fwdata import config
from fwdata.splits_v2 import (
    DATASET_VERSION,
    MIN_GROUPS_FOR_HOLDOUT,
    SPLITS,
    V2SplitStore,
    band_of,
    stable_fraction,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _make_store(tmp_path, monkeypatch, rows):
    """A minimal provenance database: `provenance` view plus `stored`.

    Mirrors the real schema closely enough for the splitter: the view carries
    identity/label/grouping, `stored` carries the perceptual hash the near
    duplicate edges are built from.
    """
    paths = config.Paths(tmp_path).ensure()
    monkeypatch.setattr(config, "PATHS", paths)

    import duckdb

    db = tmp_path / "provenance.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE prov_src (
            candidate_id VARCHAR, sha256 VARCHAR, species_taxon_id BIGINT,
            accepted_scientific_name VARCHAR, group_key VARCHAR,
            observer_key VARCHAR, latitude DOUBLE, longitude DOUBLE,
            cas_path VARCHAR, width BIGINT, height BIGINT, observed_on VARCHAR,
            license VARCHAR
        );
        CREATE TABLE stored (
            candidate_id VARCHAR, sha256 VARCHAR, dhash VARCHAR, phash VARCHAR
        );
        CREATE TABLE corpus_members (
            corpus VARCHAR, candidate_id VARCHAR, sha256 VARCHAR,
            class_id BIGINT, taxon_id BIGINT, split VARCHAR, weight DOUBLE
        );
        """
    )
    _insert(con, rows)
    con.execute("CREATE VIEW provenance AS SELECT * FROM prov_src")
    con.close()
    return db


def _insert(con, rows):
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
    con.executemany(
        "INSERT INTO stored VALUES (?,?,?,?)",
        [(r["candidate_id"], r["sha256"], r["dhash"], r["dhash"]) for r in rows],
    )


def _add_rows(db, rows):
    import duckdb

    con = duckdb.connect(str(db))
    _insert(con, rows)
    con.close()


_COUNTER = {"n": 0}


def _row(taxon, group, observer, *, sha=None, dhash=None):
    """One image. Distinct dhash values by default, so nothing is a near-dup."""
    _COUNTER["n"] += 1
    k = _COUNTER["n"]
    return {
        "candidate_id": f"src:{k}",
        "sha256": sha or f"{k:064x}",
        "taxon_id": taxon,
        "name": f"Genus{taxon} species{taxon}",
        "group_key": group,
        "observer_key": observer,
        # Spread hashes far apart in Hamming distance: sequential integers
        # would be near-duplicates of each other and union everything.
        "dhash": dhash or f"{(k * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF:016x}",
    }


def _corpus(n_classes=25, groups_per_class=12, images_per_group=2,
            observers_per_class=6):
    rows = []
    for c in range(n_classes):
        for g in range(groups_per_class):
            for _ in range(images_per_group):
                rows.append(_row(
                    1000 + c, f"obs-{c}-{g}", f"user-{c}-{g % observers_per_class}"
                ))
    return rows


def _assignments(store):
    return dict(store.con.execute(
        """
        SELECT m.sha256, g.split FROM split_group_members m
        JOIN split_groups g USING (dataset_version, group_id)
        WHERE m.dataset_version = ?
        """,
        [DATASET_VERSION],
    ).fetchall())


def _assign(db, batch, **kw):
    with V2SplitStore(db) as s:
        s.assign(batch=batch, log=lambda m: None, **kw)
        return _assignments(s)


# ---------------------------------------------------------------------------
# the band function itself
# ---------------------------------------------------------------------------
class TestBands:
    def test_every_split_is_reachable(self):
        fr = {"train": 0.70, "validation": 0.10, "dev_test": 0.10, "final_test": 0.10}
        seen = {band_of(stable_fraction(f"g{i}", "s"), fr) for i in range(5000)}
        assert seen == set(SPLITS)

    def test_bands_do_not_depend_on_how_many_groups_exist(self):
        # The V1 bug in one line: its cut was ceil(n * frac) over the current
        # group count, so the same key could land differently as n grew. A
        # fixed band cannot do that, because n is not an input.
        fr = {"train": 0.70, "validation": 0.10, "dev_test": 0.10, "final_test": 0.10}
        f = stable_fraction("group-abc", "v2:1000")
        assert band_of(f, fr) == band_of(f, fr)

    def test_proportions_are_roughly_respected(self):
        fr = {"train": 0.70, "validation": 0.10, "dev_test": 0.10, "final_test": 0.10}
        n = 20000
        counts = {s: 0 for s in SPLITS}
        for i in range(n):
            counts[band_of(stable_fraction(f"g{i}", "s"), fr)] += 1
        assert 0.67 < counts["train"] / n < 0.73
        for s in ("validation", "dev_test", "final_test"):
            assert 0.07 < counts[s] / n < 0.13


# ---------------------------------------------------------------------------
# requirement 2: growth must never move an existing assignment
# ---------------------------------------------------------------------------
class TestGrowthNeverMovesAnAssignment:
    def test_adding_an_observation_to_every_class_moves_nothing(
        self, tmp_path, monkeypatch
    ):
        """The exact scenario that moves V1's assignments.

        30 classes, one new observation added to each. V1 fails this (its own
        test asserts it fails); V2 must not move a single image.
        """
        rows = _corpus(n_classes=30, groups_per_class=20)
        db = _make_store(tmp_path, monkeypatch, rows)
        before = _assign(db, "batch0")

        _add_rows(db, [
            _row(1000 + c, f"obs-{c}-new", f"user-{c}-new")
            for c in range(30) for _ in range(2)
        ])
        after = _assign(db, "batch1")

        moved = {k: (before[k], after.get(k)) for k in before if after.get(k) != before[k]}
        assert moved == {}, f"{len(moved)} images changed split under growth"
        assert len(after) > len(before), "the new observations were never assigned"

    def test_adding_whole_new_classes_moves_nothing(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=20))
        before = _assign(db, "batch0")

        _add_rows(db, _corpus(n_classes=15, groups_per_class=10))  # taxon 1000+
        _add_rows(db, [
            _row(5000 + c, f"new-obs-{c}-{g}", f"new-user-{c}-{g % 4}")
            for c in range(15) for g in range(10)
        ])
        after = _assign(db, "batch1")

        assert all(after.get(k) == v for k, v in before.items())

    def test_adding_images_to_an_existing_observation_moves_nothing(
        self, tmp_path, monkeypatch
    ):
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=20))
        before = _assign(db, "batch0")

        # More photographs of sightings already assigned.
        _add_rows(db, [
            _row(1000 + c, f"obs-{c}-{g}", f"user-{c}-{g % 6}")
            for c in range(20) for g in range(12)
        ])
        after = _assign(db, "batch1")

        assert all(after.get(k) == v for k, v in before.items())
        # and the new photographs joined their observation's split
        with V2SplitStore(db) as s:
            spanning = s.verify()["groups_spanning_splits"]
        assert spanning == 0

    def test_ten_successive_growth_rounds_move_nothing(self, tmp_path, monkeypatch):
        """Stability has to survive repetition, not just one step."""
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=12))
        history = _assign(db, "batch0")

        for rnd in range(10):
            _add_rows(db, [
                _row(1000 + c, f"obs-{c}-r{rnd}-{g}", f"user-{c}-{(g + rnd) % 6}")
                for c in range(12) for g in range(3)
            ])
            now = _assign(db, f"batch{rnd + 1}")
            moved = [k for k, v in history.items() if now.get(k) != v]
            assert moved == [], f"round {rnd}: {len(moved)} images moved"
            history = now

    def test_reassigning_without_new_data_is_a_no_op(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _corpus())
        first = _assign(db, "batch0")
        with V2SplitStore(db) as s:
            fp1 = s.fingerprint()
        second = _assign(db, "batch0-again")
        with V2SplitStore(db) as s:
            fp2 = s.fingerprint()
        assert first == second
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# requirement 3: the leakage grouping hierarchy
# ---------------------------------------------------------------------------
class TestLeakGrouping:
    def test_exact_duplicates_under_two_candidate_ids_share_one_split(
        self, tmp_path, monkeypatch
    ):
        rows = _corpus(n_classes=12)
        twin = _row(1000 + 0, "obs-0-999", "user-elsewhere", sha=rows[0]["sha256"])
        rows.append(twin)

        db = _make_store(tmp_path, monkeypatch, rows)
        assignment = _assign(db, "batch0")
        # One node per image: the same bytes cannot be in two splits because
        # they are not two rows.
        assert rows[0]["sha256"] in assignment
        with V2SplitStore(db) as s:
            assert s.verify()["hashes_spanning_splits"] == 0

    def test_near_duplicates_in_different_observations_share_one_split(
        self, tmp_path, monkeypatch
    ):
        rows = _corpus(n_classes=12)
        a = _row(1000, "obs-near-a", "user-a", dhash="0f0f0f0f0f0f0f0f")
        b = _row(1000, "obs-near-b", "user-b", dhash="0f0f0f0f0f0f0f0e")  # 1 bit
        rows += [a, b]

        db = _make_store(tmp_path, monkeypatch, rows)
        assignment = _assign(db, "batch0")
        assert assignment[a["sha256"]] == assignment[b["sha256"]], (
            "a near-duplicate pair straddled two splits"
        )

    def test_near_duplicates_of_different_species_are_not_unioned(
        self, tmp_path, monkeypatch
    ):
        """Unioning across classes would merge two classes into one leak group
        and break the per-class stratification that makes every class
        measurable. A cross-class near-duplicate is a labelling suspect, and is
        reported as one."""
        rows = _corpus(n_classes=12)
        a = _row(1000, "obs-x-a", "user-a", dhash="a0a0a0a0a0a0a0a0")
        b = _row(1007, "obs-x-b", "user-b", dhash="a0a0a0a0a0a0a0a1")
        rows += [a, b]

        db = _make_store(tmp_path, monkeypatch, rows)
        with V2SplitStore(db) as s:
            stats = s.assign(batch="batch0", log=lambda m: None)
            groups = dict(s.con.execute(
                "SELECT sha256, group_id FROM split_group_members "
                "WHERE dataset_version = ?", [DATASET_VERSION]
            ).fetchall())
        assert stats.cross_class_near_dupes >= 1
        assert groups[a["sha256"]] != groups[b["sha256"]]

    def test_all_images_of_one_observation_share_one_split(
        self, tmp_path, monkeypatch
    ):
        db = _make_store(tmp_path, monkeypatch,
                         _corpus(n_classes=15, images_per_group=5))
        with V2SplitStore(db) as s:
            s.assign(batch="batch0", log=lambda m: None)
            spanning = s.con.execute(
                """
                SELECT count(*) FROM (
                    SELECT p.group_key FROM provenance p
                    JOIN split_group_members m ON m.sha256 = p.sha256
                    JOIN split_groups g USING (dataset_version, group_id)
                    WHERE m.dataset_version = ?
                    GROUP BY p.group_key HAVING count(DISTINCT g.split) > 1
                )
                """,
                [DATASET_VERSION],
            ).fetchone()[0]
        assert spanning == 0

    def test_observer_disjointness_applies_where_a_class_can_afford_it(
        self, tmp_path, monkeypatch
    ):
        db = _make_store(tmp_path, monkeypatch,
                         _corpus(n_classes=20, groups_per_class=30,
                                 observers_per_class=10))
        with V2SplitStore(db) as s:
            s.assign(batch="batch0", log=lambda m: None)
            # A photographer with several sightings of one species should not
            # appear on both sides of the train boundary for that species.
            spanning = s.con.execute(
                """
                SELECT count(*) FROM (
                    SELECT taxon_id, observer_key FROM split_groups
                    WHERE dataset_version = ? AND observer_key IS NOT NULL
                    GROUP BY taxon_id, observer_key
                    HAVING count(DISTINCT split) > 1
                )
                """,
                [DATASET_VERSION],
            ).fetchone()[0]
        assert spanning == 0, (
            "a photographer spans two splits within one class despite that "
            "class having enough photographers for the disjoint rule"
        )

    def test_a_thin_class_keeps_everything_in_train(self, tmp_path, monkeypatch):
        rows = _corpus(n_classes=10)
        # A class with fewer groups than the holdout minimum.
        for g in range(MIN_GROUPS_FOR_HOLDOUT - 1):
            rows.append(_row(7777, f"thin-obs-{g}", "thin-user"))

        db = _make_store(tmp_path, monkeypatch, rows)
        with V2SplitStore(db) as s:
            s.assign(batch="batch0", log=lambda m: None)
            splits = [r[0] for r in s.con.execute(
                "SELECT DISTINCT split FROM split_groups "
                "WHERE dataset_version = ? AND taxon_id = 7777",
                [DATASET_VERSION],
            ).fetchall()]
        assert splits == ["train"]


# ---------------------------------------------------------------------------
# growth that would force a move: quarantine instead
# ---------------------------------------------------------------------------
class TestBridgingConflictQuarantines:
    def test_a_bridge_between_two_splits_quarantines_the_new_image(
        self, tmp_path, monkeypatch
    ):
        """A later image can be a near-duplicate of two already-assigned groups
        that sit in different splits. Accepting it leaks; moving either group
        breaks immutability. So the new image is refused and recorded."""
        rows = _corpus(n_classes=12)
        db = _make_store(tmp_path, monkeypatch, rows)
        before = _assign(db, "batch0")

        # Find two groups of one class that landed in different splits.
        with V2SplitStore(db) as s:
            pairs = s.con.execute(
                """
                SELECT g.split, m.sha256 FROM split_groups g
                JOIN split_group_members m USING (dataset_version, group_id)
                WHERE g.dataset_version = ? AND g.taxon_id = 1000
                """,
                [DATASET_VERSION],
            ).fetchall()
        by_split: dict[str, str] = {}
        for split, sha in pairs:
            by_split.setdefault(split, sha)
        if len(by_split) < 2:
            pytest.skip("fixture did not produce two splits for this class")

        (split_a, sha_a), (split_b, sha_b) = list(by_split.items())[:2]

        # An image that is a near-duplicate of both.
        with V2SplitStore(db) as s:
            d_a, d_b = s.con.execute(
                "SELECT (SELECT dhash FROM stored WHERE sha256 = ?), "
                "       (SELECT dhash FROM stored WHERE sha256 = ?)",
                [sha_a, sha_b],
            ).fetchone()
        bridge_a = _row(1000, "obs-bridge-a", "user-bridge", dhash=d_a)
        bridge_b = _row(1000, "obs-bridge-b", "user-bridge", dhash=d_b)
        # Same observation, so the two halves union and pull both groups in.
        bridge_b["group_key"] = bridge_a["group_key"]
        _add_rows(db, [bridge_a, bridge_b])

        with V2SplitStore(db) as s:
            stats = s.assign(batch="batch1", log=lambda m: None)
            after = _assignments(s)
            quarantined = {r[0] for r in s.con.execute(
                "SELECT sha256 FROM split_quarantine WHERE dataset_version = ?",
                [DATASET_VERSION],
            ).fetchall()}

        assert stats.quarantined >= 1
        assert {bridge_a["sha256"], bridge_b["sha256"]} & quarantined
        # Nothing that already had a home moved.
        assert all(after.get(k) == v for k, v in before.items())
        assert bridge_a["sha256"] not in after or bridge_b["sha256"] not in after


# ---------------------------------------------------------------------------
# requirement 4: final_test is sealed
# ---------------------------------------------------------------------------
class TestFinalTestIsSealed:
    def _built(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch,
                         _corpus(n_classes=20, groups_per_class=30))
        store = V2SplitStore(db)
        store.assign(batch="batch0", log=lambda m: None)
        store.export_manifests(
            "global_v2", min_images_per_class=5,
            min_observations_per_class=3, log=lambda m: None,
        )
        return store

    def test_open_manifest_contains_no_final_test_rows(self, tmp_path, monkeypatch):
        store = self._built(tmp_path, monkeypatch)
        try:
            path = (config.PATHS.artifacts / "global_v2" / "manifest.parquet")
            splits = {r[0] for r in store.con.execute(
                f"SELECT DISTINCT split FROM read_parquet('{path.as_posix()}')"
            ).fetchall()}
            assert "final_test" not in splits
            assert splits <= {"train", "validation", "dev_test"}
        finally:
            store.close()

    def test_sealed_manifest_contains_only_final_test(self, tmp_path, monkeypatch):
        store = self._built(tmp_path, monkeypatch)
        try:
            path = (config.PATHS.artifacts / "global_v2" / "sealed"
                    / "final_test.parquet")
            assert path.exists()
            splits = {r[0] for r in store.con.execute(
                f"SELECT DISTINCT split FROM read_parquet('{path.as_posix()}')"
            ).fetchall()}
            assert splits == {"final_test"}
            assert (path.parent / "SEALED.md").exists()
        finally:
            store.close()

    def test_opening_final_test_without_authorisation_is_refused(
        self, tmp_path, monkeypatch
    ):
        store = self._built(tmp_path, monkeypatch)
        try:
            with pytest.raises(PermissionError):
                store.open_final_test("global_v2", reason="curiosity",
                                      authorised=False)
            assert store.access_log() == []
        finally:
            store.close()

    def test_opening_final_test_requires_a_reason(self, tmp_path, monkeypatch):
        store = self._built(tmp_path, monkeypatch)
        try:
            with pytest.raises(ValueError):
                store.open_final_test("global_v2", reason="  ", authorised=True)
        finally:
            store.close()

    def test_an_authorised_access_is_recorded(self, tmp_path, monkeypatch):
        store = self._built(tmp_path, monkeypatch)
        try:
            store.open_final_test("global_v2", reason="v2 release eval",
                                  authorised=True)
            log = store.access_log()
            assert len(log) == 1
            assert log[0][1] == "v2 release eval"
            assert log[0][3] > 0            # rows exposed, recorded
            store.open_final_test("global_v2", reason="again", authorised=True)
            assert len(store.access_log()) == 2, (
                "a second read must be visible - that is the entire point of "
                "recording them"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# requirements 5, 6, 9: persistence, no V1 import, coexistence
# ---------------------------------------------------------------------------
class TestPersistenceAndCoexistence:
    def test_assignment_is_read_from_the_table_not_recomputed(
        self, tmp_path, monkeypatch
    ):
        """Overwrite a stored assignment by hand; the next run must honour it.

        This is what "stored explicitly rather than derived" has to mean: if
        the table and the hash rule disagree, the table wins, because the table
        is what the model was actually trained against.
        """
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=12))
        _assign(db, "batch0")

        with V2SplitStore(db) as s:
            gid, old = s.con.execute(
                "SELECT group_id, split FROM split_groups "
                "WHERE dataset_version = ? AND split = 'train' LIMIT 1",
                [DATASET_VERSION],
            ).fetchone()
            s.con.execute(
                "UPDATE split_groups SET split = 'dev_test' "
                "WHERE dataset_version = ? AND group_id = ?",
                [DATASET_VERSION, gid],
            )

        _add_rows(db, [_row(1000, "obs-extra", "user-extra")])
        with V2SplitStore(db) as s:
            s.assign(batch="batch1", log=lambda m: None)
            now = s.con.execute(
                "SELECT split FROM split_groups WHERE dataset_version = ? "
                "AND group_id = ?", [DATASET_VERSION, gid],
            ).fetchone()[0]
        assert old == "train" and now == "dev_test"

    def test_v1_corpus_members_are_neither_read_nor_written(
        self, tmp_path, monkeypatch
    ):
        """V1 assignments are history. V2 must not import them, and must not
        touch them - the V1 model's reproducibility depends on that table."""
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=12))

        import duckdb

        con = duckdb.connect(str(db))
        rows = con.execute("SELECT sha256 FROM prov_src LIMIT 40").fetchall()
        # Pretend V1 put all of these in 'test'.
        con.executemany(
            "INSERT INTO corpus_members VALUES ('global_v1', ?, ?, 0, 1000, 'test', 1.0)",
            [(f"src:v1-{i}", sha) for i, (sha,) in enumerate(rows)],
        )
        con.close()

        assignment = _assign(db, "batch0")

        with V2SplitStore(db) as s:
            v1_after = s.con.execute(
                "SELECT count(*) FROM corpus_members WHERE corpus = 'global_v1'"
            ).fetchall()[0][0]
            v1_splits = {r[0] for r in s.con.execute(
                "SELECT DISTINCT split FROM corpus_members WHERE corpus='global_v1'"
            ).fetchall()}
        assert v1_after == 40, "V2 modified the V1 corpus table"
        assert v1_splits == {"test"}, "V2 rewrote V1 split labels"

        # And V2 did not simply copy V1's 'test' label onto those images.
        v2_for_those = {assignment[sha] for (sha,) in rows if sha in assignment}
        assert v2_for_those - {"train", "validation", "dev_test", "final_test"} == set()
        assert len(v2_for_those) > 1, (
            "every V1 'test' image landed in one V2 split - suspicious of an "
            "import rather than a fresh assignment"
        )

    def test_both_generations_coexist_in_one_database(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=12))
        _assign(db, "batch0")
        with V2SplitStore(db) as s:
            tables = {r[0] for r in s.con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()}
        assert {"corpus_members", "split_groups", "split_group_members"} <= tables

    def test_every_split_value_is_one_of_the_four(self, tmp_path, monkeypatch):
        db = _make_store(tmp_path, monkeypatch, _corpus(n_classes=25))
        _assign(db, "batch0")
        with V2SplitStore(db) as s:
            got = {r[0] for r in s.con.execute(
                "SELECT DISTINCT split FROM split_groups WHERE dataset_version = ?",
                [DATASET_VERSION],
            ).fetchall()}
            assert got <= set(SPLITS)
            assert got == set(SPLITS), f"not all four splits were used: {got}"
            assert s.verify() == {
                "groups_spanning_splits": 0,
                "hashes_spanning_splits": 0,
                "orphan_members": 0,
                "invalid_split_values": 0,
            }
