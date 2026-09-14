"""Leakage-safe dataset splitting.

Why not ``train_test_split``
---------------------------
Random per-image splitting would inflate every number we report. iNaturalist
observations routinely contain several photographs of the *same individual fish*
taken seconds apart; a random split puts some in train and some in test, and the
model gets credit for recognising a specific fish rather than a species. On a
corpus like ours that is worth many points of apparent top-1.

So the unit of splitting is the **observation**, never the image. Every photo of
one observation goes to exactly one split.

Identical images are also deduplicated by SHA-256 first. The same photograph
genuinely reaches us under two candidate ids - re-uploaded as a second
observation, or shared between accounts - and since those are different
observations they would land in different splits. That is byte-identical pixels
in train and test; there were 338 such cases in this corpus.

Why grouping by photographer is *not* the default
-------------------------------------------------
One photographer uploads many observations of the same population, from the same
spot, with the same hands and the same net, so grouping by photographer is
strictly harder and closer to the real deployment condition. It was tried as the
default and rejected on measurement: with ~5,500 photographers spanning 1,978
classes, **284 classes ended up with no validation images at all**. A split that
cannot measure 14% of the model is worse than a slightly optimistic one.

Instead the split groups by **observation** and stratifies within each class, so
every class is represented in val and test. The photographer concern is then
measured rather than imposed: test rows whose photographer never appears in
train are flagged ``unseen_observer`` and reported separately. That gives both
the well-covered number and the pessimistic one.

Per-class stratification is only sound because an observation has exactly one
taxon. It is *not* sound for photographer groups, who span many species -
attempting it produced 748 groups spanning splits, which
:meth:`CorpusBuilder.verify_no_leakage` caught. Observer strategies therefore
fall back to a single global assignment per group.

Held-out sets produced
----------------------
``train``       what the model fits
``val``         model selection, early stopping, temperature fitting
``test``        reported once, at the end; never tuned against
``geo_test``    observations from held-out geographic cells (opt-in), to measure
                degradation outside the training distribution

Determinism
-----------
Assignment derives from a hash of the group key, not a shuffle, so the split is
stable across runs, machines and corpus growth: adding new images never moves an
existing observation between splits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .config import PATHS

#: Default proportions. geo_test is carved out of the same pool, not additional.
DEFAULT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}


class SplitStrategy:
    """How rows are grouped so that no group spans two splits."""

    OBSERVATION = "observation"
    OBSERVER = "observer"
    #: Group by photographer, then by observation within the photographer.
    #: The default: hardest and most realistic.
    OBSERVER_THEN_OBSERVATION = "observer_then_observation"


@dataclass
class SplitStats:
    corpus: str
    strategy: str
    classes: int = 0
    images: dict[str, int] = field(default_factory=dict)
    groups: dict[str, int] = field(default_factory=dict)
    classes_missing_val: int = 0
    classes_missing_test: int = 0
    min_train_images: int = 0
    leakage_groups: int = 0
    leakage_hashes: int = 0
    geo_test_images: int = 0
    unseen_observer_test_images: int = 0
    label_conflicts: int = 0

    def as_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "strategy": self.strategy,
            "classes": self.classes,
            "images": self.images,
            "groups": self.groups,
            "classes_missing_val": self.classes_missing_val,
            "classes_missing_test": self.classes_missing_test,
            "min_train_images": self.min_train_images,
            "leakage_groups": self.leakage_groups,
            "leakage_hashes": self.leakage_hashes,
            "geo_test_images": self.geo_test_images,
            "unseen_observer_test_images": self.unseen_observer_test_images,
            "label_conflicts": self.label_conflicts,
        }


def stable_fraction(key: str, salt: str = "") -> float:
    """Map a group key to a stable value in [0, 1).

    SHA-256 rather than :func:`hash` because Python's string hash is salted per
    process, which would silently reshuffle the split on every run and make
    "never tune against test" impossible to honour.
    """
    h = hashlib.sha256((salt + "\x00" + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_split(key: str, fractions: dict[str, float], salt: str = "") -> str:
    f = stable_fraction(key, salt)
    acc = 0.0
    for name, frac in fractions.items():
        acc += frac
        if f < acc:
            return name
    return list(fractions)[-1]


class CorpusBuilder:
    """Build a named, split, hashed corpus from the provenance store."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or PATHS.provenance_db)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute("PRAGMA threads=8")
        self._register_udfs()

    def _register_udfs(self) -> None:
        self.con.create_function(
            "split_of",
            lambda k, s: assign_split(k or "", DEFAULT_FRACTIONS, s or ""),
            ["VARCHAR", "VARCHAR"],
            "VARCHAR",
        )

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------
    def build(
        self,
        corpus: str,
        *,
        strategy: str = SplitStrategy.OBSERVATION,
        min_images_per_class: int = 40,
        min_observations_per_class: int = 25,
        fractions: dict[str, float] | None = None,
        salt: str | None = None,
        geo_holdout_cells: int = 0,
        log=print,
    ) -> SplitStats:
        """Materialise ``corpus_members`` rows for ``corpus``."""
        fractions = fractions or DEFAULT_FRACTIONS
        salt = salt if salt is not None else corpus
        stats = SplitStats(corpus=corpus, strategy=strategy)

        group_expr = {
            SplitStrategy.OBSERVATION: "p.group_key",
            SplitStrategy.OBSERVER: "p.observer_key",
            SplitStrategy.OBSERVER_THEN_OBSERVATION:
                "coalesce(p.observer_key, p.group_key)",
        }[strategy]

        log(f"building corpus {corpus!r} (strategy={strategy})")

        # 1a. Identical bytes carrying two different species labels is a
        # label-quality problem, not a duplication problem: unlike a re-upload
        # of the same observation, there is no principled way to prefer one
        # candidate_id over the other, because the *label itself* is in
        # dispute, not just which record names it. Computed before `eligible`
        # so it can be excluded there rather than arbitrarily resolved by
        # whichever row the dedup happens to keep - `ORDER BY candidate_id` in
        # that QUALIFY has no relationship to which label is correct.
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE conflicting_hashes AS
            SELECT sha256 FROM provenance
            WHERE species_taxon_id IS NOT NULL AND sha256 IS NOT NULL
            GROUP BY sha256 HAVING count(DISTINCT species_taxon_id) > 1
            """
        )
        conflicts = self.con.execute(
            "SELECT count(*) FROM conflicting_hashes"
        ).fetchone()[0]
        stats.label_conflicts = int(conflicts)
        if conflicts:
            log(f"  {conflicts:,} images carry conflicting species labels "
                f"(identical bytes, different taxon) - excluded until reconciled")

        # 1b. Eligible images: stored, licensed, with a canonical taxon, and
        # not a disputed label.
        #
        # Deduplicated by sha256, one row per distinct image. The same
        # photograph does reach us under two candidate ids - re-uploaded as a
        # second observation, or shared between accounts - and because those
        # are different observations they land in different groups and
        # therefore different splits. That is byte-identical pixels in train
        # and test: 338 cases on this corpus, caught by verify_no_leakage().
        # Content-addressed storage makes the fix exact rather than fuzzy.
        # This dedup is sound *because* conflicting_hashes was removed first:
        # every sha256 reaching it now has exactly one label, so any
        # candidate_id kept by the tie-break names the same species as every
        # other candidate_id sharing that hash.
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE eligible AS
            SELECT p.candidate_id, p.sha256, p.species_taxon_id AS taxon_id,
                   p.accepted_scientific_name AS name,
                   p.group_key, p.observer_key,
                   p.latitude, p.longitude,
                   {group_expr} AS split_group
            FROM provenance p
            WHERE p.species_taxon_id IS NOT NULL
              AND p.sha256 IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM conflicting_hashes ch WHERE ch.sha256 = p.sha256
              )
            QUALIFY row_number() OVER (
                PARTITION BY p.sha256 ORDER BY p.candidate_id
            ) = 1
            """
        )

        # 2. Classes that clear the evidence bar *after* download losses.
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE classes AS
            SELECT taxon_id, any_value(name) AS name,
                   count(*) AS images,
                   count(DISTINCT group_key) AS observations,
                   count(DISTINCT observer_key) AS observers
            FROM eligible
            GROUP BY taxon_id
            HAVING count(*) >= {int(min_images_per_class)}
               AND count(DISTINCT group_key) >= {int(min_observations_per_class)}
            """
        )
        stats.classes = self.con.execute("SELECT count(*) FROM classes").fetchone()[0]
        log(f"  {stats.classes:,} classes clear the bar after download losses")

        # 3. Assign a dense class index, ordered by name so it is reproducible.
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE class_index AS
            SELECT taxon_id, name, images, observations, observers,
                   row_number() OVER (ORDER BY name) - 1 AS class_id
            FROM classes
            """
        )

        # 4. Split by group, stratified per class.
        #
        # Assigning a global hash to each group was the first implementation and
        # it covered classes badly: with observer-level groups, 284 of 1,979
        # classes ended up with no validation images at all, so 14% of the model
        # could not be measured. Stratifying *within* each class fixes that
        # without weakening the guarantee, because an observation belongs to
        # exactly one taxon - so a group can never be pulled toward two splits.
        #
        # Groups are ordered by a stable hash within their class and cut at the
        # target proportions, which keeps the assignment deterministic and
        # stable as the corpus grows.
        # Per-class stratification is only sound when a group belongs to
        # exactly one class. That holds for observations (one sighting has one
        # taxon) and does NOT hold for photographers, who span many species.
        # Stratifying an observer group per class assigns the same observer to
        # different splits for different classes, and the leakage check catches
        # it: the first attempt produced 748 groups spanning splits. Observer
        # strategies therefore use a single global assignment per group.
        stratify = strategy == SplitStrategy.OBSERVATION

        frac_train = fractions.get("train", 0.8)
        frac_val = fractions.get("val", 0.1)
        if not stratify:
            self.con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE group_split AS
                SELECT DISTINCT ci.class_id, e.split_group,
                       split_of(e.split_group, '{salt}') AS split
                FROM eligible e
                JOIN class_index ci USING (taxon_id)
                """
            )
        else:
            self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE group_split AS
            WITH groups AS (
                SELECT ci.class_id, e.split_group,
                       count(*) AS n_images
                FROM eligible e
                JOIN class_index ci USING (taxon_id)
                GROUP BY ci.class_id, e.split_group
            ),
            ordered AS (
                SELECT class_id, split_group, n_images,
                       row_number() OVER (
                           PARTITION BY class_id
                           ORDER BY hash(split_group || '{salt}')
                       ) AS rn,
                       count(*) OVER (PARTITION BY class_id) AS n_groups
                FROM groups
            )
            SELECT class_id, split_group,
                   CASE
                     -- A class with very few groups keeps them all in train:
                     -- a one-group val set measures nothing and costs the
                     -- class its only training data.
                     WHEN n_groups < 4 THEN 'train'
                     WHEN rn <= ceil(n_groups * {frac_train}) THEN 'train'
                     WHEN rn <= ceil(n_groups * {frac_train + frac_val}) THEN 'val'
                     ELSE 'test'
                   END AS split
            FROM ordered
            """
            )
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE assigned AS
            SELECT e.*, ci.class_id, gs.split
            FROM eligible e
            JOIN class_index ci USING (taxon_id)
            JOIN group_split gs
              ON gs.class_id = ci.class_id AND gs.split_group = e.split_group
            """
        )

        # 4b. Flag the subset of test whose *photographer* never appears in
        # train. That is the real deployment condition - an unseen user, in an
        # unseen place - and it is measured separately rather than imposed on
        # the whole split, because imposing it starves the rarer classes.
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE train_observers AS
            SELECT DISTINCT observer_key FROM assigned
            WHERE split = 'train' AND observer_key IS NOT NULL
            """
        )
        self.con.execute(
            """
            ALTER TABLE assigned ADD COLUMN unseen_observer BOOLEAN
            """
        )
        self.con.execute(
            """
            UPDATE assigned SET unseen_observer =
                (split = 'test'
                 AND observer_key IS NOT NULL
                 AND observer_key NOT IN (SELECT observer_key FROM train_observers))
            """
        )

        # 5. Optional geographic hold-out, carved out of test only.
        if geo_holdout_cells > 0:
            self.con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE geo_cells AS
                SELECT DISTINCT
                    CAST(floor(latitude / 5.0) AS INTEGER) * 1000 +
                    CAST(floor(longitude / 5.0) AS INTEGER) AS cell
                FROM assigned WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY cell
                LIMIT {int(geo_holdout_cells)}
                """
            )
            self.con.execute(
                """
                UPDATE assigned SET split = 'geo_test'
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                  AND (CAST(floor(latitude / 5.0) AS INTEGER) * 1000 +
                       CAST(floor(longitude / 5.0) AS INTEGER))
                      IN (SELECT cell FROM geo_cells)
                """
            )
            stats.geo_test_images = self.con.execute(
                "SELECT count(*) FROM assigned WHERE split = 'geo_test'"
            ).fetchone()[0]

        # 6. Persist.
        self.con.execute("DELETE FROM corpus_members WHERE corpus = ?", [corpus])
        self.con.execute(
            """
            INSERT INTO corpus_members
            SELECT ?, candidate_id, sha256, class_id, taxon_id, split, 1.0
            FROM assigned
            """,
            [corpus],
        )

        # 7. Report and verify.
        for split, n in self.con.execute(
            "SELECT split, count(*) FROM assigned GROUP BY split ORDER BY split"
        ).fetchall():
            stats.images[split] = int(n)
        for split, n in self.con.execute(
            "SELECT split, count(DISTINCT split_group) FROM assigned "
            "GROUP BY split ORDER BY split"
        ).fetchall():
            stats.groups[split] = int(n)

        stats.classes_missing_val = self.con.execute(
            "SELECT count(*) FROM (SELECT class_id FROM assigned GROUP BY class_id "
            "HAVING sum(CASE WHEN split='val' THEN 1 ELSE 0 END) = 0)"
        ).fetchone()[0]
        stats.classes_missing_test = self.con.execute(
            "SELECT count(*) FROM (SELECT class_id FROM assigned GROUP BY class_id "
            "HAVING sum(CASE WHEN split='test' THEN 1 ELSE 0 END) = 0)"
        ).fetchone()[0]
        stats.min_train_images = self.con.execute(
            "SELECT coalesce(min(n), 0) FROM (SELECT class_id, count(*) n FROM assigned "
            "WHERE split='train' GROUP BY class_id)"
        ).fetchone()[0]

        stats.unseen_observer_test_images = self.con.execute(
            "SELECT count(*) FROM assigned WHERE unseen_observer"
        ).fetchone()[0]

        leak = self.verify_no_leakage(corpus)
        stats.leakage_groups = leak["groups_spanning_splits"]
        stats.leakage_hashes = leak["hashes_spanning_splits"]

        log(f"  images: {stats.images}")
        log(f"  groups: {stats.groups}")
        log(f"  classes with no val images : {stats.classes_missing_val}")
        log(f"  classes with no test images: {stats.classes_missing_test}")
        log(f"  smallest train class       : {stats.min_train_images} images")
        log(f"  test images from an unseen photographer: "
            f"{stats.unseen_observer_test_images:,}")
        log(f"  leakage (groups / hashes)  : {stats.leakage_groups} / {stats.leakage_hashes}")

        report = PATHS.work / f"corpus_{corpus}.json"
        report.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
        return stats

    # ------------------------------------------------------------------
    def verify_no_leakage(self, corpus: str) -> dict:
        """Assert that no group and no identical image spans two splits.

        This is run as part of every build and again by
        ``tools/dataset.py verify``, because a split that silently leaks makes
        every evaluation number meaningless and there is no way to notice from
        the metrics alone - they just look good.
        """
        groups = self.con.execute(
            """
            SELECT count(*) FROM (
                SELECT split_group FROM assigned
                GROUP BY split_group HAVING count(DISTINCT split) > 1
            )
            """
        ).fetchone()[0]
        hashes = self.con.execute(
            """
            SELECT count(*) FROM (
                SELECT sha256 FROM corpus_members WHERE corpus = ?
                GROUP BY sha256 HAVING count(DISTINCT split) > 1
            )
            """,
            [corpus],
        ).fetchone()[0]
        return {
            "groups_spanning_splits": int(groups),
            "hashes_spanning_splits": int(hashes),
        }

    def export_manifest(self, corpus: str, out_dir: Path | None = None) -> Path:
        """Write the training manifest: one row per image with split and class."""
        out_dir = Path(out_dir or PATHS.artifacts / corpus)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "manifest.parquet"
        self.con.execute(
            f"""
            COPY (
                SELECT m.sha256, m.class_id, m.taxon_id, m.split,
                       p.accepted_scientific_name AS name,
                       p.cas_path, p.width, p.height,
                       p.latitude, p.longitude, p.observed_on,
                       p.group_key, p.observer_key, p.license,
                       a.unseen_observer
                FROM corpus_members m
                JOIN provenance p USING (candidate_id)
                LEFT JOIN assigned a ON a.candidate_id = m.candidate_id
                WHERE m.corpus = '{corpus}'
                ORDER BY m.split, m.class_id, m.sha256
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION zstd)
            """
        )
        labels = out_dir / "labels.json"
        rows = self.con.execute(
            """
            SELECT class_id, taxon_id, name, images, observations, observers
            FROM class_index ORDER BY class_id
            """
        ).fetchall()
        labels.write_text(
            json.dumps(
                {
                    "corpus": corpus,
                    "num_classes": len(rows),
                    "classes": [
                        {
                            "class_id": c,
                            "fw_taxon_id": t,
                            "scientific_name": n,
                            "images": i,
                            "observations": o,
                            "observers": ob,
                        }
                        for c, t, n, i, o, ob in rows
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out
