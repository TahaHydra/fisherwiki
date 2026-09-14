"""V2 split system: immutable, persisted, growth-stable four-way splits.

Why this exists alongside :mod:`fwdata.splits`
----------------------------------------------
V1's splitter derives splits on every build. For the OBSERVATION strategy it
ranks each class's groups by a stable hash and cuts at ``ceil(n * frac)``, where
``n`` is the class's *current* group count. The rank is stable; the **cut is
not**. Add one observation to a class and the boundary moves, so an existing
observation can change split - proven by
``tests/test_splits.py::test_growth_can_move_an_existing_observation``. That is
fine for a corpus built once, and fatal for a corpus meant to grow: an image
that was in test last month can be trained on this month, and nothing in the
metrics reveals it.

V1's module is kept, unchanged, because the V1 model and its reported numbers
have to stay reproducible. This module is V2's, and it does not import V1
assignments - a V1 split is a historical record of how that model was built,
not a claim about where an image belongs now.

The four splits
---------------
``train``       what the model fits
``validation``  model selection, early stopping, temperature fitting
``dev_test``    product development: confusion analysis, similar-species tables,
                safety cross-references, threshold and geo-prior work
``final_test``  sealed. Read once, at release, through an explicit command that
                records the access. See `SEALING` below.

The split that made V1's "read exactly once" claim untrue was not read twice by
accident - it was read repeatedly for legitimate product work, because there was
nowhere else to do that work. ``dev_test`` is that somewhere.

Stability: how it is actually guaranteed
----------------------------------------
Not by choosing a cleverer hash. By **persisting the assignment and never
recomputing it**. :func:`assign_new_groups` only ever writes rows for groups it
has never seen; existing rows are read, never updated. So growth cannot move an
existing group no matter what the assignment rule does, and the rule is free to
consult current state (for coverage) without endangering the guarantee.

The hash rule still matters for the *first* assignment of a group, and it is
deliberately a fixed-threshold hash rather than a rank-and-cut:
``stable_fraction(group_id, salt=f"{version}:{taxon_id}")`` compared against
fixed cumulative bands. Per-class salt keeps the stratification V1 needed
(every class is spread across all four splits) without making any group's
outcome depend on how many other groups its class happens to have.

Leakage grouping hierarchy
--------------------------
The unit that must never span two splits is the **leak group**: a connected
component over these edges, strongest first.

1. **Exact bytes.** Membership is keyed by ``sha256``, so two candidate records
   carrying the same photograph are the same node by construction - not merely
   deduplicated afterwards. 1,102 hashes in the current store arrive under more
   than one candidate id; under V1 a tie-break decided which record survived,
   and the losers polluted the open-set "unseen species" pool.
2. **Near duplicates.** ``dhash`` Hamming distance <= 6, computed **per taxon**
   with the exact search in :mod:`fwdata.dedupe`. A re-upload at a different
   size, or the same fish two frames apart, is the same evidence and must not
   straddle a split.

   Searching per taxon rather than globally is not an optimisation detail: only
   same-taxon pairs are ever unioned, so cross-class unions become impossible
   by construction rather than computed and then discarded. It also keeps each
   input small enough for blocked brute force, which cannot miss a pair. The
   earlier global banded index could: it required one 16-bit band to match
   exactly, which only finds every pair when the threshold is below the band
   count, and the threshold is 6 against 4 bands. Measured on the V1 CAS it
   missed 451 of the 2,960 genuine same-taxon pairs - 15%, each one a chance
   for the same photograph to sit in train and in the sealed holdout.

   The same-taxon restriction is load-bearing for a second reason. Globally,
   210,381 pairs clear the threshold and only ~3,000 are same-class; the rest
   come from a few thousand images whose dhash carries almost no signal - dark,
   uniform or low-contrast frames, 29 of them hashing to all zeroes - which
   collide with everything. Unioning those would have collapsed unrelated
   species into single leak groups. ``dataset.py dedupe`` still reports them,
   as the hash-quality signal they are rather than as label disputes.
3. **Observation / media group.** All images of one ``group_key`` - one sighting
   of one individual fish - go together. This is V1's unit and it is sound: an
   observation has exactly one taxon, which is what makes per-class
   stratification legitimate.
4. **Observer / photographer.** Deliberately **not** a union edge. Making it one
   is the OBSERVER strategy, which was measured on this corpus and rejected:
   ~5,500 photographers spanning 1,978 classes left 284 classes with no
   validation images at all. Instead observer disjointness is applied as a
   *soft, unconditional* rule at assignment time: a new group inherits the
   split any of its photographers already holds within that class.

   It used to be gated on the class having at least 8 distinct photographers,
   which was wrong in a way only growth reveals. A class assigned while it had
   5 photographers got no inheritance, so one photographer's groups landed in
   train and in the holdout; when the class later reached 8 the rule switched
   on, but immutability means those assignments can never be repaired, so that
   photographer straddles the boundary forever. The gate also made the outcome
   depend on the *current* photographer count - the same "depends on how much
   data exists right now" coupling that V2 exists to remove. Unconditional
   inheritance is state-independent and monotone: a photographer's first group
   in a class fixes the split for all of that photographer's later groups
   there. Thin classes are protected by ``MIN_GROUPS_FOR_HOLDOUT`` and the
   coverage top-up instead, which is where that protection belongs.

   Which rule decided each row is recorded in ``split_groups.rule``, so the mix
   is auditable rather than assumed.

What happens when growth merges two assigned groups
---------------------------------------------------
A new image can be a near-duplicate of images in two groups that already sit in
different splits. Moving either would violate the immutability guarantee, and
accepting the image would leak. So neither happens: the **new** image is
quarantined (recorded in ``split_quarantine``, excluded from every manifest) and
the existing assignments are left alone. Quarantine is counted and reported -
silently dropping data is how corpora rot.

SEALING
-------
``final_test`` is protected structurally, not by convention:

* :func:`export_manifests` writes ``manifest.parquet`` containing *only*
  train/validation/dev_test. Ordinary training and evaluation tooling loads that
  file and therefore **cannot** see final_test rows - not "must not", cannot.
* final_test rows go to ``sealed/final_test.parquet`` beside a ``SEALED.md``
  explaining the rule.
* Reading the sealed manifest requires :func:`open_final_test`, which refuses
  unless explicitly authorised and writes a row to ``final_test_access`` every
  time. "Read exactly once" becomes a number you can query, not a promise.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import config


def _paths():
    """Resolved lazily: tests monkeypatch ``config.PATHS``, and a module-level
    binding would quietly ignore that and write into the real data root."""
    return config.PATHS

#: Identifies this generation of split metadata inside a shared provenance DB.
#: V1 lives in ``corpus_members``; V2 lives in the tables below keyed by this.
DATASET_VERSION = "v2"

SPLITS = ("train", "validation", "dev_test", "final_test")

#: Cumulative bands are built from this order, so it must stay fixed: changing
#: the order re-bands every unassigned group (assigned ones are immutable).
DEFAULT_FRACTIONS: dict[str, float] = {
    "train": 0.70,
    "validation": 0.10,
    "dev_test": 0.10,
    "final_test": 0.10,
}

#: A class with fewer groups than this keeps all of them in train: a one-group
#: validation set measures nothing and costs a thin class its only training
#: data. V1 used the same rule for the same reason.
MIN_GROUPS_FOR_HOLDOUT = 4

#: Retained only so an existing import does not break. The observer rule is
#: unconditional now; see the module docstring for why gating it on a class's
#: current photographer count was unsound under growth.
MIN_OBSERVERS_FOR_DISJOINT = 0

#: dhash Hamming distance at or below which two images are the same evidence.
NEAR_DUPLICATE_THRESHOLD = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS split_groups (
    dataset_version VARCHAR,
    group_id        VARCHAR,
    split           VARCHAR,
    taxon_id        BIGINT,
    observer_key    VARCHAR,
    rule            VARCHAR,
    batch           VARCHAR,
    assigned_at     TIMESTAMP,
    PRIMARY KEY (dataset_version, group_id)
);

CREATE TABLE IF NOT EXISTS split_group_members (
    dataset_version VARCHAR,
    sha256          VARCHAR,
    group_id        VARCHAR,
    joined_at       TIMESTAMP,
    PRIMARY KEY (dataset_version, sha256)
);

CREATE TABLE IF NOT EXISTS split_quarantine (
    dataset_version VARCHAR,
    sha256          VARCHAR,
    reason          VARCHAR,
    detail          VARCHAR,
    quarantined_at  TIMESTAMP,
    PRIMARY KEY (dataset_version, sha256)
);

CREATE TABLE IF NOT EXISTS final_test_access (
    dataset_version VARCHAR,
    accessed_at     TIMESTAMP,
    reason          VARCHAR,
    git_commit      VARCHAR,
    rows_exposed    BIGINT
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def stable_fraction(key: str, salt: str = "") -> float:
    """Map a key to a stable value in [0, 1).

    SHA-256 rather than :func:`hash`: Python's string hash is salted per
    process, which would reshuffle assignments between runs.
    """
    h = hashlib.sha256((salt + "\x00" + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def band_of(fraction: float, fractions: dict[str, float]) -> str:
    """Which split a [0,1) fraction falls into, by fixed cumulative bands."""
    acc = 0.0
    for name in SPLITS:
        acc += fractions.get(name, 0.0)
        if fraction < acc:
            return name
    return SPLITS[-1]


@dataclass
class AssignStats:
    batch: str = ""
    images_seen: int = 0
    groups_total: int = 0
    groups_new: int = 0
    groups_existing: int = 0
    images_joined_existing: int = 0
    quarantined: int = 0
    near_duplicate_unions: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    groups_by_split: dict[str, int] = field(default_factory=dict)
    images_by_split: dict[str, int] = field(default_factory=dict)
    classes_missing: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dataset_version": DATASET_VERSION,
            "batch": self.batch,
            "images_seen": self.images_seen,
            "groups_total": self.groups_total,
            "groups_new": self.groups_new,
            "groups_existing": self.groups_existing,
            "images_joined_existing": self.images_joined_existing,
            "quarantined": self.quarantined,
            "near_duplicate_unions": self.near_duplicate_unions,
            "by_rule": self.by_rule,
            "groups_by_split": self.groups_by_split,
            "images_by_split": self.images_by_split,
            "classes_missing": self.classes_missing,
        }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smaller id wins, so component roots do not depend on insert order.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


class V2SplitStore:
    """Persistent, immutable V2 split assignments in the provenance database."""

    def __init__(self, db_path: Path | None = None, *, read_only: bool = False) -> None:
        self.db_path = Path(db_path or _paths().provenance_db)
        self.con = duckdb.connect(str(self.db_path), read_only=read_only)
        self.con.execute("PRAGMA threads=8")
        if not read_only:
            self.con.execute(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "V2SplitStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def _build_eligible(self, *, limit: int | None = None) -> None:
        """Images that may be assigned at all.

        Deliberately *not* the corpus: the class-size bar belongs to corpus
        construction, which changes as the dataset grows. Split membership is
        decided for every admissible image so that a class crossing the bar
        later does not need a single new assignment decision.
        """
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE v2_conflicting AS
            SELECT sha256 FROM provenance
            WHERE species_taxon_id IS NOT NULL AND sha256 IS NOT NULL
            GROUP BY sha256 HAVING count(DISTINCT species_taxon_id) > 1
            """
        )
        # Every (image, observation, photographer) relationship, one row each.
        #
        # group_key and observer_key are namespaced by source. They are raw
        # provider identifiers - iNaturalist observer_key is a bare integer
        # ("2046206"), 308,227 of 308,227 of them - so the moment a second
        # source lands, its user 2046206 would be silently treated as the same
        # photographer, and its observation ids could merge two species into one
        # leak group. candidate_id is already '<source>:<record_id>', so the
        # namespace is available without touching the candidates table.
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE v2_links AS
            SELECT DISTINCT
                   pr.sha256,
                   pr.species_taxon_id AS taxon_id,
                   split_part(pr.candidate_id, ':', 1) || ':' || pr.group_key
                       AS group_key,
                   split_part(pr.candidate_id, ':', 1) || ':' || pr.observer_key
                       AS observer_key
            FROM provenance pr
            WHERE pr.species_taxon_id IS NOT NULL
              AND pr.sha256 IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM v2_conflicting c WHERE c.sha256 = pr.sha256
              )
            {f'LIMIT {int(limit)}' if limit else ''}
            """
        )
        # One row per distinct image, for the attributes that genuinely are
        # per-image. taxon_id is safe to collapse because v2_conflicting already
        # removed every sha carrying more than one; dhash is a function of the
        # bytes, so identical bytes give an identical hash. The *relationships*
        # deliberately do not collapse here - see v2_links. Taking any_value of
        # group_key dropped 923 images' second observation on the real store,
        # 405 of them into observations holding other images, which is exactly
        # the edge that keeps a re-post out of the release holdout.
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE v2_eligible AS
            SELECT l.sha256,
                   min(l.taxon_id) AS taxon_id,
                   max(s.dhash)    AS dhash
            FROM v2_links l
            LEFT JOIN (
                SELECT DISTINCT pr.sha256, st.dhash
                FROM provenance pr JOIN stored st USING (candidate_id)
            ) s USING (sha256)
            GROUP BY l.sha256
            """
        )

    # ------------------------------------------------------------------
    # grouping
    # ------------------------------------------------------------------
    def _components(self, *, near_duplicates: bool, log) -> tuple[dict[str, str], int]:
        """Connected components over the leak-group edges.

        Returns ``(sha256 -> component_root, near_duplicate_unions)``.
        """
        rows = self.con.execute(
            "SELECT sha256, taxon_id, dhash FROM v2_eligible"
        ).fetchall()
        uf = _UnionFind()
        taxon_of: dict[str, int] = {}
        for sha, taxon, _dhash in rows:
            uf.find(sha)
            taxon_of[sha] = taxon

        # edge 3: every image of one observation, over *all* links rather than
        # one arbitrarily chosen per image.
        by_obs: dict[str, str] = {}
        obs_links = self.con.execute(
            "SELECT group_key, sha256 FROM v2_links "
            "WHERE group_key IS NOT NULL ORDER BY group_key, sha256"
        ).fetchall()
        for group_key, sha in obs_links:
            if sha not in taxon_of:
                continue
            first = by_obs.setdefault(group_key, sha)
            uf.union(first, sha)
        log(f"  {len(rows):,} distinct images, {len(by_obs):,} observations")

        same_class = 0
        if near_duplicates:
            from .dedupe import near_duplicate_pairs

            # Per taxon, not globally. Two reasons, and the first is the one
            # that matters: only same-taxon pairs are ever unioned, so a global
            # search computes millions of cross-class pairs it will discard.
            # The second is that per-taxon inputs are small enough for the
            # exact blocked brute force in `dedupe`, which cannot miss a pair.
            by_taxon: dict[int, list[tuple[str, str]]] = {}
            for sha, taxon, dhash in rows:
                if dhash:
                    by_taxon.setdefault(taxon, []).append((sha, dhash))
            for items in by_taxon.values():
                for a, b, _dist in near_duplicate_pairs(
                    items, threshold=NEAR_DUPLICATE_THRESHOLD
                ):
                    uf.union(a, b)              # edge 2, same taxon by construction
                    same_class += 1
            log(f"  near-duplicate pairs unioned (same taxon, exact): "
                f"{same_class:,}")

        return {sha: uf.find(sha) for sha, *_ in rows}, same_class

    # ------------------------------------------------------------------
    # assignment
    # ------------------------------------------------------------------
    def assign(
        self,
        *,
        batch: str,
        fractions: dict[str, float] | None = None,
        near_duplicates: bool = True,
        coverage_topup: bool = True,
        limit: int | None = None,
        log=print,
    ) -> AssignStats:
        """Assign every not-yet-assigned image, leaving existing rows untouched.

        This is the only way rows enter ``split_groups``/``split_group_members``
        and it never issues an UPDATE against either. Re-running it is safe and
        is how each new acquisition batch is folded in.
        """
        fractions = fractions or DEFAULT_FRACTIONS
        stats = AssignStats(batch=batch)

        log(f"assigning V2 splits (batch={batch!r})")
        self._build_eligible(limit=limit)
        stats.images_seen = self.con.execute(
            "SELECT count(*) FROM v2_eligible"
        ).fetchone()[0]

        comp, stats.near_duplicate_unions = self._components(
            near_duplicates=near_duplicates, log=log
        )

        taxon_of_sha = dict(
            self.con.execute("SELECT sha256, taxon_id FROM v2_eligible").fetchall()
        )
        # Every photographer of every image, not one per image: an image can
        # legitimately reach us under two accounts (8 such images on the real
        # store), and the observer rule has to see both or it ties the group to
        # an arbitrary one of them.
        observers_of_sha: dict[str, set[str]] = {}
        for sha, observer in self.con.execute(
            "SELECT sha256, observer_key FROM v2_links WHERE observer_key IS NOT NULL"
        ).fetchall():
            observers_of_sha.setdefault(sha, set()).add(observer)

        existing_member = dict(
            self.con.execute(
                "SELECT sha256, group_id FROM split_group_members "
                "WHERE dataset_version = ?",
                [DATASET_VERSION],
            ).fetchall()
        )
        existing_group = {
            gid: split
            for gid, split in self.con.execute(
                "SELECT group_id, split FROM split_groups WHERE dataset_version = ?",
                [DATASET_VERSION],
            ).fetchall()
        }
        quarantined = {
            row[0]
            for row in self.con.execute(
                "SELECT sha256 FROM split_quarantine WHERE dataset_version = ?",
                [DATASET_VERSION],
            ).fetchall()
        }

        # Gather components.
        members: dict[str, list[str]] = {}
        for sha, root in comp.items():
            members.setdefault(root, []).append(sha)
        stats.groups_total = len(members)

        # Observer -> split, per class, from what is already assigned. Ordered
        # so the earliest assignment wins deterministically; an unordered
        # setdefault let whichever row DuckDB returned first decide, which is
        # not something a rebuild can reproduce.
        observer_split: dict[tuple[int, str], str] = {}
        for taxon, observer, split in self.con.execute(
            "SELECT taxon_id, observer_key, split FROM split_groups "
            "WHERE dataset_version = ? AND observer_key IS NOT NULL "
            "ORDER BY assigned_at, group_id",
            [DATASET_VERSION],
        ).fetchall():
            for one in observer.split("\x1f"):
                observer_split.setdefault((taxon, one), split)

        observers_per_class = dict(
            self.con.execute(
                "SELECT taxon_id, count(DISTINCT observer_key) FROM v2_links "
                "GROUP BY taxon_id"
            ).fetchall()
        )
        def group_taxon(shas: list[str]) -> int:
            return taxon_of_sha[shas[0]]

        def group_observers(shas: list[str]) -> list[str]:
            """Every photographer contributing to this leak group, sorted."""
            out: set[str] = set()
            for s in shas:
                out |= observers_of_sha.get(s, set())
            return sorted(out)

        groups_per_class: dict[int, int] = {}
        for root, shas in members.items():
            taxon = group_taxon(shas)
            groups_per_class[taxon] = groups_per_class.get(taxon, 0) + 1

        new_groups: list[tuple] = []
        new_members: list[tuple] = []
        new_quarantine: list[tuple] = []
        now = _utcnow()

        # Which classes already have coverage, so top-up only fires where needed.
        covered: dict[tuple[int, str], int] = {}
        for taxon, split, n in self.con.execute(
            "SELECT taxon_id, split, count(*) FROM split_groups "
            "WHERE dataset_version = ? GROUP BY taxon_id, split",
            [DATASET_VERSION],
        ).fetchall():
            covered[(taxon, split)] = n

        # Deterministic order: component root, so a rebuild from the same data
        # makes the same decisions regardless of dict iteration order.
        pending: list[tuple[int, float, str, list[str]]] = []
        for root in sorted(members):
            shas = members[root]
            taxon = group_taxon(shas)
            touched = {existing_member[s] for s in shas if s in existing_member}

            if len(touched) > 1:
                splits = {existing_group.get(g) for g in touched}
                if len(splits) > 1:
                    # Growth bridged two groups sitting in different splits.
                    # Never move the old ones; refuse the new images instead.
                    detail = ", ".join(
                        f"{g}={existing_group.get(g)}" for g in sorted(touched)
                    )
                    for s in shas:
                        if s not in existing_member and s not in quarantined:
                            new_quarantine.append(
                                (DATASET_VERSION, s, "bridges_split_conflict",
                                 detail, now)
                            )
                    stats.groups_existing += 1
                    continue

            if touched:
                # Component already has a home: attach any new images to it.
                gid = sorted(touched)[0]
                for s in shas:
                    if s not in existing_member:
                        new_members.append((DATASET_VERSION, s, gid, now))
                        stats.images_joined_existing += 1
                stats.groups_existing += 1
                continue

            pending.append((taxon, stable_fraction(root, f"{DATASET_VERSION}:{taxon}"),
                            root, shas))

        # --- first assignment for never-before-seen groups -----------------
        assigned_now: dict[tuple[int, str], list[tuple[str, float]]] = {}
        decided: dict[str, tuple[str, str]] = {}     # root -> (split, rule)
        observer_of_group: dict[str, str | None] = {}
        taxon_of_group: dict[str, int] = {}
        groups_per_observer: dict[tuple[int, str | None], int] = {}

        for taxon, frac, root, shas in pending:
            observers = group_observers(shas)
            rule = "hash"
            split = band_of(frac, fractions)

            if groups_per_class.get(taxon, 0) < MIN_GROUPS_FOR_HOLDOUT:
                split, rule = "train", "thin_class"
            else:
                # Inherit from *any* photographer of this group who already has
                # a split in this class. Unconditional, deliberately: gating it
                # on a class having >= N photographers made the rule activate
                # only once a class had grown, so every photographer assigned
                # before that point could straddle train and the holdout
                # permanently - and immutability means it can never be repaired.
                # The gate also made the outcome depend on the current
                # photographer count, which is exactly the "depends on how much
                # data exists right now" coupling V2 exists to remove.
                inherited = next(
                    (observer_split[(taxon, o)] for o in observers
                     if (taxon, o) in observer_split),
                    None,
                )
                if inherited:
                    split, rule = inherited, "observer_inherit"

            decided[root] = (split, rule)
            assigned_now.setdefault((taxon, split), []).append((root, frac))
            observer_of_group[root] = observers
            taxon_of_group[root] = taxon
            for one in observers:
                groups_per_observer[(taxon, one)] = (
                    groups_per_observer.get((taxon, one), 0) + 1
                )
                observer_split.setdefault((taxon, one), split)

        # --- deterministic coverage top-up ---------------------------------
        # A per-class hash is unbiased, and once every group of a photographer
        # follows that photographer, a class effectively draws once per
        # photographer rather than once per group. Classes with few
        # photographers therefore miss splits often: measured on the V1 CAS,
        # 24 classes had no validation and 9 had no *train* data at all before
        # this pass existed in its current form.
        #
        # So where a class has photographers to spare, move an entire
        # photographer into the empty split, choosing the one whose mean hash
        # fraction sits nearest that split's band. Deterministic given the
        # batch, and it touches nothing already persisted.
        if coverage_topup:
            persisted_observer_groups: dict[tuple[int, str], int] = {}
            for taxon_id, observer, n in self.con.execute(
                "SELECT taxon_id, observer_key, count(*) FROM split_groups "
                "WHERE dataset_version = ? AND observer_key IS NOT NULL "
                "GROUP BY taxon_id, observer_key",
                [DATASET_VERSION],
            ).fetchall():
                for one in observer.split(""):
                    persisted_observer_groups[(taxon_id, one)] = (
                        persisted_observer_groups.get((taxon_id, one), 0) + int(n)
                    )

            # The unit moved is the *photographer*, not the group, because
            # that is the unit inheritance ties together. Moving one group of a
            # photographer who has others would put them on both sides of the
            # boundary - buying coverage with exactly the leak the observer rule
            # exists to prevent. Moving all of a photographer's pending groups
            # keeps disjointness intact and still fills the split.
            #
            # Only photographers with nothing persisted yet are movable:
            # anything already written is immutable.
            by_class: dict[int, dict[str, list[tuple[str, float]]]] = {}
            for taxon, frac, root, _shas in pending:
                primary = (observer_of_group.get(root) or [None])[0]
                by_class.setdefault(taxon, {}).setdefault(primary, []).append(
                    (root, frac)
                )

            for taxon, by_observer in by_class.items():
                if groups_per_class.get(taxon, 0) < MIN_GROUPS_FOR_HOLDOUT:
                    continue

                def movable(observer: str | None) -> bool:
                    """Movable = every one of this photographer's groups in this
                    class is still in train, and none is already persisted.

                    The rule that put them there is irrelevant - most groups are
                    `observer_inherit` precisely because inheritance is doing its
                    job, and requiring `hash` here meant the top-up almost never
                    found a donor. What matters is that the whole set moves
                    together and that nothing immutable is touched.
                    """
                    if persisted_observer_groups.get((taxon, observer), 0):
                        return False
                    return all(
                        decided[r][0] == "train" for r, _f in by_observer[observer]
                    )

                bands = {}
                acc = 0.0
                for name in SPLITS:
                    lo, acc = acc, acc + fractions.get(name, 0.0)
                    bands[name] = (lo + acc) / 2.0

                for name in SPLITS:
                    have = covered.get((taxon, name), 0) + sum(
                        1 for groups in by_observer.values()
                        for r, _f in groups if decided[r][0] == name
                    )
                    if have:
                        continue
                    candidates = [o for o in by_observer if movable(o)]
                    # Never strip train empty to fill a holdout: a class with no
                    # training data cannot be learned at all, which is strictly
                    # worse than a class that cannot be measured.
                    if name != "train":
                        remaining_train = [
                            o for o in by_observer
                            if any(decided[r][0] == "train"
                                   for r, _f in by_observer[o])
                        ]
                        if len(remaining_train) <= 1:
                            continue
                    if not candidates:
                        continue
                    mid = bands[name]

                    def distance(observer):
                        groups = by_observer[observer]
                        mean = sum(f for _r, f in groups) / len(groups)
                        return (abs(mean - mid), str(observer))

                    pick = min(candidates, key=distance)
                    for r, _f in by_observer[pick]:
                        decided[r] = (name, "coverage_topup")
                        for one in (observer_of_group.get(r) or []):
                            observer_split[(taxon, one)] = name

        for taxon, frac, root, shas in pending:
            split, rule = decided[root]
            # A group can have more than one photographer when the same bytes
            # reached us under two accounts. Store all of them, separated by
            # , so re-seeding observer_split on a later run sees every tie
            # rather than an arbitrary one - losing the second one there is how
            # a photographer would come to straddle two splits.
            observers = observer_of_group.get(root) or []
            new_groups.append(
                (DATASET_VERSION, root, split, taxon,
                 "".join(observers) or None, rule, batch, now)
            )
            for s in shas:
                new_members.append((DATASET_VERSION, s, root, now))
            stats.groups_new += 1
            stats.by_rule[rule] = stats.by_rule.get(rule, 0) + 1

        self._bulk_insert(
            "split_groups", new_groups,
            ["dataset_version", "group_id", "split", "taxon_id", "observer_key",
             "rule", "batch", "assigned_at"],
        )
        self._bulk_insert(
            "split_group_members", new_members,
            ["dataset_version", "sha256", "group_id", "joined_at"],
        )
        self._bulk_insert(
            "split_quarantine", new_quarantine,
            ["dataset_version", "sha256", "reason", "detail", "quarantined_at"],
        )
        stats.quarantined = len(new_quarantine)

        self._collect_stats(stats)
        log(f"  groups: {stats.groups_new:,} new, {stats.groups_existing:,} existing")
        log(f"  rules : {stats.by_rule}")
        log(f"  groups by split: {stats.groups_by_split}")
        log(f"  images by split: {stats.images_by_split}")
        if stats.quarantined:
            log(f"  QUARANTINED {stats.quarantined:,} images that would have "
                f"bridged two splits")
        return stats

    def _bulk_insert(self, table: str, rows: list[tuple], columns: list[str]) -> None:
        """Insert via Arrow rather than executemany.

        Row-at-a-time inserts of the ~300k membership rows this produces on the
        V1 CAS take longer than every other step combined - the first full run
        was still inserting after twenty minutes. Arrow makes it seconds, and
        this path runs on every acquisition batch, not just once.
        """
        if not rows:
            return
        import pyarrow as pa

        table_arrow = pa.table(
            {name: pa.array(col) for name, col in zip(columns, zip(*rows))}
        )
        self.con.register("_bulk_rows", table_arrow)
        try:
            self.con.execute(f"INSERT INTO {table} SELECT * FROM _bulk_rows")
        finally:
            self.con.unregister("_bulk_rows")

    def _collect_stats(self, stats: AssignStats) -> None:
        stats.groups_by_split = {
            s: int(n)
            for s, n in self.con.execute(
                "SELECT split, count(*) FROM split_groups WHERE dataset_version = ? "
                "GROUP BY split ORDER BY split",
                [DATASET_VERSION],
            ).fetchall()
        }
        stats.images_by_split = {
            s: int(n)
            for s, n in self.con.execute(
                """
                SELECT g.split, count(*) FROM split_group_members m
                JOIN split_groups g USING (dataset_version, group_id)
                WHERE m.dataset_version = ? GROUP BY g.split ORDER BY g.split
                """,
                [DATASET_VERSION],
            ).fetchall()
        }
        total_classes = self.con.execute(
            "SELECT count(DISTINCT taxon_id) FROM split_groups WHERE dataset_version = ?",
            [DATASET_VERSION],
        ).fetchone()[0]
        for name in SPLITS:
            with_split = self.con.execute(
                "SELECT count(DISTINCT taxon_id) FROM split_groups "
                "WHERE dataset_version = ? AND split = ?",
                [DATASET_VERSION, name],
            ).fetchone()[0]
            stats.classes_missing[name] = int(total_classes - with_split)

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------
    def verify(self) -> dict:
        """Structural checks that must hold for the assignment to mean anything."""
        groups_spanning = self.con.execute(
            """
            SELECT count(*) FROM (
                SELECT m.group_id FROM split_group_members m
                JOIN split_groups g USING (dataset_version, group_id)
                WHERE m.dataset_version = ?
                GROUP BY m.group_id HAVING count(DISTINCT g.split) > 1
            )
            """,
            [DATASET_VERSION],
        ).fetchone()[0]
        # An image is one node, so a hash spanning splits would mean the table
        # itself is corrupt rather than the algorithm being wrong.
        hashes_spanning = self.con.execute(
            """
            SELECT count(*) FROM (
                SELECT sha256 FROM split_group_members
                WHERE dataset_version = ?
                GROUP BY sha256 HAVING count(DISTINCT group_id) > 1
            )
            """,
            [DATASET_VERSION],
        ).fetchone()[0]
        orphan_members = self.con.execute(
            """
            SELECT count(*) FROM split_group_members m
            WHERE m.dataset_version = ? AND NOT EXISTS (
                SELECT 1 FROM split_groups g
                WHERE g.dataset_version = m.dataset_version
                  AND g.group_id = m.group_id)
            """,
            [DATASET_VERSION],
        ).fetchone()[0]
        bad_split = self.con.execute(
            "SELECT count(*) FROM split_groups WHERE dataset_version = ? "
            f"AND split NOT IN ({','.join('?' * len(SPLITS))})",
            [DATASET_VERSION, *SPLITS],
        ).fetchone()[0]
        return {
            "groups_spanning_splits": int(groups_spanning),
            "hashes_spanning_splits": int(hashes_spanning),
            "orphan_members": int(orphan_members),
            "invalid_split_values": int(bad_split),
        }

    def fingerprint(self) -> str:
        """A hash of every assignment, for proving a rebuild changed nothing."""
        rows = self.con.execute(
            "SELECT group_id, split FROM split_groups WHERE dataset_version = ? "
            "ORDER BY group_id",
            [DATASET_VERSION],
        ).fetchall()
        h = hashlib.sha256()
        for gid, split in rows:
            h.update(f"{gid}\x00{split}\n".encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_manifests(
        self,
        corpus: str,
        *,
        out_dir: Path | None = None,
        min_images_per_class: int = 40,
        min_observations_per_class: int = 25,
        log=print,
    ) -> dict:
        """Write the open manifest and, separately, the sealed final_test one.

        The open manifest is the only file ordinary tooling reads, and it does
        not contain final_test rows at all. That is the sealing mechanism: not a
        flag a script can forget to check, but data a script cannot reach.
        """
        out_dir = Path(out_dir or _paths().artifacts / corpus)
        sealed_dir = out_dir / "sealed"
        out_dir.mkdir(parents=True, exist_ok=True)
        sealed_dir.mkdir(parents=True, exist_ok=True)

        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE v2_rows AS
            SELECT p.sha256, p.species_taxon_id AS taxon_id,
                   p.accepted_scientific_name AS name,
                   g.split, g.group_id, g.observer_key, g.rule,
                   any_value(p.cas_path) AS cas_path,
                   any_value(p.width) AS width, any_value(p.height) AS height,
                   any_value(p.latitude) AS latitude,
                   any_value(p.longitude) AS longitude,
                   any_value(p.observed_on) AS observed_on,
                   any_value(p.license) AS license
            FROM provenance p
            JOIN split_group_members m
              ON m.sha256 = p.sha256 AND m.dataset_version = ?
            JOIN split_groups g
              ON g.group_id = m.group_id AND g.dataset_version = m.dataset_version
            WHERE NOT EXISTS (
                SELECT 1 FROM split_quarantine q
                WHERE q.dataset_version = m.dataset_version AND q.sha256 = p.sha256
            )
            GROUP BY p.sha256, p.species_taxon_id, p.accepted_scientific_name,
                     g.split, g.group_id, g.observer_key, g.rule
            """,
            [DATASET_VERSION],
        )
        # The class bar is measured on train only: whether a class is learnable
        # must not depend on how much holdout it happened to receive.
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE v2_classes AS
            SELECT taxon_id, any_value(name) AS name,
                   count(*) FILTER (WHERE split = 'train') AS train_images,
                   count(DISTINCT group_id) AS groups,
                   count(DISTINCT observer_key) AS observers
            FROM v2_rows GROUP BY taxon_id
            HAVING count(*) FILTER (WHERE split = 'train') >= {int(min_images_per_class)}
               AND count(DISTINCT group_id) >= {int(min_observations_per_class)}
            """
        )
        self.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE v2_class_index AS
            SELECT taxon_id, name, train_images, groups, observers,
                   row_number() OVER (ORDER BY name) - 1 AS class_id
            FROM v2_classes
            """
        )

        open_path = out_dir / "manifest.parquet"
        sealed_path = sealed_dir / "final_test.parquet"
        for path, where in (
            (open_path, "r.split <> 'final_test'"),
            (sealed_path, "r.split = 'final_test'"),
        ):
            self.con.execute(
                f"""
                COPY (
                    SELECT r.sha256, ci.class_id, r.taxon_id, r.split, r.name,
                           r.cas_path, r.width, r.height, r.latitude, r.longitude,
                           r.observed_on, r.license, r.group_id, r.observer_key
                    FROM v2_rows r JOIN v2_class_index ci USING (taxon_id)
                    WHERE {where}
                    ORDER BY r.split, ci.class_id, r.sha256
                ) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION zstd)
                """
            )

        rows = self.con.execute(
            "SELECT class_id, taxon_id, name, train_images, groups, observers "
            "FROM v2_class_index ORDER BY class_id"
        ).fetchall()
        (out_dir / "labels.json").write_text(
            json.dumps(
                {
                    "corpus": corpus,
                    "dataset_version": DATASET_VERSION,
                    "num_classes": len(rows),
                    "classes": [
                        {"class_id": c, "fw_taxon_id": t, "scientific_name": n,
                         "train_images": ti, "groups": g, "observers": o}
                        for c, t, n, ti, g, o in rows
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (sealed_dir / "SEALED.md").write_text(
            "# Sealed: final_test\n\n"
            "`final_test.parquet` is the release holdout for "
            f"`{corpus}` (dataset_version `{DATASET_VERSION}`).\n\n"
            "It is deliberately **not** in `../manifest.parquet`, so training\n"
            "and evaluation tooling cannot read it by accident - not as a\n"
            "matter of discipline, but because the rows are not in the file\n"
            "those tools load.\n\n"
            "Do not use it for confusion analysis, similar-species tables,\n"
            "safety cross-references, threshold tuning, calibration or\n"
            "geo-prior work. Use `dev_test` for all of that.\n\n"
            "Read it only through:\n\n"
            "    python tools/dataset.py v2-release-eval --corpus "
            f"{corpus} --reason \"<why>\" --i-am-releasing\n\n"
            "Every access appends a row to `final_test_access` in the\n"
            "provenance database, so how many times this has been read is a\n"
            "query rather than a memory.\n",
            encoding="utf-8",
        )

        counts = {
            s: int(n)
            for s, n in self.con.execute(
                "SELECT r.split, count(*) FROM v2_rows r "
                "JOIN v2_class_index ci USING (taxon_id) "
                "GROUP BY r.split ORDER BY r.split"
            ).fetchall()
        }
        result = {
            "corpus": corpus,
            "classes": len(rows),
            "images_by_split": counts,
            "open_manifest": str(open_path),
            "sealed_manifest": str(sealed_path),
        }
        log(f"  classes: {len(rows):,}")
        log(f"  images by split: {counts}")
        log(f"  wrote {open_path}  (train/validation/dev_test only)")
        log(f"  wrote {sealed_path}  (SEALED - release evaluation only)")
        return result

    # ------------------------------------------------------------------
    # the sealed door
    # ------------------------------------------------------------------
    def open_final_test(
        self, corpus: str, *, reason: str, authorised: bool,
        out_dir: Path | None = None,
    ) -> Path:
        """Return the sealed manifest path, recording the access.

        Refuses unless explicitly authorised. The record is the point: "the
        release test was read once" should be checkable against a table rather
        than believed.
        """
        if not authorised:
            raise PermissionError(
                "final_test is sealed. It is the release holdout and every read "
                "of it costs some of its value as an unbiased estimate.\n"
                "Use dev_test for development work. If this really is a release "
                "evaluation, pass --i-am-releasing and give --reason."
            )
        if not reason or not reason.strip():
            raise ValueError("an access to final_test must carry a --reason")

        out_dir = Path(out_dir or _paths().artifacts / corpus)
        sealed = out_dir / "sealed" / "final_test.parquet"
        if not sealed.exists():
            raise FileNotFoundError(f"{sealed} does not exist - export manifests first")

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=Path(__file__).resolve().parents[2],
            ).stdout.strip() or "unknown"
        except Exception:
            commit = "unknown"

        n = self.con.execute(
            f"SELECT count(*) FROM read_parquet('{sealed.as_posix()}')"
        ).fetchone()[0]
        self.con.execute(
            "INSERT INTO final_test_access VALUES (?,?,?,?,?)",
            [DATASET_VERSION, _utcnow(), reason.strip(), commit, int(n)],
        )
        return sealed

    def access_log(self) -> list[tuple]:
        return self.con.execute(
            "SELECT accessed_at, reason, git_commit, rows_exposed "
            "FROM final_test_access WHERE dataset_version = ? ORDER BY accessed_at",
            [DATASET_VERSION],
        ).fetchall()


def release_eval_authorised(flag: bool) -> bool:
    """Both a deliberate flag and an environment opt-in, or neither."""
    return bool(flag) or os.environ.get("FISHERWIKI_RELEASE_EVAL") == "1"
