"""Duplicate and leakage detection over the stored corpus.

Three distinct problems, which are easy to conflate and have different fixes:

**Exact duplicates** - identical bytes. Impossible inside the CAS by
construction (the path is the hash), but the *same image* can legitimately
arrive under two candidate ids when two sources carry the same photograph. The
fix is to keep one and record the alias, so attribution still covers both.

**Near duplicates within a class** - the same fish photographed twice, or one
photo re-uploaded at a different size. These inflate a class's apparent support
without adding information. They are *not* deleted: a burst of near-identical
frames is realistic training data. They are reported so that the per-class
counts used for the evidence bar can be adjusted.

**Cross-split leakage** - a near-duplicate pair where one image is in train and
the other in val or test. This is the serious one: it silently inflates every
reported number and nothing in the metrics reveals it. Splits are grouped by
observation and photographer precisely to prevent this, and this check verifies
that the grouping actually worked.

Matching uses the perceptual hashes computed at download time, so no image is
re-decoded. Candidate pairs are found by banding the 64-bit hashes into 16-bit
buckets, which is far cheaper than the all-pairs comparison that 300k images
would otherwise require.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import PATHS
from .provenance import ProvenanceDB

#: Hamming distance at or below which two images are considered near-duplicates.
DEFAULT_THRESHOLD = 6

#: Number of 16-bit bands a 64-bit hash is split into for candidate generation.
#: Two hashes within `threshold` bits must agree exactly on at least one band
#: when threshold < bands, which makes this an exact (not approximate) filter
#: for our threshold of 6 over 4 bands.
BANDS = 4
BAND_BITS = 16


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(h: int) -> list[tuple[int, int]]:
    return [((h >> (i * BAND_BITS)) & 0xFFFF, i) for i in range(BANDS)]


def near_duplicate_pairs(
    items: list[tuple[str, str]],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_bucket: int = 5000,
) -> list[tuple[str, str, int]]:
    """``[(key_a, key_b, distance), ...]`` for hashes within ``threshold``.

    ``items`` is ``[(key, dhash_hex), ...]``; keys are returned as given, so the
    caller decides whether a key is a candidate id, a sha256 or anything else.
    Split out of :func:`find_duplicates` so the V2 splitter can build leak
    groups from the same banded index rather than reimplementing it - two
    implementations of "are these the same photograph" would eventually
    disagree, and the split is the place where that disagreement would be
    silent and expensive.
    """
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    keys: list[str] = []
    hashes: list[int] = []
    for i, (key, dhash) in enumerate(items):
        h = int(dhash, 16)
        keys.append(key)
        hashes.append(h)
        for band in _bands(h):
            buckets[band].append(i)

    out: list[tuple[str, str, int]] = []
    for (_value, band_idx), members in buckets.items():
        if len(members) < 2 or len(members) > max_bucket:
            # A bucket with thousands of members is a degenerate hash (a solid
            # background, usually); comparing it all-pairs costs more than the
            # information is worth.
            continue
        for a_i in range(len(members)):
            a = members[a_i]
            ha = hashes[a]
            for b_i in range(a_i + 1, len(members)):
                b = members[b_i]
                hb = hashes[b]
                # A pair that agrees on several bands turns up in each of them.
                # Emit it only from the lowest such band instead of keeping a
                # global set of every pair already seen: on the current store
                # that set reaches ~17M tuples and costs gigabytes, which is
                # what made the first full run unusable rather than merely slow.
                if any(
                    ((ha >> (j * BAND_BITS)) & 0xFFFF)
                    == ((hb >> (j * BAND_BITS)) & 0xFFFF)
                    for j in range(band_idx)
                ):
                    continue
                d = _hamming(ha, hb)
                if d <= threshold:
                    out.append((keys[a], keys[b], d) if a < b
                               else (keys[b], keys[a], d))
    return out


def find_duplicates(
    *,
    threshold: int = DEFAULT_THRESHOLD,
    corpus: str | None = None,
    db_path: Path | None = None,
    log=print,
) -> dict:
    """Report exact duplicates, near-duplicates and cross-split leakage."""
    db = ProvenanceDB(db_path, read_only=True)
    try:
        if corpus:
            rows = db.con.execute(
                """
                SELECT s.candidate_id, s.sha256, s.dhash, s.phash,
                       c.accepted_scientific_name, m.split, c.group_key, c.observer_key
                FROM stored s
                JOIN candidates c USING (candidate_id)
                JOIN corpus_members m USING (candidate_id)
                WHERE m.corpus = ? AND s.dhash IS NOT NULL
                """,
                [corpus],
            ).fetchall()
        else:
            rows = db.con.execute(
                """
                SELECT s.candidate_id, s.sha256, s.dhash, s.phash,
                       c.accepted_scientific_name, NULL, c.group_key, c.observer_key
                FROM stored s
                JOIN candidates c USING (candidate_id)
                WHERE s.dhash IS NOT NULL
                """
            ).fetchall()
    finally:
        db.close()

    log(f"examining {len(rows):,} stored images")
    if not rows:
        return {"images": 0}

    # --- exact duplicates ---------------------------------------------------
    by_sha: dict[str, list[str]] = defaultdict(list)
    for cid, sha, *_ in rows:
        by_sha[sha].append(cid)
    exact = {k: v for k, v in by_sha.items() if len(v) > 1}
    log(f"exact duplicate groups (same bytes, >1 candidate id): {len(exact):,}")

    # --- near duplicates via banded hashes ----------------------------------
    by_cid = {r[0]: r for r in rows}
    near_pairs = near_duplicate_pairs(
        [(r[0], r[2]) for r in rows], threshold=threshold
    )

    log(f"near-duplicate pairs (dhash distance <= {threshold}): {len(near_pairs):,}")

    # --- classify the pairs -------------------------------------------------
    same_class = 0
    cross_class = 0
    cross_split = []
    same_group = 0
    cross_observer = 0
    for a, b, d in near_pairs:
        _, _, _, _, name_a, split_a, group_a, obs_a = by_cid[a]
        _, _, _, _, name_b, split_b, group_b, obs_b = by_cid[b]
        if name_a == name_b:
            same_class += 1
        else:
            cross_class += 1
        if group_a == group_b:
            same_group += 1
        if obs_a != obs_b:
            cross_observer += 1
        if split_a and split_b and split_a != split_b:
            cross_split.append({
                "a": a, "b": b,
                "split_a": split_a, "split_b": split_b,
                "name_a": name_a, "name_b": name_b,
                "distance": d,
            })

    report = {
        "images": len(rows),
        "exact_duplicate_groups": len(exact),
        "near_duplicate_pairs": len(near_pairs),
        "near_pairs_same_class": same_class,
        "near_pairs_cross_class": cross_class,
        "near_pairs_same_observation": same_group,
        "near_pairs_different_photographer": cross_observer,
        "cross_split_leaks": len(cross_split),
        "cross_split_examples": cross_split[:25],
        "threshold": threshold,
    }

    log(f"  same species        : {same_class:,}")
    log(f"  different species   : {cross_class:,}  "
        f"(these are label-quality suspects, not just duplicates)")
    log(f"  same observation    : {same_group:,}")
    log(f"  different photographer: {cross_observer:,}")
    if corpus:
        log(f"  CROSS-SPLIT LEAKS   : {len(cross_split):,}")
        if cross_split:
            log("")
            log("  Leakage found. Every accuracy number from this corpus is")
            log("  inflated until this is fixed.")
            for ex in cross_split[:5]:
                log(f"    {ex['name_a']} [{ex['split_a']}] ~ "
                    f"{ex['name_b']} [{ex['split_b']}] d={ex['distance']}")
    return report
