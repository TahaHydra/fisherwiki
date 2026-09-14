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

#: Bands the 64-bit hash is split into for multi-index hashing. See
#: :func:`near_duplicate_pairs` for why this is *not* the "one band must match
#: exactly" scheme it used to be.
BANDS = 4
BAND_BITS = 16

#: Above this many hashes, switch from blocked brute force to multi-index
#: hashing. Brute force is O(n^2) but has no index overhead and is trivially
#: correct, which is what we want for the per-taxon calls the splitter makes.
BRUTE_FORCE_MAX = 8192


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(h: int) -> list[tuple[int, int]]:
    return [((h >> (i * BAND_BITS)) & 0xFFFF, i) for i in range(BANDS)]


def _as_uint64(items: list[tuple[str, str]]):
    import numpy as np

    return np.array([int(d, 16) for _k, d in items], dtype=np.uint64)


def _pairs_brute_force(keys, h, threshold):
    """Every pair, blocked so memory stays bounded. Exact by construction."""
    import numpy as np

    n = len(h)
    out: list[tuple[str, str, int]] = []
    # Keep each comparison block near 32M cells; at 1 byte per popcount result
    # that is ~32 MB of scratch regardless of how large n gets.
    block = max(1, min(n, 32_000_000 // max(n, 1)))
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d = np.bitwise_count(np.bitwise_xor(h[i0:i1, None], h[None, i0:]))
        ii, jj = np.nonzero(d <= threshold)
        jj = jj + i0
        ii = ii + i0
        keep = ii < jj                      # strict upper triangle, no self-pairs
        for a, b in zip(ii[keep].tolist(), jj[keep].tolist()):
            out.append((keys[a], keys[b], int(np.bitwise_count(h[a] ^ h[b]))))
    return out


def _pairs_multi_index(keys, h, threshold, bands=BANDS, band_bits=BAND_BITS):
    """Multi-index hashing: exact, and scalable past brute force.

    The guarantee: if two hashes are within ``threshold`` bits overall then at
    least one band differs by at most ``threshold // bands`` bits, because
    otherwise every band would contribute at least ``threshold // bands + 1``
    and the total would exceed ``threshold``. So probing each band at that
    radius cannot miss a pair. Asserted below rather than assumed, since the
    previous implementation shipped the *stronger* claim (one band identical)
    without its precondition holding.
    """
    import numpy as np
    from collections import defaultdict

    radius = threshold // bands
    assert bands * (radius + 1) > threshold, (
        f"multi-index hashing needs bands*(radius+1) > threshold; "
        f"bands={bands} radius={radius} threshold={threshold}"
    )
    probes = [0] + [1 << k for k in range(band_bits)] if radius >= 1 else [0]
    if radius > 1:                          # not needed at threshold 6, but be honest
        raise NotImplementedError(
            f"probe radius {radius} > 1 is not implemented; lower the threshold "
            f"or raise BANDS"
        )

    n = len(h)
    h_int = h.tolist()
    out: list[tuple[str, str, int]] = []
    seen_emitted: set[tuple[int, int]] = set()

    for b in range(bands):
        shift = b * band_bits
        mask = (1 << band_bits) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, hv in enumerate(h_int):
            buckets[(hv >> shift) & mask].append(i)

        for value, members in buckets.items():
            for probe in probes:
                other = value ^ probe
                if probe and other < value:
                    continue                # handle each unordered value pair once
                partners = buckets.get(other)
                if not partners:
                    continue
                if probe == 0:
                    pairs = ((members[x], members[y])
                             for x in range(len(members))
                             for y in range(x + 1, len(members)))
                else:
                    pairs = ((a, c) for a in members for c in partners)
                for a, c in pairs:
                    if a == c:
                        continue
                    lo, hi = (a, c) if a < c else (c, a)
                    if (lo, hi) in seen_emitted:
                        continue
                    d = _hamming(h_int[lo], h_int[hi])
                    if d <= threshold:
                        seen_emitted.add((lo, hi))
                        out.append((keys[lo], keys[hi], d))
    return out


def near_duplicate_pairs(
    items: list[tuple[str, str]],
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[tuple[str, str, int]]:
    """``[(key_a, key_b, distance), ...]`` for every pair within ``threshold``.

    **Exact**: no pair within ``threshold`` is ever missed. That matters because
    the V2 splitter builds leak groups from these pairs, and a missed pair puts
    the same photograph in train and in the sealed release holdout - a silent
    failure that no metric reveals.

    The previous implementation banded the 64-bit hash into four 16-bit bands
    and required one band to match *exactly*. Its own comment stated the
    precondition - "threshold < bands" - and then applied it at threshold 6 with
    4 bands, where it does not hold. Two hashes differing by one bit in each of
    the four bands are 4 apart and share no band, so they were silently dropped;
    `test_four_band_counterexample_is_found` pins exactly that case.

    ``items`` is ``[(key, dhash_hex), ...]``; keys are returned as given, so the
    caller decides whether a key is a candidate id, a sha256 or anything else.
    """
    if len(items) < 2:
        return []
    keys = [k for k, _ in items]
    h = _as_uint64(items)
    if len(items) <= BRUTE_FORCE_MAX:
        return _pairs_brute_force(keys, h, threshold)
    return _pairs_multi_index(keys, h, threshold)


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
