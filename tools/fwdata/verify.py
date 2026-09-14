"""Integrity checks over the content-addressed store and the provenance DB.

Three classes of problem, all of which are silent if nobody looks:

1. **Store/DB disagreement** - a provenance row whose file is missing, or a file
   in the CAS with no provenance row. The second is the one that matters: an
   image with no provenance is an image with no licence, which is precisely
   what the whole design exists to prevent.
2. **Content drift** - a file whose bytes no longer hash to its name. Since the
   path *is* the hash, this detects bit-rot and tampering alike.
3. **Licence violations** - any stored image whose licence is not admitted by
   the policy it was collected under.

Hash verification samples by default because re-hashing 300k files takes a
while; ``--sample 0`` checks everything.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from .config import PATHS
from .licenses import License, Policy
from .provenance import ProvenanceDB


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_store(
    db: ProvenanceDB,
    *,
    sample: int = 2000,
    policy: Policy | None = None,
    log=print,
) -> dict:
    """Check the CAS against the provenance database."""
    problems: list[str] = []
    report: dict = {}

    counts = db.counts()
    report["counts"] = counts
    log(f"provenance: {counts['candidates']:,} candidates, "
        f"{counts['stored']:,} stored, {counts['failures']:,} failures")

    # --- 1. rows whose file is missing -------------------------------------
    rows = db.con.execute(
        "SELECT candidate_id, sha256, cas_path, bytes FROM stored"
    ).fetchall()
    missing = []
    for cid, sha, cas_path, nbytes in rows:
        p = PATHS.cas / cas_path
        if not p.exists():
            missing.append(cid)
        elif nbytes is not None and p.stat().st_size != nbytes:
            problems.append(f"{cas_path}: size {p.stat().st_size} != recorded {nbytes}")
    report["rows_with_missing_file"] = len(missing)
    if missing:
        problems.append(f"{len(missing)} provenance rows point at missing files")
    log(f"files referenced by provenance: {len(rows):,} "
        f"({len(missing):,} missing)")

    # --- 2. files with no provenance row -----------------------------------
    known = {r[0] for r in db.con.execute("SELECT cas_path FROM stored").fetchall()}
    orphans = 0
    on_disk = 0
    for f in PATHS.cas.rglob("*"):
        if not f.is_file() or f.suffix == ".tmp":
            continue
        on_disk += 1
        rel = str(f.relative_to(PATHS.cas)).replace("\\", "/")
        if rel not in known:
            orphans += 1
    report["files_on_disk"] = on_disk
    report["files_without_provenance"] = orphans
    if orphans:
        # Deliberately an error, not a warning: an image with no provenance row
        # is an image with no recorded licence.
        problems.append(
            f"{orphans} files in the CAS have no provenance row (no recorded licence)"
        )
    log(f"files on disk: {on_disk:,} ({orphans:,} without provenance)")

    # --- 3. content drift ---------------------------------------------------
    checkable = [r for r in rows if (PATHS.cas / r[2]).exists()]
    to_check = checkable if sample <= 0 else random.sample(
        checkable, min(sample, len(checkable))
    )
    bad_hash = []
    for cid, sha, cas_path, _ in to_check:
        actual = _sha256(PATHS.cas / cas_path)
        if actual != sha:
            bad_hash.append(cas_path)
    report["hashes_checked"] = len(to_check)
    report["hash_mismatches"] = len(bad_hash)
    if bad_hash:
        problems.append(f"{len(bad_hash)} files do not match their recorded hash")
    log(f"hashes verified: {len(to_check):,} ({len(bad_hash)} mismatched)")

    # --- 4. licence violations ---------------------------------------------
    lic_rows = db.con.execute(
        "SELECT c.license, count(*) FROM stored s JOIN candidates c "
        "USING (candidate_id) GROUP BY c.license ORDER BY 2 DESC"
    ).fetchall()
    report["stored_by_license"] = {k: int(v) for k, v in lic_rows}
    forbidden = [
        (k, v) for k, v in lic_rows
        if k in (License.UNKNOWN.value, License.ARR.value)
    ]
    if forbidden:
        problems.append(f"stored images with inadmissible licences: {forbidden}")
    if policy is not None:
        outside = [
            (k, int(v)) for k, v in lic_rows
            if k not in {x.value for x in policy.allowed}
        ]
        report["outside_policy"] = outside
        if outside:
            problems.append(
                f"stored images outside policy {policy.name!r}: {outside}"
            )
    log("stored by licence: " + ", ".join(f"{k}={v:,}" for k, v in lic_rows))

    report["ok"] = not problems
    report["problems"] = problems
    if problems:
        log("")
        for p in problems:
            log(f"PROBLEM: {p}")
    else:
        log("\nall checks passed")
    return report
