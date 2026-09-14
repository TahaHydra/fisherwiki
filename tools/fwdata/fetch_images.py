"""Download selected candidate photographs into content-addressed storage.

Selection happens *before* any bytes move, in SQL over the candidate Parquet:
licence policy, per-species caps, and a diversity-aware ordering that prefers
photographs from distinct observations and distinct photographers over many
shots of the same fish. That ordering matters more than it looks - a species
whose 300 images come from 4 observations will look well-covered and generalise
terribly.

Every stored image goes through :meth:`ImageStore.put`, so it is impossible for
a downloaded file to exist without an admissible licence and a provenance row.

The downloader is polite by construction: requests go through the shared
per-host token bucket in :mod:`fwdata.net`, concurrency is bounded, and failures
are recorded (permanently, for 404/403) so a re-run does not re-hammer dead URLs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from . import net
from .config import PATHS
from .images import ImageRejected, measure
from .licenses import Policy, get_policy, normalize
from .provenance import ImageProvenance, ImageStore, LicenseRefused, ProvenanceDB


@dataclass
class FetchStats:
    selected: int = 0
    downloaded: int = 0
    already_present: int = 0
    http_failures: int = 0
    decode_failures: int = 0
    license_refusals: int = 0
    store_errors: int = 0
    bytes_fetched: int = 0
    seconds: float = 0.0
    by_flag: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["seconds"] = round(self.seconds, 1)
        return d


SELECT_SQL = """
WITH deduped AS (
    -- One row per photo_id. iNaturalist genuinely attaches the same photo to
    -- more than one observation (6,739 cases in the 2026-08 snapshot), which
    -- would otherwise (a) violate the candidate_id primary key and (b) let the
    -- identical image land in two different splits via two different group
    -- keys - a textbook train/test leak. Tie-broken deterministically.
    SELECT *
    FROM read_parquet('{parquet}')
    WHERE license IN {licenses}
      {extra_where}
    QUALIFY row_number() OVER (
        PARTITION BY source_record_id
        ORDER BY group_key, position_in_observation
    ) = 1
),
pool AS (
    SELECT *,
           -- Prefer the first photo of each observation, then spread across
           -- observations and photographers before taking extra shots of the
           -- same individual fish.
           row_number() OVER (
               PARTITION BY canonical_name
               ORDER BY position_in_observation ASC,
                        hash(group_key) ASC,
                        hash(observer_key) ASC,
                        hash(source_record_id) ASC
           ) AS rn,
           count(*)                    OVER (PARTITION BY canonical_name) AS sp_images,
           count(DISTINCT group_key)   OVER (PARTITION BY canonical_name) AS sp_observations
    FROM deduped
)
SELECT source_dataset, source_record_id, image_url, source_url, source_taxon_id,
       original_scientific_name, accepted_scientific_name, license, license_raw,
       license_url, creator, copyright_holder, group_key, observer_key,
       latitude, longitude, positional_accuracy, observed_on, quality_grade,
       position_in_observation, declared_width, declared_height, ext,
       canonical_name, anomaly_score
FROM pool
WHERE rn <= {per_species_cap}
  AND sp_observations >= {min_observations}
ORDER BY canonical_name, rn
{limit_clause}
"""


def select_candidates(
    *,
    parquet: Path | None = None,
    policy: Policy,
    per_species_cap: int = 300,
    min_observations: int = 3,
    limit: int | None = None,
    extra_where: str = "",
) -> list[tuple]:
    """Choose which candidate photographs to fetch, without fetching anything."""
    p = Path(parquet or (PATHS.work / "candidates_inaturalist.parquet"))
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `tools/dataset.py extract` first")
    sql = SELECT_SQL.format(
        parquet=p.as_posix(),
        licenses=policy.sql_in_list(),
        per_species_cap=int(per_species_cap),
        min_observations=int(min_observations),
        extra_where=(f"AND ({extra_where})" if extra_where else ""),
        limit_clause=(f"LIMIT {int(limit)}" if limit else ""),
    )
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


COLUMNS = [
    "source_dataset", "source_record_id", "image_url", "source_url",
    "source_taxon_id", "original_scientific_name", "accepted_scientific_name",
    "license", "license_raw", "license_url", "creator", "copyright_holder",
    "group_key", "observer_key", "latitude", "longitude", "positional_accuracy",
    "observed_on", "quality_grade", "position_in_observation", "declared_width",
    "declared_height", "ext", "canonical_name", "anomaly_score",
]


_REGISTRY: "TaxonRegistry | None" = None


def _registry() -> "TaxonRegistry":
    """Lazily load the canonical taxon registry (43k rows, loaded once)."""
    global _REGISTRY
    if _REGISTRY is None:
        from .taxonomy.registry import TaxonRegistry

        _REGISTRY = TaxonRegistry()
    return _REGISTRY


def _row_to_provenance(row: tuple) -> ImageProvenance:
    d = dict(zip(COLUMNS, row))
    # Resolve our own stable taxon id here rather than leaving it null. Source
    # ids are not class identities (see taxonomy/registry.py), and a provenance
    # row without fw_taxon_id cannot be joined to a model class later.
    rec = _registry().get(d["canonical_name"] or "")
    return ImageProvenance(
        taxon_id=rec.fw_taxon_id if rec else None,
        source_dataset=d["source_dataset"],
        source_record_id=str(d["source_record_id"]),
        image_url=d["image_url"],
        source_url=d["source_url"],
        source_taxon_id=str(d["source_taxon_id"]),
        original_scientific_name=d["original_scientific_name"] or "",
        accepted_scientific_name=d["accepted_scientific_name"],
        license=normalize(d["license_raw"]),
        license_raw=d["license_raw"] or "",
        creator=d["creator"],
        copyright_holder=d["copyright_holder"],
        group_key=d["group_key"],
        observer_key=str(d["observer_key"]) if d["observer_key"] is not None else None,
        latitude=d["latitude"],
        longitude=d["longitude"],
        positional_accuracy=d["positional_accuracy"],
        observed_on=str(d["observed_on"]) if d["observed_on"] else None,
        quality_grade=d["quality_grade"],
        position_in_observation=d["position_in_observation"],
        declared_width=d["declared_width"],
        declared_height=d["declared_height"],
        ext=(d["ext"] or "jpg"),
        notes=(
            f"anomaly_score={d['anomaly_score']:.3f}"
            if d.get("anomaly_score") is not None
            else None
        ),
    )


def fetch(
    rows: list[tuple],
    *,
    policy: Policy,
    workers: int = 16,
    db: ProvenanceDB | None = None,
    log=print,
    log_every: float = 20.0,
) -> FetchStats:
    """Download ``rows`` into the CAS, recording provenance for every image."""
    t0 = time.time()
    own_db = db is None
    db = db or ProvenanceDB()
    store = ImageStore(db, policy)
    stats = FetchStats(selected=len(rows))
    lock = threading.Lock()

    provs = [_row_to_provenance(r) for r in rows]
    db.add_candidates(provs)
    log(f"registered {len(provs):,} candidates in provenance store")

    # Resume support: skip anything already in the CAS, and anything that has
    # already failed permanently (404/410, undecodable). A multi-hour fetch will
    # be interrupted, and re-running it must neither re-download what we have
    # nor re-hammer URLs that are known dead.
    db.flush()
    done: set[str] = {
        r[0]
        for r in db.con.execute(
            "SELECT candidate_id FROM stored "
            "UNION ALL "
            "SELECT candidate_id FROM fetch_failures WHERE reason = 'permanent'"
        ).fetchall()
    }
    if done:
        before = len(provs)
        provs = [
            p
            for p in provs
            if db.candidate_id(p.source_dataset, p.source_record_id) not in done
        ]
        stats.already_present = before - len(provs)
        log(f"skipping {stats.already_present:,} already fetched or permanently failed")
    if not provs:
        stats.seconds = time.time() - t0
        if own_db:
            db.close()
        return stats
    rows_remaining = len(provs)

    last_log = [time.time()]

    def one(prov: ImageProvenance) -> None:
        cid = db.candidate_id(prov.source_dataset, prov.source_record_id)
        try:
            data = net.fetch_bytes(prov.image_url)
        except net.PermanentError as exc:
            db.mark_failure(cid, "permanent", str(exc))
            with lock:
                stats.http_failures += 1
            return
        except net.DownloadError as exc:
            db.mark_failure(cid, "transient", str(exc))
            with lock:
                stats.http_failures += 1
            return

        try:
            facts = measure(data)
        except ImageRejected as exc:
            db.mark_failure(cid, "permanent", f"undecodable: {exc}")
            with lock:
                stats.decode_failures += 1
            return

        try:
            store.put(
                data,
                prov,
                sha256=facts.sha256,
                width=facts.width,
                height=facts.height,
                phash=facts.phash,
                dhash=facts.dhash,
            )
        except LicenseRefused as exc:
            # A genuine policy rejection. Permanent: re-running will not help.
            db.mark_failure(cid, "permanent", f"licence refused: {exc}")
            with lock:
                stats.license_refusals += 1
            return
        except OSError as exc:
            # Filesystem trouble is transient, not a licence problem. Counting
            # it as a refusal (as an earlier version did) made 9 disk races look
            # like 9 licence violations, which is exactly the wrong alarm.
            db.mark_failure(cid, "transient", f"store failed: {exc}")
            with lock:
                stats.store_errors += 1
            return

        for flag in facts.flags:
            db.add_quality_flag(cid, flag)
        with lock:
            stats.downloaded += 1
            stats.bytes_fetched += facts.nbytes
            for flag in facts.flags:
                stats.by_flag[flag] = stats.by_flag.get(flag, 0) + 1
            now = time.time()
            if now - last_log[0] >= log_every:
                last_log[0] = now
                el = now - t0
                done = stats.downloaded + stats.http_failures + stats.decode_failures
                rate = done / max(el, 1e-6)
                eta = (rows_remaining - done) / max(rate, 1e-6) / 60
                log(
                    f"  {done:,}/{rows_remaining:,} ({100*done/max(1,rows_remaining):.1f}%) "
                    f"{rate:.0f} img/s  {stats.bytes_fetched/1e9:.2f} GB  "
                    f"eta {eta:.0f}m  fail={stats.http_failures}"
                )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, provs))

    stats.seconds = time.time() - t0
    if own_db:
        db.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="production")
    ap.add_argument("--per-species-cap", type=int, default=300)
    ap.add_argument("--min-observations", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--where", default="", help="extra SQL predicate on candidates")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    policy = get_policy(args.policy)
    log(f"policy={policy.name} licences={sorted(x.value for x in policy.allowed)}")
    rows = select_candidates(
        policy=policy,
        per_species_cap=args.per_species_cap,
        min_observations=args.min_observations,
        limit=args.limit,
        extra_where=args.where,
    )
    log(f"selected {len(rows):,} photographs")
    stats = fetch(rows, policy=policy, workers=args.workers, log=log)
    log(json.dumps(stats.as_dict(), indent=2))
    (PATHS.work / "fetch_report.json").write_text(
        json.dumps(stats.as_dict(), indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
