"""Shared primitives for broad V2 training-data acquisition.

This module is deliberately separate from the release-corpus licence gates in
:mod:`fwdata.provenance`.  Acquisition is an engineering inventory: source
licence/rights metadata is preserved verbatim and normalised when possible, but
it is *not* used to decide whether bytes may enter the local CAS.  Release-time
corpus policy remains a separate concern and the existing production policies
are unchanged.

Long downloads are interruption-safe.  Every candidate is registered before it
is fetched; downloads land in deterministic staging paths via ``net.download_file``
(``.part`` + HTTP Range resume), then move into the content-addressed store
atomically.  Database writes are flushed before staging files are deleted, so a
power loss can cost at most the current small batch and never loses the only
copy of downloaded bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from . import net
from .config import PATHS, free_space_gb
from .images import ImageRejected, measure
from .licenses import normalize
from .provenance import CANDIDATE_COLUMNS, ImageProvenance, ProvenanceDB
from .taxonomy.registry import TaxonRecord, TaxonRegistry


class DiskReserveReached(RuntimeError):
    """Raised after stopping cleanly because the data volume is getting full."""


@dataclass
class AcquireStats:
    source: str
    selected: int = 0
    downloaded: int = 0
    already_done: int = 0
    permanent_failures: int = 0
    transient_failures: int = 0
    decode_failures: int = 0
    bytes_fetched: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class ScanState:
    """Durable per-taxon source-discovery checkpoints.

    Values are the largest per-taxon cap that was completely scanned.  Raising a
    cap later deliberately re-scans that taxon (candidate insertion is
    idempotent); keeping the same/lower cap skips it immediately.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.path = PATHS.work / "acquisition" / f"scan_{source}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: dict[str, int] = {}
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                self.completed = {
                    str(k): int(v) for k, v in (obj.get("completed") or {}).items()
                }
            except (OSError, ValueError, TypeError):
                # A broken checkpoint is not data loss: candidates already
                # registered in DuckDB are idempotent, so the safe recovery is
                # to re-scan metadata rather than to pretend work is complete.
                self.completed = {}

    def needs(self, key: str | int, cap: int) -> bool:
        return self.completed.get(str(key), -1) < int(cap)

    def mark(self, key: str | int, cap: int) -> None:
        key = str(key)
        self.completed[key] = max(int(cap), self.completed.get(key, -1))
        payload = {
            "source": self.source,
            "updated_at": time.time(),
            "completed": self.completed,
        }
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def current_support(db: ProvenanceDB) -> dict[int, int]:
    """Stored-image count by internal taxon id."""
    return {
        int(t): int(n)
        for t, n in db.con.execute(
            "SELECT species_taxon_id, count(DISTINCT sha256) "
            "FROM provenance WHERE species_taxon_id IS NOT NULL GROUP BY 1"
        ).fetchall()
    }


def prioritized_species(
    db: ProvenanceDB,
    *,
    id_attribute: str | None = None,
) -> list[TaxonRecord]:
    """Active species, thinnest existing classes first.

    ``id_attribute`` can require a source-specific id such as
    ``gbif_taxon_id``.  This makes long discovery runs useful immediately: a
    stopped run has spent its requests on classes that needed data most.
    """
    support = current_support(db)
    reg = TaxonRegistry()
    rows = [
        r for r in reg.active
        if (r.rank or "").lower() == "species"
        and (id_attribute is None or getattr(r, id_attribute, None) is not None)
    ]
    rows.sort(key=lambda r: (support.get(r.fw_taxon_id, 0), r.canonical_name))
    return rows


def _staging_root() -> Path:
    env = os.environ.get("FISHERWIKI_STAGING")
    if env:
        return Path(env)
    # Use the existing hot staging volume when present.  Falling back to the
    # data root keeps the tool portable on machines without E:.
    hot = Path(r"E:\FisherWiki\staging")
    if os.name == "nt" and hot.parent.exists():
        return hot / "acquisition"
    return PATHS.work / "acquisition" / "staging"


def _safe_ext(ext: str | None) -> str:
    e = (ext or "jpg").lower().strip().lstrip(".")
    return e if e in {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"} else "jpg"


def _stage_path(prov: ImageProvenance) -> Path:
    cid = ProvenanceDB.candidate_id(prov.source_dataset, prov.source_record_id)
    h = hashlib.sha256(cid.encode("utf-8")).hexdigest()
    return _staging_root() / prov.source_dataset / h[:2] / f"{h}.{_safe_ext(prov.ext)}"


def _store_unfiltered(
    db: ProvenanceDB,
    data: bytes,
    prov: ImageProvenance,
    *,
    sha256: str,
    width: int,
    height: int,
    phash: str | None,
    dhash: str | None,
) -> Path:
    """Write one acquired image to CAS without applying a release policy.

    The source licence fields stay in ``candidates`` exactly as usual.  The only
    thing intentionally absent is an acquisition-time allow/deny decision.
    """
    ext = _safe_ext(prov.ext)
    dest = PATHS.cas_path(sha256, ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_name(
            f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            # os.replace is atomic on the same volume.  A concurrent candidate
            # with identical bytes may win the race; replacing identical bytes
            # is harmless and leaves one CAS object.
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    cid = db.candidate_id(prov.source_dataset, prov.source_record_id)
    db.mark_stored(
        cid,
        sha256=sha256,
        nbytes=len(data),
        width=width,
        height=height,
        phash=phash,
        dhash=dhash,
        cas_path=str(dest.relative_to(PATHS.cas)).replace("\\", "/"),
    )
    return dest


def _row_to_provenance(row: tuple) -> tuple[str, ImageProvenance]:
    names = ["candidate_id"] + [n for n, _ in CANDIDATE_COLUMNS]
    d = dict(zip(names, row))
    return d["candidate_id"], ImageProvenance(
        source_dataset=d["source_dataset"],
        source_record_id=d["source_record_id"],
        image_url=d["image_url"],
        source_url=d["source_url"],
        source_taxon_id=d["source_taxon_id"],
        original_scientific_name=d["original_scientific_name"] or "",
        license=normalize(d["license"] or d["license_raw"]),
        license_raw=d["license_raw"] or "",
        creator=d["creator"],
        copyright_holder=d["copyright_holder"],
        attribution=d["attribution"],
        taxon_id=d["taxon_id"],
        accepted_scientific_name=d["accepted_scientific_name"],
        group_key=d["group_key"],
        observer_key=d["observer_key"],
        latitude=d["latitude"],
        longitude=d["longitude"],
        positional_accuracy=d["positional_accuracy"],
        observed_on=d["observed_on"],
        country_code=d["country_code"],
        quality_grade=d["quality_grade"],
        position_in_observation=d["position_in_observation"],
        declared_width=d["declared_width"],
        declared_height=d["declared_height"],
        original_filename=d["original_filename"],
        ext=d["ext"] or "jpg",
        context_tag=d["context_tag"],
        notes=d["notes"],
    )


def register_candidates(db: ProvenanceDB, records: Iterable[ImageProvenance]) -> int:
    """Register only records that resolve to a FisherWiki taxon."""
    valid = [r for r in records if r.taxon_id is not None and r.image_url]
    return db.add_candidates(valid)


def pending_count(db: ProvenanceDB, source: str) -> int:
    return int(db.con.execute(
        "SELECT count(*) FROM candidates c WHERE c.source_dataset = ? "
        "AND NOT EXISTS (SELECT 1 FROM stored s WHERE s.candidate_id=c.candidate_id) "
        "AND NOT EXISTS (SELECT 1 FROM fetch_failures f "
        "  WHERE f.candidate_id=c.candidate_id AND f.reason='permanent')",
        [source],
    ).fetchone()[0])


def fetch_registered_source(
    db: ProvenanceDB,
    source: str,
    *,
    workers: int = 16,
    batch_size: int = 512,
    reserve_gb: float = 50.0,
    progress=None,
    log=print,
) -> AcquireStats:
    """Fetch all registered pending candidates for one source.

    Keyset pagination means transient failures are tried once per invocation and
    then naturally retried on the next invocation; there is no unbounded
    in-memory "attempted" set even for multi-million-image runs.
    """
    t0 = time.time()
    stats = AcquireStats(source=source)
    stats.selected = pending_count(db, source)
    if stats.selected == 0:
        return stats

    columns = ["candidate_id"] + [n for n, _ in CANDIDATE_COLUMNS]
    select = ", ".join(f"c.{c}" for c in columns)
    last = ""
    processed = 0

    while True:
        if free_space_gb(PATHS.root) < reserve_gb:
            db.flush()
            raise DiskReserveReached(
                f"D: data reserve reached: {free_space_gb(PATHS.root):.1f} GB free; "
                f"required reserve is {reserve_gb:.1f} GB"
            )
        rows = db.con.execute(
            f"SELECT {select} FROM candidates c "
            "WHERE c.source_dataset=? AND c.candidate_id>? "
            "AND NOT EXISTS (SELECT 1 FROM stored s WHERE s.candidate_id=c.candidate_id) "
            "AND NOT EXISTS (SELECT 1 FROM fetch_failures f "
            "  WHERE f.candidate_id=c.candidate_id AND f.reason='permanent') "
            "ORDER BY c.candidate_id LIMIT ?",
            [source, last, int(batch_size)],
        ).fetchall()
        if not rows:
            break
        pairs = [_row_to_provenance(r) for r in rows]
        success_staging: list[Path] = []
        success_lock = threading.Lock()

        def one(item: tuple[str, ImageProvenance]) -> tuple[str, int]:
            cid, prov = item
            stage = _stage_path(prov)
            try:
                net.download_file(
                    prov.image_url,
                    stage,
                    min_free_gb=max(2.0, reserve_gb),
                    max_attempts=5,
                )
                data = stage.read_bytes()
                facts = measure(data)
                _store_unfiltered(
                    db,
                    data,
                    prov,
                    sha256=facts.sha256,
                    width=facts.width,
                    height=facts.height,
                    phash=facts.phash,
                    dhash=facts.dhash,
                )
                for flag in facts.flags:
                    db.add_quality_flag(cid, flag)
                with success_lock:
                    success_staging.append(stage)
                return "ok", facts.nbytes
            except net.PermanentError as exc:
                db.mark_failure(cid, "permanent", str(exc))
                return "permanent", 0
            except ImageRejected as exc:
                db.mark_failure(cid, "permanent", f"undecodable: {exc}")
                stage.unlink(missing_ok=True)
                return "decode", 0
            except (net.DownloadError, OSError) as exc:
                # Keep .part/final staging bytes for a later resume.
                db.mark_failure(cid, "transient", str(exc))
                return "transient", 0

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            results = list(ex.map(one, pairs))

        # Make candidate->CAS mappings durable *before* deleting the final
        # staging files.  This is the power-loss boundary for a batch.
        db.flush()
        for p in success_staging:
            p.unlink(missing_ok=True)

        for status, nbytes in results:
            processed += 1
            if status == "ok":
                stats.downloaded += 1
                stats.bytes_fetched += int(nbytes)
            elif status == "permanent":
                stats.permanent_failures += 1
            elif status == "decode":
                stats.decode_failures += 1
            else:
                stats.transient_failures += 1
        if progress is not None:
            progress.advance(len(results), failures=sum(s != "ok" for s, _ in results))
            progress.checkpointed(rows[-1][0])
        elif processed % 2000 < len(results):
            elapsed = max(time.time() - t0, 1e-6)
            log(
                f"  {processed:,}/{stats.selected:,} {processed/elapsed:.1f} img/s "
                f"{stats.bytes_fetched/1e9:.2f} GB "
                f"fail={stats.permanent_failures + stats.transient_failures + stats.decode_failures}"
            )
        last = rows[-1][0]

    stats.seconds = time.time() - t0
    # "already_done" is useful when the same source is re-run after an
    # interruption: total registered minus the pending work we actually saw.
    total_registered = int(db.con.execute(
        "SELECT count(*) FROM candidates WHERE source_dataset=?", [source]
    ).fetchone()[0])
    stats.already_done = max(0, total_registered - stats.selected)
    return stats
