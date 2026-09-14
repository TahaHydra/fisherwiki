"""Provenance database and content-addressed image store.

Design contract
---------------
There is exactly one way to put an image on disk: :meth:`ImageStore.put`, and it
requires a fully-populated :class:`ImageProvenance`.  There is no public helper
that writes image bytes without provenance, and the store refuses to accept a
record whose licence is ``UNKNOWN``/``ARR`` or whose licence is not admitted by
the active :class:`~fwdata.licenses.Policy`.  That is the whole point of the
module: legal cleanliness is enforced by the type system and the API surface,
not by developer discipline.

Storage layout
--------------
Images are content-addressed::

    <data_root>/cas/<sha[0:2]>/<sha[2:4]>/<sha>.<ext>

so the same photograph fetched from two sources is stored once, and a corpus is
just a list of hashes.  Deleting a corpus never deletes pixels; rebuilding a
corpus with different licence filters never re-downloads anything already held.

Tables
------
``candidates``
    One row per *candidate* photograph discovered in a bulk export, written
    before anything is downloaded.  Immutable and append-only: this is the
    reproducible record of "what we considered".
``stored``
    One row per photograph actually written to the CAS, with the bytes-level
    facts (sha256, dimensions, perceptual hash) that only exist post-download.
``fetch_failures``
    Why a candidate did not become a stored image.  Kept so that coverage gaps
    are explainable rather than mysterious.
``corpus_members``
    Materialised corpus assignments (split, weight) for a named, hashed corpus
    build.

``provenance`` is a view joining ``candidates`` and ``stored`` and is the thing
that ``DATA_PROVENANCE.md`` / ``ATTRIBUTIONS.csv`` are generated from.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pyarrow as pa

from .config import PATHS
from .licenses import License, Policy, attribution_string, normalize

SCHEMA_VERSION = 1


class ProvenanceError(RuntimeError):
    pass


class LicenseRefused(ProvenanceError):
    """Raised when a record's licence is not admissible for the active policy."""


@dataclass
class ImageProvenance:
    """Everything we must know about one photograph before it may be stored.

    Field names are stable: they are the column names in ``candidates`` and the
    headers of the exported ``ATTRIBUTIONS.csv``.
    """

    # --- identity -----------------------------------------------------------
    source_dataset: str          # 'inaturalist' | 'gbif' | 'wikimedia' | ...
    source_record_id: str        # photo_id / gbif mediaKey / commons pageid
    image_url: str               # exact URL the bytes came from
    source_url: str              # human-facing page for the record

    # --- taxonomy (as asserted by the source) -------------------------------
    source_taxon_id: str
    original_scientific_name: str

    # --- licence ------------------------------------------------------------
    license: License
    license_raw: str
    creator: str | None = None
    copyright_holder: str | None = None
    attribution: str | None = None

    # --- canonical taxonomy (filled by reconciliation) ----------------------
    taxon_id: int | None = None
    accepted_scientific_name: str | None = None

    # --- observation context ------------------------------------------------
    group_key: str | None = None       # observation uuid: leakage-safe grouping
    observer_key: str | None = None    # photographer id: second grouping level
    latitude: float | None = None
    longitude: float | None = None
    positional_accuracy: int | None = None
    observed_on: str | None = None
    country_code: str | None = None
    quality_grade: str | None = None
    position_in_observation: int | None = None

    # --- media facts asserted by the source (may differ from reality) -------
    declared_width: int | None = None
    declared_height: int | None = None
    original_filename: str | None = None
    ext: str = "jpg"

    # --- free-form ----------------------------------------------------------
    context_tag: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        # NOTE: `License` subclasses `str`, so `isinstance(self.license, str)`
        # is True for an actual License member. Normalising it again would call
        # `str(License.CC_BY)`, which on Python 3.11+ yields 'License.CC_BY'
        # rather than 'CC-BY-4.0', and that parses to UNKNOWN - silently
        # rejecting every correctly-licensed image. Test for the enum first.
        if not isinstance(self.license, License):
            self.license = normalize(self.license)
        if not self.attribution:
            self.attribution = attribution_string(
                self.creator or self.copyright_holder,
                self.license,
                self.source_dataset,
            )

    @property
    def license_url(self) -> str:
        return self.license.url

    def row(self) -> dict:
        d = asdict(self)
        d["license"] = self.license.value
        d["license_url"] = self.license_url
        return d


CANDIDATE_COLUMNS: Sequence[tuple[str, str]] = (
    ("source_dataset", "VARCHAR"),
    ("source_record_id", "VARCHAR"),
    ("image_url", "VARCHAR"),
    ("source_url", "VARCHAR"),
    ("source_taxon_id", "VARCHAR"),
    ("original_scientific_name", "VARCHAR"),
    ("license", "VARCHAR"),
    ("license_raw", "VARCHAR"),
    ("license_url", "VARCHAR"),
    ("creator", "VARCHAR"),
    ("copyright_holder", "VARCHAR"),
    ("attribution", "VARCHAR"),
    ("taxon_id", "BIGINT"),
    ("accepted_scientific_name", "VARCHAR"),
    ("group_key", "VARCHAR"),
    ("observer_key", "VARCHAR"),
    ("latitude", "DOUBLE"),
    ("longitude", "DOUBLE"),
    ("positional_accuracy", "BIGINT"),
    ("observed_on", "VARCHAR"),
    ("country_code", "VARCHAR"),
    ("quality_grade", "VARCHAR"),
    ("position_in_observation", "BIGINT"),
    ("declared_width", "BIGINT"),
    ("declared_height", "BIGINT"),
    ("original_filename", "VARCHAR"),
    ("ext", "VARCHAR"),
    ("context_tag", "VARCHAR"),
    ("notes", "VARCHAR"),
)

_DDL = f"""
CREATE TABLE IF NOT EXISTS meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id VARCHAR PRIMARY KEY,   -- '<source>:<record_id>'
    {', '.join(f'{n} {t}' for n, t in CANDIDATE_COLUMNS)},
    discovered_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stored (
    candidate_id VARCHAR,
    sha256 VARCHAR,
    bytes BIGINT,
    width BIGINT,
    height BIGINT,
    phash VARCHAR,             -- 64-bit perceptual hash, hex
    dhash VARCHAR,
    cas_path VARCHAR,
    download_timestamp TIMESTAMP,
    PRIMARY KEY (candidate_id)
);

CREATE TABLE IF NOT EXISTS fetch_failures (
    candidate_id VARCHAR,
    reason VARCHAR,
    detail VARCHAR,
    failed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus_members (
    corpus VARCHAR,
    candidate_id VARCHAR,
    sha256 VARCHAR,
    class_id BIGINT,
    taxon_id BIGINT,
    split VARCHAR,
    weight DOUBLE,
    PRIMARY KEY (corpus, candidate_id)
);

CREATE TABLE IF NOT EXISTS quality_flags (
    candidate_id VARCHAR,
    flag VARCHAR,
    score DOUBLE,
    PRIMARY KEY (candidate_id, flag)
);
"""

_PROVENANCE_VIEW = """
CREATE OR REPLACE VIEW provenance AS
SELECT
    s.sha256                       AS image_id,
    s.sha256,
    c.taxon_id                     AS species_taxon_id,
    c.accepted_scientific_name,
    c.original_scientific_name,
    c.source_dataset,
    c.source_record_id,
    c.source_url,
    c.image_url,
    c.creator,
    c.copyright_holder,
    c.license,
    c.license_url,
    c.attribution,
    s.download_timestamp,
    c.original_filename,
    c.latitude, c.longitude, c.positional_accuracy,
    c.observed_on, c.country_code,
    c.group_key, c.observer_key,
    c.quality_grade, c.context_tag,
    s.width, s.height, s.bytes, s.phash, s.cas_path,
    c.candidate_id
FROM stored s
JOIN candidates c USING (candidate_id);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ProvenanceDB:
    """Thin, explicit wrapper around the DuckDB provenance store."""

    def __init__(self, path: Path | None = None, read_only: bool = False) -> None:
        self.path = Path(path or PATHS.provenance_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        self._lock = threading.Lock()
        # Separate lock for the in-memory buffers so appending a row never
        # waits on a DuckDB flush that is already in progress.
        self._buf_lock = threading.Lock()
        self._stored_buf: list[list] = []
        self._failure_buf: list[list] = []
        self._flag_buf: list[list] = []
        if not read_only:
            self.con.execute(_DDL)
            self.con.execute(_PROVENANCE_VIEW)
            self.con.execute(
                "INSERT INTO meta VALUES ('schema_version', ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                [str(SCHEMA_VERSION)],
            )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.con.close()

    def __enter__(self) -> "ProvenanceDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- candidates --------------------------------------------------------
    @staticmethod
    def candidate_id(source_dataset: str, source_record_id: str) -> str:
        return f"{source_dataset}:{source_record_id}"

    #: DuckDB type -> pyarrow type, for the bulk-insert path below.
    _ARROW_TYPES = {
        "VARCHAR": pa.string(),
        "BIGINT": pa.int64(),
        "DOUBLE": pa.float64(),
    }

    def add_candidates(self, records: Iterable[ImageProvenance]) -> int:
        """Insert candidate provenance rows. Idempotent on ``candidate_id``.

        Uses an Arrow bulk insert rather than ``executemany``. DuckDB is
        columnar and its row-at-a-time insert path is glacial at corpus scale:
        registering 308k candidates via ``executemany`` had not completed after
        four minutes, stalling the whole fetch before a single image moved.
        Going through Arrow turns the same work into one columnar append.
        """
        rows = []
        seen: set[str] = set()
        now = _utcnow()
        for r in records:
            cid = self.candidate_id(r.source_dataset, r.source_record_id)
            # Defensive: the caller should already have deduplicated, but a
            # duplicate here would abort the whole batch on the primary key and
            # lose every row with it.
            if cid in seen:
                continue
            seen.add(cid)
            d = r.row()
            rows.append([cid] + [d.get(n) for n, _ in CANDIDATE_COLUMNS] + [now])
        if not rows:
            return 0

        cols = ["candidate_id"] + [n for n, _ in CANDIDATE_COLUMNS] + ["discovered_at"]
        types = (
            [pa.string()]
            + [self._ARROW_TYPES[t] for _, t in CANDIDATE_COLUMNS]
            + [pa.timestamp("us")]
        )
        columnar = list(zip(*rows))
        table = pa.table(
            {
                name: pa.array(list(values), type=typ)
                for name, values, typ in zip(cols, columnar, types)
            }
        )

        with self._lock:
            self.con.register("_incoming_candidates", table)
            try:
                # Anti-join rather than INSERT OR REPLACE: re-running a fetch
                # must not rewrite rows (and bump discovered_at) for candidates
                # we already know about.
                self.con.execute(
                    f"INSERT INTO candidates ({', '.join(cols)}) "
                    f"SELECT {', '.join(cols)} FROM _incoming_candidates i "
                    "WHERE NOT EXISTS ("
                    "  SELECT 1 FROM candidates c WHERE c.candidate_id = i.candidate_id)"
                )
            finally:
                self.con.unregister("_incoming_candidates")
        return len(rows)

    def candidates_for_policy(
        self, policy: Policy, *, limit: int | None = None, where: str = ""
    ) -> list[tuple]:
        """Candidate rows admissible under ``policy`` and not yet stored."""
        extra = f" AND ({where})" if where else ""
        lim = f" LIMIT {int(limit)}" if limit else ""
        sql = (
            "SELECT candidate_id, image_url, ext FROM candidates c "
            f"WHERE c.license IN {policy.sql_in_list()}{extra} "
            "AND NOT EXISTS (SELECT 1 FROM stored s WHERE s.candidate_id = c.candidate_id) "
            "AND NOT EXISTS (SELECT 1 FROM fetch_failures f "
            "                WHERE f.candidate_id = c.candidate_id AND f.reason = 'permanent')"
            f"{lim}"
        )
        return self.con.execute(sql).fetchall()

    # -- stored ------------------------------------------------------------
    # -- buffered writers --------------------------------------------------
    #
    # A single-row INSERT costs ~10 ms here, so a naive per-image write caps the
    # whole fetch at ~100 images/second no matter how many download threads are
    # running - measured, not guessed. Writes are therefore buffered in memory
    # and flushed as columnar batches. Callers see the same API.
    #
    # Durability tradeoff: up to FLUSH_EVERY rows can be lost if the process is
    # killed. That is acceptable because the CAS is the source of truth for what
    # was downloaded and `dataset.py verify --repair` can rebuild these rows
    # from it; losing a few provenance rows never loses an image or its licence.
    FLUSH_EVERY = 2000

    def _maybe_flush(self, force: bool = False) -> None:
        if not force and (
            len(self._stored_buf) < self.FLUSH_EVERY
            and len(self._failure_buf) < self.FLUSH_EVERY
            and len(self._flag_buf) < self.FLUSH_EVERY
        ):
            return
        self.flush()

    def flush(self) -> None:
        """Write all buffered rows. Safe to call at any time."""
        with self._lock:
            if self._stored_buf:
                rows, self._stored_buf = self._stored_buf, []
                self._bulk_insert(
                    "stored",
                    ["candidate_id", "sha256", "bytes", "width", "height",
                     "phash", "dhash", "cas_path", "download_timestamp"],
                    [pa.string(), pa.string(), pa.int64(), pa.int64(), pa.int64(),
                     pa.string(), pa.string(), pa.string(), pa.timestamp("us")],
                    rows,
                    conflict_key="candidate_id",
                )
            if self._failure_buf:
                rows, self._failure_buf = self._failure_buf, []
                self._bulk_insert(
                    "fetch_failures",
                    ["candidate_id", "reason", "detail", "failed_at"],
                    [pa.string(), pa.string(), pa.string(), pa.timestamp("us")],
                    rows,
                )
            if self._flag_buf:
                rows, self._flag_buf = self._flag_buf, []
                self._bulk_insert(
                    "quality_flags",
                    ["candidate_id", "flag", "score"],
                    [pa.string(), pa.string(), pa.float64()],
                    rows,
                    conflict_key="candidate_id, flag",
                )

    def _bulk_insert(
        self,
        table: str,
        cols: list[str],
        types: list,
        rows: list[list],
        conflict_key: str | None = None,
    ) -> None:
        if not rows:
            return
        columnar = list(zip(*rows))
        tbl = pa.table(
            {n: pa.array(list(v), type=t) for n, v, t in zip(cols, columnar, types)}
        )
        view = f"_incoming_{table}"
        self.con.register(view, tbl)
        try:
            if conflict_key:
                keys = [k.strip() for k in conflict_key.split(",")]
                pred = " AND ".join(f"t.{k} = i.{k}" for k in keys)
                # De-duplicate inside the batch as well as against the table:
                # the same candidate can legitimately be retried within one run.
                self.con.execute(
                    f"INSERT INTO {table} ({', '.join(cols)}) "
                    f"SELECT {', '.join(cols)} FROM ("
                    f"  SELECT *, row_number() OVER (PARTITION BY {conflict_key}) rn "
                    f"  FROM {view}) i "
                    f"WHERE i.rn = 1 AND NOT EXISTS ("
                    f"  SELECT 1 FROM {table} t WHERE {pred})"
                )
            else:
                self.con.execute(
                    f"INSERT INTO {table} ({', '.join(cols)}) "
                    f"SELECT {', '.join(cols)} FROM {view}"
                )
        finally:
            self.con.unregister(view)

    def mark_stored(
        self,
        candidate_id: str,
        *,
        sha256: str,
        nbytes: int,
        width: int | None,
        height: int | None,
        phash: str | None,
        dhash: str | None,
        cas_path: str,
    ) -> None:
        with self._buf_lock:
            self._stored_buf.append(
                [candidate_id, sha256, nbytes, width, height, phash, dhash,
                 cas_path, _utcnow()]
            )
            n = len(self._stored_buf)
        if n >= self.FLUSH_EVERY:
            self._maybe_flush()

    def mark_failure(self, candidate_id: str, reason: str, detail: str = "") -> None:
        with self._buf_lock:
            self._failure_buf.append([candidate_id, reason, detail[:500], _utcnow()])
            n = len(self._failure_buf)
        if n >= self.FLUSH_EVERY:
            self._maybe_flush()

    def add_quality_flag(self, candidate_id: str, flag: str, score: float = 1.0) -> None:
        with self._buf_lock:
            self._flag_buf.append([candidate_id, flag, float(score)])
            n = len(self._flag_buf)
        if n >= self.FLUSH_EVERY:
            self._maybe_flush()

    # -- reporting ---------------------------------------------------------
    def license_breakdown(self) -> list[tuple]:
        return self.con.execute(
            "SELECT license, count(*) AS n FROM candidates GROUP BY license ORDER BY n DESC"
        ).fetchall()

    def counts(self) -> dict[str, int]:
        self.flush()

        def one(sql: str) -> int:
            return int(self.con.execute(sql).fetchone()[0])

        return {
            "candidates": one("SELECT count(*) FROM candidates"),
            "stored": one("SELECT count(*) FROM stored"),
            "failures": one("SELECT count(*) FROM fetch_failures"),
            "distinct_taxa": one(
                "SELECT count(DISTINCT taxon_id) FROM candidates WHERE taxon_id IS NOT NULL"
            ),
        }

    def assert_no_forbidden_licenses(self, policy: Policy, corpus: str) -> None:
        """Hard gate: fail loudly if a corpus contains inadmissible media."""
        bad = self.con.execute(
            "SELECT c.license, count(*) FROM corpus_members m "
            "JOIN candidates c USING (candidate_id) "
            f"WHERE m.corpus = ? AND c.license NOT IN {policy.sql_in_list()} "
            "GROUP BY c.license",
            [corpus],
        ).fetchall()
        if bad:
            raise LicenseRefused(
                f"Corpus {corpus!r} contains media outside policy {policy.name!r}: {bad}"
            )


class ImageStore:
    """Content-addressed image store gated on provenance and licence policy."""

    def __init__(self, db: ProvenanceDB, policy: Policy, root: Path | None = None) -> None:
        self.db = db
        self.policy = policy
        self.root = Path(root or PATHS.cas)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha: str, ext: str) -> Path:
        ext = (ext or "jpg").lower().lstrip(".")
        return self.root / sha[:2] / sha[2:4] / f"{sha}.{ext}"

    def contains(self, sha: str, ext: str) -> bool:
        return self._path(sha, ext).exists()

    def put(
        self,
        data: bytes,
        prov: ImageProvenance,
        *,
        sha256: str,
        width: int | None = None,
        height: int | None = None,
        phash: str | None = None,
        dhash: str | None = None,
    ) -> Path:
        """Store ``data`` and its provenance atomically.

        Raises :class:`LicenseRefused` before touching the filesystem if the
        record's licence is not admitted by the active policy. This is the only
        supported way to add pixels to the store.
        """
        if not self.policy.admits(prov.license):
            raise LicenseRefused(
                f"Refusing to store {prov.source_dataset}:{prov.source_record_id} - "
                f"licence {prov.license.value} not admitted by policy "
                f"{self.policy.name!r}"
            )
        if prov.license is License.UNKNOWN:
            raise LicenseRefused("Refusing to store media with UNKNOWN licence")

        dest = self._path(sha256, prov.ext)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            # The temp name must be unique per writer, not derived from the
            # content hash. Two download threads can legitimately fetch the same
            # photograph at the same time (the same image reaches us under two
            # candidate ids), and a shared temp path made them collide - on
            # Windows that surfaces as WinError 32 "used by another process"
            # rather than as anything resembling the real cause.
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dest)
            except FileExistsError:
                # Another thread won the race. Identical content by definition,
                # since the path is the hash.
                tmp.unlink(missing_ok=True)
            except OSError:
                tmp.unlink(missing_ok=True)
                if not dest.exists():
                    raise

        cid = self.db.candidate_id(prov.source_dataset, prov.source_record_id)
        self.db.mark_stored(
            cid,
            sha256=sha256,
            nbytes=len(data),
            width=width,
            height=height,
            phash=phash,
            dhash=dhash,
            cas_path=str(dest.relative_to(self.root)).replace("\\", "/"),
        )
        return dest


def export_attributions(db: ProvenanceDB, out_dir: Path, corpus: str | None = None) -> dict:
    """Write ATTRIBUTIONS.csv / .json for a corpus (or the whole store).

    The CSV is the artefact that must ship with, or be linked from, any release
    that redistributes a model trained on attribution-requiring media.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if corpus:
        sql = (
            "SELECT p.* FROM provenance p JOIN corpus_members m "
            "ON m.candidate_id = p.candidate_id WHERE m.corpus = ?"
        )
        rel = db.con.execute(sql, [corpus])
    else:
        rel = db.con.execute("SELECT * FROM provenance")

    csv_path = out_dir / "ATTRIBUTIONS.csv"
    db.con.execute(
        "COPY (%s) TO '%s' (HEADER, DELIMITER ',')"
        % (
            (
                "SELECT image_id, accepted_scientific_name, source_dataset, "
                "source_record_id, source_url, image_url, creator, license, "
                "license_url, attribution FROM provenance p JOIN corpus_members m "
                "ON m.candidate_id = p.candidate_id WHERE m.corpus = '%s'" % corpus
                if corpus
                else "SELECT image_id, accepted_scientific_name, source_dataset, "
                "source_record_id, source_url, image_url, creator, license, "
                "license_url, attribution FROM provenance"
            ),
            csv_path.as_posix(),
        )
    )

    summary = {
        "generated_at": _utcnow().isoformat(),
        "corpus": corpus,
        "by_license": [
            {"license": lic, "images": n}
            for lic, n in db.con.execute(
                (
                    "SELECT license, count(*) FROM provenance p JOIN corpus_members m "
                    "ON m.candidate_id = p.candidate_id WHERE m.corpus = ? GROUP BY license"
                )
                if corpus
                else "SELECT license, count(*) FROM provenance GROUP BY license",
                [corpus] if corpus else [],
            ).fetchall()
        ],
    }
    (out_dir / "ATTRIBUTIONS.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
