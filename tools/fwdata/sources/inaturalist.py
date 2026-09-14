"""iNaturalist Open Data adapter (AWS Registry of Open Data).

Source of truth
---------------
``s3://inaturalist-open-data`` (us-east-1), mirrored over plain HTTPS at
``https://inaturalist-open-data.s3.amazonaws.com/``.  iNaturalist explicitly
directs large-scale machine-learning users at this bulk export instead of the
public API, so this adapter never touches ``api.inaturalist.org``.

Files (regenerated monthly, tab-separated, gzip):

===========================  ==========================================
``taxa.csv.gz``              taxon_id, ancestry, rank_level, rank, name, active
``observations.csv.gz``      observation_uuid, observer_id, latitude, longitude,
                             positional_accuracy, taxon_id, quality_grade, observed_on
``photos.csv.gz``            photo_uuid, photo_id, observation_uuid, observer_id,
                             extension, license, width, height, position
``observers.csv.gz``         observer_id, login, name
===========================  ==========================================

Photo URLs follow ``https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{ext}``
with ``size`` in {original, large, medium, small, thumb, square}.

Licensing
---------
The ``license`` column is per-photograph and is the *only* licence that matters
for us; the observation record licence is separate and is deliberately not used.
An **empty** licence column means All Rights Reserved - those photos are not
even present in this export, but we still normalise defensively.

Attribution is built from ``observers.name`` falling back to ``observers.login``.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

from .. import net
from ..config import PATHS

BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/"
PHOTO_URL = BASE_URL + "photos/{photo_id}/{size}.{ext}"

#: Image size variant to pull. ``medium`` is max-500px on the long edge which is
#: comfortably above our 224/320px training resolution while being ~10x smaller
#: to download than ``original``.
DEFAULT_SIZE = "medium"

BULK_FILES = {
    "taxa": "taxa.csv.gz",
    "observers": "observers.csv.gz",
    "observations": "observations.csv.gz",
    "photos": "photos.csv.gz",
}

#: Column types for DuckDB. Declared explicitly so a schema change upstream
#: fails loudly instead of being silently re-inferred.
SCHEMAS: dict[str, dict[str, str]] = {
    "taxa": {
        "taxon_id": "BIGINT",
        "ancestry": "VARCHAR",
        "rank_level": "DOUBLE",
        "rank": "VARCHAR",
        "name": "VARCHAR",
        "active": "VARCHAR",
    },
    "observers": {"observer_id": "BIGINT", "login": "VARCHAR", "name": "VARCHAR"},
    "observations": {
        "observation_uuid": "VARCHAR",
        "observer_id": "BIGINT",
        "latitude": "DOUBLE",
        "longitude": "DOUBLE",
        "positional_accuracy": "BIGINT",
        "taxon_id": "BIGINT",
        "quality_grade": "VARCHAR",
        "observed_on": "VARCHAR",
        # Added by iNaturalist since the README was written; present in the
        # 2026-08-27 snapshot. Higher values flag observations that look
        # geographically or temporally unusual for the taxon, which makes it a
        # useful (if noisy) label-quality signal. Carried through to the
        # candidate table rather than discarded.
        "anomaly_score": "DOUBLE",
    },
    "photos": {
        "photo_uuid": "VARCHAR",
        "photo_id": "BIGINT",
        "observation_uuid": "VARCHAR",
        "observer_id": "BIGINT",
        "extension": "VARCHAR",
        "license": "VARCHAR",
        "width": "BIGINT",
        "height": "BIGINT",
        "position": "BIGINT",
    },
}

#: iNaturalist taxon ids for the clades an angler would call "a fish".
#: Deliberately paraphyletic - this is a product definition, not a cladogram.
#: Verified against the 2026-08-27 taxa export:
#:   Actinopterygii 47178   ray-finned fishes      (~35.8k active species)
#:   Chondrichthyes 196614  sharks/rays/chimaeras  (~1.3k)
#:   Myxini          49099  hagfishes              (~89)
#:   Petromyzonti    49231  lampreys               (~49)
#:   Sarcopterygii   85497  lungfish + coelacanths (8; iNat places tetrapods
#:                          as siblings under Vertebrata, so this does NOT
#:                          drag in birds/mammals - checked explicitly)
FISH_ROOT_TAXA: dict[int, str] = {
    47178: "Actinopterygii",
    196614: "Chondrichthyes",
    49099: "Myxini",
    49231: "Petromyzonti",
    85497: "Sarcopterygii",
}


@dataclass(frozen=True)
class BulkFile:
    key: str
    name: str
    url: str
    size: int | None

    @property
    def local(self) -> Path:
        return PATHS.raw_source("inaturalist") / self.name


def discover() -> list[BulkFile]:
    """HEAD every bulk file and report its published size."""
    out = []
    for key, name in BULK_FILES.items():
        url = BASE_URL + name
        out.append(BulkFile(key, name, url, net.content_length(url)))
    return out


def read_csv_sql(key: str, path: Path | None = None) -> str:
    """Build a DuckDB ``read_csv`` expression with the pinned schema."""
    p = (path or (PATHS.raw_source("inaturalist") / BULK_FILES[key])).as_posix()
    cols = ", ".join(f"'{k}':'{v}'" for k, v in SCHEMAS[key].items())
    return (
        f"read_csv('{p}', delim='\\t', header=true, quote='', escape='', "
        f"columns={{{cols}}})"
    )


def photo_url(photo_id: int, ext: str, size: str = DEFAULT_SIZE) -> str:
    ext = (ext or "jpg").lower().lstrip(".")
    return PHOTO_URL.format(photo_id=photo_id, size=size, ext=ext)


def observation_url(observation_uuid: str) -> str:
    return f"https://www.inaturalist.org/observations/{observation_uuid}"


def download_bulk(
    keys: list[str] | None = None,
    *,
    workers: int = 8,
    progress: net.Progress | None = None,
) -> dict[str, str]:
    """Fetch the requested bulk files, resuming anything partially present."""
    keys = keys or list(BULK_FILES)
    files = [
        BulkFile(k, BULK_FILES[k], BASE_URL + BULK_FILES[k], None) for k in keys
    ]
    digests: dict[str, str] = {}
    for f in files:
        digests[f.key] = net.download_file_parallel(
            f.url, f.local, workers=workers, progress=progress
        )
    return digests


def peek(key: str, n: int = 5) -> list[str]:
    """Return the first ``n`` raw lines of a downloaded bulk file."""
    path = PATHS.raw_source("inaturalist") / BULK_FILES[key]
    out: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.append(line.rstrip("\n"))
    return out
