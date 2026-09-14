"""GBIF adapter: Backbone Taxonomy (bulk) and occurrence media metadata.

Two distinct uses, with two distinct licences - keeping them apart is the whole
reason this module is explicit about which is which.

**Backbone Taxonomy** (``CC BY 4.0``, verified via
``api.gbif.org/v1/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c``) is a bulk
checklist download. It gives us accepted-vs-synonym status, the accepted-name
pointer, canonical names and the full rank ladder. This is what powers
:mod:`fwdata.taxonomy`.

**Occurrence media** is *not* covered by the backbone licence, and the licence
on an occurrence *record* is not the licence on its attached *photograph*.
Any future media harvesting through GBIF must read the media-level licence and
push it through :func:`fwdata.licenses.normalize` like every other source.

The backbone is a versioned snapshot. We pin an explicit version rather than
following ``current/`` so that a rebuild months later reproduces the same
taxonomy; the version and its SHA-256 are recorded in the corpus manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import net
from ..config import PATHS

#: Pinned backbone release. Latest available as of 2026-09-13; the directory
#: listing at https://hosted-datasets.gbif.org/datasets/backbone/ shows this is
#: also what ``current/`` resolves to.
BACKBONE_VERSION = "2023-08-28"
BACKBONE_BASE = f"https://hosted-datasets.gbif.org/datasets/backbone/{BACKBONE_VERSION}/"

#: The simplified flat export. ~489 MB gzipped, tab separated, no quoting.
#: Preferred over the 971 MB full DwC-A because we need exactly these columns.
SIMPLE_URL = BACKBONE_BASE + "simple.txt.gz"

DATASET_KEY = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"
DATASET_LICENSE = "http://creativecommons.org/licenses/by/4.0/legalcode"
DATASET_CITATION = (
    "GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset "
    f"https://doi.org/10.15468/39omei accessed via GBIF.org. Version {BACKBONE_VERSION}. "
    "Licensed CC BY 4.0."
)

#: Column order of ``simple.txt.gz``. GBIF ships **no header row** and the
#: release README does not document the layout, so this was derived from the
#: data and then validated against ``api.gbif.org/v1/species/{key}`` for known
#: taxa (Perca fluviatilis 8140485 -> parent genus Perca 2334781 -> family
#: Percidae 4481 -> order Perciformes 587). The order is load-bearing; a schema
#: drift upstream will surface as a type error rather than silent mis-parsing.
#:
#: ``parent_or_accepted_key`` is overloaded by GBIF: for an accepted taxon it is
#: the parent usage, for a synonym it is the *accepted* usage. Use
#: ``is_synonym`` to decide which meaning applies.
SIMPLE_COLUMNS: list[tuple[str, str]] = [
    ("taxon_key", "BIGINT"),                 # 0
    ("parent_or_accepted_key", "BIGINT"),    # 1  see note above
    ("basionym_key", "BIGINT"),              # 2
    ("is_synonym", "VARCHAR"),               # 3  't' / 'f'
    ("status", "VARCHAR"),                   # 4  ACCEPTED/SYNONYM/DOUBTFUL/...
    ("rank", "VARCHAR"),                     # 5
    ("nomenclatural_status", "VARCHAR"),     # 6  postgres array literal
    ("constituent_key", "VARCHAR"),          # 7  source dataset uuid
    ("origin", "VARCHAR"),                   # 8
    ("source_taxon_key", "BIGINT"),          # 9
    ("kingdom_key", "BIGINT"),               # 10
    ("phylum_key", "BIGINT"),                # 11
    ("class_key", "BIGINT"),                 # 12  frequently NULL for fishes
    ("order_key", "BIGINT"),                 # 13
    ("family_key", "BIGINT"),                # 14
    ("genus_key", "BIGINT"),                 # 15
    ("species_key", "BIGINT"),               # 16
    ("name_key", "BIGINT"),                  # 17
    ("scientific_name", "VARCHAR"),          # 18  includes authorship
    ("canonical_name", "VARCHAR"),           # 19  no authorship
    ("generic_name", "VARCHAR"),             # 20
    ("specific_epithet", "VARCHAR"),         # 21
    ("infraspecific_epithet", "VARCHAR"),    # 22
    ("notho_rank", "VARCHAR"),               # 23  hybrid marker
    ("authorship", "VARCHAR"),               # 24
    ("year", "VARCHAR"),                     # 25
    ("bracket_authorship", "VARCHAR"),       # 26
    ("bracket_year", "VARCHAR"),             # 27
    ("published_in", "VARCHAR"),             # 28
    ("issues", "VARCHAR"),                   # 29  postgres array literal
]

#: GBIF's backbone does **not** link fish orders to class Actinopterygii (204):
#: ``Perciformes`` (587) has ``parentKey`` 44 = phylum Chordata, and
#: ``classKey`` is NULL for every descendant. Verified against the live API.
#: Consequence: we cannot select fish from GBIF by class. The fish taxon set is
#: therefore driven by iNaturalist (which does have a coherent fish hierarchy)
#: and GBIF is joined **by name** purely to supply synonyms and a second
#: opinion on accepted status.
CLASS_KEYS_UNRELIABLE_FOR_FISH = True


@dataclass(frozen=True)
class BackboneFile:
    url: str
    local: Path
    version: str


def backbone_file() -> BackboneFile:
    return BackboneFile(
        SIMPLE_URL,
        PATHS.raw_source("gbif") / f"backbone-simple-{BACKBONE_VERSION}.txt.gz",
        BACKBONE_VERSION,
    )


def download_backbone(progress: net.Progress | None = None, workers: int = 8) -> str:
    f = backbone_file()
    return net.download_file_parallel(
        f.url, f.local, workers=workers, progress=progress
    )


def read_backbone_sql(path: Path | None = None) -> str:
    """DuckDB ``read_csv`` expression for the pinned backbone export.

    Two details here are load-bearing and were both got wrong first time:

    ``nullstr`` must be the two characters ``\\N``. DuckDB does not process
    escape sequences inside this option, so over-escaping it to ``\\\\N``
    means the sentinel never matches and every ``\\N`` stays a literal string.

    ``ignore_errors`` is deliberately **off**. With it on, the mis-set
    ``nullstr`` above caused DuckDB to silently drop every row containing
    ``\\N`` in a BIGINT column - and because the CSV reader pushes projections
    down, the damage depended on which columns a query selected. A diagnostic
    reading only VARCHAR columns saw all 43k fish names; the real query using
    ``SELECT *`` saw 1,262. Failing loudly on a cast error is far better than a
    corpus that quietly loses 97% of its taxonomy.
    """
    p = (path or backbone_file().local).as_posix()
    cols = ", ".join(f"'{n}':'{t}'" for n, t in SIMPLE_COLUMNS)
    return (
        f"read_csv('{p}', delim='\\t', header=false, quote='', escape='', "
        f"nullstr='\\N', columns={{{cols}}})"
    )


def species_url(taxon_key: int) -> str:
    return f"https://www.gbif.org/species/{taxon_key}"
