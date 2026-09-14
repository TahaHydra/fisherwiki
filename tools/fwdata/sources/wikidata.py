"""Wikidata adapter: multilingual common names, IUCN status and cross-ids.

Why Wikidata
------------
FishBase is the obvious source for fish facts and is **CC BY-NC**, so it cannot
be used in a commercially distributable app. Wikidata is **CC0**, which makes it
the only large, structured, machine-readable source of vernacular names we can
actually ship. It also carries the identifier cross-walk (GBIF, WoRMS, iNat,
FishBase) that lets a user follow a fact back to its origin.

Coverage is uneven - some species have twenty names in twelve languages, many
have none - but a missing name is a blank field, not a wrong one, which is
exactly the failure mode this project prefers.

Properties used
---------------
=========  ==================================================
``P225``   taxon name (the scientific name; our join key)
``P1843``  taxon common name, language-tagged
``P105``   taxon rank
``P141``   IUCN conservation status
``P846``   GBIF taxon id
``P850``   WoRMS AphiaID
``P3151``  iNaturalist taxon id
``P938``   FishBase species id (recorded as a pointer only)
``P2043``  length
=========  ==================================================

Access
------
The Wikidata Query Service is the documented way to retrieve a subset like
this; we are not scraping pages. Requests are batched (``VALUES`` blocks of
~150 names), rate-limited through the shared token bucket, and sent with a
descriptive User-Agent as WMF policy requires. Roughly a dozen queries cover
our whole class list, which is well inside acceptable use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .. import net

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

#: Names per request. Large enough to be few requests, small enough to stay
#: well under the 60-second query timeout.
BATCH_SIZE = 150

SOURCE_ID = "wikidata"
SOURCE_LICENSE = "CC0-1.0"
SOURCE_CITATION = (
    "Wikidata contributors. Wikidata, the free knowledge base. "
    "Available under CC0 1.0 Universal. https://www.wikidata.org/"
)

QUERY_TEMPLATE = """
SELECT ?taxonName ?item ?itemLabel ?rankLabel ?iucnLabel
       ?gbif ?worms ?inat ?fishbase ?common ?commonLang
WHERE {
  VALUES ?taxonName { %(values)s }
  ?item wdt:P225 ?taxonName .
  OPTIONAL { ?item wdt:P105 ?rank . }
  OPTIONAL { ?item wdt:P141 ?iucn . }
  OPTIONAL { ?item wdt:P846  ?gbif . }
  OPTIONAL { ?item wdt:P850  ?worms . }
  OPTIONAL { ?item wdt:P3151 ?inat . }
  OPTIONAL { ?item wdt:P938  ?fishbase . }
  OPTIONAL {
    ?item wdt:P1843 ?common .
    BIND(LANG(?common) AS ?commonLang)
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


@dataclass
class WikidataTaxon:
    scientific_name: str
    qid: str | None = None
    label: str | None = None
    rank: str | None = None
    iucn_status: str | None = None
    gbif_taxon_id: int | None = None
    worms_aphia_id: int | None = None
    inat_taxon_id: int | None = None
    fishbase_id: str | None = None
    #: (lang, name) pairs, deduplicated, language codes as returned.
    common_names: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["common_names"] = [list(x) for x in self.common_names]
        return d


def _chunks(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _sparql_literal(s: str) -> str:
    """Escape a name for a SPARQL string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _qid_from_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rsplit("/", 1)[-1] or None


def _int_or_none(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def fetch(
    scientific_names: Iterable[str],
    *,
    batch_size: int = BATCH_SIZE,
    languages: set[str] | None = None,
    log=print,
) -> dict[str, WikidataTaxon]:
    """Look up taxa by scientific name. Returns ``{scientific_name: taxon}``.

    Names with no Wikidata match are simply absent from the result; that is a
    coverage gap, not an error.
    """
    names = sorted({n for n in scientific_names if n})
    out: dict[str, WikidataTaxon] = {}
    batches = list(_chunks(names, batch_size))
    log(f"wikidata: {len(names):,} names in {len(batches)} batches")

    for bi, batch in enumerate(batches, 1):
        query = QUERY_TEMPLATE % {
            "values": " ".join(_sparql_literal(n) for n in batch)
        }
        url = SPARQL_ENDPOINT + "?query=" + _urlquote(query) + "&format=json"
        try:
            raw = net.fetch_bytes(url, accept="application/sparql-results+json",
                                  timeout=(15, 120))
        except net.DownloadError as exc:
            log(f"  batch {bi}/{len(batches)} failed: {exc}")
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log(f"  batch {bi}/{len(batches)}: unparseable response")
            continue

        for row in data.get("results", {}).get("bindings", []):
            name = row.get("taxonName", {}).get("value")
            if not name:
                continue
            tx = out.get(name)
            if tx is None:
                tx = WikidataTaxon(scientific_name=name)
                out[name] = tx
            tx.qid = tx.qid or _qid_from_uri(row.get("item", {}).get("value"))
            tx.label = tx.label or row.get("itemLabel", {}).get("value")
            tx.rank = tx.rank or row.get("rankLabel", {}).get("value")
            tx.iucn_status = tx.iucn_status or row.get("iucnLabel", {}).get("value")
            tx.gbif_taxon_id = tx.gbif_taxon_id or _int_or_none(
                row.get("gbif", {}).get("value")
            )
            tx.worms_aphia_id = tx.worms_aphia_id or _int_or_none(
                row.get("worms", {}).get("value")
            )
            tx.inat_taxon_id = tx.inat_taxon_id or _int_or_none(
                row.get("inat", {}).get("value")
            )
            tx.fishbase_id = tx.fishbase_id or row.get("fishbase", {}).get("value")

            common = row.get("common", {}).get("value")
            lang = row.get("commonLang", {}).get("value")
            if common and lang:
                if languages and lang.split("-")[0] not in languages:
                    continue
                pair = (lang, common)
                if pair not in tx.common_names:
                    tx.common_names.append(pair)

        log(f"  batch {bi}/{len(batches)}: {len(out):,} taxa so far")
        # Be gentle with a shared public endpoint beyond the token bucket.
        time.sleep(1.0)

    return out


def _urlquote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


def source_record(retrieved_on: str):
    """Source row for :mod:`fwdata.speciesdb`."""
    from ..speciesdb import Source

    return Source(
        source_id=SOURCE_ID,
        title="Wikidata",
        publisher="Wikimedia Foundation",
        url="https://www.wikidata.org/",
        license=SOURCE_LICENSE,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        retrieved_on=retrieved_on,
        citation=SOURCE_CITATION,
        notes=(
            "Retrieved via the Wikidata Query Service. Properties used: P225 "
            "taxon name, P1843 common name, P105 rank, P141 IUCN status, "
            "P846 GBIF, P850 WoRMS, P3151 iNaturalist, P938 FishBase."
        ),
    )
