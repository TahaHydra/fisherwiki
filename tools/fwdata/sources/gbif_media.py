"""GBIF occurrence-media acquisition for V2.

This adapter intentionally excludes the iNaturalist GBIF dataset mirror because
those photographs already enter FisherWiki through the native iNaturalist bulk
export. Discovery is per FisherWiki species using the registry's GBIF taxon id,
so every registered candidate already has a stable internal class id.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

from .. import net
from ..acquisition import ScanState, register_candidates
from ..licenses import normalize
from ..provenance import ImageProvenance, ProvenanceDB
from ..taxonomy.registry import TaxonRecord

API = "https://api.gbif.org/v1/occurrence/search"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
PAGE = 300


def _json(url: str) -> dict:
    return json.loads(net.fetch_bytes(url, accept="application/json").decode("utf-8"))


def _ext_from_url(url: str) -> str:
    tail = url.split("?", 1)[0].rsplit("/", 1)[-1]
    if "." in tail:
        ext = tail.rsplit(".", 1)[-1].lower()
        if ext in {"jpg", "jpeg", "png", "webp", "tif", "tiff"}:
            return ext
    return "jpg"


def _records(occ: dict, taxon: TaxonRecord) -> list[ImageProvenance]:
    if str(occ.get("datasetKey") or "") == INAT_DATASET_KEY:
        return []
    occ_key = str(occ.get("key") or occ.get("occurrenceID") or "")
    if not occ_key:
        return []
    species_name = taxon.canonical_name
    out: list[ImageProvenance] = []
    for i, media in enumerate(occ.get("media") or []):
        if str(media.get("type") or "").lower() not in {"stillimage", "image", ""}:
            continue
        url = media.get("identifier") or media.get("references")
        if not url or not str(url).startswith(("http://", "https://")):
            continue
        raw_license = str(media.get("license") or media.get("rights") or "")
        creator = media.get("creator") or occ.get("recordedBy")
        out.append(ImageProvenance(
            source_dataset="gbif-media",
            source_record_id=f"{occ_key}:{i}",
            image_url=str(url),
            source_url=str(occ.get("references") or f"https://www.gbif.org/occurrence/{occ_key}"),
            source_taxon_id=str(occ.get("speciesKey") or occ.get("taxonKey") or taxon.gbif_taxon_id or ""),
            original_scientific_name=str(occ.get("species") or occ.get("scientificName") or species_name),
            accepted_scientific_name=species_name,
            taxon_id=taxon.fw_taxon_id,
            license=normalize(raw_license),
            license_raw=raw_license,
            creator=str(creator) if creator else None,
            copyright_holder=str(media.get("rightsHolder") or creator) if (media.get("rightsHolder") or creator) else None,
            group_key=f"gbif-occ:{occ_key}",
            observer_key=str(creator) if creator else None,
            latitude=occ.get("decimalLatitude"),
            longitude=occ.get("decimalLongitude"),
            positional_accuracy=occ.get("coordinateUncertaintyInMeters"),
            observed_on=str(occ.get("eventDate") or occ.get("dateIdentified") or "") or None,
            country_code=occ.get("countryCode"),
            quality_grade=str(occ.get("basisOfRecord") or ""),
            declared_width=media.get("width"),
            declared_height=media.get("height"),
            original_filename=media.get("title"),
            ext=_ext_from_url(str(url)),
            context_tag="specimen" if "PRESERVED" in str(occ.get("basisOfRecord") or "").upper() else None,
            notes=f"gbif_dataset={occ.get('datasetKey') or ''}",
        ))
    return out


def discover_taxon(
    db: ProvenanceDB,
    taxon: TaxonRecord,
    *,
    cap: int = 300,
    state: ScanState | None = None,
    log=print,
) -> int:
    """Register up to ``cap`` non-iNaturalist GBIF image records for one species."""
    if taxon.gbif_taxon_id is None:
        return 0
    state = state or ScanState("gbif-media")
    if not state.needs(taxon.fw_taxon_id, cap):
        return 0

    registered = 0
    offset = 0
    seen_media = 0
    while seen_media < cap:
        limit = min(PAGE, max(1, cap - seen_media))
        # GBIF occurrence-search parameters are camelCase; using snake_case is
        # silently ignored and turns a species query into a global one.
        url = API + "?" + urlencode({
            "mediaType": "StillImage",
            "taxonKey": taxon.gbif_taxon_id,
            "limit": limit,
            "offset": offset,
        })
        obj = _json(url)
        results = obj.get("results") or []
        if not results:
            break
        records: list[ImageProvenance] = []
        for occ in results:
            recs = _records(occ, taxon)
            if recs:
                records.extend(recs)
                seen_media += len(recs)
                if seen_media >= cap:
                    break
        if records:
            registered += register_candidates(db, records)
        offset += len(results)
        if obj.get("endOfRecords") or len(results) < limit or offset >= 100000:
            break
    db.flush()
    state.mark(taxon.fw_taxon_id, cap)
    if registered:
        log(f"  GBIF {taxon.canonical_name}: +{registered:,} candidates")
    return registered
