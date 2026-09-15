"""FathomNet underwater fish-image acquisition.

FathomNet exposes concept-labelled images and human-drawn boxes. We query by
scientific name through the public API. A single frame may contain multiple
species; FisherWiki registers one candidate identity per (image, taxon), and its
existing exact-hash/conflict quarantine prevents contradictory labels from
silently entering different model classes.
"""

from __future__ import annotations

import json
from urllib.parse import quote

from .. import net
from ..acquisition import ScanState, register_candidates
from ..licenses import normalize
from ..provenance import ImageProvenance, ProvenanceDB
from ..taxonomy.registry import TaxonRecord

API = "https://database.fathomnet.org/api/images/query/concept"


def _json(url: str):
    # FathomNet has had intermittent 503 periods. Discovery runs over thousands
    # of taxa, so spending the generic multi-retry budget on every taxon would
    # turn one provider outage into an all-night no-op. One short attempt here;
    # the orchestrator leaves failed taxa uncheckpointed and a later run retries.
    return json.loads(
        net.fetch_bytes(
            url,
            accept="application/json",
            max_attempts=1,
            timeout=(5, 15),
        ).decode("utf-8")
    )


def _ext(url: str) -> str:
    p = url.split("?", 1)[0]
    e = p.rsplit(".", 1)[-1].lower() if "." in p else "jpg"
    return e if e in {"jpg", "jpeg", "png", "webp", "tif", "tiff"} else "jpg"


def _concept_boxes(img: dict, name: str) -> list[dict]:
    want = name.strip().lower()
    return [
        b for b in (img.get("boundingBoxes") or [])
        if str(b.get("concept") or "").strip().lower() == want
        and not b.get("rejected", False)
    ]


def discover_taxon(
    db: ProvenanceDB,
    taxon: TaxonRecord,
    *,
    cap: int = 200,
    state: ScanState | None = None,
    log=print,
) -> int:
    state = state or ScanState("fathomnet")
    if not state.needs(taxon.fw_taxon_id, cap):
        return 0
    url = f"{API}/{quote(taxon.canonical_name, safe='')}"
    obj = _json(url)
    if not isinstance(obj, list):
        return 0

    records: list[ImageProvenance] = []
    for img in obj:
        boxes = _concept_boxes(img, taxon.canonical_name)
        if not boxes:
            continue
        image_url = img.get("url")
        uuid = str(img.get("uuid") or img.get("id") or "")
        if not image_url or not uuid:
            continue
        observers = sorted({str(b.get("observer")) for b in boxes if b.get("observer")})
        box = max(
            boxes,
            key=lambda b: int(b.get("width") or 0) * int(b.get("height") or 0),
        )
        raw_license = ""
        records.append(ImageProvenance(
            source_dataset="fathomnet",
            source_record_id=f"{uuid}:{taxon.fw_taxon_id}",
            image_url=str(image_url),
            source_url=f"https://fathomnet.org/fathomnet/#/image/{uuid}",
            source_taxon_id=str(taxon.worms_aphia_id or taxon.fw_taxon_id),
            original_scientific_name=taxon.canonical_name,
            accepted_scientific_name=taxon.canonical_name,
            taxon_id=taxon.fw_taxon_id,
            license=normalize(raw_license),
            license_raw=raw_license,
            creator=img.get("contributorsEmail"),
            copyright_holder=img.get("contributorsEmail"),
            group_key=f"fathomnet-image:{uuid}",
            observer_key="|".join(observers) or img.get("contributorsEmail"),
            latitude=img.get("latitude"),
            longitude=img.get("longitude"),
            observed_on=img.get("timestamp"),
            declared_width=img.get("width"),
            declared_height=img.get("height"),
            ext=_ext(str(image_url)),
            context_tag="underwater",
            notes=(
                f"fathomnet_box={box.get('x')},{box.get('y')},"
                f"{box.get('width')},{box.get('height')}"
            ),
        ))
        if len(records) >= cap:
            break
    n = register_candidates(db, records)
    db.flush()
    state.mark(taxon.fw_taxon_id, cap)
    if n:
        log(f"  FathomNet {taxon.canonical_name}: +{n:,} candidates")
    return n
