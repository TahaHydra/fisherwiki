"""Wikimedia Commons acquisition by scientific-name category.

Commons species categories are usually named exactly after the scientific name.
That gives a much cleaner training label than free-text search: category members
are queried in namespace 6 (files) and then resolved to original image URLs and
embedded metadata in one API call.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

from .. import net
from ..acquisition import ScanState, register_candidates
from ..licenses import normalize
from ..provenance import ImageProvenance, ProvenanceDB
from ..taxonomy.registry import TaxonRecord

API = "https://commons.wikimedia.org/w/api.php"
PAGE = 50


def _json(params: dict) -> dict:
    url = API + "?" + urlencode(params)
    return json.loads(net.fetch_bytes(url, accept="application/json").decode("utf-8"))


def _strip_html(v: object) -> str | None:
    if v is None:
        return None
    s = str(v)
    # Commons extmetadata often returns tiny HTML fragments. Attribution does
    # not need perfect rendering; stripping tags is enough for provenance.
    import re
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip() or None


def _meta_value(meta: dict, key: str) -> str:
    v = meta.get(key)
    if isinstance(v, dict):
        return str(v.get("value") or "")
    return str(v or "")


def _ext(url: str) -> str:
    path = url.split("?", 1)[0]
    e = path.rsplit(".", 1)[-1].lower() if "." in path else "jpg"
    return e if e in {"jpg", "jpeg", "png", "webp", "tif", "tiff"} else "jpg"


def discover_taxon(
    db: ProvenanceDB,
    taxon: TaxonRecord,
    *,
    cap: int = 40,
    state: ScanState | None = None,
    log=print,
) -> int:
    state = state or ScanState("wikimedia-commons")
    if not state.needs(taxon.fw_taxon_id, cap):
        return 0

    category = f"Category:{taxon.canonical_name}"
    cont: str | None = None
    titles: list[tuple[int, str]] = []
    while len(titles) < cap:
        params = {
            "action": "query", "format": "json", "formatversion": 2,
            "list": "categorymembers", "cmtitle": category,
            "cmnamespace": 6, "cmtype": "file", "cmlimit": min(500, cap - len(titles)),
        }
        if cont:
            params["cmcontinue"] = cont
        obj = _json(params)
        for row in obj.get("query", {}).get("categorymembers", []):
            titles.append((int(row["pageid"]), row["title"]))
            if len(titles) >= cap:
                break
        cont = (obj.get("continue") or {}).get("cmcontinue")
        if not cont:
            break

    registered = 0
    for start in range(0, len(titles), PAGE):
        chunk = titles[start:start + PAGE]
        if not chunk:
            continue
        obj = _json({
            "action": "query", "format": "json", "formatversion": 2,
            "pageids": "|".join(str(x[0]) for x in chunk),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|Copyrighted",
        })
        records: list[ImageProvenance] = []
        for page in obj.get("query", {}).get("pages", []):
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                continue
            mime = str(ii.get("mime") or "")
            if mime and not mime.startswith("image/"):
                continue
            url = ii.get("url")
            if not url:
                continue
            meta = ii.get("extmetadata") or {}
            raw_license = _meta_value(meta, "LicenseShortName") or _meta_value(meta, "LicenseUrl")
            creator = _strip_html(_meta_value(meta, "Artist"))
            pageid = str(page.get("pageid"))
            records.append(ImageProvenance(
                source_dataset="wikimedia-commons",
                source_record_id=pageid,
                image_url=str(url),
                source_url=str(ii.get("descriptionurl") or f"https://commons.wikimedia.org/?curid={pageid}"),
                source_taxon_id=taxon.wikidata_qid or str(taxon.fw_taxon_id),
                original_scientific_name=taxon.canonical_name,
                accepted_scientific_name=taxon.canonical_name,
                taxon_id=taxon.fw_taxon_id,
                license=normalize(raw_license),
                license_raw=raw_license,
                creator=creator,
                copyright_holder=creator,
                attribution=_strip_html(_meta_value(meta, "Credit")),
                group_key=f"commons:{pageid}",
                observer_key=creator,
                declared_width=ii.get("width"),
                declared_height=ii.get("height"),
                original_filename=str(page.get("title") or "").removeprefix("File:"),
                ext=_ext(str(url)),
                notes=f"commons_category={category}",
            ))
        if records:
            registered += register_candidates(db, records)
    db.flush()
    state.mark(taxon.fw_taxon_id, cap)
    if registered:
        log(f"  Commons {taxon.canonical_name}: +{registered:,} candidates")
    return registered
