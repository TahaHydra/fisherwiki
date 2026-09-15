#!/usr/bin/env python
"""Estimate how many usable fish photographs each external source would add.

    python tools/source_census.py

Bounded, read-only probes against public APIs - no images are downloaded and
nothing is written to the provenance store. The point is to answer "is an
adapter for this source worth writing" *before* writing it.

The number that matters is not how many photographs a source holds. It is how
many it holds that are (a) commercially redistributable, (b) not already in our
CAS, and (c) attached to a species-level scientific name. GBIF in particular
looks enormous and is mostly a mirror of iNaturalist, which this project has
already exhausted; the probe therefore breaks GBIF down by publisher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

UA = {"User-Agent": "FisherWiki/0.1 (https://github.com/TahaHydra/fisherwiki)"}
TIMEOUT = 40

#: GBIF publisher/dataset keys we already hold, so their occurrences are not
#: counted as new supply.
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"


def get(url: str, *, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                return {"_error": str(exc)[:160]}
            time.sleep(2 * (attempt + 1))
    return {}


def gbif_fish_keys() -> dict[str, int]:
    """Class-level keys for the fish groups, resolved rather than hardcoded.

    GBIF's backbone does not expose Actinopterygii as a usable class key, so
    the ray-finned key is taken from the classification of a known species
    instead of guessed.
    """
    keys: dict[str, int] = {}
    match = get("https://api.gbif.org/v1/species/match?name=Perca%20fluviatilis")
    if match.get("classKey"):
        keys[match.get("class") or "ray-finned"] = match["classKey"]
    for name in ("Elasmobranchii", "Myxini", "Petromyzonti", "Holocephali"):
        m = get(f"https://api.gbif.org/v1/species/match?name={name}&rank=CLASS")
        if m.get("usageKey"):
            keys[name] = m["usageKey"]
    return keys


def probe_gbif() -> dict:
    keys = gbif_fish_keys()
    out = {"keys": keys, "groups": {}, "publishers": {}}
    total = 0
    for name, key in keys.items():
        d = get("https://api.gbif.org/v1/occurrence/search"
                f"?mediaType=StillImage&taxonKey={key}&limit=0")
        n = d.get("count", 0)
        out["groups"][name] = n
        total += n
    out["with_images_total"] = total

    # Which publishers those images come from. iNaturalist is already ours; the
    # rest is the real marginal supply.
    big = max(keys.values(), default=None)
    if big:
        d = get("https://api.gbif.org/v1/occurrence/search"
                f"?mediaType=StillImage&taxonKey={big}&limit=0"
                "&facet=datasetKey&facetLimit=12")
        for facet in d.get("facets", []):
            for c in facet.get("counts", []):
                out["publishers"][c["name"]] = c["count"]
        inat = out["publishers"].get(INAT_DATASET, 0)
        out["ray_finned_excluding_inat"] = out["groups"].get(
            next((k for k, v in keys.items() if v == big), ""), 0) - inat
    return out


def probe_commons() -> dict:
    """Wikimedia Commons: how many files sit under the fish category trees.

    Commons has no count endpoint, so this uses the category-member count that
    the category info API exposes, which is exact for direct members and a
    lower bound for the tree.
    """
    out = {"categories": {}}
    cats = ["Category:Actinopterygii", "Category:Chondrichthyes",
            "Category:Fish by scientific name", "Category:Fish of the world"]
    titles = "|".join(urllib.parse.quote(c) for c in cats)
    d = get("https://commons.wikimedia.org/w/api.php?action=query&format=json"
            f"&prop=categoryinfo&titles={titles}")
    for page in (d.get("query", {}).get("pages", {}) or {}).values():
        info = page.get("categoryinfo") or {}
        out["categories"][page.get("title", "?")] = {
            "files": info.get("files", 0), "subcats": info.get("subcats", 0)}

    # Structured data is the better route: Commons files whose depicts (P180)
    # resolves to a taxon. Sample the search to size it.
    d2 = get("https://commons.wikimedia.org/w/api.php?action=query&format=json"
             "&list=search&srsearch=incategory%3AActinopterygii%20filemime%3Aimage%2Fjpeg"
             "&srnamespace=6&srlimit=1&srinfo=totalhits")
    out["search_totalhits_actinopterygii"] = (
        d2.get("query", {}).get("searchinfo", {}).get("totalhits"))
    return out


def probe_fathomnet() -> dict:
    """FathomNet: MBARI's underwater imagery with human-drawn boxes."""
    out = {}
    base = "https://database.fathomnet.org/api"
    d = get(f"{base}/darwincore/count/all")
    if "_error" not in d:
        out["darwincore_count"] = d
    stats = get(f"{base}/boundingboxes/count/all")
    if "_error" not in stats:
        out["bounding_boxes"] = stats
    concepts = get(f"{base}/boundingboxes/concepts")
    if isinstance(concepts, list):
        out["distinct_concepts"] = len(concepts)
    imgs = get(f"{base}/images/count/all")
    if "_error" not in imgs:
        out["images"] = imgs
    if not out:
        out["note"] = "FathomNet API did not answer; probe again or use fathomnet-py"
    return out


PROBES = {"gbif": probe_gbif, "commons": probe_commons, "fathomnet": probe_fathomnet}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=",".join(PROBES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    report = {}
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        probe = PROBES.get(name)
        if probe is None:
            print(f"unknown source {name!r}")
            continue
        t = time.time()
        print(f"probing {name} ...", flush=True)
        try:
            report[name] = probe()
        except Exception as exc:                      # a probe must never be fatal
            report[name] = {"_error": repr(exc)[:200]}
        report[name]["_seconds"] = round(time.time() - t, 1)
        print(json.dumps(report[name], indent=2)[:2200], flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
