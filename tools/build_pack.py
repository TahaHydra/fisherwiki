#!/usr/bin/env python
"""Assemble a distributable FisherWiki offline pack.

    python tools/build_pack.py --run <run_dir> --pack-id global_v1 \
        --display-name "Global Angler" --quantization int8

A pack is a ZIP containing:

    manifest.json      identity, hashes, model spec, calibration, provenance
    model.onnx         the classifier
    labels.json        class index -> fw_taxon_id mapping
    species.sqlite     the offline species database
    geoprior.bin       per-class occurrence histogram
    ATTRIBUTIONS.csv   per-image attribution for the training corpus

The manifest records the SHA-256 and exact byte length of every payload file,
and `PackVerifier` on the device refuses anything that does not match. It also
records which licence policy built the corpus, so a research-only model can
never be mistaken for a shippable one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fwdata import confusion, geoprior, regions, safety  # noqa: E402
from fwdata.config import PATHS  # noqa: E402
from fwdata.licenses import get_policy  # noqa: E402
from fwdata.provenance import ProvenanceDB, export_attributions  # noqa: E402
from fwdata.speciesdb import Source, SpeciesDatabaseBuilder  # noqa: E402
from fwdata.sources import gbif  # noqa: E402
from fwdata.sources import wikidata as wd  # noqa: E402

PACK_FORMAT_VERSION = 1
ENGINE_VERSION = 1


def log(m: str = "") -> None:
    print(m, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_spec(path: Path, arcname: str) -> dict:
    return {"path": arcname, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_species_db(
    out_path: Path,
    labels: dict,
    corpus: str,
    today: str,
    run_dir: Path | None = None,
    log=log,
) -> dict:
    """Populate the pack's SQLite database from CC0/CC-BY sources."""
    import duckdb

    wikidata_path = PATHS.work / "wikidata_taxa.json"
    wiki = {}
    if wikidata_path.exists():
        wiki = json.loads(wikidata_path.read_text(encoding="utf-8"))
    else:
        log("  note: no wikidata_taxa.json; common names will be absent")

    tax_db = PATHS.work / "taxonomy.duckdb"
    ladder: dict[str, tuple] = {}
    synonyms: dict[str, list[str]] = {}
    if tax_db.exists():
        con = duckdb.connect(str(tax_db), read_only=True)
        for name, cls, order, fam, genus in con.execute(
            "SELECT canonical_name, class_name, order_name, family_name, genus_name "
            "FROM taxa_joined WHERE active"
        ).fetchall():
            ladder[name] = (cls, order, fam, genus)
        for acc, syn in con.execute(
            "SELECT accepted_canonical, synonym_canonical FROM synonyms"
        ).fetchall():
            synonyms.setdefault(acc, []).append(syn)
        con.close()

    with SpeciesDatabaseBuilder(out_path) as db:
        gbif_src = db.add_source(Source(
            source_id="gbif-backbone",
            title="GBIF Backbone Taxonomy",
            publisher="GBIF Secretariat",
            url="https://doi.org/10.15468/39omei",
            license="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            retrieved_on=today,
            citation=gbif.DATASET_CITATION,
        ))
        wd_src = db.add_source(wd.source_record(today))
        inat_src = db.add_source(Source(
            source_id="inaturalist-open-data",
            title="iNaturalist Licensed Observation Images",
            publisher="iNaturalist",
            url="https://registry.opendata.aws/inaturalist-open-data/",
            license="MIXED-CC",
            license_url="https://github.com/inaturalist/inaturalist-open-data",
            retrieved_on=today,
            citation=(
                "iNaturalist contributors, iNaturalist Open Data. Observation "
                "coordinates used to derive occurrence ranges. Individual "
                "photograph licences are recorded per image in ATTRIBUTIONS.csv."
            ),
        ))

        taxa_rows = []
        common_rows = []
        syn_rows = []
        for c in labels["classes"]:
            fw = c["fw_taxon_id"]
            sci = c["scientific_name"]
            cls, order, fam, genus = ladder.get(sci, (None, None, None, None))
            w = wiki.get(sci, {})
            taxa_rows.append({
                "fw_taxon_id": fw,
                "scientific_name": sci,
                "rank": "species",
                "genus": genus or sci.split(" ")[0],
                "family": fam,
                "order": order,
                "class": cls,
                "gbif_taxon_id": w.get("gbif_taxon_id"),
                "inat_taxon_id": w.get("inat_taxon_id"),
                "worms_aphia_id": w.get("worms_aphia_id"),
                "wikidata_qid": w.get("qid"),
            })
            seen_lang: set[str] = set()
            for pair in w.get("common_names", []):
                lang, name = pair[0], pair[1]
                preferred = 1 if lang not in seen_lang else 0
                seen_lang.add(lang)
                common_rows.append((fw, lang, name, preferred))
            for s in synonyms.get(sci, []):
                syn_rows.append((fw, s, "synonym"))

        db.add_taxa(taxa_rows, gbif_src)
        db.add_common_names(common_rows, wd_src)
        db.add_synonyms(syn_rows, gbif_src)
        db.add_regions(regions.as_dicts())

        # IUCN status recorded as a trait, cited to Wikidata rather than to
        # IUCN directly: we did not query the Red List API, and attributing a
        # fact to a source we did not consult would be dishonest.
        trait_rows = []
        for c in labels["classes"]:
            w = wiki.get(c["scientific_name"], {})
            if w.get("iucn_status"):
                trait_rows.append({
                    "fw_taxon_id": c["fw_taxon_id"],
                    "key": "iucn_status",
                    "value_text": w["iucn_status"],
                })
        db.add_traits(trait_rows, wd_src)

        # --- curated handling safety -------------------------------------
        # Expanded from rank-level evidence (Smith & Wheeler 2006 establish
        # venom for lineages, not individual species), so a species with thin
        # data still inherits its family's warning.
        safety_sources, safety_warnings = safety.load()
        for src in safety_sources:
            db.add_source(src)
        safety_taxa = [
            {
                "fw_taxon_id": r["fw_taxon_id"],
                "scientific_name": r["scientific_name"],
                "genus": r.get("genus"),
                "family": r.get("family"),
                "order": r.get("order"),
                "class": r.get("class"),
            }
            for r in taxa_rows
        ]
        safety_rows, safety_report = safety.expand(
            safety_warnings, safety_taxa, log=log
        )
        by_source: dict[str, list[dict]] = {}
        for row in safety_rows:
            by_source.setdefault(row["source_id"], []).append(row)
        for sid, rows_for_source in by_source.items():
            db.add_safety_warnings(rows_for_source, sid)
        log(f"    safety: {safety_report['rows_generated']} warnings over "
            f"{safety_report['species_with_a_warning']} species")

        # --- measured model behaviour ------------------------------------
        # Per-class precision/recall and the confusion pairs come from the
        # held-out test evaluation. They are facts about this model, not about
        # the fish, and they are sourced as such.
        metrics = confusion.load_metrics(run_dir) if run_dir else None
        if metrics is None:
            log("    note: no class_metrics_test.json in the run directory; "
                "shipping without per-class accuracy or similar-species data. "
                "Run ml/evaluate.py --split test first.")
        per_class = confusion.class_metrics_by_index(metrics) if metrics else {}

        db.add_model_classes([
            {
                "class_index": c["class_id"],
                "fw_taxon_id": c["fw_taxon_id"],
                "train_images": c.get("images", 0),
                "train_observations": c.get("observations", 0),
                "test_precision": per_class.get(c["class_id"], {}).get("precision"),
                "test_recall": per_class.get(c["class_id"], {}).get("recall"),
                "test_support": per_class.get(c["class_id"], {}).get("support"),
            }
            for c in labels["classes"]
        ])
        if per_class:
            log(f"    per-class accuracy attached for {len(per_class)} classes")

        if metrics:
            eval_src = db.add_source(Source(
                source_id="fisherwiki-eval",
                title="FisherWiki held-out test evaluation",
                publisher="FisherWiki",
                url=None,
                license="CC-BY-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                retrieved_on=today,
                citation=(
                    "FisherWiki model evaluation on the held-out test split of "
                    f"corpus {corpus}. Measured, not asserted: rows sourced here "
                    "describe this model's behaviour, not the biology of the "
                    "species."
                ),
                notes=(
                    "similar_species rows from this source record which species "
                    "the model confuses and how often. They carry no "
                    "morphological claim."
                ),
            ))
            class_to_taxon = {c["class_id"]: c["fw_taxon_id"]
                              for c in labels["classes"]}
            dangerous = {
                int(r[0]) for r in db.con.execute(
                    "SELECT DISTINCT fw_taxon_id FROM safety_warnings "
                    "WHERE severity = 'danger'"
                ).fetchall()
            }
            similar_rows, conf_report = confusion.build_similar_species(
                metrics, class_to_taxon, dangerous, log=log
            )
            if similar_rows:
                db.add_similar_species(similar_rows, eval_src)

        # Region membership, measured from observation coordinates.
        prov = ProvenanceDB(read_only=True)
        region_rows = []
        for rid, reg in regions.REGIONS.items():
            pred = reg.sql_predicate("p.latitude", "p.longitude")
            rows = prov.con.execute(
                f"""
                SELECT m.taxon_id,
                       count(DISTINCT p.group_key) FILTER (WHERE {pred}) AS inr,
                       count(DISTINCT p.group_key) AS tot
                FROM corpus_members m
                JOIN provenance p USING (candidate_id)
                WHERE m.corpus = ? AND p.latitude IS NOT NULL
                GROUP BY m.taxon_id
                HAVING count(DISTINCT p.group_key) FILTER (WHERE {pred}) >= 5
                """,
                [corpus],
            ).fetchall()
            for taxon_id, inr, tot in rows:
                if taxon_id is None:
                    continue
                region_rows.append((taxon_id, rid, int(inr), float(inr) / max(1, tot)))
        prov.close()
        db.add_taxon_regions(region_rows, inat_src)

        db.set_metadata("pack_metadata", {
            "schema_version": str(1),
            "corpus": corpus,
            "built_at": today,
        })
        report = db.verify()

    if not report["ok"]:
        raise SystemExit(f"species database failed verification: {report['problems']}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--display-name", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--pack-version", type=int, default=1)
    ap.add_argument("--policy", default="production")
    # Matches the file ml/export.py writes. int8 means *static* int8:
    # dynamic int8 is 25x slower than fp32 on this architecture (see
    # quantize_dynamic's docstring for the measurement).
    # fp16 by default: half the size of fp32 at +0.0007 top-1, whereas INT8
    # costs 8-20 points on this architecture. See ml/export.py convert_fp16.
    ap.add_argument("--quantization", default="fp16",
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    export_dir = run_dir / "export"
    model_src = export_dir / {
        "fp16": "model_fp16.onnx",
        "int8": "model_int8.onnx",
        "fp32": "model_fp32.onnx",
    }[args.quantization]
    if not model_src.exists():
        raise SystemExit(f"{model_src} not found - run ml/export.py first")

    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    corpus = args.corpus or run_manifest["config"]["corpus"]
    labels_path = PATHS.artifacts / corpus / "labels.json"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} not found - build the corpus first")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    policy = get_policy(args.policy)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    staging = Path(args.out or (PATHS.artifacts / "packs")) / args.pack_id
    staging.mkdir(parents=True, exist_ok=True)

    log(f"building pack {args.pack_id} v{args.pack_version}")
    log(f"  corpus  : {corpus} ({labels['num_classes']} classes)")
    log(f"  model   : {model_src.name} ({model_src.stat().st_size/1e6:.1f} MB)")
    log(f"  policy  : {policy.name} (commercial-safe={policy.commercial_safe})")

    # --- species database ---------------------------------------------------
    db_path = staging / "species.sqlite"
    log("  building species database")
    db_report = build_species_db(db_path, labels, corpus, today, run_dir=run_dir)
    log(f"    {db_report['counts']}")

    # --- geo prior ----------------------------------------------------------
    geo_path = staging / "geoprior.bin"
    log("  building geographic prior")
    geoprior.build(
        out_path=geo_path,
        corpus=corpus,
        provenance_db=PATHS.provenance_db,
        num_classes=labels["num_classes"],
        log=lambda m: log("    " + m),
    )

    # --- attributions -------------------------------------------------------
    log("  exporting attributions")
    with ProvenanceDB(read_only=True) as prov:
        export_attributions(prov, staging, corpus=corpus)
    attributions = staging / "ATTRIBUTIONS.csv"

    # --- labels -------------------------------------------------------------
    labels_out = staging / "labels.json"
    labels_out.write_text(json.dumps(labels, indent=1), encoding="utf-8")

    # --- model --------------------------------------------------------------
    model_dst = staging / "model.onnx"
    model_dst.write_bytes(model_src.read_bytes())

    # --- calibration --------------------------------------------------------
    calib_path = run_dir / "calibration.json"
    calibration = {
        "temperature": 1.0,
        "unknown_threshold": 0.35,
        "margin_threshold": 0.08,
        "entropy_threshold": 0.85,
        "per_class_threshold": {},
    }
    if calib_path.exists():
        calibration.update(json.loads(calib_path.read_text(encoding="utf-8")))
    else:
        log("  WARNING: no calibration.json; shipping uncalibrated defaults. "
            "Run ml/evaluate.py --fit-calibration first for a release build.")

    cfg = run_manifest.get("config", {})
    manifest = {
        "format_version": PACK_FORMAT_VERSION,
        "pack_id": args.pack_id,
        "pack_version": args.pack_version,
        "display_name": args.display_name,
        "description": args.description,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_engine_version": ENGINE_VERSION,
        "regions": regions.as_dicts(args.regions) if args.regions else regions.as_dicts(),
        "model": {
            "file": file_spec(model_dst, "model.onnx"),
            "runtime": "onnx",
            "input_size": cfg.get("image_size", 224),
            "input_mean": [0.485, 0.456, 0.406],
            "input_std": [0.229, 0.224, 0.225],
            "input_name": "input",
            "output_name": "logits",
            "embedding_name": "embedding",
            "num_classes": labels["num_classes"],
            "architecture": cfg.get("backbone", "unknown"),
            "quantization": args.quantization,
            "calibration": calibration,
        },
        "database": file_spec(db_path, "species.sqlite"),
        "labels": file_spec(labels_out, "labels.json"),
        "geo_prior": file_spec(geo_path, "geoprior.bin"),
        "attributions": file_spec(attributions, "ATTRIBUTIONS.csv"),
        "corpus": {
            "name": corpus,
            "license_policy": policy.name,
            "commercial_safe": policy.commercial_safe,
            "image_count": run_manifest.get("num_images", 0)
            or sum(c.get("images", 0) for c in labels["classes"]),
            "class_count": labels["num_classes"],
            "sources": ["inaturalist-open-data", "gbif-backbone", "wikidata"],
            "corpus_sha256": run_manifest.get("dataset_manifest_sha256"),
            "code_commit": run_manifest.get("code_commit"),
        },
    }

    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- archive ------------------------------------------------------------
    out_pack = staging.parent / f"{args.pack_id}-v{args.pack_version}.fwpack"
    log(f"  writing {out_pack.name}")
    with zipfile.ZipFile(out_pack, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(manifest_path, "manifest.json")
        for src, arc in [
            (model_dst, "model.onnx"),
            (db_path, "species.sqlite"),
            (labels_out, "labels.json"),
            (geo_path, "geoprior.bin"),
            (attributions, "ATTRIBUTIONS.csv"),
        ]:
            z.write(src, arc)

    size_mb = out_pack.stat().st_size / 1e6
    log("")
    log(f"pack: {out_pack}  ({size_mb:.1f} MB)")
    log(f"sha256: {sha256_file(out_pack)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
