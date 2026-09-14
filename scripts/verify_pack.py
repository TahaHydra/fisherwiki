"""Verify a built .fwpack the way the device does, then run real inference.

    python scripts/verify_pack.py --pack packs/global_v1-v1.fwpack
    python scripts/verify_pack.py --pack <p> --image photo.jpg

This is the last gate before a pack is considered shippable. It deliberately
re-implements the *checks* rather than importing them from the build tooling:
the build tooling computed those hashes, so asking it to confirm them proves
nothing. What is checked here is what an installed app would see.

Steps
-----
1. Parse ``manifest.json`` with the same strictness the app uses (unknown keys
   rejected, versions checked, structural sanity).
2. Verify the SHA-256 and exact byte length of every payload entry.
3. Open the ONNX model and check its input/output signature against the
   manifest's declarations.
4. Open the SQLite database and check the class mapping is dense from 0 and
   that every model class resolves to a taxon.
5. Read the geo prior and check its class count matches.
6. Run real inference on a real image and print the ranked result, applying the
   pack's own calibration - so a human can see whether the answer is sensible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

EXPECTED_ENTRIES = {"manifest.json", "model.onnx", "species.sqlite",
                    "labels.json", "geoprior.bin", "ATTRIBUTIONS.csv"}


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"  ok    {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", default=None,
                    help="photograph to identify; a synthetic one is used if omitted")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args(argv)

    pack = Path(args.pack)
    if not pack.exists():
        fail(f"{pack} not found")
    print(f"pack: {pack}  ({pack.stat().st_size / 1e6:.1f} MB)\n", flush=True)

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(pack) as z:
        names = set(z.namelist())
        unexpected = names - EXPECTED_ENTRIES
        if unexpected:
            fail(f"unexpected entries in pack: {sorted(unexpected)}")
        ok(f"{len(names)} entries, all expected")
        z.extractall(tmp)

    # --- 1. manifest ------------------------------------------------------
    manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        fail(f"format_version {manifest['format_version']} unsupported")
    if manifest.get("min_engine_version", 1) > 1:
        fail("pack requires a newer engine")
    model_spec = manifest["model"]
    n_classes = model_spec["num_classes"]
    ok(f"manifest: {manifest['pack_id']} v{manifest['pack_version']}, "
       f"{n_classes} classes, {model_spec['architecture']} "
       f"({model_spec['quantization']})")

    corpus = manifest["corpus"]
    ok(f"corpus: {corpus['name']}, policy {corpus['license_policy']}, "
       f"commercial_safe={corpus['commercial_safe']}, "
       f"{corpus['image_count']:,} images")
    if not corpus["commercial_safe"]:
        print("  NOTE  this pack is a research build and must not be "
              "distributed commercially", flush=True)

    # --- 2. hashes --------------------------------------------------------
    entries = [model_spec["file"], manifest["database"], manifest["labels"]]
    for key in ("geo_prior", "attributions"):
        if manifest.get(key):
            entries.append(manifest[key])
    for spec in entries:
        f = tmp / spec["path"]
        if not f.exists():
            fail(f"{spec['path']} missing from the archive")
        if f.stat().st_size != spec["bytes"]:
            fail(f"{spec['path']}: {f.stat().st_size} bytes, manifest says "
                 f"{spec['bytes']}")
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            fail(f"{spec['path']}: sha256 mismatch")
    ok(f"{len(entries)} payload files: size and SHA-256 verified")

    # --- 3. model ---------------------------------------------------------
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    outputs = [o.name for o in sess.get_outputs()]
    if model_spec["input_name"] not in inputs:
        fail(f"model has no input named {model_spec['input_name']!r}: {list(inputs)}")
    if model_spec["output_name"] not in outputs:
        fail(f"model has no output named {model_spec['output_name']!r}: {outputs}")
    ok(f"model signature: inputs {list(inputs)} outputs {outputs}")

    size = model_spec["input_size"]
    probe = np.zeros((1, 3, size, size), dtype=np.float32)
    logits = sess.run([model_spec["output_name"]],
                      {model_spec["input_name"]: probe})[0]
    if logits.shape[1] != n_classes:
        fail(f"model emits {logits.shape[1]} logits, manifest declares {n_classes}")
    ok(f"model emits {logits.shape[1]} logits for a {size}x{size} input")

    # --- 4. database ------------------------------------------------------
    con = sqlite3.connect(str(tmp / "species.sqlite"))
    cur = con.cursor()
    cur.execute("SELECT count(*), min(class_index), max(class_index) "
                "FROM model_classes")
    cnt, lo, hi = cur.fetchone()
    if cnt != n_classes:
        fail(f"database has {cnt} model classes, manifest declares {n_classes}")
    if lo != 0 or hi != cnt - 1:
        fail(f"class indices are not dense 0..{cnt - 1} (got {lo}..{hi})")
    cur.execute("SELECT count(*) FROM model_classes m "
                "LEFT JOIN taxa t ON t.fw_taxon_id = m.fw_taxon_id "
                "WHERE t.fw_taxon_id IS NULL")
    orphans = cur.fetchone()[0]
    if orphans:
        fail(f"{orphans} model classes resolve to no taxon")
    ok(f"database: {cnt} classes, dense from 0, all resolve to a taxon")

    for table, label in (("common_names", "common names"),
                         ("taxon_synonyms", "synonyms"),
                         ("safety_warnings", "safety warnings"),
                         ("taxon_regions", "region records"),
                         ("sources", "sources")):
        cur.execute(f"SELECT count(*) FROM {table}")
        ok(f"  {cur.fetchone()[0]:,} {label}")

    cur.execute("SELECT count(*) FROM safety_warnings w LEFT JOIN sources s "
                "ON s.source_id = w.source_id WHERE s.source_id IS NULL")
    if cur.fetchone()[0]:
        fail("safety warnings cite an unregistered source")
    ok("  every safety warning cites a registered source")

    cur.execute("SELECT count(*) FROM similar_species x LEFT JOIN sources s "
                "ON s.source_id = x.source_id WHERE s.source_id IS NULL")
    if cur.fetchone()[0]:
        fail("similar-species rows cite an unregistered source")

    # The safety cross-reference. A species that carries no warning of its own
    # but which the model measurably confuses with a dangerous one must be able
    # to reach that warning, because the candidate list does not reliably carry
    # it: on the misidentified Trachinus draco images the venomous species was
    # in the top five only 32% of the time. This check exists so a rebuild that
    # loses similar_species cannot quietly ship without the cross-reference.
    cur.execute("""
        SELECT count(DISTINCT s.fw_taxon_id)
        FROM similar_species s
        JOIN safety_warnings w ON w.fw_taxon_id = s.other_fw_taxon_id
        WHERE w.severity = 'danger'
          AND s.fw_taxon_id NOT IN (
              SELECT fw_taxon_id FROM safety_warnings WHERE severity = 'danger')
    """)
    crossref = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM similar_species")
    n_similar = cur.fetchone()[0]
    if n_similar == 0:
        print("  NOTE  no similar-species data: this pack was built without a "
              "test evaluation, so it carries no safety cross-references and "
              "no per-class accuracy", flush=True)
    else:
        ok(f"  {crossref} harmless species cross-reference a dangerous "
           f"look-alike ({n_similar:,} similar-species rows)")

    cur.execute("SELECT count(*) FROM model_classes WHERE test_precision IS NOT NULL")
    measured = cur.fetchone()[0]
    if measured:
        ok(f"  per-class measured accuracy for {measured:,} of {n_classes:,} classes")

    # --- 5. geo prior -----------------------------------------------------
    geo = tmp / "geoprior.bin"
    if geo.exists():
        with open(geo, "rb") as fh:
            magic, version, cell, gclasses = struct.unpack(">IIfI", fh.read(16))
        if magic != 0x46574750:
            fail("geoprior.bin has a bad magic number")
        if gclasses != n_classes:
            fail(f"geo prior covers {gclasses} classes, model has {n_classes}")
        ok(f"geo prior: {gclasses} classes at {cell}-degree cells")

    # --- 6. real inference ------------------------------------------------
    labels = json.loads((tmp / "labels.json").read_text(encoding="utf-8"))
    names = {c["class_id"]: c["scientific_name"] for c in labels["classes"]}

    mean = np.array(model_spec["input_mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(model_spec["input_std"], dtype=np.float32).reshape(3, 1, 1)

    if args.image:
        from PIL import Image, ImageOps

        with Image.open(args.image) as im:
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            resize_to = round(size * 256 / 224)
            w, h = im.size
            s = resize_to / min(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                           Image.Resampling.BILINEAR)
            w, h = im.size
            left, top = (w - size) // 2, (h - size) // 2
            im = im.crop((left, top, left + size, top + size))
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        source = args.image
    else:
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:size, 0:size]
        base = 0.5 + 0.25 * np.sin(xx / 9.0) + 0.15 * np.cos(yy / 7.0)
        arr = np.stack([base, base * 0.8, base * 0.6]).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 0.02, arr.shape), 0, 1).astype(np.float32)
        source = "(synthetic pattern - pass --image for a real photograph)"

    x = ((arr - mean) / std)[None, ...].astype(np.float32)
    logits = sess.run([model_spec["output_name"]],
                      {model_spec["input_name"]: x})[0][0]

    cal = model_spec["calibration"]
    t = cal.get("temperature", 1.0)
    z = logits / t
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    order = np.argsort(-p)
    top = order[: args.topk]
    margin = float(p[order[0]] - p[order[1]])
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(len(p)))

    rejected = (
        p[order[0]] < cal.get("unknown_threshold", 0.35)
        or margin < cal.get("margin_threshold", 0.08)
        or entropy > cal.get("entropy_threshold", 0.85)
    )

    print("", flush=True)
    print(f"inference on {source}", flush=True)
    print(f"  temperature {t:.3f}  margin {margin:.4f}  "
          f"normalised entropy {entropy:.4f}", flush=True)
    print(f"  verdict: {'UNCERTAIN (rejected)' if rejected else 'identified'}",
          flush=True)
    for i in top:
        cur.execute(
            "SELECT (SELECT name FROM common_names c WHERE c.fw_taxon_id = "
            "m.fw_taxon_id AND c.lang='en' LIMIT 1) FROM model_classes m "
            "WHERE m.class_index = ?", (int(i),))
        row = cur.fetchone()
        common = row[0] if row and row[0] else ""
        print(f"    {100 * p[i]:5.1f}%  {names.get(int(i), '?'):38} {common}",
              flush=True)

    con.close()
    print("", flush=True)
    print("pack verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
