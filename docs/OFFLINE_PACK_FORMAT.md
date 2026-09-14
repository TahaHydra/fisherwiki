# Offline pack format v1

A **pack** is the unit of everything FisherWiki needs to identify fish in one
region with no network: a model, its class mapping, a species database, a
geographic prior, and the attribution data the media licences require.

Packs are **untrusted input**. They may arrive from a CDN, from a sideloaded
file, or from a file someone was sent. Every guard in
[`SafeZip`](../app/core/src/main/kotlin/com/fisherwiki/core/pack/SafeZip.kt) and
[`PackVerifier`](../app/core/src/main/kotlin/com/fisherwiki/core/pack/PackVerifier.kt)
exists because of that. Every pack is sideloaded today — the app has no network
access — and the verification path was written for a downloaded one, so it
assumes nothing about where the bytes came from.

---

## 1. Container

A ZIP archive, extension `.fwpack`:

```
manifest.json      identity, hashes, model spec, calibration, provenance
model.onnx         the classifier
labels.json        class index -> fw_taxon_id
species.sqlite     offline species database
geoprior.bin       per-class occurrence histogram   (optional)
ATTRIBUTIONS.csv   per-image attribution for the training corpus (optional)
```

No other entries are permitted: `manifest.json` declares the payload, and
extraction runs with that list as an allow-list, so a pack cannot smuggle an
unlisted file past verification by simply not mentioning it.

---

## 2. `manifest.json`

```jsonc
{
  "format_version": 1,
  "pack_id": "global_v1",              // ^[a-z0-9_]{1,64}$
  "pack_version": 3,                   // monotonic within pack_id
  "display_name": "Global Angler",
  "description": "...",
  "built_at": "2026-09-13T18:40:00+00:00",
  "min_engine_version": 1,

  "regions": [
    { "id": "europe_freshwater", "name": "Europe - freshwater",
      "water": "freshwater", "boxes": [[35.0, 71.5, -11.0, 40.0]] }
  ],

  "model": {
    "file": { "path": "model.onnx", "sha256": "…64 hex…", "bytes": 9437184 },
    "runtime": "onnx",
    "input_size": 224,
    "input_mean": [0.485, 0.456, 0.406],
    "input_std":  [0.229, 0.224, 0.225],
    "input_name": "input",
    "output_name": "logits",
    "embedding_name": "embedding",
    "num_classes": 1963,
    "architecture": "mobilenet_v3_large",
    "quantization": "int8",
    "calibration": {
      "temperature": 1.84,
      "unknown_threshold": 0.40,
      "margin_threshold": 0.08,
      "entropy_threshold": 0.85,
      "expected_calibration_error": 0.031,
      "per_class_threshold": { "417": 0.72 }
    }
  },

  "database":     { "path": "species.sqlite",  "sha256": "…", "bytes": 12582912 },
  "labels":       { "path": "labels.json",     "sha256": "…", "bytes": 184320 },
  "geo_prior":    { "path": "geoprior.bin",    "sha256": "…", "bytes": 524288 },
  "attributions": { "path": "ATTRIBUTIONS.csv","sha256": "…", "bytes": 41943040 },

  "corpus": {
    "name": "global_v1",
    "license_policy": "production",
    "commercial_safe": true,
    "image_count": 302201,
    "class_count": 1963,
    "sources": ["inaturalist-open-data", "gbif-backbone", "wikidata"],
    "corpus_sha256": "…",
    "code_commit": "…"
  }
}
```

### Fields that are load-bearing, not decorative

| Field | Why it matters |
|---|---|
| `format_version` | Checked **first**, before anything else is parsed. A future format fails cleanly instead of being half-read. |
| `min_engine_version` | A pack that relies on semantics this build does not implement is refused rather than interpreted approximately. |
| `input_mean` / `input_std` | Carried in the pack, not hard-coded. A model trained with different statistics cannot be mispaired with the wrong constants. Verified non-zero (a zero would divide by zero at preprocess). |
| `output_name` | The graph emits **logits**, never probabilities, so a model can be recalibrated by editing the manifest without re-exporting. |
| `calibration` | See §5. This is what turns a raw softmax into a number worth showing a user. |
| `corpus.commercial_safe` | `false` for a research build. Surfaced in the UI so a model that must not be distributed cannot be mistaken for one that can. |
| `corpus.corpus_sha256` + `code_commit` | Ties a shipped model to the exact training data and code that produced it. |

Manifest parsing is **strict**: unknown JSON keys are an error. A pack carrying
fields we do not understand may depend on semantics we will not apply, and
silently dropping them is how an "identification" comes to mean something other
than intended.

---

## 3. Verification order

Each step is cheap relative to the next, so a hostile or corrupt file is
rejected as early as possible.

1. Read `manifest.json` only — bounded to 4 MB, in memory.
2. Check `format_version` and `min_engine_version`.
3. Structural checks: `pack_id` pattern, plausible `input_size` (64–1024) and
   `num_classes` (1–100,000), 3-channel non-zero `input_std`, known
   `quantization`, calibration thresholds in range, every `sha256` matching
   `^[0-9a-f]{64}$`, every declared size within limits, no duplicate paths,
   `corpus.class_count <= model.num_classes`.
4. Extract, with the manifest's entry names as an allow-list.
5. Verify every extracted file's **exact byte length and SHA-256**.

Only after step 5 does a caller receive an `InstalledPack`.

Installation is then **atomic**: extraction goes to a staging directory and only
a fully verified result is moved into place, so an interrupted install cannot
leave a half-extracted pack that the engine would later load and trust.

### Archive hardening

| Attack | Guard |
|---|---|
| Zip Slip (`../../databases/app.db`) | entry names rejected if absolute, drive-qualified, containing `\`, a `..` or `.` segment, an empty segment, or NUL; **and** the resolved canonical path must be inside the destination |
| Zip bomb | per-entry cap (512 MB) and total cap (2 GB) enforced **while streaming**, not from the header, plus a declared-compression-ratio pre-check (200:1) |
| Entry-count exhaustion | 256 entries maximum |
| Symlink entries | `java.util.zip` exposes no unix mode and this extractor calls no link-creating API, so a symlink entry is written as an ordinary file containing the link target string — inert |
| Unlisted payload | entry allow-list from the manifest |
| Oversized sideload | 1.5 GB cap while copying a `content://` URI into the cache |

**Nothing in a pack is ever executed, loaded as a library, or used as a class
path.** Packs carry data only.

---

## 4. `labels.json`

```jsonc
{
  "corpus": "global_v1",
  "num_classes": 1963,
  "classes": [
    { "class_id": 0, "fw_taxon_id": 1047,
      "scientific_name": "Abramis brama",
      "images": 300, "observations": 142, "observers": 118 }
  ]
}
```

`class_id` must be **dense from 0**. The species database's `verify()` enforces
this, because a gap would silently shift the meaning of every index after it —
the worst possible failure in a system whose job is to name things correctly.

Class identity is `fw_taxon_id`, an internal id minted by
[`taxonomy/registry.py`](../tools/fwdata/taxonomy/registry.py) and committed to
the repository. Source ids (GBIF, iNaturalist) are *not* used as class
identities: they get merged, split and re-keyed upstream, and a re-keyed id
would repoint a model output at a different animal.

---

## 5. Calibration block

Raw softmax over a closed class set is systematically overconfident and has no
way to express "a species I was never trained on", let alone "a boot". The
calibration block addresses both.

| Field | Meaning |
|---|---|
| `temperature` | Divides logits before softmax. Fitted on the **validation** split by minimising NLL. Cannot change which class wins, so it never costs accuracy — it only makes the number mean something. |
| `unknown_threshold` | Minimum calibrated top-1 probability. Chosen from the measured coverage/accuracy curve, not picked by hand. |
| `margin_threshold` | Minimum gap between top-1 and top-2. Catches the confusable-species case, where the model is certain it is one of two things and has no idea which — a case that top-1 alone looks fine for. |
| `entropy_threshold` | Maximum normalised entropy. Catches "mildly attracted to forty classes at once", the signature of an out-of-distribution photograph. Normalising by `log(numClasses)` keeps one threshold meaningful across packs of different sizes. |
| `per_class_threshold` | Raises the bar for specific weak classes without penalising the rest. |

A result failing **any one** of these is reported as uncertain even if the other
two look fine. That is deliberate.

---

## 6. `geoprior.bin`

**Big-endian**, matching Java `DataInputStream`'s network byte order. The Python
writer uses `struct` `'>'` for the same reason; a mismatch would parse without
error and yield nonsense, so a round-trip test pins the two together.

```
u32   magic        0x46574750  ("FWGP")
u32   version      1
f32   cellDegrees  typically 2.0
u32   numClasses
repeat numClasses:
    u32  classTotal          total observations for this class
    u32  cellCount
    repeat cellCount:
        i32  packedCell      (latIdx << 16) | (lonIdx & 0xFFFF)
        u8   value           quantised log-frequency, 1..255
```

Values are **per-class scaled `log1p(count)`**, not linear counts. A species'
modal cell can hold thousands of records while its range edge holds one, and on
a linear scale the range edge would be indistinguishable from absent.

A cell absent from the file means **unknown**, not **absent**, and the reader
maps it to `UNKNOWN_FACTOR = 0.75` rather than to zero. See
[`GeoPrior`](../app/core/src/main/kotlin/com/fisherwiki/core/rank/GeoPrior.kt)
for why the prior is clamped to `[0.12, 3.0]` and attenuated by visual
confidence: an angler who has actually caught a vagrant, an escapee or a stocked
fish is the user who most needs the app not to argue them out of it.

---

## 7. `species.sqlite`

Schema: [`tools/fwdata/species_schema.sql`](../tools/fwdata/species_schema.sql).
That file is the single source of truth — Gradle copies it into the Kotlin test
resources so fixtures are built from exactly the DDL that ships.

Every fact table carries a `NOT NULL source_id` referencing `sources`. There is
no code path that inserts a biological claim without a citable source. Fields we
cannot source from CC0/CC-BY data are **absent**, and the app renders them as
absent. See [`SPECIES_DATA.md`](SPECIES_DATA.md).

The database is opened **read-only** on device. A pack's database is an
immutable hash-verified artefact; the user's catch log is a separate writable
database, so a pack upgrade can never touch user data.

---

## 8. Building and installing a pack

```bash
.venv/Scripts/python ml/export.py --run <run_dir> --quantize int8_static
.venv/Scripts/python tools/build_pack.py \
    --run <run_dir> --pack-id global_v1 --display-name "Global Angler"
```

Sideload for development — no backend or app store involved:

```bash
adb push packs/global_v1-v1.fwpack /sdcard/Download/
```

then **Packs → Install from file** in the app.

---

## 9. Compatibility rules

* A pack whose `format_version` differs from the build's is refused.
* A pack whose `min_engine_version` exceeds the build's is refused, with a
  message naming both numbers.
* Multiple versions of the same `pack_id` may be installed; the engine uses the
  highest.
* Different `pack_id`s may be installed side by side. Selecting between them
  for inference is a v2 feature; v1 uses the highest-versioned pack.
* Removing a pack never touches the catch log.
