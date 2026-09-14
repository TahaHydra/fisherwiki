# Architecture

```
                    OFFLINE (the phone)                 ONLINE (build machine)
   ┌───────────────────────────────────┐        ┌──────────────────────────────┐
   │  :android   camera, UI, catch log │        │  tools/   acquisition        │
   │      │                            │        │     │     provenance         │
   │      ▼                            │        │     ▼     licensing          │
   │  :core      IdentificationEngine  │  pack  │  ml/      training           │
   │             ├ Preprocessor        │◀───────│     │     evaluation         │
   │             ├ OnnxClassifier      │ .fwpack│     ▼     export             │
   │             ├ Calibration         │        │  build_pack.py               │
   │             ├ GeoPrior            │        └──────────────────────────────┘
   │             ├ CandidateRanker     │
   │             └ SpeciesRepository   │
   └───────────────────────────────────┘
```

The single most important structural decision: **`:core` is a plain Kotlin/JVM
module with no Android types.** Everything that decides what the app tells a
user — preprocessing, inference, calibration, open-set rejection, geographic
ranking, multi-photo fusion, species lookup — lives there and is unit-tested on
a workstation against the *same* ONNX Runtime Java API and the *same* `.onnx`
bytes that ship. `:android` is camera, pixels, screens and storage.

That is why a preprocessing or output-parsing mistake surfaces in CI rather than
on a riverbank with no signal.

---

## 1. Repository layout

```
app/                  Gradle project
  core/               identification engine, pure JVM, no Android
  android/            Android application
ml/                   training, evaluation, export
  fwml/               library code
  configs/            training configurations
tools/                data acquisition, provenance, taxonomy, packing
  fwdata/             library code
  dataset.py          data CLI
  build_pack.py       pack assembly
data/taxonomy/        committed: canonical taxon ID registry
docs/                 this
tests/                Python tests
scripts/              long-running one-shot jobs
packs/                built packs (payloads gitignored)
```

Bulk data never enters the repository. Images, dumps, checkpoints and packs live
under `$FISHERWIKI_DATA` (default `D:/fisherwiki-data`). What *is* committed is
everything needed to reproduce them: manifests, hashes, the taxon registry and
the code.

---

## 2. Data pipeline

```
iNaturalist Open Data (S3)     GBIF Backbone (CC BY 4.0)     Wikidata (CC0)
  observations.csv.gz 13.1 GB    simple.txt.gz 489 MB          SPARQL
  photos.csv.gz       20.1 GB
  taxa.csv.gz / observers.csv.gz
            │                          │                          │
            └──────────┬───────────────┘                          │
                       ▼                                          │
          taxonomy reconciliation  ──────────────────────────────┐ │
          43,559 canonical fw_taxon_id                           │ │
          66,754 synonyms                                        ▼ ▼
                       │                               species.sqlite
                       ▼
          candidate extraction (DuckDB)
          4,168,455 licensed fish photos
                       │
                       ▼  licence policy: CC0 / PD / CC BY
          505,347 production-safe images
                       │
                       ▼  evidence bar: >=40 images AND >=25 observations
          1,963 species-level classes
                       │
                       ▼  cap 300/species, diversity-ordered
          308,227 selected  ──▶ content-addressed store + provenance
                       │
                       ▼  group-aware split (photographer, then observation)
          train / val / test / geo_test
                       │
                       ▼
          training ──▶ calibration ──▶ ONNX export ──▶ quantise ──▶ pack
```

Design rules the pipeline enforces rather than documents:

* `ImageStore.put()` is the only way to write image bytes, and it requires a
  populated provenance record whose licence the active policy admits. There is
  no "just download it" path.
* The candidate table is written **before** any bytes move, so "what we
  considered" is reproducible independently of what we fetched.
* Storage is content-addressed (`cas/<aa>/<bb>/<sha256>.<ext>`), so changing the
  licence policy re-selects rows without re-downloading pixels.
* Every corpus build re-checks for cross-split leakage and **refuses to export a
  manifest** if it finds any.

---

## 3. The identification path, in order

1. **Decode** (`:android`). EXIF orientation applied, downsampled to ~640 px
   short edge. Both details matter: training decoded with orientation applied,
   and a 12 MP bitmap is 48 MB of heap for no benefit.
2. **Preprocess** (`:core/Preprocessor`). Resize short edge to 256/224 × input
   size, centre crop, RGB, normalise with the mean/std **from the pack
   manifest**, emit NCHW float32.
3. **Infer** (`:core/OnnxClassifier`). One session call, or a batch for Expert
   ID. Returns logits and optionally a normalised embedding.
4. **Calibrate** (`:core/Calibration`). Temperature-scale, compute top-k, margin
   and normalised entropy.
5. **Apply the geographic prior** (`:core/GeoPrior`), bounded and attenuated,
   then renormalise.
6. **Decide** (`:core/CandidateRanker`). Three independent rejection signals;
   fall back to genus by aggregation when no species claim is supportable.
7. **Look up** (`:core/SpeciesRepository`). Names, diagnostic features, similar
   species, safety warnings, sources.
8. **Render** (`:android/ResultScreen`), with the layout driven by
   `Certainty`, not by a percentage.

Steps 2–7 contain no I/O beyond reading the verified pack files, and no network
code exists anywhere in `:core`.

**Steps 5 and 6 compose without independent validation, and that is a real
gap, not a hypothetical one.** The thresholds `CandidateRanker` applies in
step 6 (`unknown_threshold`, `margin_threshold`, `entropy_threshold`, and the
temperature that produced them) are fitted in `ml/evaluate.py` against
*purely visual* calibration on the validation split - step 5 does not exist
at fitting time. Step 5 then multiplies those calibrated probabilities by a
geographic factor and renormalises before step 6 ever sees them, so the
0.0071 test-split ECE reported in `docs/MODEL.md` describes the visual-only
distribution, not the geo-adjusted one a user with location enabled actually
receives. `GeoPrior`'s factor is deliberately bounded so it cannot zero a
species out or manufacture false confidence from thin data (see its own
kdoc), which limits how badly this can go wrong, but "limited" is not the
same claim as "measured". No evaluation in this project currently compares
visual-only against visual+geo on a held-out geographic split. It is not
currently the most urgent gap to close - the Android location flow this would
calibrate is itself not wired in yet, tracked separately in `task.md` - but
closing one without the other would ship a geo-ranked confidence number this
project has never actually checked the calibration of.

---

## 4. Key decisions and why

### One global model, not per-region models

Measured, not assumed. Under the production licence policy, 1,963 species clear
the evidence bar globally. The per-region class counts sum to 3,887 because
regions overlap heavily:

| region | species classes |
|---|---|
| Indo-Pacific | 841 |
| Australia / NZ | 680 |
| North America Atlantic | 573 |
| North America freshwater | 554 |
| Africa | 277 |
| South America | 223 |
| Europe freshwater | 211 |
| Europe Atlantic | 190 |
| North America Pacific | 187 |
| Mediterranean | 151 |

So a set of regional models would collectively be **larger** than one global
model, would each see less data, and would fragment the training signal. Instead
one model covers all 1,963 classes, and regional packs differ in their species
database and geographic prior. Regional restriction happens at inference via the
geo prior, which also handles travel and introduced populations — something a
hard regional model cannot.

### ONNX Runtime, not TFLite

`torch.onnx.export` is native and reliable on Windows; the TFLite conversion
path is not. The decisive advantage is testability: the ONNX Runtime **Java API
is identical** on desktop and Android, so `:core` runs the real model on a
workstation. TFLite remains a viable future export target; the pack manifest has
a `runtime` field so adding one is a format-compatible change.

### Kotlin/JVM `:core` rather than Kotlin Multiplatform

KMP would give a shared iOS target today, but the ONNX Runtime Java API is not
multiplatform, so the inference layer would need an `expect`/`actual` split
anyway and the desktop-test advantage would be lost. Instead `:core` is written
with every platform touchpoint behind an interface (`SqliteDriver`, pixel
arrays, `File`), which keeps the iOS port a real piece of work but a
*bounded* one. See §6.

### Class identity is an internal ID

`fw_taxon_id` comes from an append-only registry committed to the repository.
GBIF and iNaturalist ids get merged, split and re-keyed upstream; a re-keyed id
would silently repoint a model output at a different animal.

---

## 5. Module dependency rules

```
:android ──▶ :core ──▶ (kotlinx-serialization, kotlinx-coroutines)
                   ──▶ ai.onnxruntime  [compileOnly]
```

* `:core` **must not** depend on anything `android.*`. This is what keeps the
  desktop tests meaningful; if it ever breaks, the tests stop testing the
  shipping code.
* `ai.onnxruntime` is `compileOnly` in `:core`. The Android module supplies
  `onnxruntime-android`, the tests supply the desktop build, so exactly one
  native runtime is present in any configuration.
* `:core` **must not** contain an HTTP client. The privacy claim is structural.

---

## 6. iOS portability — the honest position

The app is Android-only today and has not been built for iOS. What has been
done is to keep the port bounded rather than to claim it is free:

| Layer | iOS status |
|---|---|
| Identification logic, ranking, calibration, geo prior, pack verification | Kotlin with no JVM-specific APIs beyond `java.io.File` and `java.util.zip`; portable to Kotlin/Native with a KMP conversion |
| ONNX Runtime | ships an official iOS pod; the C/ObjC API differs from the Java API, so `OnnxClassifier` needs an `actual` implementation |
| SQLite | already behind `SqliteDriver`; an iOS implementation is ~80 lines |
| Image decode + EXIF | already platform-side; needs a Core Graphics equivalent of `ImageLoading` |
| UI | SwiftUI, or Compose Multiplatform |

The realistic estimate is that `:core` converts to KMP with the inference and
SQLite layers as `expect`/`actual`, and the UI is rewritten. Nothing in the pack
format, the model or the data pipeline is Android-specific.

---

## 7. Threading

* Inference runs on `Dispatchers.Default`, never the main thread. A forward pass
  is tens of milliseconds on a good phone and several hundred on a cheap one.
* ONNX Runtime is configured with 4 intra-op threads by default. Using every
  core on a big.LITTLE SoC makes the device throttle, which is *slower*.
* `IdentificationEngine` is opened once and cached, because opening it loads an
  ONNX session and ~2,000 taxon rows.
* The pack database is read-only, so concurrent reads need no coordination.

---

## 8. Failure behaviour

| Failure | Behaviour |
|---|---|
| No pack installed | UI offers to install one; never a crash or an empty result |
| Pack fails verification | refused with the specific reason; a partial install is impossible (atomic staging) |
| Pack listed but files missing | shown as "unreadable pack" with the reason, not silently hidden |
| Geo prior corrupt | degrades to neutral (purely visual ranking), never to wrong geography |
| Model/database class-count mismatch | engine refuses to open — a mismatch would name the wrong species |
| Image undecodable | reported; in training, the row is dropped rather than fed as a grey tensor with a real label |
| Model uncertain | reported as uncertain, which is a normal outcome and not an error |
