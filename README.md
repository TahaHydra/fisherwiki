# FisherWiki

Offline-first fish identification for recreational anglers.

Photograph a fish, get a species — with no signal, no account, and no
photograph ever leaving the phone.

> **Status: pre-release.** The data pipeline, the identification engine, the
> Android app and the first trained model are built and tested; a verified
> 32.6 MB pack exists. Measured top-1 on the held-out test split is **0.5336**
> over 1,978 species — but **0.3113** on photographs from photographers the
> model has never seen, which is the figure to judge it by. See
> [Current status](#current-status) and [`docs/MODEL.md`](docs/MODEL.md).
> Nothing below is claimed that has not been run.

---

## Why this exists

Most fish-ID apps send your photograph to a server, need an account, and answer
with a species name and a percentage whether or not they actually know. Three
problems with that, and this project is organised around them:

1. **Fishing happens where there is no signal.** Cloud inference is the wrong
   architecture for the use case.
2. **Your catch locations are yours.** They are not telemetry.
3. **A confident wrong answer is worse than an admitted unknown.** If the model
   does not know, it must say so — especially when the fish has venomous spines.

So: the model runs on the phone, the data stays on the phone, and
*"Identification uncertain"* is a first-class result rather than a failure.

---

## How it works

```
photo → decode (EXIF-correct) → preprocess → on-device ONNX model
      → calibrate → geographic prior → three-signal rejection
      → species, genus, or "uncertain"
```

Everything after the shutter happens locally, from a **pack**: a verified bundle
containing the model, a species database, a geographic occurrence prior and the
photo attributions. You install a pack from a file once, and it works forever
offline — the app itself has no network access at all (see
[Current status](#current-status)).

### The uncertainty logic, specifically

A softmax over a closed class set is overconfident and cannot express "a species
I was never trained on", let alone "that's a boot". So a result is rejected if
**any** of three independent signals fails:

| signal | catches |
|---|---|
| top-1 confidence | generally weak evidence |
| top-1 minus top-2 margin | two similar species — the model is sure it is one of *these two* and has no idea which |
| normalised entropy | mass spread over dozens of classes — the signature of an out-of-distribution photo |

When no species claim holds, the engine aggregates by genus. If 0.30/0.28/0.15
is spread over three *Sebastes*, no species clears the bar but *"this is a
Sebastes"* is supported at 0.73 and is genuinely useful. If the same mass is
spread over three families, nothing is claimed.

The result type makes this structural: when the engine is uncertain there is no
`best` candidate in the model at all, so the UI *cannot* render a species
headline for an answer the engine does not stand behind.

---

## Current status

Honesty about verification state, because "it builds" and "it works" are
different claims.

| Component | Status | How verified |
|---|---|---|
| Data acquisition + provenance | **done** | 307,415 images fetched, every one with a licence record |
| Licence policy enforcement | **done** | 235 Python tests; store refuses inadmissible media |
| Taxonomy reconciliation | **done** | 43,559 canonical IDs, 66,754 synonyms, verified against GBIF |
| Corpus + leakage-safe splits | **done** | 0 group leaks, 0 hash leaks, enforced by a refusal |
| Species database | **done** | Kotlin tests against the real shipping schema |
| Identification engine (`:core`) | **done** | 136 Kotlin tests, incl. cross-language parity with the Python pipeline |
| Pack format + verification | **done** | adversarial archive tests (Zip Slip, bombs, allow-list, hash mismatch) |
| Android app | **builds** | 54.8 MB APK (arm64), compiles and packages ONNX Runtime correctly |
| Trained model | **done** | 1,978 species, 0.5336 top-1 on a test split read once; calibrated to 0.0071 ECE |
| Open-set rejection | **done** | 93.3% of unseen species and 97.8% of non-fish rejected |
| Release pipeline + pack | **done** | `finalize_model.py` end-to-end; all 16 `verify_pack.py` checks pass on the built pack |
| **On-device testing** | **not done** | no physical device or emulator available here — see below |

### What has *not* been verified

* **No instrumented Android testing.** There is no device and no AVD on the
  build machine. The app compiles and packages; it has not been run on a phone.
  This is why the identification engine is a pure-JVM module — every piece of
  logic that decides what the user is told is unit-tested on the desktop against
  the same ONNX Runtime API and the same model bytes. But UI, camera and
  permissions are unexercised.
* **The preprocessing parity number does not cover the on-device decoder.**
  The measured 0.017 max-diff figure compares `Preprocessor.toTensor` against
  the Python training pipeline starting from **identical, already-decoded
  pixels** — it says nothing about `ImageLoading.kt`, the code that actually
  turns a phone photo into those pixels: `BitmapFactory`'s JPEG decode,
  power-of-2 `inSampleSize` downsampling before `Preprocessor` ever runs (a
  step training's PIL-based pipeline does not have at all), and EXIF rotation
  applied via an Android `Matrix` rather than PIL's `exif_transpose`. None of
  this is exercised by any test, on the desktop or otherwise — closing it
  needs either a device/AVD or a Robolectric shadow of `BitmapFactory`
  precise enough to trust, neither of which exists here yet.
* **Photographer generalisation is weak, and thinly measured.** Top-1 drops from
  0.5356 to **0.3113** on the 257 test images from photographers absent from
  training. The split has no same-observation leakage, but photographers span
  train and test and the model learned some of their habits along with the fish.
  257 images establishes that the gap is real and large; it is not enough to
  size it precisely. [`docs/MODEL.md §5`](docs/MODEL.md) has the full analysis,
  including why it is not a class-difficulty artefact.
* **No on-device performance numbers.** All latency figures are desktop CPU and
  are useful only for comparing export variants against each other. A phone will
  differ, likely by 3–10×.
* **No detector.** Classification runs on a centre crop of the whole frame; the
  crop-first path is written but unbenchmarked.
* **Packs are not signed.** Integrity is verified; authenticity is not. See
  [`SECURITY.md §6`](docs/SECURITY.md).
* **No in-app pack download.** The intended product downloads regional packs
  over the network; that is not built, and there is no server to download from.
  Packs are installed from a file, via the system file picker or `adb push`.
  Rather than leave an unused `INTERNET` permission behind a promise, the app
  requests no network permission at all — which makes "no photograph leaves
  the phone" checkable from the manifest rather than a claim you have to trust.
  Re-adding it is a prerequisite for building the downloader.

---

## Build and run

### Requirements

* JDK 17+ (Android Studio's bundled JBR 21 works)
* Android SDK with platform 36
* Python 3.13
* ~60 GB free disk for the full corpus (see [`DATASETS.md`](docs/DATASETS.md))

### The app

```bash
cd app
echo "sdk.dir=C:/Users/you/AppData/Local/Android/Sdk" > local.properties
./gradlew :core:test          # 136 tests, no Android SDK needed
./gradlew :android:assembleDebug
```

APKs land in `app/android/build/outputs/apk/debug/`. Install
`android-arm64-v8a-debug.apk` on a modern phone, or the universal APK if unsure.

With no pack installed the app tells you so and offers to install one. To
sideload a pack you built:

```bash
adb push packs/global_v1-v1.fwpack /sdcard/Download/
```

then **Packs → Install from file**.

### The data and model pipeline

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -q          # 235 tests

.venv/Scripts/python tools/dataset.py discover    # what is available, and its licence
```

Full rebuild — roughly 2.5 hours, dominated by the image download:

```bash
.venv/Scripts/python tools/dataset.py download --source inaturalist   # 33 GB
.venv/Scripts/python tools/dataset.py download --source gbif          # 489 MB
.venv/Scripts/python tools/taxonomy.py reconcile
.venv/Scripts/python tools/dataset.py extract
.venv/Scripts/python scripts/fetch_wikidata.py
.venv/Scripts/python tools/dataset.py fetch --policy production \
    --per-species-cap 300 --min-observations 25 --workers 64
.venv/Scripts/python tools/dataset.py build-corpus --corpus global_v1
.venv/Scripts/python tools/dataset.py verify
```

Every step is resumable. Training is documented in
[`TRAINING.md`](docs/TRAINING.md) — **read the storage section first**, it is
the difference between a 3-hour run and a 32-hour one.

---

## Privacy

| | |
|---|---|
| Account required | none |
| Photographs uploaded | never |
| Cloud inference | none |
| Analytics / crash reporting / ads | none — and none in the dependency graph |
| Location | optional, off by default, used on-device only |
| Cloud backup | disabled for every domain |

The claims are checkable rather than promissory: put the phone in airplane mode
and identify a fish, or run `./gradlew :android:dependencies` and look for an
analytics library. There isn't one. Full detail and a verification recipe in
[`PRIVACY.md`](docs/PRIVACY.md).

---

## Data and licensing

The corpus is built exclusively from media whose licence permits commercial
redistribution: **CC0, Public Domain and CC BY**. That constraint is enforced in
code — `ImageStore.put()` is the only way to write an image and refuses anything
the active policy does not admit.

It is also expensive. Of 4,168,455 licensed fish photographs available on
iNaturalist, **88% are CC BY-NC** and cannot be used. The shippable corpus is
505,347 images, which is why the model covers 1,963 species rather than the
15,466 that have at least one photo.

**FishNet is not used.** It has no licence: the project page licenses only the
website, the repository has no LICENSE file (`"license": null` via the GitHub
API), and the images come from FishBase where the per-image default is All
Rights Reserved. It is excluded from every corpus including the research one —
`research_nc` relaxes *non-commercial*, not *unknown*.

Full analysis: [`DATASET_LICENSES.md`](docs/DATASET_LICENSES.md).

---

## Repository layout

```
app/core/      identification engine — pure Kotlin/JVM, no Android types
app/android/   the application
ml/            training, evaluation, ONNX export
tools/         data acquisition, provenance, taxonomy, pack building
docs/          architecture, datasets, licensing, privacy, security
tests/         Python tests
```

`app/core` having no Android dependency is the load-bearing decision: it is what
lets the shipping identification logic be tested on a workstation against real
inference. See [`ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | system design, module rules, iOS portability |
| [DATASETS.md](docs/DATASETS.md) | sources, funnel numbers, quality handling |
| [DATASET_LICENSES.md](docs/DATASET_LICENSES.md) | what was verified, when, and what was decided |
| [MODEL.md](docs/MODEL.md) | measured accuracy, calibration, and where it fails |
| [SPECIES_DATA.md](docs/SPECIES_DATA.md) | what the offline database holds, and what it is missing |
| [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) | per-image provenance and how to audit it |
| [TRAINING.md](docs/TRAINING.md) | reproducing the model, and the storage gotcha |
| [OFFLINE_PACK_FORMAT.md](docs/OFFLINE_PACK_FORMAT.md) | pack specification |
| [PRIVACY.md](docs/PRIVACY.md) | what the app does, and how to check |
| [SECURITY.md](docs/SECURITY.md) | threat model, untrusted-pack handling, known gaps |
| [engineering-log.md](docs/engineering-log.md) | what broke and why, including my own mistakes |
| [task.md](task.md) | current state, next steps, known problems |

The engineering log is worth reading if you are evaluating this code: it records
the bugs found and, in several cases, the reasoning that was wrong first.

---

## Licence

Code: Apache-2.0 (see [LICENSE](LICENSE)).

Model packs and training data carry their own terms — photograph attributions
ship inside each pack as `ATTRIBUTIONS.csv`, and taxonomy from the GBIF Backbone
is CC BY 4.0.
