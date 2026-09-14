# FisherWiki - project status

Offline-first, privacy-first fish identification for recreational anglers.
On-device inference only. No account, no cloud, no telemetry.

Last updated: 2026-09-14

---

## Current phase

**Phase 4 - first full training run.**

Data pipeline, identification engine, pack format and Android app are built and
tested. The image cache is being prepared on SSD; training restarts once it
completes.

---

## Completed

### Data (real, not scaffolding)
- 33 GB iNaturalist bulk metadata + 489 MB GBIF backbone downloaded and pinned
- 43,559 canonical taxon IDs, 66,754 synonyms, 99.7% GBIF match rate
- 4,168,455 licensed fish photographs identified; **505,347** admissible under
  the production licence policy (CC0/PD/CC BY) — 12.1% of what exists
- **307,415 images fetched** (37.9 GB), every one with a full provenance row
- Wikidata: 1,906/1,963 species matched, 1,289 English + 599 French names,
  1,648 IUCN statuses, 127 languages
- Corpus: **1,978 classes**, 243,469 train / 30,589 val / 28,943 test,
  **zero leakage** (groups and hashes), 0 classes missing val or test
- Curated safety warnings: 346 rows over 326 species, every one sourced

### Engine (`:core`, pure JVM — 136 tests)
- Pack format with hardened verification: Zip Slip, zip bombs, entry allow-list,
  per-file SHA-256, atomic install
- Preprocessing with antialiased downscaling, **cross-language parity with the
  Python training pipeline verified to max diff 0.017**
- Calibration: temperature scaling, three-signal open-set rejection, ECE
- Geographic prior: bounded, attenuated, cross-language format round-trip tested
- Ranking with genus fallback; multi-photo fusion (sum rule)
- Species repository tested against the real shipping schema

### Python tooling (211 tests)
- Licence policy engine that fails closed
- Resumable downloader: range requests, token buckets, checksums
- Provenance store where pixels cannot be written without an admissible licence
- Corpus builder that **refuses to export a leaking split**
- Training, evaluation, ONNX export, pack assembly

### Android
- Builds a real APK: 54.8 MB arm64-v8a, ONNX Runtime natives verified
- Compose UI driven by certainty rather than by a percentage
- Catch log, pack manager, settings; backup disabled for every domain

---

## In progress

- **SSD image cache** (`tools/prepare_cache.py`) — ~30% done, ~25 min remaining
- **First full training run** — restarts when the cache completes

---

## Next

Everything on the original V1 list is done: trained, calibrated, tested once,
exported, packed, verified, open-set evaluated, documented. What follows is
ordered by expected value, and the first item is worth more than the rest
combined.

1. **More photographers per species, not more images per species.** The
   unseen-photographer result (0.3113 against 0.5356) says the corpus is
   over-weighted toward prolific contributors. Cap images per observer per
   class in the corpus build and re-measure. This is the highest-value change
   available.
2. **Observer-disjoint splitting where the data allows it.** A hybrid: observer-
   disjoint for classes with enough distinct observers, observation-grouped
   otherwise, reporting the two populations separately rather than pooling them
   into one flattering average. A naive observer-disjoint split was tried and
   abandoned because it left 284 classes with no validation images.
3. **Architecture comparison** (MobileNetV3-Large vs EfficientNet-B0) at equal
   epochs on identical data. Config exists at `ml/configs/efficientnet_b0.yaml`;
   only throughput has been compared, not accuracy.
4. **Detector crop benchmark.** `Preprocessor.expandBox` exists and is unused;
   classification runs on a centre crop of the whole frame.
5. **A licence-compatible source for `habitats` and `diagnostic_features`.**
   Both tables are empty. WoRMS could establish marine/freshwater for taxa with
   an AphiaID — see [`docs/SPECIES_DATA.md`](docs/SPECIES_DATA.md) §4.
6. **Pack signing.** Integrity is verified; authenticity is not.
7. **In-app pack download.** Not built, and the app now requests no network
   permission at all. Re-adding `INTERNET` is a prerequisite and should be a
   visible, reviewed change.
8. **Growth-stable splitting.** The OBSERVATION strategy's rank-within-class
   cut can move an existing observation to a different split when the corpus
   grows (proven by `test_growth_can_move_an_existing_observation`; see
   [`docs/DATA_PROVENANCE.md` §8](docs/DATA_PROVENANCE.md)). Either version
   corpus builds so an old test split can be pinned and re-evaluated exactly,
   or find a growth-stable assignment rule for small classes that does not
   reintroduce the 284-classes-with-no-validation-images problem the current
   rule exists to fix.
9. **Export and use the genus/family heads, or retire them.** They are trained
   (loss weights 0.2/0.1) but `ExportWrapper` never exports their logits, so
   the on-device genus fallback aggregates the species head's own probability
   mass by genus instead — a reasonable design, but not what an earlier
   version of `docs/MODEL.md` implied. Either export both heads and let the
   ranker consult them directly (measuring whether that is actually more
   accurate than aggregation first), or drop the framing that they back the
   fallback and keep them purely as an auxiliary training signal.
10. **A genuinely untouched final release-test.** `docs/MODEL.md` §5 was
    developed by inspecting confusions on the test split, and `build_pack.py`
    now populates `similar_species`/per-class accuracy from
    `class_metrics_test.json` — real, useful, but it means the held-out test
    split has become a development holdout for the *product*, not only a
    frozen number for the *model*. Consider a four-way split for V2 (train /
    val / development-test / release-test), building calibration and safety
    cross-references from val/dev-test, and reading release-test exactly once
    after everything is frozen.
11. **Clear 12 pre-existing Android lint warnings**, then flip
    `warningsAsErrors = true` in `app/android/build.gradle.kts` so they can't
    silently return: 8× `UseKtx` (`String.toUri`, `SharedPreferences.edit`),
    `ObsoleteSdkInt` (an unneeded `v26` resource qualifier — `minSdk` is
    already 26), `MonochromeLauncherIcon`, and `SelectedPhotoAccess` (Android
    14's partial photo-access API, worth a real look rather than a suppress —
    `PickVisualMedia` may already cover it, needs checking). `abortOnError`
    is already `true`; these are style/completeness, not correctness, which
    is why they were left for this pass rather than mixed into it.

---

## End-to-end spot check

The shipping pack run against five real test photographs, via
`scripts/verify_pack.py --pack <pack> --image <photo>`:

| true species | verdict | top-1 |
|---|---|---|
| *Perca fluviatilis* | **identified** | *Perca fluviatilis* 95.6% |
| *Salmo trutta* | uncertain | *Oncorhynchus mykiss* 9.4% (truth at rank 3) |
| *Forsterygion lapillum* | uncertain | *Lipophrys pholis* 62.8% (truth at rank 3) |
| *Catostomus commersonii* | uncertain | *Amia calva* 11.6% (truth outside top 5) |
| *Trachinus draco* | uncertain | *Sander canadensis* 5.3% |

One identification, four refusals, **no confidently wrong answers** — including
the case where the correct species was not in the top five at all. A 62.8%
top-1 was still refused, which is the 0.80 threshold doing what §4 of
[`docs/MODEL.md`](docs/MODEL.md) says it should.

---

## Known problems / open risks

- **No Android device or emulator available.** The app compiles and packages but
  has never run on a phone. Mitigated by keeping all decision logic in a
  pure-JVM module tested against real ONNX Runtime; UI, camera and permissions
  remain unexercised. Stated plainly in README rather than glossed.
- **Packs are not signed.** Integrity verified, authenticity not. v2 work.
- **Long tail is thin.** The smallest class has 23 training images. The genus
  fallback and per-class thresholds exist for this, but it will show up in
  per-class recall and must be reported, not hidden.
- **582 images carry conflicting species labels** (identical bytes, two taxa).
  Counted and reported; not yet used to filter.
- **ROCm-on-Windows requires `cudnn.enabled = False`** (MIOpen cannot compile
  its BatchNorm kernels — no C++ stdlib headers in the wheels). Applied only on
  that platform.
- **The corpus lives on an HDD.** Fixed for training via the SSD cache, but any
  step that re-reads the CAS (verify, dedupe, cache rebuild) is I/O bound.
- **Context tags are schema-supported but unpopulated.** The brief asks for
  `underwater / angler_hand / ground / landing_net / boat / market / specimen /
  aquarium`, and `provenance.context_tag` exists for them. iNaturalist does not
  publish anything equivalent, and inferring them reliably needs a scene
  classifier we have not built. Guessing from image statistics would put
  fabricated labels into the evaluation, which is worse than the gap, so the
  column is populated only where a tag is genuinely known (the non-fish
  negatives carry `negative:<clade>`).
  **Consequence:** the "angler-style test subset" the brief asks for is not yet
  a separate measurable set. The `unseen_observer` flag on the test split is the
  nearest honest proxy currently available (257 images) and is now measured — it
  is where the 0.3113 figure below comes from. The held-out-species and non-fish
  negative sets cover the open-set requirement properly.

---

## Metrics

Measured on the held-out test split, read after calibration was fitted on
validation. Full detail and caveats in [`docs/MODEL.md`](docs/MODEL.md).

### Recognition — test split, 28,943 images, 1,978 classes

| | |
|---|---|
| top-1 | **0.5336** |
| top-3 / top-5 | 0.6747 / 0.7234 |
| genus / family accuracy | 0.6114 / 0.6810 |
| macro F1 / weighted F1 | 0.4840 / 0.5262 |
| ECE after temperature scaling | **0.0071** (from 0.0702; T = 0.891) |

### The number that matters

| | n | top-1 |
|---|---|---|
| photographer also in train | 28,686 | 0.5356 |
| **photographer never seen** | **257** | **0.3113** |

The headline figure is inflated by photographer overlap. Observation grouping
prevented same-observation leakage, but photographers still span train and test.
95% CI on the unseen figure is [0.255, 0.368], and it is not a class-difficulty
artefact — that subset has *higher* median training support (233 vs 192).
This is the top finding of the release run and the main driver for the next
corpus build. See [`docs/MODEL.md`](docs/MODEL.md) §5.

### What reaches the user at the shipped 0.80 threshold

| | seen photographers (val) | unseen photographers |
|---|---|---|
| species answer shown | 29.9% at 93.0% precision | 15.2% at 71.8% precision |
| genus fallback | 24.1% at 78.6% precision | 20.6% at 66.0% precision |
| "not sure" | 46.0% | 64.2% |
| **useful / wrong** | **46.7% / 7.2%** | **24.5% / 11.3%** |

Accuracy falls 22 points on unseen photographers but wrong answers rise only 4:
the model goes quiet rather than confidently wrong, which is the behaviour the
uncertainty stack was built for.

### Safety cross-reference

53 of the 1,275 measured confusion pairs put a species carrying a `danger`
warning behind a prediction carrying none — the worst being *Trachinus draco*
(venomous) called *Mullus barbatus* (harmless) in 30.8% of cases. Warnings are
now collected across all candidates, and `similar_species` ships a measured
cross-reference so 49 otherwise-unwarned species reach a dangerous look-alike.
Candidate scanning alone covers only 32% of that exposure; the static
cross-reference is what closes it. See [`docs/MODEL.md`](docs/MODEL.md) §7.

### Open-set rejection

| negative set | n | rejected |
|---|---|---|
| held-out fish species (88 unseen) | 3,000 | 93.3% |
| non-fish, same source and conditions | 2,978 | 97.8% |
| synthetic | 1,000 | 100.0% |

The held-out set was originally scoped by excluded *candidate*, not excluded
*species* — 717 "unseen species" were mostly trained species leaking in
through re-uploaded duplicates. Fixing the query dropped the pool to the 88
species genuinely never trained on and raised the measured rate accordingly.
See [`docs/MODEL.md` §6](docs/MODEL.md).

### Shipped artefacts

| | |
|---|---|
| Model | ONNX fp16, 7.5 MB, 3,993,078 params (+0.0007 top-1 vs fp32, 99.9% agreement) |
| Pack | `global_v1-v1.fwpack`, 32.6 MB, all 16 `verify_pack.py` checks pass |
| Tests | 211 Python (`.venv`) + 11 torch (`.venv-train`) + 198 Kotlin (136 core + 51 cli + 11 android) |

### Hardware and pipeline

| | |
|---|---|
| fp16 matmul, RX 7800 XT | 45.5 TFLOPS (fp32: 1.8 — AMP mandatory) |
| MobileNetV3-Large training | 374 img/s (bs96, 224px, AMP); full run 8h07m |
| Image download | ~80 img/s sustained, 307,415 images in 66 min |
| Preprocessing parity (Kotlin vs Python) | max diff 0.017, rms 0.0063 |
| Inference (desktop CPU, 4 threads) | 2.19 ms fp16, 42 ms cold start |
| Debug APK (arm64-v8a) | 54.8 MB |

---

## Commands to reproduce

```bash
# environments
py -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
py -m venv .venv-train   # see docs/TRAINING.md for the torch index

# data (~2.5 h, dominated by the image download)
.venv/Scripts/python tools/dataset.py download --source inaturalist
.venv/Scripts/python tools/dataset.py download --source gbif
.venv/Scripts/python tools/taxonomy.py reconcile
.venv/Scripts/python tools/dataset.py extract
.venv/Scripts/python scripts/fetch_wikidata.py
.venv/Scripts/python tools/dataset.py fetch --policy production \
    --per-species-cap 300 --min-observations 25 --workers 64
.venv/Scripts/python tools/dataset.py build-corpus --corpus global_v1

# model
.venv-train/Scripts/python tools/prepare_cache.py \
    --corpus global_v1 --cache-root E:/fisherwiki-cache
.venv-train/Scripts/python ml/train.py --config ml/configs/global_v1.yaml
.venv-train/Scripts/python ml/evaluate.py --run <run> --split val --fit-calibration
.venv-train/Scripts/python ml/evaluate.py --run <run> --split test
.venv-train/Scripts/python ml/export.py --run <run> --quantize int8_static
.venv/Scripts/python tools/build_pack.py --run <run> \
    --pack-id global_v1 --display-name "Global Angler"

# tests
.venv/Scripts/python -m pytest tests/ -q        # 146
cd app && ./gradlew :core:test                  # 104
```
