# Contributing

Thanks for looking. This project has a few rules that are stricter than usual,
and they exist for specific reasons rather than as style preferences.

---

## The three rules that are not negotiable

### 1. Every image must be licensed, and provably so

`ImageStore.put()` is the only way to write image bytes, and it requires a
populated provenance record whose licence the active policy admits.

**Do not add a code path that writes an image without provenance.** Not for a
quick experiment, not behind a flag. If you need images outside the policy, use
the `research_nc` corpus, which is marked `commercial_safe: false` and whose
models must never ship.

If you cannot establish an image's licence, it does not go in. "Unknown" is not
a licence, and "it's only for research" is not either.

### 2. Every biological fact must cite a source

The species database has `NOT NULL source_id` on every fact table, and the
loaders refuse an unregistered source key.

**Do not add a fact because you know it is true.** If it cannot be cited it does
not ship, and the app renders an absent field as absent. This applies most
strongly to `safety_warnings`: see the editorial rules at the top of
[`data/safety/safety_warnings.yaml`](data/safety/safety_warnings.yaml).

Never generate species descriptions with a language model. The whole point of
this database is that it is not plausible-sounding.

### 3. The test split is reported, never tuned against

Fit calibration on `val`. `ml/evaluate.py` refuses `--fit-calibration --split
test`, so the tool enforces this, but do not work around it.

If you need another held-out set, add one; do not borrow test.

---

## Practical conventions

### Uncertainty is a feature, not an error state

When adding anything to the result path, remember that `Certainty.UNKNOWN` and
`Certainty.COARSE_ONLY` are **normal outcomes**. The `Identification` type has
no `best` candidate in those cases, deliberately, so the UI cannot render a
species headline for an answer the engine does not stand behind. Do not add a
nullable-dodging convenience accessor that undoes this.

### `:core` must not depend on Android

This is what makes the desktop tests meaningful — they exercise the shipping
identification logic against real ONNX Runtime and real SQLite. If `:core` ever
imports `android.*`, the tests stop testing the code that ships.

Platform differences go behind an interface (`SqliteDriver`), and pixels arrive
as an `IntArray`.

### `:core` must not contain an HTTP client

The privacy claim is structural, not a policy. Identification has no network
code anywhere beneath it, and that should stay checkable by grep.

### Prefer a refusal to a warning

Most of the bugs found in this project were caught by something refusing to
proceed, not by a number looking wrong:

* the corpus builder refusing to export a leaking split (3 distinct leaks),
* strict CSV parsing refusing to silently drop rows (97% data loss),
* size and digest verification refusing a completed download (silent corruption),
* the licence gate refusing images (a bug that rejected 199 of 200 valid ones).

None of those would have shown up as a wrong accuracy number. Several would have
produced a *better-looking* one. When you add a check, make it fail the build.

### Measure, do not assume

If you write a number in a docstring or a document, it should be one you
measured on real data. The engineering log records several cases where a
plausible-sounding rationale was simply wrong — the multi-photo fusion rule and
the image resampling filter both survived review and failed a test.

---

## Setup

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -q

cd app
echo "sdk.dir=C:/path/to/Android/Sdk" > local.properties
./gradlew :core:test          # no Android SDK needed
./gradlew :android:assembleDebug
```

Training setup is in [`docs/TRAINING.md`](docs/TRAINING.md). Read the storage
section before starting a run.

### A note on the JDK

ONNX Runtime's Windows native library fails to initialise under the JetBrains
Runtime that Android Studio bundles. The build prefers an Adoptium toolchain for
tests and falls back silently, but if `:core:test` fails with
`UnsatisfiedLinkError: onnxruntime.dll`, install a stock JDK.

---

## Before opening a pull request

```bash
.venv/Scripts/python -m pytest tests/ -q
cd app && ./gradlew :core:test :android:assembleDebug
```

Then:

* If you changed the data pipeline, run `tools/dataset.py verify` and say what
  it reported.
* If you changed anything on the identification path, say what you measured.
  "Should be equivalent" is how the resampling bug survived.
* If you found something surprising, add it to
  [`docs/engineering-log.md`](docs/engineering-log.md) — including if the
  surprise was your own earlier reasoning. That file is more useful than the
  code comments for anyone picking this up later.

---

## What would be genuinely useful

* **Instrumented Android testing.** The app has never run on a device. This is
  the largest gap in the project.
* **Pack signing.** Integrity is verified; authenticity is not.
* **Wikimedia Commons ingestion.** Per-file licences are machine-readable, so it
  passes policy; it was deferred, not rejected.
* **A detector.** Cropping the fish before classification should help, and
  `Preprocessor.expandBox` already exists for it. It needs measuring, not
  assuming.
* **More languages.** Wikidata gives us 127; the database and UI already support
  them, but only English and French have meaningful coverage.
* **Diagnostic features.** `diagnostic_features` is schema-complete and nearly
  empty, because sourcing field marks under a permissive licence is genuinely
  hard. Well-cited contributions here would improve the product more than
  anything else on this list.
