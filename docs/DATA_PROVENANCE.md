# Data provenance

Every image in this project is traceable to the record it came from, the person
who took it and the licence it carries. This document describes how that is
stored, how it is enforced, and how to reproduce or audit it.

The licensing *analysis* is in [`DATASET_LICENSES.md`](DATASET_LICENSES.md).
This is the mechanism.

---

## 1. The rule

> **There is exactly one way to put an image on disk: `ImageStore.put()`, and it
> requires a fully-populated provenance record whose licence the active policy
> admits.**

There is no helper anywhere in the codebase that writes image bytes without
provenance, and the check happens *before* the filesystem is touched. This is
enforced by API surface rather than by developer discipline, because discipline
is exactly what fails at 2 a.m. six months from now.

Two consequences worth stating:

* A file in the content-addressed store with no provenance row is an **error**,
  not a warning — `dataset.py verify` reports it as such. An image with no
  provenance row is an image with no recorded licence.
* Changing the licence policy re-selects rows but never re-downloads pixels,
  because storage is content-addressed.

---

## 2. What is recorded

One row per candidate photograph, written **before** anything is downloaded:

| field | meaning |
|---|---|
| `candidate_id` | `<source>:<record_id>`, primary key |
| `source_dataset` | `inaturalist`, `inaturalist-nonfish`, ... |
| `source_record_id` | the provider's own id (iNaturalist `photo_id`) |
| `image_url` | the exact URL the bytes came from |
| `source_url` | human-facing page for the record |
| `source_taxon_id` | taxon id **as asserted by the source** |
| `original_scientific_name` | name as asserted by the source |
| `accepted_scientific_name` | after taxonomy reconciliation |
| `taxon_id` | our canonical `fw_taxon_id` |
| `license` | normalised, from the closed vocabulary |
| `license_raw` | **exactly** as the source gave it |
| `license_url` | canonical licence URL |
| `creator`, `copyright_holder` | photographer |
| `attribution` | rendered attribution string |
| `group_key` | observation UUID — the leakage-safe split unit |
| `observer_key` | photographer id — the second grouping level |
| `latitude`, `longitude`, `positional_accuracy` | as published |
| `observed_on`, `quality_grade` | |
| `position_in_observation` | 0 = the first photo of a sighting |
| `declared_width`, `declared_height`, `ext` | as the source states them |
| `context_tag`, `notes` | e.g. `negative:Amphibia`, anomaly score |
| `discovered_at` | when we first considered it |

Then, once bytes exist, a second row records what only measurement can know:

| field | meaning |
|---|---|
| `sha256` | of the exact bytes; also the storage path |
| `bytes`, `width`, `height` | measured, not declared |
| `phash`, `dhash` | perceptual hashes for dedupe and leakage checks |
| `cas_path` | location in the content-addressed store |
| `download_timestamp` | |

`license` and `license_raw` are both kept deliberately. The normalised enum
collapses CC version numbers (`CC BY 2.0` → `CC-BY-4.0`) because policy does not
depend on the version, but attribution does, so the original string is
preserved.

---

## 3. Storage layout

```
$FISHERWIKI_DATA/
  raw/                    provider bulk dumps, exactly as published
    inaturalist/          observations, photos, taxa, observers (.csv.gz)
    gbif/                 backbone-simple-2023-08-28.txt.gz
  cas/                    content-addressed images
    ab/cd/abcd…ef.jpg
  provenance.duckdb       candidates, stored, failures, corpus_members, flags
  work/                   intermediates: candidate parquet, taxonomy, caches
  artifacts/              corpus manifests, training runs, built packs
```

Content addressing (`cas/<sha[0:2]>/<sha[2:4]>/<sha>.<ext>`) means the same
photograph obtained twice is stored once, a corpus is just a list of hashes, and
deleting a corpus never deletes pixels.

---

## 4. Failures are recorded too

`fetch_failures` holds why a candidate never became an image, classified as:

* `permanent` — 403/404/410, or undecodable. Never retried; a re-run skips it.
* `transient` — network or filesystem trouble. Retried on the next run.

This matters for honesty about coverage: 803 of 308,227 selected photographs had
been deleted at the source by the time we fetched them. That is a documented gap
rather than a silent one, and `verify` can explain any discrepancy between what
was selected and what exists.

The distinction was worth getting right. An earlier version counted filesystem
races as licence refusals, which made 9 disk errors look like 9 licence
violations — the wrong alarm entirely.

---

## 5. Quality facts, recorded not enforced

Every stored image carries measured facts: dimensions, perceptual hashes,
Laplacian-variance blur score, clipped-pixel fraction, saturation spread, and
any flags raised (`near_greyscale`, `very_dark`, `too_small`, `extreme_aspect`,
`heavily_clipped`, `very_bright`).

These are **descriptive**. Thresholds are applied later, per corpus, because the
right threshold is a product decision. A dark, blurry photo of a fish in a
landing net at dusk is a valuable training example, not garbage.

Only genuinely unusable images are hard-rejected: undecodable, zero-sized, or
single-colour.

---

## 6. Generated artefacts

```bash
.venv/Scripts/python tools/dataset.py attributions --corpus global_v1
```

produces:

* **`ATTRIBUTIONS.csv`** — one row per image: id, taxon, source, record id,
  source URL, image URL, creator, licence, licence URL, attribution string.
  This ships **inside every pack**, so a released model carries the credits for
  the photographs that made it.
* **`ATTRIBUTIONS.summary.json`** — image counts per licence.

---

## 7. Auditing

```bash
.venv/Scripts/python tools/dataset.py verify --sample 5000
```

checks four things:

1. every provenance row's file exists and matches its recorded size;
2. **no file in the store lacks a provenance row** (treated as an error);
3. sampled files still hash to their own filename — the path *is* the hash, so
   this detects bit-rot and tampering alike;
4. no stored image carries an inadmissible licence, optionally against a
   specific policy.

```bash
.venv/Scripts/python tools/dataset.py dedupe --threshold 6
```

reports exact duplicates, near-duplicates within a class, near-duplicates
*across* classes (label-quality suspects rather than duplicates), and
cross-split leakage.

---

## 8. Reproducing a corpus exactly

A corpus is reproducible from:

* the pinned provider snapshots (iNaturalist 2026-08-27, GBIF 2023-08-28 with
  its SHA-256 recorded),
* the committed taxon registry (`data/taxonomy/taxon_registry.tsv`),
* the licence policy name,
* the split salt (the corpus name),
* the code commit.

Split assignment is a SHA-256 of the group key rather than a shuffle, so it is
stable across runs, machines and corpus growth: adding new images never moves an
existing observation between splits. That is what makes "never tune against the
test set" enforceable rather than aspirational.

Every training run records the corpus manifest's SHA-256 and the code commit in
`run.json`, and both are propagated into the pack manifest — so a model in the
field can always be traced back to the exact data and code that produced it.

---

## 9. Removing an image

Because everything is keyed by SHA-256, a takedown request, a licence change or
a mislabelled image is handled precisely:

```sql
-- find it
SELECT * FROM provenance WHERE source_record_id = '12345678';
-- or by content
SELECT * FROM provenance WHERE sha256 = 'abcd…';
```

Delete the CAS file and the `stored` row, leave the `candidates` row with a
`fetch_failures` entry recording why, and rebuild the corpus. The candidate row
is kept deliberately: the record of *what was considered and then excluded* is
part of the provenance, not noise.
