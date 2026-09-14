# Species data

What the offline database actually contains, where each fact came from, and —
at least as importantly — **what it does not contain**.

The rule this document exists to make auditable: *no biological claim is
shipped without a citable source, and a field that cannot be sourced is absent
rather than invented.* The schema enforces the first half (`NOT NULL source_id`
on every fact table). This document is the honest accounting of the second.

Schema: [`tools/fwdata/species_schema.sql`](../tools/fwdata/species_schema.sql)
Access layer: [`SpeciesRepository.kt`](../app/core/src/main/kotlin/com/fisherwiki/core/db/SpeciesRepository.kt)

---

## 1. What is in the shipped pack

Measured on `global_v1-v1.fwpack`, 1,978 species.

| table | rows | coverage of the 1,978 classes | source |
|---|---|---|---|
| `taxa` | 1,978 | 100% | GBIF Backbone |
| `taxon_regions` | 4,644 | 98.9% | iNaturalist coordinates |
| `taxon_synonyms` | 11,965 | 84.1% | GBIF Backbone |
| `traits` | 1,643 | 83.1% | Wikidata |
| `common_names` | 12,497 | 76.8% | Wikidata |
| `similar_species` | 2,348 | 44.6% | measured (this model) |
| `safety_warnings` | 346 | 16.5% | Smith & Wheeler (2006), curated |
| `model_classes` | 1,978 | 100% | measured (this model) |
| **`habitats`** | **0** | **0%** | none found — see §4 |
| **`diagnostic_features`** | **0** | **0%** | none found — see §4 |

`common_names` spans 60+ languages: English 4,332, Chinese 2,954, Spanish 1,983,
French 1,409, German 207, Dutch 162, and a long tail.

---

## 2. Sources

Six, all recorded in the `sources` table and shipped inside the pack.

| source_id | what it supplies | licence |
|---|---|---|
| `gbif-backbone` | accepted names, rank, genus/family/order/class, synonyms | CC BY 4.0 |
| `wikidata` | common names, IUCN status, cross-identifiers, size traits | CC0 1.0 |
| `inaturalist-open-data` | regional presence from observation coordinates | mixed CC |
| `smith-wheeler-2006` | venomous-fish phylogeny | citation only |
| `fisherwiki-curated` | handling notes derived from the above | CC BY 4.0 |
| `fisherwiki-eval` | measured confusion, per-class accuracy | CC BY 4.0 |

Two of these need explaining.

**`smith-wheeler-2006`** is `CITATION-ONLY`: we cite the paper's *findings*
about which fish clades are venomous, at rank level. We do not reproduce its
text. Facts are not copyrightable; the expression of them is.

**`fisherwiki-eval`** is not a source of biology at all. It records what *this
model* does — which species it confuses, how often, and how accurately it
recognises each class. Rows citing it make no claim about fish. That is
deliberate: a reader can filter the database down to biological claims by
excluding a single source id.

### Why not FishBase

FishBase is the canonical fish-facts resource, and it is exactly what
`habitats` and `diagnostic_features` want. Its terms do not permit the
redistribution a commercially distributable offline pack requires — see
[`DATASET_LICENSES.md`](DATASET_LICENSES.md). Rather than paraphrase it, which
would launder the licence problem while adding a transcription-error problem,
those tables are empty.

---

## 3. Safety warnings

The highest evidentiary bar in the database, because the asymmetry is severe: a
wrong warning is bad, a wrong *absence* of a warning can put someone in
hospital.

346 warnings over 326 species, from 18 rank-level rules in
[`data/safety/safety_warnings.yaml`](../data/safety/safety_warnings.yaml).
Rules attach at family, order or class rank and expand to species, so
"weeverfishes have venomous spines" is stated once and reaches every
*Trachinus* in the pack.

| kind | severity | rank | rows |
|---|---|---|---|
| handling | caution | class | 97 |
| toxic_flesh | danger | family | 47 |
| venomous_spines | danger | family | 47 |
| sharp_spine | caution | family | 42 |
| venomous_spines | caution | order | 34 |
| bite | danger | family | 32 |
| venomous_spines | caution | family | 17 |
| bite | caution | family | 13 |
| toxic_flesh | caution | family | 11 |
| handling | info | family | 6 |

The expansion reports rules that matched nothing, which is how we know one
curated warning (`family:Chimaeridae:venomous_spines`) applies to no species in
this pack rather than having silently failed to apply.

### The cross-reference

A warning attached to the right species is useless if the model names the wrong
one. This model calls a venomous *Trachinus draco* a harmless *Mullus barbatus*
in 30.8% of test images.

So `similar_species` carries a **measured** cross-reference: 53 of the 1,275
confusion pairs cross a safety boundary, and 49 species that carry no warning of
their own now reach a dangerous look-alike's. See
[`MODEL.md §7`](MODEL.md).

Those rows assert nothing about morphology. Their `difference` column says so in
words, because a reader querying the database directly should not have to infer
it from a source id.

---

## 4. What is missing, and why

This is the part of the document that matters.

### `habitats` — empty

The brief this project was built to wants freshwater/marine/brackish per
species. The database has the table. It has no rows.

Wikidata's habitat statements on fish taxa are sparse and inconsistently
modelled; the GBIF backbone is nomenclatural and carries no ecology; FishBase
has it and cannot be redistributed. WoRMS could establish "marine" for taxa
carrying an AphiaID, which is a genuine partial path and is **not implemented**.

The tempting shortcut was to infer habitat from observation coordinates — there
are 216,878 georeferenced observations in the corpus. That would be inventing a
biological fact from a proxy, so it was not done. A fish photographed in an
estuary is not thereby brackish.

### `diagnostic_features` — empty

The brief asks for human-verifiable distinguishing features and calls them a
first-class product feature. They are the single largest gap in this pack.

No CC0/CC BY corpus of per-species morphological diagnostics was found.
Generating them with a language model was considered and rejected: the brief
forbids hallucinated taxonomic descriptions, and this is the exact case the
prohibition is for — plausible, fluent, unverifiable, and least reliable
precisely where the fish is unusual and the user most needs to be right.

The honest partial substitute now shipping is the measured confusion data. The
app cannot tell a user *how* to tell two species apart, but it can tell them
*which two* it mixes up and how often, which at least directs their own
comparison.

### `similar_species.difference` — present but not morphological

Populated with an explicit statement that no distinguishing feature is
recorded. When a licence-compatible morphological source is found those rows
should be **overwritten**, not supplemented, so the absence-marker cannot
survive alongside real content.

### Traits are thin

1,643 rows over 1,978 species is roughly one trait each, mostly IUCN status.
Maximum and typical length — two fields the brief explicitly asks for — are
sparse, because Wikidata records them for a minority of fish taxa.

---

## 5. How the app renders absence

Absence is displayed as absence. No code path substitutes a default, a
placeholder or plausible-sounding filler for a missing fact, and no screen
implies that an empty section means "nothing to report".

In particular, **an empty `safety_warnings` list is not a claim that a fish is
safe to handle** — it means this pack has no sourced warning for it. That
distinction is pinned by a test named
`species with no safety warning reports none rather than a reassurance`.

---

## 6. Rebuilding

```bash
.venv/Scripts/python tools/taxonomy.py --build
.venv/Scripts/python tools/build_pack.py --run <run_dir> --pack-id global_v1
```

`build_pack.py` refuses to write a pack whose database fails verification:
dense class indices from 0, every model class resolving to a taxon, and every
fact row citing a registered source.

Per-class accuracy and `similar_species` need `class_metrics_test.json` in the
run directory, written by `ml/evaluate.py --split test`. If it is absent the
build logs the omission and ships without them — it does not fabricate them,
and `verify_pack.py` reports the pack as carrying no cross-references.
