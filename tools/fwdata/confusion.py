"""Turn measured model confusion into shippable species data.

Two tables in the pack are populated from the held-out test evaluation rather
than from any external authority:

* ``model_classes.test_precision`` / ``test_recall`` / ``test_support`` -- how
  well this model actually recognises each species, so the app can say "this
  species is weakly recognised" instead of presenting every class as equal.
* ``similar_species`` -- which species this model mixes up, and how often.

The second exists because of a specific measured failure. The model calls a
*Trachinus draco* -- a weeverfish, venomous dorsal and opercular spines -- a
*Mullus barbatus* in 30.8% of test images, sometimes at very high confidence
with the weever nowhere near the top five. Warning on the candidate list alone
therefore does not cover it. A static, measured cross-reference does: whenever
the app names a species, it can also say what that species is known to be
confused *with*, and whether any of those carry a warning.

Honesty constraints
-------------------
Nothing here is a biological claim. A row in ``similar_species`` written by this
module asserts only "this model confused these two, at this measured rate on
this test split", attributed to a source record that says exactly that. The
``difference`` column -- which is meant to hold a human-verifiable distinguishing
feature -- is filled with an explicit statement that no such feature is recorded,
rather than with anything invented. When a real morphological source is added
later it should overwrite these rows, not sit alongside them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Wording for `similar_species.difference` when the pairing comes from
#: measurement rather than from a described morphological difference. The UI
#: keys off `source_id`, but the text must stand alone if read directly.
NO_DESCRIBED_DIFFERENCE = (
    "No distinguishing feature is recorded in this pack. "
    "This pairing comes from measured model confusion, not from a described "
    "difference; compare the photographs and the taxonomy yourself."
)

#: A pair is shipped when the model made the mistake at least this often...
MIN_COUNT = 2
#: ...and at least this fraction of the time for that species.
MIN_RATE = 0.02
#: ...unless one side carries a safety warning the other lacks, in which case a
#: single occurrence is enough. Being wrong about a venomous fish once is a
#: different kind of event from being wrong about two wrasses once.
MIN_COUNT_SAFETY = 1


@dataclass
class ConfusionReport:
    pairs_considered: int
    pairs_shipped: int
    safety_crossing: int
    classes_with_metrics: int
    examples: list[str]

    def as_dict(self) -> dict:
        return {
            "pairs_considered": self.pairs_considered,
            "pairs_shipped": self.pairs_shipped,
            "safety_crossing": self.safety_crossing,
            "classes_with_metrics": self.classes_with_metrics,
            "examples": self.examples,
        }


def load_metrics(run_dir: Path, split: str = "test") -> dict | None:
    """Read ``class_metrics_<split>.json``, or None when it has not been run.

    Returning None rather than raising is deliberate: a pack built from a run
    that has not been evaluated is a legitimate intermediate state (a smoke
    build, say). The caller logs the omission; it must not silently look like
    the model has no confusions.
    """
    p = run_dir / f"class_metrics_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def class_metrics_by_index(metrics: dict) -> dict[int, dict]:
    return {c["class_index"]: c for c in metrics.get("classes", [])}


def build_similar_species(
    metrics: dict,
    class_to_taxon: dict[int, int],
    dangerous_taxa: set[int],
    *,
    log=lambda _m: None,
) -> tuple[list[dict], ConfusionReport]:
    """Rows for ``similar_species``, both directions, from measured confusion.

    ``dangerous_taxa`` is the set of ``fw_taxon_id`` carrying a ``danger``
    severity warning. It only ever *widens* what ships: a pair below the normal
    frequency floor is admitted when exactly one side is dangerous, because the
    asymmetry is the whole point -- the user is told about the venomous fish
    they might be holding, not reassured by the harmless one they might not be.
    """
    pairs = metrics.get("confusion_pairs", [])
    rows: dict[tuple[int, int], dict] = {}
    safety_crossing = 0
    examples: list[str] = []

    for pr in pairs:
        ti, pi = pr.get("true_class"), pr.get("predicted_class")
        if ti is None or pi is None:
            continue
        true_taxon = class_to_taxon.get(ti)
        pred_taxon = class_to_taxon.get(pi)
        if not true_taxon or not pred_taxon or true_taxon == pred_taxon:
            continue

        crosses = (true_taxon in dangerous_taxa) != (pred_taxon in dangerous_taxa)
        floor = MIN_COUNT_SAFETY if crosses else MIN_COUNT
        if pr["count"] < floor or (not crosses and pr["rate"] < MIN_RATE):
            continue
        if crosses:
            safety_crossing += 1
            if len(examples) < 10:
                examples.append(
                    f"{pr['true']} -> {pr['predicted']} "
                    f"({pr['count']}, {pr['rate']:.1%})"
                )

        # Both directions. The row keyed on the *predicted* species is the one
        # that protects the user: it fires when the app shows that name.
        for a, b in ((pred_taxon, true_taxon), (true_taxon, pred_taxon)):
            key = (a, b)
            prev = rows.get(key)
            if prev is None or pr["rate"] > (prev["confusion_rate"] or 0.0):
                rows[key] = {
                    "fw_taxon_id": a,
                    "other_fw_taxon_id": b,
                    "difference": NO_DESCRIBED_DIFFERENCE,
                    "confusion_rate": float(pr["rate"]),
                }

    report = ConfusionReport(
        pairs_considered=len(pairs),
        pairs_shipped=len(rows),
        safety_crossing=safety_crossing,
        classes_with_metrics=len(metrics.get("classes", [])),
        examples=examples,
    )
    log(f"    confusion: {report.pairs_shipped} similar-species rows from "
        f"{report.pairs_considered} measured pairs")
    if safety_crossing:
        log(f"    {safety_crossing} pair(s) cross a safety boundary:")
        for e in examples:
            log(f"      {e}")
    return list(rows.values()), report
