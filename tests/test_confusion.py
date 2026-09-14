"""Measured confusion becomes shippable data only under explicit rules.

The safety-widening rule is the one that matters. A pair below the normal
frequency floor still ships when exactly one side carries a danger warning,
because the cost of the two errors is not symmetric: failing to mention a
weeverfish is not the same kind of mistake as failing to mention a wrasse.
"""

from __future__ import annotations

from fwdata import confusion


def pair(true, pred, ti, pi, count, rate):
    return {"true": true, "predicted": pred, "true_class": ti,
            "predicted_class": pi, "count": count, "rate": rate}


CLASS_TO_TAXON = {0: 100, 1: 200, 2: 300, 3: 400}


def build(pairs, dangerous=frozenset()):
    metrics = {"confusion_pairs": pairs, "classes": []}
    return confusion.build_similar_species(metrics, CLASS_TO_TAXON, set(dangerous))


def test_a_frequent_confusion_ships_in_both_directions():
    rows, report = build([pair("A", "B", 0, 1, 10, 0.4)])
    keys = {(r["fw_taxon_id"], r["other_fw_taxon_id"]) for r in rows}
    # The row keyed on the *predicted* species is the one that protects the
    # user, since that is the name the app puts on screen.
    assert (200, 100) in keys
    assert (100, 200) in keys
    assert report.pairs_shipped == 2


def test_a_rare_confusion_is_dropped():
    rows, report = build([pair("A", "B", 0, 1, 1, 0.001)])
    assert rows == []
    assert report.pairs_shipped == 0


def test_a_low_rate_confusion_is_dropped_even_when_frequent():
    # 5 occurrences, but out of a very large support: not characteristic.
    rows, _ = build([pair("A", "B", 0, 1, 5, 0.005)])
    assert rows == []


def test_a_single_safety_crossing_confusion_still_ships():
    # One occurrence, negligible rate - but B is venomous and A is not.
    rows, report = build([pair("A", "B", 0, 1, 1, 0.001)], dangerous={200})
    assert report.safety_crossing == 1
    keys = {(r["fw_taxon_id"], r["other_fw_taxon_id"]) for r in rows}
    assert (100, 200) in keys, "the harmless species must reach the warning"


def test_crossing_is_detected_in_either_direction():
    # The dangerous species being the *true* label is the important case: the
    # model named something harmless for a fish that can hurt you.
    _, report = build([pair("Venom", "Harmless", 0, 1, 1, 0.01)], dangerous={100})
    assert report.safety_crossing == 1


def test_two_dangerous_species_do_not_count_as_crossing():
    # Both venomous: no asymmetry, so the normal frequency floor applies.
    rows, report = build([pair("A", "B", 0, 1, 1, 0.001)], dangerous={100, 200})
    assert report.safety_crossing == 0
    assert rows == []


def test_two_harmless_species_do_not_count_as_crossing():
    _, report = build([pair("A", "B", 0, 1, 1, 0.001)])
    assert report.safety_crossing == 0


def test_the_highest_measured_rate_wins_for_a_duplicated_direction():
    rows, _ = build([
        pair("A", "B", 0, 1, 10, 0.40),
        pair("B", "A", 1, 0, 5, 0.10),
    ])
    by_key = {(r["fw_taxon_id"], r["other_fw_taxon_id"]): r for r in rows}
    assert by_key[(100, 200)]["confusion_rate"] == 0.40
    assert by_key[(200, 100)]["confusion_rate"] == 0.40


def test_no_row_asserts_a_morphological_difference():
    rows, _ = build([pair("A", "B", 0, 1, 10, 0.4)])
    for r in rows:
        assert r["difference"] == confusion.NO_DESCRIBED_DIFFERENCE
        # The text must say plainly that nothing is recorded, so a reader who
        # sees only the database is not misled into thinking it is guidance.
        assert "No distinguishing feature is recorded" in r["difference"]


def test_a_class_with_no_taxon_is_skipped_rather_than_guessed():
    rows, _ = build([pair("A", "B", 0, 99, 10, 0.4)])
    assert rows == []


def test_a_species_is_never_similar_to_itself():
    rows, _ = build([pair("A", "A", 0, 0, 10, 0.4)])
    assert rows == []


def test_pairs_without_class_indices_are_skipped():
    # The human-readable top-30 list has no class indices; feeding it here must
    # produce nothing rather than a bad join.
    rows, _ = build([{"true": "A", "predicted": "B", "count": 9, "rate": 0.4}])
    assert rows == []


def test_missing_metrics_file_returns_none_not_an_empty_result(tmp_path):
    # An unevaluated run is a legitimate state; it must not look like a model
    # with no confusions at all.
    assert confusion.load_metrics(tmp_path) is None


def test_load_metrics_reads_the_split_specific_file(tmp_path):
    import json
    (tmp_path / "class_metrics_test.json").write_text(
        json.dumps({"classes": [{"class_index": 0}], "confusion_pairs": []}),
        encoding="utf-8")
    m = confusion.load_metrics(tmp_path)
    assert m is not None
    assert confusion.class_metrics_by_index(m) == {0: {"class_index": 0}}
