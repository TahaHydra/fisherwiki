"""A checkpoint must not be resumed against a corpus it was not trained on.

A classifier head is a fixed-width layer whose columns mean whatever the class
map said when it was written. Point a resume at a shard set prepared with a
different class map and one of two things happens: PyTorch raises a shape
error, which is the good outcome, or the widths coincidentally match and the
run continues on labels that have quietly shifted - which converges, reports a
plausible accuracy, and is wrong.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow", reason="shard indexes are parquet")

from fwml.shards import (  # noqa: E402
    ClassMap,
    dataset_fingerprint,
    dataset_fingerprint_diff,
    write_fingerprint,
)

RECIPE = {
    "prep_impl_version": 2, "dataset_version": "v2", "long_edge": 512,
    "jpeg_quality": 90, "jpeg_optimize": False, "jpeg_progressive": False,
    "crop_pad": 0.25, "min_crop_px": 224, "whole_frame_fraction": 0.15,
    "resample": "LANCZOS", "decode_draft": True, "shard_size": 2048,
}


def _corpus(root, taxa):
    write_fingerprint(root, RECIPE)
    cm = ClassMap(root)
    cm.extend(taxa)
    cm.save()
    return dataset_fingerprint(root)


class TestDatasetFingerprint:
    def test_it_records_the_recipe_and_the_class_map(self, tmp_path):
        fp = _corpus(tmp_path, [(10, "Abramis"), (20, "Perca")])
        assert fp["num_classes"] == 2
        assert fp["preprocessing"]["long_edge"] == 512
        assert len(fp["class_map_sha256"]) == 16

    def test_the_same_corpus_fingerprints_the_same_twice(self, tmp_path):
        a = _corpus(tmp_path, [(10, "Abramis"), (20, "Perca")])
        b = dataset_fingerprint(tmp_path)
        assert dataset_fingerprint_diff(a, b) == []

    def test_adding_a_species_is_detected(self, tmp_path):
        before = _corpus(tmp_path, [(10, "Abramis"), (20, "Perca")])
        cm = ClassMap(tmp_path)
        cm.extend([(30, "Zebrasoma")])
        cm.save()
        drift = dataset_fingerprint_diff(before, dataset_fingerprint(tmp_path))
        assert any("num_classes" in d for d in drift)
        assert any("class_map_sha256" in d for d in drift)

    def test_relabelling_the_same_species_is_detected(self, tmp_path):
        """Same count, different meaning - the case a shape check cannot catch."""
        before = _corpus(tmp_path, [(10, "Abramis"), (20, "Perca")])
        path = tmp_path / ClassMap.FILENAME
        raw = json.loads(path.read_text())
        raw["classes"][0]["taxon_id"] = 99
        path.write_text(json.dumps(raw, indent=2))

        after = dataset_fingerprint(tmp_path)
        assert after["num_classes"] == before["num_classes"] == 2
        drift = dataset_fingerprint_diff(before, after)
        assert drift, "a silent relabelling passed the fingerprint check"

    def test_changed_preprocessing_is_detected(self, tmp_path):
        before = _corpus(tmp_path, [(10, "Abramis")])
        write_fingerprint(tmp_path, dict(RECIPE, long_edge=640))
        drift = dataset_fingerprint_diff(before, dataset_fingerprint(tmp_path))
        assert any("preprocessing.long_edge" in d for d in drift)

    def test_moving_the_corpus_to_another_machine_is_not_drift(self, tmp_path):
        """The same shards copied onto a rented GPU box are the same shards.
        Comparing paths would break every cloud resume."""
        before = _corpus(tmp_path / "local", [(10, "Abramis"), (20, "Perca")])
        after = _corpus(tmp_path / "rented-box", [(10, "Abramis"), (20, "Perca")])
        assert before["root"] != after["root"]
        assert dataset_fingerprint_diff(before, after) == []

    def test_an_unprepared_directory_fingerprints_to_nothing(self, tmp_path):
        fp = dataset_fingerprint(tmp_path)
        assert fp["num_classes"] is None and fp["preprocessing"] == {}
