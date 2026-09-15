"""Resuming preparation must not quietly build a corpus from two recipes.

Resume skips samples by *key*, and a key says nothing about the pixels behind
it. Without a fingerprint, changing ``--long-edge`` and re-running keeps every
old shard, skips every sample in it, and produces one dataset containing two
different preprocessings - which trains, converges, and is wrong in a way no
metric reports.

The second failure guarded here is quieter still: class ids were derived by
sorting whatever taxa the corpus currently held. Adding one species shifted the
ids of every species after it alphabetically, silently relabelling every shard
already on disk.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow", reason="shard indexes are parquet")
pytest.importorskip("PIL", reason="preparation decodes images")

from fwml.shards import (  # noqa: E402
    ClassMap,
    FingerprintMismatch,
    Sample,
    ShardWriter,
    check_fingerprint,
    fingerprint_diff,
    read_fingerprint,
    write_fingerprint,
)

FP = {
    "prep_impl_version": 2, "dataset_version": "v2", "long_edge": 512,
    "jpeg_quality": 90, "jpeg_optimize": False, "jpeg_progressive": False,
    "crop_pad": 0.25, "min_crop_px": 224, "whole_frame_fraction": 0.15,
    "resample": "LANCZOS", "decode_draft": True, "shard_size": 2048,
}


def _finalise_one_shard(out_dir):
    w = ShardWriter(out_dir, prefix="train", shard_size=1)
    w.write(Sample(key="k0", data=b"x" * 32, class_id=0, taxon_id=1,
                   sha256="0" * 64, split="train", group_id="g0"))
    w.close()


class TestFingerprintGate:
    def test_a_fresh_directory_is_accepted(self, tmp_path):
        assert check_fingerprint(tmp_path, FP, has_work=False) is None

    def test_matching_settings_resume(self, tmp_path):
        write_fingerprint(tmp_path, FP)
        assert check_fingerprint(tmp_path, dict(FP), has_work=True) == FP

    @pytest.mark.parametrize("key,value", [
        ("long_edge", 640),
        ("jpeg_quality", 95),
        ("crop_pad", 0.4),
        ("min_crop_px", 320),
        ("whole_frame_fraction", 0.3),
        ("decode_draft", False),
        ("prep_impl_version", 3),
        ("dataset_version", "v3"),
    ])
    def test_every_preprocessing_parameter_blocks_a_resume(self, tmp_path,
                                                           key, value):
        """Each of these changes the stored pixels, the stored bytes, or which
        images are eligible. None may be mixed into an existing shard set."""
        write_fingerprint(tmp_path, FP)
        wanted = dict(FP, **{key: value})
        with pytest.raises(FingerprintMismatch) as err:
            check_fingerprint(tmp_path, wanted, has_work=True)
        assert key in str(err.value), "the error must name the parameter"
        assert str(FP[key]) in str(err.value)
        assert str(value) in str(err.value)

    def test_finalised_shards_without_a_fingerprint_are_refused(self, tmp_path):
        """The real D: directory: valid tars written before fingerprints
        existed, so there is no record of how they were made."""
        _finalise_one_shard(tmp_path / "train")
        with pytest.raises(FingerprintMismatch) as err:
            check_fingerprint(tmp_path, FP, has_work=True)
        assert "--adopt-fingerprint" in str(err.value)

    def test_a_refused_run_leaves_the_directory_untouched(self, tmp_path):
        write_fingerprint(tmp_path, FP)
        _finalise_one_shard(tmp_path / "train")
        before = sorted(p.name for p in (tmp_path / "train").iterdir())
        with pytest.raises(FingerprintMismatch):
            check_fingerprint(tmp_path, dict(FP, long_edge=1024), has_work=True)
        assert sorted(p.name for p in (tmp_path / "train").iterdir()) == before
        assert read_fingerprint(tmp_path) == FP, "the stored recipe was edited"

    def test_the_diff_lists_every_difference_at_once(self, tmp_path):
        diff = fingerprint_diff(FP, dict(FP, long_edge=640, jpeg_quality=95))
        assert len(diff) == 2


class TestClassMapIsAppendOnly:
    def test_ids_are_assigned_in_name_order(self, tmp_path):
        cm = ClassMap(tmp_path)
        cm.extend([(30, "Zebrasoma flavescens"), (10, "Abramis brama"),
                   (20, "Perca fluviatilis")])
        assert [cm[10], cm[20], cm[30]] == [0, 1, 2]

    def test_a_new_species_never_moves_an_existing_id(self, tmp_path):
        """The bug this exists for: ids were re-derived by sorting the corpus,
        so inserting 'Abramis' ahead of 'Perca' renumbered every shard already
        written."""
        cm = ClassMap(tmp_path)
        cm.extend([(20, "Perca fluviatilis"), (30, "Zebrasoma flavescens")])
        cm.save()
        before = dict(cm.by_taxon)

        reopened = ClassMap(tmp_path)
        assert reopened.by_taxon == before, "ids did not survive a reload"
        added = reopened.extend([(10, "Abramis brama"), (20, "Perca fluviatilis"),
                                 (30, "Zebrasoma flavescens")])
        assert added == 1
        for taxon, class_id in before.items():
            assert reopened[taxon] == class_id, "an existing class id moved"
        assert reopened[10] == 2, "a new taxon takes the next free id"

    def test_the_saved_file_is_ordered_by_class_id(self, tmp_path):
        cm = ClassMap(tmp_path)
        cm.extend([(30, "Zebrasoma"), (10, "Abramis"), (20, "Perca")])
        cm.save()
        raw = json.loads((tmp_path / ClassMap.FILENAME).read_text())
        assert [e["class_id"] for e in raw["classes"]] == [0, 1, 2]
        assert [e["taxon_id"] for e in raw["classes"]] == [10, 20, 30]


class TestRelabelling:
    def test_stale_class_ids_are_re_derived_from_taxon_ids(self, tmp_path):
        """Shards adopted from an older run carry class ids from an older map.
        The taxon id in the sidecar is the fact; the class id is a derived
        index, so it is repaired rather than the shard rebuilt."""
        import pyarrow.parquet as pq

        from fwml.shards import relabel_sidecars, shard_sidecar

        w = ShardWriter(tmp_path, prefix="train", shard_size=2)
        for i in range(4):
            w.write(Sample(key=f"k{i}", data=b"x" * 16, class_id=99,
                           taxon_id=100 + (i % 2), sha256="0" * 64,
                           split="train", group_id="g"))
        w.close()

        assert relabel_sidecars(tmp_path, {100: 0, 101: 1}) == 2
        for shard in sorted(tmp_path.glob("*.tar")):
            table = pq.read_table(shard_sidecar(shard)).to_pylist()
            for row in table:
                assert row["class_id"] == {100: 0, 101: 1}[row["taxon_id"]]
        # Idempotent: a second pass has nothing to move.
        assert relabel_sidecars(tmp_path, {100: 0, 101: 1}) == 0

    def test_relabelling_preserves_every_other_column(self, tmp_path):
        import pyarrow.parquet as pq

        from fwml.shards import relabel_sidecars, shard_sidecar

        w = ShardWriter(tmp_path, prefix="train", shard_size=4)
        w.write(Sample(key="k0", data=b"payload", class_id=7, taxon_id=100,
                       sha256="a" * 64, split="dev_test", group_id="grp-1"))
        w.close()
        shard = next(tmp_path.glob("*.tar"))
        before = pq.read_table(shard_sidecar(shard)).to_pylist()[0]
        relabel_sidecars(tmp_path, {100: 3})
        after = pq.read_table(shard_sidecar(shard)).to_pylist()[0]
        assert after["class_id"] == 3
        for col in ("key", "shard", "offset", "length", "taxon_id", "sha256",
                    "split", "group_id"):
            assert after[col] == before[col], f"{col} changed"
