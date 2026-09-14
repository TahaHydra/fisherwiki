"""The evaluation join between predictions and manifest rows must survive drops.

`collate_drop_failures` removes images that could not be decoded, so the model
emits fewer predictions than the split has rows, and position `i` in the output
is not row `i` in the manifest. Any per-row breakdown -- accuracy by observer,
by licence, by region -- computed by zipping the two positionally is therefore
wrong the moment a single file is unreadable, and wrong *quietly*: the shapes
still broadcast, the numbers still look plausible.

These tests pin the indexed variant that makes the join sound. They are the
reason `evaluate.py` can report an unseen-photographer number at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fwml.data import (  # noqa: E402
    collate_drop_failures,
    collate_drop_failures_indexed,
)


def _item(label: int, idx: int, size: int = 4):
    return torch.full((3, size, size), float(label)), label, idx


def test_indexed_collate_preserves_index_through_drops():
    # Rows 1 and 3 failed to decode.
    batch = [_item(10, 0), _item(-1, 1), _item(20, 2), _item(-1, 3), _item(30, 4)]
    xs, ys, idx = collate_drop_failures_indexed(batch)

    assert xs.shape[0] == 3
    assert ys.tolist() == [10, 20, 30]
    # The point of the whole exercise: the surviving rows still know where they
    # came from. Positional zipping would have said 0, 1, 2.
    assert idx.tolist() == [0, 2, 4]


def test_indexed_collate_matches_unindexed_on_labels():
    batch = [_item(7, 0), _item(-1, 1), _item(9, 2)]
    xs_a, ys_a = collate_drop_failures([(x, y) for x, y, _ in batch])
    xs_b, ys_b, _ = collate_drop_failures_indexed(batch)

    assert torch.equal(xs_a, xs_b)
    assert torch.equal(ys_a, ys_b)


def test_indexed_collate_returns_none_when_everything_failed():
    assert collate_drop_failures_indexed([_item(-1, 0), _item(-1, 1)]) is None


def test_indexed_collate_index_dtype_is_long():
    # Used to index numpy arrays and torch tensors downstream.
    _, _, idx = collate_drop_failures_indexed([_item(1, 5)])
    assert idx.dtype == torch.long


def test_dataset_emits_index_only_when_asked(tmp_path):
    """`emit_index` must default off so the training path is unaffected."""
    from PIL import Image

    from fwml.data import AugmentConfig, FishDataset

    cas = tmp_path / "cas"
    (cas / "ab" / "cd").mkdir(parents=True)
    Image.new("RGB", (32, 32), (120, 90, 60)).save(cas / "ab" / "cd" / "x.jpg")
    rows = [{"class_id": 3, "cas_path": "ab/cd/x.jpg", "sha256": "abcd" * 16}]
    cfg = AugmentConfig.eval_only(8)

    plain = FishDataset(rows, cas, cfg, train=False)
    assert len(plain[0]) == 2

    indexed = FishDataset(rows, cas, cfg, train=False, emit_index=True)
    t, label, idx = indexed[0]
    assert label == 3 and idx == 0
    assert t.shape == (3, 8, 8)


def test_dataset_emits_index_for_undecodable_images(tmp_path):
    """A failure must still report which row failed, not just that one did."""
    from fwml.data import AugmentConfig, FishDataset

    cas = tmp_path / "cas"
    (cas / "ab" / "cd").mkdir(parents=True)
    (cas / "ab" / "cd" / "broken.jpg").write_bytes(b"not a jpeg")
    rows = [{"class_id": 3, "cas_path": "ab/cd/broken.jpg", "sha256": "ab" * 32}]

    ds = FishDataset(rows, cas, AugmentConfig.eval_only(8), train=False,
                     emit_index=True)
    _, label, idx = ds[0]
    assert label == -1
    assert idx == 0
