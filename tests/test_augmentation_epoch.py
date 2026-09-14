"""Augmentation must actually vary across epochs, including under real workers.

Before this was fixed, `FishDataset.__getitem__` seeded its RNG from
`(seed, idx)` only. The comment on that line claimed `(seed, epoch, idx)`, but
no `epoch` was ever threaded into the class at all, so image #18273 got the
exact same crop/flip/brightness/rotation/erase on every single epoch of a
30-epoch run. Not a crash, not an obviously-wrong metric -- the loss curve
still looked like augmentation was happening -- just a quieter kind of
overfitting to a fixed set of variants per image.

The fix has a real failure mode this file is written to catch: `train.py` runs
with `persistent_workers=True`, so worker processes are spawned once and
reused for every epoch. A plain Python attribute set on the dataset object in
the main process after that point never reaches an already-running worker --
DataLoader only re-pickles the dataset at worker *startup*. So the test that
matters is not "does set_epoch change a value on the object", it is "does a
real multi-worker persistent DataLoader actually see the new epoch" --
anything weaker would have passed on the broken version too, in the specific
sense that a bug here is exactly a bug that only shows up under workers.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fwml.data import AugmentConfig, FishDataset  # noqa: E402


@pytest.fixture()
def one_image_dataset(tmp_path):
    cas = tmp_path / "cas"
    (cas / "ab" / "cd").mkdir(parents=True)
    # A real photo-shaped image, not a flat colour: augmentation (crop,
    # rotation, colour jitter) needs texture to actually move pixel values.
    img = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            img.putpixel((x, y), ((x * 4) % 256, (y * 4) % 256, (x * y) % 256))
    img.save(cas / "ab" / "cd" / "x.jpg", quality=95)

    rows = [{"class_id": 0, "cas_path": "ab/cd/x.jpg", "sha256": "ab" * 32}]
    cfg = AugmentConfig(size=32)
    return FishDataset(rows, cas, cfg, train=True, seed=7), rows


def _tensor_for_epoch(ds: FishDataset, epoch: int) -> torch.Tensor:
    ds.set_epoch(epoch)
    t, _label = ds[0]
    return t


class TestSingleProcess:
    """The RNG formula itself, no multiprocessing involved."""

    def test_different_epochs_produce_different_augmentation(self, one_image_dataset):
        ds, _ = one_image_dataset
        a = _tensor_for_epoch(ds, 0)
        b = _tensor_for_epoch(ds, 1)
        assert not torch.equal(a, b), (
            "epoch 0 and epoch 1 produced byte-identical output - "
            "augmentation is not varying across epochs"
        )

    def test_the_same_epoch_is_reproducible(self, one_image_dataset):
        """Determinism matters as much as variation: a resumed run's
        augmentation must match a from-scratch run at the same epoch."""
        ds, _ = one_image_dataset
        a = _tensor_for_epoch(ds, 5)
        b = _tensor_for_epoch(ds, 5)
        assert torch.equal(a, b)

    def test_many_epochs_are_pairwise_distinct(self, one_image_dataset):
        """Guards against a formula that varies but collides, e.g. a period
        far shorter than a real training run's epoch count."""
        ds, _ = one_image_dataset
        tensors = [_tensor_for_epoch(ds, e) for e in range(10)]
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                assert not torch.equal(tensors[i], tensors[j]), f"epoch {i} == epoch {j}"

    def test_default_epoch_is_zero(self, one_image_dataset):
        """set_epoch(0) must match a dataset that never had set_epoch called,
        so a caller that forgets the very first call still gets a sane run
        rather than an exception or undefined behaviour."""
        ds, _ = one_image_dataset
        rows = ds.rows
        fresh = FishDataset(rows, ds.cas_root, ds.cfg, train=True, seed=7)
        t_fresh, _ = fresh[0]
        t_explicit = _tensor_for_epoch(ds, 0)
        assert torch.equal(t_fresh, t_explicit)


class TestPersistentWorkers:
    """The scenario that actually hid this bug in production."""

    def test_persistent_workers_see_a_later_set_epoch_call(self, one_image_dataset):
        ds, _ = one_image_dataset
        loader = DataLoader(
            ds, batch_size=1, num_workers=2, persistent_workers=True,
        )

        ds.set_epoch(0)
        epoch0 = next(iter(loader))[0]

        # The critical step: mutate epoch on the *already-spawned* dataset,
        # the same object the persistent workers hold a handle to, not a new
        # DataLoader. This is exactly what train_loop.py does every epoch.
        ds.set_epoch(1)
        epoch1 = next(iter(loader))[0]

        assert not torch.equal(epoch0, epoch1), (
            "a persistent-worker DataLoader did not see set_epoch(1) - "
            "the shared-memory epoch counter is not reaching the worker process"
        )

        ds.set_epoch(0)
        epoch0_again = next(iter(loader))[0]
        assert torch.equal(epoch0, epoch0_again), (
            "epoch 0 was not reproducible across separate iterations "
            "through the same persistent-worker loader"
        )

        del loader  # shut workers down before the fixture's tmp_path is removed
