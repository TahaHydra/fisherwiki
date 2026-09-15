"""The parallel preparation pipeline must produce exactly what the serial one did.

Moving decode/crop/resize/encode into a worker pool is a throughput change, and
throughput changes are allowed to make things faster. They are not allowed to
change *which* sample lands in which shard, what its class id is, or which
images are cropped - all of which become race-dependent the moment results are
consumed out of order.

The draft-decode check is the other half: letting the JPEG decoder downscale in
the DCT domain moves the crop into a different coordinate system, and a box that
is not rescaled with it crops the wrong part of the fish.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("pyarrow", reason="shard indexes are parquet")
pytest.importorskip("duckdb", reason="the corpus lives in duckdb")
PIL = pytest.importorskip("PIL", reason="preparation decodes images")

import hashlib  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

N_IMAGES = 12


def _synthetic_jpeg(seed: int, size=(1600, 1200)) -> bytes:
    """A photo-like image: smooth background, a high-contrast blob for a fish.

    Flat colour would make every resampling filter agree and the equivalence
    test vacuous, so there is real high-frequency detail in here.
    """
    im = Image.new("RGB", size)
    px = im.load()
    for y in range(0, size[1], 4):
        for x in range(0, size[0], 4):
            v = ((x * 7 + y * 13 + seed * 31) % 200) + 30
            for dy in range(4):
                for dx in range(4):
                    if x + dx < size[0] and y + dy < size[1]:
                        px[x + dx, y + dy] = (v, (v * 2) % 255, (v * 3) % 255)
    draw = ImageDraw.Draw(im)
    draw.ellipse([size[0] * 0.3, size[1] * 0.35, size[0] * 0.7, size[1] * 0.6],
                 fill=(240, 120, 40))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A tiny CAS plus the split/detection tables preparation reads."""
    import duckdb

    from fwdata import config

    root = tmp_path / "data"
    paths = config.Paths(root).ensure()
    monkeypatch.setattr(config, "PATHS", paths)
    import prepare_shards_v2 as prep
    monkeypatch.setattr(prep, "PATHS", paths)

    db = root / "provenance.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE provenance (sha256 VARCHAR, cas_path VARCHAR, "
                "accepted_scientific_name VARCHAR, species_taxon_id BIGINT)")
    con.execute("CREATE TABLE split_groups (dataset_version VARCHAR, "
                "group_id VARCHAR, split VARCHAR, taxon_id BIGINT)")
    con.execute("CREATE TABLE split_group_members (dataset_version VARCHAR, "
                "group_id VARCHAR, sha256 VARCHAR)")
    con.execute("CREATE TABLE split_quarantine (dataset_version VARCHAR, "
                "sha256 VARCHAR)")
    con.execute("CREATE TABLE detections (sha256 VARCHAR, x0 DOUBLE, y0 DOUBLE, "
                "x1 DOUBLE, y1 DOUBLE, is_primary BOOLEAN)")

    names = {100: "Abramis brama", 200: "Perca fluviatilis", 300: "Zebrasoma"}
    for i in range(N_IMAGES):
        data = _synthetic_jpeg(i)
        sha = hashlib.sha256(data).hexdigest()
        rel = f"{sha[:2]}/{sha[2:4]}/{sha}.jpg"
        dest = paths.cas / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        taxon = [100, 200, 300][i % 3]
        split = ["train", "train", "validation", "dev_test"][i % 4]
        con.execute("INSERT INTO provenance VALUES (?, ?, ?, ?)",
                    [sha, rel, names[taxon], taxon])
        # The group's taxon is deliberately wrong for one image: a real
        # observation can hold photos of several species, and the group only
        # decides the split.
        group_taxon = 999 if i == 7 else taxon
        con.execute("INSERT INTO split_groups VALUES ('v2', ?, ?, ?)",
                    [f"g{i}", split, group_taxon])
        con.execute("INSERT INTO split_group_members VALUES ('v2', ?, ?)",
                    [f"g{i}", sha])
        if i % 5 != 4:                       # one image in five has no box
            con.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, TRUE)",
                        [sha, 480.0, 420.0, 1120.0, 720.0])
    con.close()
    return {"root": root, "db": db, "paths": paths}


def _read_all(out_dir, splits):
    """Every sample in the shard set, as {key: (shard, class_id, bytes)}."""
    import pyarrow.parquet as pq

    from fwml.shards import shard_sidecar

    out = {}
    for split in splits:
        d = out_dir / split
        if not d.exists():
            continue
        for shard in sorted(d.glob("*.tar")):
            table = pq.read_table(shard_sidecar(shard)).to_pylist()
            with open(shard, "rb") as fh:
                for row in table:
                    fh.seek(row["offset"])
                    out[row["key"]] = (row["shard"], row["class_id"],
                                       fh.read(row["length"]))
    return out


SPLITS = ("train", "validation", "dev_test", "final_test")


def _build(out, corpus, **kw):
    import prepare_shards_v2 as prep

    return prep.build(out, db_path=corpus["db"], shard_size=4, splits=SPLITS,
                      log=lambda *_a, **_k: None, **kw)


class TestDeterminism:
    def test_worker_count_does_not_change_a_single_byte(self, corpus, tmp_path):
        """Shard membership, class ids and pixels are identical at any width."""
        serial = tmp_path / "serial"
        parallel = tmp_path / "parallel"
        _build(serial, corpus, workers=0)
        _build(parallel, corpus, workers=2, chunk=2)

        a, b = _read_all(serial, SPLITS), _read_all(parallel, SPLITS)
        assert set(a) == set(b) and len(a) == N_IMAGES
        assert a == b, "parallel preparation produced a different corpus"

    def test_class_ids_follow_the_persisted_map(self, corpus, tmp_path):
        from fwml.shards import ClassMap

        out = tmp_path / "out"
        _build(out, corpus, workers=0)
        cm = ClassMap(out)
        assert cm.by_taxon == {100: 0, 200: 1, 300: 2}, "name order, not row order"
        for _key, (_shard, class_id, _data) in _read_all(out, SPLITS).items():
            assert class_id in (0, 1, 2)

    def test_the_label_is_the_image_s_taxon_not_its_leak_group_s(
        self, corpus, tmp_path
    ):
        """Image 7 sits in a group whose taxon is 999 - a real case, because an
        observation can hold several species. Labelling from the group mislabels
        it and invents a class that no species owns."""
        from fwml.shards import ClassMap

        out = tmp_path / "out"
        _build(out, corpus, workers=0)
        cm = ClassMap(out)
        assert 999 not in cm.by_taxon, "the group's taxon became a class"
        assert len(cm) == 3, f"expected 3 species, got {len(cm)}"


class TestResumeEndToEnd:
    def test_an_interrupted_run_resumes_without_duplicates_or_gaps(
        self, corpus, tmp_path
    ):
        """Power-loss shape: half the corpus prepared, tmp shards left behind,
        then one resume command that finishes the job."""
        import duckdb

        import prepare_shards_v2 as prep
        from fwml.shards import scan_finalised

        out = tmp_path / "out"
        con = duckdb.connect(str(corpus["db"]), read_only=True)
        rows = prep.corpus_rows(con, SPLITS)
        con.close()
        assert len(rows) == N_IMAGES

        _build(out, corpus, workers=0, rows=rows[:6])
        first = _read_all(out, SPLITS)
        assert first, "nothing survived the first run"

        # Simulate the interruption the real run hit: a shard mid-write.
        (out / "train" / "train-09999.tar.tmp").write_bytes(b"junk")

        _build(out, corpus, workers=2, chunk=2)
        assert not list((out / "train").glob("*.tar.tmp")), "tmp shard survived"

        final = _read_all(out, SPLITS)
        assert len(final) == N_IMAGES, "a sample was lost or duplicated"
        for key, value in first.items():
            assert final[key] == value, "a completed sample was rewritten"
        assert all(not scan_finalised(out / s)["stale_tmp"] for s in SPLITS)

    def test_changing_a_setting_refuses_rather_than_mixing(self, corpus, tmp_path):
        from fwml.shards import FingerprintMismatch

        out = tmp_path / "out"
        _build(out, corpus, workers=0)
        before = _read_all(out, SPLITS)
        with pytest.raises(FingerprintMismatch) as err:
            _build(out, corpus, workers=0, long_edge=256)
        assert "long_edge" in str(err.value)
        assert _read_all(out, SPLITS) == before, "the refused run wrote anyway"


class TestDraftDecode:
    def test_drafting_crops_the_same_region_it_would_have_at_full_size(
        self, corpus, tmp_path
    ):
        """draft() rescales the image under the box. If the box is not rescaled
        with it, the crop lands somewhere else entirely - so compare pixels, not
        just sizes."""
        import numpy as np

        full = tmp_path / "full"
        drafted = tmp_path / "drafted"
        _build(full, corpus, workers=0, decode_draft=False)
        _build(drafted, corpus, workers=0, decode_draft=True)

        a, b = _read_all(full, SPLITS), _read_all(drafted, SPLITS)
        assert set(a) == set(b)
        worst = 0.0
        for key in a:
            ia = Image.open(io.BytesIO(a[key][2])).convert("RGB")
            ib = Image.open(io.BytesIO(b[key][2])).convert("RGB")
            assert ia.size == ib.size, f"{key}: drafted output changed shape"
            diff = np.abs(np.asarray(ia, dtype=float)
                          - np.asarray(ib, dtype=float)).mean()
            worst = max(worst, diff)
        # Not bit-identical - a different decode path is the whole point - but
        # the same framing. A misapplied box shows up here as tens of levels.
        assert worst < 12.0, f"drafted crops differ by {worst:.1f} mean levels"

    def test_the_fingerprint_separates_the_two_decode_paths(self, corpus, tmp_path):
        from fwml.shards import read_fingerprint

        full, drafted = tmp_path / "full", tmp_path / "drafted"
        _build(full, corpus, workers=0, decode_draft=False)
        _build(drafted, corpus, workers=0, decode_draft=True)
        assert read_fingerprint(full)["decode_draft"] is False
        assert read_fingerprint(drafted)["decode_draft"] is True


class TestSupersededResolutions:
    def test_prefer_source_drops_the_lower_resolution_copy(self, corpus, tmp_path):
        """After a 1024px re-fetch, the same photograph exists twice: two files,
        two sha256s, one observation group. Preparing both would put two
        resolutions of one photo in the same split."""
        import duckdb

        import prepare_shards_v2 as prep

        # Re-fetch image 0 at a larger size: same source photo id, new bytes.
        con = duckdb.connect(str(corpus["db"]))
        old = con.execute("SELECT sha256, cas_path, accepted_scientific_name, "
                          "species_taxon_id FROM provenance "
                          "ORDER BY sha256 LIMIT 1").fetchone()
        con.execute("ALTER TABLE provenance ADD COLUMN source_dataset VARCHAR")
        con.execute("ALTER TABLE provenance ADD COLUMN source_record_id VARCHAR")
        con.execute("UPDATE provenance SET source_dataset='inaturalist', "
                    "source_record_id=sha256")
        big = "f" * 64
        con.execute("INSERT INTO provenance VALUES (?, ?, ?, ?, ?, ?)",
                    [big, old[1], old[2], old[3], "inaturalist-large", old[0]])
        group = con.execute("SELECT group_id FROM split_group_members "
                            "WHERE sha256 = ?", [old[0]]).fetchone()[0]
        con.execute("INSERT INTO split_group_members VALUES ('v2', ?, ?)",
                    [group, big])
        con.close()

        con = duckdb.connect(str(corpus["db"]), read_only=True)
        try:
            both = prep.corpus_rows(con, SPLITS)
            preferred = prep.corpus_rows(con, SPLITS,
                                         prefer_source="inaturalist-large")
        finally:
            con.close()

        assert len(both) == N_IMAGES + 1, "precondition: both copies are assigned"
        keys = {r[0] for r in preferred}
        assert len(preferred) == N_IMAGES
        assert big in keys, "the large copy should be the one kept"
        assert old[0] not in keys, "the superseded 500px copy was still prepared"
