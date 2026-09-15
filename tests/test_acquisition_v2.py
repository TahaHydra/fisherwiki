"""Fast, network-free tests for the V2 broad acquisition layer."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from fwdata.provenance import ImageProvenance
from fwdata.taxonomy.registry import TaxonRecord
from fwdata.sources import commons_media, fathomnet_media, gbif_media


class State:
    def __init__(self):
        self.marked = []

    def needs(self, key, cap):
        return True

    def mark(self, key, cap):
        self.marked.append((str(key), int(cap)))


class DB:
    def flush(self):
        pass


def taxon(**kw):
    base = dict(
        fw_taxon_id=1234,
        canonical_name="Perca fluviatilis",
        rank="species",
        gbif_taxon_id=8140485,
    )
    base.update(kw)
    return TaxonRecord(**base)


def test_gbif_uses_real_occurrence_parameter_names(monkeypatch):
    seen = []

    def fake_json(url):
        seen.append(url)
        return {"results": [], "endOfRecords": True}

    monkeypatch.setattr(gbif_media, "_json", fake_json)
    st = State()
    assert gbif_media.discover_taxon(DB(), taxon(), cap=3, state=st) == 0
    q = parse_qs(urlsplit(seen[0]).query)
    assert q["mediaType"] == ["StillImage"]
    assert q["taxonKey"] == ["8140485"]
    assert "media_type" not in q and "taxon_key" not in q
    assert st.marked == [("1234", 3)]


def test_gbif_excludes_inaturalist_mirror():
    occ = {
        "key": 1,
        "datasetKey": gbif_media.INAT_DATASET_KEY,
        "media": [{"type": "StillImage", "identifier": "https://x/fish.jpg"}],
    }
    assert gbif_media._records(occ, taxon()) == []


def test_commons_keeps_same_file_assertions_separate_by_taxon(monkeypatch):
    calls = []

    def fake_json(params):
        calls.append(params)
        if params.get("list") == "categorymembers":
            return {"query": {"categorymembers": [{"pageid": 77, "title": "File:X.jpg"}]}}
        return {"query": {"pages": [{
            "pageid": 77,
            "title": "File:X.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/x.jpg",
                "descriptionurl": "https://commons.wikimedia.org/?curid=77",
                "mime": "image/jpeg",
                "width": 1000,
                "height": 700,
                "extmetadata": {},
            }],
        }]}}

    captured = []
    monkeypatch.setattr(commons_media, "_json", fake_json)
    monkeypatch.setattr(
        commons_media,
        "register_candidates",
        lambda db, records: captured.extend(records) or len(records),
    )
    st = State()
    assert commons_media.discover_taxon(DB(), taxon(), cap=1, state=st) == 1
    assert captured[0].source_record_id == "77:1234"
    assert captured[0].group_key == "commons:77"


def test_fathomnet_provider_failure_is_resumable(monkeypatch):
    monkeypatch.setattr(fathomnet_media, "_json", lambda url: None)
    st = State()
    assert fathomnet_media.discover_taxon(DB(), taxon(), cap=2, state=st, log=lambda x: None) == 0
    # A provider failure must not poison the durable discovery checkpoint.
    assert st.marked == []


def test_fathomnet_circuit_breaker_avoids_repeated_network_calls(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("503")

    monkeypatch.setattr(fathomnet_media, "_PROVIDER_DOWN", False)
    monkeypatch.setattr(fathomnet_media, "_LAST_ERROR", None)
    monkeypatch.setattr(fathomnet_media.net, "fetch_bytes", fail)
    assert fathomnet_media._json("https://example.invalid/one") is None
    assert fathomnet_media._json("https://example.invalid/two") is None
    assert len(calls) == 1


def test_fathomnet_requires_matching_nonrejected_box(monkeypatch):
    monkeypatch.setattr(fathomnet_media, "_json", lambda url: [{
        "uuid": "abc",
        "url": "https://example.org/fish.jpg",
        "width": 1920,
        "height": 1080,
        "boundingBoxes": [
            {"concept": "Perca fluviatilis", "x": 1, "y": 2,
             "width": 100, "height": 50, "rejected": False, "observer": "ann"},
            {"concept": "Other fish", "x": 0, "y": 0,
             "width": 20, "height": 20, "rejected": False},
        ],
    }])
    captured: list[ImageProvenance] = []
    monkeypatch.setattr(
        fathomnet_media,
        "register_candidates",
        lambda db, records: captured.extend(records) or len(records),
    )
    st = State()
    assert fathomnet_media.discover_taxon(DB(), taxon(), cap=10, state=st) == 1
    assert captured[0].source_record_id == "abc:1234"
    assert captured[0].context_tag == "underwater"
    assert "fathomnet_box=1,2,100,50" in (captured[0].notes or "")
