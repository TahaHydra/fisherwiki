"""Fetch the iNaturalist Open Data bulk metadata dumps.

Long-running (tens of GB). Safe to interrupt and re-run: completed 64 MB chunks
are never re-fetched. Progress is appended to
``<data_root>/work/fetch_inat_bulk.log``.

    python scripts/fetch_inat_bulk.py [--workers 12] [key ...]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fwdata import net  # noqa: E402
from fwdata.config import PATHS  # noqa: E402
from fwdata.sources import inaturalist as inat  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    PATHS.ensure()
    keys = args.keys or ["taxa", "observers", "observations", "photos"]

    files = inat.discover()
    by_key = {f.key: f for f in files}
    total = sum(by_key[k].size or 0 for k in keys)
    already = sum(
        by_key[k].local.stat().st_size for k in keys if by_key[k].local.exists()
    )

    log = PATHS.work / "fetch_inat_bulk.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def emit(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    emit(f"targets={keys} total={total/1e9:.2f} GB already_local={already/1e9:.2f} GB")
    net.require_free = None  # noqa: F841  (placeholder; check happens per file)

    prog = net.Progress(total_bytes=total, total_items=len(keys))
    stop = threading.Event()

    def ticker() -> None:
        while not stop.wait(30.0):
            emit(prog.line())

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    try:
        digests = inat.download_bulk(keys, workers=args.workers, progress=prog)
    finally:
        stop.set()

    for k, d in digests.items():
        f = by_key[k]
        emit(f"DONE {f.name} sha256={d} size={f.local.stat().st_size}")
    emit("ALL DONE " + prog.line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
