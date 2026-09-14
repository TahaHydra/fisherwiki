#!/usr/bin/env python
"""Run the V2 ingest stages in order, unattended, and checkpoint after each.

    python tools/v2_pipeline.py --after-pid 1234
    python tools/v2_pipeline.py --stages assign,detect,prepare

Stages are strictly sequential and each one is separately resumable. That is not
a stylistic choice: **DuckDB takes an exclusive file lock**, so acquisition,
split assignment, detection and preparation cannot touch the live provenance
database at the same time. Running them concurrently does not corrupt anything -
it simply fails to open the file - but it does mean the pipeline has to be a
chain rather than a fan-out.

Every stage is safe to re-run. `assign` only ever adds rows for images it has
not seen, `detect` skips images that already have a detection, and `prepare`
rebuilds shards from whatever is currently assigned. So an interrupted pipeline
is restarted by running it again, with no cleanup and no lost work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
PY_TRAIN = REPO / ".venv-train" / "Scripts" / "python.exe"
WORK = Path(r"D:\fisherwiki-data\work")
STATE = WORK / "v2_pipeline_state.json"


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": [], "history": []}


def _save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def _run(cmd: list[str], label: str, state: dict) -> bool:
    _log(f"START {label}: {' '.join(str(c) for c in cmd[1:4])} ...")
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=REPO)
    dt = round(time.time() - t0, 1)
    ok = proc.returncode == 0
    state["history"].append({
        "stage": label, "returncode": proc.returncode, "seconds": dt,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    if ok and label not in state["completed"]:
        state["completed"].append(label)
    _save_state(state)
    _log(f"{'DONE ' if ok else 'FAIL '} {label} in {dt}s (rc={proc.returncode})")
    return ok


def wait_for_pid(pid: int, poll: float = 20.0) -> None:
    """Block until a still-running acquisition finishes and frees the DB lock."""
    import os

    _log(f"waiting for pid {pid} to finish (it holds the database lock)")
    while True:
        try:
            if sys.platform == "win32":
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True,
                )
                if str(pid) not in out.stdout:
                    break
            else:
                os.kill(pid, 0)
        except Exception:
            break
        time.sleep(poll)
    _log(f"pid {pid} has exited")


STAGES = ("assign", "verify", "detect", "prepare")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after-pid", type=int, default=None,
                    help="wait for this process to exit before starting")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--batch", default="batch1_inat_topup")
    ap.add_argument("--shards-out", default=r"D:\fisherwiki-data\v2\shards")
    ap.add_argument("--prepare-limit", type=int, default=None)
    args = ap.parse_args(argv)

    WORK.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    if args.after_pid:
        wait_for_pid(args.after_pid)

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]

    if "assign" in wanted:
        if not _run([PY, "tools/dataset.py", "v2-assign", "--batch", args.batch],
                    f"assign:{args.batch}", state):
            _log("assignment failed - stopping rather than preparing a partial corpus")
            return 1

    if "verify" in wanted:
        if not _run([PY, "tools/dataset.py", "v2-verify"], "verify", state):
            _log("VERIFICATION FAILED - refusing to continue")
            return 1

    if "detect" in wanted:
        # Not fatal: preparation falls back to whole frames for anything without
        # a box, so a detector that dies partway still leaves a usable corpus.
        _run([PY_TRAIN, "tools/detect_fish.py", "--batch", "32"], "detect", state)

    if "prepare" in wanted:
        cmd = [PY_TRAIN, "tools/prepare_shards_v2.py", "--out", args.shards_out]
        if args.prepare_limit:
            cmd += ["--limit", str(args.prepare_limit)]
        if not _run(cmd, "prepare", state):
            return 1

    _log("pipeline complete")
    _log(f"stages completed: {state['completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
