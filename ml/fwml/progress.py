"""Persistent progress for long jobs, readable without the agent that started it.

The last big run's only progress record lived in a background shell owned by an
assistant session. When the session ended the run was still going and there was
no way to answer "how far along is it" except by watching a process. Every
expensive stage now writes a small JSON file here instead, updated in place, so
``python tools/status.py`` can answer that from a different terminal, tomorrow,
after a reboot.

Two files per stage:

``status/<stage>.json``  overwritten atomically every few seconds - counts,
                         rate, ETA, whether the stage is resumable and where its
                         durable checkpoint is.
``logs/<stage>.log``     append-only, line buffered, so a crash still leaves the
                         tail that explains it.

Writes are best effort. A status file that cannot be written must never take
down the job it is describing.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WORK = Path(r"D:\fisherwiki-data\work")


def work_dir() -> Path:
    """Resolved per call, not bound at import.

    A module-level constant reads ``FISHERWIKI_WORK`` once, at whichever moment
    the first import happens - so a test that sets it afterwards silently wrote
    its progress into the live ``D:`` status directory, where it then showed up
    in ``tools/status.py`` as a real run. Same shape as the ``PATHS`` binding
    bug that had tests writing into the production artifact store.
    """
    return Path(os.environ.get("FISHERWIKI_WORK") or DEFAULT_WORK)


def status_dir() -> Path:
    return work_dir() / "status"


def log_dir() -> Path:
    return work_dir() / "logs"

#: Don't rewrite the status file on every image; it is read by humans, not by
#: code, and a 500 MB/s NVMe still has better things to do 200 times a second.
MIN_WRITE_INTERVAL = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Progress:
    """Status writer for one stage of one pipeline.

    Usage is deliberately blunt::

        with Progress("prepare", total=479_017, resumable=True) as p:
            for ...:
                p.advance(failures=0)
            p.finish(note="...")
    """

    def __init__(
        self,
        stage: str,
        *,
        total: int | None = None,
        resumable: bool = True,
        already_done: int = 0,
        extra: dict | None = None,
        echo: bool = True,
    ) -> None:
        self.stage = stage
        self.total = total
        self.resumable = resumable
        self.already_done = int(already_done)
        self.extra = dict(extra or {})
        self.echo = echo
        self.processed = 0
        self.failures = 0
        self.state = "running"
        self.checkpoint: str | None = None
        self.note: str | None = None
        self.started = time.time()
        self._last_write = 0.0
        self._log_fh = None
        self.status_dir = status_dir()
        self.log_dir = log_dir()
        try:
            self.status_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_dir / f"{stage}.log", "a",
                                encoding="utf-8", buffering=1)
        except OSError:
            self._log_fh = None
        self.log(f"=== {stage} started {_now()} pid={os.getpid()} ===")
        self.write(force=True)

    # -- reporting ------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.echo:
            print(msg, flush=True)
        if self._log_fh is not None:
            try:
                self._log_fh.write(f"{_now()} {msg}\n")
            except (OSError, ValueError):
                pass

    def advance(self, n: int = 1, *, failures: int = 0) -> None:
        self.processed += n
        self.failures += failures
        self.write()

    def checkpointed(self, where: str) -> None:
        """Record the last point from which a resume would not lose work."""
        self.checkpoint = str(where)
        self.write(force=True)

    def snapshot(self) -> dict:
        elapsed = max(time.time() - self.started, 1e-9)
        rate = self.processed / elapsed
        remaining = None
        if self.total is not None:
            remaining = max(self.total - self.already_done - self.processed, 0)
        return {
            "stage": self.stage,
            "state": self.state,
            "pid": os.getpid(),
            "started_at": datetime.fromtimestamp(
                self.started, timezone.utc).isoformat(timespec="seconds"),
            "updated_at": _now(),
            "elapsed_s": round(elapsed, 1),
            "already_done": self.already_done,
            "processed": self.processed,
            "total": self.total,
            "remaining": remaining,
            "failures": self.failures,
            "rate_per_s": round(rate, 2),
            "eta_s": round(remaining / rate) if remaining and rate > 0 else None,
            "resumable": self.resumable,
            "last_checkpoint": self.checkpoint,
            "note": self.note,
            **self.extra,
        }

    def write(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < MIN_WRITE_INTERVAL:
            return
        self._last_write = now
        path = self.status_dir / f"{self.stage}.json"
        tmp = Path(str(path) + ".tmp")
        try:
            tmp.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    # -- lifecycle ------------------------------------------------------
    def finish(self, *, state: str = "done", note: str | None = None) -> None:
        self.state = state
        if note:
            self.note = note
        self.write(force=True)
        self.log(f"=== {self.stage} {state}: {self.processed:,} processed, "
                 f"{self.failures:,} failed ===")
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.state == "running":
            # An interrupted stage is recorded as interrupted rather than left
            # claiming to be running, so `tools/status.py` after a reboot says
            # what actually happened.
            self.finish(state="done" if exc_type is None else "interrupted",
                        note=None if exc_type is None else repr(exc)[:200])


def read_status(stage: str) -> dict | None:
    path = status_dir() / f"{stage}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def all_status() -> list[dict]:
    root = status_dir()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out
