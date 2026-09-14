"""Filesystem layout and runtime configuration.

Large artefacts (raw dumps, images, checkpoints) live *outside* the git
repository.  The location is resolved in this order:

1. ``FISHERWIKI_DATA`` environment variable.
2. ``data_root`` in ``<repo>/fisherwiki.toml`` if present.
3. Platform default (``D:/fisherwiki-data`` on Windows when D: exists,
   otherwise ``<repo>/.data``).

Only manifests, hashes and small metadata are ever written back into the repo.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    env = os.environ.get("FISHERWIKI_DATA")
    if env:
        return Path(env).expanduser().resolve()

    toml_path = REPO_ROOT / "fisherwiki.toml"
    if toml_path.exists():
        try:
            import tomllib

            cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            root = cfg.get("paths", {}).get("data_root")
            if root:
                return Path(root).expanduser().resolve()
        except Exception:  # pragma: no cover - config is advisory only
            pass

    if sys.platform == "win32":
        for drive in ("D:", "E:"):
            if Path(drive + "/").exists():
                return Path(drive + "/fisherwiki-data")
    return REPO_ROOT / ".data"


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def cas(self) -> Path:
        """Content-addressed image store: cas/<aa>/<bb>/<sha256>.<ext>."""
        return self.root / "cas"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def provenance_db(self) -> Path:
        return self.root / "provenance.duckdb"

    def raw_source(self, source: str) -> Path:
        p = self.raw / source
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cas_path(self, sha256: str, ext: str) -> Path:
        ext = ext.lower().lstrip(".")
        return self.cas / sha256[:2] / sha256[2:4] / f"{sha256}.{ext}"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.cas, self.work, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths(_default_data_root())


def free_space_gb(path: Path) -> float:
    """Free space in GB on the volume containing ``path`` (walks up if needed)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def require_free_space(path: Path, need_gb: float) -> None:
    have = free_space_gb(path)
    if have < need_gb:
        raise RuntimeError(
            f"Insufficient disk space at {path}: need ~{need_gb:.1f} GB, have {have:.1f} GB"
        )
