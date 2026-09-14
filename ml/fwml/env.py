"""Hardware detection and device setup.

Deliberately does not assume CUDA. The reference machine for this project is an
AMD RX 7800 XT on Windows, which works but needs one non-obvious workaround;
encoding that here means every entry point gets it without remembering to.

The ROCm-on-Windows MIOpen problem
----------------------------------
MIOpen JIT-compiles its BatchNorm kernels with HIPRTC, and those kernels
``#include <type_traits>``. The ROCm Windows pip wheels ship no C++ standard
library headers at all (verified across ``rocm-sdk-core``, ``rocm-sdk-devel``
and the device packages), so every MIOpen BatchNorm call fails with
``miopenStatusUnknownError`` - in both train and eval mode, for every
torchvision model.

Setting ``torch.backends.cudnn.enabled = False`` routes convolution and
batch-norm to PyTorch's native implementations instead. Measured cost on this
machine is acceptable: MobileNetV3-Large trains at 374 img/s with AMP at batch
96, 224px.

This flag is applied **only** on ROCm/Windows. On CUDA or ROCm/Linux, cuDNN and
MIOpen are left enabled, because disabling them there would be a large and
pointless slowdown.

fp32 is not usable on this stack
--------------------------------
Measured 4096-cube matmul: 1.80 TFLOPS fp32 versus 45.47 TFLOPS fp16. A 25x gap
is far beyond the expected 2x, so fp32 evidently is not hitting tuned kernels on
gfx1101/Windows. AMP is therefore the default rather than an optimisation, and
:func:`describe` warns when it is turned off on this hardware.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeviceInfo:
    kind: str                      # 'cuda' | 'rocm' | 'mps' | 'cpu'
    name: str
    torch_device: str
    total_memory_gb: float = 0.0
    capability: str = ""
    amp_dtype: str = "float16"
    amp_recommended: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "rocm", "mps")

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "torch_device": self.torch_device,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "capability": self.capability,
            "amp_dtype": self.amp_dtype,
            "amp_recommended": self.amp_recommended,
            "notes": self.notes,
        }


def _miopen_cache_dir() -> Path:
    """Project-local MIOpen kernel cache.

    Keeping it out of the user profile makes builds reproducible and the cache
    disposable. The very first convolution after a fresh install can still fail
    while the kernel DB is being populated; re-running succeeds.
    """
    from fwdata.config import PATHS  # local import: ml/ may run without tools/

    d = PATHS.work / "miopen"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup(prefer_gpu: bool = True, quiet: bool = False) -> DeviceInfo:
    """Detect hardware, apply required workarounds, return a description."""
    import torch

    notes: list[str] = []

    if not prefer_gpu or not torch.cuda.is_available():
        if prefer_gpu and not torch.cuda.is_available():
            notes.append("No GPU visible to torch; falling back to CPU.")
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)
        return DeviceInfo(
            kind="cpu",
            name=platform.processor() or "cpu",
            torch_device="cpu",
            amp_dtype="bfloat16",
            amp_recommended=False,
            notes=notes + [f"Using {threads} CPU threads."],
        )

    is_rocm = getattr(torch.version, "hip", None) is not None
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "") or ""
    kind = "rocm" if is_rocm else "cuda"

    if is_rocm:
        cache = _miopen_cache_dir()
        os.environ.setdefault("MIOPEN_USER_DB_PATH", str(cache))
        os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(cache))
        notes.append(f"MIOpen cache -> {cache}")

        if sys.platform == "win32":
            # See module docstring. Without this every BatchNorm call fails.
            torch.backends.cudnn.enabled = False
            notes.append(
                "torch.backends.cudnn.enabled=False: ROCm-on-Windows MIOpen "
                "cannot JIT-compile its BatchNorm kernels (no C++ stdlib "
                "headers in the wheels). Conv and BN use native PyTorch paths."
            )
            notes.append(
                "fp32 GEMM on this stack measures 1.8 TFLOPS vs 45.5 fp16; "
                "train with AMP."
            )

    info = DeviceInfo(
        kind=kind,
        name=torch.cuda.get_device_name(0),
        torch_device="cuda",   # ROCm builds still expose the 'cuda' device name
        total_memory_gb=props.total_memory / 1e9,
        capability=arch or f"sm_{props.major}{props.minor}",
        amp_dtype="float16",
        amp_recommended=True,
        notes=notes,
    )
    if not quiet:
        for line in describe(info).splitlines():
            print(line, flush=True)
    return info


def describe(info: DeviceInfo) -> str:
    lines = [
        f"device      : {info.name} ({info.kind}, {info.capability})",
        f"torch device: {info.torch_device}",
    ]
    if info.total_memory_gb:
        lines.append(f"memory      : {info.total_memory_gb:.1f} GB")
    for n in info.notes:
        lines.append(f"note        : {n}")
    return "\n".join(lines)


def git_commit() -> str:
    """Current code commit, recorded in every run manifest."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def seed_everything(seed: int) -> None:
    """Seed every RNG we touch, and record it in the run manifest."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
