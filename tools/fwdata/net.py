"""Resumable, polite HTTP download primitives.

Every network access in this project goes through here so that the following
properties hold uniformly:

* **Resume** - partial files are kept as ``<name>.part`` and continued with a
  ``Range`` request when the server advertises ``Accept-Ranges: bytes``.
* **Retries** - exponential backoff with jitter on transient errors; permanent
  4xx (other than 408/429) fail immediately rather than looping.
* **Rate limits** - a shared token bucket per host, plus ``Retry-After``
  compliance on 429/503.
* **Timeouts** - separate connect/read timeouts; a stalled socket cannot hang
  the pipeline forever.
* **Checksums** - optional expected SHA-256, always-computed actual SHA-256.
* **Disk-space checks** - refuses to start a download that obviously cannot fit.
* **Identifiable User-Agent** - so that data providers can contact us rather
  than silently blocking.

Nothing here bypasses robots.txt, paywalls or provider restrictions; the only
endpoints used are official bulk/public ones documented in ``docs/DATASETS.md``.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

import requests

from .config import free_space_gb

USER_AGENT = (
    "FisherWiki-dataset-tool/0.1 (offline fish identification research project; "
    "+https://github.com/fisherwiki/fisherwiki)"
)

#: Status codes worth retrying. 429/503 additionally honour Retry-After.
RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504})


class DownloadError(RuntimeError):
    pass


class PermanentError(DownloadError):
    """A failure that retrying cannot fix (404, 403, unparseable response)."""


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------


class TokenBucket:
    """Thread-safe token bucket limiting requests-per-second."""

    def __init__(self, rate_per_sec: float, burst: float | None = None) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_sec))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self.rate if self.rate > 0 else 0.05
            time.sleep(min(wait, 1.0))


class HostLimiter:
    """Per-host token buckets with a configurable default rate."""

    def __init__(self, default_rate: float = 8.0) -> None:
        self.default_rate = default_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        #: Explicit, conservative per-host limits for providers that publish one.
        self.overrides: dict[str, float] = {
            "api.gbif.org": 4.0,
            "commons.wikimedia.org": 4.0,
            "upload.wikimedia.org": 8.0,
            "api.inaturalist.org": 1.0,
            # S3 static object storage on the AWS Open Data registry:
            # sponsored precisely for bulk ML reads and sized for far more
            # than this. 100 req/s of ~130 KB objects is ~13 MB/s, which is
            # negligible for S3 while keeping us well-behaved.
            "inaturalist-open-data.s3.amazonaws.com": 100.0,
        }

    def bucket(self, host: str) -> TokenBucket:
        with self._lock:
            b = self._buckets.get(host)
            if b is None:
                rate = self.overrides.get(host, self.default_rate)
                b = TokenBucket(rate)
                self._buckets[host] = b
            return b

    def acquire(self, url: str) -> None:
        self.bucket(urlsplit(url).netloc.lower()).acquire()


LIMITER = HostLimiter()


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

_local = threading.local()


def session() -> requests.Session:
    """One :class:`requests.Session` per thread (connection pooling, not shared)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32, pool_maxsize=32, max_retries=0
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


@dataclass
class Progress:
    """Lightweight progress accounting shared across download workers."""

    total_bytes: int = 0
    done_bytes: int = 0
    total_items: int = 0
    done_items: int = 0
    failed_items: int = 0
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_bytes(self, n: int) -> None:
        with self._lock:
            self.done_bytes += n

    def item_done(self, ok: bool = True) -> None:
        with self._lock:
            self.done_items += 1
            if not ok:
                self.failed_items += 1

    @property
    def rate_mbps(self) -> float:
        el = max(1e-6, time.monotonic() - self.started)
        return self.done_bytes / el / 1e6

    def line(self) -> str:
        el = time.monotonic() - self.started
        pct = (100.0 * self.done_bytes / self.total_bytes) if self.total_bytes else 0.0
        eta = ""
        if self.total_bytes and self.done_bytes:
            remain = (self.total_bytes - self.done_bytes) / max(
                1.0, self.done_bytes / max(1e-6, el)
            )
            eta = f" eta {remain/60:.1f}m"
        items = (
            f" {self.done_items}/{self.total_items} items"
            if self.total_items
            else f" {self.done_items} items"
        )
        fail = f" ({self.failed_items} failed)" if self.failed_items else ""
        return (
            f"{self.done_bytes/1e9:.2f}/{self.total_bytes/1e9:.2f} GB {pct:5.1f}% "
            f"{self.rate_mbps:6.1f} MB/s{eta}{items}{fail}"
        )


def _sleep_backoff(attempt: int, retry_after: float | None) -> None:
    if retry_after is not None:
        time.sleep(min(retry_after, 120.0))
        return
    base = min(60.0, 1.5 * (2 ** attempt))
    time.sleep(base * (0.5 + random.random()))


def _retry_after_seconds(resp: requests.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return 30.0


def head(url: str, timeout: tuple[float, float] = (10, 30)) -> requests.Response:
    LIMITER.acquire(url)
    return session().head(url, timeout=timeout, allow_redirects=True)


def content_length(url: str) -> int | None:
    try:
        r = head(url)
        if r.status_code != 200:
            return None
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    max_attempts: int = 6,
    timeout: tuple[float, float] = (15, 120),
    progress: Progress | None = None,
    chunk: int = 1 << 20,
    min_free_gb: float = 2.0,
    on_tick: Callable[[], None] | None = None,
) -> str:
    """Download ``url`` to ``dest``, resuming a ``.part`` file if present.

    Returns the hex SHA-256 of the completed file.  Raises
    :class:`PermanentError` for non-retryable HTTP failures.

    The SHA-256 is computed incrementally *including* previously downloaded
    bytes when resuming, so a resumed download still yields a correct digest
    without a second pass over the file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        if expected_size is not None and dest.stat().st_size != expected_size:
            dest.unlink()
        else:
            digest = sha256_file(dest)
            if expected_sha256 and digest != expected_sha256:
                dest.unlink()
            else:
                if progress:
                    progress.add_bytes(dest.stat().st_size)
                    progress.item_done(True)
                return digest

    need = (expected_size or 0) / 1e9 + min_free_gb
    if free_space_gb(dest.parent) < need:
        raise DownloadError(
            f"Not enough free space for {dest.name}: need ~{need:.1f} GB, "
            f"have {free_space_gb(dest.parent):.1f} GB"
        )

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        have = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            LIMITER.acquire(url)
            with session().get(
                url, headers=headers, stream=True, timeout=timeout
            ) as resp:
                if resp.status_code in (403, 404, 410):
                    raise PermanentError(f"HTTP {resp.status_code} for {url}")
                if resp.status_code in RETRYABLE:
                    _sleep_backoff(attempt, _retry_after_seconds(resp))
                    continue
                if have and resp.status_code == 200:
                    # Server ignored Range: restart cleanly.
                    part.unlink(missing_ok=True)
                    have = 0
                elif have and resp.status_code != 206:
                    raise DownloadError(
                        f"Unexpected status {resp.status_code} resuming {url}"
                    )
                elif not have and resp.status_code != 200:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}")

                mode = "ab" if have else "wb"
                written = 0
                with open(part, mode) as fh:
                    # NOTE: decode_content=False is load-bearing, not an
                    # optimisation. S3 serves objects such as
                    # ``taxa.csv.gz`` with a stored ``Content-Encoding: gzip``
                    # header, so requests/urllib3 would transparently gunzip the
                    # body. That breaks three things at once: the bytes no
                    # longer match Content-Length, the SHA-256 no longer matches
                    # the published object, and - worst - a ``Range`` resume
                    # offsets into the *compressed* object while we would have
                    # written *decompressed* bytes, silently corrupting the
                    # file. We always persist the exact stored bytes.
                    for block in resp.raw.stream(chunk, decode_content=False):
                        if not block:
                            continue
                        fh.write(block)
                        written += len(block)
                        if progress:
                            progress.add_bytes(len(block))
                        if on_tick:
                            on_tick()
            # Completed a full body without exception.
            digest = sha256_file(part)
            if expected_sha256 and digest != expected_sha256:
                part.unlink(missing_ok=True)
                raise DownloadError(
                    f"Checksum mismatch for {url}: got {digest}, want {expected_sha256}"
                )
            if expected_size is not None and part.stat().st_size != expected_size:
                got = part.stat().st_size
                part.unlink(missing_ok=True)
                raise DownloadError(
                    f"Size mismatch for {url}: got {got}, want {expected_size}"
                )
            os.replace(part, dest)
            if progress:
                progress.item_done(True)
            return digest
        except PermanentError:
            if progress:
                progress.item_done(False)
            raise
        except (requests.RequestException, DownloadError, OSError) as exc:
            last_exc = exc
            _sleep_backoff(attempt, None)

    if progress:
        progress.item_done(False)
    raise DownloadError(f"Giving up on {url} after {max_attempts} attempts: {last_exc}")


def download_file_parallel(
    url: str,
    dest: Path,
    *,
    workers: int = 8,
    part_size: int = 64 << 20,
    progress: Progress | None = None,
    timeout: tuple[float, float] = (15, 180),
    max_attempts: int = 6,
    min_free_gb: float = 5.0,
) -> str:
    """Download one large object using concurrent HTTP range requests.

    Used only for the multi-gigabyte bulk dumps on static object storage, where
    a single TCP stream is the bottleneck rather than the provider.  Each chunk
    is written to its own ``.partNNN`` file so that an interrupted run resumes
    at chunk granularity; completed chunks are never re-fetched.

    Falls back to the sequential :func:`download_file` when the server does not
    advertise byte ranges or does not report a length.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if progress:
            progress.add_bytes(dest.stat().st_size)
        return sha256_file(dest)

    try:
        h = head(url, timeout=(10, 60))
        total = int(h.headers.get("Content-Length", 0))
        ranges_ok = h.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        total, ranges_ok = 0, False

    if not total or not ranges_ok:
        return download_file(
            url, dest, progress=progress, timeout=timeout, max_attempts=max_attempts
        )

    require_free = total / 1e9 + min_free_gb
    if free_space_gb(dest.parent) < require_free:
        raise DownloadError(
            f"Not enough free space for {dest.name}: need ~{require_free:.1f} GB, "
            f"have {free_space_gb(dest.parent):.1f} GB"
        )

    chunks: list[tuple[int, int, int]] = []
    idx = 0
    for start in range(0, total, part_size):
        end = min(start + part_size, total) - 1
        chunks.append((idx, start, end))
        idx += 1

    workdir = dest.parent / f".{dest.name}.parts"
    workdir.mkdir(parents=True, exist_ok=True)

    def fetch_chunk(spec: tuple[int, int, int]) -> None:
        i, start, end = spec
        want = end - start + 1
        cpath = workdir / f"part{i:05d}"
        if cpath.exists() and cpath.stat().st_size == want:
            if progress:
                progress.add_bytes(want)
            return
        last: Exception | None = None
        for attempt in range(max_attempts):
            try:
                LIMITER.acquire(url)
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                with session().get(
                    url, headers=headers, stream=True, timeout=timeout
                ) as resp:
                    if resp.status_code in (403, 404, 410):
                        raise PermanentError(f"HTTP {resp.status_code} for {url}")
                    if resp.status_code in RETRYABLE:
                        _sleep_backoff(attempt, _retry_after_seconds(resp))
                        continue
                    if resp.status_code != 206:
                        raise DownloadError(
                            f"Range request returned {resp.status_code} for {url}"
                        )
                    got = 0
                    tmp = cpath.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        for block in resp.raw.stream(1 << 20, decode_content=False):
                            if not block:
                                continue
                            fh.write(block)
                            got += len(block)
                            if progress:
                                progress.add_bytes(len(block))
                    if got != want:
                        if progress:
                            progress.add_bytes(-got)
                        tmp.unlink(missing_ok=True)
                        raise DownloadError(f"Short chunk {i}: {got} != {want}")
                    os.replace(tmp, cpath)
                    return
            except PermanentError:
                raise
            except (requests.RequestException, DownloadError, OSError) as exc:
                last = exc
                _sleep_backoff(attempt, None)
        raise DownloadError(f"Chunk {i} of {url} failed: {last}")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch_chunk, chunks))

    # Concatenate in order, hashing as we go.
    h256 = hashlib.sha256()
    tmp_out = dest.with_suffix(dest.suffix + ".assembling")
    with open(tmp_out, "wb") as out:
        for i, _, _ in chunks:
            cpath = workdir / f"part{i:05d}"
            with open(cpath, "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    h256.update(b)
                    out.write(b)
    if tmp_out.stat().st_size != total:
        raise DownloadError(
            f"Assembled size {tmp_out.stat().st_size} != expected {total} for {url}"
        )
    os.replace(tmp_out, dest)
    for i, _, _ in chunks:
        (workdir / f"part{i:05d}").unlink(missing_ok=True)
    try:
        workdir.rmdir()
    except OSError:
        pass
    if progress:
        progress.item_done(True)
    return h256.hexdigest()


def fetch_bytes(
    url: str,
    *,
    max_attempts: int = 5,
    timeout: tuple[float, float] = (10, 60),
    accept: str | None = None,
) -> bytes:
    """GET a small resource into memory with the same retry policy."""
    headers = {"Accept": accept} if accept else {}
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            LIMITER.acquire(url)
            resp = session().get(url, headers=headers, timeout=timeout)
            if resp.status_code in (403, 404, 410):
                raise PermanentError(f"HTTP {resp.status_code} for {url}")
            if resp.status_code in RETRYABLE:
                _sleep_backoff(attempt, _retry_after_seconds(resp))
                continue
            resp.raise_for_status()
            return resp.content
        except PermanentError:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            _sleep_backoff(attempt, None)
    raise DownloadError(f"Giving up on {url}: {last_exc}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_many(
    items: Iterable[tuple[str, Path]],
    *,
    workers: int = 8,
    progress: Progress | None = None,
    **kwargs,
) -> dict[str, str | None]:
    """Download many URLs concurrently. Returns ``{url: sha256 or None}``."""
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if progress is not None:
        progress.total_items = progress.total_items or len(items)
    results: dict[str, str | None] = {}

    def one(pair: tuple[str, Path]) -> tuple[str, str | None]:
        url, dest = pair
        try:
            return url, download_file(url, dest, progress=progress, **kwargs)
        except DownloadError:
            return url, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for url, digest in ex.map(one, items):
            results[url] = digest
    return results
