#!/usr/bin/env python3
"""yt-dlp.queue.py – Interactive yt-dlp queue"""

from __future__ import annotations

import cmd
import hashlib
import itertools
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import codecs
import re
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # Fallback when colorama not available
    class _DummyColor:
        def __getattr__(self, name):
            return ""

    Fore = Style = _DummyColor()
    HAS_COLOR = False

# ---------------------------------------------------------------------------
# yt-dlp auto-update helpers
# ---------------------------------------------------------------------------
UPDATE_STATE_FILE = Path("yt_dlp_update_state.json")
UPDATE_CHECK_INTERVAL = timedelta(days=1)
REPO_API_URL = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
DOWNLOAD_BASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/download/"
DEFAULT_BINARY_NAME = "yt-dlp.exe" if sys.platform.startswith("win") else "yt-dlp"

# Strip ANSI color codes so log pattern matching remains stable even when
# yt-dlp writes colored output.
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
FILENAME_SAFE_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")
PROGRESS_LINE_RE = re.compile(r"^\[download\]\s+\d{1,3}\.\d")
BASE_DIR = Path(__file__).resolve().parent

EXTERNAL_TOOL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "ffmpeg": ("ffmpeg",),
    "ffprobe": ("ffprobe",),
    "yt-dlp-ejs": ("yt-dlp-ejs",),
    "AtomicParsley": ("AtomicParsley", "atomicparsley"),
    "aria2c": ("aria2c",),
    "rtmpdump": ("rtmpdump",),
}

JS_RUNTIME_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "deno": ("deno",),
    "node": ("node",),
    "quickjs": ("quickjs", "qjs"),
    "quickjs-ng": ("quickjs-ng",),
    "bun": ("bun",),
}

EXTERNAL_TOOL_ORDER = tuple(EXTERNAL_TOOL_CANDIDATES.keys())
CRITICAL_TOOLS = {"ffmpeg", "ffprobe", "yt-dlp-ejs"}
JS_RUNTIME_PRIORITY = tuple(JS_RUNTIME_CANDIDATES.keys())
TOOL_NOTES: Dict[str, str] = {
    "ffmpeg": "merging + post-processing",
    "ffprobe": "stream inspection",
    "yt-dlp-ejs": "YouTube signature support",
    "AtomicParsley": "thumbnail embedding for mp4/m4a",
    "aria2c": "external segmented downloader",
    "rtmpdump": "legacy RTMP streams",
}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}


def _iter_platform_candidates(name: str) -> Iterable[str]:
    yield name
    if sys.platform.startswith("win") and not name.lower().endswith(".exe"):
        yield f"{name}.exe"


def _resolve_tool_candidate(candidate: str) -> Optional[Path]:
    local = BASE_DIR / candidate
    if local.is_file():
        return local
    found = shutil.which(candidate)
    if found:
        return Path(found)
    return None


def find_tool(names: Iterable[str]) -> Optional[Path]:
    seen: Set[str] = set()
    for raw_name in names:
        for candidate in _iter_platform_candidates(raw_name):
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            resolved = _resolve_tool_candidate(candidate)
            if resolved:
                return resolved
    return None


def detect_external_tools() -> Dict[str, Optional[Path]]:
    return {
        label: find_tool(names) for label, names in EXTERNAL_TOOL_CANDIDATES.items()
    }


def choose_js_runtime() -> Tuple[Optional[str], Optional[Path]]:
    for runtime in JS_RUNTIME_PRIORITY:
        aliases = JS_RUNTIME_CANDIDATES.get(runtime, (runtime,))
        path = find_tool(aliases)
        if path:
            return runtime, path
    return None, None


def _path_matches_system(candidate: str, path: Path) -> bool:
    located = shutil.which(candidate)
    if not located:
        return False
    try:
        return Path(located).resolve() == path.resolve()
    except OSError:
        return False


def is_youtube_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")


def _sanitize_filename(value: str, max_length: int = 120) -> str:
    slug = FILENAME_SAFE_PATTERN.sub("_", value)
    slug = re.sub(r"_+", "_", slug).strip("._")
    if not slug:
        slug = "url"
    if len(slug) > max_length:
        suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        prefix_len = max(1, max_length - len(suffix) - 1)
        slug = f"{slug[:prefix_len]}_{suffix}"
    return slug


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


class UpdateError(Exception):
    """Raised when the updater cannot complete its work."""


class UpdateRateLimit(UpdateError):
    """Raised when the updater is rate limited by the remote service."""


class UpdatePermissionError(UpdateError):
    """Raised when the updater cannot write to the target binary location."""


def _get_remote_version() -> str:
    """Return the latest yt-dlp release tag."""

    req = Request(REPO_API_URL, headers={"User-Agent": "yt-dlp-queue/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    tag = data.get("tag_name")
    if not tag:
        raise UpdateError("GitHub API response missing tag_name")
    return tag


def _get_local_version(binary_path: Path) -> Optional[str]:
    """Inspect a yt-dlp binary for its '--version' output."""

    if not binary_path.exists():
        return None

    try:
        output = subprocess.check_output(
            [str(binary_path), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    return output


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to the given destination path."""

    req = Request(url, headers={"User-Agent": "yt-dlp-queue/1.0"})
    with urlopen(req, timeout=60) as resp, open(dest, "wb") as handle:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            handle.write(chunk)


def update_yt_dlp(
    base_dir: Path | str | None = None,
    binary_name: str = DEFAULT_BINARY_NAME,
    remove_old: bool = True,
    *,
    verbose: bool = True,
) -> Tuple[Optional[str], str, bool]:
    """Check for a newer yt-dlp release and download it when needed."""

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir)

    binary_path = base_dir / binary_name
    old_binary_path = binary_path.with_suffix(binary_path.suffix + ".old")

    if not binary_path.exists() and verbose:
        print(f"{binary_name} does not exist.")

    local_version = _get_local_version(binary_path)
    remote_version = _get_remote_version()

    if local_version is not None and local_version == remote_version:
        if verbose:
            print(f"{binary_name} is up to date ({local_version}).")
        return local_version, remote_version, False

    if local_version is None:
        if verbose:
            print(f"{binary_name}: remote {remote_version}, local version not found.")
    else:
        if verbose:
            print(
                f"{binary_name}: remote {remote_version} <> {local_version} --> local is outdated."
            )

    if remove_old:
        if old_binary_path.exists():
            old_binary_path.unlink()
    else:
        if binary_path.exists():
            if old_binary_path.exists():
                old_binary_path.unlink()
            binary_path.rename(old_binary_path)

    download_url = f"{DOWNLOAD_BASE_URL}{remote_version}/{binary_name}"
    if verbose:
        print(f"Downloading new version from {download_url} ...")
    _download_file(download_url, binary_path)

    try:
        binary_path.chmod(binary_path.stat().st_mode | 0o111)
    except PermissionError:
        pass

    new_version = _get_local_version(binary_path)
    if verbose:
        print(f"Updated to version: {new_version}")
    return new_version, remote_version, True


def _read_update_state() -> Optional[datetime]:
    try:
        with open(UPDATE_STATE_FILE) as handle:
            payload = json.load(handle)
        stamp = payload.get("last_check")
        if stamp:
            return datetime.fromisoformat(stamp)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def _write_update_state(timestamp: datetime, **extra) -> None:
    payload = {"last_check": timestamp.isoformat()}
    payload.update(extra)
    try:
        with open(UPDATE_STATE_FILE, "w") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.workers = 3
        self.max_retries = 10
        self.retry_delay = 60  # seconds
        self.download_dir = None
        self.yt_dlp_path = None
        self.default_args = []
        self.auto_save = True
        self.session_file = "yt_dlp_session.json"
        self.config_file = "yt_dlp_config.json"
        self.auto_status = False
        self.flat_directories = True
        self.directory_template: Optional[str] = None
        self.keep_failed = True
        self.dump_json = False

    def to_dict(self) -> dict:
        return {
            "workers": self.workers,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "download_dir": str(self.download_dir) if self.download_dir else None,
            "yt_dlp_path": str(self.yt_dlp_path) if self.yt_dlp_path else None,
            "default_args": self.default_args,
            "auto_save": self.auto_save,
            "session_file": self.session_file,
            "auto_status": self.auto_status,
            "flat_directories": self.flat_directories,
            "directory_template": self.directory_template,
            "keep_failed": self.keep_failed,
            "dump_json": self.dump_json,
        }

    def from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


def _resolve_yt_dlp_binary(config: "Config") -> Optional[Path]:
    if config.yt_dlp_path:
        return Path(config.yt_dlp_path).expanduser()

    candidate = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if candidate:
        return Path(candidate)
    return None


def maybe_check_yt_dlp_update(
    config: "Config",
) -> Optional[Tuple[Optional[str], str, bool]]:
    now = datetime.now()
    last_check = _read_update_state()
    if last_check and now - last_check < UPDATE_CHECK_INTERVAL:
        return None

    binary_path = _resolve_yt_dlp_binary(config)
    if binary_path is None:
        _write_update_state(now, status="binary-not-found")
        return None

    target_for_access = binary_path if binary_path.exists() else binary_path.parent
    if target_for_access and not os.access(target_for_access, os.W_OK):
        _write_update_state(
            now,
            status="no-permission",
            path=str(binary_path),
        )
        raise UpdatePermissionError(
            f"Cannot update yt-dlp at {binary_path} (insufficient permissions)"
        )

    base_dir = binary_path.parent
    binary_name = binary_path.name or DEFAULT_BINARY_NAME

    try:
        result = update_yt_dlp(
            base_dir=base_dir,
            binary_name=binary_name,
            verbose=False,
        )
    except HTTPError as exc:
        status = "rate-limited" if exc.code in {403, 429} else "http-error"
        _write_update_state(
            now,
            status=status,
            code=exc.code,
            reason=getattr(exc, "reason", ""),
        )
        if status == "rate-limited":
            raise UpdateRateLimit(f"GitHub API rate limited (HTTP {exc.code})") from exc
        raise UpdateError(f"GitHub API error (HTTP {exc.code}: {exc.reason})") from exc
    except UpdateError as exc:
        _write_update_state(now, status="error", message=str(exc))
        raise
    except Exception as exc:
        _write_update_state(now, status="error", message=str(exc))
        raise UpdateError(str(exc)) from exc

    local_version, remote_version, updated = result
    _write_update_state(
        now,
        status="updated" if updated else "current",
        local_version=local_version,
        remote_version=remote_version,
    )
    return result


class PendingEntry(NamedTuple):
    url: str
    directory: Optional[Path]


class FailedEntry(NamedTuple):
    url: str
    directory: Optional[Path]


class WorkerLog:
    """Ring buffer that keeps a bounded stream of worker output."""

    def __init__(self, maxlen: int = 2000):
        self.buffer: deque[str] = deque()
        self.maxlen = maxlen
        self.offset = 0

    def append(self, message: str):
        if len(self.buffer) >= self.maxlen:
            self.buffer.popleft()
            self.offset += 1
        self.buffer.append(message)

    def get_since(self, index: int) -> Tuple[List[str], int, bool]:
        """Return entries after index, next index, and truncation flag."""
        truncated = index < self.offset
        start = max(index, self.offset)
        skip = start - self.offset
        lines = list(itertools.islice(self.buffer, skip, None))
        next_index = self.offset + len(self.buffer)
        return lines, next_index, truncated

    def snapshot(self) -> Tuple[List[str], int, int]:
        """Return a copy of the buffer and index bounds."""
        buf_copy = list(self.buffer)
        start_index = self.offset
        end_index = self.offset + len(self.buffer)
        return buf_copy, start_index, end_index


# ---------------------------------------------------------------------------
# Download tracking and statistics
# ---------------------------------------------------------------------------
class DownloadStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.retried = 0
        self.total_time = 0.0
        self.history: deque = deque(maxlen=1000)  # Keep last 1000 entries
        self.current_downloads: Dict[int, Tuple[str, Optional[Path]]] = {}

    def add_completion(self, url: str, success: bool, duration: float, worker_id: int):
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            self.total_time += duration
            self.history.append(
                {
                    "url": url,
                    "success": success,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "worker_id": worker_id,
                }
            )

            # Remove from active downloads
            self.current_downloads.pop(worker_id, None)

    def set_current_download(self, worker_id: int, url: str, directory: Optional[Path]):
        with self._lock:
            self.current_downloads[worker_id] = (url, directory)

    def record_retry(self):
        with self._lock:
            self.retried += 1

    def snapshot_active(self) -> Dict[int, Tuple[str, Optional[Path]]]:
        with self._lock:
            return dict(self.current_downloads)

    def has_active_downloads(self) -> bool:
        with self._lock:
            return bool(self.current_downloads)

    def is_worker_active(self, worker_id: int) -> bool:
        with self._lock:
            return worker_id in self.current_downloads

    def history_snapshot(self) -> List[dict]:
        with self._lock:
            return list(self.history)

    def get_summary(self) -> str:
        with self._lock:
            total = self.completed + self.failed
            if total == 0:
                return "No downloads completed yet"

            success_rate = (self.completed / total) * 100
            avg_time = self.total_time / total if total > 0 else 0
            fail_color = Fore.RED if self.failed else Fore.WHITE

            return (
                f"{Fore.WHITE}STATS: {total} downloads "
                f"({Fore.GREEN}{self.completed} success{Fore.WHITE}, "
                f"{fail_color}{self.failed} failed{Fore.WHITE}) - "
                f"{success_rate:.1f}% success rate, avg {avg_time:.1f}s per download"
                f"{Style.RESET_ALL}"
            )


# ---------------------------------------------------------------------------
# Threaded downloader
# ---------------------------------------------------------------------------
class YTDLPQueue:
    def __init__(
        self,
        config: Config,
        stats: DownloadStats,
        failed_seed: Optional[List[FailedEntry]] = None,
    ):
        self.config = config
        self.stats = stats

        # Find yt-dlp executable
        binary_path = _resolve_yt_dlp_binary(config)
        if not binary_path:
            raise RuntimeError(
                "yt-dlp executable not found, install it or set yt_dlp_path"
            )
        self.exe = str(binary_path)
        self.tool_status = detect_external_tools()
        self.js_runtime_choice = choose_js_runtime()
        self.js_runtime_arg = self._build_js_runtime_arg()

        # Thread-safe structures
        self.q: "queue.Queue[Tuple[str, int, Optional[Path]]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.paused_event = threading.Event()

        # Worker management
        self.threads: List[threading.Thread] = []
        self.next_worker_id = 0
        self._create_workers()

        # URL tracking
        self.queued_urls: Set[str] = set()
        self.completed_urls: Set[str] = set()

        # Worker output tracking
        self.worker_logs: Dict[int, WorkerLog] = {}
        self.worker_log_lock = threading.Lock()
        self.worker_log_events: Dict[int, threading.Event] = {}
        self.worker_processes: Dict[int, Optional[subprocess.Popen]] = {}
        self.worker_process_lock = threading.Lock()
        self.worker_download_dirs: Dict[int, Optional[Path]] = {}
        self.worker_dir_lock = threading.Lock()
        self.worker_progress: Dict[int, str] = {}
        self.worker_progress_lock = threading.Lock()
        self.failed_downloads_lock = threading.Lock()
        self.failed_downloads: Dict[str, FailedEntry] = {}
        self.worker_active_urls: Dict[int, str] = {}
        self.worker_active_lock = threading.Lock()
        self.unsupported_lock = threading.Lock()
        self.unsupported_details: Dict[str, str] = {}
        self.idle_callback: Optional[Callable[[], None]] = None
        self.multi_active_since_idle = False
        if failed_seed:
            self.import_failed_entries(failed_seed, replace=True)

    def _create_workers(self):
        """Create worker threads based on current config."""
        for i in range(self.config.workers):
            worker_id = self.next_worker_id
            self.next_worker_id += 1

            thread = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                daemon=True,
                name=f"ytdlp-worker-{worker_id}",
            )
            thread.start()
            self.threads.append(thread)

    def set_idle_callback(self, callback: Optional[Callable[[], None]]) -> None:
        self.idle_callback = callback

    def _default_args_have_js_runtime(self) -> bool:
        for value in self.config.default_args:
            if value == "--js-runtimes" or value.startswith("--js-runtimes="):
                return True
        return False

    def _build_js_runtime_arg(self) -> Optional[List[str]]:
        name, path = self.js_runtime_choice
        if not name or self._default_args_have_js_runtime():
            return None

        if name == "deno" and path and _path_matches_system("deno", path):
            return None

        runtime_value = name
        if path:
            runtime_value = f"{name}={path}"

        return ["--js-runtimes", runtime_value]

    # Worker tracking --------------------------------------------------
    def _ensure_worker_tracking(self, worker_id: int):
        with self.worker_log_lock:
            if worker_id not in self.worker_logs:
                self.worker_logs[worker_id] = WorkerLog()
                self.worker_log_events[worker_id] = threading.Event()
        self._ensure_worker_dir_entry(worker_id)

    def _ensure_worker_dir_entry(self, worker_id: int):
        with self.worker_dir_lock:
            if worker_id not in self.worker_download_dirs:
                self.worker_download_dirs[worker_id] = None

    def _record_worker_output(self, worker_id: int, message: str):
        self._ensure_worker_tracking(worker_id)
        with self.worker_log_lock:
            self.worker_logs[worker_id].append(message)
        self.worker_log_events[worker_id].set()

    def _maybe_notify_all_idle(self) -> None:
        if self.multi_active_since_idle and not self.stats.has_active_downloads():
            self.multi_active_since_idle = False
            if self.idle_callback:
                try:
                    self.idle_callback()
                except Exception:
                    pass

    def _set_worker_active_url(self, worker_id: int, url: Optional[str]) -> None:
        with self.worker_active_lock:
            if url is None:
                self.worker_active_urls.pop(worker_id, None)
            else:
                self.worker_active_urls[worker_id] = url

    def _get_worker_active_url(self, worker_id: int) -> Optional[str]:
        with self.worker_active_lock:
            return self.worker_active_urls.get(worker_id)

    def _flag_unsupported(self, url: str, detail: str):
        if not url:
            return
        with self.unsupported_lock:
            self.unsupported_details[url] = detail

    def _consume_unsupported(self, url: str) -> Optional[str]:
        if not url:
            return None
        with self.unsupported_lock:
            return self.unsupported_details.pop(url, None)

    def _clear_unsupported(self, url: str) -> None:
        if not url:
            return
        with self.unsupported_lock:
            self.unsupported_details.pop(url, None)

    def _set_worker_progress(self, worker_id: int, text: str) -> None:
        self._ensure_worker_tracking(worker_id)
        with self.worker_progress_lock:
            self.worker_progress[worker_id] = text
        self.worker_log_events[worker_id].set()

    def clear_worker_progress(self, worker_id: int) -> None:
        self._ensure_worker_tracking(worker_id)
        with self.worker_progress_lock:
            removed = self.worker_progress.pop(worker_id, None)
        if removed is not None:
            self.worker_log_events[worker_id].set()

    def get_worker_progress(self, worker_id: int) -> Optional[str]:
        with self.worker_progress_lock:
            return self.worker_progress.get(worker_id)

    def set_worker_download_dir(
        self, worker_id: int, directory: Optional[str]
    ) -> Optional[Path]:
        if worker_id < 0 or worker_id >= self.next_worker_id:
            raise ValueError(f"Worker {worker_id} does not exist")

        self._ensure_worker_dir_entry(worker_id)

        if directory is None:
            with self.worker_dir_lock:
                self.worker_download_dirs[worker_id] = None
            return None

        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        with self.worker_dir_lock:
            self.worker_download_dirs[worker_id] = path
        return path

    def get_worker_download_dir(self, worker_id: int) -> Optional[Path]:
        if worker_id < 0:
            return None
        self._ensure_worker_dir_entry(worker_id)
        with self.worker_dir_lock:
            return self.worker_download_dirs.get(worker_id)

    def get_worker_log_tail(
        self, worker_id: int, tail_lines: int
    ) -> Tuple[List[str], int]:
        self._ensure_worker_tracking(worker_id)
        with self.worker_log_lock:
            buffer, _, end_index = self.worker_logs[worker_id].snapshot()
        if tail_lines <= 0:
            return [], end_index
        return buffer[-tail_lines:], end_index

    def get_worker_log_since(
        self, worker_id: int, index: int
    ) -> Tuple[List[str], int, bool]:
        self._ensure_worker_tracking(worker_id)
        with self.worker_log_lock:
            lines, next_index, truncated = self.worker_logs[worker_id].get_since(index)
        return lines, next_index, truncated

    def get_worker_event(self, worker_id: int) -> threading.Event:
        self._ensure_worker_tracking(worker_id)
        return self.worker_log_events[worker_id]

    def import_failed_entries(
        self, entries: Iterable[FailedEntry], *, replace: bool = False
    ) -> None:
        if not self.config.keep_failed and not replace:
            return
        with self.failed_downloads_lock:
            if replace:
                self.failed_downloads.clear()
            for entry in entries:
                self.failed_downloads[entry.url] = entry

    def _record_failed_download(self, url: str, directory: Optional[Path]) -> None:
        if not self.config.keep_failed:
            return
        entry = FailedEntry(url, directory)
        with self.failed_downloads_lock:
            self.failed_downloads[url] = entry

    def _dump_json_metadata(self, worker_id: int, url: str, target_dir: Path) -> None:
        safe_name = _sanitize_filename(url)
        json_path = target_dir / f"{safe_name}.json"
        cmd = [self.exe] + self.config.default_args + ["-J", url]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=300,
            )
        except Exception as exc:
            self._record_worker_output(
                worker_id,
                f"[JSON] Failed to dump metadata for {url}: {exc}",
            )
            return

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "").strip().splitlines()
            detail = f" (stderr: {stderr_snippet[0][:200]})" if stderr_snippet else ""
            self._record_worker_output(
                worker_id,
                f"[JSON] yt-dlp -J exited with {result.returncode}{detail}",
            )
            return

        content = result.stdout
        if not content.strip():
            self._record_worker_output(
                worker_id,
                f"[JSON] No metadata returned for {url}",
            )
            return

        existing_text: Optional[str] = None
        if json_path.exists():
            try:
                existing_text = json_path.read_text(encoding="utf-8")
            except Exception:
                existing_text = None

        if existing_text == content:
            self._record_worker_output(
                worker_id,
                f"[JSON] Metadata unchanged for {json_path.name}",
            )
            return

        try:
            json_path.write_text(content, encoding="utf-8")
            if existing_text is None:
                status = "written"
            else:
                status = "updated"
            self._record_worker_output(
                worker_id,
                f"[JSON] Metadata {status}: {json_path}",
            )
        except Exception as exc:
            self._record_worker_output(
                worker_id,
                f"[JSON] Failed writing metadata to {json_path}: {exc}",
            )

    def _clear_failed_download(self, url: str) -> None:
        with self.failed_downloads_lock:
            self.failed_downloads.pop(url, None)

    def get_failed_downloads(self) -> List[FailedEntry]:
        with self.failed_downloads_lock:
            return list(self.failed_downloads.values())

    def failed_count(self) -> int:
        with self.failed_downloads_lock:
            return len(self.failed_downloads)

    def retry_failed(self) -> int:
        with self.failed_downloads_lock:
            entries = list(self.failed_downloads.values())
            self.failed_downloads.clear()

        if not entries:
            return 0

        requeue: List[FailedEntry] = []
        added_count = 0
        for entry in entries:
            if self.add(entry.url, entry.directory):
                added_count += 1
            else:
                requeue.append(entry)

        if requeue and self.config.keep_failed:
            with self.failed_downloads_lock:
                for entry in requeue:
                    self.failed_downloads[entry.url] = entry

        return added_count

    def _register_worker_process(self, worker_id: int, proc: subprocess.Popen):
        with self.worker_process_lock:
            self.worker_processes[worker_id] = proc

    def _clear_worker_process(self, worker_id: int):
        with self.worker_process_lock:
            self.worker_processes.pop(worker_id, None)
        self._ensure_worker_tracking(worker_id)
        self.clear_worker_progress(worker_id)
        self.worker_log_events[worker_id].set()

    def is_worker_active(self, worker_id: int) -> bool:
        if self.stats.is_worker_active(worker_id):
            return True
        with self.worker_process_lock:
            proc = self.worker_processes.get(worker_id)
        if proc is None:
            return False
        return proc.poll() is None

    def _stream_pipe(self, worker_id: int, pipe: Optional[BinaryIO], stream_name: str):
        if pipe is None:
            return
        decoder_cls = codecs.getincrementaldecoder("utf-8")
        decoder = decoder_cls(errors="replace")
        buffer: List[str] = []
        tag = f"[{stream_name}]"
        in_progress = False

        def _emit_buffer_as_log() -> None:
            nonlocal in_progress
            if buffer and not in_progress:
                line = "".join(buffer)
                clean_line = _strip_ansi(line)

                if PROGRESS_LINE_RE.match(clean_line):
                    self._set_worker_progress(worker_id, clean_line)
                    buffer.clear()
                    in_progress = False
                    return

                lowered = clean_line.lower()
                if "unsupported url" in lowered or "no suitable extractor" in lowered:
                    current_url = self._get_worker_active_url(worker_id)
                    detail = clean_line.strip() or line.strip()
                    self._flag_unsupported(current_url or "", detail)
                self._record_worker_output(worker_id, f"{tag} {line}")
                buffer.clear()
            in_progress = False

        def _process_char(ch: str) -> None:
            nonlocal in_progress
            if ch == "\r" and os.name != "nt":
                if buffer:
                    self._set_worker_progress(worker_id, "".join(buffer))
                buffer.clear()
                in_progress = True
                return

            if ch == "\n":
                if buffer:
                    _emit_buffer_as_log()
                elif not in_progress:
                    self._record_worker_output(worker_id, tag)
                self.clear_worker_progress(worker_id)
                in_progress = False
                buffer.clear()
                return

            buffer.append(ch)
            if in_progress:
                self._set_worker_progress(worker_id, "".join(buffer))

        try:
            while True:
                chunk = pipe.read(1024)
                if chunk == b"":
                    remaining = decoder.decode(b"", final=True)
                    if remaining:
                        for char in remaining:
                            _process_char(char)
                    _emit_buffer_as_log()
                    self.clear_worker_progress(worker_id)
                    break

                text = decoder.decode(chunk, final=False)
                if text:
                    for char in text:
                        _process_char(char)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def add(self, url: str, directory: Optional[Path]) -> bool:
        """Add URL to queue. Returns True if added, False if duplicate."""
        if url in self.queued_urls or url in self.completed_urls:
            return False

        self.queued_urls.add(url)
        self.q.put((url, 0, directory))
        return True

    def pause(self):
        """Pause all workers."""
        self.paused_event.set()

    def resume(self):
        """Resume all workers."""
        self.paused_event.clear()

    def is_paused(self) -> bool:
        return self.paused_event.is_set()

    def get_queue_size(self) -> int:
        return self.q.qsize()

    def get_active_downloads(self) -> Dict[int, Tuple[str, Optional[Path]]]:
        return self.stats.snapshot_active()

    def wait_empty(self):
        """Wait for queue to be empty."""
        self.q.join()

    def shutdown(self):
        """Gracefully shutdown all workers."""
        self.wait_empty()
        self.stop_event.set()

        # Send quit signals
        for _ in self.threads:
            self.q.put(("__QUIT__", 0, None))

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=10)

    def _worker(self, worker_id: int):
        """Worker thread that processes downloads."""
        while not self.stop_event.is_set():
            try:
                # Wait if paused
                while self.paused_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.1)

                if self.stop_event.is_set():
                    break

                url, retry_count, url_directory = self.q.get(timeout=1)

                if url == "__QUIT__":
                    self.q.task_done()
                    break

                self._set_worker_active_url(worker_id, url)
                self._clear_unsupported(url)

                # Determine directory resolution for command and status reporting
                worker_dir = self.get_worker_download_dir(worker_id)
                download_base = (
                    url_directory
                    or worker_dir
                    or self.config.download_dir
                    or Path.cwd()
                )

                if isinstance(download_base, Path):
                    active_dir = download_base
                else:
                    active_dir = Path(download_base).expanduser()

                try:
                    active_dir.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    self._record_worker_output(
                        worker_id,
                        f"[WARN] Could not prepare directory {active_dir}: {exc}",
                    )

                active_dir = active_dir.resolve()
                self._record_worker_output(
                    worker_id,
                    f"[INFO] Using download base directory: {active_dir}",
                )

                if self.config.dump_json:
                    self._dump_json_metadata(worker_id, url, active_dir)

                # Track current download with resolved directory
                self.stats.set_current_download(worker_id, url, active_dir)
                if not self.multi_active_since_idle:
                    active_now = self.stats.snapshot_active()
                    if len(active_now) >= 2:
                        self.multi_active_since_idle = True

                # Reset any previous progress indicator before starting a new task
                self.clear_worker_progress(worker_id)

                # Build command for yt-dlp
                cmd = [self.exe] + self.config.default_args
                if self.js_runtime_arg:
                    cmd.extend(self.js_runtime_arg)
                cmd.extend(["-P", str(active_dir)])

                template = self.config.directory_template
                if not template and not self.config.flat_directories:
                    # Provide a simple hierarchical fallback when requested
                    template = "%(uploader)s/%(title)s [%(id)s].%(ext)s"

                if template:
                    cmd.extend(["-o", template])

                if "--newline" not in cmd:
                    cmd.append("--newline")

                cmd.append(url)
                # Execute download
                start_time = time.time()
                proc: Optional[subprocess.Popen] = None
                stdout_thread: Optional[threading.Thread] = None
                stderr_thread: Optional[threading.Thread] = None
                timed_out = False
                try:
                    self._record_worker_output(worker_id, f"--- Start download: {url}")
                    popen_kwargs = {
                        "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE,
                        "bufsize": 0,
                    }

                    if sys.platform.startswith("win"):
                        try:
                            from subprocess import CREATE_NEW_PROCESS_GROUP
                        except ImportError:
                            CREATE_NEW_PROCESS_GROUP = 0x00000200
                        popen_kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
                    else:
                        popen_kwargs["preexec_fn"] = os.setsid

                    proc = subprocess.Popen(cmd, **popen_kwargs)
                    self._register_worker_process(worker_id, proc)

                    stdout_thread = threading.Thread(
                        target=self._stream_pipe,
                        args=(worker_id, proc.stdout, "STDOUT"),
                        daemon=True,
                    )
                    stderr_thread = threading.Thread(
                        target=self._stream_pipe,
                        args=(worker_id, proc.stderr, "STDERR"),
                        daemon=True,
                    )
                    stdout_thread.start()
                    stderr_thread.start()

                    try:
                        return_code = proc.wait(timeout=86400)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        self._record_worker_output(
                            worker_id,
                            "[TIMEOUT] Download exceeded 24h, terminating process",
                        )
                        proc.kill()
                        return_code = proc.wait()

                    if stdout_thread is not None:
                        stdout_thread.join(timeout=1)
                    if stderr_thread is not None:
                        stderr_thread.join(timeout=1)

                    success = (return_code == 0) and not timed_out
                    unsupported_detail = self._consume_unsupported(url)

                    should_retry = False
                    if (
                        not success
                        and not timed_out
                        and retry_count < self.config.max_retries
                    ):
                        if not unsupported_detail:
                            # Pipelines may flush "Unsupported URL" slightly after process exit.
                            time.sleep(0.2)
                            unsupported_detail = self._consume_unsupported(url)
                        should_retry = not unsupported_detail

                    if should_retry:
                        attempt = retry_count + 1
                        self.stats.record_retry()
                        self._record_worker_output(
                            worker_id,
                            f"[RETRY] Attempt {attempt}/{self.config.max_retries} scheduled in {self.config.retry_delay}s",
                        )
                        print(
                            f"{Fore.YELLOW}WARNING: Worker {worker_id}: Retrying {url} "
                            f"(attempt {attempt}/{self.config.max_retries})"
                        )
                        time.sleep(self.config.retry_delay)
                        self.q.put((url, attempt, url_directory))
                    else:
                        duration = time.time() - start_time
                        self.stats.add_completion(url, success, duration, worker_id)
                        self._maybe_notify_all_idle()

                        if success:
                            self._clear_unsupported(url)
                            self.completed_urls.add(url)
                            self._clear_failed_download(url)
                            self._record_worker_output(
                                worker_id,
                                f"[SUCCESS] Completed in {duration:.1f}s",
                            )
                            print(
                                f"{Fore.GREEN}SUCCESS: Worker {worker_id}: Completed {url} "
                                f"({duration:.1f}s)"
                            )
                        else:
                            if unsupported_detail:
                                self._record_worker_output(
                                    worker_id,
                                    f"[UNSUPPORTED] {unsupported_detail}",
                                )
                                print(
                                    f"{Fore.RED}UNSUPPORTED: Worker {worker_id}: yt-dlp rejected {url}\n  -> {unsupported_detail}"
                                )
                            elif timed_out:
                                self._record_worker_output(
                                    worker_id,
                                    "[FAIL] Download timed out after 86400s",
                                )
                                print(
                                    f"{Fore.RED}TIMEOUT: Worker {worker_id}: Timeout on {url}"
                                )
                            else:
                                self._record_worker_output(
                                    worker_id,
                                    f"[FAIL] yt-dlp exited with code {return_code}",
                                )
                                print(
                                    f"{Fore.RED}FAILED: Worker {worker_id}: Failed {url} "
                                    f"after {retry_count + 1} attempts"
                                )
                            self.queued_urls.discard(url)
                            self._record_failed_download(url, active_dir)

                        if success:
                            self.queued_urls.discard(url)

                except Exception as e:
                    if proc and proc.poll() is None:
                        proc.kill()
                    self._record_worker_output(
                        worker_id,
                        f"[ERROR] Unexpected failure: {e}",
                    )
                    print(f"{Fore.RED}ERROR: Worker {worker_id}: Error on {url}: {e}")
                    self.stats.add_completion(
                        url, False, time.time() - start_time, worker_id
                    )
                    self._maybe_notify_all_idle()
                    self.queued_urls.discard(url)
                    self._record_failed_download(url, active_dir)

                finally:
                    if proc is not None:
                        self._clear_worker_process(worker_id)
                    if stdout_thread is not None and stdout_thread.is_alive():
                        stdout_thread.join(timeout=0.5)
                    if stderr_thread is not None and stderr_thread.is_alive():
                        stderr_thread.join(timeout=0.5)
                    self._set_worker_active_url(worker_id, None)
                    self.clear_worker_progress(worker_id)
                    self.q.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Fore.RED}CRASH: Worker {worker_id} crashed: {e}")
                break


# ---------------------------------------------------------------------------
# Interactive interface
# ---------------------------------------------------------------------------
class InteractiveQueue(cmd.Cmd):
    command_aliases: Dict[str, str] = {
        "run": "start",
        "go": "start",
        "begin": "start",
        "hold": "pause",
        "stop": "pause",
        "continue": "resume",
        "unpause": "resume",
        "reload": "resume",
        "recover": "resume",
        "resume-session": "resume",
        "stat": "status",
        "info": "status",
        "log": "worker",
        "tail": "worker",
        "feed": "worker",
        "wd": "workerdir",
        "workdir": "workerdir",
        "dir": "workerdir",
        "hist": "history",
        "recent": "history",
        "cfg": "config",
        "conf": "config",
        "settings": "config",
        "sv": "save",
        "write": "save",
        "export": "save",
        "ld": "load",
        "restore": "load",
        "open": "load",
        "queue": "pending",
        "pend": "pending",
        "list": "pending",
        "links": "urls",
        "all": "urls",
        "rm": "remove",
        "del": "remove",
        "delete": "remove",
        "clear": "cls",
        "cls": "cls",
        "clq": "clearqueue",
        "clearqueue": "clearqueue",
        "clear-queue": "clearqueue",
        "flush": "clearqueue",
        "reset": "clearqueue",
        "exit": "quit",
        "q": "quit",
        "x": "quit",
        "redo": "retry",
        "rerun": "retry",
        "retry-all": "retry",
        "replay": "retry",
    }

    intro = f"{Fore.CYAN}yt-dlp Queue v2.0{Style.RESET_ALL} – type {Fore.YELLOW}?{Style.RESET_ALL} for help"
    prompt = f"{Fore.GREEN}> {Style.RESET_ALL}"

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.stats = DownloadStats()
        self.pending: List[PendingEntry] = []
        self.all_urls: Set[str] = set()  # Track all URLs ever added
        self.url_directory_overrides: Dict[str, Optional[str]] = {}
        self.dlq: Optional[YTDLPQueue] = None
        self.status_thread: Optional[threading.Thread] = None
        self.worker_dir_overrides: Dict[int, Optional[Path]] = {}
        self.session_url_directory: Optional[Path] = None
        self.async_messages: "queue.Queue[Optional[str]]" = queue.Queue()
        self._async_printer_thread: Optional[threading.Thread] = None
        self.failed_offline: Dict[str, FailedEntry] = {}
        self.auto_start_ready = threading.Event()
        self.auto_start_done = threading.Event()
        self.tool_status: Dict[str, Optional[Path]] = detect_external_tools()
        self.js_runtime_warning_shown = False

        # Load config if exists
        self._load_config()
        self._start_async_printer()
        self._start_update_check()
        self._schedule_autostart()

    # Directory helpers -------------------------------------------------
    def _notify_async(self, message: str) -> None:
        self.async_messages.put(message)

    def _start_async_printer(self) -> None:
        if self._async_printer_thread and self._async_printer_thread.is_alive():
            return

        def _printer_loop():
            while True:
                message = self.async_messages.get()
                if message is None:
                    break
                print(f"\n{message}" if message else "")

        self._async_printer_thread = threading.Thread(
            target=_printer_loop,
            name="async-printer",
            daemon=True,
        )
        self._async_printer_thread.start()

    def _refresh_tool_status(self) -> Dict[str, Optional[Path]]:
        self.tool_status = detect_external_tools()
        return self.tool_status

    def _maybe_warn_js_runtime_for_url(self, url: str) -> None:
        if not is_youtube_url(url) or self.js_runtime_warning_shown:
            return

        self._refresh_tool_status()
        runtime_name, _ = choose_js_runtime()
        if runtime_name:
            return

        print(
            f"{Fore.YELLOW}WARNING: YouTube URL queued but no JavaScript runtime (deno/node/quickjs/bun) was found.\n"
            f"Install Deno (recommended) or pass --js-runtimes to yt-dlp for full YouTube support.{Style.RESET_ALL}"
        )
        self.js_runtime_warning_shown = True

    def _stop_async_printer(self) -> None:
        if not self._async_printer_thread:
            return
        self.async_messages.put(None)
        self._async_printer_thread.join(timeout=1)
        self._async_printer_thread = None

    def _start_update_check(self):
        def _runner():
            try:
                outcome = maybe_check_yt_dlp_update(self.config)
            except UpdateRateLimit as exc:
                self._notify_async(
                    f"{Fore.YELLOW}UPDATE: yt-dlp update check rate limited, will retry later ({exc}).{Style.RESET_ALL}"
                )
                return
            except UpdatePermissionError as exc:
                self._notify_async(
                    f"{Fore.YELLOW}UPDATE: {exc}. Auto-update skipped.{Style.RESET_ALL}"
                )
                return
            except UpdateError as exc:
                self._notify_async(
                    f"{Fore.YELLOW}UPDATE: yt-dlp update check failed: {exc}{Style.RESET_ALL}"
                )
                return
            except Exception as exc:
                self._notify_async(
                    f"{Fore.YELLOW}UPDATE: Unexpected yt-dlp update error: {exc}{Style.RESET_ALL}"
                )
                return

            if not outcome:
                return

            local_version, remote_version, updated = outcome
            version_display = local_version or remote_version
            if updated:
                self._notify_async(
                    f"{Fore.GREEN}UPDATE: yt-dlp auto-updated to {version_display}{Style.RESET_ALL}"
                )
            else:
                self._notify_async(
                    f"{Fore.CYAN}UPDATE: yt-dlp already up to date ({version_display}){Style.RESET_ALL}"
                )

        threading.Thread(target=_runner, name="yt-dlp-update", daemon=True).start()

    def _print_worker_dirs(self, *, label: str = "Worker Directories"):
        print(f"\n{Fore.CYAN}{label}:{Style.RESET_ALL}")
        if self.dlq:
            for worker_id in range(self.dlq.next_worker_id):
                print(f"  Worker {worker_id}: {self._format_worker_dir(worker_id)}")
        elif self.worker_dir_overrides:
            for worker_id in sorted(self.worker_dir_overrides.keys()):
                print(f"  Worker {worker_id}: {self._format_worker_dir(worker_id)}")
        else:
            print("  (using global/default directory)")

    def _format_worker_dir(self, worker_id: int) -> str:
        directory = self._get_worker_dir(worker_id)
        return self._format_directory_label(directory)

    def _format_directory_label(self, directory: Optional[Path | str]) -> str:
        if not directory:
            return "(use global/default)"

        directory_path: Optional[Path]
        try:
            directory_path = Path(directory).expanduser()
        except Exception:
            directory_path = None

        free_info = ""
        if directory_path:
            free_label = self._format_free_space_label(directory_path)
            if free_label:
                free_info = f" ({free_label})"
        directory_str = str(directory)
        return f"{directory_str}{free_info}"

    def _remember_directory_for_url(
        self, url: str, directory: Optional[Path | str]
    ) -> None:
        if directory:
            self.url_directory_overrides[url] = str(directory)
        else:
            self.url_directory_overrides.pop(url, None)

    def _forget_directory_for_url(self, url: str) -> None:
        self.url_directory_overrides.pop(url, None)

    def _format_free_space_label(self, path_obj: Path) -> Optional[str]:
        try:
            usage = shutil.disk_usage(path_obj)
        except (OSError, ValueError):
            return None
        free_in_units = usage.free / (1024**3)
        units = ["GB", "TB", "PB", "EB", "ZB"]
        idx = 0
        while free_in_units >= 1024 and idx < len(units) - 1:
            free_in_units /= 1024
            idx += 1
        amount = f"{free_in_units:.1f}".replace(".", ",")
        return f"{amount} {units[idx]} free space"

    def _get_worker_dir(self, worker_id: int) -> Optional[str]:
        if self.dlq and worker_id < self.dlq.next_worker_id:
            target_dir = self.dlq.get_worker_download_dir(worker_id)
            if target_dir:
                return str(target_dir)
        if worker_id in self.worker_dir_overrides:
            override = self.worker_dir_overrides[worker_id]
            return str(override) if override else None
        if self.config.download_dir:
            return str(self.config.download_dir)
        return None

    # Utility methods ------------------------------------------------------
    def _validate_url(self, url: str) -> bool:
        """Basic URL validation."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _config_file(self) -> Path:
        return Path("yt_dlp_queue_config.json")

    def _config_backup_file(self) -> Path:
        config_file = self._config_file()
        return config_file.with_name(f"{config_file.stem}.backup.json")

    def _save_config(self):
        """Save current configuration and update backup."""
        data = self.config.to_dict()
        config_file = self._config_file()
        backup_file = self._config_backup_file()

        try:
            with open(config_file, "w") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:
            print(f"{Fore.RED}Failed to save config: {exc}")

        try:
            with open(backup_file, "w") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:
            print(
                f"{Fore.YELLOW}Note: Could not write config backup ({backup_file}): {exc}"
            )

    def _load_config(self):
        """Load configuration from primary file, falling back to backup if needed."""
        config_file = self._config_file()
        backup_file = self._config_backup_file()

        def _load_from(path: Path) -> bool:
            with open(path) as handle:
                data = json.load(handle)
            self.config.from_dict(data)
            return True

        loaded = False
        try:
            if config_file.exists():
                loaded = _load_from(config_file)
        except Exception as exc:
            print(f"{Fore.YELLOW}Note: Could not load config ({config_file}): {exc}")

        if not loaded and backup_file.exists():
            try:
                _load_from(backup_file)
                loaded = True
                print(
                    f"{Fore.CYAN}Loaded configuration from backup ({backup_file}).{Style.RESET_ALL}"
                )
            except Exception as exc:
                print(
                    f"{Fore.YELLOW}Note: Could not load config backup ({backup_file}): {exc}"
                )

    def _ensure_downloader(self):
        """Create downloader if it doesn't exist."""
        if self.dlq is None:
            try:
                failed_seed = (
                    list(self.failed_offline.values()) if self.failed_offline else None
                )
                self.dlq = YTDLPQueue(
                    self.config,
                    self.stats,
                    failed_seed=failed_seed,
                )
                self.dlq.set_idle_callback(
                    lambda: self._notify_async(
                        f"{Fore.CYAN}All workers completed their work and are idle again.{Style.RESET_ALL}"
                    )
                )
                if self.failed_offline:
                    self.failed_offline.clear()
                print(f"STARTED: {self.config.workers} workers")

                runtime_name, runtime_path = self.dlq.js_runtime_choice
                if runtime_name:
                    location = f" ({runtime_path})" if runtime_path else ""
                    print(
                        f"{Fore.CYAN}YouTube JS runtime: {runtime_name}{location}{Style.RESET_ALL}"
                    )
                elif not self.js_runtime_warning_shown:
                    print(
                        f"{Fore.YELLOW}WARNING: No JavaScript runtime detected; YouTube downloads may fail."
                        f" Install Deno or pass --js-runtimes.{Style.RESET_ALL}"
                    )
                    self.js_runtime_warning_shown = True

                if self.config.auto_status:
                    self._start_status_updates()
            except Exception as exc:
                print(f"{Fore.RED}Failed to start workers: {exc}")
                self.dlq = None
                return False
        return True

    def _start_status_updates(self):
        """Start background thread for status updates."""

        if self.status_thread and self.status_thread.is_alive():
            return

        def _status_loop():
            try:
                while self.dlq and not self.dlq.stop_event.is_set():
                    if not self.config.auto_status:
                        break

                    active = self.dlq.get_active_downloads()
                    if active:
                        queue_size = self.dlq.get_queue_size()
                        print(
                            f"\r{Fore.BLUE}STATUS: Active: {len(active)}, Queue: {queue_size}    ",
                            end="",
                        )
                    time.sleep(2)
            finally:
                self.status_thread = None

        self.status_thread = threading.Thread(target=_status_loop, daemon=True)
        self.status_thread.start()

    def _schedule_autostart(self):
        if self.pending:
            self.auto_start_ready.set()

        def _auto_runner():
            while not self.auto_start_done.is_set():
                self.auto_start_ready.wait()
                if self.auto_start_done.is_set():
                    break
                try:
                    time.sleep(5)
                except Exception:
                    pass
                if self.auto_start_done.is_set():
                    break
                if self.dlq or not self.pending:
                    self.auto_start_ready.clear()
                    continue
                try:
                    self.auto_start_done.set()
                    self.do_start("")
                except Exception as exc:
                    self.auto_start_done.clear()
                    self._notify_async(
                        f"{Fore.YELLOW}AUTO-START: Unable to start workers automatically: {exc}{Style.RESET_ALL}"
                    )
                    # Allow reattempt on next pending signal
                    self.auto_start_ready.clear()

        threading.Thread(target=_auto_runner, name="auto-start", daemon=True).start()

    def _signal_auto_start_ready(self) -> None:
        if not self.dlq and self.pending:
            self.auto_start_ready.set()

    def _apply_session_data(self, session_data: dict, *, merge: bool) -> int:
        pending_payload = session_data.get("pending", []) or []
        restored_entries: List[PendingEntry] = []

        if not merge:
            self.pending = []
            self.all_urls = set()
            if not self.dlq:
                self.worker_dir_overrides.clear()
            self.url_directory_overrides.clear()

        for item in pending_payload:
            directory_path: Optional[Path] = None
            if isinstance(item, dict):
                url = item.get("url")
                if not url:
                    continue
                directory_value = item.get("directory")
                if directory_value:
                    try:
                        directory_path = Path(directory_value).expanduser()
                        directory_path.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(
                            f"{Fore.YELLOW}Note: Could not prepare pending directory ({directory_value}): {exc}"
                        )
                        directory_path = None
            else:
                url = str(item)

            restored_entry = PendingEntry(url, directory_path)
            restored_entries.append(restored_entry)
            self._remember_directory_for_url(url, directory_path)

        if merge:
            self.pending.extend(restored_entries)
            self.all_urls.update(entry.url for entry in restored_entries)
        else:
            self.pending = list(restored_entries)
            self.all_urls = {entry.url for entry in restored_entries}

        if "config" in session_data:
            self.config.from_dict(session_data["config"])

        next_dir_value = session_data.get("session_url_directory")
        if next_dir_value:
            try:
                next_dir_path = Path(next_dir_value).expanduser()
                next_dir_path.mkdir(parents=True, exist_ok=True)
                self.session_url_directory = next_dir_path
            except Exception as exc:
                self.session_url_directory = None
                print(
                    f"{Fore.YELLOW}Note: Could not restore session default directory ({next_dir_value}): {exc}"
                )
        else:
            self.session_url_directory = None

        worker_dirs_data = session_data.get("worker_directories")
        if worker_dirs_data is not None:
            target_dirs: Dict[int, Optional[Path]] = {}
            for worker_id_str, directory_value in worker_dirs_data.items():
                try:
                    worker_id = int(worker_id_str)
                except (TypeError, ValueError):
                    continue

                directory_path = None
                if directory_value:
                    try:
                        directory_path = Path(directory_value).expanduser()
                        directory_path.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(
                            f"{Fore.YELLOW}Note: Could not restore worker directory ({directory_value}): {exc}"
                        )
                        directory_path = None

                target_dirs[worker_id] = directory_path

            if self.dlq:
                for worker_id, path_obj in target_dirs.items():
                    try:
                        self.dlq.set_worker_download_dir(
                            worker_id, str(path_obj) if path_obj else None
                        )
                    except Exception as exc:
                        print(
                            f"{Fore.YELLOW}Note: Could not assign worker {worker_id} directory while loading: {exc}"
                        )
            else:
                if not merge:
                    self.worker_dir_overrides.clear()
                for worker_id, path_obj in target_dirs.items():
                    self.worker_dir_overrides[worker_id] = path_obj
        elif not merge and not self.dlq:
            self.worker_dir_overrides.clear()

        failed_payload = session_data.get("failed", []) or []
        restored_failures: List[FailedEntry] = []
        for item in failed_payload:
            if isinstance(item, dict):
                url = item.get("url")
                if not url:
                    continue
                directory_value = item.get("directory")
                directory_path = None
                if directory_value:
                    try:
                        directory_path = Path(directory_value).expanduser()
                        directory_path.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(
                            f"{Fore.YELLOW}Note: Could not prepare failed directory ({directory_value}): {exc}"
                        )
                        directory_path = None
                restored_failures.append(FailedEntry(url, directory_path))
            else:
                restored_failures.append(FailedEntry(str(item), None))

        if restored_failures and not self.config.keep_failed:
            print(
                f"{Fore.YELLOW}Note: keep_failed is disabled, skipping restoration of {len(restored_failures)} failed downloads"
            )
            restored_failures = []

        if restored_failures:
            if self.dlq:
                self.dlq.import_failed_entries(restored_failures, replace=not merge)
            else:
                if not merge:
                    self.failed_offline.clear()
                for entry in restored_failures:
                    self.failed_offline[entry.url] = entry
        elif not merge and not self.dlq:
            self.failed_offline.clear()

        self._save_config()

        self._signal_auto_start_ready()

        return len(restored_entries)

    # Commands -------------------------------------------------------------
    def precmd(self, line: str) -> str:
        stripped = line.strip()
        if stripped:
            parts = stripped.split(None, 1)
            command = parts[0].lower()
            remainder = parts[1] if len(parts) > 1 else ""
            canonical = self.command_aliases.get(command)
            if canonical:
                line = f"{canonical} {remainder}".strip()
        return super().precmd(line)

    def postcmd(self, stop: bool, line: str) -> bool:
        stop = super().postcmd(stop, line)
        return stop

    def default(self, line: str) -> bool:
        """Handle URLs and unknown commands."""
        line = line.strip()
        if not line:
            return False

        # Check if it looks like a URL
        if self._validate_url(line):
            return self._add_url(line)

        print(f"{Fore.RED}Unknown command or invalid URL: {line}")
        print(f"Type {Fore.YELLOW}?{Style.RESET_ALL} for help")
        return False

    def _add_url(self, url: str) -> bool:
        """Add a URL to queue or pending list."""
        if not self._validate_url(url):
            print(f"{Fore.RED}Invalid URL: {url}")
            return False

        # Always track the URL
        self.all_urls.add(url)
        self._maybe_warn_js_runtime_for_url(url)

        directory_override = self.session_url_directory
        added_now = False

        if self.dlq is None:
            if not any(entry.url == url for entry in self.pending):
                self.pending.append(PendingEntry(url, directory_override))
                self._signal_auto_start_ready()
                dir_msg = f" (dir: {directory_override})" if directory_override else ""
                print(
                    f"{Fore.WHITE}PENDING: Added to pending: {url}{dir_msg}{Style.RESET_ALL}"
                )
                added_now = True
                self._remember_directory_for_url(url, directory_override)
            else:
                print(f"{Fore.YELLOW}WARNING: Already in pending: {url}")
        else:
            if self.dlq.add(url, directory_override):
                dir_msg = f" (dir: {directory_override})" if directory_override else ""
                print(
                    f"{Fore.WHITE}QUEUED: Queued for download: {url}{dir_msg}{Style.RESET_ALL}"
                )
                added_now = True
                self._remember_directory_for_url(url, directory_override)
            else:
                print(
                    f"{Fore.YELLOW}WARNING: Duplicate URL (already queued/completed): {url}"
                )

        if added_now and self.dlq is None:
            print(
                f"{Fore.WHITE}AUTO-START: Workers will start automatically after 5s delay{Style.RESET_ALL}"
            )
            print(
                f"{Fore.WHITE}PENDING: URL added. Use 'start' to begin downloading.{Style.RESET_ALL}"
            )

        return False

    def do_start(self, arg: str) -> bool:
        """Start workers and begin downloading."""
        if not self._ensure_downloader():
            return False

        self.auto_start_done.set()
        self.auto_start_ready.clear()

        if self.dlq.is_paused():
            self.dlq.resume()
            print(f"{Fore.GREEN}RESUMED: Resumed workers")

        # Add pending URLs and track them
        if self.worker_dir_overrides:
            for worker_id, path in list(self.worker_dir_overrides.items()):
                try:
                    self.dlq.set_worker_download_dir(
                        worker_id, str(path) if path else None
                    )
                except Exception as e:
                    print(f"{Fore.RED}Failed to set worker {worker_id} directory: {e}")
            self.worker_dir_overrides.clear()

        added = 0
        for entry in list(self.pending):
            if self.dlq.add(entry.url, entry.directory):
                added += 1
                self.pending.remove(entry)

        if added:
            print(f"{Fore.GREEN}STARTED: Added {added} URLs from pending list")

        if (
            added == 0
            and not self.dlq.get_queue_size()
            and not self.dlq.get_active_downloads()
        ):
            print(f"{Fore.YELLOW}No URLs to download. Add some URLs first!")

        return False

    def do_pause(self, arg: str) -> bool:
        """Pause all workers."""
        if self.dlq:
            self.dlq.pause()
            print(f"{Fore.YELLOW}PAUSED: Paused all workers")
        else:
            print(f"{Fore.RED}No workers running")
        return False

    def do_resume(self, arg: str) -> bool:
        """Resume paused workers or reload the last saved session when idle."""
        if self.dlq:
            self.dlq.resume()
            print(f"{Fore.GREEN}RESUMED: Resumed all workers")
            return False

        filename = arg.strip() or self.config.session_file
        session_path = Path(filename)

        if not session_path.exists():
            print(f"{Fore.RED}Session file not found: {session_path}")
            return False

        try:
            with open(session_path) as f:
                session_data = json.load(f)
        except Exception as exc:
            print(f"{Fore.RED}Failed to resume session: {exc}")
            return False

        restored_count = self._apply_session_data(session_data, merge=False)
        print(
            f"{Fore.GREEN}RESUME: Loaded session from {session_path} with {restored_count} URLs"
        )

        if "worker_directories" in session_data:
            print(
                f"Restored {len(session_data['worker_directories'])} worker directory settings"
            )

        if "total_urls_added" in session_data:
            total = session_data["total_urls_added"]
            completed = session_data.get("completed_count", 0)
            print(f"Original session: {total} total URLs, {completed} completed")

        self.do_start("")
        return False

    def do_retry(self, arg: str) -> bool:
        """Retry all failed downloads with their recorded directories."""
        if self.dlq:
            added = self.dlq.retry_failed()
            if added:
                print(f"{Fore.GREEN}RETRY: Re-queued {added} failed downloads")
                if self.dlq.is_paused():
                    print(
                        f"{Fore.YELLOW}Workers are paused, use 'resume' to continue processing"
                    )
            else:
                print(f"{Fore.CYAN}No failed downloads awaiting retry")
            return False

        if not self.failed_offline:
            print(f"{Fore.CYAN}No failed downloads recorded")
            return False

        existing_pending = {entry.url for entry in self.pending}
        added = 0
        skipped = 0
        for entry in list(self.failed_offline.values()):
            if entry.url in existing_pending:
                skipped += 1
                continue
            self.pending.append(PendingEntry(entry.url, entry.directory))
            self.all_urls.add(entry.url)
            self._remember_directory_for_url(entry.url, entry.directory)
            added += 1

        self.failed_offline.clear()

        if added:
            print(f"{Fore.GREEN}RETRY: Added {added} failed downloads back to pending")
            if skipped:
                print(f"{Fore.YELLOW}Skipped {skipped} already-pending URLs")
            print(
                f"{Fore.CYAN}Use 'start' to process the retried downloads{Style.RESET_ALL}"
            )
        else:
            print(f"{Fore.CYAN}No failed downloads were ready for retry")

        return False

    def do_status(self, arg: str) -> bool:
        """Show detailed status information."""
        print(f"\n{Fore.CYAN}=== Status Report ==={Style.RESET_ALL}")
        print(f"Workers: {self.config.workers}")
        print(f"Pending URLs: {len(self.pending)}")
        print(f"Total URLs tracked: {len(self.all_urls)}")
        failed_count = self.dlq.failed_count() if self.dlq else len(self.failed_offline)
        print(f"Failed downloads awaiting retry: {failed_count}")

        if self.dlq:
            active = self.dlq.get_active_downloads()
            queue_size = self.dlq.get_queue_size()
            completed_count = len(self.dlq.completed_urls)

            print(f"Queue size: {queue_size}")
            print(f"Active downloads: {len(active)}")
            print(f"Completed downloads: {completed_count}")
            print(f"Status: {'PAUSED' if self.dlq.is_paused() else 'RUNNING'}")

            # Calculate URLs that would be saved
            pending_url_set = {entry.url for entry in self.pending}
            urls_to_save_count = len(pending_url_set)
            for url in self.all_urls:
                if url not in self.dlq.completed_urls and url not in pending_url_set:
                    urls_to_save_count += 1
            print(f"URLs that would be saved: {urls_to_save_count}")

            if active:
                print(f"\n{Fore.YELLOW}Active Downloads:{Style.RESET_ALL}")
                for worker_id, (url, active_dir) in active.items():
                    dir_label = (
                        str(active_dir)
                        if active_dir is not None
                        else self._format_worker_dir(worker_id)
                    )
                    print(f"  Worker {worker_id}: {url}")
                    print(f"    → Directory: {dir_label}")

            self._print_worker_dirs()
        else:
            print("Workers: Not started")
            print(f"URLs that would be saved: {len(self.pending)}")
            if self.worker_dir_overrides:
                self._print_worker_dirs(label="Planned Worker Directories")

        session_label = self._format_directory_label(self.session_url_directory)
        print(f"Session default directory: {session_label}")

        tool_info = self._refresh_tool_status()
        print(f"\n{Fore.CYAN}External Tool Checks:{Style.RESET_ALL}")
        for name in EXTERNAL_TOOL_ORDER:
            path = tool_info.get(name)
            critical = name in CRITICAL_TOOLS
            color = Fore.GREEN if path else (Fore.RED if critical else Fore.YELLOW)
            status = "FOUND" if path else "MISSING"
            detail = str(path) if path else ""
            note = TOOL_NOTES.get(name)
            extra = f" – {note}" if note else ""
            if detail:
                detail = f" {detail}"
            print(f"  {color}{name:<14} {status:<7}{Style.RESET_ALL}{detail}{extra}")

        runtime_name, runtime_path = choose_js_runtime()
        runtime_color = Fore.GREEN if runtime_name else Fore.RED
        runtime_detail = (
            f"{runtime_name} @ {runtime_path}" if runtime_name else "none detected"
        )
        print(
            f"  {runtime_color}JS runtime available: {runtime_detail}{Style.RESET_ALL}"
        )

        print(f"\n{self.stats.get_summary()}")
        return False

    def do_worker(self, arg: str) -> bool:
        """Follow live stdout/stderr for a worker. Usage: worker <id> [tail_lines]"""
        if not self.dlq:
            print(f"{Fore.RED}No workers running")
            return False

        args = arg.split()
        if not args:
            print("Usage: worker <id> [tail_lines]")
            return False

        first_arg = args[0].lower()
        if first_arg in {"all", "*"}:
            try:
                tail_lines = int(args[1]) if len(args) > 1 else 25
            except ValueError:
                tail_lines = 25
            return self._follow_all_workers(tail_lines)

        try:
            worker_id = int(args[0])
        except ValueError:
            print(f"{Fore.RED}Worker id must be an integer")
            return False

        if worker_id < 0 or worker_id >= self.dlq.next_worker_id:
            print(f"{Fore.RED}No worker with id {worker_id}")
            return False

        try:
            tail_lines = int(args[1]) if len(args) > 1 else 25
        except ValueError:
            tail_lines = 25

        tail_lines = max(0, tail_lines)

        lines, index = self.dlq.get_worker_log_tail(worker_id, tail_lines)
        if lines:
            print(
                f"\n{Fore.CYAN}=== Worker {worker_id} log (last {len(lines)} lines) ==={Style.RESET_ALL}"
            )
            for line in lines:
                color = Fore.WHITE
                prefix = ""
                if line.startswith("[STDERR]"):
                    color = Fore.RED
                    prefix = line[len("[STDERR]") :].lstrip()
                elif line.startswith("[STDOUT]"):
                    color = Fore.WHITE
                    prefix = line[len("[STDOUT]") :].lstrip()
                else:
                    prefix = line
                print(f"{color}Worker {worker_id}{Style.RESET_ALL} {prefix}")
        else:
            print(f"{Fore.YELLOW}No log output from worker {worker_id} yet")

        event = self.dlq.get_worker_event(worker_id)
        event.clear()
        print(
            f"\n{Fore.CYAN}Live feed – press Ctrl+C to return to the main prompt (worker keeps running).{Style.RESET_ALL}"
        )

        index_tracker = index
        last_idle_notice = 0.0
        idle_frames = ["   ", ".  ", ".. ", "..."]
        idle_index = 0
        status_line_active = False
        last_status_text = ""
        last_progress_value: Optional[str] = None
        last_status_len = 0

        def _clear_status_line() -> None:
            nonlocal status_line_active, last_status_text, last_status_len
            if status_line_active:
                sys.stdout.write("\r" + " " * last_status_len + "\r")
                sys.stdout.flush()
                status_line_active = False
                last_status_text = ""
                last_status_len = 0

        def _show_status_line(text: str) -> None:
            nonlocal status_line_active, last_status_text, last_status_len
            if text == last_status_text and status_line_active:
                return
            sys.stdout.write("\r" + text)
            visible_len = len(text)
            if last_status_len > visible_len:
                sys.stdout.write(" " * (last_status_len - visible_len))
            sys.stdout.flush()
            status_line_active = True
            last_status_text = text
            last_status_len = visible_len

        try:
            while True:
                new_lines, next_index, truncated = self.dlq.get_worker_log_since(
                    worker_id, index_tracker
                )
                if truncated:
                    _clear_status_line()
                    print(
                        f"{Fore.YELLOW}... worker {worker_id} log truncated, showing latest entries"
                    )
                if new_lines:
                    _clear_status_line()
                    for line in new_lines:
                        color = Fore.WHITE
                        message = line
                        if line.startswith("[STDERR]"):
                            color = Fore.RED
                            message = line[len("[STDERR]") :].lstrip()
                        elif line.startswith("[STDOUT]"):
                            message = line[len("[STDOUT]") :].lstrip()
                        print(f"{color}Worker {worker_id}{Style.RESET_ALL} {message}")
                    index_tracker = next_index
                    last_idle_notice = time.time()
                else:
                    index_tracker = next_index

                progress_text = self.dlq.get_worker_progress(worker_id)
                worker_active = self.dlq.is_worker_active(worker_id)

                if progress_text and worker_active:
                    if progress_text != last_progress_value:
                        pretty = f"{Fore.YELLOW}Worker {worker_id} progress:{Style.RESET_ALL} {progress_text}"
                        _show_status_line(pretty)
                        last_progress_value = progress_text
                    last_idle_notice = time.time()
                else:
                    if last_progress_value is not None:
                        _clear_status_line()
                        last_progress_value = None

                    if not worker_active:
                        now = time.time()
                        if now - last_idle_notice > 1:
                            frame = idle_frames[idle_index % len(idle_frames)]
                            status = f"{Fore.YELLOW}Worker {worker_id} idle {frame}{Style.RESET_ALL}"
                            _show_status_line(status)
                            idle_index += 1
                            last_idle_notice = now
                    else:
                        _clear_status_line()
                        last_idle_notice = time.time()

                event.wait(timeout=1)
                event.clear()
        except KeyboardInterrupt:
            _clear_status_line()
            print(f"{Fore.GREEN}Stopped following worker {worker_id}{Style.RESET_ALL}")

        return False

    def _follow_all_workers(self, tail_lines: int) -> bool:
        if not self.dlq:
            print(f"{Fore.RED}No workers running")
            return False

        worker_ids = [wid for wid in range(self.dlq.next_worker_id)]
        if not worker_ids:
            print(f"{Fore.YELLOW}No workers available yet")
            return False

        tail_lines = max(0, tail_lines)
        print(
            f"\n{Fore.CYAN}=== Combined worker tail (last {tail_lines} lines per worker) ==={Style.RESET_ALL}"
        )
        indices: Dict[int, int] = {}
        for wid in worker_ids:
            lines, index = self.dlq.get_worker_log_tail(wid, tail_lines)
            indices[wid] = index
            if lines:
                print(f"{Fore.WHITE}--- Worker {wid} tail ---{Style.RESET_ALL}")
                for line in lines:
                    color = Fore.WHITE
                    message = line
                    if line.startswith("[STDERR]"):
                        color = Fore.RED
                        message = line[len("[STDERR]") :].lstrip()
                    elif line.startswith("[STDOUT]"):
                        message = line[len("[STDOUT]") :].lstrip()
                    print(f"{color}Worker {wid}{Style.RESET_ALL} {message}")
        print(
            f"\n{Fore.CYAN}Live combined feed – press Ctrl+C to return to the main prompt (workers keep running).{Style.RESET_ALL}"
        )

        last_progress: Dict[int, Optional[str]] = {wid: None for wid in worker_ids}

        try:
            while True:
                any_activity = False
                for wid in worker_ids:
                    new_lines, next_index, truncated = self.dlq.get_worker_log_since(
                        wid, indices[wid]
                    )
                    if truncated:
                        print(
                            f"{Fore.YELLOW}... worker {wid} log truncated, showing latest entries"
                        )
                    if new_lines:
                        any_activity = True
                        for line in new_lines:
                            color = Fore.WHITE
                            message = line
                            if line.startswith("[STDERR]"):
                                color = Fore.RED
                                message = line[len("[STDERR]") :].lstrip()
                            elif line.startswith("[STDOUT]"):
                                message = line[len("[STDOUT]") :].lstrip()
                            print(f"{color}Worker {wid}{Style.RESET_ALL} {message}")
                    indices[wid] = next_index

                    progress_text = self.dlq.get_worker_progress(wid)
                    worker_active = self.dlq.is_worker_active(wid)
                    if progress_text and worker_active:
                        if progress_text != last_progress[wid]:
                            any_activity = True
                            print(
                                f"{Fore.YELLOW}Worker {wid} progress:{Style.RESET_ALL} {progress_text}"
                            )
                            last_progress[wid] = progress_text
                    else:
                        last_progress[wid] = None

                if not any_activity:
                    time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"{Fore.GREEN}Stopped following all workers{Style.RESET_ALL}")

        return False

    def do_workerdir(self, arg: str) -> bool:
        """Manage worker download directories or set the session default directory.

        Usage examples:
          workerdir                       # list worker directories and session default
          workerdir <id>                  # show directory for worker id
          workerdir <id> <path>           # set worker directory (use 'default' to clear)
          workerdir <path>                # set session default directory for new URLs
          workerdir reset|default         # clear session default directory
        """

        tokens = shlex.split(arg, posix=False)

        def _normalize_path_input(value: str) -> str:
            text = value.strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
                text = text[1:-1]
            return text

        if not tokens:
            # label = "Worker Directories" if self.dlq else "Planned Worker Directories"
            # self._print_worker_dirs(label=label)
            session_label = self._format_directory_label(self.session_url_directory)
            print(f"Session default directory: {session_label}")
            return False

        first_token = tokens[0]

        # Try worker-id mode first
        worker_id: Optional[int]
        try:
            worker_id = int(first_token)
        except ValueError:
            worker_id = None

        if worker_id is None:
            # Treat as session-level directory override
            directive = first_token.lower()
            if directive in {"default", "none", "clear", "reset"}:
                self.session_url_directory = None
                print(
                    f"{Fore.GREEN}Session default directory cleared – using global/default location"
                )
                return False

            path_input = _normalize_path_input(" ".join(tokens))
            try:
                path_obj = Path(path_input).expanduser()
                path_obj.mkdir(parents=True, exist_ok=True)
                self.session_url_directory = path_obj
                free_info = self._format_free_space_label(path_obj)
                info_text = f" ({free_info})" if free_info else ""
                print(
                    f"{Fore.GREEN}Session default directory set to {path_obj}{info_text}"
                )
            except Exception as exc:
                print(f"{Fore.RED}Failed to set session default directory: {exc}")
            return False

        if worker_id < 0:
            print(f"{Fore.RED}Worker id must be non-negative")
            return False

        if len(tokens) == 1:
            print(f"Worker {worker_id}: {self._format_worker_dir(worker_id)}")
            return False

        path_token_raw = _normalize_path_input(" ".join(tokens[1:]))
        directory_input: Optional[str]
        if path_token_raw.lower() in {"default", "none", "clear"}:
            directory_input = None
        else:
            directory_input = path_token_raw or None

        try:
            if self.dlq:
                if worker_id >= self.dlq.next_worker_id:
                    print(
                        f"{Fore.RED}Worker {worker_id} does not exist (currently {self.dlq.next_worker_id} workers)"
                    )
                    return False
                new_path = self.dlq.set_worker_download_dir(worker_id, directory_input)
            else:
                new_path = (
                    Path(directory_input).expanduser() if directory_input else None
                )
                if directory_input:
                    new_path.mkdir(parents=True, exist_ok=True)
                self.worker_dir_overrides[worker_id] = new_path

            if directory_input is None:
                msg = "(use global/default)"
            else:
                msg = str(new_path) if new_path else directory_input
                if new_path:
                    free_info = self._format_free_space_label(new_path)
                    if free_info:
                        msg = f"{msg} ({free_info})"

            print(f"{Fore.GREEN}Worker {worker_id} directory set to: {msg}")
        except Exception as e:
            print(f"{Fore.RED}Failed to set directory: {e}")

        return False

    def do_history(self, arg: str) -> bool:
        """Show download history."""
        history = self.stats.history_snapshot()
        if not history:
            print("No download history available")
            return False

        # Parse argument for limit
        try:
            limit = int(arg) if arg.strip() else 10
        except ValueError:
            limit = 10

        print(f"\n{Fore.CYAN}=== Last {limit} Downloads ==={Style.RESET_ALL}")
        for entry in history[-limit:]:
            status = f"{Fore.GREEN}SUCCESS" if entry["success"] else f"{Fore.RED}FAILED"
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            print(f"{status} [{timestamp}] {entry['url']} ({entry['duration']:.1f}s)")

        return False

    def do_config(self, arg: str) -> bool:
        """View or modify configuration. Usage: config [key [value]]"""
        args = arg.split()

        if not args:
            # Show all config
            print(f"\n{Fore.CYAN}=== Configuration ==={Style.RESET_ALL}")
            for key, value in self.config.to_dict().items():
                print(f"{key}: {value}")
        elif len(args) == 1:
            # Show specific config
            key = args[0]
            if hasattr(self.config, key):
                print(f"{key}: {getattr(self.config, key)}")
            else:
                print(f"{Fore.RED}Unknown config key: {key}")
        else:
            # Set config
            key, value = args[0], " ".join(args[1:])
            if hasattr(self.config, key):
                # Type conversion
                current = getattr(self.config, key)
                try:
                    if isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, bool):
                        value = value.lower() in ("true", "1", "yes")
                    elif isinstance(current, list):
                        value = value.split(",") if value else []

                    if key == "directory_template" and isinstance(value, str):
                        normalized = value.strip()
                        if normalized.lower() in {
                            "",
                            "none",
                            "null",
                            "default",
                        }:
                            value = None
                        else:
                            value = normalized

                    setattr(self.config, key, value)
                    print(f"{Fore.GREEN}Set {key} = {value}")
                    self._save_config()

                    # Special handling for workers
                    if key == "workers" and self.dlq:
                        print(
                            f"{Fore.YELLOW}Note: Worker count change will take effect on restart"
                        )
                    elif key == "auto_status" and self.config.auto_status and self.dlq:
                        self._start_status_updates()
                    elif (
                        key == "flat_directories"
                        and self.config.flat_directories
                        and self.config.directory_template
                    ):
                        print(
                            f"{Fore.YELLOW}Note: flat_directories overrides directory_template for downloads"
                        )
                    elif key == "keep_failed" and not self.config.keep_failed:
                        if self.dlq:
                            self.dlq.import_failed_entries([], replace=True)
                        self.failed_offline.clear()

                except ValueError as e:
                    print(f"{Fore.RED}Invalid value for {key}: {e}")
            else:
                print(f"{Fore.RED}Unknown config key: {key}")

        return False

    def do_save(self, arg: str) -> bool:
        """Save current session to file."""
        filename = arg.strip() or self.config.session_file

        # Collect all URLs that need to be saved (not yet completed)
        urls_to_save: List[str] = []
        pending_payload: List[object] = []

        # Add pending URLs (retain their per-URL directory overrides)
        for entry in self.pending:
            urls_to_save.append(entry.url)
            if entry.directory:
                pending_payload.append(
                    {"url": entry.url, "directory": str(entry.directory)}
                )
            else:
                pending_payload.append(entry.url)

        # Add URLs that are queued but not completed
        if self.dlq:
            completed_urls = self.dlq.completed_urls
            pending_urls_set = {entry.url for entry in self.pending}
            for url in self.all_urls:
                if url not in completed_urls and url not in pending_urls_set:
                    urls_to_save.append(url)
                    directory_override = self.url_directory_overrides.get(url)
                    if directory_override:
                        pending_payload.append(
                            {"url": url, "directory": directory_override}
                        )
                    else:
                        pending_payload.append(url)

        session_data = {
            "pending": pending_payload,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "total_urls_added": len(self.all_urls),
            "completed_count": len(self.dlq.completed_urls) if self.dlq else 0,
            "session_url_directory": str(self.session_url_directory)
            if self.session_url_directory
            else None,
        }

        if self.dlq:
            failed_entries = self.dlq.get_failed_downloads()
        else:
            failed_entries = list(self.failed_offline.values())

        if failed_entries:
            session_data["failed"] = [
                {
                    "url": entry.url,
                    "directory": str(entry.directory) if entry.directory else None,
                }
                for entry in failed_entries
            ]

        worker_dir_payload: Dict[str, Optional[str]] = {}
        if self.dlq:
            for worker_id in range(self.dlq.next_worker_id):
                directory = self.dlq.get_worker_download_dir(worker_id)
                worker_dir_payload[str(worker_id)] = (
                    str(directory) if directory else None
                )
        elif self.worker_dir_overrides:
            for worker_id, path in sorted(self.worker_dir_overrides.items()):
                worker_dir_payload[str(worker_id)] = str(path) if path else None

        if worker_dir_payload:
            session_data["worker_directories"] = worker_dir_payload

        try:
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=2)
            print(f"{Fore.GREEN}SAVED: Session saved to {filename}")
            print(f"Saved {len(urls_to_save)} uncompleted URLs")
            if failed_entries:
                print(f"Saved {len(failed_entries)} failed downloads for future retry")
        except Exception as e:
            print(f"{Fore.RED}Failed to save session: {e}")

        outstanding = set(urls_to_save)
        if outstanding:
            self.url_directory_overrides = {
                url: directory
                for url, directory in self.url_directory_overrides.items()
                if directory and url in outstanding
            }
        else:
            self.url_directory_overrides.clear()

        return False

    def do_load(self, arg: str) -> bool:
        """Load session from file."""
        filename = arg.strip() or self.config.session_file

        try:
            with open(filename) as f:
                session_data = json.load(f)

        except FileNotFoundError:
            print(f"{Fore.RED}Session file not found: {filename}")
            return False
        except Exception as e:
            print(f"{Fore.RED}Failed to load session: {e}")
            return False

        restored_count = self._apply_session_data(session_data, merge=True)

        print(f"{Fore.GREEN}LOADED: Session loaded from {filename}")
        print(f"Added {restored_count} URLs to pending")

        if "total_urls_added" in session_data:
            total = session_data["total_urls_added"]
            completed = session_data.get("completed_count", 0)
            print(f"Original session: {total} total URLs, {completed} completed")

        return False

    def do_pending(self, arg: str) -> bool:
        """Show pending URLs (before workers start)."""
        if not self.pending:
            print("No pending URLs")
        else:
            print(
                f"\n{Fore.CYAN}=== Pending URLs ({len(self.pending)}) ==={Style.RESET_ALL}"
            )
            for i, entry in enumerate(self.pending):
                dir_msg = f" [dir: {entry.directory}]" if entry.directory else ""
                print(f"{i:2d}: {entry.url}{dir_msg}")
        return False

    def do_urls(self, arg: str) -> bool:
        """Show all tracked URLs and their status."""
        if not self.all_urls:
            print("No URLs tracked")
            return False

        print(
            f"\n{Fore.CYAN}=== All Tracked URLs ({len(self.all_urls)}) ==={Style.RESET_ALL}"
        )

        pending_set = {entry.url for entry in self.pending}
        completed_set = self.dlq.completed_urls if self.dlq else set()
        queued_set = self.dlq.queued_urls if self.dlq else set()

        for i, url in enumerate(sorted(self.all_urls)):
            if url in completed_set:
                status = f"{Fore.GREEN}COMPLETED"
            elif url in pending_set:
                status = f"{Fore.YELLOW}PENDING"
            elif url in queued_set:
                status = f"{Fore.BLUE}QUEUED"
            else:
                status = f"{Fore.RED}UNKNOWN"

            print(f"{i:2d}: {status}{Style.RESET_ALL} {url}")

        # Show what would be saved
        urls_to_save = [entry.url for entry in self.pending]
        if self.dlq:
            for url in self.all_urls:
                if url not in completed_set and url not in pending_set:
                    urls_to_save.append(url)

        print(
            f"\n{Fore.CYAN}URLs that would be saved: {len(urls_to_save)}{Style.RESET_ALL}"
        )
        return False

    def do_remove(self, arg: str) -> bool:
        """Remove URL from pending list. Usage: remove <index|url>"""
        if self.dlq:
            print(f"{Fore.RED}Cannot remove - workers already running")
            return False

        arg = arg.strip()
        if not arg:
            print("Usage: remove <index|url>")
            return False

        removed: Optional[PendingEntry] = None
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(self.pending):
                removed = self.pending.pop(idx)
        else:
            for entry in self.pending:
                if entry.url == arg:
                    self.pending.remove(entry)
                    removed = entry
                    break

        if removed:
            # Also remove from all_urls tracking if it was only pending
            self.all_urls.discard(removed.url)
            self._forget_directory_for_url(removed.url)
            dir_msg = f" (dir: {removed.directory})" if removed.directory else ""
            print(f"{Fore.GREEN}REMOVED: Removed: {removed.url}{dir_msg}")
        else:
            print(f"{Fore.RED}Nothing removed")

        return False

    def do_clearqueue(self, arg: str) -> bool:
        """Clear pending list."""
        if self.dlq:
            print(f"{Fore.RED}Cannot clear - workers already running")
            return False

        count = len(self.pending)

        # Remove pending URLs from all_urls tracking
        for entry in self.pending:
            self.all_urls.discard(entry.url)
            self._forget_directory_for_url(entry.url)

        self.pending.clear()
        print(f"{Fore.GREEN}CLEARED: Cleared {count} pending URLs")
        return False

    def do_cls(self, arg: str) -> bool:
        """Clear the terminal output and reprint the header."""
        command = "cls" if os.name == "nt" else "clear"
        try:
            os.system(command)
        except Exception:
            print("\n" * 50)
        print(self.intro)
        return False

    def do_quit(self, arg: str) -> bool:
        """Quit the application."""
        print("SHUTTING DOWN...")

        if self.config.auto_save and (self.pending or self.dlq):
            self.do_save("")

        if self.dlq:
            print("WAITING: Waiting for active downloads to complete...")
            self.dlq.shutdown()

        self._stop_async_printer()
        print(f"{Fore.GREEN}COMPLETE: All downloads complete – goodbye!")
        return True

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D."""
        print()
        return self.do_quit(arg)

    # Help system ----------------------------------------------------------
    def do_help(self, arg: str) -> bool:
        if not arg:
            print(f"""
{Fore.CYAN}yt-dlp Queue - Command Sheet:{Style.RESET_ALL}

{Fore.YELLOW}Basic Operations:{Style.RESET_ALL}
    <url>                  Add URL to queue
    start                  Start/resume workers (aliases: run, go, begin)
    pause                  Pause all workers (aliases: hold, stop)
    resume                 Resume workers or reload last session (aliases: continue, unpause)
    cls                    Clear the terminal display (aliases: clear)
    quit                   Shutdown and exit (aliases: exit, q, x)

{Fore.YELLOW}Queue Management:{Style.RESET_ALL}
    pending                Show pending URLs (aliases: queue, pend, list)
    urls                   Show all tracked URLs and their status (aliases: links, all)
    remove <index|url>     Remove from pending (aliases: rm, del, delete)
    clearqueue             Clear pending list (aliases: clq, clear-queue, flush, reset)
    retry                  Re-queue all failed downloads (aliases: redo, rerun)
  
{Fore.YELLOW}Monitoring:{Style.RESET_ALL}
    status                 Show detailed status (aliases: stat, info)
    history [n]            Show last n downloads (default: 10) (aliases: hist, recent)
    worker <id> [lines]    Follow live stdout/stderr for a worker (default tail: 25) (aliases: log, tail, feed)
    workerdir [...]        Manage worker directories or set the session default directory (aliases: wd, dir)
  
{Fore.YELLOW}Configuration:{Style.RESET_ALL}
    config                 Show all settings (aliases: cfg, conf, settings)
    config <key>           Show specific setting
    config <key> <value>   Set configuration value (e.g. config auto_status true)
                           Useful keys: keep_failed, flat_directories, directory_template, dump_json
  
{Fore.YELLOW}Session Management:{Style.RESET_ALL}
    save [file]            Save current session (aliases: sv, write, export)
    load [file]            Load session from file (aliases: ld, restore, open)

Type {Fore.YELLOW}help <command>{Style.RESET_ALL} for detailed help on specific commands.
""")
        else:
            mapped = self.command_aliases.get(arg.strip().lower())
            if mapped:
                arg = mapped
            super().do_help(arg)
        return False


if __name__ == "__main__":
    try:
        InteractiveQueue().cmdloop()
    except (KeyboardInterrupt, BrokenPipeError):
        print(f"\n{Fore.YELLOW}Interrupted - goodbye!")
        sys.exit(0)
