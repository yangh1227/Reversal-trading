from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Dict, Iterator, Optional


_STATE_FILENAME = "interprocess_rate_limit.json"
_SENTINEL = b"#"
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_NEXT_AT: Dict[str, float] = {}


def _runtime_dir() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata).resolve() / "AltReversalTrader"
    return Path.home().resolve() / ".alt_reversal_trader"


def default_rate_limit_state_path() -> Path:
    return _runtime_dir() / _STATE_FILENAME


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_sentinel(handle) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(_SENTINEL)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


@contextmanager
def _locked_state_file(path: Path) -> Iterator[Optional[object]]:
    _ensure_parent(path)
    handle = None
    try:
        if not path.exists():
            path.write_bytes(_SENTINEL)
        handle = path.open("r+b")
        _ensure_sentinel(handle)
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield handle
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl
            except ImportError:
                yield None
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield handle
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if handle is not None:
            handle.close()


def _read_state(handle) -> Dict[str, float]:
    if handle is None:
        return {}
    handle.seek(1)
    payload = handle.read().decode("utf-8", errors="ignore").strip()
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return {str(key): float(value) for key, value in dict(data).items()}


def _write_state(handle, state: Dict[str, float]) -> None:
    if handle is None:
        return
    encoded = json.dumps(state, separators=(",", ":")).encode("utf-8")
    handle.seek(0)
    handle.write(_SENTINEL)
    handle.write(encoded)
    handle.truncate()
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def wait_for_request_slot(
    gate_name: str,
    min_interval_seconds: float,
    *,
    state_path: Path | None = None,
) -> None:
    gate_key = str(gate_name or "default")
    interval = max(0.0, float(min_interval_seconds))
    resolved_path = Path(state_path) if state_path is not None else default_rate_limit_state_path()
    with _locked_state_file(resolved_path) as handle:
        if handle is None:
            with _FALLBACK_LOCK:
                now = time.monotonic()
                next_allowed = float(_FALLBACK_NEXT_AT.get(gate_key, 0.0))
                delay = max(0.0, next_allowed - now)
                if delay > 0:
                    time.sleep(delay)
                    now = time.monotonic()
                _FALLBACK_NEXT_AT[gate_key] = now + interval
            return
        state = _read_state(handle)
        now = time.monotonic()
        next_allowed = float(state.get(gate_key, 0.0))
        delay = max(0.0, next_allowed - now)
        if delay > 0:
            time.sleep(delay)
            now = time.monotonic()
        state[gate_key] = now + interval
        _write_state(handle, state)


def reset_request_gate_for_tests(*, gate_name: str | None = None, state_path: Path | None = None) -> None:
    resolved_path = Path(state_path) if state_path is not None else default_rate_limit_state_path()
    with _locked_state_file(resolved_path) as handle:
        if handle is None:
            with _FALLBACK_LOCK:
                if gate_name is None:
                    _FALLBACK_NEXT_AT.clear()
                else:
                    _FALLBACK_NEXT_AT.pop(str(gate_name), None)
            return
        state = _read_state(handle)
        if gate_name is None:
            state = {}
        else:
            state.pop(str(gate_name), None)
        _write_state(handle, state)
