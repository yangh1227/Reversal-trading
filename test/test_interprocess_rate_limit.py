import multiprocessing as mp
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from alt_reversal_trader import interprocess_rate_limit
from alt_reversal_trader.interprocess_rate_limit import reset_request_gate_for_tests, wait_for_request_slot


def _gate_worker(state_path: str, ready, output) -> None:
    ready.wait(5.0)
    started = time.monotonic()
    wait_for_request_slot("binance_futures", 0.2, state_path=Path(state_path))
    finished = time.monotonic()
    output.put((started, finished))


def test_wait_for_request_slot_serializes_across_processes(tmp_path) -> None:
    state_path = tmp_path / "shared_gate.json"
    reset_request_gate_for_tests(state_path=state_path)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    output = ctx.Queue()
    processes = [
        ctx.Process(target=_gate_worker, args=(str(state_path), ready, output)),
        ctx.Process(target=_gate_worker, args=(str(state_path), ready, output)),
    ]
    for process in processes:
        process.start()
    ready.set()
    results = [output.get(timeout=10.0) for _ in processes]
    for process in processes:
        process.join(timeout=10.0)
        assert process.exitcode == 0

    finished_times = sorted(result[1] for result in results)
    assert finished_times[1] - finished_times[0] >= 0.15


def test_reset_request_gate_clears_previous_deadline(tmp_path) -> None:
    state_path = tmp_path / "shared_gate.json"
    wait_for_request_slot("binance_futures", 0.2, state_path=state_path)
    reset_request_gate_for_tests(gate_name="binance_futures", state_path=state_path)
    started = time.monotonic()
    wait_for_request_slot("binance_futures", 0.2, state_path=state_path)
    elapsed = time.monotonic() - started

    assert elapsed < 0.05


def test_wait_for_request_slot_serializes_within_process_threads(tmp_path) -> None:
    state_path = tmp_path / "shared_gate.json"
    reset_request_gate_for_tests(state_path=state_path)
    ready = threading.Event()
    finished_times: list[float] = []
    errors: list[Exception] = []

    def worker() -> None:
        ready.wait(5.0)
        try:
            wait_for_request_slot("binance_futures", 0.2, state_path=state_path)
        except Exception as exc:  # pragma: no cover - assertion checks after join
            errors.append(exc)
            return
        finished_times.append(time.monotonic())

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    ready.set()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors
    assert len(finished_times) == 2
    finished_times.sort()
    assert finished_times[1] - finished_times[0] >= 0.15


def test_locked_state_file_retries_windows_deadlock_error(tmp_path) -> None:
    state_path = tmp_path / "shared_gate.json"
    calls: list[int] = []

    def fake_locking(_fd: int, mode: int, _size: int) -> None:
        calls.append(mode)
        if len(calls) == 1:
            raise OSError(36, "Resource deadlock avoided")

    fake_msvcrt = SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=fake_locking)
    with patch.object(interprocess_rate_limit, "os", interprocess_rate_limit.os), patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
        with patch.object(interprocess_rate_limit.os, "name", "nt"), patch.object(
            interprocess_rate_limit, "_WINDOWS_LOCK_RETRY_SECONDS", 0.2
        ), patch.object(interprocess_rate_limit, "_WINDOWS_LOCK_RETRY_SLEEP_SECONDS", 0.0):
            with interprocess_rate_limit._locked_state_file(state_path) as handle:
                assert handle is not None

    assert calls.count(fake_msvcrt.LK_LOCK) >= 2
    assert calls[-1] == fake_msvcrt.LK_UNLCK
