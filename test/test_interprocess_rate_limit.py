import multiprocessing as mp
from pathlib import Path
import time

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
