from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
import traceback
from typing import Optional, Tuple

import pandas as pd

from .auto_trade_runtime import resolve_latest_auto_trade_backtest
from .config import StrategySettings
from .strategy import BacktestResult


@dataclass(frozen=True)
class FavorableBacktestJob:
    symbol: str
    interval: str
    strategy_settings: StrategySettings
    history: pd.DataFrame
    seed_backtest: Optional[BacktestResult]
    fee_rate: float
    backtest_start_time: Optional[pd.Timestamp]
    history_signature: Tuple[object, ...]
    settings_signature: str


@dataclass(frozen=True)
class FavorableBacktestResultPayload:
    symbol: str
    interval: str
    backtest: Optional[BacktestResult]
    history_signature: Tuple[object, ...]
    settings_signature: str
    source: str
    error: str = ""


def run_favorable_backtest_process(command_queue: mp.Queue, result_queue: mp.Queue) -> None:
    while True:
        job = command_queue.get()
        if job is None:
            return
        if not isinstance(job, FavorableBacktestJob):
            continue
        try:
            resolution = resolve_latest_auto_trade_backtest(
                job.seed_backtest,
                job.history,
                job.strategy_settings,
                fee_rate=job.fee_rate,
                backtest_start_time=job.backtest_start_time,
            )
            result_queue.put(
                FavorableBacktestResultPayload(
                    symbol=job.symbol,
                    interval=job.interval,
                    backtest=resolution.backtest,
                    history_signature=job.history_signature,
                    settings_signature=job.settings_signature,
                    source=resolution.source,
                )
            )
        except Exception:
            result_queue.put(
                FavorableBacktestResultPayload(
                    symbol=job.symbol,
                    interval=job.interval,
                    backtest=None,
                    history_signature=job.history_signature,
                    settings_signature=job.settings_signature,
                    source="error",
                    error=traceback.format_exc(),
                )
            )


class FavorableBacktestProcess:
    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.command_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None

    def _ensure_started(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.command_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.process = self.ctx.Process(
            target=run_favorable_backtest_process,
            args=(self.command_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()

    def submit(self, job: FavorableBacktestJob) -> None:
        self._ensure_started()
        if self.command_queue is not None:
            self.command_queue.put(job)

    def drain_results(self) -> list[FavorableBacktestResultPayload]:
        if self.result_queue is None:
            return []
        results: list[FavorableBacktestResultPayload] = []
        while True:
            try:
                payload = self.result_queue.get_nowait()
            except Empty:
                break
            if isinstance(payload, FavorableBacktestResultPayload):
                results.append(payload)
        return results

    def stop(self) -> None:
        if self.command_queue is not None:
            try:
                self.command_queue.put(None)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
        self.process = None
        self.command_queue = None
        self.result_queue = None
