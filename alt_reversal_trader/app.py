from __future__ import annotations

from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, List, Optional
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from lightweight_charts.widgets import QtChart
import websocket

from .binance_futures import BinanceFuturesClient, CandidateSymbol, PositionSnapshot, resample_ohlcv, resolve_base_interval
from .config import APP_INTERVAL_OPTIONS, CHART_ENGINE_OPTIONS, PARAMETER_SPECS, AppSettings, StrategySettings
from .crash_logger import log_runtime_event
from .optimizer import OptimizationResult, optimize_symbol_intervals
from .qt_compat import (
    HORIZONTAL,
    NO_EDIT_TRIGGERS,
    PASSWORD_ECHO,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
    VERTICAL,
    WEB_ATTR_FILE_URLS,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QThread,
    QTimer,
    QUrl,
    QVBoxLayout,
    QWebEngineView,
    QWidget,
    Signal,
)
from .strategy import (
    CHART_INDICATOR_COLUMNS,
    BacktestResult,
    compact_indicator_frame,
    compute_indicators,
    estimate_warmup_bars,
    prepare_ohlcv,
    run_backtest,
)


CHART_HISTORY_BAR_LIMIT = 8_000
CHART_HISTORY_FETCH_FALLBACK_MIN_BARS = 2_000
BACKTEST_WARMUP_BAR_FLOOR = 1_500
DEFAULT_CHART_LOOKBACK_HOURS = 3
DEFAULT_CHART_RIGHT_PAD_BARS = 4
LIVE_RENDER_INTERVAL_MS = 500
OPTIMIZED_TABLE_REFRESH_MS = 250
HISTORY_CACHE_SYMBOL_LIMIT = 10


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"unsupported interval: {interval}")


def _backtest_start_time_ms(settings: AppSettings) -> int:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    return now_ms - settings.history_days * 86_400_000


def _history_fetch_start_time_ms(settings: AppSettings, interval: Optional[str] = None) -> int:
    interval_ms = _interval_to_ms(interval or settings.kline_interval)
    warmup_bars = max(estimate_warmup_bars(settings.strategy), BACKTEST_WARMUP_BAR_FLOOR)
    return _backtest_start_time_ms(settings) - warmup_bars * interval_ms


def _ws_kline_timestamp(time_ms: int) -> pd.Timestamp:
    return pd.to_datetime(int(time_ms), unit="ms", utc=True).tz_convert(None)


def _merge_live_bar(df: pd.DataFrame, bar: Dict[str, object], max_rows: Optional[int] = None) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    frame = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=columns)
    row = {key: bar[key] for key in columns}
    if frame.empty:
        frame = pd.DataFrame([row], columns=columns)
    else:
        frame["time"] = pd.to_datetime(frame["time"])
        row_time = pd.Timestamp(row["time"])
        last_time = pd.Timestamp(frame["time"].iloc[-1])
        if last_time == row_time:
            for key in columns[1:]:
                frame.at[frame.index[-1], key] = row[key]
        elif last_time < row_time:
            frame = pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
        else:
            matches = frame["time"] == row_time
            if matches.any():
                idx = frame.index[matches][-1]
                for key in columns[1:]:
                    frame.at[idx, key] = row[key]
            else:
                frame = (
                    pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
                    .drop_duplicates(subset=["time"], keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)
    return frame


def _is_provisional_exit_trade(trade, latest_time: Optional[pd.Timestamp]) -> bool:
    if latest_time is None:
        return False
    return trade.reason == "end_of_test" and pd.Timestamp(trade.exit_time) == latest_time


class KlineStreamWorker(QThread):
    kline = Signal(object)
    status = Signal(str)

    def __init__(self, symbol: str, interval: str) -> None:
        super().__init__()
        self.symbol = symbol.upper()
        self.interval = interval
        self.stream_interval = resolve_base_interval(interval)
        self._stopped = False
        self._socket = None
        self._aggregate_bar: Optional[Dict[str, object]] = None

    def stop(self) -> None:
        self._stopped = True
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

    def run(self) -> None:
        stream_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_{self.stream_interval}"
        while not self._stopped:
            try:
                self.status.emit(f"실시간 스트림 연결: {self.symbol} {self.interval}")
                self._socket = websocket.create_connection(stream_url, timeout=10)
                self._socket.settimeout(1.0)
                while not self._stopped:
                    try:
                        raw = self._socket.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    if payload.get("e") != "kline":
                        continue
                    kline = payload.get("k", {})
                    bar = {
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "time": _ws_kline_timestamp(int(kline["t"])),
                        "open": float(kline["o"]),
                        "high": float(kline["h"]),
                        "low": float(kline["l"]),
                        "close": float(kline["c"]),
                        "volume": float(kline["v"]),
                        "closed": bool(kline.get("x", False)),
                    }
                    for event in self._transform_bar(bar):
                        self.kline.emit(event)
            except Exception as exc:
                if self._stopped:
                    break
                self.status.emit(f"실시간 스트림 재연결 대기: {self.symbol} ({exc})")
                self.msleep(2000)
            finally:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except Exception:
                        pass
                    self._socket = None
        self.status.emit(f"실시간 스트림 종료: {self.symbol}")

    def _transform_bar(self, bar: Dict[str, object]) -> List[Dict[str, object]]:
        if self.interval != "2m":
            return [bar]
        if not bool(bar.get("closed")):
            return []
        bucket_time = pd.Timestamp(bar["time"]).floor("2min")
        if self._aggregate_bar is None or pd.Timestamp(self._aggregate_bar["time"]) != bucket_time:
            self._aggregate_bar = {
                "symbol": self.symbol,
                "interval": self.interval,
                "time": bucket_time,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
                "closed": False,
            }
            return [dict(self._aggregate_bar)]

        self._aggregate_bar["high"] = max(float(self._aggregate_bar["high"]), float(bar["high"]))
        self._aggregate_bar["low"] = min(float(self._aggregate_bar["low"]), float(bar["low"]))
        self._aggregate_bar["close"] = float(bar["close"])
        self._aggregate_bar["volume"] = float(self._aggregate_bar["volume"]) + float(bar["volume"])
        self._aggregate_bar["closed"] = True
        completed = dict(self._aggregate_bar)
        self._aggregate_bar = None
        return [completed]


class ScanWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

    def run(self) -> None:
        try:
            client = BinanceFuturesClient()
            candidates = client.scan_alt_candidates(
                daily_volatility_min=self.settings.daily_volatility_min,
                quote_volume_min=self.settings.quote_volume_min,
                rsi_length=self.settings.rsi_length,
                rsi_lower=self.settings.rsi_lower,
                rsi_upper=self.settings.rsi_upper,
                workers=self.settings.scan_workers,
                log_callback=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                return
            self.completed.emit(candidates)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class OptimizeWorker(QThread):
    progress = Signal(str)
    result_ready = Signal(object)
    completed = Signal()
    failed = Signal(str)

    def __init__(self, settings: AppSettings, candidates: List[CandidateSymbol]) -> None:
        super().__init__()
        self.settings = settings
        self.candidates = candidates

    def _interval_candidates(self) -> List[str]:
        if self.settings.optimize_timeframe:
            return ["1m", "2m"]
        return [self.settings.kline_interval]

    def _load_histories(
        self,
        client: BinanceFuturesClient,
        symbol: str,
        history_fetch_start_time_ms: int,
    ) -> Dict[str, pd.DataFrame]:
        intervals = self._interval_candidates()
        primary_interval = self.settings.kline_interval if self.settings.kline_interval in intervals else intervals[0]
        fetch_interval = resolve_base_interval(primary_interval if len(intervals) == 1 else "1m")
        base_history = client.historical_ohlcv(symbol, fetch_interval, start_time=history_fetch_start_time_ms)
        if base_history.empty:
            return {}
        histories: Dict[str, pd.DataFrame] = {}
        for interval in intervals:
            histories[interval] = base_history if interval == fetch_interval else resample_ohlcv(base_history, interval)
        return histories

    def run(self) -> None:
        try:
            process_count = max(1, min(int(self.settings.optimize_processes), len(self.candidates) or 1))
            self.progress.emit(f"최적화 워커 시작: 종목 {len(self.candidates)}개, 프로세스 {process_count}개")
            if process_count <= 1 or len(self.candidates) <= 1:
                self._run_sequential()
            else:
                self._run_parallel(process_count)
            if not self.isInterruptionRequested():
                self.completed.emit()
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())

    def _run_sequential(self) -> None:
        client = BinanceFuturesClient()
        backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
        interval_candidates = self._interval_candidates()
        history_fetch_start_time_ms = _history_fetch_start_time_ms(
            self.settings,
            interval="1m" if "2m" in interval_candidates else self.settings.kline_interval,
        )
        for index, candidate in enumerate(self.candidates, start=1):
            if self.isInterruptionRequested():
                return
            self.progress.emit(
                f"[{index}/{len(self.candidates)}] {candidate.symbol} "
                f"{self.settings.history_days}일 백테스트 + 웜업 K라인 로드"
            )
            histories = self._load_histories(client, candidate.symbol, history_fetch_start_time_ms)
            if not histories:
                self.progress.emit(f"{candidate.symbol}: 히스토리 없음")
                continue
            optimization, best_history = optimize_symbol_intervals(
                symbol=candidate.symbol,
                histories_by_interval=histories,
                base_settings=self.settings.strategy,
                optimize_flags=self.settings.optimize_flags,
                interval_candidates=interval_candidates,
                span_pct=self.settings.optimization_span_pct,
                steps=self.settings.optimization_steps,
                max_combinations=self.settings.max_grid_combinations,
                fee_rate=self.settings.fee_rate,
                backtest_start_time=backtest_start_time,
                should_stop=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                return
            self.result_ready.emit({"candidate": candidate, "optimization": optimization, "history": best_history})
            self.progress.emit(
                f"{candidate.symbol} [{optimization.best_interval}]: {optimization.combinations_tested}개 조합 완료, "
                f"수익률 {optimization.best_backtest.metrics.total_return_pct:.2f}%"
            )

    def _run_parallel(self, process_count: int) -> None:
        client = BinanceFuturesClient()
        backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
        interval_candidates = self._interval_candidates()
        history_fetch_start_time_ms = _history_fetch_start_time_ms(
            self.settings,
            interval="1m" if "2m" in interval_candidates else self.settings.kline_interval,
        )
        pending_candidates = deque(self.candidates)
        active_jobs: List[tuple[CandidateSymbol, pd.DataFrame, object]] = []
        submitted = 0
        completed = 0
        pool = mp.get_context("spawn").Pool(processes=process_count, maxtasksperchild=8)
        terminated = False

        def terminate_pool() -> None:
            nonlocal terminated
            if terminated:
                return
            pool.terminate()
            pool.join()
            terminated = True

        def close_pool() -> None:
            nonlocal terminated
            if terminated:
                return
            pool.close()
            pool.join()
            terminated = True

        def submit_one(candidate: CandidateSymbol, index: int) -> bool:
            self.progress.emit(
                f"[{index}/{len(self.candidates)}] {candidate.symbol} "
                f"{self.settings.history_days}일 백테스트 + 웜업 K라인 로드"
            )
            histories = self._load_histories(client, candidate.symbol, history_fetch_start_time_ms)
            if not histories:
                self.progress.emit(f"{candidate.symbol}: 히스토리 없음")
                return False
            job = pool.apply_async(
                optimize_symbol_intervals,
                kwds={
                    "symbol": candidate.symbol,
                    "histories_by_interval": histories,
                    "base_settings": self.settings.strategy,
                    "optimize_flags": self.settings.optimize_flags,
                    "interval_candidates": interval_candidates,
                    "span_pct": self.settings.optimization_span_pct,
                    "steps": self.settings.optimization_steps,
                    "max_combinations": self.settings.max_grid_combinations,
                    "fee_rate": self.settings.fee_rate,
                    "backtest_start_time": backtest_start_time,
                },
            )
            default_history = histories.get(interval_candidates[0], next(iter(histories.values())))
            active_jobs.append((candidate, default_history, job))
            self.progress.emit(f"{candidate.symbol}: 프로세스 최적화 시작 ({len(active_jobs)}/{process_count})")
            return True

        try:
            while len(active_jobs) < process_count and pending_candidates:
                if self.isInterruptionRequested():
                    terminate_pool()
                    return
                candidate = pending_candidates.popleft()
                submitted += 1
                submit_one(candidate, submitted)

            while active_jobs:
                if self.isInterruptionRequested():
                    terminate_pool()
                    return
                remaining_jobs: List[tuple[CandidateSymbol, pd.DataFrame, object]] = []
                completed_any = False
                for candidate, df, job in active_jobs:
                    if not job.ready():
                        remaining_jobs.append((candidate, df, job))
                        continue
                    optimization, best_history = job.get()
                    completed_any = True
                    completed += 1
                    self.result_ready.emit({"candidate": candidate, "optimization": optimization, "history": best_history})
                    self.progress.emit(
                        f"[{completed}/{len(self.candidates)}] {candidate.symbol} [{optimization.best_interval}]: "
                        f"{optimization.combinations_tested}개 조합 완료, "
                        f"수익률 {optimization.best_backtest.metrics.total_return_pct:.2f}%"
                    )
                active_jobs = remaining_jobs

                while len(active_jobs) < process_count and pending_candidates:
                    if self.isInterruptionRequested():
                        terminate_pool()
                        return
                    candidate = pending_candidates.popleft()
                    submitted += 1
                    submit_one(candidate, submitted)

                if not completed_any:
                    self.msleep(50)
            close_pool()
        except Exception:
            terminate_pool()
            raise


class SymbolLoadWorker(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        request_id: int,
        settings: AppSettings,
        symbol: str,
        interval: str,
        history: Optional[pd.DataFrame],
        chart_history: Optional[pd.DataFrame],
        existing_backtest: Optional[BacktestResult],
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.settings = settings
        self.symbol = symbol
        self.interval = interval
        self.history = history
        self.chart_history = chart_history
        self.existing_backtest = existing_backtest

    def run(self) -> None:
        try:
            client = BinanceFuturesClient()
            history = self.history
            if history is None:
                history = client.historical_ohlcv(
                    self.symbol,
                    self.interval,
                    start_time=_history_fetch_start_time_ms(self.settings, self.interval),
                )
            if history.empty:
                raise RuntimeError(f"{self.symbol} 히스토리 데이터가 없습니다.")
            history = prepare_ohlcv(history)
            if self.isInterruptionRequested():
                return

            chart_history = self.chart_history
            if chart_history is None:
                if len(history) >= CHART_HISTORY_FETCH_FALLBACK_MIN_BARS:
                    chart_history = history.tail(CHART_HISTORY_BAR_LIMIT).copy().reset_index(drop=True)
                else:
                    chart_history = client.historical_ohlcv_recent(
                        self.symbol,
                        self.interval,
                        bars=CHART_HISTORY_BAR_LIMIT,
                    )
                    if chart_history.empty:
                        chart_history = history.tail(CHART_HISTORY_BAR_LIMIT).copy().reset_index(drop=True)
            chart_history = prepare_ohlcv(chart_history)
            if self.isInterruptionRequested():
                return

            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
            backtest = self.existing_backtest or run_backtest(
                history,
                settings=self.settings.strategy,
                fee_rate=self.settings.fee_rate,
                backtest_start_time=backtest_start_time,
            )
            if self.isInterruptionRequested():
                return
            chart_indicators = compact_indicator_frame(
                compute_indicators(chart_history, backtest.settings),
                CHART_INDICATOR_COLUMNS,
            )
            self.loaded.emit(
                {
                    "request_id": self.request_id,
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "history": history,
                    "chart_history": chart_history,
                    "backtest": backtest,
                    "chart_indicators": chart_indicators,
                }
            )
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())


class LiveBacktestWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        symbol: str,
        history: pd.DataFrame,
        chart_history: pd.DataFrame,
        strategy_settings: StrategySettings,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.symbol = symbol
        self.history = prepare_ohlcv(history.copy())
        self.chart_history = prepare_ohlcv(chart_history.copy())
        self.strategy_settings = strategy_settings

    def run(self) -> None:
        try:
            backtest = run_backtest(
                self.history,
                settings=self.strategy_settings,
                fee_rate=self.settings.fee_rate,
                backtest_start_time=pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms"),
            )
            if self.isInterruptionRequested():
                return
            chart_indicators = compact_indicator_frame(
                compute_indicators(self.chart_history, backtest.settings),
                CHART_INDICATOR_COLUMNS,
            )
            self.completed.emit(
                {
                    "symbol": self.symbol,
                    "history": self.history,
                    "chart_history": self.chart_history,
                    "backtest": backtest,
                    "chart_indicators": chart_indicators,
                }
            )
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())


class AccountInfoWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, request_id: int, api_key: str, api_secret: str, symbol: Optional[str]) -> None:
        super().__init__()
        self.request_id = request_id
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.symbol = symbol

    def run(self) -> None:
        try:
            client = BinanceFuturesClient(self.api_key, self.api_secret)
            balance = client.get_balance_snapshot()
            positions = client.get_open_positions()
            position = next((item for item in positions if item.symbol == self.symbol), None) if self.symbol else None
            self.completed.emit(
                {
                    "request_id": self.request_id,
                    "symbol": self.symbol,
                    "balance": balance,
                    "position": position,
                    "positions": positions,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class OrderWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        leverage: int,
        side: Optional[str] = None,
        fraction: Optional[float] = None,
        close_only: bool = False,
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.symbol = symbol
        self.leverage = int(leverage)
        self.side = side
        self.fraction = fraction
        self.close_only = close_only

    def run(self) -> None:
        try:
            client = BinanceFuturesClient(self.api_key, self.api_secret)
            if self.close_only:
                result = client.close_position(self.symbol)
                message = (
                    "청산할 포지션이 없습니다."
                    if result is None
                    else f"포지션 청산 완료: orderId={result.get('orderId')}"
                )
                self.completed.emit({"symbol": self.symbol, "message": message})
                return

            if self.side is None or self.fraction is None:
                raise RuntimeError("order parameters are incomplete")
            balance = client.get_balance_snapshot()
            margin = balance.available_balance * float(self.fraction)
            client.set_leverage(self.symbol, self.leverage)
            quantity = client.build_order_quantity(self.symbol, margin, self.leverage)
            result = client.place_market_order(self.symbol, self.side, quantity)
            self.completed.emit(
                {
                    "symbol": self.symbol,
                    "message": (
                        f"주문 완료: {self.symbol} {self.side} qty={quantity} "
                        f"orderId={result.get('orderId')}"
                    ),
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class AltReversalTraderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.public_client = BinanceFuturesClient()
        self.candidates: List[CandidateSymbol] = []
        self.optimized_results: Dict[str, OptimizationResult] = {}
        self.history_cache: Dict[str, pd.DataFrame] = {}
        self.chart_history_cache: Dict[str, pd.DataFrame] = {}
        self.price_precision_cache: Dict[str, int] = {}
        self.pending_candidates: List[CandidateSymbol] = []
        self.pending_optimized_results: Dict[str, OptimizationResult] = {}
        self.pending_history_cache: Dict[str, pd.DataFrame] = {}
        self.preserve_lists_during_refresh = False
        self.open_positions: List[PositionSnapshot] = []
        self.current_position_snapshot: Optional[PositionSnapshot] = None
        self.current_symbol: Optional[str] = None
        self.current_interval = self.settings.kline_interval
        self.current_backtest: Optional[BacktestResult] = None
        self.current_chart_indicators: Optional[pd.DataFrame] = None
        self.scan_worker: Optional[ScanWorker] = None
        self.optimize_worker: Optional[OptimizeWorker] = None
        self.load_worker: Optional[SymbolLoadWorker] = None
        self.live_backtest_worker: Optional[LiveBacktestWorker] = None
        self.account_worker: Optional[AccountInfoWorker] = None
        self.order_worker: Optional[OrderWorker] = None
        self.load_request_id = 0
        self.account_request_id = 0
        self.live_recalc_pending = False
        self.pending_account_refresh = False
        self.live_stream_worker: Optional[KlineStreamWorker] = None
        self.live_pending_bar: Optional[Dict[str, object]] = None
        self._tracked_threads: set[QThread] = set()
        self.auto_refresh_minutes = 10
        self.auto_refresh_timer = QTimer(self)
        self.live_update_timer = QTimer(self)
        self.optimized_table_timer = QTimer(self)
        self.parameter_editors: Dict[str, object] = {}
        self.parameter_opt_boxes: Dict[str, QCheckBox] = {}
        self.plotly_chart_path = Path("alt_reversal_trader_plotly_chart.html").resolve()
        self.plotly_js_path = self.plotly_chart_path.with_name("plotly.min.js")
        self.chart_mode = ""
        self.chart = None
        self.chart_view = None
        self.equity_subchart = None
        self.equity_line = None
        self.entry_price_line = None
        self.supertrend_line = None
        self.zone2_line = None
        self.zone3_line = None
        self.ema_fast_line = None
        self.ema_slow_line = None
        self.position_close_buttons: List[QPushButton] = []
        self.price_label_timer = QTimer(self)

        self.setWindowTitle("Binance Alt Mean Reversion Trader")
        self.resize(1680, 960)
        self._build_ui()
        self._apply_loaded_settings()
        self._init_chart()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self._flush_live_update)
        self.optimized_table_timer.setSingleShot(True)
        self.optimized_table_timer.timeout.connect(self._flush_optimized_table)
        self.price_label_timer.setInterval(1000)
        self.price_label_timer.timeout.connect(self._refresh_live_labels)
        self.price_label_timer.start()
        self.chart_engine_combo.currentTextChanged.connect(self.on_chart_engine_changed)
        self._init_auto_refresh()
        self.statusBar().showMessage("준비됨")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(HORIZONTAL)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_body = QWidget()
        settings_layout = QVBoxLayout(settings_body)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.addWidget(self._build_api_group())
        settings_layout.addWidget(self._build_filter_group())
        settings_layout.addWidget(self._build_optimization_group())
        settings_layout.addWidget(self._build_parameter_tabs())
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_body)
        left_layout.addWidget(settings_scroll, 3)

        actions_row = QHBoxLayout()
        self.save_settings_button = QPushButton("설정 저장")
        self.scan_button = QPushButton("후보 스캔 + 최적화")
        self.refresh_balance_button = QPushButton("잔고 새로고침")
        self.save_settings_button.clicked.connect(self.save_settings_with_feedback)
        self.scan_button.clicked.connect(self.run_scan_and_optimize)
        self.refresh_balance_button.clicked.connect(self.refresh_account_info)
        actions_row.addWidget(self.save_settings_button)
        actions_row.addWidget(self.scan_button)
        actions_row.addWidget(self.refresh_balance_button)
        left_layout.addLayout(actions_row)

        lower_split = QSplitter(VERTICAL)
        lower_split.addWidget(self._build_candidate_group())
        lower_split.addWidget(self._build_optimized_group())
        lower_split.addWidget(self._build_log_group())
        lower_split.setSizes([240, 240, 160])
        left_layout.addWidget(lower_split, 4)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        summary_group = QGroupBox("Backtest Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        summary_layout.addWidget(self.summary_box)
        right_layout.addWidget(summary_group, 2)

        self.chart_host = QWidget()
        self.chart_host_layout = QVBoxLayout(self.chart_host)
        self.chart_host_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.chart_host, 7)

        header_group = QGroupBox("Selection")
        header_layout = QVBoxLayout(header_group)
        self.symbol_label = QLabel("종목: -")
        self.signal_label = QLabel("신호: -")
        self.current_price_label = QLabel("현재가: -")
        self.balance_label = QLabel("잔고: API 미입력")
        self.position_label = QLabel("포지션: -")
        header_layout.addWidget(self.symbol_label)
        header_layout.addWidget(self.signal_label)
        header_layout.addWidget(self.current_price_label)
        header_layout.addWidget(self.balance_label)
        header_layout.addWidget(self.position_label)
        right_layout.addWidget(header_group)

        right_layout.addWidget(self._build_positions_group(), 2)

        order_group = QGroupBox("Live Order")
        order_layout = QVBoxLayout(order_group)
        long_row = QHBoxLayout()
        short_row = QHBoxLayout()
        self.long_buttons = []
        self.short_buttons = []
        for fraction, text in ((0.33, "LONG 33%"), (0.50, "LONG 50%"), (0.66, "LONG 66%"), (0.99, "LONG 99%")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, value=fraction: self.place_fractional_order("BUY", value))
            self.long_buttons.append(button)
            long_row.addWidget(button)
        for fraction, text in ((0.33, "SHORT 33%"), (0.50, "SHORT 50%"), (0.66, "SHORT 66%"), (0.99, "SHORT 99%")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, value=fraction: self.place_fractional_order("SELL", value))
            self.short_buttons.append(button)
            short_row.addWidget(button)
        self.close_position_button = QPushButton("포지션 청산")
        self.close_position_button.clicked.connect(self.close_selected_position)
        order_layout.addLayout(long_row)
        order_layout.addLayout(short_row)
        order_layout.addWidget(self.close_position_button)
        right_layout.addWidget(order_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([560, 1120])

    def _build_api_group(self) -> QGroupBox:
        group = QGroupBox("Binance API")
        layout = QFormLayout(group)
        self.api_key_edit = QLineEdit()
        self.api_secret_edit = QLineEdit()
        self.api_secret_edit.setEchoMode(PASSWORD_ECHO)
        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(1, 125)
        layout.addRow("API Key", self.api_key_edit)
        layout.addRow("Secret", self.api_secret_edit)
        layout.addRow("Leverage", self.leverage_spin)
        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Market Filters")
        layout = QFormLayout(group)
        self.daily_vol_spin = QDoubleSpinBox()
        self.daily_vol_spin.setRange(0.0, 500.0)
        self.daily_vol_spin.setDecimals(2)
        self.daily_vol_spin.setSingleStep(1.0)
        self.quote_volume_spin = QDoubleSpinBox()
        self.quote_volume_spin.setRange(0.0, 10_000_000_000.0)
        self.quote_volume_spin.setDecimals(0)
        self.quote_volume_spin.setSingleStep(100_000.0)
        self.rsi_length_spin = QSpinBox()
        self.rsi_length_spin.setRange(2, 100)
        self.rsi_lower_spin = QDoubleSpinBox()
        self.rsi_lower_spin.setRange(0.0, 100.0)
        self.rsi_lower_spin.setDecimals(1)
        self.rsi_upper_spin = QDoubleSpinBox()
        self.rsi_upper_spin.setRange(0.0, 100.0)
        self.rsi_upper_spin.setDecimals(1)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(APP_INTERVAL_OPTIONS)
        self.chart_engine_combo = QComboBox()
        self.chart_engine_combo.addItems(CHART_ENGINE_OPTIONS)
        self.history_days_spin = QSpinBox()
        self.history_days_spin.setRange(1, 30)
        self.scan_workers_spin = QSpinBox()
        self.scan_workers_spin.setRange(1, 24)
        layout.addRow("1일 변동성 % >=", self.daily_vol_spin)
        layout.addRow("24h 거래량 >=", self.quote_volume_spin)
        layout.addRow("1m RSI Length", self.rsi_length_spin)
        layout.addRow("1m RSI Lower <=", self.rsi_lower_spin)
        layout.addRow("1m RSI Upper >=", self.rsi_upper_spin)
        layout.addRow("백테스트 봉", self.interval_combo)
        layout.addRow("차트 엔진", self.chart_engine_combo)
        layout.addRow("히스토리 일수", self.history_days_spin)
        layout.addRow("스캔 워커", self.scan_workers_spin)
        return group

    def _build_optimization_group(self) -> QGroupBox:
        group = QGroupBox("Optimization")
        layout = QFormLayout(group)
        self.opt_span_spin = QDoubleSpinBox()
        self.opt_span_spin.setRange(1.0, 100.0)
        self.opt_span_spin.setDecimals(1)
        self.opt_span_spin.setSingleStep(1.0)
        self.opt_steps_spin = QSpinBox()
        self.opt_steps_spin.setRange(1, 9)
        self.max_combo_spin = QSpinBox()
        self.max_combo_spin.setRange(10, 20_000)
        self.opt_process_spin = QSpinBox()
        self.opt_process_spin.setRange(1, 16)
        self.optimize_timeframe_check = QCheckBox("1m / 2m 최적화")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 5.0)
        self.fee_spin.setDecimals(4)
        self.fee_spin.setSingleStep(0.01)
        layout.addRow("범위 ±%", self.opt_span_spin)
        layout.addRow("격자 단계수", self.opt_steps_spin)
        layout.addRow("최대 조합수", self.max_combo_spin)
        layout.addRow("최적화 프로세스", self.opt_process_spin)
        layout.addRow("타임프레임", self.optimize_timeframe_check)
        layout.addRow("수수료 %", self.fee_spin)
        return group

    def _build_parameter_tabs(self) -> QGroupBox:
        group = QGroupBox("Strategy Parameters")
        outer_layout = QVBoxLayout(group)
        help_label = QLabel("입력값이 기준 전략값이며, `Opt`를 체크한 항목만 현재 값 기준으로 ±최적화 범위를 탐색합니다.")
        help_label.setWordWrap(True)
        outer_layout.addWidget(help_label)
        tabs = QTabWidget()
        outer_layout.addWidget(tabs)

        forms: Dict[str, QFormLayout] = {}
        for key, title in (("core", "Core"), ("qip", "QIP"), ("qtp", "QTP"), ("switches", "Switches")):
            page = QWidget()
            form = QFormLayout(page)
            forms[key] = form
            tabs.addTab(page, title)

        for spec in PARAMETER_SPECS:
            editor = self._make_parameter_editor(spec)
            optimize_box = QCheckBox("Opt")
            optimize_box.setChecked(bool(self.settings.optimize_flags.get(spec.key, False)))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor)
            row_layout.addWidget(optimize_box)
            forms[spec.group].addRow(spec.label, row_widget)
            self.parameter_editors[spec.key] = editor
            self.parameter_opt_boxes[spec.key] = optimize_box
        return group

    def _make_parameter_editor(self, spec):
        current = getattr(self.settings.strategy, spec.key)
        if spec.kind == "bool":
            editor = QCheckBox()
            editor.setChecked(bool(current))
            return editor
        if spec.kind == "choice":
            editor = QComboBox()
            editor.addItems(list(spec.choices))
            if current in spec.choices:
                editor.setCurrentText(current)
            return editor
        if spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.minimum), int(spec.maximum))
            editor.setSingleStep(int(spec.step or 1))
            editor.setValue(int(current))
            return editor
        editor = QDoubleSpinBox()
        editor.setRange(float(spec.minimum), float(spec.maximum))
        editor.setSingleStep(float(spec.step or 0.1))
        editor.setDecimals(4 if float(spec.step or 0.1) < 0.1 else 2)
        editor.setValue(float(current))
        return editor

    def _build_candidate_group(self) -> QGroupBox:
        group = QGroupBox("후보 종목")
        layout = QVBoxLayout(group)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(["Symbol", "DayVol%", "24h Vol", "RSI1m", "24h%", "Price"])
        self.candidate_table.setSelectionBehavior(SELECT_ROWS)
        self.candidate_table.setSelectionMode(SINGLE_SELECTION)
        self.candidate_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.itemSelectionChanged.connect(self.on_candidate_selection_changed)
        self.candidate_table.cellClicked.connect(self.on_candidate_cell_clicked)
        layout.addWidget(self.candidate_table)
        return group

    def _build_optimized_group(self) -> QGroupBox:
        group = QGroupBox("최적화 종목")
        layout = QVBoxLayout(group)
        self.optimized_table = QTableWidget(0, 8)
        self.optimized_table.setHorizontalHeaderLabels(["Symbol", "TF", "Return%", "MDD%", "Trades", "Win%", "PF", "Grid"])
        self.optimized_table.setSelectionBehavior(SELECT_ROWS)
        self.optimized_table.setSelectionMode(SINGLE_SELECTION)
        self.optimized_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.optimized_table.horizontalHeader().setStretchLastSection(True)
        self.optimized_table.itemSelectionChanged.connect(self.on_optimized_selection_changed)
        self.optimized_table.cellClicked.connect(self.on_optimized_cell_clicked)
        layout.addWidget(self.optimized_table)
        return group

    def _build_positions_group(self) -> QGroupBox:
        group = QGroupBox("Open Positions")
        layout = QVBoxLayout(group)
        self.positions_table = QTableWidget(0, 6)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Side", "Amount", "Entry", "UPnL", "Action"])
        self.positions_table.setSelectionBehavior(SELECT_ROWS)
        self.positions_table.setSelectionMode(SINGLE_SELECTION)
        self.positions_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.positions_table)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("로그")
        layout = QVBoxLayout(group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return group

    def _init_chart(self) -> None:
        self._rebuild_chart_engine(force=True)

    def _clear_chart_host(self) -> None:
        while self.chart_host_layout.count():
            item = self.chart_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.chart = None
        self.chart_view = None
        self.equity_subchart = None
        self.equity_line = None
        self.entry_price_line = None
        self.supertrend_line = None
        self.zone2_line = None
        self.zone3_line = None
        self.ema_fast_line = None
        self.ema_slow_line = None

    def _rebuild_chart_engine(self, force: bool = False) -> None:
        engine = self.chart_engine_combo.currentText() if hasattr(self, "chart_engine_combo") else self.settings.chart_engine
        if not force and engine == self.chart_mode:
            return
        self._clear_chart_host()
        if engine == "Lightweight":
            self._init_lightweight_chart()
        else:
            self._init_plotly_chart()
        self.chart_mode = engine
        if self.current_symbol and self.current_backtest:
            self.render_chart(self.current_symbol, self.current_backtest)

    def _init_plotly_chart(self) -> None:
        self.chart_view = QWebEngineView(self.chart_host)
        self.chart_view.settings().setAttribute(WEB_ATTR_FILE_URLS, True)
        self._attach_webview_crash_logger(self.chart_view, "Plotly")
        self.chart_host_layout.addWidget(self.chart_view)
        self.chart_view.setHtml(self._empty_chart_html())

    def _init_lightweight_chart(self) -> None:
        self.chart = QtChart(self.chart_host)
        chart_webview = self.chart.get_webview()
        self._attach_webview_crash_logger(chart_webview, "Lightweight")
        self.chart_host_layout.addWidget(chart_webview)
        self.chart.layout(background_color="#0f1419", text_color="#eceff4", font_size=12, font_family="Consolas")
        self.chart.legend(True)
        self.chart.candle_style(
            up_color="#8d939a",
            down_color="rgba(0, 0, 0, 0)",
            border_up_color="#8d939a",
            border_down_color="#8d939a",
            wick_up_color="#8d939a",
            wick_down_color="#8d939a",
        )
        self.chart.run_script(f"""{self.chart.id}.series.applyOptions({{
            priceLineColor: "#22f202"
        }})""")
        self.chart.volume_config(up_color="rgba(216, 243, 220, 0.30)", down_color="rgba(255, 93, 115, 0.28)")
        self.chart.watermark("ALT MR", color="rgba(240, 242, 245, 0.16)")
        self.chart.crosshair(mode="normal")
        self.chart.time_scale(time_visible=True, seconds_visible=False, min_bar_spacing=0.01, right_offset=0)
        self.equity_subchart = self.chart.create_subchart(
            position="bottom",
            width=1.0,
            height=0.28,
            sync=True,
            scale_candles_only=True,
        )
        self.equity_subchart.layout(background_color="#121922", text_color="#eceff4", font_size=11, font_family="Consolas")
        self.equity_subchart.legend(True)
        self.equity_subchart.time_scale(time_visible=True, seconds_visible=False, min_bar_spacing=0.01, right_offset=0)
        self.equity_line = self.equity_subchart.create_line(
            "Equity",
            color="rgba(108, 245, 160, 0.9)",
            width=2,
            price_line=False,
            price_label=False,
        )
        self.supertrend_line = self.chart.create_line(
            "Supertrend",
            color="rgba(255, 225, 120, 0.95)",
            width=2,
            price_line=False,
            price_label=False,
        )
        self.zone2_line = self.chart.create_line(
            "Zone 2",
            color="rgba(255, 186, 73, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.zone3_line = self.chart.create_line(
            "Zone 3",
            color="rgba(255, 123, 123, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.ema_fast_line = self.chart.create_line(
            "EMA Fast",
            color="rgba(153, 229, 255, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.ema_slow_line = self.chart.create_line(
            "EMA Slow",
            color="rgba(255, 243, 176, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self._init_lightweight_bar_close_overlay()

    def _init_lightweight_bar_close_overlay(self) -> None:
        if self.chart is None:
            return
        self.chart.run_script(
            f"""
            (() => {{
                const handler = {self.chart.id};
                if (!handler || handler.barCloseCountdown) {{
                    return;
                }}
                const label = document.createElement("div");
                Object.assign(label.style, {{
                    position: "absolute",
                    right: "6px",
                    top: "0px",
                    display: "none",
                    padding: "2px 6px",
                    borderRadius: "3px",
                    background: "#22f202",
                    color: "#0f1419",
                    fontFamily: "Consolas, monospace",
                    fontSize: "11px",
                    fontWeight: "700",
                    lineHeight: "1.25",
                    letterSpacing: "0.02em",
                    zIndex: "3200",
                    pointerEvents: "none",
                    whiteSpace: "nowrap",
                    boxShadow: "0 1px 6px rgba(0, 0, 0, 0.35)",
                }});
                handler.div.appendChild(label);
                handler.barCloseCountdown = {{
                    label,
                    state: {{ text: "", price: null }},
                }};
                handler.updateBarCloseCountdown = (text, price) => {{
                    const overlay = handler.barCloseCountdown;
                    if (!overlay) {{
                        return;
                    }}
                    overlay.state = {{ text, price }};
                    if (!text || !Number.isFinite(price)) {{
                        overlay.label.style.display = "none";
                        return;
                    }}
                    const y = handler.series.priceToCoordinate(price);
                    if (y === null || y === undefined || !Number.isFinite(y)) {{
                        overlay.label.style.display = "none";
                        return;
                    }}
                    overlay.label.textContent = text;
                    overlay.label.style.display = "block";
                    const rawTop = Math.round(y) + 18;
                    const maxTop = Math.max(4, handler.div.clientHeight - overlay.label.offsetHeight - 4);
                    const top = Math.max(4, Math.min(rawTop, maxTop));
                    overlay.label.style.top = `${{top}}px`;
                }};
                handler.hideBarCloseCountdown = () => {{
                    const overlay = handler.barCloseCountdown;
                    if (!overlay) {{
                        return;
                    }}
                    overlay.state = {{ text: "", price: null }};
                    overlay.label.style.display = "none";
                }};
                handler.chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                    const overlay = handler.barCloseCountdown;
                    if (overlay && overlay.state.text) {{
                        handler.updateBarCloseCountdown(overlay.state.text, overlay.state.price);
                    }}
                }});
                window.addEventListener("resize", () => {{
                    const overlay = handler.barCloseCountdown;
                    if (overlay && overlay.state.text) {{
                        handler.updateBarCloseCountdown(overlay.state.text, overlay.state.price);
                    }}
                }});
            }})();
            """
        )

    def _set_lightweight_bar_close_overlay(self, countdown: Optional[str], price: Optional[float]) -> None:
        if self.chart_mode != "Lightweight" or self.chart is None:
            return
        if not countdown or price is None:
            self.chart.run_script(
                f"""
                if ({self.chart.id}.hideBarCloseCountdown) {{
                    {self.chart.id}.hideBarCloseCountdown();
                }}
                """
            )
            return
        self.chart.run_script(
            f"""
            if ({self.chart.id}.updateBarCloseCountdown) {{
                {self.chart.id}.updateBarCloseCountdown({json.dumps(countdown)}, {float(price)});
            }}
            """
        )

    def _attach_webview_crash_logger(self, webview, engine_label: str) -> None:
        try:
            page = webview.page()
            signal = getattr(page, "renderProcessTerminated", None)
            if signal is None:
                return
            signal.connect(
                lambda status, exit_code, label=engine_label: self._on_webview_render_crash(label, status, exit_code)
            )
        except Exception:
            pass

    def _on_webview_render_crash(self, engine_label: str, status, exit_code: int) -> None:
        body = "\n".join(
            [
                f"engine: {engine_label}",
                f"current_symbol: {self.current_symbol}",
                f"chart_mode: {self.chart_mode}",
                f"status: {status}",
                f"exit_code: {exit_code}",
            ]
        )
        path = log_runtime_event("WebEngine Render Crash", body, open_notepad=True)
        self.log(f"{engine_label} 렌더 프로세스 종료. 로그: {path}")

    def on_chart_engine_changed(self, _engine: str) -> None:
        self.settings.chart_engine = self.chart_engine_combo.currentText()
        self._rebuild_chart_engine()

    def _apply_loaded_settings(self) -> None:
        settings = self.settings
        self.api_key_edit.setText(settings.api_key)
        self.api_secret_edit.setText(settings.api_secret)
        self.leverage_spin.setValue(settings.leverage)
        self.daily_vol_spin.setValue(settings.daily_volatility_min)
        self.quote_volume_spin.setValue(settings.quote_volume_min)
        self.rsi_length_spin.setValue(settings.rsi_length)
        self.rsi_lower_spin.setValue(settings.rsi_lower)
        self.rsi_upper_spin.setValue(settings.rsi_upper)
        self.interval_combo.setCurrentText(settings.kline_interval)
        self.chart_engine_combo.setCurrentText(settings.chart_engine)
        self.history_days_spin.setValue(settings.history_days)
        self.scan_workers_spin.setValue(settings.scan_workers)
        self.opt_span_spin.setValue(settings.optimization_span_pct)
        self.opt_steps_spin.setValue(settings.optimization_steps)
        self.max_combo_spin.setValue(settings.max_grid_combinations)
        self.opt_process_spin.setValue(settings.optimize_processes)
        self.optimize_timeframe_check.setChecked(settings.optimize_timeframe)
        self.fee_spin.setValue(settings.fee_rate * 100.0)

    def collect_settings(self) -> AppSettings:
        strategy_payload: Dict[str, object] = {}
        for spec in PARAMETER_SPECS:
            editor = self.parameter_editors[spec.key]
            if spec.kind == "bool":
                strategy_payload[spec.key] = bool(editor.isChecked())
            elif spec.kind == "choice":
                strategy_payload[spec.key] = editor.currentText()
            elif spec.kind == "int":
                strategy_payload[spec.key] = int(editor.value())
            else:
                strategy_payload[spec.key] = float(editor.value())

        optimize_flags = {key: box.isChecked() for key, box in self.parameter_opt_boxes.items()}
        return AppSettings(
            api_key=self.api_key_edit.text().strip(),
            api_secret=self.api_secret_edit.text().strip(),
            chart_engine=self.chart_engine_combo.currentText(),
            leverage=int(self.leverage_spin.value()),
            fee_rate=float(self.fee_spin.value()) / 100.0,
            history_days=int(self.history_days_spin.value()),
            kline_interval=self.interval_combo.currentText(),
            daily_volatility_min=float(self.daily_vol_spin.value()),
            quote_volume_min=float(self.quote_volume_spin.value()),
            rsi_length=int(self.rsi_length_spin.value()),
            rsi_lower=float(self.rsi_lower_spin.value()),
            rsi_upper=float(self.rsi_upper_spin.value()),
            optimization_span_pct=float(self.opt_span_spin.value()),
            optimization_steps=int(self.opt_steps_spin.value()),
            max_grid_combinations=int(self.max_combo_spin.value()),
            scan_workers=int(self.scan_workers_spin.value()),
            optimize_processes=int(self.opt_process_spin.value()),
            optimize_timeframe=bool(self.optimize_timeframe_check.isChecked()),
            strategy=StrategySettings(**strategy_payload),
            optimize_flags=optimize_flags,
        )

    def _sync_settings(self, persist: bool = False) -> AppSettings:
        previous = self.settings
        self.settings = self.collect_settings()
        if not self.current_symbol:
            self.current_interval = self.settings.kline_interval
        if (
            previous.kline_interval != self.settings.kline_interval
            or previous.history_days != self.settings.history_days
        ):
            self._stop_load_worker()
            self._stop_live_backtest_worker()
            self._stop_live_stream()
            self.history_cache.clear()
            self.chart_history_cache.clear()
            self.current_chart_indicators = None
            self.price_precision_cache.clear()
        if persist:
            self.settings.save()
            self.public_client = BinanceFuturesClient()
        return self.settings

    def save_settings(self) -> AppSettings:
        return self._sync_settings(persist=True)

    def save_settings_with_feedback(self) -> None:
        self.save_settings()
        self.log("설정을 저장했습니다. 다음 실행 때 같은 값으로 불러옵니다.")

    def log(self, message: str) -> None:
        self.log_box.appendPlainText(message)
        self.statusBar().showMessage(message, 5000)

    def _candidate_by_symbol(self, symbol: str) -> Optional[CandidateSymbol]:
        return next((candidate for candidate in self.candidates if candidate.symbol == symbol), None)

    def _resolve_price_precision(self, symbol: str, candle_df: pd.DataFrame) -> int:
        cached = self.price_precision_cache.get(symbol)
        if cached is not None:
            return cached
        precision = 2
        try:
            filters = self.public_client.get_symbol_filters(symbol)
            precision = int(filters.get("pricePrecision", precision))
            tick_size = float(filters.get("tickSize", 0.0) or 0.0)
            if tick_size > 0:
                tick_text = f"{tick_size:.16f}".rstrip("0").rstrip(".")
                if "." in tick_text:
                    precision = max(precision, len(tick_text.split(".", 1)[1]))
        except Exception:
            close_series = candle_df["close"].dropna()
            if not close_series.empty:
                price = abs(float(close_series.iloc[-1]))
                if price < 0.001:
                    precision = 8
                elif price < 0.01:
                    precision = 7
                elif price < 0.1:
                    precision = 6
                elif price < 1:
                    precision = 5
                else:
                    precision = 2
        precision = max(2, min(precision, 10))
        self.price_precision_cache[symbol] = precision
        return precision

    def _apply_lightweight_precision(self, symbol: str, candle_df: pd.DataFrame) -> None:
        if self.chart is None:
            return
        precision = self._resolve_price_precision(symbol, candle_df)
        self.chart.precision(precision)
        for line in (
            self.supertrend_line,
            self.zone2_line,
            self.zone3_line,
            self.ema_fast_line,
            self.ema_slow_line,
        ):
            if line is not None:
                line.precision(precision)

    def _default_chart_time_range(self, candle_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        if candle_df.empty:
            now = pd.Timestamp.utcnow().tz_localize(None)
            return now - pd.Timedelta(hours=DEFAULT_CHART_LOOKBACK_HOURS), now
        bar_delta = pd.Timedelta(milliseconds=_interval_to_ms(self.current_interval or self.settings.kline_interval))
        latest_time = pd.Timestamp(candle_df["time"].iloc[-1])
        end_time = latest_time + (bar_delta * DEFAULT_CHART_RIGHT_PAD_BARS)
        start_floor = pd.Timestamp(candle_df["time"].iloc[0])
        start_time = max(start_floor, latest_time - pd.Timedelta(hours=DEFAULT_CHART_LOOKBACK_HOURS))
        return start_time, end_time

    def _stash_lightweight_range(self) -> None:
        if self.chart is None:
            return
        self.chart.run_script(
            f"""
            window.__alt_lwc_view_range = {self.chart.id}.chart.timeScale().getVisibleLogicalRange();
            """
        )

    def _restore_lightweight_range(self, symbol: str) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        self.chart.run_script(
            f"""
            const range = window.__alt_lwc_view_range;
            if (range && Number.isFinite(range.from) && Number.isFinite(range.to) && range.to > range.from) {{
                {self.chart.id}.chart.timeScale().setVisibleLogicalRange(range);
                {self.equity_subchart.id}.chart.timeScale().setVisibleLogicalRange(range);
            }}
            """
        )

    def _active_backtest_settings(self, symbol: str) -> StrategySettings:
        optimization = self.optimized_results.get(symbol)
        return optimization.best_backtest.settings if optimization else self.settings.strategy

    def _track_thread(self, worker: QThread, attr_name: str) -> None:
        self._tracked_threads.add(worker)

        def _cleanup_thread() -> None:
            if getattr(self, attr_name, None) is worker:
                setattr(self, attr_name, None)
            self._tracked_threads.discard(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup_thread)

    def _prune_caches(self) -> None:
        keep_symbols = set()
        if self.current_symbol:
            keep_symbols.add(self.current_symbol)
        ordered_symbols = [
            result.symbol
            for result in sorted(
                self.optimized_results.values(),
                key=lambda item: item.best_backtest.metrics.total_return_pct,
                reverse=True,
            )[:HISTORY_CACHE_SYMBOL_LIMIT]
        ]
        keep_symbols.update(ordered_symbols)
        self.history_cache = {
            symbol: frame
            for symbol, frame in self.history_cache.items()
            if symbol in keep_symbols
        }
        chart_keep = {self.current_symbol} if self.current_symbol else set()
        self.chart_history_cache = {
            symbol: frame
            for symbol, frame in self.chart_history_cache.items()
            if symbol in chart_keep
        }

    def _stop_scan_worker(self) -> None:
        worker = self.scan_worker
        self.scan_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(10000)

    def _stop_optimize_worker(self) -> None:
        worker = self.optimize_worker
        self.optimize_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(5000)

    def _stop_load_worker(self) -> None:
        worker = self.load_worker
        self.load_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _stop_live_backtest_worker(self) -> None:
        worker = self.live_backtest_worker
        self.live_backtest_worker = None
        self.live_recalc_pending = False
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _stop_account_worker(self) -> None:
        worker = self.account_worker
        self.account_worker = None
        self.pending_account_refresh = False
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _stop_order_worker(self) -> None:
        worker = self.order_worker
        self.order_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _drain_tracked_threads(self, timeout_ms: int = 15000) -> None:
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        while True:
            running_threads = [thread for thread in list(self._tracked_threads) if thread.isRunning()]
            if not running_threads:
                return
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            wait_ms = max(1, min(500, remaining_ms))
            for thread in running_threads:
                thread.wait(wait_ms)
            QApplication.processEvents()

        running_threads = [thread for thread in list(self._tracked_threads) if thread.isRunning()]
        if not running_threads:
            return

        thread_names = ", ".join(type(thread).__name__ for thread in running_threads)
        log_runtime_event(
            "Forced Thread Shutdown",
            f"Threads still running during close: {thread_names}",
            open_notepad=False,
        )
        for thread in running_threads:
            try:
                thread.terminate()
                thread.wait(1000)
            except Exception:
                pass

    def _stop_live_stream(self) -> None:
        worker = self.live_stream_worker
        self.live_stream_worker = None
        self.live_pending_bar = None
        if self.live_update_timer.isActive():
            self.live_update_timer.stop()
        if worker is not None:
            worker.stop()
            worker.wait(1500)

    def _start_live_stream(self, symbol: str) -> None:
        self._stop_live_stream()
        worker = KlineStreamWorker(symbol, self.current_interval or self.settings.kline_interval)
        worker.kline.connect(self._queue_live_update)
        worker.status.connect(self._on_live_stream_status)
        self.live_stream_worker = worker
        self._track_thread(worker, "live_stream_worker")
        worker.start()

    def _on_live_stream_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 3000)

    def _queue_live_update(self, payload: object) -> None:
        bar = dict(payload)
        if bar.get("symbol") != self.current_symbol:
            return
        self.live_pending_bar = bar
        self.live_update_timer.setInterval(LIVE_RENDER_INTERVAL_MS if self.chart_mode == "Lightweight" else 1000)
        if not self.live_update_timer.isActive():
            self.live_update_timer.start()

    def _flush_live_update(self) -> None:
        bar = self.live_pending_bar
        self.live_pending_bar = None
        if not bar or not self.current_symbol:
            return
        symbol = str(bar["symbol"])
        history = self.history_cache.get(symbol)
        chart_history = self.chart_history_cache.get(symbol)
        if history is None or chart_history is None:
            return

        self.history_cache[symbol] = _merge_live_bar(history, bar)
        self.chart_history_cache[symbol] = _merge_live_bar(chart_history, bar, max_rows=CHART_HISTORY_BAR_LIMIT)
        if not bool(bar.get("closed")):
            if self.chart_mode == "Lightweight":
                self._apply_live_lightweight_bar(symbol, bar)
            return
        self._schedule_live_backtest(symbol)

    def _apply_live_lightweight_bar(self, symbol: str, bar: Dict[str, object]) -> None:
        if symbol != self.current_symbol or self.chart_mode != "Lightweight" or self.chart is None:
            return
        series = pd.Series(
            {
                "time": pd.Timestamp(bar["time"]),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            }
        )
        try:
            self.chart.update(series)
        except Exception:
            pass
        self._refresh_live_labels()

    def _schedule_live_backtest(self, symbol: str) -> None:
        if symbol != self.current_symbol:
            return
        if self.live_backtest_worker and self.live_backtest_worker.isRunning():
            self.live_recalc_pending = True
            return
        history = self.history_cache.get(symbol)
        chart_history = self.chart_history_cache.get(symbol)
        if history is None or chart_history is None:
            return
        worker = LiveBacktestWorker(
            self.settings,
            symbol,
            history,
            chart_history,
            self._active_backtest_settings(symbol),
        )
        worker.completed.connect(self._on_live_backtest_completed)
        worker.failed.connect(self._on_live_backtest_failed)
        self.live_backtest_worker = worker
        self._track_thread(worker, "live_backtest_worker")
        worker.start()

    def _on_live_backtest_completed(self, payload: object) -> None:
        result = dict(payload)
        symbol = str(result["symbol"])
        if symbol != self.current_symbol:
            return
        self.history_cache[symbol] = result["history"]
        self.chart_history_cache[symbol] = result["chart_history"]
        self.current_backtest = result["backtest"]
        self.current_chart_indicators = result["chart_indicators"]
        self._prune_caches()
        self.render_chart(symbol, self.current_backtest, reset_view=False)
        self.update_summary(symbol, self.current_backtest, self.optimized_results.get(symbol))
        if self.live_recalc_pending:
            self.live_recalc_pending = False
            self._schedule_live_backtest(symbol)

    def _on_live_backtest_failed(self, message: str) -> None:
        self.live_recalc_pending = False
        self.log(message)

    def _set_order_buttons_enabled(self, enabled: bool) -> None:
        for button in self.long_buttons + self.short_buttons:
            button.setEnabled(enabled)
        self.close_position_button.setEnabled(enabled)
        self._set_position_close_buttons_enabled(enabled)

    def _set_position_close_buttons_enabled(self, enabled: bool) -> None:
        for button in self.position_close_buttons:
            button.setEnabled(enabled)

    def _clear_entry_price_overlay(self) -> None:
        if self.entry_price_line is None:
            return
        try:
            self.entry_price_line.delete()
        except Exception:
            pass
        self.entry_price_line = None

    def _update_entry_price_overlay(self) -> None:
        position = self.current_position_snapshot
        if (
            self.chart_mode != "Lightweight"
            or self.chart is None
            or position is None
            or position.symbol != self.current_symbol
        ):
            self._clear_entry_price_overlay()
            return
        frame = self.chart_history_cache.get(position.symbol)
        if frame is None or frame.empty:
            frame = self.history_cache.get(position.symbol)
        if frame is None or frame.empty:
            self._clear_entry_price_overlay()
            return
        precision = self._resolve_price_precision(position.symbol, frame)
        text = f"ENTRY {position.entry_price:.{precision}f}"
        if self.entry_price_line is None:
            self.entry_price_line = self.chart.horizontal_line(
                position.entry_price,
                color="#22f202",
                width=1,
                style="solid",
                text=text,
                axis_label_visible=True,
            )
        else:
            self.entry_price_line.update(position.entry_price)
            self.entry_price_line.options(color="#22f202", style="solid", width=1, text=text)

    def _refresh_live_labels(self) -> None:
        if not hasattr(self, "current_price_label"):
            return
        symbol = self.current_symbol
        if not symbol:
            self.current_price_label.setText("현재가: -")
            self._set_lightweight_bar_close_overlay(None, None)
            return

        frame = self.chart_history_cache.get(symbol)
        if frame is None or frame.empty:
            frame = self.history_cache.get(symbol)
        if frame is None or frame.empty:
            self.current_price_label.setText("현재가: -")
            self._set_lightweight_bar_close_overlay(None, None)
            return

        latest_price = float(frame["close"].iloc[-1])
        precision = self._resolve_price_precision(symbol, frame)
        self.current_price_label.setText(f"현재가: {latest_price:.{precision}f}")

        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
        interval = self.current_interval or self.settings.kline_interval
        value = int(interval[:-1])
        unit = interval[-1]
        if unit == "m":
            floor_freq = f"{value}min"
        elif unit == "h":
            floor_freq = f"{value}h"
        elif unit == "d":
            floor_freq = f"{value}d"
        else:
            self._set_lightweight_bar_close_overlay(None, None)
            return

        bar_start = now.floor(floor_freq)
        bar_end = bar_start + pd.Timedelta(milliseconds=_interval_to_ms(interval))
        remaining_seconds = max(0, int((bar_end - now).total_seconds()))
        minutes, seconds = divmod(remaining_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            countdown = f"{minutes:02d}:{seconds:02d}"
        self._set_lightweight_bar_close_overlay(countdown, latest_price)

    def update_positions_table(self) -> None:
        if not hasattr(self, "positions_table"):
            return
        for button in self.position_close_buttons:
            button.deleteLater()
        self.position_close_buttons.clear()
        self.positions_table.setRowCount(len(self.open_positions))
        for row, position in enumerate(self.open_positions):
            side = "LONG" if position.amount > 0 else "SHORT"
            entry_text = f"{position.entry_price:.8f}".rstrip("0").rstrip(".")
            values = [
                position.symbol,
                side,
                f"{abs(position.amount):.6f}",
                entry_text,
                f"{position.unrealized_pnl:.2f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, position.symbol)
                self.positions_table.setItem(row, col, item)
            button = QPushButton("청산")
            button.clicked.connect(lambda _=False, symbol=position.symbol: self.close_position_for_symbol(symbol))
            self.positions_table.setCellWidget(row, 5, button)
            self.position_close_buttons.append(button)
        self._set_position_close_buttons_enabled(self.order_worker is None or not self.order_worker.isRunning())

    def update_candidate_table(self) -> None:
        self.candidate_table.setUpdatesEnabled(False)
        self.candidate_table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            values = [
                candidate.symbol,
                f"{candidate.daily_volatility_pct:.2f}",
                f"{candidate.quote_volume:,.0f}",
                f"{candidate.rsi_1m:.2f}",
                f"{candidate.price_change_pct:.2f}",
                f"{candidate.last_price:.6f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, candidate.symbol)
                self.candidate_table.setItem(row, col, item)
        self.candidate_table.setUpdatesEnabled(True)

    def update_optimized_table(self) -> None:
        self.optimized_table.setUpdatesEnabled(False)
        ordered = sorted(
            self.optimized_results.values(),
            key=lambda result: result.best_backtest.metrics.total_return_pct,
            reverse=True,
        )
        self.optimized_table.setRowCount(len(ordered))
        for row, result in enumerate(ordered):
            metrics = result.best_backtest.metrics
            values = [
                result.symbol,
                result.best_interval or self.settings.kline_interval,
                f"{metrics.total_return_pct:.2f}",
                f"{metrics.max_drawdown_pct:.2f}",
                str(metrics.trade_count),
                f"{metrics.win_rate_pct:.1f}",
                f"{metrics.profit_factor:.2f}",
                str(result.combinations_tested),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, result.symbol)
                self.optimized_table.setItem(row, col, item)
        self.optimized_table.setUpdatesEnabled(True)

    def _schedule_optimized_table_refresh(self) -> None:
        if not self.optimized_table_timer.isActive():
            self.optimized_table_timer.start(OPTIMIZED_TABLE_REFRESH_MS)

    def _flush_optimized_table(self) -> None:
        self.update_optimized_table()

    def _init_auto_refresh(self) -> None:
        self.auto_refresh_timer.setInterval(self.auto_refresh_minutes * 60 * 1000)
        self.auto_refresh_timer.timeout.connect(self.run_auto_refresh)
        self.auto_refresh_timer.start()
        self.log(f"자동 갱신 활성화: {self.auto_refresh_minutes}분마다 스캔+최적화")

    def _set_refresh_running(self, is_running: bool) -> None:
        self.scan_button.setEnabled(not is_running)
        self.refresh_balance_button.setEnabled(not is_running)

    def _is_refresh_running(self) -> bool:
        return bool(
            (self.scan_worker and self.scan_worker.isRunning())
            or (self.optimize_worker and self.optimize_worker.isRunning())
        )

    def run_scan_and_optimize(self, preserve_existing: bool = False) -> None:
        if self._is_refresh_running():
            return
        self.save_settings()
        self._stop_scan_worker()
        self._stop_optimize_worker()
        self._stop_load_worker()
        self._stop_live_backtest_worker()
        self.preserve_lists_during_refresh = preserve_existing
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        if not preserve_existing:
            self._stop_live_stream()
            self.candidates = []
            self.optimized_results.clear()
            self.history_cache.clear()
            self.chart_history_cache.clear()
            self.current_symbol = None
            self.current_backtest = None
            self.current_chart_indicators = None
            self.update_candidate_table()
            self.update_optimized_table()
            self.summary_box.clear()
        self.log("후보 스캔 + 최적화 시작" + (" (기존 목록 유지)" if preserve_existing else ""))
        self._set_refresh_running(True)
        self.scan_worker = ScanWorker(self.settings)
        self._track_thread(self.scan_worker, "scan_worker")
        self.scan_worker.progress.connect(self.log)
        self.scan_worker.completed.connect(self.on_scan_completed)
        self.scan_worker.failed.connect(self.on_worker_failed)
        self.scan_worker.start()

    def on_scan_completed(self, candidates: object) -> None:
        self.pending_candidates = list(candidates)
        if not self.preserve_lists_during_refresh:
            self.candidates = list(self.pending_candidates)
            self.update_candidate_table()
        self.log(f"후보 스캔 완료: {len(self.pending_candidates)}개")
        if self.pending_candidates:
            if not self.preserve_lists_during_refresh:
                self.candidate_table.selectRow(0)
            self.start_optimization(self.pending_candidates)
            return
        if self.preserve_lists_during_refresh:
            self.log("새 후보가 없어 기존 목록을 유지합니다.")
        else:
            self.log("후보가 없어 최적화를 건너뜁니다.")
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.preserve_lists_during_refresh = False
        self._set_refresh_running(False)

    def start_optimization(self, targets: List[CandidateSymbol]) -> None:
        if self.optimize_worker and self.optimize_worker.isRunning():
            return
        if not targets:
            self._set_refresh_running(False)
            return
        self.log(f"최적화 시작: {len(targets)}개 종목")
        self.optimize_worker = OptimizeWorker(self.settings, targets)
        self._track_thread(self.optimize_worker, "optimize_worker")
        self.optimize_worker.progress.connect(self.log)
        self.optimize_worker.result_ready.connect(self.on_optimization_result)
        self.optimize_worker.completed.connect(self.on_optimization_completed)
        self.optimize_worker.failed.connect(self.on_worker_failed)
        self.optimize_worker.start()

    def on_optimization_result(self, payload: object) -> None:
        result = dict(payload)
        candidate: CandidateSymbol = result["candidate"]
        optimization: OptimizationResult = result["optimization"]
        history: pd.DataFrame = result["history"]
        if self.preserve_lists_during_refresh:
            self.pending_optimized_results[candidate.symbol] = optimization
            self.pending_history_cache[candidate.symbol] = history
            return
        self.optimized_results[candidate.symbol] = optimization
        self.history_cache[candidate.symbol] = history
        self._prune_caches()
        self._schedule_optimized_table_refresh()

    def on_optimization_completed(self) -> None:
        preserved_refresh = self.preserve_lists_during_refresh
        if preserved_refresh:
            if self.pending_candidates:
                self.candidates = list(self.pending_candidates)
                self.update_candidate_table()
            if self.pending_optimized_results:
                self.optimized_results = dict(self.pending_optimized_results)
                self._flush_optimized_table()
            self.history_cache.update(self.pending_history_cache)
            self.pending_candidates = []
            self.pending_optimized_results = {}
            self.pending_history_cache = {}
            self.preserve_lists_during_refresh = False
        self._prune_caches()
        self.log(f"최적화 완료: {len(self.optimized_results)}개")
        if not preserved_refresh:
            self._flush_optimized_table()
        if not preserved_refresh and self.optimized_table.rowCount() > 0:
            self.optimized_table.selectRow(0)
        self._set_refresh_running(False)

    def on_worker_failed(self, message: str) -> None:
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.preserve_lists_during_refresh = False
        log_runtime_event("Worker Failure", message, open_notepad=False)
        self.log(message)
        self._set_refresh_running(False)
        self.show_error(message)

    def run_auto_refresh(self) -> None:
        if self._is_refresh_running():
            self.log("자동 10분 갱신 시점이지만 이전 작업이 아직 실행 중이라 건너뜁니다.")
            return
        self.log("자동 10분 갱신 시작")
        self.run_scan_and_optimize(preserve_existing=True)

    def selected_candidate_symbols(self) -> List[str]:
        selected = self.candidate_table.selectedItems()
        if not selected:
            return []
        row = selected[0].row()
        symbol_item = self.candidate_table.item(row, 0)
        return [symbol_item.text()] if symbol_item else []

    def _request_symbol_load(self, symbol: str) -> None:
        if not symbol:
            return
        optimization = self.optimized_results.get(symbol)
        target_interval = (optimization.best_interval or self.settings.kline_interval) if optimization else self.settings.kline_interval
        if self.load_worker is not None and self.load_worker.isRunning():
            return
        if (
            symbol == self.current_symbol
            and target_interval == self.current_interval
            and self.current_backtest is not None
        ):
            self.render_chart(
                symbol,
                self.current_backtest,
                reset_view=True,
                chart_indicators=self.current_chart_indicators,
            )
            return
        self.load_symbol(symbol)

    def on_candidate_selection_changed(self) -> None:
        selected = self.candidate_table.selectedItems()
        if selected:
            self._request_symbol_load(selected[0].data(USER_ROLE) or selected[0].text())

    def on_candidate_cell_clicked(self, row: int, _column: int) -> None:
        item = self.candidate_table.item(row, 0)
        if item:
            self._request_symbol_load(item.data(USER_ROLE) or item.text())

    def on_optimized_selection_changed(self) -> None:
        selected = self.optimized_table.selectedItems()
        if selected:
            self._request_symbol_load(selected[0].data(USER_ROLE) or selected[0].text())

    def on_optimized_cell_clicked(self, row: int, _column: int) -> None:
        item = self.optimized_table.item(row, 0)
        if item:
            self._request_symbol_load(item.data(USER_ROLE) or item.text())

    def load_symbol(self, symbol: str) -> None:
        self._sync_settings()
        self._stop_live_stream()
        self._stop_live_backtest_worker()
        self._stop_load_worker()
        self.current_symbol = symbol
        optimization = self.optimized_results.get(symbol)
        target_interval = (optimization.best_interval or self.settings.kline_interval) if optimization else self.settings.kline_interval
        self.current_interval = target_interval
        self.current_backtest = None
        self.current_chart_indicators = None
        self.load_request_id += 1
        cached_history = self.history_cache.get(symbol) if target_interval == self.settings.kline_interval else None
        cached_chart_history = self.chart_history_cache.get(symbol) if target_interval == self.settings.kline_interval else None
        worker = SymbolLoadWorker(
            self.load_request_id,
            self.settings,
            symbol,
            target_interval,
            cached_history,
            cached_chart_history,
            optimization.best_backtest if optimization else None,
        )
        worker.loaded.connect(self._on_symbol_loaded)
        worker.failed.connect(self._on_symbol_load_failed)
        self.load_worker = worker
        self._track_thread(worker, "load_worker")
        self.statusBar().showMessage(f"{symbol} 로드 중...", 3000)
        worker.start()

    def _on_symbol_loaded(self, payload: object) -> None:
        result = dict(payload)
        request_id = int(result["request_id"])
        symbol = str(result["symbol"])
        if request_id != self.load_request_id or symbol != self.current_symbol:
            return
        self.current_interval = str(result.get("interval", self.current_interval))
        self.history_cache[symbol] = result["history"]
        self.chart_history_cache[symbol] = result["chart_history"]
        self.current_backtest = result["backtest"]
        self.current_chart_indicators = result["chart_indicators"]
        self._prune_caches()
        optimization = self.optimized_results.get(symbol)
        self.render_chart(symbol, self.current_backtest, reset_view=True, chart_indicators=self.current_chart_indicators)
        self.update_summary(symbol, self.current_backtest, optimization)
        self.refresh_account_info()
        self._start_live_stream(symbol)

    def _on_symbol_load_failed(self, message: str) -> None:
        self.show_error(message)

    def render_chart(
        self,
        symbol: str,
        result: BacktestResult,
        reset_view: bool = True,
        chart_indicators: Optional[pd.DataFrame] = None,
    ) -> None:
        if chart_indicators is None and symbol == self.current_symbol and self.current_chart_indicators is not None:
            chart_indicators = self.current_chart_indicators
        if chart_indicators is None:
            chart_history = self.chart_history_cache.get(symbol)
            if chart_history is not None and not chart_history.empty:
                chart_indicators = compact_indicator_frame(
                    compute_indicators(prepare_ohlcv(chart_history), result.settings),
                    CHART_INDICATOR_COLUMNS,
                )
            else:
                chart_indicators = result.indicators
        indicators = chart_indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        candle_df = indicators[["time", "open", "high", "low", "close", "volume"]].copy()
        equity_df = (
            pd.DataFrame({"time": list(result.equity_curve.index), "Equity": list(result.equity_curve.values)})
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        if self.chart_mode == "Lightweight":
            self._render_lightweight_chart(symbol, candle_df, indicators, equity_df, result.trades, reset_view=reset_view)
        else:
            self._render_plotly_chart(symbol, candle_df, indicators, equity_df, result.trades, reset_view=reset_view)

        candidate = self._candidate_by_symbol(symbol)
        latest = result.latest_state
        self.symbol_label.setText(
            f"종목: {symbol} | TF {self.current_interval}"
            + (f" | DayVol {candidate.daily_volatility_pct:.2f}% | RSI1m {candidate.rsi_1m:.2f}" if candidate else "")
        )
        self.signal_label.setText(
            f"신호: Trend {latest['trend']} | Zone {latest['zone']} | "
            f"Bull {latest['final_bull']} | Bear {latest['final_bear']} | RSI {latest['rsi']:.2f}"
        )
        self._update_entry_price_overlay()
        self._refresh_live_labels()

    def _render_plotly_chart(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        reset_view: bool = True,
    ) -> None:
        if self.chart_view is None:
            self._rebuild_chart_engine(force=True)
        if not self.plotly_js_path.exists() or self.plotly_js_path.stat().st_size == 0:
            self.plotly_js_path.write_text(get_plotlyjs(), encoding="utf-8")
        range_start, range_end = self._default_chart_time_range(candle_df) if reset_view else (None, None)
        html = self._plotly_chart_html(
            symbol,
            candle_df,
            indicators,
            equity_df,
            trades,
            range_start,
            range_end,
            reset_view=reset_view,
        )
        self.plotly_chart_path.write_text(html, encoding="utf-8")
        chart_url = QUrl.fromLocalFile(str(self.plotly_chart_path))
        chart_url.setQuery(f"ts={self.plotly_chart_path.stat().st_mtime_ns}")
        self.chart_view.load(chart_url)

    def _render_lightweight_chart(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        reset_view: bool = True,
    ) -> None:
        if self.chart is None:
            self._rebuild_chart_engine(force=True)
        if not reset_view:
            self._stash_lightweight_range()
        self.chart.set(candle_df)
        self._apply_lightweight_precision(symbol, candle_df)
        self.chart.clear_markers()
        self.supertrend_line.set(indicators[["time", "supertrend"]].rename(columns={"supertrend": "Supertrend"}))
        self.zone2_line.set(indicators[["time", "zone2_line"]].rename(columns={"zone2_line": "Zone 2"}))
        self.zone3_line.set(indicators[["time", "zone3_line"]].rename(columns={"zone3_line": "Zone 3"}))
        self.ema_fast_line.set(indicators[["time", "ema_fast"]].rename(columns={"ema_fast": "EMA Fast"}))
        self.ema_slow_line.set(indicators[["time", "ema_slow"]].rename(columns={"ema_slow": "EMA Slow"}))
        self.equity_line.set(equity_df)
        range_start, range_end = self._default_chart_time_range(candle_df)
        latest_time = pd.Timestamp(candle_df["time"].iloc[-1]) if not candle_df.empty else None

        markers = []
        for trade in trades:
            markers.append(
                {
                    "time": trade.entry_time,
                    "position": "below" if trade.side == "long" else "above",
                    "shape": "arrow_up" if trade.side == "long" else "arrow_down",
                    "color": "#17c964" if trade.side == "long" else "#f31260",
                    "text": trade.zones,
                }
            )
            if not _is_provisional_exit_trade(trade, latest_time):
                markers.append(
                    {
                        "time": trade.exit_time,
                        "position": "above" if trade.side == "long" else "below",
                        "shape": "circle",
                        "color": "#f801e8",
                        "text": f"{trade.return_pct:+.1f}%",
                    }
                )
        if markers:
            self.chart.marker_list(markers)
        if reset_view:
            QTimer.singleShot(
                140,
                lambda s=symbol, st=range_start, et=range_end: self._sync_lightweight_range(s, st, et),
            )
        else:
            QTimer.singleShot(140, lambda s=symbol: self._restore_lightweight_range(s))

    def _sync_lightweight_range(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        try:
            self.chart.set_visible_range(start_time, end_time)
            self.equity_subchart.set_visible_range(start_time, end_time)
        except Exception:
            self.chart.fit()
            self.equity_subchart.fit()

    def _empty_chart_html(self) -> str:
        return """
        <html>
          <head>
            <style>
              html, body {
                margin: 0;
                height: 100%;
                background: #0f1419;
                color: #dfe6eb;
                font-family: Consolas, monospace;
              }
              .wrap {
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 0.7;
                font-size: 15px;
              }
            </style>
          </head>
          <body>
            <div class="wrap">차트 대기 중</div>
          </body>
        </html>
        """

    def _plotly_chart_html(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        range_start: Optional[pd.Timestamp],
        range_end: Optional[pd.Timestamp],
        reset_view: bool = True,
    ) -> str:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            specs=[[{"secondary_y": True}], [{}]],
        )

        volume_colors = [
            "rgba(23, 201, 100, 0.35)" if close_ >= open_ else "rgba(243, 18, 96, 0.35)"
            for open_, close_ in zip(candle_df["open"], candle_df["close"])
        ]

        fig.add_trace(
            go.Candlestick(
                x=candle_df["time"],
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
                name=symbol,
                increasing_line_color="#17c964",
                increasing_fillcolor="#17c964",
                decreasing_line_color="#f31260",
                decreasing_fillcolor="#f31260",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=candle_df["time"],
                y=candle_df["volume"],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.35,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        for column, name, color, width in (
            ("supertrend", "Supertrend", "#ffcc00", 2),
            ("zone2_line", "Zone 2", "#ff9100", 1),
            ("zone3_line", "Zone 3", "#ff1744", 1),
            ("ema_fast", "EMA Fast", "#00e5ff", 1),
            ("ema_slow", "EMA Slow", "#ffd600", 1),
        ):
            fig.add_trace(
                go.Scatter(
                    x=indicators["time"],
                    y=indicators[column],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": width},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        latest_time = pd.Timestamp(candle_df["time"].iloc[-1]) if not candle_df.empty else None
        plotted_trades = [trade for trade in trades if not _is_provisional_exit_trade(trade, latest_time)]
        long_entries = [trade for trade in trades if trade.side == "long"]
        short_entries = [trade for trade in trades if trade.side == "short"]
        if long_entries:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time for trade in long_entries],
                    y=[trade.entry_price for trade in long_entries],
                    mode="markers+text",
                    name="Long Entry",
                    text=[trade.zones for trade in long_entries],
                    textposition="bottom center",
                    marker={"symbol": "triangle-up", "size": 11, "color": "#17c964"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if short_entries:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time for trade in short_entries],
                    y=[trade.entry_price for trade in short_entries],
                    mode="markers+text",
                    name="Short Entry",
                    text=[trade.zones for trade in short_entries],
                    textposition="top center",
                    marker={"symbol": "triangle-down", "size": 11, "color": "#f31260"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if plotted_trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time for trade in plotted_trades],
                    y=[trade.exit_price for trade in plotted_trades],
                    mode="markers+text",
                    name="Exit",
                    text=[f"{trade.return_pct:+.1f}%" for trade in plotted_trades],
                    textposition="middle right",
                    marker={"symbol": "x", "size": 9, "color": "#f801e8"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        position = self.current_position_snapshot
        if position is not None and position.symbol == self.current_symbol:
            fig.add_hline(
                y=position.entry_price,
                line_color="#22f202",
                line_width=1,
                annotation_text="ENTRY",
                annotation_position="top right",
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=equity_df["time"],
                y=equity_df["Equity"],
                mode="lines",
                name="Equity",
                line={"color": "#6cf5a0", "width": 2},
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            paper_bgcolor="#0f1419",
            plot_bgcolor="#0f1419",
            font={"family": "Consolas, monospace", "color": "#dfe6eb", "size": 12},
            height=max(self.chart_host.height(), 760),
            autosize=True,
            margin={"l": 48, "r": 36, "t": 36, "b": 36},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0.0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        xaxis_options = {
            "showgrid": True,
            "gridcolor": "rgba(120, 130, 150, 0.14)",
            "zeroline": False,
        }
        if range_start is not None and range_end is not None:
            xaxis_options["range"] = [range_start, range_end]
        fig.update_xaxes(**xaxis_options)
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(120, 130, 150, 0.14)",
            zeroline=False,
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(120, 130, 150, 0.14)",
            zeroline=False,
            row=2,
            col=1,
        )

        config = {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        }
        view_key = json.dumps(f"plotly_view::{symbol}::{self.current_interval or self.settings.kline_interval}")
        default_from = json.dumps(range_start.isoformat() if range_start is not None else None)
        default_to = json.dumps(range_end.isoformat() if range_end is not None else None)
        reset_literal = "true" if reset_view else "false"
        post_script = f"""
        const gd = document.getElementById('{{plot_id}}');
        const storageKey = {view_key};
        const defaultFrom = {default_from};
        const defaultTo = {default_to};
        const shouldReset = {reset_literal};
        const applyRange = (fromValue, toValue) => {{
            if (!fromValue || !toValue) return;
            Plotly.relayout(gd, {{
                'xaxis.range': [fromValue, toValue],
                'xaxis2.range': [fromValue, toValue]
            }});
        }};
        const saveRange = (fromValue, toValue) => {{
            if (!fromValue || !toValue) return;
            localStorage.setItem(storageKey, JSON.stringify({{from: fromValue, to: toValue}}));
        }};
        gd.on('plotly_relayout', (eventData) => {{
            const fromValue = eventData['xaxis.range[0]'] ?? eventData['xaxis2.range[0]'] ?? null;
            const toValue = eventData['xaxis.range[1]'] ?? eventData['xaxis2.range[1]'] ?? null;
            if (fromValue && toValue) {{
                saveRange(fromValue, toValue);
            }}
        }});
        if (shouldReset) {{
            applyRange(defaultFrom, defaultTo);
            saveRange(defaultFrom, defaultTo);
        }} else {{
            try {{
                const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
                if (saved && saved.from && saved.to) {{
                    applyRange(saved.from, saved.to);
                }}
            }} catch (error) {{}}
        }}
        """
        return pio.to_html(
            fig,
            full_html=True,
            include_plotlyjs="directory",
            config=config,
            default_width="100%",
            default_height=f"{max(self.chart_host.height(), 760)}px",
            post_script=post_script,
        )

    def update_summary(self, symbol: str, backtest: BacktestResult, optimization: Optional[OptimizationResult]) -> None:
        metrics = backtest.metrics
        lines = [
            f"Symbol: {symbol}",
            f"Return: {metrics.total_return_pct:.2f}%",
            f"Net Profit: {metrics.net_profit:.2f}",
            f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%",
            f"Trades: {metrics.trade_count}",
            f"Win Rate: {metrics.win_rate_pct:.1f}%",
            f"Profit Factor: {metrics.profit_factor:.2f}",
            "",
            "Latest State:",
        ]
        for key, value in backtest.latest_state.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("Active Params:")
        for spec in PARAMETER_SPECS:
            lines.append(f"- {spec.key}: {getattr(backtest.settings, spec.key)}")
        if optimization:
            lines.append("")
            lines.append(
                f"Best Interval: {optimization.best_interval or self.current_interval}"
            )
            lines.append(
                f"Optimization: {optimization.combinations_tested} combos"
                + (" (trimmed)" if optimization.trimmed_grid else "")
                + f", {optimization.duration_seconds:.2f}s"
            )
        self.summary_box.setPlainText("\n".join(lines))

    def refresh_account_info(self) -> None:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self._stop_account_worker()
            self.open_positions = []
            self.current_position_snapshot = None
            self.balance_label.setText("잔고: API 미입력")
            self.position_label.setText("포지션: API 미입력")
            self.update_positions_table()
            self._clear_entry_price_overlay()
            return
        if self.account_worker is not None and self.account_worker.isRunning():
            self.pending_account_refresh = True
            return
        self.account_request_id += 1
        worker = AccountInfoWorker(
            self.account_request_id,
            self.settings.api_key,
            self.settings.api_secret,
            self.current_symbol,
        )
        worker.completed.connect(self._on_account_info_completed)
        worker.failed.connect(self._on_account_info_failed)
        self.account_worker = worker
        self._track_thread(worker, "account_worker")
        self.pending_account_refresh = False
        self.refresh_balance_button.setEnabled(False)
        self.statusBar().showMessage("잔고 조회 중...", 3000)
        worker.start()

    def _on_account_info_completed(self, payload: object) -> None:
        result = dict(payload)
        if int(result["request_id"]) != self.account_request_id:
            return
        requested_symbol = result.get("symbol")
        balance = result["balance"]
        position = result["position"]
        self.open_positions = list(result.get("positions", []))
        self.current_position_snapshot = position if (self.current_symbol and requested_symbol == self.current_symbol) else None
        self.balance_label.setText(
            f"잔고: Equity {balance.equity:.2f} USDT | Available {balance.available_balance:.2f} USDT"
        )
        if self.current_symbol and requested_symbol == self.current_symbol:
            if position is None:
                self.position_label.setText("포지션: 없음")
            else:
                side = "LONG" if position.amount > 0 else "SHORT"
                self.position_label.setText(
                    f"포지션: {side} {position.amount:.6f} @ {position.entry_price:.6f} | "
                    f"UPnL {position.unrealized_pnl:.2f}"
                )
        else:
            self.position_label.setText("포지션: 종목 미선택")
        self.update_positions_table()
        if self.current_backtest and self.current_symbol:
            if self.chart_mode == "Lightweight":
                self._update_entry_price_overlay()
            else:
                self.render_chart(self.current_symbol, self.current_backtest, reset_view=False)
        if not self._is_refresh_running():
            self.refresh_balance_button.setEnabled(True)
        if self.pending_account_refresh:
            self.pending_account_refresh = False
            self.refresh_account_info()

    def _on_account_info_failed(self, message: str) -> None:
        self.open_positions = []
        self.current_position_snapshot = None
        self.balance_label.setText("잔고 조회 실패")
        self.position_label.setText("포지션: 조회 실패")
        self.update_positions_table()
        self._clear_entry_price_overlay()
        self.log(message)
        if not self._is_refresh_running():
            self.refresh_balance_button.setEnabled(True)
        if self.pending_account_refresh:
            self.pending_account_refresh = False
            self.refresh_account_info()

    def place_fractional_order(self, side: str, fraction: float) -> None:
        self._sync_settings()
        if not self.current_symbol:
            self.show_warning("주문할 종목을 먼저 선택하세요.")
            return
        if not self.settings.api_key or not self.settings.api_secret:
            self.show_warning("API Key / Secret을 입력해야 실제 주문할 수 있습니다.")
            return
        if self.order_worker is not None and self.order_worker.isRunning():
            self.show_warning("이미 주문 처리 중입니다.")
            return
        worker = OrderWorker(
            self.settings.api_key,
            self.settings.api_secret,
            self.current_symbol,
            self.settings.leverage,
            side=side,
            fraction=fraction,
        )
        worker.completed.connect(self._on_order_completed)
        worker.failed.connect(self._on_order_failed)
        self.order_worker = worker
        self._track_thread(worker, "order_worker")
        self._set_order_buttons_enabled(False)
        self.statusBar().showMessage(f"{self.current_symbol} 주문 처리 중...", 3000)
        worker.start()

    def close_selected_position(self) -> None:
        if not self.current_symbol:
            self.show_warning("종목을 먼저 선택하세요.")
            return
        self.close_position_for_symbol(self.current_symbol)

    def close_position_for_symbol(self, symbol: str) -> None:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self.show_warning("API Key / Secret을 입력해야 실제 주문할 수 있습니다.")
            return
        if self.order_worker is not None and self.order_worker.isRunning():
            self.show_warning("이미 주문 처리 중입니다.")
            return
        worker = OrderWorker(
            self.settings.api_key,
            self.settings.api_secret,
            symbol,
            self.settings.leverage,
            close_only=True,
        )
        worker.completed.connect(self._on_order_completed)
        worker.failed.connect(self._on_order_failed)
        self.order_worker = worker
        self._track_thread(worker, "order_worker")
        self._set_order_buttons_enabled(False)
        self.statusBar().showMessage(f"{symbol} 청산 처리 중...", 3000)
        worker.start()

    def _on_order_completed(self, payload: object) -> None:
        self._set_order_buttons_enabled(True)
        result = dict(payload)
        self.log(str(result.get("message", "주문 완료")))
        self.refresh_account_info()

    def _on_order_failed(self, message: str) -> None:
        self._set_order_buttons_enabled(True)
        self.log(message)
        self.show_error("주문 처리 중 오류가 발생했습니다. 로그를 확인하세요.")

    def show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Warning", message)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.auto_refresh_timer.stop()
            self.live_update_timer.stop()
            self.optimized_table_timer.stop()
            self._stop_scan_worker()
            self._stop_optimize_worker()
            self._stop_load_worker()
            self._stop_live_backtest_worker()
            self._stop_account_worker()
            self._stop_order_worker()
            self._stop_live_stream()
            self._drain_tracked_threads()
            self.save_settings()
        finally:
            super().closeEvent(event)


def create_app() -> QApplication:
    return QApplication.instance() or QApplication([])
