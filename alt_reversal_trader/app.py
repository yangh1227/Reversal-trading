from __future__ import annotations

from collections import deque
import json
import multiprocessing as mp
from numbers import Number
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from lightweight_charts.widgets import QtChart
import websocket

from .binance_futures import (
    BalanceSnapshot,
    BinanceFuturesClient,
    CandidateSymbol,
    PositionSnapshot,
    resample_ohlcv,
    resolve_base_interval,
)
from .config import APP_INTERVAL_OPTIONS, CHART_ENGINE_OPTIONS, PARAMETER_SPECS, AppSettings, StrategySettings
from .crash_logger import log_runtime_event
from .optimizer import OptimizationResult, optimization_sort_key, optimize_symbol_interval_results
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
    QProgressBar,
    QPushButton,
    QRadioButton,
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
    evaluate_latest_state,
    estimate_warmup_bars,
    prepare_ohlcv,
    resume_backtest,
    run_backtest,
)


CHART_HISTORY_BAR_LIMIT = 8_000
BACKTEST_WARMUP_BAR_FLOOR = 1_500
DEFAULT_CHART_LOOKBACK_HOURS = 3
DEFAULT_CHART_RIGHT_PAD_BARS = 4
INITIAL_CHART_WARMUP_BARS = 300
INITIAL_CHART_BAR_FLOOR = 360
CHART_LAZY_LOAD_CHUNK_BARS = 300
CHART_LAZY_LOAD_TRIGGER_BARS = 30
RECENT_DELTA_REFRESH_BARS = 180
RECENT_DELTA_OVERLAP_BARS = 20
LIVE_RENDER_INTERVAL_MS = 120
PLOTLY_LIVE_RENDER_INTERVAL_MS = 350
OPTIMIZED_TABLE_REFRESH_MS = 250
HISTORY_CACHE_SYMBOL_LIMIT = 10
RECENT_SYMBOL_CACHE_LIMIT = 8
FULL_HISTORY_REFRESH_COOLDOWN_SECONDS = 300.0
PERFORMANCE_LOG_THRESHOLD_MS = 100.0


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
    if (df is not None and "quote_volume" in df.columns) or "quote_volume" in bar:
        columns.append("quote_volume")
    frame = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=columns)
    row_time = pd.Timestamp(bar["time"])
    row_values = [
        row_time,
        float(bar["open"]),
        float(bar["high"]),
        float(bar["low"]),
        float(bar["close"]),
        float(bar["volume"]),
    ]
    if "quote_volume" in columns:
        row_values.append(float(bar.get("quote_volume", float(bar["close"]) * float(bar["volume"]))))
    if frame.empty:
        frame = pd.DataFrame([row_values], columns=columns)
    else:
        last_time = pd.Timestamp(frame["time"].iloc[-1])
        if last_time == row_time:
            for key, value in zip(columns[1:], row_values[1:]):
                frame.at[frame.index[-1], key] = value
        elif last_time < row_time:
            frame.loc[len(frame)] = row_values
        else:
            matches = frame["time"] == row_time
            if matches.any():
                idx = frame.index[matches][-1]
                for key, value in zip(columns[1:], row_values[1:]):
                    frame.at[idx, key] = value
            else:
                frame = (
                    pd.concat([frame, pd.DataFrame([row_values], columns=columns)], ignore_index=True)
                    .drop_duplicates(subset=["time"], keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)
    return frame


def _merge_ohlcv_frames(
    base: Optional[pd.DataFrame],
    updates: Optional[pd.DataFrame],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    columns: List[str] = []
    frames: List[pd.DataFrame] = []
    for frame in (base, updates):
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
        frames.append(frame)
    if not frames:
        default_columns = ["time", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=default_columns)
    merged = pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)
    if "time" in merged.columns:
        merged["time"] = pd.to_datetime(merged["time"])
        merged = merged.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    if max_rows is not None and len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].reset_index(drop=True)
    return merged


def _frame_tail_signature(frame: Optional[pd.DataFrame]) -> Tuple[object, ...]:
    if frame is None or frame.empty or "time" not in frame.columns:
        return (None, 0)
    last = frame.iloc[-1]
    values: List[object] = [pd.Timestamp(last["time"]), int(len(frame))]
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        if column not in frame.columns:
            continue
        value = last[column]
        values.append(None if pd.isna(value) else float(value))
    return tuple(values)


def _normalize_signature_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, Number):
        return round(float(value), 10)
    return value


def _frame_window_signature(
    frame: Optional[pd.DataFrame],
    columns: Optional[List[str]] = None,
    tail_rows: int = 8,
) -> Tuple[object, ...]:
    if frame is None or frame.empty:
        return (None, 0)
    selected_columns = [column for column in (columns or list(frame.columns)) if column in frame.columns]
    if not selected_columns:
        return (None, int(len(frame)))
    sample = pd.concat(
        [
            frame.loc[:, selected_columns].head(2),
            frame.loc[:, selected_columns].tail(max(int(tail_rows), 1)),
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)
    row_signatures = []
    for _, row in sample.iterrows():
        row_signatures.append(
            tuple(
                _normalize_signature_value(pd.Timestamp(row[column]) if column == "time" else row[column])
                for column in selected_columns
            )
        )
    return (int(len(frame)), *row_signatures)


def _refresh_cached_ohlcv(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    cached: Optional[pd.DataFrame],
    min_start_time_ms: int,
    max_rows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], bool]:
    if cached is None or cached.empty:
        return cached, False
    base = prepare_ohlcv(cached.copy())
    if base.empty:
        return base, False
    compare_base = base.tail(max_rows).reset_index(drop=True) if max_rows is not None else base
    interval_ms = _interval_to_ms(interval)
    last_time = pd.Timestamp(base["time"].iloc[-1])
    refresh_start_ms = max(
        int((last_time - pd.Timedelta(milliseconds=interval_ms * 2)).timestamp() * 1000),
        int(min_start_time_ms),
    )
    refreshed = client.historical_ohlcv(symbol, interval, start_time=refresh_start_ms)
    if refreshed.empty:
        return compare_base, False
    refreshed = prepare_ohlcv(refreshed)
    merged = _merge_ohlcv_frames(base, refreshed, max_rows=max_rows)
    return merged, _frame_tail_signature(merged) != _frame_tail_signature(compare_base)


def _refresh_cached_ohlcv_recent_delta(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    cached: Optional[pd.DataFrame],
    recent_bars: int = RECENT_DELTA_REFRESH_BARS,
    overlap_bars: int = RECENT_DELTA_OVERLAP_BARS,
    max_rows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], bool, bool]:
    if cached is None or cached.empty:
        return cached, False, True
    base = prepare_ohlcv(cached.copy())
    if base.empty:
        return base, False, True
    compare_base = base.tail(max_rows).reset_index(drop=True) if max_rows is not None else base
    fetch_bars = max(int(recent_bars), int(overlap_bars) + 2)
    refreshed = client.historical_ohlcv_recent(symbol, interval, bars=fetch_bars)
    if refreshed.empty:
        return compare_base, False, False
    refreshed = prepare_ohlcv(refreshed)
    base_tail = base.tail(max(fetch_bars + max(int(overlap_bars), 0), fetch_bars)).reset_index(drop=True)
    overlap_exists = bool(
        pd.to_datetime(refreshed["time"]).isin(pd.to_datetime(base_tail["time"])).any()
    )
    if not overlap_exists and not base_tail.empty:
        interval_delta = pd.Timedelta(milliseconds=_interval_to_ms(interval))
        latest_cached_time = pd.Timestamp(base_tail["time"].iloc[-1])
        earliest_refreshed_time = pd.Timestamp(refreshed["time"].iloc[0])
        gap_too_large = earliest_refreshed_time - latest_cached_time > interval_delta
        if gap_too_large:
            return compare_base, False, True
    merged = _merge_ohlcv_frames(base, refreshed, max_rows=max_rows)
    return merged, _frame_tail_signature(merged) != _frame_tail_signature(compare_base), False


def _initial_chart_bar_limit(interval: str) -> int:
    interval_ms = _interval_to_ms(interval)
    visible_bars = max(1, int((DEFAULT_CHART_LOOKBACK_HOURS * 3_600_000) // interval_ms))
    return max(INITIAL_CHART_BAR_FLOOR, visible_bars + INITIAL_CHART_WARMUP_BARS)


def _slice_recent_ohlcv(frame: Optional[pd.DataFrame], interval: str, max_bars: Optional[int] = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    limit = max_bars or _initial_chart_bar_limit(interval)
    prepared = prepare_ohlcv(frame.copy())
    if prepared.empty:
        return prepared
    return prepared.tail(limit).reset_index(drop=True)


def _ohlcv_from_indicator_frame(frame: Optional[pd.DataFrame], interval: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    columns = [column for column in ("time", "open", "high", "low", "close", "volume", "quote_volume") if column in frame.columns]
    if not columns:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    return _slice_recent_ohlcv(frame[columns], interval)


def _frame_matches_interval(frame: Optional[pd.DataFrame], interval: str) -> bool:
    if frame is None or frame.empty or "time" not in frame.columns or len(frame) < 2:
        return True
    diffs = pd.to_datetime(frame["time"]).sort_values().diff().dropna()
    if diffs.empty:
        return True
    expected = pd.Timedelta(milliseconds=_interval_to_ms(interval))
    modes = diffs.mode()
    observed = modes.iloc[0] if not modes.empty else diffs.iloc[-1]
    return abs(observed - expected) <= pd.Timedelta(seconds=1)


def _chart_indicators_from_backtest(
    backtest: BacktestResult,
    chart_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    indicators = backtest.indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    if chart_history is not None and not chart_history.empty and "time" in chart_history.columns:
        chart_times = pd.to_datetime(chart_history["time"])
        start_time = chart_times.iloc[0]
        end_time = chart_times.iloc[-1]
        sliced = indicators[
            (pd.to_datetime(indicators["time"]) >= start_time) & (pd.to_datetime(indicators["time"]) <= end_time)
        ]
        if not sliced.empty:
            indicators = sliced.reset_index(drop=True)
    if len(indicators) > CHART_HISTORY_BAR_LIMIT:
        indicators = indicators.tail(CHART_HISTORY_BAR_LIMIT).reset_index(drop=True)
    return compact_indicator_frame(indicators, CHART_INDICATOR_COLUMNS)


def _backtest_matches_history(backtest: Optional[BacktestResult], history: Optional[pd.DataFrame]) -> bool:
    if backtest is None or history is None or history.empty or backtest.indicators.empty:
        return False
    return pd.Timestamp(backtest.indicators["time"].iloc[-1]) == pd.Timestamp(history["time"].iloc[-1])


def _history_can_resume_backtest(backtest: Optional[BacktestResult], history: Optional[pd.DataFrame]) -> bool:
    if backtest is None or history is None or history.empty or backtest.indicators.empty or backtest.cursor is None:
        return False
    history_times = pd.to_datetime(history["time"])
    backtest_last_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    return bool((history_times == backtest_last_time).any()) and pd.Timestamp(history_times.iloc[-1]) > backtest_last_time


def _is_provisional_exit_trade(trade, latest_time: Optional[pd.Timestamp]) -> bool:
    if latest_time is None:
        return False
    return trade.reason == "end_of_test" and pd.Timestamp(trade.exit_time) == latest_time


def _latest_backtest_exit_event(backtest: Optional[BacktestResult]) -> Optional[Dict[str, object]]:
    if backtest is None or backtest.indicators.empty or not backtest.trades or "time" not in backtest.indicators.columns:
        return None
    latest_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    for trade in reversed(backtest.trades):
        exit_time = pd.Timestamp(trade.exit_time)
        if exit_time < latest_time:
            break
        if exit_time != latest_time or _is_provisional_exit_trade(trade, latest_time):
            continue
        return {
            "side": str(trade.side).lower(),
            "reason": str(trade.reason),
            "bar_time": latest_time,
        }
    return None


def _auto_close_reason(position: Optional[PositionSnapshot], exit_event: Optional[Dict[str, object]]) -> Optional[str]:
    if position is None or not exit_event:
        return None
    trade_side = str(exit_event.get("side", "")).lower()
    if trade_side not in {"long", "short"}:
        return None
    if trade_side == "long" and position.amount <= 0:
        return None
    if trade_side == "short" and position.amount >= 0:
        return None
    reason = str(exit_event.get("reason", "")).strip()
    if not reason or reason == "end_of_test":
        return None
    return reason


def _auto_close_reason_text(reason: str) -> str:
    labels = {
        "trend_to_long": "추세 전환 LONG",
        "trend_to_short": "추세 전환 SHORT",
        "opposite_signal": "반대 신호",
    }
    return labels.get(reason, reason)


def _preview_exit_reason(position_qty: float, latest_state: Dict[str, object]) -> Optional[str]:
    if position_qty > 0:
        if bool(latest_state.get("trend_to_short")):
            return "trend_to_short"
        if bool(latest_state.get("final_bear")):
            return "opposite_signal"
        return None
    if position_qty < 0:
        if bool(latest_state.get("trend_to_long")):
            return "trend_to_long"
        if bool(latest_state.get("final_bull")):
            return "opposite_signal"
    return None


def _preview_entry_signal(
    cursor,
    latest_state: Dict[str, object],
    settings: StrategySettings,
) -> Optional[Tuple[str, int]]:
    if cursor is None:
        return None
    position_qty = float(getattr(cursor, "position_qty", 0.0))
    long_zone_used = list(getattr(cursor, "long_zone_used", (False, False, False)))
    short_zone_used = list(getattr(cursor, "short_zone_used", (False, False, False)))
    last_long_zone = int(getattr(cursor, "last_long_zone", 0))
    last_short_zone = int(getattr(cursor, "last_short_zone", 0))

    def reset_zones() -> None:
        nonlocal last_long_zone, last_short_zone
        long_zone_used[:] = [False, False, False]
        short_zone_used[:] = [False, False, False]
        last_long_zone = 0
        last_short_zone = 0

    if abs(position_qty) < 1e-12:
        reset_zones()

    if _preview_exit_reason(position_qty, latest_state) is not None:
        position_qty = 0.0
        reset_zones()

    if abs(position_qty) < 1e-12:
        reset_zones()

    lev_zone = int(latest_state.get("zone") or 0)
    if lev_zone not in {1, 2, 3}:
        return None

    trend = str(latest_state.get("trend", "")).upper()
    is_long_trend = trend == "LONG"
    is_short_trend = trend == "SHORT"
    final_bull = bool(latest_state.get("final_bull"))
    final_bear = bool(latest_state.get("final_bear"))

    can_long_z1 = (
        (not settings.beast_mode)
        and is_long_trend
        and final_bull
        and lev_zone == 1
        and (not long_zone_used[0])
        and last_long_zone == 0
    )
    can_long_z2 = (
        is_long_trend
        and final_bull
        and lev_zone == 2
        and (not long_zone_used[1])
        and last_long_zone in (0, 1)
    )
    can_long_z3 = (
        is_long_trend
        and final_bull
        and lev_zone == 3
        and (not long_zone_used[2])
        and last_long_zone in (0, 2)
    )
    can_short_z1 = (
        (not settings.beast_mode)
        and is_short_trend
        and final_bear
        and lev_zone == 1
        and (not short_zone_used[0])
        and last_short_zone == 0
    )
    can_short_z2 = (
        is_short_trend
        and final_bear
        and lev_zone == 2
        and (not short_zone_used[1])
        and last_short_zone in (0, 1)
    )
    can_short_z3 = (
        is_short_trend
        and final_bear
        and lev_zone == 3
        and (not short_zone_used[2])
        and last_short_zone in (0, 2)
    )

    if can_long_z1:
        return ("long", 1)
    if can_long_z2:
        return ("long", 2)
    if can_long_z3:
        return ("long", 3)
    if can_short_z1:
        return ("short", 1)
    if can_short_z2:
        return ("short", 2)
    if can_short_z3:
        return ("short", 3)
    return None


def _position_return_pct(position: PositionSnapshot) -> float:
    notional = abs(float(position.amount)) * float(position.entry_price)
    leverage = max(1, int(position.leverage) or 1)
    margin = notional / leverage if notional > 0 else 0.0
    if margin <= 0:
        return 0.0
    return float(position.unrealized_pnl) / margin * 100.0


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
                        "quote_volume": float(kline.get("q", 0.0) or 0.0),
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

        bar_time = pd.Timestamp(bar["time"])
        bucket_time = bar_time.floor("2min")
        is_first_minute = bar_time == bucket_time

        if is_first_minute:
            self._aggregate_bar = {
                "symbol": self.symbol,
                "interval": self.interval,
                "time": bucket_time,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
                "base_volume": float(bar["volume"]),
                "closed": False,
            }
            return [{key: value for key, value in self._aggregate_bar.items() if key != "base_volume"}]

        if self._aggregate_bar is None or pd.Timestamp(self._aggregate_bar["time"]) != bucket_time:
            provisional = dict(bar)
            provisional["time"] = bucket_time
            provisional["closed"] = False
            return [provisional]

        self._aggregate_bar["high"] = max(float(self._aggregate_bar["high"]), float(bar["high"]))
        self._aggregate_bar["low"] = min(float(self._aggregate_bar["low"]), float(bar["low"]))
        self._aggregate_bar["close"] = float(bar["close"])
        self._aggregate_bar["volume"] = float(self._aggregate_bar["base_volume"]) + float(bar["volume"])
        self._aggregate_bar["closed"] = bool(bar.get("closed", False))
        completed = {key: value for key, value in self._aggregate_bar.items() if key != "base_volume"}
        if completed["closed"]:
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
                use_rsi_filter=self.settings.use_rsi_filter,
                rsi_length=self.settings.rsi_length,
                rsi_lower=self.settings.rsi_lower,
                rsi_upper=self.settings.rsi_upper,
                use_atr_4h_filter=self.settings.use_atr_4h_filter,
                atr_4h_min_pct=self.settings.atr_4h_min_pct,
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
    case_plan = Signal(object)
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
            self.case_plan.emit({"candidate": candidate.symbol, "cases": len(histories)})
            interval_results = optimize_symbol_interval_results(
                symbol=candidate.symbol,
                histories_by_interval=histories,
                base_settings=self.settings.strategy,
                optimize_flags=self.settings.optimize_flags,
                interval_candidates=interval_candidates,
                span_pct=self.settings.optimization_span_pct,
                steps=self.settings.optimization_steps,
                max_combinations=self.settings.max_grid_combinations,
                fee_rate=self.settings.fee_rate,
                rank_mode=self.settings.optimization_rank_mode,
                backtest_start_time=backtest_start_time,
                should_stop=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                return
            for optimization, history in interval_results:
                if self.isInterruptionRequested():
                    return
                self.result_ready.emit({"candidate": candidate, "optimization": optimization, "history": history})
                self.progress.emit(
                    f"{candidate.symbol} [{optimization.best_interval}]: {optimization.combinations_tested}개 조합 완료, "
                    f"총점 {optimization.score:.1f} | 수익률 {optimization.best_backtest.metrics.total_return_pct:.2f}%"
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
        active_jobs: List[tuple[CandidateSymbol, object]] = []
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
            self.case_plan.emit({"candidate": candidate.symbol, "cases": len(histories)})
            job = pool.apply_async(
                optimize_symbol_interval_results,
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
                    "rank_mode": self.settings.optimization_rank_mode,
                        "backtest_start_time": backtest_start_time,
                    },
                )
            active_jobs.append((candidate, job))
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
                remaining_jobs: List[tuple[CandidateSymbol, object]] = []
                completed_any = False
                for candidate, job in active_jobs:
                    if not job.ready():
                        remaining_jobs.append((candidate, job))
                        continue
                    interval_results = job.get()
                    completed_any = True
                    completed += 1
                    for optimization, history in interval_results:
                        self.result_ready.emit({"candidate": candidate, "optimization": optimization, "history": history})
                        self.progress.emit(
                            f"[{completed}/{len(self.candidates)}] {candidate.symbol} [{optimization.best_interval}]: "
                            f"{optimization.combinations_tested}개 조합 완료, "
                            f"총점 {optimization.score:.1f} | 수익률 {optimization.best_backtest.metrics.total_return_pct:.2f}%"
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
        history_last_refresh_at: Optional[float],
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.settings = settings
        self.symbol = symbol
        self.interval = interval
        self.history = history
        self.chart_history = chart_history
        self.existing_backtest = existing_backtest
        self.history_last_refresh_at = history_last_refresh_at

    def run(self) -> None:
        try:
            client = BinanceFuturesClient()
            fetch_started_at = time.perf_counter()
            history = self.history
            history_updated = False
            refreshed_at = float(self.history_last_refresh_at or 0.0)
            recent_delta_used = False
            repair_used = False
            if history is None:
                history = client.historical_ohlcv(
                    self.symbol,
                    self.interval,
                    start_time=_history_fetch_start_time_ms(self.settings, self.interval),
                )
                history_updated = True
                refreshed_at = time.time()
            else:
                latest_bar_time = pd.Timestamp(history["time"].iloc[-1]) if not history.empty else None
                latest_bar_stale = (
                    latest_bar_time is None
                    or (pd.Timestamp.utcnow().tz_localize(None) - latest_bar_time)
                    >= pd.Timedelta(milliseconds=_interval_to_ms(self.interval) * 2)
                )
                should_refresh = (
                    latest_bar_stale
                    or
                    self.history_last_refresh_at is None
                    or (time.time() - float(self.history_last_refresh_at)) >= FULL_HISTORY_REFRESH_COOLDOWN_SECONDS
                )
                if should_refresh:
                    history, history_updated, repair_needed = _refresh_cached_ohlcv_recent_delta(
                        client,
                        self.symbol,
                        self.interval,
                        history,
                    )
                    recent_delta_used = True
                    if repair_needed:
                        history, repair_updated = _refresh_cached_ohlcv(
                            client,
                            self.symbol,
                            self.interval,
                            history,
                            _history_fetch_start_time_ms(self.settings, self.interval),
                        )
                        history_updated = history_updated or repair_updated
                        repair_used = True
                    refreshed_at = time.time()
                else:
                    history = prepare_ohlcv(history.copy())
            if history.empty:
                raise RuntimeError(f"{self.symbol} 히스토리 데이터가 없습니다.")
            history = prepare_ohlcv(history)
            if self.isInterruptionRequested():
                return
            fetch_elapsed_ms = (time.perf_counter() - fetch_started_at) * 1000.0

            chart_started_at = time.perf_counter()
            chart_history = self.chart_history
            chart_limit = _initial_chart_bar_limit(self.interval)
            if chart_history is None:
                chart_history = _slice_recent_ohlcv(history, self.interval, max_bars=chart_limit)
            else:
                target_rows = min(max(len(chart_history), chart_limit), CHART_HISTORY_BAR_LIMIT)
                chart_history = _merge_ohlcv_frames(
                    chart_history,
                    _slice_recent_ohlcv(history, self.interval, max_bars=target_rows),
                    max_rows=target_rows,
                )
            chart_history = prepare_ohlcv(chart_history)
            if self.isInterruptionRequested():
                return
            chart_elapsed_ms = (time.perf_counter() - chart_started_at) * 1000.0
            visible_slice_changed = _frame_window_signature(
                self.chart_history,
                columns=["time", "open", "high", "low", "close", "volume", "quote_volume"],
            ) != _frame_window_signature(
                chart_history,
                columns=["time", "open", "high", "low", "close", "volume", "quote_volume"],
            )

            backtest_started_at = time.perf_counter()
            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
            use_existing_backtest = (
                self.existing_backtest is not None
                and not history_updated
                and _backtest_matches_history(self.existing_backtest, history)
            )
            if use_existing_backtest:
                backtest = self.existing_backtest
            elif _history_can_resume_backtest(self.existing_backtest, history):
                backtest = resume_backtest(
                    history,
                    previous_result=self.existing_backtest,
                    settings=self.settings.strategy,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            else:
                backtest = run_backtest(
                    history,
                    settings=self.settings.strategy,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            if self.isInterruptionRequested():
                return
            chart_indicators = _chart_indicators_from_backtest(backtest, chart_history)
            backtest_elapsed_ms = (time.perf_counter() - backtest_started_at) * 1000.0
            self.loaded.emit(
                {
                    "request_id": self.request_id,
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "history": history,
                    "chart_history": chart_history,
                    "backtest": backtest,
                    "chart_indicators": chart_indicators,
                    "history_refreshed_at": refreshed_at,
                    "recent_delta_used": recent_delta_used,
                    "repair_used": repair_used,
                    "visible_slice_changed": visible_slice_changed,
                    "perf": {
                        "worker_fetch_ms": fetch_elapsed_ms,
                        "worker_chart_ms": chart_elapsed_ms,
                        "worker_backtest_ms": backtest_elapsed_ms,
                    },
                }
            )
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())


class ChartHistoryPageWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        symbol: str,
        interval: str,
        oldest_time: pd.Timestamp,
        bars: int,
        cached_history: Optional[pd.DataFrame],
    ) -> None:
        super().__init__()
        self.settings = settings
        self.symbol = symbol
        self.interval = interval
        self.oldest_time = pd.Timestamp(oldest_time)
        self.bars = int(bars)
        self.cached_history = cached_history

    def run(self) -> None:
        try:
            chunk = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            exhausted = False
            cached_history = self.cached_history
            if cached_history is not None and not cached_history.empty:
                prepared = prepare_ohlcv(cached_history.copy())
                older = prepared[pd.to_datetime(prepared["time"]) < self.oldest_time]
                if not older.empty:
                    chunk = older.tail(self.bars).reset_index(drop=True)
                    exhausted = len(older) <= self.bars

            if self.isInterruptionRequested():
                return

            if chunk.empty and not exhausted:
                interval_ms = _interval_to_ms(self.interval)
                min_start_time_ms = _history_fetch_start_time_ms(self.settings, self.interval)
                oldest_time_ms = int(self.oldest_time.timestamp() * 1000)
                if oldest_time_ms <= min_start_time_ms:
                    exhausted = True
                else:
                    start_time_ms = max(oldest_time_ms - (interval_ms * self.bars), min_start_time_ms)
                    client = BinanceFuturesClient()
                    chunk = client.historical_ohlcv(
                        self.symbol,
                        self.interval,
                        start_time=start_time_ms,
                        end_time=oldest_time_ms - 1,
                    )
                    chunk = prepare_ohlcv(chunk)
                    exhausted = start_time_ms <= min_start_time_ms or chunk.empty

            if self.isInterruptionRequested():
                return

            self.completed.emit(
                {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "oldest_time": self.oldest_time,
                    "chunk": chunk,
                    "exhausted": exhausted,
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
        existing_backtest: Optional[BacktestResult],
    ) -> None:
        super().__init__()
        self.settings = settings
        self.symbol = symbol
        self.history = prepare_ohlcv(history.copy())
        self.chart_history = prepare_ohlcv(chart_history.copy())
        self.strategy_settings = strategy_settings
        self.existing_backtest = existing_backtest

    def run(self) -> None:
        try:
            backtest_started_at = time.perf_counter()
            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
            if _history_can_resume_backtest(self.existing_backtest, self.history):
                backtest = resume_backtest(
                    self.history,
                    previous_result=self.existing_backtest,
                    settings=self.strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            else:
                backtest = run_backtest(
                    self.history,
                    settings=self.strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            if self.isInterruptionRequested():
                return
            chart_indicators = _chart_indicators_from_backtest(backtest, self.chart_history)
            backtest_elapsed_ms = (time.perf_counter() - backtest_started_at) * 1000.0
            self.completed.emit(
                {
                    "symbol": self.symbol,
                    "history": self.history,
                    "chart_history": self.chart_history,
                    "backtest": backtest,
                    "chart_indicators": chart_indicators,
                    "perf": {
                        "worker_backtest_ms": backtest_elapsed_ms,
                    },
                }
            )
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())


class AutoCloseHistoryWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        symbol: str,
        interval: str,
        history: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.symbol = symbol
        self.interval = interval
        self.history = history

    def run(self) -> None:
        try:
            history = self.history
            if history is None:
                client = BinanceFuturesClient()
                history = client.historical_ohlcv(
                    self.symbol,
                    self.interval,
                    start_time=_history_fetch_start_time_ms(self.settings, self.interval),
                )
            if history is None or history.empty:
                raise RuntimeError(f"{self.symbol} auto-close history is empty.")
            history = prepare_ohlcv(history)
            if self.isInterruptionRequested():
                return
            self.completed.emit(
                {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "history": history,
                }
            )
        except Exception:
            if not self.isInterruptionRequested():
                self.failed.emit(traceback.format_exc())


class AutoCloseSignalWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        symbol: str,
        interval: str,
        history: pd.DataFrame,
        strategy_settings: StrategySettings,
        existing_backtest: Optional[BacktestResult],
    ) -> None:
        super().__init__()
        self.settings = settings
        self.symbol = symbol
        self.interval = interval
        self.history = prepare_ohlcv(history.copy())
        self.strategy_settings = strategy_settings
        self.existing_backtest = existing_backtest

    def run(self) -> None:
        try:
            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
            use_existing_backtest = (
                self.existing_backtest is not None
                and self.existing_backtest.settings == self.strategy_settings
                and _backtest_matches_history(self.existing_backtest, self.history)
            )
            if use_existing_backtest:
                backtest = self.existing_backtest
            elif (
                self.existing_backtest is not None
                and self.existing_backtest.settings == self.strategy_settings
                and _history_can_resume_backtest(self.existing_backtest, self.history)
            ):
                backtest = resume_backtest(
                    self.history,
                    previous_result=self.existing_backtest,
                    settings=self.strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            else:
                backtest = run_backtest(
                    self.history,
                    settings=self.strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            latest_time = pd.Timestamp(self.history["time"].iloc[-1])
            if self.isInterruptionRequested():
                return
            self.completed.emit(
                {
                    "interval": self.interval,
                    "symbol": self.symbol,
                    "history": self.history,
                    "bar_time": latest_time,
                    "backtest": backtest,
                    "exit_event": _latest_backtest_exit_event(backtest),
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
        margin: Optional[float] = None,
        close_only: bool = False,
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.symbol = symbol
        self.leverage = int(leverage)
        self.side = side
        self.fraction = fraction
        self.margin = float(margin) if margin is not None else None
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

            if self.side is None or (self.fraction is None and self.margin is None):
                raise RuntimeError("order parameters are incomplete")
            if self.margin is not None:
                margin = float(self.margin)
            else:
                balance = client.get_balance_snapshot()
                margin = balance.available_balance * float(self.fraction)
            if margin <= 0:
                raise RuntimeError("order amount must be positive")
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
        self.optimized_results: Dict[Tuple[str, str], OptimizationResult] = {}
        self.history_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.chart_history_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.backtest_cache: Dict[Tuple[str, str], BacktestResult] = {}
        self.chart_indicator_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.history_refresh_times: Dict[Tuple[str, str], float] = {}
        self.price_precision_cache: Dict[str, int] = {}
        self.pending_candidates: List[CandidateSymbol] = []
        self.pending_optimized_results: Dict[Tuple[str, str], OptimizationResult] = {}
        self.pending_history_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
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
        self.chart_history_page_worker: Optional[ChartHistoryPageWorker] = None
        self.live_backtest_worker: Optional[LiveBacktestWorker] = None
        self.account_worker: Optional[AccountInfoWorker] = None
        self.order_worker: Optional[OrderWorker] = None
        self.load_request_id = 0
        self.load_request_reset_view: Dict[int, bool] = {}
        self.account_request_id = 0
        self.live_recalc_pending = False
        self.pending_account_refresh = False
        self.live_stream_worker: Optional[KlineStreamWorker] = None
        self.position_price_workers: Dict[str, KlineStreamWorker] = {}
        self.account_balance_snapshot: Optional[BalanceSnapshot] = None
        self.live_pending_bar: Optional[Dict[str, object]] = None
        self.recent_symbol_cache_keys: deque[str] = deque(maxlen=RECENT_SYMBOL_CACHE_LIMIT)
        self.chart_history_exhausted: Dict[Tuple[str, str], bool] = {}
        self.chart_history_load_pending = False
        self.chart_history_load_requested = False
        self.chart_range_bars_before = float("inf")
        self.pending_lightweight_range_shift = 0
        self.symbol_load_started_at = 0.0
        self.live_backtest_started_at = 0.0
        self.chart_render_signature: Tuple[object, ...] = (None, 0)
        self.current_lightweight_markers: List[Dict[str, object]] = []
        self.current_lightweight_rendered_markers: List[Dict[str, object]] = []
        self.current_lightweight_preview_markers: List[Dict[str, object]] = []
        self._tracked_threads: set[QThread] = set()
        self.auto_close_enabled_symbols: set[str] = set()
        self.auto_close_monitor_histories: Dict[str, pd.DataFrame] = {}
        self.auto_close_monitor_intervals: Dict[str, str] = {}
        self.auto_close_stream_workers: Dict[str, KlineStreamWorker] = {}
        self.auto_close_history_workers: Dict[str, AutoCloseHistoryWorker] = {}
        self.auto_close_signal_workers: Dict[str, AutoCloseSignalWorker] = {}
        self.auto_close_signal_pending: set[str] = set()
        self.auto_close_order_pending: set[str] = set()
        self.auto_close_queued_orders: Dict[str, Tuple[str, Optional[pd.Timestamp]]] = {}
        self.auto_close_last_trigger_time: Dict[str, pd.Timestamp] = {}
        self.order_worker_symbol: Optional[str] = None
        self.order_worker_is_auto_close = False
        self.pending_open_order_interval: Optional[str] = None
        self.auto_refresh_minutes = 10
        self.auto_refresh_timer = QTimer(self)
        self.live_update_timer = QTimer(self)
        self.optimized_table_timer = QTimer(self)
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_phase = "idle"
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
        self.chart_render_signature = (None, 0)
        self.current_lightweight_markers = []
        self.current_lightweight_rendered_markers = []
        self.current_lightweight_preview_markers = []
        self.simple_order_buttons: List[QPushButton] = []
        self.position_close_buttons: List[QPushButton] = []
        self.position_action_widgets: List[QWidget] = []
        self.position_metric_widgets: List[QWidget] = []
        self.price_label_timer = QTimer(self)

        self.setWindowTitle("Binance Alt Mean Reversion Trader")
        self.resize(1848, 1056)
        self._build_ui()
        self._apply_loaded_settings()
        self._init_chart()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self._flush_live_update)
        self.optimized_table_timer.setSingleShot(True)
        self.optimized_table_timer.timeout.connect(self._flush_optimized_table)
        self.price_label_timer.setInterval(250)
        self.price_label_timer.timeout.connect(self._refresh_live_labels)
        self.price_label_timer.start()
        self.chart_engine_combo.currentTextChanged.connect(self.on_chart_engine_changed)
        self._init_auto_refresh()
        self.statusBar().showMessage("준비됨")
        QTimer.singleShot(0, self.refresh_account_info)

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

        balance_panel = QWidget()
        balance_layout = QHBoxLayout(balance_panel)
        balance_layout.setContentsMargins(8, 2, 8, 2)
        self.symbol_label = QLabel("종목: -")
        self.signal_label = QLabel("신호: -")
        self.current_price_label = QLabel("현재가: -")
        self.position_label = QLabel("포지션: -")
        self.balance_label = QLabel("잔고: API 미입력")
        self.balance_label.setStyleSheet("font-weight: 700; font-size: 13px;")
        balance_layout.addWidget(self.balance_label)
        self.chart_interval_label = QLabel("차트TF: -")
        self.chart_interval_label.setStyleSheet("color: #111827; font-weight: 700; font-size: 12px;")
        balance_layout.addSpacing(12)
        balance_layout.addWidget(self.chart_interval_label)
        self.bar_close_countdown_label = QLabel("봉마감: -")
        self.bar_close_countdown_label.setStyleSheet("color: #d63b53; font-weight: 700; font-size: 12px;")
        balance_layout.addSpacing(12)
        balance_layout.addWidget(self.bar_close_countdown_label)
        balance_layout.addStretch(1)
        self._set_balance_label_status("API 미입력")
        self.chart_host = QWidget()
        self.chart_host_layout = QVBoxLayout(self.chart_host)
        self.chart_host_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.chart_host, 8)
        right_layout.addWidget(balance_panel)

        right_layout.addWidget(self._build_positions_group(), 2)

        order_group = QGroupBox("Live Order")
        order_layout = QVBoxLayout(order_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("주문 모드"))
        self.compound_order_radio = QRadioButton("복리")
        self.simple_order_radio = QRadioButton("단리")
        self.compound_order_radio.toggled.connect(self._on_order_mode_toggled)
        self.simple_order_radio.toggled.connect(self._on_order_mode_toggled)
        mode_row.addWidget(self.compound_order_radio)
        mode_row.addWidget(self.simple_order_radio)
        mode_row.addStretch(1)
        self.compound_order_widget = QWidget()
        compound_layout = QVBoxLayout(self.compound_order_widget)
        compound_layout.setContentsMargins(0, 0, 0, 0)
        compound_layout.setSpacing(6)
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
        compound_layout.addLayout(long_row)
        compound_layout.addLayout(short_row)
        self.simple_order_widget = QWidget()
        simple_layout = QVBoxLayout(self.simple_order_widget)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        simple_layout.setSpacing(6)
        self.simple_order_buttons = []

        simple_amount_row = QHBoxLayout()
        simple_amount_row.addWidget(QLabel("주문금액"))
        self.simple_order_amount_spin = QDoubleSpinBox()
        self.simple_order_amount_spin.setRange(1.0, 1_000_000.0)
        self.simple_order_amount_spin.setDecimals(2)
        self.simple_order_amount_spin.setSingleStep(10.0)
        self.simple_order_amount_spin.setSuffix(" USDT")
        self.simple_order_amount_spin.setToolTip("단리 공통 주문금액")
        simple_amount_row.addWidget(self.simple_order_amount_spin, 1)

        simple_button_row = QHBoxLayout()
        self.simple_long_button = QPushButton("LONG")
        self.simple_long_button.clicked.connect(lambda _=False: self.place_simple_order("BUY"))
        self.simple_long_button.setMinimumWidth(140)
        self.simple_long_button.setMinimumHeight(28)
        self.simple_order_buttons.append(self.simple_long_button)
        self.simple_short_button = QPushButton("SHORT")
        self.simple_short_button.clicked.connect(lambda _=False: self.place_simple_order("SELL"))
        self.simple_short_button.setMinimumWidth(140)
        self.simple_short_button.setMinimumHeight(28)
        self.simple_order_buttons.append(self.simple_short_button)
        simple_button_row.addWidget(self.simple_long_button, 1)
        simple_button_row.addWidget(self.simple_short_button, 1)

        simple_layout.addLayout(simple_amount_row)
        simple_layout.addLayout(simple_button_row)
        self.close_position_button = QPushButton("포지션 청산")
        self.close_position_button.clicked.connect(self.close_selected_position)
        self.close_position_button.setText("청산")
        self.close_position_button.setFixedWidth(88)
        self.close_position_button.setToolTip("선택 종목 포지션 청산")
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(self.close_position_button)
        close_row.addStretch(1)
        order_layout.addLayout(mode_row)
        order_layout.addWidget(self.compound_order_widget)
        order_layout.addWidget(self.simple_order_widget)
        order_layout.addLayout(close_row)
        right_layout.addWidget(order_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([560, 1120])

    def _on_order_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._refresh_order_mode_ui()

    def _refresh_order_mode_ui(self) -> None:
        simple_mode = self.simple_order_radio.isChecked()
        self.compound_order_widget.setVisible(not simple_mode)
        self.simple_order_widget.setVisible(simple_mode)

    def _refresh_filter_controls(self) -> None:
        rsi_enabled = bool(self.rsi_filter_check.isChecked())
        self.rsi_length_spin.setEnabled(rsi_enabled)
        self.rsi_lower_spin.setEnabled(rsi_enabled)
        self.rsi_upper_spin.setEnabled(rsi_enabled)
        atr_enabled = bool(self.atr_4h_filter_check.isChecked())
        self.atr_4h_spin.setEnabled(atr_enabled)

    def _optimization_rank_mode(self) -> str:
        if not hasattr(self, "opt_rank_mode_combo"):
            return self.settings.optimization_rank_mode
        return str(self.opt_rank_mode_combo.currentData() or self.settings.optimization_rank_mode)

    def _refresh_optimization_rank_controls(self) -> None:
        rank_mode = self._optimization_rank_mode()
        score_mode = rank_mode == "score"
        if hasattr(self, "opt_min_score_spin"):
            self.opt_min_score_spin.setEnabled(score_mode)
        if hasattr(self, "opt_min_return_spin"):
            self.opt_min_return_spin.setEnabled(not score_mode)
        if hasattr(self, "optimized_table"):
            self.update_optimized_table()

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
        self.rsi_filter_check = QCheckBox("사용")
        self.rsi_length_spin = QSpinBox()
        self.rsi_length_spin.setRange(2, 100)
        self.rsi_lower_spin = QDoubleSpinBox()
        self.rsi_lower_spin.setRange(0.0, 100.0)
        self.rsi_lower_spin.setDecimals(1)
        self.rsi_upper_spin = QDoubleSpinBox()
        self.rsi_upper_spin.setRange(0.0, 100.0)
        self.rsi_upper_spin.setDecimals(1)
        self.atr_4h_filter_check = QCheckBox("사용")
        self.atr_4h_spin = QDoubleSpinBox()
        self.atr_4h_spin.setRange(0.0, 500.0)
        self.atr_4h_spin.setDecimals(2)
        self.atr_4h_spin.setSingleStep(1.0)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(APP_INTERVAL_OPTIONS)
        self.chart_engine_combo = QComboBox()
        self.chart_engine_combo.addItems(CHART_ENGINE_OPTIONS)
        self.chart_engine_combo.setEnabled(False)
        self.history_days_spin = QSpinBox()
        self.history_days_spin.setRange(1, 30)
        self.scan_workers_spin = QSpinBox()
        self.scan_workers_spin.setRange(1, 24)
        self.rsi_filter_check.toggled.connect(self._refresh_filter_controls)
        self.atr_4h_filter_check.toggled.connect(self._refresh_filter_controls)
        layout.addRow("1일 변동성 % >=", self.daily_vol_spin)
        layout.addRow("24h 거래량 >=", self.quote_volume_spin)
        layout.addRow("1m RSI 필터", self.rsi_filter_check)
        layout.addRow("1m RSI Length", self.rsi_length_spin)
        layout.addRow("1m RSI Lower <=", self.rsi_lower_spin)
        layout.addRow("1m RSI Upper >=", self.rsi_upper_spin)
        layout.addRow("4h ATR% 필터", self.atr_4h_filter_check)
        layout.addRow("4h ATR% >=", self.atr_4h_spin)
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
        self.opt_rank_mode_combo = QComboBox()
        self.opt_rank_mode_combo.addItem("점수제", "score")
        self.opt_rank_mode_combo.addItem("수익률제", "return")
        self.opt_min_score_spin = QDoubleSpinBox()
        self.opt_min_score_spin.setRange(0.0, 100.0)
        self.opt_min_score_spin.setDecimals(1)
        self.opt_min_score_spin.setSingleStep(1.0)
        self.opt_min_return_spin = QDoubleSpinBox()
        self.opt_min_return_spin.setRange(-500.0, 10_000.0)
        self.opt_min_return_spin.setDecimals(1)
        self.opt_min_return_spin.setSingleStep(1.0)
        self.max_combo_spin = QSpinBox()
        self.max_combo_spin.setRange(10, 20_000)
        self.opt_process_spin = QSpinBox()
        self.opt_process_spin.setRange(1, 16)
        self.optimize_timeframe_check = QCheckBox("1m / 2m 최적화")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 5.0)
        self.fee_spin.setDecimals(4)
        self.fee_spin.setSingleStep(0.01)
        self.opt_rank_mode_combo.currentIndexChanged.connect(lambda _index: self._refresh_optimization_rank_controls())
        self.opt_min_score_spin.valueChanged.connect(lambda _value: self.update_optimized_table())
        self.opt_min_return_spin.valueChanged.connect(lambda _value: self.update_optimized_table())
        def _set_row_label(editor, label: str) -> None:
            field_label = layout.labelForField(editor)
            if field_label is not None:
                field_label.setText(label)
        self.optimize_timeframe_check.setText("1m / 2m 최적화")
        layout.addRow("범위 ±%", self.opt_span_spin)
        layout.addRow("격자 단계수", self.opt_steps_spin)
        layout.addRow("정렬 기준", self.opt_rank_mode_combo)
        layout.addRow("최소 총점", self.opt_min_score_spin)
        layout.addRow("최소 수익률%", self.opt_min_return_spin)
        layout.addRow("최대 조합수", self.max_combo_spin)
        layout.addRow("최적화 프로세스", self.opt_process_spin)
        layout.addRow("타임프레임", self.optimize_timeframe_check)
        layout.addRow("수수료 %", self.fee_spin)
        _set_row_label(self.opt_span_spin, "범위 스케일 (20=기본)")
        _set_row_label(self.opt_steps_spin, "항목별 샘플 상한")
        _set_row_label(self.opt_min_score_spin, "최소 총점 (점수제)")
        _set_row_label(self.opt_min_return_spin, "최소 수익률% (수익률제)")
        _set_row_label(self.max_combo_spin, "최대 조합수")
        _set_row_label(self.opt_process_spin, "최적화 프로세스")
        _set_row_label(self.optimize_timeframe_check, "타임프레임")
        _set_row_label(self.fee_spin, "수수료 %")
        self._refresh_optimization_rank_controls()
        return group

    def _build_parameter_tabs(self) -> QGroupBox:
        group = QGroupBox("Strategy Parameters")
        outer_layout = QVBoxLayout(group)
        help_label = QLabel(
            "Opt를 체크한 항목만 최적화합니다. 각 항목은 퍼센트 일괄 범위가 아니라 "
            "옵션별 기본 프로필로 탐색하고, 범위 스케일은 그 폭을 전체적으로 조절합니다."
        )
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
        self.candidate_table = QTableWidget(0, 7)
        self.candidate_table.setHorizontalHeaderLabels(["Symbol", "DayVol%", "ATR4h%", "24h Vol", "RSI1m", "24h%", "Price"])
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
        self.optimized_table = QTableWidget(0, 9)
        self.optimized_table.setHorizontalHeaderLabels(["Symbol", "TF", "Score", "Return%", "MDD%", "Trades", "Win%", "PF", "Grid"])
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
        self.positions_table = QTableWidget(0, 7)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Side", "Amount", "Entry", "UPnL", "수익률", "Action"])
        self.positions_table.setSelectionBehavior(SELECT_ROWS)
        self.positions_table.setSelectionMode(SINGLE_SELECTION)
        self.positions_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.itemSelectionChanged.connect(self.on_positions_selection_changed)
        self.positions_table.cellClicked.connect(self.on_positions_cell_clicked)
        layout.addWidget(self.positions_table)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("로그")
        layout = QVBoxLayout(group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        self.backtest_progress_label = QLabel("대기중")
        self.backtest_progress_label.setStyleSheet("color: #5f6b7a; font-size: 12px; font-weight: 700;")
        layout.addWidget(self.backtest_progress_label)
        self.backtest_progress_bar = QProgressBar()
        self.backtest_progress_bar.setRange(0, 1)
        self.backtest_progress_bar.setValue(0)
        self.backtest_progress_bar.setTextVisible(True)
        self.backtest_progress_bar.setFormat("%p%")
        self.backtest_progress_bar.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #1f2937;"
            " border-radius: 4px;"
            " background: #0f172a;"
            " color: #e5e7eb;"
            " text-align: center;"
            " min-height: 14px;"
            "}"
            "QProgressBar::chunk {"
            " background-color: #17c964;"
            " border-radius: 3px;"
            "}"
        )
        layout.addWidget(self.backtest_progress_bar)
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
            self.update_positions_table()
            return
        if engine != "Lightweight":
            self._stop_chart_history_page_worker()
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
        self.chart.events.range_change += self._on_lightweight_range_change
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
                    right: "8px",
                    top: "0px",
                    display: "none",
                    minWidth: "62px",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    border: "1px solid rgba(34, 242, 2, 0.92)",
                    background: "rgba(15, 20, 25, 0.96)",
                    color: "#22f202",
                    fontFamily: "Consolas, monospace",
                    fontSize: "11px",
                    fontWeight: "700",
                    lineHeight: "1.25",
                    textAlign: "center",
                    letterSpacing: "0.02em",
                    fontVariantNumeric: "tabular-nums",
                    zIndex: "4200",
                    pointerEvents: "none",
                    whiteSpace: "nowrap",
                    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.45)",
                }});
                handler.div.appendChild(label);
                const reposition = () => {{
                    const overlay = handler.barCloseCountdown;
                    if (!overlay || !overlay.state.text || !Number.isFinite(overlay.state.price)) {{
                        label.style.display = "none";
                        return;
                    }}
                    const y = handler.series.priceToCoordinate(overlay.state.price);
                    if (y === null || y === undefined || !Number.isFinite(y)) {{
                        label.style.display = "none";
                        return;
                    }}
                    label.textContent = overlay.state.text;
                    label.style.display = "block";
                    const labelHeight = label.offsetHeight || 18;
                    const rawTop = Math.round(y + 26);
                    const maxTop = Math.max(4, handler.div.clientHeight - labelHeight - 4);
                    const top = Math.max(4, Math.min(rawTop, maxTop));
                    label.style.top = `${{top}}px`;
                }};
                handler.barCloseCountdown = {{
                    label,
                    state: {{ text: "", price: null }},
                    reposition,
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
                    overlay.reposition();
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
                    if (overlay && overlay.state.text && overlay.reposition) {{
                        overlay.reposition();
                    }}
                }});
                const baseResize = handler.reSize.bind(handler);
                handler.reSize = () => {{
                    baseResize();
                    const overlay = handler.barCloseCountdown;
                    if (overlay && overlay.state.text && overlay.reposition) {{
                        overlay.reposition();
                    }}
                }};
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
        self.settings.chart_engine = "Lightweight"
        if self.chart_engine_combo.currentText() != "Lightweight":
            self.chart_engine_combo.setCurrentText("Lightweight")

    def _apply_loaded_settings(self) -> None:
        settings = self.settings
        self.api_key_edit.setText(settings.api_key)
        self.api_secret_edit.setText(settings.api_secret)
        self.leverage_spin.setValue(settings.leverage)
        self.daily_vol_spin.setValue(settings.daily_volatility_min)
        self.quote_volume_spin.setValue(settings.quote_volume_min)
        self.rsi_filter_check.setChecked(settings.use_rsi_filter)
        self.rsi_length_spin.setValue(settings.rsi_length)
        self.rsi_lower_spin.setValue(settings.rsi_lower)
        self.rsi_upper_spin.setValue(settings.rsi_upper)
        self.atr_4h_filter_check.setChecked(settings.use_atr_4h_filter)
        self.atr_4h_spin.setValue(settings.atr_4h_min_pct)
        self.interval_combo.setCurrentText(settings.kline_interval)
        self.chart_engine_combo.setCurrentText("Lightweight")
        self.history_days_spin.setValue(settings.history_days)
        self.scan_workers_spin.setValue(settings.scan_workers)
        self.opt_span_spin.setValue(settings.optimization_span_pct)
        self.opt_steps_spin.setValue(settings.optimization_steps)
        rank_mode_index = self.opt_rank_mode_combo.findData(settings.optimization_rank_mode)
        if rank_mode_index >= 0:
            self.opt_rank_mode_combo.setCurrentIndex(rank_mode_index)
        self.opt_min_score_spin.setValue(settings.optimization_min_score)
        self.opt_min_return_spin.setValue(settings.optimization_min_return_pct)
        self.max_combo_spin.setValue(settings.max_grid_combinations)
        self.opt_process_spin.setValue(settings.optimize_processes)
        self.optimize_timeframe_check.setChecked(settings.optimize_timeframe)
        self.fee_spin.setValue(settings.fee_rate * 100.0)
        self.simple_order_amount_spin.setValue(settings.simple_order_amount)
        if settings.order_mode == "simple":
            self.simple_order_radio.setChecked(True)
        else:
            self.compound_order_radio.setChecked(True)
        self._refresh_order_mode_ui()
        self._refresh_filter_controls()
        self._refresh_optimization_rank_controls()

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
            chart_engine="Lightweight",
            leverage=int(self.leverage_spin.value()),
            order_mode="simple" if self.simple_order_radio.isChecked() else "compound",
            simple_order_amount=float(self.simple_order_amount_spin.value()),
            fee_rate=float(self.fee_spin.value()) / 100.0,
            history_days=int(self.history_days_spin.value()),
            kline_interval=self.interval_combo.currentText(),
            daily_volatility_min=float(self.daily_vol_spin.value()),
            quote_volume_min=float(self.quote_volume_spin.value()),
            use_rsi_filter=bool(self.rsi_filter_check.isChecked()),
            rsi_length=int(self.rsi_length_spin.value()),
            rsi_lower=float(self.rsi_lower_spin.value()),
            rsi_upper=float(self.rsi_upper_spin.value()),
            use_atr_4h_filter=bool(self.atr_4h_filter_check.isChecked()),
            atr_4h_min_pct=float(self.atr_4h_spin.value()),
            optimization_span_pct=float(self.opt_span_spin.value()),
            optimization_steps=int(self.opt_steps_spin.value()),
            optimization_rank_mode=self._optimization_rank_mode(),
            optimization_min_score=float(self.opt_min_score_spin.value()),
            optimization_min_return_pct=float(self.opt_min_return_spin.value()),
            max_grid_combinations=int(self.max_combo_spin.value()),
            scan_workers=int(self.scan_workers_spin.value()),
            optimize_processes=int(self.opt_process_spin.value()),
            optimize_timeframe=bool(self.optimize_timeframe_check.isChecked()),
            strategy=StrategySettings(**strategy_payload),
            optimize_flags=optimize_flags,
            position_intervals=dict(self.settings.position_intervals),
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
            self._stop_chart_history_page_worker()
            self._stop_live_backtest_worker()
            self._stop_live_stream()
            self._stop_all_auto_close_monitors()
            self.history_cache.clear()
            self.chart_history_cache.clear()
            self.history_refresh_times.clear()
            self.chart_history_exhausted.clear()
            self.current_chart_indicators = None
            self.price_precision_cache.clear()
        if previous != self.settings:
            self.backtest_cache.clear()
            self.chart_indicator_cache.clear()
            self.auto_close_signal_pending.clear()
            self.auto_close_order_pending.clear()
            self.auto_close_queued_orders.clear()
            self.auto_close_last_trigger_time.clear()
            self._stop_all_auto_close_monitors()
            self._refresh_auto_close_monitors()
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

    def _symbol_interval_key(self, symbol: str, interval: Optional[str] = None) -> Tuple[str, str]:
        return (symbol, interval or self.current_interval or self.settings.kline_interval)

    def _optimization_results_for_symbol(self, symbol: str) -> List[OptimizationResult]:
        return [
            result
            for (result_symbol, _interval), result in self.optimized_results.items()
            if result_symbol == symbol
        ]

    def _optimization_result(self, symbol: str, interval: Optional[str] = None) -> Optional[OptimizationResult]:
        normalized_interval = str(interval or "").strip()
        if normalized_interval in APP_INTERVAL_OPTIONS:
            return self.optimized_results.get((symbol, normalized_interval))
        candidates = self._optimization_results_for_symbol(symbol)
        if not candidates:
            return None
        rank_mode = self.settings.optimization_rank_mode
        return max(candidates, key=lambda result: optimization_sort_key(result.best_backtest.metrics, rank_mode))

    def _get_history_frame(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.history_cache.get(self._symbol_interval_key(symbol, interval))

    def _set_history_frame(self, symbol: str, frame: pd.DataFrame, interval: Optional[str] = None) -> None:
        self.history_cache[self._symbol_interval_key(symbol, interval)] = frame

    def _get_chart_history_frame(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.chart_history_cache.get(self._symbol_interval_key(symbol, interval))

    def _set_chart_history_frame(self, symbol: str, frame: pd.DataFrame, interval: Optional[str] = None) -> None:
        self.chart_history_cache[self._symbol_interval_key(symbol, interval)] = frame

    def _get_pending_history_frame(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.pending_history_cache.get(self._symbol_interval_key(symbol, interval))

    def _set_pending_history_frame(self, symbol: str, frame: pd.DataFrame, interval: Optional[str] = None) -> None:
        self.pending_history_cache[self._symbol_interval_key(symbol, interval)] = frame

    def _history_last_refresh_at(self, symbol: str, interval: Optional[str] = None) -> Optional[float]:
        return self.history_refresh_times.get(self._symbol_interval_key(symbol, interval))

    def _mark_history_refreshed(self, symbol: str, refreshed_at: float, interval: Optional[str] = None) -> None:
        self.history_refresh_times[self._symbol_interval_key(symbol, interval)] = float(refreshed_at)

    def _log_perf(self, label: str, started_at: float) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= PERFORMANCE_LOG_THRESHOLD_MS:
            self.log(f"[perf] {label}: {elapsed_ms:.1f}ms")

    def _log_perf_ms(self, label: str, elapsed_ms: float) -> None:
        if float(elapsed_ms) >= PERFORMANCE_LOG_THRESHOLD_MS:
            self.log(f"[perf] {label}: {float(elapsed_ms):.1f}ms")

    def _persist_position_intervals(self) -> None:
        self.settings.save()

    def _remember_position_interval(self, symbol: str, interval: Optional[str], persist: bool = True) -> None:
        normalized = str(interval or "").strip()
        if not symbol or normalized not in APP_INTERVAL_OPTIONS:
            return
        if self.settings.position_intervals.get(symbol) == normalized:
            return
        self.settings.position_intervals[symbol] = normalized
        if persist:
            self._persist_position_intervals()

    def _forget_closed_position_intervals(self, open_symbols: set[str], persist: bool = True) -> None:
        removed = False
        for symbol in list(self.settings.position_intervals):
            if symbol in open_symbols:
                continue
            self.settings.position_intervals.pop(symbol, None)
            removed = True
        if removed and persist:
            self._persist_position_intervals()

    def _remember_missing_open_position_intervals(self, open_symbols: set[str]) -> None:
        changed = False
        for symbol in sorted(open_symbols):
            if self.settings.position_intervals.get(symbol) in APP_INTERVAL_OPTIONS:
                continue
            optimization = self._optimization_result(symbol)
            interval = (optimization.best_interval or self.settings.kline_interval) if optimization else self.settings.kline_interval
            if interval not in APP_INTERVAL_OPTIONS:
                continue
            self.settings.position_intervals[symbol] = interval
            changed = True
        if changed:
            self._persist_position_intervals()

    def _position_interval_for_symbol(self, symbol: str) -> str:
        remembered = self.settings.position_intervals.get(symbol)
        if remembered in APP_INTERVAL_OPTIONS and self._find_open_position(symbol) is not None:
            return remembered
        optimization = self._optimization_result(symbol)
        return (optimization.best_interval or self.settings.kline_interval) if optimization else self.settings.kline_interval

    def _position_symbol_text(self, symbol: str) -> str:
        return f"{symbol} [{self._position_interval_for_symbol(symbol)}]"

    def _active_interval_for_symbol(self, symbol: str) -> str:
        return self._position_interval_for_symbol(symbol)

    def _find_open_position(self, symbol: str) -> Optional[PositionSnapshot]:
        return next((position for position in self.open_positions if position.symbol == symbol), None)

    def _set_auto_close_button_state(self, button: QPushButton, enabled: bool) -> None:
        button.setCheckable(True)
        button.setChecked(enabled)
        button.setText("자동청산 ON" if enabled else "자동청산 OFF")
        button.setStyleSheet(
            """
            QPushButton {
                font-weight: 700;
                font-size: 11px;
                color: #d8dee9;
                background-color: #3b4252;
                border: 1px solid #495468;
                border-radius: 4px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #465166;
                border-color: #5b6880;
            }
            QPushButton:pressed {
                background-color: #2f3745;
            }
            QPushButton:checked {
                color: #ffffff;
                background-color: #1f8f47;
                border-color: #166636;
            }
            QPushButton:checked:hover {
                background-color: #249d50;
                border-color: #1a7a40;
            }
            QPushButton:checked:pressed {
                background-color: #17713a;
            }
            """
        )

    def _style_close_button(self, button: QPushButton) -> None:
        button.setStyleSheet(
            """
            QPushButton {
                font-weight: 700;
                font-size: 11px;
                color: #ffffff;
                background-color: #cc334f;
                border: 1px solid #9e2239;
                border-radius: 4px;
                padding: 2px 10px;
            }
            QPushButton:hover {
                background-color: #dd3b59;
                border-color: #b32741;
            }
            QPushButton:pressed {
                background-color: #a6263d;
                border-color: #7e1c2f;
            }
            QPushButton:disabled {
                color: #f3d5db;
                background-color: #b67b87;
                border-color: #95626c;
            }
            """
        )

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

    def _chart_cache_key(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> Tuple[str, str]:
        return (symbol or self.current_symbol or "", interval or self.current_interval or self.settings.kline_interval)

    def _current_chart_history(self, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        target_symbol = symbol or self.current_symbol
        if not target_symbol:
            return None
        return self._get_chart_history_frame(target_symbol, self.current_interval or self.settings.kline_interval)

    def _build_initial_chart_history(
        self,
        symbol: str,
        interval: str,
        cached_history: Optional[pd.DataFrame],
        cached_backtest: Optional[BacktestResult],
    ) -> Optional[pd.DataFrame]:
        if cached_history is not None and not cached_history.empty:
            return _slice_recent_ohlcv(cached_history, interval)
        if cached_backtest is not None and not cached_backtest.indicators.empty:
            return _ohlcv_from_indicator_frame(cached_backtest.indicators, interval)
        return None

    def _sync_chart_indicator_cache(self, symbol: str) -> None:
        if self.current_backtest is None:
            return
        chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        if chart_history is None or chart_history.empty:
            return
        self.current_chart_indicators = _chart_indicators_from_backtest(self.current_backtest, chart_history)
        self.chart_indicator_cache[self._symbol_interval_key(symbol, self.current_interval)] = self.current_chart_indicators

    def _trade_markers(self, trades, latest_time: Optional[pd.Timestamp]) -> List[Dict[str, object]]:
        markers: List[Dict[str, object]] = []
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
        return markers

    def _marker_signature(self, marker: Dict[str, object]) -> Tuple[object, ...]:
        return (
            _normalize_signature_value(pd.Timestamp(marker["time"])),
            marker.get("position"),
            marker.get("shape"),
            marker.get("color"),
            marker.get("text"),
        )

    def _compose_lightweight_markers(self, confirmed_markers: Optional[List[Dict[str, object]]] = None) -> List[Dict[str, object]]:
        markers = list(self.current_lightweight_markers if confirmed_markers is None else confirmed_markers)
        preview_markers = list(self.current_lightweight_preview_markers)
        if not preview_markers:
            return markers
        marker_keys = {
            (pd.Timestamp(marker["time"]), marker.get("position"), marker.get("shape"))
            for marker in markers
        }
        for preview in preview_markers:
            preview_key = (pd.Timestamp(preview["time"]), preview.get("position"), preview.get("shape"))
            if preview_key in marker_keys:
                continue
            markers.append(preview)
            marker_keys.add(preview_key)
        return markers

    def _render_lightweight_markers(self, confirmed_markers: Optional[List[Dict[str, object]]] = None) -> None:
        if self.chart_mode != "Lightweight" or self.chart is None:
            return
        if confirmed_markers is not None:
            self.current_lightweight_markers = list(confirmed_markers)
        rendered_markers = self._compose_lightweight_markers()
        rendered_signature = [self._marker_signature(marker) for marker in rendered_markers]
        current_signature = [self._marker_signature(marker) for marker in self.current_lightweight_rendered_markers]
        if rendered_signature == current_signature:
            return
        self.chart.clear_markers()
        if rendered_markers:
            self.chart.marker_list(rendered_markers)
        self.current_lightweight_rendered_markers = list(rendered_markers)

    def _build_live_preview_markers(self, symbol: str) -> List[Dict[str, object]]:
        if (
            symbol != self.current_symbol
            or self.chart_mode != "Lightweight"
            or self.current_backtest is None
            or self.current_backtest.cursor is None
        ):
            return []
        position_qty = float(self.current_backtest.cursor.position_qty)
        history = self._get_history_frame(symbol, self.current_interval)
        if history is None or history.empty:
            return []
        latest_state, _ = evaluate_latest_state(
            history,
            self.current_backtest.settings,
            cursor=self.current_backtest.cursor.indicator_cursor,
        )
        latest_time = pd.Timestamp(history["time"].iloc[-1])
        preview_markers: List[Dict[str, object]] = []
        reason = _preview_exit_reason(position_qty, latest_state)
        if reason is not None:
            side = "long" if position_qty > 0 else "short"
            preview_markers.append(
                {
                    "time": latest_time,
                    "position": "above" if side == "long" else "below",
                    "shape": "circle",
                    "color": "#ff9f1a",
                    "text": f"청산예상 {_auto_close_reason_text(reason)}",
                }
            )
        entry_signal = _preview_entry_signal(self.current_backtest.cursor, latest_state, self.current_backtest.settings)
        if entry_signal is not None:
            side, zone = entry_signal
            preview_markers.append(
                {
                    "time": latest_time,
                    "position": "below" if side == "long" else "above",
                    "shape": "arrow_up" if side == "long" else "arrow_down",
                    "color": "#ffb020",
                    "text": f"진입예상 {side[0].upper()}{zone}",
                }
            )
        return preview_markers

    def _refresh_live_preview_markers(self, symbol: Optional[str]) -> None:
        if self.chart_mode != "Lightweight":
            return
        target_symbol = str(symbol or "")
        preview_markers = self._build_live_preview_markers(target_symbol) if target_symbol else []
        preview_signature = [self._marker_signature(marker) for marker in preview_markers]
        current_signature = [self._marker_signature(marker) for marker in self.current_lightweight_preview_markers]
        if preview_signature == current_signature:
            return
        self.current_lightweight_preview_markers = list(preview_markers)
        self._render_lightweight_markers()

    def _build_chart_render_payload(
        self,
        symbol: str,
        result: BacktestResult,
        chart_indicators: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
        active_chart_indicators = chart_indicators
        if active_chart_indicators is None and symbol == self.current_symbol and self.current_chart_indicators is not None:
            active_chart_indicators = self.current_chart_indicators
        if active_chart_indicators is None:
            chart_history = self._get_chart_history_frame(symbol, self.current_interval)
            active_chart_indicators = _chart_indicators_from_backtest(result, chart_history)
        indicators = (
            active_chart_indicators.sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        candle_df = indicators[["time", "open", "high", "low", "close", "volume"]].copy()
        equity_df = (
            pd.DataFrame({"time": list(result.equity_curve.index), "Equity": list(result.equity_curve.values)})
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        latest_time = pd.Timestamp(candle_df["time"].iloc[-1]) if not candle_df.empty else None
        markers = self._trade_markers(result.trades, latest_time)
        return candle_df, indicators, equity_df, markers

    def _chart_render_signature_for_payload(
        self,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        markers: List[Dict[str, object]],
    ) -> Tuple[object, ...]:
        return (
            _frame_window_signature(candle_df, columns=["time", "open", "high", "low", "close", "volume"]),
            _frame_window_signature(
                indicators,
                columns=["time", "supertrend", "zone2_line", "zone3_line", "ema_fast", "ema_slow"],
            ),
            _frame_window_signature(equity_df, columns=["time", "Equity"]),
            len(markers),
            tuple(self._marker_signature(marker) for marker in markers[-4:]),
        )

    def _maybe_request_more_chart_history(self, symbol: Optional[str] = None) -> None:
        target_symbol = symbol or self.current_symbol
        if (
            not target_symbol
            or self.chart_mode != "Lightweight"
            or target_symbol != self.current_symbol
            or self.current_backtest is None
            or self.chart_history_page_worker is not None
        ):
            return
        if self.chart_range_bars_before > CHART_LAZY_LOAD_TRIGGER_BARS:
            return
        chart_history = self._current_chart_history(target_symbol)
        if chart_history is None or chart_history.empty or len(chart_history) < 2:
            return
        cache_key = self._chart_cache_key(target_symbol)
        if self.chart_history_exhausted.get(cache_key, False):
            return
        oldest_time = pd.Timestamp(chart_history["time"].iloc[0])
        cached_history = self._get_history_frame(target_symbol, self.current_interval)
        worker = ChartHistoryPageWorker(
            self.settings,
            target_symbol,
            self.current_interval,
            oldest_time,
            CHART_LAZY_LOAD_CHUNK_BARS,
            cached_history,
        )
        worker.completed.connect(self._on_chart_history_page_completed)
        worker.failed.connect(self._on_chart_history_page_failed)
        self.chart_history_page_worker = worker
        self.chart_history_load_requested = True
        self._track_thread(worker, "chart_history_page_worker")
        worker.start()

    def _on_lightweight_range_change(self, _chart, bars_before: float, _bars_after: float) -> None:
        self.chart_range_bars_before = float(bars_before)
        if self.chart_mode != "Lightweight" or not self.current_symbol:
            return
        if self.chart_history_page_worker is not None:
            self.chart_history_load_pending = self.chart_range_bars_before <= CHART_LAZY_LOAD_TRIGGER_BARS
            return
        self._maybe_request_more_chart_history(self.current_symbol)

    def _on_chart_history_page_completed(self, payload: object) -> None:
        self.chart_history_page_worker = None
        self.chart_history_load_requested = False
        result = dict(payload)
        symbol = str(result["symbol"])
        interval = str(result["interval"])
        cache_key = self._chart_cache_key(symbol, interval)
        self.chart_history_exhausted[cache_key] = bool(result.get("exhausted", False))
        if symbol != self.current_symbol or interval != self.current_interval:
            return
        chart_history = self._get_chart_history_frame(symbol, interval)
        chunk = result.get("chunk")
        if chart_history is None or chunk is None:
            return
        chunk_frame = prepare_ohlcv(pd.DataFrame(chunk).copy()) if not isinstance(chunk, pd.DataFrame) else prepare_ohlcv(chunk.copy())
        if chunk_frame.empty:
            if self.chart_history_load_pending:
                self.chart_history_load_pending = False
                QTimer.singleShot(0, lambda s=symbol: self._maybe_request_more_chart_history(s))
            return
        merged_chart_history = _merge_ohlcv_frames(chunk_frame, chart_history, max_rows=CHART_HISTORY_BAR_LIMIT)
        self.pending_lightweight_range_shift = max(0, len(merged_chart_history) - len(chart_history))
        self._set_chart_history_frame(symbol, merged_chart_history, interval)
        self._sync_chart_indicator_cache(symbol)
        if self.current_backtest is not None and self.current_chart_indicators is not None:
            self.render_chart(symbol, self.current_backtest, reset_view=False, chart_indicators=self.current_chart_indicators)
        self._prune_caches()
        should_continue = self.chart_history_load_pending or self.chart_range_bars_before <= CHART_LAZY_LOAD_TRIGGER_BARS
        self.chart_history_load_pending = False
        if should_continue:
            QTimer.singleShot(0, lambda s=symbol: self._maybe_request_more_chart_history(s))

    def _on_chart_history_page_failed(self, message: str) -> None:
        self.chart_history_page_worker = None
        self.chart_history_load_pending = False
        self.chart_history_load_requested = False
        self.log(message)

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
        shift = float(self.pending_lightweight_range_shift)
        self.pending_lightweight_range_shift = 0
        self.chart.run_script(
            f"""
            const range = window.__alt_lwc_view_range;
            if (range && Number.isFinite(range.from) && Number.isFinite(range.to) && range.to > range.from) {{
                const fromValue = range.from + {shift};
                const toValue = range.to + {shift};
                {self.equity_subchart.id}.chart.timeScale().setVisibleLogicalRange({{
                    from: fromValue,
                    to: toValue
                }});
                {self.chart.id}.chart.timeScale().setVisibleLogicalRange({{
                    from: fromValue,
                    to: toValue
                }});
            }}
            """
        )

    def _active_backtest_settings(self, symbol: str) -> StrategySettings:
        optimization = self._optimization_result(symbol, self._active_interval_for_symbol(symbol))
        return optimization.best_backtest.settings if optimization else self.settings.strategy

    def _track_thread(self, worker: QThread, attr_name: str) -> None:
        self._tracked_threads.add(worker)

        def _cleanup_thread() -> None:
            if getattr(self, attr_name, None) is worker:
                setattr(self, attr_name, None)
            self._tracked_threads.discard(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup_thread)

    def _track_mapped_thread(self, worker: QThread, registry: Dict[str, QThread], key: str) -> None:
        self._tracked_threads.add(worker)

        def _cleanup_thread() -> None:
            if registry.get(key) is worker:
                registry.pop(key, None)
            self._tracked_threads.discard(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup_thread)

    def _prune_caches(self) -> None:
        keep_symbols = set()
        if self.current_symbol:
            keep_symbols.add(self.current_symbol)
        keep_symbols.update(self.recent_symbol_cache_keys)
        ordered_symbols = [
            result.symbol
            for result in sorted(
                self.optimized_results.values(),
                key=lambda item: optimization_sort_key(item.best_backtest.metrics, self.settings.optimization_rank_mode),
                reverse=True,
            )[:HISTORY_CACHE_SYMBOL_LIMIT]
        ]
        keep_symbols.update(ordered_symbols)
        self.history_cache = {
            key: frame
            for key, frame in self.history_cache.items()
            if key[0] in keep_symbols
        }
        self.chart_history_cache = {
            key: frame
            for key, frame in self.chart_history_cache.items()
            if key[0] in keep_symbols
        }
        self.backtest_cache = {
            key: value
            for key, value in self.backtest_cache.items()
            if key[0] in keep_symbols
        }
        self.chart_indicator_cache = {
            key: value
            for key, value in self.chart_indicator_cache.items()
            if key[0] in keep_symbols
        }
        self.history_refresh_times = {
            key: value
            for key, value in self.history_refresh_times.items()
            if key[0] in keep_symbols
        }
        self.chart_history_exhausted = {
            key: value
            for key, value in self.chart_history_exhausted.items()
            if key[0] in keep_symbols
        }

    def _remember_recent_symbol(self, symbol: str) -> None:
        if not symbol:
            return
        try:
            self.recent_symbol_cache_keys.remove(symbol)
        except ValueError:
            pass
        self.recent_symbol_cache_keys.append(symbol)

    def _stop_auto_close_history_worker(self, symbol: str) -> None:
        worker = self.auto_close_history_workers.pop(symbol, None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _stop_auto_close_signal_worker(self, symbol: str) -> None:
        worker = self.auto_close_signal_workers.pop(symbol, None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)
        self.auto_close_signal_pending.discard(symbol)

    def _stop_auto_close_stream_worker(self, symbol: str) -> None:
        worker = self.auto_close_stream_workers.pop(symbol, None)
        if worker is not None:
            worker.stop()
            worker.wait(1500)

    def _stop_auto_close_monitor(self, symbol: str) -> None:
        self._stop_auto_close_history_worker(symbol)
        self._stop_auto_close_signal_worker(symbol)
        self._stop_auto_close_stream_worker(symbol)
        self.auto_close_monitor_histories.pop(symbol, None)
        self.auto_close_monitor_intervals.pop(symbol, None)

    def _clear_auto_close_symbol(self, symbol: str) -> None:
        self.auto_close_enabled_symbols.discard(symbol)
        self.auto_close_signal_pending.discard(symbol)
        self.auto_close_order_pending.discard(symbol)
        self.auto_close_queued_orders.pop(symbol, None)
        self.auto_close_last_trigger_time.pop(symbol, None)
        self._stop_auto_close_monitor(symbol)

    def _stop_all_auto_close_monitors(self) -> None:
        symbols = set(self.auto_close_history_workers) | set(self.auto_close_signal_workers) | set(self.auto_close_stream_workers)
        for symbol in list(symbols):
            self._stop_auto_close_monitor(symbol)
        self.auto_close_monitor_histories.clear()
        self.auto_close_monitor_intervals.clear()
        self.auto_close_signal_pending.clear()

    def _start_auto_close_history_worker(self, symbol: str, interval: str) -> None:
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            return
        self._stop_auto_close_history_worker(symbol)
        history = self.auto_close_monitor_histories.get(symbol)
        if not _frame_matches_interval(history, interval):
            history = self._get_history_frame(symbol, interval)
        if not _frame_matches_interval(history, interval):
            history = None
        worker = AutoCloseHistoryWorker(self.settings, symbol, interval, history)
        worker.completed.connect(self._on_auto_close_history_completed)
        worker.failed.connect(lambda message, symbol=symbol: self._on_auto_close_history_failed(symbol, message))
        self.auto_close_history_workers[symbol] = worker
        self._track_mapped_thread(worker, self.auto_close_history_workers, symbol)
        worker.start()

    def _start_auto_close_stream_worker(self, symbol: str, interval: str) -> None:
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            return
        existing = self.auto_close_stream_workers.get(symbol)
        if existing is not None and self.auto_close_monitor_intervals.get(symbol) == interval:
            return
        self._stop_auto_close_stream_worker(symbol)
        worker = KlineStreamWorker(symbol, interval)
        worker.kline.connect(self._on_auto_close_kline)
        self.auto_close_stream_workers[symbol] = worker
        self._track_mapped_thread(worker, self.auto_close_stream_workers, symbol)
        worker.start()

    def _schedule_auto_close_signal(self, symbol: str) -> None:
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            return
        history = self.auto_close_monitor_histories.get(symbol)
        if history is None or history.empty:
            return
        desired_interval = self._active_interval_for_symbol(symbol)
        if self.auto_close_monitor_intervals.get(symbol) != desired_interval:
            self._stop_auto_close_monitor(symbol)
            self._start_auto_close_history_worker(symbol, desired_interval)
            return
        worker = self.auto_close_signal_workers.get(symbol)
        if worker is not None and worker.isRunning():
            self.auto_close_signal_pending.add(symbol)
            return
        self.auto_close_signal_pending.discard(symbol)
        cache_key = self._symbol_interval_key(symbol, desired_interval)
        worker = AutoCloseSignalWorker(
            self.settings,
            symbol,
            desired_interval,
            history,
            self._active_backtest_settings(symbol),
            self.backtest_cache.get(cache_key),
        )
        worker.completed.connect(self._on_auto_close_signal_completed)
        worker.failed.connect(lambda message, symbol=symbol: self._on_auto_close_signal_failed(symbol, message))
        self.auto_close_signal_workers[symbol] = worker
        self._track_mapped_thread(worker, self.auto_close_signal_workers, symbol)
        worker.start()

    def _toggle_auto_close_for_symbol(self, symbol: str, enabled: bool) -> None:
        button = self.sender()
        if isinstance(button, QPushButton):
            self._set_auto_close_button_state(button, enabled)
        if enabled:
            self.auto_close_enabled_symbols.add(symbol)
            self.log(f"{symbol} 자동청산 활성화")
            self._refresh_auto_close_monitors()
            if symbol == self.current_symbol and self.current_backtest is not None:
                self._evaluate_backtest_auto_close(symbol, self.current_backtest)
            return
        self.log(f"{symbol} 자동청산 비활성화")
        self._clear_auto_close_symbol(symbol)
        self.update_positions_table()

    def _refresh_auto_close_monitors(self) -> None:
        open_symbols = {position.symbol for position in self.open_positions}
        for symbol in list(self.auto_close_enabled_symbols):
            if symbol not in open_symbols:
                self._clear_auto_close_symbol(symbol)

        for symbol in list(self.auto_close_history_workers):
            if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)
        for symbol in list(self.auto_close_signal_workers):
            if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)
        for symbol in list(self.auto_close_stream_workers):
            if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)

        for symbol in list(self.auto_close_enabled_symbols):
            if symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)
                if self.current_backtest is not None:
                    self._evaluate_backtest_auto_close(symbol, self.current_backtest)
                continue
            desired_interval = self._active_interval_for_symbol(symbol)
            if self.auto_close_monitor_intervals.get(symbol) != desired_interval:
                self._stop_auto_close_monitor(symbol)
            if symbol not in self.auto_close_monitor_histories:
                if symbol not in self.auto_close_history_workers:
                    self._start_auto_close_history_worker(symbol, desired_interval)
                continue
            self.auto_close_monitor_intervals[symbol] = desired_interval
            self._start_auto_close_stream_worker(symbol, desired_interval)
            self._schedule_auto_close_signal(symbol)

    def _on_auto_close_history_completed(self, payload: object) -> None:
        result = dict(payload)
        symbol = str(result["symbol"])
        interval = str(result["interval"])
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            self._stop_auto_close_monitor(symbol)
            return
        desired_interval = self._active_interval_for_symbol(symbol)
        if interval != desired_interval:
            self._stop_auto_close_monitor(symbol)
            self._start_auto_close_history_worker(symbol, desired_interval)
            return
        self.auto_close_monitor_histories[symbol] = result["history"]
        self.auto_close_monitor_intervals[symbol] = interval
        self._start_auto_close_stream_worker(symbol, interval)
        self._schedule_auto_close_signal(symbol)

    def _on_auto_close_history_failed(self, symbol: str, message: str) -> None:
        if symbol in self.auto_close_enabled_symbols:
            self.log(message)

    def _on_auto_close_kline(self, payload: object) -> None:
        bar = dict(payload)
        symbol = str(bar.get("symbol", ""))
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            return
        if not bool(bar.get("closed")):
            return
        history = self.auto_close_monitor_histories.get(symbol)
        if history is None or history.empty:
            return
        self.auto_close_monitor_histories[symbol] = _merge_live_bar(history, bar)
        self._schedule_auto_close_signal(symbol)

    def _on_auto_close_signal_completed(self, payload: object) -> None:
        result = dict(payload)
        symbol = str(result["symbol"])
        if symbol not in self.auto_close_enabled_symbols or symbol == self.current_symbol:
            return
        interval = str(result.get("interval") or self._active_interval_for_symbol(symbol))
        history = result.get("history")
        if isinstance(history, pd.DataFrame) and not history.empty:
            normalized_history = prepare_ohlcv(history.copy())
            self.auto_close_monitor_histories[symbol] = normalized_history
            self.history_cache[self._symbol_interval_key(symbol, interval)] = normalized_history
        backtest = result.get("backtest")
        if isinstance(backtest, BacktestResult):
            self.backtest_cache[self._symbol_interval_key(symbol, interval)] = backtest
        self._maybe_trigger_auto_close(symbol, result.get("exit_event"))
        if symbol in self.auto_close_signal_pending:
            self.auto_close_signal_pending.discard(symbol)
            self._schedule_auto_close_signal(symbol)

    def _on_auto_close_signal_failed(self, symbol: str, message: str) -> None:
        self.auto_close_signal_pending.discard(symbol)
        if symbol in self.auto_close_enabled_symbols:
            self.log(message)

    def _evaluate_backtest_auto_close(self, symbol: str, backtest: BacktestResult) -> None:
        if symbol not in self.auto_close_enabled_symbols:
            return
        if backtest.indicators.empty:
            return
        self._maybe_trigger_auto_close(symbol, _latest_backtest_exit_event(backtest))

    def _maybe_trigger_auto_close(
        self,
        symbol: str,
        exit_event: Optional[Dict[str, object]],
    ) -> None:
        if symbol not in self.auto_close_enabled_symbols:
            return
        position = self._find_open_position(symbol)
        if position is None:
            self._clear_auto_close_symbol(symbol)
            self.update_positions_table()
            return
        reason = _auto_close_reason(position, exit_event)
        if reason is None:
            return
        bar_time = exit_event.get("bar_time") if exit_event else None
        if symbol in self.auto_close_order_pending:
            return
        normalized_bar_time = pd.Timestamp(bar_time) if bar_time is not None else None
        if normalized_bar_time is not None and self.auto_close_last_trigger_time.get(symbol) == normalized_bar_time:
            return
        if self.order_worker is not None and self.order_worker.isRunning():
            if symbol not in self.auto_close_queued_orders:
                self.log(f"{symbol} 자동청산 대기: 기존 주문 처리 중")
            self.auto_close_queued_orders[symbol] = (reason, normalized_bar_time)
            return
        if self._submit_close_position(symbol, auto_close_reason=reason):
            self.auto_close_order_pending.add(symbol)
            self.auto_close_queued_orders.pop(symbol, None)
            if normalized_bar_time is not None:
                self.auto_close_last_trigger_time[symbol] = normalized_bar_time

    def _flush_queued_auto_close_orders(self) -> None:
        if self.order_worker is not None and self.order_worker.isRunning():
            return
        for symbol, (reason, bar_time) in list(self.auto_close_queued_orders.items()):
            if symbol not in self.auto_close_enabled_symbols:
                self.auto_close_queued_orders.pop(symbol, None)
                continue
            position = self._find_open_position(symbol)
            if position is None:
                self._clear_auto_close_symbol(symbol)
                continue
            if bar_time is not None and self.auto_close_last_trigger_time.get(symbol) == bar_time:
                self.auto_close_queued_orders.pop(symbol, None)
                continue
            if self._submit_close_position(symbol, auto_close_reason=reason):
                self.auto_close_order_pending.add(symbol)
                self.auto_close_queued_orders.pop(symbol, None)
                if bar_time is not None:
                    self.auto_close_last_trigger_time[symbol] = bar_time
                return

    def _stop_scan_worker(self) -> None:
        worker = self.scan_worker
        self.scan_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(10000)
        if self.optimize_worker is None:
            self._set_backtest_progress_idle()

    def _stop_optimize_worker(self) -> None:
        worker = self.optimize_worker
        self.optimize_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(5000)
        if self.scan_worker is None:
            self._set_backtest_progress_idle()

    def _stop_load_worker(self) -> None:
        worker = self.load_worker
        self.load_worker = None
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(1500)

    def _stop_chart_history_page_worker(self) -> None:
        worker = self.chart_history_page_worker
        self.chart_history_page_worker = None
        self.chart_history_load_pending = False
        self.chart_history_load_requested = False
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
        self.order_worker_symbol = None
        self.order_worker_is_auto_close = False
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

    def _stop_position_price_worker(self, symbol: str) -> None:
        worker = self.position_price_workers.pop(symbol, None)
        if worker is not None:
            worker.stop()
            worker.wait(1500)

    def _refresh_position_price_streams(self) -> None:
        open_symbols = {position.symbol for position in self.open_positions}
        if self.current_symbol:
            open_symbols.discard(self.current_symbol)

        for symbol in list(self.position_price_workers):
            if symbol not in open_symbols:
                self._stop_position_price_worker(symbol)

        for symbol in sorted(open_symbols):
            if symbol in self.position_price_workers:
                continue
            worker = KlineStreamWorker(symbol, "1m")
            worker.kline.connect(self._on_position_price_kline)
            self.position_price_workers[symbol] = worker
            self._track_mapped_thread(worker, self.position_price_workers, symbol)
            worker.start()

    def _on_position_price_kline(self, payload: object) -> None:
        bar = dict(payload)
        symbol = str(bar.get("symbol", ""))
        if not symbol or symbol == self.current_symbol:
            return
        self._apply_live_position_price(symbol, float(bar["close"]))

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
        if self.chart_mode == "Lightweight" and not bool(bar.get("closed")):
            self._flush_live_update()
            return
        self.live_update_timer.setInterval(
            LIVE_RENDER_INTERVAL_MS if self.chart_mode == "Lightweight" else PLOTLY_LIVE_RENDER_INTERVAL_MS
        )
        if not self.live_update_timer.isActive():
            self.live_update_timer.start()

    def _flush_live_update(self) -> None:
        bar = self.live_pending_bar
        self.live_pending_bar = None
        if not bar or not self.current_symbol:
            return
        symbol = str(bar["symbol"])
        history = self._get_history_frame(symbol, self.current_interval)
        chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        if history is None or chart_history is None:
            return

        self._set_history_frame(symbol, _merge_live_bar(history, bar), self.current_interval)
        chart_rows = min(
            max(len(chart_history), _initial_chart_bar_limit(self.current_interval)),
            CHART_HISTORY_BAR_LIMIT,
        )
        self._set_chart_history_frame(
            symbol,
            _merge_live_bar(chart_history, bar, max_rows=chart_rows),
            self.current_interval,
        )
        if bool(bar.get("closed")):
            self._mark_history_refreshed(symbol, time.time(), self.current_interval)
        self._apply_live_position_price(symbol, float(bar["close"]))
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
        self._refresh_live_preview_markers(symbol)

    def _schedule_live_backtest(self, symbol: str) -> None:
        if symbol != self.current_symbol:
            return
        if self.live_backtest_worker and self.live_backtest_worker.isRunning():
            self.live_recalc_pending = True
            return
        history = self._get_history_frame(symbol, self.current_interval)
        chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        if history is None or chart_history is None:
            return
        worker = LiveBacktestWorker(
            self.settings,
            symbol,
            history,
            chart_history,
            self._active_backtest_settings(symbol),
            self.current_backtest,
        )
        worker.completed.connect(self._on_live_backtest_completed)
        worker.failed.connect(self._on_live_backtest_failed)
        self.live_backtest_worker = worker
        self._track_thread(worker, "live_backtest_worker")
        self.live_backtest_started_at = time.perf_counter()
        worker.start()

    def _on_live_backtest_completed(self, payload: object) -> None:
        result = dict(payload)
        symbol = str(result["symbol"])
        if symbol != self.current_symbol:
            return
        perf = dict(result.get("perf") or {})
        self._log_perf_ms(f"{symbol} live worker backtest", float(perf.get("worker_backtest_ms", 0.0) or 0.0))
        previous_backtest = self.current_backtest
        previous_chart_indicators = self.current_chart_indicators
        history = result["history"]
        chart_history = result["chart_history"]
        self._set_history_frame(symbol, history, self.current_interval)
        self._set_chart_history_frame(symbol, chart_history, self.current_interval)
        self.current_backtest = result["backtest"]
        self.current_chart_indicators = result["chart_indicators"]
        cache_key = self._symbol_interval_key(symbol, self.current_interval)
        self.backtest_cache[cache_key] = self.current_backtest
        self.chart_indicator_cache[cache_key] = self.current_chart_indicators
        self._prune_caches()
        apply_started_at = time.perf_counter()
        applied_incrementally = self._apply_incremental_lightweight_backtest(
            symbol,
            previous_backtest,
            previous_chart_indicators,
            self.current_backtest,
            self.current_chart_indicators,
            chart_history,
        )
        if not applied_incrementally:
            self.render_chart(symbol, self.current_backtest, reset_view=False, chart_indicators=self.current_chart_indicators)
        self._log_perf(f"{symbol} live chart apply", apply_started_at)
        self.update_summary(symbol, self.current_backtest, self._optimization_result(symbol, self.current_interval))
        if self.live_backtest_started_at > 0:
            self._log_perf(f"{symbol} live backtest", self.live_backtest_started_at)
            self.live_backtest_started_at = 0.0
        self._evaluate_backtest_auto_close(symbol, self.current_backtest)
        if self.live_recalc_pending:
            self.live_recalc_pending = False
            self._schedule_live_backtest(symbol)

    def _on_live_backtest_failed(self, message: str) -> None:
        self.live_recalc_pending = False
        self.live_backtest_started_at = 0.0
        self.log(message)

    def _set_order_buttons_enabled(self, enabled: bool) -> None:
        for button in self.long_buttons + self.short_buttons + self.simple_order_buttons:
            button.setEnabled(enabled)
        self.compound_order_radio.setEnabled(enabled)
        self.simple_order_radio.setEnabled(enabled)
        self.simple_order_amount_spin.setEnabled(enabled)
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

    def _set_bar_close_countdown_text(self, countdown: Optional[str]) -> None:
        if not hasattr(self, "bar_close_countdown_label"):
            return
        self.bar_close_countdown_label.setText(f"봉마감: {countdown}" if countdown else "봉마감: -")

    def _set_chart_interval_text(self, interval: Optional[str]) -> None:
        if not hasattr(self, "chart_interval_label"):
            return
        self.chart_interval_label.setText(f"차트TF: {interval}" if interval else "차트TF: -")

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
        frame = self._get_chart_history_frame(position.symbol, self.current_interval)
        if frame is None or frame.empty:
            frame = self._get_history_frame(position.symbol, self.current_interval)
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
            self._set_chart_interval_text(None)
            self.current_price_label.setText("현재가: -")
            self._set_bar_close_countdown_text(None)
            self._set_lightweight_bar_close_overlay(None, None)
            return

        self._set_chart_interval_text(self.current_interval or self.settings.kline_interval)
        frame = self._get_chart_history_frame(symbol, self.current_interval)
        if frame is None or frame.empty:
            frame = self._get_history_frame(symbol, self.current_interval)
        if frame is None or frame.empty:
            self.current_price_label.setText("현재가: -")
            self._set_bar_close_countdown_text(None)
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
            self._set_bar_close_countdown_text(None)
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
        self._set_bar_close_countdown_text(countdown)
        self._set_lightweight_bar_close_overlay(countdown, latest_price)

    def _position_display_values(self, position: PositionSnapshot) -> Tuple[List[str], float, float]:
        side = "LONG" if position.amount > 0 else "SHORT"
        entry_text = f"{position.entry_price:.8f}".rstrip("0").rstrip(".")
        upnl_value = float(position.unrealized_pnl)
        return_pct = _position_return_pct(position)
        return (
            [
                self._position_symbol_text(position.symbol),
                side,
                f"{abs(position.amount):.6f}",
                entry_text,
                f"{upnl_value:.2f}",
                f"{return_pct:+.2f}%",
            ],
            upnl_value,
            return_pct,
        )

    def _pnl_color(self, value: float) -> str:
        if value > 0:
            return "#17c964"
        if value < 0:
            return "#f31260"
        return "#1f2937"

    def _build_position_metric_widget(self, text: str, value: float) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"font-weight: 700; color: {self._pnl_color(value)}; padding-left: 4px; padding-right: 4px; background: transparent;"
        )
        return label

    def _position_status_html(self, position: PositionSnapshot) -> str:
        side = "LONG" if position.amount > 0 else "SHORT"
        upnl_value = float(position.unrealized_pnl)
        return_pct = _position_return_pct(position)
        upnl_color = self._pnl_color(upnl_value)
        return_color = self._pnl_color(return_pct)
        return (
            f"<span style='color:#111827;'>포지션: {self._position_symbol_text(position.symbol)} {side} {abs(position.amount):.6f} @ {position.entry_price:.6f} | </span>"
            f"<span style='font-weight:700; color:{upnl_color};'>UPnL {upnl_value:.2f}</span> | "
            f"<span style='font-weight:700; color:{return_color};'>수익률 {return_pct:+.2f}%</span>"
        )

    def _populate_position_row(self, row: int, position: PositionSnapshot) -> None:
        values, upnl_value, return_pct = self._position_display_values(position)
        for col, value in enumerate(values):
            if col in (4, 5):
                placeholder = QTableWidgetItem("")
                placeholder.setData(USER_ROLE, position.symbol)
                self.positions_table.setItem(row, col, placeholder)
                metric_value = upnl_value if col == 4 else return_pct
                widget = self._build_position_metric_widget(value, metric_value)
                self.positions_table.setCellWidget(row, col, widget)
                continue
            item = QTableWidgetItem(value)
            item.setData(USER_ROLE, position.symbol)
            self.positions_table.setItem(row, col, item)

    def _refresh_position_status_label(self) -> None:
        if self.current_symbol:
            position = self.current_position_snapshot
            if position is None:
                self.position_label.setText("포지션: 없음")
            else:
                side = "LONG" if position.amount > 0 else "SHORT"
                self.position_label.setText(
                    f"포지션: {side} {abs(position.amount):.6f} @ {position.entry_price:.6f} | "
                    f"UPnL {position.unrealized_pnl:.2f} | 수익률 {_position_return_pct(position):+.2f}%"
                )
        else:
            self.position_label.setText("포지션: 종목 미선택")

    def _apply_live_position_price(self, symbol: str, mark_price: float) -> None:
        if mark_price <= 0:
            return
        updated_positions: List[PositionSnapshot] = []
        changed = False
        for position in self.open_positions:
            if position.symbol != symbol:
                updated_positions.append(position)
                continue
            updated_positions.append(
                PositionSnapshot(
                    symbol=position.symbol,
                    amount=position.amount,
                    entry_price=position.entry_price,
                    mark_price=mark_price,
                    unrealized_pnl=(mark_price - position.entry_price) * position.amount,
                    leverage=position.leverage,
                )
            )
            changed = True
        if not changed:
            return
        self.open_positions = updated_positions
        self._refresh_balance_label_values()
        if self.current_symbol == symbol:
            self.current_position_snapshot = self._find_open_position(symbol)
            self._refresh_position_status_label()
        if not hasattr(self, "positions_table"):
            return
        for row in range(self.positions_table.rowCount()):
            item = self.positions_table.item(row, 0)
            if item is None:
                continue
            if (item.data(USER_ROLE) or item.text()) != symbol:
                continue
            position = next((entry for entry in self.open_positions if entry.symbol == symbol), None)
            if position is not None:
                self._populate_position_row(row, position)
            break

    def _refresh_position_status_label(self) -> None:
        if self.current_symbol:
            position = self.current_position_snapshot
            if position is None:
                self.position_label.setText("포지션: 없음")
            else:
                self.position_label.setText(self._position_status_html(position))
        else:
            self.position_label.setText("포지션: 종목 미선택")

    def update_positions_table(self) -> None:
        if not hasattr(self, "positions_table"):
            return
        self.position_metric_widgets.clear()
        self.position_action_widgets.clear()
        self.position_close_buttons.clear()
        self.positions_table.clearContents()
        self.positions_table.setRowCount(len(self.open_positions))
        for row, position in enumerate(self.open_positions):
            self._populate_position_row(row, position)
            button = QPushButton("청산")
            button.clicked.connect(lambda _=False, symbol=position.symbol: self.close_position_for_symbol(symbol))
            button.setText("청산")
            button.setFixedWidth(54)
            button.setMinimumHeight(24)
            self._style_close_button(button)

            auto_button = QPushButton()
            auto_button.setFixedWidth(124)
            auto_button.setMinimumHeight(24)
            self._set_auto_close_button_state(auto_button, position.symbol in self.auto_close_enabled_symbols)
            auto_button.toggled.connect(
                lambda checked, symbol=position.symbol: self._toggle_auto_close_for_symbol(symbol, checked)
            )

            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(4)
            action_layout.addWidget(button)
            action_layout.addWidget(auto_button)
            action_layout.addStretch(1)

            self.positions_table.setCellWidget(row, 6, action_widget)
            self.position_close_buttons.append(button)
        self._set_position_close_buttons_enabled(self.order_worker is None or not self.order_worker.isRunning())

    def update_candidate_table(self) -> None:
        self.candidate_table.setUpdatesEnabled(False)
        self.candidate_table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            values = [
                candidate.symbol,
                f"{candidate.daily_volatility_pct:.2f}",
                f"{candidate.atr_4h_pct:.2f}" if pd.notna(candidate.atr_4h_pct) else "-",
                f"{candidate.quote_volume:,.0f}",
                f"{candidate.rsi_1m:.2f}" if pd.notna(candidate.rsi_1m) else "-",
                f"{candidate.price_change_pct:.2f}",
                f"{candidate.last_price:.6f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, candidate.symbol)
                self.candidate_table.setItem(row, col, item)
        self.candidate_table.setUpdatesEnabled(True)

    def _ordered_optimized_results(self) -> List[OptimizationResult]:
        rank_mode = self._optimization_rank_mode()
        minimum_score = float(self.opt_min_score_spin.value()) if hasattr(self, "opt_min_score_spin") else 0.0
        minimum_return = float(self.opt_min_return_spin.value()) if hasattr(self, "opt_min_return_spin") else 0.0
        ordered = sorted(
            self.optimized_results.values(),
            key=lambda result: optimization_sort_key(result.best_backtest.metrics, rank_mode),
            reverse=True,
        )
        if rank_mode == "return":
            return [
                result
                for result in ordered
                if float(result.best_backtest.metrics.total_return_pct) >= minimum_return
            ]
        return [result for result in ordered if result.score >= minimum_score]

    def update_optimized_table(self) -> None:
        self.optimized_table.setUpdatesEnabled(False)
        ordered = self._ordered_optimized_results()
        self.optimized_table.setRowCount(len(ordered))
        for row, result in enumerate(ordered):
            metrics = result.best_backtest.metrics
            result_interval = result.best_interval or self.settings.kline_interval
            values = [
                result.symbol,
                result_interval,
                f"{result.score:.1f}",
                f"{metrics.total_return_pct:.2f}",
                f"{metrics.max_drawdown_pct:.2f}",
                str(metrics.trade_count),
                f"{metrics.win_rate_pct:.1f}",
                f"{metrics.profit_factor:.2f}",
                str(result.combinations_tested),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, (result.symbol, result_interval))
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

    def _set_backtest_progress_idle(self, text: str = "대기중") -> None:
        self.backtest_progress_phase = "idle"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_label.setText(text)
        self.backtest_progress_bar.setRange(0, 1)
        self.backtest_progress_bar.setValue(0)
        self.backtest_progress_bar.setFormat("%p%")

    def _set_backtest_progress_scanning(self) -> None:
        self.backtest_progress_phase = "scan"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_label.setText("후보 스캔중...")
        self.backtest_progress_bar.setRange(0, 0)
        self.backtest_progress_bar.setFormat("스캔중")

    def _begin_backtest_progress(self) -> None:
        self.backtest_progress_phase = "optimize"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_label.setText("백테스트중 0/0")
        self.backtest_progress_bar.setRange(0, 0)
        self.backtest_progress_bar.setFormat("준비중")

    def _register_backtest_case_plan(self, cases: int) -> None:
        if self.backtest_progress_phase != "optimize":
            return
        if cases <= 0:
            return
        self.backtest_progress_total_cases += int(cases)
        maximum = max(self.backtest_progress_total_cases, self.backtest_progress_completed_cases, 1)
        self.backtest_progress_bar.setRange(0, maximum)
        self.backtest_progress_bar.setValue(min(self.backtest_progress_completed_cases, maximum))
        self.backtest_progress_bar.setFormat("%p%")
        self.backtest_progress_label.setText(
            f"백테스트중 {self.backtest_progress_completed_cases}/{self.backtest_progress_total_cases}"
        )

    def _advance_backtest_progress(self, symbol: str, interval: str) -> None:
        if self.backtest_progress_phase != "optimize":
            return
        self.backtest_progress_completed_cases += 1
        maximum = max(self.backtest_progress_total_cases, self.backtest_progress_completed_cases, 1)
        self.backtest_progress_bar.setRange(0, maximum)
        self.backtest_progress_bar.setValue(min(self.backtest_progress_completed_cases, maximum))
        self.backtest_progress_bar.setFormat("%p%")
        self.backtest_progress_label.setText(
            f"백테스트중 {self.backtest_progress_completed_cases}/{maximum} | {symbol} [{interval}]"
        )

    def _finish_backtest_progress(self, text: Optional[str] = None) -> None:
        maximum = max(self.backtest_progress_total_cases, self.backtest_progress_completed_cases, 1)
        self.backtest_progress_phase = "done"
        self.backtest_progress_bar.setRange(0, maximum)
        self.backtest_progress_bar.setValue(min(maximum, self.backtest_progress_completed_cases or maximum))
        self.backtest_progress_bar.setFormat("%p%")
        self.backtest_progress_label.setText(
            text
            or f"백테스트 완료 {self.backtest_progress_completed_cases}/{max(self.backtest_progress_total_cases, self.backtest_progress_completed_cases)}"
        )

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
            self.history_refresh_times.clear()
            self.backtest_cache.clear()
            self.chart_indicator_cache.clear()
            self.current_symbol = None
            self.current_backtest = None
            self.current_chart_indicators = None
            self.update_candidate_table()
            self.update_optimized_table()
            self.summary_box.clear()
        self.log("후보 스캔 + 최적화 시작" + (" (기존 목록 유지)" if preserve_existing else ""))
        self._set_backtest_progress_scanning()
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
        self._set_backtest_progress_idle("백테스트 대상 없음")
        self._set_refresh_running(False)

    def start_optimization(self, targets: List[CandidateSymbol]) -> None:
        if self.optimize_worker and self.optimize_worker.isRunning():
            return
        if not targets:
            self._set_backtest_progress_idle("백테스트 대상 없음")
            self._set_refresh_running(False)
            return
        self.log(f"최적화 시작: {len(targets)}개 종목")
        self._begin_backtest_progress()
        self.optimize_worker = OptimizeWorker(self.settings, targets)
        self._track_thread(self.optimize_worker, "optimize_worker")
        self.optimize_worker.progress.connect(self.log)
        self.optimize_worker.case_plan.connect(self.on_optimization_case_plan)
        self.optimize_worker.result_ready.connect(self.on_optimization_result)
        self.optimize_worker.completed.connect(self.on_optimization_completed)
        self.optimize_worker.failed.connect(self.on_worker_failed)
        self.optimize_worker.start()

    def on_optimization_case_plan(self, payload: object) -> None:
        result = dict(payload)
        self._register_backtest_case_plan(int(result.get("cases", 0)))

    def on_optimization_result(self, payload: object) -> None:
        result = dict(payload)
        candidate: CandidateSymbol = result["candidate"]
        optimization: OptimizationResult = result["optimization"]
        history: pd.DataFrame = result["history"]
        interval = optimization.best_interval or self.settings.kline_interval
        self._advance_backtest_progress(candidate.symbol, interval)
        cache_key = self._symbol_interval_key(candidate.symbol, optimization.best_interval or self.settings.kline_interval)
        self.backtest_cache[cache_key] = optimization.best_backtest
        self.chart_indicator_cache[cache_key] = _chart_indicators_from_backtest(optimization.best_backtest)
        if self.preserve_lists_during_refresh:
            self.pending_optimized_results[cache_key] = optimization
            self._set_pending_history_frame(candidate.symbol, history, optimization.best_interval or self.settings.kline_interval)
            return
        self.optimized_results[cache_key] = optimization
        self._set_history_frame(candidate.symbol, history, optimization.best_interval or self.settings.kline_interval)
        self._mark_history_refreshed(candidate.symbol, time.time(), optimization.best_interval or self.settings.kline_interval)
        self._prune_caches()
        self._schedule_optimized_table_refresh()
        if candidate.symbol in self.auto_close_enabled_symbols:
            self._refresh_auto_close_monitors()

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
            refreshed_at = time.time()
            for cache_key in self.pending_history_cache:
                self.history_refresh_times[cache_key] = refreshed_at
            self.pending_candidates = []
            self.pending_optimized_results = {}
            self.pending_history_cache = {}
            self.preserve_lists_during_refresh = False
        self._prune_caches()
        self.log(f"최적화 완료: {len(self.optimized_results)}개 케이스")
        if not preserved_refresh:
            self._flush_optimized_table()
        if not preserved_refresh and self.optimized_table.rowCount() > 0:
            self.optimized_table.selectRow(0)
        self._refresh_auto_close_monitors()
        self._finish_backtest_progress()
        self._set_refresh_running(False)

    def on_worker_failed(self, message: str) -> None:
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.preserve_lists_during_refresh = False
        log_runtime_event("Worker Failure", message, open_notepad=False)
        self.log(message)
        self._set_backtest_progress_idle("백테스트 실패")
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

    def _request_symbol_load(self, symbol: str, interval: Optional[str] = None) -> None:
        if not symbol:
            return
        target_interval = interval if interval in APP_INTERVAL_OPTIONS else self._active_interval_for_symbol(symbol)
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
        self.load_symbol(symbol, target_interval)

    def _item_symbol_interval(self, item: Optional[QTableWidgetItem]) -> Tuple[Optional[str], Optional[str]]:
        if item is None:
            return None, None
        payload = item.data(USER_ROLE)
        if isinstance(payload, (tuple, list)) and len(payload) >= 2:
            symbol = str(payload[0] or "").strip()
            interval = str(payload[1] or "").strip()
            return (symbol or None, interval if interval in APP_INTERVAL_OPTIONS else None)
        symbol = str(payload or item.text() or "").strip()
        return (symbol or None, None)

    def on_candidate_selection_changed(self) -> None:
        selected = self.candidate_table.selectedItems()
        if selected:
            symbol, interval = self._item_symbol_interval(selected[0])
            if symbol:
                self._request_symbol_load(symbol, interval)

    def on_candidate_cell_clicked(self, row: int, _column: int) -> None:
        item = self.candidate_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval)

    def on_optimized_selection_changed(self) -> None:
        selected = self.optimized_table.selectedItems()
        if selected:
            symbol, interval = self._item_symbol_interval(selected[0])
            if symbol:
                self._request_symbol_load(symbol, interval)

    def on_optimized_cell_clicked(self, row: int, _column: int) -> None:
        item = self.optimized_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval)

    def on_positions_selection_changed(self) -> None:
        selected = self.positions_table.selectedItems()
        if selected:
            symbol, interval = self._item_symbol_interval(selected[0])
            if symbol:
                self._request_symbol_load(symbol, interval)

    def on_positions_cell_clicked(self, row: int, _column: int) -> None:
        item = self.positions_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval)

    def load_symbol(self, symbol: str, target_interval: Optional[str] = None) -> None:
        started_at = time.perf_counter()
        self.symbol_load_started_at = started_at
        self._sync_settings()
        self._stop_live_stream()
        self._stop_live_backtest_worker()
        self._stop_load_worker()
        self._stop_chart_history_page_worker()
        self.current_symbol = symbol
        self._remember_recent_symbol(symbol)
        self.current_position_snapshot = self._find_open_position(symbol)
        self._refresh_position_status_label()
        target_interval = target_interval if target_interval in APP_INTERVAL_OPTIONS else self._active_interval_for_symbol(symbol)
        optimization = self._optimization_result(symbol, target_interval)
        cache_key = self._symbol_interval_key(symbol, target_interval)
        self.current_interval = target_interval
        cached_history = self._get_history_frame(symbol, target_interval)
        cached_chart_history = self._get_chart_history_frame(symbol, target_interval)
        cached_backtest = self.backtest_cache.get(cache_key)
        if cached_backtest is None and optimization is not None and optimization.best_interval == target_interval:
            cached_backtest = optimization.best_backtest
        if cached_chart_history is None:
            cached_chart_history = self._build_initial_chart_history(symbol, target_interval, cached_history, cached_backtest)
            if cached_chart_history is not None and not cached_chart_history.empty:
                self._set_chart_history_frame(symbol, cached_chart_history, target_interval)
        cached_chart_indicators = self.chart_indicator_cache.get(cache_key)
        if cached_backtest is not None and cached_chart_history is not None:
            cached_chart_indicators = _chart_indicators_from_backtest(cached_backtest, cached_chart_history)
        elif cached_chart_indicators is None and cached_backtest is not None:
            cached_chart_indicators = _chart_indicators_from_backtest(cached_backtest, cached_chart_history)
        self.current_backtest = cached_backtest
        self.current_chart_indicators = cached_chart_indicators
        worker_backtest = cached_backtest
        self.load_request_id += 1
        self.load_request_reset_view[self.load_request_id] = cached_backtest is None
        self.chart_range_bars_before = float("inf")
        if cached_backtest is not None:
            self.render_chart(symbol, cached_backtest, reset_view=True, chart_indicators=cached_chart_indicators)
            self._defer_symbol_post_paint(
                symbol,
                self.load_request_id,
                update_summary=True,
                refresh_account=False,
                refresh_monitors=False,
                evaluate_auto_close=False,
            )
        worker = SymbolLoadWorker(
            self.load_request_id,
            self.settings,
            symbol,
            target_interval,
            cached_history,
            cached_chart_history,
            worker_backtest,
            self._history_last_refresh_at(symbol, target_interval),
        )
        worker.loaded.connect(self._on_symbol_loaded)
        worker.failed.connect(self._on_symbol_load_failed)
        self.load_worker = worker
        self._track_thread(worker, "load_worker")
        self._log_perf(f"{symbol} click setup", started_at)
        self.statusBar().showMessage(f"{symbol} 로드 중...", 3000)
        worker.start()

    def _on_symbol_loaded(self, payload: object) -> None:
        result = dict(payload)
        request_id = int(result["request_id"])
        symbol = str(result["symbol"])
        if request_id != self.load_request_id or symbol != self.current_symbol:
            self.load_request_reset_view.pop(request_id, None)
            return
        perf = dict(result.get("perf") or {})
        self._log_perf_ms(f"{symbol} worker fetch", float(perf.get("worker_fetch_ms", 0.0) or 0.0))
        self._log_perf_ms(f"{symbol} worker chart", float(perf.get("worker_chart_ms", 0.0) or 0.0))
        self._log_perf_ms(f"{symbol} worker backtest", float(perf.get("worker_backtest_ms", 0.0) or 0.0))
        reset_view = self.load_request_reset_view.pop(request_id, True)
        self.current_interval = str(result.get("interval", self.current_interval))
        self._set_history_frame(symbol, result["history"], self.current_interval)
        refreshed_at = result.get("history_refreshed_at")
        if refreshed_at is not None:
            self._mark_history_refreshed(symbol, float(refreshed_at), self.current_interval)
        existing_chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        loaded_chart_history = result["chart_history"]
        if existing_chart_history is not None:
            target_rows = min(
                max(len(existing_chart_history), len(loaded_chart_history), _initial_chart_bar_limit(self.current_interval)),
                CHART_HISTORY_BAR_LIMIT,
            )
            loaded_chart_history = _merge_ohlcv_frames(
                existing_chart_history,
                loaded_chart_history,
                max_rows=target_rows,
            )
        self._set_chart_history_frame(symbol, loaded_chart_history, self.current_interval)
        self.current_backtest = result["backtest"]
        loaded_chart_indicators = result.get("chart_indicators")
        if existing_chart_history is None and isinstance(loaded_chart_indicators, pd.DataFrame):
            self.current_chart_indicators = loaded_chart_indicators
        else:
            self.current_chart_indicators = _chart_indicators_from_backtest(self.current_backtest, loaded_chart_history)
        cache_key = self._symbol_interval_key(symbol, self.current_interval)
        self.backtest_cache[cache_key] = self.current_backtest
        self.chart_indicator_cache[cache_key] = self.current_chart_indicators
        self._prune_caches()
        apply_started_at = time.perf_counter()
        candle_df, indicators, equity_df, markers = self._build_chart_render_payload(
            symbol,
            self.current_backtest,
            self.current_chart_indicators,
        )
        render_signature = self._chart_render_signature_for_payload(candle_df, indicators, equity_df, markers)
        needs_render = (
            reset_view
            or bool(result.get("visible_slice_changed", True))
            or render_signature != self.chart_render_signature
        )
        if needs_render:
            self.render_chart(symbol, self.current_backtest, reset_view=reset_view, chart_indicators=self.current_chart_indicators)
        else:
            self.chart_render_signature = render_signature
            self.current_lightweight_markers = list(markers) if self.chart_mode == "Lightweight" else []
            self.current_lightweight_rendered_markers = list(markers) if self.chart_mode == "Lightweight" else []
            self.current_lightweight_preview_markers = []
            self._update_entry_price_overlay()
            self._refresh_live_labels()
            self._refresh_live_preview_markers(symbol)
        self._log_perf(f"{symbol} worker apply", apply_started_at)
        if self.symbol_load_started_at > 0:
            self._log_perf(f"{symbol} symbol load ready", self.symbol_load_started_at)
            self.symbol_load_started_at = 0.0
        self._defer_symbol_post_paint(
            symbol,
            request_id,
            update_summary=True,
            refresh_account=True,
            refresh_monitors=True,
            evaluate_auto_close=True,
        )
        self._start_live_stream(symbol)

    def _on_symbol_load_failed(self, message: str) -> None:
        self.load_request_reset_view.pop(self.load_request_id, None)
        self.symbol_load_started_at = 0.0
        self.show_error(message)

    def render_chart(
        self,
        symbol: str,
        result: BacktestResult,
        reset_view: bool = True,
        chart_indicators: Optional[pd.DataFrame] = None,
    ) -> None:
        candle_df, indicators, equity_df, markers = self._build_chart_render_payload(symbol, result, chart_indicators)
        self.current_lightweight_preview_markers = []
        if self.chart_mode == "Lightweight":
            self._render_lightweight_chart(symbol, candle_df, indicators, equity_df, markers, reset_view=reset_view)
            self.current_lightweight_markers = list(markers)
        else:
            self._render_plotly_chart(symbol, candle_df, indicators, equity_df, result.trades, reset_view=reset_view)
            self.current_lightweight_markers = []
        self.chart_render_signature = self._chart_render_signature_for_payload(candle_df, indicators, equity_df, markers)

        candidate = self._candidate_by_symbol(symbol)
        latest = result.latest_state
        self.symbol_label.setText(
            f"종목: {symbol} | TF {self.current_interval}"
            + (
                f" | DayVol {candidate.daily_volatility_pct:.2f}%"
                + (f" | ATR4h {candidate.atr_4h_pct:.2f}%" if candidate and pd.notna(candidate.atr_4h_pct) else "")
                + (f" | RSI1m {candidate.rsi_1m:.2f}" if candidate and pd.notna(candidate.rsi_1m) else "")
                if candidate
                else ""
            )
        )
        self.signal_label.setText(
            f"신호: Trend {latest['trend']} | Zone {latest['zone']} | "
            f"Bull {latest['final_bull']} | Bear {latest['final_bear']} | RSI {latest['rsi']:.2f}"
        )
        self._update_entry_price_overlay()
        self._refresh_live_labels()
        self._refresh_live_preview_markers(symbol)

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
        markers: List[Dict[str, object]],
        reset_view: bool = True,
    ) -> None:
        if self.chart is None:
            self._rebuild_chart_engine(force=True)
        if not reset_view:
            self._stash_lightweight_range()
        self.chart.set(candle_df)
        self._apply_lightweight_precision(symbol, candle_df)
        self.supertrend_line.set(indicators[["time", "supertrend"]].rename(columns={"supertrend": "Supertrend"}))
        self.zone2_line.set(indicators[["time", "zone2_line"]].rename(columns={"zone2_line": "Zone 2"}))
        self.zone3_line.set(indicators[["time", "zone3_line"]].rename(columns={"zone3_line": "Zone 3"}))
        self.ema_fast_line.set(indicators[["time", "ema_fast"]].rename(columns={"ema_fast": "EMA Fast"}))
        self.ema_slow_line.set(indicators[["time", "ema_slow"]].rename(columns={"ema_slow": "EMA Slow"}))
        self.equity_line.set(equity_df)
        range_start, range_end = self._default_chart_time_range(candle_df)
        self._render_lightweight_markers(markers)
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

    def _marker_prefix_matches(
        self,
        existing_markers: List[Dict[str, object]],
        new_markers: List[Dict[str, object]],
    ) -> bool:
        if len(existing_markers) > len(new_markers):
            return False
        existing_signature = [self._marker_signature(marker) for marker in existing_markers]
        new_prefix_signature = [self._marker_signature(marker) for marker in new_markers[: len(existing_markers)]]
        return existing_signature == new_prefix_signature

    def _update_lightweight_line_point(
        self,
        line,
        label: str,
        time_value: pd.Timestamp,
        value: object,
    ) -> None:
        if line is None:
            return
        series = pd.Series(
            {
                "time": pd.Timestamp(time_value),
                label: None if pd.isna(value) else float(value),
            }
        )
        line.update(series)

    def _apply_incremental_lightweight_backtest(
        self,
        symbol: str,
        previous_backtest: Optional[BacktestResult],
        previous_indicators: Optional[pd.DataFrame],
        new_backtest: BacktestResult,
        new_indicators: Optional[pd.DataFrame],
        chart_history: pd.DataFrame,
    ) -> bool:
        if (
            symbol != self.current_symbol
            or self.chart_mode != "Lightweight"
            or self.chart is None
            or self.equity_line is None
            or previous_backtest is None
            or previous_indicators is None
            or previous_indicators.empty
            or new_indicators is None
            or new_indicators.empty
            or chart_history.empty
        ):
            return False
        previous_frame = previous_indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        new_frame = new_indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        if previous_frame.empty or new_frame.empty or len(new_frame) < len(previous_frame):
            return False
        previous_times = pd.to_datetime(previous_frame["time"]).reset_index(drop=True)
        new_times = pd.to_datetime(new_frame["time"]).reset_index(drop=True)
        appended_bar = len(new_times) == len(previous_times) + 1 and new_times.iloc[: len(previous_times)].equals(previous_times)
        same_range = len(new_times) == len(previous_times) and new_times.equals(previous_times)
        if not appended_bar and not same_range:
            return False
        try:
            latest_bar = chart_history[["time", "open", "high", "low", "close", "volume"]].iloc[-1].copy()
            self.chart.update(latest_bar)
            latest_indicator = new_frame.iloc[-1]
            latest_time = pd.Timestamp(latest_indicator["time"])
            self._update_lightweight_line_point(self.supertrend_line, "Supertrend", latest_time, latest_indicator.get("supertrend"))
            self._update_lightweight_line_point(self.zone2_line, "Zone 2", latest_time, latest_indicator.get("zone2_line"))
            self._update_lightweight_line_point(self.zone3_line, "Zone 3", latest_time, latest_indicator.get("zone3_line"))
            self._update_lightweight_line_point(self.ema_fast_line, "EMA Fast", latest_time, latest_indicator.get("ema_fast"))
            self._update_lightweight_line_point(self.ema_slow_line, "EMA Slow", latest_time, latest_indicator.get("ema_slow"))
            equity_df = (
                pd.DataFrame({"time": list(new_backtest.equity_curve.index), "Equity": list(new_backtest.equity_curve.values)})
                .sort_values("time")
                .drop_duplicates(subset=["time"])
                .reset_index(drop=True)
            )
            if equity_df.empty:
                return False
            self.equity_line.update(equity_df.iloc[-1].copy())
            new_markers = self._trade_markers(new_backtest.trades, pd.Timestamp(chart_history["time"].iloc[-1]))
            self._render_lightweight_markers(new_markers)
            candle_df = new_frame[["time", "open", "high", "low", "close", "volume"]].copy()
            self.chart_render_signature = self._chart_render_signature_for_payload(candle_df, new_frame, equity_df, new_markers)
            self._update_entry_price_overlay()
            self._refresh_live_labels()
            self._refresh_live_preview_markers(symbol)
            return True
        except Exception:
            return False

    def _defer_symbol_post_paint(
        self,
        symbol: str,
        request_id: int,
        *,
        update_summary: bool,
        refresh_account: bool,
        refresh_monitors: bool,
        evaluate_auto_close: bool,
    ) -> None:
        QTimer.singleShot(
            0,
            lambda s=symbol, rid=request_id, us=update_summary, ra=refresh_account, rm=refresh_monitors, ea=evaluate_auto_close:
            self._run_symbol_post_paint(s, rid, update_summary=us, refresh_account=ra, refresh_monitors=rm, evaluate_auto_close=ea),
        )

    def _run_symbol_post_paint(
        self,
        symbol: str,
        request_id: int,
        *,
        update_summary: bool,
        refresh_account: bool,
        refresh_monitors: bool,
        evaluate_auto_close: bool,
    ) -> None:
        if symbol != self.current_symbol or request_id != self.load_request_id:
            return
        if update_summary and self.current_backtest is not None:
            self.update_summary(symbol, self.current_backtest, self._optimization_result(symbol, self.current_interval))
        if refresh_monitors:
            self._refresh_auto_close_monitors()
        if evaluate_auto_close and self.current_backtest is not None:
            self._evaluate_backtest_auto_close(symbol, self.current_backtest)
        if refresh_account:
            self.refresh_account_info()

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
            f"Score: {optimization.score:.1f}" if optimization else "Score: -",
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

    def _balance_label_html(self, body: str) -> str:
        return (
            '<span style="color: #000000; font-weight: 700;">잔고:</span> '
            f'<span style="color: #000000; font-weight: 700;">{body}</span>'
        )

    def _set_balance_label_status(self, status: str) -> None:
        self.balance_label.setText(self._balance_label_html(status))

    def _set_balance_label_values(self, equity: float, available: float) -> None:
        body = (
            'Equity <span style="color: #1546b0; font-weight: 700;">'
            f"{equity:.2f}"
            '</span> USDT | Available <span style="color: #1546b0; font-weight: 700;">'
            f"{available:.2f}"
            "</span> USDT"
        )
        self.balance_label.setText(self._balance_label_html(body))

    def _live_total_unrealized_pnl(self) -> float:
        return float(sum(float(position.unrealized_pnl) for position in self.open_positions))

    def _refresh_balance_label_values(self) -> None:
        snapshot = self.account_balance_snapshot
        if snapshot is None:
            return
        live_unrealized = self._live_total_unrealized_pnl()
        unrealized_delta = live_unrealized - float(snapshot.unrealized_pnl)
        self._set_balance_label_values(
            float(snapshot.equity) + unrealized_delta,
            float(snapshot.available_balance) + unrealized_delta,
        )

    def refresh_account_info(self) -> None:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self._stop_account_worker()
            self.open_positions = []
            self.current_position_snapshot = None
            self.account_balance_snapshot = None
            self._refresh_position_price_streams()
            self._set_balance_label_status("API 미입력")
            self.position_label.setText("포지션: API 미입력")
            self._refresh_auto_close_monitors()
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
        open_symbols = {entry.symbol for entry in self.open_positions}
        self._forget_closed_position_intervals(open_symbols, persist=False)
        self._remember_missing_open_position_intervals(open_symbols)
        if self.current_symbol and requested_symbol == self.current_symbol:
            self.current_position_snapshot = position
        elif self.current_symbol:
            self.current_position_snapshot = self._find_open_position(self.current_symbol)
        else:
            self.current_position_snapshot = None
        self.account_balance_snapshot = balance
        self._refresh_balance_label_values()
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
        self._refresh_position_status_label()
        self._refresh_position_price_streams()
        self._refresh_auto_close_monitors()
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
        self.account_balance_snapshot = None
        self._refresh_position_price_streams()
        self._set_balance_label_status("조회 실패")
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
        self._submit_open_order(side, fraction=fraction)

    def place_simple_order(self, side: str) -> None:
        amount = float(self.simple_order_amount_spin.value())
        self._submit_open_order(side, margin=amount)

    def _submit_open_order(
        self,
        side: str,
        fraction: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> None:
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
        if margin is not None and margin <= 0:
            self.show_warning("?⑤━ 二쇰Ц 湲덉븸??0蹂대떎 而ㅼ빞 ?⑸땲??")
            return
        worker = OrderWorker(
            self.settings.api_key,
            self.settings.api_secret,
            self.current_symbol,
            self.settings.leverage,
            side=side,
            fraction=fraction,
            margin=margin,
        )
        worker.completed.connect(self._on_order_completed)
        worker.failed.connect(self._on_order_failed)
        self.order_worker = worker
        self.order_worker_symbol = self.current_symbol
        self.order_worker_is_auto_close = False
        self.pending_open_order_interval = self.current_interval
        self._track_thread(worker, "order_worker")
        self._set_order_buttons_enabled(False)
        self.statusBar().showMessage(f"{self.current_symbol} 주문 처리 중...", 3000)
        worker.start()

    def close_selected_position(self) -> None:
        if not self.current_symbol:
            self.show_warning("종목을 먼저 선택하세요.")
            return
        self.close_position_for_symbol(self.current_symbol)

    def _submit_close_position(self, symbol: str, auto_close_reason: Optional[str] = None) -> bool:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            if auto_close_reason is None:
                self.show_warning("API Key / Secret을 입력해야 실제 주문이 가능합니다.")
            else:
                self.log(f"{symbol} 자동청산 실패: API Key / Secret이 없습니다.")
            return False
        if self.order_worker is not None and self.order_worker.isRunning():
            if auto_close_reason is None:
                self.show_warning("이미 주문 처리 중입니다.")
            return False
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
        self.order_worker_symbol = symbol
        self.order_worker_is_auto_close = auto_close_reason is not None
        self.pending_open_order_interval = None
        self._track_thread(worker, "order_worker")
        self._set_order_buttons_enabled(False)
        if auto_close_reason is None:
            self.statusBar().showMessage(f"{symbol} 청산 처리 중...", 3000)
        else:
            self.log(f"{symbol} 자동청산 실행: {_auto_close_reason_text(auto_close_reason)}")
            self.statusBar().showMessage(f"{symbol} 자동청산 처리 중...", 3000)
        worker.start()
        return True

    def close_position_for_symbol(self, symbol: str) -> None:
        self._submit_close_position(symbol)

    def _on_order_completed(self, payload: object) -> None:
        self._set_order_buttons_enabled(True)
        result = dict(payload)
        order_symbol = self.order_worker_symbol or str(result.get("symbol", ""))
        if self.order_worker_is_auto_close and order_symbol:
            self.auto_close_order_pending.discard(order_symbol)
        if not self.order_worker_is_auto_close and order_symbol:
            self._remember_position_interval(order_symbol, self.pending_open_order_interval)
        self.order_worker_symbol = None
        self.order_worker_is_auto_close = False
        self.pending_open_order_interval = None
        self.log(str(result.get("message", "주문 완료")))
        self.refresh_account_info()
        QTimer.singleShot(0, self._flush_queued_auto_close_orders)

    def _on_order_failed(self, message: str) -> None:
        order_symbol = self.order_worker_symbol
        was_auto_close = self.order_worker_is_auto_close
        if was_auto_close and order_symbol:
            self.auto_close_order_pending.discard(order_symbol)
            self.auto_close_last_trigger_time.pop(order_symbol, None)
        self.order_worker_symbol = None
        self.order_worker_is_auto_close = False
        self.pending_open_order_interval = None
        self._set_order_buttons_enabled(True)
        self.log(message)
        if not was_auto_close:
            self.show_error("주문 처리 중 오류가 발생했습니다. 로그를 확인하세요.")
        QTimer.singleShot(0, self._flush_queued_auto_close_orders)

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
            self._stop_chart_history_page_worker()
            self._stop_live_backtest_worker()
            self._stop_account_worker()
            self._stop_order_worker()
            self._stop_live_stream()
            for symbol in list(self.position_price_workers):
                self._stop_position_price_worker(symbol)
            self._stop_all_auto_close_monitors()
            self._drain_tracked_threads()
            self.save_settings()
        finally:
            super().closeEvent(event)


def create_app() -> QApplication:
    return QApplication.instance() or QApplication([])
