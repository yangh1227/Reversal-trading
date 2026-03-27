from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import multiprocessing as mp
from numbers import Number
import time
from typing import Dict, List, Optional, Tuple
import traceback

import pandas as pd
from lightweight_charts.widgets import QtChart
import websocket

from .auto_trade_runtime import (
    auto_trade_signal_from_backtest,
    evaluate_auto_trade_candidate,
    history_can_resume_backtest as _history_can_resume_backtest,
    inferred_auto_trade_fraction as _inferred_auto_trade_fraction,
    pick_auto_trade_candidate,
    resolve_favorable_auto_trade_zone,
    zone_favorable_fraction,
)
from .binance_futures import (
    BalanceSnapshot,
    BinanceFuturesClient,
    CandidateSymbol,
    PositionSnapshot,
    resample_ohlcv,
    resolve_base_interval,
)
from .config import (
    APP_INTERVAL_OPTIONS,
    PARAMETER_SPECS,
    STRATEGY_TYPE_LABELS,
    STRATEGY_TYPE_OPTIONS,
    AppSettings,
    StrategySettings,
    parameter_spec_applies,
)
from .crash_logger import log_runtime_event
from .favorable_backtest_process import (
    FavorableBacktestJob,
    FavorableBacktestProcess,
    FavorableBacktestResultPayload,
)
from .live_chart_utils import (
    history_with_live_preview as _history_with_live_preview,
    merge_live_bar as _merge_live_bar,
    preview_bar_matches_context as _preview_bar_matches_context,
    seed_two_minute_aggregate as _seed_two_minute_aggregate,
    transform_two_minute_bar as _transform_two_minute_bar,
)
from .optimizer import OptimizationResult, optimization_sort_key, optimize_symbol_interval_results, parameter_value_range
from .qt_compat import (
    EVENT_KEY_PRESS,
    HORIZONTAL,
    KEY_DOWN,
    KEY_UP,
    NO_EDIT_TRIGGERS,
    PASSWORD_ECHO,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
    VERTICAL,
    QApplication,
    QCheckBox,
    QColor,
    QComboBox,
    QFontMetrics,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHeaderView,
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
    QPropertyAnimation,
    QThread,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
)
from .strategy import (
    CHART_INDICATOR_COLUMNS,
    BacktestResult,
    _preview_entry_signal,
    _preview_exit_reason,
    compact_indicator_frame,
    evaluate_latest_state,
    estimate_warmup_bars,
    fresh_entry_trigger_time,
    latest_confirmed_entry_event,
    prepare_ohlcv,
    resume_backtest,
    run_backtest,
    signal_fraction_for_zone,
)
from .telegram_notifier import TelegramNotifier
from .trade_engine import (
    EngineCloseAllPositionsCommand,
    EngineCloseOrderCommand,
    EngineHealthEvent,
    EngineLogEvent,
    EngineOpenOrderCommand,
    EngineOrderCompletedEvent,
    EngineOrderFailedEvent,
    EngineOrderSubmittedEvent,
    EngineSignalEvent,
    EngineSyncCommand,
    EngineWatchlistItem,
    TradeEngineController,
)


CHART_HISTORY_BAR_LIMIT = 8_000
BACKTEST_WARMUP_BAR_FLOOR = 1_500
DEFAULT_CHART_LOOKBACK_HOURS = 3
DEFAULT_CHART_RIGHT_PAD_BARS = 8
INITIAL_CHART_WARMUP_BARS = 300
INITIAL_CHART_BAR_FLOOR = 360
CHART_LAZY_LOAD_CHUNK_BARS = 300
CHART_LAZY_LOAD_TRIGGER_BARS = 30
RECENT_DELTA_REFRESH_BARS = 180
RECENT_DELTA_OVERLAP_BARS = 20
LIVE_RENDER_INTERVAL_MS = 120
OPTIMIZED_TABLE_REFRESH_MS = 250
OPTIMIZED_TABLE_HIGHLIGHT_REFRESH_MS = 1_000
HISTORY_CACHE_SYMBOL_LIMIT = 10
RECENT_SYMBOL_CACHE_LIMIT = 8
CANDIDATE_DEFAULT_INTERVAL = "1m"
FULL_HISTORY_REFRESH_COOLDOWN_SECONDS = 300.0
PERFORMANCE_LOG_THRESHOLD_MS = 100.0
AUTO_TRADE_INTERVAL_MS = 1_000
AUTO_CLOSE_RETRY_INTERVAL_MS = 10_000
TRADE_ENGINE_POLL_INTERVAL_MS = 100
ORDER_PENDING_RECOVERY_SECONDS = 35.0
_SNAPSHOT_KEEP = object()
OPTIMIZED_TABLE_FAVORABLE_ROW_COLOR = "#d8f0de"
OPTIMIZED_TABLE_SIGNAL_ROW_COLOR = "#f7dddd"


@dataclass
class AuthoritativeChartSnapshot:
    symbol: str
    interval: str
    confirmed_history: Optional[pd.DataFrame] = None
    confirmed_chart_history: Optional[pd.DataFrame] = None
    backtest: Optional[BacktestResult] = None
    chart_indicators: Optional[pd.DataFrame] = None
    preview_bar: Optional[Dict[str, object]] = None
    render_signature: Tuple[object, ...] = (None, 0)

    def display_history(self) -> Optional[pd.DataFrame]:
        return _history_with_live_preview(self.confirmed_history, self.preview_bar)

    def display_chart_history(self) -> Optional[pd.DataFrame]:
        return _history_with_live_preview(
            self.confirmed_chart_history,
            self.preview_bar,
            max_rows=CHART_HISTORY_BAR_LIMIT,
        )

    def latest_source_time(self) -> Optional[pd.Timestamp]:
        preview_time = pd.Timestamp(self.preview_bar["time"]) if self.preview_bar is not None else None
        chart_time = _frame_last_time(self.confirmed_chart_history)
        history_time = _frame_last_time(self.confirmed_history)
        candidates = [value for value in (preview_time, chart_time, history_time) if value is not None]
        return max(candidates) if candidates else None


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


def _optimization_result_interval(result: OptimizationResult) -> str:
    interval = str(getattr(result, "best_interval", "") or "").strip()
    return interval if interval in APP_INTERVAL_OPTIONS else CANDIDATE_DEFAULT_INTERVAL


def _ws_kline_timestamp(time_ms: int) -> pd.Timestamp:
    return pd.to_datetime(int(time_ms), unit="ms", utc=True).tz_convert(None)


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


def _slice_recent_ohlcv(
    frame: Optional[pd.DataFrame],
    interval: str,
    max_bars: Optional[int] = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    limit = max_bars or _initial_chart_bar_limit(interval)
    prepared = prepare_ohlcv(frame.copy())
    if prepared.empty:
        return prepared
    return prepared.tail(limit).reset_index(drop=True)


def _ohlcv_from_indicator_frame(
    frame: Optional[pd.DataFrame],
    interval: str,
) -> pd.DataFrame:
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
    history_signature = _history_frame_signature(history)
    if getattr(backtest, "history_signature", ()):
        return tuple(backtest.history_signature) == history_signature
    return _history_frame_signature(backtest.indicators) == history_signature


def _history_frame_signature(frame: Optional[pd.DataFrame]) -> Tuple[object, ...]:
    if frame is None or frame.empty or "time" not in frame.columns:
        return (None, 0)
    last = frame.iloc[-1]
    values: List[object] = [pd.Timestamp(last["time"]), int(len(frame))]
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            continue
        value = last[column]
        values.append(None if pd.isna(value) else float(value))
    return tuple(values)


def _frame_last_time(frame: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if frame is None or frame.empty or "time" not in frame.columns:
        return None
    return pd.Timestamp(frame["time"].iloc[-1])


def _chart_candle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("time", "open", "high", "low", "close") if column in frame.columns]
    return frame.loc[:, columns].copy() if columns else pd.DataFrame(columns=["time", "open", "high", "low", "close"])


def _is_provisional_exit_trade(trade, latest_time: Optional[pd.Timestamp]) -> bool:
    _ = latest_time
    return str(getattr(trade, "reason", "")).strip() == "end_of_test"


def _latest_backtest_exit_event(backtest: Optional[BacktestResult]) -> Optional[Dict[str, object]]:
    if backtest is None or backtest.indicators.empty or not backtest.trades or "time" not in backtest.indicators.columns:
        return None
    latest_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    for trade in reversed(backtest.trades):
        if _is_provisional_exit_trade(trade, latest_time):
            continue
        exit_time = pd.Timestamp(trade.exit_time)
        if exit_time < latest_time:
            break
        if exit_time != latest_time:
            continue
        return {
            "side": str(trade.side).lower(),
            "reason": str(trade.reason),
            "bar_time": latest_time,
        }
    return None


def _backtest_has_latest_trade_marker_change(
    previous_backtest: Optional[BacktestResult],
    new_backtest: Optional[BacktestResult],
    latest_time: Optional[pd.Timestamp],
) -> bool:
    if previous_backtest is None or new_backtest is None or latest_time is None:
        return False
    previous_trades = list(previous_backtest.trades or ())
    new_trades = list(new_backtest.trades or ())
    if len(new_trades) <= len(previous_trades):
        return False
    latest_bar_time = pd.Timestamp(latest_time)
    for trade in new_trades[len(previous_trades) :]:
        if pd.Timestamp(trade.exit_time) == latest_bar_time:
            return True
        for event_time, _event_label in list(getattr(trade, "entry_events", ()) or ()):
            if pd.Timestamp(event_time) == latest_bar_time:
                return True
    previous_open_events = list(getattr(previous_backtest, "open_entry_events", ()) or ())
    new_open_events = list(getattr(new_backtest, "open_entry_events", ()) or ())
    if len(new_open_events) > len(previous_open_events):
        for event_time, _event_label in new_open_events[len(previous_open_events) :]:
            if pd.Timestamp(event_time) == latest_bar_time:
                return True
    return False


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
        "opposite_signal": "청산신호",
        "cross_lower": "켈트너 하단 이탈",
    }
    return labels.get(reason, reason)


def _confirmed_exit_event_from_state(
    position_qty: float,
    latest_state: Dict[str, object],
    bar_time: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    reason = _preview_exit_reason(position_qty, latest_state)
    if reason is None:
        return None
    if position_qty > 0:
        side = "long"
    elif position_qty < 0:
        side = "short"
    else:
        return None
    return {
        "side": side,
        "reason": reason,
        "bar_time": pd.Timestamp(bar_time),
    }


def _confirmed_entry_event_from_state(
    cursor,
    latest_state: Dict[str, object],
    settings: StrategySettings,
    bar_time: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    signal = _preview_entry_signal(cursor, latest_state, settings)
    if signal is None:
        return None
    side, zone = signal
    if side not in {"long", "short"}:
        return None
    return {
        "side": side,
        "zone": int(zone),
        "bar_time": pd.Timestamp(bar_time),
    }


def _confirmed_entry_event_from_cursor(
    cursor,
    bar_time: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    """Detect confirmed entry directly from cursor signal fields.

    Unlike ``_confirmed_entry_event_from_state`` (which re-derives the signal
    from zone-used flags that are already consumed after the backtest runs),
    this checks whether ``last_entry_signal_time`` matches *bar_time*.
    """
    if cursor is None:
        return None
    signal_time = getattr(cursor, "last_entry_signal_time", None)
    if signal_time is None or pd.Timestamp(signal_time) != pd.Timestamp(bar_time):
        return None
    side = str(getattr(cursor, "last_entry_signal_side", "") or "").lower()
    zone = int(getattr(cursor, "last_entry_signal_zone", 0) or 0)
    if side not in {"long", "short"} or zone not in {1, 2, 3}:
        return None
    return {"side": side, "zone": int(zone), "bar_time": pd.Timestamp(bar_time)}


def _position_return_pct(position: PositionSnapshot) -> float:
    notional = abs(float(position.amount)) * float(position.entry_price)
    leverage = max(1, int(position.leverage) or 1)
    margin = notional / leverage if notional > 0 else 0.0
    if margin <= 0:
        return 0.0
    return float(position.unrealized_pnl) / margin * 100.0


def _position_notional_usdt(position: PositionSnapshot) -> float:
    return abs(float(position.amount)) * float(position.entry_price)


class KlineStreamWorker(QThread):
    kline = Signal(object)
    status = Signal(str)

    def __init__(self, symbol: str, interval: str, seed_history: Optional[pd.DataFrame] = None) -> None:
        super().__init__()
        self.symbol = symbol.upper()
        self.interval = interval
        self.stream_interval = resolve_base_interval(interval)
        self._stopped = False
        self._socket = None
        self._aggregate_bar: Optional[Dict[str, object]] = None
        self._seed_history = prepare_ohlcv(seed_history.copy()) if seed_history is not None and not seed_history.empty else None

    def _initialize_aggregate_seed(self) -> None:
        if self.interval != "2m":
            return
        recent_history = self._seed_history
        if recent_history is None:
            try:
                recent_history = BinanceFuturesClient().historical_ohlcv_recent(self.symbol, self.stream_interval, bars=2)
            except Exception:
                recent_history = None
        self._aggregate_bar = _seed_two_minute_aggregate(recent_history, self.symbol, self.interval)
        self._seed_history = None

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
                self._initialize_aggregate_seed()
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
        seed_aggregate = None
        if self._aggregate_bar is None or pd.Timestamp(self._aggregate_bar["time"]) != pd.Timestamp(bar["time"]).floor("2min"):
            self._initialize_aggregate_seed()
            seed_aggregate = self._aggregate_bar
        self._aggregate_bar, transformed = _transform_two_minute_bar(
            self._aggregate_bar,
            bar,
            seed_aggregate=seed_aggregate,
        )
        return [transformed]


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
                quote_volume_min=(self.settings.surge_quote_volume_min if self.settings.filter_preset == "급등종목"
                                  else self.settings.quote_volume_min),
                use_rsi_filter=self.settings.use_rsi_filter,
                rsi_length=self.settings.rsi_length,
                rsi_lower=self.settings.rsi_lower,
                rsi_upper=self.settings.rsi_upper,
                use_atr_4h_filter=self.settings.use_atr_4h_filter,
                atr_4h_min_pct=self.settings.atr_4h_min_pct,
                use_surge_filter=(self.settings.filter_preset == "급등종목"),
                surge_price_change_min_pct=self.settings.surge_price_change_min_pct,
                surge_rsi_30m_min=self.settings.surge_rsi_30m_min,
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
    phase_update = Signal(object)
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
        return [CANDIDATE_DEFAULT_INTERVAL]

    def _effective_optimize_flags(self) -> Dict[str, bool]:
        if self.settings.enable_parameter_optimization:
            return dict(self.settings.optimize_flags)
        return {key: False for key in self.settings.optimize_flags}

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
            phase_name = "최적화" if self.settings.enable_parameter_optimization else "기본값 백테스트"
            self.phase_update.emit(
                {
                    "phase": "optimize_start",
                    "total_candidates": len(self.candidates),
                    "process_count": process_count,
                }
            )
            self.progress.emit(f"{phase_name} 워커 시작: 종목 {len(self.candidates)}개, 프로세스 {process_count}개")
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
        optimize_flags = self._effective_optimize_flags()
        history_fetch_start_time_ms = _history_fetch_start_time_ms(
            self.settings,
            interval="1m" if "2m" in interval_candidates else CANDIDATE_DEFAULT_INTERVAL,
        )
        for index, candidate in enumerate(self.candidates, start=1):
            if self.isInterruptionRequested():
                return
            self.phase_update.emit(
                {
                    "phase": "history_loading",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                }
            )
            self.progress.emit(
                f"[{index}/{len(self.candidates)}] {candidate.symbol} "
                f"{self.settings.history_days}일 백테스트 + 웜업 K라인 로드"
            )
            histories = self._load_histories(client, candidate.symbol, history_fetch_start_time_ms)
            if not histories:
                self.progress.emit(f"{candidate.symbol}: 히스토리 없음")
                continue
            self.phase_update.emit(
                {
                    "phase": "history_ready",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                    "cases": len(histories),
                }
            )
            self.case_plan.emit({"candidate": candidate.symbol, "cases": len(histories)})
            self.phase_update.emit(
                {
                    "phase": "case_running",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                    "active_jobs": 1,
                    "process_count": 1,
                }
            )
            interval_results = optimize_symbol_interval_results(
                symbol=candidate.symbol,
                histories_by_interval=histories,
                base_settings=self.settings.strategy,
                optimize_flags=optimize_flags,
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
        optimize_flags = self._effective_optimize_flags()
        history_fetch_start_time_ms = _history_fetch_start_time_ms(
            self.settings,
            interval="1m" if "2m" in interval_candidates else CANDIDATE_DEFAULT_INTERVAL,
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
            self.phase_update.emit(
                {
                    "phase": "history_loading",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                }
            )
            self.progress.emit(
                f"[{index}/{len(self.candidates)}] {candidate.symbol} "
                f"{self.settings.history_days}일 백테스트 + 웜업 K라인 로드"
            )
            histories = self._load_histories(client, candidate.symbol, history_fetch_start_time_ms)
            if not histories:
                self.progress.emit(f"{candidate.symbol}: 히스토리 없음")
                return False
            self.phase_update.emit(
                {
                    "phase": "history_ready",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                    "cases": len(histories),
                }
            )
            self.case_plan.emit({"candidate": candidate.symbol, "cases": len(histories)})
            job = pool.apply_async(
                optimize_symbol_interval_results,
                    kwds={
                        "symbol": candidate.symbol,
                        "histories_by_interval": histories,
                        "base_settings": self.settings.strategy,
                    "optimize_flags": optimize_flags,
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
            self.phase_update.emit(
                {
                    "phase": "case_running",
                    "candidate": candidate.symbol,
                    "index": index,
                    "total_candidates": len(self.candidates),
                    "active_jobs": len(active_jobs),
                    "process_count": process_count,
                }
            )
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
        strategy_settings: Optional["StrategySettings"] = None,
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
        self.strategy_settings = strategy_settings or settings.strategy

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
                history, history_updated, repair_needed = _refresh_cached_ohlcv_recent_delta(
                    client,
                    self.symbol,
                    self.interval,
                    history,
                )
                recent_delta_used = True
                latest_bar_time = pd.Timestamp(history["time"].iloc[-1]) if history is not None and not history.empty else None
                latest_bar_stale = (
                    latest_bar_time is None
                    or (pd.Timestamp.utcnow().tz_localize(None) - latest_bar_time)
                    >= pd.Timedelta(milliseconds=_interval_to_ms(self.interval) * 2)
                )
                should_full_refresh = (
                    repair_needed
                    or latest_bar_stale
                    or self.history_last_refresh_at is None
                    or (time.time() - float(self.history_last_refresh_at)) >= FULL_HISTORY_REFRESH_COOLDOWN_SECONDS
                )
                if should_full_refresh:
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
                and self.existing_backtest.settings == self.strategy_settings
                and _backtest_matches_history(self.existing_backtest, history)
            )
            if use_existing_backtest:
                backtest = self.existing_backtest
            elif _history_can_resume_backtest(self.existing_backtest, history) and self.existing_backtest.settings == self.strategy_settings:
                backtest = resume_backtest(
                    history,
                    previous_result=self.existing_backtest,
                    settings=self.strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            else:
                backtest = run_backtest(
                    history,
                    settings=self.strategy_settings,
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
        self.resolved_auto_trade_backtest_cache: Dict[Tuple[str, str], BacktestResult] = {}
        self.resolved_auto_trade_backtest_meta: Dict[Tuple[str, str], Tuple[Tuple[object, ...], str]] = {}
        self.chart_indicator_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.history_refresh_times: Dict[Tuple[str, str], float] = {}
        self.price_precision_cache: Dict[str, int] = {}
        self.pending_candidates: List[CandidateSymbol] = []
        self.pending_optimized_results: Dict[Tuple[str, str], OptimizationResult] = {}
        self.pending_history_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.pending_backtest_cache: Dict[Tuple[str, str], BacktestResult] = {}
        self.pending_chart_indicator_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.preserve_lists_during_refresh = False
        self.open_positions: List[PositionSnapshot] = []
        self.position_strategy_by_symbol: Dict[str, StrategySettings] = dict(self.settings.position_strategy_settings)
        self.position_open_entry_events_by_symbol: Dict[str, List[Tuple[pd.Timestamp, str]]] = {
            symbol: [(pd.Timestamp(event_time), str(label)) for event_time, label in events]
            for symbol, events in self.settings.position_open_entry_events.items()
        }
        self.current_position_snapshot: Optional[PositionSnapshot] = None
        self.current_symbol: Optional[str] = None
        self.current_interval = self.settings.kline_interval
        self.current_chart_prefers_locked_position_settings = False
        self.current_backtest: Optional[BacktestResult] = None
        self.current_chart_indicators: Optional[pd.DataFrame] = None
        self.scan_worker: Optional[ScanWorker] = None
        self.optimize_worker: Optional[OptimizeWorker] = None
        self.load_worker: Optional[SymbolLoadWorker] = None
        self.chart_history_page_worker: Optional[ChartHistoryPageWorker] = None
        self.live_backtest_worker: Optional[LiveBacktestWorker] = None
        self.account_worker: Optional[AccountInfoWorker] = None
        self.order_worker: Optional[OrderWorker] = None
        self.trade_engine: Optional[TradeEngineController] = None
        self.favorable_backtest_process = FavorableBacktestProcess()
        self.load_request_id = 0
        self.load_request_reset_view: Dict[int, bool] = {}
        self.load_request_targets: Dict[int, Tuple[str, str]] = {}
        self.account_request_id = 0
        self.live_recalc_pending = False
        self.pending_account_refresh = False
        self.live_stream_worker: Optional[KlineStreamWorker] = None
        self.position_price_workers: Dict[str, KlineStreamWorker] = {}
        self.account_balance_snapshot: Optional[BalanceSnapshot] = None
        self.live_pending_bar: Optional[Dict[str, object]] = None
        self.current_live_preview_bar: Optional[Dict[str, object]] = None
        self.recent_symbol_cache_keys: deque[str] = deque(maxlen=RECENT_SYMBOL_CACHE_LIMIT)
        self.chart_history_exhausted: Dict[Tuple[str, str], bool] = {}
        self.chart_history_load_pending = False
        self.chart_history_load_requested = False
        self.chart_range_bars_before = float("inf")
        self.pending_lightweight_range_shift = 0
        self.symbol_load_started_at = 0.0
        self.live_backtest_started_at = 0.0
        self.chart_render_signature: Tuple[object, ...] = (None, 0)
        self.current_chart_snapshot: Optional[AuthoritativeChartSnapshot] = None
        self.current_lightweight_markers: List[Dict[str, object]] = []
        self.current_lightweight_rendered_markers: List[Dict[str, object]] = []
        self.current_lightweight_preview_markers: List[Dict[str, object]] = []
        self.current_lightweight_fast_entry_markers: List[Dict[str, object]] = []
        self.current_lightweight_fast_exit_markers: List[Dict[str, object]] = []
        self.suppress_positions_selection_load = False
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
        self.auto_close_last_attempt_at: Dict[str, float] = {}
        self.auto_trade_enabled = False
        self.auto_trade_requested = False
        self.auto_close_retry_timer = QTimer(self)
        self.auto_trade_timer = QTimer(self)
        self.order_pending_started_at = 0.0
        self.auto_trade_entry_pending_symbol: Optional[str] = None
        self.auto_trade_entry_pending_at: float = 0.0
        self.auto_trade_entry_pending_fraction = 0.0
        self.auto_trade_entry_pending_cursor_time = None
        self.auto_trade_entry_pending_cursor_time: Optional[pd.Timestamp] = None
        self.auto_trade_filled_fraction_by_symbol: Dict[str, float] = dict(self.settings.position_filled_fractions)
        self.auto_trade_cursor_entry_time_by_symbol: Dict[str, pd.Timestamp] = dict(
            self.settings.position_cursor_entry_times
        )
        self.auto_trade_reentry_cooldown_until: Dict[Tuple[str, str], float] = {}
        self.last_engine_entry_signal_by_key: Dict[Tuple[str, str, str], Tuple[str, int]] = {}
        self.last_engine_actionable_signal_by_key: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
        self.last_local_actionable_signal_by_key: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
        self.order_worker_symbol: Optional[str] = None
        self.order_worker_is_auto_close = False
        self.order_worker_is_auto_trade = False
        self.engine_order_pending = False
        self.engine_failed = False
        self.trade_engine_recovery_scheduled = False
        self.mobile_web_server = None
        self.pending_open_order_interval: Optional[str] = None
        self.auto_refresh_minutes = int(self.settings.auto_refresh_minutes)
        self.auto_refresh_timer = QTimer(self)
        self.live_update_timer = QTimer(self)
        self.optimized_table_timer = QTimer(self)
        self.optimized_table_highlight_timer = QTimer(self)
        self.favorable_backtest_poll_timer = QTimer(self)
        self.trade_engine_poll_timer = QTimer(self)
        self.favorable_refresh_pending: Dict[Tuple[str, str], Tuple[Tuple[object, ...], str]] = {}
        self.favorable_zone_cache: Dict[Tuple[str, str], Optional[int]] = {}
        self.optimized_actionable_signal_cache: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
        self.telegram_notifier = TelegramNotifier(log=self._log_telegram_failure)
        self.telegram_favorable_symbols: set[str] = set()
        self.backtest_summary_text = ""
        self.backtest_summary_window: Optional[QWidget] = None
        self.backtest_summary_box: Optional[QPlainTextEdit] = None
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_total_candidates = 0
        self.backtest_progress_prepared_candidates = 0
        self.backtest_progress_status_text = ""
        self.backtest_progress_phase = "idle"
        self.parameter_editors: Dict[str, object] = {}
        self.parameter_opt_boxes: Dict[str, QCheckBox] = {}
        self.parameter_opt_hint_labels: Dict[str, QLabel] = {}
        self.auto_trade_focus_settings_window: Optional[QWidget] = None
        self.auto_trade_focus_enable_check: Optional[QCheckBox] = None
        self.auto_trade_focus_mode_combo: Optional[QComboBox] = None
        self.chart_display_days_popup_spin: Optional[QSpinBox] = None
        self.chart_transition_overlay: Optional[QWidget] = None
        self.chart_transition_effect: Optional[QGraphicsOpacityEffect] = None
        self.chart_transition_animation: Optional[QPropertyAnimation] = None
        self.chart = None
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
        self.current_lightweight_fast_entry_markers = []
        self.current_lightweight_fast_exit_markers = []
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
        self.optimized_table_highlight_timer.setInterval(OPTIMIZED_TABLE_HIGHLIGHT_REFRESH_MS)
        self.optimized_table_highlight_timer.timeout.connect(self._refresh_optimized_table_highlights)
        self.optimized_table_highlight_timer.start()
        self.favorable_backtest_poll_timer.setInterval(150)
        self.favorable_backtest_poll_timer.timeout.connect(self._poll_favorable_backtest_results)
        self.favorable_backtest_poll_timer.start()
        self.trade_engine_poll_timer.setInterval(TRADE_ENGINE_POLL_INTERVAL_MS)
        self.trade_engine_poll_timer.timeout.connect(self._poll_trade_engine_events)
        self.price_label_timer.setInterval(250)
        self.price_label_timer.timeout.connect(self._refresh_live_labels)
        self.price_label_timer.start()
        self.auto_close_retry_timer.setInterval(AUTO_CLOSE_RETRY_INTERVAL_MS)
        self.auto_close_retry_timer.timeout.connect(self._run_auto_close_retry_cycle)
        self.auto_trade_timer.setInterval(AUTO_TRADE_INTERVAL_MS)
        self.auto_trade_timer.timeout.connect(self._run_auto_trade_cycle)
        self._init_auto_refresh()
        self._start_trade_engine()
        self._start_mobile_web_server()
        self.statusBar().showMessage("준비됨")
        self._refresh_auto_trade_button_state()
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
        self.backtest_summary_button = QPushButton("백테스트 요약")
        self.scan_button = QPushButton("후보 스캔 + 최적화")
        self.auto_trade_button = QPushButton()
        self.auto_trade_button.toggled.connect(self._toggle_auto_trade_mode)
        self.save_settings_button.clicked.connect(self.save_settings_with_feedback)
        self.backtest_summary_button.clicked.connect(self.show_backtest_summary)
        self.scan_button.clicked.connect(self.run_scan_and_optimize)
        actions_row.addWidget(self.save_settings_button)
        actions_row.addWidget(self.backtest_summary_button)
        actions_row.addWidget(self.scan_button)
        actions_row.addWidget(self.auto_trade_button)
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

        balance_panel = QWidget()
        balance_layout = QHBoxLayout(balance_panel)
        balance_layout.setContentsMargins(8, 2, 8, 2)
        balance_layout.setSpacing(12)
        status_strip_font_style = "font-weight: 700; font-size: 12px;"
        self.symbol_label = QLabel("종목: -")
        self.signal_label = QLabel("신호: -")
        self.current_price_label = QLabel("현재가: -")
        self.position_label = QLabel("포지션: -")
        self.balance_label = QLabel("잔고:")
        self.balance_label.setStyleSheet(f"color: #111827; {status_strip_font_style}")
        self.balance_label.setFixedHeight(18)
        balance_layout.addWidget(self.balance_label)
        self.balance_status_label = QLabel("API 미입력")
        self.balance_status_label.setStyleSheet(f"color: #111827; {status_strip_font_style}")
        self.balance_status_label.setFixedHeight(18)
        balance_layout.addWidget(self.balance_status_label)
        self.balance_equity_value_label = QLabel("")
        self.balance_equity_value_label.setStyleSheet(f"color: #1546b0; {status_strip_font_style}")
        self.balance_equity_value_label.setFixedHeight(18)
        self.balance_equity_value_label.hide()
        balance_layout.addWidget(self.balance_equity_value_label)
        self.balance_equity_unit_label = QLabel("USDT | 가용: ")
        self.balance_equity_unit_label.setStyleSheet(f"color: #111827; {status_strip_font_style}")
        self.balance_equity_unit_label.setFixedHeight(18)
        self.balance_equity_unit_label.hide()
        balance_layout.addWidget(self.balance_equity_unit_label)
        self.balance_available_value_label = QLabel("")
        self.balance_available_value_label.setStyleSheet(f"color: #1546b0; {status_strip_font_style}")
        self.balance_available_value_label.setFixedHeight(18)
        self.balance_available_value_label.hide()
        balance_layout.addWidget(self.balance_available_value_label)
        self.balance_available_unit_label = QLabel("USDT")
        self.balance_available_unit_label.setStyleSheet(f"color: #111827; {status_strip_font_style}")
        self.balance_available_unit_label.setFixedHeight(18)
        self.balance_available_unit_label.hide()
        balance_layout.addWidget(self.balance_available_unit_label)
        self.chart_interval_label = QLabel("차트TF: -")
        self.chart_interval_label.setStyleSheet(f"color: #111827; {status_strip_font_style}")
        self.chart_interval_label.setFixedHeight(18)
        balance_layout.addWidget(self.chart_interval_label)
        self.bar_close_countdown_label = QLabel("봉마감: -")
        self.bar_close_countdown_label.setStyleSheet(f"color: #d63b53; {status_strip_font_style}")
        self.bar_close_countdown_label.setFixedHeight(18)
        balance_layout.addWidget(self.bar_close_countdown_label)
        self.optimization_chart_notice_label = QLabel("")
        self.optimization_chart_notice_label.setStyleSheet(f"color: #f59e0b; {status_strip_font_style}")
        self.optimization_chart_notice_label.setFixedHeight(18)
        self.optimization_chart_notice_label.hide()
        balance_layout.addWidget(self.optimization_chart_notice_label)
        balance_layout.addStretch(1)
        self._set_balance_label_status("API 미입력")
        chart_header_row = QHBoxLayout()
        chart_header_row.setContentsMargins(0, 0, 0, 0)
        chart_header_row.setSpacing(10)
        self.chart_header_symbol_label = QLabel("-")
        self.chart_header_symbol_label.setStyleSheet(
            f"color: #ffffff; background: #1546b0; border-radius: 4px; padding: 2px 10px; {status_strip_font_style}"
        )
        chart_header_row.addWidget(self.chart_header_symbol_label)
        self.chart_header_tf_label = QLabel("TF -")
        self.chart_header_tf_label.setStyleSheet(
            f"color: #ffffff; background: #6b7280; border-radius: 4px; padding: 2px 10px; {status_strip_font_style}"
        )
        chart_header_row.addWidget(self.chart_header_tf_label)
        chart_header_row.addStretch(1)
        chart_header_row.addWidget(QLabel("차트 전환"))
        self.auto_trade_focus_enable_check = QCheckBox("사용")
        chart_header_row.addWidget(self.auto_trade_focus_enable_check)
        chart_header_row.addWidget(QLabel("기준"))
        self.auto_trade_focus_mode_combo = QComboBox()
        self.auto_trade_focus_mode_combo.addItem("예상진입신호", "preview")
        self.auto_trade_focus_mode_combo.addItem("진입신호 확정", "confirmed")
        chart_header_row.addWidget(self.auto_trade_focus_mode_combo)
        chart_header_row.addWidget(QLabel("차트 표시 시간 범위"))
        self.chart_display_days_popup_spin = QSpinBox()
        self.chart_display_days_popup_spin.setRange(1, 720)
        self.chart_display_days_popup_spin.setSuffix(" 시간")
        chart_header_row.addWidget(self.chart_display_days_popup_spin)
        self.chart_host = QWidget()
        self.chart_host_layout = QVBoxLayout(self.chart_host)
        self.chart_host_layout.setContentsMargins(0, 0, 0, 0)
        self._init_chart_transition_overlay()
        right_layout.addLayout(chart_header_row)
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
        self.close_position_button = QPushButton("전체청산")
        self.close_position_button.clicked.connect(self.close_selected_position)
        self.close_position_button.setText("전체청산")
        self.close_position_button.setFixedWidth(156)
        self.close_position_button.setToolTip("현재 보유 중인 모든 포지션 전체 청산")
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

    def _refresh_surge_info_label(self) -> None:
        if not hasattr(self, "surge_info_label"):
            return

        def _format_threshold(value: float) -> str:
            text = f"{float(value):,.1f}"
            if text.endswith(".0"):
                text = text[:-2]
            return text

        price_change_text = _format_threshold(float(self.surge_price_change_spin.value()))
        if float(self.surge_price_change_spin.value()) >= 0.0:
            price_change_text = f"+{price_change_text}"
        self.surge_info_label.setText(
            "30m RSI≥"
            + _format_threshold(float(self.surge_rsi_min_spin.value()))
            + " / 거래량≥"
            + _format_threshold(float(self.surge_quote_volume_spin.value()))
            + " / 24h≥"
            + price_change_text
            + "%"
        )

    def _refresh_filter_controls(self) -> None:
        is_surge = (
            hasattr(self, "filter_preset_combo")
            and self.filter_preset_combo.currentData() == "급등종목"
        )
        volatility_controls = [
            self.daily_vol_spin,
            self.quote_volume_spin,
            self.rsi_filter_check,
            self.rsi_length_spin,
            self.rsi_lower_spin,
            self.rsi_upper_spin,
            self.atr_4h_filter_check,
            self.atr_4h_spin,
        ]
        surge_controls = [
            getattr(self, "surge_rsi_min_spin", None),
            getattr(self, "surge_quote_volume_spin", None),
            getattr(self, "surge_price_change_spin", None),
        ]
        for ctrl in volatility_controls:
            ctrl.setEnabled(not is_surge)
        for ctrl in surge_controls:
            if ctrl is not None:
                ctrl.setEnabled(is_surge)
        if hasattr(self, "surge_info_label"):
            self._refresh_surge_info_label()
            self.surge_info_label.setVisible(is_surge)
        if not is_surge:
            rsi_enabled = bool(self.rsi_filter_check.isChecked())
            self.rsi_length_spin.setEnabled(rsi_enabled)
            self.rsi_lower_spin.setEnabled(rsi_enabled)
            self.rsi_upper_spin.setEnabled(rsi_enabled)
            atr_enabled = bool(self.atr_4h_filter_check.isChecked())
            self.atr_4h_spin.setEnabled(atr_enabled)

    def _selected_strategy_type(self) -> str:
        if hasattr(self, "strategy_type_combo"):
            return str(self.strategy_type_combo.currentData() or self.settings.strategy.strategy_type)
        return str(self.settings.strategy.strategy_type)

    def _refresh_strategy_controls(self) -> None:
        strategy_type = self._selected_strategy_type()
        for spec in PARAMETER_SPECS:
            editor = self.parameter_editors.get(spec.key)
            if editor is not None:
                editor.setEnabled(parameter_spec_applies(spec, strategy_type))
        self._refresh_candidate_optimization_controls()
        self._refresh_parameter_optimization_hints()

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

    def _parameter_editor_current_value(self, spec) -> object:
        editor = self.parameter_editors[spec.key]
        if spec.kind == "bool":
            return bool(editor.isChecked())
        if spec.kind == "choice":
            return editor.currentText()
        if spec.kind == "int":
            return int(editor.value())
        return float(editor.value())

    def _format_parameter_hint_value(self, spec, value: object) -> str:
        if spec.kind == "bool":
            return "ON" if bool(value) else "OFF"
        if spec.kind == "int":
            return str(int(value))
        if spec.kind == "float":
            step_value = spec.optimize_step if spec.optimize_step is not None else spec.step
            if step_value is None:
                return f"{float(value):.2f}"
            step_text = str(step_value)
            decimals = len(step_text.rstrip("0").split(".")[-1]) if "." in step_text else 0
            return f"{float(value):.{decimals}f}"
        return str(value)

    def _parameter_opt_hint_text(self, spec) -> str:
        if not parameter_spec_applies(spec, self._selected_strategy_type()):
            return "현재 전략 미사용"
        if hasattr(self, "parameter_optimization_check") and not self.parameter_optimization_check.isChecked():
            return "후보종목 기본값 사용"
        span_pct = float(self.opt_span_spin.value()) if hasattr(self, "opt_span_spin") else self.settings.optimization_span_pct
        steps = int(self.opt_steps_spin.value()) if hasattr(self, "opt_steps_spin") else self.settings.optimization_steps
        values = parameter_value_range(spec, self._parameter_editor_current_value(spec), span_pct, steps, True)
        if not values:
            return "-"
        if spec.kind == "bool":
            return "OFF / ON"
        if spec.kind == "choice":
            if len(values) == 1:
                return self._format_parameter_hint_value(spec, values[0])
            return f"{self._format_parameter_hint_value(spec, values[0])} ~ {self._format_parameter_hint_value(spec, values[-1])}"
        if len(values) == 1:
            return self._format_parameter_hint_value(spec, values[0])
        step_value = spec.optimize_step if spec.optimize_step is not None else spec.step
        if step_value is None:
            return f"{self._format_parameter_hint_value(spec, values[0])} ~ {self._format_parameter_hint_value(spec, values[-1])}"
        step_text = self._format_parameter_hint_value(spec, step_value)
        return (
            f"{self._format_parameter_hint_value(spec, values[0])} ~ "
            f"{self._format_parameter_hint_value(spec, values[-1])} / {step_text}"
        )

    def _refresh_parameter_opt_hint(self, spec) -> None:
        label = self.parameter_opt_hint_labels.get(spec.key)
        if label is None:
            return
        strategy_applies = parameter_spec_applies(spec, self._selected_strategy_type())
        parameter_optimization_enabled = (
            bool(self.parameter_optimization_check.isChecked())
            if hasattr(self, "parameter_optimization_check")
            else self.settings.enable_parameter_optimization
        )
        enabled = (
            strategy_applies
            and parameter_optimization_enabled
            and bool(self.parameter_opt_boxes.get(spec.key).isChecked())
            if spec.key in self.parameter_opt_boxes
            else False
        )
        color = "#6b7280" if enabled else "#8a8f98"
        label.setStyleSheet(f"color: {color}; font-size: 10px;")
        label.setText(self._parameter_opt_hint_text(spec))

    def _refresh_parameter_optimization_hints(self) -> None:
        for spec in PARAMETER_SPECS:
            self._refresh_parameter_opt_hint(spec)

    def _refresh_candidate_optimization_controls(self) -> None:
        enabled = (
            bool(self.parameter_optimization_check.isChecked())
            if hasattr(self, "parameter_optimization_check")
            else self.settings.enable_parameter_optimization
        )
        for widget in (
            getattr(self, "opt_span_spin", None),
            getattr(self, "opt_steps_spin", None),
            getattr(self, "max_combo_spin", None),
        ):
            if widget is not None:
                widget.setEnabled(enabled)
        strategy_type = self._selected_strategy_type()
        for spec in PARAMETER_SPECS:
            optimize_box = self.parameter_opt_boxes.get(spec.key)
            if optimize_box is not None:
                optimize_box.setEnabled(enabled and parameter_spec_applies(spec, strategy_type))

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
        self.strategy_type_combo = QComboBox()
        for strategy_type in STRATEGY_TYPE_OPTIONS:
            self.strategy_type_combo.addItem(STRATEGY_TYPE_LABELS.get(strategy_type, strategy_type), strategy_type)
        self.filter_preset_combo = QComboBox()
        self.filter_preset_combo.addItem("변동성", "변동성")
        self.filter_preset_combo.addItem("급등종목", "급등종목")
        self.surge_info_label = QLabel("")
        self.surge_info_label.setStyleSheet("color: gray; font-size: 10px;")
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
        self.surge_rsi_min_spin = QDoubleSpinBox()
        self.surge_rsi_min_spin.setRange(0.0, 100.0)
        self.surge_rsi_min_spin.setDecimals(1)
        self.surge_rsi_min_spin.setSingleStep(1.0)
        self.surge_quote_volume_spin = QDoubleSpinBox()
        self.surge_quote_volume_spin.setRange(0.0, 10_000_000_000.0)
        self.surge_quote_volume_spin.setDecimals(0)
        self.surge_quote_volume_spin.setSingleStep(100_000.0)
        self.surge_price_change_spin = QDoubleSpinBox()
        self.surge_price_change_spin.setRange(0.0, 500.0)
        self.surge_price_change_spin.setDecimals(1)
        self.surge_price_change_spin.setSingleStep(1.0)
        self.atr_4h_filter_check = QCheckBox("사용")
        self.atr_4h_spin = QDoubleSpinBox()
        self.atr_4h_spin.setRange(0.0, 500.0)
        self.atr_4h_spin.setDecimals(2)
        self.atr_4h_spin.setSingleStep(1.0)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(APP_INTERVAL_OPTIONS)
        self.history_days_spin = QSpinBox()
        self.history_days_spin.setRange(1, 30)
        self.history_days_spin.setSuffix(" 일")
        self.history_days_spin.setToolTip("백테스트와 최적화에 사용할 히스토리 일수")
        self.scan_workers_spin = QSpinBox()
        self.scan_workers_spin.setRange(1, 24)
        self.auto_refresh_minutes_spin = QSpinBox()
        self.auto_refresh_minutes_spin.setRange(1, 1_440)
        self.auto_refresh_minutes_spin.setSuffix(" 분")
        self.strategy_type_combo.currentIndexChanged.connect(lambda _: self._refresh_strategy_controls())
        self.filter_preset_combo.currentIndexChanged.connect(lambda _: self._refresh_filter_controls())
        self.rsi_filter_check.toggled.connect(self._refresh_filter_controls)
        self.atr_4h_filter_check.toggled.connect(self._refresh_filter_controls)
        self.surge_rsi_min_spin.valueChanged.connect(lambda *_: self._refresh_surge_info_label())
        self.surge_quote_volume_spin.valueChanged.connect(lambda *_: self._refresh_surge_info_label())
        self.surge_price_change_spin.valueChanged.connect(lambda *_: self._refresh_surge_info_label())
        layout.addRow("전략", self.strategy_type_combo)
        layout.addRow("필터 프리셋", self.filter_preset_combo)
        layout.addRow("", self.surge_info_label)
        layout.addRow("1일 변동성 % >=", self.daily_vol_spin)
        layout.addRow("24h 거래량 >=", self.quote_volume_spin)
        layout.addRow("1m RSI 필터", self.rsi_filter_check)
        layout.addRow("1m RSI Length", self.rsi_length_spin)
        layout.addRow("1m RSI Lower <=", self.rsi_lower_spin)
        layout.addRow("1m RSI Upper >=", self.rsi_upper_spin)
        layout.addRow("급등 30m RSI >=", self.surge_rsi_min_spin)
        layout.addRow("급등 24h 거래량 >=", self.surge_quote_volume_spin)
        layout.addRow("급등 24h 등락율 % >=", self.surge_price_change_spin)
        layout.addRow("4h ATR% 필터", self.atr_4h_filter_check)
        layout.addRow("4h ATR% >=", self.atr_4h_spin)
        layout.addRow("백테스트 봉", self.interval_combo)
        layout.addRow("백테스트 일수", self.history_days_spin)
        layout.addRow("스캔 워커", self.scan_workers_spin)
        layout.addRow("자동 갱신", self.auto_refresh_minutes_spin)
        return group

    def _build_optimization_group(self) -> QGroupBox:
        group = QGroupBox("Optimization")
        layout = QFormLayout(group)
        self.parameter_optimization_check = QCheckBox("후보종목 파라미터 최적화")
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
        self.auto_trade_favorable_check = QCheckBox("유리한 가격 진입 허용")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 5.0)
        self.fee_spin.setDecimals(4)
        self.fee_spin.setSingleStep(0.01)
        self.parameter_optimization_check.toggled.connect(lambda _checked: self._refresh_candidate_optimization_controls())
        self.parameter_optimization_check.toggled.connect(lambda _checked: self._refresh_parameter_optimization_hints())
        self.opt_rank_mode_combo.currentIndexChanged.connect(lambda _index: self._refresh_optimization_rank_controls())
        self.opt_min_score_spin.valueChanged.connect(lambda _value: self.update_optimized_table())
        self.opt_min_return_spin.valueChanged.connect(lambda _value: self.update_optimized_table())
        self.opt_span_spin.valueChanged.connect(lambda _value: self._refresh_parameter_optimization_hints())
        self.opt_steps_spin.valueChanged.connect(lambda _value: self._refresh_parameter_optimization_hints())
        def _set_row_label(editor, label: str) -> None:
            field_label = layout.labelForField(editor)
            if field_label is not None:
                field_label.setText(label)
        self.parameter_optimization_check.setChecked(bool(self.settings.enable_parameter_optimization))
        self.optimize_timeframe_check.setText("1m / 2m 최적화")
        self.auto_trade_favorable_check.setChecked(bool(self.settings.auto_trade_use_favorable_price))
        layout.addRow("파라미터", self.parameter_optimization_check)
        layout.addRow("범위 ±%", self.opt_span_spin)
        layout.addRow("격자 단계수", self.opt_steps_spin)
        layout.addRow("정렬 기준", self.opt_rank_mode_combo)
        layout.addRow("최소 총점", self.opt_min_score_spin)
        layout.addRow("최소 수익률%", self.opt_min_return_spin)
        layout.addRow("최대 조합수", self.max_combo_spin)
        layout.addRow("최적화 프로세스", self.opt_process_spin)
        layout.addRow("타임프레임", self.optimize_timeframe_check)
        layout.addRow("자동매매 유리한 가격", self.auto_trade_favorable_check)
        layout.addRow("수수료 %", self.fee_spin)
        _set_row_label(self.opt_span_spin, "범위 스케일 (20=기본)")
        _set_row_label(self.opt_steps_spin, "항목별 샘플 상한")
        _set_row_label(self.opt_min_score_spin, "최소 총점 (점수제)")
        _set_row_label(self.opt_min_return_spin, "최소 수익률% (수익률제)")
        _set_row_label(self.max_combo_spin, "최대 조합수")
        _set_row_label(self.opt_process_spin, "최적화 프로세스")
        _set_row_label(self.optimize_timeframe_check, "타임프레임")
        _set_row_label(self.auto_trade_favorable_check, "자동매매 유리한 가격")
        _set_row_label(self.fee_spin, "수수료 %")
        self._refresh_optimization_rank_controls()
        self._refresh_candidate_optimization_controls()
        return group

    def _build_parameter_tabs(self) -> QGroupBox:
        group = QGroupBox("Strategy Parameters")
        outer_layout = QVBoxLayout(group)
        help_label = QLabel(
            "선택한 전략에서 사용되는 항목만 반영됩니다. Opt를 체크한 항목만 최적화하고, "
            "범위 스케일은 옵션별 기본 프로필 폭을 전체적으로 조절합니다."
        )
        help_label.setWordWrap(True)
        outer_layout.addWidget(help_label)
        tabs = QTabWidget()
        outer_layout.addWidget(tabs)

        forms: Dict[str, QFormLayout] = {}
        for key, title in (
            ("core", "Core"),
            ("qip", "QIP"),
            ("qtp", "QTP"),
            ("keltner", "Keltner"),
            ("switches", "Switches"),
        ):
            page = QWidget()
            form = QFormLayout(page)
            forms[key] = form
            tabs.addTab(page, title)

        for spec in PARAMETER_SPECS:
            editor = self._make_parameter_editor(spec)
            optimize_box = QCheckBox("Opt")
            optimize_box.setChecked(bool(self.settings.optimize_flags.get(spec.key, False)))
            hint_label = QLabel()
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet("color: #6b7280; font-size: 10px;")
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor)
            row_layout.addWidget(optimize_box)
            row_layout.addWidget(hint_label, 1)
            forms[spec.group].addRow(spec.label, row_widget)
            self.parameter_editors[spec.key] = editor
            self.parameter_opt_boxes[spec.key] = optimize_box
            self.parameter_opt_hint_labels[spec.key] = hint_label
            if spec.kind == "bool":
                editor.toggled.connect(lambda _checked, current_spec=spec: self._refresh_parameter_opt_hint(current_spec))
            elif spec.kind == "choice":
                editor.currentTextChanged.connect(lambda _text, current_spec=spec: self._refresh_parameter_opt_hint(current_spec))
            else:
                editor.valueChanged.connect(lambda _value, current_spec=spec: self._refresh_parameter_opt_hint(current_spec))
            optimize_box.toggled.connect(lambda _checked, current_spec=spec: self._refresh_parameter_opt_hint(current_spec))
            self._refresh_parameter_opt_hint(spec)
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
        self.candidate_table.installEventFilter(self)
        self.candidate_table.itemSelectionChanged.connect(self.on_candidate_selection_changed)
        self.candidate_table.cellClicked.connect(self.on_candidate_cell_clicked)
        layout.addWidget(self.candidate_table)
        return group

    def _build_optimized_group(self) -> QGroupBox:
        self.optimized_group = QGroupBox("최적화 종목")
        group = self.optimized_group
        # Overlay label for favorable count — positioned in the title bar area
        self.optimized_favorable_label = QLabel("", group)
        self.optimized_favorable_label.setStyleSheet(
            "color: #ffffff; font-weight: 700; background: #1f9d55; border-radius: 8px; padding: 1px 8px;"
        )
        self.optimized_favorable_label.hide()
        self.optimized_entry_label = QLabel("", group)
        self.optimized_entry_label.setStyleSheet(
            "color: #ffffff; font-weight: 700; background: #d64550; border-radius: 8px; padding: 1px 8px;"
        )
        self.optimized_entry_label.hide()
        group.installEventFilter(self)
        layout = QVBoxLayout(group)
        self.optimized_table = QTableWidget(0, 9)
        self.optimized_table.setHorizontalHeaderLabels(["Symbol", "TF", "Score", "Return%", "MDD%", "Trades", "Win%", "PF", "Grid"])
        self.optimized_table.setSelectionBehavior(SELECT_ROWS)
        self.optimized_table.setSelectionMode(SINGLE_SELECTION)
        self.optimized_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.optimized_table.horizontalHeader().setStretchLastSection(True)
        self.optimized_table.installEventFilter(self)
        self.optimized_table.itemSelectionChanged.connect(self.on_optimized_selection_changed)
        self.optimized_table.cellClicked.connect(self.on_optimized_cell_clicked)
        layout.addWidget(self.optimized_table)
        return group

    def _reposition_favorable_label(self) -> None:
        group = getattr(self, "optimized_group", None)
        favorable_label = getattr(self, "optimized_favorable_label", None)
        entry_label = getattr(self, "optimized_entry_label", None)
        if group is None or favorable_label is None:
            return
        fm = QFontMetrics(group.font())
        title_width = fm.horizontalAdvance("최적화 종목")
        x = 9 + title_width + 8
        spacing = 6
        for label in (favorable_label, entry_label):
            if label is None or not label.isVisible():
                continue
            label.adjustSize()
            y = max(1, (group.fontMetrics().height() - label.height()) // 2)
            label.move(x, y)
            x += label.width() + spacing

    def _build_positions_group(self) -> QGroupBox:
        group = QGroupBox("Open Positions")
        layout = QVBoxLayout(group)
        self.positions_table = QTableWidget(0, 8)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Side", "Leverage", "Amount USDT", "Entry", "UPnL", "수익률", "Action"])
        self.positions_table.setSelectionBehavior(SELECT_ROWS)
        self.positions_table.setSelectionMode(SINGLE_SELECTION)
        self.positions_table.setEditTriggers(NO_EDIT_TRIGGERS)
        positions_header = self.positions_table.horizontalHeader()
        positions_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        positions_header.setStretchLastSection(True)
        self.positions_table.installEventFilter(self)
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
        self._rebuild_chart_view(force=True)

    def _init_chart_transition_overlay(self) -> None:
        if not hasattr(self, "chart_host"):
            return
        overlay = QWidget(self.chart_host)
        overlay.hide()
        overlay.setStyleSheet("background-color: rgba(8, 11, 16, 185); border-radius: 6px;")
        effect = QGraphicsOpacityEffect(overlay)
        effect.setOpacity(0.0)
        overlay.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(120)
        animation.finished.connect(self._on_chart_transition_animation_finished)
        self.chart_transition_overlay = overlay
        self.chart_transition_effect = effect
        self.chart_transition_animation = animation
        self._sync_chart_transition_overlay()

    def _sync_chart_transition_overlay(self) -> None:
        overlay = self.chart_transition_overlay
        if overlay is None or not hasattr(self, "chart_host"):
            return
        overlay.setGeometry(self.chart_host.rect())
        overlay.raise_()

    def _on_chart_transition_animation_finished(self) -> None:
        overlay = self.chart_transition_overlay
        effect = self.chart_transition_effect
        if overlay is None or effect is None:
            return
        if effect.opacity() <= 0.01:
            overlay.hide()

    def _show_chart_transition_overlay(self) -> None:
        overlay = self.chart_transition_overlay
        effect = self.chart_transition_effect
        animation = self.chart_transition_animation
        if overlay is None or effect is None or animation is None:
            return
        self._sync_chart_transition_overlay()
        animation.stop()
        overlay.show()
        overlay.raise_()
        animation.setDuration(120)
        animation.setStartValue(float(effect.opacity()))
        animation.setEndValue(1.0)
        animation.start()

    def _hide_chart_transition_overlay(self) -> None:
        overlay = self.chart_transition_overlay
        effect = self.chart_transition_effect
        animation = self.chart_transition_animation
        if overlay is None or effect is None or animation is None or not overlay.isVisible():
            return
        animation.stop()
        animation.setDuration(90)
        animation.setStartValue(float(effect.opacity()))
        animation.setEndValue(0.0)
        animation.start()

    def _schedule_chart_transition_reveal(self, symbol: str, delay_ms: int = 180) -> None:
        QTimer.singleShot(delay_ms, lambda s=symbol: self._hide_chart_transition_overlay_for_symbol(s))

    def _hide_chart_transition_overlay_for_symbol(self, symbol: str) -> None:
        if symbol != self.current_symbol:
            return
        self._hide_chart_transition_overlay()

    def _clear_chart_host(self) -> None:
        while self.chart_host_layout.count():
            item = self.chart_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.chart = None
        self.equity_subchart = None
        self.equity_line = None
        self.entry_price_line = None
        self.supertrend_line = None
        self.zone2_line = None
        self.zone3_line = None
        self.ema_fast_line = None
        self.ema_slow_line = None

    def _rebuild_chart_view(self, force: bool = False) -> None:
        if not force and self.chart is not None:
            self.update_positions_table()
            return
        self._clear_chart_host()
        self._init_lightweight_chart()
        self._sync_chart_transition_overlay()
        if self.current_symbol and self.current_backtest:
            self.render_chart(self.current_symbol, self.current_backtest)

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
        self.chart.time_scale(
            time_visible=True,
            seconds_visible=False,
            min_bar_spacing=0.01,
            right_offset=DEFAULT_CHART_RIGHT_PAD_BARS,
        )
        self.equity_subchart = self.chart.create_subchart(
            position="bottom",
            width=1.0,
            height=0.28,
            sync=True,
            scale_candles_only=True,
        )
        self.equity_subchart.layout(background_color="#121922", text_color="#eceff4", font_size=11, font_family="Consolas")
        self.equity_subchart.legend(True)
        self.equity_subchart.time_scale(
            time_visible=True,
            seconds_visible=False,
            min_bar_spacing=0.01,
            right_offset=DEFAULT_CHART_RIGHT_PAD_BARS,
        )
        self.equity_line = self.equity_subchart.create_line(
            "Equity",
            color="rgba(108, 245, 160, 0.9)",
            width=2,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.supertrend_line = self.chart.create_line(
            "Supertrend",
            color="#00D8FF",
            width=2,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.zone2_line = self.chart.create_line(
            "Zone 2",
            color="rgba(255, 186, 73, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.zone3_line = self.chart.create_line(
            "Zone 3",
            color="rgba(255, 123, 123, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.ema_fast_line = self.chart.create_line(
            "EMA Fast",
            color="rgba(153, 229, 255, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.ema_slow_line = self.chart.create_line(
            "EMA Slow",
            color="rgba(255, 243, 176, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
            crosshair_marker=False,
        )
        self.chart.events.range_change += self._on_lightweight_range_change
        self._init_lightweight_bar_close_overlay()
        self._init_lightweight_optimization_overlay()

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
        if self.chart is None:
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

    def _init_lightweight_optimization_overlay(self) -> None:
        if self.chart is None:
            return
        self.chart.run_script(
            f"""
            (() => {{
                const handler = {self.chart.id};
                if (!handler || handler.optimizationBadge) {{
                    return;
                }}
                const label = document.createElement("div");
                Object.assign(label.style, {{
                    position: "absolute",
                    left: "12px",
                    top: "10px",
                    display: "none",
                    color: "#facc15",
                    fontFamily: "Consolas, monospace",
                    fontSize: "12px",
                    fontWeight: "700",
                    letterSpacing: "0.03em",
                    textShadow: "0 1px 6px rgba(0, 0, 0, 0.65)",
                    zIndex: "4200",
                    pointerEvents: "none",
                    whiteSpace: "nowrap",
                }});
                handler.div.appendChild(label);
                handler.optimizationBadge = label;
                handler.setOptimizationBadge = (text) => {{
                    if (!handler.optimizationBadge) {{
                        return;
                    }}
                    const nextText = String(text || "").trim();
                    handler.optimizationBadge.textContent = nextText;
                    handler.optimizationBadge.style.display = nextText ? "block" : "none";
                }};
                handler.hideOptimizationBadge = () => {{
                    if (!handler.optimizationBadge) {{
                        return;
                    }}
                    handler.optimizationBadge.textContent = "";
                    handler.optimizationBadge.style.display = "none";
                }};
            }})();
            """
        )

    def _showing_optimized_chart(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> bool:
        target_symbol = str(symbol or self.current_symbol or "").strip()
        target_interval = interval if interval in APP_INTERVAL_OPTIONS else self.current_interval
        if not target_symbol or target_interval not in APP_INTERVAL_OPTIONS:
            return False
        if target_symbol != self.current_symbol or target_interval != self.current_interval:
            return False
        if self.current_backtest is None:
            return False
        optimization = self._optimization_result(target_symbol, target_interval)
        if optimization is None:
            return False
        optimized_interval = _optimization_result_interval(optimization)
        if optimized_interval != target_interval:
            return False
        return self.current_backtest.settings == optimization.best_backtest.settings

    def _set_optimization_chart_notice_text(self, visible: bool) -> None:
        if not hasattr(self, "optimization_chart_notice_label"):
            return
        if visible:
            self.optimization_chart_notice_label.setText("최적화 차트 표시중")
            self.optimization_chart_notice_label.show()
            return
        self.optimization_chart_notice_label.setText("")
        self.optimization_chart_notice_label.hide()

    def _set_lightweight_optimization_overlay(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
        showing_optimized_chart = self._showing_optimized_chart(symbol, interval)
        self._set_optimization_chart_notice_text(showing_optimized_chart)
        if self.chart is None:
            return
        if showing_optimized_chart:
            self.chart.run_script(
                f"""
                if ({self.chart.id}.setOptimizationBadge) {{
                    {self.chart.id}.setOptimizationBadge({json.dumps("최적화 차트 표시중")});
                }}
                """
            )
            return
        self.chart.run_script(
            f"""
            if ({self.chart.id}.hideOptimizationBadge) {{
                {self.chart.id}.hideOptimizationBadge();
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
                f"status: {status}",
                f"exit_code: {exit_code}",
            ]
        )
        path = log_runtime_event("WebEngine Render Crash", body, open_notepad=True)
        self.log(f"{engine_label} 렌더 프로세스 종료. 로그: {path}")

    def _apply_loaded_settings(self) -> None:
        settings = self.settings
        self.api_key_edit.setText(settings.api_key)
        self.api_secret_edit.setText(settings.api_secret)
        self.leverage_spin.setValue(settings.leverage)
        strategy_index = self.strategy_type_combo.findData(settings.strategy.strategy_type)
        if strategy_index >= 0:
            self.strategy_type_combo.setCurrentIndex(strategy_index)
        preset_index = self.filter_preset_combo.findData(settings.filter_preset)
        if preset_index >= 0:
            self.filter_preset_combo.setCurrentIndex(preset_index)
        self.daily_vol_spin.setValue(settings.daily_volatility_min)
        self.quote_volume_spin.setValue(settings.quote_volume_min)
        self.surge_quote_volume_spin.setValue(settings.surge_quote_volume_min)
        self.surge_price_change_spin.setValue(settings.surge_price_change_min_pct)
        self.surge_rsi_min_spin.setValue(settings.surge_rsi_30m_min)
        self.rsi_filter_check.setChecked(settings.use_rsi_filter)
        self.rsi_length_spin.setValue(settings.rsi_length)
        self.rsi_lower_spin.setValue(settings.rsi_lower)
        self.rsi_upper_spin.setValue(settings.rsi_upper)
        self.atr_4h_filter_check.setChecked(settings.use_atr_4h_filter)
        self.atr_4h_spin.setValue(settings.atr_4h_min_pct)
        self.interval_combo.setCurrentText(settings.kline_interval)
        self.history_days_spin.setValue(settings.history_days)
        self.scan_workers_spin.setValue(settings.scan_workers)
        self.auto_refresh_minutes_spin.setValue(settings.auto_refresh_minutes)
        self.opt_span_spin.setValue(settings.optimization_span_pct)
        self.opt_steps_spin.setValue(settings.optimization_steps)
        rank_mode_index = self.opt_rank_mode_combo.findData(settings.optimization_rank_mode)
        if rank_mode_index >= 0:
            self.opt_rank_mode_combo.setCurrentIndex(rank_mode_index)
        self.opt_min_score_spin.setValue(settings.optimization_min_score)
        self.opt_min_return_spin.setValue(settings.optimization_min_return_pct)
        self.max_combo_spin.setValue(settings.max_grid_combinations)
        self.parameter_optimization_check.setChecked(bool(settings.enable_parameter_optimization))
        self.opt_process_spin.setValue(settings.optimize_processes)
        self.optimize_timeframe_check.setChecked(settings.optimize_timeframe)
        self.auto_trade_favorable_check.setChecked(bool(settings.auto_trade_use_favorable_price))
        if self.auto_trade_focus_enable_check is not None:
            self.auto_trade_focus_enable_check.setChecked(bool(settings.auto_trade_focus_on_signal))
        if self.auto_trade_focus_mode_combo is not None:
            mode_index = self.auto_trade_focus_mode_combo.findData(settings.auto_trade_focus_signal_mode)
            if mode_index >= 0:
                self.auto_trade_focus_mode_combo.setCurrentIndex(mode_index)
        if self.chart_display_days_popup_spin is not None:
            self.chart_display_days_popup_spin.setValue(int(settings.chart_display_hours))
        self.fee_spin.setValue(settings.fee_rate * 100.0)
        self.simple_order_amount_spin.setValue(settings.simple_order_amount)
        if settings.order_mode == "simple":
            self.simple_order_radio.setChecked(True)
        else:
            self.compound_order_radio.setChecked(True)
        self._refresh_order_mode_ui()
        self._refresh_filter_controls()
        self._refresh_optimization_rank_controls()
        self._refresh_strategy_controls()
        self._refresh_parameter_optimization_hints()

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
            leverage=int(self.leverage_spin.value()),
            order_mode="simple" if self.simple_order_radio.isChecked() else "compound",
            simple_order_amount=float(self.simple_order_amount_spin.value()),
            fee_rate=float(self.fee_spin.value()) / 100.0,
            history_days=int(self.history_days_spin.value()),
            chart_display_hours=int(
                self.chart_display_days_popup_spin.value()
                if self.chart_display_days_popup_spin is not None
                else self.settings.chart_display_hours
            ),
            auto_refresh_minutes=int(self.auto_refresh_minutes_spin.value()),
            auto_trade_use_favorable_price=bool(self.auto_trade_favorable_check.isChecked()),
            auto_trade_focus_on_signal=bool(
                self.auto_trade_focus_enable_check.isChecked() if self.auto_trade_focus_enable_check is not None else self.settings.auto_trade_focus_on_signal
            ),
            auto_trade_focus_signal_mode=self._auto_trade_focus_signal_mode(),
            kline_interval=self.interval_combo.currentText(),
            filter_preset=str(self.filter_preset_combo.currentData() or "변동성"),
            daily_volatility_min=float(self.daily_vol_spin.value()),
            quote_volume_min=float(self.quote_volume_spin.value()),
            surge_quote_volume_min=float(self.surge_quote_volume_spin.value()),
            surge_price_change_min_pct=float(self.surge_price_change_spin.value()),
            surge_rsi_30m_min=float(self.surge_rsi_min_spin.value()),
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
            enable_parameter_optimization=bool(self.parameter_optimization_check.isChecked()),
            scan_workers=int(self.scan_workers_spin.value()),
            optimize_processes=int(self.opt_process_spin.value()),
            optimize_timeframe=bool(self.optimize_timeframe_check.isChecked()),
            strategy=StrategySettings(
                strategy_type=self._selected_strategy_type(),
                **strategy_payload,
            ),
            optimize_flags=optimize_flags,
            position_intervals=dict(self.settings.position_intervals),
            position_strategy_settings=dict(self.settings.position_strategy_settings),
            position_filled_fractions=dict(self.settings.position_filled_fractions),
            position_cursor_entry_times=dict(self.settings.position_cursor_entry_times),
        )

    def _sync_settings(self, persist: bool = False) -> AppSettings:
        previous = self.settings
        self.settings = self.collect_settings()
        self.auto_refresh_minutes = int(self.settings.auto_refresh_minutes)
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
            self.current_chart_snapshot = None
            self.price_precision_cache.clear()
        if previous != self.settings:
            self.backtest_cache.clear()
            self.resolved_auto_trade_backtest_cache.clear()
            self.resolved_auto_trade_backtest_meta.clear()
            self.chart_indicator_cache.clear()
            self.favorable_refresh_pending.clear()
            self.favorable_zone_cache.clear()
            self.optimized_actionable_signal_cache.clear()
        if previous.chart_display_hours != self.settings.chart_display_hours and self.current_symbol and self.current_backtest is not None:
            self.render_chart(
                self.current_symbol,
                self.current_backtest,
                reset_view=True,
                chart_indicators=self.current_chart_indicators,
                reveal_overlay=False,
            )
            self.auto_close_signal_pending.clear()
            self.auto_close_order_pending.clear()
            self.auto_close_queued_orders.clear()
            self.auto_close_last_trigger_time.clear()
            self.auto_close_last_attempt_at.clear()
            self._stop_all_auto_close_monitors()
            self._refresh_auto_close_monitors()
        if previous.auto_refresh_minutes != self.settings.auto_refresh_minutes:
            self._apply_auto_refresh_interval(log_message=True)
        self._sync_trade_engine_state()
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

    def _start_mobile_web_server(self) -> None:
        if self.mobile_web_server is not None:
            return
        try:
            from .web_mobile import MobileWebServer
        except Exception as exc:
            self.log(f"모바일 웹 비활성화: {exc}")
            return
        try:
            self.mobile_web_server = MobileWebServer(self)
            self.mobile_web_server.start()
            urls = ", ".join(self.mobile_web_server.urls)
            if urls:
                self.log(f"모바일 웹 접속 주소: {urls}")
        except Exception as exc:
            self.mobile_web_server = None
            self.log(f"모바일 웹 시작 실패: {exc}")

    def _stop_mobile_web_server(self) -> None:
        if self.mobile_web_server is None:
            return
        try:
            self.mobile_web_server.stop()
        finally:
            self.mobile_web_server = None

    def _is_order_pending(self) -> bool:
        return self.engine_order_pending or (self.order_worker is not None and self.order_worker.isRunning())

    def _trade_engine_alive(self) -> bool:
        return self.trade_engine is not None and self.trade_engine.is_alive() and not self.engine_failed

    def _start_trade_engine(self) -> None:
        self._stop_trade_engine()
        self.trade_engine_recovery_scheduled = False
        controller = TradeEngineController()
        controller.start()
        self.trade_engine = controller
        self.engine_failed = False
        self.trade_engine_poll_timer.start()
        self._sync_trade_engine_state()

    def _stop_trade_engine(self) -> None:
        self.trade_engine_poll_timer.stop()
        if self.trade_engine is not None:
            self.trade_engine.stop()
        self.trade_engine = None
        self.last_engine_entry_signal_by_key.clear()
        self.last_engine_actionable_signal_by_key.clear()
        self.last_local_actionable_signal_by_key.clear()
        self.engine_failed = False
        self.trade_engine_recovery_scheduled = False

    def _schedule_trade_engine_recovery(self, delay_ms: int = 1500) -> None:
        if self.trade_engine_recovery_scheduled:
            return
        self.trade_engine_recovery_scheduled = True
        QTimer.singleShot(delay_ms, self._recover_trade_engine_if_needed)

    def _recover_trade_engine_if_needed(self) -> None:
        self.trade_engine_recovery_scheduled = False
        if self._trade_engine_alive():
            return
        if not self.settings.api_key or not self.settings.api_secret:
            self._refresh_auto_trade_button_state()
            return
        self.log("Trade engine restarting...")
        try:
            self._start_trade_engine()
        except Exception as exc:
            self.engine_failed = True
            self.trade_engine = None
            self.log(f"Trade engine restart failed: {exc}")
            self._refresh_auto_trade_button_state()
            self._schedule_trade_engine_recovery(delay_ms=3000)
            return
        if self.auto_trade_requested and self._auto_trade_ready():
            QTimer.singleShot(0, self._activate_requested_auto_trade_if_ready)
        self._refresh_auto_trade_button_state()

    def _ensure_trade_engine_available(self) -> bool:
        if self._trade_engine_alive():
            return True
        self._recover_trade_engine_if_needed()
        return self._trade_engine_alive()

    def _trade_engine_watchlist_items(self) -> Tuple[EngineWatchlistItem, ...]:
        items: List[EngineWatchlistItem] = []
        for result in self._ordered_optimized_results():
            interval = _optimization_result_interval(result)
            items.append(
                EngineWatchlistItem(
                    symbol=result.symbol,
                    interval=interval,
                    score=float(result.score),
                    return_pct=float(result.best_backtest.metrics.total_return_pct),
                    strategy_settings=result.best_backtest.settings,
                )
            )
        return tuple(items)

    def _sync_trade_engine_state(self) -> None:
        if not self._trade_engine_alive():
            return
        try:
            self.trade_engine.send(
                EngineSyncCommand(
                    api_key=self.settings.api_key,
                    api_secret=self.settings.api_secret,
                    leverage=int(self.settings.leverage),
                    fee_rate=float(self.settings.fee_rate),
                    history_days=int(self.settings.history_days),
                    default_interval=str(self.settings.kline_interval),
                    default_strategy_settings=self.settings.strategy,
                    optimization_rank_mode=self._optimization_rank_mode(),
                    auto_trade_enabled=bool(self.auto_trade_enabled),
                    auto_trade_use_favorable_price=bool(self.settings.auto_trade_use_favorable_price),
                    auto_close_enabled_symbols=tuple(sorted(self.auto_close_enabled_symbols)),
                    position_intervals=dict(self.settings.position_intervals),
                    position_strategy_settings=dict(self.settings.position_strategy_settings),
                    position_filled_fractions=dict(self.settings.position_filled_fractions),
                    position_cursor_entry_times=dict(self.settings.position_cursor_entry_times),
                    watchlist=self._trade_engine_watchlist_items(),
                )
            )
        except Exception as exc:
            self._handle_trade_engine_failure(f"Trade engine sync failed: {exc}")

    def _poll_trade_engine_events(self) -> None:
        if self.trade_engine is None:
            return
        if not self.trade_engine.is_alive():
            self._handle_trade_engine_failure("Trade engine process stopped")
            return
        for event in self.trade_engine.drain_events():
            self._handle_trade_engine_event(event)

    def _handle_trade_engine_failure(self, message: str) -> None:
        requested_auto_trade = bool(self.auto_trade_requested or self.auto_trade_enabled)
        self.engine_failed = True
        self.auto_trade_enabled = False
        self.auto_trade_requested = requested_auto_trade
        self.engine_order_pending = False
        self.order_pending_started_at = 0.0
        self.trade_engine_poll_timer.stop()
        if self.trade_engine is not None:
            self.trade_engine.stop()
            self.trade_engine = None
        self.last_engine_entry_signal_by_key.clear()
        self.last_engine_actionable_signal_by_key.clear()
        self.last_local_actionable_signal_by_key.clear()
        self.log(message)
        self._refresh_auto_trade_button_state()
        self._set_order_buttons_enabled(True)
        self._schedule_trade_engine_recovery()

    def _handle_trade_engine_event(self, event: object) -> None:
        if isinstance(event, EngineLogEvent):
            self.log(event.message)
            return
        if isinstance(event, EngineHealthEvent):
            if event.status == "failed":
                self._handle_trade_engine_failure(event.detail or "Trade engine failed")
            elif event.detail:
                self.log(event.detail)
            return
        if isinstance(event, EngineOrderSubmittedEvent):
            self.engine_order_pending = True
            self.order_pending_started_at = time.time()
            self.order_worker_symbol = event.symbol
            self.order_worker_is_auto_close = bool(event.auto_close)
            self.order_worker_is_auto_trade = bool(event.auto_trade)
            self.pending_open_order_interval = event.interval
            actionable_changed = self._clear_actionable_signal(event.symbol, event.interval)
            if actionable_changed:
                self._refresh_optimized_table_highlights()
            if event.auto_trade:
                self.auto_trade_entry_pending_symbol = event.symbol
                self.auto_trade_entry_pending_at = time.time()
                self.auto_trade_entry_pending_fraction = float(event.fraction)
            self._set_order_buttons_enabled(False)
            return
        if isinstance(event, EngineSignalEvent):
            focus_mode = str(self.settings.auto_trade_focus_signal_mode)
            signal_key = (event.symbol, event.interval, focus_mode)
            previous_signal = self.last_engine_entry_signal_by_key.get(signal_key, ("", 0))
            previous_actionable = self.last_engine_actionable_signal_by_key.get((event.symbol, event.interval), ("", 0, ""))
            confirmed_signal = (str(event.entry_side or ""), int(event.entry_zone or 0))
            preview_signal = (str(event.preview_entry_side or ""), int(event.preview_entry_zone or 0))
            self.last_engine_entry_signal_by_key[(event.symbol, event.interval, "confirmed")] = confirmed_signal
            self.last_engine_entry_signal_by_key[(event.symbol, event.interval, "preview")] = preview_signal
            next_actionable = self._set_engine_actionable_signal(
                event.symbol,
                event.interval,
                str(event.actionable_entry_side or ""),
                int(event.actionable_entry_zone or 0),
                str(event.actionable_entry_kind or ""),
            )
            if focus_mode == "confirmed":
                next_signal = confirmed_signal
            else:
                next_signal = preview_signal
            self.last_engine_entry_signal_by_key[signal_key] = next_signal
            if next_actionable != previous_actionable:
                self._refresh_optimized_table_highlights()
            _focus_on_signal = (
                bool(self.auto_trade_focus_enable_check.isChecked())
                if self.auto_trade_focus_enable_check is not None
                else bool(self.settings.auto_trade_focus_on_signal)
            )
            if (
                _focus_on_signal
                and next_signal[0]
                and next_signal[1] > 0
                and next_signal != previous_signal
            ):
                self._request_symbol_load(
                    event.symbol,
                    event.interval,
                    prefer_locked_position_settings=False,
                )
            return
        if isinstance(event, EngineOrderCompletedEvent):
            self.engine_order_pending = False
            self.order_pending_started_at = 0.0
            self.order_worker_symbol = event.symbol
            self.order_worker_is_auto_close = bool(event.auto_close)
            self.order_worker_is_auto_trade = bool(event.auto_trade)
            self.pending_open_order_interval = event.interval
            actionable_changed = self._clear_actionable_signal(event.symbol, event.interval)
            if actionable_changed:
                self._refresh_optimized_table_highlights()
            if event.strategy_settings is not None:
                self._remember_position_strategy_settings(event.symbol, event.strategy_settings)
            if not event.auto_close:
                self._remember_position_open_entry_events_for_key(event.symbol, event.interval)
            if event.auto_trade:
                self.auto_trade_entry_pending_symbol = event.symbol
                self.auto_trade_entry_pending_at = time.time()
                self.auto_trade_entry_pending_fraction = float(event.fraction)
                if event.interval in APP_INTERVAL_OPTIONS:
                    self._request_symbol_load(
                        event.symbol,
                        event.interval,
                        prefer_locked_position_settings=True,
                    )
                self._notify_telegram_auto_trade_entry(event)
            elif event.auto_close:
                self._notify_telegram_auto_trade_close(event)
            self._on_order_completed({"symbol": event.symbol, "message": event.message})
            return
        if isinstance(event, EngineOrderFailedEvent):
            self.engine_order_pending = False
            self.order_pending_started_at = 0.0
            self.order_worker_symbol = event.symbol
            self.order_worker_is_auto_close = bool(event.auto_close)
            self.order_worker_is_auto_trade = bool(event.auto_trade)
            self.pending_open_order_interval = event.interval
            actionable_changed = self._clear_actionable_signal(event.symbol, event.interval)
            if actionable_changed:
                self._refresh_optimized_table_highlights()
            if event.auto_trade:
                self.auto_trade_entry_pending_symbol = event.symbol
                self.auto_trade_entry_pending_at = time.time()
                self.auto_trade_entry_pending_fraction = float(event.fraction)
            self._on_order_failed(event.message)

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

    def _auto_close_managed_symbols(self) -> set[str]:
        managed = set(self.auto_close_enabled_symbols)
        if self.auto_trade_enabled:
            managed.update(position.symbol for position in self.open_positions)
        return managed

    def _is_auto_close_active_for_symbol(self, symbol: str) -> bool:
        return symbol in self._auto_close_managed_symbols()

    def _refresh_auto_close_retry_timer(self) -> None:
        self.auto_close_retry_timer.stop()

    def _auto_close_retry_allowed(self, symbol: str, bar_time: Optional[pd.Timestamp]) -> bool:
        if bar_time is None:
            return True
        last_bar_time = self.auto_close_last_trigger_time.get(symbol)
        if last_bar_time != bar_time:
            return True
        last_attempt = self.auto_close_last_attempt_at.get(symbol, 0.0)
        return (time.time() - last_attempt) >= (AUTO_CLOSE_RETRY_INTERVAL_MS / 1000.0)

    def _record_auto_close_attempt(self, symbol: str, bar_time: Optional[pd.Timestamp]) -> None:
        self.auto_close_last_attempt_at[symbol] = time.time()
        if bar_time is not None:
            self.auto_close_last_trigger_time[symbol] = bar_time

    def _run_auto_close_retry_cycle(self) -> None:
        if self._trade_engine_alive():
            self._sync_trade_engine_state()
            return

    def _eligible_auto_trade_results(self) -> List[OptimizationResult]:
        return list(self._ordered_optimized_results())

    def _strategy_settings_signature(self, settings: StrategySettings) -> str:
        return json.dumps(settings.to_dict(), sort_keys=True, ensure_ascii=True)

    def _favorable_backtest_seed(
        self,
        result: OptimizationResult,
        interval: str,
        target_settings: StrategySettings,
    ) -> Optional[BacktestResult]:
        key = self._symbol_interval_key(result.symbol, interval)
        if (
            self.current_symbol == result.symbol
            and self.current_interval == interval
            and self.current_backtest is not None
            and self.current_backtest.settings == target_settings
        ):
            return self.current_backtest
        resolved_backtest = self.resolved_auto_trade_backtest_cache.get(key)
        if resolved_backtest is not None and resolved_backtest.settings == target_settings:
            return resolved_backtest
        cached_backtest = self.backtest_cache.get(key)
        if cached_backtest is not None and cached_backtest.settings == target_settings:
            return cached_backtest
        if result.best_backtest.settings == target_settings:
            return result.best_backtest
        return None

    def _enqueue_favorable_backtest_refresh(
        self,
        result: OptimizationResult,
        history: Optional[pd.DataFrame],
        target_settings: StrategySettings,
    ) -> None:
        if history is None or history.empty:
            return
        interval = _optimization_result_interval(result)
        key = self._symbol_interval_key(result.symbol, interval)
        history_signature = _history_frame_signature(history)
        settings_signature = self._strategy_settings_signature(target_settings)
        if (
            key in self.resolved_auto_trade_backtest_cache
            and self.resolved_auto_trade_backtest_meta.get(key) == (history_signature, settings_signature)
        ):
            return
        if self.favorable_refresh_pending.get(key) == (history_signature, settings_signature):
            return
        seed_backtest = self._favorable_backtest_seed(result, interval, target_settings)
        self.favorable_backtest_process.submit(
            FavorableBacktestJob(
                symbol=result.symbol,
                interval=interval,
                strategy_settings=target_settings,
                history=history.copy(),
                seed_backtest=seed_backtest,
                fee_rate=self.settings.fee_rate,
                backtest_start_time=pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms"),
                history_signature=history_signature,
                settings_signature=settings_signature,
            )
        )
        self.favorable_refresh_pending[key] = (history_signature, settings_signature)
        self.log(f"highlight_refresh_enqueued: {result.symbol}/{interval}")

    def _poll_favorable_backtest_results(self) -> None:
        payloads = self.favorable_backtest_process.drain_results()
        if not payloads:
            return
        applied_any = False
        for payload in payloads:
            key = self._symbol_interval_key(payload.symbol, payload.interval)
            self.favorable_refresh_pending.pop(key, None)
            optimization = self.optimized_results.get(key)
            current_history = self._get_history_frame(payload.symbol, payload.interval)
            current_history_signature = _history_frame_signature(current_history)
            current_settings_signature = (
                self._strategy_settings_signature(optimization.best_backtest.settings)
                if optimization is not None
                else ""
            )
            if payload.error:
                self.log(f"highlight_refresh_discarded: {payload.symbol}/{payload.interval} source={payload.source}")
                continue
            if (
                optimization is None
                or payload.backtest is None
                or payload.history_signature != current_history_signature
                or payload.settings_signature != current_settings_signature
            ):
                self.log(f"highlight_refresh_discarded: {payload.symbol}/{payload.interval} source={payload.source}")
                continue
            self.resolved_auto_trade_backtest_cache[key] = payload.backtest
            self.resolved_auto_trade_backtest_meta[key] = (payload.history_signature, payload.settings_signature)
            self.backtest_cache[key] = payload.backtest
            applied_any = True
            self.log(f"highlight_refresh_applied: {payload.symbol}/{payload.interval} source={payload.source}")
        if applied_any:
            self._refresh_optimized_table_highlights()

    def _latest_auto_trade_backtest(self, result: OptimizationResult) -> Optional[BacktestResult]:
        interval = _optimization_result_interval(result)
        target_settings = result.best_backtest.settings
        history = self._get_history_frame(result.symbol, interval)
        key = self._symbol_interval_key(result.symbol, interval)
        seed_backtest = None
        if (
            self.current_symbol == result.symbol
            and self.current_interval == interval
            and self.current_backtest is not None
            and self.current_backtest.settings == target_settings
            and _backtest_matches_history(self.current_backtest, history)
        ):
            return self.current_backtest
        resolved_backtest = self.resolved_auto_trade_backtest_cache.get(key)
        if (
            resolved_backtest is not None
            and resolved_backtest.settings == target_settings
            and self.resolved_auto_trade_backtest_meta.get(key)
            == (_history_frame_signature(history), self._strategy_settings_signature(target_settings))
            and _backtest_matches_history(resolved_backtest, history)
        ):
            return resolved_backtest
        cached_backtest = self.backtest_cache.get(key)
        if (
            cached_backtest is not None
            and cached_backtest.settings == target_settings
            and _backtest_matches_history(cached_backtest, history)
        ):
            return cached_backtest
        self._enqueue_favorable_backtest_refresh(result, history, target_settings)
        return None

    def _best_available_auto_trade_backtest_for_display(
        self,
        result: OptimizationResult,
    ) -> Optional[BacktestResult]:
        latest_backtest = self._latest_auto_trade_backtest(result)
        if latest_backtest is not None:
            return latest_backtest
        interval = _optimization_result_interval(result)
        return self._favorable_backtest_seed(
            result,
            interval,
            result.best_backtest.settings,
        )

    def _fallback_auto_trade_items(self) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        existing_keys: set[Tuple[str, str]] = set()
        for result in self._eligible_auto_trade_results():
            interval = _optimization_result_interval(result)
            existing_keys.add((result.symbol, interval))
            items.append(
                {
                    "symbol": result.symbol,
                    "interval": interval,
                    "score": float(result.score),
                    "strategy_settings": result.best_backtest.settings,
                    "backtest": self._latest_auto_trade_backtest(result),
                }
            )
        for position in self.open_positions:
            symbol = str(position.symbol)
            interval = self._position_interval_for_symbol(symbol)
            key = (symbol, interval)
            if key in existing_keys:
                continue
            strategy_settings = self._active_backtest_settings(
                symbol,
                interval,
                prefer_locked_position_settings=True,
            )
            history = self._get_history_frame(symbol, interval)
            seed_backtest = self.backtest_cache.get(self._symbol_interval_key(symbol, interval))
            if (
                seed_backtest is None
                and symbol == self.current_symbol
                and interval == self.current_interval
                and self.current_backtest is not None
                and self.current_backtest.settings == strategy_settings
            ):
                seed_backtest = self.current_backtest
            backtest = self._materialize_cached_backtest(
                symbol,
                interval,
                history,
                seed_backtest,
                strategy_settings,
            )
            if backtest is None:
                continue
            items.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "score": 0.0,
                    "strategy_settings": strategy_settings,
                    "backtest": backtest,
                }
            )
        return items

    def _toggle_auto_trade_mode(self, checked: bool) -> None:
        self.auto_trade_requested = bool(checked)
        if self.auto_trade_requested and not self._auto_trade_ready():
            self.log("자동매매 예약됨: 최적화 완료 후 자동으로 시작합니다.")
            self._sync_trade_engine_state()
            self.update_positions_table()
            self._refresh_auto_trade_button_state()
            return
        if self.auto_trade_requested:
            self._enable_auto_trade_runtime()
        else:
            self._disable_auto_trade_runtime()
        self._refresh_auto_trade_button_state()

    def _run_auto_trade_cycle(
        self,
        *,
        trigger_symbol: Optional[str] = None,
        trigger_interval: Optional[str] = None,
        trigger_bar_time: Optional[pd.Timestamp] = None,
    ) -> None:
        def clear_local_actionable_signals() -> None:
            if self.last_local_actionable_signal_by_key:
                self.last_local_actionable_signal_by_key = {}
                self._refresh_optimized_table_highlights()

        if self._trade_engine_alive():
            self._sync_trade_engine_state()
            return
        if not self.auto_trade_enabled:
            clear_local_actionable_signals()
            return
        if self._is_refresh_running():
            clear_local_actionable_signals()
            return
        if self._is_order_pending():
            clear_local_actionable_signals()
            return
        if self.account_worker is not None and self.account_worker.isRunning():
            clear_local_actionable_signals()
            return
        if self.auto_trade_entry_pending_symbol:
            if time.time() - self.auto_trade_entry_pending_at > 60.0:
                self.log(f"Auto-trade pending timeout: {self.auto_trade_entry_pending_symbol}")
                self.auto_trade_entry_pending_symbol = None
                self.auto_trade_entry_pending_fraction = 0.0
                self.auto_trade_entry_pending_cursor_time = None
                self.auto_trade_filled_fraction_by_symbol = {}
                self.auto_trade_cursor_entry_time_by_symbol = {}
                self.last_engine_actionable_signal_by_key.clear()
                self.last_local_actionable_signal_by_key.clear()
                self._refresh_optimized_table_highlights()
            else:
                clear_local_actionable_signals()
                return
        now = time.time()
        for key, cooldown_until in list(self.auto_trade_reentry_cooldown_until.items()):
            if now >= cooldown_until:
                self.auto_trade_reentry_cooldown_until.pop(key, None)
        self._refresh_auto_close_monitors()
        trigger_symbol = str(trigger_symbol or "").upper()
        trigger_interval = str(trigger_interval or self.current_interval or self.settings.kline_interval)
        normalized_trigger_time = (
            pd.Timestamp(trigger_bar_time).tz_localize(None) if trigger_bar_time is not None else None
        )
        open_positions_by_symbol = {position.symbol: position for position in self.open_positions}
        eligible_items = self._fallback_auto_trade_items()
        if not eligible_items:
            clear_local_actionable_signals()
            return
        try:
            ticker_map = self.public_client.ticker_24h()
        except Exception as exc:
            self.log(f"자동매매 시세 조회 실패: {exc}")
            ticker_map = {}
        candidates: List[Dict[str, object]] = []
        next_actionable_signals: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
        for item in eligible_items:
            symbol = str(item["symbol"])
            interval = str(item["interval"])
            if self.auto_trade_reentry_cooldown_until.get((symbol, interval), 0.0) > now:
                continue
            open_position = open_positions_by_symbol.get(symbol)
            if open_positions_by_symbol and open_position is None:
                continue
            latest_backtest = item.get("backtest")
            ticker = ticker_map.get(symbol)
            current_price = 0.0 if not ticker else float(ticker.get("lastPrice", 0.0) or 0.0)
            evaluation = evaluate_auto_trade_candidate(
                symbol=symbol,
                interval=interval,
                score=float(item["score"]),
                strategy_settings=item.get("strategy_settings"),
                latest_backtest=latest_backtest,
                current_price=current_price,
                open_position=open_position,
                remembered_interval=self.settings.position_intervals.get(symbol),
                filled_fraction=self.auto_trade_filled_fraction_by_symbol.get(
                    symbol,
                    _inferred_auto_trade_fraction(latest_backtest, open_position)
                    if latest_backtest is not None
                    else 0.0,
                ),
                remembered_cursor_entry_time=self.auto_trade_cursor_entry_time_by_symbol.get(symbol),
                allow_favorable_price_entries=bool(self.settings.auto_trade_use_favorable_price),
                trigger_symbol=trigger_symbol,
                trigger_interval=trigger_interval,
                trigger_bar_time=normalized_trigger_time,
            )
            actionable_signal = self._actionable_signal_from_evaluation(evaluation)
            if actionable_signal[0]:
                next_actionable_signals[self._symbol_interval_key(symbol, interval)] = actionable_signal
            if evaluation.reentry_position_side and latest_backtest is not None:
                self._evaluate_backtest_auto_close(symbol, latest_backtest)
                continue
            if evaluation.candidate is None:
                continue
            candidates.append(dict(evaluation.candidate))
        if next_actionable_signals != self.last_local_actionable_signal_by_key:
            self.last_local_actionable_signal_by_key = dict(next_actionable_signals)
            self._refresh_optimized_table_highlights()
        chosen = pick_auto_trade_candidate(candidates, self._optimization_rank_mode())
        if chosen is None:
            return
        clear_local_actionable_signals()
        self.auto_trade_entry_pending_cursor_time = (
            pd.Timestamp(chosen["cursor_entry_time"])
            if chosen.get("cursor_entry_time") is not None
            else None
        )
        self._submit_open_order(
            str(chosen["side"]),
            fraction=float(chosen["fraction"]),
            symbol=str(chosen["symbol"]),
            interval=str(chosen["interval"]),
            auto_trade=True,
        )

    def _get_history_frame(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.history_cache.get(self._symbol_interval_key(symbol, interval))

    def _set_history_frame(self, symbol: str, frame: pd.DataFrame, interval: Optional[str] = None) -> None:
        target_interval = interval or self.current_interval or self.settings.kline_interval
        self.history_cache[self._symbol_interval_key(symbol, target_interval)] = frame
        if symbol == self.current_symbol and target_interval == self.current_interval:
            self._sync_current_chart_snapshot(symbol, target_interval, confirmed_history=frame)

    def _get_chart_history_frame(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.chart_history_cache.get(self._symbol_interval_key(symbol, interval))

    def _set_chart_history_frame(self, symbol: str, frame: pd.DataFrame, interval: Optional[str] = None) -> None:
        target_interval = interval or self.current_interval or self.settings.kline_interval
        self.chart_history_cache[self._symbol_interval_key(symbol, target_interval)] = frame
        if symbol == self.current_symbol and target_interval == self.current_interval:
            self._sync_current_chart_snapshot(symbol, target_interval, confirmed_chart_history=frame)

    def _snapshot_matches_context(
        self,
        snapshot: Optional[AuthoritativeChartSnapshot],
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> bool:
        if snapshot is None:
            return False
        target_symbol = symbol or self.current_symbol
        target_interval = interval or self.current_interval or self.settings.kline_interval
        return snapshot.symbol == target_symbol and snapshot.interval == target_interval

    def _clear_current_chart_snapshot(self) -> None:
        self.current_chart_snapshot = None

    def _current_authoritative_snapshot(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> Optional[AuthoritativeChartSnapshot]:
        target_symbol = symbol or self.current_symbol
        target_interval = interval or self.current_interval or self.settings.kline_interval
        if not target_symbol:
            return None
        if self._snapshot_matches_context(self.current_chart_snapshot, target_symbol, target_interval):
            return self.current_chart_snapshot
        key = self._symbol_interval_key(target_symbol, target_interval)
        snapshot = AuthoritativeChartSnapshot(
            symbol=target_symbol,
            interval=target_interval,
            confirmed_history=self._get_history_frame(target_symbol, target_interval),
            confirmed_chart_history=self._get_chart_history_frame(target_symbol, target_interval),
            backtest=self.current_backtest if target_symbol == self.current_symbol and target_interval == self.current_interval else self.backtest_cache.get(key),
            chart_indicators=(
                self.current_chart_indicators
                if target_symbol == self.current_symbol and target_interval == self.current_interval
                else self.chart_indicator_cache.get(key)
            ),
            preview_bar=self._current_live_preview_for(target_symbol, target_interval),
            render_signature=self.chart_render_signature if target_symbol == self.current_symbol and target_interval == self.current_interval else (None, 0),
        )
        if target_symbol == self.current_symbol and target_interval == self.current_interval:
            self.current_chart_snapshot = snapshot
        return snapshot

    def _sync_current_chart_snapshot(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        *,
        confirmed_history: object = _SNAPSHOT_KEEP,
        confirmed_chart_history: object = _SNAPSHOT_KEEP,
        backtest: object = _SNAPSHOT_KEEP,
        chart_indicators: object = _SNAPSHOT_KEEP,
        preview_bar: object = _SNAPSHOT_KEEP,
        render_signature: object = _SNAPSHOT_KEEP,
    ) -> Optional[AuthoritativeChartSnapshot]:
        target_symbol = symbol or self.current_symbol
        target_interval = interval or self.current_interval or self.settings.kline_interval
        if not target_symbol:
            self.current_chart_snapshot = None
            return None
        base = self._current_authoritative_snapshot(target_symbol, target_interval)
        if base is None:
            return None
        snapshot = AuthoritativeChartSnapshot(
            symbol=target_symbol,
            interval=target_interval,
            confirmed_history=base.confirmed_history if confirmed_history is _SNAPSHOT_KEEP else confirmed_history,
            confirmed_chart_history=base.confirmed_chart_history if confirmed_chart_history is _SNAPSHOT_KEEP else confirmed_chart_history,
            backtest=base.backtest if backtest is _SNAPSHOT_KEEP else backtest,
            chart_indicators=base.chart_indicators if chart_indicators is _SNAPSHOT_KEEP else chart_indicators,
            preview_bar=base.preview_bar if preview_bar is _SNAPSHOT_KEEP else preview_bar,
            render_signature=base.render_signature if render_signature is _SNAPSHOT_KEEP else render_signature,
        )
        if target_symbol == self.current_symbol and target_interval == self.current_interval:
            self.current_chart_snapshot = snapshot
            self.current_backtest = snapshot.backtest
            self.current_chart_indicators = snapshot.chart_indicators
            self.chart_render_signature = snapshot.render_signature
        return snapshot

    def _clear_current_live_preview(self) -> None:
        self.current_live_preview_bar = None
        if self.current_symbol:
            self._sync_current_chart_snapshot(self.current_symbol, self.current_interval, preview_bar=None)

    def _current_live_preview_for(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        target_symbol = symbol or self.current_symbol
        target_interval = interval or self.current_interval or self.settings.kline_interval
        if not _preview_bar_matches_context(self.current_live_preview_bar, target_symbol, target_interval):
            return None
        return dict(self.current_live_preview_bar)

    def _display_history_frame(
        self,
        symbol: str,
        interval: Optional[str] = None,
        *,
        chart: bool = False,
    ) -> Optional[pd.DataFrame]:
        target_interval = interval or self.current_interval or self.settings.kline_interval
        if self._snapshot_matches_context(self.current_chart_snapshot, symbol, target_interval):
            snapshot = self.current_chart_snapshot
            if snapshot is not None:
                return snapshot.display_chart_history() if chart else snapshot.display_history()
        base_frame = self._get_chart_history_frame(symbol, target_interval) if chart else self._get_history_frame(symbol, target_interval)
        preview_bar = self._current_live_preview_for(symbol, target_interval)
        max_rows = CHART_HISTORY_BAR_LIMIT if chart else None
        return _history_with_live_preview(base_frame, preview_bar, max_rows=max_rows)

    def _reapply_live_preview_bar(self, symbol: str) -> None:
        preview_bar = self._current_live_preview_for(symbol, self.current_interval)
        if preview_bar is None:
            return
        self._apply_live_lightweight_bar(symbol, preview_bar)

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

    def _remember_position_filled_fraction(
        self,
        symbol: str,
        fraction: float,
        persist: bool = True,
    ) -> None:
        if not symbol:
            return
        normalized = max(0.0, min(signal_fraction_for_zone(3), float(fraction)))
        current = self.settings.position_filled_fractions.get(symbol)
        if current is not None and abs(current - normalized) <= 1e-9:
            return
        self.auto_trade_filled_fraction_by_symbol[symbol] = normalized
        self.settings.position_filled_fractions[symbol] = normalized
        if persist:
            self._persist_position_intervals()

    def _remember_position_cursor_entry_time(
        self,
        symbol: str,
        entry_time: Optional[pd.Timestamp],
        persist: bool = True,
    ) -> None:
        if not symbol or entry_time is None:
            return
        normalized = pd.Timestamp(entry_time).tz_localize(None)
        current = self.settings.position_cursor_entry_times.get(symbol)
        if current is not None and current == normalized:
            return
        self.auto_trade_cursor_entry_time_by_symbol[symbol] = normalized
        self.settings.position_cursor_entry_times[symbol] = normalized
        if persist:
            self._persist_position_intervals()

    def _remember_position_interval(self, symbol: str, interval: Optional[str], persist: bool = True) -> None:
        normalized = str(interval or "").strip()
        if not symbol or normalized not in APP_INTERVAL_OPTIONS:
            return
        if self.settings.position_intervals.get(symbol) == normalized:
            return
        self.settings.position_intervals[symbol] = normalized
        if persist:
            self._persist_position_intervals()

    def _remember_position_strategy_settings(
        self,
        symbol: str,
        strategy_settings: Optional[StrategySettings],
        persist: bool = True,
    ) -> None:
        if not symbol or strategy_settings is None:
            return
        if self.position_strategy_by_symbol.get(symbol) == strategy_settings and self.settings.position_strategy_settings.get(symbol) == strategy_settings:
            return
        self.position_strategy_by_symbol[symbol] = strategy_settings
        self.settings.position_strategy_settings[symbol] = strategy_settings
        if persist:
            self._persist_position_intervals()

    def _normalized_position_entry_chain(
        self,
        symbol: str,
        raw_events: List[Tuple[object, object]],
    ) -> List[Tuple[pd.Timestamp, str]]:
        chain_start = (
            self.settings.position_cursor_entry_times.get(symbol)
            or self.auto_trade_cursor_entry_time_by_symbol.get(symbol)
        )
        normalized = sorted(
            {
                (pd.Timestamp(event_time).tz_localize(None), str(event_label).strip().upper())
                for event_time, event_label in raw_events
                if str(event_label or "").strip()
            },
            key=lambda item: item[0],
        )
        if chain_start is not None:
            normalized = [
                (event_time, event_label)
                for event_time, event_label in normalized
                if event_time >= pd.Timestamp(chain_start).tz_localize(None)
            ]
        if not normalized:
            return []
        latest_label = str(normalized[-1][1] or "").strip().upper()
        active_prefix = latest_label[:1] if latest_label[:1] in {"L", "S"} else ""
        if active_prefix:
            normalized = [
                (event_time, event_label)
                for event_time, event_label in normalized
                if str(event_label or "").strip().upper().startswith(active_prefix)
            ]
        cleaned: List[Tuple[pd.Timestamp, str]] = []
        last_zone = 0
        for event_time, event_label in normalized:
            label_text = str(event_label or "").strip().upper()
            if len(label_text) < 2 or label_text[:1] not in {"L", "S"}:
                continue
            try:
                zone = int(label_text[1:])
            except Exception:
                continue
            if zone not in {1, 2, 3} or zone <= last_zone:
                continue
            cleaned.append((pd.Timestamp(event_time).tz_localize(None), label_text))
            last_zone = zone
        return cleaned

    def _remember_position_open_entry_events(
        self,
        symbol: str,
        backtest: Optional[BacktestResult],
        persist: bool = True,
    ) -> None:
        if not symbol or backtest is None:
            return
        cursor = getattr(backtest, "cursor", None)
        raw_events: List[Tuple[object, object]] = []
        has_authoritative_cursor_events = False
        if cursor is not None and abs(float(getattr(cursor, "position_qty", 0.0) or 0.0)) > 1e-12:
            raw_events = list(getattr(cursor, "zone_event_times", ()) or ())
            has_authoritative_cursor_events = bool(raw_events)
        if not raw_events:
            raw_events = list(getattr(backtest, "open_entry_events", ()) or ())
        if not raw_events:
            return
        normalized = self._normalized_position_entry_chain(symbol, raw_events)
        if not normalized:
            return
        if not has_authoritative_cursor_events:
            existing_events = self._normalized_position_entry_chain(
                symbol,
                list(self.position_open_entry_events_by_symbol.get(symbol, ())),
            )
            if existing_events:
                normalized = existing_events
        if (
            self.position_open_entry_events_by_symbol.get(symbol) == normalized
            and self.settings.position_open_entry_events.get(symbol) == normalized
        ):
            return
        self.position_open_entry_events_by_symbol[symbol] = normalized
        self.settings.position_open_entry_events[symbol] = normalized
        if persist:
            self._persist_position_intervals()

    def _remember_position_open_entry_events_for_key(
        self,
        symbol: str,
        interval: Optional[str],
        persist: bool = True,
    ) -> None:
        normalized_interval = str(interval or "").strip()
        backtest: Optional[BacktestResult] = None
        if symbol == self.current_symbol and normalized_interval == self.current_interval and self.current_backtest is not None:
            backtest = self.current_backtest
        elif normalized_interval in APP_INTERVAL_OPTIONS:
            backtest = self.backtest_cache.get((symbol, normalized_interval))
        if backtest is not None:
            self._remember_position_open_entry_events(symbol, backtest, persist=persist)

    def _forget_closed_position_intervals(self, open_symbols: set[str], persist: bool = True) -> None:
        removed = False
        for symbol in list(self.settings.position_intervals):
            if symbol in open_symbols:
                continue
            self.settings.position_intervals.pop(symbol, None)
            removed = True
        for symbol in list(self.settings.position_strategy_settings):
            if symbol in open_symbols:
                continue
            self.settings.position_strategy_settings.pop(symbol, None)
            self.position_strategy_by_symbol.pop(symbol, None)
            removed = True
        for symbol in list(self.settings.position_filled_fractions):
            if symbol in open_symbols:
                continue
            self.settings.position_filled_fractions.pop(symbol, None)
            self.auto_trade_filled_fraction_by_symbol.pop(symbol, None)
            removed = True
        for symbol in list(self.settings.position_cursor_entry_times):
            if symbol in open_symbols:
                continue
            self.settings.position_cursor_entry_times.pop(symbol, None)
            self.auto_trade_cursor_entry_time_by_symbol.pop(symbol, None)
            removed = True
        for symbol in list(self.settings.position_open_entry_events):
            if symbol in open_symbols:
                continue
            self.settings.position_open_entry_events.pop(symbol, None)
            self.position_open_entry_events_by_symbol.pop(symbol, None)
            removed = True
        if removed and persist:
            self._persist_position_intervals()

    def _remember_missing_open_position_intervals(self, open_symbols: set[str]) -> None:
        changed = False
        pending_symbol = str(self.auto_trade_entry_pending_symbol or self.order_worker_symbol or "").strip().upper()
        pending_interval = str(self.pending_open_order_interval or "").strip()
        for symbol in sorted(open_symbols):
            if symbol == pending_symbol and pending_interval in APP_INTERVAL_OPTIONS:
                interval = pending_interval
                if self.settings.position_intervals.get(symbol) != interval:
                    self.settings.position_intervals[symbol] = interval
                    changed = True
            elif self.settings.position_intervals.get(symbol) in APP_INTERVAL_OPTIONS:
                interval = self.settings.position_intervals[symbol]
            else:
                optimization = self._optimization_result(symbol)
                interval = _optimization_result_interval(optimization) if optimization else self.settings.kline_interval
                if interval in APP_INTERVAL_OPTIONS:
                    self.settings.position_intervals[symbol] = interval
                    changed = True
            if symbol in self.settings.position_strategy_settings:
                self.position_strategy_by_symbol[symbol] = self.settings.position_strategy_settings[symbol]
            else:
                optimization = self._optimization_result(symbol, interval if interval in APP_INTERVAL_OPTIONS else None)
                strategy_settings = optimization.best_backtest.settings if optimization else self.settings.strategy
                self.settings.position_strategy_settings[symbol] = strategy_settings
                self.position_strategy_by_symbol[symbol] = strategy_settings
                changed = True
            if symbol in self.settings.position_filled_fractions:
                self.auto_trade_filled_fraction_by_symbol[symbol] = self.settings.position_filled_fractions[symbol]
            if symbol in self.settings.position_cursor_entry_times:
                self.auto_trade_cursor_entry_time_by_symbol[symbol] = self.settings.position_cursor_entry_times[symbol]
            if symbol in self.settings.position_open_entry_events:
                self.position_open_entry_events_by_symbol[symbol] = self.settings.position_open_entry_events[symbol]
        if changed:
            self._persist_position_intervals()

    def _position_interval_for_symbol(self, symbol: str) -> str:
        remembered = self.settings.position_intervals.get(symbol)
        if remembered in APP_INTERVAL_OPTIONS and self._find_open_position(symbol) is not None:
            return remembered
        optimization = self._optimization_result(symbol)
        return _optimization_result_interval(optimization) if optimization else self.settings.kline_interval

    def _locked_position_strategy_settings(
        self,
        symbol: str,
        interval: Optional[str] = None,
    ) -> Optional[StrategySettings]:
        if self._find_open_position(symbol) is None:
            return None
        locked_settings = self.position_strategy_by_symbol.get(symbol)
        if locked_settings is None:
            return None
        target_interval = interval if interval in APP_INTERVAL_OPTIONS else None
        remembered_interval = self.settings.position_intervals.get(symbol)
        if target_interval is not None and remembered_interval in APP_INTERVAL_OPTIONS and target_interval != remembered_interval:
            return None
        return locked_settings

    def _position_symbol_text(self, symbol: str) -> str:
        return f"{symbol} [{self._position_interval_for_symbol(symbol)}]"

    def _active_interval_for_symbol(self, symbol: str) -> str:
        return self._position_interval_for_symbol(symbol)

    def _find_open_position(self, symbol: str) -> Optional[PositionSnapshot]:
        return next((position for position in self.open_positions if position.symbol == symbol), None)

    def _set_auto_close_button_state(
        self,
        button: QPushButton,
        enabled: bool,
        *,
        forced_by_auto_trade: bool = False,
    ) -> None:
        button.setCheckable(True)
        button.setChecked(enabled)
        if forced_by_auto_trade and enabled:
            button.setText("자동청산 ON")
            button.setToolTip("자동매매 사용 중에는 자동청산이 함께 적용됩니다.")
        else:
            button.setText("자동청산 ON" if enabled else "자동청산 OFF")
            button.setToolTip("포지션별 자동청산 사용 여부")
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

    def _set_auto_trade_button_state(self, enabled: bool, pending: bool = False) -> None:
        if not hasattr(self, "auto_trade_button"):
            return
        button = self.auto_trade_button
        button.blockSignals(True)
        button.setCheckable(True)
        button.setChecked(enabled)
        if pending:
            button.setText("자동매매 예약")
        else:
            button.setText("자동매매 ON" if enabled else "자동매매 OFF")
        button.setStyleSheet(
            """
            QPushButton {
                font-weight: 700;
                font-size: 11px;
                color: #d8dee9;
                background-color: #3b4252;
                border: 1px solid #495468;
                border-radius: 4px;
                padding: 4px 10px;
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
                background-color: #0e9f6e;
                border-color: #0b7a55;
            }
            QPushButton:checked:hover {
                background-color: #11ad79;
                border-color: #0d8b61;
            }
            QPushButton:disabled {
                color: #9aa4b2;
                background-color: #232a35;
                border-color: #313a49;
            }
            """
        )
        button.blockSignals(False)

    def _auto_trade_ready(self) -> bool:
        return bool(self.optimized_results) and bool(self.settings.api_key and self.settings.api_secret)

    def _enable_auto_trade_runtime(self) -> None:
        if self.auto_trade_enabled:
            return
        if not self._ensure_trade_engine_available():
            self.log("자동매매 예약됨: trade engine 복구 후 자동으로 다시 켜집니다.")
            self._refresh_auto_trade_button_state()
            return
        self.auto_trade_enabled = True
        self.log("자동매매 활성화")
        self.refresh_account_info()
        self._refresh_auto_close_monitors()
        self._sync_trade_engine_state()
        self.update_positions_table()

    def _disable_auto_trade_runtime(self, *, log_message: bool = True) -> None:
        if not self.auto_trade_enabled:
            return
        self.auto_trade_enabled = False
        if log_message:
            self.log("자동매매 비활성화")
        self.auto_trade_entry_pending_symbol = None
        self.auto_trade_entry_pending_fraction = 0.0
        self.auto_trade_entry_pending_cursor_time = None
        self.auto_trade_filled_fraction_by_symbol = {}
        self.auto_trade_cursor_entry_time_by_symbol = {}
        self.last_engine_actionable_signal_by_key.clear()
        self.last_local_actionable_signal_by_key.clear()
        self._refresh_auto_close_monitors()
        self._sync_trade_engine_state()
        self.update_positions_table()
        self._refresh_optimized_table_highlights()

    def _activate_requested_auto_trade_if_ready(self) -> None:
        if not self.auto_trade_requested or self.auto_trade_enabled or not self._auto_trade_ready():
            self._refresh_auto_trade_button_state()
            return
        self._enable_auto_trade_runtime()
        self._refresh_auto_trade_button_state()

    def _refresh_auto_trade_button_state(self) -> None:
        if not hasattr(self, "auto_trade_button"):
            return
        available = bool(self.settings.api_key and self.settings.api_secret)
        requested = bool(self.auto_trade_requested or self.auto_trade_enabled)
        pending = bool(self.auto_trade_requested and not self.auto_trade_enabled)
        self.auto_trade_button.setEnabled(requested or available)
        self._set_auto_trade_button_state(requested, pending=pending)

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

    def _default_lightweight_logical_range(self, candle_df: pd.DataFrame) -> tuple[float, float]:
        interval_ms = _interval_to_ms(self.current_interval or self.settings.kline_interval)
        visible_bars = max(1, int((max(1, int(self.settings.chart_display_hours)) * 3_600_000) // interval_ms))
        if candle_df.empty:
            return 0.0, float(DEFAULT_CHART_RIGHT_PAD_BARS)
        bar_count = len(candle_df)
        start_value = float(max(0, bar_count - visible_bars))
        end_value = float(max(0, bar_count - 1) + DEFAULT_CHART_RIGHT_PAD_BARS)
        return start_value, end_value

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
        self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=self.current_backtest,
            chart_indicators=self.current_chart_indicators,
        )

    def _trade_markers(self, trades, latest_time: Optional[pd.Timestamp]) -> List[Dict[str, object]]:
        markers: List[Dict[str, object]] = []
        for trade in trades:
            if _is_provisional_exit_trade(trade, latest_time):
                continue
            entry_events = list(getattr(trade, "entry_events", ()) or ())
            if not entry_events:
                entry_events = [(pd.Timestamp(trade.entry_time), str(trade.zones or ""))]
            for event_time, event_label in entry_events:
                markers.append(
                    {
                        "time": pd.Timestamp(event_time),
                        "position": "below" if trade.side == "long" else "above",
                        "shape": "arrow_up" if trade.side == "long" else "arrow_down",
                        "color": "#17c964" if trade.side == "long" else "#f31260",
                        "text": str(event_label),
                    }
                )
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

    def _open_entry_markers(
        self,
        backtest: Optional[BacktestResult],
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        if backtest is None and not symbol:
            return []
        markers: List[Dict[str, object]] = []
        merged_events: List[Tuple[pd.Timestamp, str]] = []
        use_locked_position_events = False
        if symbol:
            remembered_interval = self.settings.position_intervals.get(symbol)
            has_open_position = self._find_open_position(symbol) is not None
            prefers_locked_position_settings = (
                self.current_chart_prefers_locked_position_settings
                and symbol == self.current_symbol
                and interval == self.current_interval
            )
            remembered_events = self._normalized_position_entry_chain(
                symbol,
                list(self.position_open_entry_events_by_symbol.get(symbol, ())),
            )
            if (
                (has_open_position or prefers_locked_position_settings)
                and interval in APP_INTERVAL_OPTIONS
                and remembered_interval == interval
                and remembered_events
            ):
                merged_events.extend(remembered_events)
                use_locked_position_events = True
        if backtest is not None and not use_locked_position_events:
            merged_events.extend(
                (pd.Timestamp(event_time).tz_localize(None), str(event_label))
                for event_time, event_label in list(getattr(backtest, "open_entry_events", ()) or ())
            )
        if symbol and not use_locked_position_events:
            merged_events.extend(self.position_open_entry_events_by_symbol.get(symbol, ()))
        for event_time, event_label in sorted(set(merged_events), key=lambda item: item[0]):
            label_text = str(event_label)
            is_long = label_text.upper().startswith("L")
            markers.append(
                {
                    "time": pd.Timestamp(event_time),
                    "position": "below" if is_long else "above",
                    "shape": "arrow_up" if is_long else "arrow_down",
                    "color": "#17c964" if is_long else "#f31260",
                    "text": label_text,
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
        fast_markers = list(self.current_lightweight_fast_entry_markers) + list(self.current_lightweight_fast_exit_markers)
        if fast_markers:
            marker_keys = {
                (pd.Timestamp(marker["time"]), marker.get("position"), marker.get("shape"))
                for marker in markers
            }
            for marker in fast_markers:
                marker_key = (pd.Timestamp(marker["time"]), marker.get("position"), marker.get("shape"))
                if marker_key in marker_keys:
                    continue
                markers.append(marker)
                marker_keys.add(marker_key)
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
        if self.chart is None:
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
            or self.current_backtest is None
            or self.current_backtest.cursor is None
        ):
            return []
        position_qty = float(self.current_backtest.cursor.position_qty)
        history = self._display_history_frame(symbol, self.current_interval)
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
                    "text": "예상청산(참고)",
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
                    "text": f"{side[0].upper()}{zone} 예상진입(참고)",
                }
            )
        return preview_markers

    def _build_fast_exit_markers(self, exit_event: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
        if not exit_event:
            return []
        bar_time = exit_event.get("bar_time")
        reason = str(exit_event.get("reason", "")).strip()
        side = str(exit_event.get("side", "")).lower()
        if bar_time is None or side not in {"long", "short"} or not reason:
            return []
        return [
            {
                "time": pd.Timestamp(bar_time),
                "position": "above" if side == "long" else "below",
                "shape": "circle",
                "color": "#f801e8",
                "text": "청산신호",
            }
        ]

    def _build_fast_entry_markers(self, entry_event: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
        if not entry_event:
            return []
        bar_time = entry_event.get("bar_time")
        side = str(entry_event.get("side", "")).lower()
        zone = int(entry_event.get("zone", 0) or 0)
        if bar_time is None or side not in {"long", "short"} or zone not in {1, 2, 3}:
            return []
        return [
            {
                "time": pd.Timestamp(bar_time),
                "position": "below" if side == "long" else "above",
                "shape": "arrow_up" if side == "long" else "arrow_down",
                "color": "#17c964" if side == "long" else "#f31260",
                "text": f"{side[0].upper()}{zone} 진입신호",
            }
        ]

    def _build_active_entry_signal_markers(self) -> List[Dict[str, object]]:
        backtest = self.current_backtest
        cursor = backtest.cursor if backtest is not None else None
        if cursor is None or abs(float(cursor.position_qty)) < 1e-12:
            return []
        side = str(cursor.last_entry_signal_side or cursor.entry_side or ("long" if float(cursor.position_qty) > 0 else "short")).lower()
        zone = int(cursor.last_entry_signal_zone or (cursor.last_long_zone if side == "long" else cursor.last_short_zone) or 0)
        signal_time = cursor.last_entry_signal_time or cursor.entry_time
        if side not in {"long", "short"} or zone not in {1, 2, 3} or signal_time is None:
            return []
        return [
            {
                "time": pd.Timestamp(signal_time),
                "position": "below" if side == "long" else "above",
                "shape": "arrow_up" if side == "long" else "arrow_down",
                "color": "#17c964" if side == "long" else "#f31260",
                "text": f"{side[0].upper()}{zone}",
            }
        ]

    def _evaluate_closed_bar_auto_close(self, symbol: str, history: Optional[pd.DataFrame]) -> None:
        if (
            symbol != self.current_symbol
            or self.current_backtest is None
            or self.current_backtest.cursor is None
            or history is None
            or history.empty
        ):
            return
        # Prefer the real Binance position amount: after the backtest processes
        # an exit on the latest bar, cursor.position_qty becomes 0 even though
        # the real position is still open (auto-close order hasn't executed yet).
        real_position = (
            self.current_position_snapshot
            if self.current_position_snapshot is not None and self.current_position_snapshot.symbol == symbol
            else self._find_open_position(symbol)
        )
        if real_position is not None and abs(float(real_position.amount)) > 1e-12:
            position_qty = float(real_position.amount)
        else:
            position_qty = float(self.current_backtest.cursor.position_qty)
        latest_state, _ = evaluate_latest_state(
            history,
            self.current_backtest.settings,
            cursor=self.current_backtest.cursor.indicator_cursor,
        )
        latest_time = pd.Timestamp(history["time"].iloc[-1])
        entry_event = latest_confirmed_entry_event(self.current_backtest, latest_time)
        exit_event = _confirmed_exit_event_from_state(position_qty, latest_state, latest_time)
        next_fast_entry_markers = self._build_fast_entry_markers(entry_event)
        next_fast_exit_markers = self._build_fast_exit_markers(exit_event)
        fast_entry_signature = [self._marker_signature(marker) for marker in next_fast_entry_markers]
        current_fast_entry_signature = [self._marker_signature(marker) for marker in self.current_lightweight_fast_entry_markers]
        fast_exit_signature = [self._marker_signature(marker) for marker in next_fast_exit_markers]
        current_fast_exit_signature = [self._marker_signature(marker) for marker in self.current_lightweight_fast_exit_markers]
        preview_was_visible = bool(self.current_lightweight_preview_markers)
        self.current_lightweight_preview_markers = []
        if (
            fast_entry_signature != current_fast_entry_signature
            or fast_exit_signature != current_fast_exit_signature
            or preview_was_visible
        ):
            self.current_lightweight_fast_entry_markers = list(next_fast_entry_markers)
            self.current_lightweight_fast_exit_markers = list(next_fast_exit_markers)
            self._render_lightweight_markers()
        if exit_event is not None:
            self._maybe_trigger_auto_close(symbol, exit_event)

    def _apply_closed_bar_confirmed_backtest(
        self,
        symbol: str,
        history: Optional[pd.DataFrame],
        chart_history: Optional[pd.DataFrame],
    ) -> bool:
        if (
            symbol != self.current_symbol
            or history is None
            or history.empty
            or chart_history is None
            or chart_history.empty
        ):
            return False
        strategy_settings = self._active_backtest_settings(symbol, self.current_interval)
        previous_backtest = self.current_backtest
        previous_chart_indicators = self.current_chart_indicators
        backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
        try:
            if (
                previous_backtest is not None
                and previous_backtest.settings == strategy_settings
                and _backtest_matches_history(previous_backtest, history)
            ):
                backtest = previous_backtest
            elif (
                previous_backtest is not None
                and previous_backtest.settings == strategy_settings
                and _history_can_resume_backtest(previous_backtest, history)
            ):
                backtest = resume_backtest(
                    history,
                    previous_result=previous_backtest,
                    settings=strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            else:
                backtest = run_backtest(
                    history,
                    settings=strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            chart_indicators = _chart_indicators_from_backtest(backtest, chart_history)
        except Exception:
            return False
        self.live_recalc_pending = False
        self.current_backtest = backtest
        self.current_chart_indicators = chart_indicators
        self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=self.current_backtest,
            chart_indicators=self.current_chart_indicators,
        )
        cache_key = self._symbol_interval_key(symbol, self.current_interval)
        self.backtest_cache[cache_key] = self.current_backtest
        self.chart_indicator_cache[cache_key] = self.current_chart_indicators
        self.current_lightweight_preview_markers = []
        self.current_lightweight_fast_entry_markers = []
        self.current_lightweight_fast_exit_markers = []
        applied_incrementally = self._apply_incremental_lightweight_backtest(
            symbol,
            previous_backtest,
            previous_chart_indicators,
            self.current_backtest,
            self.current_chart_indicators,
            chart_history,
        )
        if not applied_incrementally:
            self.render_chart(
                symbol,
                self.current_backtest,
                reset_view=False,
                chart_indicators=self.current_chart_indicators,
            )
        self.update_summary(symbol, self.current_backtest, self._optimization_result(symbol, self.current_interval))
        self._evaluate_backtest_auto_close(symbol, self.current_backtest)
        if self.auto_trade_enabled and self.auto_trade_entry_pending_symbol is None:
            confirmed_bar_time = pd.Timestamp(history["time"].iloc[-1])
            QTimer.singleShot(
                0,
                lambda: self._run_auto_trade_cycle(
                    trigger_symbol=symbol,
                    trigger_interval=self.current_interval,
                    trigger_bar_time=confirmed_bar_time,
                ),
            )
        return True

    def _refresh_live_preview_markers(self, symbol: Optional[str]) -> None:
        if self.chart is None:
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
        snapshot: AuthoritativeChartSnapshot,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
        if snapshot.backtest is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
        active_chart_indicators = snapshot.chart_indicators
        if active_chart_indicators is None:
            active_chart_indicators = _chart_indicators_from_backtest(snapshot.backtest, snapshot.confirmed_chart_history)
        indicators = (
            active_chart_indicators.sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        candle_df = _chart_candle_frame(indicators)
        equity_df = (
            pd.DataFrame({"time": list(snapshot.backtest.equity_curve.index), "Equity": list(snapshot.backtest.equity_curve.values)})
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        latest_time = pd.Timestamp(candle_df["time"].iloc[-1]) if not candle_df.empty else None
        markers = self._trade_markers(snapshot.backtest.trades, latest_time)
        markers.extend(self._open_entry_markers(snapshot.backtest, snapshot.symbol, snapshot.interval))
        return candle_df, indicators, equity_df, markers

    def _clear_lightweight_volume_series(self) -> None:
        if self.chart is None:
            return
        self.chart.run_script(f"{self.chart.id}.volumeSeries.setData([])")

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
            or target_symbol != self.current_symbol
            or self.current_backtest is None
            or self.chart_history_page_worker is not None
            or self.chart is None
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
        if self.chart is None or not self.current_symbol:
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

    def _clear_stashed_lightweight_range(self) -> None:
        self.pending_lightweight_range_shift = 0
        if self.chart is None:
            return
        try:
            self.chart.run_script("window.__alt_lwc_view_range = null;")
        except Exception:
            pass

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

    def _restore_lightweight_range_via_raf(self, symbol: str) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        shift = float(self.pending_lightweight_range_shift)
        self.pending_lightweight_range_shift = 0
        equity_id = self.equity_subchart.id
        chart_id = self.chart.id
        self.chart.run_script(f"""
            requestAnimationFrame(function() {{
                requestAnimationFrame(function() {{
                    var range = window.__alt_lwc_view_range;
                    if (range && Number.isFinite(range.from) && Number.isFinite(range.to) && range.to > range.from) {{
                        var fromValue = range.from + {shift};
                        var toValue = range.to + {shift};
                        {equity_id}.chart.timeScale().setVisibleLogicalRange({{
                            from: fromValue, to: toValue
                        }});
                        {chart_id}.chart.timeScale().setVisibleLogicalRange({{
                            from: fromValue, to: toValue
                        }});
                    }}
                }});
            }});
        """)

    def _active_backtest_settings(
        self,
        symbol: str,
        interval: Optional[str] = None,
        *,
        prefer_locked_position_settings: bool = False,
    ) -> StrategySettings:
        target_interval = interval if interval in APP_INTERVAL_OPTIONS else self._active_interval_for_symbol(symbol)
        locked_settings = None
        if prefer_locked_position_settings or (
            self.current_chart_prefers_locked_position_settings
            and symbol == self.current_symbol
            and target_interval == self.current_interval
        ):
            locked_settings = self._locked_position_strategy_settings(symbol, target_interval)
        if locked_settings is not None:
            return locked_settings
        optimization = self._optimization_result(symbol, target_interval)
        return optimization.best_backtest.settings if optimization else self.settings.strategy

    def _backtest_settings_for_symbol_interval(
        self,
        symbol: str,
        interval: str,
        optimization: Optional[OptimizationResult] = None,
        *,
        prefer_locked_position_settings: bool = False,
    ) -> StrategySettings:
        locked_settings = None
        if prefer_locked_position_settings:
            locked_settings = self._locked_position_strategy_settings(symbol, interval)
        if locked_settings is not None:
            return locked_settings
        candidate = optimization or self._optimization_result(symbol, interval)
        if candidate is not None and _optimization_result_interval(candidate) == interval:
            return candidate.best_backtest.settings
        return self.settings.strategy

    def _materialize_cached_backtest(
        self,
        symbol: str,
        interval: str,
        history: Optional[pd.DataFrame],
        seed_backtest: Optional[BacktestResult],
        strategy_settings: StrategySettings,
        *,
        fast_only: bool = False,
    ) -> Optional[BacktestResult]:
        """Materialize a cached backtest for immediate display.

        When fast_only=True (used on the main/UI thread), skip full run_backtest()
        and return seed_backtest instead — the background worker will compute the
        authoritative result shortly after.
        """
        if history is None or history.empty:
            return seed_backtest
        if seed_backtest is not None and seed_backtest.settings == strategy_settings:
            if _backtest_matches_history(seed_backtest, history):
                return seed_backtest
            if _history_can_resume_backtest(seed_backtest, history):
                return resume_backtest(
                    history,
                    previous_result=seed_backtest,
                    settings=strategy_settings,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms"),
                )
        if fast_only:
            # Full run_backtest() would block the UI thread — defer to worker.
            return seed_backtest
        return run_backtest(
            history,
            settings=strategy_settings,
            fee_rate=self.settings.fee_rate,
            backtest_start_time=pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms"),
        )

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
        self.resolved_auto_trade_backtest_cache = {
            key: value
            for key, value in self.resolved_auto_trade_backtest_cache.items()
            if key[0] in keep_symbols
        }
        self.resolved_auto_trade_backtest_meta = {
            key: value
            for key, value in self.resolved_auto_trade_backtest_meta.items()
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
        self.favorable_refresh_pending = {
            key: value
            for key, value in self.favorable_refresh_pending.items()
            if key[0] in keep_symbols
        }
        self.favorable_zone_cache = {
            key: value
            for key, value in self.favorable_zone_cache.items()
            if key[0] in keep_symbols
        }
        self.optimized_actionable_signal_cache = {
            key: value
            for key, value in self.optimized_actionable_signal_cache.items()
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
        self.auto_close_last_attempt_at.pop(symbol, None)
        self._stop_auto_close_monitor(symbol)
        self._refresh_auto_close_retry_timer()

    def _stop_all_auto_close_monitors(self) -> None:
        symbols = set(self.auto_close_history_workers) | set(self.auto_close_signal_workers) | set(self.auto_close_stream_workers)
        for symbol in list(symbols):
            self._stop_auto_close_monitor(symbol)
        self.auto_close_monitor_histories.clear()
        self.auto_close_monitor_intervals.clear()
        self.auto_close_signal_pending.clear()
        self._refresh_auto_close_retry_timer()

    def _start_auto_close_history_worker(self, symbol: str, interval: str) -> None:
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
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
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
            return
        existing = self.auto_close_stream_workers.get(symbol)
        if existing is not None and self.auto_close_monitor_intervals.get(symbol) == interval:
            return
        self._stop_auto_close_stream_worker(symbol)
        worker = KlineStreamWorker(
            symbol,
            interval,
            seed_history=self._recent_stream_seed_history(symbol, interval),
        )
        worker.kline.connect(self._on_auto_close_kline)
        self.auto_close_stream_workers[symbol] = worker
        self._track_mapped_thread(worker, self.auto_close_stream_workers, symbol)
        worker.start()

    def _schedule_auto_close_signal(self, symbol: str) -> None:
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
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
            self._active_backtest_settings(
                symbol,
                desired_interval,
                prefer_locked_position_settings=True,
            ),
            self.backtest_cache.get(cache_key),
        )
        worker.completed.connect(self._on_auto_close_signal_completed)
        worker.failed.connect(lambda message, symbol=symbol: self._on_auto_close_signal_failed(symbol, message))
        self.auto_close_signal_workers[symbol] = worker
        self._track_mapped_thread(worker, self.auto_close_signal_workers, symbol)
        worker.start()

    def _toggle_auto_close_for_symbol(self, symbol: str, enabled: bool) -> None:
        if self.auto_trade_enabled:
            self.log("자동매매 활성화 중에는 자동청산 버튼을 사용할 수 없습니다.")
            self.update_positions_table()
            return
        button = self.sender()
        if isinstance(button, QPushButton):
            self._set_auto_close_button_state(button, enabled)
        if enabled:
            self.auto_close_enabled_symbols.add(symbol)
            self.log(f"{symbol} 자동청산 활성화")
            self._refresh_auto_close_monitors()
            if symbol == self.current_symbol and self.current_backtest is not None:
                self._evaluate_backtest_auto_close(symbol, self.current_backtest)
            self._refresh_auto_close_retry_timer()
            return
        self.log(f"{symbol} 자동청산 비활성화")
        self._clear_auto_close_symbol(symbol)
        self.update_positions_table()

    def _refresh_auto_close_monitors(self) -> None:
        if self._trade_engine_alive():
            self._stop_all_auto_close_monitors()
            self._sync_trade_engine_state()
            return
        open_symbols = {position.symbol for position in self.open_positions}
        for symbol in list(self.auto_close_enabled_symbols):
            if symbol not in open_symbols:
                self._clear_auto_close_symbol(symbol)
        managed_symbols = self._auto_close_managed_symbols()

        for symbol in list(self.auto_close_history_workers):
            if symbol not in managed_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)
        for symbol in list(self.auto_close_signal_workers):
            if symbol not in managed_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)
        for symbol in list(self.auto_close_stream_workers):
            if symbol not in managed_symbols or symbol == self.current_symbol:
                self._stop_auto_close_monitor(symbol)

        for symbol in list(managed_symbols):
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
        self._refresh_auto_close_retry_timer()

    def _on_auto_close_history_completed(self, payload: object) -> None:
        result = dict(payload)
        symbol = str(result["symbol"])
        interval = str(result["interval"])
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
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
        if self._is_auto_close_active_for_symbol(symbol):
            self.log(message)

    def _on_auto_close_kline(self, payload: object) -> None:
        bar = dict(payload)
        symbol = str(bar.get("symbol", ""))
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
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
        if not self._is_auto_close_active_for_symbol(symbol) or symbol == self.current_symbol:
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
        if self._is_auto_close_active_for_symbol(symbol):
            self.log(message)

    def _evaluate_backtest_auto_close(self, symbol: str, backtest: BacktestResult) -> None:
        if self._trade_engine_alive():
            return
        if not self._is_auto_close_active_for_symbol(symbol):
            return
        if backtest.indicators.empty:
            return
        self._maybe_trigger_auto_close(symbol, _latest_backtest_exit_event(backtest))

    def _maybe_trigger_auto_close(
        self,
        symbol: str,
        exit_event: Optional[Dict[str, object]],
    ) -> None:
        if self._trade_engine_alive():
            return
        if not self._is_auto_close_active_for_symbol(symbol):
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
        if not self._auto_close_retry_allowed(symbol, normalized_bar_time):
            return
        if self._is_order_pending():
            if symbol not in self.auto_close_queued_orders:
                self.log(f"{symbol} 자동청산 대기: 기존 주문 처리 중")
            self.auto_close_queued_orders[symbol] = (reason, normalized_bar_time)
            return
        if self._submit_close_position(symbol, auto_close_reason=reason):
            self.auto_close_order_pending.add(symbol)
            self.auto_close_queued_orders.pop(symbol, None)
            self._record_auto_close_attempt(symbol, normalized_bar_time)

    def _flush_queued_auto_close_orders(self) -> None:
        if self._trade_engine_alive():
            return
        if self._is_order_pending():
            return
        for symbol, (reason, bar_time) in list(self.auto_close_queued_orders.items()):
            if not self._is_auto_close_active_for_symbol(symbol):
                self.auto_close_queued_orders.pop(symbol, None)
                continue
            position = self._find_open_position(symbol)
            if position is None:
                self._clear_auto_close_symbol(symbol)
                continue
            normalized_bar_time = pd.Timestamp(bar_time) if bar_time is not None else None
            if not self._auto_close_retry_allowed(symbol, normalized_bar_time):
                continue
            if self._submit_close_position(symbol, auto_close_reason=reason):
                self.auto_close_order_pending.add(symbol)
                self.auto_close_queued_orders.pop(symbol, None)
                self._record_auto_close_attempt(symbol, normalized_bar_time)
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
            worker.wait(100)

    def _stop_chart_history_page_worker(self) -> None:
        worker = self.chart_history_page_worker
        self.chart_history_page_worker = None
        self.chart_history_load_pending = False
        self.chart_history_load_requested = False
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(100)

    def _stop_live_backtest_worker(self) -> None:
        worker = self.live_backtest_worker
        self.live_backtest_worker = None
        self.live_recalc_pending = False
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(100)

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
        self.order_worker_is_auto_trade = False
        self.engine_order_pending = False
        self.pending_open_order_interval = None
        self.auto_trade_entry_pending_symbol = None
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
        self._clear_current_live_preview()
        self.current_lightweight_preview_markers = []
        self.current_lightweight_fast_entry_markers = []
        self.current_lightweight_fast_exit_markers = []
        if self.live_update_timer.isActive():
            self.live_update_timer.stop()
        if worker is not None:
            worker.stop()
            worker.wait(100)
        if self.chart is not None:
            self._render_lightweight_markers()

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

    def _recent_stream_seed_history(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        if interval != "2m":
            return None
        try:
            return self.public_client.historical_ohlcv_recent(symbol, "1m", bars=2)
        except Exception:
            return None

    def _on_position_price_kline(self, payload: object) -> None:
        bar = dict(payload)
        symbol = str(bar.get("symbol", ""))
        if not symbol or symbol == self.current_symbol:
            return
        self._apply_live_position_price(symbol, float(bar["close"]))

    def _start_live_stream(self, symbol: str) -> None:
        self._stop_live_stream()
        interval = self.current_interval or self.settings.kline_interval
        worker = KlineStreamWorker(
            symbol,
            interval,
            seed_history=self._recent_stream_seed_history(symbol, interval),
        )
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
        # If there's a pending closed bar waiting on the timer, flush it
        # first so its confirmed backtest is not lost when the next
        # (unclosed) bar overwrites live_pending_bar.
        pending = self.live_pending_bar
        if pending is not None and bool(pending.get("closed")):
            self.live_update_timer.stop()
            self._flush_live_update()
        self.live_pending_bar = bar
        if not bool(bar.get("closed")):
            self._flush_live_update()
            return
        self.live_update_timer.setInterval(LIVE_RENDER_INTERVAL_MS)
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
        self._apply_live_position_price(symbol, float(bar["close"]))
        if not bool(bar.get("closed")):
            self.current_live_preview_bar = dict(bar)
            self._sync_current_chart_snapshot(symbol, self.current_interval, preview_bar=self.current_live_preview_bar)
            self._apply_live_lightweight_bar(symbol, bar)
            return
        self._clear_current_live_preview()
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
        self._mark_history_refreshed(symbol, time.time(), self.current_interval)
        confirmed_history = self._get_history_frame(symbol, self.current_interval)
        confirmed_chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        applied = self._apply_closed_bar_confirmed_backtest(symbol, confirmed_history, confirmed_chart_history)
        if not applied:
            self._schedule_live_backtest(symbol)
        self._evaluate_closed_bar_auto_close(symbol, confirmed_history)

    def _apply_live_lightweight_bar(self, symbol: str, bar: Dict[str, object]) -> None:
        if symbol != self.current_symbol or self.chart is None:
            return
        series = pd.Series(
            {
                "time": pd.Timestamp(bar["time"]),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
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
        active_settings = self._active_backtest_settings(symbol, self.current_interval)
        if (
            self.current_backtest is not None
            and self.current_backtest.settings == active_settings
            and _backtest_matches_history(self.current_backtest, history)
        ):
            return
        worker = LiveBacktestWorker(
            self.settings,
            symbol,
            history,
            chart_history,
            active_settings,
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
        current_history = self._get_history_frame(symbol, self.current_interval)
        if _history_frame_signature(result.get("history")) != _history_frame_signature(current_history):
            self.live_recalc_pending = False
            self._schedule_live_backtest(symbol)
            return
        perf = dict(result.get("perf") or {})
        self._log_perf_ms(f"{symbol} live worker backtest", float(perf.get("worker_backtest_ms", 0.0) or 0.0))
        previous_backtest = self.current_backtest
        previous_chart_indicators = self.current_chart_indicators
        history = result["history"]
        chart_history = result["chart_history"]
        existing_history = self._get_history_frame(symbol, self.current_interval)
        existing_chart_history = self._get_chart_history_frame(symbol, self.current_interval)
        merged_history = _merge_ohlcv_frames(history, existing_history)
        merged_chart_history = _merge_ohlcv_frames(chart_history, existing_chart_history, max_rows=CHART_HISTORY_BAR_LIMIT)
        has_newer_live_bar = False
        worker_chart_last_time = _frame_last_time(chart_history)
        merged_chart_last_time = _frame_last_time(merged_chart_history)
        if worker_chart_last_time is not None and merged_chart_last_time is not None:
            has_newer_live_bar = merged_chart_last_time > worker_chart_last_time
        preview_bar = self._current_live_preview_for(symbol, self.current_interval)
        preview_time = pd.Timestamp(preview_bar["time"]) if preview_bar is not None else None
        if preview_time is not None and (worker_chart_last_time is None or preview_time > worker_chart_last_time):
            has_newer_live_bar = True
        self._set_history_frame(symbol, merged_history, self.current_interval)
        self._set_chart_history_frame(symbol, merged_chart_history, self.current_interval)
        self.current_backtest = result["backtest"]
        self.current_chart_indicators = result["chart_indicators"]
        self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=self.current_backtest,
            chart_indicators=self.current_chart_indicators,
        )
        self.current_lightweight_fast_entry_markers = []
        self.current_lightweight_fast_exit_markers = []
        cache_key = self._symbol_interval_key(symbol, self.current_interval)
        self.backtest_cache[cache_key] = self.current_backtest
        self.chart_indicator_cache[cache_key] = self.current_chart_indicators
        self._prune_caches()
        apply_started_at = time.perf_counter()
        applied_incrementally = False
        if not has_newer_live_bar:
            applied_incrementally = self._apply_incremental_lightweight_backtest(
                symbol,
                previous_backtest,
                previous_chart_indicators,
                self.current_backtest,
                self.current_chart_indicators,
                chart_history,
                skip_candle_update=False,
            )
        if not applied_incrementally:
            self.render_chart(symbol, self.current_backtest, reset_view=False, chart_indicators=self.current_chart_indicators)
        if has_newer_live_bar:
            self._reapply_live_preview_bar(symbol)
        self._log_perf(f"{symbol} live chart apply", apply_started_at)
        self.update_summary(symbol, self.current_backtest, self._optimization_result(symbol, self.current_interval))
        if self.live_backtest_started_at > 0:
            self._log_perf(f"{symbol} live backtest", self.live_backtest_started_at)
            self.live_backtest_started_at = 0.0
        self._evaluate_backtest_auto_close(symbol, self.current_backtest)
        self._evaluate_closed_bar_auto_close(
            symbol, self._get_history_frame(symbol, self.current_interval)
        )
        if self.auto_trade_enabled and self.auto_trade_entry_pending_symbol is None:
            confirmed_history = self._get_history_frame(symbol, self.current_interval)
            confirmed_bar_time = (
                pd.Timestamp(confirmed_history["time"].iloc[-1])
                if confirmed_history is not None and not confirmed_history.empty
                else None
            )
            if confirmed_bar_time is not None:
                trigger_time = fresh_entry_trigger_time(self.current_backtest, confirmed_bar_time, self.current_interval)
                QTimer.singleShot(
                    0,
                    lambda: self._run_auto_trade_cycle(
                        trigger_symbol=symbol,
                        trigger_interval=self.current_interval,
                        trigger_bar_time=trigger_time if trigger_time is not None else confirmed_bar_time,
                    ),
                )
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
        if hasattr(self, "chart_header_tf_label"):
            self.chart_header_tf_label.setText(f"TF {interval}" if interval else "TF -")

    def _update_entry_price_overlay(self) -> None:
        position = self.current_position_snapshot
        if (
            self.chart is None
            or position is None
            or position.symbol != self.current_symbol
        ):
            self._clear_entry_price_overlay()
            return
        frame = self._display_history_frame(position.symbol, self.current_interval, chart=True)
        if frame is None or frame.empty:
            frame = self._display_history_frame(position.symbol, self.current_interval)
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
        frame = self._display_history_frame(symbol, self.current_interval, chart=True)
        if frame is None or frame.empty:
            frame = self._display_history_frame(symbol, self.current_interval)
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
        notional_usdt = _position_notional_usdt(position)
        entry_text = f"{position.entry_price:.8f}".rstrip("0").rstrip(".")
        upnl_value = float(position.unrealized_pnl)
        return_pct = _position_return_pct(position)
        return (
            [
                self._position_symbol_text(position.symbol),
                side,
                f"{position.leverage}x",
                f"{notional_usdt:.2f}",
                entry_text,
                f"{upnl_value:.2f}",
                f"{return_pct:+.2f}%",
            ],
            upnl_value,
            return_pct,
        )

    def _pnl_color(self, value: float) -> str:
        if value > 0:
            return "#1f8f47"
        if value < 0:
            return "#f31260"
        return "#1f2937"

    def _build_position_metric_widget(self, text: str, value: float) -> QLabel:
        label = QLabel(text)
        label.setProperty("metricDefaultColor", self._pnl_color(value))
        label.setStyleSheet(
            f"font-weight: 700; color: {self._pnl_color(value)}; padding-left: 4px; padding-right: 4px; background: transparent;"
        )
        return label

    def _refresh_position_metric_selection_colors(self) -> None:
        if not hasattr(self, "positions_table"):
            return
        selected_rows = {index.row() for index in self.positions_table.selectionModel().selectedRows()}
        highlight_active = self.positions_table.hasFocus()
        for row in range(self.positions_table.rowCount()):
            selected = highlight_active and row in selected_rows
            for col in (5, 6):
                widget = self.positions_table.cellWidget(row, col)
                if not isinstance(widget, QLabel):
                    continue
                default_color = str(widget.property("metricDefaultColor") or "#1f2937")
                color = "#ffffff" if selected else default_color
                widget.setStyleSheet(
                    f"font-weight: 700; color: {color}; padding-left: 4px; padding-right: 4px; background: transparent;"
                )

    def _position_status_html(self, position: PositionSnapshot) -> str:
        side = "LONG" if position.amount > 0 else "SHORT"
        notional_usdt = _position_notional_usdt(position)
        upnl_value = float(position.unrealized_pnl)
        return_pct = _position_return_pct(position)
        upnl_color = self._pnl_color(upnl_value)
        return_color = self._pnl_color(return_pct)
        return (
            f"<span style='color:#111827;'>포지션: {self._position_symbol_text(position.symbol)} {side} {notional_usdt:.2f} USDT @ {position.entry_price:.6f} | </span>"
            f"<span style='font-weight:700; color:{upnl_color};'>UPnL {upnl_value:.2f}</span> | "
            f"<span style='font-weight:700; color:{return_color};'>수익률 {return_pct:+.2f}%</span>"
        )

    def _populate_position_row(self, row: int, position: PositionSnapshot) -> None:
        values, upnl_value, return_pct = self._position_display_values(position)
        for col, value in enumerate(values):
            if col in (5, 6):
                placeholder = QTableWidgetItem("")
                placeholder.setData(USER_ROLE, position.symbol)
                self.positions_table.setItem(row, col, placeholder)
                metric_value = upnl_value if col == 5 else return_pct
                widget = self._build_position_metric_widget(value, metric_value)
                self.positions_table.setCellWidget(row, col, widget)
                continue
            item = QTableWidgetItem(value)
            item.setData(USER_ROLE, position.symbol)
            self.positions_table.setItem(row, col, item)

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
                self._refresh_position_metric_selection_colors()
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
        previous_suppress = self.suppress_positions_selection_load
        self.suppress_positions_selection_load = True
        self.positions_table.blockSignals(True)
        try:
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
                auto_close_active = position.symbol in self.auto_close_enabled_symbols or self.auto_trade_enabled
                self._set_auto_close_button_state(
                    auto_button,
                    auto_close_active,
                    forced_by_auto_trade=bool(self.auto_trade_enabled),
                )
                auto_button.setEnabled(not self.auto_trade_enabled)
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

                self.positions_table.setCellWidget(row, 7, action_widget)
                self.position_close_buttons.append(button)
            self.positions_table.resizeColumnToContents(0)
            self._set_position_close_buttons_enabled(not self._is_order_pending())
            self._refresh_position_metric_selection_colors()
        finally:
            self.positions_table.blockSignals(False)
            self.suppress_positions_selection_load = previous_suppress

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
                item.setData(USER_ROLE, (candidate.symbol, CANDIDATE_DEFAULT_INTERVAL))
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

    def _optimized_table_price_map(self, *, log_failures: bool = False) -> Dict[str, float]:
        symbols = {result.symbol for result in self._ordered_optimized_results()}
        if not symbols:
            return {}
        try:
            ticker_map = self.public_client.ticker_24h()
        except Exception as exc:
            if log_failures:
                self.log(f"최적화 종목 현재가 조회 실패: {exc}")
            return {}
        prices: Dict[str, float] = {}
        for symbol in symbols:
            ticker = ticker_map.get(symbol)
            if not ticker:
                continue
            current_price = float(ticker.get("lastPrice", 0.0) or 0.0)
            if current_price > 0:
                prices[symbol] = current_price
        return prices

    def _optimized_result_favorable_zone(self, result: OptimizationResult, current_price: Optional[float]) -> Optional[int]:
        symbol = str(result.symbol)
        interval = _optimization_result_interval(result)
        key = self._symbol_interval_key(symbol, interval)
        actionable_side, actionable_zone, actionable_kind = self._actionable_signal(symbol, interval)
        if actionable_kind == "favorable" and actionable_side and actionable_zone in {1, 2, 3}:
            self.favorable_zone_cache[key] = int(actionable_zone)
            return int(actionable_zone)
        if current_price is None or current_price <= 0:
            return self.favorable_zone_cache.get(key) if key in self.favorable_refresh_pending else None
        open_position = self._find_open_position(symbol)
        latest_backtest = self._best_available_auto_trade_backtest_for_display(result)
        if latest_backtest is None:
            return self.favorable_zone_cache.get(key) if key in self.favorable_refresh_pending else None
        filled_fraction = self.auto_trade_filled_fraction_by_symbol.get(
            symbol,
            _inferred_auto_trade_fraction(latest_backtest, open_position) if open_position is not None else 0.0,
        )
        favorable_zone = resolve_favorable_auto_trade_zone(
            latest_backtest,
            float(current_price),
            open_position,
            filled_fraction,
        )
        self.favorable_zone_cache[key] = favorable_zone
        return favorable_zone

    def _optimized_result_has_favorable_entry(self, result: OptimizationResult, current_price: Optional[float]) -> bool:
        return self._optimized_result_favorable_zone(result, current_price) is not None

    def _engine_entry_signal(
        self,
        symbol: str,
        interval: Optional[str] = None,
        *,
        mode: str = "preview",
    ) -> Tuple[str, int]:
        signal_key = (symbol, interval or self.current_interval or self.settings.kline_interval, mode)
        side, zone = self.last_engine_entry_signal_by_key.get(signal_key, ("", 0))
        normalized_side = str(side or "").lower()
        normalized_zone = int(zone or 0)
        if normalized_side not in {"long", "short"} or normalized_zone not in {1, 2, 3}:
            return "", 0
        return normalized_side, normalized_zone

    def _normalize_actionable_signal(
        self,
        side: str,
        zone: int,
        kind: str,
    ) -> Tuple[str, int, str]:
        normalized_side = str(side or "").lower()
        normalized_zone = int(zone or 0)
        normalized_kind = str(kind or "").lower()
        if normalized_side not in {"long", "short"}:
            return "", 0, ""
        if normalized_zone not in {1, 2, 3}:
            return "", 0, ""
        if normalized_kind not in {"confirmed", "favorable"}:
            return "", 0, ""
        return normalized_side, normalized_zone, normalized_kind

    def _actionable_signal_from_evaluation(
        self,
        evaluation: AutoTradeEvaluationResult,
    ) -> Tuple[str, int, str]:
        return self._normalize_actionable_signal(
            str(evaluation.signal_side or ""),
            int(evaluation.signal_zone or 0),
            str(evaluation.signal_kind or ""),
        )

    def _set_actionable_signal_cache(
        self,
        cache: Dict[Tuple[str, str], Tuple[str, int, str]],
        symbol: str,
        interval: Optional[str],
        side: str,
        zone: int,
        kind: str,
    ) -> Tuple[str, int, str]:
        key = self._symbol_interval_key(symbol, interval)
        normalized = self._normalize_actionable_signal(side, zone, kind)
        if normalized[0]:
            cache[key] = normalized
        else:
            cache.pop(key, None)
        return normalized

    def _set_engine_actionable_signal(
        self,
        symbol: str,
        interval: Optional[str],
        side: str,
        zone: int,
        kind: str,
    ) -> Tuple[str, int, str]:
        return self._set_actionable_signal_cache(
            self.last_engine_actionable_signal_by_key,
            symbol,
            interval,
            side,
            zone,
            kind,
        )

    def _set_local_actionable_signal(
        self,
        symbol: str,
        interval: Optional[str],
        side: str,
        zone: int,
        kind: str,
    ) -> Tuple[str, int, str]:
        return self._set_actionable_signal_cache(
            self.last_local_actionable_signal_by_key,
            symbol,
            interval,
            side,
            zone,
            kind,
        )

    def _clear_actionable_signal(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> bool:
        caches = (self.last_engine_actionable_signal_by_key, self.last_local_actionable_signal_by_key)
        changed = False
        if symbol is None and interval is None:
            for cache in caches:
                if cache:
                    cache.clear()
                    changed = True
            return changed
        normalized_symbol = str(symbol or "").upper()
        normalized_interval = str(interval or "")
        for cache in caches:
            for key in list(cache):
                if normalized_symbol and key[0] != normalized_symbol:
                    continue
                if normalized_interval and key[1] != normalized_interval:
                    continue
                cache.pop(key, None)
                changed = True
        return changed

    def _actionable_signal(
        self,
        symbol: str,
        interval: Optional[str] = None,
    ) -> Tuple[str, int, str]:
        key = self._symbol_interval_key(symbol, interval)
        cache = (
            self.last_engine_actionable_signal_by_key
            if self._trade_engine_alive()
            else self.last_local_actionable_signal_by_key
        )
        side, zone, kind = cache.get(key, ("", 0, ""))
        return self._normalize_actionable_signal(side, zone, kind)

    def _evaluated_actionable_signal(
        self,
        result: OptimizationResult,
        current_price: Optional[float],
    ) -> Optional[Tuple[str, int, str]]:
        interval = _optimization_result_interval(result)
        latest_backtest = self._best_available_auto_trade_backtest_for_display(result)
        if latest_backtest is None:
            return None
        trigger_bar_time: Optional[pd.Timestamp] = None
        history = self._get_history_frame(result.symbol, interval)
        if history is not None and not history.empty and "time" in history.columns:
            trigger_bar_time = pd.Timestamp(history["time"].iloc[-1]).tz_localize(None)
        elif not latest_backtest.indicators.empty and "time" in latest_backtest.indicators.columns:
            trigger_bar_time = pd.Timestamp(latest_backtest.indicators["time"].iloc[-1]).tz_localize(None)
        open_position = self._find_open_position(result.symbol)
        evaluation = evaluate_auto_trade_candidate(
            symbol=str(result.symbol),
            interval=interval,
            score=float(result.score),
            strategy_settings=result.best_backtest.settings,
            latest_backtest=latest_backtest,
            current_price=None if current_price is None or current_price <= 0 else float(current_price),
            open_position=open_position,
            remembered_interval=self.settings.position_intervals.get(result.symbol),
            filled_fraction=self.auto_trade_filled_fraction_by_symbol.get(
                result.symbol,
                _inferred_auto_trade_fraction(latest_backtest, open_position)
                if open_position is not None
                else 0.0,
            ),
            remembered_cursor_entry_time=self.auto_trade_cursor_entry_time_by_symbol.get(result.symbol),
            allow_favorable_price_entries=bool(self.settings.auto_trade_use_favorable_price),
            trigger_symbol=str(result.symbol),
            trigger_interval=interval,
            trigger_bar_time=trigger_bar_time,
        )
        actionable_signal = self._actionable_signal_from_evaluation(evaluation)
        if not actionable_signal[0]:
            return None
        return actionable_signal

    def _optimized_result_actionable_signal(
        self,
        result: OptimizationResult,
        current_price: Optional[float] = None,
    ) -> Optional[Tuple[str, int, str]]:
        interval = _optimization_result_interval(result)
        key = self._symbol_interval_key(result.symbol, interval)
        side, zone, kind = self._actionable_signal(result.symbol, interval)
        if side and zone > 0 and kind:
            return self._set_actionable_signal_cache(
                self.optimized_actionable_signal_cache,
                result.symbol,
                interval,
                side,
                zone,
                kind,
            )
        evaluated_signal = self._evaluated_actionable_signal(result, current_price)
        if evaluated_signal is not None:
            return self._set_actionable_signal_cache(
                self.optimized_actionable_signal_cache,
                result.symbol,
                interval,
                evaluated_signal[0],
                int(evaluated_signal[1]),
                evaluated_signal[2],
            )
        cached_signal = self.optimized_actionable_signal_cache.get(key)
        if key in self.favorable_refresh_pending or current_price is None or current_price <= 0:
            if cached_signal is None:
                return None
            side, zone, kind = self._normalize_actionable_signal(
                str(cached_signal[0]),
                int(cached_signal[1]),
                str(cached_signal[2]),
            )
            if side and zone > 0 and kind:
                return side, zone, kind
            return None
        self.optimized_actionable_signal_cache.pop(key, None)
        return None

    def _optimized_result_preview_signal(self, result: OptimizationResult) -> Optional[Tuple[str, int]]:
        interval = _optimization_result_interval(result)
        side, zone = self._engine_entry_signal(result.symbol, interval, mode="preview")
        if not side or zone <= 0:
            return None
        return side, zone

    def _update_optimized_status_labels(self, favorable_count: int, entry_count: int) -> None:
        favorable_label = getattr(self, "optimized_favorable_label", None)
        entry_label = getattr(self, "optimized_entry_label", None)
        if favorable_label is not None:
            if favorable_count:
                favorable_label.setText("유리" if favorable_count == 1 else f"유리 {favorable_count}")
                favorable_label.show()
            else:
                favorable_label.hide()
        if entry_label is not None:
            if entry_count:
                entry_label.setText("진입" if entry_count == 1 else f"진입 {entry_count}")
                entry_label.show()
            else:
                entry_label.hide()
        if favorable_count or entry_count:
            self._reposition_favorable_label()

    def _log_telegram_failure(self, message: str) -> None:
        if hasattr(self, "log_box"):
            self.log(message)

    def _notify_telegram(self, text: str, *, key: str, cooldown_seconds: float = 5.0) -> None:
        if not text.strip():
            return
        self.telegram_notifier.send(text.strip(), key=key, cooldown_seconds=cooldown_seconds)

    def _notify_telegram_auto_trade_entry(self, event: EngineOrderCompletedEvent) -> None:
        interval = str(event.interval or self.settings.kline_interval)
        fraction_pct = float(event.fraction or 0.0) * 100.0
        lines = [
            "[자동매매 진입]",
            f"종목: {event.symbol}",
            f"차트TF: {interval}",
        ]
        if fraction_pct > 0:
            lines.append(f"비중: {fraction_pct:.1f}%")
        lines.append(str(event.message))
        self._notify_telegram(
            "\n".join(lines),
            key=f"auto-trade-entry:{event.symbol}:{interval}:{fraction_pct:.1f}:{event.message}",
            cooldown_seconds=3.0,
        )

    def _notify_telegram_auto_trade_close(self, event: EngineOrderCompletedEvent) -> None:
        interval = str(event.interval or self.settings.kline_interval)
        position = self._find_open_position(event.symbol)
        lines = [
            "[자동매매 청산]",
            f"종목: {event.symbol}",
            f"차트TF: {interval}",
        ]
        if position is not None:
            pnl_value = float(position.unrealized_pnl)
            return_pct = _position_return_pct(position)
            lines.append(f"PnL: {pnl_value:+.2f} USDT")
            lines.append(f"수익률: {return_pct:+.2f}%")
        lines.append(str(event.message))
        self._notify_telegram(
            "\n".join(lines),
            key=f"auto-trade-close:{event.symbol}:{interval}:{event.message}",
            cooldown_seconds=3.0,
        )

    def _update_favorable_telegram_alerts(
        self,
        favorable_rows: List[Tuple[str, str, Optional[float], Optional[int]]],
    ) -> None:
        self.telegram_favorable_symbols = {
            symbol for symbol, _interval, _price, _zone in favorable_rows
        }

    def _refresh_optimized_table_highlights(self) -> None:
        if not hasattr(self, "optimized_table"):
            return
        ordered = self._ordered_optimized_results()
        if self.optimized_table.rowCount() != len(ordered):
            self.update_optimized_table()
            return
        price_map = self._optimized_table_price_map(log_failures=False)
        favorable_row_brush = QColor(OPTIMIZED_TABLE_FAVORABLE_ROW_COLOR)
        signal_row_brush = QColor(OPTIMIZED_TABLE_SIGNAL_ROW_COLOR)
        default_row_brush = self.optimized_table.palette().base().color()
        favorable_count = 0
        entry_count = 0
        favorable_rows: List[Tuple[str, str, Optional[float], Optional[int]]] = []
        for row, result in enumerate(ordered):
            result_interval = _optimization_result_interval(result)
            current_price = price_map.get(result.symbol)
            favorable_zone = self._optimized_result_favorable_zone(result, current_price)
            actionable_signal = self._optimized_result_actionable_signal(result, current_price)
            favorable_entry = favorable_zone is not None
            entry_signal = actionable_signal is not None and actionable_signal[2] == "confirmed"
            if favorable_entry:
                favorable_count += 1
                favorable_rows.append(
                    (
                        result.symbol,
                        result_interval,
                        current_price,
                        int(favorable_zone) if favorable_zone is not None else None,
                    )
                )
            if entry_signal:
                entry_count += 1
            brush = favorable_row_brush if favorable_entry else signal_row_brush if entry_signal else default_row_brush
            for col in range(self.optimized_table.columnCount()):
                item = self.optimized_table.item(row, col)
                if item is not None:
                    item.setBackground(brush)
        self._update_favorable_telegram_alerts(favorable_rows)
        self._update_optimized_status_labels(favorable_count, entry_count)

    def update_optimized_table(self) -> None:
        self.optimized_table.setUpdatesEnabled(False)
        ordered = self._ordered_optimized_results()
        price_map = self._optimized_table_price_map(log_failures=True)
        favorable_row_brush = QColor(OPTIMIZED_TABLE_FAVORABLE_ROW_COLOR)
        signal_row_brush = QColor(OPTIMIZED_TABLE_SIGNAL_ROW_COLOR)
        self.optimized_table.setRowCount(len(ordered))
        favorable_count = 0
        entry_count = 0
        favorable_rows: List[Tuple[str, str, Optional[float], Optional[int]]] = []
        for row, result in enumerate(ordered):
            metrics = result.best_backtest.metrics
            result_interval = _optimization_result_interval(result)
            current_price = price_map.get(result.symbol)
            favorable_zone = self._optimized_result_favorable_zone(result, current_price)
            actionable_signal = self._optimized_result_actionable_signal(result, current_price)
            favorable_entry = favorable_zone is not None
            entry_signal = actionable_signal is not None and actionable_signal[2] == "confirmed"
            if favorable_entry:
                favorable_count += 1
                favorable_rows.append(
                    (
                        result.symbol,
                        result_interval,
                        current_price,
                        int(favorable_zone) if favorable_zone is not None else None,
                    )
                )
            if entry_signal:
                entry_count += 1
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
                if favorable_entry:
                    item.setBackground(favorable_row_brush)
                elif entry_signal:
                    item.setBackground(signal_row_brush)
                self.optimized_table.setItem(row, col, item)
        self.optimized_table.setUpdatesEnabled(True)
        self._update_favorable_telegram_alerts(favorable_rows)
        self._update_optimized_status_labels(favorable_count, entry_count)
        self._sync_trade_engine_state()

    def _schedule_optimized_table_refresh(self) -> None:
        if not self.optimized_table_timer.isActive():
            self.optimized_table_timer.start(OPTIMIZED_TABLE_REFRESH_MS)

    def _flush_optimized_table(self) -> None:
        self.update_optimized_table()

    def _apply_auto_refresh_interval(self, *, log_message: bool = False) -> None:
        self.auto_refresh_minutes = max(1, int(self.auto_refresh_minutes))
        self.auto_refresh_timer.setInterval(self.auto_refresh_minutes * 60 * 1000)
        if log_message:
            self.log(f"자동 갱신 주기 변경: {self.auto_refresh_minutes}분")

    def _init_auto_refresh(self) -> None:
        self.auto_refresh_timer.timeout.connect(self.run_auto_refresh)
        self._apply_auto_refresh_interval()
        self.auto_refresh_timer.start()
        self.log(f"자동 갱신 활성화: {self.auto_refresh_minutes}분마다 스캔+최적화")

    def _set_refresh_running(self, is_running: bool) -> None:
        self.scan_button.setEnabled(not is_running)
        self._refresh_auto_trade_button_state()

    def _set_backtest_progress_idle(self, text: str = "대기중") -> None:
        self.backtest_progress_phase = "idle"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_total_candidates = 0
        self.backtest_progress_prepared_candidates = 0
        self.backtest_progress_status_text = text
        self.backtest_progress_label.setText(text)
        self.backtest_progress_bar.setRange(0, 1)
        self.backtest_progress_bar.setValue(0)
        self.backtest_progress_bar.setFormat("%p%")

    def _set_backtest_progress_scanning(self) -> None:
        self.backtest_progress_phase = "scan"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_total_candidates = 0
        self.backtest_progress_prepared_candidates = 0
        self.backtest_progress_status_text = "후보 스캔중..."
        self.backtest_progress_label.setText("후보 스캔중...")
        self.backtest_progress_bar.setRange(0, 0)
        self.backtest_progress_bar.setFormat("스캔중")

    def _begin_backtest_progress(self, total_candidates: int) -> None:
        self.backtest_progress_phase = "optimize"
        self.backtest_progress_total_cases = 0
        self.backtest_progress_completed_cases = 0
        self.backtest_progress_total_candidates = max(0, int(total_candidates))
        self.backtest_progress_prepared_candidates = 0
        self.backtest_progress_status_text = "준비중"
        self.backtest_progress_bar.setRange(0, 1000)
        self.backtest_progress_bar.setValue(0)
        self.backtest_progress_bar.setFormat("%p%")
        self._refresh_backtest_progress_display()

    def _refresh_backtest_progress_display(self) -> None:
        if self.backtest_progress_phase != "optimize":
            return
        total_candidates = max(self.backtest_progress_total_candidates, 1)
        prep_ratio = min(1.0, self.backtest_progress_prepared_candidates / total_candidates)
        if self.backtest_progress_total_cases > 0:
            exec_ratio = min(1.0, self.backtest_progress_completed_cases / max(self.backtest_progress_total_cases, 1))
        else:
            exec_ratio = 0.0
        progress_ratio = (prep_ratio * 0.35) + (exec_ratio * 0.65)
        self.backtest_progress_bar.setRange(0, 1000)
        self.backtest_progress_bar.setValue(int(round(progress_ratio * 1000)))
        detail = f"{self.backtest_progress_completed_cases}/{max(self.backtest_progress_total_cases, self.backtest_progress_completed_cases, 0)}"
        if self.backtest_progress_total_cases <= 0:
            detail = f"{self.backtest_progress_prepared_candidates}/{self.backtest_progress_total_candidates or 0}"
        self.backtest_progress_label.setText(f"{self.backtest_progress_status_text} | {detail}")

    def _update_backtest_progress_phase(self, payload: object) -> None:
        result = dict(payload or {})
        if self.backtest_progress_phase != "optimize":
            return
        phase = str(result.get("phase") or "")
        candidate = str(result.get("candidate") or "")
        if phase == "optimize_start":
            self.backtest_progress_total_candidates = max(0, int(result.get("total_candidates", 0)))
            process_count = max(1, int(result.get("process_count", 1)))
            self.backtest_progress_status_text = f"프로세스 준비중 ({process_count}개)"
        elif phase == "history_loading":
            total_candidates = max(0, int(result.get("total_candidates", self.backtest_progress_total_candidates)))
            if total_candidates:
                self.backtest_progress_total_candidates = total_candidates
            index = max(1, int(result.get("index", 1)))
            self.backtest_progress_status_text = f"히스토리 로드중 {index}/{self.backtest_progress_total_candidates or total_candidates} | {candidate}"
        elif phase == "history_ready":
            total_candidates = max(0, int(result.get("total_candidates", self.backtest_progress_total_candidates)))
            if total_candidates:
                self.backtest_progress_total_candidates = total_candidates
            index = max(1, int(result.get("index", 1)))
            self.backtest_progress_prepared_candidates = max(self.backtest_progress_prepared_candidates, index)
            self.backtest_progress_status_text = f"히스토리 준비완료 {self.backtest_progress_prepared_candidates}/{self.backtest_progress_total_candidates or total_candidates} | {candidate}"
        elif phase == "case_running":
            active_jobs = max(1, int(result.get("active_jobs", 1)))
            process_count = max(1, int(result.get("process_count", 1)))
            self.backtest_progress_status_text = f"프로세스 준비중 {active_jobs}/{process_count} | {candidate}"
        self._refresh_backtest_progress_display()

    def _register_backtest_case_plan(self, cases: int) -> None:
        if self.backtest_progress_phase != "optimize":
            return
        if cases <= 0:
            return
        self.backtest_progress_total_cases += int(cases)
        if not self.backtest_progress_status_text.startswith("백테스트중"):
            self.backtest_progress_status_text = "백테스트중"
        self._refresh_backtest_progress_display()

    def _advance_backtest_progress(self, symbol: str, interval: str) -> None:
        if self.backtest_progress_phase != "optimize":
            return
        self.backtest_progress_completed_cases += 1
        self.backtest_progress_status_text = f"백테스트중 | {symbol} [{interval}]"
        self._refresh_backtest_progress_display()

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

    def run_scan_and_optimize(self, preserve_existing: bool = True) -> None:
        if self._is_refresh_running():
            return
        self.save_settings()
        self._stop_scan_worker()
        self._stop_optimize_worker()
        self._stop_load_worker()
        self._stop_live_backtest_worker()
        self.preserve_lists_during_refresh = bool(preserve_existing)
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.pending_backtest_cache = {}
        self.pending_chart_indicator_cache = {}
        phase_name = "후보 스캔 + 최적화 시작" if self.settings.enable_parameter_optimization else "후보 스캔 + 기본값 백테스트 시작"
        self.log(phase_name + (" (기존 목록 유지)" if preserve_existing else ""))
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
        self.pending_backtest_cache = {}
        self.pending_chart_indicator_cache = {}
        self.preserve_lists_during_refresh = False
        self._set_backtest_progress_idle("백테스트 대상 없음")
        self._set_refresh_running(False)
        self._refresh_auto_trade_button_state()

    def start_optimization(self, targets: List[CandidateSymbol]) -> None:
        if self.optimize_worker and self.optimize_worker.isRunning():
            return
        if not targets:
            self._set_backtest_progress_idle("백테스트 대상 없음")
            self._set_refresh_running(False)
            return
        phase_name = "최적화" if self.settings.enable_parameter_optimization else "기본값 백테스트"
        self.log(f"{phase_name} 시작: {len(targets)}개 종목")
        self._begin_backtest_progress(len(targets))
        self.optimize_worker = OptimizeWorker(self.settings, targets)
        self._track_thread(self.optimize_worker, "optimize_worker")
        self.optimize_worker.progress.connect(self.log)
        self.optimize_worker.phase_update.connect(self._update_backtest_progress_phase)
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
        interval = _optimization_result_interval(optimization)
        self._advance_backtest_progress(candidate.symbol, interval)
        cache_key = self._symbol_interval_key(candidate.symbol, interval)
        if self.preserve_lists_during_refresh:
            self.pending_optimized_results[cache_key] = optimization
            self._set_pending_history_frame(candidate.symbol, history, interval)
            self.pending_backtest_cache[cache_key] = optimization.best_backtest
            self.pending_chart_indicator_cache[cache_key] = _chart_indicators_from_backtest(optimization.best_backtest)
            return
        self.backtest_cache[cache_key] = optimization.best_backtest
        self.chart_indicator_cache[cache_key] = _chart_indicators_from_backtest(optimization.best_backtest)
        self.optimized_results[cache_key] = optimization
        self._set_history_frame(candidate.symbol, history, interval)
        self._mark_history_refreshed(candidate.symbol, time.time(), interval)
        self._prune_caches()
        self._schedule_optimized_table_refresh()
        if candidate.symbol in self.auto_close_enabled_symbols:
            self._refresh_auto_close_monitors()
        self._refresh_auto_trade_button_state()

    def on_optimization_completed(self) -> None:
        preserved_refresh = self.preserve_lists_during_refresh
        if preserved_refresh:
            if self.pending_candidates:
                self.candidates = list(self.pending_candidates)
                self.update_candidate_table()
            if self.pending_optimized_results:
                self.optimized_results = dict(self.pending_optimized_results)
                self.backtest_cache.update(self.pending_backtest_cache)
                self.chart_indicator_cache.update(self.pending_chart_indicator_cache)
                self._flush_optimized_table()
            self.history_cache.update(self.pending_history_cache)
            refreshed_at = time.time()
            for cache_key in self.pending_history_cache:
                self.history_refresh_times[cache_key] = refreshed_at
            self.pending_candidates = []
            self.pending_optimized_results = {}
            self.pending_history_cache = {}
            self.pending_backtest_cache = {}
            self.pending_chart_indicator_cache = {}
            self.preserve_lists_during_refresh = False
        self._prune_caches()
        self.log(f"최적화 완료: {len(self.optimized_results)}개 케이스")
        if not preserved_refresh:
            self._flush_optimized_table()
        self._refresh_auto_close_monitors()
        self._finish_backtest_progress()
        self._set_refresh_running(False)
        self._refresh_auto_trade_button_state()
        self._activate_requested_auto_trade_if_ready()
        if self.auto_trade_enabled:
            QTimer.singleShot(0, self._run_auto_trade_cycle)

    def on_worker_failed(self, message: str) -> None:
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.pending_backtest_cache = {}
        self.pending_chart_indicator_cache = {}
        self.preserve_lists_during_refresh = False
        log_runtime_event("Worker Failure", message, open_notepad=False)
        self.log(message)
        self._set_backtest_progress_idle("백테스트 실패")
        self._set_refresh_running(False)
        self._refresh_auto_trade_button_state()
        self.show_error(message)

    def run_auto_refresh(self) -> None:
        if self._is_refresh_running():
            self.log(f"자동 {self.auto_refresh_minutes}분 갱신 시점이지만 이전 작업이 아직 실행 중이라 건너뜁니다.")
            return
        self.log(f"자동 {self.auto_refresh_minutes}분 갱신 시작")
        self.run_scan_and_optimize(preserve_existing=True)

    def selected_candidate_symbols(self) -> List[str]:
        selected = self.candidate_table.selectedItems()
        if not selected:
            return []
        row = selected[0].row()
        symbol_item = self.candidate_table.item(row, 0)
        return [symbol_item.text()] if symbol_item else []

    def _table_current_row(self, table: QTableWidget) -> int:
        selected = table.selectedItems()
        if selected:
            return selected[0].row()
        current_row = table.currentRow()
        return current_row if current_row >= 0 else -1

    def _move_symbol_table_selection(self, table: QTableWidget, direction: int) -> bool:
        row_count = table.rowCount()
        if row_count <= 0:
            return False
        current_row = self._table_current_row(table)
        if current_row < 0:
            next_row = 0 if direction >= 0 else row_count - 1
        else:
            next_row = max(0, min(row_count - 1, current_row + direction))
        if next_row == current_row and current_row >= 0:
            return True
        table.selectRow(next_row)
        table.setCurrentCell(next_row, 0)
        return True

    def _request_symbol_load(
        self,
        symbol: str,
        interval: Optional[str] = None,
        *,
        prefer_locked_position_settings: bool = False,
    ) -> None:
        if not symbol:
            return
        target_interval = interval if interval in APP_INTERVAL_OPTIONS else self._active_interval_for_symbol(symbol)
        if (
            symbol == self.current_symbol
            and target_interval == self.current_interval
            and bool(prefer_locked_position_settings) == bool(self.current_chart_prefers_locked_position_settings)
        ):
            return
        self.load_symbol(
            symbol,
            target_interval,
            prefer_locked_position_settings=prefer_locked_position_settings,
        )

    def eventFilter(self, source: object, event: object) -> bool:
        if source is getattr(self, "optimized_group", None):
            if hasattr(event, "type"):
                et = int(event.type())
                if et in (14, 17):  # QEvent::Resize=14, QEvent::Show=17
                    self._reposition_favorable_label()
        if source is getattr(self, "positions_table", None):
            if hasattr(event, "type"):
                et = int(event.type())
                if et in (8, 9, 10):  # FocusIn, FocusOut, Enter
                    self._refresh_position_metric_selection_colors()
        if source in (getattr(self, "candidate_table", None), getattr(self, "optimized_table", None)):
            if hasattr(event, "type") and event.type() == EVENT_KEY_PRESS:
                key = event.key()
                if key == KEY_UP:
                    return self._move_symbol_table_selection(source, -1)
                if key == KEY_DOWN:
                    return self._move_symbol_table_selection(source, 1)
        return super().eventFilter(source, event)

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
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=False)

    def on_candidate_cell_clicked(self, row: int, _column: int) -> None:
        item = self.candidate_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=False)

    def on_optimized_selection_changed(self) -> None:
        selected = self.optimized_table.selectedItems()
        if selected:
            symbol, interval = self._item_symbol_interval(selected[0])
            if symbol:
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=False)

    def on_optimized_cell_clicked(self, row: int, _column: int) -> None:
        item = self.optimized_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=False)

    def on_positions_selection_changed(self) -> None:
        self._refresh_position_metric_selection_colors()
        if self.suppress_positions_selection_load or not self.positions_table.hasFocus():
            return
        selected = self.positions_table.selectedItems()
        if selected:
            symbol, interval = self._item_symbol_interval(selected[0])
            if symbol:
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=True)

    def on_positions_cell_clicked(self, row: int, _column: int) -> None:
        self._refresh_position_metric_selection_colors()
        item = self.positions_table.item(row, 0)
        if item:
            symbol, interval = self._item_symbol_interval(item)
            if symbol:
                self._request_symbol_load(symbol, interval, prefer_locked_position_settings=True)

    def load_symbol(
        self,
        symbol: str,
        target_interval: Optional[str] = None,
        *,
        prefer_locked_position_settings: bool = False,
    ) -> None:
        started_at = time.perf_counter()
        self.symbol_load_started_at = started_at
        self._sync_settings()
        self._show_chart_transition_overlay()
        self._clear_stashed_lightweight_range()
        self._stop_live_stream()
        self._stop_live_backtest_worker()
        self._stop_load_worker()
        self._stop_chart_history_page_worker()
        self._clear_current_chart_snapshot()
        self.current_symbol = symbol
        if hasattr(self, "chart_header_symbol_label"):
            self.chart_header_symbol_label.setText(symbol)
        self._remember_recent_symbol(symbol)
        self.current_position_snapshot = self._find_open_position(symbol)
        self._refresh_position_status_label()
        target_interval = target_interval if target_interval in APP_INTERVAL_OPTIONS else self._active_interval_for_symbol(symbol)
        self.current_chart_prefers_locked_position_settings = bool(prefer_locked_position_settings)
        optimization = self._optimization_result(symbol, target_interval)
        cache_key = self._symbol_interval_key(symbol, target_interval)
        self.current_interval = target_interval
        locked_settings = (
            self._locked_position_strategy_settings(symbol, target_interval)
            if prefer_locked_position_settings
            else None
        )
        cached_history = self._get_history_frame(symbol, target_interval)
        cached_chart_history = self._get_chart_history_frame(symbol, target_interval)
        seed_backtest = self.backtest_cache.get(cache_key)
        if locked_settings is not None and seed_backtest is not None and seed_backtest.settings != locked_settings:
            seed_backtest = None
        if (
            seed_backtest is None
            and optimization is not None
            and optimization.best_interval == target_interval
            and (locked_settings is None or optimization.best_backtest.settings == locked_settings)
        ):
            seed_backtest = optimization.best_backtest
        initial_settings = self._backtest_settings_for_symbol_interval(
            symbol,
            target_interval,
            optimization,
            prefer_locked_position_settings=prefer_locked_position_settings,
        )
        should_refresh_initial_backtest = (
            cached_history is not None
            and not cached_history.empty
            and (
                seed_backtest is None
                or seed_backtest.settings != initial_settings
                or not _backtest_matches_history(seed_backtest, cached_history)
            )
        )
        cached_backtest = (
            self._materialize_cached_backtest(symbol, target_interval, cached_history, seed_backtest, initial_settings, fast_only=True)
            if should_refresh_initial_backtest
            else seed_backtest
        )
        if cached_backtest is not None:
            self.backtest_cache[cache_key] = cached_backtest
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
        self._sync_current_chart_snapshot(
            symbol,
            target_interval,
            confirmed_history=cached_history,
            confirmed_chart_history=cached_chart_history,
            backtest=cached_backtest,
            chart_indicators=cached_chart_indicators,
            preview_bar=None,
            render_signature=(None, 0),
        )
        worker_backtest = cached_backtest
        self.load_request_id += 1
        self.load_request_reset_view[self.load_request_id] = True
        self.load_request_targets[self.load_request_id] = (symbol, target_interval)
        self.chart_range_bars_before = float("inf")
        worker = SymbolLoadWorker(
            self.load_request_id,
            self.settings,
            symbol,
            target_interval,
            cached_history,
            cached_chart_history,
            worker_backtest,
            self._history_last_refresh_at(symbol, target_interval),
            strategy_settings=initial_settings,
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
        expected_target = self.load_request_targets.get(request_id)
        result_interval = str(result.get("interval", self.current_interval))
        if (
            request_id != self.load_request_id
            or symbol != self.current_symbol
            or expected_target != (symbol, result_interval)
        ):
            self.load_request_reset_view.pop(request_id, None)
            self.load_request_targets.pop(request_id, None)
            return
        perf = dict(result.get("perf") or {})
        self._log_perf_ms(f"{symbol} worker fetch", float(perf.get("worker_fetch_ms", 0.0) or 0.0))
        self._log_perf_ms(f"{symbol} worker chart", float(perf.get("worker_chart_ms", 0.0) or 0.0))
        self._log_perf_ms(f"{symbol} worker backtest", float(perf.get("worker_backtest_ms", 0.0) or 0.0))
        reset_view = self.load_request_reset_view.pop(request_id, True)
        self.load_request_targets.pop(request_id, None)
        self.current_interval = result_interval
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
        self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=self.current_backtest,
            chart_indicators=self.current_chart_indicators,
        )
        cache_key = self._symbol_interval_key(symbol, self.current_interval)
        self.backtest_cache[cache_key] = self.current_backtest
        self.chart_indicator_cache[cache_key] = self.current_chart_indicators
        self._prune_caches()
        apply_started_at = time.perf_counter()
        snapshot = self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=self.current_backtest,
            chart_indicators=self.current_chart_indicators,
        )
        if snapshot is None:
            return
        candle_df, indicators, equity_df, markers = self._build_chart_render_payload(snapshot)
        render_signature = self._chart_render_signature_for_payload(candle_df, indicators, equity_df, markers)
        needs_render = (
            reset_view
            or bool(result.get("visible_slice_changed", True))
            or render_signature != self.chart_render_signature
        )
        if needs_render:
            self.render_chart(symbol, self.current_backtest, reset_view=reset_view, chart_indicators=self.current_chart_indicators, _precomputed_payload=(candle_df, indicators, equity_df, markers))
        else:
            self.chart_render_signature = render_signature
            self._sync_current_chart_snapshot(symbol, self.current_interval, render_signature=render_signature)
            self.current_lightweight_markers = list(markers)
            self.current_lightweight_rendered_markers = list(markers)
            self.current_lightweight_preview_markers = []
            self.current_lightweight_fast_entry_markers = []
            self.current_lightweight_fast_exit_markers = []
            self._update_entry_price_overlay()
            self._refresh_live_labels()
            self._refresh_live_preview_markers(symbol)
            self._set_lightweight_optimization_overlay(symbol, self.current_interval)
            self._schedule_chart_transition_reveal(symbol, delay_ms=120)
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
        self.load_request_targets.pop(self.load_request_id, None)
        self.symbol_load_started_at = 0.0
        self._hide_chart_transition_overlay()
        self.show_error(message)

    def render_chart(
        self,
        symbol: str,
        result: BacktestResult,
        reset_view: bool = True,
        chart_indicators: Optional[pd.DataFrame] = None,
        reveal_overlay: bool = True,
        _precomputed_payload: Optional[tuple] = None,
    ) -> None:
        snapshot = self._sync_current_chart_snapshot(
            symbol,
            self.current_interval,
            backtest=result,
            chart_indicators=chart_indicators,
        )
        if snapshot is None:
            return
        if _precomputed_payload is not None:
            candle_df, indicators, equity_df, markers = _precomputed_payload
        else:
            candle_df, indicators, equity_df, markers = self._build_chart_render_payload(snapshot)
        self.current_lightweight_preview_markers = []
        self.current_lightweight_fast_entry_markers = []
        self.current_lightweight_fast_exit_markers = []
        self._render_lightweight_chart(
            symbol,
            candle_df,
            indicators,
            equity_df,
            markers,
            reset_view=reset_view,
            reveal_overlay=reveal_overlay,
        )
        self.current_lightweight_markers = list(markers)
        render_signature = self._chart_render_signature_for_payload(candle_df, indicators, equity_df, markers)
        self.chart_render_signature = render_signature
        self._sync_current_chart_snapshot(symbol, self.current_interval, render_signature=render_signature)

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
        if hasattr(self, "chart_header_symbol_label"):
            self.chart_header_symbol_label.setText(symbol)
        if hasattr(self, "chart_header_tf_label"):
            self.chart_header_tf_label.setText(f"TF {self.current_interval}" if self.current_interval else "TF -")
        strategy_label = STRATEGY_TYPE_LABELS.get(result.settings.strategy_type, result.settings.strategy_type)
        if result.settings.strategy_type == "keltner_trend":
            pending_price = latest.get("pending_entry_price")
            pending_text = (
                f" | Pending {pending_price:.6f}"
                if pending_price is not None and pd.notna(pending_price)
                else ""
            )
            self.signal_label.setText(
                f"신호: {strategy_label} | Trend {latest['trend']} | "
                f"UpperCross {latest['final_bull']} | LowerCross {latest['final_bear']}{pending_text}"
            )
        else:
            self.signal_label.setText(
                f"신호: {strategy_label} | Trend {latest['trend']} | Zone {latest['zone']} | "
                f"Bull {latest['final_bull']} | Bear {latest['final_bear']} | RSI {latest['rsi']:.2f}"
            )
        self._update_entry_price_overlay()
        self._refresh_live_labels()
        self._refresh_live_preview_markers(symbol)
        self._set_lightweight_optimization_overlay(symbol, self.current_interval)

    def _render_lightweight_chart(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        markers: List[Dict[str, object]],
        reset_view: bool = True,
        reveal_overlay: bool = True,
    ) -> None:
        if self.chart is None:
            self._rebuild_chart_view(force=True)
        if not reset_view:
            self._stash_lightweight_range()
        self.chart.set(candle_df)
        self._clear_lightweight_volume_series()
        self._apply_lightweight_precision(symbol, candle_df)
        self.supertrend_line.set(indicators[["time", "supertrend"]].rename(columns={"supertrend": "Supertrend"}))
        self.zone2_line.set(indicators[["time", "zone2_line"]].rename(columns={"zone2_line": "Zone 2"}))
        self.zone3_line.set(indicators[["time", "zone3_line"]].rename(columns={"zone3_line": "Zone 3"}))
        self.ema_fast_line.set(indicators[["time", "ema_fast"]].rename(columns={"ema_fast": "EMA Fast"}))
        self.ema_slow_line.set(indicators[["time", "ema_slow"]].rename(columns={"ema_slow": "EMA Slow"}))
        self.equity_line.set(equity_df)
        range_from, range_to = self._default_lightweight_logical_range(candle_df)
        self._render_lightweight_markers(markers)
        if reset_view:
            self._sync_lightweight_range_via_raf(symbol, range_from, range_to)
        else:
            self._restore_lightweight_range_via_raf(symbol)
        if reveal_overlay:
            self._schedule_chart_transition_reveal(symbol, delay_ms=250 if reset_view else 220)

    def _sync_lightweight_range(self, symbol: str, range_from: float, range_to: float) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        try:
            self.chart.run_script(
                f"""
                {self.equity_subchart.id}.chart.timeScale().setVisibleLogicalRange({{
                    from: {float(range_from)},
                    to: {float(range_to)}
                }});
                {self.chart.id}.chart.timeScale().setVisibleLogicalRange({{
                    from: {float(range_from)},
                    to: {float(range_to)}
                }});
                """
            )
        except Exception:
            self.chart.fit()
            self.equity_subchart.fit()

    def _sync_lightweight_range_via_raf(self, symbol: str, range_from: float, range_to: float) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        equity_id = self.equity_subchart.id
        chart_id = self.chart.id
        rf = float(range_from)
        rt = float(range_to)
        try:
            self.chart.run_script(f"""
                requestAnimationFrame(function() {{
                    requestAnimationFrame(function() {{
                        try {{
                            {equity_id}.chart.timeScale().setVisibleLogicalRange({{
                                from: {rf}, to: {rt}
                            }});
                            {chart_id}.chart.timeScale().setVisibleLogicalRange({{
                                from: {rf}, to: {rt}
                            }});
                        }} catch(e) {{}}
                    }});
                }});
            """)
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
        skip_candle_update: bool = False,
    ) -> bool:
        if (
            symbol != self.current_symbol
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
        tracked_columns = [
            column
            for column in ("time", "open", "high", "low", "close", "supertrend", "zone2_line", "zone3_line", "ema_fast", "ema_slow")
            if column in previous_frame.columns and column in new_frame.columns
        ]
        if tracked_columns:
            previous_compare = previous_frame.loc[:, tracked_columns].reset_index(drop=True)
            if appended_bar:
                next_compare = new_frame.iloc[: len(previous_frame)].loc[:, tracked_columns].reset_index(drop=True)
                if not previous_compare.equals(next_compare):
                    return False
            elif len(previous_frame) > 1:
                previous_prefix = previous_compare.iloc[:-1].reset_index(drop=True)
                next_prefix = new_frame.iloc[:-1].loc[:, tracked_columns].reset_index(drop=True)
                if not previous_prefix.equals(next_prefix):
                    return False
        latest_chart_time = pd.Timestamp(chart_history["time"].iloc[-1])
        try:
            if not skip_candle_update:
                latest_bar = _chart_candle_frame(chart_history).iloc[-1].copy()
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
            new_markers = self._trade_markers(new_backtest.trades, latest_chart_time)
            new_markers.extend(self._open_entry_markers(new_backtest, symbol, self.current_interval))
            self._render_lightweight_markers(new_markers)
            candle_df = _chart_candle_frame(new_frame)
            render_signature = self._chart_render_signature_for_payload(candle_df, new_frame, equity_df, new_markers)
            self.chart_render_signature = render_signature
            self._sync_current_chart_snapshot(symbol, self.current_interval, render_signature=render_signature)
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

    def update_summary(self, symbol: str, backtest: BacktestResult, optimization: Optional[OptimizationResult]) -> None:
        metrics = backtest.metrics
        strategy_label = STRATEGY_TYPE_LABELS.get(backtest.settings.strategy_type, backtest.settings.strategy_type)
        lines = [
            f"Symbol: {symbol}",
            f"Strategy: {strategy_label}",
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
        lines.append(f"- strategy_type: {backtest.settings.strategy_type}")
        for spec in PARAMETER_SPECS:
            lines.append(f"- {spec.key}: {getattr(backtest.settings, spec.key)}")
        if optimization:
            lines.append("")
            lines.append(
                f"Best Interval: {_optimization_result_interval(optimization)}"
            )
            lines.append(
                f"Optimization: {optimization.combinations_tested} combos"
                + (" (trimmed)" if optimization.trimmed_grid else "")
                + f", {optimization.duration_seconds:.2f}s"
            )
        self.backtest_summary_text = "\n".join(lines)
        if self.backtest_summary_box is not None:
            self.backtest_summary_box.setPlainText(self.backtest_summary_text)

    def show_backtest_summary(self) -> None:
        if not self.backtest_summary_text.strip():
            QMessageBox.information(self, "백테스트 서머리", "표시할 백테스트 서머리가 없습니다.")
            return
        if self.backtest_summary_window is not None and self.backtest_summary_window.isVisible():
            self.backtest_summary_window.hide()
            return
        if self.backtest_summary_window is None:
            window = QWidget(None, windowTitle="백테스트 서머리")
            window.resize(520, 680)
            layout = QVBoxLayout(window)
            summary_box = QPlainTextEdit()
            summary_box.setReadOnly(True)
            layout.addWidget(summary_box)
            self.backtest_summary_window = window
            self.backtest_summary_box = summary_box
        assert self.backtest_summary_box is not None
        self.backtest_summary_box.setPlainText(self.backtest_summary_text)
        self.backtest_summary_window.show()
        self.backtest_summary_window.raise_()
        self.backtest_summary_window.activateWindow()

    def _auto_trade_focus_signal_mode(self) -> str:
        if self.auto_trade_focus_mode_combo is None:
            return str(self.settings.auto_trade_focus_signal_mode)
        return str(self.auto_trade_focus_mode_combo.currentData() or self.settings.auto_trade_focus_signal_mode)

    def _ensure_auto_trade_focus_settings_window(self) -> None:
        if self.auto_trade_focus_settings_window is not None:
            return
        window = QWidget(None, windowTitle="차트전환 설정")
        window.resize(320, 160)
        layout = QFormLayout(window)
        enable_check = QCheckBox("자동매매 신호 시 차트 전환")
        enable_check.setChecked(bool(self.settings.auto_trade_focus_on_signal))
        mode_combo = QComboBox()
        mode_combo.addItem("예상진입신호", "preview")
        mode_combo.addItem("진입신호 확정", "confirmed")
        mode_index = mode_combo.findData(self.settings.auto_trade_focus_signal_mode)
        if mode_index >= 0:
            mode_combo.setCurrentIndex(mode_index)
        chart_display_days_spin = QSpinBox()
        chart_display_days_spin.setRange(1, 720)
        chart_display_days_spin.setSuffix(" 시간")
        chart_display_days_spin.setValue(int(self.settings.chart_display_hours))
        layout.addRow("사용", enable_check)
        layout.addRow("기준", mode_combo)
        layout.addRow("차트 표시 시간 범위", chart_display_days_spin)
        self.auto_trade_focus_settings_window = window
        self.auto_trade_focus_enable_check = enable_check
        self.auto_trade_focus_mode_combo = mode_combo
        self.chart_display_days_popup_spin = chart_display_days_spin

    def show_auto_trade_focus_settings(self) -> None:
        self._ensure_auto_trade_focus_settings_window()
        assert self.auto_trade_focus_settings_window is not None
        if self.auto_trade_focus_settings_window.isVisible():
            self.auto_trade_focus_settings_window.hide()
            return
        assert self.auto_trade_focus_enable_check is not None
        assert self.auto_trade_focus_mode_combo is not None
        assert self.chart_display_days_popup_spin is not None
        self.auto_trade_focus_enable_check.setChecked(bool(self.settings.auto_trade_focus_on_signal))
        mode_index = self.auto_trade_focus_mode_combo.findData(self.settings.auto_trade_focus_signal_mode)
        if mode_index >= 0:
            self.auto_trade_focus_mode_combo.setCurrentIndex(mode_index)
        self.chart_display_days_popup_spin.setValue(int(self.settings.chart_display_hours))
        self.auto_trade_focus_settings_window.show()
        self.auto_trade_focus_settings_window.raise_()
        self.auto_trade_focus_settings_window.activateWindow()

    def _set_balance_label_status(self, status: str) -> None:
        self.balance_status_label.setText(status)
        self.balance_status_label.show()
        self.balance_equity_value_label.hide()
        self.balance_equity_unit_label.hide()
        self.balance_available_value_label.hide()
        self.balance_available_unit_label.hide()

    def _set_balance_label_values(self, equity: float, available: float) -> None:
        self.balance_status_label.hide()
        self.balance_equity_value_label.setText(f"{equity:.2f}")
        self.balance_equity_unit_label.setText("USDT | 가용: ")
        self.balance_available_value_label.setText(f"{available:.2f}")
        self.balance_available_unit_label.setText("USDT")
        self.balance_equity_value_label.show()
        self.balance_equity_unit_label.show()
        self.balance_available_value_label.show()
        self.balance_available_unit_label.show()

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

    def _recover_stale_auto_trade_pending_state(self, open_symbols: set[str]) -> None:
        pending_symbol = str(self.order_worker_symbol or "").strip().upper()
        pending_interval = str(self.pending_open_order_interval or "").strip()
        pending_age = (
            max(0.0, time.time() - float(self.order_pending_started_at))
            if self.order_pending_started_at > 0
            else 0.0
        )
        if (
            self.engine_order_pending
            and self.order_worker_is_auto_trade
            and pending_symbol
            and pending_symbol in open_symbols
        ):
            if pending_interval in APP_INTERVAL_OPTIONS:
                self._remember_position_interval(pending_symbol, pending_interval, persist=False)
            self.engine_order_pending = False
            self.order_worker_symbol = None
            self.order_worker_is_auto_close = False
            self.order_worker_is_auto_trade = False
            self.pending_open_order_interval = None
            self.order_pending_started_at = 0.0
            self.auto_trade_entry_pending_symbol = None
            self.auto_trade_entry_pending_fraction = 0.0
            self.auto_trade_entry_pending_cursor_time = None
            self._set_order_buttons_enabled(True)
            self.log(f"{pending_symbol} 자동매매 pending 상태를 계좌 동기화로 복구했습니다.")
            return
        if not self.engine_order_pending or not pending_symbol:
            return
        if pending_symbol == "*" and not open_symbols:
            self.engine_order_pending = False
            self.order_worker_symbol = None
            self.order_worker_is_auto_close = False
            self.order_worker_is_auto_trade = False
            self.pending_open_order_interval = None
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            self.log("전체청산 pending 상태를 계좌 동기화로 복구했습니다.")
            return
        if (
            not self.order_worker_is_auto_close
            and not self.order_worker_is_auto_trade
            and pending_interval in APP_INTERVAL_OPTIONS
            and pending_symbol in open_symbols
        ):
            self._remember_position_interval(pending_symbol, pending_interval, persist=False)
            self.engine_order_pending = False
            self.order_worker_symbol = None
            self.order_worker_is_auto_close = False
            self.order_worker_is_auto_trade = False
            self.pending_open_order_interval = None
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            self.log(f"{pending_symbol} 수동주문 pending 상태를 계좌 동기화로 복구했습니다.")
            return
        if pending_symbol != "*" and pending_interval not in APP_INTERVAL_OPTIONS and pending_symbol not in open_symbols:
            self.engine_order_pending = False
            self.order_worker_symbol = None
            self.order_worker_is_auto_close = False
            self.order_worker_is_auto_trade = False
            self.pending_open_order_interval = None
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            self.log(f"{pending_symbol} 청산 pending 상태를 계좌 동기화로 복구했습니다.")
            return
        if pending_age >= ORDER_PENDING_RECOVERY_SECONDS:
            self.engine_order_pending = False
            self.order_worker_symbol = None
            self.order_worker_is_auto_close = False
            self.order_worker_is_auto_trade = False
            self.pending_open_order_interval = None
            self.order_pending_started_at = 0.0
            self.auto_trade_entry_pending_symbol = None
            self.auto_trade_entry_pending_fraction = 0.0
            self.auto_trade_entry_pending_cursor_time = None
            self._set_order_buttons_enabled(True)
            self.log(f"{pending_symbol} 주문 pending timeout을 복구했습니다.")

    def refresh_account_info(self) -> None:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self._stop_account_worker()
            if self.auto_trade_enabled or self.auto_trade_requested:
                self.auto_trade_requested = False
                self.auto_trade_enabled = False
                self.auto_trade_timer.stop()
                self.auto_trade_entry_pending_symbol = None
                self.auto_trade_entry_pending_fraction = 0.0
                self.auto_trade_entry_pending_cursor_time = None
                self.auto_trade_filled_fraction_by_symbol = {}
                self.auto_trade_cursor_entry_time_by_symbol = {}
                self.log("API 키가 없어 자동매매를 비활성화했습니다.")
            self.open_positions = []
            self.position_strategy_by_symbol = {}
            self.current_position_snapshot = None
            self.account_balance_snapshot = None
            self._refresh_position_price_streams()
            self._set_balance_label_status("API 미입력")
            self.position_label.setText("포지션: API 미입력")
            self._refresh_auto_close_monitors()
            self.update_positions_table()
            self._refresh_optimized_table_highlights()
            self._clear_entry_price_overlay()
            self._refresh_auto_trade_button_state()
            self._sync_trade_engine_state()
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
        self._recover_stale_auto_trade_pending_state(open_symbols)
        self.position_strategy_by_symbol = {
            symbol: strategy_settings
            for symbol, strategy_settings in self.position_strategy_by_symbol.items()
            if symbol in open_symbols
        }
        self.auto_trade_filled_fraction_by_symbol = {
            symbol: fraction
            for symbol, fraction in self.auto_trade_filled_fraction_by_symbol.items()
            if symbol in open_symbols
        }
        self.settings.position_filled_fractions = {
            symbol: fraction
            for symbol, fraction in self.settings.position_filled_fractions.items()
            if symbol in open_symbols
        }
        self.auto_trade_cursor_entry_time_by_symbol = {
            symbol: entry_time
            for symbol, entry_time in self.auto_trade_cursor_entry_time_by_symbol.items()
            if symbol in open_symbols
        }
        self.settings.position_cursor_entry_times = {
            symbol: entry_time
            for symbol, entry_time in self.settings.position_cursor_entry_times.items()
            if symbol in open_symbols
        }
        self._forget_closed_position_intervals(open_symbols, persist=False)
        self._remember_missing_open_position_intervals(open_symbols)
        for symbol in sorted(open_symbols):
            self._remember_position_open_entry_events_for_key(
                symbol,
                self.settings.position_intervals.get(symbol),
                persist=False,
            )
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
        self._sync_trade_engine_state()
        if self.current_backtest and self.current_symbol:
            self._update_entry_price_overlay()
        self._refresh_auto_trade_button_state()
        if self.auto_trade_entry_pending_symbol and self._find_open_position(self.auto_trade_entry_pending_symbol) is not None:
            self.auto_trade_entry_pending_symbol = None
        if self.auto_trade_enabled:
            QTimer.singleShot(0, self._run_auto_trade_cycle)
        if self.pending_account_refresh:
            self.pending_account_refresh = False
            self.refresh_account_info()

    def _on_account_info_failed(self, message: str) -> None:
        self.open_positions = []
        self.position_strategy_by_symbol = {}
        self.position_open_entry_events_by_symbol = {}
        self.current_position_snapshot = None
        self.account_balance_snapshot = None
        self._refresh_position_price_streams()
        self._set_balance_label_status("조회 실패")
        self.position_label.setText("포지션: 조회 실패")
        self.update_positions_table()
        self._clear_entry_price_overlay()
        self.log(message)
        self.auto_trade_entry_pending_symbol = None
        self.auto_trade_entry_pending_fraction = 0.0
        self.auto_trade_entry_pending_cursor_time = None
        self.auto_trade_filled_fraction_by_symbol = {}
        self.auto_trade_cursor_entry_time_by_symbol = {}
        self._refresh_auto_close_retry_timer()
        self._refresh_auto_trade_button_state()
        self._sync_trade_engine_state()
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
        *,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        auto_trade: bool = False,
    ) -> Tuple[bool, str]:
        self._sync_settings()
        target_symbol = str(symbol or self.current_symbol or "").strip()
        target_interval = str(interval or self.current_interval or self.settings.kline_interval).strip()
        if not target_symbol:
            message = "주문할 종목을 먼저 선택하세요."
            self.show_warning(message)
            return False, message
        if not self.settings.api_key or not self.settings.api_secret:
            if auto_trade:
                message = "자동매매 진입 실패: API Key / Secret이 없습니다."
                self.log(message)
            else:
                message = "API Key / Secret을 입력해야 실제 주문할 수 있습니다."
                self.show_warning(message)
            return False, message
        if self._is_order_pending():
            if auto_trade:
                message = f"{target_symbol} 자동매매 진입 대기: 기존 주문 처리 중"
                self.log(message)
            else:
                message = "이미 주문 처리 중입니다."
                self.show_warning(message)
            return False, message
        if margin is not None and margin <= 0:
            if auto_trade:
                message = f"{target_symbol} 자동매매 진입 실패: 주문금액은 0보다 커야 합니다."
                self.log(message)
            else:
                message = "주문금액은 0보다 커야 합니다."
                self.show_warning(message)
            return False, message
        if not self._ensure_trade_engine_available():
            message = "Trade engine is not available."
            self.show_error(message)
            return False, message
        optimization = self._optimization_result(target_symbol, target_interval)
        strategy_settings = optimization.best_backtest.settings if optimization else self.settings.strategy
        self.order_worker_symbol = target_symbol
        self.order_worker_is_auto_close = False
        self.order_worker_is_auto_trade = auto_trade
        self.engine_order_pending = True
        self.order_pending_started_at = time.time()
        self.pending_open_order_interval = target_interval
        if auto_trade:
            self.auto_trade_entry_pending_symbol = target_symbol
            self.auto_trade_entry_pending_at = time.time()
            self.auto_trade_entry_pending_fraction = float(fraction or 0.0)
            self.auto_trade_entry_pending_cursor_time = None
            self._request_symbol_load(target_symbol, target_interval)
        else:
            self.auto_trade_entry_pending_fraction = 0.0
            self.auto_trade_entry_pending_cursor_time = None
        self._set_order_buttons_enabled(False)
        if auto_trade:
            self.log(
                f"{target_symbol} 자동매매 진입 실행: {side} "
                f"{int(round((fraction or 0.0) * 100)) if fraction is not None else 0}%"
            )
        self.statusBar().showMessage(f"{target_symbol} 주문 처리 중...", 3000)
        try:
            self.trade_engine.send(
                EngineOpenOrderCommand(
                    symbol=target_symbol,
                    interval=target_interval,
                    side=side,
                    leverage=int(self.settings.leverage),
                    fraction=fraction,
                    margin=margin,
                    auto_trade=auto_trade,
                    strategy_settings=strategy_settings,
                )
            )
        except Exception as exc:
            self.engine_order_pending = False
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            message = f"Trade engine open order failed: {exc}"
            self._on_order_failed(message)
            return False, message
        return True, f"{target_symbol} 주문 요청이 접수되었습니다."

    def close_selected_position(self) -> None:
        if not self.current_symbol:
            self.show_warning("종목을 먼저 선택하세요.")
            return
        yes_button = getattr(QMessageBox, "Yes", None)
        no_button = getattr(QMessageBox, "No", None)
        if yes_button is None or no_button is None:
            standard_button = getattr(QMessageBox, "StandardButton", None)
            yes_button = standard_button.Yes
            no_button = standard_button.No
        open_symbols = sorted(position.symbol for position in self.open_positions)
        if not open_symbols:
            self.show_warning("청산할 포지션이 없습니다.")
            return
        answer = QMessageBox.question(
            self,
            "전체청산 확인",
            f"보유 포지션 {len(open_symbols)}개를 모두 전체청산할까요?\n" + ", ".join(open_symbols),
            yes_button | no_button,
            no_button,
        )
        if answer != yes_button:
            return
        self.close_all_positions()

    def close_all_positions(self) -> None:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self.show_warning("API Key / Secret을 입력해야 실제 주문이 가능합니다.")
            return
        if self._is_order_pending():
            self.show_warning("이미 주문 처리 중입니다.")
            return
        if not self.open_positions:
            self.show_warning("청산할 포지션이 없습니다.")
            return
        if not self._ensure_trade_engine_available():
            self.show_error("Trade engine is not available.")
            return
        self.order_worker_symbol = "*"
        self.order_worker_is_auto_close = False
        self.order_worker_is_auto_trade = False
        self.engine_order_pending = True
        self.order_pending_started_at = time.time()
        self.pending_open_order_interval = None
        self._set_order_buttons_enabled(False)
        self.statusBar().showMessage("전체 포지션 청산 처리 중...", 3000)
        try:
            self.trade_engine.send(EngineCloseAllPositionsCommand())
        except Exception as exc:
            self.engine_order_pending = False
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            self._on_order_failed(f"Trade engine close-all failed: {exc}")

    def _submit_close_position(self, symbol: str, auto_close_reason: Optional[str] = None) -> bool:
        self._sync_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            if auto_close_reason is None:
                self.show_warning("API Key / Secret을 입력해야 실제 주문이 가능합니다.")
            else:
                self.log(f"{symbol} 자동청산 실패: API Key / Secret이 없습니다.")
            return False
        if self._is_order_pending():
            if auto_close_reason is None:
                self.show_warning("이미 주문 처리 중입니다.")
            return False
        if not self._ensure_trade_engine_available():
            if auto_close_reason is None:
                self.show_error("Trade engine is not available.")
            else:
                self.log(f"{symbol} 자동청산 실패: trade engine unavailable")
            return False
        self.order_worker_symbol = symbol
        self.order_worker_is_auto_close = auto_close_reason is not None
        self.order_worker_is_auto_trade = False
        self.engine_order_pending = True
        self.order_pending_started_at = time.time()
        self.pending_open_order_interval = None
        self._set_order_buttons_enabled(False)
        if auto_close_reason is None:
            self.statusBar().showMessage(f"{symbol} 청산 처리 중...", 3000)
        else:
            self.log(f"{symbol} 자동청산 실행: {_auto_close_reason_text(auto_close_reason)}")
            self.statusBar().showMessage(f"{symbol} 자동청산 처리 중...", 3000)
        try:
            self.trade_engine.send(
                EngineCloseOrderCommand(
                    symbol=symbol,
                    reason=auto_close_reason,
                    auto_close=auto_close_reason is not None,
                )
            )
        except Exception as exc:
            self.engine_order_pending = False
            self.order_pending_started_at = 0.0
            self._set_order_buttons_enabled(True)
            self._on_order_failed(f"Trade engine close order failed: {exc}")
            return False
        return True

    def close_position_for_symbol(self, symbol: str) -> None:
        self._submit_close_position(symbol)

    def _on_order_completed(self, payload: object) -> None:
        self.engine_order_pending = False
        self.order_pending_started_at = 0.0
        self._set_order_buttons_enabled(True)
        result = dict(payload)
        message_text = str(result.get("message", ""))
        order_symbol = self.order_worker_symbol or str(result.get("symbol", ""))
        close_interval = self._position_interval_for_symbol(order_symbol) if order_symbol else ""
        if self.order_worker_is_auto_close and order_symbol:
            self.auto_close_order_pending.discard(order_symbol)
            self.auto_close_last_attempt_at[order_symbol] = time.time()
        if (
            order_symbol
            and close_interval in APP_INTERVAL_OPTIONS
            and (self.order_worker_is_auto_close or "close completed" in message_text.lower())
        ):
            self.auto_trade_reentry_cooldown_until[(order_symbol, close_interval)] = time.time() + 60.0
        if not self.order_worker_is_auto_close and order_symbol:
            self._remember_position_interval(order_symbol, self.pending_open_order_interval)
        was_auto_trade = self.order_worker_is_auto_trade
        if was_auto_trade and order_symbol:
            current_fraction = float(self.auto_trade_filled_fraction_by_symbol.get(order_symbol, 0.0))
            self._remember_position_filled_fraction(
                order_symbol,
                min(
                    signal_fraction_for_zone(3),
                    current_fraction + float(self.auto_trade_entry_pending_fraction or 0.0),
                ),
                persist=True,
            )
            if self.auto_trade_entry_pending_cursor_time is not None:
                self._remember_position_cursor_entry_time(
                    order_symbol,
                    self.auto_trade_entry_pending_cursor_time,
                    persist=True,
                )
        self.order_worker_symbol = None
        self.order_worker_is_auto_close = False
        self.order_worker_is_auto_trade = False
        self.pending_open_order_interval = None
        self.auto_trade_entry_pending_fraction = 0.0
        self.log(str(result.get("message", "주문 완료")))
        self.refresh_account_info()
        QTimer.singleShot(0, self._flush_queued_auto_close_orders)
        if was_auto_trade:
            QTimer.singleShot(0, self._run_auto_trade_cycle)

    def _on_order_failed(self, message: str) -> None:
        self.engine_order_pending = False
        self.order_pending_started_at = 0.0
        order_symbol = self.order_worker_symbol
        was_auto_close = self.order_worker_is_auto_close
        was_auto_trade = self.order_worker_is_auto_trade
        if was_auto_close and order_symbol:
            self.auto_close_order_pending.discard(order_symbol)
            self.auto_close_last_attempt_at[order_symbol] = time.time()
        if was_auto_trade:
            self.auto_trade_entry_pending_symbol = None
            self.auto_trade_entry_pending_fraction = 0.0
            self.auto_trade_entry_pending_cursor_time = None
        self.order_worker_symbol = None
        self.order_worker_is_auto_close = False
        self.order_worker_is_auto_trade = False
        self.pending_open_order_interval = None
        self._set_order_buttons_enabled(True)
        self.log(message)
        if not was_auto_close and not was_auto_trade:
            self.show_error("주문 처리 중 오류가 발생했습니다. 로그를 확인하세요.")
        QTimer.singleShot(0, self._flush_queued_auto_close_orders)

    def show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Warning", message)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._sync_chart_transition_overlay()
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.auto_refresh_timer.stop()
            self.live_update_timer.stop()
            self.optimized_table_timer.stop()
            self.optimized_table_highlight_timer.stop()
            self.favorable_backtest_poll_timer.stop()
            self.auto_close_retry_timer.stop()
            self.auto_trade_timer.stop()
            self.trade_engine_poll_timer.stop()
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
            self._stop_trade_engine()
            self._stop_mobile_web_server()
            self.favorable_backtest_process.stop()
            self._drain_tracked_threads()
            self.save_settings()
        finally:
            super().closeEvent(event)


def create_app() -> QApplication:
    return QApplication.instance() or QApplication([])
