from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

from .config import StrategySettings

RESULT_INDICATOR_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "supertrend",
    "zone2_line",
    "zone3_line",
    "ema_fast",
    "ema_slow",
    "rsi",
    "trend_to_long",
    "trend_to_short",
    "final_bull",
    "final_bear",
]

CHART_INDICATOR_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "supertrend",
    "zone2_line",
    "zone3_line",
    "ema_fast",
    "ema_slow",
    "trend_to_long",
    "trend_to_short",
    "final_bull",
    "final_bear",
]


SENSITIVITY_MULTIPLIERS = {
    "1-Ultra Fine Max": 0.3,
    "2-Ultra Fine": 0.5,
    "3-Fine Max": 0.65,
    "4-Fine": 0.8,
    "5-Normal": 1.0,
    "6-Broad Min": 1.25,
    "7-Broad": 1.5,
    "8-Broad Max": 1.75,
    "9-Ultra Broad": 2.0,
    "10-Ultra Broad Max": 2.5,
}

PREPARED_OHLCV_ATTR = "_alt_prepared_ohlcv"


@dataclass(frozen=True)
class TradeRecord:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    reason: str
    zones: str


@dataclass(frozen=True)
class StrategyMetrics:
    total_return_pct: float
    net_profit: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    profit_factor: float


@dataclass(frozen=True)
class BacktestCursor:
    processed_bars: int
    last_time: pd.Timestamp
    equity: float
    position_qty: float
    avg_entry_price: float
    entry_side: str
    entry_time: pd.Timestamp
    entry_price: float
    zone_events: Tuple[str, ...]
    gross_profit: float
    gross_loss: float
    trade_count: int
    win_count: int
    long_zone_used: Tuple[bool, bool, bool]
    short_zone_used: Tuple[bool, bool, bool]
    last_long_zone: int
    last_short_zone: int
    last_equity_value: float
    indicator_cursor: Optional["IndicatorCursor"] = None


@dataclass(frozen=True)
class IndicatorCursor:
    payload: Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class BacktestResult:
    settings: StrategySettings
    metrics: StrategyMetrics
    trades: List[TradeRecord]
    indicators: pd.DataFrame
    latest_state: Dict[str, object]
    equity_curve: pd.Series
    cursor: Optional[BacktestCursor] = None
    history_signature: Tuple[object, ...] = ()


def compact_indicator_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in df.columns]
    if not selected:
        return df.tail(1).copy()
    return df.loc[:, selected].copy()


def _indicator_history_signature(df: pd.DataFrame) -> Tuple[object, ...]:
    if df.empty or "time" not in df.columns:
        return (None, 0)
    last = df.iloc[-1]
    values: List[object] = [pd.Timestamp(last["time"]), int(len(df))]
    for column in ("open", "high", "low", "close", "volume"):
        if column not in df.columns:
            continue
        value = last[column]
        values.append(None if pd.isna(value) else float(value))
    return tuple(values)


def _strip_provisional_trade(trades: List[TradeRecord], cursor: Optional[BacktestCursor]) -> List[TradeRecord]:
    if not trades or cursor is None:
        return list(trades)
    last_trade = trades[-1]
    if last_trade.reason == "end_of_test" and pd.Timestamp(last_trade.exit_time) == pd.Timestamp(cursor.last_time):
        return list(trades[:-1])
    return list(trades)


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.attrs.get(PREPARED_OHLCV_ATTR):
        return df
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame[["open", "high", "low", "close", "volume"]] = frame[["open", "high", "low", "close", "volume"]].astype(float)
    frame = frame.sort_values("time").reset_index(drop=True)
    frame.attrs[PREPARED_OHLCV_ATTR] = True
    return frame


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_ohlcv(df)


def _coerce_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _encode_cursor_value(value: object) -> object:
    if isinstance(value, deque):
        value = tuple(value)
    if isinstance(value, tuple):
        return tuple(_encode_cursor_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_encode_cursor_value(item) for item in value)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(round(value, 15))
    return value


def _decode_cursor_value(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(_decode_cursor_value(item) for item in value)
    if value is None:
        return float("nan")
    return value


class _EwmTracker:
    def __init__(self, length: int, alpha: float, snapshot: Optional[Tuple[int, object]] = None) -> None:
        self.length = int(length)
        self.alpha = float(alpha)
        self.count = 0
        self.value = float("nan")
        if snapshot is not None:
            self.count = int(snapshot[0])
            self.value = _coerce_float(_decode_cursor_value(snapshot[1]))

    def update(self, value: float) -> float:
        numeric = _coerce_float(value)
        if not math.isfinite(numeric):
            return self.value
        if self.count == 0 or not math.isfinite(self.value):
            self.value = numeric
        else:
            self.value = ((1.0 - self.alpha) * self.value) + (self.alpha * numeric)
        self.count += 1
        if self.count < self.length:
            return float("nan")
        return self.value

    def snapshot(self) -> Tuple[int, object]:
        return (int(self.count), _encode_cursor_value(self.value))


class _RollingWindow:
    def __init__(self, size: int, values: Iterable[object] = ()) -> None:
        self.size = max(1, int(size))
        self.values: Deque[float] = deque(maxlen=self.size)
        self.sum = 0.0
        self.sumsq = 0.0
        self.finite_count = 0
        for value in values:
            self.push(_coerce_float(_decode_cursor_value(value)))

    def push(self, value: float) -> None:
        numeric = _coerce_float(value)
        if len(self.values) == self.size:
            removed = self.values[0]
            if math.isfinite(removed):
                self.sum -= removed
                self.sumsq -= removed * removed
                self.finite_count -= 1
        self.values.append(numeric)
        if math.isfinite(numeric):
            self.sum += numeric
            self.sumsq += numeric * numeric
            self.finite_count += 1

    def mean(self, min_periods: int) -> float:
        if self.finite_count < max(int(min_periods), 1):
            return float("nan")
        return self.sum / max(self.finite_count, 1)

    def std(self, min_periods: int, ddof: int = 1) -> float:
        required = max(int(min_periods), 1)
        if self.finite_count < required or self.finite_count <= ddof:
            return float("nan")
        numerator = self.sumsq - ((self.sum * self.sum) / self.finite_count)
        variance = numerator / max(self.finite_count - ddof, 1)
        return math.sqrt(max(variance, 0.0))

    def min(self, min_periods: int = 1) -> float:
        if self.finite_count < max(int(min_periods), 1):
            return float("nan")
        finite_values = [value for value in self.values if math.isfinite(value)]
        return min(finite_values) if finite_values else float("nan")

    def max(self, min_periods: int = 1) -> float:
        if self.finite_count < max(int(min_periods), 1):
            return float("nan")
        finite_values = [value for value in self.values if math.isfinite(value)]
        return max(finite_values) if finite_values else float("nan")

    def snapshot(self) -> Tuple[object, ...]:
        return tuple(_encode_cursor_value(value) for value in self.values)


class _PivotTracker:
    def __init__(self, left: int, values: Iterable[object] = ()) -> None:
        self.left = max(1, int(left))
        self.values: Deque[float] = deque(maxlen=self.left + 2)
        for value in values:
            self.values.append(_coerce_float(_decode_cursor_value(value)))

    def update_high(self, value: float) -> float:
        self.values.append(_coerce_float(value))
        if len(self.values) < self.left + 2:
            return float("nan")
        candidate = self.values[-2]
        if not math.isfinite(candidate):
            return float("nan")
        return candidate if candidate >= max(self.values) else float("nan")

    def update_low(self, value: float) -> float:
        self.values.append(_coerce_float(value))
        if len(self.values) < self.left + 2:
            return float("nan")
        candidate = self.values[-2]
        if not math.isfinite(candidate):
            return float("nan")
        return candidate if candidate <= min(self.values) else float("nan")

    def snapshot(self) -> Tuple[object, ...]:
        return tuple(_encode_cursor_value(value) for value in self.values)


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if not math.isfinite(avg_gain) or not math.isfinite(avg_loss):
        return float("nan")
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _rsi_series(values: pd.Series, length: int) -> pd.Series:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return pd.Series(dtype=float)
    gain_tracker = _EwmTracker(length, 1.0 / max(int(length), 1))
    loss_tracker = _EwmTracker(length, 1.0 / max(int(length), 1))
    result = np.full(len(series), np.nan, dtype=float)
    prev_close = float("nan")
    for index, close_value in enumerate(series.to_numpy(dtype=float, copy=False)):
        if math.isfinite(prev_close):
            delta = close_value - prev_close
            avg_gain = gain_tracker.update(max(delta, 0.0))
            avg_loss = loss_tracker.update(max(-delta, 0.0))
            if math.isfinite(avg_gain) and math.isfinite(avg_loss):
                result[index] = _rsi_from_avgs(avg_gain, avg_loss)
        prev_close = close_value
    return pd.Series(result, index=series.index, dtype=float)


def rsi_last_value(values: pd.Series, length: int = 14) -> float:
    series = _rsi_series(values, length)
    if series.empty:
        return float("nan")
    value = series.iloc[-1]
    return float(value) if pd.notna(value) else float("nan")


def _indicator_constants(settings: StrategySettings) -> Dict[str, float]:
    sensitivity_mult = SENSITIVITY_MULTIPLIERS.get(settings.sensitivity_mode, 1.0)
    zz_left = max(2, int(round(settings.zz_len_raw * sensitivity_mult)))
    qtp_sens = settings.qtp_sensitivity / 100.0
    return {
        "zone3_thresh": 0.5 * settings.zone_sensitivity,
        "zone2_thresh": 1.0 * settings.zone_sensitivity,
        "zz_left": zz_left,
        "atr_mult": settings.atr_mult_raw / sensitivity_mult,
        "qtp_sens": qtp_sens,
        "qtp_pivot_left": int(round(settings.qtp_max_pvt_left - (settings.qtp_max_pvt_left - settings.qtp_min_pvt_left) * qtp_sens)),
        "qtp_rsi_upper": 70 - round(10 * qtp_sens),
        "qtp_rsi_lower": 30 + round(10 * qtp_sens),
        "qtp_stoch_upper": 80 - round(10 * qtp_sens),
        "qtp_stoch_lower": 20 + round(10 * qtp_sens),
        "qtp_score_thr": 55 - round(15 * qtp_sens),
    }


def _build_indicator_state(settings: StrategySettings, cursor: Optional[IndicatorCursor] = None) -> Dict[str, Any]:
    payload = {key: _decode_cursor_value(value) for key, value in (cursor.payload if cursor is not None else ())}
    constants = _indicator_constants(settings)
    state: Dict[str, Any] = {
        "processed_rows": int(payload.get("processed_rows", 0) or 0),
        "prev_close": _coerce_float(payload.get("prev_close")),
        "prev_close_2": _coerce_float(payload.get("prev_close_2")),
        "prev_high": _coerce_float(payload.get("prev_high")),
        "prev_low": _coerce_float(payload.get("prev_low")),
        "prev_is_long_trend": bool(payload.get("prev_is_long_trend", False)),
        "prev_is_short_trend": bool(payload.get("prev_is_short_trend", False)),
        "supertrend_value": _coerce_float(payload.get("supertrend_value")),
        "supertrend_direction": _coerce_float(payload.get("supertrend_direction")),
        "supertrend_final_upper": _coerce_float(payload.get("supertrend_final_upper")),
        "supertrend_final_lower": _coerce_float(payload.get("supertrend_final_lower")),
        "prev_qip_rsi": _coerce_float(payload.get("prev_qip_rsi")),
        "prev_qip_vol_ma": _coerce_float(payload.get("prev_qip_vol_ma")),
        "prev_volume": _coerce_float(payload.get("prev_volume")),
        "prev_qip_ema_fast": _coerce_float(payload.get("prev_qip_ema_fast")),
        "prev_qip_ema_slow": _coerce_float(payload.get("prev_qip_ema_slow")),
        "prev_macd_raw": _coerce_float(payload.get("prev_macd_raw")),
        "prev_qip_low_roll": _coerce_float(payload.get("prev_qip_low_roll")),
        "prev_qip_high_roll": _coerce_float(payload.get("prev_qip_high_roll")),
        "prev_atr14": _coerce_float(payload.get("prev_atr14")),
        "last_pl_rsi": _coerce_float(payload.get("last_pl_rsi")),
        "last_ph_rsi": _coerce_float(payload.get("last_ph_rsi")),
        "last_pl_val": _coerce_float(payload.get("last_pl_val")),
        "last_ph_val": _coerce_float(payload.get("last_ph_val")),
        "last_pl_macd": _coerce_float(payload.get("last_pl_macd")),
        "last_ph_macd": _coerce_float(payload.get("last_ph_macd")),
        "prev_qtp_rsi": _coerce_float(payload.get("prev_qtp_rsi")),
        "prev_qtp_stoch": _coerce_float(payload.get("prev_qtp_stoch")),
        "prev_qtp_zdev": _coerce_float(payload.get("prev_qtp_zdev")),
        "prev_qtp_bull_rev": bool(payload.get("prev_qtp_bull_rev", False)),
        "prev_qtp_bear_rev": bool(payload.get("prev_qtp_bear_rev", False)),
        "prev_qtp_vol_spike": bool(payload.get("prev_qtp_vol_spike", False)),
        "prev_qtp_up_trend": bool(payload.get("prev_qtp_up_trend", False)),
        "prev_qtp_dn_trend": bool(payload.get("prev_qtp_dn_trend", False)),
        "prev_qtp_roc1": _coerce_float(payload.get("prev_qtp_roc1")),
        "prev_qtp_roc2": _coerce_float(payload.get("prev_qtp_roc2")),
        "constants": constants,
        "supertrend_atr": _EwmTracker(settings.atr_period, 1.0 / max(int(settings.atr_period), 1), payload.get("supertrend_atr_state")),
        "atr14": _EwmTracker(14, 1.0 / 14.0, payload.get("atr14_state")),
        "qtp_atr": _EwmTracker(settings.qtp_atr_len, 1.0 / max(int(settings.qtp_atr_len), 1), payload.get("qtp_atr_state")),
        "ema12": _EwmTracker(12, 2.0 / (12 + 1), payload.get("ema12_state")),
        "ema26": _EwmTracker(26, 2.0 / (26 + 1), payload.get("ema26_state")),
        "qip_ema_fast": _EwmTracker(settings.qip_ema_fast, 2.0 / (settings.qip_ema_fast + 1), payload.get("qip_ema_fast_state")),
        "qip_ema_slow": _EwmTracker(settings.qip_ema_slow, 2.0 / (settings.qip_ema_slow + 1), payload.get("qip_ema_slow_state")),
        "qtp_ema_fast": _EwmTracker(settings.qtp_ema_fast_len, 2.0 / (settings.qtp_ema_fast_len + 1), payload.get("qtp_ema_fast_state")),
        "qtp_ema_slow": _EwmTracker(settings.qtp_ema_slow_len, 2.0 / (settings.qtp_ema_slow_len + 1), payload.get("qtp_ema_slow_state")),
        "qtp_basis": _EwmTracker(settings.qtp_dev_lookback, 2.0 / (settings.qtp_dev_lookback + 1), payload.get("qtp_basis_state")),
        "qip_rsi_gain": _EwmTracker(settings.qip_rsi_len, 1.0 / max(int(settings.qip_rsi_len), 1), payload.get("qip_rsi_gain_state")),
        "qip_rsi_loss": _EwmTracker(settings.qip_rsi_len, 1.0 / max(int(settings.qip_rsi_len), 1), payload.get("qip_rsi_loss_state")),
        "qtp_rsi_gain": _EwmTracker(settings.qtp_rsi_len, 1.0 / max(int(settings.qtp_rsi_len), 1), payload.get("qtp_rsi_gain_state")),
        "qtp_rsi_loss": _EwmTracker(settings.qtp_rsi_len, 1.0 / max(int(settings.qtp_rsi_len), 1), payload.get("qtp_rsi_loss_state")),
        "qip_volume_window": _RollingWindow(settings.vol_ma_len, payload.get("qip_volume_window", ())),
        "qip_low_window": _RollingWindow(max(int(constants["zz_left"]) * 2, 1), payload.get("qip_low_window", ())),
        "qip_high_window": _RollingWindow(max(int(constants["zz_left"]) * 2, 1), payload.get("qip_high_window", ())),
        "qip_pivot_high": _PivotTracker(int(constants["zz_left"]), payload.get("qip_pivot_high_window", ())),
        "qip_pivot_low": _PivotTracker(int(constants["zz_left"]), payload.get("qip_pivot_low_window", ())),
        "qtp_stoch_high_window": _RollingWindow(settings.qtp_stoch_len, payload.get("qtp_stoch_high_window", ())),
        "qtp_stoch_low_window": _RollingWindow(settings.qtp_stoch_len, payload.get("qtp_stoch_low_window", ())),
        "qtp_volume_window": _RollingWindow(settings.qtp_vol_len, payload.get("qtp_volume_window", ())),
        "qtp_dev_window": _RollingWindow(settings.qtp_dev_lookback, payload.get("qtp_dev_window", ())),
        "qtp_pivot_high": _PivotTracker(int(constants["qtp_pivot_left"]), payload.get("qtp_pivot_high_window", ())),
        "qtp_pivot_low": _PivotTracker(int(constants["qtp_pivot_left"]), payload.get("qtp_pivot_low_window", ())),
    }
    return state


def _snapshot_indicator_state(state: Dict[str, Any]) -> IndicatorCursor:
    payload = {
        "processed_rows": int(state["processed_rows"]),
        "prev_close": state["prev_close"],
        "prev_close_2": state["prev_close_2"],
        "prev_high": state["prev_high"],
        "prev_low": state["prev_low"],
        "prev_is_long_trend": bool(state["prev_is_long_trend"]),
        "prev_is_short_trend": bool(state["prev_is_short_trend"]),
        "supertrend_value": state["supertrend_value"],
        "supertrend_direction": state["supertrend_direction"],
        "supertrend_final_upper": state["supertrend_final_upper"],
        "supertrend_final_lower": state["supertrend_final_lower"],
        "prev_qip_rsi": state["prev_qip_rsi"],
        "prev_qip_vol_ma": state["prev_qip_vol_ma"],
        "prev_volume": state["prev_volume"],
        "prev_qip_ema_fast": state["prev_qip_ema_fast"],
        "prev_qip_ema_slow": state["prev_qip_ema_slow"],
        "prev_macd_raw": state["prev_macd_raw"],
        "prev_qip_low_roll": state["prev_qip_low_roll"],
        "prev_qip_high_roll": state["prev_qip_high_roll"],
        "prev_atr14": state["prev_atr14"],
        "last_pl_rsi": state["last_pl_rsi"],
        "last_ph_rsi": state["last_ph_rsi"],
        "last_pl_val": state["last_pl_val"],
        "last_ph_val": state["last_ph_val"],
        "last_pl_macd": state["last_pl_macd"],
        "last_ph_macd": state["last_ph_macd"],
        "prev_qtp_rsi": state["prev_qtp_rsi"],
        "prev_qtp_stoch": state["prev_qtp_stoch"],
        "prev_qtp_zdev": state["prev_qtp_zdev"],
        "prev_qtp_bull_rev": bool(state["prev_qtp_bull_rev"]),
        "prev_qtp_bear_rev": bool(state["prev_qtp_bear_rev"]),
        "prev_qtp_vol_spike": bool(state["prev_qtp_vol_spike"]),
        "prev_qtp_up_trend": bool(state["prev_qtp_up_trend"]),
        "prev_qtp_dn_trend": bool(state["prev_qtp_dn_trend"]),
        "prev_qtp_roc1": state["prev_qtp_roc1"],
        "prev_qtp_roc2": state["prev_qtp_roc2"],
        "supertrend_atr_state": state["supertrend_atr"].snapshot(),
        "atr14_state": state["atr14"].snapshot(),
        "qtp_atr_state": state["qtp_atr"].snapshot(),
        "ema12_state": state["ema12"].snapshot(),
        "ema26_state": state["ema26"].snapshot(),
        "qip_ema_fast_state": state["qip_ema_fast"].snapshot(),
        "qip_ema_slow_state": state["qip_ema_slow"].snapshot(),
        "qtp_ema_fast_state": state["qtp_ema_fast"].snapshot(),
        "qtp_ema_slow_state": state["qtp_ema_slow"].snapshot(),
        "qtp_basis_state": state["qtp_basis"].snapshot(),
        "qip_rsi_gain_state": state["qip_rsi_gain"].snapshot(),
        "qip_rsi_loss_state": state["qip_rsi_loss"].snapshot(),
        "qtp_rsi_gain_state": state["qtp_rsi_gain"].snapshot(),
        "qtp_rsi_loss_state": state["qtp_rsi_loss"].snapshot(),
        "qip_volume_window": state["qip_volume_window"].snapshot(),
        "qip_low_window": state["qip_low_window"].snapshot(),
        "qip_high_window": state["qip_high_window"].snapshot(),
        "qip_pivot_high_window": state["qip_pivot_high"].snapshot(),
        "qip_pivot_low_window": state["qip_pivot_low"].snapshot(),
        "qtp_stoch_high_window": state["qtp_stoch_high_window"].snapshot(),
        "qtp_stoch_low_window": state["qtp_stoch_low_window"].snapshot(),
        "qtp_volume_window": state["qtp_volume_window"].snapshot(),
        "qtp_dev_window": state["qtp_dev_window"].snapshot(),
        "qtp_pivot_high_window": state["qtp_pivot_high"].snapshot(),
        "qtp_pivot_low_window": state["qtp_pivot_low"].snapshot(),
    }
    return IndicatorCursor(payload=tuple(sorted((key, _encode_cursor_value(value)) for key, value in payload.items())))

def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    return _rsi_series(series.astype(float), length)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return _rma(tr, length)


def _macd_line(close: pd.Series) -> pd.Series:
    return _ema(close, 12) - _ema(close, 26)


def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    lowest = low.rolling(length, min_periods=length).min()
    highest = high.rolling(length, min_periods=length).max()
    return (close - lowest) / (highest - lowest).replace(0.0, np.nan) * 100.0


def _supertrend(df: pd.DataFrame, length: int, factor: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    atr = _atr(high, low, close, length)
    hl2 = (high + low) / 2.0
    upperband = hl2 + factor * atr
    lowerband = hl2 - factor * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    valid_start = next((i for i, value in enumerate(atr) if np.isfinite(value)), None)
    if valid_start is None:
        return supertrend, direction, atr

    supertrend.iat[valid_start] = upperband.iat[valid_start]
    direction.iat[valid_start] = 1.0

    for i in range(valid_start + 1, len(df)):
        prev_final_upper = final_upper.iat[i - 1]
        prev_final_lower = final_lower.iat[i - 1]
        if not np.isfinite(prev_final_upper):
            prev_final_upper = upperband.iat[i - 1]
        if not np.isfinite(prev_final_lower):
            prev_final_lower = lowerband.iat[i - 1]

        if upperband.iat[i] < prev_final_upper or close.iat[i - 1] > prev_final_upper:
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = prev_final_upper

        if lowerband.iat[i] > prev_final_lower or close.iat[i - 1] < prev_final_lower:
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = prev_final_lower

        if supertrend.iat[i - 1] == prev_final_upper:
            if close.iat[i] <= final_upper.iat[i]:
                supertrend.iat[i] = final_upper.iat[i]
                direction.iat[i] = 1.0
            else:
                supertrend.iat[i] = final_lower.iat[i]
                direction.iat[i] = -1.0
        else:
            if close.iat[i] >= final_lower.iat[i]:
                supertrend.iat[i] = final_lower.iat[i]
                direction.iat[i] = -1.0
            else:
                supertrend.iat[i] = final_upper.iat[i]
                direction.iat[i] = 1.0
    return supertrend, direction, atr


def _pivot_high(values: np.ndarray, left: int, right: int = 1) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if left <= 0 or len(values) <= left + right:
        return result
    for pivot_idx in range(left, len(values) - right):
        window = values[pivot_idx - left : pivot_idx + right + 1]
        if np.isfinite(values[pivot_idx]) and values[pivot_idx] >= np.nanmax(window):
            result[pivot_idx + right] = values[pivot_idx]
    return result


def _pivot_low(values: np.ndarray, left: int, right: int = 1) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if left <= 0 or len(values) <= left + right:
        return result
    for pivot_idx in range(left, len(values) - right):
        window = values[pivot_idx - left : pivot_idx + right + 1]
        if np.isfinite(values[pivot_idx]) and values[pivot_idx] <= np.nanmin(window):
            result[pivot_idx + right] = values[pivot_idx]
    return result


def _previous_occurrence(condition: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    last_value = np.nan
    for idx, flag in enumerate(condition):
        result[idx] = last_value
        if flag and np.isfinite(values[idx]):
            last_value = values[idx]
    return result


def estimate_warmup_bars(settings: StrategySettings) -> int:
    sensitivity_mult = SENSITIVITY_MULTIPLIERS.get(settings.sensitivity_mode, 1.0)
    zz_left = max(2, int(round(settings.zz_len_raw * sensitivity_mult)))
    lookbacks = [
        settings.atr_period + 2,
        settings.qip_rsi_len + 2,
        settings.vol_ma_len + 2,
        settings.qip_ema_slow + 2,
        zz_left * 2 + settings.atr_period + 5,
        settings.qtp_ema_slow_len + 2,
        settings.qtp_rsi_len + 2,
        settings.qtp_stoch_len + 2,
        settings.qtp_atr_len + 2,
        settings.qtp_dev_lookback * 2 + settings.qtp_atr_len + 5,
        settings.qtp_vol_len + 2,
        settings.qtp_max_pvt_left + 2,
    ]
    return max(max(lookbacks) * 3, 300)


def _cached(cache: Optional[Dict[Tuple[object, ...], object]], key: Tuple[object, ...], builder):
    if cache is None:
        return builder()
    if key not in cache:
        cache[key] = builder()
    return cache[key]


def _stream_indicator_rows(
    df: pd.DataFrame,
    settings: StrategySettings,
    cursor: Optional[IndicatorCursor] = None,
    start_index: int = 0,
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]], IndicatorCursor]:
    frame = _ensure_ohlcv(df)
    if frame.empty:
        return [], None, IndicatorCursor(payload=())
    state = _build_indicator_state(settings, cursor)
    constants = state["constants"]
    actual_start = max(int(start_index), int(state["processed_rows"]))
    rows: List[Dict[str, object]] = []
    latest_row: Optional[Dict[str, object]] = None
    time_values = pd.to_datetime(frame["time"]).to_numpy(copy=False)
    open_values = frame["open"].to_numpy(dtype=float, copy=False)
    high_values = frame["high"].to_numpy(dtype=float, copy=False)
    low_values = frame["low"].to_numpy(dtype=float, copy=False)
    close_values = frame["close"].to_numpy(dtype=float, copy=False)
    volume_values = frame["volume"].to_numpy(dtype=float, copy=False)
    quote_volume_values = frame["quote_volume"].to_numpy(dtype=float, copy=False) if "quote_volume" in frame.columns else None

    for index in range(actual_start, len(frame)):
        time_value = pd.Timestamp(time_values[index])
        open_value = float(open_values[index])
        high_value = float(high_values[index])
        low_value = float(low_values[index])
        close_value = float(close_values[index])
        volume_value = float(volume_values[index])
        prev_close = _coerce_float(state["prev_close"])
        prev_close_2 = _coerce_float(state["prev_close_2"])
        prev_high = _coerce_float(state["prev_high"])
        prev_low = _coerce_float(state["prev_low"])

        tr_candidates = [high_value - low_value]
        if math.isfinite(prev_close):
            tr_candidates.append(abs(high_value - prev_close))
            tr_candidates.append(abs(low_value - prev_close))
        tr_value = max(tr_candidates)

        atr_st = state["supertrend_atr"].update(tr_value)
        atr14 = state["atr14"].update(tr_value)
        qtp_atr = state["qtp_atr"].update(tr_value)

        hl2 = (high_value + low_value) / 2.0
        upperband = hl2 + (settings.factor * atr_st) if math.isfinite(atr_st) else float("nan")
        lowerband = hl2 - (settings.factor * atr_st) if math.isfinite(atr_st) else float("nan")

        prev_supertrend = _coerce_float(state["supertrend_value"])
        prev_direction = _coerce_float(state["supertrend_direction"])
        prev_final_upper = _coerce_float(state["supertrend_final_upper"])
        prev_final_lower = _coerce_float(state["supertrend_final_lower"])
        supertrend = float("nan")
        direction = float("nan")
        final_upper = upperband
        final_lower = lowerband
        if math.isfinite(atr_st):
            if not math.isfinite(prev_supertrend):
                supertrend = upperband
                direction = 1.0
            else:
                if not math.isfinite(prev_final_upper):
                    prev_final_upper = upperband
                if not math.isfinite(prev_final_lower):
                    prev_final_lower = lowerband
                final_upper = upperband if (upperband < prev_final_upper or (math.isfinite(prev_close) and prev_close > prev_final_upper)) else prev_final_upper
                final_lower = lowerband if (lowerband > prev_final_lower or (math.isfinite(prev_close) and prev_close < prev_final_lower)) else prev_final_lower
                if math.isfinite(prev_direction) and prev_direction > 0:
                    if close_value <= final_upper:
                        supertrend = final_upper
                        direction = 1.0
                    else:
                        supertrend = final_lower
                        direction = -1.0
                else:
                    if close_value >= final_lower:
                        supertrend = final_lower
                        direction = -1.0
                    else:
                        supertrend = final_upper
                        direction = 1.0
        state["supertrend_value"] = supertrend
        state["supertrend_direction"] = direction
        state["supertrend_final_upper"] = final_upper
        state["supertrend_final_lower"] = final_lower

        dist_atr = abs(close_value - supertrend) / atr_st if math.isfinite(supertrend) and math.isfinite(atr_st) and atr_st != 0 else float("nan")
        lev_zone = 3 if math.isfinite(dist_atr) and dist_atr < constants["zone3_thresh"] else (2 if math.isfinite(dist_atr) and dist_atr < constants["zone2_thresh"] else 1)
        is_long_trend = math.isfinite(direction) and direction < 0
        is_short_trend = math.isfinite(direction) and direction > 0
        trend_to_long = is_long_trend and not bool(state["prev_is_long_trend"])
        trend_to_short = is_short_trend and not bool(state["prev_is_short_trend"])

        ema12 = state["ema12"].update(close_value)
        ema26 = state["ema26"].update(close_value)
        macd_raw = (ema12 - ema26) if math.isfinite(ema12) and math.isfinite(ema26) else float("nan")
        qip_ema_fast = state["qip_ema_fast"].update(close_value)
        qip_ema_slow = state["qip_ema_slow"].update(close_value)
        qtp_ema_fast = state["qtp_ema_fast"].update(close_value)
        qtp_ema_slow = state["qtp_ema_slow"].update(close_value)
        qtp_basis = state["qtp_basis"].update(close_value)

        qip_rsi_full = float("nan")
        qtp_rsi = float("nan")
        if math.isfinite(prev_close):
            delta = close_value - prev_close
            avg_gain_qip = state["qip_rsi_gain"].update(max(delta, 0.0))
            avg_loss_qip = state["qip_rsi_loss"].update(max(-delta, 0.0))
            if math.isfinite(avg_gain_qip) and math.isfinite(avg_loss_qip):
                qip_rsi_full = _rsi_from_avgs(avg_gain_qip, avg_loss_qip)
            avg_gain_qtp = state["qtp_rsi_gain"].update(max(delta, 0.0))
            avg_loss_qtp = state["qtp_rsi_loss"].update(max(-delta, 0.0))
            if math.isfinite(avg_gain_qtp) and math.isfinite(avg_loss_qtp):
                qtp_rsi = _rsi_from_avgs(avg_gain_qtp, avg_loss_qtp)

        qip_rsi_val = _coerce_float(state["prev_qip_rsi"])
        vol_curr = _coerce_float(state["prev_volume"])
        vol_ma_prev = state["qip_volume_window"].mean(settings.vol_ma_len)
        qip_ema_f = _coerce_float(state["prev_qip_ema_fast"])
        qip_ema_s = _coerce_float(state["prev_qip_ema_slow"])
        macd_line = _coerce_float(state["prev_macd_raw"])
        low_roll_prev = state["qip_low_window"].min(1)
        high_roll_prev = state["qip_high_window"].max(1)
        atr14_prev = _coerce_float(state["prev_atr14"])

        pivot_h_value = state["qip_pivot_high"].update_high(high_value)
        pivot_l_value = state["qip_pivot_low"].update_low(low_value)
        ph_confirmed = math.isfinite(pivot_h_value)
        pl_confirmed = math.isfinite(pivot_l_value)
        ph_valid = ph_confirmed and math.isfinite(pivot_h_value) and math.isfinite(low_roll_prev) and math.isfinite(atr14_prev) and ((pivot_h_value - low_roll_prev) > (atr14_prev * constants["atr_mult"]))
        pl_valid = pl_confirmed and math.isfinite(pivot_l_value) and math.isfinite(high_roll_prev) and math.isfinite(atr14_prev) and ((high_roll_prev - pivot_l_value) > (atr14_prev * constants["atr_mult"]))

        qip_rsi_bull_ok = True if not settings.qip_use_rsi_zone else (math.isfinite(qip_rsi_val) and qip_rsi_val <= settings.qip_rsi_bull_max)
        qip_rsi_bear_ok = True if not settings.qip_use_rsi_zone else (math.isfinite(qip_rsi_val) and qip_rsi_val >= settings.qip_rsi_bear_min)

        bull_div = pl_valid and math.isfinite(qip_rsi_val) and math.isfinite(state["last_pl_rsi"]) and math.isfinite(pivot_l_value) and math.isfinite(state["last_pl_val"]) and (qip_rsi_val > state["last_pl_rsi"]) and (pivot_l_value < state["last_pl_val"])
        bear_div = ph_valid and math.isfinite(qip_rsi_val) and math.isfinite(state["last_ph_rsi"]) and math.isfinite(pivot_h_value) and math.isfinite(state["last_ph_val"]) and (qip_rsi_val < state["last_ph_rsi"]) and (pivot_h_value > state["last_ph_val"])
        bull_macd_div = pl_valid and math.isfinite(macd_line) and math.isfinite(state["last_pl_macd"]) and math.isfinite(pivot_l_value) and math.isfinite(state["last_pl_val"]) and (macd_line > state["last_pl_macd"]) and (pivot_l_value < state["last_pl_val"])
        bear_macd_div = ph_valid and math.isfinite(macd_line) and math.isfinite(state["last_ph_macd"]) and math.isfinite(pivot_h_value) and math.isfinite(state["last_ph_val"]) and (macd_line < state["last_ph_macd"]) and (pivot_h_value > state["last_ph_val"])

        bull_score = float(pl_valid)
        bear_score = float(ph_valid)
        if settings.use_rsi_div:
            bull_score += float(bull_div)
            bear_score += float(bear_div)
        if settings.use_macd_div:
            bull_score += float(bull_macd_div)
            bear_score += float(bear_macd_div)
        if settings.use_volume:
            bull_score += float(math.isfinite(vol_curr) and math.isfinite(vol_ma_prev) and vol_curr > (vol_ma_prev * 1.2))
            bear_score += float(math.isfinite(vol_curr) and math.isfinite(vol_ma_prev) and vol_curr > (vol_ma_prev * 1.2))
        if settings.use_ema_conf:
            bull_score += float(math.isfinite(qip_ema_f) and math.isfinite(qip_ema_s) and qip_ema_f > qip_ema_s)
            bear_score += float(math.isfinite(qip_ema_f) and math.isfinite(qip_ema_s) and qip_ema_f < qip_ema_s)
        qip_final_bull = bool(settings.use_qip) and pl_valid and bull_score >= settings.min_score and qip_rsi_bull_ok
        qip_final_bear = bool(settings.use_qip) and ph_valid and bear_score >= settings.min_score and qip_rsi_bear_ok

        if pl_confirmed and math.isfinite(qip_rsi_val):
            state["last_pl_rsi"] = qip_rsi_val
        if ph_confirmed and math.isfinite(qip_rsi_val):
            state["last_ph_rsi"] = qip_rsi_val
        if pl_confirmed and math.isfinite(pivot_l_value):
            state["last_pl_val"] = pivot_l_value
        if ph_confirmed and math.isfinite(pivot_h_value):
            state["last_ph_val"] = pivot_h_value
        if pl_confirmed and math.isfinite(macd_line):
            state["last_pl_macd"] = macd_line
        if ph_confirmed and math.isfinite(macd_line):
            state["last_ph_macd"] = macd_line

        state["qip_low_window"].push(low_value)
        state["qip_high_window"].push(high_value)
        state["qip_volume_window"].push(volume_value)

        state["qtp_stoch_high_window"].push(high_value)
        state["qtp_stoch_low_window"].push(low_value)
        stoch_high = state["qtp_stoch_high_window"].max(settings.qtp_stoch_len)
        stoch_low = state["qtp_stoch_low_window"].min(settings.qtp_stoch_len)
        qtp_stoch = ((close_value - stoch_low) / (stoch_high - stoch_low) * 100.0) if math.isfinite(stoch_high) and math.isfinite(stoch_low) and stoch_high != stoch_low else float("nan")

        state["qtp_volume_window"].push(volume_value)
        qtp_vol_sma = state["qtp_volume_window"].mean(settings.qtp_vol_len)
        qtp_dev = ((close_value - qtp_basis) / qtp_atr) if math.isfinite(qtp_basis) and math.isfinite(qtp_atr) and qtp_atr != 0 else float("nan")
        state["qtp_dev_window"].push(qtp_dev)
        qtp_d_mean = state["qtp_dev_window"].mean(settings.qtp_dev_lookback)
        qtp_d_std = state["qtp_dev_window"].std(settings.qtp_dev_lookback, ddof=1)
        qtp_zdev = ((qtp_dev - qtp_d_mean) / qtp_d_std) if math.isfinite(qtp_dev) and math.isfinite(qtp_d_mean) and math.isfinite(qtp_d_std) and qtp_d_std != 0 else float("nan")

        qtp_body = abs(close_value - open_value)
        qtp_range = high_value - low_value
        qtp_upper_wick = high_value - max(open_value, close_value)
        qtp_lower_wick = min(open_value, close_value) - low_value
        qtp_bull_rev = qtp_range > 0 and qtp_lower_wick > (qtp_body * 1.2) and close_value > open_value
        qtp_bear_rev = qtp_range > 0 and qtp_upper_wick > (qtp_body * 1.2) and close_value < open_value
        qtp_vol_spike = math.isfinite(qtp_vol_sma) and volume_value > (qtp_vol_sma * (1.1 + (0.5 * constants["qtp_sens"])))
        qtp_up_trend = math.isfinite(qtp_ema_fast) and math.isfinite(qtp_ema_slow) and qtp_ema_fast > qtp_ema_slow
        qtp_dn_trend = math.isfinite(qtp_ema_fast) and math.isfinite(qtp_ema_slow) and qtp_ema_fast < qtp_ema_slow
        qtp_roc1 = (close_value - prev_close) if math.isfinite(prev_close) else float("nan")
        qtp_roc2 = (prev_close - prev_close_2) if math.isfinite(prev_close) and math.isfinite(prev_close_2) else float("nan")

        qtp_bot_score = 0.0
        qtp_top_score = 0.0
        qtp_bot_score += 18.0 if math.isfinite(state["prev_qtp_rsi"]) and state["prev_qtp_rsi"] < constants["qtp_rsi_lower"] else 0.0
        qtp_top_score += 18.0 if math.isfinite(state["prev_qtp_rsi"]) and state["prev_qtp_rsi"] > constants["qtp_rsi_upper"] else 0.0
        qtp_bot_score += 12.0 if math.isfinite(state["prev_qtp_stoch"]) and state["prev_qtp_stoch"] < constants["qtp_stoch_lower"] else 0.0
        qtp_top_score += 12.0 if math.isfinite(state["prev_qtp_stoch"]) and state["prev_qtp_stoch"] > constants["qtp_stoch_upper"] else 0.0
        qtp_bot_score += 22.0 if math.isfinite(state["prev_qtp_zdev"]) and state["prev_qtp_zdev"] < -(1.0 + ((1.0 - constants["qtp_sens"]) * 0.8)) else 0.0
        qtp_top_score += 22.0 if math.isfinite(state["prev_qtp_zdev"]) and state["prev_qtp_zdev"] > (1.0 + ((1.0 - constants["qtp_sens"]) * 0.8)) else 0.0
        qtp_bot_score += 16.0 if state["prev_qtp_bull_rev"] else 0.0
        qtp_top_score += 16.0 if state["prev_qtp_bear_rev"] else 0.0
        qtp_bot_score += 10.0 if state["prev_qtp_vol_spike"] and math.isfinite(prev_close) and math.isfinite(prev_low) and prev_close > prev_low else 0.0
        qtp_top_score += 10.0 if state["prev_qtp_vol_spike"] and math.isfinite(prev_close) and math.isfinite(prev_high) and prev_close < prev_high else 0.0
        qtp_bot_score += 10.0 if math.isfinite(state["prev_qtp_roc2"]) and math.isfinite(state["prev_qtp_roc1"]) and state["prev_qtp_roc2"] < 0 and state["prev_qtp_roc1"] > state["prev_qtp_roc2"] else 0.0
        qtp_top_score += 10.0 if math.isfinite(state["prev_qtp_roc2"]) and math.isfinite(state["prev_qtp_roc1"]) and state["prev_qtp_roc2"] > 0 and state["prev_qtp_roc1"] < state["prev_qtp_roc2"] else 0.0
        if settings.qtp_use_trend:
            qtp_bot_score += 8.0 if state["prev_qtp_up_trend"] else 0.0
            qtp_top_score += 8.0 if state["prev_qtp_dn_trend"] else 0.0
        qtp_bot_score = min(qtp_bot_score, 100.0)
        qtp_top_score = min(qtp_top_score, 100.0)

        qtp_rsi_bull_ok = True if not settings.qtp_use_rsi_zone else (math.isfinite(state["prev_qtp_rsi"]) and state["prev_qtp_rsi"] <= settings.qtp_rsi_bull_max)
        qtp_rsi_bear_ok = True if not settings.qtp_use_rsi_zone else (math.isfinite(state["prev_qtp_rsi"]) and state["prev_qtp_rsi"] >= settings.qtp_rsi_bear_min)
        qtp_p_low = state["qtp_pivot_low"].update_low(low_value)
        qtp_p_high = state["qtp_pivot_high"].update_high(high_value)
        qtp_p_low_confirmed = math.isfinite(qtp_p_low)
        qtp_p_high_confirmed = math.isfinite(qtp_p_high)
        qtp_final_bull = bool(settings.use_qtp) and qtp_p_low_confirmed and qtp_bot_score >= constants["qtp_score_thr"] and qtp_rsi_bull_ok
        qtp_final_bear = bool(settings.use_qtp) and qtp_p_high_confirmed and qtp_top_score >= constants["qtp_score_thr"] and qtp_rsi_bear_ok

        final_bull = bool(qip_final_bull or qtp_final_bull)
        final_bear = bool(qip_final_bear or qtp_final_bear)
        sign = 1.0 if is_long_trend else -1.0
        zone2_line = supertrend + (sign * constants["zone2_thresh"] * atr_st) if math.isfinite(supertrend) and math.isfinite(atr_st) else float("nan")
        zone3_line = supertrend + (sign * constants["zone3_thresh"] * atr_st) if math.isfinite(supertrend) and math.isfinite(atr_st) else float("nan")

        row: Dict[str, object] = {
            "time": time_value,
            "open": open_value,
            "high": high_value,
            "low": low_value,
            "close": close_value,
            "volume": volume_value,
            "supertrend": supertrend,
            "direction": direction,
            "dist_atr": dist_atr,
            "lev_zone": lev_zone,
            "zone2_line": zone2_line,
            "zone3_line": zone3_line,
            "is_long_trend": bool(is_long_trend),
            "is_short_trend": bool(is_short_trend),
            "trend_to_long": bool(trend_to_long),
            "trend_to_short": bool(trend_to_short),
            "qip_final_bull": bool(qip_final_bull),
            "qip_final_bear": bool(qip_final_bear),
            "qtp_final_bull": bool(qtp_final_bull),
            "qtp_final_bear": bool(qtp_final_bear),
            "final_bull": bool(final_bull),
            "final_bear": bool(final_bear),
            "bull_score": bull_score,
            "bear_score": bear_score,
            "qtp_bot_score": qtp_bot_score,
            "qtp_top_score": qtp_top_score,
            "ema_fast": qip_ema_fast,
            "ema_slow": qip_ema_slow,
            "rsi": qip_rsi_full,
        }
        if quote_volume_values is not None:
            row["quote_volume"] = float(quote_volume_values[index])
        rows.append(row)
        latest_row = row

        state["processed_rows"] = int(index + 1)
        state["prev_close_2"] = prev_close
        state["prev_close"] = close_value
        state["prev_high"] = high_value
        state["prev_low"] = low_value
        state["prev_is_long_trend"] = bool(is_long_trend)
        state["prev_is_short_trend"] = bool(is_short_trend)
        state["prev_qip_rsi"] = qip_rsi_full
        state["prev_qip_vol_ma"] = state["qip_volume_window"].mean(settings.vol_ma_len)
        state["prev_volume"] = volume_value
        state["prev_qip_ema_fast"] = qip_ema_fast
        state["prev_qip_ema_slow"] = qip_ema_slow
        state["prev_macd_raw"] = macd_raw
        state["prev_qip_low_roll"] = state["qip_low_window"].min(1)
        state["prev_qip_high_roll"] = state["qip_high_window"].max(1)
        state["prev_atr14"] = atr14
        state["prev_qtp_rsi"] = qtp_rsi
        state["prev_qtp_stoch"] = qtp_stoch
        state["prev_qtp_zdev"] = qtp_zdev
        state["prev_qtp_bull_rev"] = bool(qtp_bull_rev)
        state["prev_qtp_bear_rev"] = bool(qtp_bear_rev)
        state["prev_qtp_vol_spike"] = bool(qtp_vol_spike)
        state["prev_qtp_up_trend"] = bool(qtp_up_trend)
        state["prev_qtp_dn_trend"] = bool(qtp_dn_trend)
        state["prev_qtp_roc1"] = qtp_roc1
        state["prev_qtp_roc2"] = qtp_roc2

    return rows, latest_row, _snapshot_indicator_state(state)


def evaluate_latest_state(
    df: pd.DataFrame,
    settings: StrategySettings,
    cursor: Optional[IndicatorCursor] = None,
) -> Tuple[Dict[str, object], Optional[IndicatorCursor]]:
    rows, latest_row, latest_cursor = _stream_indicator_rows(df, settings, cursor=cursor)
    if latest_row is None:
        return {}, latest_cursor
    latest_series = pd.Series(latest_row)
    return _build_latest_state(latest_series), latest_cursor


def _indicator_cursor_processed_rows(cursor: Optional[IndicatorCursor]) -> int:
    if cursor is None:
        return 0
    for key, value in cursor.payload:
        if key == "processed_rows":
            return int(value or 0)
    return 0


def _empty_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    indicators = frame.iloc[0:0].copy()
    extra_columns = [
        "supertrend",
        "direction",
        "dist_atr",
        "lev_zone",
        "zone2_line",
        "zone3_line",
        "is_long_trend",
        "is_short_trend",
        "trend_to_long",
        "trend_to_short",
        "qip_final_bull",
        "qip_final_bear",
        "qtp_final_bull",
        "qtp_final_bear",
        "final_bull",
        "final_bear",
        "bull_score",
        "bear_score",
        "qtp_bot_score",
        "qtp_top_score",
        "ema_fast",
        "ema_slow",
        "rsi",
    ]
    for column in extra_columns:
        if column not in indicators.columns:
            indicators[column] = pd.Series(dtype=float)
    return indicators


def _stream_indicator_frame(
    df: pd.DataFrame,
    settings: StrategySettings,
    cursor: Optional[IndicatorCursor] = None,
    start_index: int = 0,
) -> Tuple[pd.DataFrame, IndicatorCursor]:
    frame = _ensure_ohlcv(df)
    rows, _, indicator_cursor = _stream_indicator_rows(frame, settings, cursor=cursor, start_index=start_index)
    if rows:
        indicators = pd.DataFrame(rows)
    else:
        indicators = _empty_indicator_frame(frame)
    indicators.attrs[PREPARED_OHLCV_ATTR] = True
    indicators.attrs["indicator_cursor"] = indicator_cursor
    return indicators, indicator_cursor


def compute_indicators(
    df: pd.DataFrame,
    settings: StrategySettings,
    cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> pd.DataFrame:
    frame = _ensure_ohlcv(df)
    cache_key = ("stream_indicators", id(frame), settings)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, pd.DataFrame):
            return cached.copy()
    indicators, indicator_cursor = _stream_indicator_frame(frame, settings)
    indicators.attrs["indicator_cursor"] = indicator_cursor
    if cache is not None:
        cache[cache_key] = indicators.copy()
    return indicators


def _normalize_backtest_start_time(backtest_start_time: Optional[pd.Timestamp | str]) -> Optional[pd.Timestamp]:
    if backtest_start_time is None:
        return None
    timestamp = pd.Timestamp(backtest_start_time)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _select_active_indicators(
    indicators: pd.DataFrame,
    backtest_start_time: Optional[pd.Timestamp | str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_start = _normalize_backtest_start_time(backtest_start_time)
    if test_start is None:
        return indicators, indicators
    times = indicators["time"].to_numpy(dtype="datetime64[ns]", copy=False)
    start_index = int(times.searchsorted(test_start.to_datetime64(), side="left"))
    active_indicators = indicators.iloc[start_index:].reset_index(drop=True)
    if active_indicators.empty:
        return active_indicators, indicators.tail(1).reset_index(drop=True)
    return active_indicators, active_indicators


def _build_latest_state(latest: pd.Series) -> Dict[str, object]:
    lev_zone_raw = latest.get("lev_zone", 0)
    lev_zone = int(lev_zone_raw) if pd.notna(lev_zone_raw) else 0
    return {
        "trend": "LONG" if bool(latest["is_long_trend"]) else "SHORT",
        "zone": lev_zone,
        "trend_to_long": bool(latest["trend_to_long"]),
        "trend_to_short": bool(latest["trend_to_short"]),
        "final_bull": bool(latest["final_bull"]),
        "final_bear": bool(latest["final_bear"]),
        "qip_bull": bool(latest["qip_final_bull"]),
        "qip_bear": bool(latest["qip_final_bear"]),
        "qtp_bull": bool(latest["qtp_final_bull"]),
        "qtp_bear": bool(latest["qtp_final_bear"]),
        "rsi": float(latest["rsi"]) if np.isfinite(latest["rsi"]) else float("nan"),
        "dist_atr": float(latest["dist_atr"]) if np.isfinite(latest["dist_atr"]) else float("nan"),
    }


def _run_backtest_core(
    indicators: pd.DataFrame,
    settings: StrategySettings,
    fee_rate: float,
    starting_equity: float,
    backtest_start_time: Optional[pd.Timestamp | str],
    include_details: bool,
    resume_cursor: Optional[BacktestCursor] = None,
    existing_trades: Optional[List[TradeRecord]] = None,
    existing_equity_curve: Optional[pd.Series] = None,
    indicator_cursor: Optional[IndicatorCursor] = None,
) -> Tuple[StrategyMetrics, List[TradeRecord], pd.Series, pd.DataFrame, Dict[str, object], Optional[BacktestCursor]]:
    latest = indicators.iloc[-1]
    active_indicators, result_indicators = _select_active_indicators(indicators, backtest_start_time)

    active_count = len(active_indicators)
    close_values = active_indicators["close"].to_numpy(dtype=float, copy=False)
    time_values = active_indicators["time"].to_numpy(copy=False)
    lev_zone_values = active_indicators["lev_zone"].fillna(0.0).to_numpy(dtype=int, copy=False)
    is_long_trend_values = active_indicators["is_long_trend"].to_numpy(dtype=bool, copy=False)
    is_short_trend_values = active_indicators["is_short_trend"].to_numpy(dtype=bool, copy=False)
    trend_to_long_values = active_indicators["trend_to_long"].to_numpy(dtype=bool, copy=False)
    trend_to_short_values = active_indicators["trend_to_short"].to_numpy(dtype=bool, copy=False)
    final_bull_values = active_indicators["final_bull"].to_numpy(dtype=bool, copy=False)
    final_bear_values = active_indicators["final_bear"].to_numpy(dtype=bool, copy=False)

    start_index = min(max(int(resume_cursor.processed_bars), 0), active_count) if resume_cursor is not None else 0
    if resume_cursor is None:
        equity = float(starting_equity)
        position_qty = 0.0
        avg_entry_price = 0.0
        entry_side = ""
        entry_time = pd.Timestamp(result_indicators["time"].iloc[0])
        entry_price = 0.0
        zone_events: List[str] = []
        trades: List[TradeRecord] = []
        gross_profit = 0.0
        gross_loss = 0.0
        trade_count = 0
        win_count = 0
        long_zone_used = [False, False, False]
        short_zone_used = [False, False, False]
        last_long_zone = 0
        last_short_zone = 0
    else:
        equity = float(resume_cursor.equity)
        position_qty = float(resume_cursor.position_qty)
        avg_entry_price = float(resume_cursor.avg_entry_price)
        entry_side = str(resume_cursor.entry_side)
        entry_time = pd.Timestamp(resume_cursor.entry_time)
        entry_price = float(resume_cursor.entry_price)
        zone_events = list(resume_cursor.zone_events)
        trades = list(existing_trades or [])
        gross_profit = float(resume_cursor.gross_profit)
        gross_loss = float(resume_cursor.gross_loss)
        trade_count = int(resume_cursor.trade_count)
        win_count = int(resume_cursor.win_count)
        long_zone_used = list(resume_cursor.long_zone_used)
        short_zone_used = list(resume_cursor.short_zone_used)
        last_long_zone = int(resume_cursor.last_long_zone)
        last_short_zone = int(resume_cursor.last_short_zone)
    equity_curve_values = np.empty(active_count, dtype=float) if active_count else np.empty(0, dtype=float)
    if start_index > 0 and existing_equity_curve is not None and len(existing_equity_curve) >= start_index:
        equity_curve_values[:start_index] = existing_equity_curve.iloc[:start_index].to_numpy(dtype=float, copy=False)
    def reset_zones() -> None:
        nonlocal last_long_zone, last_short_zone
        long_zone_used[:] = [False, False, False]
        short_zone_used[:] = [False, False, False]
        last_long_zone = 0
        last_short_zone = 0

    def add_position(side: str, price: float, time_value: pd.Timestamp, zone: int) -> None:
        nonlocal equity, position_qty, avg_entry_price, entry_side, entry_time, entry_price, last_long_zone, last_short_zone
        allocation_pct = settings.entry_size_pct / 100.0
        if settings.beast_mode and zone in (2, 3):
            if zone == 2:
                allocation_pct = 0.5
            elif zone == 3 and ((side == "long" and last_long_zone == 0) or (side == "short" and last_short_zone == 0)):
                allocation_pct = 1.0
            else:
                allocation_pct = 0.5

        qty = (equity * allocation_pct) / max(price, 1e-9)
        equity -= qty * price * fee_rate
        signed_qty = qty if side == "long" else -qty

        if abs(position_qty) < 1e-12:
            position_qty = signed_qty
            avg_entry_price = price
            entry_side = side
            entry_time = time_value
            entry_price = price
            if include_details:
                zone_events[:] = [f"{side[0].upper()}{zone}"]
        else:
            total_qty = abs(position_qty) + qty
            avg_entry_price = ((avg_entry_price * abs(position_qty)) + (price * qty)) / max(total_qty, 1e-12)
            position_qty += signed_qty
            if include_details:
                zone_events.append(f"{side[0].upper()}{zone}")

        if side == "long":
            long_zone_used[zone - 1] = True
            last_long_zone = zone
        else:
            short_zone_used[zone - 1] = True
            last_short_zone = zone

    def close_position(price: float, time_value: pd.Timestamp, reason: str) -> None:
        nonlocal equity, position_qty, avg_entry_price, entry_side, entry_time, entry_price
        nonlocal gross_profit, gross_loss, trade_count, win_count
        if abs(position_qty) < 1e-12 or not entry_side:
            return
        pnl = position_qty * (price - avg_entry_price)
        fee = abs(position_qty) * price * fee_rate
        net_pnl = pnl - fee
        equity += net_pnl
        trade_count += 1
        if net_pnl > 0:
            gross_profit += net_pnl
            win_count += 1
        else:
            gross_loss += abs(net_pnl)
        if include_details:
            gross_cost = abs(position_qty) * avg_entry_price
            return_pct = (pnl / gross_cost * 100.0) if gross_cost else 0.0
            trades.append(
                TradeRecord(
                    side=entry_side,
                    entry_time=entry_time,
                    exit_time=time_value,
                    entry_price=float(entry_price),
                    exit_price=float(price),
                    quantity=float(abs(position_qty)),
                    pnl=float(net_pnl),
                    return_pct=float(return_pct),
                    reason=reason,
                    zones=",".join(zone_events),
                )
            )
            zone_events.clear()
        position_qty = 0.0
        avg_entry_price = 0.0
        entry_side = ""
        entry_price = 0.0
        reset_zones()

    for index in range(start_index, active_count):
        current_price = close_values[index]
        current_time = pd.Timestamp(time_values[index])

        if abs(position_qty) < 1e-12:
            reset_zones()

        if trend_to_short_values[index] and position_qty > 0:
            close_position(current_price, current_time, "trend_to_short")
        if trend_to_long_values[index] and position_qty < 0:
            close_position(current_price, current_time, "trend_to_long")
        if final_bear_values[index] and position_qty > 0:
            close_position(current_price, current_time, "opposite_signal")
        if final_bull_values[index] and position_qty < 0:
            close_position(current_price, current_time, "opposite_signal")

        if abs(position_qty) < 1e-12:
            reset_zones()

        lev_zone = lev_zone_values[index]
        can_long_z1 = (not settings.beast_mode) and is_long_trend_values[index] and final_bull_values[index] and lev_zone == 1 and (not long_zone_used[0]) and last_long_zone == 0
        can_long_z2 = is_long_trend_values[index] and final_bull_values[index] and lev_zone == 2 and (not long_zone_used[1]) and last_long_zone in (0, 1)
        can_long_z3 = is_long_trend_values[index] and final_bull_values[index] and lev_zone == 3 and (not long_zone_used[2]) and last_long_zone in (0, 2)
        can_short_z1 = (not settings.beast_mode) and is_short_trend_values[index] and final_bear_values[index] and lev_zone == 1 and (not short_zone_used[0]) and last_short_zone == 0
        can_short_z2 = is_short_trend_values[index] and final_bear_values[index] and lev_zone == 2 and (not short_zone_used[1]) and last_short_zone in (0, 1)
        can_short_z3 = is_short_trend_values[index] and final_bear_values[index] and lev_zone == 3 and (not short_zone_used[2]) and last_short_zone in (0, 2)

        if can_long_z1:
            add_position("long", current_price, current_time, 1)
        if can_long_z2:
            add_position("long", current_price, current_time, 2)
        if can_long_z3:
            add_position("long", current_price, current_time, 3)
        if can_short_z1:
            add_position("short", current_price, current_time, 1)
        if can_short_z2:
            add_position("short", current_price, current_time, 2)
        if can_short_z3:
            add_position("short", current_price, current_time, 3)

        unrealized = position_qty * (current_price - avg_entry_price) if abs(position_qty) > 1e-12 else 0.0
        equity_curve_values[index] = equity + unrealized

    cursor: Optional[BacktestCursor] = None
    if active_count:
        last_equity_value = (
            float(resume_cursor.last_equity_value)
            if resume_cursor is not None and start_index >= active_count
            else float(equity_curve_values[active_count - 1])
        )
        cursor = BacktestCursor(
            processed_bars=active_count,
            last_time=pd.Timestamp(time_values[active_count - 1]),
            equity=float(equity),
            position_qty=float(position_qty),
            avg_entry_price=float(avg_entry_price),
            entry_side=str(entry_side),
            entry_time=pd.Timestamp(entry_time),
            entry_price=float(entry_price),
            zone_events=tuple(zone_events),
            gross_profit=float(gross_profit),
            gross_loss=float(gross_loss),
            trade_count=int(trade_count),
            win_count=int(win_count),
            long_zone_used=tuple(bool(flag) for flag in long_zone_used),
            short_zone_used=tuple(bool(flag) for flag in short_zone_used),
            last_long_zone=int(last_long_zone),
            last_short_zone=int(last_short_zone),
            last_equity_value=last_equity_value,
            indicator_cursor=indicator_cursor,
        )

    if abs(position_qty) > 1e-12 and active_count:
        close_position(close_values[-1], pd.Timestamp(time_values[-1]), "end_of_test")
        equity_curve_values[-1] = equity

    if equity_curve_values.size:
        peaks = np.maximum.accumulate(equity_curve_values)
        drawdown_pct = np.divide(
            equity_curve_values - peaks,
            peaks,
            out=np.zeros_like(equity_curve_values),
            where=peaks != 0.0,
        ) * 100.0
        max_drawdown_pct = abs(float(drawdown_pct.min()))
    else:
        max_drawdown_pct = 0.0

    win_rate = (win_count / trade_count * 100.0) if trade_count else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf" if gross_profit > 0 else 0.0)

    if include_details:
        if equity_curve_values.size:
            curve = pd.Series(equity_curve_values, index=active_indicators["time"].iloc[: len(equity_curve_values)], dtype=float)
        else:
            curve = pd.Series([starting_equity], index=[result_indicators["time"].iloc[0]], dtype=float)
    else:
        curve = pd.Series(dtype=float)

    metrics = StrategyMetrics(
        total_return_pct=((equity - starting_equity) / starting_equity) * 100.0,
        net_profit=equity - starting_equity,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=trade_count,
        win_rate_pct=win_rate,
        profit_factor=float(profit_factor if math.isfinite(profit_factor) else 999.0),
    )
    latest_state = _build_latest_state(latest) if include_details else {}
    indicator_frame = compact_indicator_frame(result_indicators, RESULT_INDICATOR_COLUMNS) if include_details else result_indicators.tail(1)
    return metrics, trades, curve, indicator_frame, latest_state, cursor


def run_backtest_metrics(
    df: pd.DataFrame,
    settings: StrategySettings,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
    backtest_start_time: Optional[pd.Timestamp | str] = None,
    indicator_cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> StrategyMetrics:
    indicators = compute_indicators(df, settings, cache=indicator_cache)
    metrics, _, _, _, _, _ = _run_backtest_core(
        indicators,
        settings=settings,
        fee_rate=fee_rate,
        starting_equity=starting_equity,
        backtest_start_time=backtest_start_time,
        include_details=False,
    )
    return metrics


def run_backtest(
    df: pd.DataFrame,
    settings: StrategySettings,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
    backtest_start_time: Optional[pd.Timestamp | str] = None,
    indicator_cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> BacktestResult:
    indicators = compute_indicators(df, settings, cache=indicator_cache)
    indicator_cursor = indicators.attrs.get("indicator_cursor")
    metrics, trades, curve, result_indicators, latest_state, cursor = _run_backtest_core(
        indicators,
        settings=settings,
        fee_rate=fee_rate,
        starting_equity=starting_equity,
        backtest_start_time=backtest_start_time,
        include_details=True,
        indicator_cursor=indicator_cursor,
    )
    return BacktestResult(
        settings=settings,
        metrics=metrics,
        trades=trades,
        indicators=result_indicators,
        latest_state=latest_state,
        equity_curve=curve,
        cursor=cursor,
        history_signature=_indicator_history_signature(result_indicators),
    )


def resume_backtest(
    df: pd.DataFrame,
    previous_result: Optional[BacktestResult],
    settings: StrategySettings,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
    backtest_start_time: Optional[pd.Timestamp | str] = None,
    indicator_cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> BacktestResult:
    if previous_result is None or previous_result.cursor is None or previous_result.settings != settings:
        return run_backtest(
            df,
            settings=settings,
            fee_rate=fee_rate,
            starting_equity=starting_equity,
            backtest_start_time=backtest_start_time,
            indicator_cache=indicator_cache,
        )
    previous_indicators = previous_result.indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    previous_indicator_cursor = previous_result.cursor.indicator_cursor
    full_history = _ensure_ohlcv(df)
    previous_processed_rows = _indicator_cursor_processed_rows(previous_indicator_cursor)
    if previous_indicators.empty or previous_indicator_cursor is None or previous_processed_rows <= 0 or len(full_history) <= previous_processed_rows:
        return run_backtest(
            df,
            settings=settings,
            fee_rate=fee_rate,
            starting_equity=starting_equity,
            backtest_start_time=backtest_start_time,
            indicator_cache=indicator_cache,
        )
    prefix_last = full_history.iloc[previous_processed_rows - 1]
    previous_last = previous_indicators.iloc[-1]
    compare_columns = ["time", "open", "high", "low", "close", "volume"]
    prefix_matches = True
    for column in compare_columns:
        if column not in previous_last.index:
            continue
        left = prefix_last[column]
        right = previous_last[column]
        if column == "time":
            if pd.Timestamp(left) != pd.Timestamp(right):
                prefix_matches = False
                break
        elif not np.isclose(float(left), float(right), equal_nan=True, atol=1e-12):
            prefix_matches = False
            break
    if not prefix_matches:
        return run_backtest(
            df,
            settings=settings,
            fee_rate=fee_rate,
            starting_equity=starting_equity,
            backtest_start_time=backtest_start_time,
            indicator_cache=indicator_cache,
        )

    new_indicators, indicator_cursor = _stream_indicator_frame(
        full_history,
        settings,
        cursor=previous_indicator_cursor,
        start_index=previous_processed_rows,
    )
    if new_indicators.empty:
        return previous_result

    indicators = pd.concat([previous_indicators, new_indicators], ignore_index=True)
    indicators = indicators.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    indicators.attrs["indicator_cursor"] = indicator_cursor

    existing_curve = previous_result.equity_curve.copy()
    if not existing_curve.empty:
        existing_curve = existing_curve.iloc[: len(previous_indicators)].copy()
        existing_curve.iloc[-1] = previous_result.cursor.last_equity_value
    metrics, trades, curve, result_indicators, latest_state, cursor = _run_backtest_core(
        indicators,
        settings=settings,
        fee_rate=fee_rate,
        starting_equity=starting_equity,
        backtest_start_time=backtest_start_time,
        include_details=True,
        resume_cursor=previous_result.cursor,
        existing_trades=_strip_provisional_trade(previous_result.trades, previous_result.cursor),
        existing_equity_curve=existing_curve,
        indicator_cursor=indicator_cursor,
    )
    return BacktestResult(
        settings=settings,
        metrics=metrics,
        trades=trades,
        indicators=result_indicators,
        latest_state=latest_state,
        equity_curve=curve,
        cursor=cursor,
        history_signature=_indicator_history_signature(result_indicators),
    )
