from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import pandas_ta as pta

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
class BacktestResult:
    settings: StrategySettings
    metrics: StrategyMetrics
    trades: List[TradeRecord]
    indicators: pd.DataFrame
    latest_state: Dict[str, object]
    equity_curve: pd.Series


def compact_indicator_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in df.columns]
    if not selected:
        return df.tail(1).copy()
    return df.loc[:, selected].copy()


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


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    rsi = pta.rsi(series.astype(float), length=length, talib=False)
    if rsi is None:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return rsi.reindex(series.index)


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


def compute_indicators(
    df: pd.DataFrame,
    settings: StrategySettings,
    cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> pd.DataFrame:
    frame = _ensure_ohlcv(df)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    volume = frame["volume"]

    supertrend, direction, atr_st = _cached(
        cache,
        ("supertrend", settings.atr_period, settings.factor),
        lambda: _supertrend(frame, settings.atr_period, settings.factor),
    )
    dist_atr = (close - supertrend).abs() / atr_st.replace(0.0, np.nan)
    zone3_thresh = 0.5 * settings.zone_sensitivity
    zone2_thresh = 1.0 * settings.zone_sensitivity
    lev_zone = np.where(dist_atr < zone3_thresh, 3, np.where(dist_atr < zone2_thresh, 2, 1))

    is_long_trend = direction < 0
    is_short_trend = direction > 0
    trend_to_long = is_long_trend & ~is_long_trend.shift(1, fill_value=False)
    trend_to_short = is_short_trend & ~is_short_trend.shift(1, fill_value=False)

    sens_mult = SENSITIVITY_MULTIPLIERS.get(settings.sensitivity_mode, 1.0)
    zz_left = max(2, int(round(settings.zz_len_raw * sens_mult)))
    atr_mult = settings.atr_mult_raw / sens_mult
    atr_val = _cached(cache, ("atr", 14), lambda: _atr(high, low, close, 14))
    macd_raw = _cached(cache, ("macd_line",), lambda: _macd_line(close))

    pivot_h_series = _cached(
        cache,
        ("pivot_high_series", zz_left, 1),
        lambda: pd.Series(_pivot_high(high.to_numpy(dtype=float), zz_left, 1), index=frame.index),
    )
    pivot_l_series = _cached(
        cache,
        ("pivot_low_series", zz_left, 1),
        lambda: pd.Series(_pivot_low(low.to_numpy(dtype=float), zz_left, 1), index=frame.index),
    )
    ph_confirmed = pivot_h_series.notna()
    pl_confirmed = pivot_l_series.notna()

    lowest_low_prev = _cached(
        cache,
        ("rolling_low_shift", zz_left * 2),
        lambda: low.rolling(zz_left * 2, min_periods=1).min().shift(1),
    )
    highest_high_prev = _cached(
        cache,
        ("rolling_high_shift", zz_left * 2),
        lambda: high.rolling(zz_left * 2, min_periods=1).max().shift(1),
    )
    ph_valid = ph_confirmed & ((pivot_h_series - lowest_low_prev) > (atr_val.shift(1) * atr_mult))
    pl_valid = pl_confirmed & ((highest_high_prev - pivot_l_series) > (atr_val.shift(1) * atr_mult))

    qip_rsi_full = _cached(cache, ("rsi", settings.qip_rsi_len), lambda: _rsi(close, settings.qip_rsi_len))
    qip_rsi_val = qip_rsi_full.shift(1)
    vol_ma = _cached(
        cache,
        ("volume_mean_shift", settings.vol_ma_len),
        lambda: volume.rolling(settings.vol_ma_len, min_periods=settings.vol_ma_len).mean().shift(1),
    )
    vol_curr = volume.shift(1)
    qip_ema_fast_full = _cached(cache, ("ema", settings.qip_ema_fast), lambda: _ema(close, settings.qip_ema_fast))
    qip_ema_slow_full = _cached(cache, ("ema", settings.qip_ema_slow), lambda: _ema(close, settings.qip_ema_slow))
    qip_ema_f = qip_ema_fast_full.shift(1)
    qip_ema_s = qip_ema_slow_full.shift(1)
    macd_line = macd_raw.shift(1)

    qip_rsi_bull_ok = pd.Series(True, index=frame.index) if not settings.qip_use_rsi_zone else (qip_rsi_val <= settings.qip_rsi_bull_max)
    qip_rsi_bear_ok = pd.Series(True, index=frame.index) if not settings.qip_use_rsi_zone else (qip_rsi_val >= settings.qip_rsi_bear_min)

    prev_pl_rsi = _cached(
        cache,
        ("previous_occurrence", "pl_rsi", zz_left, settings.qip_rsi_len),
        lambda: _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), qip_rsi_val.to_numpy(dtype=float)),
    )
    prev_ph_rsi = _cached(
        cache,
        ("previous_occurrence", "ph_rsi", zz_left, settings.qip_rsi_len),
        lambda: _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), qip_rsi_val.to_numpy(dtype=float)),
    )
    prev_pl_val = _cached(
        cache,
        ("previous_occurrence", "pl_val", zz_left),
        lambda: _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), pivot_l_series.to_numpy(dtype=float)),
    )
    prev_ph_val = _cached(
        cache,
        ("previous_occurrence", "ph_val", zz_left),
        lambda: _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), pivot_h_series.to_numpy(dtype=float)),
    )
    prev_pl_macd = _cached(
        cache,
        ("previous_occurrence", "pl_macd", zz_left),
        lambda: _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), macd_line.to_numpy(dtype=float)),
    )
    prev_ph_macd = _cached(
        cache,
        ("previous_occurrence", "ph_macd", zz_left),
        lambda: _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), macd_line.to_numpy(dtype=float)),
    )

    bull_div = pl_valid & (qip_rsi_val > prev_pl_rsi) & (pivot_l_series < prev_pl_val)
    bear_div = ph_valid & (qip_rsi_val < prev_ph_rsi) & (pivot_h_series > prev_ph_val)
    bull_macd_div = pl_valid & (macd_line > prev_pl_macd) & (pivot_l_series < prev_pl_val)
    bear_macd_div = ph_valid & (macd_line < prev_ph_macd) & (pivot_h_series > prev_ph_val)

    bull_score = pd.Series(0, index=frame.index, dtype=float)
    bull_score += pl_valid.astype(float)
    bull_score += bull_div.astype(float) if settings.use_rsi_div else 0.0
    bull_score += bull_macd_div.astype(float) if settings.use_macd_div else 0.0
    bull_score += (vol_curr > vol_ma * 1.2).astype(float) if settings.use_volume else 0.0
    bull_score += (qip_ema_f > qip_ema_s).astype(float) if settings.use_ema_conf else 0.0

    bear_score = pd.Series(0, index=frame.index, dtype=float)
    bear_score += ph_valid.astype(float)
    bear_score += bear_div.astype(float) if settings.use_rsi_div else 0.0
    bear_score += bear_macd_div.astype(float) if settings.use_macd_div else 0.0
    bear_score += (vol_curr > vol_ma * 1.2).astype(float) if settings.use_volume else 0.0
    bear_score += (qip_ema_f < qip_ema_s).astype(float) if settings.use_ema_conf else 0.0

    qip_final_bull = (pl_valid & (bull_score >= settings.min_score) & qip_rsi_bull_ok) if settings.use_qip else pd.Series(False, index=frame.index)
    qip_final_bear = (ph_valid & (bear_score >= settings.min_score) & qip_rsi_bear_ok) if settings.use_qip else pd.Series(False, index=frame.index)

    qtp_sens = settings.qtp_sensitivity / 100.0
    qtp_pivot_left = int(round(settings.qtp_max_pvt_left - (settings.qtp_max_pvt_left - settings.qtp_min_pvt_left) * qtp_sens))
    qtp_rsi_upper = 70 - round(10 * qtp_sens)
    qtp_rsi_lower = 30 + round(10 * qtp_sens)
    qtp_stoch_upper = 80 - round(10 * qtp_sens)
    qtp_stoch_lower = 20 + round(10 * qtp_sens)
    qtp_score_thr = 55 - round(15 * qtp_sens)

    qtp_ema_fast_val = _cached(cache, ("ema", settings.qtp_ema_fast_len), lambda: _ema(close, settings.qtp_ema_fast_len))
    qtp_ema_slow_val = _cached(cache, ("ema", settings.qtp_ema_slow_len), lambda: _ema(close, settings.qtp_ema_slow_len))
    qtp_atr = _cached(cache, ("atr", settings.qtp_atr_len), lambda: _atr(high, low, close, settings.qtp_atr_len))
    qtp_rsi = _cached(cache, ("rsi", settings.qtp_rsi_len), lambda: _rsi(close, settings.qtp_rsi_len))
    qtp_stoch = _cached(cache, ("stoch", settings.qtp_stoch_len), lambda: _stoch(close, high, low, settings.qtp_stoch_len))
    qtp_vol_sma = _cached(
        cache,
        ("volume_mean", settings.qtp_vol_len),
        lambda: volume.rolling(settings.qtp_vol_len, min_periods=settings.qtp_vol_len).mean(),
    )

    qtp_basis = _cached(cache, ("ema", settings.qtp_dev_lookback), lambda: _ema(close, settings.qtp_dev_lookback))
    qtp_dev = (close - qtp_basis) / qtp_atr.replace(0.0, np.nan)
    qtp_d_mean = _cached(
        cache,
        ("qtp_dev_mean", settings.qtp_dev_lookback, settings.qtp_atr_len),
        lambda: qtp_dev.rolling(settings.qtp_dev_lookback, min_periods=settings.qtp_dev_lookback).mean(),
    )
    qtp_d_std = _cached(
        cache,
        ("qtp_dev_std", settings.qtp_dev_lookback, settings.qtp_atr_len),
        lambda: qtp_dev.rolling(settings.qtp_dev_lookback, min_periods=settings.qtp_dev_lookback).std(),
    )
    qtp_zdev = (qtp_dev - qtp_d_mean) / qtp_d_std.replace(0.0, np.nan)

    qtp_body = (close - open_).abs()
    qtp_range = high - low
    open_close_max = np.maximum(open_.to_numpy(dtype=float), close.to_numpy(dtype=float))
    open_close_min = np.minimum(open_.to_numpy(dtype=float), close.to_numpy(dtype=float))
    qtp_upper_wick = high - pd.Series(open_close_max, index=frame.index)
    qtp_lower_wick = pd.Series(open_close_min, index=frame.index) - low

    qtp_bull_rev = (qtp_range > 0) & (qtp_lower_wick > qtp_body * 1.2) & (close > open_)
    qtp_bear_rev = (qtp_range > 0) & (qtp_upper_wick > qtp_body * 1.2) & (close < open_)
    qtp_vol_spike = volume > qtp_vol_sma * (1.1 + 0.5 * qtp_sens)
    qtp_up_trend = qtp_ema_fast_val > qtp_ema_slow_val
    qtp_dn_trend = qtp_ema_fast_val < qtp_ema_slow_val
    qtp_roc1 = close - close.shift(1)
    qtp_roc2 = close.shift(1) - close.shift(2)

    qtp_bot_score = pd.Series(0.0, index=frame.index)
    qtp_top_score = pd.Series(0.0, index=frame.index)
    qtp_bot_score += np.where(qtp_rsi.shift(1) < qtp_rsi_lower, 18, 0)
    qtp_top_score += np.where(qtp_rsi.shift(1) > qtp_rsi_upper, 18, 0)
    qtp_bot_score += np.where(qtp_stoch.shift(1) < qtp_stoch_lower, 12, 0)
    qtp_top_score += np.where(qtp_stoch.shift(1) > qtp_stoch_upper, 12, 0)
    qtp_bot_score += np.where(qtp_zdev.shift(1) < -(1.0 + (1.0 - qtp_sens) * 0.8), 22, 0)
    qtp_top_score += np.where(qtp_zdev.shift(1) > (1.0 + (1.0 - qtp_sens) * 0.8), 22, 0)
    qtp_bot_score += np.where(qtp_bull_rev.shift(1), 16, 0)
    qtp_top_score += np.where(qtp_bear_rev.shift(1), 16, 0)
    qtp_bot_score += np.where(qtp_vol_spike.shift(1) & (close.shift(1) > low.shift(1)), 10, 0)
    qtp_top_score += np.where(qtp_vol_spike.shift(1) & (close.shift(1) < high.shift(1)), 10, 0)
    qtp_bot_score += np.where((qtp_roc2.shift(1) < 0) & (qtp_roc1.shift(1) > qtp_roc2.shift(1)), 10, 0)
    qtp_top_score += np.where((qtp_roc2.shift(1) > 0) & (qtp_roc1.shift(1) < qtp_roc2.shift(1)), 10, 0)
    if settings.qtp_use_trend:
        qtp_bot_score += np.where(qtp_up_trend.shift(1), 8, 0)
        qtp_top_score += np.where(qtp_dn_trend.shift(1), 8, 0)
    qtp_bot_score = qtp_bot_score.clip(upper=100)
    qtp_top_score = qtp_top_score.clip(upper=100)

    qtp_rsi_bull_ok = pd.Series(True, index=frame.index) if not settings.qtp_use_rsi_zone else (qtp_rsi.shift(1) <= settings.qtp_rsi_bull_max)
    qtp_rsi_bear_ok = pd.Series(True, index=frame.index) if not settings.qtp_use_rsi_zone else (qtp_rsi.shift(1) >= settings.qtp_rsi_bear_min)
    qtp_p_low = _cached(
        cache,
        ("pivot_low", qtp_pivot_left, 1),
        lambda: _pivot_low(low.to_numpy(dtype=float), qtp_pivot_left, 1),
    )
    qtp_p_high = _cached(
        cache,
        ("pivot_high", qtp_pivot_left, 1),
        lambda: _pivot_high(high.to_numpy(dtype=float), qtp_pivot_left, 1),
    )
    qtp_p_low_series = pd.Series(np.isfinite(qtp_p_low), index=frame.index)
    qtp_p_high_series = pd.Series(np.isfinite(qtp_p_high), index=frame.index)
    qtp_final_bull = (qtp_p_low_series & (qtp_bot_score >= qtp_score_thr) & qtp_rsi_bull_ok) if settings.use_qtp else pd.Series(False, index=frame.index)
    qtp_final_bear = (qtp_p_high_series & (qtp_top_score >= qtp_score_thr) & qtp_rsi_bear_ok) if settings.use_qtp else pd.Series(False, index=frame.index)

    final_bull = qip_final_bull | qtp_final_bull
    final_bear = qip_final_bear | qtp_final_bear
    sign = np.where(is_long_trend, 1.0, -1.0)

    indicators = frame.copy()
    indicators["supertrend"] = supertrend
    indicators["direction"] = direction
    indicators["dist_atr"] = dist_atr
    indicators["lev_zone"] = lev_zone
    indicators["zone2_line"] = supertrend + sign * zone2_thresh * atr_st
    indicators["zone3_line"] = supertrend + sign * zone3_thresh * atr_st
    indicators["is_long_trend"] = is_long_trend
    indicators["is_short_trend"] = is_short_trend
    indicators["trend_to_long"] = trend_to_long
    indicators["trend_to_short"] = trend_to_short
    indicators["qip_final_bull"] = qip_final_bull
    indicators["qip_final_bear"] = qip_final_bear
    indicators["qtp_final_bull"] = qtp_final_bull
    indicators["qtp_final_bear"] = qtp_final_bear
    indicators["final_bull"] = final_bull
    indicators["final_bear"] = final_bear
    indicators["bull_score"] = bull_score
    indicators["bear_score"] = bear_score
    indicators["qtp_bot_score"] = qtp_bot_score
    indicators["qtp_top_score"] = qtp_top_score
    indicators["ema_fast"] = qip_ema_fast_full
    indicators["ema_slow"] = qip_ema_slow_full
    indicators["rsi"] = qip_rsi_full
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
    return {
        "trend": "LONG" if bool(latest["is_long_trend"]) else "SHORT",
        "zone": int(latest["lev_zone"]),
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
) -> Tuple[StrategyMetrics, List[TradeRecord], pd.Series, pd.DataFrame, Dict[str, object]]:
    latest = indicators.iloc[-1]
    active_indicators, result_indicators = _select_active_indicators(indicators, backtest_start_time)

    active_count = len(active_indicators)
    close_values = active_indicators["close"].to_numpy(dtype=float, copy=False)
    time_values = active_indicators["time"].to_numpy(copy=False)
    lev_zone_values = active_indicators["lev_zone"].to_numpy(dtype=int, copy=False)
    is_long_trend_values = active_indicators["is_long_trend"].to_numpy(dtype=bool, copy=False)
    is_short_trend_values = active_indicators["is_short_trend"].to_numpy(dtype=bool, copy=False)
    trend_to_long_values = active_indicators["trend_to_long"].to_numpy(dtype=bool, copy=False)
    trend_to_short_values = active_indicators["trend_to_short"].to_numpy(dtype=bool, copy=False)
    final_bull_values = active_indicators["final_bull"].to_numpy(dtype=bool, copy=False)
    final_bear_values = active_indicators["final_bear"].to_numpy(dtype=bool, copy=False)

    equity = float(starting_equity)
    position_qty = 0.0
    avg_entry_price = 0.0
    entry_side = ""
    entry_time = pd.Timestamp(result_indicators["time"].iloc[0])
    entry_price = 0.0
    zone_events: List[str] = []
    trades: List[TradeRecord] = []
    equity_curve_values = np.empty(active_count, dtype=float) if active_count else np.empty(0, dtype=float)

    gross_profit = 0.0
    gross_loss = 0.0
    trade_count = 0
    win_count = 0

    long_zone_used = [False, False, False]
    short_zone_used = [False, False, False]
    last_long_zone = 0
    last_short_zone = 0
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

    for index in range(active_count):
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
    return metrics, trades, curve, indicator_frame, latest_state


def run_backtest_metrics(
    df: pd.DataFrame,
    settings: StrategySettings,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
    backtest_start_time: Optional[pd.Timestamp | str] = None,
    indicator_cache: Optional[Dict[Tuple[object, ...], object]] = None,
) -> StrategyMetrics:
    indicators = compute_indicators(df, settings, cache=indicator_cache)
    metrics, _, _, _, _ = _run_backtest_core(
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
    metrics, trades, curve, result_indicators, latest_state = _run_backtest_core(
        indicators,
        settings=settings,
        fee_rate=fee_rate,
        starting_equity=starting_equity,
        backtest_start_time=backtest_start_time,
        include_details=True,
    )
    return BacktestResult(
        settings=settings,
        metrics=metrics,
        trades=trades,
        indicators=result_indicators,
        latest_state=latest_state,
        equity_curve=curve,
    )
