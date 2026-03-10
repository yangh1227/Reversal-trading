from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import numpy as np
import pandas as pd
import pandas_ta as pta

from .config import StrategySettings


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


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame[["open", "high", "low", "close", "volume"]] = frame[["open", "high", "low", "close", "volume"]].astype(float)
    return frame.sort_values("time").reset_index(drop=True)


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

    for i in range(len(df)):
        if i == 0:
            supertrend.iat[i] = upperband.iat[i]
            direction.iat[i] = 1.0
            continue

        if upperband.iat[i] < final_upper.iat[i - 1] or close.iat[i - 1] > final_upper.iat[i - 1]:
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if lowerband.iat[i] > final_lower.iat[i - 1] or close.iat[i - 1] < final_lower.iat[i - 1]:
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        if supertrend.iat[i - 1] == final_upper.iat[i - 1]:
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


def compute_indicators(df: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    frame = _ensure_ohlcv(df)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    volume = frame["volume"]

    supertrend, direction, atr_st = _supertrend(frame, settings.atr_period, settings.factor)
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
    atr_val = _atr(high, low, close, 14)
    macd_raw = _macd_line(close)

    pivot_h = _pivot_high(high.to_numpy(dtype=float), zz_left, 1)
    pivot_l = _pivot_low(low.to_numpy(dtype=float), zz_left, 1)
    pivot_h_series = pd.Series(pivot_h, index=frame.index)
    pivot_l_series = pd.Series(pivot_l, index=frame.index)
    ph_confirmed = pivot_h_series.notna()
    pl_confirmed = pivot_l_series.notna()

    lowest_low_prev = low.rolling(zz_left * 2, min_periods=1).min().shift(1)
    highest_high_prev = high.rolling(zz_left * 2, min_periods=1).max().shift(1)
    ph_valid = ph_confirmed & ((pivot_h_series - lowest_low_prev) > (atr_val.shift(1) * atr_mult))
    pl_valid = pl_confirmed & ((highest_high_prev - pivot_l_series) > (atr_val.shift(1) * atr_mult))

    qip_rsi_full = _rsi(close, settings.qip_rsi_len)
    qip_rsi_val = qip_rsi_full.shift(1)
    vol_ma = volume.rolling(settings.vol_ma_len, min_periods=settings.vol_ma_len).mean().shift(1)
    vol_curr = volume.shift(1)
    qip_ema_f = _ema(close, settings.qip_ema_fast).shift(1)
    qip_ema_s = _ema(close, settings.qip_ema_slow).shift(1)
    macd_line = macd_raw.shift(1)

    qip_rsi_bull_ok = pd.Series(True, index=frame.index) if not settings.qip_use_rsi_zone else (qip_rsi_val <= settings.qip_rsi_bull_max)
    qip_rsi_bear_ok = pd.Series(True, index=frame.index) if not settings.qip_use_rsi_zone else (qip_rsi_val >= settings.qip_rsi_bear_min)

    prev_pl_rsi = _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), qip_rsi_val.to_numpy(dtype=float))
    prev_ph_rsi = _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), qip_rsi_val.to_numpy(dtype=float))
    prev_pl_val = _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), pivot_l_series.to_numpy(dtype=float))
    prev_ph_val = _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), pivot_h_series.to_numpy(dtype=float))
    prev_pl_macd = _previous_occurrence(pl_confirmed.to_numpy(dtype=bool), macd_line.to_numpy(dtype=float))
    prev_ph_macd = _previous_occurrence(ph_confirmed.to_numpy(dtype=bool), macd_line.to_numpy(dtype=float))

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

    qtp_ema_fast_val = _ema(close, settings.qtp_ema_fast_len)
    qtp_ema_slow_val = _ema(close, settings.qtp_ema_slow_len)
    qtp_atr = _atr(high, low, close, settings.qtp_atr_len)
    qtp_rsi = _rsi(close, settings.qtp_rsi_len)
    qtp_stoch = _stoch(close, high, low, settings.qtp_stoch_len)
    qtp_vol_sma = volume.rolling(settings.qtp_vol_len, min_periods=settings.qtp_vol_len).mean()

    qtp_basis = _ema(close, settings.qtp_dev_lookback)
    qtp_dev = (close - qtp_basis) / qtp_atr.replace(0.0, np.nan)
    qtp_d_mean = qtp_dev.rolling(settings.qtp_dev_lookback, min_periods=settings.qtp_dev_lookback).mean()
    qtp_d_std = qtp_dev.rolling(settings.qtp_dev_lookback, min_periods=settings.qtp_dev_lookback).std()
    qtp_zdev = (qtp_dev - qtp_d_mean) / qtp_d_std.replace(0.0, np.nan)

    qtp_body = (close - open_).abs()
    qtp_range = high - low
    qtp_upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    qtp_lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

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
    qtp_p_low = _pivot_low(low.to_numpy(dtype=float), qtp_pivot_left, 1)
    qtp_p_high = _pivot_high(high.to_numpy(dtype=float), qtp_pivot_left, 1)
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
    indicators["ema_fast"] = _ema(close, settings.qip_ema_fast)
    indicators["ema_slow"] = _ema(close, settings.qip_ema_slow)
    indicators["rsi"] = _rsi(close, settings.qip_rsi_len)
    return indicators


def run_backtest(
    df: pd.DataFrame,
    settings: StrategySettings,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
) -> BacktestResult:
    indicators = compute_indicators(df, settings)
    equity = float(starting_equity)
    position_qty = 0.0
    avg_entry_price = 0.0
    open_trade: Optional[dict] = None
    trades: List[TradeRecord] = []
    equity_curve: List[float] = []

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
        nonlocal equity, position_qty, avg_entry_price, open_trade, last_long_zone, last_short_zone
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
            open_trade = {
                "side": side,
                "entry_time": time_value,
                "entry_price": price,
                "zone_events": [f"{side[0].upper()}{zone}"],
            }
        else:
            total_qty = abs(position_qty) + qty
            avg_entry_price = ((avg_entry_price * abs(position_qty)) + (price * qty)) / max(total_qty, 1e-12)
            position_qty += signed_qty
            if open_trade is not None:
                open_trade["zone_events"].append(f"{side[0].upper()}{zone}")

        if side == "long":
            long_zone_used[zone - 1] = True
            last_long_zone = zone
        else:
            short_zone_used[zone - 1] = True
            last_short_zone = zone

    def close_position(price: float, time_value: pd.Timestamp, reason: str) -> None:
        nonlocal equity, position_qty, avg_entry_price, open_trade
        if abs(position_qty) < 1e-12 or open_trade is None:
            return
        pnl = position_qty * (price - avg_entry_price)
        fee = abs(position_qty) * price * fee_rate
        equity += pnl - fee
        gross_cost = abs(position_qty) * avg_entry_price
        return_pct = (pnl / gross_cost * 100.0) if gross_cost else 0.0
        trades.append(
            TradeRecord(
                side=str(open_trade["side"]),
                entry_time=pd.Timestamp(open_trade["entry_time"]),
                exit_time=pd.Timestamp(time_value),
                entry_price=float(open_trade["entry_price"]),
                exit_price=float(price),
                quantity=float(abs(position_qty)),
                pnl=float(pnl - fee),
                return_pct=float(return_pct),
                reason=reason,
                zones=",".join(open_trade["zone_events"]),
            )
        )
        position_qty = 0.0
        avg_entry_price = 0.0
        open_trade = None
        reset_zones()

    for row in indicators.itertuples(index=False):
        current_price = float(row.close)
        current_time = pd.Timestamp(row.time)

        if abs(position_qty) < 1e-12:
            reset_zones()

        if row.trend_to_short and position_qty > 0:
            close_position(current_price, current_time, "trend_to_short")
        if row.trend_to_long and position_qty < 0:
            close_position(current_price, current_time, "trend_to_long")
        if row.final_bear and position_qty > 0:
            close_position(current_price, current_time, "opposite_signal")
        if row.final_bull and position_qty < 0:
            close_position(current_price, current_time, "opposite_signal")

        if abs(position_qty) < 1e-12:
            reset_zones()

        can_long_z1 = (not settings.beast_mode) and row.is_long_trend and row.final_bull and row.lev_zone == 1 and (not long_zone_used[0]) and last_long_zone == 0
        can_long_z2 = row.is_long_trend and row.final_bull and row.lev_zone == 2 and (not long_zone_used[1]) and last_long_zone in (0, 1)
        can_long_z3 = row.is_long_trend and row.final_bull and row.lev_zone == 3 and (not long_zone_used[2]) and last_long_zone in (0, 2)
        can_short_z1 = (not settings.beast_mode) and row.is_short_trend and row.final_bear and row.lev_zone == 1 and (not short_zone_used[0]) and last_short_zone == 0
        can_short_z2 = row.is_short_trend and row.final_bear and row.lev_zone == 2 and (not short_zone_used[1]) and last_short_zone in (0, 1)
        can_short_z3 = row.is_short_trend and row.final_bear and row.lev_zone == 3 and (not short_zone_used[2]) and last_short_zone in (0, 2)

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
        equity_curve.append(equity + unrealized)

    if abs(position_qty) > 1e-12:
        final_row = indicators.iloc[-1]
        close_position(float(final_row["close"]), pd.Timestamp(final_row["time"]), "end_of_test")
        equity_curve[-1] = equity

    gross_profit = sum(max(trade.pnl, 0.0) for trade in trades)
    gross_loss = abs(sum(min(trade.pnl, 0.0) for trade in trades))
    trade_count = len(trades)
    win_rate = (sum(1 for trade in trades if trade.pnl > 0) / trade_count * 100.0) if trade_count else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf" if gross_profit > 0 else 0.0)

    if equity_curve:
        curve_index = indicators["time"].iloc[: len(equity_curve)]
        curve = pd.Series(equity_curve, index=curve_index, dtype=float)
    else:
        curve = pd.Series([starting_equity], index=[indicators["time"].iloc[0]], dtype=float)
    peaks = curve.cummax()
    drawdown_pct = ((curve - peaks) / peaks.replace(0.0, np.nan)).fillna(0.0) * 100.0
    max_drawdown_pct = abs(float(drawdown_pct.min())) if not drawdown_pct.empty else 0.0

    latest = indicators.iloc[-1]
    latest_state = {
        "trend": "LONG" if bool(latest["is_long_trend"]) else "SHORT",
        "zone": int(latest["lev_zone"]),
        "final_bull": bool(latest["final_bull"]),
        "final_bear": bool(latest["final_bear"]),
        "qip_bull": bool(latest["qip_final_bull"]),
        "qip_bear": bool(latest["qip_final_bear"]),
        "qtp_bull": bool(latest["qtp_final_bull"]),
        "qtp_bear": bool(latest["qtp_final_bear"]),
        "rsi": float(latest["rsi"]) if np.isfinite(latest["rsi"]) else float("nan"),
        "dist_atr": float(latest["dist_atr"]) if np.isfinite(latest["dist_atr"]) else float("nan"),
    }

    metrics = StrategyMetrics(
        total_return_pct=((equity - starting_equity) / starting_equity) * 100.0,
        net_profit=equity - starting_equity,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=trade_count,
        win_rate_pct=win_rate,
        profit_factor=float(profit_factor if math.isfinite(profit_factor) else 999.0),
    )
    return BacktestResult(
        settings=settings,
        metrics=metrics,
        trades=trades,
        indicators=indicators,
        latest_state=latest_state,
        equity_curve=curve,
    )
