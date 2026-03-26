from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import pandas as pd

from .binance_futures import PositionSnapshot
from .config import APP_INTERVAL_OPTIONS, StrategySettings
from .strategy import (
    active_auto_trade_signal as strategy_active_auto_trade_signal,
    BacktestResult,
    latest_confirmed_entry_event,
    resume_backtest,
    run_backtest,
    signal_fraction_for_zone,
)


@dataclass(frozen=True)
class AutoTradeEvaluationResult:
    candidate: Optional[Dict[str, object]] = None
    reentry_position_side: str = ""
    signal_side: str = ""
    signal_zone: int = 0
    signal_kind: str = ""


@dataclass(frozen=True)
class ResolvedAutoTradeBacktest:
    backtest: Optional[BacktestResult] = None
    source: str = "none"


def backtest_matches_history(backtest: Optional[BacktestResult], history: Optional[pd.DataFrame]) -> bool:
    if backtest is None or history is None or history.empty or backtest.indicators.empty:
        return False
    history_signature = getattr(backtest, "history_signature", ())
    if history_signature:
        history_times = pd.to_datetime(history["time"])
        last_row = history.iloc[-1]
        values: List[object] = [pd.Timestamp(history_times.iloc[-1]), int(len(history))]
        for column in ("open", "high", "low", "close", "volume"):
            if column not in history.columns:
                continue
            value = last_row[column]
            values.append(None if pd.isna(value) else float(value))
        return tuple(history_signature) == tuple(values)
    if "time" not in backtest.indicators.columns or "time" not in history.columns:
        return False
    indicator_times = pd.to_datetime(backtest.indicators["time"])
    history_times = pd.to_datetime(history["time"])
    if indicator_times.empty or history_times.empty:
        return False
    return (
        len(backtest.indicators) == len(history)
        and pd.Timestamp(indicator_times.iloc[-1]) == pd.Timestamp(history_times.iloc[-1])
    )


def resolve_latest_auto_trade_backtest(
    seed_backtest: Optional[BacktestResult],
    history: Optional[pd.DataFrame],
    strategy_settings: StrategySettings,
    *,
    fee_rate: float = 0.0004,
    starting_equity: float = 1_000.0,
    backtest_start_time: Optional[pd.Timestamp | str] = None,
) -> ResolvedAutoTradeBacktest:
    usable_seed = (
        seed_backtest
        if seed_backtest is not None and seed_backtest.settings == strategy_settings
        else None
    )
    if history is None or history.empty:
        return ResolvedAutoTradeBacktest(usable_seed, "cached" if usable_seed is not None else "none")
    if usable_seed is not None:
        if backtest_matches_history(usable_seed, history):
            return ResolvedAutoTradeBacktest(usable_seed, "cached")
        if history_can_resume_backtest(usable_seed, history):
            return ResolvedAutoTradeBacktest(
                resume_backtest(
                    history,
                    previous_result=usable_seed,
                    settings=strategy_settings,
                    fee_rate=fee_rate,
                    starting_equity=starting_equity,
                    backtest_start_time=backtest_start_time,
                ),
                "resumed",
            )
    return ResolvedAutoTradeBacktest(
        run_backtest(
            history,
            settings=strategy_settings,
            fee_rate=fee_rate,
            starting_equity=starting_equity,
            backtest_start_time=backtest_start_time,
        ),
        "materialized",
    )


def resolve_favorable_auto_trade_zone(
    latest_backtest: Optional[BacktestResult],
    current_price: Optional[float],
    open_position: Optional[PositionSnapshot],
    filled_fraction: float,
) -> Optional[int]:
    if latest_backtest is None or current_price is None or current_price <= 0:
        return None
    signal = auto_trade_signal_from_backtest(latest_backtest)
    if signal is None:
        return None
    signal_side = str(signal.get("side") or "")
    if open_position is not None:
        position_side = "long" if float(open_position.amount) > 0 else "short"
        if signal_side != position_side:
            return None
    favorable_zone, favorable_fraction = favorable_signal_state(
        signal_side,
        float(current_price),
        float(signal.get("price") or 0.0),
        dict(signal.get("zone_prices") or {}),
        float(filled_fraction),
    )
    if favorable_zone <= 0 or favorable_fraction <= 1e-9:
        return None
    return favorable_zone


def history_can_resume_backtest(backtest: Optional[BacktestResult], history: Optional[pd.DataFrame]) -> bool:
    if backtest is None or history is None or history.empty or backtest.indicators.empty or backtest.cursor is None:
        return False
    history_times = pd.to_datetime(history["time"])
    backtest_last_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    return bool((history_times == backtest_last_time).any()) and pd.Timestamp(history_times.iloc[-1]) > backtest_last_time


def auto_trade_signal_from_backtest(backtest: BacktestResult) -> Optional[Dict[str, object]]:
    return strategy_active_auto_trade_signal(backtest)


def favorable_zone_for_price(
    side: str,
    current_price: float,
    signal_price: float,
    zone_prices: Dict[int, float],
) -> Optional[int]:
    for zone in (3, 2, 1):
        zone_price = float(zone_prices.get(zone, 0.0) or 0.0)
        if zone == 1 and zone_price <= 0:
            zone_price = float(signal_price or 0.0)
        if zone_price <= 0:
            continue
        favorable = current_price < zone_price if side == "long" else current_price > zone_price
        if favorable:
            return zone
    return None


def favorable_signal_state(
    side: str,
    current_price: float,
    signal_price: float,
    zone_prices: Dict[int, float],
    filled_fraction: float,
) -> tuple[int, float]:
    favorable_zone = favorable_zone_for_price(
        side,
        float(current_price),
        float(signal_price),
        dict(zone_prices or {}),
    )
    if favorable_zone is None:
        return 0, 0.0
    return favorable_zone, max(0.0, signal_fraction_for_zone(favorable_zone) - float(filled_fraction))


def zone_favorable_fraction(
    side: str,
    current_price: float,
    signal_price: float,
    zone_prices: Dict[int, float],
    filled_fraction: float,
) -> float:
    _favorable_zone, favorable_fraction = favorable_signal_state(
        side,
        float(current_price),
        float(signal_price),
        dict(zone_prices or {}),
        float(filled_fraction),
    )
    return favorable_fraction


def favorable_auto_trade_fraction(
    latest_backtest: Optional[BacktestResult],
    current_price: Optional[float],
    open_position: Optional[PositionSnapshot],
    filled_fraction: float,
) -> float:
    if latest_backtest is None or current_price is None or current_price <= 0:
        return 0.0
    signal = auto_trade_signal_from_backtest(latest_backtest)
    if signal is None:
        return 0.0
    side = str(signal["side"])
    if open_position is not None:
        position_side = "long" if float(open_position.amount) > 0 else "short"
        if side != position_side:
            return 0.0
    return zone_favorable_fraction(
        side,
        float(current_price),
        float(signal["price"]),
        dict(signal.get("zone_prices") or {}),
        float(filled_fraction),
    )


def inferred_auto_trade_fraction(backtest: BacktestResult, position: Optional[PositionSnapshot]) -> float:
    cursor = backtest.cursor
    if cursor is None or position is None or abs(float(position.amount)) < 1e-12:
        return 0.0
    side = "long" if float(position.amount) > 0 else "short"
    zone = 0
    signal_side = str(cursor.last_entry_signal_side or "").lower()
    signal_zone = int(cursor.last_entry_signal_zone or 0)
    if signal_side == side and signal_zone in {1, 2, 3}:
        zone = signal_zone
    fallback_zone = int(cursor.last_long_zone if side == "long" else cursor.last_short_zone)
    if fallback_zone in {1, 2, 3}:
        zone = max(zone, fallback_zone)
    return signal_fraction_for_zone(zone) if zone in {1, 2, 3} else 0.0


def pick_auto_trade_candidate(candidates: List[Dict[str, object]], rank_mode: str) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    if rank_mode == "return":
        best_return = max(float(item["return_pct"]) for item in candidates)
        return_tied = [item for item in candidates if float(item["return_pct"]) == best_return]
        best_score = max(float(item["score"]) for item in return_tied)
        score_tied = [item for item in return_tied if float(item["score"]) == best_score]
        return random.choice(score_tied)
    best_score = max(float(item["score"]) for item in candidates)
    score_tied = [item for item in candidates if float(item["score"]) == best_score]
    best_return = max(float(item["return_pct"]) for item in score_tied)
    return_tied = [item for item in score_tied if float(item["return_pct"]) == best_return]
    return random.choice(return_tied)


def evaluate_auto_trade_candidate(
    *,
    symbol: str,
    interval: str,
    score: float,
    strategy_settings: Optional[StrategySettings],
    latest_backtest: Optional[BacktestResult],
    current_price: Optional[float],
    open_position: Optional[PositionSnapshot],
    remembered_interval: Optional[str],
    filled_fraction: float,
    remembered_cursor_entry_time: Optional[pd.Timestamp],
    allow_favorable_price_entries: bool = True,
    from_initial_load: bool = False,
    trigger_symbol: str = "",
    trigger_interval: str = "",
    trigger_bar_time: Optional[pd.Timestamp] = None,
) -> AutoTradeEvaluationResult:
    if latest_backtest is None:
        return AutoTradeEvaluationResult()
    if open_position is not None and remembered_interval in APP_INTERVAL_OPTIONS and remembered_interval != interval:
        return AutoTradeEvaluationResult()
    signal = auto_trade_signal_from_backtest(latest_backtest)
    signal_price = 0.0
    zone_prices: Dict[int, float] = {}
    side = ""
    signal_zone = 0
    fraction = 0.0
    cursor_entry_time = None
    actionable_side = ""
    actionable_zone = 0
    actionable_kind = ""
    if signal is not None:
        signal_price = float(signal["price"])
        zone_prices = dict(signal.get("zone_prices") or {})
        side = str(signal["side"])
        signal_zone = int(signal.get("zone") or 0)
        fraction = float(signal["fraction"])
        cursor_entry_time = signal.get("cursor_entry_time")
    normalized_trigger_time = pd.Timestamp(trigger_bar_time).tz_localize(None) if trigger_bar_time is not None else None
    confirmed_entry = (
        latest_confirmed_entry_event(latest_backtest, normalized_trigger_time)
        if normalized_trigger_time is not None and symbol == trigger_symbol and interval == trigger_interval
        else None
    )
    has_fresh_confirmed_entry = (
        signal is not None
        and confirmed_entry is not None
        and str(confirmed_entry["side"]) == side
        and int(confirmed_entry["zone"]) == int(signal["zone"])
    )
    fresh_confirmed_additional_entry = False
    if open_position is not None and confirmed_entry is not None:
        position_side = "long" if float(open_position.amount) > 0 else "short"
        confirmed_side = str(confirmed_entry["side"])
        confirmed_zone = int(confirmed_entry["zone"])
        confirmed_fraction = signal_fraction_for_zone(confirmed_zone)
        if (
            confirmed_side == position_side
            and confirmed_zone in {1, 2, 3}
            and confirmed_fraction > float(filled_fraction) + 1e-9
        ):
            side = confirmed_side
            signal_zone = confirmed_zone
            fraction = confirmed_fraction
            fresh_confirmed_additional_entry = True
            has_fresh_confirmed_entry = True
            actionable_side = confirmed_side
            actionable_zone = confirmed_zone
            actionable_kind = "confirmed"
            if latest_backtest.cursor is not None and latest_backtest.cursor.entry_time is not None:
                cursor_entry_time = pd.Timestamp(latest_backtest.cursor.entry_time)
    if signal is None and not fresh_confirmed_additional_entry:
        return AutoTradeEvaluationResult()
    if not has_fresh_confirmed_entry and (current_price is None or current_price <= 0):
        return AutoTradeEvaluationResult()
    latest_state = latest_backtest.latest_state
    if latest_state and open_position is not None:
        if side == "long" and (bool(latest_state.get("trend_to_short")) or bool(latest_state.get("final_bear"))):
            return AutoTradeEvaluationResult()
        if side == "short" and (bool(latest_state.get("trend_to_long")) or bool(latest_state.get("final_bull"))):
            return AutoTradeEvaluationResult()
    if open_position is not None:
        position_side = "long" if float(open_position.amount) > 0 else "short"
        if side != position_side:
            return AutoTradeEvaluationResult()
        if remembered_cursor_entry_time is not None and cursor_entry_time is not None:
            if pd.Timestamp(remembered_cursor_entry_time).tz_localize(None) != pd.Timestamp(cursor_entry_time).tz_localize(None):
                return AutoTradeEvaluationResult(reentry_position_side=position_side)
        fraction = max(0.0, fraction - float(filled_fraction))
        if fraction <= 1e-9:
            return AutoTradeEvaluationResult()
        if not has_fresh_confirmed_entry:
            if not allow_favorable_price_entries:
                return AutoTradeEvaluationResult()
            favorable_zone, favorable_fraction = favorable_signal_state(
                side,
                float(current_price),
                signal_price,
                zone_prices,
                float(filled_fraction),
            )
            if favorable_zone <= 0 or favorable_fraction <= 1e-9:
                return AutoTradeEvaluationResult()
            fraction = min(fraction, favorable_fraction)
            actionable_side = side
            actionable_zone = favorable_zone
            actionable_kind = "favorable"
    else:
        # 초기 로드 시에는 신호봉이 과거이므로 confirmed_entry여도 현재가 존 검사 적용
        if not has_fresh_confirmed_entry or from_initial_load:
            if not allow_favorable_price_entries and not from_initial_load:
                return AutoTradeEvaluationResult()
            if current_price is None or current_price <= 0:
                return AutoTradeEvaluationResult()
            favorable_zone, favorable_fraction = favorable_signal_state(
                side,
                float(current_price),
                signal_price,
                zone_prices,
                0.0,
            )
            if favorable_zone <= 0 or favorable_fraction <= 1e-9:
                return AutoTradeEvaluationResult()
            fraction = min(fraction, favorable_fraction)
            actionable_side = side
            actionable_zone = favorable_zone
            actionable_kind = "favorable"
    if has_fresh_confirmed_entry and actionable_kind != "confirmed":
        actionable_side = side
        actionable_zone = signal_zone
        actionable_kind = "confirmed"
    if actionable_side not in {"long", "short"}:
        return AutoTradeEvaluationResult()
    if actionable_zone not in {1, 2, 3}:
        return AutoTradeEvaluationResult()
    if actionable_kind not in {"confirmed", "favorable"}:
        return AutoTradeEvaluationResult()
    candidate = {
        "symbol": symbol,
        "interval": interval,
        "side": "BUY" if side == "long" else "SELL",
        "score": float(score),
        "return_pct": float(latest_backtest.metrics.total_return_pct),
        "fraction": float(fraction),
        "strategy_settings": strategy_settings,
        "cursor_entry_time": cursor_entry_time,
        "signal_side": actionable_side,
        "signal_zone": int(actionable_zone),
        "signal_kind": actionable_kind,
    }
    return AutoTradeEvaluationResult(
        candidate=candidate,
        signal_side=actionable_side,
        signal_zone=int(actionable_zone),
        signal_kind=actionable_kind,
    )
