from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import time
import pandas as pd

from .config import (
    DEFAULT_OPTIMIZATION_PROFILE_SCALE,
    PARAMETER_SPECS,
    ParameterSpec,
    StrategySettings,
    parameter_spec_applies,
)
from .strategy import BacktestResult, StrategyMetrics, prepare_ohlcv, run_backtest, run_backtest_metrics


@dataclass(frozen=True)
class OptimizationResult:
    symbol: str
    best_interval: str
    best_backtest: BacktestResult
    score: float
    combinations_tested: int
    duration_seconds: float
    trimmed_grid: bool


PARAMETER_SPEC_BY_KEY = {spec.key: spec for spec in PARAMETER_SPECS}
OPTIMIZATION_RESULT_CACHE: Dict[Tuple[object, ...], OptimizationResult] = {}


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def _winrate_trade_component(metrics: StrategyMetrics) -> float:
    trade_count = max(float(metrics.trade_count), 1.0)
    return _clamp_score(float(metrics.win_rate_pct) * math.log(trade_count) * 0.2)


def _profit_factor_component(metrics: StrategyMetrics) -> float:
    profit_factor = float(metrics.profit_factor)
    if not math.isfinite(profit_factor):
        return 100.0 if profit_factor > 0 else 0.0
    return _clamp_score(50.0 + (50.0 * math.tanh((profit_factor - 1.0) / 1.25)))


def score_optimization_metrics(metrics: StrategyMetrics) -> float:
    return_component = 50.0 + (50.0 * math.tanh(float(metrics.total_return_pct) / 35.0))
    win_component = _winrate_trade_component(metrics)
    mdd_component = _clamp_score(100.0 - (float(metrics.max_drawdown_pct) * 2.5))
    profit_factor_component = _profit_factor_component(metrics)
    total_score = (
        (return_component * 0.45)
        + (win_component * 0.20)
        + (profit_factor_component * 0.20)
        + (mdd_component * 0.15)
    )
    return round(_clamp_score(total_score), 2)


def optimization_sort_key(metrics: StrategyMetrics, rank_mode: str) -> Tuple[float, ...]:
    score = score_optimization_metrics(metrics)
    if rank_mode == "return":
        return (
            float(metrics.total_return_pct),
            score,
            float(metrics.win_rate_pct),
            float(metrics.profit_factor),
            -float(metrics.max_drawdown_pct),
            float(metrics.trade_count),
        )
    return (
        score,
        float(metrics.total_return_pct),
        _winrate_trade_component(metrics),
        _profit_factor_component(metrics),
        -float(metrics.max_drawdown_pct),
        float(metrics.trade_count),
        float(metrics.profit_factor),
    )


def _history_signature(df: pd.DataFrame) -> Tuple[object, ...]:
    if df.empty or "time" not in df.columns:
        return (None, 0)
    last = df.iloc[-1]
    values: List[object] = [pd.Timestamp(last["time"]), int(len(df))]
    for column in ("open", "high", "low", "close", "volume"):
        if column not in df.columns:
            continue
        values.append(float(last[column]))
    return tuple(values)


def optimize_symbol_process_entry(
    symbol: str,
    df: pd.DataFrame,
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    span_pct: float,
    steps: int,
    max_combinations: int,
    fee_rate: float,
    rank_mode: str = "score",
    backtest_start_time: pd.Timestamp | str | None = None,
    result_interval: Optional[str] = None,
) -> OptimizationResult:
    return optimize_symbol(
        symbol=symbol,
        df=df,
        base_settings=base_settings,
        optimize_flags=optimize_flags,
        span_pct=span_pct,
        steps=steps,
        max_combinations=max_combinations,
        fee_rate=fee_rate,
        rank_mode=rank_mode,
        backtest_start_time=backtest_start_time,
        result_interval=result_interval,
    )


def _profile_scale(span_pct: float) -> float:
    return max(0.25, float(span_pct) / DEFAULT_OPTIMIZATION_PROFILE_SCALE)


def _numeric_step(spec: ParameterSpec):
    if spec.optimize_step is not None:
        return spec.optimize_step
    if spec.step is not None:
        return spec.step
    return 1 if spec.kind == "int" else 0.1


def _snap_numeric_value(spec: ParameterSpec, value, step):
    if spec.kind == "int":
        step = max(1, int(step or 1))
        snapped = int(round(float(value) / step) * step)
    else:
        step = float(step or 0.01)
        precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        snapped = round(round(float(value) / step) * step, precision)
    if spec.minimum is not None:
        snapped = max(spec.minimum, snapped)
    if spec.maximum is not None:
        snapped = min(spec.maximum, snapped)
    return snapped


def _thin_values(values: Sequence, target_count: int, base_index: int) -> List:
    items = list(values)
    if target_count <= 0:
        return []
    if len(items) <= target_count:
        return items
    if target_count == 1:
        return [items[base_index]]

    indices = {int(round((len(items) - 1) * idx / (target_count - 1))) for idx in range(target_count)}
    indices.add(base_index)
    protected = {0, len(items) - 1, base_index}

    while len(indices) < target_count:
        candidates = [idx for idx in range(len(items)) if idx not in indices]
        if not candidates:
            break
        best_idx = max(candidates, key=lambda idx: min(abs(idx - selected) for selected in indices))
        indices.add(best_idx)

    while len(indices) > target_count:
        removable = [idx for idx in sorted(indices) if idx not in protected]
        if not removable:
            removable = [idx for idx in sorted(indices) if idx != base_index]
        if not removable:
            break
        drop_idx = min(
            removable,
            key=lambda idx: (
                min(abs(idx - other) for other in indices if other != idx),
                abs(idx - base_index),
            ),
        )
        indices.remove(drop_idx)

    return [items[idx] for idx in sorted(indices)]


def _choice_range(spec: ParameterSpec, base_value, span_pct: float, steps: int) -> List:
    choices = list(spec.choices)
    if base_value not in choices:
        return [base_value]
    center = choices.index(base_value)
    radius = max(1, int(round(spec.optimize_choice_radius * _profile_scale(span_pct))))
    start = max(0, center - radius)
    end = min(len(choices), center + radius + 1)
    values = choices[start:end]
    base_index = values.index(base_value)
    return _thin_values(values, min(max(steps, 1), len(values)), base_index)


def _numeric_range(spec: ParameterSpec, base_value, span_pct: float, steps: int) -> List:
    original_base = int(base_value) if spec.kind == "int" else float(base_value)
    step = _numeric_step(spec)
    span = spec.optimize_span
    if span is None:
        span = max(step, abs(float(base_value)) * 0.2)
    effective_span = max(step, float(span) * _profile_scale(span_pct))
    half_steps = max(1, int(round(effective_span / float(step))))
    values: List = []
    for offset in range(-half_steps, half_steps + 1):
        value = _snap_numeric_value(spec, float(base_value) + (offset * float(step)), step)
        if value not in values:
            values.append(value)
    base_value = original_base
    if spec.minimum is not None:
        base_value = max(spec.minimum, base_value)
    if spec.maximum is not None:
        base_value = min(spec.maximum, base_value)
    if base_value not in values:
        values.append(base_value)
    values = sorted(values)
    base_index = values.index(base_value)
    return _thin_values(values, min(max(steps, 1), len(values)), base_index)


def parameter_value_range(spec: ParameterSpec, base_value, span_pct: float, steps: int, enabled: bool) -> List:
    if not enabled:
        return [base_value]

    if spec.kind == "bool":
        return [False, True]

    if spec.kind == "choice":
        return _choice_range(spec, base_value, span_pct, steps)

    return _numeric_range(spec, base_value, span_pct, steps)


def _value_range(spec: ParameterSpec, base_value, span_pct: float, steps: int, enabled: bool) -> List:
    return parameter_value_range(spec, base_value, span_pct, steps, enabled)


def _settings_are_valid(settings: StrategySettings) -> bool:
    if settings.strategy_type == "mean_reversion":
        if settings.qip_ema_fast >= settings.qip_ema_slow:
            return False
        if settings.qtp_ema_fast_len >= settings.qtp_ema_slow_len:
            return False
        if settings.qtp_min_pvt_left > settings.qtp_max_pvt_left:
            return False
        if settings.qip_rsi_bull_max >= settings.qip_rsi_bear_min:
            return False
        if settings.qtp_rsi_bull_max >= settings.qtp_rsi_bear_min:
            return False
    return True


def generate_parameter_grid(
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    span_pct: float,
    steps: int,
    max_combinations: int,
) -> Tuple[List[StrategySettings], bool]:
    strategy_type = str(base_settings.strategy_type)
    values_by_key = {
        spec.key: _value_range(
            spec,
            getattr(base_settings, spec.key),
            span_pct=span_pct,
            steps=steps,
            enabled=bool(optimize_flags.get(spec.key, False)) and parameter_spec_applies(spec, strategy_type),
        )
        for spec in PARAMETER_SPECS
    }

    def count() -> int:
        total = 1
        for values in values_by_key.values():
            total *= max(1, len(values))
        return total

    trimmed = False
    while count() > max_combinations:
        trimmed = True
        candidates = [key for key, values in values_by_key.items() if len(values) > 1]
        if not candidates:
            break
        key = max(
            candidates,
            key=lambda current: (
                PARAMETER_SPEC_BY_KEY[current].optimize_priority,
                len(values_by_key[current]),
            ),
        )
        current_values = values_by_key[key]
        next_size = max(1, len(current_values) - (2 if len(current_values) > 4 else 1))
        base_value = getattr(base_settings, key)
        if base_value in current_values:
            base_index = current_values.index(base_value)
        else:
            base_index = len(current_values) // 2
        values_by_key[key] = _thin_values(current_values, next_size, base_index)

    keys = list(values_by_key.keys())
    grid: List[StrategySettings] = []
    seen_settings: set[StrategySettings] = set()
    for combo in product(*(values_by_key[key] for key in keys)):
        payload = {key: combo[idx] for idx, key in enumerate(keys)}
        settings = replace(base_settings, **payload)
        if not _settings_are_valid(settings):
            continue
        if settings in seen_settings:
            continue
        seen_settings.add(settings)
        grid.append(settings)
    return grid, trimmed


def optimize_symbol(
    symbol: str,
    df: pd.DataFrame,
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    span_pct: float,
    steps: int,
    max_combinations: int,
    fee_rate: float,
    rank_mode: str = "score",
    backtest_start_time: pd.Timestamp | str | None = None,
    should_stop: Optional[Callable[[], bool]] = None,
    result_interval: Optional[str] = None,
) -> OptimizationResult:
    started = time.perf_counter()
    grid, trimmed = generate_parameter_grid(base_settings, optimize_flags, span_pct, steps, max_combinations)
    prepared_df = prepare_ohlcv(df)
    cache_key = (
        symbol,
        result_interval or "",
        _history_signature(prepared_df),
        base_settings,
        tuple(sorted(optimize_flags.items())),
        float(span_pct),
        int(steps),
        int(max_combinations),
        float(fee_rate),
        str(rank_mode),
        pd.Timestamp(backtest_start_time) if backtest_start_time is not None else None,
    )
    cached_result = OPTIMIZATION_RESULT_CACHE.get(cache_key)
    if cached_result is not None:
        return replace(cached_result, duration_seconds=0.0)
    indicator_cache: Dict[tuple[object, ...], object] = {}
    best_metrics: StrategyMetrics | None = None
    best_settings: StrategySettings | None = None
    combinations_tested = 0

    for settings in grid:
        if should_stop and should_stop():
            break
        current = run_backtest_metrics(
            prepared_df,
            settings=settings,
            fee_rate=fee_rate,
            backtest_start_time=backtest_start_time,
            indicator_cache=indicator_cache,
        )
        combinations_tested += 1
        if best_metrics is None:
            best_settings = settings
            best_metrics = current
            continue

        best = best_metrics
        current_key = optimization_sort_key(current, rank_mode)
        best_key = optimization_sort_key(best, rank_mode)
        if current_key > best_key:
            best_settings = settings
            best_metrics = current

    if best_settings is None or best_metrics is None:
        raise RuntimeError(f"optimization cancelled for {symbol}" if should_stop and should_stop() else f"no optimization result for {symbol}")

    best_result: BacktestResult = run_backtest(
        prepared_df,
        settings=best_settings,
        fee_rate=fee_rate,
        backtest_start_time=backtest_start_time,
        indicator_cache=indicator_cache,
    )

    result = OptimizationResult(
        symbol=symbol,
        best_interval=result_interval or "",
        best_backtest=best_result,
        score=score_optimization_metrics(best_result.metrics),
        combinations_tested=combinations_tested,
        duration_seconds=time.perf_counter() - started,
        trimmed_grid=trimmed,
    )
    OPTIMIZATION_RESULT_CACHE[cache_key] = result
    return result


def optimize_symbol_intervals(
    symbol: str,
    histories_by_interval: Dict[str, pd.DataFrame],
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    interval_candidates: List[str],
    span_pct: float,
    steps: int,
    max_combinations: int,
    fee_rate: float,
    rank_mode: str = "score",
    backtest_start_time: pd.Timestamp | str | None = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[OptimizationResult, pd.DataFrame]:
    started = time.perf_counter()
    interval_results = optimize_symbol_interval_results(
        symbol=symbol,
        histories_by_interval=histories_by_interval,
        base_settings=base_settings,
        optimize_flags=optimize_flags,
        interval_candidates=interval_candidates,
        span_pct=span_pct,
        steps=steps,
        max_combinations=max_combinations,
        fee_rate=fee_rate,
        rank_mode=rank_mode,
        backtest_start_time=backtest_start_time,
        should_stop=should_stop,
    )
    best_result: Optional[OptimizationResult] = None
    best_history: Optional[pd.DataFrame] = None
    total_combinations = 0
    trimmed_any = False

    for current, history in interval_results:
        total_combinations += current.combinations_tested
        trimmed_any = trimmed_any or current.trimmed_grid
        if best_result is None:
            best_result = current
            best_history = history
            continue
        current_metrics = current.best_backtest.metrics
        best_metrics = best_result.best_backtest.metrics
        current_key = optimization_sort_key(current_metrics, rank_mode)
        best_key = optimization_sort_key(best_metrics, rank_mode)
        if current_key > best_key:
            best_result = current
            best_history = history

    if best_result is None or best_history is None:
        raise RuntimeError(f"optimization cancelled for {symbol}" if should_stop and should_stop() else f"no optimization result for {symbol}")

    return (
        OptimizationResult(
            symbol=symbol,
            best_interval=best_result.best_interval,
            best_backtest=best_result.best_backtest,
            score=best_result.score,
            combinations_tested=total_combinations,
            duration_seconds=time.perf_counter() - started,
            trimmed_grid=trimmed_any,
        ),
        best_history,
    )


def optimize_symbol_interval_results(
    symbol: str,
    histories_by_interval: Dict[str, pd.DataFrame],
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    interval_candidates: List[str],
    span_pct: float,
    steps: int,
    max_combinations: int,
    fee_rate: float,
    rank_mode: str = "score",
    backtest_start_time: pd.Timestamp | str | None = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Tuple[OptimizationResult, pd.DataFrame]]:
    interval_results: List[Tuple[OptimizationResult, pd.DataFrame]] = []

    for interval in interval_candidates:
        if should_stop and should_stop():
            break
        history = histories_by_interval.get(interval)
        if history is None or history.empty:
            continue
        current = optimize_symbol(
            symbol=symbol,
            df=history,
            base_settings=base_settings,
            optimize_flags=optimize_flags,
            span_pct=span_pct,
            steps=steps,
            max_combinations=max_combinations,
            fee_rate=fee_rate,
            rank_mode=rank_mode,
            backtest_start_time=backtest_start_time,
            should_stop=should_stop,
            result_interval=interval,
        )
        interval_results.append((current, history))

    if not interval_results:
        raise RuntimeError(f"optimization cancelled for {symbol}" if should_stop and should_stop() else f"no optimization result for {symbol}")

    interval_results.sort(
        key=lambda item: optimization_sort_key(item[0].best_backtest.metrics, rank_mode),
        reverse=True,
    )
    return interval_results
