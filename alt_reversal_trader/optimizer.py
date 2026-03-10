from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Tuple
import time

import pandas as pd

from .config import PARAMETER_SPECS, ParameterSpec, StrategySettings
from .strategy import BacktestResult, run_backtest


@dataclass(frozen=True)
class OptimizationResult:
    symbol: str
    best_backtest: BacktestResult
    combinations_tested: int
    duration_seconds: float
    trimmed_grid: bool


def _value_range(spec: ParameterSpec, base_value, span_pct: float, steps: int, enabled: bool) -> List:
    if not enabled:
        return [base_value]

    if spec.kind == "bool":
        return [False, True]

    if spec.kind == "choice":
        choices = list(spec.choices)
        if base_value not in choices:
            return [base_value]
        center = choices.index(base_value)
        span = max(1, round(len(choices) * (span_pct / 100.0)))
        return choices[max(0, center - span) : min(len(choices), center + span + 1)]

    values: List = []
    ratios = [1.0] if steps <= 1 else [1.0 - (span_pct / 100.0) + ((2 * span_pct / 100.0) * idx / (steps - 1)) for idx in range(steps)]
    for ratio in ratios:
        value = float(base_value) * ratio
        if spec.kind == "int":
            step = int(spec.step or 1)
            value = int(round(value / step) * step)
        else:
            step = float(spec.step or 0.01)
            precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
            value = round(round(value / step) * step, precision)
        if spec.minimum is not None:
            value = max(spec.minimum, value)
        if spec.maximum is not None:
            value = min(spec.maximum, value)
        if value not in values:
            values.append(value)
    if base_value not in values:
        values.append(base_value)
    return sorted(values)


def generate_parameter_grid(
    base_settings: StrategySettings,
    optimize_flags: Dict[str, bool],
    span_pct: float,
    steps: int,
    max_combinations: int,
) -> Tuple[List[StrategySettings], bool]:
    values_by_key = {
        spec.key: _value_range(
            spec,
            getattr(base_settings, spec.key),
            span_pct=span_pct,
            steps=steps,
            enabled=bool(optimize_flags.get(spec.key, False)),
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
        key = max(values_by_key, key=lambda current: len(values_by_key[current]))
        current_values = values_by_key[key]
        if len(current_values) <= 1:
            break
        mid = current_values[len(current_values) // 2]
        reduced = [current_values[0], mid, current_values[-1]]
        deduped = []
        for value in reduced:
            if value not in deduped:
                deduped.append(value)
        values_by_key[key] = deduped if len(deduped) < len(current_values) else [mid]

    keys = list(values_by_key.keys())
    grid: List[StrategySettings] = []
    for combo in product(*(values_by_key[key] for key in keys)):
        payload = {key: combo[idx] for idx, key in enumerate(keys)}
        grid.append(replace(base_settings, **payload))
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
) -> OptimizationResult:
    started = time.perf_counter()
    grid, trimmed = generate_parameter_grid(base_settings, optimize_flags, span_pct, steps, max_combinations)
    best_result: BacktestResult | None = None

    for settings in grid:
        result = run_backtest(df, settings=settings, fee_rate=fee_rate)
        if best_result is None:
            best_result = result
            continue

        current = result.metrics
        best = best_result.metrics
        current_key = (
            current.total_return_pct,
            -current.max_drawdown_pct,
            current.win_rate_pct,
            current.trade_count,
            current.profit_factor,
        )
        best_key = (
            best.total_return_pct,
            -best.max_drawdown_pct,
            best.win_rate_pct,
            best.trade_count,
            best.profit_factor,
        )
        if current_key > best_key:
            best_result = result

    if best_result is None:
        raise RuntimeError(f"no optimization result for {symbol}")

    return OptimizationResult(
        symbol=symbol,
        best_backtest=best_result,
        combinations_tested=len(grid),
        duration_seconds=time.perf_counter() - started,
        trimmed_grid=trimmed,
    )
