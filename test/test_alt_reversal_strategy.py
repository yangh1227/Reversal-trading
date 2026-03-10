import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alt_reversal_trader.config import DEFAULT_OPTIMIZE_FLAGS, StrategySettings
from alt_reversal_trader.optimizer import generate_parameter_grid
from alt_reversal_trader.strategy import run_backtest


def make_sample_ohlcv(rows: int = 500) -> pd.DataFrame:
    time_index = pd.date_range("2026-01-01", periods=rows, freq="min")
    base = pd.Series(range(rows), dtype=float)
    close = 100 + (base * 0.03) + (base % 15 - 7) * 0.4
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.6
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.6
    volume = 5000 + (base % 20) * 75
    return pd.DataFrame(
        {
            "time": time_index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_backtest_smoke() -> None:
    df = make_sample_ohlcv()
    result = run_backtest(df, settings=StrategySettings())
    assert result.metrics.trade_count >= 0
    assert not result.indicators.empty
    assert {"supertrend", "final_bull", "final_bear"}.issubset(result.indicators.columns)


def test_parameter_grid_respects_limits() -> None:
    flags = DEFAULT_OPTIMIZE_FLAGS.copy()
    flags["use_volume"] = True
    flags["use_qip"] = True
    grid, trimmed = generate_parameter_grid(
        StrategySettings(),
        optimize_flags=flags,
        span_pct=20.0,
        steps=5,
        max_combinations=150,
    )
    assert len(grid) <= 150
    assert trimmed is True
