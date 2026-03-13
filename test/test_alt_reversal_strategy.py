import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alt_reversal_trader.config import DEFAULT_OPTIMIZE_FLAGS, StrategySettings
from alt_reversal_trader.binance_futures import BinanceFuturesClient, resample_ohlcv
from alt_reversal_trader.optimizer import generate_parameter_grid, optimize_symbol_intervals, score_optimization_metrics
from alt_reversal_trader.strategy import StrategyMetrics
from alt_reversal_trader.strategy import resume_backtest, run_backtest, run_backtest_metrics


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
    assert result.indicators["supertrend"].notna().any()
    assert result.indicators["ema_fast"].notna().any()


def test_backtest_respects_start_time_window() -> None:
    df = make_sample_ohlcv(800)
    start_time = df["time"].iloc[500]
    result = run_backtest(df, settings=StrategySettings(), backtest_start_time=start_time)
    assert not result.indicators.empty
    assert result.indicators["time"].iloc[0] >= start_time
    assert result.equity_curve.index[0] >= start_time


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


def test_parameter_grid_uses_profiled_ranges() -> None:
    flags = {key: False for key in DEFAULT_OPTIMIZE_FLAGS}
    flags["atr_period"] = True
    flags["factor"] = True
    grid, trimmed = generate_parameter_grid(
        StrategySettings(),
        optimize_flags=flags,
        span_pct=20.0,
        steps=5,
        max_combinations=100,
    )
    assert trimmed is False
    assert sorted({settings.atr_period for settings in grid}) == [6, 8, 10, 12, 14]
    assert sorted({settings.factor for settings in grid}) == [2.2, 2.6, 3.0, 3.4, 3.8]


def test_parameter_grid_filters_invalid_combinations() -> None:
    base = StrategySettings(
        qip_ema_fast=40,
        qip_ema_slow=42,
        qtp_ema_fast_len=10,
        qtp_ema_slow_len=12,
        qtp_min_pvt_left=4,
        qtp_max_pvt_left=5,
        qip_rsi_bull_max=45,
        qip_rsi_bear_min=47,
        qtp_rsi_bull_max=44,
        qtp_rsi_bear_min=54,
    )
    flags = {key: False for key in DEFAULT_OPTIMIZE_FLAGS}
    for key in (
        "qip_ema_fast",
        "qip_ema_slow",
        "qtp_ema_fast_len",
        "qtp_ema_slow_len",
        "qtp_min_pvt_left",
        "qtp_max_pvt_left",
        "qip_rsi_bull_max",
        "qip_rsi_bear_min",
        "qtp_rsi_bull_max",
        "qtp_rsi_bear_min",
    ):
        flags[key] = True
    grid, _ = generate_parameter_grid(
        base,
        optimize_flags=flags,
        span_pct=40.0,
        steps=7,
        max_combinations=800,
    )
    assert grid
    for settings in grid:
        assert settings.qip_ema_fast < settings.qip_ema_slow
        assert settings.qtp_ema_fast_len < settings.qtp_ema_slow_len
        assert settings.qtp_min_pvt_left <= settings.qtp_max_pvt_left
        assert settings.qip_rsi_bull_max < settings.qip_rsi_bear_min
        assert settings.qtp_rsi_bull_max < settings.qtp_rsi_bear_min


def test_backtest_metrics_match_full_result() -> None:
    df = make_sample_ohlcv(900)
    start_time = df["time"].iloc[300]
    full = run_backtest(df, settings=StrategySettings(), backtest_start_time=start_time)
    metrics_only = run_backtest_metrics(df, settings=StrategySettings(), backtest_start_time=start_time)
    assert metrics_only == full.metrics


def test_resume_backtest_matches_full_rebuild() -> None:
    df = make_sample_ohlcv(960)
    start_time = df["time"].iloc[240]
    partial = df.iloc[:720].reset_index(drop=True)
    extended = df.iloc[:780].reset_index(drop=True)
    initial = run_backtest(partial, settings=StrategySettings(), backtest_start_time=start_time)
    resumed = resume_backtest(extended, initial, settings=StrategySettings(), backtest_start_time=start_time)
    full = run_backtest(extended, settings=StrategySettings(), backtest_start_time=start_time)
    assert resumed.metrics == full.metrics
    assert resumed.latest_state == full.latest_state
    assert resumed.trades == full.trades
    assert resumed.cursor == full.cursor
    pd.testing.assert_frame_equal(resumed.indicators.reset_index(drop=True), full.indicators.reset_index(drop=True))
    pd.testing.assert_series_equal(resumed.equity_curve, full.equity_curve)


def test_resample_ohlcv_to_2m() -> None:
    df = make_sample_ohlcv(6)
    resampled = resample_ohlcv(df, "2m")
    assert len(resampled) == 3
    assert resampled["time"].iloc[0] == df["time"].iloc[0]
    assert resampled["open"].iloc[0] == df["open"].iloc[0]
    assert resampled["close"].iloc[0] == df["close"].iloc[1]


def test_optimize_symbol_intervals_returns_best_interval() -> None:
    df = make_sample_ohlcv(720)
    histories = {
        "1m": df,
        "2m": resample_ohlcv(df, "2m"),
    }
    flags = DEFAULT_OPTIMIZE_FLAGS.copy()
    flags["atr_period"] = True
    result, history = optimize_symbol_intervals(
        symbol="TESTUSDT",
        histories_by_interval=histories,
        base_settings=StrategySettings(),
        optimize_flags=flags,
        interval_candidates=["1m", "2m"],
        span_pct=10.0,
        steps=3,
        max_combinations=50,
        fee_rate=0.0005,
        backtest_start_time=df["time"].iloc[200],
    )
    assert result.best_interval in {"1m", "2m"}
    assert 0.0 <= result.score <= 100.0
    assert not history.empty


def test_optimization_score_weights_return_most() -> None:
    higher_return = StrategyMetrics(
        total_return_pct=32.0,
        net_profit=320.0,
        max_drawdown_pct=12.0,
        trade_count=20,
        win_rate_pct=58.0,
        profit_factor=1.7,
    )
    lower_return = StrategyMetrics(
        total_return_pct=18.0,
        net_profit=180.0,
        max_drawdown_pct=10.0,
        trade_count=20,
        win_rate_pct=60.0,
        profit_factor=1.8,
    )
    assert score_optimization_metrics(higher_return) > score_optimization_metrics(lower_return)


def test_get_open_positions_recomputes_unrealized_pnl() -> None:
    client = BinanceFuturesClient()
    client._request = lambda *args, **kwargs: [  # type: ignore[method-assign]
        {
            "symbol": "TESTUSDT",
            "positionAmt": "2.5",
            "entryPrice": "100",
            "markPrice": "104",
            "unRealizedProfit": "9999",
            "leverage": "3",
        }
    ]
    positions = client.get_open_positions()
    assert len(positions) == 1
    assert abs(positions[0].unrealized_pnl - 10.0) < 1e-9
