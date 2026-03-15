import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alt_reversal_trader.config import DEFAULT_OPTIMIZE_FLAGS, StrategySettings
from alt_reversal_trader.binance_futures import BinanceFuturesClient, resample_ohlcv
from alt_reversal_trader.live_chart_utils import history_with_live_preview, seed_two_minute_aggregate
from alt_reversal_trader.optimizer import (
    generate_parameter_grid,
    optimization_sort_key,
    optimize_symbol_interval_results,
    optimize_symbol_intervals,
    score_optimization_metrics,
)
from alt_reversal_trader.strategy import StrategyMetrics
from alt_reversal_trader.strategy import (
    BacktestCursor,
    BacktestResult,
    latest_confirmed_entry_event,
    incremental_signal_fraction_for_entry,
    resume_backtest,
    run_backtest,
    run_backtest_metrics,
    signal_fraction_for_zone,
)


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


def make_confirmed_entry_backtest(
    *,
    open_entry_events: tuple[tuple[pd.Timestamp, str], ...],
    cursor_signal_time: str | None,
    cursor_signal_side: str,
    cursor_signal_zone: int,
) -> BacktestResult:
    latest_time = pd.Timestamp("2026-01-01 00:10:00")
    cursor = BacktestCursor(
        processed_bars=1,
        last_time=latest_time,
        equity=1000.0,
        position_qty=1.0,
        avg_entry_price=100.0,
        entry_side="long",
        entry_time=latest_time,
        entry_price=100.0,
        zone_events=("L1",),
        zone_event_times=open_entry_events,
        gross_profit=0.0,
        gross_loss=0.0,
        trade_count=0,
        win_count=0,
        long_zone_used=(True, False, False),
        short_zone_used=(False, False, False),
        last_long_zone=1,
        last_short_zone=0,
        last_entry_signal_time=None if cursor_signal_time is None else pd.Timestamp(cursor_signal_time),
        last_entry_signal_price=100.0,
        last_entry_signal_side=cursor_signal_side,
        last_entry_signal_zone=cursor_signal_zone,
        last_equity_value=1000.0,
    )
    indicators = pd.DataFrame(
        {
            "time": [latest_time],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        }
    )
    return BacktestResult(
        settings=StrategySettings(),
        metrics=StrategyMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0),
        trades=[],
        open_entry_events=open_entry_events,
        indicators=indicators,
        latest_state={},
        equity_curve=pd.Series([1000.0], index=[latest_time]),
        cursor=cursor,
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
    assert sorted({settings.factor for settings in grid}) == [1.8, 2.4, 3.0, 3.6, 4.2]


def test_signal_fraction_targets_use_total_position_sizes() -> None:
    assert signal_fraction_for_zone(1) == 0.33
    assert signal_fraction_for_zone(2) == 0.50
    assert signal_fraction_for_zone(3) == 0.99
    assert round(incremental_signal_fraction_for_entry(2, 0), 2) == 0.50
    assert round(incremental_signal_fraction_for_entry(2, 1), 2) == 0.17
    assert round(incremental_signal_fraction_for_entry(3, 2), 2) == 0.49


def test_latest_confirmed_entry_event_prefers_open_entry_events() -> None:
    backtest = make_confirmed_entry_backtest(
        open_entry_events=((pd.Timestamp("2026-01-01 00:10:00"), "S3"),),
        cursor_signal_time="2026-01-01 00:10:00",
        cursor_signal_side="long",
        cursor_signal_zone=1,
    )

    event = latest_confirmed_entry_event(backtest, pd.Timestamp("2026-01-01 00:10:00"))

    assert event == {
        "side": "short",
        "zone": 3,
        "bar_time": pd.Timestamp("2026-01-01 00:10:00"),
    }


def test_latest_confirmed_entry_event_falls_back_to_cursor_signal() -> None:
    backtest = make_confirmed_entry_backtest(
        open_entry_events=(),
        cursor_signal_time="2026-01-01 00:10:00",
        cursor_signal_side="short",
        cursor_signal_zone=2,
    )

    event = latest_confirmed_entry_event(backtest, pd.Timestamp("2026-01-01 00:10:00"))

    assert event == {
        "side": "short",
        "zone": 2,
        "bar_time": pd.Timestamp("2026-01-01 00:10:00"),
    }


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


def test_optimize_symbol_interval_results_returns_each_interval_case() -> None:
    df = make_sample_ohlcv(720)
    histories = {
        "1m": df,
        "2m": resample_ohlcv(df, "2m"),
    }
    flags = DEFAULT_OPTIMIZE_FLAGS.copy()
    flags["atr_period"] = True
    interval_results = optimize_symbol_interval_results(
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
    assert [result.best_interval for result, _history in interval_results] == ["1m", "2m"] or [
        result.best_interval for result, _history in interval_results
    ] == ["2m", "1m"]
    assert {result.best_interval for result, _history in interval_results} == {"1m", "2m"}


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


def test_optimization_score_rewards_more_confirmed_trades_and_profit_factor() -> None:
    fewer_trades = StrategyMetrics(
        total_return_pct=20.0,
        net_profit=200.0,
        max_drawdown_pct=12.0,
        trade_count=10,
        win_rate_pct=58.0,
        profit_factor=1.4,
    )
    more_trades_better_pf = StrategyMetrics(
        total_return_pct=20.0,
        net_profit=200.0,
        max_drawdown_pct=12.0,
        trade_count=40,
        win_rate_pct=58.0,
        profit_factor=2.1,
    )
    assert score_optimization_metrics(more_trades_better_pf) > score_optimization_metrics(fewer_trades)


def test_optimization_sort_key_can_prefer_return_mode() -> None:
    higher_score_lower_return = StrategyMetrics(
        total_return_pct=18.0,
        net_profit=180.0,
        max_drawdown_pct=6.0,
        trade_count=36,
        win_rate_pct=68.0,
        profit_factor=2.3,
    )
    higher_return_lower_score = StrategyMetrics(
        total_return_pct=26.0,
        net_profit=260.0,
        max_drawdown_pct=15.0,
        trade_count=12,
        win_rate_pct=48.0,
        profit_factor=1.2,
    )
    assert score_optimization_metrics(higher_score_lower_return) > score_optimization_metrics(higher_return_lower_score)
    assert optimization_sort_key(higher_return_lower_score, "return") > optimization_sort_key(
        higher_score_lower_return,
        "return",
    )


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


def test_history_with_live_preview_appends_without_mutating_source() -> None:
    confirmed = make_sample_ohlcv(3)
    preview_bar = {
        "symbol": "TESTUSDT",
        "interval": "1m",
        "time": confirmed["time"].iloc[-1] + pd.Timedelta(minutes=1),
        "open": 111.0,
        "high": 112.0,
        "low": 110.0,
        "close": 111.5,
        "volume": 77.0,
        "quote_volume": 8585.5,
    }
    displayed = history_with_live_preview(confirmed, preview_bar)
    assert displayed is not None
    assert len(displayed) == len(confirmed) + 1
    assert displayed["time"].iloc[-1] == preview_bar["time"]
    assert float(displayed["quote_volume"].iloc[-1]) == preview_bar["quote_volume"]
    assert len(confirmed) == 3


def test_history_with_live_preview_replaces_same_timestamp() -> None:
    confirmed = make_sample_ohlcv(3)
    preview_bar = {
        "symbol": "TESTUSDT",
        "interval": "1m",
        "time": confirmed["time"].iloc[-1],
        "open": 201.0,
        "high": 203.0,
        "low": 200.0,
        "close": 202.0,
        "volume": 91.0,
    }
    displayed = history_with_live_preview(confirmed, preview_bar)
    assert displayed is not None
    assert len(displayed) == len(confirmed)
    assert float(displayed["close"].iloc[-1]) == 202.0
    assert float(confirmed["close"].iloc[-1]) != 202.0


def test_seed_two_minute_aggregate_uses_only_first_minute_seed() -> None:
    first_minute = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:02:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 20.0],
            "quote_volume": [1000.0, 2100.0],
        }
    )
    seed = seed_two_minute_aggregate(first_minute, "TESTUSDT", "2m")
    assert seed is not None
    assert seed["time"] == pd.Timestamp("2026-01-01 00:02:00")
    assert float(seed["base_volume"]) == 20.0
    assert float(seed["base_quote_volume"]) == 2100.0

    second_minute = first_minute.copy()
    second_minute.loc[1, "time"] = pd.Timestamp("2026-01-01 00:03:00")
    no_seed = seed_two_minute_aggregate(second_minute, "TESTUSDT", "2m")
    assert no_seed is None
