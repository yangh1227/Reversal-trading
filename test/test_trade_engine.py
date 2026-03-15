import multiprocessing as mp

import pandas as pd

from alt_reversal_trader.binance_futures import PositionSnapshot
from alt_reversal_trader.config import StrategySettings
from alt_reversal_trader.strategy import run_backtest
from alt_reversal_trader.trade_engine import EngineWatchlistItem, _EngineSymbolState, _OrderExecutionResult, _TradeEngine


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


def make_position(symbol: str = "TESTUSDT") -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        amount=1.0,
        entry_price=100.0,
        mark_price=101.0,
        unrealized_pnl=1.0,
        leverage=3,
    )


def test_trade_engine_prefers_locked_strategy_for_open_positions() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    locked_settings = StrategySettings(atr_period=7)
    watchlist_settings = StrategySettings(atr_period=21)
    engine.position_strategy_by_symbol["TESTUSDT"] = locked_settings
    engine.open_positions["TESTUSDT"] = make_position()
    engine.watchlist[("TESTUSDT", "1m")] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=1.0,
        return_pct=2.0,
        strategy_settings=watchlist_settings,
    )

    assert engine._settings_for_key("TESTUSDT", "1m") == locked_settings

    engine.open_positions.clear()

    assert engine._settings_for_key("TESTUSDT", "1m") == watchlist_settings


def test_trade_engine_keeps_locked_position_strategy_when_watchlist_updates() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    key = ("TESTUSDT", "1m")
    locked_settings = StrategySettings(atr_period=7)
    watchlist_settings = StrategySettings(atr_period=21)
    history = make_sample_ohlcv(320)
    backtest = run_backtest(history, settings=locked_settings)
    engine.open_positions["TESTUSDT"] = make_position()
    engine.position_intervals["TESTUSDT"] = "1m"
    engine.position_strategy_by_symbol["TESTUSDT"] = locked_settings
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=10.0,
        return_pct=20.0,
        strategy_settings=watchlist_settings,
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=locked_settings,
        history=history,
        backtest=backtest,
        loading=False,
    )

    engine._ensure_active_states()

    state = engine.symbol_states[key]
    assert state.strategy_settings == locked_settings
    assert state.backtest == backtest
    assert state.loading is False


def test_trade_engine_rebuilds_changed_state_from_cached_history_without_refetch() -> None:
    class NoFetchClient:
        def historical_ohlcv(self, *args, **kwargs):
            raise AssertionError("historical_ohlcv should not be called when cached history exists")

    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.client = NoFetchClient()
    engine._start_stream = lambda key: None
    engine._emit_signal_event = lambda state: None
    engine._evaluate_backtest_auto_close = lambda state: None
    engine._evaluate_auto_trade = lambda: None
    key = ("TESTUSDT", "1m")
    old_settings = StrategySettings(atr_period=7)
    new_settings = StrategySettings(atr_period=21)
    history = make_sample_ohlcv(320)
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=10.0,
        return_pct=20.0,
        strategy_settings=new_settings,
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=old_settings,
        history=history,
        backtest=run_backtest(history, settings=old_settings),
        loading=False,
    )

    engine._ensure_active_states()

    state = engine.symbol_states[key]
    assert state.strategy_settings == new_settings
    assert state.backtest is None
    assert state.loading is True

    engine._load_one_pending_state()

    assert state.loading is False
    assert state.backtest is not None
    assert state.backtest.settings == new_settings


def test_trade_engine_drops_symbol_after_auto_close_reports_no_open_position() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.open_positions["TESTUSDT"] = make_position()
    engine.filled_fraction_by_symbol["TESTUSDT"] = 0.5
    engine.auto_close_last_trigger_time["TESTUSDT"] = pd.Timestamp("2026-01-01 00:00:00")
    engine.auto_close_last_attempt_at["TESTUSDT"] = 123.0
    engine.order_result_queue.put(
        _OrderExecutionResult(
            symbol="TESTUSDT",
            success=True,
            message="TESTUSDT close skipped: no open position",
            auto_close=True,
            auto_trade=False,
            interval="1m",
            fraction=0.0,
            strategy_settings=None,
            no_open_position=True,
        )
    )
    engine._refresh_positions = lambda force=False: None

    engine._drain_order_results()

    assert "TESTUSDT" not in engine.open_positions
    assert "TESTUSDT" not in engine.filled_fraction_by_symbol
    assert "TESTUSDT" not in engine.auto_close_last_trigger_time
    assert "TESTUSDT" not in engine.auto_close_last_attempt_at
