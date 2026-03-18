import multiprocessing as mp
import alt_reversal_trader.trade_engine as trade_engine_module
from dataclasses import replace
import threading
import time

import pandas as pd

from alt_reversal_trader.binance_futures import PositionSnapshot
from alt_reversal_trader.config import DEFAULT_HISTORY_DAYS, StrategySettings
from alt_reversal_trader.strategy import BacktestCursor, BacktestResult, StrategyMetrics, TradeRecord, run_backtest
from alt_reversal_trader.trade_engine import (
    EngineSyncCommand,
    EngineWatchlistItem,
    _EngineSymbolState,
    _OrderRequest,
    _OrderExecutor,
    _OrderExecutionResult,
    _TradeEngine,
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


def make_position(symbol: str = "TESTUSDT") -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        amount=1.0,
        entry_price=100.0,
        mark_price=101.0,
        unrealized_pnl=1.0,
        leverage=3,
    )


def test_trade_engine_defaults_history_days_to_shared_config_default() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())

    assert engine.history_days == DEFAULT_HISTORY_DAYS


def make_signal_backtest(
    *,
    side: str = "long",
    zone: int = 3,
    signal_time: str = "2026-01-01 00:10:00",
    latest_time: str | None = None,
    entry_time: str | None = None,
    price: float = 100.0,
    position_qty: float | None = None,
) -> BacktestResult:
    signal_timestamp = pd.Timestamp(signal_time)
    latest_timestamp = pd.Timestamp(latest_time or signal_time)
    entry_timestamp = pd.Timestamp(entry_time or signal_time)
    label = f"{side[0].upper()}{zone}"
    actual_position_qty = position_qty if position_qty is not None else (1.0 if side == "long" else -1.0)
    cursor = BacktestCursor(
        processed_bars=1,
        last_time=latest_timestamp,
        equity=1000.0,
        position_qty=actual_position_qty,
        avg_entry_price=price,
        entry_side=side,
        entry_time=entry_timestamp,
        entry_price=price,
        zone_events=(label,),
        zone_event_times=((signal_timestamp, label),),
        gross_profit=0.0,
        gross_loss=0.0,
        trade_count=0,
        win_count=0,
        long_zone_used=(zone >= 1, zone >= 2, zone >= 3) if side == "long" else (False, False, False),
        short_zone_used=(zone >= 1, zone >= 2, zone >= 3) if side == "short" else (False, False, False),
        last_long_zone=zone if side == "long" else 0,
        last_short_zone=zone if side == "short" else 0,
        last_entry_signal_time=signal_timestamp,
        last_entry_signal_price=price,
        last_entry_signal_side=side,
        last_entry_signal_zone=zone,
        last_equity_value=1000.0,
    )
    indicators = pd.DataFrame(
        {
            "time": [latest_timestamp],
            "open": [price],
            "high": [price + 1.0],
            "low": [price - 1.0],
            "close": [price],
            "volume": [1000.0],
            "zone2_line": [price - 1.0 if side == "long" else price + 1.0],
            "zone3_line": [price - 2.0 if side == "long" else price + 2.0],
            "trend_to_long": [False],
            "trend_to_short": [False],
            "final_bull": [False],
            "final_bear": [False],
        }
    )
    return BacktestResult(
        settings=StrategySettings(),
        metrics=StrategyMetrics(1.0, 1.0, 0.0, 1, 100.0, 1.0),
        trades=[],
        open_entry_events=((signal_timestamp, label),),
        indicators=indicators,
        latest_state={},
        equity_curve=pd.Series([1000.0], index=[latest_timestamp]),
        cursor=cursor,
    )


def make_stale_multi_zone_backtest() -> BacktestResult:
    indicators = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2026-01-01 00:10:00"),
                pd.Timestamp("2026-01-01 00:20:00"),
                pd.Timestamp("2026-01-01 00:30:00"),
            ],
            "open": [100.0, 120.0, 110.0],
            "high": [101.0, 121.0, 111.0],
            "low": [99.0, 119.0, 109.0],
            "close": [100.0, 120.0, 110.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "trend_to_long": [False, False, False],
            "trend_to_short": [False, False, False],
            "final_bull": [False, False, False],
            "final_bear": [False, False, False],
        }
    )
    cursor = BacktestCursor(
        processed_bars=3,
        last_time=pd.Timestamp("2026-01-01 00:30:00"),
        equity=1000.0,
        position_qty=-1.0,
        avg_entry_price=110.0,
        entry_side="short",
        entry_time=pd.Timestamp("2026-01-01 00:10:00"),
        entry_price=100.0,
        zone_events=("S2", "S3"),
        zone_event_times=(
            (pd.Timestamp("2026-01-01 00:10:00"), "S2"),
            (pd.Timestamp("2026-01-01 00:20:00"), "S3"),
        ),
        gross_profit=0.0,
        gross_loss=0.0,
        trade_count=0,
        win_count=0,
        long_zone_used=(False, False, False),
        short_zone_used=(False, True, True),
        last_long_zone=0,
        last_short_zone=3,
        last_entry_signal_time=pd.Timestamp("2026-01-01 00:20:00"),
        last_entry_signal_price=120.0,
        last_entry_signal_side="short",
        last_entry_signal_zone=3,
        last_equity_value=1000.0,
    )
    return BacktestResult(
        settings=StrategySettings(),
        metrics=StrategyMetrics(1.0, 1.0, 0.0, 1, 100.0, 1.0),
        trades=[],
        open_entry_events=(
            (pd.Timestamp("2026-01-01 00:10:00"), "S2"),
            (pd.Timestamp("2026-01-01 00:20:00"), "S3"),
        ),
        indicators=indicators,
        latest_state={},
        equity_curve=pd.Series([1000.0], index=[pd.Timestamp("2026-01-01 00:30:00")]),
        cursor=cursor,
    )


def make_stale_signal_backtest_after_exit() -> BacktestResult:
    backtest = make_signal_backtest(
        side="short",
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:30:00",
        price=100.0,
        position_qty=0.0,
    )
    trades = [
        TradeRecord(
            side="short",
            entry_time=pd.Timestamp("2026-01-01 00:15:00"),
            exit_time=pd.Timestamp("2026-01-01 00:20:00"),
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
            pnl=-1.0,
            return_pct=-1.0,
            reason="trend_to_long",
            zones="S2",
            entry_events=((pd.Timestamp("2026-01-01 00:15:00"), "S2"),),
        )
    ]
    indicators = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01 00:30:00")],
            "open": [101.0],
            "high": [102.0],
            "low": [100.0],
            "close": [101.0],
            "volume": [1000.0],
            "zone2_line": [101.0],
            "zone3_line": [102.0],
            "trend_to_long": [False],
            "trend_to_short": [False],
            "final_bull": [False],
            "final_bear": [False],
        }
    )
    return BacktestResult(
        settings=backtest.settings,
        metrics=backtest.metrics,
        trades=trades,
        open_entry_events=backtest.open_entry_events,
        indicators=indicators,
        latest_state={},
        equity_curve=pd.Series([1000.0], index=[pd.Timestamp("2026-01-01 00:30:00")]),
        cursor=backtest.cursor,
    )


def make_stale_signal_backtest_after_latest_state_exit() -> BacktestResult:
    backtest = make_signal_backtest(
        side="short",
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:30:00",
        price=100.0,
        position_qty=0.0,
    )
    indicators = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01 00:30:00")],
            "open": [101.0],
            "high": [102.0],
            "low": [100.0],
            "close": [101.0],
            "volume": [1000.0],
            "zone2_line": [101.0],
            "zone3_line": [102.0],
            "trend_to_long": [True],
            "trend_to_short": [False],
            "final_bull": [True],
            "final_bear": [False],
        }
    )
    return replace(
        backtest,
        indicators=indicators,
        latest_state={"trend_to_long": True, "final_bull": True},
    )


class FakeTickerClient:
    def __init__(self, price_by_symbol: dict[str, float]) -> None:
        self.price_by_symbol = dict(price_by_symbol)

    def ticker_24h(self) -> dict[str, dict[str, float]]:
        return {
            symbol: {"lastPrice": price}
            for symbol, price in self.price_by_symbol.items()
        }


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


def test_trade_engine_sync_restores_locked_position_strategies() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    locked_settings = StrategySettings(atr_period=7)
    engine._refresh_positions = lambda force=False: None

    engine._apply_sync(
        EngineSyncCommand(
            api_key="key",
            api_secret="secret",
            leverage=3,
            fee_rate=0.0005,
            history_days=3,
            default_interval="1m",
            default_strategy_settings=StrategySettings(),
            optimization_rank_mode="score",
            auto_trade_enabled=False,
            auto_close_enabled_symbols=(),
            position_intervals={"TESTUSDT": "1m"},
            position_strategy_settings={"TESTUSDT": locked_settings},
            position_filled_fractions={"TESTUSDT": 0.5},
            position_cursor_entry_times={"TESTUSDT": pd.Timestamp("2026-01-01 00:10:00")},
            watchlist=(),
        )
    )

    assert engine.position_intervals["TESTUSDT"] == "1m"
    assert engine.position_strategy_by_symbol["TESTUSDT"] == locked_settings
    assert engine.filled_fraction_by_symbol["TESTUSDT"] == 0.5
    assert engine.auto_trade_cursor_entry_time["TESTUSDT"] == pd.Timestamp("2026-01-01 00:10:00")


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


def test_trade_engine_groups_active_symbols_by_base_interval() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.watchlist[("AAAUSDT", "1m")] = EngineWatchlistItem(
        symbol="AAAUSDT",
        interval="1m",
        score=1.0,
        return_pct=1.0,
        strategy_settings=StrategySettings(),
    )
    engine.watchlist[("BBBUSDT", "2m")] = EngineWatchlistItem(
        symbol="BBBUSDT",
        interval="2m",
        score=1.0,
        return_pct=1.0,
        strategy_settings=StrategySettings(),
    )
    engine.watchlist[("CCCUSDT", "5m")] = EngineWatchlistItem(
        symbol="CCCUSDT",
        interval="5m",
        score=1.0,
        return_pct=1.0,
        strategy_settings=StrategySettings(),
    )

    grouped = engine._desired_stream_targets_by_base_interval()

    assert grouped == {
        "1m": {
            "AAAUSDT": ("1m",),
            "BBBUSDT": ("2m",),
        },
        "5m": {
            "CCCUSDT": ("5m",),
        },
    }


def test_trade_engine_refreshes_stream_targets_when_watchlist_changes() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    stream_snapshots: list[dict[str, dict[str, tuple[str, ...]]]] = []
    engine._refresh_streams = lambda: stream_snapshots.append(engine._desired_stream_targets_by_base_interval())
    engine.watchlist[("AAAUSDT", "1m")] = EngineWatchlistItem(
        symbol="AAAUSDT",
        interval="1m",
        score=1.0,
        return_pct=1.0,
        strategy_settings=StrategySettings(),
    )

    engine._ensure_active_states()

    engine.watchlist.pop(("AAAUSDT", "1m"))
    engine.watchlist[("BBBUSDT", "2m")] = EngineWatchlistItem(
        symbol="BBBUSDT",
        interval="2m",
        score=1.0,
        return_pct=1.0,
        strategy_settings=StrategySettings(),
    )

    engine._ensure_active_states()

    assert stream_snapshots == [
        {"1m": {"AAAUSDT": ("1m",)}},
        {"1m": {"BBBUSDT": ("2m",)}},
    ]


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
    assert state.backtest is not None
    assert state.loading is True
    assert state.needs_reload is True
    assert state.loaded_strategy_settings == old_settings

    engine._load_one_pending_state()

    assert state.loading is False
    assert state.backtest is not None
    assert state.backtest.settings == new_settings
    assert state.loaded_strategy_settings == new_settings
    assert state.needs_reload is False


def test_trade_engine_buffers_closed_bar_until_state_history_is_ready() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    key = ("TESTUSDT", "2m")
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="2m",
        strategy_settings=StrategySettings(),
        history=None,
        backtest=None,
        loading=False,
    )
    closed_bar = {
        "symbol": "TESTUSDT",
        "interval": "2m",
        "time": pd.Timestamp("2026-01-01 00:10:00"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "closed": True,
    }

    engine._handle_closed_bar("TESTUSDT", "2m", closed_bar)

    assert engine.symbol_states[key].loading is True
    assert engine.symbol_states[key].pending_closed_bar == closed_bar


def test_trade_engine_initial_load_triggers_fresh_confirmed_latest_bar() -> None:
    class HistoryClient:
        def historical_ohlcv(self, *args, **kwargs):
            return make_sample_ohlcv(20)

    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = HistoryClient()
    key = ("TESTUSDT", "2m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="2m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="2m",
        strategy_settings=StrategySettings(),
        history=None,
        backtest=None,
        loading=True,
    )
    signal_time = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("2min")
    captured_calls: list[dict[str, object]] = []
    original_run_backtest = trade_engine_module.run_backtest
    engine._start_stream = lambda key: None
    engine._emit_signal_event = lambda state: None
    engine._evaluate_backtest_auto_close = lambda state: None
    engine._evaluate_auto_trade = lambda **kwargs: captured_calls.append(kwargs)
    trade_engine_module.run_backtest = lambda *args, **kwargs: make_signal_backtest(
        side="short",
        zone=3,
        signal_time=str(signal_time),
        latest_time=str(signal_time),
        price=100.0,
    )
    try:
        engine._load_one_pending_state()
    finally:
        trade_engine_module.run_backtest = original_run_backtest

    assert captured_calls == [
        {
            "trigger_symbol": "TESTUSDT",
            "trigger_interval": "2m",
            "trigger_bar_time": signal_time,
        }
    ]


def test_trade_engine_prioritizes_open_position_state_reload() -> None:
    class HistoryClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def historical_ohlcv(self, symbol, interval, start_time=None):
            del interval, start_time
            self.calls.append(str(symbol))
            return make_sample_ohlcv(30)

    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.client = HistoryClient()
    engine._start_stream = lambda key: None
    engine._emit_signal_event = lambda state: None
    engine._evaluate_backtest_auto_close = lambda state: None
    engine._evaluate_auto_trade = lambda **kwargs: None
    engine.watchlist[("WATCHUSDT", "1m")] = EngineWatchlistItem(
        symbol="WATCHUSDT",
        interval="1m",
        score=2.0,
        return_pct=3.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[("WATCHUSDT", "1m")] = _EngineSymbolState(
        symbol="WATCHUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        loading=True,
        reload_requested_at=10.0,
    )
    engine.open_positions["POSUSDT"] = make_position("POSUSDT")
    engine.position_intervals["POSUSDT"] = "1m"
    engine.symbol_states[("POSUSDT", "1m")] = _EngineSymbolState(
        symbol="POSUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        loading=True,
        reload_requested_at=20.0,
    )

    engine._load_one_pending_state()

    assert engine.client.calls == ["POSUSDT"]


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
            close_order=True,
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


def test_trade_engine_uses_rest_price_when_stream_price_is_stale() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_signal_backtest(side="short", zone=2, signal_time="2026-01-01 00:15:00", price=100.0)
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
        loaded_strategy_settings=StrategySettings(),
        last_price_update_at=time.time() - 10.0,
        stale_since=1.0,
    )
    engine.latest_stream_price_by_symbol["TESTUSDT"] = 95.0
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "TESTUSDT"


def test_trade_engine_emits_degraded_and_recovered_health_events_for_stale_position_stream() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.api_key = "key"
    engine.api_secret = "secret"
    engine.streams["1m"] = (None, threading.Event(), tuple())  # type: ignore[arg-type]
    engine.base_stream_started_at["1m"] = 100.0
    engine.base_stream_last_payload_at["1m"] = 100.0
    engine.open_positions["TESTUSDT"] = make_position()
    engine.position_intervals["TESTUSDT"] = "1m"
    state = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=make_signal_backtest(zone=2),
        loaded_strategy_settings=StrategySettings(),
        last_stream_payload_at=100.0,
        last_price_update_at=100.0,
    )
    engine.symbol_states[("TESTUSDT", "1m")] = state
    events: list[object] = []
    refresh_calls: list[bool] = []
    stopped: list[str] = []
    refreshed: list[bool] = []
    engine.emit = lambda event: events.append(event)
    engine._refresh_positions = lambda force=False: refresh_calls.append(force)
    engine._stop_stream = lambda key: stopped.append(str(key))
    engine._refresh_streams = lambda: refreshed.append(True)

    engine._update_stream_health(now=104.0)

    assert state.stale_since == 104.0
    assert refresh_calls == [True]
    assert stopped == ["1m"]
    assert refreshed == [True]
    assert any(getattr(event, "status", "") == "degraded" for event in events)

    state.last_stream_payload_at = 105.0
    state.last_price_update_at = 105.0
    engine.base_stream_last_payload_at["1m"] = 105.0

    engine._update_stream_health(now=105.5)

    assert state.stale_since == 0.0
    assert any(getattr(event, "status", "") == "recovered" for event in events)


def test_trade_engine_keeps_open_position_symbol_active_outside_watchlist() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 80.0})
    engine.open_positions["TESTUSDT"] = make_position("TESTUSDT")
    engine.position_intervals["TESTUSDT"] = "1m"
    engine.filled_fraction_by_symbol["TESTUSDT"] = 0.5
    backtest = make_signal_backtest(side="long", zone=3, signal_time="2026-01-01 00:15:00", price=100.0)
    engine.symbol_states[("TESTUSDT", "1m")] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
        loaded_strategy_settings=StrategySettings(),
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "TESTUSDT"
    assert submitted[0]["fraction"] > 0.0


def test_trade_engine_uses_persisted_filled_fraction_after_restart_for_additional_entry() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine._refresh_positions = lambda force=False: None
    backtest = make_signal_backtest(
        side="long",
        zone=3,
        signal_time="2026-01-01 00:20:00",
        latest_time="2026-01-01 00:25:00",
        entry_time="2026-01-01 00:10:00",
        price=100.0,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._apply_sync(
        EngineSyncCommand(
            api_key="key",
            api_secret="secret",
            leverage=2,
            fee_rate=0.0005,
            history_days=DEFAULT_HISTORY_DAYS,
            default_interval="1m",
            default_strategy_settings=StrategySettings(),
            optimization_rank_mode="score",
            auto_trade_enabled=True,
            auto_close_enabled_symbols=(),
            position_intervals={"TESTUSDT": "1m"},
            position_strategy_settings={"TESTUSDT": StrategySettings()},
            position_filled_fractions={"TESTUSDT": 0.5},
            position_cursor_entry_times={"TESTUSDT": pd.Timestamp("2026-01-01 00:10:00")},
            watchlist=(
                EngineWatchlistItem(
                    symbol="TESTUSDT",
                    interval="1m",
                    score=7.0,
                    return_pct=12.0,
                    strategy_settings=StrategySettings(),
                ),
            ),
        )
    )
    engine.client = FakeTickerClient({"TESTUSDT": 80.0})
    engine.open_positions["TESTUSDT"] = make_position("TESTUSDT")
    engine.position_intervals["TESTUSDT"] = "1m"
    engine.symbol_states[("TESTUSDT", "1m")] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
        loaded_strategy_settings=StrategySettings(),
    )

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "TESTUSDT"
    assert round(float(submitted[0]["fraction"]), 2) == 0.49


def test_trade_engine_drops_symbol_after_successful_close_before_refresh() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.open_positions["TESTUSDT"] = make_position()
    engine.filled_fraction_by_symbol["TESTUSDT"] = 0.5
    engine.auto_trade_cursor_entry_time["TESTUSDT"] = pd.Timestamp("2026-01-01 00:00:00")
    engine.order_result_queue.put(
        _OrderExecutionResult(
            symbol="TESTUSDT",
            success=True,
            message="TESTUSDT close completed: orderId=1",
            auto_close=False,
            auto_trade=False,
            close_order=True,
            interval="1m",
            fraction=0.0,
            strategy_settings=None,
            no_open_position=False,
        )
    )
    engine._refresh_positions = lambda force=False: None

    engine._drain_order_results()

    assert "TESTUSDT" not in engine.open_positions
    assert "TESTUSDT" not in engine.filled_fraction_by_symbol
    assert "TESTUSDT" not in engine.auto_trade_cursor_entry_time


def test_trade_engine_enters_fresh_confirmed_new_signal_on_trigger_bar() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 120.0})
    backtest = make_signal_backtest(zone=2, signal_time="2026-01-01 00:15:00")
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade(
        trigger_symbol="TESTUSDT",
        trigger_interval="1m",
        trigger_bar_time=pd.Timestamp("2026-01-01 00:15:00"),
    )

    assert len(submitted) == 1
    assert submitted[0]["side"] == "BUY"
    assert round(float(submitted[0]["fraction"]), 2) == 0.50


def test_trade_engine_enters_fresh_confirmed_additional_signal_for_open_position() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 120.0})
    backtest = make_signal_backtest(
        zone=3,
        signal_time="2026-01-01 00:20:00",
        entry_time="2026-01-01 00:10:00",
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    engine.open_positions["TESTUSDT"] = make_position()
    engine.position_intervals["TESTUSDT"] = "1m"
    engine.filled_fraction_by_symbol["TESTUSDT"] = 0.5
    engine.auto_trade_cursor_entry_time["TESTUSDT"] = pd.Timestamp("2026-01-01 00:10:00")
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade(
        trigger_symbol="TESTUSDT",
        trigger_interval="1m",
        trigger_bar_time=pd.Timestamp("2026-01-01 00:20:00"),
    )

    assert len(submitted) == 1
    assert submitted[0]["side"] == "BUY"
    assert round(float(submitted[0]["fraction"]), 2) == 0.49


def test_trade_engine_skips_stale_confirmed_signal_without_matching_trigger_bar() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 120.0})
    backtest = make_signal_backtest(
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:16:00",
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade(
        trigger_symbol="TESTUSDT",
        trigger_interval="1m",
        trigger_bar_time=pd.Timestamp("2026-01-01 00:16:00"),
    )

    assert submitted == []


def test_trade_engine_skips_favorable_reentry_after_latest_state_exit_signal() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_stale_signal_backtest_after_latest_state_exit()
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert submitted == []


def test_order_executor_formats_insufficient_margin_errors_for_logs() -> None:
    request = _OrderRequest(
        symbol="TESTUSDT",
        side="BUY",
        leverage=2,
        apply_leverage=True,
        fraction=0.5,
        margin=None,
        price=None,
        interval="1m",
        auto_close=False,
        auto_trade=True,
        reason=None,
        strategy_settings=StrategySettings(),
        close_side=None,
        close_quantity=None,
        api_key="key",
        api_secret="secret",
    )

    message = _OrderExecutor._format_order_failure_message(
        request,
        "RuntimeError: [-2019] Margin is insufficient.",
    )

    assert "insufficient margin" in message.lower()
    assert "TESTUSDT" in message


def test_trade_engine_enqueue_close_order_uses_cached_position_quantity() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.client = object()  # type: ignore[assignment]
    engine.api_key = "key"
    engine.api_secret = "secret"
    engine.open_positions["TESTUSDT"] = make_position()

    engine._enqueue_close_order("TESTUSDT", reason="manual", auto_close=False)

    _priority, _sequence, request = engine.order_request_queue.get_nowait()
    assert request.close_side == "SELL"
    assert request.close_quantity == 1.0


def test_trade_engine_enqueue_open_order_reuses_stream_price_and_cached_leverage() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.client = FakeTickerClient({"TESTUSDT": 101.0})
    engine.api_key = "key"
    engine.api_secret = "secret"
    engine.leverage = 3
    engine.latest_stream_price_by_symbol["TESTUSDT"] = 102.5
    engine.applied_leverage_by_symbol["TESTUSDT"] = 3

    engine._enqueue_open_order(
        symbol="TESTUSDT",
        interval="1m",
        side="BUY",
        fraction=0.5,
        auto_trade=False,
        strategy_settings=StrategySettings(),
    )

    _priority, _sequence, request = engine.order_request_queue.get_nowait()
    assert request.price == 102.5
    assert request.apply_leverage is False


def test_trade_engine_enters_on_favorable_price_without_fresh_confirmed_trigger() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 95.0})
    backtest = make_signal_backtest(
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:16:00",
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["side"] == "BUY"
    assert round(float(submitted[0]["fraction"]), 2) == 0.50


def test_trade_engine_skips_favorable_price_entry_when_toggle_disabled() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.auto_trade_use_favorable_price = False
    engine.client = FakeTickerClient({"TESTUSDT": 95.0})
    backtest = make_signal_backtest(
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:16:00",
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert submitted == []


def test_trade_engine_enters_on_displayed_stale_signal_even_without_backtest_position() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_signal_backtest(
        side="short",
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:16:00",
        price=100.0,
        position_qty=0.0,
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["side"] == "SELL"
    assert round(float(submitted[0]["fraction"]), 2) == 0.50


def test_trade_engine_skips_stale_signal_after_exit_until_new_entry_signal() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_stale_signal_backtest_after_exit()
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert submitted == []


def test_trade_engine_allows_new_short_entry_even_if_opposite_exit_flags_are_set() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_signal_backtest(
        side="short",
        zone=2,
        signal_time="2026-01-01 00:15:00",
        latest_time="2026-01-01 00:16:00",
        price=100.0,
    )
    backtest = replace(
        backtest,
        latest_state={
            "trend_to_long": True,
            "final_bull": True,
        },
    )
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["side"] == "SELL"
    assert round(float(submitted[0]["fraction"]), 2) == 0.50


def test_trade_engine_limits_stale_short_entry_to_s2_when_price_is_between_s2_and_s3() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    engine.client = FakeTickerClient({"TESTUSDT": 110.0})
    backtest = make_stale_multi_zone_backtest()
    key = ("TESTUSDT", "1m")
    engine.watchlist[key] = EngineWatchlistItem(
        symbol="TESTUSDT",
        interval="1m",
        score=7.0,
        return_pct=12.0,
        strategy_settings=StrategySettings(),
    )
    engine.symbol_states[key] = _EngineSymbolState(
        symbol="TESTUSDT",
        interval="1m",
        strategy_settings=StrategySettings(),
        backtest=backtest,
    )
    submitted: list[dict[str, object]] = []
    engine._enqueue_open_order = lambda **kwargs: submitted.append(kwargs)

    engine._evaluate_auto_trade()

    assert len(submitted) == 1
    assert submitted[0]["side"] == "SELL"
    assert round(float(submitted[0]["fraction"]), 2) == 0.50


def test_trade_engine_price_update_triggers_symbol_scoped_auto_trade_eval() -> None:
    engine = _TradeEngine(mp.Queue(), mp.Queue())
    engine.auto_trade_enabled = True
    calls: list[dict[str, object]] = []
    original = engine._evaluate_auto_trade
    engine._evaluate_auto_trade = lambda **kwargs: calls.append(kwargs)

    engine._handle_price_update(
        "TESTUSDT",
        "1m",
        pd.Timestamp("2026-01-01 00:10:00"),
        101.5,
    )

    engine._evaluate_auto_trade = original

    assert engine.latest_stream_price_by_symbol["TESTUSDT"] == 101.5
    assert calls == [{"trigger_symbol": "TESTUSDT", "trigger_interval": "1m"}]
