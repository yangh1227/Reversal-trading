import multiprocessing as mp

import pandas as pd

from alt_reversal_trader.binance_futures import PositionSnapshot
from alt_reversal_trader.config import StrategySettings
from alt_reversal_trader.strategy import BacktestCursor, BacktestResult, StrategyMetrics, run_backtest
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


def make_signal_backtest(
    *,
    side: str = "long",
    zone: int = 3,
    signal_time: str = "2026-01-01 00:10:00",
    latest_time: str | None = None,
    entry_time: str | None = None,
    price: float = 100.0,
) -> BacktestResult:
    signal_timestamp = pd.Timestamp(signal_time)
    latest_timestamp = pd.Timestamp(latest_time or signal_time)
    entry_timestamp = pd.Timestamp(entry_time or signal_time)
    label = f"{side[0].upper()}{zone}"
    position_qty = 1.0 if side == "long" else -1.0
    cursor = BacktestCursor(
        processed_bars=1,
        last_time=latest_timestamp,
        equity=1000.0,
        position_qty=position_qty,
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
