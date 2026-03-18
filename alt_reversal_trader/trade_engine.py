from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from queue import Empty, PriorityQueue, Queue
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

import pandas as pd
import websocket

from .auto_trade_runtime import (
    auto_trade_signal_from_backtest as _auto_trade_signal_from_backtest,
    evaluate_auto_trade_candidate,
    favorable_auto_trade_fraction,
    history_can_resume_backtest as _history_can_resume_backtest,
    inferred_auto_trade_fraction as _inferred_auto_trade_fraction,
    pick_auto_trade_candidate,
)
from .binance_futures import BinanceFuturesClient, PositionSnapshot, resolve_base_interval
from .config import APP_INTERVAL_OPTIONS, DEFAULT_HISTORY_DAYS, StrategySettings
from .live_chart_utils import merge_live_bar as _merge_live_bar
from .live_chart_utils import seed_two_minute_aggregate as _seed_two_minute_aggregate
from .live_chart_utils import transform_two_minute_bar as _transform_two_minute_bar
from .strategy import (
    BacktestResult,
    evaluate_latest_state,
    estimate_warmup_bars,
    latest_confirmed_entry_event,
    prepare_ohlcv,
    resume_backtest,
    run_backtest,
    signal_fraction_for_zone,
)


AUTO_TRADE_RECHECK_INTERVAL_SECONDS = 1.0
AUTO_CLOSE_RETRY_INTERVAL_SECONDS = 10.0
POSITION_REFRESH_INTERVAL_SECONDS = 1.0
COMMAND_POLL_TIMEOUT_SECONDS = 0.25
STREAM_RECONNECT_DELAY_SECONDS = 2.0
BACKTEST_WARMUP_BAR_FLOOR = 1_500
PENDING_ORDER_TIMEOUT_SECONDS = 30.0
STREAM_PRICE_EVAL_MIN_INTERVAL_SECONDS = 0.2
STREAM_STALE_SECONDS = 3.0
STREAM_FORCE_RELOAD_SECONDS = 10.0
STALE_SYMBOL_EVAL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class EngineWatchlistItem:
    symbol: str
    interval: str
    score: float
    return_pct: float
    strategy_settings: StrategySettings


@dataclass(frozen=True)
class EngineSyncCommand:
    api_key: str
    api_secret: str
    leverage: int
    fee_rate: float
    history_days: int
    default_interval: str
    default_strategy_settings: StrategySettings
    optimization_rank_mode: str
    auto_trade_enabled: bool
    auto_trade_use_favorable_price: bool
    auto_close_enabled_symbols: Tuple[str, ...]
    position_intervals: Dict[str, str]
    position_strategy_settings: Dict[str, StrategySettings]
    position_filled_fractions: Dict[str, float]
    position_cursor_entry_times: Dict[str, pd.Timestamp]
    watchlist: Tuple[EngineWatchlistItem, ...]


@dataclass(frozen=True)
class EngineOpenOrderCommand:
    symbol: str
    interval: str
    side: str
    leverage: int
    fraction: Optional[float] = None
    margin: Optional[float] = None
    auto_trade: bool = False
    strategy_settings: Optional[StrategySettings] = None


@dataclass(frozen=True)
class EngineCloseOrderCommand:
    symbol: str
    reason: Optional[str] = None
    auto_close: bool = False


@dataclass(frozen=True)
class EngineStopCommand:
    pass


@dataclass(frozen=True)
class EngineLogEvent:
    message: str


@dataclass(frozen=True)
class EngineHealthEvent:
    status: str
    detail: str = ""


@dataclass(frozen=True)
class EngineOrderCompletedEvent:
    symbol: str
    message: str
    auto_close: bool = False
    auto_trade: bool = False
    interval: Optional[str] = None
    fraction: float = 0.0
    strategy_settings: Optional[StrategySettings] = None


@dataclass(frozen=True)
class EngineOrderSubmittedEvent:
    symbol: str
    auto_close: bool = False
    auto_trade: bool = False
    interval: Optional[str] = None
    fraction: float = 0.0


@dataclass(frozen=True)
class EngineOrderFailedEvent:
    symbol: str
    message: str
    auto_close: bool = False
    auto_trade: bool = False
    interval: Optional[str] = None
    fraction: float = 0.0


@dataclass(frozen=True)
class EngineSignalEvent:
    symbol: str
    interval: str
    entry_side: str = ""
    entry_zone: int = 0
    preview_entry_side: str = ""
    preview_entry_zone: int = 0
    exit_reason: str = ""
    bar_time: Optional[pd.Timestamp] = None


@dataclass
class _EngineSymbolState:
    symbol: str
    interval: str
    strategy_settings: StrategySettings
    score: float = 0.0
    return_pct: float = 0.0
    history: Optional[pd.DataFrame] = None
    backtest: Optional[BacktestResult] = None
    loaded_strategy_settings: Optional[StrategySettings] = None
    loading: bool = False
    needs_reload: bool = False
    force_history_refetch: bool = False
    pending_closed_bar: Optional[Dict[str, object]] = None
    priority_class: str = "watchlist"
    last_stream_payload_at: float = 0.0
    last_price_update_at: float = 0.0
    stale_since: float = 0.0
    reload_requested_at: float = 0.0


@dataclass(frozen=True)
class _OrderRequest:
    symbol: str
    side: Optional[str]
    leverage: int
    apply_leverage: bool
    fraction: Optional[float]
    margin: Optional[float]
    price: Optional[float]
    interval: Optional[str]
    auto_close: bool
    auto_trade: bool
    reason: Optional[str]
    strategy_settings: Optional[StrategySettings]
    close_side: Optional[str]
    close_quantity: Optional[float]
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class _OrderExecutionResult:
    symbol: str
    success: bool
    message: str
    auto_close: bool
    auto_trade: bool
    close_order: bool
    interval: Optional[str]
    fraction: float
    strategy_settings: Optional[StrategySettings]
    no_open_position: bool = False


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"unsupported interval: {interval}")


def _ws_kline_timestamp(time_ms: int) -> pd.Timestamp:
    return pd.to_datetime(int(time_ms), unit="ms", utc=True).tz_convert(None)


def _backtest_start_time_ms(history_days: int) -> int:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    return now_ms - max(int(history_days), 1) * 86_400_000


def _fresh_initial_trigger_bar_time(
    backtest: Optional[BacktestResult],
    interval: str,
    now_time: Optional[pd.Timestamp] = None,
) -> Optional[pd.Timestamp]:
    if backtest is None or backtest.indicators.empty or "time" not in backtest.indicators.columns:
        return None
    latest_bar_time = pd.Timestamp(backtest.indicators["time"].iloc[-1]).tz_localize(None)
    if latest_confirmed_entry_event(backtest, latest_bar_time) is None:
        return None
    reference_time = now_time
    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if latest_bar_time > reference_time:
        return None
    freshness_window = pd.Timedelta(milliseconds=_interval_to_ms(interval) * 1.5)
    if reference_time - latest_bar_time > freshness_window:
        return None
    return latest_bar_time


def _history_fetch_start_time_ms(history_days: int, interval: str, strategy_settings: StrategySettings) -> int:
    interval_ms = _interval_to_ms(interval)
    warmup_bars = max(estimate_warmup_bars(strategy_settings), BACKTEST_WARMUP_BAR_FLOOR)
    return _backtest_start_time_ms(history_days) - warmup_bars * interval_ms


def _is_provisional_exit_trade(trade, latest_time: Optional[pd.Timestamp]) -> bool:
    if latest_time is None:
        return False
    return trade.reason == "end_of_test" and pd.Timestamp(trade.exit_time) == latest_time


def _latest_backtest_exit_event(backtest: Optional[BacktestResult]) -> Optional[Dict[str, object]]:
    if backtest is None or backtest.indicators.empty or not backtest.trades or "time" not in backtest.indicators.columns:
        return None
    latest_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    for trade in reversed(backtest.trades):
        exit_time = pd.Timestamp(trade.exit_time)
        if exit_time < latest_time:
            break
        if exit_time != latest_time or _is_provisional_exit_trade(trade, latest_time):
            continue
        return {
            "side": str(trade.side).lower(),
            "reason": str(trade.reason),
            "bar_time": latest_time,
        }
    return None


def _preview_exit_reason(position_qty: float, latest_state: Dict[str, object]) -> Optional[str]:
    if position_qty > 0:
        if bool(latest_state.get("trend_to_short")):
            return "trend_to_short"
        if bool(latest_state.get("final_bear")):
            return "opposite_signal"
        return None
    if position_qty < 0:
        if bool(latest_state.get("trend_to_long")):
            return "trend_to_long"
        if bool(latest_state.get("final_bull")):
            return "opposite_signal"
    return None


def _preview_entry_signal(cursor, latest_state: Dict[str, object], settings: StrategySettings) -> Optional[Tuple[str, int]]:
    if cursor is None:
        return None
    position_qty = float(getattr(cursor, "position_qty", 0.0))
    long_zone_used = list(getattr(cursor, "long_zone_used", (False, False, False)))
    short_zone_used = list(getattr(cursor, "short_zone_used", (False, False, False)))
    last_long_zone = int(getattr(cursor, "last_long_zone", 0))
    last_short_zone = int(getattr(cursor, "last_short_zone", 0))

    def reset_zones() -> None:
        nonlocal last_long_zone, last_short_zone
        long_zone_used[:] = [False, False, False]
        short_zone_used[:] = [False, False, False]
        last_long_zone = 0
        last_short_zone = 0

    if abs(position_qty) < 1e-12:
        reset_zones()
    if _preview_exit_reason(position_qty, latest_state) is not None:
        position_qty = 0.0
        reset_zones()
    if abs(position_qty) < 1e-12:
        reset_zones()

    lev_zone = int(latest_state.get("zone") or 0)
    if lev_zone not in {1, 2, 3}:
        return None

    trend = str(latest_state.get("trend", "")).upper()
    is_long_trend = trend == "LONG"
    is_short_trend = trend == "SHORT"
    final_bull = bool(latest_state.get("final_bull"))
    final_bear = bool(latest_state.get("final_bear"))

    can_long_z1 = (not settings.beast_mode) and is_long_trend and final_bull and lev_zone == 1 and (not long_zone_used[0]) and last_long_zone == 0
    can_long_z2 = is_long_trend and final_bull and lev_zone == 2 and (not long_zone_used[1]) and last_long_zone in (0, 1)
    can_long_z3 = is_long_trend and final_bull and lev_zone == 3 and (not long_zone_used[2]) and last_long_zone in (0, 2)
    can_short_z1 = (not settings.beast_mode) and is_short_trend and final_bear and lev_zone == 1 and (not short_zone_used[0]) and last_short_zone == 0
    can_short_z2 = is_short_trend and final_bear and lev_zone == 2 and (not short_zone_used[1]) and last_short_zone in (0, 1)
    can_short_z3 = is_short_trend and final_bear and lev_zone == 3 and (not short_zone_used[2]) and last_short_zone in (0, 2)

    if can_long_z1:
        return ("long", 1)
    if can_long_z2:
        return ("long", 2)
    if can_long_z3:
        return ("long", 3)
    if can_short_z1:
        return ("short", 1)
    if can_short_z2:
        return ("short", 2)
    if can_short_z3:
        return ("short", 3)
    return None


def _confirmed_exit_event_from_position_backtest(
    position: Optional[PositionSnapshot],
    backtest: Optional[BacktestResult],
) -> Optional[Dict[str, object]]:
    if (
        position is None
        or backtest is None
        or backtest.indicators.empty
        or "time" not in backtest.indicators.columns
    ):
        return None
    amount = float(position.amount)
    if abs(amount) < 1e-12:
        return None
    latest_state = dict(backtest.latest_state or {})
    reason: Optional[str] = None
    side = "long" if amount > 0 else "short"
    if amount > 0:
        if bool(latest_state.get("trend_to_short")):
            reason = "trend_to_short"
        elif bool(latest_state.get("final_bear")):
            reason = "opposite_signal"
    else:
        if bool(latest_state.get("trend_to_long")):
            reason = "trend_to_long"
        elif bool(latest_state.get("final_bull")):
            reason = "opposite_signal"
    if reason is None:
        return None
    latest_time = pd.Timestamp(backtest.indicators["time"].iloc[-1])
    return {
        "side": side,
        "reason": reason,
        "bar_time": latest_time,
    }


def _auto_close_reason(position: Optional[PositionSnapshot], exit_event: Optional[Dict[str, object]]) -> Optional[str]:
    if position is None or not exit_event:
        return None
    trade_side = str(exit_event.get("side", "")).lower()
    if trade_side not in {"long", "short"}:
        return None
    if trade_side == "long" and position.amount <= 0:
        return None
    if trade_side == "short" and position.amount >= 0:
        return None
    reason = str(exit_event.get("reason", "")).strip()
    if not reason or reason == "end_of_test":
        return None
    return reason


class _EngineMultiplexKlineStream(threading.Thread):
    def __init__(
        self,
        base_interval: str,
        targets_by_symbol: Dict[str, Tuple[str, ...]],
        event_queue: Queue,
        stop_event: threading.Event,
        seed_history_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.base_interval = str(base_interval)
        self.targets_by_symbol = {
            str(symbol).upper(): tuple(sorted({str(interval) for interval in intervals}))
            for symbol, intervals in dict(targets_by_symbol).items()
            if intervals
        }
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.socket = None
        self.seed_history_by_symbol = {
            str(symbol).upper(): prepare_ohlcv(history.copy())
            for symbol, history in dict(seed_history_by_symbol or {}).items()
            if history is not None and not history.empty
        }
        self.aggregate_bars: Dict[Tuple[str, str], Optional[Dict[str, object]]] = {}

    def _emit_log(self, message: str) -> None:
        self.event_queue.put(("log", message))

    def _emit_bar(self, bar: Dict[str, object]) -> None:
        self.event_queue.put(("bar", str(bar["symbol"]).upper(), str(bar["interval"]), bar))

    def _emit_price(self, bar: Dict[str, object], interval: str) -> None:
        self.event_queue.put(
            ("price", str(bar["symbol"]).upper(), str(interval), pd.Timestamp(bar["time"]), float(bar["close"]))
        )

    def _initialize_aggregate_seed(self, symbol: str, interval: str) -> Optional[Dict[str, object]]:
        if interval != "2m":
            return None
        seed_history = self.seed_history_by_symbol.pop(str(symbol).upper(), None)
        return _seed_two_minute_aggregate(seed_history, str(symbol).upper(), interval)

    def _transform_bar_for_interval(
        self,
        symbol: str,
        interval: str,
        bar: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        if interval != "2m":
            return dict(bar) if bool(bar.get("closed")) else None
        key = (str(symbol).upper(), str(interval))
        aggregate_bar = self.aggregate_bars.get(key)
        seed_aggregate = None
        if aggregate_bar is None or pd.Timestamp(aggregate_bar["time"]) != pd.Timestamp(bar["time"]).floor("2min"):
            seed_aggregate = self._initialize_aggregate_seed(symbol, interval)
        next_aggregate, transformed = _transform_two_minute_bar(
            aggregate_bar,
            bar,
            seed_aggregate=seed_aggregate,
        )
        self.aggregate_bars[key] = next_aggregate
        return transformed if bool(transformed.get("closed")) else None

    def run(self) -> None:
        stream_names = [
            f"{symbol.lower()}@kline_{self.base_interval}"
            for symbol in sorted(self.targets_by_symbol)
        ]
        if not stream_names:
            return
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(stream_names)}"
        while not self.stop_event.is_set():
            try:
                self.socket = websocket.create_connection(stream_url, timeout=10)
                self.socket.settimeout(1.0)
                self._emit_log(
                    f"Trade engine multiplex stream connected: {self.base_interval} symbols={len(self.targets_by_symbol)}"
                )
                while not self.stop_event.is_set():
                    try:
                        raw = self.socket.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    data = payload.get("data", payload)
                    if data.get("e") != "kline":
                        continue
                    kline = data.get("k", {})
                    symbol = str(data.get("s", "") or kline.get("s", "") or "").upper()
                    target_intervals = self.targets_by_symbol.get(symbol)
                    if not symbol or not target_intervals:
                        continue
                    source_bar = {
                        "symbol": symbol,
                        "interval": self.base_interval,
                        "time": _ws_kline_timestamp(int(kline["t"])),
                        "open": float(kline["o"]),
                        "high": float(kline["h"]),
                        "low": float(kline["l"]),
                        "close": float(kline["c"]),
                        "volume": float(kline["v"]),
                        "quote_volume": float(kline.get("q", 0.0) or 0.0),
                        "closed": bool(kline.get("x", False)),
                    }
                    for target_interval in target_intervals:
                        target_bar = dict(source_bar)
                        target_bar["interval"] = target_interval
                        self._emit_price(target_bar, target_interval)
                        transformed = self._transform_bar_for_interval(symbol, target_interval, target_bar)
                        if transformed is not None:
                            self._emit_bar(transformed)
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self._emit_log(f"Trade engine multiplex stream reconnecting: {self.base_interval} ({exc})")
                time.sleep(STREAM_RECONNECT_DELAY_SECONDS)
            finally:
                if self.socket is not None:
                    try:
                        self.socket.close()
                    except Exception:
                        pass
                    self.socket = None


class _OrderExecutor(threading.Thread):
    def __init__(self, request_queue: PriorityQueue, result_queue: Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                _priority, _sequence, request = self.request_queue.get(timeout=0.5)
            except Empty:
                continue
            if request is None:
                continue
            try:
                client = BinanceFuturesClient(request.api_key, request.api_secret)
                if request.side is None:
                    if request.close_side is not None and request.close_quantity is not None and request.close_quantity > 0:
                        result = client.place_market_order(
                            request.symbol,
                            request.close_side,
                            request.close_quantity,
                            reduce_only=True,
                        )
                        no_open_position = False
                    else:
                        result = client.close_position(request.symbol)
                        no_open_position = result is None
                    message = (
                        f"{request.symbol} close skipped: no open position"
                        if no_open_position
                        else f"{request.symbol} close completed: orderId={result.get('orderId')}"
                    )
                    payload = _OrderExecutionResult(
                        symbol=request.symbol,
                        success=True,
                        message=message,
                        auto_close=request.auto_close,
                        auto_trade=request.auto_trade,
                        close_order=True,
                        interval=request.interval,
                        fraction=float(request.fraction or 0.0),
                        strategy_settings=request.strategy_settings,
                        no_open_position=no_open_position,
                    )
                else:
                    if request.margin is not None:
                        margin = float(request.margin)
                    else:
                        balance = client.get_balance_snapshot()
                        margin = balance.available_balance * float(request.fraction or 0.0)
                    if margin <= 0:
                        raise RuntimeError("order amount must be positive")
                    if request.apply_leverage:
                        client.set_leverage(request.symbol, request.leverage)
                    quantity = client.build_order_quantity(
                        request.symbol,
                        margin,
                        request.leverage,
                        price=request.price,
                    )
                    result = client.place_market_order(request.symbol, request.side, quantity)
                    payload = _OrderExecutionResult(
                        symbol=request.symbol,
                        success=True,
                        message=(
                            f"Order completed: {request.symbol} {request.side} qty={quantity} "
                            f"orderId={result.get('orderId')}"
                        ),
                        auto_close=request.auto_close,
                        auto_trade=request.auto_trade,
                        close_order=False,
                        interval=request.interval,
                        fraction=float(request.fraction or 0.0),
                        strategy_settings=request.strategy_settings,
                        no_open_position=False,
                    )
                self.result_queue.put(payload)
            except Exception:
                payload = _OrderExecutionResult(
                    symbol=request.symbol,
                    success=False,
                    message=self._format_order_failure_message(request, traceback.format_exc()),
                    auto_close=request.auto_close,
                    auto_trade=request.auto_trade,
                    close_order=request.side is None,
                    interval=request.interval,
                    fraction=float(request.fraction or 0.0),
                    strategy_settings=request.strategy_settings,
                    no_open_position=False,
                )
                try:
                    self.result_queue.put(payload)
                except Exception:
                    # Queue itself failed — log only. Pending timeout (#1) handles unlock.
                    traceback.print_exc()


    @staticmethod
    def _format_order_failure_message(request: _OrderRequest, raw_error: str) -> str:
        message = str(raw_error or "").strip()
        compact = " ".join(line.strip() for line in message.splitlines() if line.strip())
        if not compact:
            return f"{request.symbol} order failed"
        lowered = compact.lower()
        if (
            "insufficient margin" in lowered
            or "margin is insufficient" in lowered
            or "balance is insufficient" in lowered
            or "insufficient balance" in lowered
            or "[-2019]" in lowered
            or "code=-2019" in lowered
        ):
            return f"{request.symbol} order failed: insufficient margin ({compact})"
        if request.side is None:
            return f"{request.symbol} close failed: {compact}"
        return f"{request.symbol} order failed: {compact}"


class TradeEngineController:
    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.command_queue: Optional[mp.Queue] = None
        self.event_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.command_queue = self.ctx.Queue()
        self.event_queue = self.ctx.Queue()
        self.process = self.ctx.Process(
            target=run_trade_engine_process,
            args=(self.command_queue, self.event_queue),
            daemon=True,
        )
        self.process.start()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def send(self, command: object) -> None:
        if self.command_queue is None or not self.is_alive():
            raise RuntimeError("trade engine is not running")
        self.command_queue.put(command)

    def drain_events(self, limit: int = 100) -> List[object]:
        if self.event_queue is None:
            return []
        events: List[object] = []
        for _ in range(max(limit, 0)):
            try:
                events.append(self.event_queue.get_nowait())
            except Empty:
                break
        return events

    def stop(self, timeout: float = 5.0) -> None:
        if self.process is None:
            return
        if self.command_queue is not None and self.process.is_alive():
            try:
                self.command_queue.put(EngineStopCommand())
            except Exception:
                pass
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
        self.process = None
        self.command_queue = None
        self.event_queue = None


class _TradeEngine:
    def __init__(self, command_queue: mp.Queue, event_queue: mp.Queue) -> None:
        self.command_queue = command_queue
        self.event_queue = event_queue
        self.internal_queue: Queue = Queue()
        self.order_request_queue: PriorityQueue = PriorityQueue()
        self.order_result_queue: Queue = Queue()
        self.stop_event = threading.Event()
        self.order_executor = _OrderExecutor(self.order_request_queue, self.order_result_queue, self.stop_event)
        self.client: Optional[BinanceFuturesClient] = None
        self.api_key = ""
        self.api_secret = ""
        self.leverage = 1
        self.fee_rate = 0.0005
        self.history_days = DEFAULT_HISTORY_DAYS
        self.default_interval = "1m"
        self.default_strategy_settings = StrategySettings()
        self.optimization_rank_mode = "score"
        self.auto_trade_enabled = False
        self.auto_trade_use_favorable_price = True
        self.auto_close_enabled_symbols: set[str] = set()
        self.position_intervals: Dict[str, str] = {}
        self.watchlist: Dict[Tuple[str, str], EngineWatchlistItem] = {}
        self.symbol_states: Dict[Tuple[str, str], _EngineSymbolState] = {}
        self.streams: Dict[str, Tuple[_EngineMultiplexKlineStream, threading.Event, Tuple[Tuple[str, Tuple[str, ...]], ...]]] = {}
        self.open_positions: Dict[str, PositionSnapshot] = {}
        self.filled_fraction_by_symbol: Dict[str, float] = {}
        self.auto_trade_cursor_entry_time: Dict[str, pd.Timestamp] = {}
        self.position_strategy_by_symbol: Dict[str, StrategySettings] = {}
        self.applied_leverage_by_symbol: Dict[str, int] = {}
        self.pending_order_symbols: Dict[str, float] = {}
        self.auto_close_last_trigger_time: Dict[str, pd.Timestamp] = {}
        self.auto_close_last_attempt_at: Dict[str, float] = {}
        self.latest_stream_price_by_symbol: Dict[str, float] = {}
        self.last_stream_price_eval_at: Dict[Tuple[str, str], float] = {}
        self.last_stale_symbol_eval_at: Dict[Tuple[str, str], float] = {}
        self.base_stream_last_payload_at: Dict[str, float] = {}
        self.base_stream_started_at: Dict[str, float] = {}
        self.base_stream_last_restart_at: Dict[str, float] = {}
        self.stream_health_degraded = False
        self.last_auto_trade_check_at = 0.0
        self.last_position_refresh_at = 0.0
        self.order_sequence = 0

    def emit(self, event: object) -> None:
        try:
            self.event_queue.put(event)
        except Exception:
            pass

    def log(self, message: str) -> None:
        self.emit(EngineLogEvent(message))

    def run(self) -> None:
        self.order_executor.start()
        self.emit(EngineHealthEvent("ready", "trade engine started"))
        while not self.stop_event.is_set():
            self._drain_commands()
            self._drain_order_results()
            self._drain_internal_events()
            now = time.time()
            self._update_stream_health(now=now)
            if self.api_key and self.api_secret and now - self.last_position_refresh_at >= POSITION_REFRESH_INTERVAL_SECONDS:
                self._refresh_positions()
            if self.api_key and self.api_secret and now - self.last_auto_trade_check_at >= AUTO_TRADE_RECHECK_INTERVAL_SECONDS:
                self.last_auto_trade_check_at = now
                self._evaluate_auto_trade()
                self._retry_auto_close_orders()
            self._load_one_pending_state()
            self._expire_stale_pending_orders()
            time.sleep(COMMAND_POLL_TIMEOUT_SECONDS)
        self._stop_all_streams()
        self.order_request_queue.put((99, self.order_sequence, None))

    def stop(self) -> None:
        self.stop_event.set()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except Empty:
                break
            if isinstance(command, EngineStopCommand):
                self.stop()
                return
            if isinstance(command, EngineSyncCommand):
                self._apply_sync(command)
                continue
            if isinstance(command, EngineOpenOrderCommand):
                self._handle_manual_open(command)
                continue
            if isinstance(command, EngineCloseOrderCommand):
                self._handle_manual_close(command)

    def _drain_internal_events(self) -> None:
        while True:
            try:
                payload = self.internal_queue.get_nowait()
            except Empty:
                break
            event_type = payload[0]
            if event_type == "log":
                self.log(str(payload[1]))
            elif event_type == "bar":
                _event_type, symbol, interval, bar = payload
                self._handle_closed_bar(symbol, interval, bar)
            elif event_type == "price":
                _event_type, symbol, interval, bar_time, price = payload
                self._handle_price_update(symbol, interval, pd.Timestamp(bar_time), float(price))

    def _drain_order_results(self) -> None:
        while True:
            try:
                result = self.order_result_queue.get_nowait()
            except Empty:
                break
            if not isinstance(result, _OrderExecutionResult):
                continue
            self.pending_order_symbols.pop(result.symbol, None)
            if result.close_order and result.success:
                self.open_positions.pop(result.symbol, None)
                self.filled_fraction_by_symbol.pop(result.symbol, None)
                self.auto_trade_cursor_entry_time.pop(result.symbol, None)
                self.applied_leverage_by_symbol.pop(result.symbol, None)
            if result.no_open_position:
                self.open_positions.pop(result.symbol, None)
                self.filled_fraction_by_symbol.pop(result.symbol, None)
                self.auto_trade_cursor_entry_time.pop(result.symbol, None)
                self.applied_leverage_by_symbol.pop(result.symbol, None)
            if result.auto_close and result.success:
                self.auto_close_last_trigger_time.pop(result.symbol, None)
                self.auto_close_last_attempt_at.pop(result.symbol, None)
            if result.success:
                if result.interval in APP_INTERVAL_OPTIONS:
                    self.position_intervals[result.symbol] = str(result.interval)
                if result.strategy_settings is not None:
                    self.position_strategy_by_symbol[result.symbol] = result.strategy_settings
                if not result.close_order:
                    self.applied_leverage_by_symbol[result.symbol] = int(self.leverage)
                if result.auto_trade:
                    current_fraction = float(self.filled_fraction_by_symbol.get(result.symbol, 0.0))
                    self.filled_fraction_by_symbol[result.symbol] = min(
                        signal_fraction_for_zone(3),
                        current_fraction + float(result.fraction or 0.0),
                    )
            self._refresh_positions(force=True)
            if result.success:
                self.emit(
                    EngineOrderCompletedEvent(
                        symbol=result.symbol,
                        message=result.message,
                        auto_close=result.auto_close,
                        auto_trade=result.auto_trade,
                        interval=result.interval,
                        fraction=result.fraction,
                        strategy_settings=result.strategy_settings,
                    )
                )
            else:
                self.emit(
                    EngineOrderFailedEvent(
                        symbol=result.symbol,
                        message=result.message,
                        auto_close=result.auto_close,
                        auto_trade=result.auto_trade,
                        interval=result.interval,
                        fraction=result.fraction,
                    )
                )

    def _expire_stale_pending_orders(self) -> None:
        now = time.time()
        for symbol in list(self.pending_order_symbols):
            if now - self.pending_order_symbols[symbol] > PENDING_ORDER_TIMEOUT_SECONDS:
                self.pending_order_symbols.pop(symbol, None)
                self.log(f"{symbol} pending order timed out after {PENDING_ORDER_TIMEOUT_SECONDS}s")

    def _apply_sync(self, command: EngineSyncCommand) -> None:
        creds_changed = (command.api_key != self.api_key) or (command.api_secret != self.api_secret)
        self.api_key = command.api_key.strip()
        self.api_secret = command.api_secret.strip()
        self.leverage = max(1, int(command.leverage))
        self.fee_rate = float(command.fee_rate)
        self.history_days = max(1, int(command.history_days))
        self.default_interval = command.default_interval if command.default_interval in APP_INTERVAL_OPTIONS else "1m"
        self.default_strategy_settings = command.default_strategy_settings
        self.optimization_rank_mode = command.optimization_rank_mode if command.optimization_rank_mode in {"score", "return"} else "score"
        self.auto_trade_enabled = bool(command.auto_trade_enabled)
        self.auto_trade_use_favorable_price = bool(command.auto_trade_use_favorable_price)
        self.auto_close_enabled_symbols = {str(symbol) for symbol in command.auto_close_enabled_symbols}
        self.position_intervals = {
            str(symbol): str(interval)
            for symbol, interval in dict(command.position_intervals).items()
            if str(interval) in APP_INTERVAL_OPTIONS
        }
        self.position_strategy_by_symbol = {
            str(symbol): settings
            for symbol, settings in dict(command.position_strategy_settings).items()
            if isinstance(settings, StrategySettings)
        }
        self.filled_fraction_by_symbol = {
            str(symbol): max(0.0, min(signal_fraction_for_zone(3), float(fraction)))
            for symbol, fraction in dict(command.position_filled_fractions).items()
            if str(symbol or "").strip()
        }
        self.auto_trade_cursor_entry_time = {
            str(symbol): pd.Timestamp(entry_time).tz_localize(None)
            for symbol, entry_time in dict(command.position_cursor_entry_times).items()
            if str(symbol or "").strip()
        }
        self.watchlist = {
            (item.symbol, item.interval): item
            for item in command.watchlist
            if item.interval in APP_INTERVAL_OPTIONS
        }
        if self.api_key and self.api_secret:
            if creds_changed or self.client is None:
                self.client = BinanceFuturesClient(self.api_key, self.api_secret)
            self._refresh_positions(force=True)
        else:
            self.client = None
            self.open_positions.clear()
            self.filled_fraction_by_symbol.clear()
            self.auto_trade_cursor_entry_time.clear()
        self._ensure_active_states()

    def _refresh_positions(self, force: bool = False) -> None:
        if self.client is None:
            self.open_positions = {}
            self.last_position_refresh_at = time.time()
            return
        if not force and time.time() - self.last_position_refresh_at < POSITION_REFRESH_INTERVAL_SECONDS:
            return
        positions = self.client.get_open_positions()
        # Guard against empty API response when we expect positions (transient API failure)
        if not positions and self.open_positions:
            self.last_position_refresh_at = time.time()
            return
        self.open_positions = {position.symbol: position for position in positions}
        open_symbols = set(self.open_positions)
        self.filled_fraction_by_symbol = {
            symbol: fraction
            for symbol, fraction in self.filled_fraction_by_symbol.items()
            if symbol in open_symbols
        }
        self.auto_trade_cursor_entry_time = {
            symbol: t
            for symbol, t in self.auto_trade_cursor_entry_time.items()
            if symbol in open_symbols
        }
        for symbol in list(self.position_strategy_by_symbol):
            if symbol not in open_symbols and symbol not in {item.symbol for item in self.watchlist.values()}:
                self.position_strategy_by_symbol.pop(symbol, None)
        self.last_position_refresh_at = time.time()
        self._ensure_active_states()

    def _active_symbol_keys(self) -> set[Tuple[str, str]]:
        keys = set(self.watchlist.keys())
        for symbol in self.open_positions:
            interval = self.position_intervals.get(symbol)
            if interval not in APP_INTERVAL_OPTIONS:
                interval = next((key_interval for key_symbol, key_interval in self.watchlist if key_symbol == symbol), self.default_interval)
            keys.add((symbol, interval))
        return keys

    def _settings_for_key(self, symbol: str, interval: str) -> StrategySettings:
        locked_settings = self.position_strategy_by_symbol.get(symbol)
        if symbol in self.open_positions and locked_settings is not None:
            return locked_settings
        watchlist_item = self.watchlist.get((symbol, interval))
        if watchlist_item is not None:
            return watchlist_item.strategy_settings
        if locked_settings is not None:
            return locked_settings
        return self.default_strategy_settings

    def _priority_class_for_key(self, key: Tuple[str, str], state: Optional[_EngineSymbolState] = None) -> str:
        symbol, _interval = key
        if symbol in self.open_positions:
            return "position"
        if (
            symbol in self.pending_order_symbols
            or symbol in self.auto_close_enabled_symbols
            or symbol in self.auto_close_last_trigger_time
            or (state is not None and state.pending_closed_bar is not None)
            or (state is not None and state.stale_since > 0)
        ):
            return "triggered"
        return "watchlist"

    def _mark_state_for_reload(
        self,
        state: _EngineSymbolState,
        *,
        force_history_refetch: bool = False,
        now: Optional[float] = None,
    ) -> None:
        state.loading = True
        state.needs_reload = True
        state.force_history_refetch = state.force_history_refetch or force_history_refetch
        state.reload_requested_at = float(now if now is not None else time.time())

    def _pending_state_sort_key(self, key: Tuple[str, str]) -> Tuple[int, float, str, str]:
        state = self.symbol_states[key]
        priority_rank = {
            "position": 0,
            "triggered": 1,
            "watchlist": 2,
        }.get(self._priority_class_for_key(key, state), 2)
        requested_at = float(state.reload_requested_at or 0.0)
        if requested_at <= 0.0:
            requested_at = time.time()
        return (priority_rank, requested_at, key[0], key[1])

    def _stream_payload_age_seconds(self, state: _EngineSymbolState, now: float) -> Optional[float]:
        base_interval = resolve_base_interval(state.interval)
        if base_interval not in self.streams:
            return None
        reference_time = max(
            float(state.last_stream_payload_at or 0.0),
            float(self.base_stream_last_payload_at.get(base_interval, 0.0) or 0.0),
            float(self.base_stream_started_at.get(base_interval, 0.0) or 0.0),
        )
        if reference_time <= 0.0:
            return None
        return max(0.0, now - reference_time)

    def _stream_price_is_fresh(self, symbol: str, interval: str, now: Optional[float] = None) -> bool:
        state = self.symbol_states.get((str(symbol).upper(), str(interval)))
        if state is None:
            return False
        reference_now = float(now if now is not None else time.time())
        if state.stale_since > 0:
            return False
        last_price_update_at = float(state.last_price_update_at or 0.0)
        if last_price_update_at <= 0.0:
            return False
        return reference_now - last_price_update_at < STREAM_STALE_SECONDS

    def _update_stream_health(self, *, now: Optional[float] = None) -> None:
        reference_now = float(now if now is not None else time.time())
        stale_position_symbols: set[str] = set()
        stale_watchlist_symbols: set[str] = set()
        bases_to_restart: set[str] = set()
        should_force_position_refresh = False
        for key in sorted(self._active_symbol_keys()):
            state = self.symbol_states.get(key)
            if state is None:
                continue
            state.priority_class = self._priority_class_for_key(key, state)
            payload_age = self._stream_payload_age_seconds(state, reference_now)
            if payload_age is None:
                continue
            symbol = key[0]
            interval = key[1]
            base_interval = resolve_base_interval(interval)
            is_stale = payload_age >= STREAM_STALE_SECONDS
            if is_stale:
                if state.stale_since <= 0.0:
                    state.stale_since = reference_now
                if state.reload_requested_at <= 0.0:
                    state.reload_requested_at = reference_now
                if symbol in self.open_positions:
                    stale_position_symbols.add(symbol)
                    should_force_position_refresh = True
                else:
                    stale_watchlist_symbols.add(symbol)
                if reference_now - float(self.base_stream_last_restart_at.get(base_interval, 0.0) or 0.0) >= STREAM_STALE_SECONDS:
                    bases_to_restart.add(base_interval)
                if (
                    reference_now - float(state.stale_since) >= STREAM_FORCE_RELOAD_SECONDS
                    and not state.loading
                ):
                    self._mark_state_for_reload(state, force_history_refetch=True, now=reference_now)
                if (
                    self.auto_trade_enabled
                    and reference_now - float(self.last_stale_symbol_eval_at.get(key, 0.0) or 0.0)
                    >= STALE_SYMBOL_EVAL_INTERVAL_SECONDS
                ):
                    self.last_stale_symbol_eval_at[key] = reference_now
                    self._evaluate_auto_trade(trigger_symbol=symbol, trigger_interval=interval)
            elif state.stale_since > 0.0:
                state.stale_since = 0.0
                self.last_stale_symbol_eval_at.pop(key, None)
        if should_force_position_refresh and self.api_key and self.api_secret:
            self._refresh_positions(force=True)
        for base_interval in sorted(bases_to_restart):
            self.base_stream_last_restart_at[base_interval] = reference_now
            self._stop_stream(base_interval)
        if bases_to_restart:
            self._refresh_streams()
        has_stale = bool(stale_position_symbols or stale_watchlist_symbols)
        if has_stale and not self.stream_health_degraded:
            self.stream_health_degraded = True
            self.emit(
                EngineHealthEvent(
                    "degraded",
                    (
                        "Trade engine degraded: "
                        f"stale position streams={len(stale_position_symbols)} "
                        f"watchlist streams={len(stale_watchlist_symbols)}"
                    ),
                )
            )
        elif not has_stale and self.stream_health_degraded:
            self.stream_health_degraded = False
            self.emit(EngineHealthEvent("recovered", "Trade engine stream health recovered"))

    def _ensure_active_states(self) -> None:
        now = time.time()
        active_keys = self._active_symbol_keys()
        for key in list(self.symbol_states):
            if key not in active_keys and key[0] not in self.open_positions:
                self.symbol_states.pop(key, None)
        for key in active_keys:
            symbol, interval = key
            state = self.symbol_states.get(key)
            settings = self._settings_for_key(symbol, interval)
            if state is None:
                watch = self.watchlist.get(key)
                self.symbol_states[key] = _EngineSymbolState(
                    symbol=symbol,
                    interval=interval,
                    strategy_settings=settings,
                    loaded_strategy_settings=settings,
                    score=float(watch.score) if watch is not None else 0.0,
                    return_pct=float(watch.return_pct) if watch is not None else 0.0,
                    loading=False,
                    priority_class=self._priority_class_for_key(key),
                )
            else:
                if state.backtest is not None and state.loaded_strategy_settings is None:
                    state.loaded_strategy_settings = state.strategy_settings
                if state.strategy_settings != settings:
                    state.strategy_settings = settings
                    self._mark_state_for_reload(state, now=now)
                else:
                    state.strategy_settings = settings
                watch = self.watchlist.get(key)
                state.score = float(watch.score) if watch is not None else state.score
                state.return_pct = float(watch.return_pct) if watch is not None else state.return_pct
                state.priority_class = self._priority_class_for_key(key, state)
        for key in active_keys:
            self._ensure_state_loaded(key)
        self._refresh_streams()

    def _ensure_state_loaded(self, key: Tuple[str, str]) -> None:
        state = self.symbol_states[key]
        if state.backtest is not None and state.loaded_strategy_settings is None:
            state.loaded_strategy_settings = state.strategy_settings
        if state.loading:
            return
        if state.history is None or state.history.empty:
            self._mark_state_for_reload(state)
            return
        if state.backtest is None:
            self._mark_state_for_reload(state)
            return
        if state.needs_reload or state.force_history_refetch:
            self._mark_state_for_reload(state, force_history_refetch=state.force_history_refetch)

    def _load_one_pending_state(self) -> None:
        pending_keys = [key for key, state in self.symbol_states.items() if state.loading]
        if not pending_keys:
            return
        key = sorted(pending_keys, key=self._pending_state_sort_key)[0]
        state = self.symbol_states[key]
        try:
            if self.client is None:
                return
            history = None if state.force_history_refetch else state.history
            had_existing_history = history is not None and not history.empty
            if history is None or history.empty:
                start_time = _history_fetch_start_time_ms(self.history_days, state.interval, state.strategy_settings)
                history = self.client.historical_ohlcv(state.symbol, state.interval, start_time=start_time)
            history = prepare_ohlcv(history)
            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.history_days), unit="ms")
            backtest = run_backtest(
                history,
                settings=state.strategy_settings,
                fee_rate=self.fee_rate,
                backtest_start_time=backtest_start_time,
            )
            trigger_bar_time: Optional[pd.Timestamp] = None
            pending_closed_bar = dict(state.pending_closed_bar) if state.pending_closed_bar is not None else None
            if pending_closed_bar is not None:
                history = _merge_live_bar(history, pending_closed_bar)
                if _history_can_resume_backtest(backtest, history):
                    backtest = resume_backtest(
                        history,
                        previous_result=backtest,
                        settings=state.strategy_settings,
                        fee_rate=self.fee_rate,
                        backtest_start_time=backtest_start_time,
                    )
                else:
                    backtest = run_backtest(
                        history,
                        settings=state.strategy_settings,
                        fee_rate=self.fee_rate,
                        backtest_start_time=backtest_start_time,
                    )
                trigger_bar_time = pd.Timestamp(pending_closed_bar["time"]).tz_localize(None)
                state.pending_closed_bar = None
            elif not had_existing_history:
                trigger_bar_time = _fresh_initial_trigger_bar_time(backtest, state.interval)
            state.history = history
            state.backtest = backtest
            state.loaded_strategy_settings = state.strategy_settings
            state.needs_reload = False
            state.force_history_refetch = False
            state.reload_requested_at = 0.0
            self._start_stream(key)
            self._emit_signal_event(state)
            self._evaluate_backtest_auto_close(state)
            if trigger_bar_time is not None:
                self._evaluate_auto_trade(
                    trigger_symbol=state.symbol,
                    trigger_interval=state.interval,
                    trigger_bar_time=trigger_bar_time,
                )
            else:
                self._evaluate_auto_trade()
        except Exception:
            self.log(traceback.format_exc())
        finally:
            state.loading = False

    def _stream_targets_signature(
        self,
        targets_by_symbol: Dict[str, Tuple[str, ...]],
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        return tuple(
            sorted(
                (
                    str(symbol).upper(),
                    tuple(sorted({str(interval) for interval in intervals})),
                )
                for symbol, intervals in dict(targets_by_symbol).items()
                if intervals
            )
        )

    def _desired_stream_targets_by_base_interval(self) -> Dict[str, Dict[str, Tuple[str, ...]]]:
        grouped: Dict[str, Dict[str, set[str]]] = {}
        for symbol, interval in self._active_symbol_keys():
            base_interval = resolve_base_interval(interval)
            symbol_map = grouped.setdefault(base_interval, {})
            symbol_map.setdefault(str(symbol).upper(), set()).add(str(interval))
        return {
            base_interval: {
                symbol: tuple(sorted(intervals))
                for symbol, intervals in symbol_map.items()
            }
            for base_interval, symbol_map in grouped.items()
        }

    def _refresh_streams(self) -> None:
        desired = self._desired_stream_targets_by_base_interval()
        for base_interval in list(self.streams):
            if base_interval not in desired:
                self._stop_stream(base_interval)
                continue
            _stream, _stop_event, signature = self.streams[base_interval]
            desired_signature = self._stream_targets_signature(desired[base_interval])
            if signature != desired_signature:
                self._stop_stream(base_interval)
        for base_interval, targets_by_symbol in desired.items():
            if base_interval in self.streams or self.client is None:
                continue
            seed_history_by_symbol: Dict[str, pd.DataFrame] = {}
            if base_interval == "1m":
                for symbol, intervals in targets_by_symbol.items():
                    if "2m" not in intervals:
                        continue
                    try:
                        history = self.client.historical_ohlcv_recent(symbol, "1m", bars=2)
                    except Exception:
                        history = None
                    if history is not None and not history.empty:
                        seed_history_by_symbol[symbol] = history
            stop_event = threading.Event()
            signature = self._stream_targets_signature(targets_by_symbol)
            stream = _EngineMultiplexKlineStream(
                base_interval,
                targets_by_symbol,
                self.internal_queue,
                stop_event,
                seed_history_by_symbol=seed_history_by_symbol,
            )
            self.streams[base_interval] = (stream, stop_event, signature)
            started_at = time.time()
            self.base_stream_started_at[base_interval] = started_at
            self.base_stream_last_payload_at[base_interval] = started_at
            stream.start()

    def _start_stream(self, key: Tuple[str, str]) -> None:
        del key
        self._refresh_streams()

    def _stop_stream(self, key: object) -> None:
        if isinstance(key, tuple):
            base_interval = resolve_base_interval(str(key[1]))
        else:
            base_interval = str(key)
        stream_entry = self.streams.pop(base_interval, None)
        if stream_entry is None:
            return
        stream, stop_event, _signature = stream_entry
        stop_event.set()
        stream.join(timeout=1.5)
        self.base_stream_last_payload_at.pop(base_interval, None)
        self.base_stream_started_at.pop(base_interval, None)

    def _stop_all_streams(self) -> None:
        for key in list(self.streams):
            self._stop_stream(key)

    def _handle_closed_bar(self, symbol: str, interval: str, bar: Dict[str, object]) -> None:
        key = (symbol, interval)
        state = self.symbol_states.get(key)
        if state is None:
            state = _EngineSymbolState(
                symbol=symbol,
                interval=interval,
                strategy_settings=self._settings_for_key(symbol, interval),
                priority_class=self._priority_class_for_key(key),
            )
            self.symbol_states[key] = state
        state.priority_class = self._priority_class_for_key(key, state)
        history = state.history
        if state.loading or history is None or history.empty:
            state.pending_closed_bar = dict(bar)
            if not state.loading:
                self._mark_state_for_reload(state)
            return
        state.history = _merge_live_bar(history, bar)
        backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.history_days), unit="ms")
        if _history_can_resume_backtest(state.backtest, state.history):
            state.backtest = resume_backtest(
                state.history,
                previous_result=state.backtest,
                settings=state.strategy_settings,
                fee_rate=self.fee_rate,
                backtest_start_time=backtest_start_time,
            )
        else:
            state.backtest = run_backtest(
                state.history,
                settings=state.strategy_settings,
                fee_rate=self.fee_rate,
                backtest_start_time=backtest_start_time,
            )
        state.loaded_strategy_settings = state.strategy_settings
        state.needs_reload = False
        state.force_history_refetch = False
        state.reload_requested_at = 0.0
        self._emit_signal_event(state)
        self._evaluate_backtest_auto_close(state)
        self._evaluate_auto_trade(
            trigger_symbol=symbol,
            trigger_interval=interval,
            trigger_bar_time=pd.Timestamp(bar["time"]),
        )

    def _handle_price_update(
        self,
        symbol: str,
        interval: str,
        bar_time: pd.Timestamp,
        price: float,
    ) -> None:
        if price <= 0:
            return
        symbol = str(symbol).upper()
        interval = str(interval)
        now = time.time()
        self.latest_stream_price_by_symbol[symbol] = float(price)
        self.base_stream_last_payload_at[resolve_base_interval(interval)] = now
        state = self.symbol_states.get((symbol, interval))
        if state is not None:
            state.last_stream_payload_at = now
            state.last_price_update_at = now
            state.stale_since = 0.0
            state.priority_class = self._priority_class_for_key((symbol, interval), state)
            self.last_stale_symbol_eval_at.pop((symbol, interval), None)
        if not self.auto_trade_enabled:
            return
        key = (symbol, interval)
        last_eval = float(self.last_stream_price_eval_at.get(key, 0.0))
        if now - last_eval < STREAM_PRICE_EVAL_MIN_INTERVAL_SECONDS:
            return
        self.last_stream_price_eval_at[key] = now
        self._evaluate_auto_trade(trigger_symbol=symbol, trigger_interval=interval)

    def _emit_signal_event(self, state: _EngineSymbolState) -> None:
        if state.backtest is None:
            return
        signal = _auto_trade_signal_from_backtest(state.backtest)
        exit_event = _latest_backtest_exit_event(state.backtest)
        preview_side = ""
        preview_zone = 0
        if state.history is not None and not state.history.empty and state.backtest.cursor is not None:
            latest_state, _ = evaluate_latest_state(
                state.history,
                state.backtest.settings,
                cursor=state.backtest.cursor.indicator_cursor,
            )
            preview_signal = _preview_entry_signal(state.backtest.cursor, latest_state, state.backtest.settings)
            if preview_signal is not None:
                preview_side, preview_zone = preview_signal
        self.emit(
            EngineSignalEvent(
                symbol=state.symbol,
                interval=state.interval,
                entry_side="" if signal is None else str(signal["side"]),
                entry_zone=0 if signal is None else int(signal["zone"]),
                preview_entry_side=preview_side,
                preview_entry_zone=preview_zone,
                exit_reason="" if exit_event is None else str(exit_event.get("reason", "")),
                bar_time=None if exit_event is None else pd.Timestamp(exit_event.get("bar_time")),
            )
        )

    def _evaluate_backtest_auto_close(self, state: _EngineSymbolState) -> None:
        if state.backtest is None:
            return
        position = self.open_positions.get(state.symbol)
        exit_event = _confirmed_exit_event_from_position_backtest(position, state.backtest)
        if exit_event is None:
            exit_event = _latest_backtest_exit_event(state.backtest)
        if exit_event is None:
            return
        self._maybe_trigger_auto_close(state.symbol, exit_event)

    def _managed_auto_close_symbols(self) -> set[str]:
        managed = set(self.auto_close_enabled_symbols)
        if self.auto_trade_enabled:
            managed.update(self.open_positions)
        return managed

    def _auto_trade_items(self) -> List[EngineWatchlistItem]:
        items = list(self.watchlist.values())
        existing_keys = {(item.symbol, item.interval) for item in items}
        for symbol in sorted(self.open_positions):
            interval = self.position_intervals.get(symbol)
            if interval not in APP_INTERVAL_OPTIONS:
                interval = next((key_interval for key_symbol, key_interval in self.watchlist if key_symbol == symbol), self.default_interval)
            key = (symbol, interval)
            if key in existing_keys:
                continue
            state = self.symbol_states.get(key)
            items.append(
                EngineWatchlistItem(
                    symbol=symbol,
                    interval=interval,
                    score=0.0 if state is None else float(state.score),
                    return_pct=0.0 if state is None else float(state.return_pct),
                    strategy_settings=self._settings_for_key(symbol, interval),
                )
            )
        return items

    def _state_for_symbol(self, symbol: str) -> Optional[_EngineSymbolState]:
        interval = self.position_intervals.get(symbol)
        if interval in APP_INTERVAL_OPTIONS:
            return self.symbol_states.get((symbol, interval))
        for key, state in self.symbol_states.items():
            if key[0] == symbol:
                return state
        return None

    def _maybe_trigger_auto_close(self, symbol: str, exit_event: Optional[Dict[str, object]]) -> None:
        position = self.open_positions.get(symbol)
        reason = _auto_close_reason(position, exit_event)
        if reason is None or position is None:
            return
        if symbol in self.pending_order_symbols:
            return
        bar_time = pd.Timestamp(exit_event["bar_time"]) if exit_event and exit_event.get("bar_time") is not None else None
        last_bar_time = self.auto_close_last_trigger_time.get(symbol)
        last_attempt = self.auto_close_last_attempt_at.get(symbol, 0.0)
        if bar_time is not None and last_bar_time == bar_time and time.time() - last_attempt < AUTO_CLOSE_RETRY_INTERVAL_SECONDS:
            return
        self.auto_close_last_attempt_at[symbol] = time.time()
        if bar_time is not None:
            self.auto_close_last_trigger_time[symbol] = bar_time
        self._enqueue_close_order(symbol, reason, auto_close=True)

    def _trigger_reentry_auto_close(self, symbol: str, position_side: str, backtest: BacktestResult) -> None:
        """Trigger auto-close when the backtest closed and re-entered (close→reopen).

        Scans backtest trades for the most recent non-end_of_test exit matching
        *position_side* and uses its reason to trigger auto-close for the real
        exchange position.
        """
        if symbol in self.pending_order_symbols:
            return
        for trade in reversed(backtest.trades):
            if str(trade.reason) == "end_of_test":
                continue
            if str(trade.side).lower() == position_side:
                exit_event = {
                    "side": position_side,
                    "reason": str(trade.reason),
                    "bar_time": pd.Timestamp(trade.exit_time),
                }
                self._maybe_trigger_auto_close(symbol, exit_event)
                return

    def _retry_auto_close_orders(self) -> None:
        for symbol in sorted(self._managed_auto_close_symbols()):
            if symbol in self.pending_order_symbols:
                continue
            state = self._state_for_symbol(symbol)
            if state is None or state.backtest is None:
                continue
            position = self.open_positions.get(symbol)
            exit_event = _confirmed_exit_event_from_position_backtest(position, state.backtest)
            if exit_event is None:
                exit_event = _latest_backtest_exit_event(state.backtest)
            if exit_event is None:
                continue
            self._maybe_trigger_auto_close(symbol, exit_event)

    def _latest_auto_trade_backtest(self, item: EngineWatchlistItem) -> Optional[BacktestResult]:
        state = self.symbol_states.get((item.symbol, item.interval))
        if state is None:
            return None
        if state.needs_reload and item.symbol not in self.open_positions:
            return None
        return state.backtest

    def _evaluate_auto_trade(
        self,
        *,
        trigger_symbol: Optional[str] = None,
        trigger_interval: Optional[str] = None,
        trigger_bar_time: Optional[pd.Timestamp] = None,
    ) -> None:
        if not self.auto_trade_enabled or self.client is None:
            return
        trigger_symbol = str(trigger_symbol or "").upper()
        trigger_interval = str(trigger_interval or "")
        normalized_trigger_time = (
            pd.Timestamp(trigger_bar_time).tz_localize(None) if trigger_bar_time is not None else None
        )
        eligible_items = self._auto_trade_items()
        if not eligible_items:
            return
        ticker_map: Optional[Dict[str, Dict[str, object]]] = None

        def current_price_for(symbol: str, interval: str) -> Optional[float]:
            live_price = float(self.latest_stream_price_by_symbol.get(symbol, 0.0) or 0.0)
            if live_price > 0 and self._stream_price_is_fresh(symbol, interval):
                return live_price
            nonlocal ticker_map
            if ticker_map is None:
                try:
                    ticker_map = self.client.ticker_24h()
                except Exception as exc:
                    self.log(f"Trade engine auto-trade ticker lookup failed: {exc}")
                    ticker_map = {}
            ticker = ticker_map.get(symbol) if ticker_map is not None else None
            if not ticker:
                return None
            current_price = float(ticker.get("lastPrice", 0.0) or 0.0)
            return current_price if current_price > 0 else None

        open_positions_by_symbol = dict(self.open_positions)
        candidates: List[Dict[str, object]] = []
        for item in eligible_items:
            if item.symbol in self.pending_order_symbols:
                continue
            if trigger_symbol and item.symbol != trigger_symbol:
                continue
            if trigger_interval and item.interval != trigger_interval:
                continue
            open_position = open_positions_by_symbol.get(item.symbol)
            if open_positions_by_symbol and open_position is None:
                continue
            if open_position is not None:
                remembered_interval = self.position_intervals.get(item.symbol)
                if remembered_interval in APP_INTERVAL_OPTIONS and remembered_interval != item.interval:
                    continue
            latest_backtest = self._latest_auto_trade_backtest(item)
            if latest_backtest is None:
                continue
            current_price = current_price_for(item.symbol, item.interval)
            evaluation = evaluate_auto_trade_candidate(
                symbol=item.symbol,
                interval=item.interval,
                score=float(item.score),
                strategy_settings=item.strategy_settings,
                latest_backtest=latest_backtest,
                current_price=current_price,
                open_position=open_position,
                remembered_interval=self.position_intervals.get(item.symbol),
                filled_fraction=self.filled_fraction_by_symbol.get(
                    item.symbol,
                    _inferred_auto_trade_fraction(latest_backtest, open_position)
                    if open_position is not None
                    else 0.0,
                ),
                remembered_cursor_entry_time=self.auto_trade_cursor_entry_time.get(item.symbol),
                allow_favorable_price_entries=self.auto_trade_use_favorable_price,
                trigger_symbol=trigger_symbol,
                trigger_interval=trigger_interval,
                trigger_bar_time=normalized_trigger_time,
            )
            if evaluation.reentry_position_side:
                self._trigger_reentry_auto_close(item.symbol, evaluation.reentry_position_side, latest_backtest)
                continue
            if evaluation.candidate is None:
                continue
            candidates.append(dict(evaluation.candidate))
            continue
            # wasteful enter→close→re-enter cycle on the same bar.
            if open_position is not None:
                position_side = "long" if float(open_position.amount) > 0 else "short"
                if side != position_side:
                    continue
                # Detect close→reopen: the backtest closed the position and
                # re-entered, but the real exchange position hasn't been closed
                # yet.  Skip entry and let auto-close handle the old position.
                cursor_entry_time = signal.get("cursor_entry_time")
                remembered = self.auto_trade_cursor_entry_time.get(item.symbol)
                if remembered is not None and cursor_entry_time is not None:
                    if pd.Timestamp(remembered).tz_localize(None) != pd.Timestamp(cursor_entry_time).tz_localize(None):
                        self._trigger_reentry_auto_close(item.symbol, position_side, latest_backtest)
                        continue
                filled_fraction = self.filled_fraction_by_symbol.get(
                    item.symbol,
                    _inferred_auto_trade_fraction(latest_backtest, open_position),
                )
                fraction = max(0.0, fraction - float(filled_fraction))
                if fraction <= 1e-9:
                    continue
                if not has_fresh_confirmed_entry:
                    favorable_fraction = _zone_favorable_fraction(
                        side, current_price, signal_price, zone_prices, float(filled_fraction),
                    )
                    if favorable_fraction <= 1e-9:
                        continue
                    fraction = min(fraction, favorable_fraction)
                # Additional entry (e.g. L2→L3): enter immediately when the
                # signal first appears (same bar).  On subsequent bars, use
                # zone-level favorable price checks so that each zone enters
                # independently at its own price threshold.
            else:
                if not has_fresh_confirmed_entry:
                    favorable_fraction = _zone_favorable_fraction(
                        side, current_price, signal_price, zone_prices, 0.0,
                    )
                    if favorable_fraction <= 1e-9:
                        continue
                    fraction = min(fraction, favorable_fraction)
            candidates.append(
                {
                    "symbol": item.symbol,
                    "interval": item.interval,
                    "side": "BUY" if side == "long" else "SELL",
                    "score": float(item.score),
                    "return_pct": float(latest_backtest.metrics.total_return_pct),
                    "fraction": float(fraction),
                    "strategy_settings": item.strategy_settings,
                    "cursor_entry_time": signal.get("cursor_entry_time"),
                }
            )
        chosen = pick_auto_trade_candidate(candidates, self.optimization_rank_mode)
        if chosen is None:
            return
        # Remember the cursor entry_time on first entry so we can detect
        # close→reopen cycles on subsequent evaluations.
        symbol = str(chosen["symbol"])
        if symbol not in self.auto_trade_cursor_entry_time:
            cet = chosen.get("cursor_entry_time")
            if cet is not None:
                self.auto_trade_cursor_entry_time[symbol] = pd.Timestamp(cet)
        self._enqueue_open_order(
            symbol=symbol,
            interval=str(chosen["interval"]),
            side=str(chosen["side"]),
            fraction=float(chosen["fraction"]),
            auto_trade=True,
            strategy_settings=chosen.get("strategy_settings"),
        )

    def _enqueue_open_order(
        self,
        *,
        symbol: str,
        interval: str,
        side: str,
        fraction: Optional[float] = None,
        margin: Optional[float] = None,
        auto_trade: bool = False,
        strategy_settings: Optional[StrategySettings] = None,
    ) -> None:
        if self.client is None or not self.api_key or not self.api_secret:
            self.emit(
                EngineOrderFailedEvent(
                    symbol=symbol,
                    message=f"{symbol} order failed: API credentials are missing",
                    auto_trade=auto_trade,
                    interval=interval,
                    fraction=float(fraction or 0.0),
                )
            )
            return
        if symbol in self.pending_order_symbols:
            self.emit(
                EngineOrderFailedEvent(
                    symbol=symbol,
                    message=f"{symbol} order skipped: existing engine order is pending",
                    auto_trade=auto_trade,
                    interval=interval,
                    fraction=float(fraction or 0.0),
                )
            )
            return
        self.pending_order_symbols[symbol] = time.time()
        self.emit(
            EngineOrderSubmittedEvent(
                symbol=symbol,
                auto_close=False,
                auto_trade=auto_trade,
                interval=interval,
                fraction=float(fraction or 0.0),
            )
        )
        latest_price = float(self.latest_stream_price_by_symbol.get(symbol, 0.0) or 0.0)
        if latest_price <= 0:
            try:
                latest_price = float(self.client.get_latest_price(symbol))
            except Exception:
                latest_price = 0.0
        apply_leverage = self.applied_leverage_by_symbol.get(symbol) != int(self.leverage)
        self.order_sequence += 1
        self.order_request_queue.put(
            (
                1,
                self.order_sequence,
                _OrderRequest(
                    symbol=symbol,
                    side=side,
                    leverage=self.leverage,
                    apply_leverage=apply_leverage,
                    fraction=fraction,
                    margin=margin,
                    price=latest_price if latest_price > 0 else None,
                    interval=interval,
                    auto_close=False,
                    auto_trade=auto_trade,
                    reason=None,
                    strategy_settings=strategy_settings,
                    close_side=None,
                    close_quantity=None,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                ),
            )
        )

    def _enqueue_close_order(self, symbol: str, reason: Optional[str], auto_close: bool) -> None:
        if self.client is None or not self.api_key or not self.api_secret:
            self.emit(
                EngineOrderFailedEvent(
                    symbol=symbol,
                    message=f"{symbol} close failed: API credentials are missing",
                    auto_close=auto_close,
                )
            )
            return
        if symbol in self.pending_order_symbols:
            self.emit(
                EngineOrderFailedEvent(
                    symbol=symbol,
                    message=f"{symbol} close skipped: existing engine order is pending",
                    auto_close=auto_close,
                    interval=self.position_intervals.get(symbol),
                )
            )
            return
        self.pending_order_symbols[symbol] = time.time()
        self.emit(
            EngineOrderSubmittedEvent(
                symbol=symbol,
                auto_close=auto_close,
                auto_trade=False,
                interval=self.position_intervals.get(symbol),
            )
        )
        position = self.open_positions.get(symbol)
        close_side: Optional[str] = None
        close_quantity: Optional[float] = None
        if position is not None and abs(float(position.amount)) > 1e-12:
            close_side = "SELL" if float(position.amount) > 0 else "BUY"
            close_quantity = abs(float(position.amount))
        self.order_sequence += 1
        self.order_request_queue.put(
            (
                0,
                self.order_sequence,
                _OrderRequest(
                    symbol=symbol,
                    side=None,
                    leverage=self.leverage,
                    apply_leverage=False,
                    fraction=None,
                    margin=None,
                    price=None,
                    interval=self.position_intervals.get(symbol),
                    auto_close=auto_close,
                    auto_trade=False,
                    reason=reason,
                    strategy_settings=self.position_strategy_by_symbol.get(symbol),
                    close_side=close_side,
                    close_quantity=close_quantity,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                ),
            )
        )

    def _handle_manual_open(self, command: EngineOpenOrderCommand) -> None:
        self._enqueue_open_order(
            symbol=command.symbol,
            interval=command.interval,
            side=command.side,
            fraction=command.fraction,
            margin=command.margin,
            auto_trade=command.auto_trade,
            strategy_settings=command.strategy_settings,
        )

    def _handle_manual_close(self, command: EngineCloseOrderCommand) -> None:
        self._enqueue_close_order(command.symbol, command.reason, auto_close=command.auto_close)


def run_trade_engine_process(command_queue: mp.Queue, event_queue: mp.Queue) -> None:
    engine = _TradeEngine(command_queue, event_queue)
    try:
        engine.run()
    except Exception:
        try:
            event_queue.put(EngineHealthEvent("failed", traceback.format_exc()))
        except Exception:
            pass
