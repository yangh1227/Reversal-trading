from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import hashlib
import hmac
import math
import re
import threading
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from .strategy import rsi_last_value


FAPI_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"
TICKER_24H_PATH = "/fapi/v1/ticker/24hr"
KLINES_PATH = "/fapi/v1/klines"
ORDER_PATH = "/fapi/v1/order"
ACCOUNT_PATH = "/fapi/v2/account"
POSITION_RISK_PATH = "/fapi/v2/positionRisk"
LEVERAGE_PATH = "/fapi/v1/leverage"
REQUEST_MIN_INTERVAL_SECONDS = 0.25
CUSTOM_INTERVAL_BASE = {
    "2m": "1m",
}


@dataclass(frozen=True)
class CandidateSymbol:
    symbol: str
    last_price: float
    price_change_pct: float
    quote_volume: float
    daily_volatility_pct: float
    rsi_1m: float


@dataclass(frozen=True)
class BalanceSnapshot:
    total_wallet_balance: float
    available_balance: float
    equity: float
    unrealized_pnl: float


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    amount: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int


def _normalize_unrealized_pnl(amount: float, entry_price: float, mark_price: float, fallback: float) -> float:
    if math.isfinite(amount) and math.isfinite(entry_price) and math.isfinite(mark_price) and entry_price > 0 and mark_price > 0:
        calculated = (mark_price - entry_price) * amount
        if math.isfinite(calculated):
            return 0.0 if abs(calculated) < 1e-10 else float(calculated)
    return float(fallback)


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


def _rsi_with_pandas_ta(series: pd.Series, length: int = 14) -> float:
    return rsi_last_value(series.astype(float), length=length)


def _daily_volatility_from_klines(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    if len(df) == 1:
        row = df.iloc[-1]
        return float((float(row["high"]) - float(row["low"])) / max(abs(float(row["low"])), 1e-9) * 100.0)
    prev_close = float(df.iloc[-2]["close"])
    row = df.iloc[-1]
    high = float(row["high"])
    low = float(row["low"])
    true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return float(true_range / max(abs(low), 1e-9) * 100.0)


def _rows_to_ohlcv_frame(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "quote_volume"])
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    frame = frame.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric_cols] = frame[numeric_cols].astype(float)
    frame["time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True).dt.tz_convert(None)
    now_ms = int(time.time() * 1000)
    frame = frame[frame["close_time_ms"].astype(np.int64) <= now_ms]
    return frame[["time", "open", "high", "low", "close", "volume", "quote_volume"]].reset_index(drop=True)


def resolve_base_interval(interval: str) -> str:
    return CUSTOM_INTERVAL_BASE.get(interval, interval)


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        columns = ["time", "open", "high", "low", "close", "volume"]
        if "quote_volume" in df.columns:
            columns.append("quote_volume")
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    numeric_columns = [column for column in ("open", "high", "low", "close", "volume", "quote_volume") if column in frame.columns]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame = frame.sort_values("time").reset_index(drop=True)

    if interval != "2m":
        ordered_columns = ["time", "open", "high", "low", "close", "volume"]
        if "quote_volume" in frame.columns:
            ordered_columns.append("quote_volume")
        return frame[ordered_columns].reset_index(drop=True)

    indexed = frame.set_index("time")
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in indexed.columns:
        aggregations["quote_volume"] = "sum"
    resampled = indexed.resample("2min", label="left", closed="left").agg(aggregations)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    ordered_columns = ["time", "open", "high", "low", "close", "volume"]
    if "quote_volume" in resampled.columns:
        ordered_columns.append("quote_volume")
    return resampled[ordered_columns].reset_index(drop=True)


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


def _format_rate_limit_error(path: str, response: requests.Response) -> str:
    wait_seconds = _retry_after_seconds(response)
    wait_text = f"{wait_seconds:.0f}초 후 재시도" if wait_seconds is not None else "잠시 후 재시도"
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    message = str(payload.get("msg", "") or "")
    matched = re.search(r"until (\d{13})", message)
    if matched:
        banned_until_ms = int(matched.group(1))
        local_until = datetime.fromtimestamp(banned_until_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return (
            f"Binance 요청 제한으로 IP가 일시 차단되었습니다 ({path}, HTTP 418). "
            f"{wait_text}, 해제 예정 시각: {local_until}."
        )
    return f"Binance 요청 제한에 걸렸습니다 ({path}, HTTP {response.status_code}). {wait_text}."


def _round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    text = str(step).rstrip("0")
    precision = len(text.split(".")[-1]) if "." in text else 0
    return round(math.floor(value / step) * step, precision)


class BinanceFuturesClient:
    _request_gate_lock = threading.Lock()
    _next_request_at = 0.0

    def __init__(self, api_key: str = "", api_secret: str = "", base_url: str = FAPI_BASE_URL) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = base_url.rstrip("/")
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def _wait_for_request_slot(cls) -> None:
        with cls._request_gate_lock:
            now = time.monotonic()
            delay = max(0.0, cls._next_request_at - now)
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            cls._next_request_at = now + REQUEST_MIN_INTERVAL_SECONDS

    def _cached(self, key: str, ttl_seconds: float) -> Any:
        with self._cache_lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            timestamp, value = entry
            if time.time() - timestamp > ttl_seconds:
                self._cache.pop(key, None)
                return None
            return value

    def _store_cache(self, key: str, value: Any) -> Any:
        with self._cache_lock:
            self._cache[key] = (time.time(), value)
        return value

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        retries: int = 3,
    ) -> Any:
        params = dict(params or {})
        headers: Dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("API key/secret not configured")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 60_000
            query_string = urlencode(params)
            params["signature"] = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key

        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                self._wait_for_request_slot()
                response = requests.request(method, url, params=params, headers=headers, timeout=20)
                if response.status_code == 418:
                    raise RuntimeError(_format_rate_limit_error(path, response))
                if response.status_code == 429:
                    wait_seconds = _retry_after_seconds(response) or (1.5 * (attempt + 1))
                    if attempt == retries - 1:
                        raise RuntimeError(_format_rate_limit_error(path, response))
                    time.sleep(wait_seconds)
                    continue
                if response.status_code >= 500:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == retries - 1:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Binance request failed for {path}: {last_error}")

    def exchange_info(self) -> Dict[str, Any]:
        cached = self._cached("exchange_info", 600)
        if cached is not None:
            return cached
        return self._store_cache("exchange_info", self._request("GET", EXCHANGE_INFO_PATH))

    def ticker_24h(self) -> Dict[str, Dict[str, Any]]:
        cached = self._cached("ticker_24h", 10)
        if cached is not None:
            return cached
        payload = self._request("GET", TICKER_24H_PATH)
        return self._store_cache("ticker_24h", {entry["symbol"]: entry for entry in payload})

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        ttl_seconds: float = 5.0,
    ) -> List[List[Any]]:
        key = f"klines:{symbol}:{interval}:{limit}:{start_time}:{end_time}"
        cached = self._cached(key, ttl_seconds)
        if cached is not None:
            return cached
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._store_cache(key, self._request("GET", KLINES_PATH, params=params))

    def historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        base_interval = resolve_base_interval(interval)
        if base_interval != interval:
            base_df = self.historical_ohlcv(symbol, base_interval, start_time=start_time, end_time=end_time)
            return resample_ohlcv(base_df, interval)
        step_ms = _interval_to_ms(interval)
        end_ms = end_time or int(time.time() * 1000)
        cursor = int(start_time)
        rows: List[List[Any]] = []
        while cursor < end_ms:
            batch = self.klines(symbol, interval, limit=1500, start_time=cursor, end_time=end_ms, ttl_seconds=0.0)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + step_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1500:
                break
        return _rows_to_ohlcv_frame(rows)

    def historical_ohlcv_recent(
        self,
        symbol: str,
        interval: str,
        bars: int,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        base_interval = resolve_base_interval(interval)
        if base_interval != interval:
            base_bars = max(int(bars), 1) * max(1, _interval_to_ms(interval) // _interval_to_ms(base_interval)) + 2
            base_df = self.historical_ohlcv_recent(symbol, base_interval, bars=base_bars, end_time=end_time)
            resampled = resample_ohlcv(base_df, interval)
            return resampled.tail(max(int(bars), 1)).reset_index(drop=True)
        target_bars = max(int(bars), 1)
        cursor_end = int(end_time or time.time() * 1000)
        rows: List[List[Any]] = []

        while target_bars > 0:
            batch_limit = min(target_bars, 1500)
            batch = self.klines(symbol, interval, limit=batch_limit, end_time=cursor_end, ttl_seconds=0.0)
            if not batch:
                break
            rows.extend(batch)
            target_bars -= len(batch)
            earliest_open_time = int(batch[0][0])
            if len(batch) < batch_limit or earliest_open_time <= 0:
                break
            cursor_end = earliest_open_time - 1

        return _rows_to_ohlcv_frame(rows)

    def usdt_perpetual_symbols(self) -> List[str]:
        symbols = []
        for entry in self.exchange_info().get("symbols", []):
            if (
                entry.get("status") == "TRADING"
                and entry.get("quoteAsset") == "USDT"
                and entry.get("contractType") == "PERPETUAL"
            ):
                symbols.append(entry["symbol"])
        return symbols

    def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        for entry in self.exchange_info().get("symbols", []):
            if entry.get("symbol") != symbol:
                continue
            filters = {flt["filterType"]: flt for flt in entry.get("filters", [])}
            lot_size = filters.get("LOT_SIZE", {})
            market_lot = filters.get("MARKET_LOT_SIZE", lot_size)
            price_filter = filters.get("PRICE_FILTER", {})
            notional_filter = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
            return {
                "pricePrecision": int(entry.get("pricePrecision", 8)),
                "quantityPrecision": int(entry.get("quantityPrecision", 8)),
                "stepSize": float(lot_size.get("stepSize", 0.0) or 0.0),
                "marketStepSize": float(market_lot.get("stepSize", lot_size.get("stepSize", 0.0)) or 0.0),
                "minQty": float(lot_size.get("minQty", 0.0) or 0.0),
                "marketMinQty": float(market_lot.get("minQty", lot_size.get("minQty", 0.0)) or 0.0),
                "tickSize": float(price_filter.get("tickSize", 0.0) or 0.0),
                "minNotional": float(notional_filter.get("minNotional", 0.0) or 0.0),
            }
        raise KeyError(f"unknown symbol: {symbol}")

    def get_latest_price(self, symbol: str) -> float:
        ticker = self.ticker_24h().get(symbol)
        if not ticker:
            raise RuntimeError(f"ticker not found for {symbol}")
        return float(ticker.get("lastPrice", 0.0) or 0.0)

    def get_balance_snapshot(self) -> BalanceSnapshot:
        payload = self._request("GET", ACCOUNT_PATH, signed=True)
        wallet = float(payload.get("totalWalletBalance", 0.0))
        available = float(payload.get("availableBalance", 0.0))
        unrealized = float(payload.get("totalUnrealizedProfit", 0.0))
        equity = float(payload.get("totalMarginBalance", wallet + unrealized))
        return BalanceSnapshot(wallet, available, equity, unrealized)

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        for position in self.get_open_positions():
            if position.symbol == symbol:
                return position
        return None

    def get_open_positions(self) -> List[PositionSnapshot]:
        payload = self._request("GET", POSITION_RISK_PATH, signed=True)
        positions: List[PositionSnapshot] = []
        for entry in payload:
            amount = float(entry.get("positionAmt", 0.0))
            if abs(amount) < 1e-12:
                continue
            entry_price = float(entry.get("entryPrice", 0.0))
            mark_price = float(entry.get("markPrice", 0.0))
            api_unrealized = float(entry.get("unRealizedProfit", 0.0))
            positions.append(
                PositionSnapshot(
                    symbol=str(entry.get("symbol", "")),
                    amount=amount,
                    entry_price=entry_price,
                    mark_price=mark_price,
                    unrealized_pnl=_normalize_unrealized_pnl(amount, entry_price, mark_price, api_unrealized),
                    leverage=int(float(entry.get("leverage", 0.0) or 0.0)),
                )
            )
        positions.sort(key=lambda item: (item.symbol,))
        return positions

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return self._request("POST", LEVERAGE_PATH, params={"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def build_order_quantity(self, symbol: str, usdt_margin: float, leverage: int, price: Optional[float] = None) -> float:
        market_price = price or self.get_latest_price(symbol)
        filters = self.get_symbol_filters(symbol)
        quantity = (float(usdt_margin) * float(leverage)) / max(market_price, 1e-9)
        step = filters["marketStepSize"] or filters["stepSize"]
        quantity = _round_down(quantity, step)
        min_qty = filters["marketMinQty"] or filters["minQty"]
        if quantity < min_qty:
            raise RuntimeError(f"{symbol} quantity below minimum ({quantity} < {min_qty})")
        min_notional = filters["minNotional"]
        if min_notional and quantity * market_price < min_notional:
            raise RuntimeError(f"{symbol} order notional below minimum")
        return quantity

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        filters = self.get_symbol_filters(symbol)
        step = filters["marketStepSize"] or filters["stepSize"]
        normalized_qty = _round_down(quantity, step)
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": normalized_qty,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", ORDER_PATH, params=params, signed=True)

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        position = self.get_position(symbol)
        if position is None:
            return None
        side = "SELL" if position.amount > 0 else "BUY"
        return self.place_market_order(symbol, side, abs(position.amount), reduce_only=True)

    def scan_alt_candidates(
        self,
        daily_volatility_min: float,
        quote_volume_min: float,
        rsi_length: int,
        rsi_lower: float,
        rsi_upper: float,
        workers: int = 8,
        log_callback: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[CandidateSymbol]:
        def log(message: str) -> None:
            if log_callback:
                log_callback(message)

        ticker_map = self.ticker_24h()
        pre_symbols = []
        for symbol in self.usdt_perpetual_symbols():
            if should_stop and should_stop():
                return []
            ticker = ticker_map.get(symbol)
            if not ticker:
                continue
            if float(ticker.get("quoteVolume", 0.0) or 0.0) >= quote_volume_min:
                pre_symbols.append(symbol)
        log(f"거래량 사전선별 {len(pre_symbols)}개: 24h 거래량 {quote_volume_min:,.0f} USDT 이상")

        def enrich(symbol: str) -> Optional[CandidateSymbol]:
            if should_stop and should_stop():
                return None
            ticker = ticker_map[symbol]
            daily_df = _rows_to_ohlcv_frame(self.klines(symbol, "1d", limit=3, ttl_seconds=0.0))
            daily_vol = _daily_volatility_from_klines(daily_df)
            if not np.isfinite(daily_vol) or daily_vol < daily_volatility_min:
                return None
            if should_stop and should_stop():
                return None
            minute_limit = min(max(rsi_length * 3, 60), 99)
            minute_df = _rows_to_ohlcv_frame(self.klines(symbol, "1m", limit=minute_limit, ttl_seconds=0.0))
            if len(minute_df) < max(rsi_length + 5, 30):
                return None
            rsi_value = _rsi_with_pandas_ta(minute_df["close"], rsi_length)
            if not np.isfinite(rsi_value) or not (rsi_value <= rsi_lower or rsi_value >= rsi_upper):
                return None
            return CandidateSymbol(
                symbol=symbol,
                last_price=float(ticker.get("lastPrice", 0.0) or 0.0),
                price_change_pct=float(ticker.get("priceChangePercent", 0.0) or 0.0),
                quote_volume=float(ticker.get("quoteVolume", 0.0) or 0.0),
                daily_volatility_pct=float(daily_vol),
                rsi_1m=float(rsi_value),
            )

        candidates: List[CandidateSymbol] = []
        if not pre_symbols:
            return candidates

        worker_count = max(1, min(int(workers or 1), len(pre_symbols), 4))
        if workers > worker_count:
            log(f"요청 제한 방지를 위해 스캔 워커를 {worker_count}개로 제한합니다.")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(enrich, symbol): symbol for symbol in pre_symbols}
            done = 0
            for future in as_completed(futures):
                if should_stop and should_stop():
                    break
                done += 1
                result = future.result()
                if result is not None:
                    candidates.append(result)
                if done % 10 == 0 or done == len(pre_symbols):
                    log(f"후보 스캔 진행 {done}/{len(pre_symbols)}")

        candidates.sort(
            key=lambda item: (
                item.daily_volatility_pct,
                item.quote_volume,
                abs(item.rsi_1m - 50.0),
            ),
            reverse=True,
        )
        log(
            f"후보 필터 완료 {len(candidates)}개: "
            f"24h 거래량 {quote_volume_min:,.0f}+ / 일변동성 {daily_volatility_min:.2f}%+ / "
            f"1m RSI <= {rsi_lower:.2f} or >= {rsi_upper:.2f}"
        )
        return candidates
