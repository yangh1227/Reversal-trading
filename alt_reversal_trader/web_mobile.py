from __future__ import annotations

import asyncio
import hmac
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .binance_futures import _interval_to_ms
from .qt_compat import QObject, Signal


WEB_USERNAME = "yangh1227"
WEB_PASSWORD = "yang1227"
WEB_BIND_HOST = "0.0.0.0"
WEB_DEFAULT_PORT = 8765
WEB_SESSION_MAX_AGE_SECONDS = 60 * 60 * 2
WEB_LOGIN_RATE_LIMIT_ATTEMPTS = 5
WEB_LOGIN_RATE_LIMIT_WINDOW_SECONDS = 60 * 10
WEB_PRICE_CACHE_SECONDS = 2.0
WEB_MAX_REQUEST_BYTES = 64 * 1024
WEB_VENDOR_JS_PATH = (
    Path(__file__).resolve().parent.parent
    / "node_modules"
    / "lightweight-charts"
    / "dist"
    / "lightweight-charts.standalone.production.js"
)
WEB_STATIC_DIR = Path(__file__).resolve().parent / "web_static"
WEB_SESSION_COOKIE = "altrev_mobile_session"


def _serialize_timestamp(value: object) -> Optional[str]:
    if value in {None, ""}:
        return None
    try:
        return pd.Timestamp(value).tz_localize(None).isoformat()
    except Exception:
        return None


def _serialize_series_frame(frame: pd.DataFrame, value_column: str) -> list[dict[str, object]]:
    if frame.empty or value_column not in frame.columns:
        return []
    return [
        {
            "time": _serialize_timestamp(row["time"]),
            "value": None if pd.isna(row[value_column]) else float(row[value_column]),
        }
        for _, row in frame.iterrows()
        if row.get("time") is not None
    ]


def _decimal_places(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    except Exception:
        return 0
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _price_format_from_frames(candle_df: pd.DataFrame, indicators: pd.DataFrame) -> dict[str, object]:
    precision = 0
    for column in ("open", "high", "low", "close"):
        if column in candle_df.columns:
            precision = max(precision, max((_decimal_places(v) for v in candle_df[column].tail(300)), default=0))
    precision = max(precision, 2)
    return {"precision": precision, "minMove": 10 ** (-precision)}


def _format_compact_number(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    try:
        return f"{float(value):.12f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _safe_port(start_port: int = WEB_DEFAULT_PORT) -> int:
    for port in range(start_port, start_port + 20):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("No available web server port found in 8765-8784")


def _local_ip_addresses() -> list[str]:
    addresses: set[str] = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        for addrinfo in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = str(addrinfo[4][0])
            if ip and not ip.startswith("127."):
                addresses.add(ip)
    except Exception:
        pass
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = str(probe.getsockname()[0])
        if ip and not ip.startswith("127."):
            addresses.add(ip)
    except Exception:
        pass
    finally:
        if probe is not None:
            probe.close()
    return sorted(addresses)


class _UiInvoker(QObject):
    dispatch = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.dispatch.connect(self._execute)

    def _execute(self, payload: dict[str, object]) -> None:
        done = payload["done"]
        try:
            payload["result"] = payload["fn"]()
        except Exception as exc:
            payload["error"] = exc
        finally:
            done.set()

    def call(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        done = threading.Event()
        payload: dict[str, object] = {
            "fn": fn,
            "done": done,
            "result": None,
            "error": None,
        }
        self.dispatch.emit(payload)
        if not done.wait(timeout):
            raise TimeoutError("Timed out waiting for UI thread")
        if payload["error"] is not None:
            raise payload["error"]  # type: ignore[misc]
        return payload["result"]

    def post(self, fn: Callable[[], Any]) -> None:
        payload: dict[str, object] = {
            "fn": fn,
            "done": threading.Event(),
            "result": None,
            "error": None,
        }
        self.dispatch.emit(payload)


@dataclass
class _SessionState:
    username: str
    csrf_token: str
    expires_at: float


class MobileWebServer:
    def __init__(self, window: Any) -> None:
        self.window = window
        self.invoker = _UiInvoker()
        self._price_cache: dict[str, float] = {}
        self._price_cache_at = 0.0
        self._sessions: Dict[str, _SessionState] = {}
        self._login_attempts: Dict[str, list[float]] = {}
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self.port = _safe_port()
        self._app = self._build_app()

    @property
    def urls(self) -> list[str]:
        return [f"http://{host}:{self.port}" for host in _local_ip_addresses()]

    def start(self) -> None:
        if self._thread is not None:
            return
        config = uvicorn.Config(
            self._app,
            host=WEB_BIND_HOST,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="MobileWebServer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._server = None
        self._thread = None

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Alt Reversal Trader Mobile", docs_url=None, redoc_url=None, openapi_url=None)
        app.mount("/mobile-static", StaticFiles(directory=str(WEB_STATIC_DIR)), name="mobile-static")

        @app.exception_handler(Exception)
        async def handle_uncaught_exception(_request: Request, exc: Exception) -> JSONResponse:
            try:
                self.window.log(f"Mobile web error: {exc}")
            except Exception:
                pass
            return JSONResponse({"detail": "모바일 웹 요청 처리 중 오류가 발생했습니다."}, status_code=500)

        @app.middleware("http")
        async def harden_http(request: Request, call_next):
            content_length = str(request.headers.get("content-length", "")).strip()
            if content_length.isdigit() and int(content_length) > WEB_MAX_REQUEST_BYTES:
                return JSONResponse({"detail": "요청 본문이 너무 큽니다."}, status_code=413)
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self' ws: wss:; "
                "font-src 'self' data:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            return response

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse((WEB_STATIC_DIR / "mobile.html").read_text(encoding="utf-8"))

        @app.get("/vendor/lightweight-charts.js")
        def vendor_lightweight_charts() -> FileResponse:
            return FileResponse(str(WEB_VENDOR_JS_PATH), media_type="application/javascript")

        @app.get("/api/me")
        def api_me(request: Request) -> JSONResponse:
            session = self._session_from_request(request)
            return JSONResponse(
                {
                    "authenticated": session is not None,
                    "username": session.username if session else None,
                    "csrfToken": session.csrf_token if session else None,
                    "urls": self.urls,
                }
            )

        @app.post("/api/login")
        async def api_login(request: Request) -> JSONResponse:
            ip = self._client_ip(request)
            self._enforce_login_rate_limit(ip)
            payload = await request.json()
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", ""))
            if not (
                hmac.compare_digest(username, WEB_USERNAME)
                and hmac.compare_digest(password, WEB_PASSWORD)
            ):
                self._record_login_attempt(ip)
                raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
            session_token = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(24)
            self._sessions[session_token] = _SessionState(
                username=username,
                csrf_token=csrf_token,
                expires_at=time.time() + WEB_SESSION_MAX_AGE_SECONDS,
            )
            self._login_attempts.pop(ip, None)
            response = JSONResponse({"ok": True, "csrfToken": csrf_token, "username": username})
            response.set_cookie(
                WEB_SESSION_COOKIE,
                session_token,
                max_age=WEB_SESSION_MAX_AGE_SECONDS,
                httponly=True,
                samesite="lax",
            )
            return response

        @app.post("/api/logout")
        async def api_logout(request: Request) -> JSONResponse:
            session_token = request.cookies.get(WEB_SESSION_COOKIE, "")
            if session_token:
                self._sessions.pop(session_token, None)
            response = JSONResponse({"ok": True})
            response.delete_cookie(WEB_SESSION_COOKIE)
            return response

        @app.get("/api/dashboard")
        def api_dashboard(request: Request) -> JSONResponse:
            self._require_session(request)
            return JSONResponse(self.invoker.call(self._build_dashboard_state))

        @app.get("/api/chart/current")
        def api_chart_current(request: Request) -> JSONResponse:
            self._require_session(request)
            return JSONResponse(self.invoker.call(self._build_current_chart_payload))

        @app.websocket("/ws/live")
        async def ws_live(websocket: WebSocket) -> None:
            session = self._session_from_websocket(websocket)
            if session is None:
                await websocket.close(code=4401)
                return
            origin = str(websocket.headers.get("origin", "")).strip()
            if origin:
                expected_origin = f"{websocket.url.scheme.replace('ws', 'http')}://{websocket.headers.get('host', '')}".rstrip("/")
                if origin.rstrip("/") != expected_origin:
                    await websocket.close(code=4403)
                    return
            await websocket.accept()
            try:
                while True:
                    dashboard = self.invoker.call(self._build_dashboard_state)
                    chart = self.invoker.call(self._build_current_chart_payload)
                    await websocket.send_json({"type": "dashboard", "data": dashboard})
                    await websocket.send_json({"type": "chart", "data": chart})
                    await asyncio.sleep(1.0)
            except WebSocketDisconnect:
                return

        @app.post("/api/chart/select")
        async def api_chart_select(request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            payload = await request.json()
            symbol = str(payload.get("symbol", "")).strip().upper()
            interval = str(payload.get("interval", "")).strip()
            if not symbol:
                raise HTTPException(status_code=400, detail="종목이 필요합니다.")
            return JSONResponse(self.invoker.call(lambda: self._select_symbol(symbol, interval)))

        @app.post("/api/order/fractional")
        async def api_order_fractional(request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            payload = await request.json()
            symbol = str(payload.get("symbol", "")).strip().upper()
            interval = str(payload.get("interval", "")).strip()
            side = str(payload.get("side", "")).strip().upper()
            fraction = float(payload.get("fraction", 0.0) or 0.0)
            if side not in {"BUY", "SELL"}:
                raise HTTPException(status_code=400, detail="BUY 또는 SELL만 가능합니다.")
            self.invoker.post(lambda: self._submit_fractional_order(symbol, interval, side, fraction))
            return JSONResponse({"ok": True, "queued": True})

        @app.post("/api/order/simple")
        async def api_order_simple(request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            payload = await request.json()
            symbol = str(payload.get("symbol", "")).strip().upper()
            interval = str(payload.get("interval", "")).strip()
            side = str(payload.get("side", "")).strip().upper()
            amount = float(payload.get("amount", 0.0) or 0.0)
            if side not in {"BUY", "SELL"}:
                raise HTTPException(status_code=400, detail="BUY 또는 SELL만 가능합니다.")
            self.invoker.post(lambda: self._submit_simple_order(symbol, interval, side, amount))
            return JSONResponse({"ok": True, "queued": True})

        @app.post("/api/positions/close-all")
        async def api_close_all(request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            self.invoker.post(self._close_all_positions)
            return JSONResponse({"ok": True, "queued": True})

        @app.post("/api/positions/{symbol}/close")
        async def api_close_symbol(symbol: str, request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            self.invoker.post(lambda: self._close_position(str(symbol or "").strip().upper()))
            return JSONResponse({"ok": True, "queued": True})

        @app.post("/api/positions/{symbol}/auto-close")
        async def api_toggle_auto_close(symbol: str, request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            payload = await request.json()
            return JSONResponse(
                self.invoker.call(
                    lambda: self._toggle_auto_close(str(symbol or "").strip().upper(), bool(payload.get("enabled", False)))
                )
            )

        @app.post("/api/auto-trade")
        async def api_toggle_auto_trade(request: Request) -> JSONResponse:
            self._require_session(request, require_csrf=True)
            payload = await request.json()
            return JSONResponse(self.invoker.call(lambda: self._toggle_auto_trade(bool(payload.get("enabled", False)))))

        return app

    def _session_from_request(self, request: Request) -> Optional[_SessionState]:
        token = request.cookies.get(WEB_SESSION_COOKIE, "")
        return self._session_from_token(token)

    def _session_from_websocket(self, websocket: WebSocket) -> Optional[_SessionState]:
        token = websocket.cookies.get(WEB_SESSION_COOKIE, "")
        return self._session_from_token(token)

    def _session_from_token(self, token: str) -> Optional[_SessionState]:
        if not token:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        if session.expires_at <= time.time():
            self._sessions.pop(token, None)
            return None
        return session

    def _require_session(self, request: Request, require_csrf: bool = False) -> _SessionState:
        session = self._session_from_request(request)
        if session is None:
            raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
        if require_csrf:
            origin = str(request.headers.get("origin", "")).strip()
            if origin:
                expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}".rstrip("/")
                if origin.rstrip("/") != expected_origin:
                    raise HTTPException(status_code=403, detail="허용되지 않은 요청 Origin입니다.")
            token = str(request.headers.get("X-CSRF-Token", "")).strip()
            if not token or not hmac.compare_digest(token, session.csrf_token):
                raise HTTPException(status_code=403, detail="잘못된 요청 토큰입니다.")
        return session

    def _client_ip(self, request: Request) -> str:
        client = request.client
        return str(client.host if client else "unknown")

    def _enforce_login_rate_limit(self, ip: str) -> None:
        now = time.time()
        attempts = [ts for ts in self._login_attempts.get(ip, []) if now - ts < WEB_LOGIN_RATE_LIMIT_WINDOW_SECONDS]
        self._login_attempts[ip] = attempts
        if len(attempts) >= WEB_LOGIN_RATE_LIMIT_ATTEMPTS:
            raise HTTPException(status_code=429, detail="로그인 시도가 너무 많습니다. 잠시 후 다시 시도하세요.")

    def _record_login_attempt(self, ip: str) -> None:
        self._login_attempts.setdefault(ip, []).append(time.time())

    def _cached_price_map(self) -> dict[str, float]:
        now = time.time()
        if now - self._price_cache_at <= WEB_PRICE_CACHE_SECONDS:
            return dict(self._price_cache)
        symbols = [result.symbol for result in self.window._ordered_optimized_results()]
        if not symbols:
            self._price_cache = {}
            self._price_cache_at = now
            return {}
        try:
            self._price_cache = self.window._optimized_table_price_map(log_failures=False)
        except Exception:
            self._price_cache = {}
        self._price_cache_at = now
        return dict(self._price_cache)

    def _current_bar_close_deadline_ms(self) -> Optional[int]:
        interval = str(self.window.current_interval or self.window.settings.kline_interval or "").strip()
        if not interval:
            return None
        try:
            value = int(interval[:-1])
        except Exception:
            return None
        unit = interval[-1]
        if unit == "m":
            floor_freq = f"{value}min"
        elif unit == "h":
            floor_freq = f"{value}h"
        elif unit == "d":
            floor_freq = f"{value}d"
        else:
            return None
        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
        bar_start = now.floor(floor_freq)
        bar_end = bar_start + pd.Timedelta(milliseconds=_interval_to_ms(interval))
        return int(bar_end.timestamp() * 1000)

    def _current_countdown(self) -> Optional[str]:
        deadline_ms = self._current_bar_close_deadline_ms()
        if deadline_ms is None:
            return None
        remaining_seconds = max(0, int((deadline_ms - int(time.time() * 1000)) / 1000))
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _build_dashboard_state(self) -> dict[str, object]:
        if self.window.current_symbol is None:
            first = next(iter(self.window._ordered_optimized_results()), None)
            if first is not None:
                self.window._request_symbol_load(first.symbol, first.best_interval or self.window.settings.kline_interval)
        price_map = self._cached_price_map()
        ordered_results = self.window._ordered_optimized_results()
        favorable_symbols: list[str] = []
        optimized_rows: list[dict[str, object]] = []
        for result in ordered_results:
            metrics = result.best_backtest.metrics
            favorable = self.window._optimized_result_has_favorable_entry(result, price_map.get(result.symbol))
            if favorable:
                favorable_symbols.append(result.symbol)
            optimized_rows.append(
                {
                    "symbol": result.symbol,
                    "interval": result.best_interval or self.window.settings.kline_interval,
                    "score": round(float(result.score), 2),
                    "returnPct": round(float(metrics.total_return_pct), 2),
                    "mddPct": round(float(metrics.max_drawdown_pct), 2),
                    "trades": int(metrics.trade_count),
                    "favorable": favorable,
                    "currentPrice": _format_compact_number(price_map.get(result.symbol)),
                    "isCurrent": result.symbol == self.window.current_symbol,
                }
            )
        positions = []
        for position in self.window.open_positions:
            values, upnl_value, return_pct = self.window._position_display_values(position)
            positions.append(
                {
                    "symbol": position.symbol,
                    "side": values[1],
                    "leverage": values[2],
                    "amountUsdt": values[3],
                    "entryPrice": values[4],
                    "upnl": round(upnl_value, 2),
                    "returnPct": round(return_pct, 2),
                    "autoCloseEnabled": position.symbol in self.window.auto_close_enabled_symbols or self.window.auto_trade_enabled,
                }
            )
        snapshot = self.window.account_balance_snapshot
        equity = available = None
        if snapshot is not None:
            live_unrealized = self.window._live_total_unrealized_pnl()
            unrealized_delta = live_unrealized - float(snapshot.unrealized_pnl)
            equity = round(float(snapshot.equity) + unrealized_delta, 2)
            available = round(float(snapshot.available_balance) + unrealized_delta, 2)
        return {
            "serverUrls": self.urls,
            "current": {
                "symbol": self.window.current_symbol,
                "interval": self.window.current_interval,
                "countdown": self._current_countdown(),
                "barCloseDeadlineMs": self._current_bar_close_deadline_ms(),
                "signalText": self.window.signal_label.text() if hasattr(self.window, "signal_label") else "",
                "chartVersion": repr(self.window.chart_render_signature),
            },
            "balance": {"equity": equity, "available": available},
            "autoTradeEnabled": bool(self.window.auto_trade_enabled or self.window.auto_trade_requested),
            "simpleOrderAmount": float(self.window.simple_order_amount_spin.value()),
            "optimized": optimized_rows,
            "favorableSymbols": favorable_symbols,
            "positions": positions,
        }

    def _build_current_chart_payload(self) -> dict[str, object]:
        symbol = self.window.current_symbol
        interval = self.window.current_interval
        if not symbol or self.window.current_backtest is None:
            return {"symbol": symbol, "interval": interval, "ready": False}
        preview_bar = self.window._current_live_preview_for(symbol, interval)
        snapshot = self.window._sync_current_chart_snapshot(
            symbol,
            interval,
            backtest=self.window.current_backtest,
            chart_indicators=self.window.current_chart_indicators,
            preview_bar=preview_bar,
        )
        if snapshot is None:
            return {"symbol": symbol, "interval": interval, "ready": False}
        candle_df, indicators, _, markers = self.window._build_chart_render_payload(snapshot)
        live_chart_history = snapshot.display_chart_history()
        if live_chart_history is not None and not live_chart_history.empty:
            candle_df = (
                live_chart_history[["time", "open", "high", "low", "close"]]
                .sort_values("time")
                .drop_duplicates(subset=["time"])
                .reset_index(drop=True)
            )
        return {
            "ready": True,
            "symbol": symbol,
            "interval": interval,
            "priceFormat": _price_format_from_frames(candle_df, indicators),
            "candles": [
                {
                    "time": _serialize_timestamp(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for _, row in candle_df.iterrows()
            ],
            "indicators": {
                "supertrend": _serialize_series_frame(indicators[["time", "supertrend"]], "supertrend") if "supertrend" in indicators else [],
                "zone2": _serialize_series_frame(indicators[["time", "zone2_line"]], "zone2_line") if "zone2_line" in indicators else [],
                "zone3": _serialize_series_frame(indicators[["time", "zone3_line"]], "zone3_line") if "zone3_line" in indicators else [],
                "emaFast": _serialize_series_frame(indicators[["time", "ema_fast"]], "ema_fast") if "ema_fast" in indicators else [],
                "emaSlow": _serialize_series_frame(indicators[["time", "ema_slow"]], "ema_slow") if "ema_slow" in indicators else [],
            },
            "markers": [
                {
                    "time": _serialize_timestamp(marker.get("time")),
                    "position": marker.get("position"),
                    "shape": marker.get("shape"),
                    "color": marker.get("color"),
                    "text": marker.get("text"),
                }
                for marker in self.window._compose_lightweight_markers(markers)
            ],
        }

    def _select_symbol(self, symbol: str, interval: str) -> dict[str, object]:
        optimization = self.window._optimization_result(symbol)
        chosen_interval = interval or (
            (optimization.best_interval if optimization else None)
            or self.window._position_interval_for_symbol(symbol)
            or self.window.settings.kline_interval
        )
        self.window._request_symbol_load(symbol, chosen_interval, prefer_locked_position_settings=True)
        return {"ok": True, "symbol": symbol, "interval": chosen_interval}

    def _submit_fractional_order(self, symbol: str, interval: str, side: str, fraction: float) -> dict[str, object]:
        if fraction <= 0:
            raise HTTPException(status_code=400, detail="비율은 0보다 커야 합니다.")
        self.window._submit_open_order(side, fraction=float(fraction), symbol=symbol or None, interval=interval or None)
        return {"ok": True}

    def _submit_simple_order(self, symbol: str, interval: str, side: str, amount: float) -> dict[str, object]:
        if amount <= 0:
            raise HTTPException(status_code=400, detail="주문 금액은 0보다 커야 합니다.")
        self.window._submit_open_order(side, margin=float(amount), symbol=symbol or None, interval=interval or None)
        return {"ok": True}

    def _close_all_positions(self) -> dict[str, object]:
        self.window.close_all_positions()
        return {"ok": True}

    def _close_position(self, symbol: str) -> dict[str, object]:
        if not symbol:
            raise HTTPException(status_code=400, detail="종목이 필요합니다.")
        self.window.close_position_for_symbol(symbol)
        return {"ok": True}

    def _toggle_auto_close(self, symbol: str, enabled: bool) -> dict[str, object]:
        if not symbol:
            raise HTTPException(status_code=400, detail="종목이 필요합니다.")
        self.window._toggle_auto_close_for_symbol(symbol, enabled)
        return {"ok": True, "enabled": enabled}

    def _toggle_auto_trade(self, enabled: bool) -> dict[str, object]:
        self.window.auto_trade_button.setChecked(bool(enabled))
        return {"ok": True, "enabled": bool(enabled)}
