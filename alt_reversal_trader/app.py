from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from lightweight_charts.widgets import QtChart
import websocket

from .binance_futures import BinanceFuturesClient, CandidateSymbol
from .config import APP_INTERVAL_OPTIONS, CHART_ENGINE_OPTIONS, PARAMETER_SPECS, AppSettings, StrategySettings
from .optimizer import OptimizationResult, optimize_symbol
from .qt_compat import (
    HORIZONTAL,
    NO_EDIT_TRIGGERS,
    PASSWORD_ECHO,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
    VERTICAL,
    WEB_ATTR_FILE_URLS,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QThread,
    QTimer,
    QUrl,
    QVBoxLayout,
    QWebEngineView,
    QWidget,
    Signal,
)
from .strategy import BacktestResult, compute_indicators, estimate_warmup_bars, run_backtest


CHART_HISTORY_BAR_LIMIT = 20_000
BACKTEST_WARMUP_BAR_FLOOR = 1_500
DEFAULT_CHART_LOOKBACK_HOURS = 12
LIVE_RENDER_INTERVAL_MS = 500


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


def _backtest_start_time_ms(settings: AppSettings) -> int:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    return now_ms - settings.history_days * 86_400_000


def _history_fetch_start_time_ms(settings: AppSettings) -> int:
    interval_ms = _interval_to_ms(settings.kline_interval)
    warmup_bars = max(estimate_warmup_bars(settings.strategy), BACKTEST_WARMUP_BAR_FLOOR)
    return _backtest_start_time_ms(settings) - warmup_bars * interval_ms


def _ws_kline_timestamp(time_ms: int) -> pd.Timestamp:
    return pd.to_datetime(int(time_ms), unit="ms", utc=True).tz_convert(None)


def _merge_live_bar(df: pd.DataFrame, bar: Dict[str, object], max_rows: Optional[int] = None) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    frame = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=columns)
    row = {key: bar[key] for key in columns}
    if frame.empty:
        frame = pd.DataFrame([row], columns=columns)
    else:
        frame["time"] = pd.to_datetime(frame["time"])
        row_time = pd.Timestamp(row["time"])
        last_time = pd.Timestamp(frame["time"].iloc[-1])
        if last_time == row_time:
            for key in columns[1:]:
                frame.at[frame.index[-1], key] = row[key]
        elif last_time < row_time:
            frame = pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
        else:
            matches = frame["time"] == row_time
            if matches.any():
                idx = frame.index[matches][-1]
                for key in columns[1:]:
                    frame.at[idx, key] = row[key]
            else:
                frame = (
                    pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
                    .drop_duplicates(subset=["time"], keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)
    return frame


class KlineStreamWorker(QThread):
    kline = Signal(object)
    status = Signal(str)

    def __init__(self, symbol: str, interval: str) -> None:
        super().__init__()
        self.symbol = symbol.upper()
        self.interval = interval
        self._stopped = False
        self._socket = None

    def stop(self) -> None:
        self._stopped = True
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

    def run(self) -> None:
        stream_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_{self.interval}"
        while not self._stopped:
            try:
                self.status.emit(f"실시간 스트림 연결: {self.symbol} {self.interval}")
                self._socket = websocket.create_connection(stream_url, timeout=10)
                self._socket.settimeout(1.0)
                while not self._stopped:
                    try:
                        raw = self._socket.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    if payload.get("e") != "kline":
                        continue
                    kline = payload.get("k", {})
                    self.kline.emit(
                        {
                            "symbol": self.symbol,
                            "interval": self.interval,
                            "time": _ws_kline_timestamp(int(kline["t"])),
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"]),
                            "closed": bool(kline.get("x", False)),
                        }
                    )
            except Exception as exc:
                if self._stopped:
                    break
                self.status.emit(f"실시간 스트림 재연결 대기: {self.symbol} ({exc})")
                self.msleep(2000)
            finally:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except Exception:
                        pass
                    self._socket = None
        self.status.emit(f"실시간 스트림 종료: {self.symbol}")


class ScanWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

    def run(self) -> None:
        try:
            client = BinanceFuturesClient()
            candidates = client.scan_alt_candidates(
                daily_volatility_min=self.settings.daily_volatility_min,
                quote_volume_min=self.settings.quote_volume_min,
                rsi_length=self.settings.rsi_length,
                rsi_lower=self.settings.rsi_lower,
                rsi_upper=self.settings.rsi_upper,
                workers=self.settings.scan_workers,
                log_callback=self.progress.emit,
            )
            self.completed.emit(candidates)
        except Exception as exc:
            self.failed.emit(str(exc))


class OptimizeWorker(QThread):
    progress = Signal(str)
    result_ready = Signal(object)
    completed = Signal()
    failed = Signal(str)

    def __init__(self, settings: AppSettings, candidates: List[CandidateSymbol]) -> None:
        super().__init__()
        self.settings = settings
        self.candidates = candidates

    def run(self) -> None:
        try:
            client = BinanceFuturesClient()
            backtest_start_time_ms = _backtest_start_time_ms(self.settings)
            history_fetch_start_time_ms = _history_fetch_start_time_ms(self.settings)
            for index, candidate in enumerate(self.candidates, start=1):
                self.progress.emit(
                    f"[{index}/{len(self.candidates)}] {candidate.symbol} "
                    f"{self.settings.history_days}일 백테스트 + 웜업 K라인 로드"
                )
                df = client.historical_ohlcv(
                    candidate.symbol,
                    self.settings.kline_interval,
                    start_time=history_fetch_start_time_ms,
                )
                if df.empty:
                    self.progress.emit(f"{candidate.symbol}: 히스토리 없음")
                    continue
                optimization = optimize_symbol(
                    symbol=candidate.symbol,
                    df=df,
                    base_settings=self.settings.strategy,
                    optimize_flags=self.settings.optimize_flags,
                    span_pct=self.settings.optimization_span_pct,
                    steps=self.settings.optimization_steps,
                    max_combinations=self.settings.max_grid_combinations,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=pd.to_datetime(backtest_start_time_ms, unit="ms"),
                )
                self.result_ready.emit({"candidate": candidate, "optimization": optimization, "history": df})
                self.progress.emit(
                    f"{candidate.symbol}: {optimization.combinations_tested}개 조합 완료, "
                    f"수익률 {optimization.best_backtest.metrics.total_return_pct:.2f}%"
                )
            self.completed.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class AltReversalTraderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.candidates: List[CandidateSymbol] = []
        self.optimized_results: Dict[str, OptimizationResult] = {}
        self.history_cache: Dict[str, pd.DataFrame] = {}
        self.chart_history_cache: Dict[str, pd.DataFrame] = {}
        self.pending_candidates: List[CandidateSymbol] = []
        self.pending_optimized_results: Dict[str, OptimizationResult] = {}
        self.pending_history_cache: Dict[str, pd.DataFrame] = {}
        self.preserve_lists_during_refresh = False
        self.current_symbol: Optional[str] = None
        self.current_backtest: Optional[BacktestResult] = None
        self.scan_worker: Optional[ScanWorker] = None
        self.optimize_worker: Optional[OptimizeWorker] = None
        self.live_stream_worker: Optional[KlineStreamWorker] = None
        self.live_pending_bar: Optional[Dict[str, object]] = None
        self.auto_refresh_minutes = 10
        self.auto_refresh_timer = QTimer(self)
        self.live_update_timer = QTimer(self)
        self.parameter_editors: Dict[str, object] = {}
        self.parameter_opt_boxes: Dict[str, QCheckBox] = {}
        self.plotly_chart_path = Path("alt_reversal_trader_plotly_chart.html").resolve()
        self.plotly_js_path = self.plotly_chart_path.with_name("plotly.min.js")
        self.chart_mode = ""
        self.chart = None
        self.chart_view = None
        self.equity_subchart = None
        self.equity_line = None
        self.supertrend_line = None
        self.zone2_line = None
        self.zone3_line = None
        self.ema_fast_line = None
        self.ema_slow_line = None

        self.setWindowTitle("Binance Alt Mean Reversion Trader")
        self.resize(1680, 960)
        self._build_ui()
        self._apply_loaded_settings()
        self._init_chart()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self._flush_live_update)
        self.chart_engine_combo.currentTextChanged.connect(self.on_chart_engine_changed)
        self._init_auto_refresh()
        self.statusBar().showMessage("준비됨")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(HORIZONTAL)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_body = QWidget()
        settings_layout = QVBoxLayout(settings_body)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.addWidget(self._build_api_group())
        settings_layout.addWidget(self._build_filter_group())
        settings_layout.addWidget(self._build_optimization_group())
        settings_layout.addWidget(self._build_parameter_tabs())
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_body)
        left_layout.addWidget(settings_scroll, 3)

        actions_row = QHBoxLayout()
        self.save_settings_button = QPushButton("설정 저장")
        self.scan_button = QPushButton("후보 스캔 + 최적화")
        self.refresh_balance_button = QPushButton("잔고 새로고침")
        self.save_settings_button.clicked.connect(self.save_settings_with_feedback)
        self.scan_button.clicked.connect(self.run_scan_and_optimize)
        self.refresh_balance_button.clicked.connect(self.refresh_account_info)
        actions_row.addWidget(self.save_settings_button)
        actions_row.addWidget(self.scan_button)
        actions_row.addWidget(self.refresh_balance_button)
        left_layout.addLayout(actions_row)

        lower_split = QSplitter(VERTICAL)
        lower_split.addWidget(self._build_candidate_group())
        lower_split.addWidget(self._build_optimized_group())
        lower_split.addWidget(self._build_log_group())
        lower_split.setSizes([240, 240, 160])
        left_layout.addWidget(lower_split, 4)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        header_group = QGroupBox("Selection")
        header_layout = QVBoxLayout(header_group)
        self.symbol_label = QLabel("종목: -")
        self.signal_label = QLabel("신호: -")
        self.balance_label = QLabel("잔고: API 미입력")
        self.position_label = QLabel("포지션: -")
        header_layout.addWidget(self.symbol_label)
        header_layout.addWidget(self.signal_label)
        header_layout.addWidget(self.balance_label)
        header_layout.addWidget(self.position_label)
        right_layout.addWidget(header_group)

        self.chart_host = QWidget()
        self.chart_host_layout = QVBoxLayout(self.chart_host)
        self.chart_host_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.chart_host, 7)

        summary_group = QGroupBox("Backtest Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        summary_layout.addWidget(self.summary_box)
        right_layout.addWidget(summary_group, 2)

        order_group = QGroupBox("Live Order")
        order_layout = QVBoxLayout(order_group)
        long_row = QHBoxLayout()
        short_row = QHBoxLayout()
        self.long_buttons = []
        self.short_buttons = []
        for fraction, text in ((0.33, "LONG 33%"), (0.50, "LONG 50%"), (0.99, "LONG 99%")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, value=fraction: self.place_fractional_order("BUY", value))
            self.long_buttons.append(button)
            long_row.addWidget(button)
        for fraction, text in ((0.33, "SHORT 33%"), (0.50, "SHORT 50%"), (0.99, "SHORT 99%")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, value=fraction: self.place_fractional_order("SELL", value))
            self.short_buttons.append(button)
            short_row.addWidget(button)
        self.close_position_button = QPushButton("포지션 청산")
        self.close_position_button.clicked.connect(self.close_selected_position)
        order_layout.addLayout(long_row)
        order_layout.addLayout(short_row)
        order_layout.addWidget(self.close_position_button)
        right_layout.addWidget(order_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([560, 1120])

    def _build_api_group(self) -> QGroupBox:
        group = QGroupBox("Binance API")
        layout = QFormLayout(group)
        self.api_key_edit = QLineEdit()
        self.api_secret_edit = QLineEdit()
        self.api_secret_edit.setEchoMode(PASSWORD_ECHO)
        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(1, 125)
        layout.addRow("API Key", self.api_key_edit)
        layout.addRow("Secret", self.api_secret_edit)
        layout.addRow("Leverage", self.leverage_spin)
        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Market Filters")
        layout = QFormLayout(group)
        self.daily_vol_spin = QDoubleSpinBox()
        self.daily_vol_spin.setRange(0.0, 500.0)
        self.daily_vol_spin.setDecimals(2)
        self.daily_vol_spin.setSingleStep(1.0)
        self.quote_volume_spin = QDoubleSpinBox()
        self.quote_volume_spin.setRange(0.0, 10_000_000_000.0)
        self.quote_volume_spin.setDecimals(0)
        self.quote_volume_spin.setSingleStep(100_000.0)
        self.rsi_length_spin = QSpinBox()
        self.rsi_length_spin.setRange(2, 100)
        self.rsi_lower_spin = QDoubleSpinBox()
        self.rsi_lower_spin.setRange(0.0, 100.0)
        self.rsi_lower_spin.setDecimals(1)
        self.rsi_upper_spin = QDoubleSpinBox()
        self.rsi_upper_spin.setRange(0.0, 100.0)
        self.rsi_upper_spin.setDecimals(1)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(APP_INTERVAL_OPTIONS)
        self.chart_engine_combo = QComboBox()
        self.chart_engine_combo.addItems(CHART_ENGINE_OPTIONS)
        self.history_days_spin = QSpinBox()
        self.history_days_spin.setRange(1, 30)
        self.scan_workers_spin = QSpinBox()
        self.scan_workers_spin.setRange(1, 24)
        layout.addRow("1일 변동성 % >=", self.daily_vol_spin)
        layout.addRow("24h 거래량 >=", self.quote_volume_spin)
        layout.addRow("1m RSI Length", self.rsi_length_spin)
        layout.addRow("1m RSI Lower <=", self.rsi_lower_spin)
        layout.addRow("1m RSI Upper >=", self.rsi_upper_spin)
        layout.addRow("백테스트 봉", self.interval_combo)
        layout.addRow("차트 엔진", self.chart_engine_combo)
        layout.addRow("히스토리 일수", self.history_days_spin)
        layout.addRow("스캔 워커", self.scan_workers_spin)
        return group

    def _build_optimization_group(self) -> QGroupBox:
        group = QGroupBox("Optimization")
        layout = QFormLayout(group)
        self.opt_span_spin = QDoubleSpinBox()
        self.opt_span_spin.setRange(1.0, 100.0)
        self.opt_span_spin.setDecimals(1)
        self.opt_span_spin.setSingleStep(1.0)
        self.opt_steps_spin = QSpinBox()
        self.opt_steps_spin.setRange(1, 9)
        self.max_combo_spin = QSpinBox()
        self.max_combo_spin.setRange(10, 20_000)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 5.0)
        self.fee_spin.setDecimals(4)
        self.fee_spin.setSingleStep(0.01)
        layout.addRow("범위 ±%", self.opt_span_spin)
        layout.addRow("격자 단계수", self.opt_steps_spin)
        layout.addRow("최대 조합수", self.max_combo_spin)
        layout.addRow("수수료 %", self.fee_spin)
        return group

    def _build_parameter_tabs(self) -> QGroupBox:
        group = QGroupBox("Strategy Parameters")
        outer_layout = QVBoxLayout(group)
        help_label = QLabel("입력값이 기준 전략값이며, `Opt`를 체크한 항목만 현재 값 기준으로 ±최적화 범위를 탐색합니다.")
        help_label.setWordWrap(True)
        outer_layout.addWidget(help_label)
        tabs = QTabWidget()
        outer_layout.addWidget(tabs)

        forms: Dict[str, QFormLayout] = {}
        for key, title in (("core", "Core"), ("qip", "QIP"), ("qtp", "QTP"), ("switches", "Switches")):
            page = QWidget()
            form = QFormLayout(page)
            forms[key] = form
            tabs.addTab(page, title)

        for spec in PARAMETER_SPECS:
            editor = self._make_parameter_editor(spec)
            optimize_box = QCheckBox("Opt")
            optimize_box.setChecked(bool(self.settings.optimize_flags.get(spec.key, False)))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor)
            row_layout.addWidget(optimize_box)
            forms[spec.group].addRow(spec.label, row_widget)
            self.parameter_editors[spec.key] = editor
            self.parameter_opt_boxes[spec.key] = optimize_box
        return group

    def _make_parameter_editor(self, spec):
        current = getattr(self.settings.strategy, spec.key)
        if spec.kind == "bool":
            editor = QCheckBox()
            editor.setChecked(bool(current))
            return editor
        if spec.kind == "choice":
            editor = QComboBox()
            editor.addItems(list(spec.choices))
            if current in spec.choices:
                editor.setCurrentText(current)
            return editor
        if spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.minimum), int(spec.maximum))
            editor.setSingleStep(int(spec.step or 1))
            editor.setValue(int(current))
            return editor
        editor = QDoubleSpinBox()
        editor.setRange(float(spec.minimum), float(spec.maximum))
        editor.setSingleStep(float(spec.step or 0.1))
        editor.setDecimals(4 if float(spec.step or 0.1) < 0.1 else 2)
        editor.setValue(float(current))
        return editor

    def _build_candidate_group(self) -> QGroupBox:
        group = QGroupBox("후보 종목")
        layout = QVBoxLayout(group)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(["Symbol", "DayVol%", "24h Vol", "RSI1m", "24h%", "Price"])
        self.candidate_table.setSelectionBehavior(SELECT_ROWS)
        self.candidate_table.setSelectionMode(SINGLE_SELECTION)
        self.candidate_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.itemSelectionChanged.connect(self.on_candidate_selection_changed)
        self.candidate_table.cellClicked.connect(self.on_candidate_cell_clicked)
        layout.addWidget(self.candidate_table)
        return group

    def _build_optimized_group(self) -> QGroupBox:
        group = QGroupBox("최적화 종목")
        layout = QVBoxLayout(group)
        self.optimized_table = QTableWidget(0, 7)
        self.optimized_table.setHorizontalHeaderLabels(["Symbol", "Return%", "MDD%", "Trades", "Win%", "PF", "Grid"])
        self.optimized_table.setSelectionBehavior(SELECT_ROWS)
        self.optimized_table.setSelectionMode(SINGLE_SELECTION)
        self.optimized_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.optimized_table.horizontalHeader().setStretchLastSection(True)
        self.optimized_table.itemSelectionChanged.connect(self.on_optimized_selection_changed)
        self.optimized_table.cellClicked.connect(self.on_optimized_cell_clicked)
        layout.addWidget(self.optimized_table)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("로그")
        layout = QVBoxLayout(group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return group

    def _init_chart(self) -> None:
        self._rebuild_chart_engine(force=True)

    def _clear_chart_host(self) -> None:
        while self.chart_host_layout.count():
            item = self.chart_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.chart = None
        self.chart_view = None
        self.equity_subchart = None
        self.equity_line = None
        self.supertrend_line = None
        self.zone2_line = None
        self.zone3_line = None
        self.ema_fast_line = None
        self.ema_slow_line = None

    def _rebuild_chart_engine(self, force: bool = False) -> None:
        engine = self.chart_engine_combo.currentText() if hasattr(self, "chart_engine_combo") else self.settings.chart_engine
        if not force and engine == self.chart_mode:
            return
        self._clear_chart_host()
        if engine == "Lightweight":
            self._init_lightweight_chart()
        else:
            self._init_plotly_chart()
        self.chart_mode = engine
        if self.current_symbol and self.current_backtest:
            self.render_chart(self.current_symbol, self.current_backtest)

    def _init_plotly_chart(self) -> None:
        self.chart_view = QWebEngineView(self.chart_host)
        self.chart_view.settings().setAttribute(WEB_ATTR_FILE_URLS, True)
        self.chart_host_layout.addWidget(self.chart_view)
        self.chart_view.setHtml(self._empty_chart_html())

    def _init_lightweight_chart(self) -> None:
        self.chart = QtChart(self.chart_host)
        self.chart_host_layout.addWidget(self.chart.get_webview())
        self.chart.layout(background_color="#0f1419", text_color="#eceff4", font_size=12, font_family="Consolas")
        self.chart.legend(True)
        self.chart.candle_style(
            up_color="#d8f3dc",
            down_color="rgba(0, 0, 0, 0)",
            border_up_color="#d8f3dc",
            border_down_color="#ff5d73",
            wick_up_color="#d8f3dc",
            wick_down_color="#ff5d73",
        )
        self.chart.volume_config(up_color="rgba(216, 243, 220, 0.30)", down_color="rgba(255, 93, 115, 0.28)")
        self.chart.watermark("ALT MR", color="rgba(240, 242, 245, 0.16)")
        self.chart.crosshair(mode="normal")
        self.chart.time_scale(time_visible=True, seconds_visible=False, min_bar_spacing=0.01, right_offset=0)
        self.equity_subchart = self.chart.create_subchart(
            position="bottom",
            width=1.0,
            height=0.28,
            sync=True,
            scale_candles_only=True,
        )
        self.equity_subchart.layout(background_color="#121922", text_color="#eceff4", font_size=11, font_family="Consolas")
        self.equity_subchart.legend(True)
        self.equity_subchart.time_scale(time_visible=True, seconds_visible=False, min_bar_spacing=0.01, right_offset=0)
        self.equity_line = self.equity_subchart.create_line(
            "Equity",
            color="rgba(108, 245, 160, 0.9)",
            width=2,
            price_line=False,
            price_label=False,
        )
        self.supertrend_line = self.chart.create_line(
            "Supertrend",
            color="rgba(255, 225, 120, 0.95)",
            width=2,
            price_line=False,
            price_label=False,
        )
        self.zone2_line = self.chart.create_line(
            "Zone 2",
            color="rgba(255, 186, 73, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.zone3_line = self.chart.create_line(
            "Zone 3",
            color="rgba(255, 123, 123, 0.72)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.ema_fast_line = self.chart.create_line(
            "EMA Fast",
            color="rgba(153, 229, 255, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
        )
        self.ema_slow_line = self.chart.create_line(
            "EMA Slow",
            color="rgba(255, 243, 176, 0.82)",
            width=1,
            price_line=False,
            price_label=False,
        )

    def on_chart_engine_changed(self, _engine: str) -> None:
        self.settings.chart_engine = self.chart_engine_combo.currentText()
        self._rebuild_chart_engine()

    def _apply_loaded_settings(self) -> None:
        settings = self.settings
        self.api_key_edit.setText(settings.api_key)
        self.api_secret_edit.setText(settings.api_secret)
        self.leverage_spin.setValue(settings.leverage)
        self.daily_vol_spin.setValue(settings.daily_volatility_min)
        self.quote_volume_spin.setValue(settings.quote_volume_min)
        self.rsi_length_spin.setValue(settings.rsi_length)
        self.rsi_lower_spin.setValue(settings.rsi_lower)
        self.rsi_upper_spin.setValue(settings.rsi_upper)
        self.interval_combo.setCurrentText(settings.kline_interval)
        self.chart_engine_combo.setCurrentText(settings.chart_engine)
        self.history_days_spin.setValue(settings.history_days)
        self.scan_workers_spin.setValue(settings.scan_workers)
        self.opt_span_spin.setValue(settings.optimization_span_pct)
        self.opt_steps_spin.setValue(settings.optimization_steps)
        self.max_combo_spin.setValue(settings.max_grid_combinations)
        self.fee_spin.setValue(settings.fee_rate * 100.0)

    def collect_settings(self) -> AppSettings:
        strategy_payload: Dict[str, object] = {}
        for spec in PARAMETER_SPECS:
            editor = self.parameter_editors[spec.key]
            if spec.kind == "bool":
                strategy_payload[spec.key] = bool(editor.isChecked())
            elif spec.kind == "choice":
                strategy_payload[spec.key] = editor.currentText()
            elif spec.kind == "int":
                strategy_payload[spec.key] = int(editor.value())
            else:
                strategy_payload[spec.key] = float(editor.value())

        optimize_flags = {key: box.isChecked() for key, box in self.parameter_opt_boxes.items()}
        return AppSettings(
            api_key=self.api_key_edit.text().strip(),
            api_secret=self.api_secret_edit.text().strip(),
            chart_engine=self.chart_engine_combo.currentText(),
            leverage=int(self.leverage_spin.value()),
            fee_rate=float(self.fee_spin.value()) / 100.0,
            history_days=int(self.history_days_spin.value()),
            kline_interval=self.interval_combo.currentText(),
            daily_volatility_min=float(self.daily_vol_spin.value()),
            quote_volume_min=float(self.quote_volume_spin.value()),
            rsi_length=int(self.rsi_length_spin.value()),
            rsi_lower=float(self.rsi_lower_spin.value()),
            rsi_upper=float(self.rsi_upper_spin.value()),
            optimization_span_pct=float(self.opt_span_spin.value()),
            optimization_steps=int(self.opt_steps_spin.value()),
            max_grid_combinations=int(self.max_combo_spin.value()),
            scan_workers=int(self.scan_workers_spin.value()),
            strategy=StrategySettings(**strategy_payload),
            optimize_flags=optimize_flags,
        )

    def save_settings(self) -> AppSettings:
        previous = self.settings
        self.settings = self.collect_settings()
        if (
            previous.kline_interval != self.settings.kline_interval
            or previous.history_days != self.settings.history_days
        ):
            self._stop_live_stream()
            self.history_cache.clear()
            self.chart_history_cache.clear()
        self.settings.save()
        return self.settings

    def save_settings_with_feedback(self) -> None:
        self.save_settings()
        self.log("설정을 저장했습니다. 다음 실행 때 같은 값으로 불러옵니다.")

    def log(self, message: str) -> None:
        self.log_box.appendPlainText(message)
        self.statusBar().showMessage(message, 5000)

    def _candidate_by_symbol(self, symbol: str) -> Optional[CandidateSymbol]:
        return next((candidate for candidate in self.candidates if candidate.symbol == symbol), None)

    def _resolve_price_precision(self, symbol: str, candle_df: pd.DataFrame) -> int:
        precision = 2
        try:
            filters = BinanceFuturesClient().get_symbol_filters(symbol)
            precision = int(filters.get("pricePrecision", precision))
            tick_size = float(filters.get("tickSize", 0.0) or 0.0)
            if tick_size > 0:
                tick_text = f"{tick_size:.16f}".rstrip("0").rstrip(".")
                if "." in tick_text:
                    precision = max(precision, len(tick_text.split(".", 1)[1]))
        except Exception:
            close_series = candle_df["close"].dropna()
            if not close_series.empty:
                price = abs(float(close_series.iloc[-1]))
                if price < 0.001:
                    precision = 8
                elif price < 0.01:
                    precision = 7
                elif price < 0.1:
                    precision = 6
                elif price < 1:
                    precision = 5
                else:
                    precision = 2
        return max(2, min(precision, 10))

    def _apply_lightweight_precision(self, symbol: str, candle_df: pd.DataFrame) -> None:
        if self.chart is None:
            return
        precision = self._resolve_price_precision(symbol, candle_df)
        self.chart.precision(precision)
        for line in (
            self.supertrend_line,
            self.zone2_line,
            self.zone3_line,
            self.ema_fast_line,
            self.ema_slow_line,
        ):
            if line is not None:
                line.precision(precision)

    def _default_chart_time_range(self, candle_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        if candle_df.empty:
            now = pd.Timestamp.utcnow().tz_localize(None)
            return now - pd.Timedelta(hours=DEFAULT_CHART_LOOKBACK_HOURS), now
        end_time = pd.Timestamp(candle_df["time"].iloc[-1])
        start_floor = pd.Timestamp(candle_df["time"].iloc[0])
        start_time = max(start_floor, end_time - pd.Timedelta(hours=DEFAULT_CHART_LOOKBACK_HOURS))
        return start_time, end_time

    def _active_backtest_settings(self, symbol: str) -> StrategySettings:
        optimization = self.optimized_results.get(symbol)
        return optimization.best_backtest.settings if optimization else self.settings.strategy

    def _stop_live_stream(self) -> None:
        worker = self.live_stream_worker
        self.live_stream_worker = None
        self.live_pending_bar = None
        if self.live_update_timer.isActive():
            self.live_update_timer.stop()
        if worker is not None:
            worker.stop()
            worker.wait(1500)

    def _start_live_stream(self, symbol: str) -> None:
        self._stop_live_stream()
        worker = KlineStreamWorker(symbol, self.settings.kline_interval)
        worker.kline.connect(self._queue_live_update)
        worker.status.connect(self._on_live_stream_status)
        self.live_stream_worker = worker
        worker.start()

    def _on_live_stream_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 3000)

    def _queue_live_update(self, payload: object) -> None:
        bar = dict(payload)
        if bar.get("symbol") != self.current_symbol:
            return
        self.live_pending_bar = bar
        self.live_update_timer.setInterval(LIVE_RENDER_INTERVAL_MS if self.chart_mode == "Lightweight" else 1000)
        if not self.live_update_timer.isActive():
            self.live_update_timer.start()

    def _flush_live_update(self) -> None:
        bar = self.live_pending_bar
        self.live_pending_bar = None
        if not bar or not self.current_symbol:
            return
        symbol = str(bar["symbol"])
        history = self.history_cache.get(symbol)
        chart_history = self.chart_history_cache.get(symbol)
        if history is None or chart_history is None:
            return

        self.history_cache[symbol] = _merge_live_bar(history, bar)
        self.chart_history_cache[symbol] = _merge_live_bar(chart_history, bar, max_rows=CHART_HISTORY_BAR_LIMIT)

        backtest = run_backtest(
            self.history_cache[symbol],
            settings=self._active_backtest_settings(symbol),
            fee_rate=self.settings.fee_rate,
            backtest_start_time=pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms"),
        )
        self.current_backtest = backtest
        self.render_chart(symbol, backtest, reset_view=False)
        self.update_summary(symbol, backtest, self.optimized_results.get(symbol))

    def update_candidate_table(self) -> None:
        self.candidate_table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            values = [
                candidate.symbol,
                f"{candidate.daily_volatility_pct:.2f}",
                f"{candidate.quote_volume:,.0f}",
                f"{candidate.rsi_1m:.2f}",
                f"{candidate.price_change_pct:.2f}",
                f"{candidate.last_price:.6f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, candidate.symbol)
                self.candidate_table.setItem(row, col, item)

    def update_optimized_table(self) -> None:
        ordered = sorted(
            self.optimized_results.values(),
            key=lambda result: result.best_backtest.metrics.total_return_pct,
            reverse=True,
        )
        self.optimized_table.setRowCount(len(ordered))
        for row, result in enumerate(ordered):
            metrics = result.best_backtest.metrics
            values = [
                result.symbol,
                f"{metrics.total_return_pct:.2f}",
                f"{metrics.max_drawdown_pct:.2f}",
                str(metrics.trade_count),
                f"{metrics.win_rate_pct:.1f}",
                f"{metrics.profit_factor:.2f}",
                str(result.combinations_tested),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, result.symbol)
                self.optimized_table.setItem(row, col, item)

    def _init_auto_refresh(self) -> None:
        self.auto_refresh_timer.setInterval(self.auto_refresh_minutes * 60 * 1000)
        self.auto_refresh_timer.timeout.connect(self.run_auto_refresh)
        self.auto_refresh_timer.start()
        self.log(f"자동 갱신 활성화: {self.auto_refresh_minutes}분마다 스캔+최적화")

    def _set_refresh_running(self, is_running: bool) -> None:
        self.scan_button.setEnabled(not is_running)
        self.refresh_balance_button.setEnabled(not is_running)

    def _is_refresh_running(self) -> bool:
        return bool(
            (self.scan_worker and self.scan_worker.isRunning())
            or (self.optimize_worker and self.optimize_worker.isRunning())
        )

    def run_scan_and_optimize(self, preserve_existing: bool = False) -> None:
        if self._is_refresh_running():
            return
        self.save_settings()
        self.preserve_lists_during_refresh = preserve_existing
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        if not preserve_existing:
            self._stop_live_stream()
            self.candidates = []
            self.optimized_results.clear()
            self.history_cache.clear()
            self.chart_history_cache.clear()
            self.current_symbol = None
            self.current_backtest = None
            self.update_candidate_table()
            self.update_optimized_table()
            self.summary_box.clear()
        self.log("후보 스캔 + 최적화 시작" + (" (기존 목록 유지)" if preserve_existing else ""))
        self._set_refresh_running(True)
        self.scan_worker = ScanWorker(self.settings)
        self.scan_worker.progress.connect(self.log)
        self.scan_worker.completed.connect(self.on_scan_completed)
        self.scan_worker.failed.connect(self.on_worker_failed)
        self.scan_worker.start()

    def on_scan_completed(self, candidates: object) -> None:
        self.pending_candidates = list(candidates)
        if not self.preserve_lists_during_refresh:
            self.candidates = list(self.pending_candidates)
            self.update_candidate_table()
        self.log(f"후보 스캔 완료: {len(self.pending_candidates)}개")
        if self.pending_candidates:
            if not self.preserve_lists_during_refresh:
                self.candidate_table.selectRow(0)
            self.start_optimization(self.pending_candidates)
            return
        if self.preserve_lists_during_refresh:
            self.log("새 후보가 없어 기존 목록을 유지합니다.")
        else:
            self.log("후보가 없어 최적화를 건너뜁니다.")
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.preserve_lists_during_refresh = False
        self._set_refresh_running(False)

    def start_optimization(self, targets: List[CandidateSymbol]) -> None:
        if self.optimize_worker and self.optimize_worker.isRunning():
            return
        if not targets:
            self._set_refresh_running(False)
            return
        self.log(f"최적화 시작: {len(targets)}개 종목")
        self.optimize_worker = OptimizeWorker(self.settings, targets)
        self.optimize_worker.progress.connect(self.log)
        self.optimize_worker.result_ready.connect(self.on_optimization_result)
        self.optimize_worker.completed.connect(self.on_optimization_completed)
        self.optimize_worker.failed.connect(self.on_worker_failed)
        self.optimize_worker.start()

    def on_optimization_result(self, payload: object) -> None:
        result = dict(payload)
        candidate: CandidateSymbol = result["candidate"]
        optimization: OptimizationResult = result["optimization"]
        history: pd.DataFrame = result["history"]
        if self.preserve_lists_during_refresh:
            self.pending_optimized_results[candidate.symbol] = optimization
            self.pending_history_cache[candidate.symbol] = history
            return
        self.optimized_results[candidate.symbol] = optimization
        self.history_cache[candidate.symbol] = history
        self.update_optimized_table()

    def on_optimization_completed(self) -> None:
        preserved_refresh = self.preserve_lists_during_refresh
        if preserved_refresh:
            if self.pending_candidates:
                self.candidates = list(self.pending_candidates)
                self.update_candidate_table()
            if self.pending_optimized_results:
                self.optimized_results = dict(self.pending_optimized_results)
                self.update_optimized_table()
            self.history_cache.update(self.pending_history_cache)
            self.pending_candidates = []
            self.pending_optimized_results = {}
            self.pending_history_cache = {}
            self.preserve_lists_during_refresh = False
        self.log(f"최적화 완료: {len(self.optimized_results)}개")
        if not preserved_refresh and self.optimized_table.rowCount() > 0:
            self.optimized_table.selectRow(0)
        self._set_refresh_running(False)

    def on_worker_failed(self, message: str) -> None:
        self.pending_candidates = []
        self.pending_optimized_results = {}
        self.pending_history_cache = {}
        self.preserve_lists_during_refresh = False
        self.log(message)
        self._set_refresh_running(False)
        self.show_error(message)

    def run_auto_refresh(self) -> None:
        if self._is_refresh_running():
            self.log("자동 10분 갱신 시점이지만 이전 작업이 아직 실행 중이라 건너뜁니다.")
            return
        self.log("자동 10분 갱신 시작")
        self.run_scan_and_optimize(preserve_existing=True)

    def selected_candidate_symbols(self) -> List[str]:
        selected = self.candidate_table.selectedItems()
        if not selected:
            return []
        row = selected[0].row()
        symbol_item = self.candidate_table.item(row, 0)
        return [symbol_item.text()] if symbol_item else []

    def _request_symbol_load(self, symbol: str) -> None:
        if not symbol:
            return
        if symbol == self.current_symbol and self.current_backtest is not None:
            return
        self.load_symbol(symbol)

    def on_candidate_selection_changed(self) -> None:
        selected = self.candidate_table.selectedItems()
        if selected:
            self._request_symbol_load(selected[0].data(USER_ROLE) or selected[0].text())

    def on_candidate_cell_clicked(self, row: int, _column: int) -> None:
        item = self.candidate_table.item(row, 0)
        if item:
            self._request_symbol_load(item.data(USER_ROLE) or item.text())

    def on_optimized_selection_changed(self) -> None:
        selected = self.optimized_table.selectedItems()
        if selected:
            self._request_symbol_load(selected[0].data(USER_ROLE) or selected[0].text())

    def on_optimized_cell_clicked(self, row: int, _column: int) -> None:
        item = self.optimized_table.item(row, 0)
        if item:
            self._request_symbol_load(item.data(USER_ROLE) or item.text())

    def load_symbol(self, symbol: str) -> None:
        try:
            self._stop_live_stream()
            self.current_symbol = symbol
            self.save_settings()
            backtest_start_time = pd.to_datetime(_backtest_start_time_ms(self.settings), unit="ms")
            history = self.history_cache.get(symbol)
            if history is None:
                client = BinanceFuturesClient()
                history = client.historical_ohlcv(
                    symbol,
                    self.settings.kline_interval,
                    start_time=_history_fetch_start_time_ms(self.settings),
                )
                self.history_cache[symbol] = history
            if history.empty:
                self.show_warning(f"{symbol} 히스토리 데이터가 없습니다.")
                return

            chart_history = self.chart_history_cache.get(symbol)
            if chart_history is None:
                client = BinanceFuturesClient()
                chart_history = client.historical_ohlcv_recent(
                    symbol,
                    self.settings.kline_interval,
                    bars=CHART_HISTORY_BAR_LIMIT,
                )
                if chart_history.empty:
                    chart_history = history.copy()
                else:
                    chart_history = (
                        pd.concat([chart_history, history], ignore_index=True)
                        .drop_duplicates(subset=["time"])
                        .sort_values("time")
                        .reset_index(drop=True)
                    )
                self.chart_history_cache[symbol] = chart_history

            optimization = self.optimized_results.get(symbol)
            backtest = (
                optimization.best_backtest
                if optimization
                else run_backtest(
                    history,
                    settings=self.settings.strategy,
                    fee_rate=self.settings.fee_rate,
                    backtest_start_time=backtest_start_time,
                )
            )
            self.current_backtest = backtest
            self.render_chart(symbol, backtest)
            self.update_summary(symbol, backtest, optimization)
            self.refresh_account_info()
            self._start_live_stream(symbol)
        except Exception as exc:
            self.show_error(str(exc))

    def render_chart(self, symbol: str, result: BacktestResult, reset_view: bool = True) -> None:
        chart_history = self.chart_history_cache.get(symbol)
        if chart_history is not None and not chart_history.empty:
            chart_indicators = compute_indicators(chart_history, result.settings)
        else:
            chart_indicators = result.indicators
        indicators = chart_indicators.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        candle_df = indicators[["time", "open", "high", "low", "close", "volume"]].copy()
        equity_df = (
            pd.DataFrame({"time": list(result.equity_curve.index), "Equity": list(result.equity_curve.values)})
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )
        if self.chart_mode == "Lightweight":
            self._render_lightweight_chart(symbol, candle_df, indicators, equity_df, result.trades, reset_view=reset_view)
        else:
            self._render_plotly_chart(symbol, candle_df, indicators, equity_df, result.trades, reset_view=reset_view)

        candidate = self._candidate_by_symbol(symbol)
        latest = result.latest_state
        self.symbol_label.setText(
            f"종목: {symbol}"
            + (f" | DayVol {candidate.daily_volatility_pct:.2f}% | RSI1m {candidate.rsi_1m:.2f}" if candidate else "")
        )
        self.signal_label.setText(
            f"신호: Trend {latest['trend']} | Zone {latest['zone']} | "
            f"Bull {latest['final_bull']} | Bear {latest['final_bear']} | RSI {latest['rsi']:.2f}"
        )

    def _render_plotly_chart(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        reset_view: bool = True,
    ) -> None:
        if self.chart_view is None:
            self._rebuild_chart_engine(force=True)
        if not self.plotly_js_path.exists() or self.plotly_js_path.stat().st_size == 0:
            self.plotly_js_path.write_text(get_plotlyjs(), encoding="utf-8")
        range_start, range_end = self._default_chart_time_range(candle_df) if reset_view else (None, None)
        html = self._plotly_chart_html(symbol, candle_df, indicators, equity_df, trades, range_start, range_end)
        self.plotly_chart_path.write_text(html, encoding="utf-8")
        chart_url = QUrl.fromLocalFile(str(self.plotly_chart_path))
        chart_url.setQuery(f"ts={self.plotly_chart_path.stat().st_mtime_ns}")
        self.chart_view.load(chart_url)

    def _render_lightweight_chart(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        reset_view: bool = True,
    ) -> None:
        if self.chart is None:
            self._rebuild_chart_engine(force=True)
        self.chart.set(candle_df)
        self._apply_lightweight_precision(symbol, candle_df)
        self.chart.clear_markers()
        self.supertrend_line.set(indicators[["time", "supertrend"]].rename(columns={"supertrend": "Supertrend"}))
        self.zone2_line.set(indicators[["time", "zone2_line"]].rename(columns={"zone2_line": "Zone 2"}))
        self.zone3_line.set(indicators[["time", "zone3_line"]].rename(columns={"zone3_line": "Zone 3"}))
        self.ema_fast_line.set(indicators[["time", "ema_fast"]].rename(columns={"ema_fast": "EMA Fast"}))
        self.ema_slow_line.set(indicators[["time", "ema_slow"]].rename(columns={"ema_slow": "EMA Slow"}))
        self.equity_line.set(equity_df)
        range_start, range_end = self._default_chart_time_range(candle_df)

        markers = []
        for trade in trades:
            markers.append(
                {
                    "time": trade.entry_time,
                    "position": "below" if trade.side == "long" else "above",
                    "shape": "arrow_up" if trade.side == "long" else "arrow_down",
                    "color": "#17c964" if trade.side == "long" else "#f31260",
                    "text": trade.zones,
                }
            )
            markers.append(
                {
                    "time": trade.exit_time,
                    "position": "above" if trade.side == "long" else "below",
                    "shape": "circle",
                    "color": "#94a3b8",
                    "text": f"{trade.return_pct:+.1f}%",
                }
            )
        if markers:
            self.chart.marker_list(markers)
        if reset_view:
            QTimer.singleShot(
                140,
                lambda s=symbol, st=range_start, et=range_end: self._sync_lightweight_range(s, st, et),
            )

    def _sync_lightweight_range(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        if symbol != self.current_symbol or self.chart is None or self.equity_subchart is None:
            return
        try:
            self.chart.set_visible_range(start_time, end_time)
            self.equity_subchart.set_visible_range(start_time, end_time)
        except Exception:
            self.chart.fit()
            self.equity_subchart.fit()

    def _empty_chart_html(self) -> str:
        return """
        <html>
          <head>
            <style>
              html, body {
                margin: 0;
                height: 100%;
                background: #0f1419;
                color: #dfe6eb;
                font-family: Consolas, monospace;
              }
              .wrap {
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 0.7;
                font-size: 15px;
              }
            </style>
          </head>
          <body>
            <div class="wrap">차트 대기 중</div>
          </body>
        </html>
        """

    def _plotly_chart_html(
        self,
        symbol: str,
        candle_df: pd.DataFrame,
        indicators: pd.DataFrame,
        equity_df: pd.DataFrame,
        trades,
        range_start: Optional[pd.Timestamp],
        range_end: Optional[pd.Timestamp],
    ) -> str:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            specs=[[{"secondary_y": True}], [{}]],
        )

        volume_colors = [
            "rgba(23, 201, 100, 0.35)" if close_ >= open_ else "rgba(243, 18, 96, 0.35)"
            for open_, close_ in zip(candle_df["open"], candle_df["close"])
        ]

        fig.add_trace(
            go.Candlestick(
                x=candle_df["time"],
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
                name=symbol,
                increasing_line_color="#17c964",
                increasing_fillcolor="#17c964",
                decreasing_line_color="#f31260",
                decreasing_fillcolor="#f31260",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=candle_df["time"],
                y=candle_df["volume"],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.35,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        for column, name, color, width in (
            ("supertrend", "Supertrend", "#ffcc00", 2),
            ("zone2_line", "Zone 2", "#ff9100", 1),
            ("zone3_line", "Zone 3", "#ff1744", 1),
            ("ema_fast", "EMA Fast", "#00e5ff", 1),
            ("ema_slow", "EMA Slow", "#ffd600", 1),
        ):
            fig.add_trace(
                go.Scatter(
                    x=indicators["time"],
                    y=indicators[column],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": width},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        long_entries = [trade for trade in trades if trade.side == "long"]
        short_entries = [trade for trade in trades if trade.side == "short"]
        if long_entries:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time for trade in long_entries],
                    y=[trade.entry_price for trade in long_entries],
                    mode="markers+text",
                    name="Long Entry",
                    text=[trade.zones for trade in long_entries],
                    textposition="bottom center",
                    marker={"symbol": "triangle-up", "size": 11, "color": "#17c964"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if short_entries:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time for trade in short_entries],
                    y=[trade.entry_price for trade in short_entries],
                    mode="markers+text",
                    name="Short Entry",
                    text=[trade.zones for trade in short_entries],
                    textposition="top center",
                    marker={"symbol": "triangle-down", "size": 11, "color": "#f31260"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time for trade in trades],
                    y=[trade.exit_price for trade in trades],
                    mode="markers+text",
                    name="Exit",
                    text=[f"{trade.return_pct:+.1f}%" for trade in trades],
                    textposition="middle right",
                    marker={"symbol": "x", "size": 9, "color": "#94a3b8"},
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=equity_df["time"],
                y=equity_df["Equity"],
                mode="lines",
                name="Equity",
                line={"color": "#6cf5a0", "width": 2},
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            paper_bgcolor="#0f1419",
            plot_bgcolor="#0f1419",
            font={"family": "Consolas, monospace", "color": "#dfe6eb", "size": 12},
            height=max(self.chart_host.height(), 760),
            autosize=True,
            margin={"l": 48, "r": 36, "t": 36, "b": 36},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0.0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        xaxis_options = {
            "showgrid": True,
            "gridcolor": "rgba(120, 130, 150, 0.14)",
            "zeroline": False,
        }
        if range_start is not None and range_end is not None:
            xaxis_options["range"] = [range_start, range_end]
        fig.update_xaxes(**xaxis_options)
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(120, 130, 150, 0.14)",
            zeroline=False,
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(120, 130, 150, 0.14)",
            zeroline=False,
            row=2,
            col=1,
        )

        config = {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        }
        return pio.to_html(
            fig,
            full_html=True,
            include_plotlyjs="directory",
            config=config,
            default_width="100%",
            default_height=f"{max(self.chart_host.height(), 760)}px",
        )

    def update_summary(self, symbol: str, backtest: BacktestResult, optimization: Optional[OptimizationResult]) -> None:
        metrics = backtest.metrics
        lines = [
            f"Symbol: {symbol}",
            f"Return: {metrics.total_return_pct:.2f}%",
            f"Net Profit: {metrics.net_profit:.2f}",
            f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%",
            f"Trades: {metrics.trade_count}",
            f"Win Rate: {metrics.win_rate_pct:.1f}%",
            f"Profit Factor: {metrics.profit_factor:.2f}",
            "",
            "Latest State:",
        ]
        for key, value in backtest.latest_state.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("Active Params:")
        for spec in PARAMETER_SPECS:
            lines.append(f"- {spec.key}: {getattr(backtest.settings, spec.key)}")
        if optimization:
            lines.append("")
            lines.append(
                f"Optimization: {optimization.combinations_tested} combos"
                + (" (trimmed)" if optimization.trimmed_grid else "")
                + f", {optimization.duration_seconds:.2f}s"
            )
        self.summary_box.setPlainText("\n".join(lines))

    def refresh_account_info(self) -> None:
        self.save_settings()
        if not self.settings.api_key or not self.settings.api_secret:
            self.balance_label.setText("잔고: API 미입력")
            self.position_label.setText("포지션: API 미입력")
            return
        try:
            client = BinanceFuturesClient(self.settings.api_key, self.settings.api_secret)
            balance = client.get_balance_snapshot()
            self.balance_label.setText(
                f"잔고: Equity {balance.equity:.2f} USDT | Available {balance.available_balance:.2f} USDT"
            )
            if self.current_symbol:
                position = client.get_position(self.current_symbol)
                if position is None:
                    self.position_label.setText("포지션: 없음")
                else:
                    side = "LONG" if position.amount > 0 else "SHORT"
                    self.position_label.setText(
                        f"포지션: {side} {position.amount:.6f} @ {position.entry_price:.6f} | "
                        f"UPnL {position.unrealized_pnl:.2f}"
                    )
            else:
                self.position_label.setText("포지션: 종목 미선택")
        except Exception as exc:
            self.balance_label.setText(f"잔고 조회 실패: {exc}")
            self.position_label.setText("포지션: 조회 실패")

    def place_fractional_order(self, side: str, fraction: float) -> None:
        self.save_settings()
        if not self.current_symbol:
            self.show_warning("주문할 종목을 먼저 선택하세요.")
            return
        if not self.settings.api_key or not self.settings.api_secret:
            self.show_warning("API Key / Secret을 입력해야 실제 주문할 수 있습니다.")
            return
        try:
            client = BinanceFuturesClient(self.settings.api_key, self.settings.api_secret)
            balance = client.get_balance_snapshot()
            margin = balance.available_balance * fraction
            client.set_leverage(self.current_symbol, self.settings.leverage)
            quantity = client.build_order_quantity(self.current_symbol, margin, self.settings.leverage)
            result = client.place_market_order(self.current_symbol, side, quantity)
            self.log(f"주문 완료: {self.current_symbol} {side} qty={quantity} orderId={result.get('orderId')}")
            self.refresh_account_info()
        except Exception as exc:
            self.show_error(str(exc))

    def close_selected_position(self) -> None:
        self.save_settings()
        if not self.current_symbol:
            self.show_warning("종목을 먼저 선택하세요.")
            return
        if not self.settings.api_key or not self.settings.api_secret:
            self.show_warning("API Key / Secret을 입력해야 실제 주문할 수 있습니다.")
            return
        try:
            client = BinanceFuturesClient(self.settings.api_key, self.settings.api_secret)
            result = client.close_position(self.current_symbol)
            self.log("청산할 포지션이 없습니다." if result is None else f"포지션 청산 완료: orderId={result.get('orderId')}")
            self.refresh_account_info()
        except Exception as exc:
            self.show_error(str(exc))

    def show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Warning", message)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._stop_live_stream()
            self.save_settings()
        finally:
            super().closeEvent(event)


def create_app() -> QApplication:
    return QApplication.instance() or QApplication([])
