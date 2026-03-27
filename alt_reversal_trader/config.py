from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import pandas as pd


APP_CONFIG_FILENAME = "alt_reversal_trader_config.json"
APP_INTERVAL_OPTIONS = ("1m", "2m", "3m", "5m", "15m")
OPTIMIZATION_RANK_MODE_OPTIONS = ("score", "return")
AUTO_TRADE_FOCUS_SIGNAL_MODE_OPTIONS = ("preview", "confirmed")
DEFAULT_HISTORY_DAYS = 3
DEFAULT_CHART_DISPLAY_HOURS = 24
DEFAULT_OPTIMIZATION_PROFILE_SCALE = 20.0
STRATEGY_TYPE_OPTIONS = ("mean_reversion", "keltner_trend")
STRATEGY_TYPE_LABELS = {
    "mean_reversion": "역추세",
    "keltner_trend": "켈트너채널(추세)",
}
KELTNER_BAND_STYLE_OPTIONS = (
    "Average True Range",
    "True Range",
    "Range",
)
QIP_SENSITIVITY_OPTIONS = (
    "1-Ultra Fine Max",
    "2-Ultra Fine",
    "3-Fine Max",
    "4-Fine",
    "5-Normal",
    "6-Broad Min",
    "7-Broad",
    "8-Broad Max",
    "9-Ultra Broad",
    "10-Ultra Broad Max",
)


def _app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _user_config_dir() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata).resolve() / "AltReversalTrader"
    return Path.home().resolve() / ".alt_reversal_trader"


def default_config_path() -> Path:
    return _user_config_dir() / APP_CONFIG_FILENAME


APP_CONFIG_PATH = default_config_path()


def _load_path_candidates(path: Path) -> List[Path]:
    candidates = [path]
    legacy_candidates = [
        _app_base_dir() / path.name,
        Path.cwd() / path.name,
    ]
    for legacy_path in legacy_candidates:
        if legacy_path not in candidates:
            candidates.append(legacy_path)
    return candidates


def default_optimize_process_count() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count - 1))


@dataclass(frozen=True)
class StrategySettings:
    strategy_type: str = "mean_reversion"
    atr_period: int = 10
    factor: float = 17.0
    zone_sensitivity: float = 8.0
    sensitivity_mode: str = "10-Ultra Broad Max"
    zz_len_raw: int = 5
    atr_mult_raw: float = 1.5
    use_volume: bool = True
    use_rsi_div: bool = True
    use_macd_div: bool = True
    use_ema_conf: bool = True
    min_score: int = 3
    qip_rsi_len: int = 14
    vol_ma_len: int = 20
    qip_ema_fast: int = 21
    qip_ema_slow: int = 55
    qip_use_rsi_zone: bool = True
    qip_rsi_bull_max: int = 40
    qip_rsi_bear_min: int = 60
    qtp_sensitivity: int = 10
    qtp_ema_fast_len: int = 20
    qtp_ema_slow_len: int = 50
    qtp_use_trend: bool = True
    qtp_rsi_len: int = 14
    qtp_stoch_len: int = 14
    qtp_atr_len: int = 14
    qtp_dev_lookback: int = 50
    qtp_vol_len: int = 20
    qtp_min_pvt_left: int = 2
    qtp_max_pvt_left: int = 8
    qtp_use_rsi_zone: bool = True
    qtp_rsi_bull_max: int = 30
    qtp_rsi_bear_min: int = 70
    use_qip: bool = True
    use_qtp: bool = True
    beast_mode: bool = True
    entry_size_pct: float = 10.0
    keltner_length: int = 20
    keltner_multiplier: float = 2.0
    keltner_use_ema: bool = True
    keltner_band_style: str = "Average True Range"
    keltner_atr_length: int = 10

    def __post_init__(self) -> None:
        strategy_type = str(self.strategy_type or STRATEGY_TYPE_OPTIONS[0]).strip()
        if strategy_type not in STRATEGY_TYPE_OPTIONS:
            strategy_type = STRATEGY_TYPE_OPTIONS[0]
        object.__setattr__(self, "strategy_type", strategy_type)
        band_style = str(self.keltner_band_style or KELTNER_BAND_STYLE_OPTIONS[0]).strip()
        if band_style not in KELTNER_BAND_STYLE_OPTIONS:
            band_style = KELTNER_BAND_STYLE_OPTIONS[0]
        object.__setattr__(self, "keltner_band_style", band_style)
        object.__setattr__(self, "keltner_length", max(1, int(self.keltner_length)))
        object.__setattr__(self, "keltner_multiplier", max(0.1, float(self.keltner_multiplier)))
        object.__setattr__(self, "keltner_use_ema", bool(self.keltner_use_ema))
        object.__setattr__(self, "keltner_atr_length", max(1, int(self.keltner_atr_length)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_position_strategy_settings(
    payload: Dict[str, Any] | None,
) -> Dict[str, StrategySettings]:
    normalized: Dict[str, StrategySettings] = {}
    for symbol, raw_settings in dict(payload or {}).items():
        symbol_text = str(symbol or "").strip()
        if not symbol_text:
            continue
        if isinstance(raw_settings, StrategySettings):
            normalized[symbol_text] = raw_settings
            continue
        if not isinstance(raw_settings, dict):
            continue
        normalized[symbol_text] = StrategySettings(
            **{k: v for k, v in raw_settings.items() if k in StrategySettings.__dataclass_fields__}
        )
    return normalized


def _normalize_position_cursor_entry_times(
    payload: Dict[str, Any] | None,
) -> Dict[str, pd.Timestamp]:
    normalized: Dict[str, pd.Timestamp] = {}
    for symbol, raw_time in dict(payload or {}).items():
        symbol_text = str(symbol or "").strip()
        if not symbol_text or raw_time in {None, ""}:
            continue
        try:
            normalized[symbol_text] = pd.Timestamp(raw_time).tz_localize(None)
        except Exception:
            continue
    return normalized


def _normalize_position_open_entry_events(
    payload: Dict[str, Any] | None,
) -> Dict[str, List[Tuple[pd.Timestamp, str]]]:
    normalized: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}
    for symbol, raw_events in dict(payload or {}).items():
        symbol_text = str(symbol or "").strip()
        if not symbol_text:
            continue
        events: List[Tuple[pd.Timestamp, str]] = []
        for raw_event in list(raw_events or []):
            if isinstance(raw_event, dict):
                raw_time = raw_event.get("time")
                raw_label = raw_event.get("label")
            elif isinstance(raw_event, (list, tuple)) and len(raw_event) >= 2:
                raw_time, raw_label = raw_event[0], raw_event[1]
            else:
                continue
            try:
                event_time = pd.Timestamp(raw_time).tz_localize(None)
            except Exception:
                continue
            label_text = str(raw_label or "").strip()
            if not label_text:
                continue
            events.append((event_time, label_text))
        if events:
            deduped = sorted(set(events), key=lambda item: item[0])
            normalized[symbol_text] = deduped
    return normalized


@dataclass
class AppSettings:
    api_key: str = ""
    api_secret: str = ""
    leverage: int = 2
    order_mode: str = "compound"
    simple_order_amount: float = 50.0
    fee_rate: float = 0.0005
    history_days: int = DEFAULT_HISTORY_DAYS
    chart_display_hours: int = DEFAULT_CHART_DISPLAY_HOURS
    auto_refresh_minutes: int = 30
    auto_trade_use_favorable_price: bool = True
    auto_trade_focus_on_signal: bool = True
    auto_trade_focus_signal_mode: str = "preview"
    kline_interval: str = "1m"
    filter_preset: str = "변동성"
    daily_volatility_min: float = 20.0
    quote_volume_min: float = 10_000_000.0
    use_rsi_filter: bool = False
    rsi_length: int = 14
    rsi_lower: float = 40.0
    rsi_upper: float = 60.0
    use_atr_4h_filter: bool = False
    atr_4h_min_pct: float = 10.0
    optimization_span_pct: float = 20.0
    optimization_steps: int = 5
    optimization_rank_mode: str = "score"
    optimization_min_score: float = 70.0
    optimization_min_return_pct: float = 0.0
    max_grid_combinations: int = 300
    enable_parameter_optimization: bool = True
    scan_workers: int = 4
    optimize_processes: int = field(default_factory=default_optimize_process_count)
    optimize_timeframe: bool = True
    strategy: StrategySettings = field(default_factory=StrategySettings)
    optimize_flags: Dict[str, bool] = field(default_factory=dict)
    position_intervals: Dict[str, str] = field(default_factory=dict)
    position_strategy_settings: Dict[str, StrategySettings] = field(default_factory=dict)
    position_filled_fractions: Dict[str, float] = field(default_factory=dict)
    position_cursor_entry_times: Dict[str, pd.Timestamp] = field(default_factory=dict)
    position_open_entry_events: Dict[str, List[Tuple[pd.Timestamp, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.order_mode not in {"compound", "simple"}:
            self.order_mode = "compound"
        self.simple_order_amount = max(1.0, float(self.simple_order_amount))
        self.atr_4h_min_pct = max(0.0, float(self.atr_4h_min_pct))
        self.chart_display_hours = max(1, int(self.chart_display_hours))
        if self.optimization_rank_mode not in OPTIMIZATION_RANK_MODE_OPTIONS:
            self.optimization_rank_mode = OPTIMIZATION_RANK_MODE_OPTIONS[0]
        self.optimization_min_score = max(0.0, float(self.optimization_min_score))
        self.optimization_min_return_pct = float(self.optimization_min_return_pct)
        self.enable_parameter_optimization = bool(self.enable_parameter_optimization)
        self.scan_workers = max(1, int(self.scan_workers))
        self.optimize_processes = max(1, int(self.optimize_processes))
        self.auto_refresh_minutes = max(1, int(self.auto_refresh_minutes))
        self.auto_trade_use_favorable_price = bool(self.auto_trade_use_favorable_price)
        self.auto_trade_focus_on_signal = bool(self.auto_trade_focus_on_signal)
        if self.auto_trade_focus_signal_mode not in AUTO_TRADE_FOCUS_SIGNAL_MODE_OPTIONS:
            self.auto_trade_focus_signal_mode = AUTO_TRADE_FOCUS_SIGNAL_MODE_OPTIONS[0]
        if not self.optimize_flags:
            self.optimize_flags = DEFAULT_OPTIMIZE_FLAGS.copy()
        self.position_intervals = {
            str(symbol): str(interval)
            for symbol, interval in dict(self.position_intervals).items()
            if str(interval) in APP_INTERVAL_OPTIONS
        }
        self.position_strategy_settings = _normalize_position_strategy_settings(self.position_strategy_settings)
        self.position_filled_fractions = {
            str(symbol): max(0.0, min(0.99, float(fraction)))
            for symbol, fraction in dict(self.position_filled_fractions).items()
            if str(symbol or "").strip()
        }
        self.position_cursor_entry_times = _normalize_position_cursor_entry_times(self.position_cursor_entry_times)
        self.position_open_entry_events = _normalize_position_open_entry_events(self.position_open_entry_events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "leverage": self.leverage,
            "order_mode": self.order_mode,
            "simple_order_amount": self.simple_order_amount,
            "fee_rate": self.fee_rate,
            "history_days": self.history_days,
            "chart_display_hours": self.chart_display_hours,
            "auto_refresh_minutes": self.auto_refresh_minutes,
            "auto_trade_use_favorable_price": self.auto_trade_use_favorable_price,
            "auto_trade_focus_on_signal": self.auto_trade_focus_on_signal,
            "auto_trade_focus_signal_mode": self.auto_trade_focus_signal_mode,
            "kline_interval": self.kline_interval,
            "filter_preset": self.filter_preset,
            "daily_volatility_min": self.daily_volatility_min,
            "quote_volume_min": self.quote_volume_min,
            "use_rsi_filter": self.use_rsi_filter,
            "rsi_length": self.rsi_length,
            "rsi_lower": self.rsi_lower,
            "rsi_upper": self.rsi_upper,
            "use_atr_4h_filter": self.use_atr_4h_filter,
            "atr_4h_min_pct": self.atr_4h_min_pct,
            "optimization_span_pct": self.optimization_span_pct,
            "optimization_steps": self.optimization_steps,
            "optimization_rank_mode": self.optimization_rank_mode,
            "optimization_min_score": self.optimization_min_score,
            "optimization_min_return_pct": self.optimization_min_return_pct,
            "max_grid_combinations": self.max_grid_combinations,
            "enable_parameter_optimization": self.enable_parameter_optimization,
            "scan_workers": self.scan_workers,
            "optimize_processes": self.optimize_processes,
            "optimize_timeframe": self.optimize_timeframe,
            "strategy": self.strategy.to_dict(),
            "optimize_flags": dict(self.optimize_flags),
            "position_intervals": dict(self.position_intervals),
            "position_strategy_settings": {
                symbol: settings.to_dict()
                for symbol, settings in self.position_strategy_settings.items()
            },
            "position_filled_fractions": dict(self.position_filled_fractions),
            "position_cursor_entry_times": {
                symbol: timestamp.isoformat()
                for symbol, timestamp in self.position_cursor_entry_times.items()
            },
            "position_open_entry_events": {
                symbol: [
                    {"time": event_time.isoformat(), "label": label}
                    for event_time, label in events
                ]
                for symbol, events in self.position_open_entry_events.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        payload = dict(data or {})
        strategy_payload = payload.pop("strategy", {}) or {}
        optimize_flags = payload.pop("optimize_flags", {}) or {}
        position_intervals = payload.pop("position_intervals", {}) or {}
        position_strategy_settings = payload.pop("position_strategy_settings", {}) or {}
        position_filled_fractions = payload.pop("position_filled_fractions", {}) or {}
        position_cursor_entry_times = payload.pop("position_cursor_entry_times", {}) or {}
        position_open_entry_events = payload.pop("position_open_entry_events", {}) or {}
        simple_order_amount = payload.pop("simple_order_amount", None)
        chart_display_hours = payload.pop("chart_display_hours", None)
        legacy_chart_display_days = payload.pop("chart_display_days", None)
        legacy_simple_long = payload.pop("simple_long_order_amount", None)
        legacy_simple_short = payload.pop("simple_short_order_amount", None)
        if simple_order_amount is None:
            simple_order_amount = legacy_simple_long if legacy_simple_long is not None else legacy_simple_short
        if simple_order_amount is not None:
            payload["simple_order_amount"] = simple_order_amount
        if chart_display_hours is None and legacy_chart_display_days is not None:
            try:
                chart_display_hours = max(1, int(legacy_chart_display_days)) * 24
            except Exception:
                chart_display_hours = DEFAULT_CHART_DISPLAY_HOURS
        if chart_display_hours is not None:
            payload["chart_display_hours"] = chart_display_hours
        strategy = StrategySettings(
            **{k: v for k, v in strategy_payload.items() if k in StrategySettings.__dataclass_fields__}
        )
        settings = cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})
        settings.strategy = strategy
        merged_flags = DEFAULT_OPTIMIZE_FLAGS.copy()
        merged_flags.update({k: bool(v) for k, v in optimize_flags.items() if k in merged_flags})
        settings.optimize_flags = merged_flags
        settings.position_intervals = {
            str(symbol): str(interval)
            for symbol, interval in dict(position_intervals).items()
            if str(interval) in APP_INTERVAL_OPTIONS
        }
        settings.position_strategy_settings = _normalize_position_strategy_settings(position_strategy_settings)
        settings.position_filled_fractions = {
            str(symbol): max(0.0, min(0.99, float(fraction)))
            for symbol, fraction in dict(position_filled_fractions).items()
            if str(symbol or "").strip()
        }
        settings.position_cursor_entry_times = _normalize_position_cursor_entry_times(position_cursor_entry_times)
        settings.position_open_entry_events = _normalize_position_open_entry_events(position_open_entry_events)
        return settings

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        resolved_path = Path(path) if path is not None else default_config_path()
        for candidate in _load_path_candidates(resolved_path):
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as file:
                settings = cls.from_dict(json.load(file))
            if candidate != resolved_path:
                settings.save(resolved_path)
            return settings
        return cls()

    def save(self, path: Path | None = None) -> None:
        resolved_path = Path(path) if path is not None else default_config_path()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
        temp_path.replace(resolved_path)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    group: str
    kind: str
    minimum: Any = None
    maximum: Any = None
    step: Any = None
    choices: Sequence[Any] = ()
    optimize_default: bool = False
    optimize_span: Any = None
    optimize_step: Any = None
    optimize_priority: int = 5
    optimize_choice_radius: int = 1
    strategies: Sequence[str] = ()


def parameter_spec_applies(spec: ParameterSpec, strategy_type: str) -> bool:
    if not spec.strategies:
        return True
    return str(strategy_type or STRATEGY_TYPE_OPTIONS[0]) in tuple(spec.strategies)


PARAMETER_SPECS: List[ParameterSpec] = [
    ParameterSpec("atr_period", "ATR Length", "core", "int", 2, 100, 1, optimize_default=True, optimize_span=4, optimize_step=1, optimize_priority=3, strategies=("mean_reversion",)),
    ParameterSpec("factor", "Factor", "core", "float", 0.5, 20.0, 0.1, optimize_default=True, optimize_span=2.0, optimize_step=0.1, optimize_priority=1, strategies=("mean_reversion",)),
    ParameterSpec("zone_sensitivity", "Zone Sensitivity", "core", "float", 0.1, 10.0, 0.1, optimize_default=True, optimize_span=1.5, optimize_step=0.1, optimize_priority=1, strategies=("mean_reversion",)),
    ParameterSpec("entry_size_pct", "Entry Size %", "core", "float", 1.0, 100.0, 1.0, optimize_span=6.0, optimize_step=2.0, optimize_priority=8, strategies=STRATEGY_TYPE_OPTIONS),
    ParameterSpec("sensitivity_mode", "QIP Sensitivity Mode", "qip", "choice", choices=QIP_SENSITIVITY_OPTIONS, optimize_choice_radius=2, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("zz_len_raw", "QIP Pivot Length", "qip", "int", 2, 50, 1, optimize_default=True, optimize_span=2, optimize_step=1, optimize_priority=2, strategies=("mean_reversion",)),
    ParameterSpec("atr_mult_raw", "QIP ATR Multiplier", "qip", "float", 0.1, 5.0, 0.1, optimize_default=True, optimize_span=0.5, optimize_step=0.1, optimize_priority=2, strategies=("mean_reversion",)),
    ParameterSpec("use_volume", "QIP Volume Filter", "qip", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("use_rsi_div", "QIP RSI Divergence", "qip", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("use_macd_div", "QIP MACD Divergence", "qip", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("use_ema_conf", "QIP EMA Confirm", "qip", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("min_score", "QIP Min Score", "qip", "int", 1, 5, 1, optimize_default=True, optimize_span=1, optimize_step=1, optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qip_rsi_len", "QIP RSI Length", "qip", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("vol_ma_len", "QIP Volume MA", "qip", "int", 2, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("qip_ema_fast", "QIP EMA Fast", "qip", "int", 2, 200, 1, optimize_span=8, optimize_step=2, optimize_priority=6, strategies=("mean_reversion",)),
    ParameterSpec("qip_ema_slow", "QIP EMA Slow", "qip", "int", 3, 400, 1, optimize_span=16, optimize_step=4, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("qip_use_rsi_zone", "QIP RSI Zone", "qip", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qip_rsi_bull_max", "QIP RSI Bull Max", "qip", "int", 10, 60, 1, optimize_span=6, optimize_step=2, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qip_rsi_bear_min", "QIP RSI Bear Min", "qip", "int", 40, 90, 1, optimize_span=6, optimize_step=2, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_sensitivity", "QTP Sensitivity", "qtp", "int", 1, 100, 1, optimize_default=True, optimize_span=15, optimize_step=5, optimize_priority=4, strategies=("mean_reversion",)),
    ParameterSpec("qtp_ema_fast_len", "QTP EMA Fast", "qtp", "int", 1, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("qtp_ema_slow_len", "QTP EMA Slow", "qtp", "int", 2, 200, 1, optimize_span=16, optimize_step=4, optimize_priority=7, strategies=("mean_reversion",)),
    ParameterSpec("qtp_use_trend", "QTP Trend Filter", "qtp", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qtp_rsi_len", "QTP RSI Length", "qtp", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_stoch_len", "QTP Stoch Length", "qtp", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_atr_len", "QTP ATR Length", "qtp", "int", 1, 100, 1, optimize_span=4, optimize_step=1, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_dev_lookback", "QTP Deviation Lookback", "qtp", "int", 10, 200, 1, optimize_span=15, optimize_step=5, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_vol_len", "QTP Volume SMA", "qtp", "int", 2, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_min_pvt_left", "QTP Min Pivot Left", "qtp", "int", 1, 20, 1, optimize_span=2, optimize_step=1, optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qtp_max_pvt_left", "QTP Max Pivot Left", "qtp", "int", 1, 30, 1, optimize_span=3, optimize_step=1, optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qtp_use_rsi_zone", "QTP RSI Zone", "qtp", "bool", optimize_priority=9, strategies=("mean_reversion",)),
    ParameterSpec("qtp_rsi_bull_max", "QTP RSI Bull Max", "qtp", "int", 10, 50, 1, optimize_span=5, optimize_step=1, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("qtp_rsi_bear_min", "QTP RSI Bear Min", "qtp", "int", 50, 90, 1, optimize_span=5, optimize_step=1, optimize_priority=8, strategies=("mean_reversion",)),
    ParameterSpec("keltner_length", "KC Length", "keltner", "int", 1, 200, 1, optimize_default=True, optimize_span=10, optimize_step=2, optimize_priority=3, strategies=("keltner_trend",)),
    ParameterSpec("keltner_multiplier", "KC Multiplier", "keltner", "float", 0.5, 10.0, 0.1, optimize_default=True, optimize_span=1.0, optimize_step=0.1, optimize_priority=2, strategies=("keltner_trend",)),
    ParameterSpec("keltner_use_ema", "KC Use EMA", "keltner", "bool", optimize_priority=8, strategies=("keltner_trend",)),
    ParameterSpec("keltner_band_style", "KC Band Style", "keltner", "choice", choices=KELTNER_BAND_STYLE_OPTIONS, optimize_choice_radius=1, optimize_priority=5, strategies=("keltner_trend",)),
    ParameterSpec("keltner_atr_length", "KC ATR Length", "keltner", "int", 1, 200, 1, optimize_span=6, optimize_step=2, optimize_priority=4, strategies=("keltner_trend",)),
    ParameterSpec("use_qip", "Enable QIP", "switches", "bool", optimize_priority=10, strategies=("mean_reversion",)),
    ParameterSpec("use_qtp", "Enable QTP", "switches", "bool", optimize_priority=10, strategies=("mean_reversion",)),
    ParameterSpec("beast_mode", "Beast Mode", "switches", "bool", optimize_priority=10, strategies=("mean_reversion",)),
]

DEFAULT_OPTIMIZE_FLAGS: Dict[str, bool] = {
    "atr_period": True,
    "factor": True,
    "zone_sensitivity": True,
    "entry_size_pct": False,
    "sensitivity_mode": True,
    "zz_len_raw": True,
    "atr_mult_raw": True,
    "use_volume": True,
    "use_rsi_div": True,
    "use_macd_div": True,
    "use_ema_conf": True,
    "min_score": True,
    "qip_rsi_len": True,
    "vol_ma_len": True,
    "qip_ema_fast": True,
    "qip_ema_slow": True,
    "qip_use_rsi_zone": True,
    "qip_rsi_bull_max": True,
    "qip_rsi_bear_min": True,
    "qtp_sensitivity": True,
    "qtp_ema_fast_len": True,
    "qtp_ema_slow_len": True,
    "qtp_use_trend": True,
    "qtp_rsi_len": True,
    "qtp_stoch_len": True,
    "qtp_atr_len": True,
    "qtp_dev_lookback": True,
    "qtp_vol_len": True,
    "qtp_min_pvt_left": True,
    "qtp_max_pvt_left": True,
    "qtp_use_rsi_zone": True,
    "qtp_rsi_bull_max": True,
    "qtp_rsi_bear_min": True,
    "keltner_length": True,
    "keltner_multiplier": True,
    "keltner_use_ema": True,
    "keltner_band_style": True,
    "keltner_atr_length": True,
    "use_qip": True,
    "use_qtp": True,
    "beast_mode": True,
}
