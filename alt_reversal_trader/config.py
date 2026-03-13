from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json


APP_CONFIG_PATH = Path("alt_reversal_trader_config.json")
APP_INTERVAL_OPTIONS = ("1m", "2m", "3m", "5m", "15m")
CHART_ENGINE_OPTIONS = ("Lightweight",)
OPTIMIZATION_RANK_MODE_OPTIONS = ("score", "return")
DEFAULT_OPTIMIZATION_PROFILE_SCALE = 20.0
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


def default_optimize_process_count() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count - 1))


@dataclass(frozen=True)
class StrategySettings:
    atr_period: int = 10
    factor: float = 3.0
    zone_sensitivity: float = 1.0
    sensitivity_mode: str = "5-Normal"
    zz_len_raw: int = 5
    atr_mult_raw: float = 1.5
    use_volume: bool = True
    use_rsi_div: bool = True
    use_macd_div: bool = False
    use_ema_conf: bool = True
    min_score: int = 2
    qip_rsi_len: int = 14
    vol_ma_len: int = 20
    qip_ema_fast: int = 21
    qip_ema_slow: int = 55
    qip_use_rsi_zone: bool = True
    qip_rsi_bull_max: int = 40
    qip_rsi_bear_min: int = 60
    qtp_sensitivity: int = 60
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
    beast_mode: bool = False
    entry_size_pct: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppSettings:
    api_key: str = ""
    api_secret: str = ""
    chart_engine: str = "Lightweight"
    leverage: int = 3
    order_mode: str = "compound"
    simple_order_amount: float = 50.0
    fee_rate: float = 0.0005
    history_days: int = 5
    kline_interval: str = "1m"
    daily_volatility_min: float = 25.0
    quote_volume_min: float = 1_000_000.0
    use_rsi_filter: bool = True
    rsi_length: int = 14
    rsi_lower: float = 40.0
    rsi_upper: float = 60.0
    use_atr_4h_filter: bool = True
    atr_4h_min_pct: float = 10.0
    optimization_span_pct: float = 20.0
    optimization_steps: int = 5
    optimization_rank_mode: str = "score"
    optimization_min_score: float = 0.0
    optimization_min_return_pct: float = 0.0
    max_grid_combinations: int = 729
    scan_workers: int = 4
    optimize_processes: int = field(default_factory=default_optimize_process_count)
    optimize_timeframe: bool = True
    strategy: StrategySettings = field(default_factory=StrategySettings)
    optimize_flags: Dict[str, bool] = field(default_factory=dict)
    position_intervals: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chart_engine = CHART_ENGINE_OPTIONS[0]
        if self.order_mode not in {"compound", "simple"}:
            self.order_mode = "compound"
        self.simple_order_amount = max(1.0, float(self.simple_order_amount))
        self.atr_4h_min_pct = max(0.0, float(self.atr_4h_min_pct))
        if self.optimization_rank_mode not in OPTIMIZATION_RANK_MODE_OPTIONS:
            self.optimization_rank_mode = OPTIMIZATION_RANK_MODE_OPTIONS[0]
        self.optimization_min_score = max(0.0, float(self.optimization_min_score))
        self.optimization_min_return_pct = float(self.optimization_min_return_pct)
        self.scan_workers = max(1, int(self.scan_workers))
        self.optimize_processes = max(1, int(self.optimize_processes))
        if not self.optimize_flags:
            self.optimize_flags = DEFAULT_OPTIMIZE_FLAGS.copy()
        self.position_intervals = {
            str(symbol): str(interval)
            for symbol, interval in dict(self.position_intervals).items()
            if str(interval) in APP_INTERVAL_OPTIONS
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "chart_engine": self.chart_engine,
            "leverage": self.leverage,
            "order_mode": self.order_mode,
            "simple_order_amount": self.simple_order_amount,
            "fee_rate": self.fee_rate,
            "history_days": self.history_days,
            "kline_interval": self.kline_interval,
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
            "scan_workers": self.scan_workers,
            "optimize_processes": self.optimize_processes,
            "optimize_timeframe": self.optimize_timeframe,
            "strategy": self.strategy.to_dict(),
            "optimize_flags": dict(self.optimize_flags),
            "position_intervals": dict(self.position_intervals),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        payload = dict(data or {})
        strategy_payload = payload.pop("strategy", {}) or {}
        optimize_flags = payload.pop("optimize_flags", {}) or {}
        position_intervals = payload.pop("position_intervals", {}) or {}
        simple_order_amount = payload.pop("simple_order_amount", None)
        legacy_simple_long = payload.pop("simple_long_order_amount", None)
        legacy_simple_short = payload.pop("simple_short_order_amount", None)
        if simple_order_amount is None:
            simple_order_amount = legacy_simple_long if legacy_simple_long is not None else legacy_simple_short
        if simple_order_amount is not None:
            payload["simple_order_amount"] = simple_order_amount
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
        return settings

    @classmethod
    def load(cls, path: Path = APP_CONFIG_PATH) -> "AppSettings":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def save(self, path: Path = APP_CONFIG_PATH) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)


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


PARAMETER_SPECS: List[ParameterSpec] = [
    ParameterSpec("atr_period", "ATR Length", "core", "int", 2, 100, 1, optimize_default=True, optimize_span=4, optimize_step=1, optimize_priority=3),
    ParameterSpec("factor", "Factor", "core", "float", 0.5, 20.0, 0.1, optimize_default=True, optimize_span=0.8, optimize_step=0.1, optimize_priority=1),
    ParameterSpec("zone_sensitivity", "Zone Sensitivity", "core", "float", 0.1, 10.0, 0.1, optimize_default=True, optimize_span=0.4, optimize_step=0.1, optimize_priority=1),
    ParameterSpec("entry_size_pct", "Entry Size %", "core", "float", 1.0, 100.0, 1.0, optimize_span=6.0, optimize_step=2.0, optimize_priority=8),
    ParameterSpec("sensitivity_mode", "QIP Sensitivity Mode", "qip", "choice", choices=QIP_SENSITIVITY_OPTIONS, optimize_choice_radius=2, optimize_priority=7),
    ParameterSpec("zz_len_raw", "QIP Pivot Length", "qip", "int", 2, 50, 1, optimize_default=True, optimize_span=2, optimize_step=1, optimize_priority=2),
    ParameterSpec("atr_mult_raw", "QIP ATR Multiplier", "qip", "float", 0.1, 5.0, 0.1, optimize_default=True, optimize_span=0.5, optimize_step=0.1, optimize_priority=2),
    ParameterSpec("use_volume", "QIP Volume Filter", "qip", "bool", optimize_priority=9),
    ParameterSpec("use_rsi_div", "QIP RSI Divergence", "qip", "bool", optimize_priority=9),
    ParameterSpec("use_macd_div", "QIP MACD Divergence", "qip", "bool", optimize_priority=9),
    ParameterSpec("use_ema_conf", "QIP EMA Confirm", "qip", "bool", optimize_priority=9),
    ParameterSpec("min_score", "QIP Min Score", "qip", "int", 1, 5, 1, optimize_default=True, optimize_span=1, optimize_step=1, optimize_priority=9),
    ParameterSpec("qip_rsi_len", "QIP RSI Length", "qip", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=7),
    ParameterSpec("vol_ma_len", "QIP Volume MA", "qip", "int", 2, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=7),
    ParameterSpec("qip_ema_fast", "QIP EMA Fast", "qip", "int", 2, 200, 1, optimize_span=8, optimize_step=2, optimize_priority=6),
    ParameterSpec("qip_ema_slow", "QIP EMA Slow", "qip", "int", 3, 400, 1, optimize_span=16, optimize_step=4, optimize_priority=7),
    ParameterSpec("qip_use_rsi_zone", "QIP RSI Zone", "qip", "bool", optimize_priority=9),
    ParameterSpec("qip_rsi_bull_max", "QIP RSI Bull Max", "qip", "int", 10, 60, 1, optimize_span=6, optimize_step=2, optimize_priority=8),
    ParameterSpec("qip_rsi_bear_min", "QIP RSI Bear Min", "qip", "int", 40, 90, 1, optimize_span=6, optimize_step=2, optimize_priority=8),
    ParameterSpec("qtp_sensitivity", "QTP Sensitivity", "qtp", "int", 1, 100, 1, optimize_default=True, optimize_span=15, optimize_step=5, optimize_priority=4),
    ParameterSpec("qtp_ema_fast_len", "QTP EMA Fast", "qtp", "int", 1, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=7),
    ParameterSpec("qtp_ema_slow_len", "QTP EMA Slow", "qtp", "int", 2, 200, 1, optimize_span=16, optimize_step=4, optimize_priority=7),
    ParameterSpec("qtp_use_trend", "QTP Trend Filter", "qtp", "bool", optimize_priority=9),
    ParameterSpec("qtp_rsi_len", "QTP RSI Length", "qtp", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=8),
    ParameterSpec("qtp_stoch_len", "QTP Stoch Length", "qtp", "int", 2, 50, 1, optimize_span=4, optimize_step=1, optimize_priority=8),
    ParameterSpec("qtp_atr_len", "QTP ATR Length", "qtp", "int", 1, 100, 1, optimize_span=4, optimize_step=1, optimize_priority=8),
    ParameterSpec("qtp_dev_lookback", "QTP Deviation Lookback", "qtp", "int", 10, 200, 1, optimize_span=15, optimize_step=5, optimize_priority=8),
    ParameterSpec("qtp_vol_len", "QTP Volume SMA", "qtp", "int", 2, 100, 1, optimize_span=8, optimize_step=2, optimize_priority=8),
    ParameterSpec("qtp_min_pvt_left", "QTP Min Pivot Left", "qtp", "int", 1, 20, 1, optimize_span=2, optimize_step=1, optimize_priority=9),
    ParameterSpec("qtp_max_pvt_left", "QTP Max Pivot Left", "qtp", "int", 1, 30, 1, optimize_span=3, optimize_step=1, optimize_priority=9),
    ParameterSpec("qtp_use_rsi_zone", "QTP RSI Zone", "qtp", "bool", optimize_priority=9),
    ParameterSpec("qtp_rsi_bull_max", "QTP RSI Bull Max", "qtp", "int", 10, 50, 1, optimize_span=5, optimize_step=1, optimize_priority=8),
    ParameterSpec("qtp_rsi_bear_min", "QTP RSI Bear Min", "qtp", "int", 50, 90, 1, optimize_span=5, optimize_step=1, optimize_priority=8),
    ParameterSpec("use_qip", "Enable QIP", "switches", "bool", optimize_priority=10),
    ParameterSpec("use_qtp", "Enable QTP", "switches", "bool", optimize_priority=10),
    ParameterSpec("beast_mode", "Beast Mode", "switches", "bool", optimize_priority=10),
]

DEFAULT_OPTIMIZE_FLAGS: Dict[str, bool] = {spec.key: spec.optimize_default for spec in PARAMETER_SPECS}
