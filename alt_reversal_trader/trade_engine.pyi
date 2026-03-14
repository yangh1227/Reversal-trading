from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import StrategySettings


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
    auto_close_enabled_symbols: Tuple[str, ...]
    position_intervals: Dict[str, str]
    watchlist: Tuple[EngineWatchlistItem, ...]


@dataclass(frozen=True)
class EngineOpenOrderCommand:
    symbol: str
    interval: str
    side: str
    leverage: int
    fraction: Optional[float] = ...
    margin: Optional[float] = ...
    auto_trade: bool = ...
    strategy_settings: Optional[StrategySettings] = ...


@dataclass(frozen=True)
class EngineCloseOrderCommand:
    symbol: str
    reason: Optional[str] = ...
    auto_close: bool = ...


@dataclass(frozen=True)
class EngineStopCommand:
    pass


@dataclass(frozen=True)
class EngineLogEvent:
    message: str


@dataclass(frozen=True)
class EngineHealthEvent:
    status: str
    detail: str = ...


@dataclass(frozen=True)
class EngineOrderSubmittedEvent:
    symbol: str
    auto_close: bool = ...
    auto_trade: bool = ...
    interval: Optional[str] = ...
    fraction: float = ...


@dataclass(frozen=True)
class EngineOrderCompletedEvent:
    symbol: str
    message: str
    auto_close: bool = ...
    auto_trade: bool = ...
    interval: Optional[str] = ...
    fraction: float = ...
    strategy_settings: Optional[StrategySettings] = ...


@dataclass(frozen=True)
class EngineOrderFailedEvent:
    symbol: str
    message: str
    auto_close: bool = ...
    auto_trade: bool = ...
    interval: Optional[str] = ...
    fraction: float = ...


@dataclass(frozen=True)
class EngineSignalEvent:
    symbol: str
    interval: str
    entry_side: str = ...
    entry_zone: int = ...
    exit_reason: str = ...
    bar_time: object = ...


class TradeEngineController:
    def __init__(self) -> None: ...
    def start(self) -> None: ...
    def is_alive(self) -> bool: ...
    def send(self, command: object) -> None: ...
    def drain_events(self, limit: int = ...) -> List[object]: ...
    def stop(self, timeout: float = ...) -> None: ...
