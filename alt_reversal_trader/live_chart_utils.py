from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from .strategy import prepare_ohlcv


def merge_live_bar(df: Optional[pd.DataFrame], bar: Dict[str, object], max_rows: Optional[int] = None) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    if (df is not None and "quote_volume" in df.columns) or "quote_volume" in bar:
        columns.append("quote_volume")
    frame = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=columns)
    if list(frame.columns) != columns:
        frame = frame.reindex(columns=columns)
    row_time = pd.Timestamp(bar["time"])
    row_values = [
        row_time,
        float(bar["open"]),
        float(bar["high"]),
        float(bar["low"]),
        float(bar["close"]),
        float(bar["volume"]),
    ]
    if "quote_volume" in columns:
        row_values.append(float(bar.get("quote_volume", float(bar["close"]) * float(bar["volume"]))))
    if frame.empty:
        frame = pd.DataFrame([row_values], columns=columns)
    else:
        last_time = pd.Timestamp(frame["time"].iloc[-1])
        if last_time == row_time:
            for key, value in zip(columns[1:], row_values[1:]):
                frame.at[frame.index[-1], key] = value
        elif last_time < row_time:
            frame.loc[len(frame)] = row_values
        else:
            matches = frame["time"] == row_time
            if matches.any():
                idx = frame.index[matches][-1]
                for key, value in zip(columns[1:], row_values[1:]):
                    frame.at[idx, key] = value
            else:
                frame = (
                    pd.concat([frame, pd.DataFrame([row_values], columns=columns)], ignore_index=True)
                    .drop_duplicates(subset=["time"], keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)
    return frame


def preview_bar_matches_context(
    preview_bar: Optional[Dict[str, object]],
    symbol: Optional[str],
    interval: Optional[str],
) -> bool:
    if preview_bar is None or not symbol or not interval:
        return False
    return (
        str(preview_bar.get("symbol", "")).strip() == str(symbol).strip()
        and str(preview_bar.get("interval", "")).strip() == str(interval).strip()
    )


def history_with_live_preview(
    frame: Optional[pd.DataFrame],
    preview_bar: Optional[Dict[str, object]],
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if preview_bar is None:
        return frame
    base_frame = frame.copy() if frame is not None else pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    return merge_live_bar(base_frame, preview_bar, max_rows=max_rows)


def seed_two_minute_aggregate(
    recent_history: Optional[pd.DataFrame],
    symbol: str,
    interval: str,
) -> Optional[Dict[str, object]]:
    if interval != "2m" or recent_history is None or recent_history.empty:
        return None
    prepared = prepare_ohlcv(recent_history.copy())
    if prepared.empty:
        return None
    last_row = prepared.iloc[-1]
    last_time = pd.Timestamp(last_row["time"])
    bucket_time = last_time.floor("2min")
    if last_time != bucket_time:
        return None
    seed = {
        "symbol": symbol,
        "interval": interval,
        "time": bucket_time,
        "open": float(last_row["open"]),
        "high": float(last_row["high"]),
        "low": float(last_row["low"]),
        "close": float(last_row["close"]),
        "volume": float(last_row["volume"]),
        "base_volume": float(last_row["volume"]),
        "closed": False,
    }
    if "quote_volume" in last_row.index:
        seed["quote_volume"] = float(last_row["quote_volume"])
        seed["base_quote_volume"] = float(last_row["quote_volume"])
    return seed


def transform_two_minute_bar(
    aggregate_bar: Optional[Dict[str, object]],
    bar: Dict[str, object],
    seed_aggregate: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    """Advance a 2m aggregate with a 1m source bar.

    Returns the next aggregate state and the current aggregated event. The
    returned event is a preview bar until the second minute closes.
    """
    bar_time = pd.Timestamp(bar["time"])
    bucket_time = bar_time.floor("2min")
    is_first_minute = bar_time == bucket_time

    def _visible_bar(payload: Dict[str, object]) -> Dict[str, object]:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"base_volume", "base_quote_volume"}
        }

    def _new_bucket() -> Dict[str, object]:
        payload = {
            "symbol": str(bar["symbol"]),
            "interval": str(bar["interval"]),
            "time": bucket_time,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
            "base_volume": float(bar["volume"]),
            "closed": False,
        }
        quote_volume = float(bar.get("quote_volume", 0.0) or 0.0)
        if quote_volume > 0:
            payload["quote_volume"] = quote_volume
            payload["base_quote_volume"] = quote_volume
        return payload

    if is_first_minute:
        next_aggregate = _new_bucket()
        return next_aggregate, _visible_bar(next_aggregate)

    working = dict(aggregate_bar) if aggregate_bar is not None else None
    if working is None or pd.Timestamp(working["time"]) != bucket_time:
        candidate = dict(seed_aggregate) if seed_aggregate is not None else None
        if candidate is not None and pd.Timestamp(candidate["time"]) == bucket_time:
            working = candidate

    if working is None or pd.Timestamp(working["time"]) != bucket_time:
        provisional = dict(bar)
        provisional["time"] = bucket_time
        provisional["closed"] = False
        return None, provisional

    working["high"] = max(float(working["high"]), float(bar["high"]))
    working["low"] = min(float(working["low"]), float(bar["low"]))
    working["close"] = float(bar["close"])
    working["volume"] = float(working["base_volume"]) + float(bar["volume"])
    if "base_quote_volume" in working or "quote_volume" in bar:
        working["quote_volume"] = float(working.get("base_quote_volume", 0.0)) + float(bar.get("quote_volume", 0.0) or 0.0)
    working["closed"] = bool(bar.get("closed", False))
    visible = _visible_bar(working)
    if visible["closed"]:
        return None, visible
    return working, visible
