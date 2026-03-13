from __future__ import annotations

from typing import Dict, Optional

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
