import pandas as pd

from alt_reversal_trader.live_chart_utils import transform_two_minute_bar


def test_transform_two_minute_bar_emits_bucket_preview_on_first_minute() -> None:
    aggregate, event = transform_two_minute_bar(
        None,
        {
            "symbol": "TESTUSDT",
            "interval": "2m",
            "time": pd.Timestamp("2026-01-01 00:10:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1005.0,
            "closed": True,
        },
    )

    assert aggregate is not None
    assert event["time"] == pd.Timestamp("2026-01-01 00:10:00")
    assert event["closed"] is False
    assert event["close"] == 100.5


def test_transform_two_minute_bar_closes_bucket_on_second_minute_close() -> None:
    aggregate, _event = transform_two_minute_bar(
        None,
        {
            "symbol": "TESTUSDT",
            "interval": "2m",
            "time": pd.Timestamp("2026-01-01 00:10:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1005.0,
            "closed": True,
        },
    )
    aggregate, event = transform_two_minute_bar(
        aggregate,
        {
            "symbol": "TESTUSDT",
            "interval": "2m",
            "time": pd.Timestamp("2026-01-01 00:11:00"),
            "open": 100.5,
            "high": 103.0,
            "low": 100.0,
            "close": 102.0,
            "volume": 12.0,
            "quote_volume": 1224.0,
            "closed": True,
        },
    )

    assert aggregate is None
    assert event["time"] == pd.Timestamp("2026-01-01 00:10:00")
    assert event["closed"] is True
    assert event["high"] == 103.0
    assert event["low"] == 99.0
    assert event["close"] == 102.0
    assert event["volume"] == 22.0
