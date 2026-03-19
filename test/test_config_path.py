import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd


CONFIG_MODULE_PATH = Path(__file__).resolve().parents[1] / "alt_reversal_trader" / "config.py"


def _load_config_module():
    spec = importlib.util.spec_from_file_location("alt_reversal_trader_config_test_module", CONFIG_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestConfigPath(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _load_config_module()

    def test_load_without_existing_file_uses_updated_code_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config_dir = base_dir / "appdata"
            app_dir = base_dir / "app"
            cwd_dir = base_dir / "cwd"
            config_dir.mkdir()
            app_dir.mkdir()
            cwd_dir.mkdir()

            original_cwd = Path.cwd()
            try:
                os.chdir(cwd_dir)
                with patch.object(self.config, "_user_config_dir", return_value=config_dir), patch.object(
                    self.config, "_app_base_dir", return_value=app_dir
                ):
                    loaded = self.config.AppSettings.load()
            finally:
                os.chdir(original_cwd)

            self.assertEqual(loaded.leverage, 2)
            self.assertEqual(loaded.history_days, 3)
            self.assertEqual(loaded.chart_display_hours, 24)
            self.assertEqual(loaded.auto_refresh_minutes, 30)
            self.assertTrue(loaded.auto_trade_use_favorable_price)
            self.assertTrue(loaded.auto_trade_focus_on_signal)
            self.assertEqual(loaded.auto_trade_focus_signal_mode, "preview")
            self.assertEqual(loaded.daily_volatility_min, 20.0)
            self.assertEqual(loaded.quote_volume_min, 10_000_000.0)
            self.assertFalse(loaded.use_rsi_filter)
            self.assertFalse(loaded.use_atr_4h_filter)
            self.assertEqual(loaded.optimization_min_score, 70.0)
            self.assertEqual(loaded.max_grid_combinations, 300)
            self.assertEqual(loaded.optimize_processes, self.config.default_optimize_process_count())
            self.assertEqual(loaded.strategy.factor, 17.0)
            self.assertEqual(loaded.strategy.zone_sensitivity, 8.0)
            self.assertEqual(loaded.strategy.sensitivity_mode, "10-Ultra Broad Max")
            self.assertTrue(loaded.strategy.use_macd_div)
            self.assertEqual(loaded.strategy.min_score, 3)
            self.assertEqual(loaded.strategy.qtp_sensitivity, 10)
            self.assertTrue(loaded.strategy.beast_mode)
            self.assertTrue(loaded.optimize_flags["beast_mode"])
            self.assertFalse(loaded.optimize_flags["entry_size_pct"])

    def test_save_uses_user_config_dir_instead_of_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config_dir = base_dir / "appdata"
            other_cwd = base_dir / "other"
            config_dir.mkdir()
            other_cwd.mkdir()

            settings = self.config.AppSettings(
                kline_interval="5m",
                order_mode="simple",
                simple_order_amount=123.0,
                auto_refresh_minutes=45,
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                with patch.object(self.config, "_user_config_dir", return_value=config_dir):
                    settings.save()
            finally:
                os.chdir(original_cwd)

            canonical_path = config_dir / self.config.APP_CONFIG_FILENAME
            legacy_path = other_cwd / self.config.APP_CONFIG_FILENAME

            self.assertTrue(canonical_path.exists())
            self.assertFalse(legacy_path.exists())
            saved_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["kline_interval"], "5m")
            self.assertEqual(saved_payload["order_mode"], "simple")
            self.assertEqual(saved_payload["simple_order_amount"], 123.0)
            self.assertEqual(saved_payload["auto_refresh_minutes"], 45)

    def test_load_falls_back_to_legacy_app_dir_file_and_migrates_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            app_dir = base_dir / "app"
            config_dir = base_dir / "appdata"
            app_dir.mkdir()
            config_dir.mkdir()

            legacy_path = app_dir / self.config.APP_CONFIG_FILENAME
            legacy_payload = {
                "kline_interval": "15m",
                "order_mode": "simple",
                "simple_order_amount": 88.5,
            }
            legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

            with patch.object(self.config, "_user_config_dir", return_value=config_dir), patch.object(
                self.config, "_app_base_dir", return_value=app_dir
            ):
                loaded = self.config.AppSettings.load()

            canonical_path = config_dir / self.config.APP_CONFIG_FILENAME

            self.assertEqual(loaded.kline_interval, "15m")
            self.assertEqual(loaded.order_mode, "simple")
            self.assertEqual(loaded.simple_order_amount, 88.5)
            self.assertTrue(canonical_path.exists())
            migrated_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated_payload["kline_interval"], "15m")
            self.assertEqual(migrated_payload["order_mode"], "simple")
            self.assertEqual(migrated_payload["simple_order_amount"], 88.5)

    def test_position_strategy_settings_round_trip_through_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / self.config.APP_CONFIG_FILENAME
            locked_settings = self.config.StrategySettings(atr_period=7, factor=3.5)
            settings = self.config.AppSettings(
                auto_trade_use_favorable_price=False,
                auto_trade_focus_on_signal=False,
                auto_trade_focus_signal_mode="confirmed",
                chart_display_hours=12,
                position_intervals={"BTCUSDT": "5m"},
                position_strategy_settings={"BTCUSDT": locked_settings},
                position_filled_fractions={"BTCUSDT": 0.5},
                position_cursor_entry_times={"BTCUSDT": pd.Timestamp("2026-01-01 00:10:00")},
            )

            settings.save(config_path)
            loaded = self.config.AppSettings.load(config_path)

            self.assertEqual(loaded.position_intervals["BTCUSDT"], "5m")
            self.assertFalse(loaded.auto_trade_use_favorable_price)
            self.assertFalse(loaded.auto_trade_focus_on_signal)
            self.assertEqual(loaded.auto_trade_focus_signal_mode, "confirmed")
            self.assertEqual(loaded.chart_display_hours, 12)
            self.assertEqual(loaded.position_strategy_settings["BTCUSDT"], locked_settings)
            self.assertEqual(loaded.position_filled_fractions["BTCUSDT"], 0.5)
            self.assertEqual(loaded.position_cursor_entry_times["BTCUSDT"], pd.Timestamp("2026-01-01 00:10:00"))

    def test_legacy_chart_display_days_migrates_to_hours(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / self.config.APP_CONFIG_FILENAME
            config_path.write_text(json.dumps({"chart_display_days": 2}), encoding="utf-8")

            loaded = self.config.AppSettings.load(config_path)

            self.assertEqual(loaded.chart_display_hours, 48)

    def test_default_app_settings_use_dynamic_optimize_process_count(self) -> None:
        settings = self.config.AppSettings()

        self.assertEqual(settings.optimize_processes, self.config.default_optimize_process_count())

    def test_legacy_chart_engine_field_is_ignored_on_load(self) -> None:
        settings = self.config.AppSettings.from_dict(
            {
                "chart_engine": "Plotly",
                "kline_interval": "2m",
            }
        )

        self.assertEqual(settings.kline_interval, "2m")
        self.assertFalse(hasattr(settings, "chart_engine"))

    def test_save_omits_legacy_chart_engine_field(self) -> None:
        payload = self.config.AppSettings().to_dict()

        self.assertNotIn("chart_engine", payload)


if __name__ == "__main__":
    unittest.main()
