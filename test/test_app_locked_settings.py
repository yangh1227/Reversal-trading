import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "alt_reversal_trader" / "app.py"


def _load_app_module_ast() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def _window_method_node(method_name: str) -> ast.FunctionDef:
    module = _load_app_module_ast()
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "AltReversalTraderWindow":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"AltReversalTraderWindow.{method_name} not found")


def _call_has_locked_settings_keyword(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg != "prefer_locked_position_settings":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant):
            return bool(value.value) is True
    return False


def _self_method_calls(method_name: str, called_name: str) -> list[ast.Call]:
    method = _window_method_node(method_name)
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == called_name
    ]


def test_schedule_auto_close_signal_requests_locked_position_settings() -> None:
    method = _window_method_node("_schedule_auto_close_signal")
    matching_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_active_backtest_settings"
    ]

    assert matching_calls, "_schedule_auto_close_signal should call _active_backtest_settings"
    assert any(_call_has_locked_settings_keyword(call) for call in matching_calls)


def test_handle_trade_engine_event_reloads_auto_trade_with_locked_settings() -> None:
    matching_calls = _self_method_calls("_handle_trade_engine_event", "_request_symbol_load")

    assert matching_calls, "_handle_trade_engine_event should call _request_symbol_load"
    assert any(_call_has_locked_settings_keyword(call) for call in matching_calls)


def test_lightweight_chart_initializes_optimization_overlay() -> None:
    matching_calls = _self_method_calls("_init_lightweight_chart", "_init_lightweight_optimization_overlay")

    assert matching_calls, "_init_lightweight_chart should initialize the optimization overlay"


def test_render_chart_refreshes_optimization_overlay() -> None:
    matching_calls = _self_method_calls("render_chart", "_set_lightweight_optimization_overlay")

    assert matching_calls, "render_chart should refresh the optimization overlay"
    assert "최적화 차트 표시중" in APP_PATH.read_text(encoding="utf-8")


def test_lightweight_optimization_overlay_updates_header_notice() -> None:
    matching_calls = _self_method_calls("_set_lightweight_optimization_overlay", "_set_optimization_chart_notice_text")

    assert matching_calls, "_set_lightweight_optimization_overlay should also update the header notice"
    assert "optimization_chart_notice_label" in APP_PATH.read_text(encoding="utf-8")


def test_showing_optimized_chart_depends_on_current_backtest_settings() -> None:
    method = _window_method_node("_showing_optimized_chart")
    source_segment = ast.get_source_segment(APP_PATH.read_text(encoding="utf-8"), method) or ""

    assert "self.current_backtest is None" in source_segment
    assert "self.current_backtest.settings == optimization.best_backtest.settings" in source_segment
    assert "self.current_chart_prefers_locked_position_settings" not in source_segment


def test_close_selected_position_requires_confirmation() -> None:
    method = _window_method_node("close_selected_position")
    source_segment = ast.get_source_segment(APP_PATH.read_text(encoding="utf-8"), method) or ""

    assert "QMessageBox.question" in source_segment
    assert "전체청산 확인" in source_segment
    assert "포지션을 전체청산할까요?" in source_segment


def test_live_order_close_button_uses_total_close_label() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'QPushButton("전체청산")' in source
    assert 'setFixedWidth(156)' in source


def test_preview_and_fast_markers_use_unified_signal_text() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert '"text": "예상청산신호"' in source
    assert '예상진입신호' in source
    assert '"opposite_signal": "예상청산신호"' in source
    assert "_auto_close_reason_text(reason)" not in ast.get_source_segment(
        source, _window_method_node("_build_fast_exit_markers")
    )


def test_closed_bar_markers_use_confirmed_entry_helper() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_evaluate_closed_bar_auto_close"),
    ) or ""

    assert "latest_confirmed_entry_event(self.current_backtest, latest_time)" in source_segment


def test_closed_bar_backtest_schedules_auto_trade_with_trigger_bar_time() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_apply_closed_bar_confirmed_backtest"),
    ) or ""

    assert "trigger_symbol=symbol" in source_segment
    assert "trigger_interval=self.current_interval" in source_segment
    assert "trigger_bar_time=confirmed_bar_time" in source_segment


def test_live_backtest_completion_always_reschedules_on_history_mismatch() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_on_live_backtest_completed"),
    ) or ""

    mismatch_block = source_segment.split(
        'if _history_frame_signature(result.get("history")) != _history_frame_signature(current_history):',
        1,
    )[1].split("perf = dict", 1)[0]

    assert "self.live_recalc_pending = False" in mismatch_block
    assert "self._schedule_live_backtest(symbol)" in mismatch_block
    assert "if self.live_recalc_pending:" not in mismatch_block


def test_optimization_completion_does_not_auto_select_first_result_row() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("on_optimization_completed"),
    ) or ""

    assert "self.optimized_table.selectRow(0)" not in source_segment


def test_auto_trade_toggle_uses_requested_reservation_state() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_toggle_auto_trade_mode"),
    ) or ""

    assert "self.auto_trade_requested = bool(checked)" in source_segment
    assert "self._auto_trade_ready()" in source_segment
    assert "self._enable_auto_trade_runtime()" in source_segment
    assert "self._disable_auto_trade_runtime()" in source_segment
    assert "if checked and not self.optimized_results" not in source_segment


def test_auto_trade_button_refresh_does_not_require_optimized_results_to_enable() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_refresh_auto_trade_button_state"),
    ) or ""

    assert "available = bool(self.settings.api_key and self.settings.api_secret)" in source_segment
    assert "requested = bool(self.auto_trade_requested or self.auto_trade_enabled)" in source_segment
    assert "self.auto_trade_button.setEnabled(requested or available)" in source_segment
    assert "self.optimized_results" not in source_segment


def test_auto_trade_ready_ignores_engine_failed_lockout() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_auto_trade_ready"),
    ) or ""

    assert "self.optimized_results" in source_segment
    assert "self.settings.api_key and self.settings.api_secret" in source_segment
    assert "engine_failed" not in source_segment


def test_trade_engine_failure_schedules_recovery_and_preserves_request() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_handle_trade_engine_failure"),
    ) or ""

    assert "requested_auto_trade = bool(self.auto_trade_requested or self.auto_trade_enabled)" in source_segment
    assert "self.auto_trade_requested = requested_auto_trade" in source_segment
    assert "self._schedule_trade_engine_recovery()" in source_segment


def test_auto_trade_runtime_ensures_trade_engine_is_available() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_enable_auto_trade_runtime"),
    ) or ""

    assert "self._ensure_trade_engine_available()" in source_segment
    assert "trade engine 복구 후 자동으로 다시 켜집니다." in source_segment


def test_optimization_completion_activates_requested_auto_trade() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("on_optimization_completed"),
    ) or ""

    assert "self._activate_requested_auto_trade_if_ready()" in source_segment


def test_lightweight_indicator_lines_disable_crosshair_markers() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_init_lightweight_chart"),
    ) or ""

    assert source_segment.count("crosshair_marker=False") >= 6


def test_collect_settings_preserves_auto_refresh_minutes_and_locked_positions() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("collect_settings"),
    ) or ""

    assert "auto_refresh_minutes=int(self.auto_refresh_minutes_spin.value())" in source_segment
    assert "position_strategy_settings=dict(self.settings.position_strategy_settings)" in source_segment
    assert "position_filled_fractions=dict(self.settings.position_filled_fractions)" in source_segment
    assert "position_cursor_entry_times=dict(self.settings.position_cursor_entry_times)" in source_segment


def test_optimized_table_marks_favorable_rows_light_green() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("update_optimized_table"),
    ) or ""

    assert "OPTIMIZED_TABLE_FAVORABLE_ROW_COLOR" in APP_PATH.read_text(encoding="utf-8")
    assert "favorable_auto_trade_fraction(" in APP_PATH.read_text(encoding="utf-8")
    assert "item.setBackground(favorable_row_brush)" in source_segment


def test_optimized_table_favorable_highlight_avoids_stale_optimization_backtests() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_optimized_result_has_favorable_entry"),
    ) or ""

    assert "latest_backtest: Optional[BacktestResult] = None" in source_segment
    assert "return result.best_backtest" not in source_segment
    assert "if latest_backtest is None:" in source_segment


def test_window_init_starts_optimized_table_highlight_timer() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("__init__"),
    ) or ""

    assert "self.optimized_table_highlight_timer = QTimer(self)" in source_segment
    assert "self.optimized_table_highlight_timer.setInterval(OPTIMIZED_TABLE_HIGHLIGHT_REFRESH_MS)" in source_segment
    assert "self.optimized_table_highlight_timer.timeout.connect(self._refresh_optimized_table_highlights)" in source_segment
    assert "self.optimized_table_highlight_timer.start()" in source_segment


def test_optimized_table_highlight_refresh_restores_palette_base_for_non_favorable_rows() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_refresh_optimized_table_highlights"),
    ) or ""

    assert "self.optimized_table.palette().base().color()" in source_segment
    assert "QColor()" not in source_segment


def test_kline_stream_worker_uses_shared_two_minute_transform_helper() -> None:
    module = _load_app_module_ast()
    source = APP_PATH.read_text(encoding="utf-8")

    helper_import_present = "transform_two_minute_bar as _transform_two_minute_bar" in source
    transform_method_source = ""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "KlineStreamWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_transform_bar":
                    transform_method_source = ast.get_source_segment(source, item) or ""
                    break

    assert helper_import_present
    assert "_transform_two_minute_bar(" in transform_method_source


def test_run_auto_refresh_logs_use_configured_minutes() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("run_auto_refresh"),
    ) or ""

    assert 'f"자동 {self.auto_refresh_minutes}분 갱신 시점이지만 이전 작업이 아직 실행 중이라 건너뜁니다."' in source_segment
    assert 'f"자동 {self.auto_refresh_minutes}분 갱신 시작"' in source_segment


def test_run_scan_and_optimize_defaults_to_preserve_existing() -> None:
    method = _window_method_node("run_scan_and_optimize")

    assert method.args.defaults
    default = method.args.defaults[-1]
    assert isinstance(default, ast.Constant)
    assert default.value is True


def test_window_init_keeps_refresh_preserve_flag_as_idle_false() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("__init__"),
    ) or ""

    assert "self.preserve_lists_during_refresh = False" in source_segment
    assert "bool(preserve_existing)" not in source_segment


def test_run_scan_and_optimize_does_not_clear_live_results_before_refresh() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("run_scan_and_optimize"),
    ) or ""

    assert "self.preserve_lists_during_refresh = bool(preserve_existing)" in source_segment
    assert "self.optimized_results = {}" not in source_segment
    assert "self.history_cache = {}" not in source_segment
    assert "self.backtest_cache = {}" not in source_segment
    assert "self.current_symbol = None" not in source_segment


def test_filter_group_does_not_expose_chart_engine_selector() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_filter_group"),
    ) or ""

    assert "chart_engine_combo" not in source_segment
    assert "차트 엔진" not in source_segment


def test_app_source_no_longer_keeps_legacy_chart_engine_state() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "chart_mode" not in source
    assert "CHART_ENGINE_OPTIONS" not in source


def test_filter_group_exposes_chart_backtest_history_days_setting() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_filter_group"),
    ) or ""

    assert 'self.history_days_spin.setSuffix(" 일")' in source_segment
    assert 'self.history_days_spin.setToolTip("차트 로드와 백테스트에 사용할 히스토리 일수")' in source_segment
    assert 'layout.addRow("차트/백테스트 일수", self.history_days_spin)' in source_segment


def test_optimization_result_buffers_pending_backtests_when_preserving_lists() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("on_optimization_result"),
    ) or ""

    assert "self.pending_backtest_cache[cache_key] = optimization.best_backtest" in source_segment
    assert "self.pending_chart_indicator_cache[cache_key] = _chart_indicators_from_backtest(optimization.best_backtest)" in source_segment


def test_optimization_completion_swaps_pending_results_atomically() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("on_optimization_completed"),
    ) or ""

    assert "self.optimized_results = dict(self.pending_optimized_results)" in source_segment
    assert "self.backtest_cache.update(self.pending_backtest_cache)" in source_segment
    assert "self.chart_indicator_cache.update(self.pending_chart_indicator_cache)" in source_segment
    assert "self.history_cache.update(self.pending_history_cache)" in source_segment


def test_fallback_auto_trade_cycle_uses_shared_runtime_evaluator() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_run_auto_trade_cycle"),
    ) or ""

    assert "evaluate_auto_trade_candidate(" in source_segment
    assert "pick_auto_trade_candidate(" in source_segment
    assert "self._pick_auto_trade_candidate(" not in source_segment
