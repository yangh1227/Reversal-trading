import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "alt_reversal_trader" / "app.py"
WEB_MOBILE_PATH = Path(__file__).resolve().parents[1] / "alt_reversal_trader" / "web_mobile.py"
MOBILE_JS_PATH = Path(__file__).resolve().parents[1] / "alt_reversal_trader" / "web_static" / "mobile.js"


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


def test_engine_signal_event_auto_focuses_chart_when_favorable_price_toggle_is_off() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_handle_trade_engine_event"),
    ) or ""

    assert "self.settings.auto_trade_focus_on_signal" in source_segment
    assert 'focus_mode == "confirmed"' in source_segment
    assert 'event.preview_entry_side' in source_segment
    assert 'event.preview_entry_zone' in source_segment
    assert 'event.entry_side' in source_segment
    assert 'event.entry_zone' in source_segment
    assert 'event.actionable_entry_side' in source_segment
    assert 'event.actionable_entry_zone' in source_segment
    assert 'event.actionable_entry_kind' in source_segment
    assert "next_signal != previous_signal" in source_segment
    assert "prefer_locked_position_settings=False" in source_segment


def test_engine_signal_event_caches_preview_confirmed_and_actionable_signals_separately() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_handle_trade_engine_event"),
    ) or ""

    assert 'self.last_engine_entry_signal_by_key[(event.symbol, event.interval, "confirmed")] = confirmed_signal' in source_segment
    assert 'self.last_engine_entry_signal_by_key[(event.symbol, event.interval, "preview")] = preview_signal' in source_segment
    assert "self._set_engine_actionable_signal(" in source_segment


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
    assert "보유 포지션" in source_segment
    assert "개를 모두 전체청산할까요?" in source_segment
    assert "self.close_all_positions()" in source_segment


def test_close_all_positions_sends_engine_close_all_command() -> None:
    method = _window_method_node("close_all_positions")
    source_segment = ast.get_source_segment(APP_PATH.read_text(encoding="utf-8"), method) or ""

    assert "EngineCloseAllPositionsCommand()" in source_segment
    assert "self.order_pending_started_at = time.time()" in source_segment
    assert 'self.statusBar().showMessage("전체 포지션 청산 처리 중...", 3000)' in source_segment
    assert 'self.show_warning("청산할 포지션이 없습니다.")' in source_segment


def test_live_order_close_button_uses_total_close_label() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'QPushButton("전체청산")' in source
    assert 'setFixedWidth(156)' in source
    assert 'setToolTip("현재 보유 중인 모든 포지션 전체 청산")' in source


def test_app_starts_mobile_web_server_on_boot() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "self.mobile_web_server = None" in source
    assert "self._start_mobile_web_server()" in source


def test_close_event_stops_mobile_web_server() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("closeEvent"),
    ) or ""

    assert "self._stop_mobile_web_server()" in source_segment


def test_preview_markers_are_reference_only_and_fast_markers_use_actionable_text() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert '"text": "예상청산(참고)"' in source
    assert "예상진입(참고)" in source
    assert '"opposite_signal": "청산신호"' in source
    assert '"text": "청산신호"' in source
    assert "진입신호" in source
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
    assert "chart_display_hours=int(" in source_segment
    assert "auto_trade_use_favorable_price=bool(self.auto_trade_favorable_check.isChecked())" in source_segment
    assert "auto_trade_focus_on_signal=bool(" in source_segment
    assert "auto_trade_focus_signal_mode=self._auto_trade_focus_signal_mode()" in source_segment
    assert "position_strategy_settings=dict(self.settings.position_strategy_settings)" in source_segment
    assert "position_filled_fractions=dict(self.settings.position_filled_fractions)" in source_segment
    assert "position_cursor_entry_times=dict(self.settings.position_cursor_entry_times)" in source_segment


def test_optimized_table_marks_favorable_rows_light_green() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("update_optimized_table"),
    ) or ""

    assert "OPTIMIZED_TABLE_FAVORABLE_ROW_COLOR" in APP_PATH.read_text(encoding="utf-8")
    assert "_optimized_result_favorable_zone(result, current_price)" in source_segment
    assert "favorable_entry = favorable_zone is not None" in source_segment
    assert "item.setBackground(favorable_row_brush)" in source_segment


def test_optimized_table_marks_confirmed_signal_rows_light_red() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("update_optimized_table"),
    ) or ""

    assert "OPTIMIZED_TABLE_SIGNAL_ROW_COLOR" in APP_PATH.read_text(encoding="utf-8")
    assert "_optimized_result_actionable_signal(result, current_price)" in source_segment
    assert 'actionable_signal[2] == "confirmed"' in source_segment
    assert "item.setBackground(signal_row_brush)" in source_segment
    assert "elif entry_signal:" in source_segment


def test_optimized_table_favorable_highlight_avoids_stale_optimization_backtests() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_optimized_result_favorable_zone"),
    ) or ""

    assert "actionable_side, actionable_zone, actionable_kind = self._actionable_signal(symbol, interval)" in source_segment
    assert 'if actionable_kind == "favorable" and actionable_side and actionable_zone in {1, 2, 3}:' in source_segment
    assert "latest_backtest = self._best_available_auto_trade_backtest_for_display(result)" in source_segment
    assert "resolve_favorable_auto_trade_zone(" in source_segment
    assert "return self.favorable_zone_cache.get(key) if key in self.favorable_refresh_pending else None" in source_segment
    assert "if latest_backtest is None:" in source_segment


def test_optimized_result_actionable_signal_can_compute_directly_from_shared_evaluator() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_evaluated_actionable_signal"),
    ) or ""

    assert "latest_backtest = self._best_available_auto_trade_backtest_for_display(result)" in source_segment
    assert "evaluate_auto_trade_candidate(" in source_segment
    assert "trigger_symbol=str(result.symbol)" in source_segment
    assert "trigger_interval=interval" in source_segment


def test_best_available_auto_trade_backtest_for_display_falls_back_to_seed_backtest() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_best_available_auto_trade_backtest_for_display"),
    ) or ""

    assert "latest_backtest = self._latest_auto_trade_backtest(result)" in source_segment
    assert "return self._favorable_backtest_seed(" in source_segment


def test_optimized_result_actionable_signal_keeps_last_display_signal_during_async_refresh() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_optimized_result_actionable_signal"),
    ) or ""

    assert "self.optimized_actionable_signal_cache" in source_segment
    assert "self._actionable_signal(result.symbol, interval)" in source_segment
    assert "if key in self.favorable_refresh_pending or current_price is None or current_price <= 0:" in source_segment
    assert "cached_signal = self.optimized_actionable_signal_cache.get(key)" in source_segment


def test_window_init_starts_optimized_table_highlight_timer() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("__init__"),
    ) or ""

    assert "self.optimized_table_highlight_timer = QTimer(self)" in source_segment
    assert "self.optimized_table_highlight_timer.setInterval(OPTIMIZED_TABLE_HIGHLIGHT_REFRESH_MS)" in source_segment
    assert "self.optimized_table_highlight_timer.timeout.connect(self._refresh_optimized_table_highlights)" in source_segment
    assert "self.optimized_table_highlight_timer.start()" in source_segment
    assert "self.favorable_backtest_poll_timer = QTimer(self)" in source_segment
    assert "self.optimized_actionable_signal_cache" in source_segment
    assert "self.favorable_backtest_poll_timer.timeout.connect(self._poll_favorable_backtest_results)" in source_segment
    assert "self.favorable_backtest_poll_timer.start()" in source_segment


def test_latest_auto_trade_backtest_uses_background_refresh_instead_of_materializing_in_ui() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_latest_auto_trade_backtest"),
    ) or ""

    assert "_enqueue_favorable_backtest_refresh(" in source_segment
    assert "resolve_latest_auto_trade_backtest(" not in source_segment
    assert "run_backtest(" not in source_segment
    assert "resume_backtest(" not in source_segment


def test_close_event_stops_favorable_backtest_process() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("closeEvent"),
    ) or ""

    assert "self.favorable_backtest_poll_timer.stop()" in source_segment
    assert "self.favorable_backtest_process.stop()" in source_segment


def test_prune_caches_keeps_optimized_actionable_display_cache_in_sync() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_prune_caches"),
    ) or ""

    assert "self.optimized_actionable_signal_cache = {" in source_segment
    assert "self.favorable_refresh_pending = {" in source_segment
    assert "self.resolved_auto_trade_backtest_cache = {" in source_segment


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
    assert 'self.history_days_spin.setToolTip("백테스트와 최적화에 사용할 히스토리 일수")' in source_segment
    assert 'layout.addRow("백테스트 일수", self.history_days_spin)' in source_segment


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
    assert "self._actionable_signal_from_evaluation(evaluation)" in source_segment
    assert "next_actionable_signals" in source_segment
    assert "pick_auto_trade_candidate(" in source_segment
    assert "allow_favorable_price_entries=bool(self.settings.auto_trade_use_favorable_price)" in source_segment
    assert "self._pick_auto_trade_candidate(" not in source_segment


def test_submit_open_order_returns_result_for_mobile_api_validation() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_submit_open_order"),
    ) or ""

    assert "-> Tuple[bool, str]" in source_segment
    assert "return False, message" in source_segment
    assert 'return True, f"{target_symbol} 주문 요청이 접수되었습니다."' in source_segment


def test_account_refresh_recovers_stale_auto_trade_pending_state() -> None:
    helper_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_recover_stale_auto_trade_pending_state"),
    ) or ""
    completed_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_on_account_info_completed"),
    ) or ""

    assert "self.engine_order_pending" in helper_source
    assert "self.order_worker_is_auto_trade" in helper_source
    assert "pending_symbol in open_symbols" in helper_source
    assert "pending_interval = str(self.pending_open_order_interval or \"\").strip()" in helper_source
    assert "pending_age = (" in helper_source
    assert "self._remember_position_interval(pending_symbol, pending_interval, persist=False)" in helper_source
    assert 'if pending_symbol == "*" and not open_symbols:' in helper_source
    assert "if pending_age >= ORDER_PENDING_RECOVERY_SECONDS:" in helper_source
    assert "self._set_order_buttons_enabled(True)" in helper_source
    assert "self._recover_stale_auto_trade_pending_state(open_symbols)" in completed_source


def test_account_refresh_prefers_pending_auto_trade_interval_over_stale_saved_interval() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_remember_missing_open_position_intervals"),
    ) or ""

    assert "pending_symbol = str(self.auto_trade_entry_pending_symbol or self.order_worker_symbol or \"\").strip().upper()" in source_segment
    assert "pending_interval = str(self.pending_open_order_interval or \"\").strip()" in source_segment
    assert "if symbol == pending_symbol and pending_interval in APP_INTERVAL_OPTIONS:" in source_segment
    assert "self.settings.position_intervals[symbol] = interval" in source_segment


def test_trade_engine_sync_includes_favorable_price_toggle() -> None:
    source_segment = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_sync_trade_engine_state"),
    ) or ""

    assert "auto_trade_use_favorable_price=bool(self.settings.auto_trade_use_favorable_price)" in source_segment


def test_positions_table_includes_leverage_column() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    display_source = ast.get_source_segment(source, _window_method_node("_position_display_values")) or ""
    populate_source = ast.get_source_segment(source, _window_method_node("_populate_position_row")) or ""
    update_source = ast.get_source_segment(source, _window_method_node("update_positions_table")) or ""

    assert 'self.positions_table = QTableWidget(0, 8)' in source
    assert '["Symbol", "Side", "Leverage", "Amount USDT", "Entry", "UPnL", "수익률", "Action"]' in source
    assert 'f"{position.leverage}x"' in display_source
    assert "if col in (5, 6):" in populate_source
    assert "self.positions_table.setCellWidget(row, 7, action_widget)" in update_source


def test_backtest_summary_moves_to_button_dialog_and_frees_chart_space() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    build_ui_source = ast.get_source_segment(source, _window_method_node("_build_ui")) or ""
    update_source = ast.get_source_segment(source, _window_method_node("update_summary")) or ""
    show_source = ast.get_source_segment(source, _window_method_node("show_backtest_summary")) or ""

    assert 'self.backtest_summary_button = QPushButton("백테스트 요약")' in build_ui_source
    assert "self.backtest_summary_button.clicked.connect(self.show_backtest_summary)" in build_ui_source
    assert "actions_row.addWidget(self.backtest_summary_button)" in build_ui_source
    assert 'summary_group = QGroupBox("Backtest Summary")' not in build_ui_source
    assert "self.summary_box" not in source
    assert "self.backtest_summary_text = \"\\n\".join(lines)" in update_source
    assert "self.backtest_summary_box.setPlainText(self.backtest_summary_text)" in update_source
    assert "self.backtest_summary_window is not None and self.backtest_summary_window.isVisible()" in show_source
    assert "self.backtest_summary_window.hide()" in show_source
    assert 'QWidget(None, windowTitle="백테스트 서머리")' in show_source
    assert 'QMessageBox.information(self, "백테스트 서머리", "표시할 백테스트 서머리가 없습니다.")' in show_source


def test_chart_focus_settings_controls_are_inline_above_chart() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    build_ui_source = ast.get_source_segment(source, _window_method_node("_build_ui")) or ""

    assert "right_layout.addLayout(chart_header_row)" in build_ui_source
    assert 'chart_header_row.addWidget(QLabel("차트 전환"))' in build_ui_source
    assert 'self.auto_trade_focus_enable_check = QCheckBox("사용")' in build_ui_source
    assert 'self.auto_trade_focus_mode_combo.addItem("예상진입신호", "preview")' in build_ui_source
    assert 'self.auto_trade_focus_mode_combo.addItem("진입신호 확정", "confirmed")' in build_ui_source
    assert 'chart_header_row.addWidget(QLabel("차트 표시 시간 범위"))' in build_ui_source
    assert "self.chart_display_days_popup_spin = QSpinBox()" in build_ui_source
    assert 'self.chart_display_days_popup_spin.setSuffix(" 시간")' in build_ui_source


def test_chart_display_days_controls_initial_visible_range_only() -> None:
    default_range_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_default_lightweight_logical_range"),
    ) or ""
    initial_history_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_initial_chart_history"),
    ) or ""

    assert "self.settings.chart_display_hours" in default_range_source
    assert "3_600_000" in default_range_source
    assert "lookback_days" not in initial_history_source


def test_backtest_progress_uses_staged_status_updates() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    begin_source = ast.get_source_segment(source, _window_method_node("_begin_backtest_progress")) or ""
    refresh_source = ast.get_source_segment(source, _window_method_node("_refresh_backtest_progress_display")) or ""
    update_source = ast.get_source_segment(source, _window_method_node("_update_backtest_progress_phase")) or ""
    start_source = ast.get_source_segment(source, _window_method_node("start_optimization")) or ""

    assert "self.backtest_progress_bar.setRange(0, 1000)" in begin_source
    assert "progress_ratio = (prep_ratio * 0.35) + (exec_ratio * 0.65)" in refresh_source
    assert '"phase": "history_loading"' in source
    assert '"phase": "history_ready"' in source
    assert '"phase": "case_running"' in source
    assert "self.optimize_worker.phase_update.connect(self._update_backtest_progress_phase)" in start_source
    assert 'self.backtest_progress_status_text = f"히스토리 로드중' in update_source
    assert 'self.backtest_progress_status_text = f"프로세스 준비중' in update_source


def test_optimized_favorable_badge_uses_green_chip_style() -> None:
    build_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_optimized_group"),
    ) or ""
    update_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_update_optimized_status_labels"),
    ) or ""

    assert "background: #1f9d55" in build_source
    assert "color: #ffffff" in build_source
    assert 'favorable_label.setText("유리" if favorable_count == 1 else f"유리 {favorable_count}")' in update_source


def test_optimized_entry_badge_uses_red_chip_style() -> None:
    build_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_optimized_group"),
    ) or ""
    update_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_update_optimized_status_labels"),
    ) or ""

    assert 'self.optimized_entry_label = QLabel("", group)' in build_source
    assert "background: #d64550" in build_source
    assert 'entry_label.setText("진입" if entry_count == 1 else f"진입 {entry_count}")' in update_source


def test_mobile_dashboard_state_exports_actionable_signal_entries() -> None:
    source = WEB_MOBILE_PATH.read_text(encoding="utf-8")

    assert "_optimized_result_favorable_zone(result, current_price)" in source
    assert "_optimized_result_actionable_signal(result, current_price)" in source
    assert '"orderPending": bool(self.window._is_order_pending())' in source
    assert '"signalEntries": signal_entries' in source
    assert '"actionable": actionable' in source
    assert '"actionableSide": actionable_side' in source
    assert '"actionableKind": actionable_kind' in source
    assert '"kind": "favorable"' in source
    assert "auto_trade_focus_signal_mode" not in source


def test_mobile_order_endpoints_wait_for_ui_validation() -> None:
    source = WEB_MOBILE_PATH.read_text(encoding="utf-8")

    assert 'return JSONResponse(self.invoker.call(lambda: self._submit_fractional_order(symbol, interval, side, fraction)))' in source
    assert 'return JSONResponse(self.invoker.call(lambda: self._submit_simple_order(symbol, interval, side, amount)))' in source
    assert "ok, message = self.window._submit_open_order(" in source
    assert "raise HTTPException(status_code=self._submission_error_status(message), detail=message)" in source


def test_mobile_frontend_renders_actionable_signal_entries_and_red_cards() -> None:
    source = MOBILE_JS_PATH.read_text(encoding="utf-8")

    assert "renderFavorable(state.favorableEntries || [], state.signalEntries || [])" in source
    assert "actionableSignalCode(entry)" in source
    assert 'return suffix;' in source
    assert 'item.favorable ? " favorable" : ""' in source
    assert '!item.favorable && item.actionableKind === "confirmed" ? " signal" : ""' in source


def test_mobile_frontend_blocks_duplicate_manual_orders_while_server_order_is_pending() -> None:
    source = MOBILE_JS_PATH.read_text(encoding="utf-8")

    assert "let orderRequestPending = false;" in source
    assert "let serverOrderPending = false;" in source
    assert "function isOrderActionPending()" in source
    assert "setServerOrderPending(!!state.orderPending);" in source
    assert "if (isOrderActionPending()) {" in source
    assert 'showToast("이미 주문 처리 중입니다.", "info", 2200);' in source
    assert "if (payload?.queued) {" in source


def test_status_strip_uses_uniform_spacing_and_label_heights() -> None:
    build_ui_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _window_method_node("_build_ui"),
    ) or ""

    assert "balance_layout.setSpacing(12)" in build_ui_source
    assert "self.balance_label.setFixedHeight(18)" in build_ui_source
    assert "self.balance_status_label = QLabel(" in build_ui_source
    assert "self.balance_equity_value_label = QLabel(" in build_ui_source
    assert 'self.balance_equity_value_label.setStyleSheet(f"color: #1546b0; {status_strip_font_style}")' in build_ui_source
    assert "self.balance_available_value_label = QLabel(" in build_ui_source
    assert 'self.balance_available_value_label.setStyleSheet(f"color: #1546b0; {status_strip_font_style}")' in build_ui_source
    assert "self.chart_interval_label.setFixedHeight(18)" in build_ui_source
    assert "self.bar_close_countdown_label.setFixedHeight(18)" in build_ui_source
    assert "self.optimization_chart_notice_label.setFixedHeight(18)" in build_ui_source
    assert "balance_layout.addSpacing(12)" not in build_ui_source
