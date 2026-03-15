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
