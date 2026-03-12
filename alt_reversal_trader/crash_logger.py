from __future__ import annotations

import faulthandler
import os
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import sys
import threading
import traceback
from typing import Optional


CRASH_LOG_DIR = Path("alt_reversal_trader_crash_logs").resolve()
CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
_FAULT_LOG_FILE = None
_QT_MESSAGE_HANDLER_INSTALLED = False


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _new_log_path(prefix: str) -> Path:
    return CRASH_LOG_DIR / f"{prefix}_{_timestamp()}.txt"


def _base_header(title: str) -> str:
    return "\n".join(
        [
            f"[{title}]",
            f"time: {datetime.now().isoformat()}",
            f"python: {sys.version}",
            f"platform: {platform.platform()}",
            f"pid: {os.getpid()}",
            f"thread: {threading.current_thread().name}",
            f"cwd: {Path.cwd()}",
            "",
        ]
    )


def open_log_in_notepad(path: Path) -> None:
    try:
        subprocess.Popen(["notepad.exe", str(path)])
    except Exception:
        pass


def write_log(title: str, body: str, open_notepad: bool = False) -> Path:
    path = _new_log_path("crash")
    path.write_text(_base_header(title) + body, encoding="utf-8")
    latest = CRASH_LOG_DIR / "crash_latest.txt"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    if open_notepad:
        open_log_in_notepad(path)
    return path


def log_runtime_event(title: str, body: str, open_notepad: bool = False) -> Path:
    path = _new_log_path("event")
    path.write_text(_base_header(title) + body, encoding="utf-8")
    latest = CRASH_LOG_DIR / "event_latest.txt"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    if open_notepad:
        open_log_in_notepad(path)
    return path


def _format_exception(exc_type, exc_value, exc_traceback) -> str:
    return "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))


def _handle_unhandled_exception(exc_type, exc_value, exc_traceback) -> None:
    path = write_log("Unhandled Exception", _format_exception(exc_type, exc_value, exc_traceback), open_notepad=True)
    sys.stderr.write(f"\nUnhandled exception logged to: {path}\n")


def _threading_excepthook(args) -> None:
    body = "".join(
        [
            f"thread_name: {args.thread.name if args.thread else 'unknown'}\n\n",
            _format_exception(args.exc_type, args.exc_value, args.exc_traceback),
        ]
    )
    write_log("Unhandled Thread Exception", body, open_notepad=True)


def _unraisable_hook(args) -> None:
    body = "".join(
        [
            f"object: {args.object!r}\n",
            f"err_msg: {args.err_msg}\n\n",
            _format_exception(args.exc_type, args.exc_value, args.exc_traceback),
        ]
    )
    write_log("Unraisable Exception", body, open_notepad=False)


def install_crash_logging() -> None:
    global _FAULT_LOG_FILE
    if _FAULT_LOG_FILE is None:
        fault_path = CRASH_LOG_DIR / "fatal_faults_latest.txt"
        _FAULT_LOG_FILE = open(fault_path, "a", encoding="utf-8")
        faulthandler.enable(_FAULT_LOG_FILE, all_threads=True)
    sys.excepthook = _handle_unhandled_exception
    threading.excepthook = _threading_excepthook
    sys.unraisablehook = _unraisable_hook


def log_qt_fatal_message(message: str) -> Path:
    return write_log("Qt Fatal Message", message, open_notepad=True)


def install_qt_message_logging() -> None:
    global _QT_MESSAGE_HANDLER_INSTALLED
    if _QT_MESSAGE_HANDLER_INSTALLED:
        return
    try:
        from PyQt5.QtCore import qInstallMessageHandler
    except Exception:
        try:
            from PySide6.QtCore import qInstallMessageHandler
        except Exception:
            try:
                from PyQt6.QtCore import qInstallMessageHandler
            except Exception:
                return

    def handler(mode, context, message):
        mode_code = int(mode) if not isinstance(mode, int) else mode
        body = "\n".join(
            [
                f"mode: {mode_code}",
                f"file: {getattr(context, 'file', '')}",
                f"line: {getattr(context, 'line', '')}",
                f"function: {getattr(context, 'function', '')}",
                "",
                str(message),
            ]
        )
        if mode_code == 3:
            write_log("Qt Fatal Message", body, open_notepad=True)
        else:
            log_runtime_event("Qt Message", body, open_notepad=False)

    qInstallMessageHandler(handler)
    _QT_MESSAGE_HANDLER_INSTALLED = True
