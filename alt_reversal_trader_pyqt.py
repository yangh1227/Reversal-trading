import multiprocessing
import traceback

from alt_reversal_trader.app import AltReversalTraderWindow, create_app
from alt_reversal_trader.crash_logger import install_crash_logging, install_qt_message_logging, write_log
from alt_reversal_trader.qt_compat import app_exec


def main() -> int:
    install_crash_logging()
    install_qt_message_logging()
    try:
        app = create_app()
        window = AltReversalTraderWindow()
        window.show()
        return app_exec(app)
    except Exception:
        write_log("Main Loop Exception", traceback.format_exc(), open_notepad=True)
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
