from alt_reversal_trader.app import AltReversalTraderWindow, create_app
from alt_reversal_trader.qt_compat import app_exec


def main() -> int:
    app = create_app()
    window = AltReversalTraderWindow()
    window.show()
    return app_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
