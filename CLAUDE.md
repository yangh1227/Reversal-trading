# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 안내 문서입니다.

## 프로젝트 개요

이 저장소는 두 가지 독립적인 컴포넌트로 구성됩니다.

1. **`lightweight_charts/`** — TradingView Lightweight Charts JS 라이브러리의 Python 래퍼. `pip install lightweight-charts`로 설치 가능한 재사용 패키지.
2. **`alt_reversal_trader/`** — PyQt5 기반 데스크톱 자동매매 앱. Binance Futures 실계좌 연동, 백테스팅, 파라미터 최적화 기능 포함.

## 명령어

```bash
# 데스크톱 앱 실행
py -3.12 alt_reversal_trader_pyqt.py

# 전체 테스트 실행
py -3.12 -m pytest

# 특정 테스트 파일 실행
py -3.12 -m pytest test/test_trade_engine.py

# 키워드로 테스트 필터링
py -3.12 -m pytest test/test_trade_engine.py -k auto_trade

# 커밋 전 문법 점검
py -3.12 -m py_compile alt_reversal_trader/app.py alt_reversal_trader/trade_engine.py

# TypeScript 차트 플러그인 빌드 → lightweight_charts/js/ 로 복사
npm run build

# 차트 플러그인 개발용 Vite 서버 실행
npm run dev
```

## 아키텍처

### lightweight_charts/ (차트 라이브러리)

- **`abstract.py`** — 핵심 클래스: `Window`(JS 실행 및 이벤트 루프), `SeriesCommon`(캔들/라인 시리즈 기반), `AbstractChart`(데이터, 드로잉, 이벤트, 서브차트). 모든 차트 상태와 JS 통신이 여기에 집중됩니다.
- **`chart.py`** — 데스크톱 구현: pywebview를 서브프로세스로 래핑하고, 멀티프로세싱 큐로 비동기 JS 평가를 처리합니다.
- **`widgets.py`** — 프레임워크 어댑터: `QtChart`(PyQt5/6), `JupyterChart`(주피터 노트북).
- **`polygon.py`** — Polygon.io 시장 데이터 연동.
- **`src/`** — TypeScript 드로잉 플러그인(추세선, 수평선, 수직선, 박스). rollup으로 `dist/bundle.js`로 컴파일된 후 `lightweight_charts/js/`로 복사됩니다.

### alt_reversal_trader/ (자동매매 앱)

- **`app.py`** (~7000줄) — `MainWindow`: 전체 PyQt UI, 탭 구성(전략 / 백테스트 / 라이브 / 포지션 / 설정), 실시간 차트 렌더링, 최적화 워크플로우. 앱 전체의 중심 조율자.
- **`trade_engine.py`** — `_TradeEngine`: 실시간 실행 루프, `_OrderExecutor`: Binance 주문 생명주기 관리, `_OrderRequest`: 주문 커맨드 데이터클래스. WebSocket 가격 피드, 포지션 추적, 레이트 리밋 복구 처리.
- **`strategy.py`** — `run_backtest()`: 과거 데이터 시뮬레이션, `evaluate_latest_state()`: 실시간 신호 생성. 지표: Supertrend, EMA, RSI, MACD 다이버전스. 3단계 존 기반 진입 시스템. Numba JIT 선택적 사용.
- **`binance_futures.py`** — `BinanceFuturesClient`: OHLCV, 포지션, 주문, 잔고를 위한 REST + WebSocket.
- **`config.py`** — `StrategySettings`, `AppSettings` 데이터클래스, JSON 영속성, 이전 버전 설정 마이그레이션. 저장 경로: `%APPDATA%\AltReversalTrader\`(Windows) 또는 `~/.alt_reversal_trader/`.
- **`optimizer.py`** — 멀티프로세싱을 활용한 전략 파라미터 그리드 탐색.
- **`auto_trade_runtime.py`** — 백테스트 결과에서 자동매매 후보 심볼/인터벌 선택.
- **`live_chart_utils.py`** — 틱 데이터 집계, 실시간 봉과 과거 데이터 병합.
- **`interprocess_rate_limit.py`** — 프로세스 간 공유 레이트 리미터, 복구 인터벌 지원.
- **`qt_compat.py`** — PyQt5/PyQt6 통합 임포트 레이어.

### 테스트

`test/` 디렉토리에 pytest 기반 테스트가 있습니다. 가장 큰 파일은 `test_trade_engine.py`. `test/util.py`의 `Tester` 기반 클래스가 Chart 초기화/정리를 담당합니다. 테스트 데이터는 `examples/1_setting_data/ohlcv.csv`에 있습니다.

## 코딩 규칙

- Python: 공백 4칸 들여쓰기, 함수/변수 `snake_case`, 클래스 `PascalCase`, 상수 `UPPER_SNAKE_CASE`.
- `app.py` 등 UI 파일에서 한글 문자열 사용은 허용됩니다.
- `app.py`, `trade_engine.py` 같은 대형 파일에서 중복을 늘리기보다 공용 모듈로 추출하는 것을 우선합니다.
- `config.py` 설정 구조 변경 시 기존 사용자 설정이 깨지지 않도록 마이그레이션을 반드시 처리합니다.
- 커밋 형식: `fix: ...`, `feat: ...`, `refactor: ...`, `chore: ...` — 커밋당 하나의 행동 변화.
- 타입 체크는 Pyright 사용 (`pyrightconfig.json`이 `alt_reversal_trader/`, `lightweight_charts/`, `test/` 를 포함).
