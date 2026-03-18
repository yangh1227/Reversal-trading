# Repository Guidelines

## 프로젝트 구조
- `alt_reversal_trader/`: PyQt 기반 자동매매 앱, 전략 로직, Binance 클라이언트, 트레이드 엔진이 들어 있습니다.
- `lightweight_charts/`: 재사용 가능한 Python 차트 패키지와 번들된 JS 자산입니다.
- `src/`: 차트 플러그인용 TypeScript 소스입니다.
- `test/`: `pytest` 중심의 테스트 모음입니다. 전략, 엔진, 설정 저장, UI 관련 회귀 테스트가 포함됩니다.
- `examples/`: 예제 OHLCV 데이터와 테스트용 자산입니다.
- `alt_reversal_trader_pyqt.py`: 데스크톱 앱 실행 진입점입니다.

## 빌드, 테스트, 개발 명령
- `py -3.12 alt_reversal_trader_pyqt.py`: 로컬에서 프로그램 실행
- `py -3.12 -m pytest`: 전체 Python 테스트 실행
- `py -3.12 -m pytest test\test_trade_engine.py -k auto_trade`: 자동매매 관련 테스트만 빠르게 확인
- `py -3.12 -m py_compile alt_reversal_trader\app.py alt_reversal_trader\trade_engine.py`: 커밋 전 문법 점검
- `npm run build`: TS 자산 빌드 후 `lightweight_charts/js`로 복사
- `npm run dev`: 차트 플러그인 개발용 Vite 서버 실행

## 코딩 스타일과 네이밍
- Python은 공백 4칸 들여쓰기를 사용합니다.
- 함수/변수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE_CASE`를 따릅니다.
- `app.py`, `trade_engine.py` 같은 대형 파일은 직접 중복을 늘리기보다 공용 모듈로 추출하는 쪽을 우선합니다.
- 기존 파일이 한글 UI 문자열을 포함하지 않는다면 기본적으로 ASCII를 유지합니다.

## 테스트 가이드
- 새 기능이나 버그 수정에는 가능한 한 회귀 테스트를 같이 추가합니다.
- 테스트 이름은 `test_<동작>()` 또는 `test_<주제>.py` 형식을 사용합니다.
- 전략, 자동매매, 실시간 봉 처리, 설정 저장 로직을 건드렸다면 관련 타깃 테스트 후 전체 `pytest`까지 확인합니다.

## 커밋 및 PR 규칙
- 최근 히스토리처럼 `fix: ...`, `refactor: ...`, `chore: ...` 형식을 사용합니다.
- 한 커밋에는 한 가지 행동 변화만 담는 것이 좋습니다.
- PR에는 변경 이유, 핵심 영향 범위, 실행한 테스트 명령, UI 변경 시 스크린샷을 포함합니다.

## 보안 및 설정 주의사항
- API Key, Secret, 실계좌 정보는 절대 커밋하지 않습니다.
- 사용자 설정은 저장소 밖 경로에도 저장되므로, 설정 구조를 바꿀 때는 기존 값이 깨지지 않게 마이그레이션을 고려합니다.
