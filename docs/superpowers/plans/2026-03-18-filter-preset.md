# Filter Preset (변동성 / 급등종목) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 Market Filters를 "변동성" 프리셋으로 분류하고, 30m RSI≥65 / 거래량≥10M / 24h 등락률≥+15% 조건의 "급등종목" 프리셋을 추가한다.

**Architecture:** `AppSettings`에 `filter_preset` 문자열 필드를 추가하고, `scan_alt_candidates()`에 `use_surge_filter` 파라미터를 추가하여 급등 모드 필터 로직을 분기한다. UI는 필터 그룹 최상단에 프리셋 콤보박스를 추가하고 선택에 따라 기존 컨트롤을 활성/비활성화한다. 전략 로직은 변경하지 않는다.

**Tech Stack:** Python 3.12, PyQt5, pandas-ta, Binance REST API

---

### Task 1: config.py — filter_preset 필드 추가

**Files:**
- Modify: `alt_reversal_trader/config.py:162` (daily_volatility_min 바로 위)
- Modify: `alt_reversal_trader/config.py:233` (to_dict daily_volatility_min 바로 위)

- [ ] **Step 1: AppSettings 데이터클래스에 filter_preset 필드 추가**

`daily_volatility_min: float = 20.0` 바로 위에 추가:
```python
    filter_preset: str = "변동성"
```

- [ ] **Step 2: to_dict()에 filter_preset 직렬화 추가**

`"daily_volatility_min": self.daily_volatility_min,` 바로 위에 추가:
```python
            "filter_preset": self.filter_preset,
```

- [ ] **Step 3: from_dict()는 자동 처리 확인**

`from_dict()`는 `cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})` 패턴을 사용하므로 별도 수정 불필요. JSON에 없으면 기본값 "변동성"으로 fallback됨.

- [ ] **Step 4: 문법 검사**
```bash
py -3.12 -m py_compile alt_reversal_trader/config.py
```
Expected: 출력 없음 (성공)

- [ ] **Step 5: Commit**
```bash
git add alt_reversal_trader/config.py
git commit -m "feat: add filter_preset field to AppSettings"
```

---

### Task 2: binance_futures.py — 급등종목 필터 로직 추가

**Files:**
- Modify: `alt_reversal_trader/binance_futures.py:533-642`

급등종목 모드 조건:
- 24h 거래량 >= `quote_volume_min` (기존 pre-filter 재사용, 고정 10M은 UI에서 설정)
- 24h 등락률 >= +15.0% (ticker `priceChangePercent`, 추가 API 호출 없음)
- 30m RSI >= 65.0 (30m 봉 fetch → RSI 계산)
- daily_volatility 체크 **스킵**

- [ ] **Step 1: scan_alt_candidates() 시그니처에 use_surge_filter 파라미터 추가**

```python
    def scan_alt_candidates(
        self,
        daily_volatility_min: float,
        quote_volume_min: float,
        use_rsi_filter: bool,
        rsi_length: int,
        rsi_lower: float,
        rsi_upper: float,
        use_atr_4h_filter: bool,
        atr_4h_min_pct: float,
        use_surge_filter: bool = False,
        workers: int = 8,
        log_callback: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[CandidateSymbol]:
```

- [ ] **Step 2: enrich() 함수 내부에 급등종목 분기 추가**

현재 `enrich()` 내부 daily_vol 체크 이후:

```python
        def enrich(symbol: str) -> Optional[CandidateSymbol]:
            if should_stop and should_stop():
                return None
            ticker = ticker_map[symbol]

            if use_surge_filter:
                # 24h 등락률 >= +15%
                price_change = float(ticker.get("priceChangePercent", 0.0) or 0.0)
                if price_change < 15.0:
                    return None
                # 30m RSI >= 65
                rsi_30m_limit = min(max(14 * 3, 60), 99)
                rsi_30m_df = _rows_to_ohlcv_frame(self.klines(symbol, "30m", limit=rsi_30m_limit, ttl_seconds=0.0))
                if len(rsi_30m_df) < max(14 + 5, 30):
                    return None
                rsi_30m_value = _rsi_with_pandas_ta(rsi_30m_df["close"], 14)
                if not np.isfinite(rsi_30m_value) or rsi_30m_value < 65.0:
                    return None
                if should_stop and should_stop():
                    return None
                return CandidateSymbol(
                    symbol=symbol,
                    last_price=float(ticker.get("lastPrice", 0.0) or 0.0),
                    price_change_pct=price_change,
                    quote_volume=float(ticker.get("quoteVolume", 0.0) or 0.0),
                    daily_volatility_pct=0.0,
                    rsi_1m=float(rsi_30m_value),
                    atr_4h_pct=float("nan"),
                )

            # 기존 변동성 모드 (아래 기존 코드 유지)
            daily_df = _rows_to_ohlcv_frame(self.klines(symbol, "1d", limit=3, ttl_seconds=0.0))
            daily_vol = _daily_volatility_from_klines(daily_df)
            # ... (기존 코드 그대로)
```

- [ ] **Step 3: 로그 메시지에 급등종목 필터 정보 추가**

`active_filters` 구성 부분을 분기 처리:

```python
        if use_surge_filter:
            active_filters = [
                f"24h 거래량 {quote_volume_min:,.0f}+",
                "24h 등락률 +15%+",
                "30m RSI >= 65",
            ]
        else:
            active_filters = [
                f"24h 거래량 {quote_volume_min:,.0f}+",
                f"일변동성 {daily_volatility_min:.2f}%+",
            ]
            if use_rsi_filter:
                active_filters.append(f"1m RSI <= {rsi_lower:.2f} or >= {rsi_upper:.2f}")
            else:
                active_filters.append("1m RSI OFF")
            if use_atr_4h_filter:
                active_filters.append(f"4h ATR% >= {atr_4h_min_pct:.2f}")
            else:
                active_filters.append("4h ATR% OFF")
```

- [ ] **Step 4: 문법 검사**
```bash
py -3.12 -m py_compile alt_reversal_trader/binance_futures.py
```
Expected: 출력 없음

- [ ] **Step 5: Commit**
```bash
git add alt_reversal_trader/binance_futures.py
git commit -m "feat: add use_surge_filter mode to scan_alt_candidates"
```

---

### Task 3: app.py — UI 프리셋 콤보박스 및 연동

**Files:**
- Modify: `alt_reversal_trader/app.py`
  - `_build_filter_group()` ~2004
  - `_refresh_filter_controls()` ~1908
  - `_apply_loaded_settings()` ~2692
  - `collect_settings()` ~2739
  - `ScanWorker.run()` ~800

#### 3-A: _build_filter_group()에 프리셋 콤보박스 추가

- [ ] **Step 1: filter_preset_combo 위젯 생성 및 레이아웃 최상단 삽입**

`_build_filter_group()` 내 `self.daily_vol_spin = ...` 이전에:

```python
        self.filter_preset_combo = QComboBox()
        self.filter_preset_combo.addItem("변동성", "변동성")
        self.filter_preset_combo.addItem("급등종목", "급등종목")
        self.surge_info_label = QLabel("30m RSI≥65 / 거래량≥10M / 24h≥+15%")
        self.surge_info_label.setStyleSheet("color: gray; font-size: 10px;")
```

레이아웃 addRow는 `layout.addRow("1일 변동성 % >=", self.daily_vol_spin)` 이전에:

```python
        layout.addRow("필터 프리셋", self.filter_preset_combo)
        layout.addRow("", self.surge_info_label)
```

콤보박스 시그널 연결 (`self.rsi_filter_check.toggled.connect(...)` 근처):

```python
        self.filter_preset_combo.currentIndexChanged.connect(lambda _: self._refresh_filter_controls())
```

#### 3-B: _refresh_filter_controls() 업데이트

- [ ] **Step 2: 프리셋에 따라 변동성 컨트롤 활성/비활성화**

기존 메서드를 아래로 교체:

```python
    def _refresh_filter_controls(self) -> None:
        is_surge = (
            hasattr(self, "filter_preset_combo")
            and self.filter_preset_combo.currentData() == "급등종목"
        )
        volatility_controls = [
            self.daily_vol_spin,
            self.quote_volume_spin,
            self.rsi_filter_check,
            self.rsi_length_spin,
            self.rsi_lower_spin,
            self.rsi_upper_spin,
            self.atr_4h_filter_check,
            self.atr_4h_spin,
        ]
        for ctrl in volatility_controls:
            ctrl.setEnabled(not is_surge)
        if hasattr(self, "surge_info_label"):
            self.surge_info_label.setVisible(is_surge)
        if not is_surge:
            rsi_enabled = bool(self.rsi_filter_check.isChecked())
            self.rsi_length_spin.setEnabled(rsi_enabled)
            self.rsi_lower_spin.setEnabled(rsi_enabled)
            self.rsi_upper_spin.setEnabled(rsi_enabled)
            atr_enabled = bool(self.atr_4h_filter_check.isChecked())
            self.atr_4h_spin.setEnabled(atr_enabled)
```

#### 3-C: _apply_loaded_settings() 업데이트

- [ ] **Step 3: 로드 시 filter_preset_combo 값 복원**

`self._refresh_filter_controls()` 호출 바로 위에:

```python
        preset_index = self.filter_preset_combo.findData(settings.filter_preset)
        if preset_index >= 0:
            self.filter_preset_combo.setCurrentIndex(preset_index)
```

#### 3-D: collect_settings() 업데이트

- [ ] **Step 4: filter_preset 값을 AppSettings 생성 시 포함**

`AppSettings(...)` 생성 블록에 `kline_interval=...` 근처에 추가:

```python
            filter_preset=str(self.filter_preset_combo.currentData() or "변동성"),
```

#### 3-E: ScanWorker.run() — use_surge_filter 전달

- [ ] **Step 5: scan_alt_candidates 호출에 use_surge_filter 파라미터 추가**

```python
            candidates = client.scan_alt_candidates(
                daily_volatility_min=self.settings.daily_volatility_min,
                quote_volume_min=self.settings.quote_volume_min,
                use_rsi_filter=self.settings.use_rsi_filter,
                rsi_length=self.settings.rsi_length,
                rsi_lower=self.settings.rsi_lower,
                rsi_upper=self.settings.rsi_upper,
                use_atr_4h_filter=self.settings.use_atr_4h_filter,
                atr_4h_min_pct=self.settings.atr_4h_min_pct,
                use_surge_filter=(self.settings.filter_preset == "급등종목"),
                workers=self.settings.scan_workers,
                log_callback=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
```

- [ ] **Step 6: 문법 검사**
```bash
py -3.12 -m py_compile alt_reversal_trader/app.py alt_reversal_trader/trade_engine.py
```
Expected: 출력 없음

- [ ] **Step 7: Commit**
```bash
git add alt_reversal_trader/app.py
git commit -m "feat: add filter preset combo (변동성/급등종목) to Market Filters UI"
```
