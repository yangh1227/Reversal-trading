# Shift+Drag Measure Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shift+마우스 드래그 시 차트에 가격 변동률(%)을 표시하는 사각형 오버레이를 추가한다. 드래그 방향이 위(상승)면 초록, 아래(하락)면 빨강. 마우스를 놓아도 유지되고, Shift 없는 클릭 시 사라진다.

**Architecture:** 새 TypeScript 모듈 `src/shift-drag-measure/`를 PluginBase 서브클래스로 구현한다. `handler.ts`에서 Shift+mousedown/mousemove/mouseup 이벤트를 감지해 plugin에 가격 범위를 전달한다. Plugin은 전체 너비에 걸쳐 사각형을 렌더링하고 중앙에 % 텍스트를 그린다. 빌드 후 `lightweight_charts/js/`에 반영된다.

**Tech Stack:** TypeScript, lightweight-charts ISeriesPrimitive API, fancy-canvas (CanvasRenderingTarget2D), rollup(npm run build)

---

## 파일 구조

| 역할 | 파일 |
|------|------|
| 신규: Plugin 클래스 | `src/shift-drag-measure/shift-drag-measure.ts` |
| 신규: Pane View (좌표 변환) | `src/shift-drag-measure/pane-view.ts` |
| 신규: Pane Renderer (canvas 렌더링) | `src/shift-drag-measure/pane-renderer.ts` |
| 수정: Handler에 이벤트 통합 | `src/general/handler.ts` |
| 수정: Export 추가 | `src/index.ts` |

---

## Task 1: pane-renderer.ts 작성

**Files:**
- Create: `src/shift-drag-measure/pane-renderer.ts`

- [ ] **Step 1: 파일 생성**

```typescript
// src/shift-drag-measure/pane-renderer.ts
import { ISeriesPrimitivePaneRenderer } from "lightweight-charts";
import { CanvasRenderingTarget2D } from "fancy-canvas";

export class ShiftDragMeasureRenderer implements ISeriesPrimitivePaneRenderer {
    private _startY: number | null;
    private _endY: number | null;
    private _pct: number | null;

    constructor(startY: number | null, endY: number | null, pct: number | null) {
        this._startY = startY;
        this._endY = endY;
        this._pct = pct;
    }

    draw(target: CanvasRenderingTarget2D): void {
        if (this._startY === null || this._endY === null || this._pct === null) return;

        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            const { width } = scope.bitmapSize;

            const y1 = Math.round(this._startY! * scope.verticalPixelRatio);
            const y2 = Math.round(this._endY! * scope.verticalPixelRatio);
            const top = Math.min(y1, y2);
            const bottom = Math.max(y1, y2);
            const height = bottom - top;
            if (height < 1) return;

            // 상승(endY < startY = 화면 위 = 가격 높음) → 초록, 하락 → 빨강
            const isUp = this._endY! < this._startY!;
            const fillColor = isUp ? 'rgba(0, 200, 100, 0.20)' : 'rgba(255, 80, 80, 0.20)';
            const strokeColor = isUp ? 'rgba(0, 200, 100, 0.75)' : 'rgba(255, 80, 80, 0.75)';

            // 사각형 채우기
            ctx.fillStyle = fillColor;
            ctx.fillRect(0, top, width, height);

            // 테두리 (BoxPaneRenderer 패턴과 동일하게 offset 없이)
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = Math.round(scope.verticalPixelRatio);
            ctx.strokeRect(0, top, width, height);

            // % 텍스트 (중앙)
            const sign = this._pct! >= 0 ? '+' : '';
            const label = `${sign}${this._pct!.toFixed(2)}%`;
            const fontSize = Math.round(13 * scope.verticalPixelRatio);
            ctx.font = `bold ${fontSize}px sans-serif`;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.90)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, width / 2, top + height / 2);
        });
    }
}
```

- [ ] **Step 2: 문법 확인 (빌드는 Task 4에서)**

---

## Task 2: pane-view.ts 작성

**Files:**
- Create: `src/shift-drag-measure/pane-view.ts`

- [ ] **Step 1: 파일 생성**

```typescript
// src/shift-drag-measure/pane-view.ts
import { ISeriesPrimitivePaneView } from "lightweight-charts";
import { ShiftDragMeasure } from "./shift-drag-measure";
import { ShiftDragMeasureRenderer } from "./pane-renderer";

export class ShiftDragMeasurePaneView implements ISeriesPrimitivePaneView {
    private _source: ShiftDragMeasure;
    private _startY: number | null = null;
    private _endY: number | null = null;
    private _pct: number | null = null;

    constructor(source: ShiftDragMeasure) {
        this._source = source;
    }

    update(): void {
        const start = this._source.startPrice;
        const end = this._source.endPrice;
        if (start === null || end === null) {
            this._startY = null;
            this._endY = null;
            this._pct = null;
            return;
        }
        const series = this._source.series;
        this._startY = series.priceToCoordinate(start) ?? null;
        this._endY = series.priceToCoordinate(end) ?? null;
        if (start !== 0) {
            this._pct = (end - start) / Math.abs(start) * 100;
        } else {
            this._pct = null;
        }
    }

    renderer(): ShiftDragMeasureRenderer {
        return new ShiftDragMeasureRenderer(this._startY, this._endY, this._pct);
    }
}
```

---

## Task 3: shift-drag-measure.ts (Plugin 클래스) 작성

**Files:**
- Create: `src/shift-drag-measure/shift-drag-measure.ts`

- [ ] **Step 1: 파일 생성**

```typescript
// src/shift-drag-measure/shift-drag-measure.ts
import { ISeriesPrimitivePaneView } from "lightweight-charts";
import { PluginBase } from "../plugin-base";
import { ShiftDragMeasurePaneView } from "./pane-view";

export class ShiftDragMeasure extends PluginBase {
    public startPrice: number | null = null;
    public endPrice: number | null = null;

    private _paneViews: ShiftDragMeasurePaneView[];

    constructor() {
        super();
        this._paneViews = [new ShiftDragMeasurePaneView(this)];
    }

    paneViews(): ISeriesPrimitivePaneView[] {
        return this._paneViews;
    }

    setPrices(start: number, end: number): void {
        this.startPrice = start;
        this.endPrice = end;
        this.requestUpdate();
    }

    clear(): void {
        this.startPrice = null;
        this.endPrice = null;
        this.requestUpdate();
    }

    isActive(): boolean {
        return this.startPrice !== null;
    }
}
```

---

## Task 4: handler.ts에 이벤트 통합

**Files:**
- Modify: `src/general/handler.ts`

- [ ] **Step 1: import 추가**

`handler.ts` 상단 import 블록 끝에 추가:
```typescript
import { ShiftDragMeasure } from "../shift-drag-measure/shift-drag-measure";
import { Coordinate } from "lightweight-charts";
```

(`Coordinate`는 이미 lightweight-charts에서 re-export됨. 상단 import에 추가)

- [ ] **Step 2: Handler 클래스에 필드 추가**

`Handler` 클래스의 public 필드 선언부(`public _seriesList` 라인 아래)에 추가:
```typescript
private _shiftDragMeasure: ShiftDragMeasure;
```

- [ ] **Step 3: constructor에서 초기화 및 이벤트 등록**

`constructor` 내부, `this.reSize()` 호출 바로 위에 아래 코드를 삽입:

```typescript
// Shift+Drag 가격 측정 오버레이
this._shiftDragMeasure = new ShiftDragMeasure();
this.series.attachPrimitive(this._shiftDragMeasure);

let _sdmDragging = false;
let _sdmStartPrice: number | null = null;

this.div.addEventListener('mousedown', (e: MouseEvent) => {
    if (e.shiftKey) {
        const rect = this.div.getBoundingClientRect();
        const y = (e.clientY - rect.top) as Coordinate;
        const price = this.series.coordinateToPrice(y);
        if (price !== null) {
            _sdmStartPrice = price;
            _sdmDragging = true;
            this._shiftDragMeasure.setPrices(price, price);
        }
        e.preventDefault();  // 차트 드래그 스크롤 방지 (stopPropagation 제거 — chart 이벤트 차단 방지)
    } else {
        if (this._shiftDragMeasure.isActive()) {
            this._shiftDragMeasure.clear();
        }
    }
});

// document 리스너는 드래그가 div 밖으로 나가도 추적하기 위해 필요.
// 다중 Handler 인스턴스에서의 누적을 막기 위해 _sdmDragging 가드로 처리.
const _sdmMouseMove = (e: MouseEvent) => {
    if (!_sdmDragging || _sdmStartPrice === null) return;
    const rect = this.div.getBoundingClientRect();
    const y = (e.clientY - rect.top) as Coordinate;
    const price = this.series.coordinateToPrice(y);
    if (price !== null) {
        this._shiftDragMeasure.setPrices(_sdmStartPrice, price);
    }
};
const _sdmMouseUp = () => {
    _sdmDragging = false;
    // 마우스 업 시 사각형 유지 (clear 하지 않음)
};
document.addEventListener('mousemove', _sdmMouseMove);
document.addEventListener('mouseup', _sdmMouseUp);
```

---

## Task 5: index.ts에 export 추가

**Files:**
- Modify: `src/index.ts`

- [ ] **Step 1: export 라인 추가**

`src/index.ts` 끝에 추가:
```typescript
export * from './shift-drag-measure/shift-drag-measure';
```

---

## Task 6: 빌드 및 검증

- [ ] **Step 1: TypeScript 빌드**

```bash
cd C:\Users\Dystopia\Desktop\lightweight-charts-python-main
npm run build
```

Expected: 에러 없이 `dist/bundle.js` 생성 후 `lightweight_charts/js/`에 복사 완료

- [ ] **Step 2: 문법 오류 확인**

빌드 로그에서 TypeScript 에러가 없는지 확인. 에러 발생 시 해당 파일 수정 후 재빌드.

- [ ] **Step 3: 앱 실행 후 동작 확인**

```bash
py -3.12 alt_reversal_trader_pyqt.py
```

체크리스트:
- [ ] Shift+마우스 드래그 → 색 사각형 + % 텍스트 표시
- [ ] 위로 드래그 → 초록 사각형, 양수 %
- [ ] 아래로 드래그 → 빨강 사각형, 음수 %
- [ ] 마우스 놓아도 사각형 유지
- [ ] Shift 없이 차트 클릭 → 사각형 사라짐

- [ ] **Step 4: 커밋**

```bash
git add src/shift-drag-measure/ src/general/handler.ts src/index.ts lightweight_charts/js/
git commit -m "feat: add shift+drag price measure overlay"
```
