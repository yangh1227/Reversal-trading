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
        // Y 좌표 직접 사용 (priceToCoordinate 변환 없음 — null 반환 문제 완전 제거)
        this._startY = this._source.startY;
        this._endY = this._source.endY;

        const start = this._source.startPrice;
        const end = this._source.endPrice;
        if (start !== null && end !== null && start !== 0) {
            this._pct = (end - start) / Math.abs(start) * 100;
        } else {
            this._pct = null;
        }
    }

    renderer(): ShiftDragMeasureRenderer {
        return new ShiftDragMeasureRenderer(this._startY, this._endY, this._pct);
    }
}
