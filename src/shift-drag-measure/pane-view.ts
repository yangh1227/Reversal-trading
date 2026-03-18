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
