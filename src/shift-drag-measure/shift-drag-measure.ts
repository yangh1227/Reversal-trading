// src/shift-drag-measure/shift-drag-measure.ts
import { ISeriesPrimitivePaneView } from "lightweight-charts";
import { PluginBase } from "../plugin-base";
import { ShiftDragMeasurePaneView } from "./pane-view";

export class ShiftDragMeasure extends PluginBase {
    // Y 픽셀 좌표 직접 저장 (priceToCoordinate 변환 체인 제거)
    public startY: number | null = null;
    public endY: number | null = null;
    // 퍼센트 계산용 가격 (null이면 % 표시 생략)
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

    setPoints(startY: number, endY: number, startPrice: number | null, endPrice: number | null): void {
        this.startY = startY;
        this.endY = endY;
        this.startPrice = startPrice;
        this.endPrice = endPrice;
        this.requestUpdate();
    }

    clear(): void {
        this.startY = null;
        this.endY = null;
        this.startPrice = null;
        this.endPrice = null;
        this.requestUpdate();
    }

    isActive(): boolean {
        return this.startY !== null;
    }
}
