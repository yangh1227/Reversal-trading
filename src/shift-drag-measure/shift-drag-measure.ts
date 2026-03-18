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
