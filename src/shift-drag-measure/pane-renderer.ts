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
