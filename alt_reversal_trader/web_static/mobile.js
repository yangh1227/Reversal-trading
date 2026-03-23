let csrfToken = null;
let chart = null;
let candleSeries = null;
let lineSeries = {};
let entryPriceLine = null;
let currentPriceLine = null;
let currentChartKey = "";
let dashboardTimer = null;
let chartTimer = null;
let liveSocket = null;
let liveSocketRetryTimer = null;
let countdownTimer = null;
let countdownDeadlineMs = null;
let foregroundSyncTimer = null;
let recoverUiTimer = null;
let recoveryToastCooldownUntil = 0;
let autoTradeTogglePending = false;
let orderActionPending = false;

const els = {
  loginView: document.getElementById("login-view"),
  appView: document.getElementById("app-view"),
  loginUsername: document.getElementById("login-username"),
  loginPassword: document.getElementById("login-password"),
  loginSubmit: document.getElementById("login-submit"),
  loginError: document.getElementById("login-error"),
  equityValue: document.getElementById("equity-value"),
  availableValue: document.getElementById("available-value"),
  currentSymbol: document.getElementById("current-symbol"),
  currentInterval: document.getElementById("current-interval"),
  barCountdown: document.getElementById("bar-countdown"),
  favorableCount: document.getElementById("favorable-count"),
  favorableList: document.getElementById("favorable-list"),
  optimizedList: document.getElementById("optimized-list"),
  positionsList: document.getElementById("positions-list"),
  closeAllButton: document.getElementById("close-all-button"),
  autoTradeToggle: document.getElementById("auto-trade-toggle"),
  chartContainer: document.getElementById("chart-container"),
  chartLoading: document.getElementById("chart-loading"),
  refreshButton: document.getElementById("refresh-button"),
  logoutButton: document.getElementById("logout-button"),
  simpleAmount: document.getElementById("simple-amount"),
  modeCompound: document.getElementById("mode-compound"),
  modeSimple: document.getElementById("mode-simple"),
  compoundOrders: document.getElementById("compound-orders"),
  simpleOrders: document.getElementById("simple-orders"),
  simpleLong: document.getElementById("simple-long"),
  simpleShort: document.getElementById("simple-short"),
  toastContainer: document.getElementById("toast-container"),
};

function showToast(message, type = "info", duration = 3000) {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  els.toastContainer.appendChild(toast);
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add("show")));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (csrfToken && options.method && options.method !== "GET") {
    headers.set("X-CSRF-Token", csrfToken);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || payload.message || "요청 실패");
    }
    return payload;
  });
}

function showChartLoading(show) {
  if (els.chartLoading) {
    els.chartLoading.style.display = show ? "flex" : "none";
  }
}

function setOrderActionPending(pending) {
  orderActionPending = !!pending;
  [
    els.closeAllButton,
    els.simpleLong,
    els.simpleShort,
    ...document.querySelectorAll("[data-fraction]"),
  ].forEach((element) => {
    if (element) {
      element.disabled = orderActionPending;
    }
  });
}

function initChart() {
  if (chart) {
    return;
  }
  chart = LightweightCharts.createChart(els.chartContainer, {
    autoSize: true,
    layout: { background: { color: "#0b1220" }, textColor: "#d1d5db" },
    grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    localization: { priceFormatter: (price) => formatCompactPrice(price, "") },
    rightPriceScale: {
      borderColor: "#1f2937",
      minimumWidth: 0,
      entireTextOnly: false,
      scaleMargins: { top: 0.08, bottom: 0.08 },
    },
    timeScale: { borderColor: "#1f2937", timeVisible: true, secondsVisible: false },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#a3acb7",
    downColor: "rgba(0, 0, 0, 0)",
    borderVisible: true,
    borderUpColor: "#a3acb7",
    borderDownColor: "#a3acb7",
    wickUpColor: "#a3acb7",
    wickDownColor: "#a3acb7",
    priceLineVisible: false,
    lastValueVisible: false,
  });
  lineSeries = {
    supertrend: chart.addLineSeries({ color: "#00d2ff", lineWidth: 2, priceLineVisible: false, lastValueVisible: false }),
    zone2: chart.addLineSeries({ color: "#d4a62a", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }),
    zone3: chart.addLineSeries({ color: "#d36b6b", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }),
    emaFast: chart.addLineSeries({ color: "#9ed8f2", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }),
    emaSlow: chart.addLineSeries({ color: "#e3d995", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }),
  };
}

function parsePrice(value) {
  const parsed = parseFloat(String(value ?? "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function formatCompactPrice(value, fallback = "-") {
  const parsed = parsePrice(value);
  if (parsed == null) {
    return fallback;
  }
  return parsed.toFixed(8).replace(/\.?0+$/, "");
}

function setPriceLine(target, options) {
  if (!candleSeries) {
    return target;
  }
  if (!options || !Number.isFinite(options.price)) {
    if (target) {
      candleSeries.removePriceLine(target);
    }
    return null;
  }
  if (!target) {
    return candleSeries.createPriceLine(options);
  }
  target.applyOptions(options);
  return target;
}

function updateReferenceLines(payload) {
  const currentSymbol = window.__dashboardState?.current?.symbol || "";
  const positions = window.__dashboardState?.positions || [];
  const currentPosition = positions.find((item) => item.symbol === currentSymbol) || null;
  const entryPrice = parsePrice(currentPosition?.entryPrice);
  const candles = payload?.candles || [];
  const latestCandle = candles.length ? candles[candles.length - 1] : null;
  const currentPrice = parsePrice(latestCandle?.close);

  entryPriceLine = setPriceLine(entryPriceLine, entryPrice == null ? null : {
    price: entryPrice,
    color: "#24fc0c",
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
  });
  currentPriceLine = setPriceLine(currentPriceLine, currentPrice == null ? null : {
    price: currentPrice,
    color: "#24fc0c",
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    axisLabelVisible: true,
  });
}

function queueUiRecovery(delayMs = 1500) {
  if (recoverUiTimer) {
    clearTimeout(recoverUiTimer);
    recoverUiTimer = null;
  }
  recoverUiTimer = setTimeout(() => {
    recoverUiTimer = null;
    if (els.appView.classList.contains("hidden")) {
      return;
    }
    if (Date.now() >= recoveryToastCooldownUntil) {
      recoveryToastCooldownUntil = Date.now() + 4000;
      showToast("연결 복구 중...", "info", 2000);
    }
    queueForegroundSync(0);
  }, Math.max(0, Number(delayMs) || 0));
}

function applyPriceFormat(format) {
  if (!format || !candleSeries) {
    return;
  }
  const precision = Number(format.precision);
  const minMove = Number(format.minMove);
  if (!Number.isFinite(precision) || !Number.isFinite(minMove) || precision < 0 || minMove <= 0) {
    return;
  }
  const fmt = (price) => {
    const p = parsePrice(price);
    return p == null ? "" : p.toFixed(precision).replace(/\.?0+$/, "");
  };
  const priceFormat = { type: "custom", precision, minMove, formatter: fmt };
  chart.applyOptions({ localization: { priceFormatter: fmt } });
  candleSeries.applyOptions({ priceFormat });
  Object.values(lineSeries).forEach((series) => series.applyOptions({ priceFormat }));
}

function toUnix(timeText) {
  return Math.floor(new Date(timeText).getTime() / 1000);
}

function markerShape(shape) {
  if (shape === "arrow_up") return "arrowUp";
  if (shape === "arrow_down") return "arrowDown";
  return "circle";
}

function markerPosition(position) {
  return position === "below" ? "belowBar" : "aboveBar";
}

function applyChartPayload(payload, options = {}) {
  if (!payload || !payload.ready) {
    return;
  }
  initChart();
  applyPriceFormat(payload.priceFormat);
  candleSeries.setData(
    (payload.candles || []).map((candle) => ({
      time: toUnix(candle.time),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }))
  );
  lineSeries.supertrend.setData(
    (payload.indicators?.supertrend || [])
      .filter((item) => item.value !== null)
      .map((item) => ({ time: toUnix(item.time), value: item.value }))
  );
  lineSeries.zone2.setData(
    (payload.indicators?.zone2 || [])
      .filter((item) => item.value !== null)
      .map((item) => ({ time: toUnix(item.time), value: item.value }))
  );
  lineSeries.zone3.setData(
    (payload.indicators?.zone3 || [])
      .filter((item) => item.value !== null)
      .map((item) => ({ time: toUnix(item.time), value: item.value }))
  );
  lineSeries.emaFast.setData(
    (payload.indicators?.emaFast || [])
      .filter((item) => item.value !== null)
      .map((item) => ({ time: toUnix(item.time), value: item.value }))
  );
  lineSeries.emaSlow.setData(
    (payload.indicators?.emaSlow || [])
      .filter((item) => item.value !== null)
      .map((item) => ({ time: toUnix(item.time), value: item.value }))
  );
  candleSeries.setMarkers(
    (payload.markers || [])
      .filter((marker) => marker.time)
      .map((marker) => ({
        time: toUnix(marker.time),
        position: markerPosition(marker.position),
        shape: markerShape(marker.shape),
        color: marker.color,
        text: marker.text || "",
      }))
      .sort((a, b) => a.time - b.time)
  );
  updateReferenceLines(payload);
  chart.resize(els.chartContainer.clientWidth, els.chartContainer.clientHeight, true);
  if (options.fitContent) {
    chart.timeScale().fitContent();
  }
}

async function refreshChart(force = false) {
  const dashboard = window.__dashboardState;
  if (!dashboard?.current?.symbol) {
    return;
  }
  const chartKey = `${dashboard.current.symbol}:${dashboard.current.interval}:${dashboard.current.chartVersion || ""}`;
  if (!force && chartKey === currentChartKey) {
    return;
  }
  initChart();
  showChartLoading(true);
  try {
    const payload = await api("/api/chart/current");
    if (!payload.ready) {
      return;
    }
    currentChartKey = chartKey;
    applyChartPayload(payload, { fitContent: true });
  } finally {
    showChartLoading(false);
  }
}

function renderCountdown(deadlineMs) {
  if (!deadlineMs || !Number.isFinite(deadlineMs)) {
    els.barCountdown.textContent = "-";
    return;
  }
  const remainingSeconds = Math.max(0, Math.floor((deadlineMs - Date.now()) / 1000));
  const hours = Math.floor(remainingSeconds / 3600);
  const minutes = Math.floor((remainingSeconds % 3600) / 60);
  const seconds = remainingSeconds % 60;
  els.barCountdown.textContent = hours > 0
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function startCountdown(deadlineMs) {
  countdownDeadlineMs = Number(deadlineMs) || null;
  if (countdownTimer) {
    clearInterval(countdownTimer);
    countdownTimer = null;
  }
  renderCountdown(countdownDeadlineMs);
  if (!countdownDeadlineMs) {
    return;
  }
  countdownTimer = setInterval(() => renderCountdown(countdownDeadlineMs), 1000);
}

function renderFavorable(favorableSymbols, signalSymbols) {
  els.favorableList.innerHTML = "";
  const total = favorableSymbols.length + signalSymbols.length;
  els.favorableCount.textContent = String(total);
  if (!total) {
    els.favorableList.innerHTML = '<span class="muted">현재 없음</span>';
    return;
  }
  favorableSymbols.forEach((symbol) => {
    const chip = document.createElement("span");
    chip.className = "chip symbol";
    chip.textContent = symbol;
    chip.onclick = () => selectSymbol(symbol);
    els.favorableList.appendChild(chip);
  });
  signalSymbols.forEach((symbol) => {
    if (favorableSymbols.includes(symbol)) return;
    const chip = document.createElement("span");
    chip.className = "chip symbol signal-chip";
    chip.textContent = `⚡ ${symbol}`;
    chip.onclick = () => selectSymbol(symbol);
    els.favorableList.appendChild(chip);
  });
}

function parseUpnlNum(upnl) {
  const value = parseFloat(String(upnl).replace(/[^\d.\-+]/g, ""));
  return Number.isNaN(value) ? 0 : value;
}

function renderOptimized(items) {
  els.optimizedList.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = `list-item${item.favorable ? " favorable" : ""}`;
    row.innerHTML = `
      <div class="list-title">
        <strong>${item.symbol}</strong>
        <span class="interval-badge">${item.interval}</span>
      </div>
      <div class="list-meta">Score ${item.score} · Return ${item.returnPct}% · MDD ${item.mddPct}% · Trades ${item.trades}</div>
      <div class="list-meta">${item.currentPrice ? `현재가 ${formatCompactPrice(item.currentPrice)}` : "현재가 -"}</div>
      <div class="list-actions">
        <button class="ghost chart-view-btn">차트 보기</button>
      </div>
    `;
    row.querySelector("button").onclick = () => {
      selectSymbol(item.symbol, item.interval);
      setTimeout(() => els.chartContainer.closest(".card")?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
    };
    els.optimizedList.appendChild(row);
  });
}

function renderPositions(items) {
  els.positionsList.innerHTML = "";
  if (!items.length) {
    els.positionsList.innerHTML = '<div class="muted">보유 포지션이 없습니다.</div>';
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    const isLong = /BUY|LONG/i.test(String(item.side));
    const sideKey = isLong ? "long" : "short";
    const sideLabel = isLong ? "LONG" : "SHORT";
    const upnlNum = parseUpnlNum(item.upnl);
    const upnlCls = upnlNum >= 0 ? "positive" : "negative";

    row.className = `list-item ${sideKey}-pos`;
    row.style.cursor = "pointer";
    row.innerHTML = `
      <div class="list-title">
        <strong>${item.symbol}</strong>
        <span class="side-badge ${sideKey}">${sideLabel} ${item.leverage}</span>
        ${item.interval ? `<span class="pos-interval">${item.interval}</span>` : ""}
      </div>
      <div class="list-meta">진입 ${formatCompactPrice(item.entryPrice)} · ${item.amountUsdt} USDT</div>
      <div class="pos-pnl-row">
        <span class="return-pct ${upnlCls}">${item.returnPct >= 0 ? "+" : ""}${item.returnPct}%</span>
        <span class="upnl-secondary">${item.upnl >= 0 ? "+" : ""}${item.upnl} USDT</span>
      </div>
      <div class="list-actions">
        <button class="danger">청산</button>
        <label class="switch-wrap">
          <span>자동청산</span>
          <span class="toggle">
            <input type="checkbox" ${item.autoCloseEnabled ? "checked" : ""}>
            <span class="toggle-track"></span>
          </span>
        </label>
      </div>
    `;
    row.addEventListener("click", () => selectSymbol(item.symbol));
    const closeButton = row.querySelector("button");
    const autoCloseCheckbox = row.querySelector('input[type="checkbox"]');
    closeButton.disabled = orderActionPending;
    autoCloseCheckbox.disabled = orderActionPending;
    closeButton.onclick = (event) => {
      event.stopPropagation();
      runOrderAction(() => closePosition(item.symbol), "포지션 청산 접수", "포지션 청산 요청 실패").catch(() => {});
    };
    autoCloseCheckbox.onchange = (event) => {
      event.stopPropagation();
      runOrderAction(
        () => toggleAutoClose(item.symbol, event.target.checked),
        event.target.checked ? "자동청산 ON 반영" : "자동청산 OFF 반영",
        "자동청산 설정 실패",
        { syncDelayMs: 0, showQueuedToast: false }
      ).catch(() => {
        event.target.checked = !event.target.checked;
      });
    };
    els.positionsList.appendChild(row);
  });
}

function applyDashboardState(state) {
  window.__dashboardState = state;
  els.equityValue.textContent = state.balance.equity == null ? "-" : `${state.balance.equity.toFixed(2)} USDT`;
  els.availableValue.textContent = state.balance.available == null ? "-" : `${state.balance.available.toFixed(2)} USDT`;
  els.currentSymbol.textContent = state.current.symbol || "차트 로드 중";
  els.currentInterval.textContent = state.current.interval || "-";
  startCountdown(state.current.barCloseDeadlineMs);

  if (!autoTradeTogglePending) {
    els.autoTradeToggle.checked = !!state.autoTradeEnabled;
  }
  els.autoTradeToggle.disabled = autoTradeTogglePending;
  els.simpleAmount.value = state.simpleOrderAmount ?? 50;

  renderFavorable(state.favorableSymbols || [], state.signalSymbols || []);
  renderOptimized(state.optimized || []);
  renderPositions(state.positions || []);
}

async function refreshDashboard(forceChart = false) {
  const state = await api("/api/dashboard");
  applyDashboardState(state);
  await refreshChart(forceChart);
}

async function runOrderAction(action, successMessage, fallbackErrorMessage, options = {}) {
  const syncDelayMs = Number(options.syncDelayMs ?? 600) || 0;
  const showQueuedToast = options.showQueuedToast !== false;
  if (orderActionPending) {
    return false;
  }
  setOrderActionPending(true);
  try {
    const payload = await action();
    if (showQueuedToast) {
      showToast(successMessage || "요청 접수", "success");
    }
    queueForegroundSync(syncDelayMs);
    return payload;
  } catch (error) {
    queueUiRecovery(0);
    queueForegroundSync(0);
    showToast(error?.message || fallbackErrorMessage || "요청 실패", "error", 3500);
    throw error;
  } finally {
    setTimeout(() => {
      setOrderActionPending(false);
      refreshDashboard(false).catch(() => {});
    }, 250);
  }
}

function connectLiveSocket() {
  if (liveSocket && (liveSocket.readyState === WebSocket.OPEN || liveSocket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  if (liveSocketRetryTimer) {
    clearTimeout(liveSocketRetryTimer);
    liveSocketRetryTimer = null;
  }
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  liveSocket = new WebSocket(`${scheme}://${window.location.host}/ws/live`);

  liveSocket.onmessage = (event) => {
    let payload = {};
    try {
      payload = JSON.parse(event.data || "{}");
    } catch (_error) {
      queueUiRecovery(250);
      return;
    }
    if (payload.type === "dashboard" && payload.data) {
      applyDashboardState(payload.data);
      return;
    }
    if (payload.type === "chart" && payload.data) {
      currentChartKey = `${payload.data.symbol || ""}:${payload.data.interval || ""}:${window.__dashboardState?.current?.chartVersion || ""}`;
      showChartLoading(false);
      applyChartPayload(payload.data, { fitContent: false });
    }
  };

  liveSocket.onclose = () => {
    liveSocket = null;
    queueUiRecovery(250);
    liveSocketRetryTimer = setTimeout(() => {
      if (!els.appView.classList.contains("hidden")) {
        connectLiveSocket();
      }
    }, 2000);
  };

  liveSocket.onerror = () => {
    if (liveSocket) {
      liveSocket.close();
    }
  };
}

function queueForegroundSync(delayMs = 0) {
  if (foregroundSyncTimer) {
    clearTimeout(foregroundSyncTimer);
    foregroundSyncTimer = null;
  }
  foregroundSyncTimer = setTimeout(() => {
    foregroundSyncTimer = null;
    if (els.appView.classList.contains("hidden")) {
      return;
    }
    currentChartKey = "";
    connectLiveSocket();
    refreshDashboard(true).catch(console.error);
  }, Math.max(0, Number(delayMs) || 0));
}

function showLogin(show) {
  els.loginView.classList.toggle("hidden", !show);
  els.appView.classList.toggle("hidden", show);
}

async function login() {
  els.loginError.textContent = "";
  try {
    const payload = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({
        username: els.loginUsername.value,
        password: els.loginPassword.value,
      }),
    });
    csrfToken = payload.csrfToken;
    showLogin(false);
    await refreshDashboard(true);
    connectLiveSocket();
  } catch (error) {
    els.loginError.textContent = error.message;
  }
}

async function bootstrap() {
  try {
    const payload = await api("/api/me");
    if (!payload.authenticated) {
      showLogin(true);
      return;
    }
    csrfToken = payload.csrfToken;
    showLogin(false);
    await refreshDashboard(true);
    connectLiveSocket();
  } catch {
    showLogin(true);
  }
}

function startPolling() {
  stopPolling();
  dashboardTimer = setInterval(() => refreshDashboard(false).catch(console.error), 2000);
  chartTimer = setInterval(() => refreshChart(false).catch(console.error), 5000);
}

function stopPolling() {
  if (dashboardTimer) clearInterval(dashboardTimer);
  if (chartTimer) clearInterval(chartTimer);
  if (countdownTimer) clearInterval(countdownTimer);
  if (liveSocketRetryTimer) clearTimeout(liveSocketRetryTimer);
  if (foregroundSyncTimer) clearTimeout(foregroundSyncTimer);
  if (recoverUiTimer) clearTimeout(recoverUiTimer);
  if (liveSocket) {
    liveSocket.onclose = null;
    liveSocket.close();
  }
  dashboardTimer = null;
  chartTimer = null;
  countdownTimer = null;
  liveSocketRetryTimer = null;
  foregroundSyncTimer = null;
  recoverUiTimer = null;
  liveSocket = null;
}

async function logout() {
  await api("/api/logout", { method: "POST" });
  stopPolling();
  csrfToken = null;
  currentChartKey = "";
  countdownDeadlineMs = null;
  showLogin(true);
}

async function selectSymbol(symbol, interval = "") {
  await api("/api/chart/select", {
    method: "POST",
    body: JSON.stringify({ symbol, interval }),
  });
  currentChartKey = "";
  showChartLoading(true);
  setTimeout(() => refreshDashboard(true).catch((error) => showToast(error.message, "error")), 250);
}

async function submitFractional(side, fraction) {
  const current = window.__dashboardState?.current || {};
  await api("/api/order/fractional", {
    method: "POST",
    body: JSON.stringify({ symbol: current.symbol, interval: current.interval, side, fraction }),
  });
}

async function submitSimple(side) {
  const current = window.__dashboardState?.current || {};
  await api("/api/order/simple", {
    method: "POST",
    body: JSON.stringify({
      symbol: current.symbol,
      interval: current.interval,
      side,
      amount: Number(els.simpleAmount.value || 0),
    }),
  });
}

async function closeAll() {
  await api("/api/positions/close-all", { method: "POST" });
}

async function closePosition(symbol) {
  await api(`/api/positions/${encodeURIComponent(symbol)}/close`, { method: "POST" });
}

async function toggleAutoClose(symbol, enabled) {
  await api(`/api/positions/${encodeURIComponent(symbol)}/auto-close`, {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

async function toggleAutoTrade(enabled) {
  await api("/api/auto-trade", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

async function submitAutoTradeToggle(enabled) {
  const currentEnabled = !!window.__dashboardState?.autoTradeEnabled;
  if (autoTradeTogglePending) {
    els.autoTradeToggle.checked = currentEnabled;
    return;
  }
  if (enabled === currentEnabled) {
    els.autoTradeToggle.checked = currentEnabled;
    return;
  }
  autoTradeTogglePending = true;
  els.autoTradeToggle.disabled = true;
  els.autoTradeToggle.checked = currentEnabled;
  try {
    await toggleAutoTrade(enabled);
    await refreshDashboard(true);
    const actualEnabled = !!window.__dashboardState?.autoTradeEnabled;
    els.autoTradeToggle.checked = actualEnabled;
    showToast(actualEnabled ? "자동매매 활성화" : "자동매매 비활성화");
  } catch (error) {
    els.autoTradeToggle.checked = currentEnabled;
    showToast(error.message, "error");
  } finally {
    autoTradeTogglePending = false;
    els.autoTradeToggle.disabled = false;
  }
}

function setOrderMode(mode) {
  els.modeCompound.classList.toggle("active", mode === "compound");
  els.modeSimple.classList.toggle("active", mode === "simple");
  els.compoundOrders.classList.toggle("hidden", mode !== "compound");
  els.simpleOrders.classList.toggle("hidden", mode !== "simple");
}

document.querySelectorAll("[data-fraction]").forEach((button) => {
  button.addEventListener("click", () => {
    runOrderAction(
      () => submitFractional(button.dataset.side, Number(button.dataset.fraction)),
      "주문 접수",
      "주문 요청 실패"
    ).catch(() => {});
  });
});

els.simpleLong.addEventListener("click", () =>
  runOrderAction(() => submitSimple("BUY"), "LONG 주문 접수", "LONG 주문 요청 실패").catch(() => {})
);

els.simpleShort.addEventListener("click", () =>
  runOrderAction(() => submitSimple("SELL"), "SHORT 주문 접수", "SHORT 주문 요청 실패").catch(() => {})
);

els.closeAllButton.addEventListener("click", () => {
  if (!confirm("모든 포지션을 청산하시겠습니까?")) {
    return;
  }
  runOrderAction(() => closeAll(), "전체 청산 접수", "전체 청산 요청 실패").catch(() => {});
});

els.autoTradeToggle.addEventListener("change", (event) => {
  submitAutoTradeToggle(event.target.checked).catch((error) => showToast(error.message, "error"));
});

els.modeCompound.addEventListener("click", () => setOrderMode("compound"));
els.modeSimple.addEventListener("click", () => setOrderMode("simple"));

els.refreshButton.addEventListener("click", () =>
  refreshDashboard(true).catch((error) => showToast(error.message, "error"))
);

els.logoutButton.addEventListener("click", () =>
  logout().catch((error) => showToast(error.message, "error"))
);

els.loginSubmit.addEventListener("click", () => login());

[els.loginUsername, els.loginPassword].forEach((element) =>
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      login();
    }
  })
);

bootstrap().catch((error) => {
  els.loginError.textContent = error.message;
  showLogin(true);
});

window.addEventListener("error", () => {
  queueUiRecovery(250);
});

window.addEventListener("unhandledrejection", () => {
  queueUiRecovery(250);
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    queueForegroundSync(0);
  }
});

window.addEventListener("pageshow", () => {
  queueForegroundSync(0);
});

window.addEventListener("focus", () => {
  queueForegroundSync(100);
});
