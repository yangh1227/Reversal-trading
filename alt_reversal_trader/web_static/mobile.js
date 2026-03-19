let csrfToken = null;
let chart = null;
let candleSeries = null;
let lineSeries = {};
let currentChartKey = "";
let dashboardTimer = null;
let chartTimer = null;

const els = {
  loginView: document.getElementById("login-view"),
  appView: document.getElementById("app-view"),
  loginUsername: document.getElementById("login-username"),
  loginPassword: document.getElementById("login-password"),
  loginSubmit: document.getElementById("login-submit"),
  loginError: document.getElementById("login-error"),
  serverUrls: document.getElementById("server-urls"),
  equityValue: document.getElementById("equity-value"),
  availableValue: document.getElementById("available-value"),
  currentSymbol: document.getElementById("current-symbol"),
  currentInterval: document.getElementById("current-interval"),
  barCountdown: document.getElementById("bar-countdown"),
  signalText: document.getElementById("signal-text"),
  favorableCount: document.getElementById("favorable-count"),
  favorableList: document.getElementById("favorable-list"),
  optimizedList: document.getElementById("optimized-list"),
  positionsList: document.getElementById("positions-list"),
  closeAllButton: document.getElementById("close-all-button"),
  autoTradeToggle: document.getElementById("auto-trade-toggle"),
  chartContainer: document.getElementById("chart-container"),
  refreshButton: document.getElementById("refresh-button"),
  logoutButton: document.getElementById("logout-button"),
  simpleAmount: document.getElementById("simple-amount"),
  modeCompound: document.getElementById("mode-compound"),
  modeSimple: document.getElementById("mode-simple"),
  compoundOrders: document.getElementById("compound-orders"),
  simpleOrders: document.getElementById("simple-orders"),
  simpleLong: document.getElementById("simple-long"),
  simpleShort: document.getElementById("simple-short"),
};

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

function initChart() {
  if (chart) return;
  chart = LightweightCharts.createChart(els.chartContainer, {
    autoSize: true,
    layout: { background: { color: "#0b1220" }, textColor: "#d1d5db" },
    grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: "#1f2937" },
    timeScale: { borderColor: "#1f2937", timeVisible: true, secondsVisible: false },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#d1d5db",
    downColor: "#d1d5db",
    borderVisible: false,
    wickUpColor: "#d1d5db",
    wickDownColor: "#d1d5db",
  });
  lineSeries = {
    supertrend: chart.addLineSeries({ color: "#00d2ff", lineWidth: 2 }),
    zone2: chart.addLineSeries({ color: "#d4a62a", lineWidth: 1 }),
    zone3: chart.addLineSeries({ color: "#d36b6b", lineWidth: 1 }),
    emaFast: chart.addLineSeries({ color: "#9ed8f2", lineWidth: 1 }),
    emaSlow: chart.addLineSeries({ color: "#e3d995", lineWidth: 1 }),
  };
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

async function refreshChart(force = false) {
  const dashboard = window.__dashboardState;
  if (!dashboard?.current?.symbol) return;
  const chartKey = `${dashboard.current.symbol}:${dashboard.current.interval}:${dashboard.current.chartVersion || ""}`;
  if (!force && chartKey === currentChartKey) return;
  initChart();
  const payload = await api("/api/chart/current");
  if (!payload.ready) return;
  currentChartKey = chartKey;
  candleSeries.setData(payload.candles.map((c) => ({ time: toUnix(c.time), open: c.open, high: c.high, low: c.low, close: c.close })));
  lineSeries.supertrend.setData(payload.indicators.supertrend.filter((x) => x.value !== null).map((x) => ({ time: toUnix(x.time), value: x.value })));
  lineSeries.zone2.setData(payload.indicators.zone2.filter((x) => x.value !== null).map((x) => ({ time: toUnix(x.time), value: x.value })));
  lineSeries.zone3.setData(payload.indicators.zone3.filter((x) => x.value !== null).map((x) => ({ time: toUnix(x.time), value: x.value })));
  lineSeries.emaFast.setData(payload.indicators.emaFast.filter((x) => x.value !== null).map((x) => ({ time: toUnix(x.time), value: x.value })));
  lineSeries.emaSlow.setData(payload.indicators.emaSlow.filter((x) => x.value !== null).map((x) => ({ time: toUnix(x.time), value: x.value })));
  candleSeries.setMarkers(payload.markers.filter((m) => m.time).map((m) => ({
    time: toUnix(m.time),
    position: markerPosition(m.position),
    shape: markerShape(m.shape),
    color: m.color,
    text: m.text || "",
  })));
  chart.timeScale().fitContent();
}

function renderFavorable(symbols) {
  els.favorableList.innerHTML = "";
  els.favorableCount.textContent = String(symbols.length);
  if (!symbols.length) {
    els.favorableList.innerHTML = '<span class="muted">현재 없음</span>';
    return;
  }
  symbols.forEach((symbol) => {
    const chip = document.createElement("span");
    chip.className = "chip symbol";
    chip.textContent = symbol;
    chip.onclick = () => selectSymbol(symbol);
    els.favorableList.appendChild(chip);
  });
}

function renderOptimized(items) {
  els.optimizedList.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = `list-item${item.favorable ? " favorable" : ""}`;
    row.innerHTML = `
      <div class="list-title">
        <strong>${item.symbol}</strong>
        <span>${item.interval}</span>
      </div>
      <div class="list-meta">Score ${item.score} | Return ${item.returnPct}% | MDD ${item.mddPct}% | Trades ${item.trades}</div>
      <div class="list-meta">${item.currentPrice ? `현재가 ${item.currentPrice}` : "현재가 -"}</div>
      <div class="list-actions">
        <button class="ghost">차트 보기</button>
      </div>
    `;
    row.querySelector("button").onclick = () => selectSymbol(item.symbol, item.interval);
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
    row.className = "list-item";
    row.innerHTML = `
      <div class="list-title"><strong>${item.symbol}</strong><span>${item.side} · ${item.leverage}</span></div>
      <div class="list-meta">진입 ${item.entryPrice} | 금액 ${item.amountUsdt} USDT</div>
      <div class="list-meta">UPnL ${item.upnl} | 수익률 ${item.returnPct}%</div>
      <div class="list-actions">
        <button class="danger">청산</button>
        <label class="switch-wrap"><span>자동청산</span><input type="checkbox" ${item.autoCloseEnabled ? "checked" : ""}></label>
      </div>
    `;
    row.querySelector("button").onclick = () => closePosition(item.symbol);
    row.querySelector('input[type="checkbox"]').onchange = (event) => toggleAutoClose(item.symbol, event.target.checked);
    els.positionsList.appendChild(row);
  });
}

async function refreshDashboard(forceChart = false) {
  const state = await api("/api/dashboard");
  window.__dashboardState = state;
  els.serverUrls.textContent = state.serverUrls.join("  |  ");
  els.equityValue.textContent = state.balance.equity == null ? "-" : `${state.balance.equity.toFixed(2)} USDT`;
  els.availableValue.textContent = state.balance.available == null ? "-" : `${state.balance.available.toFixed(2)} USDT`;
  els.currentSymbol.textContent = state.current.symbol || "차트 로드 중";
  els.currentInterval.textContent = state.current.interval || "-";
  els.barCountdown.textContent = state.current.countdown || "-";
  els.signalText.textContent = state.current.signalText || "";
  els.autoTradeToggle.checked = !!state.autoTradeEnabled;
  els.simpleAmount.value = state.simpleOrderAmount ?? 50;
  renderFavorable(state.favorableSymbols || []);
  renderOptimized(state.optimized || []);
  renderPositions(state.positions || []);
  await refreshChart(forceChart);
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
    startPolling();
  } catch (error) {
    els.loginError.textContent = error.message;
  }
}

async function bootstrap() {
  const payload = await api("/api/me");
  if (!payload.authenticated) {
    showLogin(true);
    return;
  }
  csrfToken = payload.csrfToken;
  showLogin(false);
  await refreshDashboard(true);
  startPolling();
}

function startPolling() {
  stopPolling();
  dashboardTimer = setInterval(() => refreshDashboard(false).catch(console.error), 2000);
  chartTimer = setInterval(() => refreshChart(false).catch(console.error), 5000);
}

function stopPolling() {
  if (dashboardTimer) clearInterval(dashboardTimer);
  if (chartTimer) clearInterval(chartTimer);
  dashboardTimer = null;
  chartTimer = null;
}

async function logout() {
  await api("/api/logout", { method: "POST" });
  stopPolling();
  csrfToken = null;
  showLogin(true);
}

async function selectSymbol(symbol, interval = "") {
  await api("/api/chart/select", {
    method: "POST",
    body: JSON.stringify({ symbol, interval }),
  });
  currentChartKey = "";
  setTimeout(() => refreshDashboard(true).catch(console.error), 400);
}

async function submitFractional(side, fraction) {
  const current = window.__dashboardState?.current || {};
  await api("/api/order/fractional", {
    method: "POST",
    body: JSON.stringify({
      symbol: current.symbol,
      interval: current.interval,
      side,
      fraction,
    }),
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

function setOrderMode(mode) {
  els.modeCompound.classList.toggle("active", mode === "compound");
  els.modeSimple.classList.toggle("active", mode === "simple");
  els.compoundOrders.classList.toggle("hidden", mode !== "compound");
  els.simpleOrders.classList.toggle("hidden", mode !== "simple");
}

document.querySelectorAll("[data-fraction]").forEach((button) => {
  button.addEventListener("click", () => submitFractional(button.dataset.side, Number(button.dataset.fraction)).catch((error) => alert(error.message)));
});
els.simpleLong.addEventListener("click", () => submitSimple("BUY").catch((error) => alert(error.message)));
els.simpleShort.addEventListener("click", () => submitSimple("SELL").catch((error) => alert(error.message)));
els.closeAllButton.addEventListener("click", () => closeAll().catch((error) => alert(error.message)));
els.autoTradeToggle.addEventListener("change", (event) => toggleAutoTrade(event.target.checked).catch((error) => alert(error.message)));
els.modeCompound.addEventListener("click", () => setOrderMode("compound"));
els.modeSimple.addEventListener("click", () => setOrderMode("simple"));
els.refreshButton.addEventListener("click", () => refreshDashboard(true).catch((error) => alert(error.message)));
els.logoutButton.addEventListener("click", () => logout().catch((error) => alert(error.message)));
els.loginSubmit.addEventListener("click", () => login());

bootstrap().catch((error) => {
  els.loginError.textContent = error.message;
  showLogin(true);
});
