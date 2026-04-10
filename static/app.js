const state = {
  dashboard: null,
  predictions: [],
  filteredPredictions: [],
  history: [],
  admin: null,
  selectedMatchId: null,
  detail: null,
  refreshTimer: null,
};

const elements = {
  schedulerStatus: document.getElementById("scheduler-status"),
  databaseStatus: document.getElementById("database-status"),
  lastRunAt: document.getElementById("last-run-at"),
  metricTotalMatches: document.getElementById("metric-total-matches"),
  metricPublished: document.getElementById("metric-published"),
  metricValueBets: document.getElementById("metric-value-bets"),
  metricRoi: document.getElementById("metric-roi"),
  metricHitRate: document.getElementById("metric-hit-rate"),
  metricProfit: document.getElementById("metric-profit"),
  refreshBtn: document.getElementById("refresh-btn"),
  searchInput: document.getElementById("search-input"),
  confidenceFilter: document.getElementById("confidence-filter"),
  valueOnlyToggle: document.getElementById("value-only-toggle"),
  predictionsList: document.getElementById("predictions-list"),
  detailBadge: document.getElementById("detail-badge"),
  detailCard: document.getElementById("detail-card"),
  historyList: document.getElementById("history-list"),
  historyCount: document.getElementById("history-count"),
  runNowBtn: document.getElementById("run-now-btn"),
  adminState: document.getElementById("admin-state"),
  adminInterval: document.getElementById("admin-interval"),
  adminLastStatus: document.getElementById("admin-last-status"),
  adminReport: document.getElementById("admin-report"),
  runsList: document.getElementById("runs-list"),
};

function badgeClass(status) {
  if (["success", "published", "ok"].includes(status)) return "badge success";
  if (["running", "queued"].includes(status)) return "badge running";
  if (["error", "failed"].includes(status)) return "badge danger";
  if (["warning", "degraded"].includes(status)) return "badge warning";
  return "badge neutral";
}

function shortDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("fr-FR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function numberPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function currency(value) {
  return `${Number(value || 0).toFixed(2)}$`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Erreur serveur");
  }
  return payload;
}

function renderDashboard() {
  const dashboard = state.dashboard;
  if (!dashboard) return;

  const scheduler = dashboard.scheduler || {};
  elements.schedulerStatus.textContent = scheduler.running ? "running" : scheduler.last_status || "idle";
  elements.databaseStatus.textContent = "ok";
  elements.lastRunAt.textContent = shortDate(scheduler.last_finished_at || scheduler.last_started_at);

  elements.metricTotalMatches.textContent = dashboard.today.total_matches;
  elements.metricPublished.textContent = dashboard.today.published_predictions;
  elements.metricValueBets.textContent = dashboard.today.value_bets;
  elements.metricRoi.textContent = `${Number(dashboard.performance.roi_pct || 0).toFixed(1)}%`;
  elements.metricHitRate.textContent = `${Number(dashboard.performance.hit_rate_pct || 0).toFixed(1)}%`;
  elements.metricProfit.textContent = currency(dashboard.performance.total_profit || 0);
}

function applyFilters() {
  const search = elements.searchInput.value.trim().toLowerCase();
  const confidence = elements.confidenceFilter.value;
  const valueOnly = elements.valueOnlyToggle.checked;

  state.filteredPredictions = state.predictions.filter((item) => {
    const haystack = `${item.home_name} ${item.away_name} ${item.league_name}`.toLowerCase();
    if (search && !haystack.includes(search)) return false;
    if (confidence !== "ALL" && item.confidence_level !== confidence) return false;
    if (valueOnly && !item.is_value_bet) return false;
    return true;
  });
}

function renderPredictionsList() {
  applyFilters();

  if (!state.filteredPredictions.length) {
    elements.predictionsList.innerHTML = `
      <div class="empty-state">
        Aucun pronostic publie ne correspond aux filtres actifs.
      </div>
    `;
    return;
  }

  elements.predictionsList.innerHTML = state.filteredPredictions.map((prediction) => {
    const isActive = prediction.match_id === state.selectedMatchId;
    return `
      <article class="prediction-card ${isActive ? "active" : ""}" data-match-id="${prediction.match_id}">
        <div class="prediction-topline">
          <div>
            <div class="prediction-title">${prediction.home_name} vs ${prediction.away_name}</div>
            <div class="prediction-meta">${prediction.league_name} · ${shortDate(prediction.match_date)}</div>
          </div>
          <span class="${badgeClass(prediction.confidence_level)}">${prediction.confidence_level || "LOW"}</span>
        </div>

        <div class="prediction-grid">
          <div class="prediction-chip">
            <span class="label">Pronostic</span>
            <strong>${prediction.recommendation}</strong>
          </div>
          <div class="prediction-chip">
            <span class="label">Cote</span>
            <strong>${Number(prediction.recommended_odd || 0).toFixed(2)}</strong>
          </div>
          <div class="prediction-chip">
            <span class="label">Edge</span>
            <strong>${Number(prediction.edge_pct || 0).toFixed(1)}%</strong>
          </div>
          <div class="prediction-chip">
            <span class="label">Value</span>
            <strong>${prediction.is_value_bet ? "VALUE" : "NO VALUE"}</strong>
          </div>
        </div>

        <p class="prediction-copy">${prediction.explanation || "Explication non disponible."}</p>
      </article>
    `;
  }).join("");

  document.querySelectorAll(".prediction-card").forEach((card) => {
    card.addEventListener("click", async () => {
      state.selectedMatchId = Number(card.dataset.matchId);
      renderPredictionsList();
      await loadPredictionDetail(state.selectedMatchId);
    });
  });
}

function renderDetail() {
  const detail = state.detail;
  if (!detail) {
    elements.detailBadge.className = "badge neutral";
    elements.detailBadge.textContent = "Aucune selection";
    elements.detailCard.className = "detail-card empty";
    elements.detailCard.innerHTML = `
      <h3>Aucun pronostic selectionne</h3>
      <p>Choisis un match publie dans la liste pour voir les probabilites, la value bet, l'explication et le Kelly.</p>
    `;
    return;
  }

  elements.detailBadge.className = badgeClass(detail.publication_status || "published");
  elements.detailBadge.textContent = detail.publication_status || "published";

  elements.detailCard.className = "detail-card";
  elements.detailCard.innerHTML = `
    <div class="detail-header">
      <div>
        <h3>${detail.home_name} vs ${detail.away_name}</h3>
        <p>${detail.league_name} · ${shortDate(detail.match_date)}</p>
      </div>
      <div class="${badgeClass(detail.confidence_level)}">${detail.confidence_level || "LOW"}</div>
    </div>

    <div class="probability-grid">
      <div class="probability-card">
        <span class="label">H1</span>
        <strong>${numberPct(detail.final_prob_h1)}</strong>
        <small>cote ${Number(detail.odds_h1 || 0).toFixed(2)}</small>
      </div>
      <div class="probability-card">
        <span class="label">H2</span>
        <strong>${numberPct(detail.final_prob_h2)}</strong>
        <small>cote ${Number(detail.odds_h2 || 0).toFixed(2)}</small>
      </div>
      <div class="probability-card">
        <span class="label">EQ</span>
        <strong>${numberPct(detail.final_prob_eq)}</strong>
        <small>cote ${Number(detail.odds_eq || 0).toFixed(2)}</small>
      </div>
    </div>

    <div class="detail-grid">
      <div class="detail-cell">
        <span class="label">Pronostic final</span>
        <strong>${detail.recommendation}</strong>
      </div>
      <div class="detail-cell">
        <span class="label">Value bet</span>
        <strong>${detail.is_value_bet ? "VALUE" : "NO VALUE"}</strong>
      </div>
      <div class="detail-cell">
        <span class="label">Edge</span>
        <strong>${Number(detail.edge_pct || 0).toFixed(2)}%</strong>
      </div>
      <div class="detail-cell">
        <span class="label">Mise Kelly</span>
        <strong>${currency(detail.suggested_stake)}</strong>
      </div>
      <div class="detail-cell">
        <span class="label">Facteur cle</span>
        <strong>${detail.key_factor || "N/A"}</strong>
      </div>
      <div class="detail-cell">
        <span class="label">Resultat reel</span>
        <strong>${detail.actual_result || detail.match_result || "N/A"}</strong>
      </div>
    </div>

    <article class="detail-note">
      <span class="label">Explication</span>
      <p>${detail.explanation || detail.llm_analysis || "Aucune explication disponible."}</p>
    </article>
  `;
}

function renderHistory() {
  elements.historyCount.textContent = `${state.history.length} lignes`;
  if (!state.history.length) {
    elements.historyList.innerHTML = `<div class="empty-state">Aucun pronostic historise et grade pour le moment.</div>`;
    return;
  }

  elements.historyList.innerHTML = state.history.map((entry) => `
    <article class="history-row">
      <div>
        <strong>${entry.home_name} vs ${entry.away_name}</strong>
        <div class="subtle">${entry.league_name} · ${shortDate(entry.match_date)}</div>
      </div>
      <div>${entry.recommendation}</div>
      <div>${entry.actual_result || "N/A"}</div>
      <div class="${entry.is_correct ? "up" : "down"}">${entry.is_correct ? "WIN" : "LOSS"}</div>
      <div>${currency(entry.profit_loss)}</div>
    </article>
  `).join("");
}

function renderAdmin() {
  const admin = state.admin;
  if (!admin) return;

  elements.adminState.textContent = admin.running ? "running" : "idle";
  elements.adminInterval.textContent = `${Math.round((admin.interval_seconds || 0) / 60)} min`;
  elements.adminLastStatus.textContent = admin.last_status || "-";

  if (admin.last_report) {
    elements.adminReport.className = "admin-report";
    elements.adminReport.textContent = JSON.stringify(admin.last_report, null, 2);
  } else {
    elements.adminReport.className = "admin-report empty-state";
    elements.adminReport.textContent = "Aucun rapport de cycle disponible pour le moment.";
  }

  const runs = admin.recent_runs || [];
  if (!runs.length) {
    elements.runsList.innerHTML = `<div class="empty-state">Aucun job scheduler historise.</div>`;
    return;
  }

  elements.runsList.innerHTML = runs.map((run) => `
    <article class="run-card">
      <div class="prediction-topline">
        <div>
          <strong>${run.job_name}</strong>
          <div class="subtle">${shortDate(run.started_at)} · ${run.trigger_source || "scheduler"}</div>
        </div>
        <span class="${badgeClass(run.status)}">${run.status}</span>
      </div>
      <div class="run-grid">
        <div><span class="label">Matchs</span><strong>${run.matches_scanned || 0}</strong></div>
        <div><span class="label">Publies</span><strong>${run.predictions_published || 0}</strong></div>
        <div><span class="label">Warnings</span><strong>${run.warnings || 0}</strong></div>
      </div>
      ${run.error_message ? `<p class="subtle danger-text">${run.error_message}</p>` : ""}
    </article>
  `).join("");
}

async function loadPredictionDetail(matchId) {
  try {
    state.detail = await fetchJson(`/api/predictions/${matchId}`);
    renderDetail();
  } catch (error) {
    state.detail = null;
    renderDetail();
  }
}

async function loadAll() {
  const [health, dashboard, predictions, history, admin] = await Promise.all([
    fetchJson("/health"),
    fetchJson("/api/dashboard"),
    fetchJson("/api/predictions/today"),
    fetchJson("/api/predictions/history?limit=20"),
    fetchJson("/api/admin/status"),
  ]);

  elements.databaseStatus.textContent = health.database ? "ok" : "degraded";
  state.dashboard = dashboard;
  state.predictions = predictions.predictions || [];
  state.history = history.history || [];
  state.admin = admin;

  if (!state.selectedMatchId && state.predictions.length) {
    state.selectedMatchId = state.predictions[0].match_id;
  }

  renderDashboard();
  renderPredictionsList();
  renderHistory();
  renderAdmin();

  if (state.selectedMatchId) {
    await loadPredictionDetail(state.selectedMatchId);
  } else {
    state.detail = null;
    renderDetail();
  }
}

async function triggerRunNow() {
  elements.runNowBtn.disabled = true;
  try {
    await fetchJson("/api/admin/run-now", { method: "POST" });
    await loadAll();
  } catch (error) {
    console.error(error);
  } finally {
    window.setTimeout(() => {
      elements.runNowBtn.disabled = false;
    }, 1500);
  }
}

function wireEvents() {
  elements.refreshBtn.addEventListener("click", loadAll);
  elements.searchInput.addEventListener("input", renderPredictionsList);
  elements.confidenceFilter.addEventListener("change", renderPredictionsList);
  elements.valueOnlyToggle.addEventListener("change", renderPredictionsList);
  elements.runNowBtn.addEventListener("click", triggerRunNow);
}

async function bootstrap() {
  wireEvents();
  await loadAll();
  state.refreshTimer = window.setInterval(loadAll, 30000);
}

bootstrap().catch((error) => {
  console.error(error);
});
