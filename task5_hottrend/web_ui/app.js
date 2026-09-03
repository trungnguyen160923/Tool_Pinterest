const state = {
  data: null,
  activeTab: "accepted",
  pathSnapshot: null,
};

const $ = (selector) => document.querySelector(selector);

function value(id) {
  return document.getElementById(id).value;
}

function slugifyNiche(input) {
  const slug = String(input || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug || "niche";
}

function pathsForNiche(niche) {
  const slug = slugifyNiche(niche);
  return {
    trendOutput: `${slug}_trend_output`,
    trendPackage: `${slug}_trend_output/trend_package.json`,
    crawlOutput: `${slug}_crawl_output`,
  };
}

function currentPathValues() {
  return {
    trendOutput: value("trendOutput"),
    trendPackage: value("trendPackage"),
    crawlOutput: value("crawlOutput"),
  };
}

function pathsMatchSnapshot() {
  const current = currentPathValues();
  const snapshot = state.pathSnapshot;
  return (
    snapshot &&
    current.trendOutput === snapshot.trendOutput &&
    current.trendPackage === snapshot.trendPackage &&
    current.crawlOutput === snapshot.crawlOutput
  );
}

function applyPathValues(paths) {
  document.getElementById("trendOutput").value = paths.trendOutput;
  document.getElementById("trendPackage").value = paths.trendPackage;
  document.getElementById("crawlOutput").value = paths.crawlOutput;
  state.pathSnapshot = { ...paths };
}

function syncPathsFromNiche() {
  if (!pathsMatchSnapshot()) return;
  applyPathValues(pathsForNiche(value("niche")));
}

function syncTrendPackagePath() {
  const output = value("trendOutput").replace(/[\\/]+$/, "");
  document.getElementById("trendPackage").value = `${output}/trend_package.json`;
}

function escapeHtml(input) {
  return String(input ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setStatus(text) {
  $("#status").textContent = text;
}

function renderMetrics(data) {
  const counts = data.manifest?.counts || {};
  const download = data.manifest?.download_stats || {};
  const gate = data.manifest?.gate_stats || {};
  const roles = gate.vision_roles || {};
  const raw = data.raw_results?.results || [];
  const trends = data.trend_package?.trends || [];
  const rejected = data.rejected_images || [];
  const accepted = counts.hot_product_images ?? 0;
  const candidates = counts.image_candidates ?? 0;
  const acceptRate = candidates ? `${Math.round((accepted / candidates) * 100)}%` : "—";
  const crawledTrendCount = new Set((data.manifest?.discovery_audit?.queries || []).map((item) => item.trend_id)).size;
  $("#metrics").innerHTML = [
    ["Package trends", trends.length],
    ["Crawled trends", crawledTrendCount || "—"],
    ["Raw URLs", counts.raw_results ?? raw.length],
    ["Downloaded", download.downloaded ?? 0],
    ["Candidates", candidates],
    ["Final accepted", accepted],
    ["Final accept rate", acceptRate],
    ["Vision accepted", gate.vision_accepted ?? "—"],
    ["Vision absent", roles.ABSENT ?? 0],
    ["Vision primary", roles.PRIMARY ?? 0],
    ["Vision secondary", roles.SECONDARY ?? 0],
    ["Product policy", data.manifest?.product_policy?.display_name || "auto"],
    ["Final rejected", counts.rejected_images ?? rejected.length],
    ["Provider", data.manifest?.provider || "—"],
    ["Vision", data.manifest?.vision_mode || "—"],
    ["Python", data.paths?.python || "—"],
    ["Output", data.paths?.crawl_output || "crawl_output"],
  ]
    .map(([label, val]) => `<div class="metric"><small>${escapeHtml(label)}</small><strong>${escapeHtml(val)}</strong></div>`)
    .join("");
}

function imageSrc(item) {
  if (item.local_path) {
    return `/file?path=${encodeURIComponent(item.local_path)}`;
  }
  return item.image_url || "";
}

function renderAccepted(data) {
  const items = data.hot_product_images || [];
  if (!items.length) {
    return `<div class="empty">No accepted images yet. Try stricter product queries or run crawler with Bing Images.</div>`;
  }
  return `<div class="gallery">${items
    .map(
      (item) => `
      <article class="card">
        <img class="thumb" src="${escapeHtml(imageSrc(item))}" alt="${escapeHtml(item.trend)}" loading="lazy">
        <div class="card-body">
          <div class="card-title">#${escapeHtml(item.rank)} · ${escapeHtml(item.trend)}</div>
          <div class="muted">${escapeHtml(item.query)}</div>
          <div class="chips">
            <span class="chip good">score ${escapeHtml(item.image_score)}</span>
            <span class="chip">${escapeHtml(item.product_role)}</span>
            <span class="chip">${escapeHtml(item.target_product_type || "type?")}</span>
            <span class="chip">vis ${escapeHtml(item.product_visibility)}</span>
            <span class="chip">trend ${escapeHtml(item.trend_relevance)}</span>
          </div>
          <p class="muted">${escapeHtml(item.detected_product || item.reason)}</p>
          ${item.pin_url ? `<a href="${escapeHtml(item.pin_url)}" target="_blank" rel="noreferrer">Open pin</a>` : ""}
        </div>
      </article>`
    )
    .join("")}</div>`;
}

function renderTrends(data) {
  const trends = data.trend_package?.trends || [];
  if (!trends.length) return `<div class="empty">No trend package loaded.</div>`;
  return `<div class="table-wrap"><table>
    <thead><tr><th>Trend</th><th>Strength</th><th>Fit</th><th>Relationship</th><th>Queries</th><th>Reason</th></tr></thead>
    <tbody>${trends
      .map(
        (trend) => `<tr>
          <td><strong>${escapeHtml(trend.trend)}</strong><br><span class="muted">${escapeHtml(trend.trend_id)}</span></td>
          <td>${escapeHtml(trend.trend_strength)}</td>
          <td>${escapeHtml(trend.semantic_fit)}</td>
          <td>${escapeHtml(trend.relationship)}</td>
          <td>${(trend.queries || []).map((q) => `<span class="chip">${escapeHtml(q.query || q)}</span>`).join(" ")}</td>
          <td>${escapeHtml(trend.reason)}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;
}

function renderQueries(data) {
  const queries = data.manifest?.discovery_audit?.queries || [];
  if (!queries.length) return `<div class="empty">No discovery audit yet.</div>`;
  return `<div class="table-wrap"><table>
    <thead><tr><th>Trend</th><th>Query</th><th>Found</th><th>Error</th></tr></thead>
    <tbody>${queries
      .map(
        (item) => `<tr>
          <td>${escapeHtml(item.trend)}</td>
          <td>${escapeHtml(item.query)}</td>
          <td>${escapeHtml(item.count)}</td>
          <td>${escapeHtml(item.error || "")}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;
}

function renderRejected(data) {
  const items = data.rejected_images || [];
  if (!items.length) return `<div class="empty">No rejected images.</div>`;
  return `<div class="table-wrap"><table>
    <thead><tr><th>Image</th><th>Trend</th><th>Query</th><th>Role</th><th>Type</th><th>Main</th><th>Policy</th><th>Vision reason</th></tr></thead>
    <tbody>${items
      .slice(0, 250)
      .map(
        (item) => `<tr>
          <td>${item.local_path || item.image_url ? `<img class="thumb" style="width:120px" src="${escapeHtml(imageSrc(item))}" loading="lazy">` : ""}</td>
          <td>${escapeHtml(item.trend)}</td>
          <td>${escapeHtml(item.query)}</td>
          <td>${escapeHtml(item.product_role || item.target_product_role || "")}</td>
          <td>${escapeHtml(item.target_product_type || "")}</td>
          <td>${escapeHtml(item.main_subject || "")}</td>
          <td>${escapeHtml(item.reason || item.download_error || item.gate_reason || "")}</td>
          <td>${escapeHtml(item.vision_reason || item.detected_product || "")}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;
}

function renderRaw(data) {
  const items = data.raw_results?.results || [];
  if (!items.length) return `<div class="empty">No raw results.</div>`;
  return `<div class="table-wrap"><table>
    <thead><tr><th>Source</th><th>Trend</th><th>Query</th><th>Image URL</th><th>Pin</th></tr></thead>
    <tbody>${items
      .slice(0, 300)
      .map(
        (item) => `<tr>
          <td>${escapeHtml(item.source)}</td>
          <td>${escapeHtml(item.trend)}</td>
          <td>${escapeHtml(item.query)}</td>
          <td><a href="${escapeHtml(item.image_url)}" target="_blank" rel="noreferrer">${escapeHtml(item.image_url)}</a></td>
          <td>${item.pin_url ? `<a href="${escapeHtml(item.pin_url)}" target="_blank" rel="noreferrer">pin</a>` : ""}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;
}

function renderMain() {
  const data = state.data || {};
  renderMetrics(data);
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === state.activeTab);
  });
  const renderers = {
    accepted: renderAccepted,
    trends: renderTrends,
    queries: renderQueries,
    rejected: renderRejected,
    raw: renderRaw,
  };
  $("#content").innerHTML = renderers[state.activeTab](data);
  $("#log").textContent = (data.status?.log || []).join("");
  setStatus(data.status?.running ? `Running ${data.status.job}` : "Idle");
}

async function refresh() {
  const params = new URLSearchParams({
    trend_output: value("trendOutput"),
    crawl_output: value("crawlOutput"),
  });
  state.data = await api(`/api/data?${params.toString()}`);
  renderMain();
}

async function runTrendFinder() {
  syncTrendPackagePath();
  const output = value("trendOutput");
  await api("/api/run/trends", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      niche: value("niche"),
      region: value("region"),
      output,
      max_trends: value("maxTrends"),
    }),
  });
  await refresh();
}

async function runCrawler() {
  await api("/api/run/crawler", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: value("trendPackage"),
      output: value("crawlOutput"),
      provider: value("provider"),
      max_trends: value("crawlTrends"),
      max_images_per_query: value("maxImages"),
      max_queries_per_trend: value("maxQueries"),
      max_downloads: value("maxDownloads"),
      top_images: value("topImages"),
      min_image_score: value("minImageScore"),
      vision_mode: value("visionMode"),
      accepted_product_roles: value("roles"),
      min_product_visibility: value("minVisibility"),
      min_trend_relevance: value("minTrendRelevance"),
    }),
  });
  await refresh();
}

async function runBrowserLogin() {
  await api("/api/run/browser-login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ timeout: 600 }),
  });
  await refresh();
}

async function clearData(scope) {
  const label = scope === "all" ? "trend + crawl data" : "crawl data";
  const ok = window.confirm(`Delete ${label}? This removes output files inside task5_hottrend only.`);
  if (!ok) return;
  await api("/api/clear-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scope,
      trend_output: value("trendOutput"),
      crawl_output: value("crawlOutput"),
    }),
  });
  await refresh();
}

document.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.matches(".tab")) {
    state.activeTab = target.dataset.tab;
    renderMain();
  }
  if (target.id === "refresh") {
    await refresh();
  }
  if (target.id === "runTrend") {
    await runTrendFinder();
  }
  if (target.id === "runCrawler") {
    await runCrawler();
  }
  if (target.id === "loginPinterest") {
    await runBrowserLogin();
  }
  if (target.id === "clearCrawl") {
    await clearData("crawl");
  }
  if (target.id === "clearAll") {
    await clearData("all");
  }
});

document.getElementById("trendOutput").addEventListener("change", syncTrendPackagePath);
document.getElementById("niche").addEventListener("input", syncPathsFromNiche);
state.pathSnapshot = currentPathValues();

setInterval(refresh, 2500);
refresh().catch((error) => {
  setStatus(error.message);
});
