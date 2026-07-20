const state = {
  figures: [],
  currentFigure: null,
  events: [],
  currentEvent: null,
  thinking: "enabled",
  running: false,
};

const $ = (selector) => document.querySelector(selector);
const figureOrder = [
  "mao-zedong", "charles-darwin", "abraham-lincoln", "winston-churchill",
  "mahatma-gandhi", "nelson-mandela", "albert-einstein", "simon-bolivar",
  "liliuokalani", "mustafa-kemal-ataturk", "rachel-carson",
];

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let body = {};
  try { body = await response.json(); } catch (_) { body = {}; }
  if (!response.ok) {
    const error = new Error(body.detail || "请求失败");
    error.status = response.status;
    throw error;
  }
  return body;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function prettyDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "long", day: "numeric" }).format(date);
}

function renderFigureList() {
  const list = $("#figure-list");
  list.replaceChildren();
  $("#figure-count").textContent = String(state.figures.length).padStart(2, "0");
  state.figures.forEach((figure, index) => {
    const button = el("button", `figure-row accent-${figure.accent}`);
    button.type = "button";
    button.classList.toggle("active", state.currentFigure?.figure_id === figure.figure_id);
    const number = el("span", "figure-number", String(index + 1).padStart(2, "0"));
    const copy = el("span", "figure-row-copy");
    copy.append(el("b", "", figure.name), el("small", "", `${figure.life} · ${figure.field}`));
    button.append(number, copy, el("i", "", "→"));
    button.addEventListener("click", () => selectFigure(figure.figure_id));
    list.append(button);
  });
}

function evidenceCard(item) {
  const card = el("article", "source-card");
  const head = el("div", "source-head");
  head.append(el("span", "", item.evidence_id.toUpperCase()), el("b", "", item.confidence));
  const kind = el("p", "source-kind", `${item.date} · ${item.kind} · ${item.proximity}`);
  const summary = el("p", "source-summary", item.summary);
  const link = el("a", "source-link", `${item.source_title} ↗`);
  link.href = item.source_url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  card.append(head, kind, summary, link);
  return card;
}

function hypothesisCard(item, index) {
  const card = el("article", "hypothesis-card");
  const head = el("div", "hypothesis-head");
  head.append(el("span", "", `H${index + 1}`), el("strong", "", `${Math.round(item.probability * 100)}%`));
  card.append(head, el("h4", "", item.label), el("p", "", item.basis));
  const meter = el("div", "hypothesis-meter");
  const fill = el("i");
  fill.style.width = `${item.probability * 100}%`;
  meter.append(fill);
  card.append(meter, el("small", "", `依据：${item.evidence_ids.join(" · ")}`));
  return card;
}

function textList(items, className = "") {
  const list = el("ul", className);
  items.forEach((item) => list.append(el("li", "", item)));
  return list;
}

function renderFigure(profile) {
  state.currentFigure = profile;
  renderFigureList();
  const slice = profile.slices[0];
  const root = $("#figure-dossier");
  root.replaceChildren();
  root.className = `figure-dossier accent-${profile.accent}`;

  const mast = el("header", "dossier-mast");
  const identity = el("div", "dossier-identity");
  identity.append(
    el("p", "native-name", profile.native_name),
    el("h3", "", profile.name),
    el("p", "figure-tagline", profile.tagline),
  );
  const seal = el("div", "figure-seal");
  seal.append(el("b", "", profile.name.slice(0, 1)), el("span", "", profile.life));
  mast.append(identity, seal);

  const facts = el("div", "dossier-facts");
  [["领域", profile.field], ["档案规模", profile.archive_scale], ["当前状态", profile.status], ["数字切片", `${profile.slices.length} 个`]]
    .forEach(([label, value]) => {
      const item = el("div");
      item.append(el("span", "", label), el("b", "", value));
      facts.append(item);
    });

  const note = el("div", "method-note");
  note.append(el("span", "", "阅读边界"), el("p", "", profile.method_note));

  const sliceHead = el("section", "slice-head");
  const chronology = el("div", "slice-chronology");
  chronology.append(el("span", "", "DIGITAL SLICE / 01"), el("b", "", slice.date_label));
  const sliceCopy = el("div");
  sliceCopy.append(el("h3", "", slice.title), el("p", "slice-question", slice.question), el("p", "slice-context", slice.context));
  sliceHead.append(chronology, sliceCopy);

  const boundary = el("section", "knowledge-boundary");
  const known = el("article");
  known.append(el("p", "panel-label", `截止 ${slice.date} · 当时已知`), textList(slice.known_at_cutoff));
  const unknown = el("article", "unknown-panel");
  unknown.append(el("p", "panel-label", "当时未知 / 不得泄漏"), textList(slice.unknown_at_cutoff));
  boundary.append(known, unknown);

  const hypothesisSection = el("section", "dossier-section");
  const hypothesisTitle = el("div", "dossier-section-title");
  hypothesisTitle.append(el("div", "", "01"), el("h3", "", "竞争性行动假设"), el("p", "", "概率是研究原型的先验展示，不是历史事实"));
  const hypotheses = el("div", "hypothesis-grid");
  slice.hypotheses.forEach((item, index) => hypotheses.append(hypothesisCard(item, index)));
  hypothesisSection.append(hypothesisTitle, hypotheses);

  const tensionSection = el("section", "tension-section");
  tensionSection.append(el("span", "", "核心冲突"));
  const chips = el("div");
  slice.tensions.forEach((item) => chips.append(el("b", "", item)));
  tensionSection.append(chips);

  const evidenceSection = el("section", "dossier-section evidence-section");
  const evidenceTitle = el("div", "dossier-section-title");
  evidenceTitle.append(el("div", "", "02"), el("h3", "", "证据与反证入口"), el("p", "", "打开原始档案核对，而不是相信界面本身"));
  const evidenceGrid = el("div", "source-grid");
  slice.evidence.forEach((item) => evidenceGrid.append(evidenceCard(item)));
  evidenceSection.append(evidenceTitle, evidenceGrid);

  root.append(mast, facts, note, sliceHead, boundary, hypothesisSection, tensionSection, evidenceSection);
}

async function selectFigure(figureId) {
  try {
    const profile = await api(`/api/public/figures/${encodeURIComponent(figureId)}`);
    renderFigure(profile);
  } catch (error) {
    $("#figure-dossier").textContent = `人物档案暂时不可用：${error.message}`;
  }
}

async function initializePublic() {
  try {
    const data = await api("/api/public/figures");
    state.figures = data.figures.sort((a, b) => figureOrder.indexOf(a.figure_id) - figureOrder.indexOf(b.figure_id));
    renderFigureList();
    if (state.figures.length) await selectFigure(state.figures[0].figure_id);
  } catch (error) {
    $("#figure-dossier").textContent = `档案馆暂时不可用：${error.message}`;
  }
}

async function openPrivate() {
  $("#public-site").hidden = true;
  $("#private-shell").hidden = false;
  window.scrollTo(0, 0);
  try {
    const session = await api("/api/session");
    if (session.authenticated) return showPrivateApp();
    $("#login-view").hidden = false;
    $("#private-app").hidden = true;
    $("#logout-button").hidden = true;
    $("#login-message").textContent = session.configured ? "" : "私人镜像服务尚未配置访问凭据";
  } catch (_) {
    $("#login-message").textContent = "暂时无法连接私人镜像服务";
  }
}

function closePrivate() {
  $("#private-shell").hidden = true;
  $("#public-site").hidden = false;
  window.scrollTo(0, 0);
}

function optionLabel(id) {
  return state.currentEvent?.options.find((item) => item.option_id === id)?.description || id || "暂无记录";
}

async function showPrivateApp() {
  $("#login-view").hidden = true;
  $("#private-app").hidden = false;
  $("#logout-button").hidden = false;
  const data = await api("/api/events");
  state.events = data.events;
  renderPrivateEventList();
  if (state.events.length) await selectPrivateEvent(state.events[0].episode_id);
}

function renderPrivateEventList() {
  const list = $("#event-list");
  list.replaceChildren();
  $("#event-count").textContent = state.events.length;
  state.events.forEach((event, index) => {
    const button = el("button", state.currentEvent?.episode_id === event.episode_id ? "active" : "");
    button.type = "button";
    button.append(el("span", "", String(index + 1).padStart(2, "0")), el("b", "", event.question), el("small", "", event.era || event.domain));
    button.addEventListener("click", () => selectPrivateEvent(event.episode_id));
    list.append(button);
  });
  $("#private-empty").hidden = Boolean(state.events.length);
}

async function selectPrivateEvent(episodeId) {
  state.currentEvent = await api(`/api/events/${encodeURIComponent(episodeId)}`);
  renderPrivateEventList();
  $("#private-event").hidden = false;
  $("#result-section").hidden = true;
  $("#private-tags").textContent = `${state.currentEvent.domain} · ${state.currentEvent.persona_state.era || "未标记时代"}`;
  $("#event-question").textContent = state.currentEvent.question;
  $("#event-cutoff").textContent = `历史截点 · ${prettyDate(state.currentEvent.cutoff_at)}`;
  const evidence = $("#evidence-list");
  evidence.replaceChildren();
  state.currentEvent.evidence.forEach((item) => {
    const card = el("article");
    card.append(el("span", "", `${item.evidence_id} · ${prettyDate(item.observed_at)}`), el("p", "", item.summary));
    evidence.append(card);
  });
  const options = $("#option-grid");
  options.replaceChildren();
  state.currentEvent.options.forEach((item, index) => {
    const card = el("article");
    card.append(el("span", "", String.fromCharCode(65 + index)), el("p", "", item.description));
    options.append(card);
  });
}

function renderPrediction(result) {
  $("#result-section").hidden = false;
  $("#model-judgement").textContent = optionLabel(result.prediction.predicted_judgement);
  $("#model-action").textContent = `预测行动：${optionLabel(result.prediction.predicted_action)}`;
  $("#actual-judgement").textContent = optionLabel(result.actual.judgement);
  $("#actual-action").textContent = `实际行动：${optionLabel(result.actual.action)}`;
  const list = $("#probability-list");
  list.replaceChildren();
  Object.entries(result.prediction.option_probabilities).sort((a, b) => b[1] - a[1]).forEach(([id, value]) => {
    const row = el("div");
    const copy = el("p");
    copy.append(el("span", "", optionLabel(id)), el("b", "", `${Math.round(value * 100)}%`));
    const meter = el("i");
    const fill = el("b");
    fill.style.width = `${value * 100}%`;
    meter.append(fill);
    row.append(copy, meter);
    list.append(row);
  });
  $("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPrediction() {
  if (!state.currentEvent || state.running) return;
  state.running = true;
  $("#run-button").disabled = true;
  $("#run-message").textContent = "正在严格按历史截面重建…";
  try {
    const result = await api("/api/predict", { method: "POST", body: JSON.stringify({ episode_id: state.currentEvent.episode_id, thinking: state.thinking }) });
    renderPrediction(result);
    $("#run-message").textContent = "完成。本次输入未包含截止点之后的信息。";
  } catch (error) {
    $("#run-message").textContent = error.message;
  } finally {
    state.running = false;
    $("#run-button").disabled = false;
  }
}

$("#open-private").addEventListener("click", openPrivate);
$("#back-public").addEventListener("click", closePrivate);
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#login-message").textContent = "正在验证…";
  try {
    await api("/api/login", { method: "POST", body: JSON.stringify({ password: $("#password").value }) });
    $("#password").value = "";
    await showPrivateApp();
  } catch (error) { $("#login-message").textContent = error.message; }
});
$("#logout-button").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST", body: "{}" }); } catch (_) {}
  await openPrivate();
});
document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => {
  state.thinking = button.dataset.thinking;
  document.querySelectorAll(".mode").forEach((item) => item.classList.toggle("active", item === button));
}));
$("#run-button").addEventListener("click", runPrediction);
document.querySelectorAll("[data-figure-jump]").forEach((button) => button.addEventListener("click", async () => {
  document.querySelector("#figures").scrollIntoView({ behavior: "smooth" });
  await selectFigure(button.dataset.figureJump);
}));

initializePublic();
