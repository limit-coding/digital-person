const state = {
  figures: [],
  currentFigure: null,
  events: [],
  currentEvent: null,
  personalProfile: null,
  privateView: "profile",
  thinking: "enabled",
  running: false,
  models: [],
  currentPeriodId: null,
  currentSliceId: null,
  askModelIds: [],
  askMode: "grounded",
  askMessages: [],
  asking: false,
  personalAskModelIds: [],
  personalAskMessages: [],
  personalAsking: false,
  personalCloudConsent: false,
  replayModelIds: [],
  replayCloudConsent: false,
  replayResults: [],
};

const $ = (selector) => document.querySelector(selector);
const figureOrder = [
  "mao-zedong", "charles-darwin", "abraham-lincoln", "winston-churchill",
  "mahatma-gandhi", "nelson-mandela", "albert-einstein", "simon-bolivar",
  "liliuokalani", "mustafa-kemal-ataturk", "rachel-carson", "ludwig-van-beethoven", "isaac-newton",
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
  if (!profile.slices.some((item) => item.slice_id === state.currentSliceId)) state.currentSliceId = profile.slices[0]?.slice_id || null;
  const sliceIndex = profile.slices.findIndex((item) => item.slice_id === state.currentSliceId);
  const slice = profile.slices[Math.max(0, sliceIndex)];
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
  chronology.append(el("span", "", `DIGITAL SLICE / ${String(sliceIndex + 1).padStart(2, "0")}`), el("b", "", slice.date_label));
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

  const sliceTabs = el("div", "slice-tabs");
  profile.slices.forEach((item, index) => {
    const button = el("button", item.slice_id === state.currentSliceId ? "active" : "");
    button.type = "button";
    button.append(el("span", "", String(index + 1).padStart(2, "0")), el("b", "", item.date_label), el("small", "", item.title));
    button.addEventListener("click", () => {
      state.currentSliceId = item.slice_id;
      const matchingPeriod = profile.periods?.find((period) => period.source_slice_ids?.includes(item.slice_id));
      if (matchingPeriod) state.currentPeriodId = matchingPeriod.period_id;
      state.askMessages = [];
      renderFigure(profile);
    });
    sliceTabs.append(button);
  });
  root.append(mast, facts, note);
  if (profile.slices.length > 1) root.append(sliceTabs);
  root.append(sliceHead, boundary, hypothesisSection, tensionSection, evidenceSection, renderQuestionStudio(profile));
}

function askList(title, items, className = "") {
  const section = el("section", `ask-answer-list ${className}`.trim());
  section.append(el("h5", "", title));
  if (!items.length) section.append(el("p", "ask-none", "本轮没有可列出的内容"));
  else section.append(textList(items));
  return section;
}

function comparisonEntryCard(entry) {
  const answer = el("article", "ask-message answer mirror-answer-card");
  if (entry.error) {
    answer.append(el("span", "", entry.modelLabel), el("h4", "", "暂未形成判断"), el("p", "", entry.error));
    return answer;
  }
  const scopeLabel = entry.result.period_label || entry.result.mirror_title;
  answer.append(
    el("span", "", `${scopeLabel} · ${entry.result.model.label}`),
    el("h4", "mirror-judgement", entry.result.judgement),
    el("p", "ask-main-answer", entry.result.answer),
  );
  const layers = el("div", "ask-answer-layers");
  layers.append(askList("记录支持", entry.result.supported_claims), askList("模型推演", entry.result.speculative_claims, "speculative"), askList("仍然未知", entry.result.unknowns, "unknown"));
  answer.append(layers, el("small", "", entry.result.boundary_note));
  return answer;
}

function comparisonHistory(messages) {
  const turns = [];
  messages.forEach((item) => {
    turns.push({role: "user", content: item.question});
    const combined = item.results.filter((entry) => entry.result).map((entry) => `${entry.result.model.label}：${entry.result.answer}`).join("\n").slice(0, 1500);
    if (combined) turns.push({role: "assistant", content: combined});
  });
  return turns.slice(-10);
}

function renderQuestionStudio(profile) {
  const periods = profile.periods || [];
  if (!periods.some((item) => item.period_id === state.currentPeriodId)) state.currentPeriodId = periods[0]?.period_id || null;
  const period = periods.find((item) => item.period_id === state.currentPeriodId);
  const studio = el("section", "ask-studio");
  studio.id = "ask-studio";
  const head = el("header", "ask-head");
  const copy = el("div");
  copy.append(el("p", "overline", "ASK / PERIOD-SPECIFIC MIRROR"), el("h3", "", "不只回放事件，直接问这个时期的 TA。"), el("span", "", "事实问题、价值冲突或反事实脑洞都可以；推测部分会单独标出。"));
  head.append(copy, el("b", "", `${periods.length} 个时期`));
  studio.append(head);

  const periodTabs = el("div", "period-tabs");
  periods.forEach((item) => {
    const button = el("button", item.period_id === state.currentPeriodId ? "active" : "");
    button.type = "button";
    button.append(el("span", "", item.range), el("b", "", item.label));
    button.addEventListener("click", () => {
      state.currentPeriodId = item.period_id;
      if (item.source_slice_ids?.length) state.currentSliceId = item.source_slice_ids[0];
      state.askMessages = [];
      renderFigure(profile);
      $("#ask-studio").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    periodTabs.append(button);
  });
  studio.append(periodTabs);

  if (period) {
    const frame = el("article", "period-frame");
    frame.append(el("span", "", period.headline), el("p", "", period.context));
    const anchors = el("div");
    (period.anchors || []).slice(0, 3).forEach((item) => anchors.append(el("b", "", item)));
    frame.append(anchors);
    studio.append(frame);
  }

  const controls = el("div", "ask-controls");
  const modelPicker = el("div", "ask-model-picker");
  modelPicker.append(el("span", "", "参与判断的模型 · 可多选"));
  const modelChoices = el("div", "ask-model-choices");
  state.models.forEach((model) => {
    const selected = state.askModelIds.includes(model.model_id);
    const button = el("button", selected ? "active" : "", `${model.label}${model.available ? "" : " · 未配置"}`);
    button.type = "button";
    button.disabled = !model.available;
    button.setAttribute("aria-pressed", String(selected));
    button.addEventListener("click", () => {
      if (selected && state.askModelIds.length > 1) state.askModelIds = state.askModelIds.filter((id) => id !== model.model_id);
      else if (!selected && state.askModelIds.length < 4) state.askModelIds = [...state.askModelIds, model.model_id];
      renderFigure(profile);
    });
    modelChoices.append(button);
  });
  modelPicker.append(modelChoices);
  const modeGroup = el("div", "ask-mode");
  [{id: "grounded", label: "史料优先"}, {id: "counterfactual", label: "反事实推演"}].forEach((mode) => {
    const button = el("button", state.askMode === mode.id ? "active" : "", mode.label);
    button.type = "button";
    button.addEventListener("click", () => { state.askMode = mode.id; renderFigure(profile); });
    modeGroup.append(button);
  });
  controls.append(modelPicker, modeGroup);
  studio.append(controls);

  const thread = el("div", "ask-thread");
  if (!state.askMessages.length) {
    const starters = el("div", "ask-starters");
    ["如果你看到今天的人工智能，会最先质疑什么？", "你这个时期最不愿牺牲的东西是什么？", "如果关键条件反过来，你可能怎样改选？"].forEach((text) => {
      const button = el("button", "", text);
      button.type = "button";
      button.addEventListener("click", () => { const input = $("#period-question"); input.value = text; input.focus(); });
      starters.append(button);
    });
    thread.append(starters);
  }
  state.askMessages.forEach((message) => {
    const question = el("article", "ask-message user");
    question.append(el("span", "", "你的问题"), el("p", "", message.question));
    const comparison = el("div", "mirror-comparison");
    message.results.forEach((entry) => comparison.append(comparisonEntryCard(entry)));
    thread.append(question, comparison);
  });
  studio.append(thread);

  const form = el("form", "ask-form");
  const input = el("textarea");
  input.id = "period-question";
  input.name = "question";
  input.placeholder = `问 ${profile.name} 的“${period?.label || "当前"}”时期…`;
  input.maxLength = 1200;
  input.required = true;
  const submit = el("button", "", state.asking ? `正在等待 ${state.askModelIds.length} 个模型…` : `让 ${state.askModelIds.length} 个模型同时判断 ↗`);
  submit.type = "submit";
  submit.disabled = state.asking || !period;
  form.append(input, submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || state.asking) return;
    state.asking = true;
    renderFigure(profile);
    try {
      const history = comparisonHistory(state.askMessages);
      const selectedModels = state.models.filter((model) => state.askModelIds.includes(model.model_id));
      const results = await Promise.all(selectedModels.map(async (model) => {
        try {
          const result = await api("/api/ask", { method: "POST", body: JSON.stringify({figure_id: profile.figure_id, period_id: state.currentPeriodId, question, model_id: model.model_id, mode: state.askMode, history}) });
          return {modelLabel: model.label, result};
        } catch (error) {
          const message = error.status === 401 ? "需要先登录私人会话" : error.message;
          return {modelLabel: model.label, error: message};
        }
      }));
      state.askMessages.push({question, results});
    } finally {
      state.asking = false;
      renderFigure(profile);
      $("#ask-studio").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
  studio.append(form, el("p", "ask-footnote", "每个模型独立作答，页面不让任何一个模型替其他模型总结；人物、时期和私人租户之间仍保持隔离。"));
  return studio;
}

async function selectFigure(figureId) {
  try {
    if (state.currentFigure?.figure_id !== figureId) {
      state.currentPeriodId = null;
      state.currentSliceId = null;
      state.askMessages = [];
    }
    const profile = await api(`/api/public/figures/${encodeURIComponent(figureId)}`);
    renderFigure(profile);
  } catch (error) {
    $("#figure-dossier").textContent = `人物档案暂时不可用：${error.message}`;
  }
}

async function initializePublic() {
  try {
    const [data, models] = await Promise.all([api("/api/public/figures"), api(`/api/public/models?ts=${Date.now()}`)]);
    state.models = models.models;
    const externalModels = state.models.filter((item) => item.available && item.model_id !== "evidence-synthesis");
    const defaultModelIds = (externalModels.length ? externalModels : state.models.filter((item) => item.available)).slice(0, 4).map((item) => item.model_id);
    state.askModelIds = [...defaultModelIds];
    state.personalAskModelIds = [...defaultModelIds];
    state.replayModelIds = [...defaultModelIds];
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
  const [data, profile] = await Promise.all([api("/api/events"), api("/api/me/profile")]);
  state.events = data.events;
  state.personalProfile = profile;
  state.privateView = "profile";
  state.currentEvent = null;
  renderPersonalProfile();
  renderPrivateEventList();
  showPersonalProfile();
}

function renderPersonalProfile() {
  const profile = state.personalProfile;
  if (!profile) return;
  $("#profile-title").textContent = profile.title;
  $("#profile-subtitle").textContent = profile.subtitle;
  $("#profile-confidence").textContent = `${profile.confidence.label} · ${Math.round(profile.confidence.score * 100)}%`;
  $("#profile-confidence-note").textContent = profile.confidence.note;
  const stats = $("#profile-stats");
  stats.replaceChildren();
  [[profile.coverage.event_count, "决策切片"], [profile.coverage.domain_count, "覆盖领域"], [profile.coverage.evidence_count, "证据锚点"], [profile.coverage.observed_action_count, "已观察行动"]].forEach(([value, label]) => {
    const card = el("article"); card.append(el("b", "", value), el("span", "", label)); stats.append(card);
  });
  const signals = $("#profile-signals");
  signals.replaceChildren();
  profile.signals.forEach((signal, index) => {
    const card = el("article");
    card.append(el("span", "", String(index + 1).padStart(2, "0")), el("h3", "", signal.label), el("b", "", signal.value), el("p", "", signal.note), el("small", "", signal.support));
    signals.append(card);
  });
  const domains = $("#profile-domains");
  domains.replaceChildren();
  profile.domains.forEach((domain) => {
    const row = el("div"); const copy = el("p"); copy.append(el("span", "", domain.label), el("b", "", `${domain.count} 个`));
    const meter = el("i"); const fill = el("b"); fill.style.width = `${domain.share * 100}%`; meter.append(fill); row.append(copy, meter); domains.append(row);
  });
  const tensions = $("#profile-tensions");
  tensions.replaceChildren();
  profile.active_tensions.slice(0, 6).forEach((item) => {
    const row = el("article"); row.append(el("span", "", item.domain), el("p", "", item.text)); tensions.append(row);
  });
  $("#profile-boundary").textContent = profile.data_boundary.note;
  renderPersonalAskStudio();
}

function renderPersonalAskStudio() {
  const choices = $("#personal-model-choices");
  choices.replaceChildren();
  state.models.forEach((model) => {
    const selected = state.personalAskModelIds.includes(model.model_id);
    const button = el("button", selected ? "active" : "", `${model.label}${model.available ? "" : " · 未配置"}`);
    button.type = "button";
    button.disabled = !model.available;
    button.setAttribute("aria-pressed", String(selected));
    button.addEventListener("click", () => {
      if (selected && state.personalAskModelIds.length > 1) state.personalAskModelIds = state.personalAskModelIds.filter((id) => id !== model.model_id);
      else if (!selected && state.personalAskModelIds.length < 4) state.personalAskModelIds = [...state.personalAskModelIds, model.model_id];
      renderPersonalAskStudio();
    });
    choices.append(button);
  });
  const consent = $("#personal-cloud-consent");
  consent.checked = state.personalCloudConsent;
  const selectedModels = state.models.filter((model) => state.personalAskModelIds.includes(model.model_id));
  const needsConsent = selectedModels.some((model) => !model.local);
  const submit = $("#personal-ask-button");
  submit.textContent = state.personalAsking ? `正在等待 ${selectedModels.length} 个模型…` : `让 ${selectedModels.length} 个模型同时判断 ↗`;
  submit.disabled = state.personalAsking || !selectedModels.length || (needsConsent && !state.personalCloudConsent);
  $("#personal-ask-status").textContent = needsConsent && !state.personalCloudConsent
    ? "勾选授权后才会向云模型发送聚合指标；聊天原文始终不会上传。"
    : "这是工作模型，不是人格诊断；不同模型的分歧会原样保留。";

  const thread = $("#personal-ask-thread");
  thread.replaceChildren();
  state.personalAskMessages.forEach((message) => {
    const question = el("article", "ask-message user");
    question.append(el("span", "", "你问自己的镜像"), el("p", "", message.question));
    const comparison = el("div", "mirror-comparison");
    message.results.forEach((entry) => comparison.append(comparisonEntryCard(entry)));
    thread.append(question, comparison);
  });
}

async function runPersonalQuestion(event) {
  event.preventDefault();
  const input = $("#personal-question");
  const question = input.value.trim();
  if (!question || state.personalAsking) return;
  const selectedModels = state.models.filter((model) => state.personalAskModelIds.includes(model.model_id));
  if (!selectedModels.length) return;
  state.personalAsking = true;
  renderPersonalAskStudio();
  try {
    const history = comparisonHistory(state.personalAskMessages);
    const results = await Promise.all(selectedModels.map(async (model) => {
      try {
        const result = await api("/api/me/ask", {method: "POST", body: JSON.stringify({question, model_id: model.model_id, history, allow_cloud: state.personalCloudConsent})});
        return {modelLabel: model.label, result};
      } catch (error) {
        return {modelLabel: model.label, error: error.message};
      }
    }));
    state.personalAskMessages.push({question, results});
    input.value = "";
  } finally {
    state.personalAsking = false;
    renderPersonalAskStudio();
    $(".personal-ask-studio").scrollIntoView({behavior: "smooth", block: "start"});
  }
}

function showPersonalProfile() {
  state.privateView = "profile";
  state.currentEvent = null;
  $("#personal-profile").hidden = false;
  $("#private-event").hidden = true;
  $("#private-empty").hidden = true;
  renderPrivateEventList();
}

function renderPrivateEventList() {
  const list = $("#event-list");
  list.replaceChildren();
  $("#event-count").textContent = state.events.length;
  $("#profile-nav").classList.toggle("active", state.privateView === "profile");
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
  state.privateView = "event";
  state.currentEvent = await api(`/api/events/${encodeURIComponent(episodeId)}`);
  state.replayResults = [];
  renderPrivateEventList();
  $("#personal-profile").hidden = true;
  $("#private-empty").hidden = true;
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
  renderReplayControls();
}

function replayModels() {
  return state.models.filter((model) => model.available && model.model_id !== "evidence-synthesis");
}

function renderReplayControls() {
  const choices = $("#replay-model-choices");
  choices.replaceChildren();
  replayModels().forEach((model) => {
    const selected = state.replayModelIds.includes(model.model_id);
    const button = el("button", selected ? "active" : "", model.label);
    button.type = "button";
    button.setAttribute("aria-pressed", String(selected));
    button.addEventListener("click", () => {
      if (selected && state.replayModelIds.length > 1) state.replayModelIds = state.replayModelIds.filter((id) => id !== model.model_id);
      else if (!selected && state.replayModelIds.length < 4) state.replayModelIds = [...state.replayModelIds, model.model_id];
      renderReplayControls();
    });
    choices.append(button);
  });
  const consent = $("#replay-cloud-consent");
  consent.checked = state.replayCloudConsent;
  const selectedModels = replayModels().filter((model) => state.replayModelIds.includes(model.model_id));
  const needsConsent = selectedModels.some((model) => !model.local);
  const button = $("#run-button");
  button.textContent = state.running ? `正在等待 ${selectedModels.length} 个模型…` : `让 ${selectedModels.length} 个模型回放 ↗`;
  button.disabled = state.running || !selectedModels.length || (needsConsent && !state.replayCloudConsent);
  if (needsConsent && !state.replayCloudConsent) {
    $("#run-message").textContent = "勾选授权后，才会发送脱敏历史截面；事件结果始终留在服务端。";
  } else if (!state.running && !state.replayResults.length) {
    $("#run-message").textContent = "";
  }
}

function replayResultCard(entry, actual) {
  const card = el("article", "replay-model-card");
  card.append(el("span", "replay-model-name", entry.modelLabel));
  if (entry.error) {
    card.append(el("b", "replay-error", "本次调用失败"), el("small", "", entry.error));
    return card;
  }
  const prediction = entry.result.prediction;
  const judgement = optionLabel(prediction.predicted_judgement);
  const action = optionLabel(prediction.predicted_action);
  const actionHit = prediction.predicted_action === actual.action;
  const verdict = el("div", "replay-verdict");
  verdict.append(el("span", "", "当时会判断"), el("b", "", judgement), el("span", "", "最终会行动"), el("strong", "", action));
  const hit = el("i", actionHit ? "hit" : "miss", actionHit ? "行动命中真实记录" : "与真实行动不同");
  card.append(verdict, hit);
  const probabilities = el("div", "private-probabilities compact");
  Object.entries(prediction.option_probabilities).sort((a, b) => b[1] - a[1]).forEach(([id, value]) => {
    const row = el("div");
    const copy = el("p");
    copy.append(el("span", "", optionLabel(id)), el("b", "", `${Math.round(value * 100)}%`));
    const meter = el("i");
    const fill = el("b");
    fill.style.width = `${value * 100}%`;
    meter.append(fill);
    row.append(copy, meter);
    probabilities.append(row);
  });
  card.append(probabilities);
  return card;
}

function renderPrediction(results) {
  $("#result-section").hidden = false;
  const successful = results.filter((entry) => entry.result);
  const actual = successful[0]?.result.actual || {judgement: null, action: null};
  const judgements = new Set(successful.map((entry) => entry.result.prediction.predicted_judgement));
  const actions = new Set(successful.map((entry) => entry.result.prediction.predicted_action));
  const agreement = $("#replay-agreement");
  agreement.replaceChildren();
  if (successful.length > 1) {
    agreement.append(
      el("span", judgements.size === 1 ? "agree" : "disagree", judgements.size === 1 ? "判断一致" : "判断有分歧"),
      el("span", actions.size === 1 ? "agree" : "disagree", actions.size === 1 ? "行动一致" : "行动有分歧"),
    );
  } else {
    agreement.append(el("span", "disagree", "仅一个模型成功"));
  }
  $("#replay-difference").textContent = successful.map((entry) => {
    const prediction = entry.result.prediction;
    return `${entry.modelLabel}：判断“${optionLabel(prediction.predicted_judgement)}”，行动“${optionLabel(prediction.predicted_action)}”`;
  }).join("；");
  const comparison = $("#replay-comparison");
  comparison.replaceChildren();
  results.forEach((entry) => comparison.append(replayResultCard(entry, actual)));
  $("#actual-judgement").textContent = optionLabel(actual.judgement);
  $("#actual-action").textContent = `实际行动：${optionLabel(actual.action)}`;
  $("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPrediction() {
  if (!state.currentEvent || state.running) return;
  const selectedModels = replayModels().filter((model) => state.replayModelIds.includes(model.model_id));
  if (!selectedModels.length) return;
  state.running = true;
  renderReplayControls();
  $("#run-message").textContent = `正在让 ${selectedModels.length} 个模型分别重建同一历史截面…`;
  try {
    const results = await Promise.all(selectedModels.map(async (model) => {
      try {
        const result = await api("/api/predict", {method: "POST", body: JSON.stringify({episode_id: state.currentEvent.episode_id, thinking: state.thinking, model_id: model.model_id, allow_cloud: state.replayCloudConsent})});
        return {modelLabel: model.label, result};
      } catch (error) {
        return {modelLabel: model.label, error: error.message};
      }
    }));
    state.replayResults = results;
    renderPrediction(results);
    $("#run-message").textContent = "完成。上方先显示结论分歧，再展示各自概率；所有模型都未看到截止点后的真实结果。";
  } finally {
    state.running = false;
    renderReplayControls();
  }
}

$("#open-private").addEventListener("click", openPrivate);
$("#profile-nav").addEventListener("click", showPersonalProfile);
$("#personal-cloud-consent").addEventListener("change", (event) => {
  state.personalCloudConsent = event.target.checked;
  renderPersonalAskStudio();
});
$("#replay-cloud-consent").addEventListener("change", (event) => {
  state.replayCloudConsent = event.target.checked;
  renderReplayControls();
});
$("#personal-ask-form").addEventListener("submit", runPersonalQuestion);
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
