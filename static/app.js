const conversationListEl = document.getElementById("conversationList");
const conversationTitleEl = document.getElementById("conversationTitle");
const conversationMetaEl = document.getElementById("conversationMeta");
const newConversationBtn = document.getElementById("newConversationBtn");
const renameConversationBtn = document.getElementById("renameConversationBtn");
const deleteConversationBtn = document.getElementById("deleteConversationBtn");
const messagesEl = document.getElementById("messages");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

const detailPanel = document.getElementById("detailPanel");
const detailTitle = document.getElementById("detailTitle");
const detailContent = document.getElementById("detailContent");
const detailClose = document.getElementById("detailClose");

detailClose.addEventListener("click", () => {
  detailPanel.classList.add("hidden");
});

function showDetail(title, obj) {
  detailTitle.textContent = title;
  detailContent.textContent = JSON.stringify(obj, null, 2);
  detailPanel.classList.remove("hidden");
}

function scrollMessagesToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMarkdown(el, mdText) {
  const markedLib = window.marked;
  if (!markedLib) {
    el.textContent = mdText;
    return;
  }
  let html = markedLib.parse(mdText ?? "", { gfm: true, breaks: true });
  if (window.DOMPurify) {
    html = window.DOMPurify.sanitize(html);
  }
  el.innerHTML = html;
}

function enableAutoResizeTextarea(ta) {
  if (!ta) return;

  const resize = () => {
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  };

  let composing = false;

  ta.addEventListener("compositionstart", () => {
    composing = true;
  });
  ta.addEventListener("compositionend", () => {
    composing = false;
    requestAnimationFrame(resize);
  });
  ta.addEventListener("input", () => {
    if (composing) return;
    requestAnimationFrame(resize);
  });

  requestAnimationFrame(resize);
}

enableAutoResizeTextarea(questionInput);

function mkBlock(type) {
  const div = document.createElement("div");
  div.className = `block ${type}`;
  return div;
}

function mkQuestionBlock(text) {
  const b = mkBlock("question");
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = "Question";
  b.appendChild(title);

  const body = document.createElement("div");
  body.textContent = text;
  b.appendChild(body);
  return b;
}

function mkHistoryAnswerBlock(text) {
  const b = mkBlock("answer");
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = "Answer";
  b.appendChild(title);

  const finalText = document.createElement("div");
  finalText.className = "final-text";
  renderMarkdown(finalText, text ?? "");
  b.appendChild(finalText);
  return b;
}

function mkAnswerBlock() {
  const b = mkBlock("answer");

  const title = document.createElement("div");
  title.className = "title";
  title.textContent = "Answer";
  b.appendChild(title);

  const thinkingDetails = document.createElement("details");
  thinkingDetails.className = "section";
  thinkingDetails.open = true;

  const thinkingSummary = document.createElement("summary");
  thinkingSummary.textContent = "Workflow Thinking";
  thinkingDetails.appendChild(thinkingSummary);

  const thinkingPre = document.createElement("pre");
  thinkingDetails.appendChild(thinkingPre);

  const statusDetails = document.createElement("details");
  statusDetails.className = "section";
  statusDetails.open = true;

  const statusSummary = document.createElement("summary");
  statusSummary.textContent = "Status / Logs";

  const thinkingHint = document.createElement("div");
  thinkingHint.className = "thinking-hint";
  thinkingHint.textContent = "";
  // thinkingHint.textContent = "这里展示的是面向用户的工作流思考摘要，不是模型原始 CoT。";
  thinkingDetails.appendChild(thinkingHint);
  statusDetails.appendChild(statusSummary);

  const statusPre = document.createElement("pre");
  statusDetails.appendChild(statusPre);

  const evidenceDetails = document.createElement("details");
  evidenceDetails.className = "section";
  evidenceDetails.open = false;

  const evidenceSummary = document.createElement("summary");
  evidenceSummary.textContent = "Evidence";
  evidenceDetails.appendChild(evidenceSummary);

  const evidenceList = document.createElement("div");
  evidenceList.className = "evidence-list";
  evidenceDetails.appendChild(evidenceList);

  const finalSection = document.createElement("div");
  finalSection.className = "section";

  const finalTitle = document.createElement("div");
  finalTitle.className = "final-title";
  finalTitle.textContent = "Final Answer";
  finalSection.appendChild(finalTitle);

  const finalText = document.createElement("div");
  finalText.className = "final-text";
  finalSection.appendChild(finalText);

  b.appendChild(thinkingDetails);
  b.appendChild(statusDetails);
  b.appendChild(evidenceDetails);
  b.appendChild(finalSection);

  return {
    block: b,
    thinkingDetails,
    thinkingPre,
    statusDetails,
    statusPre,
    evidenceDetails,
    evidenceList,
    finalText,
    answerStarted: false,
    answerMd: "",
  };
}

function finalizeAnswerBlock(answerState) {
  if (!answerState) return;

  const noThinking = !answerState.thinkingPre.textContent.trim();
  const noStatus = !answerState.statusPre.textContent.trim();
  const noEvidence = !answerState.evidenceList.childElementCount;

  answerState.thinkingDetails.hidden = noThinking;
  answerState.statusDetails.hidden = noStatus;
  answerState.evidenceDetails.hidden = noEvidence;

  if (answerState.answerStarted) {
    if (!noThinking) answerState.thinkingDetails.open = false;
    if (!noStatus) answerState.statusDetails.open = false;
    if (!noEvidence) answerState.evidenceDetails.open = false;
  }
}

function applyAnswerMessage(answerState, msg) {
  if (!answerState || !msg || !msg.type) return;

  if (msg.type === "thinking_round") {
    answerState.thinkingPre.textContent += `\n\n【思考 ${msg.round}】\n`;
    return;
  }

  if (msg.type === "thinking") {
    answerState.thinkingPre.textContent += msg.text ?? "";
    return;
  }

  if (msg.type === "status") {
    answerState.statusPre.textContent += msg.text ?? "";
    return;
  }

  if (msg.type === "evidence") {
    const items = msg.items ?? [];
    answerState.evidenceList.innerHTML = "";
    for (const it of items) {
      renderEvidenceItem(it, answerState.evidenceList);
    }
    return;
  }

  if (msg.type === "answer") {
    if (!answerState.answerStarted) {
      answerState.answerStarted = true;
      answerState.thinkingDetails.open = false;
      answerState.statusDetails.open = false;
      answerState.evidenceDetails.open = false;
    }
    answerState.answerMd = (answerState.answerMd ?? "") + (msg.text ?? "");
    renderMarkdown(answerState.finalText, answerState.answerMd);
    return;
  }

  if (msg.type === "error") {
    const trace = msg.trace ? `\n${msg.trace}` : "";
    answerState.answerMd = (answerState.answerMd ?? "") + `\n\n\`\`\`\n[error] ${msg.message ?? "unknown"}${trace}\n\`\`\`\n`;
    renderMarkdown(answerState.finalText, answerState.answerMd);
  }
}

function storedItemToStreamMessage(item) {
  if (!item || typeof item !== "object") return null;

  const kind = String(item.kind || "message");
  const payload = item.payload && typeof item.payload === "object" ? item.payload : null;

  if (kind === "message") {
    if (item.role === "assistant") {
      return { type: "answer", text: item.content ?? "" };
    }
    return null;
  }

  if (payload && payload.type) {
    return payload;
  }

  if (kind === "thinking_round") {
    const round = payload && payload.round != null ? payload.round : "?";
    return { type: "thinking_round", round };
  }

  if (kind === "thinking" || kind === "status") {
    return { type: kind, text: item.content ?? "" };
  }

  if (kind === "evidence") {
    return { type: "evidence", items: [] };
  }

  if (kind === "error") {
    return { type: "error", message: item.content ?? "unknown" };
  }

  return null;
}

function renderEvidenceItem(item, evidenceList) {
  const wrap = document.createElement("div");
  wrap.className = "evidence-item";

  const header = document.createElement("div");
  header.className = "evidence-header";
  header.textContent = `${item.cid}  score=${Number(item.score ?? 0).toFixed(4)}`;
  wrap.appendChild(header);

  const graphEl = document.createElement("div");
  graphEl.className = "graph";
  wrap.appendChild(graphEl);

  evidenceList.appendChild(wrap);

  const nodes = (item.graph?.nodes ?? []).map((n) => ({
    data: {
      id: n.eid,
      label: n.name || (n.labels?.[0] ?? n.eid),
      eid: n.eid,
      labels: n.labels ?? [],
    },
  }));

  const edges = (item.graph?.edges ?? []).map((e) => ({
    data: {
      id: e.rid,
      source: e.source,
      target: e.target,
      label: e.type || "REL",
      rid: e.rid,
      type: e.type,
    },
  }));

  const cy = cytoscape({
    container: graphEl,
    elements: { nodes, edges },
    layout: { name: "cose" },
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": 10,
          "text-wrap": "wrap",
          "text-max-width": 56,
          "text-valign": "center",
          "text-halign": "center",
          "background-color": "#2b7cff",
          color: "#111",
          "border-width": 1,
          "border-color": "#111",
          shape: "ellipse",
          width: 60,
          height: 60,
        },
      },
      {
        selector: "edge",
        style: {
          label: "data(label)",
          "font-size": 9,
          "curve-style": "bezier",
          "target-arrow-shape": "triangle",
          width: 1.5,
          "line-color": "#333",
          "target-arrow-color": "#333",
          "text-background-opacity": 1,
          "text-background-color": "#fff",
          "text-background-padding": "2px",
        },
      },
    ],
  });

  cy.on("tap", "node", async (evt) => {
    const n = evt.target.data();
    try {
      const res = await fetch(`/api/node/${encodeURIComponent(n.eid)}`);
      const data = await res.json();
      showDetail(`Node ${n.eid}`, data);
    } catch (e) {
      showDetail("Error", { message: String(e) });
    }
  });

  cy.on("tap", "edge", async (evt) => {
    const e = evt.target.data();
    try {
      const res = await fetch(`/api/rel/${encodeURIComponent(e.rid)}`);
      const data = await res.json();
      showDetail(`Rel ${e.rid}`, data);
    } catch (err) {
      showDetail("Error", { message: String(err) });
    }
  });
}

const state = {
  ws: null,
  wsReady: false,
  conversations: [],
  currentConversationId: "",
  active: null,
};

function ensureNoActiveConversationAction() {
  if (state.active) {
    alert("当前会话正在生成回答，请等待完成后再切换或编辑会话。");
    return false;
  }
  return true;
}

async function requestJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const message = data?.detail || data?.message || text || `${res.status} ${res.statusText}`;
    throw new Error(String(message));
  }
  return data;
}

function formatBeijingTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function updateConversationHeader() {
  const current = state.conversations.find((item) => item.id === state.currentConversationId);
  if (!current) {
    conversationTitleEl.textContent = "未选择会话";
    conversationMetaEl.textContent = "";
    return;
  }
  conversationTitleEl.textContent = current.title || "未命名会话";
  const count = Number(current.message_count ?? 0);
  conversationMetaEl.textContent = `ID: ${current.id} · 消息数: ${count} · 更新时间: ${formatBeijingTime(current.updated_at)}（北京时间）`;
}

function renderConversationList() {
  conversationListEl.innerHTML = "";
  for (const item of state.conversations) {
    const btn = document.createElement("button");
    btn.className = "conversation-item";
    if (item.id === state.currentConversationId) {
      btn.classList.add("active");
    }

    const title = document.createElement("div");
    title.className = "conversation-item-title";
    title.textContent = item.title || "未命名会话";
    btn.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "conversation-item-meta";
    meta.textContent = `${item.message_count ?? 0}条消息`;
    btn.appendChild(meta);

    btn.addEventListener("click", () => {
      selectConversation(item.id);
    });

    conversationListEl.appendChild(btn);
  }
  updateConversationHeader();
}

async function refreshConversations({ autoSelect = true } = {}) {
  const data = await requestJson("/api/conversations");
  state.conversations = Array.isArray(data?.items) ? data.items : [];

  if (!state.conversations.length) {
    const created = await requestJson("/api/conversations", { method: "POST", body: JSON.stringify({}) });
    state.conversations = [created];
  }

  if (autoSelect) {
    const exists = state.conversations.some((item) => item.id === state.currentConversationId);
    if (!exists) {
      state.currentConversationId = state.conversations[0]?.id || "";
    }
  }

  renderConversationList();
}

async function loadMessages(conversationId) {
  messagesEl.innerHTML = "";
  if (!conversationId) {
    updateConversationHeader();
    return;
  }
  const data = await requestJson(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
  const items = Array.isArray(data?.items) ? data.items : [];

  let currentAnswer = null;
  const historyAnswers = [];

  const ensureHistoryAnswerBlock = () => {
    if (currentAnswer) return currentAnswer;
    currentAnswer = mkAnswerBlock();
    messagesEl.appendChild(currentAnswer.block);
    historyAnswers.push(currentAnswer);
    return currentAnswer;
  };

  for (const item of items) {
    const kind = String(item.kind || "message");

    if (kind === "message" && item.role === "user") {
      currentAnswer = null;
      messagesEl.appendChild(mkQuestionBlock(item.content ?? ""));
      continue;
    }

    const streamMsg = storedItemToStreamMessage(item);
    if (!streamMsg) continue;

    const answerState = ensureHistoryAnswerBlock();
    applyAnswerMessage(answerState, streamMsg);
  }

  for (const answerState of historyAnswers) {
    finalizeAnswerBlock(answerState);
  }

  updateConversationHeader();
  scrollMessagesToBottom();
}

async function selectConversation(conversationId) {
  if (!ensureNoActiveConversationAction()) return;
  state.currentConversationId = conversationId;
  renderConversationList();
  await loadMessages(conversationId);
}

async function createConversation() {
  if (!ensureNoActiveConversationAction()) return;
  const title = window.prompt("请输入新会话名称（可留空）", "") ?? "";
  const created = await requestJson("/api/conversations", {
    method: "POST",
    body: JSON.stringify(title.trim() ? { title: title.trim() } : {}),
  });
  await refreshConversations({ autoSelect: false });
  state.currentConversationId = created.id;
  renderConversationList();
  await loadMessages(created.id);
}

async function renameConversation() {
  if (!ensureNoActiveConversationAction()) return;
  const current = state.conversations.find((item) => item.id === state.currentConversationId);
  if (!current) {
    alert("请先选择会话。");
    return;
  }
  const title = window.prompt("请输入新的会话名称", current.title || "") ?? "";
  if (!title.trim()) return;
  await requestJson(`/api/conversations/${encodeURIComponent(current.id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title: title.trim() }),
  });
  await refreshConversations({ autoSelect: false });
  renderConversationList();
  updateConversationHeader();
}

async function deleteConversation() {
  if (!ensureNoActiveConversationAction()) return;
  const current = state.conversations.find((item) => item.id === state.currentConversationId);
  if (!current) {
    alert("请先选择会话。");
    return;
  }
  const ok = window.confirm(`确认删除会话“${current.title || current.id}”？`);
  if (!ok) return;
  await requestJson(`/api/conversations/${encodeURIComponent(current.id)}`, { method: "DELETE" });
  state.currentConversationId = "";
  await refreshConversations({ autoSelect: true });
  await loadMessages(state.currentConversationId);
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);

  state.ws.onopen = () => {
    state.wsReady = true;
  };

  state.ws.onclose = () => {
    state.wsReady = false;
    setTimeout(connectWS, 1000);
  };

  state.ws.onmessage = (ev) => {
    if (!state.active) return;
    let msg = null;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      msg = { type: "status", text: ev.data };
    }

    if (msg.type === "conversation_meta") {
      if (msg.conversation_id) {
        state.currentConversationId = msg.conversation_id;
      }
      return;
    }

    if (["thinking_round", "thinking", "status", "evidence", "answer", "error"].includes(msg.type)) {
      applyAnswerMessage(state.active, msg);
      if (msg.type === "answer" || msg.type === "error") {
        scrollMessagesToBottom();
      }
      return;
    }

    if (msg.type === "done") {
      const finishedConversationId = state.active.conversationId;
      state.active = null;
      refreshConversations({ autoSelect: false }).then(() => {
        if (state.currentConversationId === finishedConversationId) {
          renderConversationList();
          updateConversationHeader();
        }
      }).catch((err) => {
        console.error(err);
      });
      return;
    }
  };
}

async function sendQuestion() {
  const text = questionInput.value.trim();
  if (!text) return;
  if (!state.wsReady || !state.ws) {
    alert("WebSocket not ready, please wait...");
    return;
  }
  if (!state.currentConversationId) {
    alert("请先选择一个会话。");
    return;
  }
  if (state.active) {
    alert("当前会话正在生成回答，请稍候。");
    return;
  }

  const qBlock = mkQuestionBlock(text);
  const a = mkAnswerBlock();
  messagesEl.appendChild(qBlock);
  messagesEl.appendChild(a.block);
  state.active = { ...a, conversationId: state.currentConversationId };

  state.ws.send(JSON.stringify({
    type: "question",
    text,
    conversation_id: state.currentConversationId,
  }));

  questionInput.value = "";
  questionInput.style.height = "auto";
  scrollMessagesToBottom();
}

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) {
    e.preventDefault();
    sendQuestion().catch((err) => {
      alert(String(err));
    });
  }
});

sendBtn.addEventListener("click", () => {
  sendQuestion().catch((err) => {
    alert(String(err));
  });
});

newConversationBtn.addEventListener("click", () => {
  createConversation().catch((err) => {
    alert(String(err));
  });
});

renameConversationBtn.addEventListener("click", () => {
  renameConversation().catch((err) => {
    alert(String(err));
  });
});

deleteConversationBtn.addEventListener("click", () => {
  deleteConversation().catch((err) => {
    alert(String(err));
  });
});

async function bootstrap() {
  connectWS();
  await refreshConversations({ autoSelect: true });
  await loadMessages(state.currentConversationId);
}

bootstrap().catch((err) => {
  alert(String(err));
});
