const streamEl = document.getElementById("stream");
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

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
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

function mkBlock(type) {
  const div = document.createElement("div");
  div.className = `block ${type}`;
  return div;
}

function mkInputBlock() {
  const b = mkBlock("input");

  const ta = document.createElement("textarea");
  ta.placeholder = "输入问题，Ctrl+Enter 发送";
  b.appendChild(ta);

  enableAutoResizeTextarea(ta);

  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      const text = ta.value.trim();
      if (!text) return;
      sendQuestion(text, b);
    }
  });

  setTimeout(() => ta.focus(), 50);

  return b;
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
  thinkingSummary.textContent = "Model Thinking";
  thinkingDetails.appendChild(thinkingSummary);

  const thinkingPre = document.createElement("pre");
  thinkingPre.dataset.role = "thinking";
  thinkingDetails.appendChild(thinkingPre);

  const statusDetails = document.createElement("details");
  statusDetails.className = "section";
  statusDetails.open = true;

  const statusSummary = document.createElement("summary");
  statusSummary.textContent = "Status / Logs";
  statusDetails.appendChild(statusSummary);

  const statusPre = document.createElement("pre");
  statusPre.dataset.role = "status";
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
  finalTitle.style.fontWeight = "700";
  finalTitle.textContent = "Final Answer";
  finalSection.appendChild(finalTitle);

  const finalText = document.createElement("div");
  finalText.className = "final-text";
  finalText.dataset.role = "answer";
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
  };
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

let ws = null;
let wsReady = false;
let active = null;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    wsReady = true;
  };

  ws.onclose = () => {
    wsReady = false;
    setTimeout(connectWS, 1000);
  };

  ws.onmessage = (ev) => {
    if (!active) return;
    let msg = null;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      msg = { type: "status", text: ev.data };
    }

    if (msg.type === "thinking_round") {
      active.thinkingPre.textContent += `\n\n【思考 ${msg.round}】\n`;
      return;
    }

    if (msg.type === "thinking") {
      active.thinkingPre.textContent += msg.text ?? "";
      return;
    }

    if (msg.type === "status") {
      active.statusPre.textContent += msg.text ?? "";
      return;
    }

    if (msg.type === "evidence") {
      const items = msg.items ?? [];
      active.evidenceList.innerHTML = "";
      for (const it of items) {
        renderEvidenceItem(it, active.evidenceList);
      }
      return;
    }

    if (msg.type === "answer") {
      if (!active.answerStarted) {
        active.answerStarted = true;
        active.thinkingDetails.open = false;
        active.statusDetails.open = false;
        active.evidenceDetails.open = false;
      }
      active.answerMd = (active.answerMd ?? "") + (msg.text ?? "");
      renderMarkdown(active.finalText, active.answerMd);
      scrollToBottom();
      return;
    }

    if (msg.type === "error") {
      const trace = msg.trace ? `\n${msg.trace}` : "";
      active.answerMd = (active.answerMd ?? "") + `\n\n\
\`\`\`\n[error] ${msg.message ?? "unknown"}${trace}\n\`\`\`\n`;
      renderMarkdown(active.finalText, active.answerMd);
      scrollToBottom();
      return;
    }

    if (msg.type === "done") {
      active = null;
      streamEl.appendChild(mkInputBlock());
      scrollToBottom();
      return;
    }
  };
}

connectWS();

function sendQuestion(text, inputBlock) {
  if (!wsReady || !ws) {
    alert("WebSocket not ready, please wait...");
    return;
  }

  const qBlock = mkQuestionBlock(text);
  const a = mkAnswerBlock();

  streamEl.replaceChild(qBlock, inputBlock);
  streamEl.appendChild(a.block);
  active = a;

  ws.send(JSON.stringify({ type: "question", text }));
  scrollToBottom();
}

streamEl.appendChild(mkInputBlock());
