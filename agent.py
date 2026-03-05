# -*- coding: utf-8 -*-
"""
agent.py

Overall QA orchestration logic.

Design (updated):
- After receiving a question, the agent FIRST thinks.
- Retrieval is downgraded to a TOOL that the agent may (or may not) use.
- The agent itself decides when to search, and provides search keywords.
- If evidence is insufficient, the agent may call search multiple times.

This file contains no GPU imports and can be safely imported by server.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Prompts
# =============================================================================

DECIDE_SYSTEM = """你是一个答题智能体（agent）。
你有一个工具：graph_search，用于从Neo4j知识图谱中检索证据。

当前步骤：你只能“决定是否要调用工具”，不要回答用户问题本身。
你必须只输出一段 JSON（不要加任何其它文字/解释/Markdown），格式二选一：
1) {"action":"search","keywords":["关键词1","关键词2", ...]}
2) {"action":"final"}

要求：
- keywords 应该是简短的实体名/关键短语/术语（1-6个），用于图谱检索。
- 如果现有证据不足以严谨回答，就选 action=search，并给出新的 keywords。
"""

DECIDE_TEMPLATE = """用户问题：
{question}

当前已获得的证据摘要（可能为空）：
{evidence_brief}
"""

ANSWER_SYSTEM = "You are a careful assistant that strictly grounds answers in the provided context."

ANSWER_TEMPLATE = r"""
你是一个严谨的助手。只能使用下面提供的 Context 来回答问题：
- 如果 Context 信息不足，请明确说“资料不足，无法确定”，并说明缺了什么。
- 每当你引用 Context 中的事实，都要用 [C1] [C2] 这样的引用标注。
- 不要编造任何 Context 中不存在的细节。

Context:
{context}

Question:
{query_text}

Answer:
""".strip()


def build_answer_messages(context: str, question: str) -> List[Dict[str, str]]:
    prompt = ANSWER_TEMPLATE.format(context=context, query_text=question)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# Helpers
# =============================================================================

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_obj(text: str) -> Optional[str]:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text.strip())
    return m.group(0).strip() if m else None


def parse_decision(text: str) -> Optional[Dict[str, Any]]:
    """Parse the agent decision JSON from model output (robust to extra tokens)."""
    raw = _extract_json_obj(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    action = str(obj.get("action", "")).strip().lower()
    if action not in ("search", "final"):
        return None
    if action == "search":
        kws = obj.get("keywords", [])
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list):
            return None
        cleaned = [str(k).strip() for k in kws if str(k).strip()]
        obj["keywords"] = cleaned[:6]
        if not obj["keywords"]:
            return None
    return obj


def brief_evidence(evidence_items: List[Dict[str, Any]], max_items: int = 8) -> str:
    if not evidence_items:
        return "(空)"
    lines: List[str] = []
    for it in evidence_items[:max_items]:
        cid = it.get("cid", "")
        node = it.get("node") or {}
        name = (node.get("name") or "").strip()
        labels = node.get("labels") or []
        labels_s = ",".join([str(x) for x in labels]) if labels else ""
        if name:
            lines.append(f"- {cid}: {name} ({labels_s})")
        else:
            lines.append(f"- {cid}: ({labels_s})")
    if len(evidence_items) > max_items:
        lines.append(f"... (+{len(evidence_items) - max_items} more)")
    return "\n".join(lines)


def _renumber_context_and_items(
    existing_items: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
    new_context: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Append new evidence to existing with stable unique C-ids.

    - existing: C1..CN
    - new:       C1..CM   -> renumbered to C(N+1)..C(N+M)
    """
    offset = len(existing_items)
    if offset <= 0:
        return new_items, new_context

    # Renumber items
    renumbered: List[Dict[str, Any]] = []
    for i, it in enumerate(new_items, start=1):
        cid_new = f"C{offset + i}"
        it2 = dict(it)
        it2["cid"] = cid_new
        renumbered.append(it2)

    # Renumber context text: replace "Ck:" headings only (best-effort).
    ctx = new_context
    for i in range(1, len(new_items) + 1):
        old = f"C{i}:"
        new = f"C{offset + i}:"
        ctx = re.sub(rf"(?m)^{re.escape(old)}$", new, ctx)

    merged_items = existing_items + renumbered
    return merged_items, ctx


# =============================================================================
# Search tool wrapper
# =============================================================================

@dataclass
class GraphSearchTool:
    """A thin wrapper around graphrag.retrieve_evidence."""

    driver: Any
    top_k: int
    expand_k: int
    fulltext_index_name: str
    database: Optional[str] = None

    def search(self, keywords: Any) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        from graphrag import retrieve_evidence  # local import (lightweight)

        return retrieve_evidence(
            driver=self.driver,
            keywords=keywords,
            top_k=int(self.top_k),
            expand_k=int(self.expand_k),
            fulltext_index_name=str(self.fulltext_index_name),
            database=self.database,
        )


# =============================================================================
# Agent core logic
# =============================================================================

def run_agent_sync(
    *,
    question: str,
    llm_get_response: Callable[[List[Dict[str, str]], bool, str], str],
    search_tool: GraphSearchTool,
    max_search_rounds: int = 3,
    print_debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Synchronous agent loop (used by CLI).

    llm_get_response(messages, think, print_type) -> raw_text
    """
    q = (question or "").strip()
    if not q:
        return "Empty question", []

    all_items: List[Dict[str, Any]] = []
    all_context_parts: List[str] = []

    # Decide/search loop
    for r in range(1, max_search_rounds + 1):
        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(question=q, evidence_brief=evidence_brief)
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": decide_prompt},
        ]

        if print_debug:
            print(f"\n[agent] decide round={r}")

        raw = llm_get_response(decide_messages, True, "stream")
        directive_text = raw.split("</think>")[-1].strip()
        decision = parse_decision(directive_text)

        if print_debug:
            print(f"[agent] decision_raw={directive_text!r}")
            print(f"[agent] decision={decision}")

        if not decision or decision["action"] == "final":
            break

        keywords = decision["keywords"]
        print(f"\n[tool] graph_search keywords: {keywords}\n")

        items, ctx, meta = search_tool.search(keywords)

        if print_debug:
            print(f"[tool] meta={meta}")

        if not items or not ctx.strip():
            print("[tool] no evidence returned.\n")
            # continue to next decide round, letting the model adjust keywords
            continue

        # Merge and renumber
        merged_items, renumbered_ctx = _renumber_context_and_items(all_items, items, ctx)
        all_items = merged_items
        all_context_parts.append(renumbered_ctx)

        # Print references like previous CLI behavior (full context chunk)
        print("[References]")
        print(renumbered_ctx)
        print("=" * 80)

    # Final answer
    context_text = "\n\n".join([p for p in all_context_parts if p.strip()]).strip()
    if not context_text:
        final = "资料不足，无法确定（未获得可用的检索上下文）。"
        print("\nA>")
        print(final)
        print()
        return final, all_items

    messages = build_answer_messages(context=context_text, question=q)
    raw = llm_get_response(messages, True, "stream")
    answer = raw.split("</think>")[-1].strip()

    print("\nA>")
    print("=" * 80)
    print(answer)
    print()

    return answer, all_items


async def run_agent_async(
    *,
    question: str,
    llm_call: Callable[[List[Dict[str, str]], bool, bool], Awaitable[str]],
    search_tool: GraphSearchTool,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    max_search_rounds: int = 3,
) -> None:
    """Async agent loop (used by WebSocket server).

    llm_call(messages, forward_answer_events, forward_done_event) -> answer_text (concatenated)
      - For *decision* steps: forward_answer_events=False, forward_done_event=False
      - For *final* step:    forward_answer_events=True,  forward_done_event=True
    """
    q = (question or "").strip()
    if not q:
        await emit({"type": "error", "message": "Empty question"})
        await emit({"type": "done"})
        return

    all_items: List[Dict[str, Any]] = []
    all_context_parts: List[str] = []

    # Decide/search loop
    for r in range(1, max_search_rounds + 1):
        await emit({"type": "thinking", "phase": "agent", "text": f"思考：是否需要检索（round={r})...\n"})

        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(question=q, evidence_brief=evidence_brief)
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": decide_prompt},
        ]

        try:
            decision_answer_text = await llm_call(decide_messages, False, False)
        except Exception as e:
            await emit({"type": "error", "message": f"LLM failed: {e}"})
            await emit({"type": "done"})
            return

        decision = parse_decision((decision_answer_text or "").strip())

        if not decision:
            await emit({"type": "thinking", "phase": "agent", "text": "解析决策失败，直接进入最终回答（可能无检索）。\n"})
            break

        if decision["action"] == "final":
            await emit({"type": "thinking", "phase": "agent", "text": "决定：不再检索，直接作答。\n"})
            break

        keywords = decision["keywords"]
        await emit({"type": "thinking", "phase": "tool", "text": f"启用检索 graph_search keywords={keywords}\n"})

        try:
            items, ctx, meta = search_tool.search(keywords)
        except Exception as e:
            await emit({"type": "error", "message": f"Retrieve failed: {e}"})
            await emit({"type": "done"})
            return

        await emit({"type": "thinking", "phase": "tool", "text": f"检索完成：{meta}\n"})

        if not items or not ctx.strip():
            await emit({"type": "thinking", "phase": "tool", "text": "检索结果为空，尝试重新思考关键词...\n"})
            continue

        # Merge and renumber
        merged_items, renumbered_ctx = _renumber_context_and_items(all_items, items, ctx)
        all_items = merged_items
        all_context_parts.append(renumbered_ctx)

        # Send cumulative evidence to UI (keeps UI logic simple)
        await emit({"type": "evidence", "items": all_items})

    # Final answer
    context_text = "\n\n".join([p for p in all_context_parts if p.strip()]).strip()

    if not context_text:
        await emit({"type": "answer", "text": "资料不足，无法确定（未获得可用的检索上下文）。"})
        await emit({"type": "done"})
        return

    await emit({"type": "thinking", "phase": "llm", "text": "开始生成答案...\n"})
    messages = build_answer_messages(context=context_text, question=q)

    # Stream final answer from GPU worker (server side)
    try:
        await llm_call(messages, True, True)
    except Exception as e:
        await emit({"type": "error", "message": f"LLM failed: {e}"})
        await emit({"type": "done"})
