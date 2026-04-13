# -*- coding: utf-8 -*-
"""
agent.py

Overall QA orchestration logic.

Design:
- After receiving a question, the agent first thinks about whether retrieval is needed.
- Retrieval is a tool that the agent may (or may not) use.
- The agent decides when to search, and provides search keywords.
- Search in this branch is a direct Neo4j scan and does not rely on any index.

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
你有一个工具：graph_search，用于在 Neo4j 图中按关键词直接检索证据（不依赖索引）。

当前步骤：你只能“决定是否要调用工具”，不要回答用户问题本身。
你必须只输出一段 JSON（不要加任何其它文字/解释/Markdown），格式二选一：
1) {"action":"search","keywords":["关键词1","关键词2", ...]}
2) {"action":"final"}

要求：
- keywords 应该是简短的实体名/关键短语/术语（1-6个），用于图中检索。
- 如果现有证据不足以严谨回答，就选 action=search，并给出新的 keywords。
- 用户当前问题可能依赖前面对话，请结合 Conversation History 一起判断。
"""

DECIDE_TEMPLATE = """Conversation History（可能为空）：
{history_text}

用户当前问题：
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
- 用户当前问题可能依赖 Conversation History，请先正确理解历史对话，再基于 Context 作答。

Conversation History:
{history_text}

Context:
{context}

Question:
{query_text}

Answer:
""".strip()


def format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_messages: int = 12, max_chars: int = 1200) -> str:
    if not chat_history:
        return "(空)"

    lines: List[str] = []
    for msg in chat_history[-max_messages:]:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        if role == "assistant":
            who = "Assistant"
        elif role == "system":
            who = "System"
        else:
            who = "User"
        lines.append(f"{who}: {content}")

    return "\n".join(lines) if lines else "(空)"


def build_answer_messages(
    context: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    prompt = ANSWER_TEMPLATE.format(
        context=context,
        query_text=question,
        history_text=format_chat_history(chat_history),
    )
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
    for item in evidence_items[:max_items]:
        cid = item.get("cid", "")
        node = item.get("node") or {}
        name = (node.get("name") or "").strip()
        labels = node.get("labels") or []
        labels_text = ",".join([str(x) for x in labels]) if labels else ""
        if name:
            lines.append(f"- {cid}: {name} ({labels_text})")
        else:
            lines.append(f"- {cid}: ({labels_text})")
    if len(evidence_items) > max_items:
        lines.append(f"... (+{len(evidence_items) - max_items} more)")
    return "\n".join(lines)


def _renumber_context_and_items(
    existing_items: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
    new_context: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Append new evidence to existing with stable unique C-ids."""
    offset = len(existing_items)
    if offset <= 0:
        return new_items, new_context

    renumbered: List[Dict[str, Any]] = []
    for i, item in enumerate(new_items, start=1):
        cid_new = f"C{offset + i}"
        item2 = dict(item)
        item2["cid"] = cid_new
        renumbered.append(item2)

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
    database: Optional[str] = None

    def search(self, keywords: Any) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        from graphrag import retrieve_evidence

        return retrieve_evidence(
            driver=self.driver,
            keywords=keywords,
            top_k=int(self.top_k),
            expand_k=int(self.expand_k),
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
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Synchronous agent loop (used by CLI)."""
    q = (question or "").strip()
    if not q:
        return "Empty question", []

    all_items: List[Dict[str, Any]] = []
    all_context_parts: List[str] = []
    history_text = format_chat_history(chat_history)

    for round_idx in range(1, max_search_rounds + 1):
        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(
            question=q,
            evidence_brief=evidence_brief,
            history_text=history_text,
        )
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": decide_prompt},
        ]

        if print_debug:
            print(f"\n[agent] decide round={round_idx}")

        raw = llm_get_response(decide_messages, False, "stream")
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
            continue

        merged_items, renumbered_ctx = _renumber_context_and_items(all_items, items, ctx)
        all_items = merged_items
        all_context_parts.append(renumbered_ctx)

        print("[References]")
        print(renumbered_ctx)
        print("=" * 80)

    context_text = "\n\n".join([part for part in all_context_parts if part.strip()]).strip()
    if not context_text:
        final = "资料不足，无法确定（未获得可用的检索上下文）。"
        print("\nA>")
        print(final)
        print()
        return final, all_items

    messages = build_answer_messages(context=context_text, question=q, chat_history=chat_history)
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
    llm_call: Callable[[List[Dict[str, str]], bool, bool, bool], Awaitable[str]],
    search_tool: GraphSearchTool,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    max_search_rounds: int = 3,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Async agent loop (used by WebSocket server)."""
    q = (question or "").strip()
    if not q:
        await emit({"type": "error", "message": "Empty question"})
        await emit({"type": "done"})
        return ""

    all_items: List[Dict[str, Any]] = []
    all_context_parts: List[str] = []
    history_text = format_chat_history(chat_history)

    for round_idx in range(1, max_search_rounds + 1):
        await emit({"type": "status", "phase": "agent", "text": f"判断是否需要检索（round={round_idx})...\n"})

        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(
            question=q,
            evidence_brief=evidence_brief,
            history_text=history_text,
        )
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": decide_prompt},
        ]

        try:
            decision_answer_text = await llm_call(decide_messages, False, False, False)
        except Exception as exc:
            await emit({"type": "error", "message": f"LLM failed: {exc}"})
            await emit({"type": "done"})
            return ""

        decision = parse_decision((decision_answer_text or "").strip())

        if not decision:
            await emit({"type": "status", "phase": "agent", "text": "解析决策失败，直接进入最终回答（可能无检索）。\n"})
            break

        if decision["action"] == "final":
            await emit({"type": "status", "phase": "agent", "text": "决定：不再检索，直接作答。\n"})
            break

        keywords = decision["keywords"]
        await emit({"type": "status", "phase": "tool", "text": f"启用检索 graph_search keywords={keywords}\n"})

        try:
            items, ctx, meta = search_tool.search(keywords)
        except Exception as exc:
            await emit({"type": "error", "message": f"Retrieve failed: {exc}"})
            await emit({"type": "done"})
            return ""

        await emit({"type": "status", "phase": "tool", "text": f"检索完成：{meta}\n"})

        if not items or not ctx.strip():
            await emit({"type": "status", "phase": "tool", "text": "检索结果为空，尝试重新思考关键词...\n"})
            continue

        merged_items, renumbered_ctx = _renumber_context_and_items(all_items, items, ctx)
        all_items = merged_items
        all_context_parts.append(renumbered_ctx)

        await emit({"type": "evidence", "items": all_items})

    context_text = "\n\n".join([part for part in all_context_parts if part.strip()]).strip()

    if not context_text:
        final_text = "资料不足，无法确定（未获得可用的检索上下文）。"
        await emit({"type": "answer", "text": final_text})
        await emit({"type": "done"})
        return final_text

    await emit({"type": "status", "phase": "llm", "text": "开始生成答案...\n"})
    messages = build_answer_messages(context=context_text, question=q, chat_history=chat_history)

    try:
        answer_text = await llm_call(messages, True, True, True)
    except Exception as exc:
        await emit({"type": "error", "message": f"LLM failed: {exc}"})
        await emit({"type": "done"})
        return ""

    return answer_text.split("</think>")[-1].strip()
