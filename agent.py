# -*- coding: utf-8 -*-
"""
agent.py

Overall QA orchestration logic.

Design:
- After receiving a question, the agent first thinks about whether a tool is needed.
- Tools are optional helpers; the agent decides whether to use one and which one to use.
- graph_search is currently one small general-purpose tool among the available tools.
- The user-facing "thinking" stream is a structured workflow summary, not raw model CoT.

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

DECIDE_SYSTEM = """你是一个答题智能体（agent）的流程规划器。

你可以选择：
1) 直接进入最终回答；
2) 先调用一个工具，再继续判断。

可用工具：
{tools_text}

当前步骤：你只能决定“下一步怎么做”，不要回答用户问题本身。
你必须只输出一段 JSON（不要加任何其它文字/解释/Markdown），格式二选一：
1) {{"action":"use_tool","tool":"工具名","keywords":["关键词1","关键词2", ...],"reason":"一句简短中文原因"}}
2) {{"action":"final","reason":"一句简短中文原因"}}

要求：
- 如果问题依赖外部证据、数据库内容或工具结果，优先选择 use_tool。
- 如果可以直接基于当前对话、稳定知识或一般推理回答，选择 final。
- tool 必须是可用工具之一。
- keywords 应该是简短的实体名/关键短语/术语（1-6个）；如果所选工具不需要关键词，也保留一个最核心短语。
- 用户当前问题可能依赖前面对话，请结合 Conversation History 一起判断。
- reason 必须简短、直接、给用户可读。
"""

DECIDE_TEMPLATE = """Conversation History（可能为空）：
{history_text}

用户当前问题：
{question}

当前已获得的证据摘要（可能为空）：
{evidence_brief}
"""

ANSWER_WITH_CONTEXT_SYSTEM = "You are a careful assistant that strictly grounds answers in the provided context."

ANSWER_WITH_CONTEXT_TEMPLATE = r"""
你是一个严谨的助手。只能使用下面提供的 Context 来回答问题：
- 如果 Context 信息不足，请明确说“资料不足，无法确定”，并说明缺了什么。
- 每当你引用 Context 中的事实，都要用 [C1] [C2] 这样的引用标注。
- 如果证据之间并不完全一致，请明确指出不确定点。
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

ANSWER_DIRECT_SYSTEM = "You are a careful assistant."

ANSWER_DIRECT_TEMPLATE = r"""
你是一个严谨的助手。当前没有使用外部工具证据，或者当前问题不必依赖工具。
请基于 Conversation History、稳定知识和一般推理来回答：
- 不要假装你查过数据库或外部资料。
- 如果这个问题本来依赖最新事实、专门资料或你并不确定的信息，请明确说“不确定”，并说明还需要什么证据。
- 用户当前问题可能依赖 Conversation History，请先正确理解历史对话。
- 回答要直接、清楚，不要输出 JSON。

Conversation History:
{history_text}

Question:
{query_text}

Answer:
""".strip()


# =============================================================================
# History and prompt builders
# =============================================================================


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
    prompt = ANSWER_WITH_CONTEXT_TEMPLATE.format(
        context=context,
        query_text=question,
        history_text=format_chat_history(chat_history),
    )
    return [
        {"role": "system", "content": ANSWER_WITH_CONTEXT_SYSTEM},
        {"role": "user", "content": prompt},
    ]



def build_direct_answer_messages(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    prompt = ANSWER_DIRECT_TEMPLATE.format(
        query_text=question,
        history_text=format_chat_history(chat_history),
    )
    return [
        {"role": "system", "content": ANSWER_DIRECT_SYSTEM},
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
    if action == "search":
        action = "use_tool"
        obj["tool"] = obj.get("tool") or "graph_search"
    if action not in ("use_tool", "final"):
        return None

    reason = str(obj.get("reason", "")).strip()
    if not reason:
        reason = "基于当前信息进行下一步选择。"
    obj["reason"] = reason
    obj["action"] = action

    if action == "use_tool":
        tool = str(obj.get("tool", "")).strip() or "graph_search"
        kws = obj.get("keywords", [])
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list):
            return None
        cleaned = [str(k).strip() for k in kws if str(k).strip()]
        cleaned = cleaned[:6]
        if not cleaned:
            return None
        obj["tool"] = tool
        obj["keywords"] = cleaned

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
        ctx = re.sub(rf"(?m)^{re.escape(old)}", new, ctx)

    merged_items = existing_items + renumbered
    return merged_items, ctx



def _build_tools_text(tool_registry: Dict[str, "ToolDef"]) -> str:
    lines: List[str] = []
    for name, tool in tool_registry.items():
        lines.append(f"- {name}: {tool.description}")
    return "\n".join(lines) if lines else "- （暂无工具）"



def _summarize_items_for_user(evidence_items: List[Dict[str, Any]], max_names: int = 3) -> str:
    if not evidence_items:
        return "尚未拿到有效证据。"

    names: List[str] = []
    for item in evidence_items:
        node = item.get("node") or {}
        name = (node.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= max_names:
            break

    if names:
        preview = "、".join(names)
        return f"已拿到 {len(evidence_items)} 条证据，重点涉及：{preview}。"
    return f"已拿到 {len(evidence_items)} 条证据，结论开始收敛。"


# =============================================================================
# Search tool wrapper and registry
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


@dataclass
class ToolDef:
    name: str
    description: str
    run: Callable[[Dict[str, Any]], Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]



def build_tool_registry(search_tool: GraphSearchTool) -> Dict[str, ToolDef]:
    return {
        "graph_search": ToolDef(
            name="graph_search",
            description="在 Neo4j 图中按关键词直接检索证据，适合查实体、关系、术语和数据库内事实。",
            run=lambda decision: search_tool.search(decision.get("keywords", [])),
        )
    }


# =============================================================================
# Agent core logic helpers
# =============================================================================


async def _emit_workflow_step(
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    round_idx: int,
    title: str,
    detail: str,
) -> None:
    await emit({"type": "thinking_round", "round": round_idx})
    await emit({"type": "thinking", "round": round_idx, "text": f"{title}\n{detail}\n"})


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
    tool_registry = build_tool_registry(search_tool)
    tools_text = _build_tools_text(tool_registry)
    direct_answer_allowed = False

    for round_idx in range(1, max_search_rounds + 1):
        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(
            question=q,
            evidence_brief=evidence_brief,
            history_text=history_text,
        )
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM.format(tools_text=tools_text)},
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

        if not decision:
            direct_answer_allowed = not all_items
            break

        if decision["action"] == "final":
            direct_answer_allowed = not all_items
            break

        tool_name = str(decision.get("tool", "")).strip()
        tool_def = tool_registry.get(tool_name)
        if not tool_def:
            if print_debug:
                print(f"[agent] unknown tool={tool_name!r}; fallback to final")
            direct_answer_allowed = not all_items
            break

        if print_debug:
            print(f"\n[tool] {tool_name} keywords: {decision.get('keywords', [])}\n")

        items, ctx, meta = tool_def.run(decision)

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

    if context_text:
        messages = build_answer_messages(context=context_text, question=q, chat_history=chat_history)
        raw = llm_get_response(messages, True, "stream")
        answer = raw.split("</think>")[-1].strip()
    elif direct_answer_allowed:
        messages = build_direct_answer_messages(question=q, chat_history=chat_history)
        raw = llm_get_response(messages, True, "stream")
        answer = raw.split("</think>")[-1].strip()
    else:
        answer = "资料不足，无法确定（未获得可用的检索上下文）。"

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
    tool_registry = build_tool_registry(search_tool)
    tools_text = _build_tools_text(tool_registry)
    direct_answer_allowed = False
    thinking_step = 1

    await _emit_workflow_step(
        emit,
        thinking_step,
        "先判断这个问题是否依赖外部信息，以及是否需要调用工具。",
        "我会先做路线选择；如果需要证据，再决定使用哪个工具。",
    )
    thinking_step += 1

    for round_idx in range(1, max_search_rounds + 1):
        await emit({"type": "status", "phase": "agent", "text": f"判断是否需要调用工具（round={round_idx})...\n"})

        evidence_brief = brief_evidence(all_items)
        decide_prompt = DECIDE_TEMPLATE.format(
            question=q,
            evidence_brief=evidence_brief,
            history_text=history_text,
        )
        decide_messages = [
            {"role": "system", "content": DECIDE_SYSTEM.format(tools_text=tools_text)},
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
            direct_answer_allowed = not all_items
            await _emit_workflow_step(
                emit,
                thinking_step,
                "工具决策未能可靠解析，改为保守处理。",
                "当前先不继续尝试工具选择；若没有证据，就直接给出谨慎回答。",
            )
            thinking_step += 1
            await emit({"type": "status", "phase": "agent", "text": "解析决策失败，改为保守回答路径。\n"})
            break

        if decision["action"] == "final":
            direct_answer_allowed = not all_items
            await _emit_workflow_step(
                emit,
                thinking_step,
                "已完成工具选择判断。",
                f"当前不调用工具。原因：{decision.get('reason', '当前信息已足够直接回答。')}",
            )
            thinking_step += 1
            await emit({"type": "status", "phase": "agent", "text": "决定：当前不调用工具，直接作答。\n"})
            break

        tool_name = str(decision.get("tool", "")).strip()
        tool_def = tool_registry.get(tool_name)
        if not tool_def:
            direct_answer_allowed = not all_items
            await _emit_workflow_step(
                emit,
                thinking_step,
                "模型给出了一个当前不可用的工具。",
                f"工具名：{tool_name or '(空)'}。当前回退到直接回答路径。",
            )
            thinking_step += 1
            await emit({"type": "status", "phase": "agent", "text": f"未知工具：{tool_name}，回退为直接作答。\n"})
            break

        keywords = decision["keywords"]
        await _emit_workflow_step(
            emit,
            thinking_step,
            "已判定需要调用工具，正在核对证据。",
            f"当前优先选择工具：{tool_name}。原因：{decision.get('reason', '')}\n本轮关键词：{keywords}",
        )
        thinking_step += 1
        await emit({"type": "status", "phase": "tool", "text": f"启用工具 {tool_name} keywords={keywords}\n"})

        try:
            items, ctx, meta = tool_def.run(decision)
        except Exception as exc:
            await emit({"type": "error", "message": f"Tool failed ({tool_name}): {exc}"})
            await emit({"type": "done"})
            return ""

        await emit({"type": "status", "phase": "tool", "text": f"工具执行完成：{meta}\n"})

        if not items or not ctx.strip():
            await _emit_workflow_step(
                emit,
                thinking_step,
                "本轮工具没有拿到有效证据。",
                "我会继续调整关键词，或在下一轮重新判断是否该换一个工具。",
            )
            thinking_step += 1
            await emit({"type": "status", "phase": "tool", "text": "工具结果为空，准备重新思考下一步。\n"})
            continue

        merged_items, renumbered_ctx = _renumber_context_and_items(all_items, items, ctx)
        all_items = merged_items
        all_context_parts.append(renumbered_ctx)

        await _emit_workflow_step(
            emit,
            thinking_step,
            "已拿到主要证据，结论开始收敛。",
            _summarize_items_for_user(all_items),
        )
        thinking_step += 1

        await emit({"type": "evidence", "items": all_items})

    context_text = "\n\n".join([part for part in all_context_parts if part.strip()]).strip()

    if context_text:
        await _emit_workflow_step(
            emit,
            thinking_step,
            "当前判断：将基于已获得证据生成答案。",
            "如果证据仍有缺口或冲突，我会在最终回答里明确保留不确定性。",
        )
        thinking_step += 1
        await emit({"type": "status", "phase": "llm", "text": "开始基于证据生成答案...\n"})
        messages = build_answer_messages(context=context_text, question=q, chat_history=chat_history)
        try:
            answer_text = await llm_call(messages, False, True, True)
        except Exception as exc:
            await emit({"type": "error", "message": f"LLM failed: {exc}"})
            await emit({"type": "done"})
            return ""
        return answer_text.split("</think>")[-1].strip()

    if direct_answer_allowed:
        await _emit_workflow_step(
            emit,
            thinking_step,
            "当前判断：可以直接回答，不依赖工具证据。",
            "下面会直接给出回答；如果问题其实依赖外部事实，我会明确说明不确定性。",
        )
        thinking_step += 1
        await emit({"type": "status", "phase": "llm", "text": "开始直接生成答案...\n"})
        messages = build_direct_answer_messages(question=q, chat_history=chat_history)
        try:
            answer_text = await llm_call(messages, False, True, True)
        except Exception as exc:
            await emit({"type": "error", "message": f"LLM failed: {exc}"})
            await emit({"type": "done"})
            return ""
        return answer_text.split("</think>")[-1].strip()

    final_text = "资料不足，无法确定（未获得可用证据；如需继续，可再尝试其他工具或补充更具体线索）。"
    await _emit_workflow_step(
        emit,
        thinking_step,
        "当前判断：现有证据不足，暂时无法可靠收敛到结论。",
        "这次不继续强行回答，避免把未经验证的推测当成结论。",
    )
    await emit({"type": "answer", "text": final_text})
    await emit({"type": "done"})
    return final_text
