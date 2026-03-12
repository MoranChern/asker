# -*- coding: utf-8 -*-
"""
gpu_worker.py

A short-lived GPU process for Qwen (llama.cpp) inference.

- Reads ONE JSON request from stdin.
- Streams JSON lines to stdout (JSONL), so the web server can forward them to the browser in real time.
- Exits immediately after finishing, so GPU memory is released.

Request JSON schema (minimal):
{
  "messages": [{"role":"system|user|assistant","content":"..."}],
  "think": true,
  "max_tokens": optional int,
  "temperature": optional float,
  ...
}

Output JSONL events:
- {"type":"status","phase":"worker_progress","text":"..."}
- {"type":"thinking_round","round":1}
- {"type":"thinking","round":1,"text":"..."}
- {"type":"answer","text":"..."}
- {"type":"error","message":"...","trace":"..."}
- {"type":"done"}
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from typing import Any, Dict, List


OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
MAX_TAG_LEN = max(len(OPEN_TAG), len(CLOSE_TAG))


def emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def emit_status(phase: str, text: str) -> None:
    emit({"type": "status", "phase": phase, "text": text})


def _get_req() -> Dict[str, Any]:
    raw = sys.stdin.read()
    raw = (raw or "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _safe_messages(x: Any) -> List[Dict[str, str]]:
    if not isinstance(x, list):
        raise ValueError("messages must be a list")
    out: List[Dict[str, str]] = []
    for i, m in enumerate(x):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be a dict")
        role = m.get("role")
        content = m.get("content", "")
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": str(content)})
    if not out:
        raise ValueError("messages cannot be empty")
    return out


def _stream_with_thinking(stream: Any) -> None:
    state = "outside"
    buf = ""
    round_idx = 0

    for chunk in stream:
        if "choices" not in chunk or not chunk["choices"]:
            continue
        delta = chunk["choices"][0].get("delta", {})
        if "content" not in delta:
            continue

        text = delta["content"]
        if not text:
            continue

        buf += text

        while True:
            if state == "outside":
                pos = buf.find(OPEN_TAG)
                if pos != -1:
                    if pos > 0:
                        emit({"type": "answer", "text": buf[:pos]})
                    buf = buf[pos + len(OPEN_TAG) :]
                    round_idx += 1
                    emit({"type": "thinking_round", "round": round_idx})
                    state = "think"
                    continue

                safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
                if safe_len > 0:
                    emit({"type": "answer", "text": buf[:safe_len]})
                    buf = buf[safe_len:]
                break

            pos = buf.find(CLOSE_TAG)
            if pos != -1:
                if pos > 0:
                    emit({"type": "thinking", "round": round_idx, "text": buf[:pos]})
                buf = buf[pos + len(CLOSE_TAG) :]
                state = "outside"
                continue

            safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
            if safe_len > 0:
                emit({"type": "thinking", "round": round_idx, "text": buf[:safe_len]})
                buf = buf[safe_len:]
            break

    if buf:
        if state == "think":
            emit({"type": "thinking", "round": round_idx, "text": buf})
        else:
            emit({"type": "answer", "text": buf})


def _stream_without_thinking(stream: Any) -> None:
    """Emit answer only, even if the model still leaks <think> blocks."""
    state = "outside"
    buf = ""

    for chunk in stream:
        if "choices" not in chunk or not chunk["choices"]:
            continue
        delta = chunk["choices"][0].get("delta", {})
        if "content" not in delta:
            continue

        text = delta["content"]
        if not text:
            continue

        buf += text

        while True:
            if state == "outside":
                pos = buf.find(OPEN_TAG)
                if pos != -1:
                    if pos > 0:
                        emit({"type": "answer", "text": buf[:pos]})
                    buf = buf[pos + len(OPEN_TAG) :]
                    state = "think"
                    continue

                safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
                if safe_len > 0:
                    emit({"type": "answer", "text": buf[:safe_len]})
                    buf = buf[safe_len:]
                break

            pos = buf.find(CLOSE_TAG)
            if pos != -1:
                buf = buf[pos + len(CLOSE_TAG) :]
                state = "outside"
                continue

            safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
            if safe_len > 0:
                buf = buf[safe_len:]
            break

    if buf and state == "outside":
        emit({"type": "answer", "text": buf})


def main() -> None:
    try:
        emit_status("worker_progress", "gpu_worker: init\n")

        try:
            import constants as C
            from llama_cpp import Llama
        except Exception as e:
            emit({"type": "error", "message": f"gpu_worker import failed: {e}", "trace": traceback.format_exc(limit=50)})
            emit({"type": "done"})
            return

        req = _get_req()
        messages = _safe_messages(req.get("messages", []))
        think = bool(req.get("think", True))

        max_tokens = int(req.get("max_tokens", C.QWEN_MAX_TOKENS))
        temperature = float(req.get("temperature", C.QWEN_TEMPERATURE))
        top_k = int(req.get("top_k", C.QWEN_TOP_K))
        top_p = float(req.get("top_p", C.QWEN_TOP_P))
        min_p = float(req.get("min_p", C.QWEN_MIN_P))
        presence_penalty = float(req.get("presence_penalty", C.QWEN_PRESENCE_PENALTY))

        if getattr(C, "CUDA_VISIBLE_DEVICES", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(C.CUDA_VISIBLE_DEVICES)

        emit_status("worker_progress", f"gpu_worker: loading model... (think={think})\n")

        llm = Llama(
            C.GENERAL_MODEL_PATH,
            n_ctx=C.N_CTX,
            n_gpu_layers=C.N_GPU_LAYERS,
            verbose=False,
        )

        emit_status("worker_progress", "gpu_worker: model loaded, generating...\n")

        if not think:
            messages[-1]["content"] = (messages[-1].get("content", "") or "") + " /no_think"

        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        if think:
            _stream_with_thinking(stream)
        else:
            _stream_without_thinking(stream)

        emit({"type": "done"})

        try:
            del llm
        except Exception:
            pass
        gc.collect()

    except Exception as e:
        emit({"type": "error", "message": str(e), "trace": traceback.format_exc(limit=50)})
        emit({"type": "done"})


if __name__ == "__main__":
    main()
