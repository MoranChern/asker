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
import inspect
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional


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


def _configure_cuda_visible_devices(constants_mod: Any) -> List[str]:
    raw = getattr(constants_mod, "CUDA_VISIBLE_DEVICES", None)
    if raw is None:
        return []
    value = str(raw).strip()
    if not value:
        return []
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return [part.strip() for part in value.split(",") if part.strip()]


def _auto_tensor_split(device_count: int) -> Optional[List[float]]:
    if device_count <= 1:
        return None
    main_gpu_share = max(0.7, 1.0 - 0.1 * (device_count - 1))
    return [main_gpu_share] + [1.0] * (device_count - 1)


def _parse_tensor_split(raw: Any, device_count: int) -> Optional[List[float]]:
    if device_count <= 1:
        return None

    if raw is None:
        return _auto_tensor_split(device_count)

    values: List[float]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return _auto_tensor_split(device_count)
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    else:
        raise ValueError("CUDA_TENSOR_SPLIT must be a comma-separated string or a list/tuple of numbers")

    if len(values) != device_count:
        raise ValueError(
            f"CUDA_TENSOR_SPLIT length mismatch: expected {device_count}, got {len(values)}"
        )
    if any(v <= 0 for v in values):
        raise ValueError("CUDA_TENSOR_SPLIT values must be > 0")
    return values


def _resolve_split_mode(llama_cpp_mod: Any, raw: Any) -> Optional[int]:
    mode = str(raw or "layer").strip().lower()
    mapping = {
        "none": getattr(llama_cpp_mod, "LLAMA_SPLIT_MODE_NONE", None),
        "layer": getattr(llama_cpp_mod, "LLAMA_SPLIT_MODE_LAYER", None),
        "row": getattr(llama_cpp_mod, "LLAMA_SPLIT_MODE_ROW", None),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported CUDA_SPLIT_MODE: {raw!r}")
    return mapping[mode]


def _build_llama_kwargs(constants_mod: Any, llama_cls: Any, llama_cpp_mod: Any, visible_devices: List[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model_path": constants_mod.GENERAL_MODEL_PATH,
        "n_ctx": constants_mod.N_CTX,
        "n_gpu_layers": constants_mod.N_GPU_LAYERS,
        "verbose": False,
    }

    sig = inspect.signature(llama_cls.__init__)
    supported = sig.parameters
    device_count = len(visible_devices)

    if "offload_kqv" in supported:
        kwargs["offload_kqv"] = bool(getattr(constants_mod, "QWEN_OFFLOAD_KQV", True))

    if "flash_attn" in supported:
        kwargs["flash_attn"] = bool(
            getattr(constants_mod, "QWEN_FLASH_ATTN", device_count > 1 or int(constants_mod.N_CTX) >= 16384)
        )

    if "n_batch" in supported:
        kwargs["n_batch"] = int(getattr(constants_mod, "QWEN_N_BATCH", 512))

    if "n_ubatch" in supported:
        default_ubatch = 256 if device_count > 1 else kwargs.get("n_batch", 512)
        kwargs["n_ubatch"] = min(int(kwargs.get("n_batch", 512)), int(getattr(constants_mod, "QWEN_N_UBATCH", default_ubatch)))

    if device_count > 1:
        split_mode = _resolve_split_mode(
            llama_cpp_mod,
            getattr(constants_mod, "CUDA_SPLIT_MODE", "layer"),
        )
        tensor_split = _parse_tensor_split(
            getattr(constants_mod, "CUDA_TENSOR_SPLIT", None),
            device_count,
        )
        main_gpu = int(getattr(constants_mod, "CUDA_MAIN_GPU", 0))

        if "split_mode" in supported and split_mode is not None:
            kwargs["split_mode"] = split_mode
        if "tensor_split" in supported and tensor_split is not None:
            kwargs["tensor_split"] = tensor_split
        if "main_gpu" in supported:
            kwargs["main_gpu"] = main_gpu

    return kwargs


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
            visible_devices = _configure_cuda_visible_devices(C)
            import llama_cpp
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

        llama_kwargs = _build_llama_kwargs(C, Llama, llama_cpp, visible_devices)
        emit_status(
            "worker_progress",
            (
                "gpu_worker: loading model... "
                f"(think={think}, visible_gpus={visible_devices or ['default']}, "
                f"llama_kwargs={json.dumps(llama_kwargs, ensure_ascii=False, default=str)})\n"
            ),
        )

        llm = Llama(**llama_kwargs)

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
