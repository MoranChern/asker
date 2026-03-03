# -*- coding: utf-8 -*-
"""Qwen (llama-cpp) client for this project.

- Uses constants.py for configuration.
- get_response signature: get_response(self, messages: List[Dict], think: bool, print_type: str) -> str
- Does NOT print any separators. Separators should be handled by the main application.
"""

import os
from typing import List, Dict, Optional

from llama_cpp import Llama
import constants as C


class LLM_QWEN_Standalone:
    def __init__(
        self,
        model_path: str = C.GENERAL_MODEL_PATH,
        cuda_visible_devices: Optional[str] = C.CUDA_VISIBLE_DEVICES,
        n_ctx: int = C.N_CTX,
        n_gpu_layers: int = C.N_GPU_LAYERS,
        verbose: bool = False,
    ):
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

        self.llm = Llama(
            model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            del self.llm
        except Exception:
            pass

    def get_response(self, messages: List[Dict], think: bool, print_type: str) -> str:
        """Return model output.

        Behavior:
        - think=False: appends " /no_think" to the last message and returns final answer (strip before </think>)
        - think=True:
            * print_type="stream": prints ONLY thought blocks (<think>...</think>) round-by-round,
              and returns the full raw text (thoughts + answer). The caller (main) should print separators
              and the final answer.
            * print_type in {"result","none"}: does not do special splitting; returns raw text.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list of dicts")

        msgs = [dict(m) for m in messages]
        if "content" not in msgs[-1]:
            msgs[-1]["content"] = ""

        if not think:
            msgs[-1]["content"] += " /no_think"

        stream = self.llm.create_chat_completion(
            messages=msgs,
            stream=True,
            max_tokens=C.QWEN_MAX_TOKENS,
            temperature=C.QWEN_TEMPERATURE,
            top_k=C.QWEN_TOP_K,
            top_p=C.QWEN_TOP_P,
            min_p=C.QWEN_MIN_P,
            presence_penalty=C.QWEN_PRESENCE_PENALTY,
        )

        OPEN_TAG = "<think>"
        CLOSE_TAG = "</think>"
        MAX_TAG_LEN = max(len(OPEN_TAG), len(CLOSE_TAG))

        res = ""

        if print_type == "stream" and think:
            # Print ONLY thoughts, in rounds
            state = "outside"  # or "think"
            buf = ""
            round_idx = 0

            def _trim_outside_buf():
                nonlocal buf
                safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
                if safe_len > 0:
                    buf = buf[safe_len:]

            for chunk in stream:
                if "choices" not in chunk or not chunk["choices"]:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                if "content" not in delta:
                    continue
                text = delta["content"]
                res += text
                buf += text

                while True:
                    if state == "outside":
                        pos = buf.find(OPEN_TAG)
                        if pos != -1:
                            buf = buf[pos + len(OPEN_TAG):]  # discard outside prefix
                            round_idx += 1
                            print(f"\n\n【思考 {round_idx}】\n", end="", flush=True)
                            state = "think"
                            continue
                        _trim_outside_buf()
                        break
                    else:
                        pos = buf.find(CLOSE_TAG)
                        if pos != -1:
                            if pos > 0:
                                print(buf[:pos], end="", flush=True)
                            buf = buf[pos + len(CLOSE_TAG):]
                            print("\n", end="", flush=True)
                            state = "outside"
                            continue

                        safe_len = max(0, len(buf) - (MAX_TAG_LEN - 1))
                        if safe_len > 0:
                            print(buf[:safe_len], end="", flush=True)
                            buf = buf[safe_len:]
                        break

            print()  # newline after streaming thoughts

        else:
            # Raw streaming/collecting
            for chunk in stream:
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        text = delta["content"]
                        if print_type == "stream":
                            print(text, end="", flush=True)
                        res += text
            if print_type == "stream":
                print()

        if not think:
            res = res.split(CLOSE_TAG)[-1].strip()

        if print_type == "result":
            print(res)

        return res
