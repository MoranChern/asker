# -*- coding: utf-8 -*-
"""Qwen (llama-cpp) client for this project.

- Uses constants.py for configuration.
- get_response signature: get_response(self, messages: List[Dict], think: bool, print_type: str) -> str
- Does NOT print any separators. Separators should be handled by the main application.
"""

import inspect
import os
from typing import Any, Dict, List, Optional

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
        visible_devices = self._configure_cuda_visible_devices(cuda_visible_devices)

        from llama_cpp import Llama
        import llama_cpp

        llama_kwargs = self._build_llama_kwargs(
            llama_cls=Llama,
            llama_cpp_mod=llama_cpp,
            visible_devices=visible_devices,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

        self.llm = Llama(**llama_kwargs)

    @staticmethod
    def _configure_cuda_visible_devices(cuda_visible_devices: Optional[str]) -> List[str]:
        if cuda_visible_devices is None:
            return []
        value = str(cuda_visible_devices).strip()
        if not value:
            return []
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        return [part.strip() for part in value.split(",") if part.strip()]

    @staticmethod
    def _auto_tensor_split(device_count: int) -> Optional[List[float]]:
        if device_count <= 1:
            return None
        main_gpu_share = max(0.7, 1.0 - 0.1 * (device_count - 1))
        return [main_gpu_share] + [1.0] * (device_count - 1)

    @classmethod
    def _parse_tensor_split(cls, raw: Any, device_count: int) -> Optional[List[float]]:
        if device_count <= 1:
            return None

        if raw is None:
            return cls._auto_tensor_split(device_count)

        values: List[float]
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return cls._auto_tensor_split(device_count)
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

    @staticmethod
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

    @classmethod
    def _build_llama_kwargs(
        cls,
        *,
        llama_cls: Any,
        llama_cpp_mod: Any,
        visible_devices: List[str],
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }

        sig = inspect.signature(llama_cls.__init__)
        supported = sig.parameters
        device_count = len(visible_devices)

        if "offload_kqv" in supported:
            kwargs["offload_kqv"] = bool(getattr(C, "QWEN_OFFLOAD_KQV", True))

        if "flash_attn" in supported:
            kwargs["flash_attn"] = bool(getattr(C, "QWEN_FLASH_ATTN", device_count > 1 or int(n_ctx) >= 16384))

        if "n_batch" in supported:
            kwargs["n_batch"] = int(getattr(C, "QWEN_N_BATCH", 512))

        if "n_ubatch" in supported:
            default_ubatch = 256 if device_count > 1 else kwargs.get("n_batch", 512)
            kwargs["n_ubatch"] = min(int(kwargs.get("n_batch", 512)), int(getattr(C, "QWEN_N_UBATCH", default_ubatch)))

        if device_count > 1:
            split_mode = cls._resolve_split_mode(
                llama_cpp_mod,
                getattr(C, "CUDA_SPLIT_MODE", "layer"),
            )
            tensor_split = cls._parse_tensor_split(
                getattr(C, "CUDA_TENSOR_SPLIT", None),
                device_count,
            )
            main_gpu = int(getattr(C, "CUDA_MAIN_GPU", 0))

            if "split_mode" in supported and split_mode is not None:
                kwargs["split_mode"] = split_mode
            if "tensor_split" in supported and tensor_split is not None:
                kwargs["tensor_split"] = tensor_split
            if "main_gpu" in supported:
                kwargs["main_gpu"] = main_gpu

        return kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            del self.llm
        except Exception:
            pass

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        if not text:
            return ""

        open_tag = "<think>"
        close_tag = "</think>"
        out: List[str] = []
        i = 0
        in_think = False

        while i < len(text):
            if not in_think:
                j = text.find(open_tag, i)
                if j == -1:
                    out.append(text[i:])
                    break
                out.append(text[i:j])
                i = j + len(open_tag)
                in_think = True
            else:
                j = text.find(close_tag, i)
                if j == -1:
                    break
                i = j + len(close_tag)
                in_think = False

        return "".join(out).strip()

    def get_response(self, messages: List[Dict], think: bool, print_type: str) -> str:
        """Return model output.

        Behavior:
        - think=False: appends " /no_think" to the last message and suppresses thought blocks in returned text.
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

        open_tag = "<think>"
        close_tag = "</think>"
        max_tag_len = max(len(open_tag), len(close_tag))

        res = ""

        if print_type == "stream" and think:
            state = "outside"
            buf = ""
            round_idx = 0

            def _trim_outside_buf():
                nonlocal buf
                safe_len = max(0, len(buf) - (max_tag_len - 1))
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
                        pos = buf.find(open_tag)
                        if pos != -1:
                            buf = buf[pos + len(open_tag):]
                            round_idx += 1
                            print(f"\n\n【思考 {round_idx}】\n", end="", flush=True)
                            state = "think"
                            continue
                        _trim_outside_buf()
                        break

                    pos = buf.find(close_tag)
                    if pos != -1:
                        if pos > 0:
                            print(buf[:pos], end="", flush=True)
                        buf = buf[pos + len(close_tag):]
                        print("\n", end="", flush=True)
                        state = "outside"
                        continue

                    safe_len = max(0, len(buf) - (max_tag_len - 1))
                    if safe_len > 0:
                        print(buf[:safe_len], end="", flush=True)
                        buf = buf[safe_len:]
                    break

            print()

        else:
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
            res = self._strip_think_blocks(res)

        if print_type == "result":
            print(res)

        return res
