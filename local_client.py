# -*- coding: utf-8 -*-
"""
local_client.py

A standalone terminal client for the GraphRAG web backend.

Notes:
- It only talks to the backend over HTTP/WebSocket.
- It only depends on Python standard library plus project-local `constants`.
- It must not import any other project module.
- When no arguments are provided, it enters multi-turn interactive chat mode.
- By default it reads server address from constants.SERVER_BIND_HOST / SERVER_BIND_PORT.
- If the default bind host is 0.0.0.0, it will use localhost for client connection.

Usage examples:
  python local_client.py
  python local_client.py --ws-url ws://127.0.0.1:8000/ws
  python local_client.py --host 127.0.0.1 --port 8000
  python local_client.py --question "介绍一下某个实体"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    import constants
except Exception:
    constants = None  # type: ignore[assignment]


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def get_default_host_port() -> Tuple[str, int]:
    host = "127.0.0.1"
    port = 8000

    if constants is not None:
        host = str(getattr(constants, "SERVER_BIND_HOST", host) or host).strip()
        try:
            port = int(getattr(constants, "SERVER_BIND_PORT", port))
        except Exception:
            port = 8000

    if host == "0.0.0.0":
        host = "localhost"

    return host, port


class WebSocketError(Exception):
    pass


class SimpleWebSocketClient:
    """Minimal RFC6455 client for text frames, ping/pong and close."""

    def __init__(self, ws_url: str, timeout: float = 60.0) -> None:
        self.ws_url = ws_url
        self.timeout = float(timeout)
        self.sock: Optional[socket.socket] = None
        self._closed = False

    def connect(self) -> None:
        if self.sock is not None:
            return

        parsed = urllib.parse.urlparse(self.ws_url)
        scheme = parsed.scheme.lower()
        if scheme not in ("ws", "wss"):
            raise WebSocketError(f"Unsupported WebSocket scheme: {scheme}")

        host = parsed.hostname
        if not host:
            raise WebSocketError("WebSocket URL missing host")

        port = parsed.port
        if port is None:
            port = 443 if scheme == "wss" else 80

        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        raw_sock = socket.create_connection((host, port), timeout=self.timeout)
        raw_sock.settimeout(self.timeout)

        if scheme == "wss":
            context = ssl.create_default_context()
            sock: socket.socket = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            sock = raw_sock

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_hdr = host if parsed.port is None else f"{host}:{port}"
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_hdr}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: local_client/1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        sock.sendall(req)

        response = self._recv_http_response(sock)
        status_line, headers = self._parse_http_response(response)
        if not status_line.startswith("HTTP/1.1 101") and not status_line.startswith("HTTP/1.0 101"):
            raise WebSocketError(f"WebSocket handshake failed: {status_line}")

        accept = headers.get("sec-websocket-accept", "")
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
        if accept != expected:
            raise WebSocketError("WebSocket handshake failed: invalid Sec-WebSocket-Accept")

        self.sock = sock
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.sock is not None:
                try:
                    frame = self._build_frame(opcode=0x8, payload=b"")
                    self.sock.sendall(frame)
                except Exception:
                    pass
                try:
                    self.sock.close()
                finally:
                    self.sock = None
        finally:
            self.sock = None

    def send_text(self, text: str) -> None:
        if self.sock is None:
            raise WebSocketError("WebSocket is not connected")
        payload = text.encode("utf-8")
        frame = self._build_frame(opcode=0x1, payload=payload)
        self.sock.sendall(frame)

    def recv_text(self) -> Optional[str]:
        if self.sock is None:
            raise WebSocketError("WebSocket is not connected")

        fragments: List[bytes] = []
        current_opcode: Optional[int] = None

        while True:
            fin, opcode, payload = self._read_frame()

            if opcode == 0x8:
                self.close()
                return None
            if opcode == 0x9:
                self._send_control(opcode=0xA, payload=payload)
                continue
            if opcode == 0xA:
                continue
            if opcode not in (0x0, 0x1):
                continue

            if opcode == 0x1:
                current_opcode = 0x1
                fragments = [payload]
            elif opcode == 0x0:
                if current_opcode != 0x1:
                    continue
                fragments.append(payload)

            if fin:
                return b"".join(fragments).decode("utf-8", errors="replace")

    def _send_control(self, opcode: int, payload: bytes) -> None:
        if self.sock is None:
            return
        frame = self._build_frame(opcode=opcode, payload=payload)
        self.sock.sendall(frame)

    def _read_exact(self, n: int) -> bytes:
        assert self.sock is not None
        chunks: List[bytes] = []
        remaining = n
        while remaining > 0:
            data = self.sock.recv(remaining)
            if not data:
                raise WebSocketError("Connection closed while reading frame")
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _read_frame(self) -> Tuple[bool, int, bytes]:
        header = self._read_exact(2)
        b1, b2 = header[0], header[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F

        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]

        mask_key = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""

        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        return fin, opcode, payload

    def _build_frame(self, opcode: int, payload: bytes) -> bytes:
        fin_opcode = 0x80 | (opcode & 0x0F)
        mask_bit = 0x80
        n = len(payload)

        if n < 126:
            header = bytes([fin_opcode, mask_bit | n])
        elif n <= 0xFFFF:
            header = bytes([fin_opcode, mask_bit | 126]) + struct.pack("!H", n)
        else:
            header = bytes([fin_opcode, mask_bit | 127]) + struct.pack("!Q", n)

        mask_key = os.urandom(4)
        masked_payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return header + mask_key + masked_payload

    @staticmethod
    def _recv_http_response(sock: socket.socket) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 65536:
                raise WebSocketError("HTTP response header too large")
        return bytes(data)

    @staticmethod
    def _parse_http_response(data: bytes) -> Tuple[str, Dict[str, str]]:
        try:
            header_blob = data.split(b"\r\n\r\n", 1)[0].decode("utf-8", errors="replace")
        except Exception as e:
            raise WebSocketError(f"Invalid HTTP response: {e}")

        lines = header_blob.split("\r\n")
        if not lines:
            raise WebSocketError("Empty HTTP response")

        status_line = lines[0].strip()
        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        return status_line, headers


def derive_ws_url(host: str, port: int, use_ssl: bool) -> str:
    scheme = "wss" if use_ssl else "ws"
    return f"{scheme}://{host}:{port}/ws"


def derive_http_base(ws_url: str) -> str:
    parsed = urllib.parse.urlparse(ws_url)
    if parsed.scheme not in ("ws", "wss"):
        raise ValueError(f"Invalid WebSocket URL: {ws_url}")
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if http_scheme == "https" else 80
    return f"{http_scheme}://{host}:{port}"


def fetch_health(http_base: str, timeout: float) -> Optional[Dict[str, Any]]:
    url = http_base.rstrip("/") + "/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        if isinstance(data, dict):
            return data
        return {"raw": data}
    except urllib.error.HTTPError as e:
        return {"ok": False, "http_error": e.code, "reason": str(e)}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


class LocalClientApp:
    def __init__(self, ws_url: str, timeout: float = 60.0, show_health: bool = True) -> None:
        self.ws_url = ws_url
        self.timeout = float(timeout)
        self.show_health = show_health
        self.ws = SimpleWebSocketClient(ws_url=ws_url, timeout=timeout)
        self._answer_started = False
        self._last_evidence_count = 0
        self._printed_thinking_header = False

    def start(self) -> None:
        if self.show_health:
            http_base = derive_http_base(self.ws_url)
            health = fetch_health(http_base, self.timeout)
            if health is not None:
                print(f"[health] {json.dumps(health, ensure_ascii=False)}")
        self.ws.connect()
        print(f"Connected to {self.ws_url}")
        print("Type your question. Type 'exit' to quit.\n")

    def close(self) -> None:
        self.ws.close()

    def ask(self, question: str) -> int:
        q = (question or "").strip()
        if not q:
            return 0

        self._answer_started = False
        self._last_evidence_count = 0
        self._printed_thinking_header = False

        payload = {"type": "question", "text": q}
        self.ws.send_text(json.dumps(payload, ensure_ascii=False))

        while True:
            raw = self.ws.recv_text()
            if raw is None:
                print("\n[error] server disconnected")
                return 2

            try:
                msg = json.loads(raw)
            except Exception:
                print(f"\n[raw] {raw}")
                continue

            status = self._handle_message(msg)
            if status == "done":
                if self._answer_started:
                    print()
                print()
                return 0
            if status == "fatal":
                print()
                return 1

    def repl(self) -> int:
        try:
            while True:
                try:
                    q = input("Q> ").strip()
                except EOFError:
                    print()
                    break

                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q.lower() in (":health", "/health"):
                    http_base = derive_http_base(self.ws_url)
                    health = fetch_health(http_base, self.timeout)
                    print(json.dumps(health or {}, ensure_ascii=False, indent=2))
                    continue

                rc = self.ask(q)
                if rc != 0:
                    return rc
        except KeyboardInterrupt:
            print()
        return 0

    def _handle_message(self, msg: Dict[str, Any]) -> str:
        typ = str(msg.get("type", "")).strip().lower()

        if typ == "thinking":
            phase = str(msg.get("phase", "thinking"))
            text = str(msg.get("text", ""))
            self._print_thinking(phase, text)
            return "continue"

        if typ == "evidence":
            items = msg.get("items") or []
            if isinstance(items, list):
                self._print_evidence(items)
            return "continue"

        if typ == "answer":
            text = str(msg.get("text", ""))
            self._print_answer(text)
            return "continue"

        if typ == "error":
            message = str(msg.get("message", "unknown error"))
            print(f"\n[error] {message}")
            trace = msg.get("trace")
            if trace:
                print(str(trace).rstrip())
            return "continue"

        if typ == "done":
            return "done"

        print(f"\n[event] {json.dumps(msg, ensure_ascii=False)}")
        return "continue"

    def _print_thinking(self, phase: str, text: str) -> None:
        if not text:
            return
        if not self._printed_thinking_header:
            print("[Thinking]")
            self._printed_thinking_header = True
        prefix = f"[{phase}] "
        for line in text.splitlines(True):
            if line.endswith("\n"):
                print(prefix + line[:-1])
            else:
                print(prefix + line)

    def _print_evidence(self, items: List[Dict[str, Any]]) -> None:
        new_items = items[self._last_evidence_count:]
        if not new_items:
            return

        print("[References]")
        for item in new_items:
            cid = str(item.get("cid", "?"))
            score = item.get("score")
            node = item.get("node") or {}
            graph = item.get("graph") or {}
            name = str(node.get("name") or "").strip()
            labels = node.get("labels") or []
            labels_s = ", ".join(str(x) for x in labels) if labels else "-"
            desc = str(node.get("desc") or "").strip()
            if len(desc) > 200:
                desc = desc[:200] + "..."
            nodes = graph.get("nodes") or []
            edges = graph.get("edges") or []

            head = f"{cid}: {name or '(unnamed)'}"
            if score is not None:
                try:
                    head += f" | score={float(score):.4f}"
                except Exception:
                    head += f" | score={score}"
            print(head)
            print(f"  labels: {labels_s}")
            if desc:
                print(f"  desc: {desc}")
            print(f"  graph: nodes={len(nodes)}, edges={len(edges)}")

        self._last_evidence_count = len(items)
        print("-" * 80)

    def _print_answer(self, text: str) -> None:
        if not self._answer_started:
            print("A>")
            print("=" * 80)
            self._answer_started = True
        print(text, end="", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    default_host, default_port = get_default_host_port()

    parser = argparse.ArgumentParser(description="Standalone local client for GraphRAG server backend")
    parser.add_argument("--ws-url", default="", help="WebSocket endpoint, e.g. ws://127.0.0.1:8000/ws")
    parser.add_argument("--host", default=default_host, help="Server host when --ws-url is omitted")
    parser.add_argument("--port", type=int, default=default_port, help="Server port when --ws-url is omitted")
    parser.add_argument("--ssl", action="store_true", help="Use wss:// when --ws-url is omitted")
    parser.add_argument("--timeout", type=float, default=60.0, help="Socket/HTTP timeout in seconds")
    parser.add_argument("--no-health", action="store_true", help="Do not call /api/health on startup")
    parser.add_argument("--question", default="", help="Ask one question and exit")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ws_url = args.ws_url.strip() or derive_ws_url(str(args.host), int(args.port), bool(args.ssl))

    app = LocalClientApp(ws_url=ws_url, timeout=float(args.timeout), show_health=not bool(args.no_health))

    try:
        app.start()
        if args.question:
            return app.ask(args.question)
        return app.repl()
    except (ConnectionError, OSError, WebSocketError, socket.timeout) as e:
        _stderr(f"[fatal] {e}")
        return 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
