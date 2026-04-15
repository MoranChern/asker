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
- Supports multi-conversation management via the server's SQLite-backed APIs.

Usage examples:
  python local_client.py
  python local_client.py --ws-url ws://127.0.0.1:8000/ws
  python local_client.py --host 127.0.0.1 --port 8000
  python local_client.py --question "介绍一下某个实体"
  python local_client.py --new-conversation --conversation-title "测试会话"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
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

        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            port = 443 if scheme == "wss" else 80

        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        raw_sock = socket.create_connection((host, port), timeout=self.timeout)
        raw_sock.settimeout(self.timeout)

        if scheme == "wss":
            ctx = ssl.create_default_context()
            server_hostname = host
            sock = ctx.wrap_socket(raw_sock, server_hostname=server_hostname)
        else:
            sock = raw_sock

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"User-Agent: local_client.py\r\n"
            f"\r\n"
        )
        sock.sendall(req.encode("utf-8"))

        resp = self._recv_http_response(sock)
        status_line, headers = self._parse_http_response(resp)

        if not status_line.startswith("HTTP/1.1 101") and not status_line.startswith("HTTP/1.0 101"):
            sock.close()
            raise WebSocketError(f"WebSocket handshake failed: {status_line}")

        accept = headers.get("sec-websocket-accept", "")
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
        if accept != expected:
            sock.close()
            raise WebSocketError("Invalid Sec-WebSocket-Accept in handshake response")

        self.sock = sock
        self._closed = False

    def close(self) -> None:
        if self.sock is None or self._closed:
            return
        try:
            frame = self._build_frame(0x8, b"")
            self.sock.sendall(frame)
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None
            self._closed = True

    def send_text(self, text: str) -> None:
        if self.sock is None:
            raise WebSocketError("WebSocket not connected")
        frame = self._build_frame(0x1, text.encode("utf-8"))
        self.sock.sendall(frame)

    def recv_text(self) -> Optional[str]:
        if self.sock is None:
            raise WebSocketError("WebSocket not connected")

        parts: List[bytes] = []
        while True:
            fin, opcode, payload = self._read_frame()

            if opcode == 0x8:  # close
                self.close()
                return None
            if opcode == 0x9:  # ping
                pong = self._build_frame(0xA, payload)
                self.sock.sendall(pong)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode not in (0x0, 0x1):
                continue

            parts.append(payload)
            if fin:
                break

        try:
            return b"".join(parts).decode("utf-8")
        except UnicodeDecodeError:
            return b"".join(parts).decode("utf-8", errors="replace")

    def _read_exact(self, n: int) -> bytes:
        if self.sock is None:
            raise WebSocketError("WebSocket not connected")
        chunks = []
        got = 0
        while got < n:
            chunk = self.sock.recv(n - got)
            if not chunk:
                raise WebSocketError("Connection closed while reading frame")
            chunks.append(chunk)
            got += len(chunk)
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


def http_json_request(
    method: str,
    url: str,
    timeout: float,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    data = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from e
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return body


def api_list_conversations(http_base: str, timeout: float) -> List[Dict[str, Any]]:
    data = http_json_request("GET", http_base.rstrip("/") + "/api/conversations", timeout)
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def api_create_conversation(http_base: str, timeout: float, title: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if title.strip():
        payload["title"] = title.strip()
    data = http_json_request("POST", http_base.rstrip("/") + "/api/conversations", timeout, payload=payload)
    return data if isinstance(data, dict) else {}


def api_get_messages(http_base: str, timeout: float, conversation_id: str) -> List[Dict[str, Any]]:
    data = http_json_request(
        "GET",
        http_base.rstrip("/") + f"/api/conversations/{urllib.parse.quote(conversation_id)}/messages",
        timeout,
    )
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def api_rename_conversation(http_base: str, timeout: float, conversation_id: str, title: str) -> Dict[str, Any]:
    data = http_json_request(
        "PATCH",
        http_base.rstrip("/") + f"/api/conversations/{urllib.parse.quote(conversation_id)}",
        timeout,
        payload={"title": title},
    )
    return data if isinstance(data, dict) else {}


def api_delete_conversation(http_base: str, timeout: float, conversation_id: str) -> Dict[str, Any]:
    data = http_json_request(
        "DELETE",
        http_base.rstrip("/") + f"/api/conversations/{urllib.parse.quote(conversation_id)}",
        timeout,
    )
    return data if isinstance(data, dict) else {}


class LocalClientApp:
    def __init__(
        self,
        ws_url: str,
        timeout: float = 60.0,
        show_health: bool = True,
        preferred_conversation_id: str = "",
    ) -> None:
        self.ws_url = ws_url
        self.http_base = derive_http_base(ws_url)
        self.timeout = float(timeout)
        self.show_health = show_health
        self.preferred_conversation_id = preferred_conversation_id.strip()
        self.ws = SimpleWebSocketClient(ws_url=ws_url, timeout=timeout)
        self._answer_started = False
        self._last_evidence_count = 0
        self._printed_thinking_header = False
        self.current_conversation_id = ""
        self.current_conversation_title = ""
        self.conversations: List[Dict[str, Any]] = []

    def start(self) -> None:
        if self.show_health:
            health = fetch_health(self.http_base, self.timeout)
            if health is not None:
                print(f"[health] {json.dumps(health, ensure_ascii=False)}")
        self.ws.connect()
        print(f"Connected to {self.ws_url}")
        self.refresh_conversations()
        self.ensure_active_conversation(self.preferred_conversation_id)
        print("Type your question. Type 'exit' to quit.")
        print("Use :help to view conversation commands.\n")
        self.print_current_conversation()

    def close(self) -> None:
        self.ws.close()

    def refresh_conversations(self) -> List[Dict[str, Any]]:
        self.conversations = api_list_conversations(self.http_base, self.timeout)
        return self.conversations

    def ensure_active_conversation(self, preferred_id: str = "") -> None:
        self.refresh_conversations()
        selected: Optional[Dict[str, Any]] = None
        if preferred_id:
            for item in self.conversations:
                if str(item.get("id", "")) == preferred_id:
                    selected = item
                    break
            if selected is None:
                raise RuntimeError(f"conversation not found: {preferred_id}")

        if selected is None and self.current_conversation_id:
            for item in self.conversations:
                if str(item.get("id", "")) == self.current_conversation_id:
                    selected = item
                    break

        if selected is None and self.conversations:
            selected = self.conversations[0]

        if selected is None:
            selected = self.create_conversation("")

        self.current_conversation_id = str(selected.get("id", ""))
        self.current_conversation_title = str(selected.get("title", ""))

    def create_conversation(self, title: str = "") -> Dict[str, Any]:
        item = api_create_conversation(self.http_base, self.timeout, title)
        self.refresh_conversations()
        self.current_conversation_id = str(item.get("id", ""))
        self.current_conversation_title = str(item.get("title", ""))
        print(f"[conversation] created {self.current_conversation_title} ({self.current_conversation_id})")
        return item

    def select_conversation(self, identifier: str) -> Dict[str, Any]:
        item = self._resolve_conversation(identifier)
        self.current_conversation_id = str(item.get("id", ""))
        self.current_conversation_title = str(item.get("title", ""))
        self.print_current_conversation()
        return item

    def rename_current_or_target(self, identifier: str, title: Optional[str] = None) -> Dict[str, Any]:
        target = self._resolve_conversation(identifier) if title is not None else self._require_current_conversation()
        new_title = title if title is not None else identifier
        if not str(new_title or "").strip():
            raise RuntimeError("new title must not be empty")
        item = api_rename_conversation(
            self.http_base,
            self.timeout,
            str(target.get("id", "")),
            str(new_title).strip(),
        )
        self.refresh_conversations()
        if str(item.get("id", "")) == self.current_conversation_id:
            self.current_conversation_title = str(item.get("title", ""))
        print(f"[conversation] renamed to {item.get('title', '')} ({item.get('id', '')})")
        return item

    def delete_current_or_target(self, identifier: str = "") -> None:
        target = self._resolve_conversation(identifier) if identifier else self._require_current_conversation()
        target_id = str(target.get("id", ""))
        target_title = str(target.get("title", ""))
        api_delete_conversation(self.http_base, self.timeout, target_id)
        print(f"[conversation] deleted {target_title} ({target_id})")
        self.refresh_conversations()
        if target_id == self.current_conversation_id:
            self.current_conversation_id = ""
            self.current_conversation_title = ""
        self.ensure_active_conversation("")
        self.print_current_conversation()

    def show_conversations(self) -> None:
        self.refresh_conversations()
        if not self.conversations:
            print("[conversation] no conversations")
            return
        print("[conversation] list")
        for idx, item in enumerate(self.conversations, start=1):
            conv_id = str(item.get("id", ""))
            marker = "*" if conv_id == self.current_conversation_id else " "
            title = str(item.get("title", ""))
            count = item.get("message_count", 0)
            updated = str(item.get("updated_at", ""))
            print(f" {marker} {idx:>2}. {title} | id={conv_id} | messages={count} | updated={updated}")

    def show_current_messages(self, limit: int = 20) -> None:
        conv = self._require_current_conversation()
        items = api_get_messages(self.http_base, self.timeout, str(conv.get("id", "")))
        if limit > 0:
            items = items[-limit:]
        print(f"[conversation] messages in {conv.get('title', '')} ({conv.get('id', '')})")
        if not items:
            print("  (empty)")
            return
        for item in items:
            role = str(item.get("role", ""))
            kind = str(item.get("kind", "message"))
            content = str(item.get("content", "")).strip()
            if len(content) > 300:
                content = content[:300] + "..."
            if kind == "message":
                print(f"- {role}: {content}")
            else:
                print(f"- {kind}/{role}: {content}")

    def print_current_conversation(self) -> None:
        conv = self._require_current_conversation()
        print(f"[conversation] current: {conv.get('title', '')} ({conv.get('id', '')})")

    def ask(self, question: str) -> int:
        q = (question or "").strip()
        if not q:
            return 0

        conv = self._require_current_conversation()

        self._answer_started = False
        self._last_evidence_count = 0
        self._printed_thinking_header = False

        payload = {
            "type": "question",
            "text": q,
            "conversation_id": str(conv.get("id", "")),
        }
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
                self.refresh_conversations()
                self.ensure_active_conversation(self.current_conversation_id)
                return 0
            if status == "fatal":
                print()
                return 1

    def repl(self) -> int:
        try:
            while True:
                try:
                    prompt_title = self.current_conversation_title or "未命名会话"
                    q = input(f"[{prompt_title}] Q> ").strip()
                except EOFError:
                    print()
                    break

                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q.startswith(":") or q.startswith("/"):
                    rc = self.handle_command(q)
                    if rc != 0:
                        return rc
                    continue

                rc = self.ask(q)
                if rc != 0:
                    return rc
        except KeyboardInterrupt:
            print()
        return 0

    def handle_command(self, raw: str) -> int:
        line = raw[1:].strip()
        if not line:
            return 0
        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"[command error] {e}")
            return 0
        if not parts:
            return 0

        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd == "help":
                self.print_help()
                return 0
            if cmd == "health":
                health = fetch_health(self.http_base, self.timeout)
                print(json.dumps(health or {}, ensure_ascii=False, indent=2))
                return 0
            if cmd in ("list", "ls"):
                self.show_conversations()
                return 0
            if cmd == "current":
                self.print_current_conversation()
                return 0
            if cmd == "new":
                title = " ".join(args).strip()
                self.create_conversation(title)
                return 0
            if cmd == "use":
                if not args:
                    print("usage: :use <conversation_id|index>")
                    return 0
                self.select_conversation(args[0])
                return 0
            if cmd == "rename":
                if not args:
                    print("usage: :rename <new title>  OR  :rename <conversation_id|index> <new title>")
                    return 0
                if len(args) == 1:
                    self.rename_current_or_target(args[0], None)
                else:
                    self.rename_current_or_target(args[0], " ".join(args[1:]).strip())
                return 0
            if cmd == "delete":
                identifier = args[0] if args else ""
                target = self._resolve_conversation(identifier) if identifier else self._require_current_conversation()
                confirm = input(f"Delete conversation '{target.get('title', '')}'? [y/N] ").strip().lower()
                if confirm in ("y", "yes"):
                    self.delete_current_or_target(identifier)
                else:
                    print("[conversation] delete cancelled")
                return 0
            if cmd == "show":
                limit = 20
                if args:
                    try:
                        limit = int(args[0])
                    except Exception:
                        print("usage: :show [limit]")
                        return 0
                self.show_current_messages(limit)
                return 0
        except Exception as e:
            print(f"[command error] {e}")
            return 0

        print(f"Unknown command: {cmd}. Use :help")
        return 0

    def print_help(self) -> None:
        print("Commands:")
        print("  :help                          show this help")
        print("  :list                          list all conversations")
        print("  :new [title]                   create and select a conversation")
        print("  :use <conversation_id|index>   select a conversation")
        print("  :rename <new title>            rename current conversation")
        print("  :rename <id|index> <title>     rename a specific conversation")
        print("  :delete [conversation_id|index] delete a conversation")
        print("  :current                       show current conversation")
        print("  :show [limit]                  show recent stored messages")
        print("  :health                        show backend health")
        print("  exit / quit                    exit the client")

    def _require_current_conversation(self) -> Dict[str, Any]:
        if not self.current_conversation_id:
            self.ensure_active_conversation("")
        for item in self.conversations:
            if str(item.get("id", "")) == self.current_conversation_id:
                return item
        self.ensure_active_conversation(self.current_conversation_id)
        for item in self.conversations:
            if str(item.get("id", "")) == self.current_conversation_id:
                return item
        raise RuntimeError("no active conversation")

    def _resolve_conversation(self, identifier: str) -> Dict[str, Any]:
        self.refresh_conversations()
        ident = str(identifier or "").strip()
        if not ident:
            return self._require_current_conversation()

        for item in self.conversations:
            if str(item.get("id", "")) == ident:
                return item

        if ident.isdigit():
            idx = int(ident)
            if 1 <= idx <= len(self.conversations):
                return self.conversations[idx - 1]

        raise RuntimeError(f"conversation not found: {ident}")

    def _handle_message(self, msg: Dict[str, Any]) -> str:
        typ = str(msg.get("type", "")).strip().lower()

        if typ == "conversation_meta":
            conv_id = str(msg.get("conversation_id", "")).strip()
            conv_title = str(msg.get("conversation_title", "")).strip()
            if conv_id:
                self.current_conversation_id = conv_id
            if conv_title:
                self.current_conversation_title = conv_title
            return "continue"

        if typ == "thinking_round":
            round_idx = msg.get("round")
            self._print_thinking("thinking", f"\n【思考 {round_idx}】\n")
            return "continue"

        if typ == "thinking":
            phase = str(msg.get("phase", "thinking"))
            text = str(msg.get("text", ""))
            self._print_thinking(phase, text)
            return "continue"

        if typ == "status":
            phase = str(msg.get("phase", "status"))
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
    parser.add_argument("--conversation-id", default="", help="Select a specific conversation ID on startup")
    parser.add_argument("--new-conversation", action="store_true", help="Create a new conversation on startup")
    parser.add_argument("--conversation-title", default="", help="Title for --new-conversation")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ws_url = args.ws_url.strip() or derive_ws_url(str(args.host), int(args.port), bool(args.ssl))

    app = LocalClientApp(
        ws_url=ws_url,
        timeout=float(args.timeout),
        show_health=not bool(args.no_health),
        preferred_conversation_id=str(args.conversation_id or ""),
    )

    try:
        app.start()
        if args.new_conversation:
            app.create_conversation(str(args.conversation_title or ""))
        elif args.conversation_id:
            app.select_conversation(str(args.conversation_id))

        if args.question:
            return app.ask(args.question)
        return app.repl()
    except (ConnectionError, OSError, WebSocketError, socket.timeout, RuntimeError) as e:
        _stderr(f"[fatal] {e}")
        return 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
