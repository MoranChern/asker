# -*- coding: utf-8 -*-
"""
server.py

Backend for the web app (must be runnable with no args):

  python server.py

Key constraints:
- The server process itself must NOT use GPU and must not import llama_cpp.
- When GPU is needed (LLM generation), spawn a short-lived worker process (gpu_worker.py),
  stream outputs, then exit to release GPU memory.
- Streaming: send "thinking" and "answer" tokens in real time.
- Status / progress logs are emitted separately from model thinking.
- Evidence: show a graph (nodes/edges). Clicking nodes/edges can fetch details from backend.

Current behavior:
- Answering is orchestrated by agent.py.
- Retrieval is a tool; the agent decides if/when to use it.
- Retrieval in this branch is direct Neo4j scanning and does not use any index.
- Conversation and message history are persisted in SQLite.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import neo4j
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase  # type: ignore

import constants as C
from agent import GraphSearchTool, run_agent_async


# -----------------------------------------------------------------------------
# Neo4j
# -----------------------------------------------------------------------------
NEO4J_URI = C.NEO4J_URI
NEO4J_USER = C.NEO4J_USER
NEO4J_PASSWORD = C.NEO4J_PASSWORD
NEO4J_DATABASE = C.NEO4J_DATABASE

TOP_K = int(C.SERVER_TOP_K)
EXPAND_K = int(C.SERVER_EXPAND_K)


def create_driver() -> neo4j.Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


_DRIVER: Optional[neo4j.Driver] = None


def neo4j_run(cypher: str, params: Optional[dict] = None) -> List[dict]:
    assert _DRIVER is not None
    params = params or {}
    records, _, _ = _DRIVER.execute_query(
        cypher,
        params,
        database_=NEO4J_DATABASE,
        routing_=neo4j.RoutingControl.READ,
    )
    return [dict(r) for r in records]


def json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe_value(x) for k, x in v.items()}
    return str(v)


def json_safe(obj: Any) -> Any:
    return json_safe_value(obj)


# -----------------------------------------------------------------------------
# Paths / SQLite
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHAT_DB_PATH = BASE_DIR / "chat_history.sqlite3"
HISTORY_MESSAGE_LIMIT = 12


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CHAT_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _conversation_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "message_count": int(row["message_count"] or 0),
    }


def _message_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "conversation_id": str(row["conversation_id"]),
        "role": str(row["role"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
    }


def init_chat_db() -> None:
    with db_connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id ON messages(conversation_id, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC, created_at DESC)"
        )

        count = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
        if count == 0:
            _create_conversation_locked(conn, None)

        conn.commit()


def _default_conversation_title_locked(conn: sqlite3.Connection) -> str:
    count = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
    return f"新会话 {count + 1}"


def _create_conversation_locked(conn: sqlite3.Connection, title: Optional[str]) -> Dict[str, Any]:
    now = utc_now_iso()
    conv_id = uuid.uuid4().hex
    conv_title = (title or "").strip() or _default_conversation_title_locked(conn)
    conn.execute(
        "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, conv_title, now, now),
    )
    row = conn.execute(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               COALESCE(COUNT(m.id), 0) AS message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.id = ?
        GROUP BY c.id, c.title, c.created_at, c.updated_at
        """,
        (conv_id,),
    ).fetchone()
    assert row is not None
    return _conversation_row_to_dict(row)


def list_conversations_db() -> List[Dict[str, Any]]:
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COALESCE(COUNT(m.id), 0) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id, c.title, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC, c.created_at DESC, c.id DESC
            """
        ).fetchall()
    return [_conversation_row_to_dict(row) for row in rows]


def get_conversation_db(conversation_id: str) -> Optional[Dict[str, Any]]:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COALESCE(COUNT(m.id), 0) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id, c.title, c.created_at, c.updated_at
            """,
            (conversation_id,),
        ).fetchone()
    return _conversation_row_to_dict(row) if row is not None else None


def create_conversation_db(title: Optional[str]) -> Dict[str, Any]:
    with db_connect() as conn:
        data = _create_conversation_locked(conn, title)
        conn.commit()
    return data


def rename_conversation_db(conversation_id: str, title: str) -> Dict[str, Any]:
    new_title = (title or "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="title must not be empty")

    with db_connect() as conn:
        exists = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="conversation not found")

        now = utc_now_iso()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, now, conversation_id),
        )
        conn.commit()

    data = get_conversation_db(conversation_id)
    if data is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return data


def delete_conversation_db(conversation_id: str) -> None:
    with db_connect() as conn:
        row = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def get_messages_db(conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if get_conversation_db(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    with db_connect() as conn:
        if limit is None:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM (
                    SELECT id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, int(limit)),
            ).fetchall()
    return [_message_row_to_dict(row) for row in rows]


def save_message_db(conversation_id: str, role: str, content: str) -> Dict[str, Any]:
    msg_role = str(role or "").strip().lower()
    if msg_role not in ("system", "user", "assistant"):
        raise ValueError(f"invalid role: {role}")
    if get_conversation_db(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    now = utc_now_iso()
    with db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, msg_role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE id = ?
            """,
            (int(cur.lastrowid),),
        ).fetchone()
    assert row is not None
    return _message_row_to_dict(row)


def get_or_create_latest_conversation() -> Dict[str, Any]:
    items = list_conversations_db()
    if items:
        return items[0]
    return create_conversation_db(None)


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DRIVER
    init_chat_db()
    _DRIVER = create_driver()
    try:
        await asyncio.to_thread(_DRIVER.verify_connectivity)
    except Exception as exc:
        print(f"[warn] Neo4j connectivity check failed: {exc}")
    yield
    if _DRIVER is not None:
        try:
            _DRIVER.close()
        except Exception:
            pass
        _DRIVER = None


app = FastAPI(title="GraphRAG WebApp", version="0.3", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_GPU_SEM = asyncio.Semaphore(int(C.SERVER_GPU_WORKERS))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    ok = _DRIVER is not None
    return JSONResponse(
        {
            "ok": ok,
            "neo4j_uri": NEO4J_URI,
            "neo4j_db": NEO4J_DATABASE,
            "chat_db": str(CHAT_DB_PATH),
            "conversation_count": len(list_conversations_db()),
        }
    )


@app.get("/api/conversations")
def list_conversations_api() -> JSONResponse:
    return JSONResponse({"items": list_conversations_db()})


@app.post("/api/conversations")
def create_conversation_api(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    title = None
    if isinstance(payload, dict):
        title = payload.get("title")
    item = create_conversation_db(None if title is None else str(title))
    return JSONResponse(item)


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages_api(conversation_id: str) -> JSONResponse:
    return JSONResponse({"items": get_messages_db(conversation_id)})


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation_api(
    conversation_id: str,
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> JSONResponse:
    title = ""
    if isinstance(payload, dict):
        title = str(payload.get("title", ""))
    item = rename_conversation_db(conversation_id, title)
    return JSONResponse(item)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation_api(conversation_id: str) -> JSONResponse:
    delete_conversation_db(conversation_id)
    return JSONResponse({"ok": True, "deleted_id": conversation_id})


@app.get("/api/node/{eid}")
def node_detail(eid: str) -> JSONResponse:
    if _DRIVER is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not ready")
    cypher = """
    MATCH (n) WHERE elementId(n) = $eid
    RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props,
           coalesce(n.Name, n.name, '') AS name,
           coalesce(n.Description, n.description, '') AS desc
    """
    rows = neo4j_run(cypher, {"eid": eid})
    if not rows:
        raise HTTPException(status_code=404, detail="node not found")
    out = dict(rows[0])
    out["props"] = json_safe(out.get("props", {}))
    return JSONResponse(out)


@app.get("/api/rel/{rid}")
def rel_detail(rid: str) -> JSONResponse:
    if _DRIVER is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not ready")
    cypher = """
    MATCH (a)-[r]-(b) WHERE elementId(r) = $rid
    RETURN elementId(r) AS rid, type(r) AS type, properties(r) AS props,
           elementId(a) AS source, elementId(b) AS target
    """
    rows = neo4j_run(cypher, {"rid": rid})
    if not rows:
        raise HTTPException(status_code=404, detail="relationship not found")
    out = dict(rows[0])
    out["props"] = json_safe(out.get("props", {}))
    return JSONResponse(out)


# -----------------------------------------------------------------------------
# GPU worker spawning (JSONL streaming)
# -----------------------------------------------------------------------------

async def _spawn_gpu_worker(
    *,
    messages: List[Dict[str, str]],
    think: bool,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    forward_answer_events: bool = True,
    forward_done_event: bool = True,
) -> str:
    """Spawn gpu_worker.py and forward JSONL events."""

    async def send(obj: Dict[str, Any]) -> None:
        await emit(obj)

    async def status(phase: str, text: str) -> None:
        await send({"type": "status", "phase": phase, "text": text})

    await status("worker_progress", "gpu_worker: waiting for GPU slot...\n")

    answer_buf: List[str] = []

    async with _GPU_SEM:
        await status("worker_progress", "gpu_worker: slot acquired, starting process...\n")

        seen_done = False
        seen_error = False
        stderr_lines: List[str] = []
        stdout_nonjson: List[str] = []

        proc: Optional[asyncio.subprocess.Process] = None
        heartbeat_task: Optional[asyncio.Task] = None
        stderr_task: Optional[asyncio.Task] = None
        stop_heartbeat = asyncio.Event()

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(BASE_DIR / "gpu_worker.py"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdin and proc.stdout and proc.stderr

            await status("worker_progress", f"gpu_worker: started pid={proc.pid}\n")

            async def _heartbeat() -> None:
                tick = 0
                while not stop_heartbeat.is_set():
                    tick += 1
                    await status("worker_progress", f"gpu_worker: running... (tick={tick})\n")
                    await asyncio.sleep(float(getattr(C, "SERVER_WORKER_HEARTBEAT_SEC", 2.0)))

            heartbeat_task = asyncio.create_task(_heartbeat())

            async def _drain_stderr() -> None:
                try:
                    while True:
                        line = await proc.stderr.readline()
                        if not line:
                            break
                        txt = line.decode("utf-8", errors="replace")
                        stderr_lines.append(txt)
                        if txt.strip():
                            await send({"type": "status", "phase": "worker_stderr", "text": txt})
                except WebSocketDisconnect:
                    raise
                except Exception:
                    return

            stderr_task = asyncio.create_task(_drain_stderr())

            req = {"messages": messages, "think": bool(think)}
            payload = json.dumps(req, ensure_ascii=False).encode("utf-8")
            proc.stdin.write(payload)
            await proc.stdin.drain()
            proc.stdin.close()

            await status("worker_progress", f"gpu_worker: request sent (bytes={len(payload)})\n")

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
                if not s:
                    continue

                try:
                    obj = json.loads(s)
                except Exception:
                    stdout_nonjson.append(s)
                    obj = {"type": "status", "phase": "worker_stdout", "text": s + "\n"}

                typ = obj.get("type")

                if typ == "answer":
                    txt = obj.get("text") or ""
                    answer_buf.append(str(txt))

                if typ == "answer" and not forward_answer_events:
                    continue
                if typ == "done" and not forward_done_event:
                    seen_done = True
                    break

                if typ == "error":
                    seen_error = True
                    trace = obj.get("trace") or ""
                    msg = obj.get("message") or "gpu_worker error"
                    print(f"[gpu_worker error] {msg}")
                    if trace:
                        print(trace)
                        await send({"type": "status", "phase": "worker_traceback", "text": trace + "\n"})
                    await send(obj)
                    continue

                await send(obj)

                if typ == "done":
                    seen_done = True
                    break

            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=float(getattr(C, "SERVER_WORKER_TERMINATE_TIMEOUT_SEC", 2.0)),
                )
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(
                        proc.wait(),
                        timeout=float(getattr(C, "SERVER_WORKER_KILL_TIMEOUT_SEC", 2.0)),
                    )
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            rc = proc.returncode
            await status("worker_progress", f"gpu_worker: exited (returncode={rc})\n")

            if (rc is not None and rc != 0) and not seen_error:
                trace = "".join(stderr_lines).strip()
                if not trace and stdout_nonjson:
                    trace = "\n".join(stdout_nonjson).strip()
                if not trace:
                    trace = "(no stderr captured)"
                await send({"type": "error", "message": f"gpu_worker crashed (returncode={rc})", "trace": trace})

            if forward_done_event and not seen_done:
                await send({"type": "done"})

        finally:
            stop_heartbeat.set()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            if stderr_task is not None:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    return "".join(answer_buf)


# -----------------------------------------------------------------------------
# WebSocket: agent-driven loop
# -----------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def emit(obj: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await emit({"type": "error", "message": "Invalid JSON"})
                continue

            if msg.get("type") != "question":
                await emit({"type": "error", "message": "Unknown message type"})
                continue

            question = str(msg.get("text", "")).strip()
            if not question:
                await emit({"type": "error", "message": "Empty question"})
                continue

            if _DRIVER is None:
                await emit({"type": "error", "message": "Neo4j driver not ready"})
                await emit({"type": "done"})
                continue

            conv_id = str(msg.get("conversation_id", "")).strip()
            if conv_id:
                conv = get_conversation_db(conv_id)
                if conv is None:
                    await emit({"type": "error", "message": f"conversation not found: {conv_id}"})
                    await emit({"type": "done"})
                    continue
            else:
                conv = get_or_create_latest_conversation()
                conv_id = str(conv["id"])

            await emit({
                "type": "conversation_meta",
                "conversation_id": conv_id,
                "conversation_title": conv.get("title", ""),
            })

            history_messages = [
                {"role": item["role"], "content": item["content"]}
                for item in get_messages_db(conv_id, limit=HISTORY_MESSAGE_LIMIT)
            ]

            tool = GraphSearchTool(
                driver=_DRIVER,
                top_k=TOP_K,
                expand_k=EXPAND_K,
                database=NEO4J_DATABASE,
            )

            async def llm_call(
                messages: List[Dict[str, str]],
                think: bool,
                forward_answer_events: bool,
                forward_done_event: bool,
            ) -> str:
                return await _spawn_gpu_worker(
                    messages=messages,
                    think=think,
                    emit=emit,
                    forward_answer_events=forward_answer_events,
                    forward_done_event=forward_done_event,
                )

            try:
                answer_text = await run_agent_async(
                    question=question,
                    llm_call=llm_call,
                    search_tool=tool,
                    emit=emit,
                    max_search_rounds=3,
                    chat_history=history_messages,
                )
                if answer_text:
                    save_message_db(conv_id, "user", question)
                    save_message_db(conv_id, "assistant", answer_text)
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                tb = traceback.format_exc(limit=50)
                print("[server] agent exception:", exc)
                print(tb)
                try:
                    await emit({"type": "error", "message": f"server exception: {exc}", "trace": tb})
                    await emit({"type": "done"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    host = C.SERVER_BIND_HOST
    port = int(C.SERVER_BIND_PORT)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
