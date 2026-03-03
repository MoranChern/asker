# -*- coding: utf-8 -*-
"""
server.py

Backend for the web APP (must be runnable with NO args):

  python server.py

Key constraints from requirements:
- The server process itself must NOT use GPU and must not import llama_cpp.
- When GPU is needed (LLM generation), spawn a short-lived worker process (gpu_worker.py),
  stream outputs, then kill/exit to release GPU memory.
- Streaming: send "thinking" and "answer" tokens in real time.
- Evidence: show a graph (nodes/edges). Clicking nodes/edges can fetch details from backend.

This server uses:
- FastAPI + WebSocket for streaming
- Neo4j driver for evidence retrieval (CPU-only)
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase  # type: ignore
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import constants as C


# -----------------------------------------------------------------------------
# Small text helpers (kept here to avoid importing heavy GraphRAG modules)
# -----------------------------------------------------------------------------

def lucene_escape(s: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", s)


def extract_terms(question: str) -> List[str]:
    quoted = []
    for p in [r'"([^"]+)"', r"'([^']+)'", r"“([^”]+)”", r"「([^」]+)」"]:
        quoted.extend([t.strip() for t in re.findall(p, question) if t.strip()])
    if quoted:
        return quoted[:3]

    cleaned = re.sub(r"[，,。.!?；;:：()\[\]{}<>《》“”‘’\"'`]", " ", question)
    toks = [t.strip() for t in cleaned.split() if t.strip()]
    stop = {"是什么", "什么", "解释", "介绍", "请问", "如何", "为什么", "怎么", "定义", "meaning", "define", "explain", "what"}
    toks = [t for t in toks if t not in stop]
    toks.sort(key=len, reverse=True)
    return toks[:3]


# -----------------------------------------------------------------------------
# Neo4j
# -----------------------------------------------------------------------------

# IMPORTANT: Per requirement, server parameters MUST come from constants.py only.
# Do NOT use environment variables here.
NEO4J_URI = C.NEO4J_URI
NEO4J_USER = C.NEO4J_USER
NEO4J_PASSWORD = C.NEO4J_PASSWORD
NEO4J_DATABASE = C.NEO4J_DATABASE

FULLTEXT_INDEX_NAME = C.SERVER_FULLTEXT_INDEX_NAME
TOP_K = int(C.SERVER_TOP_K)
EXPAND_K = int(C.SERVER_EXPAND_K)


def create_driver() -> neo4j.Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


_DRIVER: Optional[neo4j.Driver] = None


def neo4j_run(cypher: str, params: Optional[dict] = None) -> List[dict]:
    assert _DRIVER is not None
    params = params or {}
    records, _, _ = _DRIVER.execute_query(cypher, params, database_=NEO4J_DATABASE, routing_=neo4j.RoutingControl.READ)
    return [dict(r) for r in records]


def index_exists(name: str) -> bool:
    rows = neo4j_run("SHOW INDEXES YIELD name RETURN name")
    return any(r.get("name") == name for r in rows)




def json_safe_value(v: Any) -> Any:
    """Convert Neo4j values to JSON-serializable types (fallback to str)."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe_value(x) for k, x in v.items()}
    # Neo4j temporal/spatial types, custom classes...
    return str(v)


def json_safe(obj: Any) -> Any:
    return json_safe_value(obj)

# -----------------------------------------------------------------------------
# Evidence retrieval (safe fulltext + neighbor expansion)
# -----------------------------------------------------------------------------

def retrieve_evidence(question: str, top_k: int = TOP_K, expand_k: int = EXPAND_K) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Return (evidence_items, context_text, debug_meta)."""
    if not question.strip():
        return [], "", {"reason": "empty"}

    terms = extract_terms(question)
    if not terms:
        return [], "", {"reason": "no_terms"}

    lucene = " OR ".join([f'"{lucene_escape(t)}"' for t in terms])

    # Prefer fulltext index if it exists; otherwise fallback to slow contains-search.
    use_fulltext = False
    try:
        use_fulltext = index_exists(FULLTEXT_INDEX_NAME)
    except Exception:
        use_fulltext = False

    if use_fulltext:
        cypher = """
        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT $top_k
        OPTIONAL MATCH (node)-[r]-(m)
        WITH node, score, collect(DISTINCT {
          rel_eid: elementId(r),
          rel_type: type(r),
          m_eid: elementId(m),
          m_labels: labels(m),
          m_name: coalesce(m.Name, m.name, ''),
          m_desc: coalesce(m.Description, m.description, '')
        })[0..$k] AS neighbors
        RETURN
          elementId(node) AS eid,
          labels(node) AS labels,
          coalesce(node.Name, node.name, '') AS name,
          coalesce(node.Description, node.description, '') AS desc,
          score AS score,
          neighbors
        ORDER BY score DESC
        """
        rows = neo4j_run(
            cypher,
            {"index": FULLTEXT_INDEX_NAME, "q": lucene, "top_k": int(top_k), "k": int(expand_k)},
        )
        meta = {"mode": "fulltext", "terms": terms, "lucene": lucene, "index": FULLTEXT_INDEX_NAME}
    else:
        # Fallback (no index): match by contains on Name/Description
        # NOTE: This is slower; keep LIMIT small.
        cypher = """
        WITH $terms AS terms
        MATCH (n)
        WHERE any(t IN terms WHERE
          toLower(coalesce(n.Name, n.name, '')) CONTAINS toLower(t)
          OR toLower(coalesce(n.Description, n.description, '')) CONTAINS toLower(t)
        )
        WITH n LIMIT $top_k
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, collect(DISTINCT {
          rel_eid: elementId(r),
          rel_type: type(r),
          m_eid: elementId(m),
          m_labels: labels(m),
          m_name: coalesce(m.Name, m.name, ''),
          m_desc: coalesce(m.Description, m.description, '')
        })[0..$k] AS neighbors
        RETURN
          elementId(n) AS eid,
          labels(n) AS labels,
          coalesce(n.Name, n.name, '') AS name,
          coalesce(n.Description, n.description, '') AS desc,
          0.0 AS score,
          neighbors
        LIMIT $top_k
        """
        rows = neo4j_run(cypher, {"terms": terms, "top_k": int(top_k), "k": int(expand_k)})
        meta = {"mode": "fallback_contains", "terms": terms, "lucene": lucene, "index_missing": True}

    evidence_items: List[Dict[str, Any]] = []
    context_chunks: List[str] = []

    for idx, r in enumerate(rows, start=1):
        cid = f"C{idx}"
        eid = r.get("eid")
        labels = r.get("labels") or []
        name = r.get("name") or ""
        desc = r.get("desc") or ""
        score = r.get("score", 0.0)
        neighbors = r.get("neighbors") or []

        # Build graph elements (nodes + edges)
        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def add_node(n_eid: str, n_labels: List[str], n_name: str, n_desc: str):
            if not n_eid:
                return
            if n_eid not in nodes_map:
                nodes_map[n_eid] = {
                    "eid": n_eid,
                    "labels": n_labels,
                    "name": n_name,
                    "desc": n_desc,
                }

        add_node(eid, labels, name, desc)

        for nb in neighbors:
            m_eid = nb.get("m_eid")
            m_labels = nb.get("m_labels") or []
            m_name = nb.get("m_name") or ""
            m_desc = nb.get("m_desc") or ""
            add_node(m_eid, m_labels, m_name, m_desc)

            rid = nb.get("rel_eid")
            rtype = nb.get("rel_type") or ""
            if rid and eid and m_eid:
                edges.append(
                    {
                        "rid": rid,
                        "type": rtype,
                        "source": eid,
                        "target": m_eid,
                    }
                )

        # Context text (for grounding)
        lines: List[str] = []
        lines.append(f"{cid}:")
        lines.append(f"Node(eid={eid}, labels={labels})")
        if name:
            lines.append(f"Name: {name}")
        if desc:
            lines.append(f"Description: {str(desc)[:1200]}")
        if neighbors:
            lines.append("Neighbors (1-hop, sampled):")
            for nb in neighbors:
                rel = nb.get("rel_type")
                nb_labels = nb.get("m_labels")
                nb_name = nb.get("m_name", "")
                nb_desc = nb.get("m_desc", "")
                if isinstance(nb_desc, str) and len(nb_desc) > 300:
                    nb_desc = nb_desc[:300] + "..."
                lines.append(f"- {rel} -> (labels={nb_labels}) {nb_name} | {nb_desc}")

        context_chunks.append("\n".join(lines))

        evidence_items.append(
            {
                "cid": cid,
                "score": score,
                "node": {"eid": eid, "labels": labels, "name": name, "desc": desc},
                "graph": {
                    "nodes": list(nodes_map.values()),
                    "edges": edges,
                },
            }
        )

    context_text = "\n\n".join(context_chunks)
    return evidence_items, context_text, meta


# -----------------------------------------------------------------------------
# Prompt template (same as graphrag.py)
# -----------------------------------------------------------------------------

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


def build_messages(context: str, question: str) -> List[Dict[str, str]]:
    prompt = ANSWER_TEMPLATE.format(context=context, query_text=question)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": prompt},
    ]


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DRIVER
    _DRIVER = create_driver()
    try:
        await asyncio.to_thread(_DRIVER.verify_connectivity)
    except Exception as e:
        print(f"[warn] Neo4j connectivity check failed: {e}")
    yield
    if _DRIVER is not None:
        try:
            _DRIVER.close()
        except Exception:
            pass
        _DRIVER = None


app = FastAPI(title="GraphRAG WebApp", version="0.1", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global GPU worker concurrency limit (avoid GPU OOM)
_GPU_SEM = asyncio.Semaphore(int(C.SERVER_GPU_WORKERS))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    ok = _DRIVER is not None
    return JSONResponse({"ok": ok, "neo4j_uri": NEO4J_URI, "neo4j_db": NEO4J_DATABASE})


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
    out['props'] = json_safe(out.get('props', {}))
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
    out['props'] = json_safe(out.get('props', {}))
    return JSONResponse(out)


# -----------------------------------------------------------------------------
# WebSocket streaming
# -----------------------------------------------------------------------------

async def _spawn_gpu_worker(messages: List[Dict[str, str]], websocket: WebSocket) -> None:
    """Spawn gpu_worker.py and forward JSONL events to websocket.

    Requirements implemented:
    1) Each invocation outputs server-side progress updates to the client ("Thinking" stream).
    2) Capture ALL gpu_worker exceptions (including import-time crashes) and forward traceback.
    """

    async def send(obj: Dict[str, Any]) -> None:
        # Let WebSocketDisconnect bubble up, but swallow JSON serialization issues.
        await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    async def progress(text: str) -> None:
        await send({"type": "thinking", "phase": "worker_progress", "text": text})

    async def send_error(message: str, trace: str = "") -> None:
        payload: Dict[str, Any] = {"type": "error", "message": message}
        if trace:
            payload["trace"] = trace
        await send(payload)

    await progress("gpu_worker: waiting for GPU slot...\n")

    async with _GPU_SEM:
        await progress("gpu_worker: slot acquired, starting process...\n")

        seen_done = False
        seen_error = False
        stderr_lines: List[str] = []
        stdout_nonjson: List[str] = []

        proc = None
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

            await progress(f"gpu_worker: started pid={proc.pid}\n")

            # Heartbeat progress while worker is alive (covers model loading time where worker may be silent)
            async def _heartbeat() -> None:
                tick = 0
                while not stop_heartbeat.is_set():
                    tick += 1
                    await progress(f"gpu_worker: running... (tick={tick})\n")
                    await asyncio.sleep(float(getattr(C, "SERVER_WORKER_HEARTBEAT_SEC", 2.0)))

            heartbeat_task = asyncio.create_task(_heartbeat())

            # Drain stderr and keep a copy (for traceback)
            async def _drain_stderr() -> None:
                try:
                    while True:
                        line = await proc.stderr.readline()
                        if not line:
                            break
                        txt = line.decode("utf-8", errors="replace")
                        stderr_lines.append(txt)
                        # Surface stderr to UI as well (often contains traceback)
                        if txt.strip():
                            await send({"type": "thinking", "phase": "worker_stderr", "text": txt})
                except WebSocketDisconnect:
                    raise
                except Exception:
                    return

            stderr_task = asyncio.create_task(_drain_stderr())

            # Send request
            req = {"messages": messages, "think": True}
            payload = json.dumps(req, ensure_ascii=False).encode("utf-8")
            proc.stdin.write(payload)
            await proc.stdin.drain()
            proc.stdin.close()

            await progress(f"gpu_worker: request sent (bytes={len(payload)})\n")

            # Read JSONL from stdout
            event_count = 0
            last_event_ts = time.time()

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
                if not s:
                    continue

                event_count += 1
                last_event_ts = time.time()

                try:
                    obj = json.loads(s)
                except Exception:
                    stdout_nonjson.append(s)
                    obj = {"type": "thinking", "phase": "worker_stdout", "text": s + "\n"}

                # If worker sends an error with traceback, ensure we surface traceback clearly
                if obj.get("type") == "error":
                    seen_error = True
                    trace = obj.get("trace") or ""
                    msg = obj.get("message") or "gpu_worker error"
                    # Print to server console
                    print(f"[gpu_worker error] {msg}")
                    if trace:
                        print(trace)
                        # Also surface traceback to UI explicitly
                        await send({"type": "thinking", "phase": "worker_traceback", "text": trace + "\n"})
                    await send(obj)
                    # NOTE: do not break; worker may still emit "done"
                    continue

                await send(obj)

                if obj.get("type") == "done":
                    seen_done = True
                    break

            # Wait for process end
            try:
                await asyncio.wait_for(proc.wait(), timeout=float(getattr(C, "SERVER_WORKER_TERMINATE_TIMEOUT_SEC", 2.0)))
            except Exception:
                # best-effort terminate/kill
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=float(getattr(C, "SERVER_WORKER_KILL_TIMEOUT_SEC", 2.0)))
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            rc = proc.returncode
            await progress(f"gpu_worker: exited (returncode={rc})\n")

            # If worker crashed without emitting a structured error, synthesize one with traceback from stderr.
            if (rc is not None and rc != 0) and not seen_error:
                trace = "".join(stderr_lines).strip()
                if not trace and stdout_nonjson:
                    trace = "\n".join(stdout_nonjson).strip()
                if not trace:
                    trace = "(no stderr captured)"
                await send_error(f"gpu_worker crashed (returncode={rc})", trace=trace)

            # Ensure client always receives done to unblock UI
            if not seen_done:
                await send({"type": "done"})

        except WebSocketDisconnect:
            # Client disconnected; ensure worker is terminated quickly
            if proc is not None:
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
            raise

        except Exception as e:
            tb = traceback.format_exc(limit=50)
            print("[server] _spawn_gpu_worker exception:", e)
            print(tb)
            try:
                await send_error(f"server exception while running gpu_worker: {e}", trace=tb)
                await send({"type": "done"})
            except Exception:
                # If websocket is already dead, just swallow.
                pass

        finally:
            stop_heartbeat.set()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except Exception:
                    pass

            if stderr_task is not None:
                stderr_task.cancel()
                try:
                    await stderr_task
                except Exception:
                    pass

            # Final safeguard: ensure worker is not left behind
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


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}, ensure_ascii=False))
                continue

            if msg.get("type") != "question":
                await websocket.send_text(json.dumps({"type": "error", "message": "Unknown message type"}, ensure_ascii=False))
                continue

            question = str(msg.get("text", "")).strip()
            if not question:
                await websocket.send_text(json.dumps({"type": "error", "message": "Empty question"}, ensure_ascii=False))
                continue

            # 1) Retrieval (query generation is part of thinking)
            await websocket.send_text(json.dumps({"type": "thinking", "phase": "retrieval", "text": "生成检索词...\n"}, ensure_ascii=False))

            if _DRIVER is None:
                await websocket.send_text(json.dumps({"type": "error", "message": "Neo4j driver not ready"}, ensure_ascii=False))
                await websocket.send_text(json.dumps({"type": "done"}, ensure_ascii=False))
                continue

            try:
                evidence_items, context_text, meta = await asyncio.to_thread(retrieve_evidence, question, TOP_K, EXPAND_K)
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Retrieve failed: {e}"}, ensure_ascii=False))
                await websocket.send_text(json.dumps({"type": "done"}, ensure_ascii=False))
                continue

            await websocket.send_text(json.dumps({"type": "thinking", "phase": "retrieval", "text": f"检索完成：{meta}\n"}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"type": "evidence", "items": evidence_items}, ensure_ascii=False))

            if not context_text.strip():
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "answer",
                            "text": "资料不足，无法确定（检索上下文为空）。",
                        },
                        ensure_ascii=False,
                    )
                )
                await websocket.send_text(json.dumps({"type": "done"}, ensure_ascii=False))
                continue

            # 2) LLM generation (GPU worker)
            await websocket.send_text(json.dumps({"type": "thinking", "phase": "llm", "text": "开始生成答案...\n"}, ensure_ascii=False))
            messages = build_messages(context=context_text, question=question)

            try:
                await _spawn_gpu_worker(messages=messages, websocket=websocket)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Worker failed: {e}"}, ensure_ascii=False))
                await websocket.send_text(json.dumps({"type": "done"}, ensure_ascii=False))

    except WebSocketDisconnect:
        return


def main() -> None:
    # Must run without args
    import uvicorn  # local import

    host = C.SERVER_BIND_HOST
    port = int(C.SERVER_BIND_PORT)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
