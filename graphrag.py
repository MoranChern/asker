# -*- coding: utf-8 -*-
"""
graphrag.py

GraphRAG *retrieval tool* + (optional) index build utilities.

Per updated design:
- Answering / agent orchestration lives in agent.py.
- This module provides ONLY:
  1) Neo4j connection helpers
  2) Evidence retrieval (given keywords from the agent)
  3) Index build helpers (CLI/offline), implemented with lazy imports

IMPORTANT:
- Do NOT import llama_cpp here.
- Keep imports lightweight at module import time (server.py imports this module).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase  # type: ignore

# ---- Optional centralized constants (preferred) ----
try:
    import constants as _constants  # type: ignore
except Exception:
    _constants = None


def _cget(name: str, default):
    if _constants is None:
        return default
    return getattr(_constants, name, default)


# =============================================================================
# CONFIG
# =============================================================================
# For CLI usage we allow env overrides; server.py is required to read only constants.py.
NEO4J_URI = os.getenv("NEO4J_URI", _cget("NEO4J_URI", "bolt://localhost:58287"))
NEO4J_USER = os.getenv("NEO4J_USER", _cget("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", _cget("NEO4J_PASSWORD", "password"))
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", _cget("NEO4J_DATABASE", "neo4j"))

# RAG tagging / embedding
RAG_LABEL = os.getenv("GRAPHRAG_RAG_LABEL", "RAG")
EMBEDDING_PROPERTY = os.getenv("GRAPHRAG_EMBEDDING_PROPERTY", "embedding")

VECTOR_INDEX_NAME = os.getenv("GRAPHRAG_VECTOR_INDEX", "graphrag_vector_rag")
FULLTEXT_INDEX_NAME = os.getenv("GRAPHRAG_FULLTEXT_INDEX", "graphrag_fulltext_rag")

TOP_K = int(os.getenv("GRAPHRAG_TOP_K", "8"))
EXPAND_K = int(os.getenv("GRAPHRAG_EXPAND_K", "20"))

EMBED_MODEL = os.getenv(
    "GRAPHRAG_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBED_DEVICE = os.getenv("GRAPHRAG_EMBED_DEVICE", "auto").lower()

TEXT_PROPS = ["Name", "Description", "name", "description"]


# =============================================================================
# Neo4j helpers
# =============================================================================

def create_driver() -> neo4j.Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def neo4j_run(driver: neo4j.Driver, cypher: str, params: Optional[dict] = None, *, database: Optional[str] = None) -> List[dict]:
    params = params or {}
    records, _, _ = driver.execute_query(
        cypher,
        params,
        database_=(database or NEO4J_DATABASE),
        routing_=neo4j.RoutingControl.READ,
    )
    return [dict(r) for r in records]


def index_exists(driver: neo4j.Driver, name: str) -> bool:
    rows = neo4j_run(driver, "SHOW INDEXES YIELD name RETURN name")
    return any(r.get("name") == name for r in rows)


# =============================================================================
# Retrieval tool (keywords -> evidence)
# =============================================================================

_LUCENE_ESCAPE_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def lucene_escape(s: str) -> str:
    return _LUCENE_ESCAPE_RE.sub(r"\\\1", s)


def _normalize_keywords(keywords: Any) -> List[str]:
    """Normalize keywords input from the agent."""
    if keywords is None:
        return []
    if isinstance(keywords, str):
        kw = keywords.strip()
        return [kw] if kw else []
    if isinstance(keywords, list):
        out: List[str] = []
        for x in keywords:
            if x is None:
                continue
            t = str(x).strip()
            if t:
                out.append(t)
        return out
    # fallback
    t = str(keywords).strip()
    return [t] if t else []


def retrieve_evidence(
    driver: neo4j.Driver,
    keywords: Any,
    *,
    top_k: int = TOP_K,
    expand_k: int = EXPAND_K,
    fulltext_index_name: str = FULLTEXT_INDEX_NAME,
    database: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Retrieve evidence given agent-provided keywords.

    Returns (evidence_items, context_text, meta).

    evidence_items item schema matches the web UI:
      {
        "cid": "C1",
        "score": float,
        "node": {...},
        "graph": {"nodes":[...], "edges":[...]}
      }

    context_text:
      "C1:\nNode(...)\n...\n\nC2:\n..."
    """
    kws = _normalize_keywords(keywords)
    if not kws:
        return [], "", {"reason": "empty_keywords"}

    db = database or NEO4J_DATABASE

    use_fulltext = False
    try:
        use_fulltext = index_exists(driver, fulltext_index_name)
    except Exception:
        use_fulltext = False

    lucene = " OR ".join([f'"{lucene_escape(k)}"' for k in kws])

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
            driver,
            cypher,
            {"index": fulltext_index_name, "q": lucene, "top_k": int(top_k), "k": int(expand_k)},
            database=db,
        )
        meta = {"mode": "fulltext", "keywords": kws, "lucene": lucene, "index": fulltext_index_name}
    else:
        # Fallback (no index): match by contains on Name/Description (slower, keep LIMIT small).
        cypher = """
        WITH $kws AS kws
        MATCH (n)
        WHERE any(t IN kws WHERE
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
        rows = neo4j_run(driver, cypher, {"kws": kws, "top_k": int(top_k), "k": int(expand_k)}, database=db)
        meta = {"mode": "fallback_contains", "keywords": kws, "lucene": lucene, "index_missing": True}

    evidence_items: List[Dict[str, Any]] = []
    context_chunks: List[str] = []

    for idx, r in enumerate(rows, start=1):
        cid = f"C{idx}"
        eid = r.get("eid")
        labels = r.get("labels") or []
        name = r.get("name") or ""
        desc = r.get("desc") or ""
        score = float(r.get("score") or 0.0)
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
                edges.append({"rid": rid, "type": rtype, "source": eid, "target": m_eid})

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
                "graph": {"nodes": list(nodes_map.values()), "edges": edges},
            }
        )

    context_text = "\n\n".join(context_chunks)
    return evidence_items, context_text, meta


# =============================================================================
# Index build utilities (CLI/offline) - LAZY imports
# =============================================================================

def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--"
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def tag_rag_nodes(
    driver: neo4j.Driver,
    batch_size: int = 10000,
    verbose: bool = False,
    min_print_interval_sec: float = 2.0,
) -> int:
    """Tag nodes that have any non-empty text-ish property with :RAG_LABEL."""
    cypher = f"""
    MATCH (n)
    WHERE any(p IN $props WHERE n[p] IS NOT NULL AND toString(n[p]) <> '')
      AND NOT n:`{RAG_LABEL}`
    WITH n
    LIMIT $limit
    SET n:`{RAG_LABEL}`
    RETURN count(n) AS tagged
    """

    count_cypher = f"""
    MATCH (n)
    WHERE any(p IN $props WHERE n[p] IS NOT NULL AND toString(n[p]) <> '')
      AND NOT n:`{RAG_LABEL}`
    RETURN count(n) AS total
    """

    total = 0
    batch = 0
    t0 = time.time()
    last_print = 0.0

    target: Optional[int] = None
    if verbose:
        try:
            rows = neo4j_run(driver, count_cypher, {"props": TEXT_PROPS})
            target = int(rows[0]["total"]) if rows else None
        except Exception:
            target = None

        if target is not None:
            print(f"[tag] start batch_size={batch_size} target={target}", flush=True)
        else:
            print(f"[tag] start batch_size={batch_size} target=unknown", flush=True)

    while True:
        batch += 1
        rows = neo4j_run(driver, cypher, {"props": TEXT_PROPS, "limit": int(batch_size)})
        tagged = int(rows[0]["tagged"]) if rows else 0
        if tagged <= 0:
            break

        total += tagged

        if verbose:
            now = time.time()
            if last_print == 0.0 or (now - last_print) >= float(min_print_interval_sec):
                elapsed = max(1e-6, now - t0)
                rate = total / elapsed

                eta_sec: Optional[float] = None
                pct_str = ""
                if target is not None and target > 0:
                    remain = max(0, target - total)
                    eta_sec = (remain / rate) if rate > 1e-9 else None
                    pct = min(100.0, (total / target) * 100.0)
                    pct_str = f" {pct:.2f}%"

                eta_str = _format_eta(eta_sec)
                tgt_str = str(target) if target is not None else "?"
                print(
                    f"[tag] batch={batch} +{tagged} total={total}/{tgt_str}{pct_str} "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f} nodes/s eta={eta_str}",
                    flush=True,
                )
                last_print = now

    if verbose:
        elapsed = max(1e-6, time.time() - t0)
        print(f"[tag] done total={total} elapsed={elapsed:.1f}s", flush=True)

    return total


def make_embedder():
    """Create SentenceTransformer embedder (lazy import)."""
    from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings  # local import

    kwargs: Dict[str, Any] = {}
    if EMBED_DEVICE in ("cpu", "cuda", "mps"):
        kwargs["device"] = EMBED_DEVICE
    return SentenceTransformerEmbeddings(model=EMBED_MODEL, **kwargs)


def build_indexes(
    driver: neo4j.Driver,
    embedder,
    similarity: str = "cosine",
    tag_batch_size: int = 10000,
    verbose: bool = False,
    min_print_interval_sec: float = 2.0,
) -> int:
    """Ensure :RAG_LABEL exists, create fulltext + vector index if needed. Returns embedding dim."""
    from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index  # local import

    if verbose:
        print("[build] tagging RAG nodes (batched)...", flush=True)
    tag_rag_nodes(
        driver,
        batch_size=tag_batch_size,
        verbose=verbose,
        min_print_interval_sec=min_print_interval_sec,
    )

    if verbose:
        print("[build] ensuring fulltext index...", flush=True)
    if not index_exists(driver, FULLTEXT_INDEX_NAME):
        create_fulltext_index(
            driver,
            FULLTEXT_INDEX_NAME,
            label=RAG_LABEL,
            node_properties=TEXT_PROPS,
            fail_if_exists=False,
            neo4j_database=NEO4J_DATABASE,
        )

    if verbose:
        print("[build] probing embedding dimension...", flush=True)
    dim = len(embedder.embed_query("dimension probe"))

    if verbose:
        print("[build] ensuring vector index...", flush=True)
    if not index_exists(driver, VECTOR_INDEX_NAME):
        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label=RAG_LABEL,
            embedding_property=EMBEDDING_PROPERTY,
            dimensions=dim,
            similarity_fn=similarity,
            fail_if_exists=False,
            neo4j_database=NEO4J_DATABASE,
        )

    if verbose:
        print(f"[build] indexes ready (dim={dim})", flush=True)

    return dim


def embed_missing_nodes(
    driver: neo4j.Driver,
    embedder,
    batch_size: int = 128,
    max_nodes: int = 0,
    force_reembed: bool = False,
    verbose: bool = False,
    min_print_interval_sec: float = 2.0,
) -> int:
    """Compute embeddings for nodes that miss EMBEDDING_PROPERTY (or force)."""
    from neo4j_graphrag.indexes import upsert_vectors  # local import

    total = 0
    batch = 0
    t0 = time.time()
    last_print = 0.0

    where = "true" if force_reembed else f"n.`{EMBEDDING_PROPERTY}` IS NULL"

    count_cypher = f"""
    MATCH (n:`{RAG_LABEL}`)
    WHERE {where}
      AND (
        coalesce(n.Name, n.name, '') <> ''
        OR coalesce(n.Description, n.description, '') <> ''
      )
    RETURN count(n) AS total
    """

    target: Optional[int] = None
    if verbose:
        try:
            rows = neo4j_run(driver, count_cypher)
            raw_target = int(rows[0]["total"]) if rows else 0
            if max_nodes and max_nodes > 0:
                target = min(raw_target, int(max_nodes))
            else:
                target = raw_target
        except Exception:
            target = None

        tgt_str = str(target) if target is not None else "unknown"
        print(
            f"[embed] start batch_size={batch_size} target={tgt_str} force_reembed={force_reembed}",
            flush=True,
        )

    while True:
        batch += 1

        limit = int(batch_size)
        if max_nodes and (max_nodes - total) < limit:
            limit = max(0, int(max_nodes - total))
        if limit <= 0:
            break

        cypher = f"""
        MATCH (n:`{RAG_LABEL}`)
        WHERE {where}
          AND (
            coalesce(n.Name, n.name, '') <> ''
            OR coalesce(n.Description, n.description, '') <> ''
          )
        WITH n
        RETURN
          elementId(n) AS eid,
          coalesce(n.Name, n.name, '') AS name,
          coalesce(n.Description, n.description, '') AS desc
        LIMIT $limit
        """

        rows = neo4j_run(driver, cypher, {"limit": limit})
        if not rows:
            break

        eids: List[str] = []
        texts: List[str] = []
        for r in rows:
            name = (r.get("name") or "").strip()
            desc = (r.get("desc") or "").strip()
            txt = f"{name}\n{desc}".strip()
            if not txt:
                continue
            eids.append(r["eid"])
            texts.append(txt)

        if not eids:
            continue

        # Embed
        try:
            vecs = embedder.model.encode(texts)  # type: ignore[attr-defined]
            embeddings = vecs.tolist() if hasattr(vecs, "tolist") else [embedder.embed_query(t) for t in texts]
        except Exception:
            embeddings = [embedder.embed_query(t) for t in texts]

        # Upsert
        upsert_vectors(
            driver,
            ids=eids,
            embedding_property=EMBEDDING_PROPERTY,
            embeddings=embeddings,
            neo4j_database=NEO4J_DATABASE,
        )

        total += len(eids)

        if verbose:
            now = time.time()
            if last_print == 0.0 or (now - last_print) >= float(min_print_interval_sec):
                elapsed = max(1e-6, now - t0)
                rate = total / elapsed

                eta_sec: Optional[float] = None
                pct_str = ""
                tgt_str = str(target) if target is not None else "?"
                if target is not None and target > 0:
                    remain = max(0, target - total)
                    eta_sec = (remain / rate) if rate > 1e-9 else None
                    pct = min(100.0, (total / target) * 100.0)
                    pct_str = f" {pct:.2f}%"

                eta_str = _format_eta(eta_sec)
                print(
                    f"[embed] batch={batch} +{len(eids)} total={total}/{tgt_str}{pct_str} "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f} nodes/s eta={eta_str}",
                    flush=True,
                )
                last_print = now

        if max_nodes and total >= max_nodes:
            break

    if verbose:
        elapsed = max(1e-6, time.time() - t0)
        print(f"[embed] done total={total} elapsed={elapsed:.1f}s", flush=True)

    return total


def ensure_fulltext_index(driver: neo4j.Driver) -> None:
    """Best-effort ensure the fulltext index exists (writes)."""
    try:
        from neo4j_graphrag.indexes import create_fulltext_index  # local import

        tag_rag_nodes(driver)
        if not index_exists(driver, FULLTEXT_INDEX_NAME):
            create_fulltext_index(
                driver,
                FULLTEXT_INDEX_NAME,
                label=RAG_LABEL,
                node_properties=TEXT_PROPS,
                fail_if_exists=False,
                neo4j_database=NEO4J_DATABASE,
            )
    except Exception:
        # Keep silent; retrieval can still work via fallback contains-search.
        pass
