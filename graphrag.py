# -*- coding: utf-8 -*-
"""
graphrag.py

Neo4j-based retrieval helpers for the no-index version.

Per current design:
- Answering / agent orchestration lives in agent.py.
- This module provides only:
  1) Neo4j connection helpers
  2) Evidence retrieval by direct keyword scan in the graph

IMPORTANT:
- Do NOT import llama_cpp here.
- Do NOT create or use Neo4j fulltext/vector indexes here.
- Keep imports lightweight at module import time (server.py imports this module).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase  # type: ignore

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
# For local CLI usage we still allow env overrides; server.py must read constants.py.
NEO4J_URI = os.getenv("NEO4J_URI", _cget("NEO4J_URI", "bolt://localhost:58287"))
NEO4J_USER = os.getenv("NEO4J_USER", _cget("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", _cget("NEO4J_PASSWORD", "password"))
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", _cget("NEO4J_DATABASE", "neo4j"))

TOP_K = int(os.getenv("GRAPHRAG_TOP_K", "8"))
EXPAND_K = int(os.getenv("GRAPHRAG_EXPAND_K", "20"))


# =============================================================================
# Neo4j helpers
# =============================================================================

def create_driver() -> neo4j.Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def neo4j_run(
    driver: neo4j.Driver,
    cypher: str,
    params: Optional[dict] = None,
    *,
    database: Optional[str] = None,
) -> List[dict]:
    params = params or {}
    records, _, _ = driver.execute_query(
        cypher,
        params,
        database_=(database or NEO4J_DATABASE),
        routing_=neo4j.RoutingControl.READ,
    )
    return [dict(r) for r in records]


# =============================================================================
# Retrieval tool (keywords -> evidence)
# =============================================================================

def _normalize_keywords(keywords: Any) -> List[str]:
    if keywords is None:
        return []
    if isinstance(keywords, str):
        text = keywords.strip()
        return [text] if text else []
    if isinstance(keywords, list):
        out: List[str] = []
        for item in keywords:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(keywords).strip()
    return [text] if text else []


def retrieve_evidence(
    driver: neo4j.Driver,
    keywords: Any,
    *,
    top_k: int = TOP_K,
    expand_k: int = EXPAND_K,
    database: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Retrieve evidence by scanning node text directly, without any index.

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
    kws_raw = _normalize_keywords(keywords)
    if not kws_raw:
        return [], "", {"reason": "empty_keywords"}

    kws = [k.lower() for k in kws_raw]
    db = database or NEO4J_DATABASE

    cypher = """
    WITH [kw IN $kws WHERE trim(kw) <> ''] AS kws
    MATCH (n)
    WITH
      n,
      [kw IN kws WHERE toLower(coalesce(n.Name, n.name, '')) CONTAINS kw] AS name_hits,
      [kw IN kws WHERE toLower(coalesce(n.Description, n.description, '')) CONTAINS kw] AS desc_hits
    WITH
      n,
      size(name_hits) AS name_hit_count,
      size(desc_hits) AS desc_hit_count
    WHERE name_hit_count > 0 OR desc_hit_count > 0
    WITH
      n,
      (toFloat(name_hit_count) * 2.0 + toFloat(desc_hit_count)) AS score
    ORDER BY score DESC, size(coalesce(n.Description, n.description, '')) DESC, coalesce(n.Name, n.name, '') ASC
    LIMIT $top_k
    OPTIONAL MATCH (n)-[r]-(m)
    WITH n, score, collect(DISTINCT {
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
      score AS score,
      neighbors
    ORDER BY score DESC, name ASC
    """

    rows = neo4j_run(
        driver,
        cypher,
        {"kws": kws, "top_k": int(top_k), "k": int(expand_k)},
        database=db,
    )
    meta = {
        "mode": "contains_scan",
        "keywords": kws_raw,
        "database": db,
        "top_k": int(top_k),
        "expand_k": int(expand_k),
    }

    evidence_items: List[Dict[str, Any]] = []
    context_chunks: List[str] = []

    for idx, row in enumerate(rows, start=1):
        cid = f"C{idx}"
        eid = row.get("eid")
        labels = row.get("labels") or []
        name = row.get("name") or ""
        desc = row.get("desc") or ""
        score = float(row.get("score") or 0.0)
        neighbors = row.get("neighbors") or []

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def add_node(node_eid: str, node_labels: List[str], node_name: str, node_desc: str) -> None:
            if not node_eid:
                return
            if node_eid not in nodes_map:
                nodes_map[node_eid] = {
                    "eid": node_eid,
                    "labels": node_labels,
                    "name": node_name,
                    "desc": node_desc,
                }

        add_node(eid, labels, name, desc)

        for nb in neighbors:
            m_eid = nb.get("m_eid")
            m_labels = nb.get("m_labels") or []
            m_name = nb.get("m_name") or ""
            m_desc = nb.get("m_desc") or ""
            add_node(m_eid, m_labels, m_name, m_desc)

            rid = nb.get("rel_eid")
            rel_type = nb.get("rel_type") or ""
            if rid and eid and m_eid:
                edges.append({"rid": rid, "type": rel_type, "source": eid, "target": m_eid})

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
                rel_type = nb.get("rel_type")
                nb_labels = nb.get("m_labels")
                nb_name = nb.get("m_name", "")
                nb_desc = nb.get("m_desc", "")
                if isinstance(nb_desc, str) and len(nb_desc) > 300:
                    nb_desc = nb_desc[:300] + "..."
                lines.append(f"- {rel_type} -> (labels={nb_labels}) {nb_name} | {nb_desc}")

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
