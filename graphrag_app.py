# -*- coding: utf-8 -*-
"""
graphrag_app.py

GraphRAG "Second Edition" (Hybrid + Vectors + Neighbor Expansion) for Neo4j 5.x/2026.x
===================================================================================

This is a more "production-ready" GraphRAG skeleton that upgrades v1 by adding:

A) Vector embeddings + Neo4j native vector index
B) Hybrid retrieval (vector + fulltext) using Neo4j's official `neo4j-graphrag`
C) 1-hop neighborhood expansion to give better "entity explanation" context
D) Robust Text2Cypher fallback (read-only, LIMIT enforced, retry with error feedback)

Key design choices:
- Indexing (writes) is explicit via `--build-index`. Query-time is read-only.
- Only nodes with text-ish properties (Name/Description/name/description) are tagged with a RAG label.
- One vector index + one fulltext index are created on that label.
- Embeddings are stored on the node property `embedding` as LIST<FLOAT>.
- Uses SentenceTransformers as the local embedding model via neo4j-graphrag's built-in embedder.
- Uses your local Qwen (llama.cpp) from qwen.py for generation & Text2Cypher.

Requirements (minimum):
    pip install -U neo4j neo4j-graphrag pydantic

For embeddings (local):
    pip install -U "neo4j-graphrag[sentence-transformers]"

Run:
    python graphrag_app.py
    python graphrag_app.py --build-index

Environment variables (optional):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    GRAPHRAG_RAG_LABEL              (default: "RAG")
    GRAPHRAG_EMBEDDING_PROPERTY     (default: "embedding")
    GRAPHRAG_VECTOR_INDEX           (default: "graphrag_vector_rag")
    GRAPHRAG_FULLTEXT_INDEX         (default: "graphrag_fulltext_rag")
    GRAPHRAG_EMBED_MODEL            (default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    GRAPHRAG_EMBED_DEVICE           (default: "cuda"|"cpu"|"mps"|"auto")  (default: "auto")
    GRAPHRAG_TOP_K                  (default: 8)
    GRAPHRAG_EXPAND_K               (default: 20)
    GRAPHRAG_T2C_MAX_ATTEMPTS       (default: 3)
    GRAPHRAG_T2C_LIMIT              (default: 50)

Notes:
- If you cannot or do not want to download HF models online, set GRAPHRAG_EMBED_MODEL
  to a local SentenceTransformer path.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase  # type: ignore
from neo4j.exceptions import CypherSyntaxError, Neo4jError  # type: ignore

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate, Text2CypherTemplate
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

from neo4j_graphrag.llm.base import LLMInterfaceV2
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage

from neo4j_graphrag.retrievers import HybridRetriever  # official hybrid retriever
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, upsert_vectors

# ---- Your local LLM wrapper (llama.cpp) ----
from qwen import LLM_QWEN_Standalone



# ---- Optional centralized constants (preferred) ----
# If constants.py exists in the same folder, we use it as the default configuration,
# while still allowing environment variables to override.
try:
    import constants as _constants  # type: ignore
except Exception:
    _constants = None

def _cget(name: str, default):
    if _constants is None:
        return default
    return getattr(_constants, name, default)

# =============================================================================
# CONFIG (edit here or via env vars)
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", _cget("NEO4J_URI", "bolt://localhost:58287"))
NEO4J_USER = os.getenv("NEO4J_USER", _cget("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", _cget("NEO4J_PASSWORD", "password"))
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", _cget("NEO4J_DATABASE", "neo4j"))
RAG_LABEL = os.getenv("GRAPHRAG_RAG_LABEL", "RAG")
EMBEDDING_PROPERTY = os.getenv("GRAPHRAG_EMBEDDING_PROPERTY", "embedding")

VECTOR_INDEX_NAME = os.getenv("GRAPHRAG_VECTOR_INDEX", "graphrag_vector_rag")
FULLTEXT_INDEX_NAME = os.getenv("GRAPHRAG_FULLTEXT_INDEX", "graphrag_fulltext_rag")

TOP_K = int(os.getenv("GRAPHRAG_TOP_K", "8"))
EXPAND_K = int(os.getenv("GRAPHRAG_EXPAND_K", "20"))

T2C_MAX_ATTEMPTS = int(os.getenv("GRAPHRAG_T2C_MAX_ATTEMPTS", "3"))
T2C_LIMIT = int(os.getenv("GRAPHRAG_T2C_LIMIT", "50"))

EMBED_MODEL = os.getenv(
    "GRAPHRAG_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBED_DEVICE = os.getenv("GRAPHRAG_EMBED_DEVICE", "auto").lower()


# =============================================================================
# Common utilities
# =============================================================================

def print_sep(char: str = "=", n: int = 80) -> None:
    print(char * n)



FORBIDDEN_CYPHER_KEYWORDS = [
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "DROP", "REMOVE",
    "CALL apoc.", "LOAD CSV", "FOREACH", "GRANT", "REVOKE",
]

_INVALID_LABEL_PIPE_RE = re.compile(r"\([^)]*:\s*`?[\w ]+`?\s*\|", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\b\s+(\d+)", re.IGNORECASE)


def cypher_is_safe_readonly(cypher: str) -> Tuple[bool, str]:
    up = cypher.upper()
    for kw in FORBIDDEN_CYPHER_KEYWORDS:
        if kw.upper() in up:
            return False, f"Forbidden Cypher keyword detected: {kw}"
    return True, "ok"


def cypher_has_invalid_label_pipe(cypher: str) -> bool:
    return bool(_INVALID_LABEL_PIPE_RE.search(cypher))


def ensure_limit(cypher: str, limit: int) -> str:
    m = _LIMIT_RE.search(cypher)
    if not m:
        return cypher.rstrip().rstrip(";") + f"\nLIMIT {limit}"
    try:
        n = int(m.group(1))
        if n > limit:
            return _LIMIT_RE.sub(f"LIMIT {limit}", cypher, count=1)
    except Exception:
        pass
    return cypher


def looks_like_definition_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["是什么", "定义", "meaning", "define", "what is", "explain", "介绍"])


def lucene_escape(s: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", s)


def extract_terms(question: str) -> List[str]:
    # Prefer quoted; otherwise pick longer tokens
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


def neo4j_run(driver: neo4j.Driver, cypher: str, params: Optional[dict] = None) -> List[dict]:
    params = params or {}
    records, _, _ = driver.execute_query(cypher, params, database_=NEO4J_DATABASE)
    return [dict(r) for r in records]


def index_exists(driver: neo4j.Driver, name: str) -> bool:
    rows = neo4j_run(driver, "SHOW INDEXES YIELD name RETURN name")
    return any(r.get("name") == name for r in rows)


# =============================================================================
# LLM adapter (Qwen via llama.cpp) for neo4j-graphrag
# =============================================================================

class QwenLLM(LLMInterfaceV2):
    supports_structured_output: bool = False

    def __init__(self, model_name: str = "qwen-local", model_params: Optional[dict] = None):
        super().__init__(model_name=model_name, model_params=model_params or {})
        self._qwen = LLM_QWEN_Standalone(verbose=False)


    def get_response(self, messages: List[Dict], think: bool, print_type: str) -> str:
        """Wrapper for underlying Qwen standalone client."""
        return self._qwen.get_response(messages=messages, think=think, print_type=print_type)


    def _to_messages(
        self,
        prompt: str,
        message_history: Optional[List[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system_instruction:
            msgs.append({"role": "system", "content": system_instruction})
        if message_history:
            for m in message_history:
                msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def invoke(
        self,
        input: str,
        message_history: Optional[List[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        msgs = self._to_messages(input, message_history=message_history, system_instruction=system_instruction)
        out = self.get_response(messages=msgs, think=bool(self.model_params.get("think", True)), print_type=str(self.model_params.get("print_type", "stream")))
        return LLMResponse(content=out)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[List[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        return self.invoke(input, message_history=message_history, system_instruction=system_instruction)


# =============================================================================
# Index building (writes)
# =============================================================================

TEXT_PROPS = ["Name", "Description", "name", "description"]


def tag_rag_nodes(driver: neo4j.Driver) -> int:
    """
    Tag nodes that have any text-ish property with :RAG_LABEL.
    Returns the number of nodes tagged in this run (best effort).
    """
    cypher = f"""
    MATCH (n)
    WHERE any(p IN $props WHERE n[p] IS NOT NULL)
    SET n:`{RAG_LABEL}`
    RETURN count(n) AS tagged
    """
    rows = neo4j_run(driver, cypher, {"props": TEXT_PROPS})
    return int(rows[0]["tagged"]) if rows else 0


def build_indexes(driver: neo4j.Driver, embedder: SentenceTransformerEmbeddings, similarity: str = "cosine") -> int:
    """
    Ensure :RAG_LABEL exists, create fulltext + vector index if needed.
    Returns embedding dimension.
    """
    tag_rag_nodes(driver)

    # Fulltext index on Name/Description variants
    if not index_exists(driver, FULLTEXT_INDEX_NAME):
        create_fulltext_index(
            driver,
            FULLTEXT_INDEX_NAME,
            label=RAG_LABEL,
            node_properties=TEXT_PROPS,
            fail_if_exists=False,
            neo4j_database=NEO4J_DATABASE,
        )

    # Determine embedding dimension from model
    dim = len(embedder.embed_query("dimension probe"))

    # Vector index on embedding property
    if not index_exists(driver, VECTOR_INDEX_NAME):
        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label=RAG_LABEL,
            embedding_property=EMBEDDING_PROPERTY,
            dimensions=dim,
            similarity_fn=similarity,  # "cosine" or "euclidean"
            fail_if_exists=False,
            neo4j_database=NEO4J_DATABASE,
        )

    return dim


def make_text_for_embedding(name: str, desc: str) -> str:
    name = (name or "").strip()
    desc = (desc or "").strip()
    if name and desc:
        return f"{name}\n{desc}"
    return name or desc


def embed_missing_nodes(
    driver: neo4j.Driver,
    embedder: SentenceTransformerEmbeddings,
    batch_size: int = 128,
    max_nodes: int = 0,
    force_reembed: bool = False,
) -> int:
    """
    Compute embeddings for nodes (:RAG_LABEL) that don't have EMBEDDING_PROPERTY (or force).
    Uses upsert_vectors() from neo4j-graphrag for efficient writes.

    max_nodes=0 => no limit (process all).
    Returns count embedded.
    """
    total = 0
    while True:
        where = "true" if force_reembed else f"n.`{EMBEDDING_PROPERTY}` IS NULL"
        cypher = f"""
        MATCH (n:`{RAG_LABEL}`)
        WHERE {where}
        WITH n
        RETURN
          elementId(n) AS eid,
          coalesce(n.Name, n.name, '') AS name,
          coalesce(n.Description, n.description, '') AS desc
        LIMIT $limit
        """
        rows = neo4j_run(driver, cypher, {"limit": batch_size})
        if not rows:
            break

        eids: List[str] = []
        texts: List[str] = []
        for r in rows:
            txt = make_text_for_embedding(r.get("name", ""), r.get("desc", ""))
            if not txt.strip():
                continue
            eids.append(r["eid"])
            texts.append(txt)

        if not eids:
            # nothing embeddable in this batch; stop to avoid looping forever
            break

        # Batch encode if possible (SentenceTransformerEmbeddings exposes .model)
        try:
            vecs = embedder.model.encode(texts)  # type: ignore[attr-defined]
            # numpy array or torch tensor -> tolist
            if hasattr(vecs, "tolist"):
                embeddings = vecs.tolist()
            else:
                embeddings = [embedder.embed_query(t) for t in texts]
        except Exception:
            embeddings = [embedder.embed_query(t) for t in texts]

        upsert_vectors(
            driver,
            ids=eids,
            embedding_property=EMBEDDING_PROPERTY,
            embeddings=embeddings,
            neo4j_database=NEO4J_DATABASE,
        )

        total += len(eids)
        if max_nodes and total >= max_nodes:
            break

    return total


# =============================================================================
# Retriever: Hybrid (vector+fulltext) + 1-hop expansion
# =============================================================================

class HybridNeighborhoodRetriever(Retriever):
    """
    Uses neo4j-graphrag HybridRetriever for recall, then expands each hit with 1-hop neighborhood.

    This solves the practical issue:
      "Graph has the node but retrieval returns no useful context"
    by always attaching local neighborhood context.
    """

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        hybrid: HybridRetriever,
        expand_k: int = EXPAND_K,
        neo4j_database: Optional[str] = None,
    ) -> None:
        super().__init__(driver=driver, neo4j_database=neo4j_database or NEO4J_DATABASE)
        self.hybrid = hybrid
        self.expand_k = expand_k

    def _expand_neighbors(self, eids: List[str]) -> Dict[str, dict]:
        cypher = f"""
        UNWIND $eids AS eid
        MATCH (n) WHERE elementId(n) = eid
        OPTIONAL MATCH (n)-[r]-(m)
        WITH eid, n, collect(DISTINCT {{
          rel_type: type(r),
          m_eid: elementId(m),
          m_labels: labels(m),
          m_name: coalesce(m.Name, m.name, ''),
          m_desc: coalesce(m.Description, m.description, '')
        }})[0..$k] AS neighbors
        RETURN
          eid,
          labels(n) AS n_labels,
          coalesce(n.Name, n.name, '') AS n_name,
          coalesce(n.Description, n.description, '') AS n_desc,
          neighbors
        """
        rows = neo4j_run(self.driver, cypher, {"eids": eids, "k": int(self.expand_k)})
        return {r["eid"]: r for r in rows}

    def get_search_results(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RawSearchResult:
        # delegate to official HybridRetriever
        raw = self.hybrid.get_search_results(query_text=query_text, top_k=top_k)
        return raw

    def search(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RetrieverResult:
        raw = self.hybrid.get_search_results(query_text=query_text, top_k=top_k)

        # Extract elementIds
        hits: List[Tuple[str, float]] = []
        for rec in raw.records:
            node = rec.get("node")
            score = rec.get("score")
            # neo4j Node uses .element_id in v5+
            eid = getattr(node, "element_id", None)
            if eid is None:
                # fallback: try read from record
                eid = rec.get("id") or rec.get("eid")
            if not isinstance(eid, str):
                continue
            hits.append((eid, float(score) if score is not None else 0.0))

        if not hits:
            return RetrieverResult(items=[], metadata={"__retriever": "HybridNeighborhoodRetriever", "raw_metadata": raw.metadata})

        expansions = self._expand_neighbors([h[0] for h in hits])

        items: List[RetrieverResultItem] = []
        for eid, score in hits:
            exp = expansions.get(eid, {})
            name = exp.get("n_name", "")
            desc = exp.get("n_desc", "")
            labels = exp.get("n_labels", [])
            neighbors = exp.get("neighbors", []) or []

            lines = []
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
                    nb_desc = (nb_desc[:300] + "...") if isinstance(nb_desc, str) and len(nb_desc) > 300 else nb_desc
                    lines.append(f"- {rel} -> (labels={nb_labels}) {nb_name} | {nb_desc}")

            items.append(
                RetrieverResultItem(
                    content="\n".join(lines),
                    metadata={"score": score, "eid": eid, "name": name, "retriever": "HybridNeighborhoodRetriever"},
                )
            )

        return RetrieverResult(
            items=items,
            metadata={"__retriever": "HybridNeighborhoodRetriever", "raw_metadata": raw.metadata},
        )


# =============================================================================
# Robust Text2Cypher fallback (read-only + retry)
# =============================================================================

@dataclass
class Text2CypherAttempt:
    attempt: int
    prompt: str
    cypher: str
    error: Optional[str] = None


class RobustText2CypherRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        llm: LLMInterfaceV2,
        neo4j_schema: Optional[str] = None,
        max_attempts: int = T2C_MAX_ATTEMPTS,
        limit: int = T2C_LIMIT,
        neo4j_database: Optional[str] = None,
    ):
        super().__init__(driver=driver, neo4j_database=neo4j_database or NEO4J_DATABASE)
        self.llm = llm
        self.max_attempts = max_attempts
        self.limit = limit

        if neo4j_schema is None:
            try:
                neo4j_schema = get_schema(driver)
            except Exception:
                neo4j_schema = "(schema unavailable)"
        self.neo4j_schema = neo4j_schema

        self.prompt_template = Text2CypherTemplate(
            template=r"""
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{schema}

IMPORTANT RULES (MUST FOLLOW):
1) Read-only ONLY. Do NOT use CREATE/MERGE/DELETE/SET/DROP/REMOVE/LOAD CSV/CALL apoc.*.
2) Always return a SMALL result set: include "LIMIT {limit}" (or smaller).
3) Do NOT use "(:Label1|Label2)" in MATCH node patterns. Neo4j MATCH does not support label OR with '|'.
   If you must match multiple labels, use:
   - MATCH (n) WHERE n:Label1 OR n:Label2 ...
   - or UNION.
4) Prefer using properties that exist in schema. Use coalesce() to handle missing properties safely.

Input:
{query_text}

Return ONLY the Cypher statement, without triple backticks or any other text.
Cypher query:
Examples (may be empty):
{examples}

""".strip(),
            expected_inputs=["schema", "query_text", "limit"],
        )

    def _build_prompt(self, query_text: str, extra: str = "") -> str:
        base = self.prompt_template.format(schema=self.neo4j_schema, query_text=query_text, limit=str(self.limit))
        return base + ("\n\n" + extra.strip() if extra else "")

    def _generate_cypher(self, prompt: str) -> str:
        cypher = self.llm.invoke(prompt).content.strip()
        cypher = ensure_limit(cypher, self.limit)
        ok, msg = cypher_is_safe_readonly(cypher)
        if not ok:
            raise ValueError(msg)
        if cypher_has_invalid_label_pipe(cypher):
            raise ValueError("Invalid label OR syntax detected (use WHERE ... OR ... instead of :A|B in MATCH).")
        return cypher

    def get_search_results(self, query_text: str) -> RawSearchResult:
        attempts: List[Text2CypherAttempt] = []
        last_error: Optional[str] = None

        for i in range(1, self.max_attempts + 1):
            extra = ""
            if last_error:
                extra = (
                    "The previous Cypher failed. Fix the query.\n"
                    f"Error:\n{last_error}\n"
                    "Return ONLY the corrected Cypher.\n"
                )
            prompt = self._build_prompt(query_text, extra=extra)
            print(f"[query-gen] text2cypher attempt {i}: generating Cypher...")
            try:
                cypher = self._generate_cypher(prompt)
                print("[query-gen] generated Cypher:")
                print(cypher)
                print_sep("-")
                records, _, _ = self.driver.execute_query(
                    cypher,
                    {},
                    database_=self.neo4j_database,
                    routing_=neo4j.RoutingControl.READ,
                )
                attempts.append(Text2CypherAttempt(attempt=i, prompt=prompt, cypher=cypher))
                return RawSearchResult(
                    records=records,
                    metadata={"mode": "text2cypher", "cypher": cypher, "attempts": [a.__dict__ for a in attempts]},
                )
            except (CypherSyntaxError, Neo4jError, ValueError) as e:
                err_msg = getattr(e, "message", None) or str(e)
                print(f"[query-gen] text2cypher attempt {i} failed: {err_msg}")
                print_sep("-")
                last_error = err_msg
                attempts.append(Text2CypherAttempt(attempt=i, prompt=prompt, cypher="", error=last_error))
                continue

        return RawSearchResult(
            records=[],
            metadata={"mode": "text2cypher", "error": last_error, "attempts": [a.__dict__ for a in attempts]},
        )

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(content=json.dumps(dict(record), ensure_ascii=False), metadata={"retriever": "RobustText2CypherRetriever"})


# =============================================================================
# Safe Fulltext fallback (handles Lucene special chars better for entity names)
# =============================================================================

class SafeFulltextNeighborhoodRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver, fulltext_index_name: str, expand_k: int = EXPAND_K):
        super().__init__(driver=driver, neo4j_database=NEO4J_DATABASE)
        self.fulltext_index_name = fulltext_index_name
        self.expand_k = expand_k

    def get_search_results(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RawSearchResult:
        terms = extract_terms(query_text)
        if not terms:
            return RawSearchResult(records=[], metadata={"mode": "safe_fulltext", "terms": []})
        lucene = " OR ".join([f'"{lucene_escape(t)}"' for t in terms])
        print(f"[query-gen] fulltext terms: {terms}")
        print(f"[query-gen] fulltext lucene: {lucene}")
        print_sep("-")
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT $top_k
        OPTIONAL MATCH (node)-[r]-(m)
        WITH node, score, collect(DISTINCT {{
          rel_type: type(r),
          m_labels: labels(m),
          m_name: coalesce(m.Name, m.name, ''),
          m_desc: coalesce(m.Description, m.description, '')
        }})[0..$k] AS neighbors
        RETURN
          elementId(node) AS eid,
          labels(node) AS labels,
          coalesce(node.Name, node.name, '') AS name,
          coalesce(node.Description, node.description, '') AS desc,
          score AS score,
          neighbors
        ORDER BY score DESC
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            {"index": self.fulltext_index_name, "q": lucene, "top_k": top_k, "k": int(self.expand_k)},
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        return RawSearchResult(records=records, metadata={"mode": "safe_fulltext", "terms": terms, "lucene": lucene})

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        eid = record.get("eid")
        labels = record.get("labels")
        name = record.get("name")
        desc = record.get("desc")
        score = record.get("score")
        neighbors = record.get("neighbors") or []
        lines = []
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
                nb_desc = (nb_desc[:300] + "...") if isinstance(nb_desc, str) and len(nb_desc) > 300 else nb_desc
                lines.append(f"- {rel} -> (labels={nb_labels}) {nb_name} | {nb_desc}")
        return RetrieverResultItem(
            content="\n".join(lines),
            metadata={"score": score, "eid": eid, "node_name": name, "retriever": "SafeFulltextNeighborhoodRetriever"},
        )


# =============================================================================
# Composite Retriever: Hybrid -> Fulltext (if needed) -> Text2Cypher
# =============================================================================

class CompositeRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        hybrid_nb: Optional[HybridNeighborhoodRetriever],
        safe_ft: SafeFulltextNeighborhoodRetriever,
        t2c: RobustText2CypherRetriever,
    ):
        super().__init__(driver=driver, neo4j_database=NEO4J_DATABASE)
        self.hybrid_nb = hybrid_nb
        self.safe_ft = safe_ft
        self.t2c = t2c

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        return RawSearchResult(records=[], metadata={})

    def search(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RetrieverResult:
        meta: Dict[str, Any] = {"router": "hybrid->fulltext->t2c"}
        items: List[RetrieverResultItem] = []

        # 1) Try hybrid if available
        if self.hybrid_nb is not None:
            try:
                r1 = self.hybrid_nb.search(query_text=query_text, top_k=top_k)
                meta["hybrid"] = r1.metadata
                items.extend(r1.items)
            except Exception as e:
                meta["hybrid_error"] = str(e)

        # 2) If definition question or hybrid empty, try safe fulltext
        if looks_like_definition_question(query_text) or len(items) == 0:
            r2 = self.safe_ft.search(query_text=query_text, top_k=top_k)
            meta["fulltext"] = r2.metadata
            # merge (dedupe by content)
            existing = {it.content for it in items}
            for it in r2.items:
                if it.content not in existing:
                    items.append(it)
                    existing.add(it.content)

        # 3) If still empty, fallback to text2cypher
        if len(items) == 0:
            r3 = self.t2c.search(query_text=query_text)
            meta["t2c"] = r3.metadata
            items.extend(r3.items)

        # Renumber for citations: C1, C2, ...
        numbered: List[RetrieverResultItem] = []
        for i, it in enumerate(items, start=1):
            numbered.append(RetrieverResultItem(content=f"C{i}:\n{it.content}", metadata=it.metadata))

        return RetrieverResult(items=numbered, metadata=meta)


# =============================================================================
# Answer prompt (grounded)
# =============================================================================

ANSWER_PROMPT = RagTemplate(
    template=r"""
你是一个严谨的助手。只能使用下面提供的 Context 来回答问题：
- 如果 Context 信息不足，请明确说“资料不足，无法确定”，并说明缺了什么。
- 每当你引用 Context 中的事实，都要用 [C1] [C2] 这样的引用标注。
- 不要编造任何 Context 中不存在的细节。


Examples (may be empty):
{examples}

Context:
{context}

Question:
{query_text}

Answer:
""".strip(),
    expected_inputs=["context", "query_text", "examples"],
    system_instructions="You are a careful assistant that strictly grounds answers in the provided context.",
)


# =============================================================================
# App builder
# =============================================================================

def make_embedder() -> SentenceTransformerEmbeddings:
    """
    Create SentenceTransformer embedder with a reasonable device choice.
    """
    kwargs: Dict[str, Any] = {}
    if EMBED_DEVICE in ("cpu", "cuda", "mps"):
        kwargs["device"] = EMBED_DEVICE
    # If auto: let sentence-transformers decide
    return SentenceTransformerEmbeddings(model=EMBED_MODEL, **kwargs)


def build_app(driver: neo4j.Driver, enable_hybrid: bool = True) -> GraphRAG:
    llm = QwenLLM(model_name="qwen-local")

    # Schema for Text2Cypher prompt
    try:
        schema = get_schema(driver)
    except Exception:
        schema = None

    t2c = RobustText2CypherRetriever(driver=driver, llm=llm, neo4j_schema=schema)

    # Safe fulltext (works even without vectors)
    safe_ft = SafeFulltextNeighborhoodRetriever(driver, FULLTEXT_INDEX_NAME, expand_k=EXPAND_K)

    hybrid_nb: Optional[HybridNeighborhoodRetriever] = None
    if enable_hybrid and index_exists(driver, VECTOR_INDEX_NAME) and index_exists(driver, FULLTEXT_INDEX_NAME):
        try:
            embedder = make_embedder()
            hybrid = HybridRetriever(
                driver,
                VECTOR_INDEX_NAME,
                FULLTEXT_INDEX_NAME,
                embedder=embedder,
                neo4j_database=NEO4J_DATABASE,
            )
            hybrid_nb = HybridNeighborhoodRetriever(driver=driver, hybrid=hybrid, expand_k=EXPAND_K, neo4j_database=NEO4J_DATABASE)
        except Exception as e:
            # If embeddings stack isn't installed, degrade gracefully
            print(f"[warn] Hybrid retriever disabled: {e}")

    retriever = CompositeRetriever(driver=driver, hybrid_nb=hybrid_nb, safe_ft=safe_ft, t2c=t2c)
    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=ANSWER_PROMPT)
    rag._neo4j_driver = driver  # type: ignore[attr-defined]
    return rag


# =============================================================================
# CLI
# =============================================================================

def cmd_build_index(args: argparse.Namespace) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Fail fast on bad credentials / unreachable Neo4j
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        embedder = make_embedder()
        dim = build_indexes(driver, embedder, similarity=args.similarity)
        print(f"[ok] Indexes ready. Embedding dimension = {dim}")

        embedded = embed_missing_nodes(
            driver,
            embedder,
            batch_size=args.batch_size,
            max_nodes=args.max_nodes,
            force_reembed=args.force_reembed,
        )
        print(f"[ok] Embedded nodes updated: {embedded}")

    finally:
        driver.close()


def cmd_chat(args: argparse.Namespace) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Fail fast on bad credentials / unreachable Neo4j
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        # Ensure fulltext index at least (cheap); vector index/embeddings are optional
        try:
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
        except Exception as e:
            print(f"[warn] Could not ensure fulltext index: {e}")

        if not index_exists(driver, VECTOR_INDEX_NAME):
            print("[info] Vector index not found. Hybrid retrieval will be disabled until you run:")
            print("       python graphrag_app.py --build-index")

        rag = build_app(driver, enable_hybrid=not args.no_hybrid)

        print("Neo4j GraphRAG v2 (Hybrid + Neighborhood) started.")
        print(f"Neo4j: {NEO4J_URI} (db={NEO4J_DATABASE})")
        print(f"Indexes: vector={VECTOR_INDEX_NAME} fulltext={FULLTEXT_INDEX_NAME}")
        print("Type your question. Type 'exit' to quit.\n")

        while True:
            q = input("Q> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break

            # --- Retrieve first: print references BEFORE summarizing ---
            retriever_result = rag.retriever.search(query_text=q, top_k=TOP_K)
            if not retriever_result.items:
                print("\nA>")
                print("资料不足，无法确定（检索上下文为空）。")
                print()
                continue

            print("\n[References]")
            for item in retriever_result.items:
                print(item.content)
                print()
            print_sep('=')

            context = "\n\n".join([it.content for it in retriever_result.items])
            prompt = ANSWER_PROMPT.format(context=context, query_text=q, examples="")

            print("\nA>")
            # Stream thoughts (qwen prints NO separators). Separators are printed HERE in main.
            raw = rag.llm.get_response(
                messages=rag.llm._to_messages(prompt, system_instruction=ANSWER_PROMPT.system_instructions),
                think=True,
                print_type="stream",
            )
            answer = raw.split("</think>")[-1].strip()
            print_sep('=')
            print(answer)
            print()

            if args.debug:
                print("\n--- Retrieval Debug ---")
                print(json.dumps(retriever_result.metadata or {}, ensure_ascii=False, indent=2))
                print("\n--- Context (top) ---")
                for item in (retriever_result.items or [])[: min(6, len(retriever_result.items))]:
                    print(item.content)
                    print()

            print()

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo4j GraphRAG v2 (Hybrid + Neighborhood)")
    parser.add_argument("--build-index", action="store_true", help="Create indexes and populate embeddings (writes).")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size for --build-index.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Max nodes to embed for --build-index (0 = all).")
    parser.add_argument("--force-reembed", action="store_true", help="Re-embed even if embedding exists.")
    parser.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine", help="Vector index similarity function.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval even if indexes exist.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval metadata and context snippets.")
    args = parser.parse_args()

    if args.build_index:
        cmd_build_index(args)
    else:
        cmd_chat(args)


if __name__ == "__main__":
    main()
