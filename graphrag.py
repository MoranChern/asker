# -*- coding: utf-8 -*-
"""
graphrag.py

GraphRAG business logic extracted from the original graphrag_app.py.

Design goals:
- Keep GraphRAG / Neo4j retrieval, indexing, prompt templates here.
- Keep *application entrypoints* (CLI loop, web server, etc.) outside (see test.py, server.py).
- Avoid importing llama_cpp / GPU code at import-time:
  Qwen adapter performs a lazy import of qwen.py only when instantiated.

This file can be imported safely by server.py without triggering GPU usage,
as long as you do NOT instantiate QwenLLM / LLM_QWEN_Standalone.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase  # type: ignore
from neo4j.exceptions import CypherSyntaxError, Neo4jError  # type: ignore

# neo4j-graphrag core (lightweight; does not load local GPU model)
from neo4j_graphrag.generation.prompts import RagTemplate, Text2CypherTemplate
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

from neo4j_graphrag.llm.base import LLMInterfaceV2
from neo4j_graphrag.types import LLMMessage


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
# CONFIG (edit constants.py or via env vars)
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

FORBIDDEN_CYPHER_KEYWORDS = _cget(
    "FORBIDDEN_CYPHER_KEYWORDS",
    [
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "DROP",
        "REMOVE",
        "CALL apoc.",
        "LOAD CSV",
        "FOREACH",
        "GRANT",
        "REVOKE",
    ],
)

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
    stop = {
        "是什么",
        "什么",
        "解释",
        "介绍",
        "请问",
        "如何",
        "为什么",
        "怎么",
        "定义",
        "meaning",
        "define",
        "explain",
        "what",
    }
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


def create_driver() -> neo4j.Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# =============================================================================
# LLM adapter (Qwen via llama.cpp) for neo4j-graphrag
# =============================================================================

class QwenLLM(LLMInterfaceV2):
    """neo4j-graphrag LLM adapter.

    NOTE: This class performs a *lazy import* of qwen.py so that simply importing
    graphrag.py does NOT import llama_cpp or touch GPU.
    """

    supports_structured_output: bool = False

    def __init__(self, model_name: str = "qwen-local", model_params: Optional[dict] = None):
        super().__init__(model_name=model_name, model_params=model_params or {})
        # Lazy import to avoid server.py importing llama_cpp accidentally.
        from qwen import LLM_QWEN_Standalone  # local import

        self._qwen = LLM_QWEN_Standalone(verbose=False)

    def get_response(self, messages: List[Dict], think: bool, print_type: str) -> str:
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
    ):
        msgs = self._to_messages(input, message_history=message_history, system_instruction=system_instruction)
        out = self.get_response(
            messages=msgs,
            think=bool(self.model_params.get("think", True)),
            print_type=str(self.model_params.get("print_type", "stream")),
        )
        # neo4j-graphrag expects .content property in return type; use a tiny shim.
        from neo4j_graphrag.llm.types import LLMResponse  # local import

        return LLMResponse(content=out)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[List[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ):
        return self.invoke(input, message_history=message_history, system_instruction=system_instruction)


# =============================================================================
# Index building (writes)
# =============================================================================

TEXT_PROPS = ["Name", "Description", "name", "description"]


def tag_rag_nodes(driver: neo4j.Driver) -> int:
    """Tag nodes that have any text-ish property with :RAG_LABEL."""
    cypher = f"""
    MATCH (n)
    WHERE any(p IN $props WHERE n[p] IS NOT NULL)
    SET n:`{RAG_LABEL}`
    RETURN count(n) AS tagged
    """
    rows = neo4j_run(driver, cypher, {"props": TEXT_PROPS})
    return int(rows[0]["tagged"]) if rows else 0


def make_embedder():
    """Create SentenceTransformer embedder (lazy import)."""
    from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings  # local import

    kwargs: Dict[str, Any] = {}
    if EMBED_DEVICE in ("cpu", "cuda", "mps"):
        kwargs["device"] = EMBED_DEVICE
    return SentenceTransformerEmbeddings(model=EMBED_MODEL, **kwargs)


def build_indexes(driver: neo4j.Driver, embedder, similarity: str = "cosine") -> int:
    """Ensure :RAG_LABEL exists, create fulltext + vector index if needed. Returns embedding dim."""
    # Lazy imports to avoid heavy deps unless called
    from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index  # local import

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

    dim = len(embedder.embed_query("dimension probe"))

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

    return dim


def make_text_for_embedding(name: str, desc: str) -> str:
    name = (name or "").strip()
    desc = (desc or "").strip()
    if name and desc:
        return f"{name}\n{desc}"
    return name or desc


def embed_missing_nodes(
    driver: neo4j.Driver,
    embedder,
    batch_size: int = 128,
    max_nodes: int = 0,
    force_reembed: bool = False,
) -> int:
    """Compute embeddings for nodes that miss EMBEDDING_PROPERTY (or force)."""
    from neo4j_graphrag.indexes import upsert_vectors  # local import

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
            break

        try:
            vecs = embedder.model.encode(texts)  # type: ignore[attr-defined]
            embeddings = vecs.tolist() if hasattr(vecs, "tolist") else [embedder.embed_query(t) for t in texts]
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
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver, hybrid, expand_k: int = EXPAND_K, neo4j_database: Optional[str] = None) -> None:
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
          rel_eid: elementId(r),
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
        return self.hybrid.get_search_results(query_text=query_text, top_k=top_k)

    def search(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RetrieverResult:
        raw = self.hybrid.get_search_results(query_text=query_text, top_k=top_k)

        hits: List[Tuple[str, float]] = []
        for rec in raw.records:
            node = rec.get("node")
            score = rec.get("score")
            eid = getattr(node, "element_id", None)
            if eid is None:
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
                    metadata={
                        "score": score,
                        "eid": eid,
                        "name": name,
                        "labels": labels,
                        "desc": desc,
                        "neighbors": neighbors,
                        "retriever": "HybridNeighborhoodRetriever",
                    },
                )
            )

        return RetrieverResult(items=items, metadata={"__retriever": "HybridNeighborhoodRetriever", "raw_metadata": raw.metadata})


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
            try:
                cypher = self._generate_cypher(prompt)
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
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT $top_k
        OPTIONAL MATCH (node)-[r]-(m)
        WITH node, score, collect(DISTINCT {{
          rel_type: type(r),
          rel_eid: elementId(r),
          m_eid: elementId(m),
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
            metadata={
                "score": score,
                "eid": eid,
                "labels": labels,
                "name": name,
                "desc": desc,
                "neighbors": neighbors,
                "retriever": "SafeFulltextNeighborhoodRetriever",
            },
        )


# =============================================================================
# Composite Retriever: Hybrid -> Fulltext (if needed) -> Text2Cypher
# =============================================================================

class CompositeRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver, hybrid_nb: Optional[HybridNeighborhoodRetriever], safe_ft: SafeFulltextNeighborhoodRetriever, t2c: RobustText2CypherRetriever):
        super().__init__(driver=driver, neo4j_database=NEO4J_DATABASE)
        self.hybrid_nb = hybrid_nb
        self.safe_ft = safe_ft
        self.t2c = t2c

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        return RawSearchResult(records=[], metadata={})

    def search(self, query_text: str, top_k: int = TOP_K, **kwargs: Any) -> RetrieverResult:
        meta: Dict[str, Any] = {"router": "hybrid->fulltext->t2c"}
        items: List[RetrieverResultItem] = []

        if self.hybrid_nb is not None:
            try:
                r1 = self.hybrid_nb.search(query_text=query_text, top_k=top_k)
                meta["hybrid"] = r1.metadata
                items.extend(r1.items)
            except Exception as e:
                meta["hybrid_error"] = str(e)

        if looks_like_definition_question(query_text) or len(items) == 0:
            r2 = self.safe_ft.search(query_text=query_text, top_k=top_k)
            meta["fulltext"] = r2.metadata
            existing = {it.content for it in items}
            for it in r2.items:
                if it.content not in existing:
                    items.append(it)
                    existing.add(it.content)

        if len(items) == 0:
            r3 = self.t2c.search(query_text=query_text)
            meta["t2c"] = r3.metadata
            items.extend(r3.items)

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
# GraphRAG builder (CLI usage)
# =============================================================================

def build_retriever(driver: neo4j.Driver, enable_hybrid: bool = True) -> CompositeRetriever:
    """Build the CompositeRetriever used by CLI test.py."""
    llm = QwenLLM(model_name="qwen-local")

    try:
        schema = get_schema(driver)
    except Exception:
        schema = None

    t2c = RobustText2CypherRetriever(driver=driver, llm=llm, neo4j_schema=schema)

    safe_ft = SafeFulltextNeighborhoodRetriever(driver, FULLTEXT_INDEX_NAME, expand_k=EXPAND_K)

    hybrid_nb: Optional[HybridNeighborhoodRetriever] = None
    if enable_hybrid and index_exists(driver, VECTOR_INDEX_NAME) and index_exists(driver, FULLTEXT_INDEX_NAME):
        try:
            # Lazy import embedder + official hybrid retriever
            from neo4j_graphrag.retrievers import HybridRetriever  # local import

            embedder = make_embedder()
            hybrid = HybridRetriever(
                driver,
                VECTOR_INDEX_NAME,
                FULLTEXT_INDEX_NAME,
                embedder=embedder,
                neo4j_database=NEO4J_DATABASE,
            )
            hybrid_nb = HybridNeighborhoodRetriever(driver=driver, hybrid=hybrid, expand_k=EXPAND_K, neo4j_database=NEO4J_DATABASE)
        except Exception:
            # Degrade gracefully if embeddings stack isn't installed
            hybrid_nb = None

    return CompositeRetriever(driver=driver, hybrid_nb=hybrid_nb, safe_ft=safe_ft, t2c=t2c)


def build_prompt(context: str, question: str) -> str:
    return ANSWER_PROMPT.format(context=context, query_text=question, examples="")


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
        # Keep silent; CLI can still run with text2cypher fallback
        pass
