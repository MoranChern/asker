# -*- coding: utf-8 -*-
"""
main.py

CLI entrypoint (local test main program).

Usage:
  python main.py
  python main.py --build-index

Updated behavior:
- Answering is orchestrated by agent.py.
- Retrieval is a TOOL; the agent decides if/when to use it, and provides keywords.
"""

from __future__ import annotations

import argparse

from agent import GraphSearchTool, run_agent_sync
from graphrag import (
    create_driver,
    ensure_fulltext_index,
    TOP_K,
    EXPAND_K,
    FULLTEXT_INDEX_NAME,
    make_embedder,
    build_indexes,
    embed_missing_nodes,
)


def cmd_build_index(args: argparse.Namespace) -> None:
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        embedder = make_embedder()
        print("[step] Building indexes (batched tagging + index creation)...", flush=True)
        dim = build_indexes(driver, embedder, similarity=args.similarity, verbose=True)
        print("[step] Index build finished.", flush=True)
        print(f"[ok] Indexes ready. Embedding dimension = {dim}")

        embedded = embed_missing_nodes(
            driver,
            embedder,
            batch_size=args.batch_size,
            max_nodes=args.max_nodes,
            force_reembed=args.force_reembed,
            verbose=True,
        )
        print(f"[ok] Embedded nodes updated: {embedded}")
    finally:
        driver.close()


def cmd_chat(args: argparse.Namespace) -> None:
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        # Best-effort ensure fulltext index exists (writes); retrieval can still work via fallback if missing.
        ensure_fulltext_index(driver)

        # Lazy import to avoid build-index needing llama_cpp
        from qwen import LLM_QWEN_Standalone  # local import

        tool = GraphSearchTool(
            driver=driver,
            top_k=int(TOP_K),
            expand_k=int(EXPAND_K),
            fulltext_index_name=str(FULLTEXT_INDEX_NAME),
        )

        print("Neo4j GraphRAG Agent started.")
        print("Type your question. Type 'exit' to quit.\n")

        with LLM_QWEN_Standalone(verbose=False) as llm:
            while True:
                q = input("Q> ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break

                run_agent_sync(
                    question=q,
                    llm_get_response=llm.get_response,
                    search_tool=tool,
                    max_search_rounds=3,
                    print_debug=bool(args.debug),
                )

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo4j GraphRAG Agent (tool-using retrieval)")
    parser.add_argument("--build-index", action="store_true", help="Create indexes and populate embeddings (writes).")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size for --build-index.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Max nodes to embed for --build-index (0 = all).")
    parser.add_argument("--force-reembed", action="store_true", help="Re-embed even if embedding exists.")
    parser.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine", help="Vector index similarity function.")
    parser.add_argument("--debug", action="store_true", help="Print agent debug info.")
    args = parser.parse_args()

    if args.build_index:
        cmd_build_index(args)
    else:
        cmd_chat(args)


if __name__ == "__main__":
    main()
