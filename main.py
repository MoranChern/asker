# -*- coding: utf-8 -*-
"""
main.py

CLI entrypoint (main program logic) extracted from the original graphrag_app.py.

Usage:
  python main.py
  python main.py --build-index
"""

from __future__ import annotations

import argparse
import json

from graphrag import (
    create_driver,
    ensure_fulltext_index,
    build_retriever,
    build_prompt,
    ANSWER_PROMPT,
    TOP_K,
    VECTOR_INDEX_NAME,
    FULLTEXT_INDEX_NAME,
    index_exists,
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
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        ensure_fulltext_index(driver)

        if not index_exists(driver, VECTOR_INDEX_NAME):
            print("[info] Vector index not found. Hybrid retrieval will be disabled until you run:")
            print("       python main.py --build-index")

        retriever = build_retriever(driver, enable_hybrid=not args.no_hybrid)

        print("Neo4j GraphRAG v2 (Hybrid + Neighborhood) started.")
        print("Type your question. Type 'exit' to quit.\n")

        while True:
            q = input("Q> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break

            retriever_result = retriever.search(query_text=q, top_k=TOP_K)
            if not retriever_result.items:
                print("\nA>")
                print("资料不足，无法确定（检索上下文为空）。")
                print()
                continue

            print("\n[References]")
            for item in retriever_result.items:
                print(item.content)
                print()
            print("=" * 80)

            context = "\n\n".join([it.content for it in retriever_result.items])
            prompt = build_prompt(context=context, question=q)

            print("\nA>")
            # Use the LLM adapter behind retriever (Text2Cypher uses llm too).
            # Here we just call it directly via QwenLLM streaming behavior.
            llm = retriever.t2c.llm  # type: ignore[attr-defined]
            raw = llm.get_response(
                messages=llm._to_messages(prompt, system_instruction=ANSWER_PROMPT.system_instructions),  # type: ignore[attr-defined]
                think=True,
                print_type="stream",
            )

            answer = raw.split("</think>")[-1].strip()
            print("=" * 80)
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
