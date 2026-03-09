# -*- coding: utf-8 -*-
"""
index_manager.py

Standalone CLI for GraphRAG index maintenance.

Usage examples:
  python index_manager.py --build
  python index_manager.py --drop-all-indexes
  python index_manager.py --list-indexes

Notes:
- This file is intended to replace the old index-related entrypoint behavior that used to live in main.py.
- It can build indexes and embeddings, or drop all indexes in the configured Neo4j database.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any

from graphrag import (
    NEO4J_DATABASE,
    build_indexes,
    create_driver,
    drop_all_indexes,
    embed_missing_nodes,
    list_indexes,
    make_embedder,
)


def _print_index_table(indexes: List[Dict[str, Any]]) -> None:
    if not indexes:
        print("[info] No indexes found.")
        return

    for idx in indexes:
        name = idx.get("name") or ""
        typ = idx.get("type") or ""
        entity_type = idx.get("entityType") or ""
        labels_or_types = idx.get("labelsOrTypes") or []
        properties = idx.get("properties") or []
        state = idx.get("state") or ""
        owning_constraint = idx.get("owningConstraint") or ""
        print(
            f"- name={name} | type={typ} | entityType={entity_type} | "
            f"labelsOrTypes={labels_or_types} | properties={properties} | "
            f"state={state} | owningConstraint={owning_constraint}"
        )


def cmd_build(args: argparse.Namespace) -> None:
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        embedder = make_embedder()

        print(f"[step] Building indexes in database: {NEO4J_DATABASE}", flush=True)
        dim = build_indexes(
            driver,
            embedder,
            similarity=args.similarity,
            tag_batch_size=args.tag_batch_size,
            verbose=True,
        )
        print(f"[ok] Index build finished. Embedding dimension = {dim}", flush=True)

        embedded = embed_missing_nodes(
            driver,
            embedder,
            batch_size=args.batch_size,
            max_nodes=args.max_nodes,
            force_reembed=args.force_reembed,
            verbose=True,
        )
        print(f"[ok] Embedded nodes updated: {embedded}", flush=True)
    finally:
        driver.close()


def cmd_drop_all_indexes(args: argparse.Namespace) -> None:
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        before = list_indexes(driver)
        print(f"[step] Existing indexes in database: {NEO4J_DATABASE}", flush=True)
        _print_index_table(before)

        dropped = drop_all_indexes(
            driver,
            include_constraint_owned=args.include_constraint_owned,
            verbose=True,
        )
        print(f"[ok] Dropped indexes: {len(dropped)}", flush=True)
        for name in dropped:
            print(f"  - {name}")

        after = list_indexes(driver)
        print("[step] Remaining indexes:", flush=True)
        _print_index_table(after)
    finally:
        driver.close()


def cmd_list_indexes(args: argparse.Namespace) -> None:
    driver = create_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[error] Neo4j connection failed (check URI/user/password, and whether Neo4j is running): {e}")
        raise

    try:
        indexes = list_indexes(driver)
        print(f"[step] Current indexes in database: {NEO4J_DATABASE}", flush=True)
        _print_index_table(indexes)
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GraphRAG index manager")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="Create/update GraphRAG indexes and embeddings.")
    action.add_argument("--drop-all-indexes", action="store_true", help="Drop all indexes in the configured Neo4j database.")
    action.add_argument("--list-indexes", action="store_true", help="List all indexes in the configured Neo4j database.")

    parser.add_argument("--tag-batch-size", type=int, default=10000, help="Batch size when tagging RAG nodes during index build.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size for --build.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Max nodes to embed for --build (0 = all).")
    parser.add_argument("--force-reembed", action="store_true", help="Re-embed even if embedding already exists.")
    parser.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine", help="Vector index similarity function.")
    parser.add_argument(
        "--include-constraint-owned",
        action="store_true",
        help="Also drop indexes owned by constraints. Use with care.",
    )

    args = parser.parse_args()

    if args.build:
        cmd_build(args)
    elif args.drop_all_indexes:
        cmd_drop_all_indexes(args)
    else:
        cmd_list_indexes(args)


if __name__ == "__main__":
    main()
