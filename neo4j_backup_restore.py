from __future__ import annotations

import lzma
import os
import pickle
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from constants import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from neo4j import GraphDatabase

BACKUP_FILE = SCRIPT_DIR / "data.bak"
TMP_BACKUP_FILE = SCRIPT_DIR / "data.bak.tmp"
BACKUP_FORMAT_VERSION = 1
BACKUP_NODE_BLOCK_SIZE = 2000
BACKUP_REL_BLOCK_SIZE = 5000
DELETE_BATCH_SIZE = 20000
SQLITE_IN_CLAUSE_CHUNK = 900
PROGRESS_INTERVAL_SEC = 1.0
LZMA_PRESET = 9 | lzma.PRESET_EXTREME
PICKLE_PROTOCOL = 5
FETCH_SIZE = 1000
INDEX_ONLINE_POLL_INTERVAL_SEC = 1.0


def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def format_count(n: int | float) -> str:
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.2f}"
    return f"{int(n):,}"


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    whole = int(seconds + 0.5)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Progress:
    def __init__(self, phase: str, total: int | None, unit: str):
        self.phase = phase
        self.total = total
        self.unit = unit
        self.done = 0
        self.start = time.monotonic()
        self.last_print = 0.0

    def advance(self, delta: int) -> None:
        self.done += delta
        self.maybe_print()

    def maybe_print(self, force: bool = False, extra: str = "") -> None:
        now = time.monotonic()
        if not force and now - self.last_print < PROGRESS_INTERVAL_SEC:
            return
        self.last_print = now
        elapsed = now - self.start
        rate = (self.done / elapsed) if elapsed > 0 else 0.0
        rate_str = f"{rate:,.1f}{self.unit}/s" if rate > 0 else f"0/{self.unit}s"
        if self.total is None:
            msg = f"[{self.phase}] 已处理 {format_count(self.done)} {self.unit}, 速度 {rate_str}, 已耗时 {format_duration(elapsed)}"
        else:
            ratio = (self.done / self.total) if self.total else 1.0
            ratio = min(max(ratio, 0.0), 1.0)
            pct = ratio * 100.0
            eta = ((self.total - self.done) / rate) if rate > 0 and self.done < self.total else 0.0
            msg = (
                f"[{self.phase}] {format_count(self.done)}/{format_count(self.total)} {self.unit} "
                f"({pct:6.2f}%), ETA {format_duration(eta)}, 速度 {rate_str}, 已耗时 {format_duration(elapsed)}"
            )
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)

    def finish(self, extra: str = "") -> None:
        self.maybe_print(force=True, extra=extra)


def print_stage(title: str) -> None:
    line = "=" * 24
    print(f"\n{line} {title} {line}", flush=True)


@contextmanager
def temp_sqlite_map() -> Iterator[sqlite3.Connection]:
    fd, path = tempfile.mkstemp(prefix="neo4j_restore_map_", suffix=".sqlite3")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -200000")
        conn.execute(
            "CREATE TABLE node_map (old_eid TEXT PRIMARY KEY, new_eid TEXT NOT NULL) WITHOUT ROWID"
        )
        yield conn
    finally:
        conn.close()
        try:
            os.remove(path)
        except OSError:
            pass


def make_driver():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
    )
    driver.verify_connectivity()
    return driver


def scalar_from_result(result, key: str = "value") -> Any:
    record = result.single()
    if record is None:
        raise RuntimeError("查询未返回结果")
    return record[key]


def iter_chunks(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def fetch_schema(tx) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    constraints = []
    result = tx.run(
        """
        SHOW CONSTRAINTS
        YIELD name, createStatement
        WHERE createStatement IS NOT NULL
        RETURN name, createStatement
        ORDER BY name
        """
    )
    for record in result:
        constraints.append({"name": record["name"], "create": record["createStatement"]})

    indexes = []
    result = tx.run(
        """
        SHOW INDEXES
        YIELD name, createStatement, owningConstraint
        WHERE createStatement IS NOT NULL AND owningConstraint IS NULL
        RETURN name, createStatement
        ORDER BY name
        """
    )
    for record in result:
        indexes.append({"name": record["name"], "create": record["createStatement"]})

    return constraints, indexes


def write_pickle(stream, obj: Any) -> None:
    pickle.dump(obj, stream, protocol=PICKLE_PROTOCOL)


def load_pickle(stream) -> Any:
    return pickle.load(stream)


def backup_database() -> None:
    start_ts = time.monotonic()
    if TMP_BACKUP_FILE.exists():
        TMP_BACKUP_FILE.unlink()

    try:
        print_stage("连接 Neo4j")
        with make_driver() as driver:
            print(f"已连接: {NEO4J_URI} / db={NEO4J_DATABASE}", flush=True)

            print_stage("读取数据库概况与 Schema")
            with driver.session(database=NEO4J_DATABASE, fetch_size=FETCH_SIZE) as session:
                tx = session.begin_transaction()
                try:
                    node_total = scalar_from_result(
                        tx.run("MATCH (n) RETURN count(n) AS value")
                    )
                    rel_total = scalar_from_result(
                        tx.run("MATCH ()-[r]->() RETURN count(r) AS value")
                    )
                    constraints, indexes = fetch_schema(tx)

                    meta = {
                        "kind": "meta",
                        "version": BACKUP_FORMAT_VERSION,
                        "format": "lzma+pickle-stream",
                        "created_at": now_local_str(),
                        "database": NEO4J_DATABASE,
                        "uri": NEO4J_URI,
                        "node_total": int(node_total),
                        "relationship_total": int(rel_total),
                        "constraints": constraints,
                        "indexes": indexes,
                    }

                    print(
                        "概况: "
                        f"节点 {format_count(node_total)}, 关系 {format_count(rel_total)}, "
                        f"约束 {format_count(len(constraints))}, 索引 {format_count(len(indexes))}",
                        flush=True,
                    )

                    print_stage(f"写入备份文件 -> {BACKUP_FILE}")
                    with lzma.open(TMP_BACKUP_FILE, "wb", preset=LZMA_PRESET) as out:
                        write_pickle(out, meta)

                        node_progress = Progress("备份节点", int(node_total), "nodes")
                        result = tx.run(
                            "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
                        )
                        block: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
                        for record in result:
                            block.append(
                                (
                                    record["eid"],
                                    tuple(sorted(record["labels"])),
                                    record["props"],
                                )
                            )
                            node_progress.advance(1)
                            if len(block) >= BACKUP_NODE_BLOCK_SIZE:
                                write_pickle(out, {"kind": "nodes", "rows": block})
                                block = []
                        if block:
                            write_pickle(out, {"kind": "nodes", "rows": block})
                        node_progress.finish()

                        rel_progress = Progress("备份关系", int(rel_total), "rels")
                        result = tx.run(
                            """
                            MATCH ()-[r]->()
                            RETURN elementId(startNode(r)) AS start_eid,
                                   elementId(endNode(r)) AS end_eid,
                                   type(r) AS rel_type,
                                   properties(r) AS props
                            """
                        )
                        block2: list[tuple[str, str, str, dict[str, Any]]] = []
                        for record in result:
                            block2.append(
                                (
                                    record["start_eid"],
                                    record["end_eid"],
                                    record["rel_type"],
                                    record["props"],
                                )
                            )
                            rel_progress.advance(1)
                            if len(block2) >= BACKUP_REL_BLOCK_SIZE:
                                write_pickle(out, {"kind": "rels", "rows": block2})
                                block2 = []
                        if block2:
                            write_pickle(out, {"kind": "rels", "rows": block2})
                        rel_progress.finish()

                        write_pickle(
                            out,
                            {
                                "kind": "end",
                                "node_total": int(node_total),
                                "relationship_total": int(rel_total),
                            },
                        )
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise

        TMP_BACKUP_FILE.replace(BACKUP_FILE)
        elapsed = time.monotonic() - start_ts
        size = BACKUP_FILE.stat().st_size if BACKUP_FILE.exists() else 0
        print_stage("备份完成")
        print(f"文件: {BACKUP_FILE}", flush=True)
        print(f"大小: {format_bytes(size)}", flush=True)
        print(f"总耗时: {format_duration(elapsed)}", flush=True)
    except Exception:
        if TMP_BACKUP_FILE.exists():
            try:
                TMP_BACKUP_FILE.unlink()
            except OSError:
                pass
        raise


def load_backup_meta() -> dict[str, Any]:
    if not BACKUP_FILE.exists():
        raise FileNotFoundError(f"未找到备份文件: {BACKUP_FILE}")
    with lzma.open(BACKUP_FILE, "rb") as fp:
        meta = load_pickle(fp)
    if not isinstance(meta, dict) or meta.get("kind") != "meta":
        raise RuntimeError("备份文件格式错误: 缺少 meta 头")
    if meta.get("version") != BACKUP_FORMAT_VERSION:
        raise RuntimeError(
            f"备份版本不匹配: 文件版本={meta.get('version')} 当前程序版本={BACKUP_FORMAT_VERSION}"
        )
    return meta


def list_existing_indexes(session) -> list[str]:
    result = session.run(
        """
        SHOW INDEXES
        YIELD name, owningConstraint
        WHERE owningConstraint IS NULL
        RETURN name
        ORDER BY name
        """
    )
    return [record["name"] for record in result]


def list_existing_constraints(session) -> list[str]:
    result = session.run(
        "SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name"
    )
    return [record["name"] for record in result]


def quote_name(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def drop_existing_schema(session) -> None:
    indexes = list_existing_indexes(session)
    constraints = list_existing_constraints(session)

    print_stage("清理现有 Schema")
    idx_progress = Progress("删除现有索引", len(indexes), "indexes")
    for name in indexes:
        session.run(f"DROP INDEX {quote_name(name)} IF EXISTS").consume()
        idx_progress.advance(1)
    idx_progress.finish()

    cons_progress = Progress("删除现有约束", len(constraints), "constraints")
    for name in constraints:
        session.run(f"DROP CONSTRAINT {quote_name(name)} IF EXISTS").consume()
        cons_progress.advance(1)
    cons_progress.finish()


def delete_relationships_in_batches(session) -> None:
    total = int(
        scalar_from_result(
            session.run("MATCH ()-[r]->() RETURN count(r) AS value")
        )
    )
    progress = Progress("清空现有关系", total, "rels")
    while True:
        deleted = int(
            scalar_from_result(
                session.run(
                    "MATCH ()-[r]->() WITH r LIMIT $limit DELETE r RETURN count(*) AS value",
                    limit=DELETE_BATCH_SIZE,
                )
            )
        )
        if deleted == 0:
            break
        progress.advance(deleted)
    progress.finish()


def delete_nodes_in_batches(session) -> None:
    total = int(scalar_from_result(session.run("MATCH (n) RETURN count(n) AS value")))
    progress = Progress("清空现有节点", total, "nodes")
    while True:
        deleted = int(
            scalar_from_result(
                session.run(
                    "MATCH (n) WITH n LIMIT $limit DELETE n RETURN count(*) AS value",
                    limit=DELETE_BATCH_SIZE,
                )
            )
        )
        if deleted == 0:
            break
        progress.advance(deleted)
    progress.finish()


def fetch_mapping_many(conn: sqlite3.Connection, old_eids: Iterable[str]) -> dict[str, str]:
    keys = list(dict.fromkeys(old_eids))
    if not keys:
        return {}
    out: dict[str, str] = {}
    for chunk in iter_chunks(keys, SQLITE_IN_CLAUSE_CHUNK):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"SELECT old_eid, new_eid FROM node_map WHERE old_eid IN ({placeholders})"
        rows = conn.execute(sql, chunk).fetchall()
        for old_eid, new_eid in rows:
            out[old_eid] = new_eid
    return out


def restore_node_rows(session, conn: sqlite3.Connection, rows: list[tuple[str, tuple[str, ...], dict[str, Any]]]) -> int:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for old_eid, labels, props in rows:
        grouped[tuple(labels)].append({"old_eid": old_eid, "props": props})

    restored = 0
    for labels, group_rows in grouped.items():
        if labels:
            result = session.run(
                """
                UNWIND $rows AS row
                CREATE (n:$($labels))
                SET n = row.props
                RETURN row.old_eid AS old_eid, elementId(n) AS new_eid
                """,
                rows=group_rows,
                labels=list(labels),
            )
        else:
            result = session.run(
                """
                UNWIND $rows AS row
                CREATE (n)
                SET n = row.props
                RETURN row.old_eid AS old_eid, elementId(n) AS new_eid
                """,
                rows=group_rows,
            )
        mappings = [(record["old_eid"], record["new_eid"]) for record in result]
        if len(mappings) != len(group_rows):
            raise RuntimeError("节点恢复结果数量异常")
        conn.executemany(
            "INSERT INTO node_map(old_eid, new_eid) VALUES (?, ?)", mappings
        )
        conn.commit()
        restored += len(group_rows)
    return restored


def restore_rel_rows(session, conn: sqlite3.Connection, rows: list[tuple[str, str, str, dict[str, Any]]]) -> int:
    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for start_old, end_old, rel_type, props in rows:
        grouped[rel_type].append((start_old, end_old, props))

    restored = 0
    for rel_type, rel_rows in grouped.items():
        needed = [old for item in rel_rows for old in item[:2]]
        mapping = fetch_mapping_many(conn, needed)
        payload = []
        for start_old, end_old, props in rel_rows:
            start_new = mapping.get(start_old)
            end_new = mapping.get(end_old)
            if start_new is None or end_new is None:
                raise RuntimeError(
                    f"关系恢复失败: 找不到端点映射 rel_type={rel_type} start={start_old} end={end_old}"
                )
            payload.append(
                {"start_eid": start_new, "end_eid": end_new, "props": props}
            )

        created = int(
            scalar_from_result(
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a) WHERE elementId(a) = row.start_eid
                    MATCH (b) WHERE elementId(b) = row.end_eid
                    CREATE (a)-[r:$($rel_type)]->(b)
                    SET r = row.props
                    RETURN count(*) AS value
                    """,
                    rows=payload,
                    rel_type=rel_type,
                )
            )
        )
        if created != len(payload):
            raise RuntimeError(
                f"关系恢复结果数量异常: 期望 {len(payload)}，实际 {created}"
            )
        restored += created
    return restored


def create_constraints(session, constraints: list[dict[str, str]]) -> None:
    progress = Progress("重建约束", len(constraints), "constraints")
    for item in constraints:
        session.run(item["create"]).consume()
        progress.advance(1)
    progress.finish()


def create_indexes(session, indexes: list[dict[str, str]]) -> None:
    progress = Progress("提交索引创建", len(indexes), "indexes")
    for item in indexes:
        session.run(item["create"]).consume()
        progress.advance(1)
    progress.finish()

    if not indexes:
        return

    names = [item["name"] for item in indexes]
    print_stage("等待索引 ONLINE")
    start = time.monotonic()
    last_print = 0.0
    while True:
        rows = list(
            session.run(
                """
                SHOW INDEXES
                YIELD name, state, populationPercent
                WHERE name IN $names
                RETURN name, state, populationPercent
                ORDER BY name
                """,
                names=names,
            )
        )
        if not rows:
            raise RuntimeError("索引创建后未能查询到任何索引状态")
        failed = [r for r in rows if r["state"] == "FAILED"]
        if failed:
            raise RuntimeError(
                "索引创建失败: " + ", ".join(r["name"] for r in failed)
            )
        online = sum(1 for r in rows if r["state"] == "ONLINE")
        avg_percent = sum(
            float(r["populationPercent"] or (100.0 if r["state"] == "ONLINE" else 0.0))
            for r in rows
        ) / len(rows)
        elapsed = time.monotonic() - start
        ratio = max(0.0, min(avg_percent / 100.0, 1.0))
        rate = ratio / elapsed if elapsed > 0 else 0.0
        eta = ((1.0 - ratio) / rate) if rate > 0 and ratio < 1.0 else 0.0
        now = time.monotonic()
        if now - last_print >= PROGRESS_INTERVAL_SEC or online == len(rows):
            last_print = now
            print(
                f"[等待索引 ONLINE] {online}/{len(rows)} 已上线, 总体 {avg_percent:6.2f}%, "
                f"ETA {format_duration(eta)}, 已耗时 {format_duration(elapsed)}",
                flush=True,
            )
        if online == len(rows):
            break
        time.sleep(INDEX_ONLINE_POLL_INTERVAL_SEC)


def restore_database() -> None:
    start_ts = time.monotonic()
    meta = load_backup_meta()

    print_stage(f"读取备份文件 -> {BACKUP_FILE}")
    print(
        "备份信息: "
        f"节点 {format_count(meta['node_total'])}, 关系 {format_count(meta['relationship_total'])}, "
        f"约束 {format_count(len(meta['constraints']))}, 索引 {format_count(len(meta['indexes']))}, "
        f"创建时间 {meta['created_at']}",
        flush=True,
    )

    print_stage("连接 Neo4j")
    with make_driver() as driver, temp_sqlite_map() as map_conn:
        print(f"已连接: {NEO4J_URI} / db={NEO4J_DATABASE}", flush=True)
        with driver.session(database=NEO4J_DATABASE, fetch_size=FETCH_SIZE) as session:
            drop_existing_schema(session)
            delete_relationships_in_batches(session)
            delete_nodes_in_batches(session)

            print_stage("写回节点与关系")
            node_progress = Progress("恢复节点", int(meta["node_total"]), "nodes")
            rel_progress = Progress("恢复关系", int(meta["relationship_total"]), "rels")
            end_seen = False

            with lzma.open(BACKUP_FILE, "rb") as fp:
                head = load_pickle(fp)
                if head.get("kind") != "meta":
                    raise RuntimeError("备份文件格式错误: 头部不是 meta")

                while True:
                    try:
                        item = load_pickle(fp)
                    except EOFError:
                        break
                    kind = item.get("kind")
                    if kind == "nodes":
                        restored = restore_node_rows(session, map_conn, item["rows"])
                        node_progress.advance(restored)
                    elif kind == "rels":
                        restored = restore_rel_rows(session, map_conn, item["rows"])
                        rel_progress.advance(restored)
                    elif kind == "end":
                        end_seen = True
                        break
                    else:
                        raise RuntimeError(f"备份文件格式错误: 未知块类型 {kind!r}")

            node_progress.finish()
            rel_progress.finish()
            if not end_seen:
                raise RuntimeError("备份文件不完整: 缺少 end 块")

            print_stage("重建 Schema")
            create_constraints(session, meta["constraints"])
            create_indexes(session, meta["indexes"])

    elapsed = time.monotonic() - start_ts
    print_stage("恢复完成")
    print(f"总耗时: {format_duration(elapsed)}", flush=True)


def parse_action() -> str:
    if len(sys.argv) > 2:
        raise SystemExit("只支持 0 或 1 个动作参数: backup / restore")
    if len(sys.argv) == 2:
        raw = sys.argv[1].strip().lower()
    else:
        raw = input("请选择操作 [1=backup 2=restore] > ").strip().lower()

    mapping = {
        "1": "backup",
        "backup": "backup",
        "b": "backup",
        "备份": "backup",
        "2": "restore",
        "restore": "restore",
        "r": "restore",
        "还原": "restore",
    }
    action = mapping.get(raw)
    if not action:
        raise SystemExit("无法识别的操作，只支持 backup / restore")
    return action


def confirm_restore() -> None:
    prompt = (
        "警告: 还原前会先删除 Neo4j 当前数据库中的全部节点、关系、索引、约束。\n"
        "输入 YES 继续 > "
    )
    answer = input(prompt).strip()
    if answer != "YES":
        raise SystemExit("已取消还原")


def main() -> None:
    action = parse_action()
    if action == "backup":
        backup_database()
    elif action == "restore":
        confirm_restore()
        restore_database()
    else:
        raise SystemExit(f"未知动作: {action}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n执行失败: {type(exc).__name__}: {exc}", flush=True)
        raise
