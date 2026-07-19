"""Build and query a local SQLite FTS5 index over digital-mirror records."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_index(records_path: Path, database_path: Path) -> dict[str, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE records (
                record_id TEXT NOT NULL UNIQUE,
                source_relative_path TEXT NOT NULL,
                locator TEXT NOT NULL,
                source_group TEXT NOT NULL,
                record_kind TEXT NOT NULL,
                author_role TEXT NOT NULL,
                occurred_at TEXT,
                text TEXT NOT NULL,
                allowed_uses_json TEXT NOT NULL,
                provenance_flags_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE records_fts USING fts5(
                text,
                content='records',
                content_rowid='rowid',
                tokenize='trigram'
            );
            """
        )
        count = 0
        for record in iter_jsonl(records_path):
            cursor = connection.execute(
                """
                INSERT INTO records (
                    record_id, source_relative_path, locator, source_group,
                    record_kind, author_role, occurred_at, text,
                    allowed_uses_json, provenance_flags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["record_id"],
                    record["source_relative_path"],
                    record["locator"],
                    record["source_group"],
                    record["record_kind"],
                    record["author_role"],
                    record["occurred_at"],
                    record["text"],
                    json.dumps(record["allowed_uses"], ensure_ascii=False),
                    json.dumps(record["provenance_flags"], ensure_ascii=False),
                ),
            )
            connection.execute(
                "INSERT INTO records_fts(rowid, text) VALUES (?, ?)",
                (cursor.lastrowid, record["text"]),
            )
            count += 1
        connection.commit()
        return {"record_count": count}
    finally:
        connection.close()


def fts_query(query: str) -> str:
    terms = [term for term in query.replace('"', " ").split() if len(term) >= 3]
    if not terms:
        raise ValueError("FTS trigram queries need at least one term of three characters")
    return " AND ".join(f'"{term}"' for term in terms)


def search(
    database_path: Path,
    query: str,
    limit: int = 10,
    author_role: str | None = None,
    source_group: str | None = None,
    allowed_use: str | None = None,
    before: str | None = None,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        where = ["records_fts MATCH ?"]
        parameters: list[Any] = [fts_query(query)]
        if author_role:
            where.append("r.author_role = ?")
            parameters.append(author_role)
        if source_group:
            where.append("r.source_group = ?")
            parameters.append(source_group)
        if allowed_use:
            where.append("EXISTS (SELECT 1 FROM json_each(r.allowed_uses_json) WHERE value = ?)")
            parameters.append(allowed_use)
        if before:
            # A strict historical replay cannot assume an undated document existed.
            where.append(
                "r.occurred_at IS NOT NULL AND datetime(r.occurred_at) < datetime(?)"
            )
            parameters.append(before)
        parameters.append(limit)
        rows = connection.execute(
            f"""
            SELECT
                r.record_id,
                r.source_relative_path,
                r.locator,
                r.source_group,
                r.record_kind,
                r.author_role,
                r.occurred_at,
                r.text,
                r.allowed_uses_json,
                bm25(records_fts) AS score
            FROM records_fts
            JOIN records r ON r.rowid = records_fts.rowid
            WHERE {' AND '.join(where)}
            ORDER BY score
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [
            {
                "record_id": row["record_id"],
                "source_relative_path": row["source_relative_path"],
                "locator": row["locator"],
                "source_group": row["source_group"],
                "record_kind": row["record_kind"],
                "author_role": row["author_role"],
                "occurred_at": row["occurred_at"],
                "text": row["text"],
                "allowed_uses": json.loads(row["allowed_uses_json"]),
                "score": row["score"],
            }
            for row in rows
        ]
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("records", type=Path)
    build_parser.add_argument("--database", type=Path, required=True)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--database", type=Path, required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--author-role")
    search_parser.add_argument("--source-group")
    search_parser.add_argument("--allowed-use")
    search_parser.add_argument("--before")
    args = parser.parse_args()

    if args.command == "build":
        print(json.dumps(build_index(args.records, args.database), indent=2))
        return
    results = search(
        args.database,
        args.query,
        limit=args.limit,
        author_role=args.author_role,
        source_group=args.source_group,
        allowed_use=args.allowed_use,
        before=args.before,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
