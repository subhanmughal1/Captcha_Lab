"""SQLite log of every attempt, used for the reporting dashboard.
Only metadata is stored: no tokens, no secrets, no images."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from . import config

_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT,
            challenge_type TEXT, method TEXT, success INTEGER, status TEXT,
            confidence REAL, processing_ms REAL, cost_usd REAL)""")
        _conn.commit()
    return _conn


def record(kind: str, challenge_type: str, method: str, success: bool | None,
           status: str, processing_ms: float, confidence: float | None = None,
           cost_usd: float = 0.0) -> None:
    conn().execute(
        "INSERT INTO attempts (ts, kind, challenge_type, method, success, status,"
        " confidence, processing_ms, cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), kind, challenge_type, method,
         None if success is None else int(success), status, confidence, processing_ms, cost_usd))
    conn().commit()


def stats() -> dict:
    c = conn()
    total, ok, failed, unknown, avg_ms, cost = c.execute(
        "SELECT COUNT(*), COALESCE(SUM(success=1),0), COALESCE(SUM(success=0),0),"
        " COALESCE(SUM(success IS NULL),0), COALESCE(AVG(processing_ms),0),"
        " COALESCE(SUM(cost_usd),0) FROM attempts").fetchone()
    breakdown = [dict(r) for r in c.execute(
        "SELECT kind, challenge_type, method, COUNT(*) AS attempts,"
        " COALESCE(SUM(success=1),0) AS successful, COALESCE(SUM(success=0),0) AS failed,"
        " ROUND(AVG(processing_ms),2) AS avg_ms, ROUND(SUM(cost_usd),4) AS cost_usd"
        " FROM attempts GROUP BY kind, challenge_type, method ORDER BY attempts DESC")]
    recent = [dict(r) for r in c.execute(
        "SELECT ts, kind, challenge_type, method, success, status, confidence, processing_ms"
        " FROM attempts ORDER BY id DESC LIMIT 25")]
    return {"total_attempts": total, "successful": ok, "failed": failed, "unscored": unknown,
            "avg_processing_ms": round(avg_ms, 2), "total_cost_usd": round(cost, 4),
            "est_cost_per_challenge_usd": round(cost / total, 6) if total else 0.0,
            "breakdown": breakdown, "recent": recent}


def reset() -> None:
    conn().execute("DELETE FROM attempts")
    conn().commit()
