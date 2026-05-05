import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS dashboard_actions (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        action TEXT NOT NULL,
        game_code TEXT NOT NULL,
        game_name TEXT NOT NULL,
        status TEXT NOT NULL,
        exit_code INTEGER NOT NULL,
        command TEXT NOT NULL,
        summary TEXT NOT NULL,
        archive_dir TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendation_records (
        game_code TEXT NOT NULL,
        target_issue TEXT NOT NULL,
        run_id TEXT NOT NULL,
        rank INTEGER NOT NULL,
        generated_at TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        numbers TEXT NOT NULL,
        status TEXT NOT NULL,
        draw_numbers TEXT NOT NULL,
        prize_level TEXT NOT NULL,
        cost REAL NOT NULL,
        payout REAL NOT NULL,
        PRIMARY KEY (game_code, target_issue, run_id, rank, numbers)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dashboard_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        game_code TEXT NOT NULL,
        game_name TEXT NOT NULL,
        action TEXT NOT NULL,
        status TEXT NOT NULL,
        summary TEXT NOT NULL
    )
    """,
)


def init_history_db(db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for statement in SCHEMA:
            conn.execute(statement)
        conn.commit()


def record_dashboard_action(db_path: Path, record: Dict[str, Any]) -> None:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO dashboard_actions (
                job_id, created_at, action, game_code, game_name, status,
                exit_code, command, summary, archive_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("job_id", "")),
                str(record.get("created_at", "")),
                str(record.get("action", "")),
                str(record.get("game_code", "")),
                str(record.get("game_name", "")),
                str(record.get("status", "")),
                int(record.get("exit_code", 0) or 0),
                str(record.get("command", "")),
                str(record.get("summary", "")),
                str(record.get("archive_dir", "")),
            ),
        )
        conn.commit()


def load_dashboard_actions(db_path: Path, limit: int = 100) -> List[Dict[str, Any]]:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, job_id, action, game_code, game_name, status,
                   exit_code, command, summary, archive_dir
            FROM dashboard_actions
            ORDER BY created_at DESC, job_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def sync_recommendation_records(db_path: Path, records: Iterable[Any]) -> None:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO recommendation_records (
                game_code, target_issue, run_id, rank, generated_at,
                strategy_name, numbers, status, draw_numbers, prize_level,
                cost, payout
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.game_code,
                    record.target_issue,
                    record.run_id,
                    int(record.rank),
                    record.generated_at,
                    record.strategy_name,
                    record.numbers,
                    record.status,
                    record.draw_numbers,
                    record.prize_level,
                    float(record.cost or 0),
                    float(record.payout or 0),
                )
                for record in records
            ],
        )
        conn.commit()


def save_dashboard_config(db_path: Path, config: Dict[str, str]) -> None:
    init_history_db(db_path)
    allowed = {
        "llm_provider",
        "llm_base_url",
        "llm_model",
        "llm_api_key",
    }
    with sqlite3.connect(db_path) as conn:
        for key, value in config.items():
            if key not in allowed:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO dashboard_config (key, value) VALUES (?, ?)",
                (key, str(value or "")),
            )
        conn.commit()


def load_dashboard_config(db_path: Path) -> Dict[str, str]:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM dashboard_config").fetchall()
    config = {str(key): str(value) for key, value in rows}
    api_key = config.get("llm_api_key", "")
    config["llm_api_key_masked"] = mask_api_key(api_key)
    return config


def mask_api_key(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    if len(value) <= 3:
        return "***"
    prefix = value[:3] if value.startswith("sk-") else value[:4]
    return f"{prefix}***"


def record_training_record(db_path: Path, record: Dict[str, Any]) -> None:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO training_records (
                created_at, game_code, game_name, action, status, summary
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("created_at", "")),
                str(record.get("game_code", "")),
                str(record.get("game_name", "")),
                str(record.get("action", "")),
                str(record.get("status", "")),
                str(record.get("summary", "")),
            ),
        )
        conn.commit()


def load_training_records(db_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, game_code, game_name, action, status, summary
            FROM training_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
