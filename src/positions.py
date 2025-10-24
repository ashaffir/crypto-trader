from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Store DB under CONTROL_DIR so bot and UI share it via mounted volume
DEFAULT_DB_PATH = os.getenv(
    "POSITIONS_DB",
    os.path.join(
        os.getenv("CONTROL_DIR", os.path.join("data", "control")), "positions.sqlite"
    ),
)


@dataclass
class Position:
    id: Optional[int]
    symbol: str
    direction: str  # "long" or "short"
    leverage: int
    opened_ts_ms: int
    qty: Optional[float] = None
    entry_px: Optional[float] = None
    notional: Optional[float] = None
    confidence: Optional[float] = None
    closed_ts_ms: Optional[int] = None
    exit_px: Optional[float] = None
    pnl: Optional[float] = None
    llm_model: Optional[str] = None
    best_favorable_px: Optional[float] = None
    close_reason: Optional[str] = None
    venue: Optional[str] = None  # "spot" | "futures"
    exec_mode: Optional[str] = None  # "paper" | "live"


class PositionStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    leverage INTEGER NOT NULL,
                    opened_ts_ms INTEGER NOT NULL,
                    qty REAL,
                    entry_px REAL,
                    notional REAL,
                    confidence REAL,
                    closed_ts_ms INTEGER,
                    exit_px REAL,
                    pnl REAL,
                    llm_model TEXT,
                    best_favorable_px REAL,
                    close_reason TEXT,
                    venue TEXT,
                    exec_mode TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(symbol, closed_ts_ms)"
            )
            # Best-effort migrations for older DBs
            try:
                cols = {
                    r[1]
                    for r in conn.execute("PRAGMA table_info(positions)").fetchall()
                }
                if "llm_model" not in cols:
                    conn.execute("ALTER TABLE positions ADD COLUMN llm_model TEXT")
                if "best_favorable_px" not in cols:
                    conn.execute(
                        "ALTER TABLE positions ADD COLUMN best_favorable_px REAL"
                    )
                if "close_reason" not in cols:
                    conn.execute("ALTER TABLE positions ADD COLUMN close_reason TEXT")
                if "notional" not in cols:
                    conn.execute("ALTER TABLE positions ADD COLUMN notional REAL")
                if "venue" not in cols:
                    conn.execute("ALTER TABLE positions ADD COLUMN venue TEXT")
                if "exec_mode" not in cols:
                    conn.execute("ALTER TABLE positions ADD COLUMN exec_mode TEXT")
            except Exception:
                pass

    # ---- CRUD ----
    def open_position(
        self,
        symbol: str,
        direction: str,
        leverage: int,
        opened_ts_ms: int,
        qty: Optional[float] = None,
        entry_px: Optional[float] = None,
        confidence: Optional[float] = None,
        llm_model: Optional[str] = None,
        venue: Optional[str] = None,
        exec_mode: Optional[str] = None,
    ) -> int:
        direction_norm = "long" if direction in ("buy", "long") else "short"
        # Compute notional exposure if data available: qty * entry_px * leverage
        notional_value: Optional[float] = None
        try:
            if qty is not None and entry_px is not None and leverage is not None:
                notional_value = float(qty) * float(entry_px) * int(leverage)
        except Exception:
            notional_value = None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO positions(
                    symbol, direction, leverage, opened_ts_ms, qty, entry_px, notional,
                    confidence, llm_model, best_favorable_px, venue, exec_mode
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    symbol.upper(),
                    direction_norm,
                    int(leverage),
                    int(opened_ts_ms),
                    qty,
                    entry_px,
                    notional_value,
                    confidence,
                    llm_model,
                    entry_px,
                    (venue if venue in ("spot", "futures") else None),
                    (exec_mode if exec_mode in ("paper", "live") else None),
                ],
            )
            return int(cur.lastrowid)

    def close_position(
        self,
        position_id: int,
        closed_ts_ms: int,
        exit_px: Optional[float] = None,
        pnl: Optional[float] = None,
        close_reason: Optional[str] = None,
    ) -> None:
        # If pnl is not provided, try to compute it using stored entry, qty and leverage
        computed_pnl = pnl
        try:
            if computed_pnl is None and exit_px is not None:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT direction, entry_px, qty, leverage FROM positions WHERE id=?",
                        [int(position_id)],
                    ).fetchone()
                    if (
                        row is not None
                        and row["entry_px"] is not None
                        and row["qty"] is not None
                    ):
                        entry = float(row["entry_px"])  # type: ignore
                        qty = float(row["qty"])  # type: ignore
                        lev = int(row["leverage"]) if row["leverage"] is not None else 1  # type: ignore
                        direction = str(row["direction"]) if row["direction"] is not None else "long"  # type: ignore
                        if direction == "long":
                            computed_pnl = (float(exit_px) - entry) * qty * lev
                        else:
                            computed_pnl = (entry - float(exit_px)) * qty * lev
        except Exception:
            # Best effort; keep provided pnl (possibly None)
            pass
        with self._connect() as conn:
            conn.execute(
                "UPDATE positions SET closed_ts_ms=?, exit_px=?, pnl=?, close_reason=? WHERE id=?",
                [
                    int(closed_ts_ms),
                    exit_px,
                    computed_pnl,
                    close_reason,
                    int(position_id),
                ],
            )

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM positions WHERE closed_ts_ms IS NULL"
        args: List[Any] = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol.upper())
        with self._connect() as conn:
            rows = conn.execute(q, args).fetchall()
            return [dict(r) for r in rows]

    def get_latest_open_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE symbol=? AND closed_ts_ms IS NULL ORDER BY id DESC LIMIT 1",
                [symbol.upper()],
            ).fetchone()
            return dict(row) if row else None

    def count_open(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) as c FROM positions WHERE closed_ts_ms IS NULL"
            ).fetchone()
            return int(row["c"]) if row else 0

    def all_positions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM positions ORDER BY id DESC"
                ).fetchall()
            ]

    def update_best_favorable(self, position_id: int, price: float) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT direction, best_favorable_px FROM positions WHERE id=?",
                [int(position_id)],
            ).fetchone()
            if not row:
                return
            direction = (
                str(row["direction"]) if row["direction"] is not None else "long"
            )
            cur_best = row["best_favorable_px"]
            new_best = price
            try:
                if cur_best is not None:
                    if direction == "long":
                        new_best = max(float(cur_best), float(price))
                    else:
                        new_best = min(float(cur_best), float(price))
            except Exception:
                new_best = price
            conn.execute(
                "UPDATE positions SET best_favorable_px=? WHERE id=?",
                [new_best, int(position_id)],
            )


__all__ = ["Position", "PositionStore", "DEFAULT_DB_PATH"]
