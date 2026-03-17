"""
db_manager.py
Handles all MySQL operations for the People Counter system.

Key behaviors:
  - On startup: reads today's last saved IN/OUT from DB → resumes from there.
  - On new day: starts from 0 automatically.
  - On code stop/crash: next run resumes from last saved value.
  - Periodic sync saves live counts to DB.
"""

import mysql.connector
from mysql.connector import Error
from datetime import date
import threading
import time
import logging

logger = logging.getLogger("DBManager")


# ─────────────────────────────────────────────
#  Schema (run once)
# ─────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS daily_counts (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    count_date  DATE         NOT NULL,
    camera_id   VARCHAR(64)  NOT NULL DEFAULT 'default',
    people_in   INT          NOT NULL DEFAULT 0,
    people_out  INT          NOT NULL DEFAULT 0,
    last_update DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_date_cam (count_date, camera_id)
);
"""


class DBManager:
    def __init__(self, cfg: dict, camera_id: str = "default"):
        self.cfg       = cfg
        self.camera_id = camera_id
        self.conn      = None
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

        # live counters (set after calling load_today())
        self.people_in  = 0
        self.people_out = 0

    # ──────────────────────────────────────────
    #  Connection helpers
    # ──────────────────────────────────────────
    def connect(self) -> bool:
        try:
            self.conn = mysql.connector.connect(
                host     = self.cfg["host"],
                port     = int(self.cfg.get("port", 3306)),
                user     = self.cfg["user"],
                password = self.cfg["password"],
                database = self.cfg["database"],
                autocommit=False,
            )
            self._ensure_schema()
            logger.info("MySQL connected.")
            return True
        except Error as e:
            logger.error(f"MySQL connect failed: {e}")
            self.conn = None
            return False

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute(SCHEMA_SQL)
        self.conn.commit()
        cur.close()

    def _reconnect(self) -> bool:
        try:
            if self.conn and self.conn.is_connected():
                return True
        except Exception:
            pass
        logger.warning("Reconnecting to MySQL …")
        return self.connect()

    # ──────────────────────────────────────────
    #  Load today's data on startup
    # ──────────────────────────────────────────
    def load_today(self) -> tuple[int, int]:
        """
        Returns (people_in, people_out) for today from DB.
        Returns (0, 0) if no record yet → fresh day.
        Sets self.people_in / self.people_out automatically.
        """
        if not self._reconnect():
            logger.warning("Cannot load today — DB unavailable. Starting from 0.")
            self.people_in  = 0
            self.people_out = 0
            return 0, 0

        today = date.today().isoformat()
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT people_in, people_out FROM daily_counts "
                "WHERE count_date = %s AND camera_id = %s",
                (today, self.camera_id),
            )
            row = cur.fetchone()
            cur.close()

            if row:
                self.people_in, self.people_out = int(row[0]), int(row[1])
                logger.info(
                    f"Resumed from DB — IN: {self.people_in}, OUT: {self.people_out}"
                )
            else:
                self.people_in  = 0
                self.people_out = 0
                logger.info("New day — starting from 0.")

            return self.people_in, self.people_out

        except Error as e:
            logger.error(f"load_today error: {e}")
            self.people_in  = 0
            self.people_out = 0
            return 0, 0

    # ──────────────────────────────────────────
    #  Save (upsert) current counters
    # ──────────────────────────────────────────
    def save(self, people_in: int, people_out: int):
        """Upsert today's row with latest counts."""
        with self._lock:
            self.people_in  = people_in
            self.people_out = people_out

        if not self._reconnect():
            logger.warning("DB unavailable — skipping save.")
            return

        today = date.today().isoformat()
        sql = """
            INSERT INTO daily_counts (count_date, camera_id, people_in, people_out)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                people_in   = VALUES(people_in),
                people_out  = VALUES(people_out),
                last_update = CURRENT_TIMESTAMP
        """
        try:
            cur = self.conn.cursor()
            cur.execute(sql, (today, self.camera_id, people_in, people_out))
            self.conn.commit()
            cur.close()
        except Error as e:
            logger.error(f"save error: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass

    # ──────────────────────────────────────────
    #  Background auto-sync thread
    # ──────────────────────────────────────────
    def start_sync(self, get_counts_fn, interval_sec: int = 5):
        """
        Starts background thread that calls get_counts_fn() → (in, out)
        and saves to DB every interval_sec seconds.
        """
        self._running = True

        def _loop():
            while self._running:
                time.sleep(interval_sec)
                try:
                    in_cnt, out_cnt = get_counts_fn()
                    self.save(in_cnt, out_cnt)
                    logger.debug(f"Auto-sync: IN={in_cnt} OUT={out_cnt}")
                except Exception as e:
                    logger.error(f"Sync loop error: {e}")

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info(f"DB auto-sync started (every {interval_sec}s).")

    def stop_sync(self):
        self._running = False

    # ──────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────
    def close(self):
        self.stop_sync()
        try:
            if self.conn and self.conn.is_connected():
                self.conn.close()
                logger.info("MySQL connection closed.")
        except Exception:
            pass
