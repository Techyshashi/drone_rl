"""
NETRA Command Center - persistent experiment storage (SQLite).

Design rules
  * Every completed episode is written to disk immediately, inside its own transaction.
  * Missing measurements are stored as NULL, never as 0.
  * A write that fails is NOT dropped: it is appended to a JSONL spool file next to the database and
    re-imported on the next successful write / next start-up. The caller also gets a DatabaseError so the
    UI can show it.
  * (experiment_id, episode_number) is UNIQUE and inserts use INSERT OR IGNORE, so retries or UI refreshes
    can never create duplicates.
  * All timestamps are UTC ISO-8601 with millisecond precision, e.g. 2026-10-04T12:30:15.123Z.
"""
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone

NETRA_VERSION = "1.0.0"
SCHEMA_VERSION = 1
DB_NAME = "netra_experiments.db"
SPOOL_NAME = "pending_episodes.jsonl"

EXPERIMENT_STATUSES = ("RUNNING", "COMPLETED", "CANCELLED", "INTERRUPTED", "FAILED")
EPISODE_OUTCOMES = ("SUCCESS", "COLLISION", "TIMEOUT", "OTHER_FAILURE", "INTERRUPTED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id        TEXT PRIMARY KEY,
    started_at           TEXT NOT NULL,
    ended_at             TEXT,
    last_activity        TEXT,
    status               TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','CANCELLED','INTERRUPTED','FAILED')),
    end_reason           TEXT,
    reflex_mode          TEXT NOT NULL CHECK (reflex_mode IN ('ON','OFF')),
    policy_path          TEXT,
    policy_file          TEXT,
    policy_sha256        TEXT,
    vecnorm_path         TEXT,
    vecnorm_file         TEXT,
    vecnorm_sha256       TEXT,
    goal_setting         INTEGER,          -- 0 = random each episode, 1..6 = fixed survivor
    goal_label           TEXT,
    configured_episodes  INTEGER NOT NULL,
    sim_speed            REAL,
    headless             INTEGER,
    software_version     TEXT,
    git_commit           TEXT,
    config_json          TEXT,             -- full model / environment configuration snapshot
    previous_experiment_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_exp_started ON experiments (started_at);
CREATE INDEX IF NOT EXISTS ix_exp_mode    ON experiments (reflex_mode);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id            TEXT PRIMARY KEY,
    experiment_id         TEXT NOT NULL REFERENCES experiments (experiment_id) ON DELETE CASCADE,
    episode_number        INTEGER NOT NULL,
    reflex_mode           TEXT NOT NULL CHECK (reflex_mode IN ('ON','OFF')),
    goal_id               INTEGER,          -- survivor 1..6 actually used in this episode
    outcome               TEXT NOT NULL CHECK (outcome IN ('SUCCESS','COLLISION','TIMEOUT','OTHER_FAILURE','INTERRUPTED')),
    status                TEXT NOT NULL CHECK (status IN ('COMPLETED','INTERRUPTED')),
    success               INTEGER,          -- NULL for interrupted episodes
    collision             INTEGER,
    duration_s            REAL,
    path_length_m         REAL,
    final_goal_dist_m     REAL,
    remaining_survivors   INTEGER,          -- NULL: each episode targets exactly one survivor
    reflex_activation_pct REAL,             -- NULL when reflex is OFF
    min_clearance_m       REAL,             -- minimum nearest-obstacle range seen in telemetry
    telemetry_samples     INTEGER,
    policy_actions_json   TEXT,
    fused_actions_json    TEXT,
    termination_reason    TEXT,
    error_details         TEXT,
    started_at            TEXT,
    ended_at              TEXT,
    recorded_at           TEXT NOT NULL,
    UNIQUE (experiment_id, episode_number)
);
CREATE INDEX IF NOT EXISTS ix_ep_exp     ON episodes (experiment_id);
CREATE INDEX IF NOT EXISTS ix_ep_mode    ON episodes (reflex_mode);
CREATE INDEX IF NOT EXISTS ix_ep_outcome ON episodes (outcome);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""

EPISODE_COLUMNS = [
    "episode_id", "experiment_id", "episode_number", "reflex_mode", "goal_id", "outcome", "status", "success",
    "collision", "duration_s", "path_length_m", "final_goal_dist_m", "remaining_survivors",
    "reflex_activation_pct", "min_clearance_m", "telemetry_samples", "policy_actions_json", "fused_actions_json",
    "termination_reason", "error_details", "started_at", "ended_at", "recorded_at",
]
EXPERIMENT_COLUMNS = [
    "experiment_id", "started_at", "ended_at", "last_activity", "status", "end_reason", "reflex_mode",
    "policy_path", "policy_file", "policy_sha256", "vecnorm_path", "vecnorm_file", "vecnorm_sha256",
    "goal_setting", "goal_label", "configured_episodes", "sim_speed", "headless", "software_version",
    "git_commit", "config_json", "previous_experiment_id",
]


class DatabaseError(Exception):
    """A write or read failed. For episode writes the record was spooled to disk first."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def local_date_to_utc(d, end_of_day=False):
    """datetime.date (local calendar day) -> UTC ISO string at the start / end of that local day."""
    t = datetime(d.year, d.month, d.day, 23, 59, 59, 999000) if end_of_day else datetime(d.year, d.month, d.day)
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_to_local_str(s, fmt="%Y-%m-%d %H:%M:%S"):
    if not s:
        return ""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone().strftime(fmt)
    except ValueError:
        return s


def app_data_dir():
    """Stable per-user data folder (override with NETRA_DATA_DIR)."""
    env = os.environ.get("NETRA_DATA_DIR")
    if env:
        base = env
    elif sys.platform.startswith("win"):
        base = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "NETRA")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/NETRA")
    else:
        base = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "netra")
    os.makedirs(base, exist_ok=True)
    return base


class ExperimentDatabase:
    def __init__(self, path=None):
        self.data_dir = os.path.dirname(path) if path else app_data_dir()
        if path:
            os.makedirs(self.data_dir or ".", exist_ok=True)
        self.path = path or os.path.join(self.data_dir, DB_NAME)
        self.spool_path = os.path.join(self.data_dir, SPOOL_NAME)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.executescript(SCHEMA)
            self._exec("INSERT OR IGNORE INTO schema_info (key, value) VALUES ('schema_version', ?)",
                       (str(SCHEMA_VERSION),))
        except sqlite3.Error as e:
            raise DatabaseError(f"cannot open database {self.path}: {e}") from e

    # ------------------------------------------------------------------ low level
    def _exec(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params)

    def _tx(self, fn):
        """Run fn(cursor_exec) inside BEGIN IMMEDIATE ... COMMIT; roll back on any error."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    out = fn(self._conn.execute)
                    self._conn.execute("COMMIT")
                    return out
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            except sqlite3.Error as e:
                raise DatabaseError(str(e)) from e

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------------ settings
    def get_setting(self, key, default=None):
        try:
            r = self._exec("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return default
        return r["value"] if r else default

    def all_settings(self):
        try:
            return {r["key"]: r["value"] for r in self._exec("SELECT key, value FROM app_settings")}
        except sqlite3.Error:
            return {}

    def set_settings(self, items):
        now = utc_now()

        def fn(ex):
            for k, v in items.items():
                ex("INSERT INTO app_settings (key, value, updated_at) VALUES (?,?,?) "
                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                   (k, None if v is None else str(v), now))
        self._tx(fn)

    # ------------------------------------------------------------------ experiments
    def create_experiment(self, rec):
        rec = dict(rec)
        rec.setdefault("started_at", utc_now())
        rec.setdefault("last_activity", rec["started_at"])
        rec.setdefault("status", "RUNNING")
        cols = [c for c in EXPERIMENT_COLUMNS if c in rec]
        sql = f"INSERT INTO experiments ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        self._tx(lambda ex: ex(sql, [rec[c] for c in cols]))
        return rec["experiment_id"]

    def set_experiment_status(self, experiment_id, status, reason=None):
        if status not in EXPERIMENT_STATUSES:
            raise ValueError(status)
        now = utc_now()
        ended = None if status == "RUNNING" else now
        self._tx(lambda ex: ex("UPDATE experiments SET status = ?, ended_at = ?, end_reason = ?, "
                               "last_activity = ? WHERE experiment_id = ?",
                               (status, ended, reason, now, experiment_id)))

    def touch_experiment(self, experiment_id):
        try:
            self._tx(lambda ex: ex("UPDATE experiments SET last_activity = ? WHERE experiment_id = ?",
                                   (utc_now(), experiment_id)))
        except DatabaseError:
            pass

    def recover_orphans(self):
        """Experiments still marked RUNNING belong to a session that ended unexpectedly."""
        rows = self._exec("SELECT experiment_id, last_activity, started_at FROM experiments "
                          "WHERE status = 'RUNNING'").fetchall()
        ids = []

        def fn(ex):
            for r in rows:
                ex("UPDATE experiments SET status = 'INTERRUPTED', ended_at = ?, "
                   "end_reason = 'previous session ended unexpectedly' WHERE experiment_id = ?",
                   (r["last_activity"] or r["started_at"], r["experiment_id"]))
                ids.append(r["experiment_id"])
        if rows:
            self._tx(fn)
        return ids

    def get_experiment(self, experiment_id):
        r = self._exec("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        return dict(r) if r else None

    def list_experiments(self, date_from=None, date_to=None, reflex=None, goal=None, outcome=None,
                         search=None, experiment_ids=None):
        """Filters on experiment fields; goal / outcome match experiments that contain such an episode."""
        where, p = [], []
        if date_from:
            where.append("e.started_at >= ?")
            p.append(date_from)
        if date_to:
            where.append("e.started_at <= ?")
            p.append(date_to)
        if reflex:
            where.append("e.reflex_mode = ?")
            p.append(reflex)
        if search:
            where.append("e.experiment_id LIKE ? ESCAPE '\\'")
            p.append("%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if experiment_ids:
            where.append(f"e.experiment_id IN ({','.join('?' * len(experiment_ids))})")
            p.extend(experiment_ids)
        sub, sp = [], []
        if goal:
            sub.append("x.goal_id = ?")
            sp.append(int(goal))
        if outcome:
            sub.append("x.outcome = ?")
            sp.append(outcome)
        if sub:
            where.append("EXISTS (SELECT 1 FROM episodes x WHERE x.experiment_id = e.experiment_id AND "
                         + " AND ".join(sub) + ")")
            p.extend(sp)
        sql = "SELECT e.* FROM experiments e" + (" WHERE " + " AND ".join(where) if where else "") \
              + " ORDER BY e.started_at DESC"
        try:
            return [dict(r) for r in self._exec(sql, p).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(str(e)) from e

    # ------------------------------------------------------------------ episodes
    @staticmethod
    def _episode_values(rec):
        rec = dict(rec)
        rec.setdefault("recorded_at", utc_now())
        for c in EPISODE_COLUMNS:
            rec.setdefault(c, None)
        return rec

    def _insert_episode(self, ex, rec):
        cur = ex(f"INSERT OR IGNORE INTO episodes ({','.join(EPISODE_COLUMNS)}) "
                 f"VALUES ({','.join('?' * len(EPISODE_COLUMNS))})", [rec[c] for c in EPISODE_COLUMNS])
        if cur.rowcount:
            ex("UPDATE experiments SET last_activity = ? WHERE experiment_id = ?",
               (rec["recorded_at"], rec["experiment_id"]))
        return cur.rowcount == 1

    def save_episode(self, rec):
        """Insert one episode. Returns True if inserted, False if it already existed (duplicate ignored).
        On failure the record is spooled to disk and DatabaseError is raised (nothing is silently lost)."""
        rec = self._episode_values(rec)
        if rec["outcome"] not in EPISODE_OUTCOMES:
            raise ValueError(f"bad outcome {rec['outcome']!r}")
        try:
            self.flush_spool()
            return self._tx(lambda ex: self._insert_episode(ex, rec))
        except DatabaseError as e:
            self._spool(rec)
            raise DatabaseError(f"{e} - episode {rec['episode_id']} saved to {self.spool_path}") from e

    def _spool(self, rec):
        try:
            with open(self.spool_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def spool_size(self):
        try:
            with open(self.spool_path, encoding="utf-8") as f:
                return sum(1 for ln in f if ln.strip())
        except OSError:
            return 0

    def flush_spool(self):
        """Import spooled episodes. Returns number imported. Lines that still fail stay in the file."""
        if not os.path.isfile(self.spool_path):
            return 0
        with open(self.spool_path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        keep, done = [], 0
        for ln in lines:
            try:
                rec = self._episode_values(json.loads(ln))
                self._tx(lambda ex, r=rec: self._insert_episode(ex, r))
                done += 1
            except (ValueError, DatabaseError):
                keep.append(ln)
        if keep:
            with open(self.spool_path, "w", encoding="utf-8") as f:
                f.write("\n".join(keep) + "\n")
        else:
            os.remove(self.spool_path)
        return done

    def get_episodes(self, experiment_ids=None, reflex=None, outcome=None, goal=None,
                     date_from=None, date_to=None):
        where, p = [], []
        if experiment_ids is not None:
            if not experiment_ids:
                return []
            where.append(f"p.experiment_id IN ({','.join('?' * len(experiment_ids))})")
            p.extend(experiment_ids)
        if reflex:
            where.append("p.reflex_mode = ?")
            p.append(reflex)
        if outcome:
            where.append("p.outcome = ?")
            p.append(outcome)
        if goal:
            where.append("p.goal_id = ?")
            p.append(int(goal))
        if date_from:
            where.append("COALESCE(p.ended_at, p.recorded_at) >= ?")
            p.append(date_from)
        if date_to:
            where.append("COALESCE(p.ended_at, p.recorded_at) <= ?")
            p.append(date_to)
        sql = ("SELECT p.* FROM episodes p" + (" WHERE " + " AND ".join(where) if where else "")
               + " ORDER BY p.recorded_at, p.experiment_id, p.episode_number")
        try:
            return [dict(r) for r in self._exec(sql, p).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(str(e)) from e

    def next_episode_number(self, experiment_id):
        r = self._exec("SELECT COALESCE(MAX(episode_number), 0) AS m FROM episodes WHERE experiment_id = ?",
                       (experiment_id,)).fetchone()
        return int(r["m"]) + 1

    def count_completed(self, experiment_id):
        r = self._exec("SELECT COUNT(*) AS c FROM episodes WHERE experiment_id = ? AND status = 'COMPLETED'",
                       (experiment_id,)).fetchone()
        return int(r["c"])

    # ------------------------------------------------------------------ maintenance
    def backup(self, dest=None):
        """Consistent copy of the database using SQLite's online backup API."""
        if dest is None:
            bdir = os.path.join(self.data_dir, "backups")
            os.makedirs(bdir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(bdir, f"netra_experiments_{stamp}.db")
        try:
            with self._lock:
                out = sqlite3.connect(dest)
                try:
                    self._conn.backup(out)
                finally:
                    out.close()
        except sqlite3.Error as e:
            raise DatabaseError(f"backup failed: {e}") from e
        return dest

    def clear_history(self):
        """Delete all experiments and episodes (settings are kept). Always backs up first. Returns backup path."""
        dest = self.backup()
        self._tx(lambda ex: (ex("DELETE FROM episodes"), ex("DELETE FROM experiments")))
        if os.path.isfile(self.spool_path):
            os.replace(self.spool_path, dest + ".spool.jsonl")
        return dest

    def counts(self):
        e = self._exec("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]
        p = self._exec("SELECT COUNT(*) AS c FROM episodes").fetchone()["c"]
        return int(e), int(p)
