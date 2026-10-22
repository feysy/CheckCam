"""
Storage layer — SQLite + sqlite-vec for clip metadata and vector search.

Schema:
  clips:       Relational metadata (file_path, timestamp, processed flag, action)
  clip_vectors: vec0 virtual table with embeddings linked by shared rowid
"""

import sqlite3
import numpy as np
import structlog

logger = structlog.get_logger()


def _serialize_float32(vector: list[float] | np.ndarray) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return np.array(vector, dtype=np.float32).tobytes()


class Database:
    """SQLite + sqlite-vec database manager."""

    def __init__(self, db_path: str, embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        """Open connection and initialize schema."""
        import sqlite_vec

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        self._create_tables()
        logger.info("database_ready", path=self.db_path,
                     embedding_dim=self.embedding_dim)

    def _create_tables(self):
        """Create metadata, vector, actions, and rules tables if they don't exist."""
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                action TEXT DEFAULT NULL,
                thumbnail_path TEXT DEFAULT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_clips_processed
                ON clips(processed);

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                command TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL UNIQUE,
                action_id INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (action_id) REFERENCES actions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_rules_action_id
                ON rules(action_id);
        """)

        # vec0 virtual table — rowid links to clips.id
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS clip_vectors
            USING vec0(embedding float[{self.embedding_dim}])
        """)
        self.conn.commit()

        # Seed initial actions and rules if empty
        self._seed_default_rules()

    def _seed_default_rules(self):
        """Seed default actions and initial rules if tables are empty."""
        # Check / migrate thumbnail_path column for existing databases
        try:
            self.conn.execute("ALTER TABLE clips ADD COLUMN thumbnail_path TEXT DEFAULT NULL")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Already exists

        action_count = self.conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        if action_count == 0:
            self.conn.executescript("""
                INSERT INTO actions (id, name, command) VALUES
                    (1, 'discard', 'DISCARD'),
                    (2, 'save', 'SAVE'),
                    (3, 'alert', 'ALERT');
            """)
            self.conn.commit()
            logger.info("default_actions_seeded")

        rule_count = self.conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        if rule_count == 0:
            default_rules = [
                # Discard (action_id=1)
                ("an empty room or hallway", 1),
                ("tree branches or shadows blowing in the wind", 1),
                ("car headlights passing across the wall", 1),
                # Save (action_id=2)
                ("a person walking", 2),
                ("a dog or cat moving around", 2),
                ("a vehicle pulling into the driveway", 2),
                # Alert (action_id=3)
                ("a person with a bottle", 3),
                ("smoke, fire, or unusual hazardous situation", 3),
                ("a delivery driver dropping off a package", 3),
            ]
            self.conn.executemany(
                "INSERT OR IGNORE INTO rules (description, action_id) VALUES (?, ?)",
                default_rules
            )
            self.conn.commit()
            logger.info("default_rules_seeded", count=len(default_rules))

    # ── Clip CRUD ─────────────────────────────────────────────────────

    def insert_clip(self, file_path: str, timestamp: str,
                    duration_sec: float) -> int:
        """Insert a new clip record with processed=0. Returns clip ID."""
        cursor = self.conn.execute(
            """INSERT INTO clips (file_path, timestamp, duration_sec, processed)
               VALUES (?, ?, ?, 0)""",
            (file_path, timestamp, duration_sec)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_unprocessed_clips(self) -> list[dict]:
        """Return all clips where processed=0."""
        rows = self.conn.execute(
            "SELECT * FROM clips WHERE processed = 0"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_processed(self, clip_id: int, action: str, thumbnail_path: str | None = None):
        """Mark a clip as processed with action and optional thumbnail path."""
        self.conn.execute(
            "UPDATE clips SET processed = 1, action = ?, thumbnail_path = ? WHERE id = ?",
            (action, thumbnail_path, clip_id)
        )
        self.conn.commit()

    def update_thumbnail(self, clip_id: int, thumbnail_path: str):
        """Update thumbnail path for a clip."""
        self.conn.execute(
            "UPDATE clips SET thumbnail_path = ? WHERE id = ?",
            (thumbnail_path, clip_id)
        )
        self.conn.commit()

    def get_clip_by_id(self, clip_id: int) -> dict | None:
        """Fetch a single clip by ID."""
        row = self.conn.execute(
            "SELECT * FROM clips WHERE id = ?", (clip_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_recent_clips(self, limit: int = 30) -> list[dict]:
        """Fetch recent saved/alerted clips for the UI dashboard."""
        rows = self.conn.execute("""
            SELECT
                id AS clip_id,
                file_path,
                timestamp,
                duration_sec,
                action,
                thumbnail_path
            FROM clips
            WHERE processed = 1 AND action IN ('SAVED', 'ALERTED')
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── Vector Operations ─────────────────────────────────────────────

    def insert_vector(self, clip_id: int, embedding: np.ndarray):
        """Insert a normalized embedding vector linked to a clip by rowid."""
        self.conn.execute(
            "INSERT INTO clip_vectors(rowid, embedding) VALUES (?, ?)",
            (clip_id, _serialize_float32(embedding))
        )
        self.conn.commit()

    def search_similar(self, query_embedding: np.ndarray,
                       limit: int = 10) -> list[dict]:
        """
        KNN search: find clips with embeddings closest to query_embedding.
        Returns list of dicts with clip_id, distance, file_path, timestamp, action, thumbnail_path.
        """
        rows = self.conn.execute("""
            SELECT
                cv.rowid AS clip_id,
                cv.distance,
                c.file_path,
                c.timestamp,
                c.duration_sec,
                c.action,
                c.thumbnail_path
            FROM clip_vectors cv
            JOIN clips c ON c.id = cv.rowid
            WHERE cv.embedding MATCH ? AND k = ?
            ORDER BY cv.distance
        """, (_serialize_float32(query_embedding), limit)).fetchall()
        return [dict(r) for r in rows]

    # ── Actions & Rules CRUD ──────────────────────────────────────────
    def get_actions(self) -> list[dict]:
        """Fetch all available actions."""
        rows = self.conn.execute(
            "SELECT id, name, command FROM actions ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_action(self, name: str, command: str) -> int:
        """Insert a new action. Returns action ID."""
        cursor = self.conn.execute(
            "INSERT INTO actions (name, command) VALUES (?, ?)",
            (name, command)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_rules(self) -> list[dict]:
        """Fetch all rules joined with their target action details."""
        rows = self.conn.execute("""
            SELECT
                r.id,
                r.description,
                r.action_id,
                a.name AS action_name,
                a.command AS action_command
            FROM rules r
            JOIN actions a ON a.id = r.action_id
            ORDER BY r.id
        """).fetchall()
        return [dict(r) for r in rows]

    def add_rule(self, description: str, action_id: int) -> int:
        """Insert a new classification rule. Returns rule ID."""
        cursor = self.conn.execute(
            "INSERT INTO rules (description, action_id) VALUES (?, ?)",
            (description, action_id)
        )
        self.conn.commit()
        return cursor.lastrowid

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule by ID."""
        cursor = self.conn.execute(
            "DELETE FROM rules WHERE id = ?", (rule_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("database_closed")
