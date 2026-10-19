"""Servants 数据库管理类，使用 SQLite 存储从化者数据。"""

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Sequence

DB_PATH = Path(__file__).resolve().parent.parent / "servants.db"

CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS servants (
    id          INTEGER PRIMARY KEY,
    name_cn     TEXT NOT NULL,
    name_jp     TEXT NOT NULL,
    name_en     TEXT NOT NULL,
    page_url    TEXT,
    avatar      TEXT,
    card_rarity TEXT,
    servant_class   TEXT,
    np_color    TEXT,
    np_type     TEXT,
    cards       TEXT,
    faction     TEXT,
    get         TEXT,
    atk_max     INTEGER,
    hp_max      INTEGER
)
"""

CREATE_AVATAR_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS avatar (
    id          INTEGER PRIMARY KEY,
    servant_id  INTEGER NOT NULL,
    file        TEXT NOT NULL,
    left        INTEGER NOT NULL,
    top         INTEGER NOT NULL,
    right       INTEGER NOT NULL,
    bottom      INTEGER NOT NULL
)
"""

INSERT_SQL = """\
INSERT OR REPLACE INTO servants
    (id, name_cn, name_jp, name_en, page_url, avatar,
     card_rarity, servant_class, np_color, np_type,
     cards, faction, get, atk_max, hp_max)
VALUES
    (:id, :name_cn, :name_jp, :name_en, :page_url, :avatar,
     :card_rarity, :servant_class, :np_color, :np_type,
     :cards, :faction, :get, :atk_max, :hp_max)
"""


def _normalize_row(row: dict) -> dict:
    """将 JSON / CSV 行统一为数据库写入所需的格式。"""
    for key in ("cards", "faction", "get"):
        val = row.get(key)
        if isinstance(val, list):
            row[key] = "&".join(val)

    for key in ("atk_max", "hp_max"):
        val = row.get(key)
        if val is not None and val != "":
            row[key] = int(val)
        else:
            row[key] = None

    row["id"] = int(row["id"])
    return row


class ServantDB:
    """SQLite 数据库管理类，管理 servants 表。"""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self):
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.execute(CREATE_AVATAR_TABLE_SQL)
        self.conn.commit()

    # ── 导入 ──────────────────────────────────────────────

    def import_from_json(self, path: str | Path) -> int:
        """从 JSON 文件批量导入，返回实际写入的记录数。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [_normalize_row(r) for r in data]
        return self._upsert(rows)

    def import_from_csv(self, path: str | Path) -> int:
        """从 CSV 文件批量导入，返回实际写入的记录数。"""
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(_normalize_row(dict(r)))
        return self._upsert(rows)

    def import_from_json_str(self, text: str) -> int:
        """从 JSON 字符串批量导入。"""
        data = json.loads(text)
        rows = [_normalize_row(r) for r in data]
        return self._upsert(rows)

    def import_from_csv_str(self, text: str) -> int:
        """从 CSV 字符串批量导入。"""
        rows = []
        reader = csv.DictReader(io.StringIO(text))
        for r in reader:
            rows.append(_normalize_row(dict(r)))
        return self._upsert(rows)

    def _upsert(self, rows: Sequence[dict]) -> int:
        """按 id 去重后批量写入（INSERT OR REPLACE）。"""
        seen: dict[int, dict] = {}
        for r in rows:
            seen[r["id"]] = r
        self.conn.executemany(INSERT_SQL, seen.values())
        self.conn.commit()
        return len(seen)

    # ── 删除 ──────────────────────────────────────────────

    def delete_by_id(self, servant_id: int) -> bool:
        """按 id 删除单条记录，返回是否实际删除。"""
        cur = self.conn.execute("DELETE FROM servants WHERE id = ?", (servant_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_ids(self, servant_ids: Sequence[int]) -> int:
        """批量删除，返回实际删除的记录数。"""
        placeholders = ",".join("?" for _ in servant_ids)
        cur = self.conn.execute(
            f"DELETE FROM servants WHERE id IN ({placeholders})",
            servant_ids,
        )
        self.conn.commit()
        return cur.rowcount

    def delete_all(self) -> int:
        """清空整张表，返回删除的记录数。"""
        cur = self.conn.execute("DELETE FROM servants")
        self.conn.commit()
        return cur.rowcount

    # ── 查询 ──────────────────────────────────────────────

    def get_by_id(self, servant_id: int) -> dict | None:
        """按 id 查询单条记录。"""
        row = self.conn.execute(
            "SELECT * FROM servants WHERE id = ?", (servant_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        """获取全部记录。"""
        rows = self.conn.execute("SELECT * FROM servants ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """返回表中记录总数。"""
        return self.conn.execute("SELECT COUNT(*) FROM servants").fetchone()[0]

    def close(self):
        self.conn.close()

    # ── avatar 表操作 ────────────────────────────────────

    def save_avatar(self, file: str, left: int, top: int, right: int, bottom: int,
                    servant_id: int | None = None, avatar_id: int | None = None) -> int:
        """保存 avatar 记录。传 avatar_id 则更新，否则新增。返回记录 id。"""
        if avatar_id is not None:
            self.conn.execute(
                "UPDATE avatar SET file = ?, left = ?, top = ?, right = ?, bottom = ? "
                "WHERE id = ?",
                (file, left, top, right, bottom, avatar_id),
            )
            self.conn.commit()
            return avatar_id
        else:
            cur = self.conn.execute(
                "INSERT INTO avatar (servant_id, file, left, top, right, bottom) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (servant_id, file, left, top, right, bottom),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_avatars_by_servant(self, servant_id: int) -> list[dict]:
        """获取某个从者的全部 avatar 记录。"""
        rows = self.conn.execute(
            "SELECT * FROM avatar WHERE servant_id = ? ORDER BY id",
            (servant_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_avatar_by_id(self, avatar_id: int) -> dict | None:
        """按 id 查询单条 avatar 记录。"""
        row = self.conn.execute(
            "SELECT * FROM avatar WHERE id = ?", (avatar_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_avatars(self) -> list[dict]:
        """获取全部 avatar 记录。"""
        rows = self.conn.execute(
            "SELECT * FROM avatar ORDER BY servant_id, id"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_avatar(self, avatar_id: int) -> bool:
        """按 id 删除单条 avatar 记录。"""
        cur = self.conn.execute(
            "DELETE FROM avatar WHERE id = ?", (avatar_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_avatars_by_servant(self, servant_id: int) -> int:
        """删除某个从者的全部 avatar 记录。"""
        cur = self.conn.execute(
            "DELETE FROM avatar WHERE servant_id = ?", (servant_id,)
        )
        self.conn.commit()
        return cur.rowcount
