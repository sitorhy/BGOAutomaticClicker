"""
将 avatar_detect_results.json 导入数据库的 avatar 表
"""
import json
import sqlite3
from pathlib import Path

from chaldea.config import ROOT_DIR, TEMP_DIR

DB_PATH = ROOT_DIR / "servants.db"

INSERT_SQL = """\
INSERT INTO avatar (servant_id, file, left, top, right, bottom)
VALUES (?, ?, ?, ?, ?, ?)
"""


def import_avatar_json(json_path: str | Path) -> int:
    """从 JSON 文件导入 avatar 检测结果到数据库，返回导入记录数。"""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON 格式错误: 期望数组")

    conn = sqlite3.connect(str(DB_PATH))
    # conn.execute("DELETE FROM avatar")

    try:
        records = [
            (r["servant_id"], "/".join(Path(r["file"]).parts[1:]), r["left"], r["top"], r["right"], r["bottom"])
            for r in data
        ]
        conn.executemany(INSERT_SQL, records)
        conn.commit()
        print(f"导入完成: {len(records)} 条记录已写入 avatar 表")
    finally:
        conn.close()

    return len(records)


if __name__ == "__main__":
    JSON_FILE_PATH = TEMP_DIR / "avatar_detect_results.json"

    import_avatar_json(JSON_FILE_PATH)
