"""
检查 avatar 表中的裁剪结果，将裁剪后的头像输出到 temp/check 目录
"""
import sqlite3
from pathlib import Path

from PIL import Image

from chaldea.config import RES_DIR, ROOT_DIR

DB_PATH = ROOT_DIR / "servants.db"
OUTPUT_DIR = ROOT_DIR / "temp" / "check"


def check_avatar():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM avatar ORDER BY servant_id, id").fetchall()
    conn.close()

    print(f"共 {len(rows)} 条 avatar 记录")

    assets_dir = RES_DIR / "assets"
    success = 0
    skipped = 0

    for row in rows:
        file_path = row["file"].replace("\\", "/")
        img_path = assets_dir / str(row["servant_id"]) / file_path

        if not img_path.exists():
            skipped += 1
            continue

        left = row["left"]
        top = row["top"]
        right = row["right"]
        bottom = row["bottom"]

        img = Image.open(img_path)
        cropped = img.crop((left, top, right, bottom))

        out_name = f"{row['servant_id']}_{Path(file_path).stem}.png"
        out_path = OUTPUT_DIR / out_name
        cropped.save(out_path)
        success += 1

    print(f"裁剪完成: {success} 张, 跳过: {skipped} 张")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    check_avatar()
