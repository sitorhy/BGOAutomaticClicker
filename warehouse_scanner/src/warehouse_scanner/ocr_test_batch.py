import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from ocr_test import MASK_IMG_PATH, ROOT, detect_cards, draw_boxes, detect_grids

# hsv_test 全量验证得到的列中心真值（滚动只影响行位置，列固定）
GT_COL_CENTERS = [199.0, 432.5, 667.5, 901.5, 1136.5, 1370.5]
COL_TOL = 60.0
ROW_TOL = 15.0


def col_dev(cx: float) -> float:
    center = min(GT_COL_CENTERS, key=lambda c: abs(c - cx))
    return cx - center


def row_gap_median(tops: list) -> float:
    rows = []
    for t in sorted(tops):
        if not rows or t - rows[-1] > ROW_TOL:
            rows.append(t)
    gaps = [b - a for a, b in zip(rows, rows[1:])]
    return median(gaps) if gaps else 0.0


def main() -> None:
    engine = RapidOCR()
    mask_img = Image.open(MASK_IMG_PATH).convert("L")
    images = sorted((ROOT / "res" / "input_chain_01").glob("*.png"))

    print("文件, 卡片数, 列偏差均值, 列偏差最大, 行距中位数, 超限数")
    total = 0
    for p in images:
        # _, _, card_boxes = detect_cards(engine, p)
        _, _, card_boxes = detect_grids(engine, p)
        total += len(card_boxes)

        centers = []
        tops = []
        for box in card_boxes:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            centers.append((min(xs) + max(xs)) / 2)
            tops.append(min(ys))

        abs_devs = [abs(col_dev(c)) for c in centers]
        mean_dev = sum(abs_devs) / len(abs_devs) if abs_devs else 0.0
        max_dev = max(abs_devs, default=0.0)
        over = sum(1 for d in abs_devs if d > COL_TOL)
        gap = row_gap_median(tops)

        if card_boxes:
            out_dir = ROOT / "temp" / f"{p.stem}_ocr_test"
            out_dir.mkdir(parents=True, exist_ok=True)
            composited = Image.composite(
                Image.open(p).convert("RGB"), mask_img.convert("RGB"), mask_img
            )
            draw_boxes(composited, card_boxes, out_dir / "card_boxes.png")

        print(
            f"{p.stem}, {len(card_boxes)}, {mean_dev:.1f}, {max_dev:.1f}, "
            f"{gap:.1f}, {over}"
        )

    print(f"合计: {total} 张卡片 / {len(images)} 张截图, 平均 {total / len(images):.1f}")


if __name__ == "__main__":
    main()
