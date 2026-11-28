import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).parent.parent.parent
DEFAULT_IMG_PATH = ROOT / "res" / "input_chain_01" / "Screenshot_20261120-023336.png"
MASK_IMG_PATH = ROOT / "res" / "mask" / "检测区域蒙版.png"

LEVEL_PATTERN = re.compile(r"(\d+\s?/\s?\d+)", re.IGNORECASE)
CARD_H = 225

# 相对右上角的偏移，因为文字是右对齐的，具体取值还跟截图的分辨率有关
LEFT_OFFSET, TOP_OFFSET, RIGHT_OFFSET, BOTTOM_OFFSET = (-200, -10, 10, 215)

def make_binary_v(img_path: Path) -> Image.Image:
    # 提炼自 hsv_test.py：蒙版复合 -> HSV -> V 通道 -> 二值化
    mask_img = Image.open(MASK_IMG_PATH).convert("L")
    composited = Image.composite(
        Image.open(img_path).convert("RGB"), mask_img.convert("RGB"), mask_img
    )

    # hsv = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2HSV)
    return composited
    # return Image.fromarray(hsv[:, :, 2])


def card_box_from_text(box: list) -> list:
    # 等级文本右对齐，卡片边框由文本右上角推出（偏移与截图分辨率相关）
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    right_top = int(max(xs)), int(min(ys))
    return [
        [right_top[0] + LEFT_OFFSET, right_top[1] + TOP_OFFSET],
        [right_top[0] + RIGHT_OFFSET, right_top[1] + TOP_OFFSET],
        [right_top[0] + RIGHT_OFFSET, right_top[1] + BOTTOM_OFFSET],
        [right_top[0] + LEFT_OFFSET, right_top[1] + BOTTOM_OFFSET],
    ]

# 计算 box 中心点
def get_box_center(box: list[list[int]]):
    left_top = box[0]
    right_bottom = box[2]
    return int((right_bottom[0] + left_top[0]) / 2), int((right_bottom[1] + left_top[1]) / 2)

# 对盒子中心点进行一维聚类 划分网格
def cluster_grid(boxes: list[list[list[int]]]):
    centers = [get_box_center(box) for box in boxes]
    widths = [pts[2][0] - pts[0][0] for pts in boxes]
    heights = [pts[2][1] - pts[0][1] for pts in boxes]
    width_avg = int(sum(widths)) / len(widths)
    height_avg = int(sum(heights)) / len(heights)

    min_x = min(pts[0][0] for pts in boxes)
    min_y = min(pts[0][1] for pts in boxes)
    max_x = max(pts[2][0] for pts in boxes)
    max_y = max(pts[2][1] for pts in boxes)

    print(f"范围 ({min_x}, {min_y}, {max_x}, {max_y}) 平均尺寸 {width_avg} x {height_avg}")

    gap_x, gap_y = 25, 25

    # print(f"中心列表 {centers}")

    # 按 y 排序
    centers_sorted = sorted(centers, key=lambda c: c[1])

    # 相邻两个框中心y坐标差值大于ty认为是新的一行
    # 例如 112
    ty = int(height_avg * 0.5)
    tx = int(width_avg * 0.5)

    # 中心点已按 y 排序，两个相邻的框超出 ty 表示这里应该断行
    grid: list[list[tuple[int, int] | None]] = [
        [centers_sorted[0]]
    ]
    for (index, point_back) in enumerate(centers_sorted[1:], 1):
        point_front = centers_sorted[index - 1]
        grid_row = 0
        # 卡片之间有 gap_y = 25 - 30 px 的行间距，所以需要加上 5 的测量误差
        # 例如 dy = 241
        dy = abs(point_back[1] - point_front[1])
        while dy + point_front[1] > (min_y + ty) + (ty * grid_row * 2) + (gap_y * max(0, grid_row - 1)) + 5:
            grid_row = grid_row + 1
        while len(grid) < grid_row:
            grid.append([])
        grid[grid_row - 1].append(point_back)

    # 再每行按照 x 排序
    for (row_index, row) in enumerate(grid):
        grid[row_index] = sorted(row, key=lambda c: c[0])
        # 微调（可选），对齐底部
        if len(grid[row_index]) > 0:
            min_row_y = min(y for _, y in grid[row_index])
            for (col_index, p) in enumerate(grid[row_index]):
                x, y = p
                grid[row_index][col_index] = (x, min_row_y)
        else:
            print(f"空行 {row_index}")

    # 有概率识别失败出现空洞，计算框框的列坐标
    for (row_index, row) in enumerate(grid):
        # 防止空行
        if len(row) > 0:
            grid_row_next: list[tuple[int, int] | None] = [
                row[0]
            ]

            for (pts_index, point_back) in enumerate(row[1:], 1):
                point_front = row[pts_index - 1]
                col_index = 0
                dx = abs(point_back[0] - point_front[0])
                while dx + point_front[0] > (min_x + tx) + (tx * col_index * 2) + (gap_x * max(0, col_index - 1)) + 5:
                    col_index = col_index + 1
                while len(grid_row_next) < col_index:
                    # 预填充 补全空缺
                    # 注意这里补充的是中心点
                    center = (
                        (min_x + tx) + (tx * len(grid_row_next) * 2) + (gap_x * len(grid_row_next)),
                        (min_y + ty) + (ty * row_index * 2) + (gap_y * max(0, row_index - 1))
                    )
                    grid_row_next.append(center)
                grid_row_next[col_index - 1] = point_back

            grid[row_index] = grid_row_next

    return grid, width_avg, height_avg

def detect_cards(engine: RapidOCR, img_path: Path) -> tuple:
    result, _ = engine(np.array(make_binary_v(img_path)))
    if not result:
        return [], [], []
    text_boxes = [box for box, text, _ in result if LEVEL_PATTERN.search(text)]
    card_boxes = [card_box_from_text(box) for box in text_boxes]

    #for (index, i) in enumerate(card_boxes):
    #    print(f"box_{index}: {i}")

    return result, text_boxes, card_boxes

def draw_boxes(img: Image.Image, boxes: list, out_path: Path) -> None:
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img)
    for box in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        draw.rectangle(
            [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            outline="red",
            width=3,
        )
    img.save(out_path)

def detect_grids(engine, img_path):
    result, text_boxes, card_boxes = detect_cards(engine, img_path)

    # 划分网格
    grid, width_avg, height_avg = cluster_grid(card_boxes)

    grid_boxes = []

    for grid_index, grid_row in enumerate(grid):
        for col_index, box in enumerate(grid_row):
            if not box is None:
                grid_boxes.append([
                    [box[0] - int(width_avg / 2), box[1] - int(height_avg / 2)],
                    [box[0] + int(width_avg / 2), box[1] + int(height_avg / 2)],
                    [box[0] + int(width_avg / 2), box[1] + int(height_avg / 2)],
                    [box[0] - int(width_avg / 2), box[1] - int(height_avg / 2)]
                ])

    if len(grid_boxes) > 0:
        card_boxes = grid_boxes

    return text_boxes, result, card_boxes

def main(img_path: Path) -> None:
    out_dir = ROOT / "temp" / f"{img_path.stem}_ocr_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    make_binary_v(img_path).save(out_dir / "binary_v.png")

    engine = RapidOCR()

    text_boxes, result, card_boxes = detect_grids(engine, img_path)

    print(f"检测到 {len(card_boxes)} 个卡片")
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        print(f"\033[33m{score:.2f}] {text!r} bbox=({int(min(xs))},{int(min(ys))},{int(max(xs))},{int(max(ys))})\033[0m")
        if LEVEL_PATTERN.search(text):

            print(
                f"[{score:.2f}] {text!r} "
                f"bbox=({int(min(xs))},{int(min(ys))},{int(max(xs))},{int(max(ys))})"
            )

    mask_img = Image.open(MASK_IMG_PATH).convert("L")
    composited = Image.composite(
        Image.open(img_path).convert("RGB"), mask_img.convert("RGB"), mask_img
    )

    if text_boxes:
        draw_boxes(composited, text_boxes, out_dir / "text_boxes.png")
        print(f"共找到 {len(text_boxes)} 处疑似等级信息，已标记到 text_boxes.png")
    else:
        print("未找到疑似等级信息")

    if card_boxes:
        draw_boxes(composited.copy(), card_boxes, out_dir / "card_boxes.png")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG_PATH
    if not target.exists():
        raise FileNotFoundError(target)
    main(target)
