import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import label

ROOT = Path(__file__).parent.parent.parent
DEFAULT_IMG_PATH = ROOT / "res" / "input_chain_01" / "Screenshot_20261120-023458.png"
MASK_IMG_PATH = ROOT / "res" / "mask" / "检测区域蒙版.png"

CARD_COLS = 6
CARD_ROWS = 3
CARD_W = 224
CARD_H = 225

def hsv_test(img_path: Path) -> None:
    out_dir = ROOT / "temp" / f"{img_path.stem}_hsv_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    mask_img = Image.open(MASK_IMG_PATH).convert("L")

    input_img = Image.open(img_path).convert("RGB")
    composited = Image.composite(input_img, mask_img.convert("RGB"), mask_img)
    composited_path = out_dir / "input_chain.png"
    composited.save(composited_path)

    img = cv2.imread(str(composited_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(str(out_dir / "channel_h.png"), hsv[:, :, 0])
    cv2.imwrite(str(out_dir / "channel_s.png"), hsv[:, :, 1])
    cv2.imwrite(str(out_dir / "channel_v.png"), hsv[:, :, 2])

    # S 通道：二值化 -> 膨胀 -> 连通域画框
    grey_s = Image.open(out_dir / "channel_s.png").convert("L")
    table = [0 if i < 180 else 255 for i in range(256)]
    binary_img = grey_s.point(table, mode="L")
    binary_img.save(out_dir / "binary_s_test.png")

    expanded_white = binary_img.filter(ImageFilter.MaxFilter(size=3))
    expanded_white.save(out_dir / "binary_s_test_expanded.png")

    mask = np.array(expanded_white) > 0
    labeled, num_features = label(mask, structure=np.ones((3, 3), dtype=int))
    print(f"S 通道合并后共检测到 {num_features} 个区域")

    vis = expanded_white.convert("RGB")
    draw = ImageDraw.Draw(vis)
    for region_id in range(1, num_features + 1):
        ys, xs = np.where(labeled == region_id)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    vis.save(out_dir / "binary_s_test_expanded_regions.png")

    # V 通道：二值化 -> 膨胀 -> 降噪 -> 轮廓碎片 -> 网格拟合
    grey_v = Image.open(out_dir / "channel_v.png").convert("L")
    table = [0 if i < 220 else 255 for i in range(256)]
    binary_img = grey_v.point(table, mode="L")
    binary_img.save(out_dir / "binary_v_test.png")

    expanded_white = binary_img.filter(ImageFilter.MaxFilter(size=5))
    expanded_white.save(out_dir / "binary_v_test_expanded.png")

    # 降噪：过滤掉边界框过小的连通区域
    min_w, min_h = 32, 32
    mask = np.array(expanded_white) > 0
    labeled, num_features = label(mask, structure=np.ones((3, 3), dtype=int))
    denoised = mask.copy()
    for region_id in range(1, num_features + 1):
        ys, xs = np.where(labeled == region_id)
        if xs.max() - xs.min() + 1 < min_w and ys.max() - ys.min() + 1 < min_h:
            denoised[labeled == region_id] = False
    denoised_img = Image.fromarray((denoised * 255).astype(np.uint8))
    denoised_img.save(out_dir / "binary_v_test_denoised.png")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG_PATH
    if not target.exists():
        raise FileNotFoundError(target)
    hsv_test(target)
