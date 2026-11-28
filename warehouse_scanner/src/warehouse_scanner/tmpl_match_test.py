import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMG_PATH = ROOT / "res" / "input_chain_01" / "Screenshot_20261120-023123.png"
DEFAULT_TEMPLATE_PATH = ROOT / "res" / "徽标" / "金卡Saber.png"


def match_template(image: np.ndarray, template: np.ndarray, threshold: float) -> list[dict]:
    detections = []
    for scale in np.arange(0.25, 1.001, 0.01):
        width = round(template.shape[1] * scale)
        height = round(template.shape[0] * scale)
        resized = cv2.resize(
            template,
            (width, height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        result = cv2.matchTemplate(
            image, resized[:, :, :3], cv2.TM_CCOEFF_NORMED, mask=resized[:, :, 3]
        )
        while True:
            _, score, _, location = cv2.minMaxLoc(result)
            if score < threshold:
                break
            detections.append(
                {
                    "x": location[0],
                    "y": location[1],
                    "width": width,
                    "height": height,
                    "score": round(float(score), 6),
                    "scale": round(float(scale), 2),
                }
            )
            x, y = location
            result[
                max(0, y - height // 2):y + height // 2 + 1,
                max(0, x - width // 2):x + width // 2 + 1,
            ] = -1

    detections.sort(key=lambda item: item["score"], reverse=True)
    kept = []
    for candidate in detections:
        for detection in kept:
            x1 = max(candidate["x"], detection["x"])
            y1 = max(candidate["y"], detection["y"])
            x2 = min(candidate["x"] + candidate["width"], detection["x"] + detection["width"])
            y2 = min(candidate["y"] + candidate["height"], detection["y"] + detection["height"])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = (
                candidate["width"] * candidate["height"]
                + detection["width"] * detection["height"]
                - intersection
            )
            if intersection / union > 0.3:
                break
        else:
            kept.append(candidate)
    return kept


def tmpl_match_test(
    img_path: Path,
    template_path: Path,
    threshold: float = 0.8,
    out_dir: Path | None = None,
) -> list[dict]:
    image = cv2.imread(str(img_path))
    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if image is None or template is None:
        raise FileNotFoundError(img_path if image is None else template_path)
    if template.ndim != 3 or template.shape[2] != 4:
        raise ValueError("模板必须为带 Alpha 通道的 BGRA 图像")

    detections = match_template(image, template, threshold)
    if out_dir is None:
        out_dir = ROOT / "temp" / f"{img_path.stem}_tmpl_match_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    for index, detection in enumerate(detections, 1):
        x, y = detection["x"], detection["y"]
        right = x + detection["width"]
        bottom = y + detection["height"]
        cv2.rectangle(annotated, (x, y), (right, bottom), (0, 0, 255), 2)
        label = f"{index}: {detection['score']:.3f} @ {detection['scale']:.2f}x"
        cv2.putText(
            annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"match_{index:02d}.png"), image[y:bottom, x:right])

    output_path = out_dir / "annotated.png"
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"无法保存标注图: {output_path}")
    report = {
        "image": str(img_path),
        "template": str(template_path),
        "threshold": threshold,
        "scale_range": [0.25, 1.0],
        "scale_step": 0.01,
        "detections": detections,
    }
    (out_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for index, detection in enumerate(detections, 1):
        print(
            f"匹配 {index}: ({detection['x']}, {detection['y']}), "
            f"{detection['width']}x{detection['height']}, "
            f"score={detection['score']:.6f}, scale={detection['scale']:.2f}"
        )
    print(f"输出目录: {out_dir}")
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Alpha 掩码进行多尺度模板匹配")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMG_PATH)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    tmpl_match_test(args.image, args.template, args.threshold, args.out_dir)
