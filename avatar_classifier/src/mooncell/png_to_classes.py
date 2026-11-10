#!/usr/bin/env python3
"""将徽标 PNG 原图缩放到目标尺寸后贴到透明画布，生成训练用图。

按 --preset 选择内置预设（画布尺寸 / 目标缩放尺寸 / 粘贴偏移），
从 --input-dir 读取图片，处理结果输出到 --out-dir；
也可通过命令行参数 --canvas --target --offset 动态覆盖预设：
    徽标:         画布 160x160, 目标 38x38, 偏移 (14, 7)
    冠位徽标:     画布 160x160, 目标 45x45, 偏移 (11, 4)
    助战徽标:     画布 170x400, 目标 62x62, 偏移 (6, 7)
    冠位助战徽标: 画布 170x400, 目标 76x76, 偏移 (0, 3)

运行命令（项目根目录下执行）：
    .venv/Scripts/python.exe -m mooncell.png_to_classes --preset 徽标 --input-dir res/徽标 --out-dir res/foreground
    .venv/Scripts/python.exe -m mooncell.png_to_classes --preset 冠位徽标 --input-dir res/冠位徽标 --out-dir res/foreground
    .venv/Scripts/python.exe -m mooncell.png_to_classes --preset 助战徽标 --input-dir res/助战徽标 --out-dir res/foreground_support
    .venv/Scripts/python.exe -m mooncell.png_to_classes --preset 冠位助战徽标 --input-dir res/冠位徽标 --out-dir res/foreground_support
    .venv/Scripts/python.exe -m mooncell.png_to_classes --preset 徽标 --input-dir res/徽标 --out-dir res/foreground --target 40x40 --offset 12,6
"""

import argparse
from pathlib import Path

from PIL import Image

# 各预设的生成配置，键为预设名
PRESETS: dict[str, dict] = {
    "徽标": {"canvas": (160, 160), "target": (38, 38), "offset": (14, 7)},
    "冠位徽标": {"canvas": (160, 160), "target": (45, 45), "offset": (11, 4)},
    "助战徽标": {"canvas": (170, 400), "target": (62, 62), "offset": (6, 7)},
    "冠位助战徽标": {"canvas": (170, 400), "target": (76, 76), "offset": (0, 3)},
}


def convert(src: Path, dst_dir: Path, canvas_size: tuple[int, int],
            target_size: tuple[int, int], offset: tuple[int, int]) -> Path:
    with Image.open(src) as im:
        im = im.convert("RGBA")
        im = im.resize(target_size, Image.LANCZOS)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.paste(im, offset, im)  # 第三参数为 alpha 遮罩, 保留透明边缘

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    canvas.save(dst)
    return dst


def parse_size(text: str) -> tuple[int, int]:
    """解析 "160x160" / "160,160" / "160 160" 为 (宽, 高)。"""
    parts = [p for p in text.replace("x", ",").replace(" ", ",").split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"尺寸格式错误: {text}（应形如 160x160）")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"尺寸必须为正数: {text}")
    return w, h


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="徽标 PNG 转训练图：按 --preset 加载预设，处理 --input-dir 图片并输出到 --out-dir")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS),
                        help="预设名（如 徽标、冠位徽标、助战徽标、冠位助战徽标）")
    parser.add_argument("--input-dir", required=True, help="源图片目录（如 res/徽标）")
    parser.add_argument("--out-dir", required=True, help="输出目录（如 res/foreground）")
    parser.add_argument("--canvas", type=parse_size, help="覆盖画布尺寸，如 160x160")
    parser.add_argument("--target", type=parse_size, help="覆盖目标缩放尺寸，如 38x38")
    parser.add_argument("--offset", type=parse_size, help="覆盖粘贴偏移 x,y，如 14,7")
    args = parser.parse_args(argv)

    # 按预设名加载配置，命令行参数优先覆盖
    config = dict(PRESETS[args.preset])
    for key in ("canvas", "target", "offset"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"预设: {args.preset} -> 源 {root}, 画布 {config['canvas']}, "
          f"目标 {config['target']}, 偏移 {config['offset']}, 输出 {out_dir}")

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".png")
    if not files:
        print(f"{root} 下没有找到 PNG 图片")
        return 1

    for src in files:
        dst = convert(src, out_dir, config["canvas"], config["target"], config["offset"])
        print(f"  {src.relative_to(root)} -> {dst.name}")
    print(f"完成: {len(files)} 张, 输出目录 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
