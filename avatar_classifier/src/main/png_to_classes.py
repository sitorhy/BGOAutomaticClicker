#!/usr/bin/env python3
"""将徽标 PNG 原图缩放到目标尺寸后贴到透明画布，生成训练用图。

内置各目录的生成配置（画布尺寸 / 目标缩放尺寸 / 粘贴偏移），
也可通过命令行参数 --canvas --target --offset 动态覆盖：
    徽标:     画布 160x160, 目标 38x38, 偏移 (14, 7)
    冠位徽标: 画布 160x160, 目标 45x45, 偏移 (11, 4)

运行命令（项目根目录下执行）：
    .venv/Scripts/python.exe -m main.png_to_classes res/徽标 --out-dir res/foreground
    .venv/Scripts/python.exe -m main.png_to_classes res/冠位徽标 --out-dir res/foreground
    .venv/Scripts/python.exe -m main.png_to_classes res/徽标 --out-dir res/foreground --target 40x40 --offset 12,6
"""

import argparse
from pathlib import Path

from PIL import Image

# 各目录内置预设配置，键为源目录名
PRESETS: dict[str, dict] = {
    "徽标": {"canvas": (160, 160), "target": (38, 38), "offset": (14, 7)},
    "冠位徽标": {"canvas": (160, 160), "target": (45, 45), "offset": (11, 4)},
}

# 目录名未命中预设时的兜底配置
DEFAULT_CONFIG = {"canvas": (160, 160), "target": (45, 45), "offset": (11, 4)}


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
        description="徽标 PNG 转训练图：按目录预设（可用参数覆盖）缩放并贴到透明画布")
    parser.add_argument("src_dir", help="源图片目录（如 res/徽标、res/冠位徽标）")
    parser.add_argument("--canvas", type=parse_size, help="画布尺寸，如 160x160")
    parser.add_argument("--target", type=parse_size, help="目标缩放尺寸，如 38x38")
    parser.add_argument("--offset", type=parse_size, help="粘贴偏移 x,y，如 14,7")
    parser.add_argument("--out-dir", help="输出目录，默认为源目录的上级目录下的 classes")
    args = parser.parse_args(argv)

    root = Path(args.src_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1

    # 按目录名取预设，命令行参数优先覆盖
    config = dict(PRESETS.get(root.name, DEFAULT_CONFIG))
    for key in ("canvas", "target", "offset"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    print(f"配置: {root.name} -> 画布 {config['canvas']}, "
          f"目标 {config['target']}, 偏移 {config['offset']}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else root.parent / "classes"
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
