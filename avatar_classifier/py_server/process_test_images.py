"""将 test 目录下的指定图片处理为前景图标。

对每个源图片：创建 132 x 145 的透明背景画布，将图片缩放为 36 x 36，
粘贴到画布的 (x=2, y=2) 处，并另存到 foreground 目录（存在则覆盖）。
"""

from pathlib import Path

from PIL import Image

# 项目根目录（脚本位于根目录下）
ROOT_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT_DIR / "res" / "test"
FOREGROUND_DIR = ROOT_DIR / "res" / "foreground"

# 需要处理的源文件名
SOURCE_FILES = [
    "冠位Alterego.png",
    "冠位Archer.png",
    "冠位Assassin.png",
    "冠位Avenger.png",
    "冠位Berserker.png",
    "冠位Caster.png",
    "冠位Foreigner.png",
    "冠位Lancer.png",
    "冠位MoonCancer.png",
    "冠位Pretender.png",
    "冠位Rider.png",
    "冠位Ruler.png",
    "冠位Saber.png",
    "冠位Shielder.png",
    "冠位UnBeast.png",
    "金卡Shielder.png",
    "铜卡Shielder.png",
    "银卡Shielder.png",
]

CANVAS_SIZE = (132, 145)
ICON_SIZE = (36, 36)
PASTE_POS = (2, 2)


def process_image(src_path: Path, dst_path: Path) -> None:
    # 创建透明背景画布
    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))

    # 打开源图并缩放为 36 x 36
    with Image.open(src_path) as img:
        icon = img.convert("RGBA").resize(ICON_SIZE, Image.LANCZOS)

    # 粘贴到指定位置（使用 icon 自身作为掩码以保留透明度）
    canvas.paste(icon, PASTE_POS, icon)

    # 另存到 foreground 目录，存在则覆盖
    canvas.save(dst_path, "PNG")


def main() -> None:
    FOREGROUND_DIR.mkdir(parents=True, exist_ok=True)

    for name in SOURCE_FILES:
        src_path = TEST_DIR / name
        dst_path = FOREGROUND_DIR / name

        if not src_path.exists():
            print(f"[跳过] 源文件不存在: {src_path}")
            continue

        process_image(src_path, dst_path)
        print(f"[完成] {src_path.name} -> {dst_path}")


if __name__ == "__main__":
    main()
