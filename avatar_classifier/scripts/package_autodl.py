"""打 autoDL 全量包：只收 chaldea 合成+训练真正用到的东西。

为什么不直接打包整个仓库：
+ mooncell 侧的 train.py 会拉起 xgboost / scikit-learn，face-detect-* 是另一套前后端，
  远端机器一概用不上；chaldea 的依赖已经隔离到 torch / torchvision / Pillow / numpy。
+ res/assets 有 1.7G，另外单独压缩上传，本脚本只放合成需要的其余资源（掩码、底板、徽标、背景）。
+ 远端不 pip install 本项目，run_all.sh 直接用 PYTHONPATH=src 跑，省掉构建后端和联网拉代码。

用法：
    uv run python scripts/package_autodl.py                     # 产出 dist/autodl_chaldea.tar.gz
    uv run python scripts/package_autodl.py --no-background     # 不含 res/background（142M）
    uv run python scripts/package_autodl.py --no-torch-weights  # 不含 resnet18 预训练权重
"""
import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "autodl_chaldea"

# chaldea 运行链的模块清单：合成入口 + 配置 + 链路 + 两条流水线 + 微调
CHALDEA_MODULES = [
    "__main__.py",
    "config.py",
    "chain.py",
    "composite.py",
    "composite_support.py",
    "finetune.py",
    "train_finetune_full.py",
]
# 合成必读的资源目录（assets 除外，另外单独传）
RES_DIRS = ["mask", "foreground", "foreground_support"]
BACKGROUND_DIR = "background"
# torchvision 首次构造 resnet18 会去 download.pytorch.org 取权重，远端多半连不上，预置进包
TORCH_WEIGHTS = ["resnet18-f37072fd.pth"]

REQUIREMENTS = """\
# 合成与训练实际 import 的三方库。
# torch / torchvision 用镜像自带的 CUDA 版本，不要在这里 pip 装，否则会覆盖成 CPU 版。
pillow>=11
numpy>=1.26
"""

# 纯字面量直接写盘，不要 .format()：shell 的 ${VAR:-default} 单个大括号就是最终形态。
RUN_SCRIPT = """\
#!/usr/bin/env bash
# autoDL 全量：合成所有从者的头像卡与助战卡，然后全量微调。
# 解压后一般只需要改下面两行指向资源盘和数据盘。
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"

# res/assets 解压到 $CHALDEA_RES_DIR/assets，其余资源本包已经带好
export CHALDEA_RES_DIR="${CHALDEA_RES_DIR:-$PKG_DIR/res}"
# 全量合成有几十 GB，务必指到数据盘
export CHALDEA_TEMP_DIR="${CHALDEA_TEMP_DIR:-/root/autodl-tmp/temp}"

export PYTHONPATH="$PKG_DIR/src"
export TORCH_HOME="$PKG_DIR/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-20}"

$PYTHON -m chaldea all
$PYTHON -m chaldea support all
$PYTHON -m chaldea.train_finetune_full \\
    --record "$CHALDEA_TEMP_DIR/composite.json" "$CHALDEA_TEMP_DIR/composite_support.json" \\
    --output-path "$CHALDEA_TEMP_DIR/finetune_full" \\
    --epochs "$EPOCHS" --workers 8

echo "权重与标签表: $CHALDEA_TEMP_DIR/finetune_full"
"""


def stage_package(with_background: bool, with_torch_weights: bool) -> Path:
    stage = ROOT / "dist" / PACKAGE_NAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    src_out = stage / "src" / "chaldea"
    src_out.mkdir(parents=True)
    for name in CHALDEA_MODULES:
        src = ROOT / "src" / "chaldea" / name
        if not src.is_file():
            raise SystemExit(f"缺少模块 {src}，检查 CHALDEA_MODULES 清单")
        shutil.copy2(src, src_out / name)
    print(f"代码: {len(CHALDEA_MODULES)} 个模块 -> src/chaldea/")

    shutil.copy2(ROOT / "servants.db", stage / "servants.db")
    print("数据库: servants.db")

    res_dirs = RES_DIRS + ([BACKGROUND_DIR] if with_background else [])
    for name in res_dirs:
        src = ROOT / "res" / name
        if not src.is_dir():
            raise SystemExit(f"缺少资源目录 {src}")
        shutil.copytree(src, stage / "res" / name)
    print(f"资源: {', '.join(res_dirs)}（res/assets 需另外解压到 $CHALDEA_RES_DIR/assets）")

    if with_torch_weights:
        cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        copied = [f for f in TORCH_WEIGHTS if (cache / f).is_file()]
        if copied:
            out = stage / "torch" / "hub" / "checkpoints"
            out.mkdir(parents=True)
            for f in copied:
                shutil.copy2(cache / f, out / f)
            print(f"预训练权重: {', '.join(copied)}")
        else:
            print(f"跳过预训练权重：本地缓存 {cache} 里没有，远端首次运行需要能联网下载")

    (stage / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (stage / "run_all.sh").write_text(RUN_SCRIPT, encoding="utf-8", newline="\n")
    (stage / "README.md").write_text(
        "# 解到远端之后\n"
        "```shell\n"
        "tar -xzf autodl_chaldea.tar.gz && cd autodl_chaldea\n"
        "pip install -r requirements.txt          # 只装 pillow / numpy\n"
        "tar -xzf assets.tar.gz -C res/           # 解压后是 res/assets/<从者id>/...\n"
        "bash run_all.sh                          # 路径不对就改 run_all.sh 顶部两行\n"
        "```\n"
        "+ 头像链与助战链都要跑，训练集混放两类素材；只跑一条等于换了个分布去训练。\n"
        "+ 产物落盘在 $CHALDEA_TEMP_DIR：temp/{dir_name}/ 是素材，composite*.json 是类别记录，\n"
        "  finetune_full/ 是权重与 label_map.json。\n"
        "+ 助战卡识别时只喂卡面上半部分（立绘约占高度 60%），下半部的礼装美术会把预测带偏。\n",
        encoding="utf-8",
        newline="\n",
    )
    return stage


def make_archive(stage: Path) -> Path:
    out = stage.parent / f"{stage.name}.tar.gz"
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w:gz") as tar:
        tar.add(stage, arcname=PACKAGE_NAME)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 autoDL 全量合成+训练所需的最小集合")
    parser.add_argument("--no-background", action="store_true", help="不打包 res/background（142M）")
    parser.add_argument("--no-torch-weights", action="store_true", help="不打包本地 resnet18 预训练权重")
    args = parser.parse_args()

    stage = stage_package(not args.no_background, not args.no_torch_weights)
    archive = make_archive(stage)
    print(f"\n归档: {archive}  {archive.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
