"""全量微调训练脚本：类别表来自合成流水线的目录记录，不是手工挑的从者清单。

与 mooncell/train_finetune.py 的三点差别：
+ 类别 = record（默认 temp/composite_support.json）里 completed>0 的 dir_name，一个目录一类。
  label_map.json 的数组下标即分类头标签，条目里的 id 字段用于识别时把类归并回从者，
  例如 [{dir:"1"}, {dir:"1_1"}] 是标签 0 和 1，两者都指向 id=1 的玛修。
+ 样本取自 chaldea 合成流水线的落盘结构 temp/{dir_name}/，目录里的图片全取，
  头像素材与助战卡面混在一起训，不做文件名区分。
+ 超参全部走命令行，默认值与 train_finetune.py 一致；不再硬编码小规模 servant_ids。

模型与数据管线复用 mooncell.train_finetune，ResNet 那套不在这里抄第二份。

用法：
    python -m chaldea.train_finetune_full
    python -m chaldea.train_finetune_full --record temp/composite_support.json temp/composite.json
"""
import argparse
import json
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from chaldea.composite import DB_PATH, Servant
from chaldea.config import TEMP_DIR
from mooncell.train_finetune import DEVICE, AvatarDataset, FineTuneModel

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}
LABEL_MAP_FILE = "label_map.json"
# 与 test_model_finetune.py 的权重文件名保持一致，模型目录一换就能复用现成推理脚本
MODEL_FILE = "finetuned_resnet18.pth"

DEFAULT_IMAGES_DIR = TEMP_DIR
DEFAULT_OUTPUT_PATH = TEMP_DIR / "finetune_full"
DEFAULT_RECORD_FILES = [TEMP_DIR / "composite_support.json"]


def count_images(dir_path: Path) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_record_dirs(record_files: list[Path]) -> dict[str, int]:
    """合并多份合成记录，返回 dir_name -> servant_id；completed<=0 的目录样本不完整，不进训练集"""
    dirs: dict[str, int] = {}
    for record_file in record_files:
        if not record_file.exists():
            raise FileNotFoundError(f"记录文件不存在: {record_file}")
        rows = json.loads(record_file.read_text(encoding="utf-8"))
        incomplete = [row["dir_name"] for row in rows if row.get("completed", 0) <= 0]
        if incomplete:
            print(f"跳过 {len(incomplete)} 个未完成目录（completed<=0）: {', '.join(incomplete[:10])}")
        for row in rows:
            if row.get("completed", 0) <= 0:
                continue
            dirs.setdefault(row["dir_name"], int(row["servant_id"]))
        print(f"✓ 已读取记录 {record_file}（{len(rows)} 行）")
    return dirs


def resolve_classes(record_files: list[Path], images_dir: Path) -> list[tuple[str, int]]:
    """把记录里的目录翻成 (dir_name, servant_id) 类别表，只保留真的有图片的目录"""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片根目录不存在: {images_dir}")

    dirs = load_record_dirs(record_files)
    classes = []
    empty = []
    for dir_name, servant_id in dirs.items():
        if count_images(images_dir / dir_name) == 0:
            empty.append(dir_name)
            continue
        classes.append((dir_name, servant_id))
    if empty:
        print(f"跳过 {len(empty)} 个记录里的目录（{images_dir} 下取不到图片）: {', '.join(empty[:10])}")

    # 先按从者 id 再按目录名排序，保证 label_map.json 的下标稳定
    classes.sort(key=lambda c: (c[1], c[0]))

    extra = sorted(d.name for d in images_dir.iterdir() if d.is_dir() and d.name not in dirs and count_images(d))
    if extra:
        print(f"⚠ {len(extra)} 个图片目录不在记录里（首轮建标签表时不会成为类别）: {', '.join(extra[:10])}")
    return classes


def load_servants(db_path: Path) -> dict[int, Servant]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return {row["id"]: Servant(**dict(row)) for row in conn.execute("select * from servants")}
    finally:
        conn.close()


def build_label_map(classes: list[tuple[str, int]], servants: dict[int, Servant], output_path: Path) -> list[Servant]:
    label_map = []
    for dir_name, servant_id in classes:
        servant = servants.get(servant_id)
        if servant is None:
            print(f"跳过 {dir_name}：servants 表无 id={servant_id}")
            continue
        entry = Servant(**asdict(servant))
        entry.dir_name = dir_name
        label_map.append(entry)

    file_path = output_path / LABEL_MAP_FILE
    # 包含日文，不能用默认编码写
    file_path.write_text(json.dumps([asdict(s) for s in label_map], indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"✓ 标签映射表已保存到 {file_path}（{len(label_map)} 类）")
    return label_map


def load_label_map(classes: list[tuple[str, int]], images_dir: Path, output_path: Path) -> list[Servant]:
    """复用已有 label_map.json，保持类别下标与分类头尺寸不变（缺样本的类只是本轮训不到）"""
    file_path = output_path / LABEL_MAP_FILE
    label_map = [Servant(**row) for row in json.loads(file_path.read_text(encoding="utf-8"))]

    empty = [s.dir_name for s in label_map if count_images(images_dir / s.dir_name) == 0]
    if empty:
        print(f"⚠ {len(empty)} 类取不到样本，仍占着分类头下标: {', '.join(empty[:10])}")
    known = {s.dir_name for s in label_map}
    new = sorted(dir_name for dir_name, _ in classes if dir_name not in known)
    if new:
        print(f"⚠ 记录里新增 {len(new)} 个目录未进本轮（加入会打乱标签下标）: {', '.join(new[:10])}")
    print(f"✓ 已加载 {file_path}（{len(label_map)} 类）")
    return label_map


def split_dir_images(
    dir_path: Path,
    label: int,
    test_size: float,
    random_state: int,
    max_per_dir: int,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    files = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if max_per_dir > 0 and len(files) > max_per_dir:
        # 用目录名派生种子，同一目录每次重跑取到同一批样本
        rng = random.Random(f"{dir_path.name}:{random_state}")
        files = rng.sample(files, max_per_dir)
    if len(files) < 2:
        return files, [], [label] * len(files), []
    return train_test_split(files, [label] * len(files), test_size=test_size, random_state=random_state)


def build_loaders(label_map: list[Servant], images_dir: Path, model: FineTuneModel, args) -> tuple[DataLoader, DataLoader]:
    labels = {s.dir_name: i for i, s in enumerate(label_map)}
    trained = [s for s in label_map if count_images(images_dir / s.dir_name) > 0]
    if not trained:
        raise SystemExit(f"label_map.json 的 {len(label_map)} 个类别在 {images_dir} 下都取不到样本")
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [
            pool.submit(split_dir_images, images_dir / s.dir_name, labels[s.dir_name], args.test_size,
                        args.random_state, args.max_per_dir)
            for s in trained
        ]
        splits = [f.result() for f in futures]

    train_files = [p for split in splits for p in split[0]]
    val_files = [p for split in splits for p in split[1]]
    train_labels = [l for split in splits for l in split[2]]
    val_labels = [l for split in splits for l in split[3]]

    per_class = [len(t) + len(v) for t, v in zip((s[0] for s in splits), (s[1] for s in splits))]
    print(f"分类头类别数: {len(label_map)}（归并到 {len({s.id for s in label_map})} 个从者 id），本轮有样本 {len(trained)} 类")
    print(f"单类样本数: 最少 {min(per_class)}, 最多 {max(per_class)}, 合计 {sum(per_class)}")
    print(f"训练集: {len(train_files)} 张 / 验证集: {len(val_files)} 张")
    if not train_files or not val_files:
        raise ValueError("训练集或验证集为空，检查 --images-dir 与 --record")

    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=DEVICE == "cuda")
    train_loader = DataLoader(AvatarDataset(train_files, train_labels, model.transform_train), shuffle=True, **common)
    val_loader = DataLoader(AvatarDataset(val_files, val_labels, model.transform_eval), shuffle=False, **common)
    return train_loader, val_loader


def train(images_dir: Path, output_path: Path, label_map: list[Servant], args) -> None:
    model = FineTuneModel(num_classes=len(label_map), lr=args.lr, weight_decay=args.weight_decay)
    train_loader, val_loader = build_loaders(label_map, images_dir, model, args)

    print(f"\n开始微调（device={DEVICE}, epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}）...")
    best_acc, best_epoch = 0.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = model.train_epoch(train_loader)
        val_acc, _ = model.evaluate(val_loader)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.model.state_dict(), output_path / MODEL_FILE)
        print(f"Epoch {epoch:>3}/{args.epochs}  loss={loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_acc:.4f}(e{best_epoch})  "
              f"用时 {time.perf_counter() - started:.0f}s")

    print(f"\n✓ 最佳验证准确率 {best_acc:.4f}（第 {best_epoch} 轮），权重已存到 {output_path / MODEL_FILE}")

    model.model.load_state_dict(torch.load(output_path / MODEL_FILE, map_location=DEVICE, weights_only=True))
    val_acc, per_class = model.evaluate(val_loader)
    worst = sorted(per_class.items(), key=lambda kv: kv[1][1] / kv[1][0])[:args.show_classes]
    print(f"\n重载最佳权重后验证准确率: {val_acc:.4f}")
    print(f"验证准确率最低的 {len(worst)} 类:")
    for label_idx, (cnt, ok) in worst:
        servant = label_map[label_idx]
        print(f"  label={label_idx:>4} dir={servant.dir_name:<10} id={servant.id:<5} "
              f"{servant.name_cn}: {ok}/{cnt} = {ok / cnt:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="按合成记录标定标签的全量微调训练")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                        help=f"训练集图片根目录，每个类别一个子目录（默认: {DEFAULT_IMAGES_DIR}）")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help=f"权重与 label_map.json 输出目录（默认: {DEFAULT_OUTPUT_PATH}）")
    parser.add_argument("--record", type=Path, nargs="+", default=DEFAULT_RECORD_FILES,
                        help=f"合成记录文件，可给多份取并集（默认: {DEFAULT_RECORD_FILES}）")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help=f"从者数据库（默认: {DB_PATH}）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2, help="每个类别划入验证集的比例")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader 读图进程数")
    parser.add_argument("--random-state", type=int, default=42, help="划分与抽样种子")
    parser.add_argument("--max-per-dir", type=int, default=0, help="每个类别最多采样多少张，0 表示不限")
    parser.add_argument("--max-dirs", type=int, default=0,
                        help="只把前 N 个目录收进 label_map.json，用于冒烟测试，0 表示全部；已有标签表时不生效")
    parser.add_argument("--show-classes", type=int, default=20, help="收尾打印最差类别的条数")
    parser.add_argument("--overwrite-label-map", action="store_true",
                        help="重建 label_map.json（会改变分类头下标，旧权重作废）")
    args = parser.parse_args()

    print(f"图片目录: {args.images_dir}\n输出目录: {args.output_path}\n记录文件: {args.record}")
    args.output_path.mkdir(parents=True, exist_ok=True)

    classes = resolve_classes(args.record, args.images_dir)
    if args.max_dirs > 0:
        classes = classes[:args.max_dirs]
    if not classes:
        raise SystemExit("记录与图片目录的交集为空，没有可训练的类别")
    print(f"候选类别: {len(classes)} 个目录")

    if (args.output_path / LABEL_MAP_FILE).exists() and not args.overwrite_label_map:
        label_map = load_label_map(classes, args.images_dir, args.output_path)
    else:
        label_map = build_label_map(classes, load_servants(args.db_path), args.output_path)
    if not label_map:
        raise SystemExit("标签映射表为空，检查 --record 与 --images-dir")

    train(args.images_dir, args.output_path, label_map, args)


if __name__ == "__main__":
    main()
