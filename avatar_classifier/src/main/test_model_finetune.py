import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from main.config import RES_DIR, TEMP_DIR
from main.test_model import (
    EXPECTED_MAP_FILE,
    evaluate_results,
    load_expected_map,
    load_label_map,
    print_results,
)
from main.train_finetune import DEVICE, FineTuneModel

# 默认测试目录，可通过命令行参数 --test-input 覆盖
TEST_INPUT_ROOT = RES_DIR / "test" / "grp02"

# 默认模型目录（存放 label_map.json / finetuned_resnet18.pth），可通过命令行参数 --model-root 覆盖
MODEL_ROOT = TEMP_DIR / "grp01_finetune"

# 微调权重文件名（train_finetune.py 保存的 state_dict）
MODEL_FILE = "finetuned_resnet18.pth"


def load_finetuned_model(model_path: Path, num_classes: int) -> nn.Module:
    """重建与训练时同结构的 ResNet18（含替换后的 fc），加载 state_dict"""
    model = models.resnet18(weights=None)
    # 必须先换头再加载权重，否则 fc 的 shape (1000 vs num_classes) 对不上
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_with_finetuned_model(
    model: nn.Module,
    img_paths: list[Path],
    label_map: list[dict],
) -> list[dict]:
    """微调模型直接输出 logits，softmax 后即类别概率（无需外部分类器）"""
    # 与训练验证集流水线完全一致（Resize + ToTensor + Normalize，无增强）
    transform = FineTuneModel._base_transforms()

    results = []
    total = len(img_paths)
    for i, img_path in enumerate(img_paths):
        print(f"  [{i + 1:>3}/{total}] {img_path.name}")
        x = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = model(x).softmax(1)[0].cpu().numpy()

        idx = int(probs.argmax())
        top3_indices = np.argsort(probs)[-3:][::-1]
        candidates = [
            f"{label_map[j]['name_cn']}(id={label_map[j]['id']}, p={probs[j]:.4f})"
            for j in top3_indices
        ]
        results.append({
            "img_index": i,
            "matched_name": label_map[idx]["name_cn"],
            "matched_id": label_map[idx]["id"],
            "confidence": float(probs[idx]),
            "top3_candidates": candidates,
        })
    return results


def main(test_input_root: Path, model_root: Path):
    test_img_paths = sorted(test_input_root.glob("*.png"))
    if not test_img_paths:
        raise FileNotFoundError(f"在 {test_input_root} 下未找到任何 .png 测试图片")
    test_img_names = [p.name for p in test_img_paths]

    label_map = load_label_map(model_root)
    print(f"✓ 已加载 label_map.json，共 {len(label_map)} 个从者类别")

    expected_map = load_expected_map(test_input_root)
    if expected_map is not None:
        print(f"✓ 已加载期望从者映射 {EXPECTED_MAP_FILE}，共 {len(expected_map)} 条")
    else:
        print(f"⚠ 未找到 {test_input_root / EXPECTED_MAP_FILE}，跳过正确率评估")

    model_path = model_root / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"未找到微调权重: {model_path}")
    model = load_finetuned_model(model_path, num_classes=len(label_map))
    print(f"✓ 已加载微调模型 {model_path.name}（ResNet18 + {len(label_map)} 类分类头）")

    print(f"\n正在对 {len(test_img_paths)} 张测试图片进行预测...")
    results = predict_with_finetuned_model(model, test_img_paths, label_map)

    print_results(MODEL_FILE, results, test_img_names)

    if expected_map is not None:
        evaluate_results(MODEL_FILE, results, test_img_names, label_map, expected_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用微调后的 ResNet18 模型测试头像识别")
    parser.add_argument(
        "--test-input",
        type=Path,
        default=TEST_INPUT_ROOT,
        help=f"测试图片目录（默认: {TEST_INPUT_ROOT}）",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
        help=f"模型及 label_map.json 所在目录（默认: {MODEL_ROOT}）",
    )
    args = parser.parse_args()
    print(f"测试图片目录: {args.test_input}")
    print(f"模型目录: {args.model_root}")
    main(args.test_input, args.model_root)