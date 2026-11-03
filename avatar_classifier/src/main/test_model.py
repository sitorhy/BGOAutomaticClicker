import argparse
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from xgboost import XGBClassifier

from main.config import RES_DIR, TEMP_DIR
from main.train import FeatureExtractor

# 默认测试目录，可通过命令行参数 --test-input 覆盖
TEST_INPUT_ROOT = RES_DIR / "test" / "grp01"

# 默认模型目录（存放 label_map.json / xgboost_model.*），可通过命令行参数 --model-root 覆盖
MODEL_ROOT = TEMP_DIR / "grp01"

# 期望从者映射文件名，位于测试目录下，格式: {图片文件名: 从者 id}
EXPECTED_MAP_FILE = "expected.json"


def load_label_map(model_root: Path) -> list[dict]:
    """加载 label_map.json，返回从者列表（索引即模型输出的类别 ID）"""
    label_map_path = model_root / "label_map.json"
    return json.loads(label_map_path.read_text(encoding="utf-8"))


def load_expected_map(test_input_root: Path) -> dict[str, int] | None:
    """加载测试目录下的期望从者映射文件，不存在则返回 None"""
    path = test_input_root / EXPECTED_MAP_FILE
    if not path.exists():
        return None
    return {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def evaluate_results(
    model_name: str,
    results: list[dict],
    img_names: list[str],
    label_map: list[dict],
    expected_map: dict[str, int],
):
    """对比预测结果与期望从者，输出正确率与置信度统计"""
    id_to_servant = {s["id"]: s for s in label_map}
    total = len(results)
    evaluated = 0
    correct = 0
    conf_all: list[float] = []
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    wrong_lines: list[str] = []

    print(f"\n{'=' * 60}")
    print(f"  正确率评估: {model_name}")
    print(f"{'=' * 60}")

    for r in results:
        name = img_names[r["img_index"]]
        expected_id = expected_map.get(name)
        if expected_id is None:
            print(f"  ? [{r['img_index'] + 1:>2}] {name}: 期望映射中无记录，跳过")
            continue
        evaluated += 1

        conf = r["confidence"]
        conf_str = f"{conf:.4f}" if conf is not None else "N/A"
        is_correct = r["matched_id"] == expected_id
        if is_correct:
            correct += 1
        if conf is not None:
            conf_all.append(conf)
            (conf_correct if is_correct else conf_wrong).append(conf)

        mark = "✓" if is_correct else "✗"
        expected_name = id_to_servant.get(expected_id, {}).get("name_cn", "?")
        print(
            f"  {mark} [{r['img_index'] + 1:>2}] {name:<20}"
            f" 期望 {expected_name}(id={expected_id})"
            f" → 预测 {r['matched_name']}(id={r['matched_id']})  置信度 {conf_str}"
        )
        if not is_correct:
            wrong_lines.append(
                f"  [{r['img_index'] + 1:>2}] {name}: 期望 {expected_name}(id={expected_id}),"
                f" 预测 {r['matched_name']}(id={r['matched_id']}), 置信度 {conf_str}"
            )

    accuracy = correct / evaluated if evaluated else 0.0
    print(f"\n  已评估: {evaluated}/{total} 张（期望映射共 {len(expected_map)} 条）")
    print(f"  正确率: {correct}/{evaluated} = {accuracy:.2%}")
    if conf_all:
        print(
            f"  平均置信度: {np.mean(conf_all):.4f}"
            f"（最低 {min(conf_all):.4f} / 最高 {max(conf_all):.4f}）"
        )
        if conf_correct:
            print(f"    - 正确样本平均置信度: {np.mean(conf_correct):.4f}")
        if conf_wrong:
            print(f"    - 错误样本平均置信度: {np.mean(conf_wrong):.4f}")
    if wrong_lines:
        print("\n  错误样本（期望 → 预测）:")
        print("\n".join(wrong_lines))
    else:
        print("  ✓ 全部预测正确")


def predict_with_json_model(model_path: Path, features: np.ndarray, label_map: list[dict]) -> list[dict]:
    """使用 xgboost_model.json 预测"""
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    pred_indices = clf.predict(features)
    pred_probs = clf.predict_proba(features)

    results = []
    for i, idx in enumerate(pred_indices):
        servant = label_map[idx]
        top_prob = pred_probs[i][idx]
        # 取概率前 3 的候选
        top3_indices = np.argsort(pred_probs[i])[-3:][::-1]
        candidates = [
            f"{label_map[j]['name_cn']}(id={label_map[j]['id']}, p={pred_probs[i][j]:.4f})"
            for j in top3_indices
        ]
        results.append({
            "img_index": i,
            "matched_name": servant["name_cn"],
            "matched_id": servant["id"],
            "confidence": float(top_prob),
            "top3_candidates": candidates,
        })
    return results


def predict_with_onnx_model(model_path: Path, features: np.ndarray, label_map: list[dict]) -> list[dict]:
    """使用 xgboost_model.onnx 预测"""
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    # 获取输出信息
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]

    # ONNX 模型通常有两个输出：label 和 probabilities
    ort_inputs = {input_name: features.astype(np.float32)}
    ort_outputs = session.run(output_names, ort_inputs)

    # 推断哪个输出是标签、哪个是概率
    # onnxmltools 导出的 XGBoost 多分类：第一个输出是 label，第二个是 probabilities
    pred_labels = ort_outputs[0].flatten()
    pred_probs = ort_outputs[1] if len(ort_outputs) > 1 else None

    results = []
    for i, label_idx in enumerate(pred_labels):
        idx = int(label_idx)
        servant = label_map[idx]

        if pred_probs is not None:
            top_prob = pred_probs[i][idx]
            top3_indices = np.argsort(pred_probs[i])[-3:][::-1]
            candidates = [
                f"{label_map[j]['name_cn']}(id={label_map[j]['id']}, p={pred_probs[i][j]:.4f})"
                for j in top3_indices
            ]
        else:
            top_prob = None
            candidates = []

        results.append({
            "img_index": i,
            "matched_name": servant["name_cn"],
            "matched_id": servant["id"],
            "confidence": float(top_prob) if top_prob is not None else None,
            "top3_candidates": candidates,
        })
    return results


def print_results(model_name: str, results: list[dict], img_names: list[str]):
    print(f"\n{'=' * 60}")
    print(f"  模型: {model_name}")
    print(f"{'=' * 60}")
    for r in results:
        i = r["img_index"]
        conf_str = f"{r['confidence']:.4f}" if r["confidence"] is not None else "N/A"
        print(f"\n[{i + 1:>2}] {img_names[i]}")
        print(f"     匹配从者: {r['matched_name']}  (id={r['matched_id']})")
        print(f"     置信度:   {conf_str}")
        if r["top3_candidates"]:
            print(f"     Top-3 候选:")
            for c in r["top3_candidates"]:
                print(f"       - {c}")


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

    extractor = FeatureExtractor.get()
    print(f"✓ 已初始化 FeatureExtractor（ResNet18, 512 维特征）")

    print(f"\n正在提取 {len(test_img_paths)} 张测试图片的特征...")
    features = extractor.extract_batch(test_img_paths, show_progress=True)
    print(f"✓ 特征提取完成，形状: {features.shape}")

    model_results: list[tuple[str, list[dict]]] = []

    # JSON 模型预测
    json_model_path = model_root / "xgboost_model.json"
    if json_model_path.exists():
        json_results = predict_with_json_model(json_model_path, features, label_map)
        print_results("xgboost_model.json", json_results, test_img_names)
        model_results.append(("xgboost_model.json", json_results))
    else:
        print(f"\n⚠ 未找到 {json_model_path}，跳过 JSON 模型预测")

    # ONNX 模型预测
    # onnx_model_path = model_root / "xgboost_model.onnx"
    # if onnx_model_path.exists():
    #     onnx_results = predict_with_onnx_model(onnx_model_path, features, label_map)
    #     print_results("xgboost_model.onnx", onnx_results, test_img_names)
    #     model_results.append(("xgboost_model.onnx", onnx_results))
    # else:
    #     print(f"\n⚠ 未找到 {onnx_model_path}，跳过 ONNX 模型预测")
    #
    # # 对比两个模型的预测结果是否一致
    # if json_model_path.exists() and onnx_model_path.exists():
    #     print(f"\n{'=' * 60}")
    #     print(f"  一致性对比")
    #     print(f"{'=' * 60}")
    #     all_match = True
    #     for jr, onr in zip(json_results, onnx_results):
    #         match = jr["matched_id"] == onr["matched_id"]
    #         if not match:
    #             all_match = False
    #         status = "✓" if match else "✗"
    #         print(
    #             f"  {status} [{jr['img_index'] + 1:>2}] {test_img_names[jr['img_index']]:<16}"
    #             f"  JSON: id={jr['matched_id']:<4}"
    #             f"  ONNX: id={onr['matched_id']:<4}"
    #         )
    #     if all_match:
    #         print("\n✓ 两个模型预测结果完全一致")
    #     else:
    #         print("\n✗ 存在预测不一致的样本")

    # 正确率与置信度评估
    if expected_map is not None:
        for model_name, results in model_results:
            evaluate_results(model_name, results, test_img_names, label_map, expected_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用已训练模型测试头像识别")
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
