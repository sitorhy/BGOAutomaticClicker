import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from xgboost import XGBClassifier

from main.config import RES_DIR, TEMP_DIR
from main.train import FeatureExtractor

test_img_names = [
    "154_王哈.png",
    "175_水泥.png",
    "214_瓦尔基里.png",
    "214_瓦尔基里2.png",
    "215_CBA.png",
    "215_CBA2.png",
    "268_仇凛.png",
    "284_C呆.png",
    "284_C呆2.png",
    "351_公主.png",
    "357_RBA.png",
    "444_U奥.png"
]

test_img_paths = [RES_DIR / "test" / "grp01" / p for p in test_img_names]


def load_label_map() -> list[dict]:
    """加载 label_map.json，返回从者列表（索引即模型输出的类别 ID）"""
    label_map_path = TEMP_DIR / "label_map.json"
    return json.loads(label_map_path.read_text())


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


def main():
    label_map = load_label_map()
    print(f"✓ 已加载 label_map.json，共 {len(label_map)} 个从者类别")

    extractor = FeatureExtractor.get()
    print(f"✓ 已初始化 FeatureExtractor（ResNet18, 512 维特征）")

    print(f"\n正在提取 {len(test_img_paths)} 张测试图片的特征...")
    features = extractor.extract_batch(test_img_paths, show_progress=True)
    print(f"✓ 特征提取完成，形状: {features.shape}")

    # JSON 模型预测
    json_model_path = TEMP_DIR / "xgboost_model.json"
    if json_model_path.exists():
        json_results = predict_with_json_model(json_model_path, features, label_map)
        print_results("xgboost_model.json", json_results, test_img_names)
    else:
        print(f"\n⚠ 未找到 {json_model_path}，跳过 JSON 模型预测")

    # ONNX 模型预测
    # onnx_model_path = TEMP_DIR / "xgboost_model.onnx"
    # if onnx_model_path.exists():
    #     onnx_results = predict_with_onnx_model(onnx_model_path, features, label_map)
    #     print_results("xgboost_model.onnx", onnx_results, test_img_names)
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


if __name__ == "__main__":
    main()
