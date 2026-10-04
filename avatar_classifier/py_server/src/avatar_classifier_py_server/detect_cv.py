"""
头像位置检测：OpenCV 多尺度模板匹配

在立绘（目标图像 target）中搜索头像（模板图像 template）的位置，
支持掩码（mask）排除边框干扰。在 [min_scale, max_scale] 间线性插值取 steps 个尺度，
逐尺度缩放模板后做归一化匹配，取匹配度最高者为最终结果。

参考 test.py 的 AvatarClipChainNode / test_avatar_cv_detect 实现。
"""
from pathlib import Path

import numpy as np
import cv2

# res 根目录：py_server/res（detect_cv.py 位于 py_server/src/avatar_classifier_py_server/ 下）
_RES_ROOT = Path(__file__).resolve().parents[2] / "res"
# YuNet 人脸检测模型默认路径（OpenCV Zoo: face_detection_yunet_2023mar.onnx）
DEFAULT_FACE_MODEL = _RES_ROOT / "model" / "face_detection_yunet_2023mar.onnx"


def _imread_unicode(path: str | Path, flags: int) -> np.ndarray:
    """读取图片，兼容 Windows 中文路径。

    cv2.imread 在 Windows 上不支持中文路径，会静默返回 None，
    因此先用 np.fromfile 读字节流，再交给 cv2.imdecode 解码。
    """
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if arr is None:
        raise ValueError(f"图片解析失败: {path}")
    return arr


def detect_avatar(
        target: str | Path,
        template: str | Path,
        mask: str | Path | None = None,
        min_scale: float = 0.9,
        max_scale: float = 1.25,
        steps: int = 10,
) -> list[dict]:
    """多尺度模板匹配，检测头像在立绘中的位置。

    在 [min_scale, max_scale] 之间线性插值取 steps 个缩放比，将模板（头像）
    按各缩放比 resized 后在目标（立绘）中做带掩码的归一化模板匹配，
    统计头像在各尺度下的最大匹配度，返回按匹配度降序的结果列表。

    Args:
        target: 目标图像路径（立绘）
        template: 模板图像路径（头像）
        mask: 掩码图像路径，灰度图（白色=参与匹配，黑色=忽略），用于抠出头像核心内容、排除边框干扰；None 表示不使用掩码
        min_scale: 最小缩放比（相对模板原尺寸），默认 0.9
        max_scale: 最大缩放比，默认 1.25
        steps: 缩放步数（在 min~max 之间均匀取 steps 个尺度），默认 10

    Returns:
        每个缩放尺度的匹配结果列表，按匹配度降序排列。每项包含:
            scale         - 缩放比
            template_size - (宽, 高) 缩放后的模板尺寸
            score         - 匹配度，越大越好
            x, y          - 匹配区域左上角在目标图中的坐标
            w, h          - 匹配区域的宽高
            rect          - (x, y, x+w, y+h) 便捷元组
    """
    # ── 读取图像 ──
    # target / template 按彩色读入（BGR），mask 按灰度读入
    target_img = _imread_unicode(target, cv2.IMREAD_COLOR)
    template_img = _imread_unicode(template, cv2.IMREAD_COLOR)
    mask_img: np.ndarray | None = None
    if mask is not None:
        mask_img = _imread_unicode(mask, cv2.IMREAD_GRAYSCALE)

    th, tw = template_img.shape[:2]  # 模板原始高、宽
    target_h, target_w = target_img.shape[:2]

    # 带掩码时 OpenCV 仅可靠支持 TM_SQDIFF / TM_CCORR_NORMED（见 test.py 注释），
    # 这里统一用 TM_CCORR_NORMED，最大值即最佳匹配；无掩码时用 TM_CCOEFF_NORMED
    method = cv2.TM_CCORR_NORMED if mask_img is not None else cv2.TM_CCOEFF_NORMED

    results: list[dict] = []

    # ── 线性插值取 steps 个尺度：从 min_scale 到 max_scale 均匀分布 ──
    scales = np.linspace(min_scale, max_scale, steps)
    for scale in scales:
        # 当前尺度下的模板尺寸（至少 1 像素）
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))

        # 缩放后的模板超出目标尺寸时无法匹配，跳过该尺度
        if new_h > target_h or new_w > target_w:
            continue

        # ── 缩放模板（双线性插值）与掩码（最近邻插值，保持二值不引入中间值）──
        scaled_template = cv2.resize(template_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        scaled_mask: np.ndarray | None = None
        if mask_img is not None:
            scaled_mask = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # ── 归一化模板匹配 ──
        if scaled_mask is not None:
            result = cv2.matchTemplate(target_img, scaled_template, method, mask=scaled_mask)
        else:
            result = cv2.matchTemplate(target_img, scaled_template, method)

        # 获取最佳匹配位置
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        # │          │        │      │
        # │          │        │      └── 最大值的位置 (x, y)
        # │          │        └── 最小值的位置 (x, y)
        # │          └── 最大值（匹配分数）
        # └── 最小值（匹配分数）

        best_x, best_y = int(max_loc[0]), int(max_loc[1])

        results.append({
            "scale": float(scale),
            "template_size": (new_w, new_h),
            "score": float(max_val),
            "x": best_x,
            "y": best_y,
            "w": new_w,
            "h": new_h,
            "rect": (best_x, best_y, best_x + new_w, best_y + new_h),
        })

    if not results:
        raise ValueError("所有缩放尺度均超出目标图像尺寸，无法匹配")

    # 按匹配度降序，头像在立绘中的最大匹配度即 results[0]
    results.sort(key=lambda r: r["score"], reverse=True)

    best = results[0]
    x1, y1, x2, y2 = best["rect"]
    print(f"\n🎯 最佳匹配:")
    print(f"  缩放比:   {best['scale']:.3f}")
    print(f"  模板尺寸: {best['template_size'][0]}×{best['template_size'][1]}")
    print(f"  匹配度:   {best['score']:.4f}")
    print(f"  位置:     ({best['x']}, {best['y']})")
    print(f"  区域:     ({x1}, {y1}) → ({x2}, {y2})")

    return results


def detect_face(
        image: str | Path,
        model_path: str | Path | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
) -> list[dict]:
    """使用 OpenCV 的 FaceDetectorYN（YuNet 人脸检测模型）检测图像中的人脸。

    FaceDetectorYN.create(model, config, input_size, score_threshold, nms_threshold, top_k)
    中 config 传空字符串（YuNet 为单文件 ONNX，无需额外配置文件），input_size 为 (宽, 高)。
    detect() 返回的 faces 为 N×15 数组，每行依次为：
        [0]x [1]y [2]w [3]h
        [4][5] 左眼   [6][7] 右眼   [8][9] 鼻尖   [10][11] 嘴角左   [12][13] 嘴角右
        [14] 置信度

    Args:
        image: 待检测图像路径（立绘/头像等，自动兼容中文路径）
        model_path: YuNet ONNX 模型路径，None 时使用 DEFAULT_FACE_MODEL
        score_threshold: 人脸置信度阈值，低于该值的检测被过滤，默认 0.6
        nms_threshold: 非极大值抑制阈值，默认 0.3
        top_k: 保留的候选框上限，默认 5000

    Returns:
        检测到的人脸列表，按置信度降序排列。每项包含:
            score     - 置信度 [0, 1]
            x, y      - 人脸框左上角坐标
            w, h      - 人脸框宽高
            rect      - (x, y, x+w, y+h) 便捷元组
            landmarks - 5 个关键点：left_eye / right_eye / nose_tip / mouth_left / mouth_right
    """
    model = Path(model_path) if model_path is not None else DEFAULT_FACE_MODEL
    if not model.exists():
        raise FileNotFoundError(
            f"未找到 YuNet 人脸检测模型: {model}\n"
            f"请从 OpenCV Zoo 下载 face_detection_yunet_2023mar.onnx 并放入该路径，"
            f"或通过 model_path 参数指定。"
        )

    # FaceDetectorYN 按 BGR 读入
    img = _imread_unicode(image, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 创建检测器并设定输入尺寸（与待检测图像一致）
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (w, h), score_threshold, nms_threshold, top_k,
    )
    detector.setInputSize((w, h))

    # detect 返回 (retval, faces)，faces 为 N×15；未检到人脸时为空
    _retval, faces = detector.detect(img)

    results: list[dict] = []
    if faces is None:
        return results
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        landmarks = {
            "left_eye": (int(face[4]), int(face[5])),
            "right_eye": (int(face[6]), int(face[7])),
            "nose_tip": (int(face[8]), int(face[9])),
            "mouth_left": (int(face[10]), int(face[11])),
            "mouth_right": (int(face[12]), int(face[13])),
        }
        results.append({
            "score": float(face[14]),
            "x": x,
            "y": y,
            "w": fw,
            "h": fh,
            "rect": (x, y, x + fw, y + fh),
            "landmarks": landmarks,
        })

    # 按置信度降序
    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n🙂 共检测到 {len(results)} 张人脸")
    for i, r in enumerate(results):
        print(f"  #{i} 置信度={r['score']:.3f} 区域={r['rect']}")

    return results
