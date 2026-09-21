"""
头像位置检测：多尺度模板匹配

在立绘（目标图像）中搜索头像（模板图像）的位置，
支持掩码排除边框干扰，在指定缩放范围内逐尺度匹配。
"""
import numpy as np
from scipy.signal import fftconvolve


def detect_avatar(
        target: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        scale_min: float = 0.3,
        scale_max: float = 2.0,
        scale_steps: int = 20,
) -> list[dict]:
    """多尺度模板匹配，检测头像在立绘中的位置。

    使用归一化互相关 (NCC) 衡量匹配度，通过 FFT 卷积加速滑动窗口。
    掩码为二值图（白色=参与匹配，黑色=忽略），用于抠出头像核心内容、排除边框干扰。

    Args:
        target: 目标图像（立绘），H×W×3 (RGB)
        template: 模板图像（头像），h×w×3
        mask: 掩码图，与模板同尺寸，只有黑白两色，h×w×(1|3|4)
        scale_min: 最小缩放比（相对模板原尺寸）
        scale_max: 最大缩放比
        scale_steps: 缩放步数（在 min~max 之间均匀取 scale_steps 个尺度）

    Returns:
        每个缩放尺度的匹配结果列表，按匹配度降序排列。
        每项包含:
            scale       - 缩放比
            template_size - (宽, 高) 缩放后的模板尺寸
            score       - 匹配度 [-1, 1]，越大越好
            x, y        - 匹配区域左上角在目标图中的坐标
            w, h        - 匹配区域的宽高
            rect        - (x, y, x+w, y+h) 便捷元组
    """
    from PIL import Image

    results = []
    th, tw = target.shape[:2]

    """
    |	ndarray.flags		|	有关数组内存布局的信息。					
    |	ndarray.shape		|	数组维度的元组。							
    |	ndarray.strides		|	遍历数组时每个维度中的字节元组。			
    |	ndarray.ndim		|	数组维数。								
    |	ndarray.data		|	Python缓冲区对象指向数组的数据的开头。		
    |	ndarray.size		|	数组中的元素数。							
    |	ndarray.itemsize	|	一个数组元素的长度，以字节为单位。			
    |	ndarray.nbytes		|	数组元素消耗的总字节数。					
    |	ndarray.base		|	如果内存来自其他对象，则为基础对象。		
    """
    # ── 掩码预处理：转灰度 → 二值化 → 归一化到 {0, 1} ──
    if mask.ndim == 3:
        mask_gray = np.mean(mask[:, :, :3], axis=2)
    else:
        mask_gray = mask.astype(np.float64)
    mask_binary = (mask_gray > 127).astype(np.float64)  # 二值化，避免插值引入中间值

    # 确保掩码与模板同尺寸
    mh, mw = mask_binary.shape
    if mh != template.shape[0] or mw != template.shape[1]:
        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize((template.shape[1], template.shape[0]), Image.NEAREST)
        mask_binary = (np.asarray(mask_img, dtype=np.float64) > 127).astype(np.float64)

    # ── 模板灰度化 ──
    tmpl_gray = template.mean(axis=2) if template.ndim == 3 else template.astype(np.float64)
    tmpl_f = tmpl_gray.astype(np.float64)

    # ── 目标灰度化 ──
    target_gray = target.mean(axis=2) if target.ndim == 3 else target.astype(np.float64)

    scales = np.linspace(scale_min, scale_max, scale_steps)

    for scale in scales:
        # ── 缩放模板与掩码 ──
        new_w = max(1, int(tmpl_f.shape[1] * scale))
        new_h = max(1, int(tmpl_f.shape[0] * scale))

        if new_h > th or new_w > tw:
            continue

        tmpl_img = Image.fromarray(np.clip(tmpl_f, 0, 255).astype(np.uint8), mode="L")
        tmpl_img = tmpl_img.resize((new_w, new_h), Image.BILINEAR)
        scaled = np.asarray(tmpl_img, dtype=np.float64)

        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)
        scaled_mask = (np.asarray(mask_img, dtype=np.float64) > 127).astype(np.float64)

        # ── FFT 加速全位置 NCC ──
        # 利用 fftconvolve 一次性算出所有位置的三个关键量：
        #   cross_map[i,j]  = Σ(mask × scaled × target_patch)
        #   t_sum_map[i,j]  = Σ(mask × target_patch)
        #   t_sq_map[i,j]   = Σ(mask × target_patch²)
        # 然后 NCC = (cross - mt·t_sum/n) / √(t_var · s_var)
        #   where t_var = mt2 - mt²/n,  s_var = t_sq_map - t_sum²/n
        #
        # 注意：fftconvolve(A, K[::-1,::-1], 'valid') 等价于互相关
        #         输出 (i,j) = Σ_{u,v} A[i+u, j+v] · K[u, v]

        mt_sum_s = float((scaled_mask * scaled).sum())  # Σ(mask × scaled_tmpl) 当前尺度
        mt2_sum_s = float((scaled_mask * scaled ** 2).sum())
        n_s = max(float(scaled_mask.sum()), 1e-8)
        t_var_s = max(mt2_sum_s - mt_sum_s ** 2 / n_s, 0.0)

        if t_var_s < 1e-12:
            continue

        # 互相关：Σ(mask × tmpl × target_patch)
        kernel_cross = (scaled_mask * scaled)[::-1, ::-1]
        cross_map = fftconvolve(target_gray, kernel_cross, mode="valid")

        # 目标加权和：Σ(mask × target_patch)
        kernel_sum = scaled_mask[::-1, ::-1]
        t_sum_map = fftconvolve(target_gray, kernel_sum, mode="valid")

        # 目标加权平方和：Σ(mask × target_patch²)
        t_sq_map = fftconvolve(target_gray ** 2, kernel_sum, mode="valid")

        # NCC 分子 & 分母
        numerator = cross_map - mt_sum_s * t_sum_map / n_s
        s_var_map = np.maximum(t_sq_map - t_sum_map ** 2 / n_s, 0.0)
        denominator = np.sqrt(s_var_map * t_var_s)

        ncc_map = np.full_like(numerator, -2.0)
        valid = denominator > 1e-8
        ncc_map[valid] = numerator[valid] / denominator[valid]
        # 兜底裁剪浮点误差
        np.clip(ncc_map, -1.0, 1.0, out=ncc_map)

        best_score = float(ncc_map.max())
        best_pos = np.unravel_index(int(ncc_map.argmax()), ncc_map.shape)
        best_y, best_x = int(best_pos[0]), int(best_pos[1])

        results.append({
            "scale": float(scale),
            "template_size": (new_w, new_h),
            "score": best_score,
            "x": best_x,
            "y": best_y,
            "w": new_w,
            "h": new_h,
            "rect": (best_x, best_y, best_x + new_w, best_y + new_h),
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    # ── 最佳匹配 ──
    best = results[0]
    x1, y1, x2, y2 = best["rect"]
    print(f"\n🎯 最佳匹配:")
    print(f"  缩放比:  {best['scale']:.3f}")
    print(f"  模板尺寸: {best['template_size'][0]}×{best['template_size'][1]}")
    print(f"  匹配度:  {best['score']:.4f}  ({best['score'] * 100:.1f}%)")
    print(f"  位置:    ({best['x']}, {best['y']})")
    print(f"  区域:    ({x1}, {y1}) → ({x2}, {y2})")

    bar_len = 30
    filled = int(max(0, best["score"]) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}]")

    # ── 可视化输出到 temp/ ──
    _save_detect_visualization(
        target=target,
        template=template,
        mask=mask,
        best=best,
    )


    return results


def _save_detect_visualization(target, template, mask, best):
    """将掩码匹配结果可视化，输出到 temp/ 目录。"""
    from PIL import Image, ImageDraw
    from .config import ROOT_DIR

    temp_dir = ROOT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)

    bx, by = best["x"], best["y"]
    bw, bh = best["w"], best["h"]

    # ── 缩放掩码到最佳匹配尺寸 ──
    if mask.ndim == 3:
        mask_gray = np.mean(mask[:, :, :3], axis=2)
    else:
        mask_gray = mask.astype(np.float64)
    mask_bin = (mask_gray > 127).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_bin, mode="L")
    mask_img = mask_img.resize((bw, bh), Image.NEAREST)
    mask_arr = np.array(mask_img)

    # ── 1. 掩码层：立绘上仅保留掩码白色区域，其余黑色（RGBA） ──
    layer = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    layer[:, :, 3] = 255
    roi = target[by:by + bh, bx:bx + bw]
    layer_roi = layer[by:by + bh, bx:bx + bw]
    white = mask_arr > 127
    layer_roi[white, :3] = roi[white]
    layer_roi[white, 3] = 255
    Image.fromarray(layer, mode="RGBA").save(
        temp_dir / "detect_mask_layer.png"
    )

    # ── 2. 叠加预览：掩码外部半透明遮罩 + 绿色边框 ──
    vis = Image.fromarray(target).convert("RGBA")
    overlay_arr = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    overlay_arr[:, :, 3] = 100  # 默认半透明黑
    overlay_roi = overlay_arr[by:by + bh, bx:bx + bw]
    overlay_roi[white] = [0, 0, 0, 0]  # 掩码内部透明
    vis = Image.alpha_composite(vis, Image.fromarray(overlay_arr, "RGBA"))
    ImageDraw.Draw(vis).rectangle(
        [bx, by, bx + bw - 1, by + bh - 1],
        outline=(0, 255, 0, 200), width=2,
    )
    vis.save(temp_dir / "detect_mask_layer.png")

    print(f"\n📁 可视化已保存到 temp/:")
