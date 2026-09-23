import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
import cv2

"""
测试用例
.venv/Scripts/python.exe -m unittest avatar_classifier_py_server.test.TestUnit.test_avatar_detect
"""
class TestUnit(unittest.TestCase):
    def test_avatar_cv_detect(self):
        mask_path = Path(__file__).parent.parent.parent / 'res' / 'test' / 'mooncell头像探测掩码.jpg'
        template_path = Path(__file__).parent.parent.parent / 'res' / 'test' / 'Servant481.jpg'
        target_path = Path(__file__).parent.parent.parent / 'res' / 'test' / '哈贝特洛特(Pretender)一破.png'
        
        # cv2.imread 在 Windows 上不支持中文路径，会静默返回 None
        # 解决方案：cv2.imdecode(np.fromfile(...)) 先读字节再解码
        # np.uint8 指通道精度 2^8 = 256
        mask_img = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        template_img = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        h, w = template_img.shape[:2]

        # 执行带掩码的模板匹配
        # 注意：带掩码时，必须使用 cv2.TM_SQDIFF 或 cv2.TM_CCORR_NORMED
        result = cv2.matchTemplate(target_img, template_img, cv2.TM_CCORR_NORMED, mask=mask_img)

        # 获取最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # │          │        │      │
        # │          │        │      └── 最大值的位置 (x, y)
        # │          │        └── 最小值的位置 (x, y)
        # │          └── 最大值（匹配分数）
        # └── 最小值（匹配分数）

        # 如果使用 cv2.TM_CCORR_NORMED，最大值 max_loc 是最佳匹配点
        # 如果使用 cv2.TM_SQDIFF，最小值 min_loc 是最佳匹配点
        # 匹配方法	         最佳值在	     原因
        # TM_SQDIFF	        min_loc	        差异越小越匹配，最小值 = 最佳
        # TM_SQDIFF_NORMED	min_loc	        同上
        # TM_CCORR	        max_loc	        相关性越大越匹配，最大值 = 最佳
        # TM_CCORR_NORMED	max_loc	        同上（你代码用的就是这个）
        # TM_CCOEFF	        max_loc	        相关系数越大越匹配
        # TM_CCOEFF_NORMED	max_loc	        同上
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        print(f"最佳匹配位置: {top_left} -> {bottom_right}")


        # 5. 绘制结果
        cv2.rectangle(target_img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow('Result', target_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_avatar_detect(self):
        mask_path = Path(__file__).parent.parent.parent / 'res' / 'test' / 'mooncell头像探测掩码.jpg'
        mask_img = Image.open(mask_path)
        print(f"掩码图片尺寸: {mask_img.size}")
        print(f"掩码图片模式: {mask_img.mode}")
        # mask_img.show()

        # Image.convert(mode) 用于把图像转换到指定的色彩模式（color mode）。
        # 参数 mode 是一个字符串，传入不同值表示不同的像素存储方式。
        # 下面是 Pillow 常用取值及含义：    
        # "1"	二值图（纯黑纯白，阈值 128）	1 bit
        # "L"	灰度图（0=黑，255=白）	8 bit
        # "I"	32 位整型灰度	32 bit
        # "F"	32 位浮点灰度	32 bit
        # "RGB"	红绿蓝	真彩色，红绿蓝三通道（你代码里用的就是这个）
        # "RGBA"	RGB + Alpha 透明通道
        # "CMYK"	印刷四色（青、品红、黄、黑）
        # "P"	调色板模式（8 位索引，如 GIF）
        # mask_img_gray = mask_img.convert("L")
        # mask_img_binary.show()

        # ndarray 重载了 > 运算符
        # 对每一个元素进行广播比较。
        # 对彩色图片进行 > 127 比较，结果是元素类型为[bool, bool, bool] 的ndarray
        # 因此先转为灰度图（元素为 int8）, 在转为 bool ndarray，最后转为只有 0 / 1 的二值图
        mask_grey = mask_img.convert("L")
        # PIL Image 与 NumPy ndarray 的结构对比
        # Image: 图像对象，封装了像素数据 + 图像元信息（模式、尺寸、格式等）
        #       通过 .getpixel((x, y)) 方法，坐标顺序是 (x, y) 即 (列, 行)
        #       默认不可变，修改需调用方法（如 .putpixel()、.convert()），返回新对象
        #       无向量化运算，只能逐像素操作	支持广播、向量
        # ndarray: 通用 N 维数组，纯粹的数值容器
        #       通过 arr[y, x] 索引，顺序是 (行, 列) 即 (y, x)
        #       原地可变，arr[0, 0] = 255 直接修改
        #       支持广播、向量化运算（arr * 2、arr > 127）
        mask_binary = (np.array(mask_grey) > 127).astype(np.float64)

        # 加载头像图片，和掩码图进行逻辑 &，白色（1）的不会保留，黑色（0）的部分过滤
        template_path = Path(__file__).parent.parent.parent / 'res' / 'test' / 'Servant481.jpg'
        template_img = Image.open(template_path)
        # template_img.show()
        print(f"模板图片色彩模式: {template_img.mode}")
        print(f"模板图片尺寸: {template_img.size}")
        
        # 将掩码图缩放到模板图的大小（这里是一样的，省略掉）
    
        # 掩码图和模板图进行逻辑与操作
        # 这里假设原图为 [128,255,129] (灰度，每个像素只有一个值)
        # [128,255,129] x [0,1,0]  =>
        # 128 * 0 + 255 * 1 + 129 * 0 => [0, 255, 0]
        template_array = np.asarray(template_img.convert("L"), dtype=np.float64)
        #template_mask_array = template_array * mask_binary
        # template_mask_img = Image.fromarray(template_mask_array.astype(np.uint8))
        # template_mask_img.show()

        # ── 手搓 NCC 模板匹配 ──────────────────────────────────────────
        #
        # NCC（归一化互相关）公式：
        #
        #             Σ(mask × tmpl × target_patch) - mt·t_sum/n
        #   NCC = ──────────────────────────────────────────────────────
        #           √(s_var × t_var)
        #
        #   其中：
        #     mt      = Σ(mask × tmpl)              掩码加权模板总和
        #     n       = Σ(mask)                      掩码有效像素数
        #     t_sum   = Σ(mask × target_patch)       掩码加权目标 patch 总和
        #     t_sq    = Σ(mask × target_patch²)      掩码加权目标 patch 平方和
        #     s_var   = Σ(mask × tmpl²) - mt²/n     模板方差
        #     t_var   = t_sq - t_sum²/n              目标 patch 方差
        #
        # NCC 范围 [-1, 1]，1 表示完全匹配

        # 1. 加载目标图像（立绘）并灰度化
        target_path = Path(__file__).parent.parent.parent / 'res' / 'test' / '哈贝特洛特(Pretender)一破.png'
        target_img = Image.open(target_path)
        target_gray = np.asarray(target_img.convert("L"), dtype=np.float64)
        print(f"目标图片尺寸: {target_img.size}")

        mt_sum_s = float((mask_binary * template_array).sum())  # Σ(mask × scaled_tmpl) 当前尺度
        mt2_sum_s = float((mask_binary * template_array ** 2).sum())
        n_s = max(float(mask_binary.sum()), 1e-8)
        t_var_s = max(mt2_sum_s - mt_sum_s ** 2 / n_s, 0.0)

        # 互相关：Σ(mask × tmpl × target_patch)
        kernel_cross = (mask_binary * template_array)[::-1, ::-1]
        cross_map = fftconvolve(target_gray, kernel_cross, mode="valid")

        # 目标加权和：Σ(mask × target_patch)
        kernel_sum = mask_binary[::-1, ::-1]
        t_sum_map = fftconvolve(target_gray, kernel_sum, mode="valid")

        # 目标加权平方和：Σ(mask × target_patch²)
        t_sq_map = fftconvolve(target_gray ** 2, kernel_sum, mode="valid")

        # NCC 分子 & 分母
        numerator = cross_map - mt_sum_s * t_sum_map / n_s
        s_var_map = np.maximum(t_sq_map - t_sum_map ** 2 / n_s, 0.0)
        denominator = np.sqrt(s_var_map * t_var_s)

        # NCC 分子 & 分母
        #        cross - mt·t_sum/n
        #  NCC = ─────────────────────────
        #       sqrt(s_var × t_var)
        ncc_map = np.full_like(numerator, -2.0)
        valid = denominator > 1e-8
        ncc_map[valid] = numerator[valid] / denominator[valid]
        # 兜底裁剪浮点误差
        np.clip(ncc_map, -1.0, 1.0, out=ncc_map)

        best_score = float(ncc_map.max())
        best_pos = np.unravel_index(int(ncc_map.argmax()), ncc_map.shape)
        best_y, best_x = int(best_pos[0]), int(best_pos[1])

        result = {
            "scale": 1.0,
            "template_size": (template_array.shape[1], template_array.shape[0]),
            "score": best_score,
            "x": best_x,
            "y": best_y,
            "w": template_array.shape[1],
            "h": template_array.shape[0],
            "rect": (best_x, best_y, best_x + template_array.shape[1], best_y + template_array.shape[0]),
        }
        print(f"最佳匹配分值: {result}")

        vis = target_img.copy()
        ImageDraw.Draw(vis).rectangle([best_x, best_y, best_x + template_array.shape[1], best_y + template_array.shape[0]], outline=(255, 0, 0), width=2)
        vis.show()


if __name__ == '__main__':
    unittest.main()
