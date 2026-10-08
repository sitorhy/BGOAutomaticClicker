import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
import cv2

from .merge_chain import (
    merge_layers,
    MergeChainNodeGroup,
    BackgroundChainNode,
    AvatarClipChainNode,
    ExpandClipChainNode,
    LabelChainNode,
    StatusChainNode,
    StarsChainNode,
    ClassChainNode,
)

"""
测试用例
.venv/Scripts/python.exe -m unittest avatar_classifier_py_server.test.TestUnit.test_avatar_detect
"""

# res 资源根目录，配置中的图片路径均基于它解析
RES_DIR = Path(__file__).parent.parent.parent / 'res'


class TestUnit(unittest.TestCase):
    def test_avatar_cv_detect(self):
        mask_path = Path(__file__).parent.parent.parent / \
                    'res' / 'test' / 'mooncell头像探测掩码.jpg'
        template_path = Path(__file__).parent.parent.parent / \
                        'res' / 'test' / 'Servant481.jpg'
        target_path = Path(__file__).parent.parent.parent / \
                      'res' / 'test' / '哈贝特洛特(Pretender)一破.png'

        # cv2.imread 在 Windows 上不支持中文路径，会静默返回 None
        # 解决方案：cv2.imdecode(np.fromfile(...)) 先读字节再解码
        # np.uint8 指通道精度 2^8 = 256
        mask_img = cv2.imdecode(np.fromfile(
            mask_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        template_img = cv2.imdecode(np.fromfile(
            template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(np.fromfile(
            target_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        h, w = template_img.shape[:2]

        # 执行带掩码的模板匹配
        # 注意：带掩码时，必须使用 cv2.TM_SQDIFF 或 cv2.TM_CCORR_NORMED
        result = cv2.matchTemplate(
            target_img, template_img, cv2.TM_CCORR_NORMED, mask=mask_img)

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
        mask_path = Path(__file__).parent.parent.parent / \
                    'res' / 'test' / 'mooncell头像探测掩码.jpg'
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
        template_path = Path(__file__).parent.parent.parent / \
                        'res' / 'test' / 'Servant481.jpg'
        template_img = Image.open(template_path)
        # template_img.show()
        print(f"模板图片色彩模式: {template_img.mode}")
        print(f"模板图片尺寸: {template_img.size}")

        # 将掩码图缩放到模板图的大小（这里是一样的，省略掉）

        # 掩码图和模板图进行逻辑与操作
        # 这里假设原图为 [128,255,129] (灰度，每个像素只有一个值)
        # [128,255,129] x [0,1,0]  =>
        # 128 * 0 + 255 * 1 + 129 * 0 => [0, 255, 0]
        template_array = np.asarray(
            template_img.convert("L"), dtype=np.float64)
        # template_mask_array = template_array * mask_binary
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
        target_path = Path(__file__).parent.parent.parent / \
                      'res' / 'test' / '哈贝特洛特(Pretender)一破.png'
        target_img = Image.open(target_path)
        target_gray = np.asarray(target_img.convert("L"), dtype=np.float64)
        print(f"目标图片尺寸: {target_img.size}")

        mt_sum_s = float((mask_binary * template_array).sum()
                         )  # Σ(mask × scaled_tmpl) 当前尺度
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
        ImageDraw.Draw(vis).rectangle([best_x, best_y, best_x + template_array.shape[1],
                                       best_y + template_array.shape[0]], outline=(255, 0, 0), width=2)
        vis.show()

    def test_pic_merge(self):
        # 裁剪位置 (192, 140) -> (324, 284)
        target_path = Path(__file__).parent.parent.parent / \
                      'res' / 'test' / '哈贝特洛特(Pretender)一破.png'
        target_img = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # cv2.imshow('img_target', target_img)
        # cv2.waitKey(0)
        # v2.destroyAllWindows()
        target_img_clip = target_img[133:288, 183:333]
        # cv2.imshow('img_target_clip', target_img_clip)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if target_img_clip.shape[2] == 4:
            target_img_clip = cv2.cvtColor(target_img_clip, cv2.COLOR_BGRA2RGBA)
        else:
            target_img_clip = cv2.cvtColor(target_img_clip, cv2.COLOR_BGR2RGB)

        # 创建背景图，用黑色底填充，大小与裁剪区域相同
        blank_img = Image.new(
            'RGBA', (150, 155), (0, 0, 0, 255))
        # blank_img.show()

        # 验证数据
        # cv2.imshow('blank_img', cv2.cvtColor(np.asarray(blank_img), cv2.COLOR_RGBA2BGRA))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 图层配置拟定
        layers = [
            {
                'name': 'background',  # 唯一标识， 动态修改配置时可能会用到
                'template': np.asarray(blank_img)
            },
            {
                'name': 'frame',
                'template': Path(__file__).parent.parent.parent / 'res' / 'foreground' / 'L0_金框.png',
            },
            {
                'name': 'avatar',
                'mask': Path(__file__).parent.parent.parent / 'res' / 'mask' / 'BGO头像裁剪掩码.png',
                'template': target_img_clip,
            },
            {
                'name': 'stars',
                'template': Path(__file__).parent.parent.parent / 'res' / 'foreground' / 'L1_4星.png',
            },
            {
                'name': 'status',
                'template': Path(__file__).parent.parent.parent / 'res' / 'foreground' / 'L1_满破标.png',
            },
            {
                'name': 'class',
                'template': Path(__file__).parent.parent.parent / 'res' / 'foreground' / '金卡Breakser.png',
            },
            {
                'fit': 'fill',  # contain / cover / fill
                'name': 'label',
                'template': Path(__file__).parent.parent.parent / 'res' / 'foreground' / 'L1_金标.png',
            },
        ]
        canvas = merge_layers(layers)
        output_img = Path(__file__).parent.parent.parent / 'temp' / 'output.png'
        if not output_img.parent.exists():
            output_img.parent.mkdir(parents=True)
        canvas.save(output_img)

    def test_merge_layers(self):
        test_merge_layers()

    def test_merge_layers_chained(self):
        test_merge_layers_chained()


def test_merge_layers():
    """
    头像合成链（统一输入输出结构，7 个阶段，每个阶段对应一个序列）：

    ── 数据流 ────────────────────────────────────────────────────────────────
    每个阶段接收一个 “序列集”，输出一个 “序列集”，上一阶段的输出即下一阶段的输入，
    阶段间无需做数据适配（stage_in(N+1) == stage_out(N)）。

    阶段流转（STAGE_ORDER，分组键即序列集的检索入口）：
      1 background 输入[default] 项×1        → 输出[default]  叠加背景基底（最底层）  ×1
      2 avatar     输入[default] 项×1(立绘)   → 输出[default]  探测生成 t1...tn        ×n
      3 expand     输入[default] 项×n        → 输出[金,银,铜,铁,冠位]  按边框分组展开并叠加边框 m 种
      4 label      输入 各头像分组 项×z       → 输出分组不变  叠加标签 q 种   ×(z×q)
      5 status     输入 各头像分组 项×z       → 输出分组不变  叠加满破 k 种   ×(z×k)
      6 stars      输入 各头像分组 项×z       → 输出分组不变  叠加星标 p 种   ×(z×p)
      7 class      输入 各头像分组 项×z       → 输出分组不变  叠加职介 r 种   ×(z×r)
      （z 为当前分组内累计项数；每个阶段只读上一阶段的输出集，写完自己的输出集后旧项删除）

    ── 各阶段详述 ────────────────────────────────────────────────────────────
    阶段一：背景基底（background，分组内叠加，分组透传）
    输入 = 输出分组 "default"；叠加黑色底画布（template 为 'blank' 占位，运行时由调用方填充），
    作为序列项 layers 的最底层。项数 ×1。

    阶段二：头像序列初始化（avatar，生成，分组透传）
    输入：分组 "default"，立绘图片（每项 template = 一张立绘）
    2.1 探测头像位置，获取头像序列的生成范围，根据步进插值获取头像区域序列：
        记步进为 step（配置 avatar_cfg.step，step = 10 表示从头像的最大和最小范围间取 10 个矩形）
        记头像区域序列为 r1, r2, r3 .... rn, 1 <= n <= step
        r1 ... rn 从输入立绘中截取的头像切片为 a1, .... an
        a1 ... an 保存为临时图片 c1, .... cn（实际实现可添加前缀加以区分）
        c1, .... cn 分别和 avatar_cfg.mask（"BGO头像裁剪掩码.png"，灰度图）进行逻辑与运算
        （putalpha，注意掩码是灰度图），回传为 c1, .... cn，得到边缘透明的头像图片序列
    输出：分组 "default" 不变，序列集 { items: [c1 ... cn] }，每个 c.layers = [背景层, 头像层(clip)]

    阶段三：分组展开 + 叠加边框（expand，按边框分组展开并合成，分组透传）
    输入：分组 "default" 的序列集（t1 ... tn）
    3.1 头像分组数量不会超过边框种类，实际就是按边框分组：把 "default" 的每项按头像分组
        （"金"、"银"、"铜"、"铁"、"冠位" ... × 进阶组合 pair）逐个复制展开，
        group / pair 赋值到序列项；记分组边框序列 b1 ... bm（L0_金框 / L0_银框 /
        L0_铜框 / L0_铁框 / L0_冠位框 ...），按归属把边框叠加合成到序列项，
        c.layers += [边框层]。铜卡可一路突破为金卡/冠位，体现为多个 pair。
    输出：分组为各头像分组，序列项携带 group 与 pair，c.layers += [边框层]，供后续阶段按归属取图

    阶段四：叠加标签（label，分组内扩展，分组透传）
    输入 = 阶段三输出。记当前分组标签序列 l1 ... lq（fit=fill），每项与 l1 ... lq 叠加合成，项数 ×q。
    输出分组 = 输入分组，c.layers += [标签层]

    阶段五：添加满破标记（status，分组内扩展，分组透传）
    输入 = 阶段四输出。记满破标记序列 c1' ... ck', 0 <= k <= 4，默认所有从者满破
    （只有 "L1_满破标.png"），实际 k = 1；每项分别叠加 c1' ... ck'，项数 ×k。
    输出分组 = 输入分组，c.layers += [满破层]

    阶段六：添加稀有度标记（stars，分组内扩展，分组透传）
    每个分组对应一个标记序列 s1 ... sp（p 为整数），按 pair 取图。
    例如铜卡有 "L1_1星.png"、"L1_2星.png"；而金卡最多，有 "L1_4星.png"、"L1_5星.png"，
    "L1_2星再临.png" 也可以是金卡的星标。每项与 s1 ... sp 叠加合成，项数 ×p。
    输出分组 = 输入分组，c.layers += [星标层]

    阶段七：添加职介标记（class，分组内扩展，分组透传）
    跟阶段六一样，每个分组对应一个职介标记序列 f1 ... fr（r 为整数），按 pair 取图，
    铜卡序列为 "铜卡Saber.png" 等。每项与 f1 ... fr 叠加合成，项数 ×r。
    输出分组 = 输入分组，c.layers += [职介层]，即为最终头像合成结果序列

    统一输入输出收益：阶段接口一致（序列集 -> 序列集），任意两个阶段可直接串联；
    拆分后 expand / label 职责单一，新增阶段（如灵衣层）只需在 STAGE_ORDER 插入
    阶段名、实现 stage_branches 的分支与分组归属配置即可自动接入流水线。
    """

    # 输出目录，供各阶段节点做输入输出解耦配置（此处指向内存盘）
    output_dir = Path("F:\\")

    fg_dir = RES_DIR / 'foreground'

    BackgroundChainNode(
        input=[
            MergeChainNodeGroup(
                name='default',
                layers=[
                    {
                        'template': RES_DIR / 'mask' / 'BGO头像裁剪掩码.png',
                    }
                ]
            )
        ],
        out_put_file=output_dir / 'output_background_img.png',
    ).output()

    AvatarClipChainNode(
        input=[
            MergeChainNodeGroup(
                name='default',
                layers=[
                    {
                        'template': RES_DIR / 'test' / 'Servant481.jpg',
                        'mask': RES_DIR / 'mask' / 'mooncell头像探测掩码.jpg',
                    }
                ]
            )
        ],
        target_image_path=RES_DIR / 'test' / '哈贝特洛特(Pretender)一破.png',
        clip_mask=RES_DIR / 'mask' / 'BGO头像裁剪掩码.png',
        out_dir=output_dir,
    ).output()

    # 上一阶段：分组展开与边框合成，结果已落盘于 output_dir/{分组}/，此处暂时注释
    clip_paths = sorted(
        output_dir.glob('avatar_clip_*.png'),
        key=lambda p: int(p.stem.split('_')[-1]),
    )

    ExpandClipChainNode(
        input=[
            MergeChainNodeGroup(
                name='default',
                layers=[
                    {
                        'template': clip_path,
                    }
                    for clip_path in clip_paths
                ]
            )
        ],
        out_dir=output_dir,
        frame_paths=[
            {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
            {'name': '银', 'frame': fg_dir / 'L0_银框.png'},
            {'name': '铜', 'frame': fg_dir / 'L0_铜框.png'},
            {'name': '铁', 'frame': fg_dir / 'L0_铁框.png'},
            {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
            {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
        ],
    ).output()

    # 从磁盘收集上一阶段落盘的分组图片（output_dir/{分组}/*.png），作为后续叠加阶段的输入
    def collect_groups(root: Path) -> list[MergeChainNodeGroup]:
        return [
            MergeChainNodeGroup(
                name=d.name,
                layers=[{'template': f} for f in sorted(d.glob('*.png'))],
            )
            for d in root.iterdir()
            if d.is_dir()
        ]

    LabelChainNode(
        input=collect_groups(output_dir),
        out_dir=output_dir,
        label_paths=[
            {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
            {'name': '银', 'labels': [fg_dir / 'L1_银标.png']},
            {'name': '铜', 'labels': [fg_dir / 'L1_铜标.png']},
            {'name': '铁', 'labels': [fg_dir / 'L1_铁标.png']},
            {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
            {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
        ],
    ).output()

    StatusChainNode(
        input=collect_groups(output_dir),
        out_dir=output_dir,
        status_paths=[
            {'name': name, 'status': [fg_dir / 'L1_满破标.png']}
            for name in ('金', '银', '铜', '铁', '冠位', '满级冠位')
        ],
    ).output()

    # 金/冠位/满级冠位共用同一套完整星标序列
    gold_stars = [
        'L1_1星再临', 'L1_1星冠位', 'L1_2星再临', 'L1_2星冠位',
        'L1_3星再临', 'L1_3星冠位', 'L1_4星', 'L1_4星再临', 'L1_4星冠位',
        'L1_5星', 'L1_5星再临', 'L1_5星冠位',
    ]
    stars_chain = {
        '金': gold_stars,
        '银': ['L1_1星再临', 'L1_2星再临', 'L1_3星'],
        '铜': ['L1_1星再临', 'L1_2星'],
        '铁': ['L1_2星再临', 'L1_2星'],
        '冠位': gold_stars,
        '满级冠位': gold_stars,
    }
    StarsChainNode(
        input=collect_groups(output_dir),
        out_dir=output_dir,
        stars_paths=[
            {'name': name, 'stars': [fg_dir / f'{p}.png' for p in paths]}
            for name, paths in stars_chain.items()
        ],
    ).output()

    ClassChainNode(
        input=collect_groups(output_dir),
        out_dir=output_dir,
        class_paths=[
            {'name': '金', 'class': [fg_dir / '金卡Breakser.png']},
            {'name': '银', 'class': [fg_dir / '银卡Berserker.png']},
            {'name': '铜', 'class': [fg_dir / '铜卡Berserker.png']},
            {'name': '铁', 'class': []},
            {'name': '冠位', 'class': []},
            {'name': '满级冠位', 'class': []},
        ],
    ).output()


def test_merge_layers_chained():
    """
    照抄 test_merge_layers，但做两点调整：
      1. 排除阶段一 BackgroundChainNode，从阶段二 AvatarClipChainNode 开始；
      2. 不再从磁盘 glob / collect_groups 收集中间产物，而是把每个 ChainNode 的
         .output() 返回值直接作为下一个 ChainNode 的 input，内存中逐阶段串联。

    阶段流转（与 test_merge_layers 一致，仅去掉 background）：
      avatar → expand → label → status → stars → class
    """

    # 输出目录，供各阶段节点做输入输出解耦配置（此处指向内存盘）
    output_dir = Path("F:\\")

    fg_dir = RES_DIR / 'foreground'

    # 阶段二：头像裁剪，输出直接交给下一阶段（不再回读磁盘上的 avatar_clip_*.png）
    avatar_groups = AvatarClipChainNode(
        input=[
            MergeChainNodeGroup(
                name='default',
                layers=[
                    {
                        'template': RES_DIR / 'test' / 'Servant481.jpg',
                        'mask': RES_DIR / 'mask' / 'mooncell头像探测掩码.jpg',
                    }
                ]
            )
        ],
        target_image_path=RES_DIR / 'test' / '哈贝特洛特(Pretender)一破.png',
        clip_mask=RES_DIR / 'mask' / 'BGO头像裁剪掩码.png',
        out_dir=output_dir,
    ).output()

    # 阶段三：分组展开 + 叠加边框，输入为上一阶段输出的 default 分组
    expand_groups = ExpandClipChainNode(
        input=avatar_groups,
        out_dir=output_dir,
        frame_paths=[
            {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
            {'name': '银', 'frame': fg_dir / 'L0_银框.png'},
            {'name': '铜', 'frame': fg_dir / 'L0_铜框.png'},
            {'name': '铁', 'frame': fg_dir / 'L0_铁框.png'},
            {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
            {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
        ],
    ).output()

    # 阶段四：叠加标签，输入为上一阶段输出的各头像分组
    label_groups = LabelChainNode(
        input=expand_groups,
        out_dir=output_dir,
        label_paths=[
            {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
            {'name': '银', 'labels': [fg_dir / 'L1_银标.png']},
            {'name': '铜', 'labels': [fg_dir / 'L1_铜标.png']},
            {'name': '铁', 'labels': [fg_dir / 'L1_铁标.png']},
            {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
            {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
        ],
    ).output()

    # 阶段五：添加满破标记
    status_groups = StatusChainNode(
        input=label_groups,
        out_dir=output_dir,
        status_paths=[
            {'name': name, 'status': [fg_dir / 'L1_满破标.png']}
            for name in ('金', '银', '铜', '铁', '冠位', '满级冠位')
        ],
    ).output()

    # 阶段六：添加稀有度标记（金/冠位/满级冠位共用同一套完整星标序列）
    gold_stars = [
        'L1_1星再临', 'L1_1星冠位', 'L1_2星再临', 'L1_2星冠位',
        'L1_3星再临', 'L1_3星冠位', 'L1_4星', 'L1_4星再临', 'L1_4星冠位',
        'L1_5星', 'L1_5星再临', 'L1_5星冠位',
    ]
    stars_chain = {
        '金': gold_stars,
        '银': ['L1_1星再临', 'L1_2星再临', 'L1_3星'],
        '铜': ['L1_1星再临', 'L1_2星'],
        '铁': ['L1_2星再临', 'L1_2星'],
        '冠位': gold_stars,
        '满级冠位': gold_stars,
    }
    stars_groups = StarsChainNode(
        input=status_groups,
        out_dir=output_dir,
        stars_paths=[
            {'name': name, 'stars': [fg_dir / f'{p}.png' for p in paths]}
            for name, paths in stars_chain.items()
        ],
    ).output()

    # 阶段七：添加职介标记，输出即为最终头像合成结果序列
    ClassChainNode(
        input=stars_groups,
        out_dir=output_dir,
        class_paths=[
            {'name': '金', 'class': [fg_dir / '金卡Breakser.png']},
            {'name': '银', 'class': [fg_dir / '银卡Berserker.png']},
            {'name': '铜', 'class': [fg_dir / '铜卡Berserker.png']},
            {'name': '铁', 'class': []},
            {'name': '冠位', 'class': [fg_dir / '冠位Berserker.png']},
            {'name': '满级冠位', 'class': [fg_dir / '冠位Berserker.png']},
        ],
    ).output()


if __name__ == '__main__':
    unittest.main()
