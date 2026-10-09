import json
import os
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from typing_extensions import NotRequired, TypedDict
import cv2

from .detect_cv import detect_avatar

# 图片图层/合成工具与合并链节点（MergeChainNode 及其各阶段子类）
# 从 test.py 抽离，供测试与业务代码复用。各节点的图片输入输出路径与配置
# 均通过构造函数注入，实现输入输出解耦。

FitMode = Literal['fill', 'contain', 'cover']


class LayerSpec(TypedDict, total=False):
    """单个图层/标记图片的描述（对应注释中的 b1、l1、c1、s1、f1 等序列元素）"""
    # 唯一标识，动态修改配置时可能会用到
    name: str
    # 模板图片路径（str | PathLike）或已加载好的 ndarray
    template: PathLike | str | np.ndarray
    # 裁剪掩码图片路径，灰度图，合成前通过 putalpha 转为透明通道
    mask: NotRequired[PathLike | str]
    # 尺寸不适应于画布时的缩放策略: contain / cover / fill, None 表示居中放置
    fit: NotRequired[FitMode | None]


def parse_layer_image(template: PathLike | str | np.ndarray | None) -> np.ndarray | None:
    if isinstance(template, np.ndarray):
        # 调用方传入的 ndarray 应已是 RGB(A) 格式，不做通道转换
        return template
    if isinstance(template, str | PathLike):
        nd_arr = cv2.imdecode(np.fromfile(
            template, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # cv2.imdecode 返回 BGR(A) 通道顺序，需转为 RGB(A) 才能被 PIL 正确解读
        # x轴y轴位置和颜色数据三个维度 ndim = 3,  shape 为 [高，宽，通道数]
        if nd_arr.ndim == 3 and nd_arr.shape[2] == 4:
            nd_arr = cv2.cvtColor(nd_arr, cv2.COLOR_BGRA2RGBA)
        elif nd_arr.ndim == 3 and nd_arr.shape[2] == 3:
            nd_arr = cv2.cvtColor(nd_arr, cv2.COLOR_BGR2RGB)
        return nd_arr
    return None


def merge_layers(layers: list[LayerSpec]):
    canvas: np.ndarray | None = None
    for layer in layers:
        template_nd_arr = parse_layer_image(layer['template'])
        print(
            f"Layer: {layer['name'] if 'name' in layer else 'anonymous'}, Template: {template_nd_arr.shape}")

        """ 创建画布 ，以第一幅图的大小为基准 """
        if canvas is None:
            canvas = Image.new(
                'RGBA', (template_nd_arr.shape[1], template_nd_arr.shape[0]), (0, 0, 0, 0))

        """ 将图层绘制到画布上 """
        template_img = Image.fromarray(template_nd_arr)
        # alpha_composite 要求两张图模式一致（均为 RGBA），需转换
        if template_img.mode != 'RGBA':
            template_img = template_img.convert('RGBA')
        # template_img.show()

        if 'mask' in layer:
            mask = Image.open(layer['mask']).convert('L')
            if mask.size != template_img.size:
                mask = mask.resize(template_img.size)
            template_img.putalpha(mask)

        # 如果所有图层尺寸一致，可直接合成；否则根据 fit 的取值做大小适应后再合成
        if template_img.size == canvas.size:
            canvas = Image.alpha_composite(canvas, template_img)
        else:
            template_img = fit_layer_to_canvas(template_img, canvas.size, layer.get('fit'))
            temp_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            temp_layer.paste(template_img, (0, 0), template_img)
            canvas = Image.alpha_composite(canvas, temp_layer)

    return canvas


def fit_layer_to_canvas(img: Image.Image, canvas_size: tuple[int, int], fit: str | None):
    """根据 fit 模式将图层缩放到画布大小并居中放置

    fit 取值:
        None  : 不缩放，保持原尺寸，居中放置在画布上（原有兜底行为）
        fill  : 直接拉伸到与画布一样大小（不保持宽高比）
        contain  : 等比缩放使图层完整容纳在画布内（可能留边）
        cover    : 等比缩放使图层完全覆盖画布（可能超出裁剪）
    """
    cw, ch = canvas_size
    iw, ih = img.size

    if fit is None:
        # 保持原尺寸，居中平移到画布上
        fitted = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        fitted.paste(img, ((cw - iw) // 2, (ch - ih) // 2), img)
        return fitted

    if fit == 'fill':
        return img.resize((cw, ch))

    if fit == 'contain':
        scale = min(cw / iw, ch / ih)
        resized = img.resize((max(1, round(iw * scale)), max(1, round(ih * scale))))
        fitted = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        fitted.paste(resized, ((cw - resized.width) // 2, (ch - resized.height) // 2), resized)
        return fitted

    if fit == 'cover':
        scale = max(cw / iw, ch / ih)
        resized = img.resize((max(1, round(iw * scale)), max(1, round(ih * scale))))
        # 居中裁剪到画布大小
        left = (resized.width - cw) // 2
        top = (resized.height - ch) // 2
        return resized.crop((left, top, left + cw, top + ch))

    raise ValueError(f'未知的 fit 模式: {fit}')


class MergeChainNodeGroup:
    name: str
    layers: list[LayerSpec]

    def __init__(self, name: str = "", layers=None):
        if layers is None:
            layers = []
        self.name = name
        self.layers = layers


class MergeChainNode:
    input: list[MergeChainNodeGroup]

    def __init__(self, input: list[MergeChainNodeGroup]):
        self.input = input

    def output(self) -> list[MergeChainNodeGroup]:
        pass


class BackgroundChainNode(MergeChainNode):
    # 背景图落盘路径，构造对象时指定（输入输出解耦）
    out_put_file: PathLike

    def __init__(self, input: list[MergeChainNodeGroup], out_put_file: PathLike):
        super().__init__(input)
        self.out_put_file = Path(out_put_file)

    def output(self) -> list[MergeChainNodeGroup]:
        first_input_group = self.input[0]
        background_img_path = first_input_group.layers[0].get('template')
        if background_img_path is None or not Path(background_img_path).exists():
            raise ValueError(f"背景图不存在: {background_img_path}")
        print(f"background_img_path: {background_img_path}")
        img_arr = parse_layer_image(background_img_path)
        if img_arr is None:
            raise ValueError(f"背景图解析失败: {background_img_path}")
        width, height = img_arr.shape[1], img_arr.shape[0]
        print(f"width: {width}, height: {height}")
        # 宽高取出后 img_arr 不再被使用，主动 del 释放其底层像素缓冲区
        del img_arr

        output_background_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
        # output_background_img.show()

        self.out_put_file.parent.mkdir(parents=True, exist_ok=True)
        output_background_img.save(self.out_put_file)

        return [
            MergeChainNodeGroup(
                name='default',
                layers=[
                    {
                        'template': self.out_put_file,
                    }
                ]
            )
        ]


class AvatarClipChainNode(MergeChainNode):
    steps = 10
    min_scale = 0.8
    max_scale = 1.5
    

    def __init__(self, input: list[MergeChainNodeGroup], target_image_path: PathLike, clip_mask: PathLike, out_dir: PathLike, output_size: tuple[int, int]):
        super().__init__(input)
        # 立绘（待探测目标）路径，构造时指定
        self.target_image_path = Path(target_image_path)
        # 头像裁剪掩码（合成 canvas 时通用叠加），构造时指定
        self.clip_mask = Path(clip_mask)
        # 裁剪产物落盘目录，构造时指定
        self.out_dir = Path(out_dir)
        # 裁剪产物尺寸，基于检测位置扩充至指定大小
        self.output_size = output_size

    def output(self) -> list[MergeChainNodeGroup]:
        """
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
        """
        first_input_group = self.input[0]
        avatar_img_path = first_input_group.layers[-1].get('template')
        if avatar_img_path is None or not Path(avatar_img_path).exists():
            raise ValueError(f"头像文件不存在: {avatar_img_path}")
        print(f"avatar_img_path: {avatar_img_path}")
        avatar_img_arr = parse_layer_image(avatar_img_path)
        if avatar_img_arr is None:
            raise ValueError(f"头像解析失败: {avatar_img_path}")
        width, height = avatar_img_arr.shape[1], avatar_img_arr.shape[0]
        # avatar_img_arr = cv2.cvtColor(avatar_img_arr, cv2.COLOR_BGR2RGB)
        #
        # avatar_mask_file_name = first_input_group.layers[0].get('mask')
        # avatar_mask_img_arr: np.ndarray | None = None
        # if not avatar_mask_file_name is None:
        #     avatar_mask_img_path = Path(avatar_mask_file_name)
        #     print(f"avatar_mask_img_path: {avatar_mask_img_path}")
        #     avatar_mask_img_arr = cv2.imdecode(np.fromfile(avatar_mask_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # cv2.imshow("avatar_img_arr", avatar_img_arr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        target_image_arr = cv2.imdecode(np.fromfile(self.target_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        target_image_arr = cv2.cvtColor(target_image_arr, cv2.COLOR_BGR2RGB)

        # 调用 detect_cv.detect_avatar 做带掩码的多尺度模板匹配，返回按匹配度降序的结果列表
        avatar_mask_file_name = first_input_group.layers[-1].get('mask')
        avatar_mask_img_path = Path(avatar_mask_file_name)
        results = detect_avatar(
            target=self.target_image_path,
            template=avatar_img_path,
            mask=avatar_mask_img_path,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            steps=self.steps,
        )
        if not results:
            raise ValueError("头像检测未返回任何匹配结果")

        best = results[0]

        # best 的 rect 是检测图（立绘）中匹配到的区域，尺寸等于该尺度下缩放后的模板；
        # 而后续图层合并、前景素材均按 output_size 设计，直接按 rect 裁剪再合并会引入缩放。
        # 因此沿 best 的中心向外扩充裁剪区域，使其大小等于 output_size，
        # 让从立绘裁剪出的区域本身即为 output_size，合并时不再需要缩放。
        out_w, out_h = self.output_size
        bx1, by1, bx2, by2 = best['rect']
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2
        best = {**best, 'rect': (
            int(bcx - out_w / 2), int(bcy - out_h / 2),
            int(bcx + out_w / 2), int(bcy + out_h / 2),
        )}

        step = (self.max_scale - self.min_scale) / self.steps
        rects = []
        # 裁剪/合并基准尺寸改为 output_size，best 已按此尺寸扩充，基准裁剪无需缩放
        dx = out_w
        dy = out_h
        center_x = best['rect'][0] + dx / 2
        center_y = best['rect'][1] + dy / 2
        for i in range(int(self.steps / 2)):
            scale = self.min_scale + i * step
            print(f"scale = {scale}")
            # x1, y1, x2, y2 = best["rect"]
            rects.append([
                int(center_x - dx / 2 * scale),
                int(center_y - dy / 2 * scale),
                int(center_x + dx / 2 * scale),
                int(center_y + dy / 2 * scale)
            ])
        # x1, y1 = best["rect"]
        rects.append([int(best["rect"][0]), int(best["rect"][1]), int(best["rect"][0] + dx), int(best["rect"][1] + dy)])
        for i in range(int(self.steps / 2) - 1):
            scale = self.min_scale + (i + int(self.steps / 2) + 1) * step
            print(f"scale = {scale}")
            # x1, y1, x2, y2 = best["rect"]
            # x1 = x1 - dx * scale
            # X2 = x2 + y2 * scale
            rects.append([
                int(center_x - dx / 2 * scale),
                int(center_y - dy / 2 * scale),
                int(center_x + dx / 2 * scale),
                int(center_y + dy / 2 * scale)
            ])
        print(json.dumps(rects, indent=4))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        layers = []
        for index, rect in enumerate(rects):
            x1, y1, x2, y2 = rect
            clip_arr = target_image_arr[y1:y2, x1:x2]

            clip_path = self.out_dir / f"avatar_clip_{index}.png"
            img = Image.fromarray(clip_arr)
            img = img.resize((dx, dy))
            img.save(clip_path)
            print(f"头像裁剪已保存: {clip_path}")
            print(f"匹配区域: ({x1}, {y1}) -> ({x2}, {y2})")

            canvas = merge_layers(first_input_group.layers[:-1] + [
                {
                    'template': clip_path,
                    'mask': self.clip_mask
                }
            ])

            canvas.save(clip_path)

            layers.append({
                'template': clip_path
            })

        return [MergeChainNodeGroup(name='default', layers=layers)]


class ExpandClipChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, frame_paths: list[dict]):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 各头像分组的边框配置，构造时指定
        self.frame_paths = frame_paths

    def output(self) -> list[MergeChainNodeGroup]:
        first_input_group = self.input[0]
        frame_paths = self.frame_paths

        # 以 out_dir 为根目录，按 frame_paths 的 name 分别创建子文件夹 root / f"{name}"
        root = self.out_dir
        groups: list[MergeChainNodeGroup] = []
        for index, frame in enumerate(frame_paths):
            frame_dir = root / f"{frame['name']}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            layers: list[LayerSpec] = []
            for avatar_foreground in first_input_group.layers:
                avatar_foreground_path = avatar_foreground.get('template')
                canvas = merge_layers([
                    {
                        'template': frame['frame'],
                    },
                    {
                        'template': avatar_foreground_path,
                    }
                ])
                out_path = frame_dir / os.path.basename(avatar_foreground_path)
                canvas.save(out_path)
                # 以刚保存的图片路径作为图层 template
                layers.append({'template': out_path})

            # 以 name 作为分组名称，分组目录下的图片作为图层 template
            groups.append(MergeChainNodeGroup(name=frame['name'], layers=layers))

        return groups


class LabelChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, label_paths: list[dict], delete_old: bool = False):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 各分组标签图层配置，构造时指定
        self.label_paths = label_paths
        # 新分组图片生成后是否移除上一流程生成的旧图片，默认 False（不删除）
        self.delete_old = delete_old

    def output(self) -> list[MergeChainNodeGroup]:
        root = self.out_dir
        # 分组名 -> labels 图层配置
        labels_map = {str(cfg['name']): cfg['labels'] for cfg in self.label_paths}

        groups: list[MergeChainNodeGroup] = []
        for group in self.input:
            labels = labels_map.get(str(group.name), [])
            group_dir = root / group.name
            group_dir.mkdir(parents=True, exist_ok=True)

            new_layers: list[LayerSpec] = []
            old_paths: list = []
            # 交叉合成：每个输入图层 × 每个标签图层，labels=2 且 layers=10 时输出 20 个图层
            for layer in group.layers:
                avatar_path = layer.get('template')
                if avatar_path is None:
                    continue
                old_paths.append(avatar_path)
                for label_index, label_path in enumerate(labels):
                    # 标签叠加在分组图之上，按标签序号命名以区分同一头像的不同标签产物
                    canvas = merge_layers([
                        {'template': avatar_path},
                        {'template': label_path, 'fit': 'fill'},
                    ])
                    out_path = group_dir / f"{Path(str(avatar_path)).stem}_label{label_index}.png"
                    canvas.save(out_path)
                    new_layers.append({'template': out_path})

            # 输出分组名称不变，仅图层发生变化
            groups.append(MergeChainNodeGroup(name=group.name, layers=new_layers))

            # 新图片生成后，按需移除上一流程生成的旧图片
            if self.delete_old:
                for old_path in old_paths:
                    p = Path(str(old_path))
                    if p.exists():
                        p.unlink()

        return groups


class StatusChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, status_paths: list[dict], delete_old: bool = False):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 各分组满破标记图层配置，构造时指定
        self.status_paths = status_paths
        # 新分组图片生成后是否移除上一流程生成的旧图片，默认 False（不删除）
        self.delete_old = delete_old

    def output(self) -> list[MergeChainNodeGroup]:
        root = self.out_dir
        # 分组名 -> 满破标记图层配置
        status_map = {str(cfg['name']): cfg['status'] for cfg in self.status_paths}

        groups: list[MergeChainNodeGroup] = []
        for group in self.input:
            markers = status_map.get(str(group.name), [])
            # 本阶段该分组无标记配置时原样透传图层，不生成新图也不删除旧图
            if not markers:
                groups.append(MergeChainNodeGroup(name=group.name, layers=list(group.layers)))
                continue

            group_dir = root / group.name
            group_dir.mkdir(parents=True, exist_ok=True)

            new_layers: list[LayerSpec] = []
            old_paths: list = []
            # 交叉合成：每个输入图层 × 每个满破标记图层
            for layer in group.layers:
                avatar_path = layer.get('template')
                if avatar_path is None:
                    continue
                old_paths.append(avatar_path)
                for marker_index, marker_path in enumerate(markers):
                    canvas = merge_layers([
                        {'template': avatar_path},
                        {'template': marker_path, 'fit': 'fill'},
                    ])
                    out_path = group_dir / f"{Path(str(avatar_path)).stem}_status{marker_index}.png"
                    canvas.save(out_path)
                    new_layers.append({'template': out_path})

            # 输出分组名称不变，仅图层发生变化
            groups.append(MergeChainNodeGroup(name=group.name, layers=new_layers))

            # 新图片生成后，按需移除上一流程生成的旧图片
            if self.delete_old:
                for old_path in old_paths:
                    p = Path(str(old_path))
                    if p.exists():
                        p.unlink()

        return groups


class StarsChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, stars_paths: list[dict], delete_old: bool = False):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 各分组稀有度星标图层配置，构造时指定
        self.stars_paths = stars_paths
        # 新分组图片生成后是否移除上一流程生成的旧图片，默认 False（不删除）
        self.delete_old = delete_old

    def output(self) -> list[MergeChainNodeGroup]:
        root = self.out_dir
        # 分组名 -> 稀有度星标图层配置
        stars_map = {str(cfg['name']): cfg['stars'] for cfg in self.stars_paths}

        groups: list[MergeChainNodeGroup] = []
        for group in self.input:
            markers = stars_map.get(str(group.name), [])
            # 本阶段该分组无标记配置时原样透传图层，不生成新图也不删除旧图
            if not markers:
                groups.append(MergeChainNodeGroup(name=group.name, layers=list(group.layers)))
                continue

            group_dir = root / group.name
            group_dir.mkdir(parents=True, exist_ok=True)

            new_layers: list[LayerSpec] = []
            old_paths: list = []
            # 交叉合成：每个输入图层 × 每个星标图层
            for layer in group.layers:
                avatar_path = layer.get('template')
                if avatar_path is None:
                    continue
                old_paths.append(avatar_path)
                for marker_index, marker_path in enumerate(markers):
                    canvas = merge_layers([
                        {'template': avatar_path},
                        {'template': marker_path, 'fit': 'fill'},
                    ])
                    out_path = group_dir / f"{Path(str(avatar_path)).stem}_stars{marker_index}.png"
                    canvas.save(out_path)
                    new_layers.append({'template': out_path})

            # 输出分组名称不变，仅图层发生变化
            groups.append(MergeChainNodeGroup(name=group.name, layers=new_layers))

            # 新图片生成后，按需移除上一流程生成的旧图片
            if self.delete_old:
                for old_path in old_paths:
                    p = Path(str(old_path))
                    if p.exists():
                        p.unlink()

        return groups


class ClassChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, class_paths: list[dict], delete_old: bool = False):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 各分组职介标记图层配置，构造时指定
        self.class_paths = class_paths
        # 新分组图片生成后是否移除上一流程生成的旧图片，默认 False（不删除）
        self.delete_old = delete_old

    def output(self) -> list[MergeChainNodeGroup]:
        root = self.out_dir
        # 分组名 -> 职介标记图层配置
        class_map = {str(cfg['name']): cfg['class'] for cfg in self.class_paths}

        groups: list[MergeChainNodeGroup] = []
        for group in self.input:
            markers = class_map.get(str(group.name), [])
            # 本阶段该分组无标记配置时原样透传图层，不生成新图也不删除旧图
            if not markers:
                groups.append(MergeChainNodeGroup(name=group.name, layers=list(group.layers)))
                continue

            group_dir = root / group.name
            group_dir.mkdir(parents=True, exist_ok=True)

            new_layers: list[LayerSpec] = []
            old_paths: list = []
            # 交叉合成：每个输入图层 × 每个职介标记图层
            for layer in group.layers:
                avatar_path = layer.get('template')
                if avatar_path is None:
                    continue
                old_paths.append(avatar_path)
                for marker_index, marker_path in enumerate(markers):
                    canvas = merge_layers([
                        {'template': avatar_path},
                        {'template': marker_path, 'fit': 'fill'},
                    ])
                    out_path = group_dir / f"{Path(str(avatar_path)).stem}_class{marker_index}.png"
                    canvas.save(out_path)
                    new_layers.append({'template': out_path})

            # 输出分组名称不变，仅图层发生变化
            groups.append(MergeChainNodeGroup(name=group.name, layers=new_layers))

            # 新图片生成后，按需移除上一流程生成的旧图片
            if self.delete_old:
                for old_path in old_paths:
                    p = Path(str(old_path))
                    if p.exists():
                        p.unlink()

        return groups

class NormalizeChainNode(MergeChainNode):
    def __init__(self, input: list[MergeChainNodeGroup], out_dir: PathLike, output_size: tuple[int, int], background_color: tuple[int, int, int, int] = (0, 0, 0, 255), delete_old: bool = False):
        super().__init__(input)
        # 分组图片输出根目录，构造时指定
        self.out_dir = Path(out_dir)
        # 新分组图片生成后是否移除上一流程生成的旧图片，默认 False（不删除）
        self.delete_old = delete_old
        # 输出图片大小
        self.output_size = output_size
        # 输出图片背景色
        self.background_color = background_color
        

    def output(self) -> list[MergeChainNodeGroup]:
        root = self.out_dir
        cw, ch = self.output_size
        groups: list[MergeChainNodeGroup] = []
        for group in self.input:
            group_dir = root / group.name
            group_dir.mkdir(parents=True, exist_ok=True)

            new_layers: list[LayerSpec] = []
            old_paths: list = []
            for layer in group.layers:
                img_path = layer.get('template')
                if img_path is None:
                    continue
                old_paths.append(img_path)

                canvas = Image.new('RGBA', (cw, ch), self.background_color)

                # 读取图片并统一为 RGBA（parse_layer_image 已完成 BGR(A)->RGB(A) 转换）
                img_arr = parse_layer_image(img_path)
                if img_arr is None:
                    raise ValueError(f"图片解析失败: {img_path}")
                img = Image.fromarray(img_arr)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # 以画布中心为基准计算偏移，将图片居中放置
                offset = ((cw - img.width) // 2, (ch - img.height) // 2)
                # 第 3 个参数传入 img 自身作为 mask，保留透明区域不被覆盖
                canvas.paste(img, offset, img)

                out_path = group_dir / f"{Path(str(img_path)).stem}_normalized.png"
                print(f"保存图片: {out_path}")
                canvas.save(out_path)
                new_layers.append({'template': out_path})

            # 输出分组名称不变，仅图层发生变化
            groups.append(MergeChainNodeGroup(name=group.name, layers=new_layers))

            # 新图片生成后，按需移除上一流程生成的旧图片
            if self.delete_old:
                for old_path in old_paths:
                    p = Path(str(old_path))
                    if p.exists():
                        p.unlink()

        return groups
                