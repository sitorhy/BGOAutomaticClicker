from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from chaldea.config import BG_DIR, RES_DIR, TEMP_DIR


ImageInput = Union[Path, np.ndarray, Image.Image]

YELLOW = "\033[93m"
RESET = "\033[0m"

# 头像裁剪掩码，头像类素材默认按它设置透明通道
AVATAR_MASK_IMG = RES_DIR / "mask" / "BGO头像裁剪掩码.png"


class ChainNode:
    def __init__(self, image: ImageInput, alpha_mask: Union[Path, Image.Image, None] = None):
        if isinstance(image, Path):
            self._image_path = image
            self._image: Image.Image | None = None
        else:
            self._image_path = None
            self._image = self._to_image(image)
        self._image = self._image.convert("RGBA") if self._image is not None else None

        if isinstance(alpha_mask, Path):
            self._mask_path = alpha_mask
            self._mask: Image.Image | None = None
        elif alpha_mask is not None:
            self._mask_path = None
            self._mask = alpha_mask.convert("L")
        else:
            self._mask_path = None
            self._mask = None

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            self._image = Image.open(self._image_path).convert("RGBA")
        return self._image

    @image.setter
    def image(self, value: ImageInput) -> None:
        # 原地覆盖内存图（统一转 RGBA），image_path 仍指向原始文件不变
        self._image = self._to_image(value).convert("RGBA")

    @property
    def alpha_mask(self) -> Image.Image | None:
        if self._mask is None and self._mask_path is not None:
            self._mask = Image.open(self._mask_path).convert("L")
        return self._mask

    @staticmethod
    def _to_image(source: ImageInput) -> Image.Image:
        if isinstance(source, Path):
            return Image.open(source)
        if isinstance(source, np.ndarray):
            return Image.fromarray(source)
        return source

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)  # 触发懒加载

    def merge(self, foreground: ImageInput) -> ChainNode:
        fg_img = self._to_image(foreground).convert("RGBA")
        fg_img = fg_img.resize(self.image.size, Image.Resampling.LANCZOS)

        base = self.image
        if self.alpha_mask is not None:
            mask = self.alpha_mask
            if mask.size != base.size:
                mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            r, g, b, _ = base.split()
            base = Image.merge("RGBA", (r, g, b, mask))

        result = Image.alpha_composite(base, fg_img)
        return ChainNode(result)

class Chain:
    def __init__(self, nodes: list[ChainNode] | None = None):
        self.nodes = nodes or []

    def output(self) -> list[ChainNode]:
        return self.nodes

# 删除输入的序列文件
class UnlinkChain(Chain):
    def __init__(self, nodes: list[ChainNode]):
        super().__init__(nodes)

    def output(self) -> list[ChainNode]:
        for node in self.nodes:
            if node.image_path is not None and isinstance(node.image_path, Path):
                node.image_path.unlink(missing_ok=True)
                print(f"Deleted {node.image_path}")
        self.nodes = []
        return []





class CharaClassifyChain(Chain):
    def __init__(self, input_dir: Path):
        super().__init__()
        self.input_dir = Path(input_dir)

    def output(self) -> list[ChainNode]:
        nodes = []
        for subdir_name in ["charaGraph/ascension", "charaGraph/costume"]:
            subdir = self.input_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for img_path in subdir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        nodes.append(ChainNode(img_path))
        return nodes


class FaceClassifyChain(Chain):
    def __init__(self, input_dir: Path):
        super().__init__()
        self.input_dir = Path(input_dir)

    def output(self) -> list[ChainNode]:
        nodes = []
        for subdir_name in ["faces/ascension", "faces/costume"]:
            subdir = self.input_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for img_path in subdir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        nodes.append(ChainNode(img_path))
        return nodes


class NarrowFigureClassifyChain(Chain):
    def __init__(self, input_dir: Path):
        super().__init__()
        self.input_dir = Path(input_dir)

    def output(self) -> list[ChainNode]:
        nodes = []
        for subdir_name in ["narrowFigure/ascension", "narrowFigure/costume"]:
            subdir = self.input_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for img_path in subdir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        nodes.append(ChainNode(img_path))
        return nodes


class CommandsClassifyChain(Chain):
    def __init__(self, input_dir: Path):
        super().__init__()
        self.input_dir = Path(input_dir)

    def output(self) -> list[ChainNode]:
        nodes = []
        for subdir_name in ["commands/ascension", "commands/costume"]:
            subdir = self.input_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for img_path in subdir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        nodes.append(ChainNode(img_path))
        return nodes


class StatusClassifyChain(Chain):
    def __init__(self, input_dir: Path):
        super().__init__()
        self.input_dir = Path(input_dir)

    def output(self) -> list[ChainNode]:
        nodes = []
        for subdir_name in ["status/ascension", "status/costume"]:
            subdir = self.input_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for img_path in subdir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        nodes.append(ChainNode(img_path))
        return nodes


class ExpandClipChain(Chain):
    def __init__(
            self,
            target_img: Path,
            rect: tuple[int, int, int, int],
            ratio: float,
            min_scale: float = 0.85,
            max_scale: float = 1.5,
            steps: int = 5,
            output_dir: Path = TEMP_DIR,
            mask_img: Path | None = None,
    ):
        super().__init__()
        self.target_img = Path(target_img)
        self.rect = rect
        self.ratio = ratio
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.steps = steps
        self.output_dir = Path(output_dir)
        self.mask_img = Path(mask_img) if mask_img is not None else None

    def _base_rect(self, img_size: tuple[int, int]) -> tuple[int, int, int, int]:
        left, top, right, bottom = self.rect
        width = right - left
        height = bottom - top
        current_ratio = width / height

        cx = (left + right) / 2
        cy = (top + bottom) / 2

        # 沿中心向外扩展，使长宽比等于 ratio
        if current_ratio > self.ratio:
            new_height = width / self.ratio
            new_left, new_right = left, right
            new_top = cy - new_height / 2
            new_bottom = cy + new_height / 2
        elif current_ratio < self.ratio:
            new_width = height * self.ratio
            new_top, new_bottom = top, bottom
            new_left = cx - new_width / 2
            new_right = cx + new_width / 2
        else:
            new_left, new_top, new_right, new_bottom = left, top, right, bottom

        return self._clamp_rect(new_left, new_top, new_right, new_bottom, img_size)

    @staticmethod
    def _clamp_rect(
            left: float, top: float, right: float, bottom: float,
            img_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        img_width, img_height = img_size
        # 范围尺寸不超过图片尺寸
        width = min(right - left, img_width)
        height = min(bottom - top, img_height)
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        left = cx - width / 2
        right = cx + width / 2
        top = cy - height / 2
        bottom = cy + height / 2

        # 超出图片范围时沿同维度平移回图片内
        if left < 0:
            right -= left
            left = 0
        if right > img_width:
            left -= (right - img_width)
            right = img_width
        if top < 0:
            bottom -= top
            top = 0
        if bottom > img_height:
            top -= (bottom - img_height)
            bottom = img_height

        return int(left), int(top), int(right), int(bottom)

    def output(self) -> list[ChainNode]:
        img = Image.open(self.target_img).convert("RGBA")
        base_rect = self._base_rect(img.size)
        base_left, base_top, base_right, base_bottom = base_rect
        base_width = base_right - base_left
        base_height = base_bottom - base_top
        base_cx = (base_left + base_right) / 2
        base_cy = (base_top + base_bottom) / 2

        # steps 个线性插值缩放值，加上基准 1.0，共 steps + 1 个
        scales = [self.min_scale + (self.max_scale - self.min_scale) * i / (self.steps - 1)
                  for i in range(self.steps)] + [1.0]

        rects = []
        for scale in scales:
            scaled_width = base_width * scale
            scaled_height = base_height * scale
            rect = self._clamp_rect(
                base_cx - scaled_width / 2,
                base_cy - scaled_height / 2,
                base_cx + scaled_width / 2,
                base_cy + scaled_height / 2,
                img.size,
            )
            if rect not in rects:
                rects.append(rect)

        mask = Image.open(self.mask_img).convert("L") if self.mask_img is not None else None

        nodes = []
        for idx, rect in enumerate(rects):
            clip = img.crop(rect)
            if mask is not None:
                clip = clip.resize(mask.size, Image.Resampling.LANCZOS)
                clip.putalpha(mask)
            output_path = self.output_dir / f"{self.target_img.stem}_bg_{idx}.png"
            node = ChainNode(clip)
            node.save(output_path)
            nodes.append(ChainNode(output_path))
        return nodes


class OverlayChain(Chain):
    """图层叠加链的公共骨架：输入图序列 × 叠加图层交叉合成后落盘。

    子类实现 overlay_layers / output_name，按需覆写 prepare_target，
    并用 layer_on_top / skip_existing 声明图层顺序与重复输出策略。
    """

    # 叠加图层在输入图之上；置 False 则作为背景垫在输入图下方
    layer_on_top = True
    # 输出文件已存在时跳过写图，仅复用其路径
    skip_existing = False
    # 本进程内各叠加链已写出的文件，用于发现跨链/跨图层的同名输出
    _produced: set[Path] = set()

    def __init__(self, input_nodes: list[ChainNode], output_dir: Path = TEMP_DIR):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_dir = Path(output_dir)

    def prepare_target(self, target: Image.Image) -> Image.Image:
        return target

    def overlay_layers(self) -> list[Image.Image]:
        raise NotImplementedError

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        raise NotImplementedError

    def output(self) -> list[ChainNode]:
        layers = self.overlay_layers()

        nodes = []
        for target_idx, node in enumerate(self.input_nodes):
            target = self.prepare_target(node.image)
            for layer_idx, layer in enumerate(layers):
                top = layer.resize(target.size, Image.Resampling.LANCZOS)
                if self.layer_on_top:
                    result = Image.alpha_composite(target, top)
                else:
                    result = Image.alpha_composite(top, target)

                output_path = self.output_dir / self.output_name(node.image_path.stem, layer_idx, target_idx)
                if output_path in OverlayChain._produced:
                    print(f"{YELLOW}输出文件重名，本次进程已写出过 {output_path}{RESET}")
                OverlayChain._produced.add(output_path)

                if self.skip_existing and output_path.exists():
                    print(f"{YELLOW}跳过已存在的合成图 {output_path}{RESET}")
                else:
                    ChainNode(result).save(output_path)
                nodes.append(ChainNode(output_path))
        return nodes


class FrameChain(OverlayChain):
    layer_on_top = False

    def __init__(
            self,
            name: str,
            frame: Path,
            expand_chain: list[ChainNode],
            output_dir: Path = TEMP_DIR,
            mask: Path | None = None,
    ):
        super().__init__(expand_chain, output_dir)
        self.name = name
        self.frame = Path(frame)
        self.mask = Path(mask) if mask is not None else None

    def prepare_target(self, target: Image.Image) -> Image.Image:
        if self.mask is None or not self.mask.exists():
            return target
        mask = Image.open(self.mask).convert("L")
        # resize 返回新对象，putalpha 不会污染输入节点缓存的原图
        target = target.resize(mask.size, Image.Resampling.LANCZOS)
        target.putalpha(mask)
        return target

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(self.frame).convert("RGBA")]

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        return f"{stem}_{self.name}_{target_idx}.png"


class LabelChain(OverlayChain):
    def __init__(
            self,
            frame_chain: list[ChainNode],
            label_list: list[Path],
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(frame_chain, output_dir)
        self.label_list = [Path(p) for p in label_list]

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.label_list]

    def output_name(self, stem: str, label_idx: int, target_idx: int) -> str:
        return f"{stem}_label_{label_idx}_{target_idx}.png"


class StatusChain(OverlayChain):
    def __init__(
            self,
            label_chain: list[ChainNode],
            status_list: list[Path],
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(label_chain, output_dir)
        self.status_list = [Path(p) for p in status_list]

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.status_list]

    def output_name(self, stem: str, status_idx: int, target_idx: int) -> str:
        return f"{stem}_status_{status_idx}_{target_idx}.png"


class StarsChain(OverlayChain):
    skip_existing = True

    def __init__(
            self,
            status_chain: list[ChainNode],
            output_dir: Path,
            stars_list: list[Path],
            rank_name: str,
    ):
        super().__init__(status_chain, output_dir)
        self.stars_list = [Path(p) for p in stars_list]
        self.rank_name = rank_name

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.stars_list]

    def output_name(self, stem: str, stars_idx: int, target_idx: int) -> str:
        return f"{stem}_stars_{self.rank_name}_{stars_idx}_{target_idx}.png"


class ClassChain(OverlayChain):
    def __init__(
            self,
            stars_chain: list[ChainNode],
            class_list: list[Path],
            class_name: str,
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(stars_chain, output_dir)
        self.class_list = [Path(p) for p in class_list]
        self.class_name = class_name

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.class_list]

    def output_name(self, stem: str, class_idx: int, target_idx: int) -> str:
        return f"{stem}_{self.class_name}_{class_idx}_{target_idx}.png"


class PostChain(Chain):
    def __init__(self, chains: list[Chain]):
        super().__init__()
        self.chains = chains

    def output(self) -> list[ChainNode]:
        nodes = []
        for chain in self.chains:
            nodes.extend(chain.output())
        return nodes


class BackgroundClipChain(Chain):
    """背景随机裁剪链：把 bg_dir 下每张原图随机裁出 output_size 大小的切片。

    切片只留在内存里（ChainNode 持有图片对象），不落盘。
    """

    def __init__(self, bg_dir: Path = BG_DIR, output_size: tuple[int, int] = (280, 280)):
        super().__init__()
        self.bg_dir = Path(bg_dir)
        self.output_size = output_size

    def output(self) -> list[ChainNode]:
        clip_w, clip_h = self.output_size
        nodes = []
        for bg_path in sorted(self.bg_dir.iterdir()):
            if not bg_path.is_file() or bg_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            with Image.open(bg_path) as bg_img:
                if bg_img.width < clip_w or bg_img.height < clip_h:
                    print(f"{YELLOW}背景图 {bg_path.name} 尺寸 {bg_img.size} 小于切片 {self.output_size}，跳过{RESET}")
                    continue
                left = random.randint(0, bg_img.width - clip_w)
                top = random.randint(0, bg_img.height - clip_h)
                tile = bg_img.crop((left, top, left + clip_w, top + clip_h)).convert("RGBA")
            nodes.append(ChainNode(tile))
        return nodes


class BackgroundChain(Chain):
    """随机背景链：为 chain 里每张图随机挑一张背景切片垫在其后方，覆盖写回 node.image。

    画布尺寸以 node.image 为准：背景切片 resize 到该尺寸铺满底层，node.image 保持原始
    尺寸与透明通道叠在背景之上，因此只有它的透明区域会露出背景。结果原地覆盖，
    方便后续链直接复用同一批节点。
    """

    def __init__(self, bg_nodes: list[ChainNode], chain: list[ChainNode]):
        super().__init__()
        self.bg_nodes = bg_nodes
        self.chain = chain

    def output(self) -> list[ChainNode]:
        if not self.bg_nodes:
            print(f"{YELLOW}BackgroundChain 无可用背景切片，跳过{RESET}")
            return self.chain
        for node in self.chain:
            fg = node.image
            # resize 返回新图，不会污染被复用且带缓存的背景节点
            bg = random.choice(self.bg_nodes).image.resize(fg.size, Image.Resampling.LANCZOS)
            node.image = Image.alpha_composite(bg, fg)
            node.image.save(node.image_path)
        return self.chain


class FaceMixChain(Chain):
    """背景混合链：状态图边缘高斯模糊后居中叠在背景切片上，弱化立绘的背景特征依赖。

    画布大小以背景切片为准；前景先按 zoom 放大留出边缘模糊带，叠合时超出画布的部分被裁掉。
    每次叠合再对 subject 施加 scale_range 内的随机缩放和 offset_range 内的随机偏移，
    增强位置与尺度扰动（offset_range=0、scale_range=(1,1) 时退化为纯居中叠加）。
    """

    zoom: float = 1.5
    blur_radius: int = 15
    margin: int = 40

    def __init__(
            self,
            status_nodes: list[ChainNode],
            bg_nodes: list[ChainNode],
            output_dir: Path = TEMP_DIR,
            alpha: float = 0.92,
            scale_range: tuple[float, float] = (1.0, 1.5),
            offset_range: int = 20,
    ):
        super().__init__()
        self.status_nodes = status_nodes
        self.bg_nodes = bg_nodes
        self.output_dir = Path(output_dir)
        self.alpha = alpha
        # 合成 subject 时的随机缩放区间与随机偏移像素半径（0 表示不偏移）
        self.scale_range = scale_range
        self.offset_range = offset_range

    def _blur_edges(self, target: Image.Image) -> Image.Image:
        """放大 target，把其边缘带替换成高斯模糊结果，中心保持清晰"""
        zoom_w = max(1, round(target.width * self.zoom))
        zoom_h = max(1, round(target.height * self.zoom))
        left = (zoom_w - target.width) // 2
        top = (zoom_h - target.height) // 2

        canvas = Image.new("RGBA", (zoom_w, zoom_h), (255, 255, 255, 0))
        canvas.alpha_composite(target, (left, top))

        # “边缘白、中心黑”遮罩：Image.composite 在 mask=255 处取模糊图、mask=0 处取清晰图
        mask = Image.new("L", (zoom_w, zoom_h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, left + target.width, top + target.height], fill=255)
        draw.rectangle(
            [left + self.margin, top + self.margin,
             left + target.width - self.margin, top + target.height - self.margin],
            fill=0,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=self.margin / 2))

        result = Image.composite(
            canvas.filter(ImageFilter.GaussianBlur(radius=self.blur_radius)), canvas, mask
        )

        if self.alpha is not None and self.alpha < 1.0:
            r, g, b, a = result.split()
            a = a.point(lambda p: int(p * self.alpha))
            result = Image.merge("RGBA", (r, g, b, a))
        return result

    def _random_scale(self, target: Image.Image) -> Image.Image:
        """在 scale_range 内随机取一个比例缩放 target，比例恒为 1 时原样返回"""
        lo, hi = self.scale_range
        if lo == hi == 1.0:
            return target
        scale = random.uniform(lo, hi)
        size = (max(1, round(target.width * scale)), max(1, round(target.height * scale)))
        return target.resize(size, Image.Resampling.LANCZOS)

    def output(self) -> list[ChainNode]:
        nodes = []
        for target_idx, status_node in enumerate(self.status_nodes):
            base_subject = self._blur_edges(status_node.image)
            for bg_idx, bg_node in enumerate(self.bg_nodes):
                # copy 后再叠加，避免污染被复用且带缓存的背景节点
                canvas = bg_node.image.copy()
                # 每次合成都随机缩放 subject，居中后再叠加随机偏移，超出画布的部分被裁掉
                subject = self._random_scale(base_subject)
                offset_x = (canvas.width - subject.width) // 2
                offset_y = (canvas.height - subject.height) // 2
                if self.offset_range > 0:
                    offset_x += random.randint(-self.offset_range, self.offset_range)
                    offset_y += random.randint(-self.offset_range, self.offset_range)
                canvas.alpha_composite(subject, (offset_x, offset_y))

                output_path = self.output_dir / f"{status_node.image_path.stem}_mesh_{bg_idx}_{target_idx}.png"
                ChainNode(canvas).save(output_path)
                nodes.append(ChainNode(output_path))
        return nodes


class ExpandCropChain(Chain):
    """礼装位遮盖链：把头像裁剪图的礼装位替换成随机背景切片。

    礼装位显示的是好友所带概念礼装，与本从者身份无关且不可控，整块盖掉避免模型学到
    这块内容。产物是孤立的训练素材，不再作为后续叠加链的输入。
    两张掩码分工不同：bg_mask_img 圈出礼装位，只决定背景切片盖哪里；
    mask_img 决定产物自身的透明通道，即头像的裁剪形状。
    输出尺寸以 bg_mask_img 为准，背景切片和裁剪图都先对齐到该尺寸，调用方无需预处理。
    """

    def __init__(
            self,
            frame_chain: list[ChainNode],
            bg_mask_img: Path,
            bg_nodes: list[ChainNode]
    ):
        super().__init__()
        self.frame_chain = frame_chain
        self.bg_mask_img = Path(bg_mask_img)
        self.bg_nodes = bg_nodes

    def output(self) -> list[ChainNode]:
        bg_mask = Image.open(self.bg_mask_img).convert("L")

        # 再原图的基础上修改
        for target_idx, frame_node in enumerate(self.frame_chain):
            # 模拟外围背景
            bg_index = random.randint(0, len(self.bg_nodes) - 1)

            # 模拟礼装
            bg_clip_index = random.randint(0, len(self.bg_nodes) - 1)

            bg_img = self.bg_nodes[bg_index].image.copy()
            bg_clip = self.bg_nodes[bg_clip_index].image.copy()
            bg_clip.putalpha(bg_mask)

            # 叠加
            bg_img.paste(frame_node.image, (0,0), frame_node.image)
            bg_img.paste(bg_clip, (0,0), bg_clip)
            if frame_node.image_path is not None:
                bg_img.save(frame_node.image_path)

        return self.frame_chain


class CopyChain(Chain):
    """文件复制链：把输入节点的图片文件原样复制到输出目录"""

    def __init__(self, input_nodes: list[ChainNode], output_dir: Path = TEMP_DIR):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_dir = Path(output_dir)

    def output(self) -> list[ChainNode]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        nodes = []
        for node in self.input_nodes:
            output_path = self.output_dir / node.image_path.name
            shutil.copy2(node.image_path, output_path)
            nodes.append(ChainNode(output_path))
        return nodes


class NarrowFigureFrameChain(Chain):
    """助战背景链：把立绘按 offset 落在 L0 背景画布上，画布尺寸以 L0 为准。

    不能套 OverlayChain：OverlayChain 把图层 resize 到 target 尺寸，而这里要保留
    立绘原始尺寸、只移动落点，缩放方向是反的。
    """

    def __init__(
            self,
            name: str,
            frame: Path,
            narrow_figure_chain: list[ChainNode],
            output_dir: Path = TEMP_DIR,
            offset: tuple[int, int] = (11, 12),
            mask: Path | None = None,
    ):
        super().__init__()
        self.name = name
        self.frame = Path(frame)
        self.narrow_figure_chain = narrow_figure_chain
        self.output_dir = Path(output_dir)
        self.offset = offset
        self.mask = Path(mask) if mask is not None else None

    def output(self) -> list[ChainNode]:
        nodes = []
        for target_idx, node in enumerate(self.narrow_figure_chain):
            # 每张输出都从背景原图重新起一张画布，避免叠加互相污染
            canvas = Image.open(self.frame).convert("RGBA")
            canvas.alpha_composite(node.image, self.offset)

            if self.mask is not None:
                mask = Image.open(self.mask).convert("L")
                if mask.size != canvas.size:
                    mask = mask.resize(canvas.size, Image.Resampling.LANCZOS)
                r, g, b, a = canvas.split()
                canvas = Image.merge("RGBA", (r, g, b, ImageChops.multiply(a, mask)))

            output_path = self.output_dir / f"{node.image_path.stem}_{self.name}_{target_idx}.png"
            ChainNode(canvas).save(output_path)
            nodes.append(ChainNode(output_path))
        return nodes


class NarrowFigureBorderChain(OverlayChain):
    """助战边框链：L1 描边图层覆盖在卡面之上"""

    def __init__(
            self,
            frame_chain: list[ChainNode],
            border_list: list[Path],
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(frame_chain, output_dir)
        self.border_list = [Path(p) for p in border_list]

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.border_list]

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        return f"{stem}_border_{layer_idx}_{target_idx}.png"


class NarrowFigureLabelChain(OverlayChain):
    """助战标签链：L2 助战序号图层覆盖在卡面之上"""

    def __init__(
            self,
            border_chain: list[ChainNode],
            label_list: list[Path],
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(border_chain, output_dir)
        self.label_list = [Path(p) for p in label_list]

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.label_list]

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        return f"{stem}_label_{layer_idx}_{target_idx}.png"


class NarrowFigureStarsChain(OverlayChain):
    """助战星级链：L3 星级图层覆盖在卡面之上，rank_name 带出分路避免撞名"""

    def __init__(
            self,
            label_chain: list[ChainNode],
            stars_list: list[Path],
            rank_name: str,
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(label_chain, output_dir)
        self.stars_list = [Path(p) for p in stars_list]
        self.rank_name = rank_name

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.stars_list]

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        return f"{stem}_stars_{self.rank_name}_{layer_idx}_{target_idx}.png"


class NarrowFigureClassChain(OverlayChain):
    """助战职阶链：职阶徽标图层覆盖在卡面之上，为最终阶段产物"""

    def __init__(
            self,
            stars_chain: list[ChainNode],
            class_list: list[Path],
            class_name: str,
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__(stars_chain, output_dir)
        self.class_list = [Path(p) for p in class_list]
        self.class_name = class_name

    def overlay_layers(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGBA") for p in self.class_list]

    def output_name(self, stem: str, layer_idx: int, target_idx: int) -> str:
        return f"{stem}_{self.class_name}_{layer_idx}_{target_idx}.png"


class NarrowFigureCraftChain(Chain):
    """礼装合成链：在合成完成的助战卡面上，用随机背景切片盖住礼装位。

    礼装位显示的是好友所带概念礼装，与本从者身份无关且不可控，
    用随机背景遮掉，避免模型学到这块内容。切片按掩码画布尺寸裁剪，
    掩码亮度直接作为切片 alpha，切片与卡面同为 170x400，直接对齐叠合。
    """

    def __init__(
            self,
            class_chain: list[ChainNode],
            crop_mask: Path,
            bg_nodes: list[ChainNode],
            background_list: list[Path] | None = None,
            output_dir: Path = TEMP_DIR,
    ):
        super().__init__()
        self.class_chain = class_chain
        self.crop_mask = Path(crop_mask)
        self.background_list = background_list
        self.output_dir = Path(output_dir)
        self.bg_nodes = bg_nodes

    def _tiles(self) -> list[Image.Image]:
        """随机挑背景，按掩码画布尺寸裁一块并把掩码亮度设为 alpha"""
        mask = Image.open(self.crop_mask)
        mask_w, mask_h = mask.size

        tiles = []
        for node in self.bg_nodes:
            bg_img = node.image
            bg_img = bg_img.convert("RGBA")
            if bg_img.width < mask_w or bg_img.height < mask_h:
                scale = max(mask_w / bg_img.width, mask_h / bg_img.height)
                bg_img = bg_img.resize(
                    (round(bg_img.width * scale), round(bg_img.height * scale)),
                    Image.Resampling.LANCZOS)
            left = random.randint(0, bg_img.width - mask_w)
            top = random.randint(0, bg_img.height - mask_h)
            tile = bg_img.crop((left, top, left + mask_w, top + mask_h))

            tile.putalpha(mask.convert("L"))
            tiles.append(tile)
        return tiles

    def output(self) -> list[ChainNode]:
        tiles = self._tiles()
        nodes = []
        for target_idx, node in enumerate(self.class_chain):
            # 随机选一张背景
            tile_index = random.randint(0, len(tiles) - 1)
            # alpha_composite 返回新图，不污染卡面节点的缓存
            canvas = Image.alpha_composite(node.image, tiles[tile_index])

            output_path = self.output_dir / f"{node.image_path.stem}_craft_{target_idx}.png"
            ChainNode(canvas).save(output_path)
            nodes.append(ChainNode(output_path))
        return nodes


class ChaldeaJsonMerger:
    def __init__(self, json_files: list[Path]):
        self.json_files = [Path(f) for f in json_files]

    def merge(self) -> dict[int, dict]:
        result = {}
        for f in self.json_files:
            data = json.loads(f.read_text(encoding="utf-8"))
            for servant in data:
                result[servant["collectionNo"]] = servant
        return result


class CharaFacePairer:
    def __init__(self, chara_chain: CharaClassifyChain, face_chain: FaceClassifyChain, merged_json: dict):
        self.chara_chain = chara_chain
        self.face_chain = face_chain
        self.merged_json = merged_json
        self.collection_no = int(chara_chain.input_dir.name)

    def output(self) -> list[tuple[ChainNode, ChainNode]]:
        servant = self.merged_json.get(self.collection_no)
        if not servant:
            return []

        chara_by_name = {}
        for node in self.chara_chain.output():
            if node.image_path:
                chara_by_name[node.image_path.name] = node

        face_by_name = {}
        for node in self.face_chain.output():
            if node.image_path:
                face_by_name[node.image_path.name] = node

        pairs = []
        extra = servant.get("extraAssets", {})

        for category in ["ascension", "costume"]:
            chara_graph = extra.get("charaGraph", {}).get(category, {})
            faces = extra.get("faces", {}).get(category, {})

            for key in chara_graph:
                if key not in faces:
                    continue
                chara_filename = chara_graph[key].rsplit("/", 1)[-1]
                face_filename = faces[key].rsplit("/", 1)[-1]

                chara_node = chara_by_name.get(chara_filename)
                face_node = face_by_name.get(face_filename)

                if chara_node and face_node:
                    pairs.append((chara_node, face_node))

        return pairs


@dataclass
class AvatarDetectResult:
    chara_path: Path
    face_path: Path
    scale: float
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class AvatarDetectChain:
    def __init__(
            self,
            pairer: CharaFacePairer,
            min_scale: float = 0.85,
            max_scale: float = 3.0,
            steps: int = 25,
    ):
        self.pairer = pairer
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.steps = steps

    def output(self) -> list[AvatarDetectResult]:
        from mooncell.detect_cv import detect_avatar

        results = []
        for chara_node, face_node in self.pairer.output():
            chara_path = chara_node.image_path
            face_path = face_node.image_path
            if not chara_path or not face_path:
                continue

            detect_results = detect_avatar(
                target=chara_path,
                template=face_path,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                steps=self.steps,
            )
            best = detect_results[0]
            results.append(AvatarDetectResult(
                chara_path=chara_path,
                face_path=face_path,
                scale=best["scale"],
                score=best["score"],
                x=best["x"],
                y=best["y"],
                w=best["w"],
                h=best["h"],
            ))

        return results
