from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from chaldea.config import TEMP_DIR


ImageInput = Union[Path, np.ndarray, Image.Image]


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
