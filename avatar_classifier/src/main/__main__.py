import csv
import random
import shutil
import warnings
from dataclasses import dataclass
from os import listdir
from pathlib import Path

from PIL import Image

from main.detect_cv import detect_face, detect_anime_face
from main.merge_chain import ClassifyChainNode, BackgroundChainNode, MergeChainNodeGroup
from .config import RES_DIR, ROOT_DIR, BG_DIR, TEMP_DIR

OUT_PUT_DIR = TEMP_DIR

@dataclass
class Servant:
    id: int
    name_cn: str
    name_jp: str
    name_en: str
    page_url: str
    avatar: str
    card_rarity: str
    servant_class: str
    np_color: str
    np_type: str
    cards: str
    faction: str
    get: str
    atk_max: int | None
    hp_max: int | None

    @classmethod
    def from_csv(cls, path: Path | None = None) -> list["Servant"]:
        """从 servants.csv 读取所有 Servant 数据。"""
        if path is None:
            path = ROOT_DIR / "scripts" / "servants.csv"
        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [
                cls(
                    id=int(row["id"]),
                    name_cn=row["name_cn"],
                    name_jp=row["name_jp"],
                    name_en=row["name_en"],
                    page_url=row["page_url"],
                    avatar=row["avatar"],
                    card_rarity=row["card_rarity"],
                    servant_class=row["servant_class"],
                    np_color=row["np_color"],
                    np_type=row["np_type"],
                    cards=row["cards"],
                    faction=row["faction"],
                    get=row["get"],
                    atk_max=int(row["atk_max"]) if row["atk_max"] else None,
                    hp_max=int(row["hp_max"]) if row["hp_max"] else None,
                )
                for row in reader
            ]


def main():
    servants = Servant.from_csv()

    #单例测试
    #for s in servants[:1]:
    for s in servants[:1]:
        servant_id = s.id
        servant_res_dir = RES_DIR / "assets" / str(servant_id)
        if servant_res_dir.exists():
            print(f"Servant {servant_id} res folder exists.")
            # 对资源突破进行分拣
            classify_servant_res = ClassifyChainNode(input_dir=servant_res_dir).output()
            for c in classify_servant_res:
                # 分组
                group = c.name
                # 图片列表，提取 template 字段, 分类节点输出的类型必定是 Path
                img_list: list[Path] = [layer["template"] for layer in c.layers]

                match group:
                    # case 'status':
                        # 再临阶段
                    #    process_status_img_list(s, img_list)
                    #    break
                    # case 'avatar':
                        # wiki的头像图，用于立绘的头像模板匹配
                    #    process_avatar_img_list(s, img_list)
                    #    break
                    case 'character':
                        process_character_img_list(s, img_list)
                        break
                    case _:
                        print(f"Unknown group: {group}")

        else:
            warnings.warn(f"Servant {servant_id} res folder does not exist.")


def process_status_img_list(s: Servant, img_list: list[Path]):
    """
    背景融合
    降低立绘背景的特征依赖
    默认背景图尺寸大于 output_size, 比输入图片略大即可
    过大的背景图没有意义，还会报 DecompressionBombWarning:
    """
    output_size = (280, 280)

    # 单例测试
    # bg_img_list = listdir(BG_DIR)[-6:-5]
    bg_img_list = listdir(BG_DIR)

    bg_cropped_list = []
    for index, bg in enumerate(bg_img_list):
        bg_img = Image.open(BG_DIR / bg)
        print(f"背景图 {bg} {bg_img.size}")

        # 随机裁剪位置
        x = random.randint(0, bg_img.width - output_size[0])
        y = random.randint(0, bg_img.height - output_size[1])

        # 裁剪 (left, upper, right, lower)
        cropped_bg = bg_img.crop((x, y, x + output_size[0], y + output_size[1]))
        print(f"裁剪位置: ({x}, {y}), 裁剪后尺寸: {cropped_bg.size}")

        # 输入使用Path类型，存到临时文件夹，省点内存
        tmp_cropped_bg_path = OUT_PUT_DIR / f"{s.id}_bg_{index}.png"
        cropped_bg.convert("RGBA").save(tmp_cropped_bg_path)
        print(f"保存背景图 {tmp_cropped_bg_path}")

        bg_cropped_list.append(tmp_cropped_bg_path)

    for target_image_path in img_list:
        BackgroundChainNode(
            input=[
                MergeChainNodeGroup(
                    name='default',
                    layers=[
                        {
                            'template': bg_path_cropped
                        } for bg_path_cropped in bg_cropped_list
                    ]
                )
            ],
            alpha=0.85,
            target_image_path=target_image_path,
            out_dir=OUT_PUT_DIR,
            output_size=output_size,
        ).output()

    #删除临时文件
    for tmp_cropped_bg_path in bg_cropped_list:
        tmp_cropped_bg_path.unlink()


def process_avatar_img_list(s: Servant, img_list: list[Path]):
    # wiki 的头像直接复制即可
    for avatar in img_list:
        shutil.copy(avatar, OUT_PUT_DIR)


def process_character_img_list(s: Servant, img_list: list[Path]):
    # 满破、灵衣等没有提供头像图，实际上头像合成还要依赖 detect_face
    for character_img in img_list:
        print(f"检测：{character_img.name}")
        results = detect_anime_face(character_img)
        print(results)


if __name__ == '__main__':
    main()
