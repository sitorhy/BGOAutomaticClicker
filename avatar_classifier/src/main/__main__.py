import csv
import random
import shutil
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import listdir
from pathlib import Path

from PIL import Image

from main.config import RES_DIR, ROOT_DIR, BG_DIR, TEMP_DIR
from main.merge_chain import ClassifyChainNode, BackgroundChainNode, MergeChainNodeGroup, \
    ExpandClipChainNode, LabelChainNode, StatusChainNode, StarsChainNode, ClassChainNode

OUT_PUT_DIR = TEMP_DIR
OUT_PUT_DIR.mkdir(exist_ok=True)


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

    # 单例测试
    # for s in servants[:1]:
    for s in servants:
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
                    case 'status':
                        process_status_img_list(s, img_list)
                    case 'avatar':
                       process_avatar_img_list(s, img_list)
                    case _:
                        print(f"Unknown group: {group}")

        else:
            warnings.warn(f"Servant {servant_id} res folder does not exist.")

    # character 分支单独执行，基于数据库全量数据生成头像
    process_character_img_list()


def process_one_status_image(target_image_path, bg_cropped_list, output_dir, output_size):
    """处理单张状态图片的背景融合。"""
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
        alpha=0.92,
        target_image_path=target_image_path,
        out_dir=output_dir,
        output_size=output_size,
    ).output()


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

    output_dir = OUT_PUT_DIR / f"{s.id}"
    output_dir.mkdir(exist_ok=True)

    # 使用多线程处理每张图片
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [
            pool.submit(
                process_one_status_image,
                target_image_path,
                bg_cropped_list,
                output_dir,
                output_size
            )
            for target_image_path in img_list
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理状态图片时出错: {e}")

    # 删除临时文件
    for tmp_cropped_bg_path in bg_cropped_list:
        tmp_cropped_bg_path.unlink()


def process_avatar_img_list(s: Servant, img_list: list[Path]):
    # wiki 的头像直接复制即可
    for avatar in img_list:
        Path(OUT_PUT_DIR / f"{s.id}").mkdir(parents=True, exist_ok=True)
        shutil.copy(avatar, OUT_PUT_DIR / f"{s.id}")


def process_one_servant(svt, conn, img_mask_path, output_base_dir, res_dir, fg_dir):
    """处理单个从者的头像生成，每个线程独立打开 img_mask。"""
    img_mask = Image.open(img_mask_path)
    try:
        is_revoke = svt['revoke']

        # 筛选出可以获取的从者条目
        if is_revoke == 0:
            print(f"Processing {svt['id']} {svt['name_cn']} {svt['servant_class']}")
            avatar_infos = conn.execute("SELECT * FROM avatar WHERE servant_id = ?", (int(svt['id']),)).fetchall()
            for index, info in enumerate(avatar_infos):
                img_file_name = info['file']
                target_image_path = res_dir / "assets" / f"{svt['id']}" / img_file_name
                img = Image.open(target_image_path).convert("RGBA")

                left = int(info['left'])
                top = int(info['top'])
                right = int(info['right'])
                bottom = int(info['bottom'])

                # 微调比例，沿中心向外扩大裁剪范围，使长宽比等于 AVATRT_MASK_IMG_PATH
                mask_ratio = img_mask.size[0] / img_mask.size[1]  # width / height
                width = right - left
                height = bottom - top
                current_ratio = width / height

                cx = (left + right) / 2
                cy = (top + bottom) / 2

                if current_ratio > mask_ratio:
                    # 当前裁剪太宽，需要增加高度
                    new_height = width / mask_ratio
                    new_left, new_right = left, right
                    new_top = cy - new_height / 2
                    new_bottom = cy + new_height / 2
                elif current_ratio < mask_ratio:
                    # 当前裁剪太高，需要增加宽度
                    new_width = height * mask_ratio
                    new_top, new_bottom = top, bottom
                    new_left = cx - new_width / 2
                    new_right = cx + new_width / 2
                else:
                    new_left, new_top, new_right, new_bottom = left, top, right, bottom

                # 边界检查：如果超出图片范围，沿同维度平移回图片内
                img_width, img_height = img.size
                if new_left < 0:
                    new_right -= new_left
                    new_left = 0
                if new_right > img_width:
                    new_left -= (new_right - img_width)
                    new_right = img_width
                if new_top < 0:
                    new_bottom -= new_top
                    new_top = 0
                if new_bottom > img_height:
                    new_top -= (new_bottom - img_height)
                    new_bottom = img_height

                # 基准裁剪范围（已调整为 mask 比例并完成边界回退）
                base_left, base_top, base_right, base_bottom = int(new_left), int(new_top), int(new_right), int(
                    new_bottom)
                base_width = base_right - base_left
                base_height = base_bottom - base_top
                base_cx = (base_left + base_right) / 2
                base_cy = (base_top + base_bottom) / 2

                min_scale = 0.8
                max_scale = 1.5
                steps = 5

                # 生成 steps 个线性插值的缩放值，加上原始 1.0，共 steps + 1 个
                scales = [min_scale + (max_scale - min_scale) * i / (steps - 1) for i in range(steps)] + [1.0]

                output_dir = output_base_dir / f"{svt['id']}"
                output_dir.mkdir(exist_ok=True)
                img_width, img_height = img.size

                for scale_idx, scale in enumerate(scales):
                    # 沿基准中心按 scale 倍数缩放裁剪范围
                    scaled_width = base_width * scale
                    scaled_height = base_height * scale
                    s_left = base_cx - scaled_width / 2
                    s_top = base_cy - scaled_height / 2
                    s_right = base_cx + scaled_width / 2
                    s_bottom = base_cy + scaled_height / 2

                    # 边界检查：超出图片范围时沿同维度平移回图片内
                    if s_left < 0:
                        s_right -= s_left
                        s_left = 0
                    if s_right > img_width:
                        s_left -= (s_right - img_width)
                        s_right = img_width
                    if s_top < 0:
                        s_bottom -= s_top
                        s_top = 0
                    if s_bottom > img_height:
                        s_top -= (s_bottom - img_height)
                        s_bottom = img_height

                    s_left, s_top, s_right, s_bottom = int(s_left), int(s_top), int(s_right), int(s_bottom)

                    img_clip = img.crop((s_left, s_top, s_right, s_bottom))
                    zoom_img_clip = img_clip.resize(img_mask.size)

                    output_avatar_clip_path = output_dir / f"avatar_source_{svt['id']}_{index}_{scale_idx}.png"
                    # 输出裁剪源
                    zoom_img_clip.save(output_avatar_clip_path)

                    # 裁剪源设置 alpha 通道
                    zoom_img_clip.putalpha(img_mask.convert("L"))
                    output_avatar_clip_path = output_dir / f"avatar_input_{svt['id']}_{index}_{scale_idx}.png"
                    zoom_img_clip.save(output_avatar_clip_path)

                    card_rarity = svt['card_rarity']

                    default_frame_paths = [
                        {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
                        {'name': '银', 'frame': fg_dir / 'L0_银框.png'},
                        {'name': '铜', 'frame': fg_dir / 'L0_铜框.png'},
                        {'name': '铁', 'frame': fg_dir / 'L0_铁框.png'},
                        {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
                        {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
                    ]

                    frame_paths_for_card_rarity = {
                        '银卡': [
                            {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
                            {'name': '银', 'frame': fg_dir / 'L0_银框.png'},
                            {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
                            {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
                        ],
                        '金卡': [
                            {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
                            {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
                            {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
                        ],
                        '铜卡': [
                            {'name': '金', 'frame': fg_dir / 'L0_金框.png'},
                            {'name': '银', 'frame': fg_dir / 'L0_银框.png'},
                            {'name': '铜', 'frame': fg_dir / 'L0_铜框.png'},
                            {'name': '铁', 'frame': fg_dir / 'L0_铁框.png'},
                            {'name': '冠位', 'frame': fg_dir / 'L0_冠位框.png'},
                            {'name': '满级冠位', 'frame': fg_dir / 'L0_满级冠位框.png'},
                        ]
                    }

                    if card_rarity in frame_paths_for_card_rarity:
                        frame_paths = frame_paths_for_card_rarity[card_rarity]
                    else:
                        frame_paths = default_frame_paths

                    expand_chain_node_output = ExpandClipChainNode(
                        input=[
                            MergeChainNodeGroup(
                                name='default',
                                layers=[
                                    {
                                        'template': output_avatar_clip_path,
                                    }
                                ]
                            )
                        ],
                        out_dir=output_dir,
                        frame_paths=frame_paths,
                    ).output()

                    default_label_paths = [
                        {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
                        {'name': '银', 'labels': [fg_dir / 'L1_银标.png']},
                        {'name': '铜', 'labels': [fg_dir / 'L1_铜标.png']},
                        {'name': '铁', 'labels': [fg_dir / 'L1_铁标.png']},
                        {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
                        {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
                    ]

                    frame_paths_for_card_rarity = {
                        '银卡': [
                            {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
                            {'name': '银', 'labels': [fg_dir / 'L1_银标.png']},
                            {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
                            {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
                        ],
                        '金卡': [
                            {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
                            {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
                            {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
                        ],
                        '铜卡': [
                            {'name': '金', 'labels': [fg_dir / 'L1_金标.png', fg_dir / 'L1_满级金标.png']},
                            {'name': '银', 'labels': [fg_dir / 'L1_银标.png']},
                            {'name': '铜', 'labels': [fg_dir / 'L1_铜标.png']},
                            {'name': '冠位', 'labels': [fg_dir / 'L1_冠位标.png']},
                            {'name': '满级冠位', 'labels': [fg_dir / 'L1_满级冠位标.png']},
                        ]
                    }

                    if card_rarity in frame_paths_for_card_rarity:
                        label_paths = frame_paths_for_card_rarity[card_rarity]
                    else:
                        label_paths = default_label_paths

                    label_groups = LabelChainNode(
                        input=expand_chain_node_output,
                        out_dir=output_dir,
                        label_paths=label_paths,
                    ).output()

                    # 固定，默认全部满破
                    status_groups = StatusChainNode(
                        input=label_groups,
                        out_dir=output_dir,
                        status_paths=[
                            {'name': name, 'status': [fg_dir / 'L1_满破标.png']}
                            for name in ('金', '银', '铜', '铁', '冠位', '满级冠位')
                        ],
                    ).output()

                    stars_chain_for_card_rarity = {
                        '金卡': [
                            {
                                'name': '金',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_4星.png',
                                    'L1_4星再临.png',
                                    'L1_5星.png',
                                    'L1_5星再临.png',
                                ]]
                            },
                            {
                                'name': '冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_4星满级.png',
                                    'L1_5星满级.png',
                                ]]
                            },
                            {
                                'name': '满级冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_4星冠位.png',
                                    'L1_5星冠位.png'
                                ]]
                            }
                        ],
                        '银卡': [
                            {
                                'name': '银',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_3星.png',
                                    'L1_3星再临.png',
                                ]]
                            },
                            {
                                'name': '冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_3星冠位.png',
                                ]]
                            },
                            {
                                'name': '满级冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_3星满级.png'
                                ]]
                            }
                        ],
                        '铜卡': [
                            {
                                'name': '铜',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_1星.png',
                                    'L1_1星再临.png',
                                    'L1_2星.png',
                                    'L1_2星再临.png',
                                ]]
                            },
                            {
                                'name': '冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_1星冠位.png',
                                    'L1_2星冠位.png',
                                ]]
                            },
                            {
                                'name': '满级冠位',
                                'stars': [Path(RES_DIR / "foreground" / x) for x in [
                                    'L1_1星满级.png',
                                    'L1_2星满级.png'
                                ]]
                            }
                        ],
                    }

                    if card_rarity in stars_chain_for_card_rarity:
                        stars_paths = stars_chain_for_card_rarity[card_rarity]
                    else:
                        stars_paths = stars_chain_for_card_rarity['铜卡']

                    stars_groups = StarsChainNode(
                        input=status_groups,
                        out_dir=output_dir,
                        stars_paths=stars_paths,
                    ).output()

                    servant_class = svt['servant_class']

                    # 验证职阶是否在已知范围内
                    valid_classes = [
                        'Shielder', 'Saber', 'Archer', 'Lancer', 'Rider', 'Caster',
                        'Assassin', 'Berserker', 'Ruler', 'Avenger', 'Alterego',
                        'MoonCancer', 'Foreigner', 'Pretender', 'Beast'
                    ]
                    if servant_class not in valid_classes:
                        warnings.warn(f"未知的职阶: {servant_class}")

                    class_chain_for_card_rarity = {
                        '金卡': [
                            {'name': '金', 'class': [fg_dir / f'金卡{servant_class}.png']},
                            {'name': '冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                            {'name': '满级冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                        ],
                        '银卡': [
                            {'name': '金', 'class': [fg_dir / f'金卡{servant_class}.png']},
                            {'name': '银', 'class': [fg_dir / f'银卡{servant_class}.png']},
                            {'name': '冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                            {'name': '满级冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                        ],
                        '铜卡': [
                            {'name': '金', 'class': [fg_dir / f'金卡{servant_class}.png']},
                            {'name': '银', 'class': [fg_dir / f'银卡{servant_class}.png']},
                            {'name': '铜', 'class': [fg_dir / f'铜卡{servant_class}.png']},
                            {'name': '冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                            {'name': '满级冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                        ],
                        # 黑卡只有一张
                        '黑卡': [
                            {'name': '铁', 'class': [fg_dir / f'黑卡{servant_class}.png']},
                            {'name': '冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                            {'name': '满级冠位', 'class': [fg_dir / f'冠位{servant_class}.png']},
                        ]
                    }

                    if card_rarity in class_chain_for_card_rarity:
                        class_paths = class_chain_for_card_rarity[card_rarity]
                    else:
                        class_paths = class_chain_for_card_rarity['黑卡']

                    # 检查职阶文件是否存在，不存在则从 class 中移除
                    filtered_class_paths = []
                    for item in class_paths:
                        existing_paths = []
                        for path in item['class']:
                            if path.exists():
                                existing_paths.append(path)
                            else:
                                warnings.warn(f"职阶文件不存在，已跳过: {path}")
                        if existing_paths:
                            filtered_class_paths.append({**item, 'class': existing_paths})
                    class_paths = filtered_class_paths

                    ClassChainNode(
                        input=stars_groups,
                        out_dir=output_dir,
                        class_paths=class_paths,
                    ).output()
                
                    # ── 助战合成流程 ──
                    # 背景图 L0 映射：card_rarity -> 助战框底图
                    # frame_bg_map = {
                    #     '金卡': fg_dir / '金助战框.png',
                    #     '银卡': fg_dir / '银助战框.png',
                    #     '铜卡': fg_dir / '铜助战框.png',
                    #     '': fg_dir / '铁助战框.png',       # 黑卡 card_rarity 为空字符串
                    #     '黑卡': fg_dir / '铁助战框.png',
                    #     '冠位': fg_dir / '冠位助战框.png',
                    # }
                
                    # 前景图序列：铜卡可升级为银卡和金卡，从当前稀有度开始取，一直到冠位
                    # support_paths_for_card_rarity = {
                    #     '铜卡': [
                    #         {'name': '铜', 'support': [fg_dir / '铜助战框前景.png', fg_dir / '铜助战框前景2.png']},
                    #         {'name': '银', 'support': [fg_dir / '银助战框前景.png', fg_dir / '银助战框前景2.png']},
                    #         {'name': '金', 'support': [fg_dir / '金助战框前景.png', fg_dir / '金助战框前景2.png']},
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    #     '银卡': [
                    #         {'name': '银', 'support': [fg_dir / '银助战框前景.png', fg_dir / '银助战框前景2.png']},
                    #         {'name': '金', 'support': [fg_dir / '金助战框前景.png', fg_dir / '金助战框前景2.png']},
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    #     '金卡': [
                    #         {'name': '金', 'support': [fg_dir / '金助战框前景.png', fg_dir / '金助战框前景2.png']},
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    #     # 黑卡只有一张，不可升级
                    #     '': [
                    #         {'name': '铁', 'support': [fg_dir / '铁助战框前景.png', fg_dir / '铁助战框前景2.png']},
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    #     '黑卡': [
                    #         {'name': '铁', 'support': [fg_dir / '铁助战框前景.png', fg_dir / '铁助战框前景2.png']},
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    #     '冠位': [
                    #         {'name': '冠位', 'support': [fg_dir / '冠位助战框前景.png', fg_dir / '冠位助战框前景2.png']},
                    #     ],
                    # }
                    #
                    # output_dir = OUT_PUT_DIR / f"{svt['id']}" / 'support'
                    # output_dir.mkdir(parents=True, exist_ok=True)

                    # 拍扁目录：将子目录中的文件移动到 output_dir，然后删除子目录
                    for sub in output_dir.iterdir():
                        if sub.is_dir():
                            for file in sub.iterdir():
                                if file.is_file():
                                    shutil.move(str(file), str(output_dir / file.name))
                            shutil.rmtree(sub)

                img.close()
    finally:
        img_mask.close()


def process_character_img_list():
    DB_PATH = ROOT_DIR / "servants.db"
    AVATRT_MASK_IMG_PATH = RES_DIR / "mask" / "BGO头像裁剪掩码.png"
    fg_dir = RES_DIR / "foreground"

    # 主线程创建连接，只读场景下加 check_same_thread=False 可跨线程共享
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    servants = conn.execute("select * from servants").fetchall()

    # 使用多线程处理每个从者
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [
            pool.submit(
                process_one_servant,
                svt,
                conn,
                AVATRT_MASK_IMG_PATH,
                OUT_PUT_DIR,
                RES_DIR,
                fg_dir
            )
            for svt in servants
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理从者时出错: {e}")

    conn.close()


if __name__ == '__main__':
    main()
