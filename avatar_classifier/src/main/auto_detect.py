"""
根据仅有的wiki头像自动探测立绘头像位置
"""
import re
import sqlite3
import warnings
from pathlib import Path

from PIL import Image

from main.__main__ import Servant
from main.config import RES_DIR, ROOT_DIR
from main.detect_cv import detect_avatar
from main.merge_chain import ClassifyChainNode

def main():
    servants = Servant.from_csv()

    DB_PATH = ROOT_DIR / "servants.db"

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conn.execute('DELETE FROM avatar')

    # 排除凑数的从者
    for s in [s for s in servants if not s.get == '无法获得']:
        servant_id = s.id
        servant_res_dir = RES_DIR / "assets" / str(servant_id)
        if servant_res_dir.exists():
            # print(f"Servant {servant_id} res folder exists.")
            # 对资源突破进行分拣
            classify_servant_res = ClassifyChainNode(input_dir=servant_res_dir).output()

            avatar_group = next((i for i in classify_servant_res if i.name == 'avatar'), None)

            if not avatar_group is None:
                # print(f"Servant {servant_id} has avatar res.")
                characters_res = next((j for j in classify_servant_res if j.name == 'character'), None)
                if not characters_res is None:
                    # print(f"Servant {servant_id} has character res.")
                    # 构建根据从者名字 + 数字的正则匹配（如 "吉尔伽美什2.png" 或 "吉尔伽美什_3.png"）
                    name_pattern = '|'.join(
                        re.escape(n) for n in [s.name_cn, s.name_jp, s.name_en] if n
                    )
                    name_num_re = re.compile(rf'(?:{name_pattern})[_\-]?\d') if name_pattern else None
                    # 构建根据 servant_id 补全三位 + 字母 A-D 的正则匹配（如 "008A.png"、"008B.png"）
                    id_letter_re = re.compile(rf'^{servant_id:03d}[A-D]', re.IGNORECASE)

                    # 筛选命名规范的，不规范的只能手动
                    templates = [
                        x['template'] for
                        x in characters_res.layers
                        if ('卡面' in x['template'].name and not '灵衣' in x['template'].name) or
                           (('破' in x['template'].name) and (not '满破' in x['template'].name)) or
                           ('初始' in x['template'].name) or
                           ('再临' in x['template'].name) or
                           ('灵基' in x['template'].name) or
                           ('立绘' in x['template'].name) or
                           ('一' in x['template'].name) or
                           ('二' in x['template'].name) or
                           ('三' in x['template'].name) or
                           ('四' in x['template'].name) or
                           ('Svt' in x['template'].name) or
                           ('Herc' in x['template'].name) or
                           ('从者' in x['template'].name) or
                           ('charagraph' in x['template'].name) or
                           ('charagragh' in x['template'].name) or
                           (name_num_re and name_num_re.search(x['template'].name)) or
                           (id_letter_re.search(x['template'].name))
                    ]

                    if len(templates) == 0:
                        # 兜底：筛选 characters_res.layers 中尺寸为 512x724 的图片
                        fallback = [
                            x['template'] for x in characters_res.layers
                            if Image.open(x['template']).size == (512, 724)
                        ]
                        if fallback:
                            print(f"Servant {servant_id} fallback: found {len(fallback)} images with size 512x724")
                            templates = fallback
                        else:
                            warnings.warn(f"Servant {servant_id} has no match res.")

                    templates = templates[:3] # 不要满破图
                    print(f"{servant_id} match {len(templates)}")
                    if len(templates) == 0:
                        # 需要手动的立绘
                        warnings.warn(f"Servant {servant_id} has no match res.")
                    else:
                        avatar_path: Path = avatar_group.layers[0]['template']
                        mask_path = RES_DIR / "mask" / "mooncell头像探测掩码.jpg"
                        records = []

                        # 逐张立绘匹配，并校验头像位置是否在上半部分
                        for template in templates:
                            results = detect_avatar(target=template, template=avatar_path, mask=mask_path)
                            best = results[0]

                            template_height = Image.open(template).size[1]
                            if best['rect'][1] > template_height / 2:
                                print(f"Servant {servant_id} [{template.name}] avatar in lower half "
                                      f"(top={best['rect'][1]}, height={template_height}), skipping.")
                                continue

                            record = {
                                'servant_id': servant_id,
                                'file': template.name,
                                'left': best['rect'][0],
                                'top': best['rect'][1],
                                'right': best['rect'][2],
                                'bottom': best['rect'][3]
                            }
                            print(f"Servant {servant_id} [{template.name}] match: {record}")
                            records.append(record)

                        if len(records) == 0:
                            warnings.warn(f"Servant {servant_id} has no valid avatar detection result.")
                        else:
                            for r in records:
                                conn.execute(
                                    "INSERT INTO avatar (servant_id, file, left, top, right, bottom) "
                                    "VALUES (?, ?, ?, ?, ?, ?)",
                                    (servant_id, r['file'], r['left'], r['top'], r['right'], r['bottom']),
                                )
                            conn.commit()

        else:
            warnings.warn(f"Servant {servant_id} res folder does not exist.")


    conn.close()

if __name__ == '__main__':
    main()
