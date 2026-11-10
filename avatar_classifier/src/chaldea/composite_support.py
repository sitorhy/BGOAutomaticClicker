"""助战立绘合成流水线：按 src/chaldea/助战立绘合成流水线.md 批量产出助战卡面训练素材。

单个从者目录的处理链路为 立绘 -> Frame/Border/Label/Stars/Class 五段叠加，
再以合成出的卡面为底板做礼装合成，目录之间按 dir_name 并行，
最终素材只取礼装合成结果，产物与头像素材同目录混放。
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable

from chaldea.chain import (
    ChainNode,
    NarrowFigureClassifyChain,
    NarrowFigureFrameChain,
    NarrowFigureBorderChain,
    NarrowFigureLabelChain,
    NarrowFigureStarsChain,
    NarrowFigureClassChain,
    NarrowFigureCraftChain,
    PostChain,
)
from chaldea.composite import (
    CompositeRecord,
    Servant,
    DB_PATH,
    YELLOW,
    RESET,
    _group_of,
    _servant_id_of,
)
from chaldea.config import RES_DIR, TEMP_DIR

FG_DIR = RES_DIR / "foreground_support"
CRAFT_MASK = FG_DIR / "礼装掩码.png"

# 立绘 148x375 落在 170x400 画布上的居中偏移
CHAIN_OFFSET = (11, 12)
# 礼装合成每个从者随机挑选的背景张数，即每张卡面的礼装变体数
CRAFT_SIZE = 1

# 背景分路：每个稀有度一组分支，分支名 -> L0 背景图
FRAME_CONFIG: dict[str, dict[str, Path]] = {
    "金卡": {
        "金": FG_DIR / "L0_金.png",
        "冠位": FG_DIR / "L0_冠位.png",
    },
    "银卡": {
        "金": FG_DIR / "L0_金.png",
        "银": FG_DIR / "L0_银.png",
        "冠位": FG_DIR / "L0_冠位.png",
    },
    "铜卡": {
        "铜": FG_DIR / "L0_铜.png",
        "金": FG_DIR / "L0_金.png",
        "银": FG_DIR / "L0_银.png",
        "冠位": FG_DIR / "L0_冠位.png",
    },
    "铁卡": {
        "铁": FG_DIR / "L0_铁.png",
        "冠位": FG_DIR / "L0_冠位.png",
    },
}

BORDER_CONFIG: dict[str, dict[str, list[Path]]] = {
    "金卡": {
        "金": [FG_DIR / "L1_金.png", FG_DIR / "L1_金2.png"],
        "冠位": [FG_DIR / "L1_冠位.png", FG_DIR / "L1_冠位2.png"],
    },
    "银卡": {
        "金": [FG_DIR / "L1_金.png", FG_DIR / "L1_金2.png"],
        "银": [FG_DIR / "L1_银.png", FG_DIR / "L1_银2.png"],
        "冠位": [FG_DIR / "L1_冠位.png", FG_DIR / "L1_冠位2.png"],
    },
    "铜卡": {
        "铜": [FG_DIR / "L1_铜.png", FG_DIR / "L1_铜2.png"],
        "金": [FG_DIR / "L1_金.png", FG_DIR / "L1_金2.png"],
        "银": [FG_DIR / "L1_银.png", FG_DIR / "L1_银2.png"],
        "冠位": [FG_DIR / "L1_冠位.png", FG_DIR / "L1_冠位2.png"],
    },
    "铁卡": {
        "铁": [FG_DIR / "L1_铁.png", FG_DIR / "L1_铁2.png"],
        "冠位": [FG_DIR / "L1_冠位.png", FG_DIR / "L1_冠位2.png"],
    },
}

# 助战序号图层各分路一致，只有一份
LABEL_LIST: list[Path] = [
    FG_DIR / "L2_助战序号.png",
    FG_DIR / "L2_助战序号2.png",
    FG_DIR / "L2_助战序号3.png",
]

STARS_CONFIG: dict[str, dict[str, list[Path]]] = {
    "金卡": {
        "金": [
            FG_DIR / "L3_4星.png", FG_DIR / "L3_4星再临.png",
            FG_DIR / "L3_5星.png", FG_DIR / "L3_5星再临.png",
        ],
        "冠位": [
            FG_DIR / "L3_4星冠位.png", FG_DIR / "L3_4星满级.png",
            FG_DIR / "L3_5星冠位.png", FG_DIR / "L3_5星满级.png",
        ],
    },
    "银卡": {
        "金": [FG_DIR / "L3_3星再临.png"],
        "银": [FG_DIR / "L3_3星.png"],
        "冠位": [FG_DIR / "L3_3星满级.png", FG_DIR / "L3_3星冠位.png"],
    },
    "铜卡": {
        "铜": [FG_DIR / "L3_1星.png", FG_DIR / "L3_2星.png"],
        "金": [FG_DIR / "L3_1星再临.png", FG_DIR / "L3_2星再临.png"],
        "银": [FG_DIR / "L3_1星再临.png", FG_DIR / "L3_2星再临.png"],
        "冠位": [
            FG_DIR / "L3_1星冠位.png", FG_DIR / "L3_1星满级.png",
            FG_DIR / "L3_2星满级.png", FG_DIR / "L3_2星冠位.png",
        ],
    },
    "铁卡": {
        "铁": [FG_DIR / "L3_2星.png", FG_DIR / "L3_2星再临.png"],
        "冠位": [FG_DIR / "L3_2星满级.png", FG_DIR / "L3_2星冠位.png"],
    },
}

# 职阶徽标图层文件名模板，{servant_class} 由从者数据填充
CLASS_CONFIG: dict[str, dict[str, str]] = {
    "金卡": {
        "金": "金卡{}.png",
        "冠位": "冠位{}.png",
    },
    "银卡": {
        "金": "金卡{}.png",
        "银": "银卡{}.png",
        "冠位": "冠位{}.png",
    },
    "铜卡": {
        "铜": "铜卡{}.png",
        "金": "金卡{}.png",
        "银": "银卡{}.png",
        "冠位": "冠位{}.png",
    },
    "铁卡": {
        "铁": "黑卡{}.png",
        "冠位": "冠位{}.png",
    },
}


def _process_work_dir(servant: Servant, work_dir: Path, output_dir: Path) -> list[ChainNode]:
    """处理单个从者目录，返回最终训练素材节点"""
    dir_output = output_dir / servant.dir_name

    narrow_figure_chain = NarrowFigureClassifyChain(work_dir).output()
    group = _group_of(servant.card_rarity)
    class_chains = []
    for branch_name, frame in FRAME_CONFIG[group].items():
        rank_name = f"{group}_{branch_name}"
        frame_chain = NarrowFigureFrameChain(
            name=rank_name,
            frame=frame,
            narrow_figure_chain=narrow_figure_chain,
            output_dir=dir_output,
            offset=CHAIN_OFFSET,
        ).output()
        border_chain = NarrowFigureBorderChain(
            frame_chain=frame_chain,
            border_list=BORDER_CONFIG[group][branch_name],
            output_dir=dir_output,
        ).output()
        label_chain = NarrowFigureLabelChain(
            border_chain=border_chain,
            label_list=LABEL_LIST,
            output_dir=dir_output,
        ).output()
        stars_chain = NarrowFigureStarsChain(
            label_chain=label_chain,
            stars_list=STARS_CONFIG[group][branch_name],
            rank_name=rank_name,
            output_dir=dir_output,
        ).output()
        class_chains.append(NarrowFigureClassChain(
            stars_chain=stars_chain,
            class_list=[FG_DIR / CLASS_CONFIG[group][branch_name].format(servant.servant_class)],
            class_name=rank_name,
            output_dir=dir_output,
        ))

    # 卡面只是礼装合成的底板，不作为最终素材返回
    cards = PostChain(class_chains).output()
    craft_nodes = NarrowFigureCraftChain(
        class_chain=cards,
        crop_mask=CRAFT_MASK,
        size=CRAFT_SIZE,
        output_dir=dir_output,
    ).output()
    return craft_nodes


def composite_support(
    input_dir: Path = RES_DIR / "assets",
    sub_dir_filter: Callable[[str], bool] = lambda dir_name: True,
    output_dir: Path = TEMP_DIR,
    record_file: Path | None = None,
    max_workers: int | None = None,
    db_path: Path | None = None,
) -> list[ChainNode]:
    """遍历 input_dir 子目录执行助战合成流水线，返回全部最终素材节点"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    record_file = Path(record_file) if record_file is not None else TEMP_DIR / "composite_support.json"
    db_path = Path(db_path) if db_path is not None else DB_PATH

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        servants_by_id = {row["id"]: Servant(**dict(row)) for row in conn.execute("SELECT * FROM servants")}
    finally:
        conn.close()

    servants: list[Servant] = []
    for work_dir in input_dir.iterdir():
        if not work_dir.is_dir() or not sub_dir_filter(work_dir.name):
            continue
        servant_id = _servant_id_of(work_dir.name)
        if servant_id is None:
            print(f"{YELLOW}跳过 {work_dir.name}：目录名无法解析从者 id{RESET}")
            continue
        servant = servants_by_id.get(servant_id)
        if servant is None:
            print(f"{YELLOW}跳过 {work_dir.name}：servants 表无 id={servant_id} 的记录{RESET}")
            continue
        if servant.revoke == 1:
            print(f"{YELLOW}跳过 {work_dir.name}：从者已失效 (revoke=1){RESET}")
            continue
        if not (work_dir / "narrowFigure").is_dir():
            print(f"{YELLOW}跳过 {work_dir.name}：无 narrowFigure 立绘{RESET}")
            continue
        servants.append(replace(servant, dir_name=work_dir.name))

    servants.sort(key=lambda s: (s.id, s.dir_name))

    record = CompositeRecord(record_file)
    record.sync([
        {"servant_id": s.id, "dir_name": s.dir_name, "servant_cn": s.name_cn, "completed": 0}
        for s in servants
    ])
    pending = [s for s in servants if not record.is_completed(s.dir_name)]

    print(f"待处理: {len(pending)} 个目录 (跳过已完成 {len(servants) - len(pending)} 个)")
    print(f"线程数: {max_workers or '自动'}, 输出目录: {output_dir}, 记录文件: {record_file}")

    if not pending:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    nodes: list[ChainNode] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_work_dir, servant, input_dir / servant.dir_name, output_dir): servant.dir_name
            for servant in pending
        }
        for future in as_completed(futures):
            dir_name = futures[future]
            produced = future.result()
            record.mark_completed(dir_name)
            nodes.extend(produced)
            print(f"  [{dir_name}] 完成: {len(produced)} 张礼装合成图")

    print(f"\n总计: {len(nodes)} 张礼装合成图已输出到 {output_dir}")
    return nodes
