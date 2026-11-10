"""合成流水线主体：按 src/chaldea/合成流水线.md 批量产出训练素材。

单个从者目录的处理链路为 立绘 -> ExpandClipChain 裁剪 -> Frame/Label/Status/
Stars/Class 五段叠加 -> PostChain 汇总 -> FaceMixChain 背景混合 -> CopyChain 头像复制，
目录之间按 dir_name 并行。
"""
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from PIL import Image

from chaldea.chain import (
    ChainNode,
    CharaClassifyChain,
    FaceClassifyChain,
    ExpandClipChain,
    FrameChain,
    LabelChain,
    StatusChain,
    StarsChain,
    ClassChain,
    PostChain,
    StatusClassifyChain,
    BackgroundClipChain,
    FaceMixChain,
    CopyChain,
)
from chaldea.config import BG_DIR, RES_DIR, ROOT_DIR, TEMP_DIR

DB_PATH = ROOT_DIR / "servants.db"
FG_DIR = RES_DIR / "foreground"
MASK_IMG = RES_DIR / "mask" / "BGO头像裁剪掩码.png"

MIN_SCALE = 0.85
MAX_SCALE = 1.5
STEPS = 3

# 状态图层固定按满破产出
STATUS_LIST = [FG_DIR / "L1_满破标.png"]

# 边框分路：每个稀有度一组分支，分支顺序即分路顺序
FRAME_CONFIG: dict[str, list[dict]] = {
    "金卡": [
        {"name": "金", "frame": FG_DIR / "L0_金框.png"},
        {"name": "冠位", "frame": FG_DIR / "L0_冠位框.png"},
        {"name": "满级冠位", "frame": FG_DIR / "L0_满级冠位框.png"},
    ],
    "银卡": [
        {"name": "金", "frame": FG_DIR / "L0_金框.png"},
        {"name": "银", "frame": FG_DIR / "L0_银框.png"},
        {"name": "冠位", "frame": FG_DIR / "L0_冠位框.png"},
        {"name": "满级冠位", "frame": FG_DIR / "L0_满级冠位框.png"},
    ],
    "铜卡": [
        {"name": "金", "frame": FG_DIR / "L0_金框.png"},
        {"name": "银", "frame": FG_DIR / "L0_银框.png"},
        {"name": "铜", "frame": FG_DIR / "L0_铜框.png"},
        {"name": "冠位", "frame": FG_DIR / "L0_冠位框.png"},
        {"name": "满级冠位", "frame": FG_DIR / "L0_满级冠位框.png"},
    ],
    "铁卡": [
        {"name": "铁", "frame": FG_DIR / "L0_铁框.png"},
        {"name": "冠位", "frame": FG_DIR / "L0_冠位框.png"},
        {"name": "满级冠位", "frame": FG_DIR / "L0_满级冠位框.png"},
    ],
}

LABEL_CONFIG: dict[str, dict[str, list[Path]]] = {
    "金卡": {
        "金": [FG_DIR / "L1_金标.png", FG_DIR / "L1_满级金标.png"],
        "冠位": [FG_DIR / "L1_冠位标.png"],
        "满级冠位": [FG_DIR / "L1_满级冠位标.png"],
    },
    "银卡": {
        "金": [FG_DIR / "L1_金标.png", FG_DIR / "L1_满级金标.png"],
        "银": [FG_DIR / "L1_银标.png"],
        "冠位": [FG_DIR / "L1_冠位标.png"],
        "满级冠位": [FG_DIR / "L1_满级冠位标.png"],
    },
    "铜卡": {
        "金": [FG_DIR / "L1_金标.png", FG_DIR / "L1_满级金标.png"],
        "银": [FG_DIR / "L1_银标.png"],
        "铜": [FG_DIR / "L1_铜标.png"],
        "冠位": [FG_DIR / "L1_冠位标.png"],
        "满级冠位": [FG_DIR / "L1_满级冠位标.png"],
    },
    "铁卡": {
        "铁": [FG_DIR / "L1_铁标.png"],
        "冠位": [FG_DIR / "L1_冠位标.png"],
        "满级冠位": [FG_DIR / "L1_满级冠位标.png"],
    },
}

STARS_CONFIG: dict[str, dict[str, list[Path]]] = {
    "金卡": {
        "金": [FG_DIR / "L1_4星.png", FG_DIR / "L1_4星再临.png", FG_DIR / "L1_5星.png", FG_DIR / "L1_5星再临.png"],
        "冠位": [FG_DIR / "L1_4星满级.png", FG_DIR / "L1_5星满级.png"],
        "满级冠位": [FG_DIR / "L1_4星冠位.png", FG_DIR / "L1_5星冠位.png"],
    },
    "银卡": {
        "金": [FG_DIR / "L1_3星再临.png"],
        "银": [FG_DIR / "L1_3星.png"],
        "冠位": [FG_DIR / "L1_3星冠位.png"],
        "满级冠位": [FG_DIR / "L1_3星满级.png"],
    },
    "铜卡": {
        "金": [FG_DIR / "L1_1星再临.png", FG_DIR / "L1_2星再临.png"],
        "银": [FG_DIR / "L1_1星再临.png", FG_DIR / "L1_2星再临.png"],
        "铜": [FG_DIR / "L1_1星.png", FG_DIR / "L1_2星.png"],
        "冠位": [FG_DIR / "L1_1星冠位.png", FG_DIR / "L1_2星冠位.png"],
        "满级冠位": [FG_DIR / "L1_1星满级.png", FG_DIR / "L1_2星满级.png"],
    },
    "铁卡": {
        "铁": [FG_DIR / "L1_2星.png", FG_DIR / "L1_2星再临.png"],
        "冠位": [FG_DIR / "L1_1星冠位.png", FG_DIR / "L1_2星冠位.png"],
        "满级冠位": [FG_DIR / "L1_1星满级.png", FG_DIR / "L1_2星满级.png"],
    },
}

# 职阶图层文件名模板，{servant_class} 由从者数据填充
CLASS_CONFIG: dict[str, dict[str, str]] = {
    "金卡": {
        "金": "金卡{}.png",
        "冠位": "冠位{}.png",
        "满级冠位": "冠位{}.png",
    },
    "银卡": {
        "金": "金卡{}.png",
        "银": "银卡{}.png",
        "冠位": "冠位{}.png",
        "满级冠位": "冠位{}.png",
    },
    "铜卡": {
        "金": "金卡{}.png",
        "银": "银卡{}.png",
        "铜": "铜卡{}.png",
        "冠位": "冠位{}.png",
        "满级冠位": "冠位{}.png",
    },
    "铁卡": {
        "铁": "黑卡{}.png",
        "冠位": "冠位{}.png",
        "满级冠位": "冠位{}.png",
    },
}

YELLOW = "\033[93m"
RESET = "\033[0m"


@dataclass
class Servant:
    """从者实体，字段与 servants 表列一一对应，dir_name 为工作区目录名"""
    id: int
    name_cn: str
    name_jp: str
    name_en: str
    page_url: str | None = None
    avatar: str | None = None
    card_rarity: str | None = None
    servant_class: str | None = None
    np_color: str | None = None
    np_type: str | None = None
    cards: str | None = None
    faction: str | None = None
    get: str | None = None
    atk_max: int | None = None
    hp_max: int | None = None
    revoke: int = 0
    dir_name: str = ""


class CompositeRecord:
    """线程安全的合成进度记录，文件不存在时自动创建。

    记录格式为 [{servant_id, dir_name, servant_cn, completed}]，
    以 dir_name 作为处理与否的键，completed=1 的目录下次直接跳过。
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._lock = threading.Lock()
        self._rows: list[dict] = []

    def sync(self, entries: list[dict]):
        """合并本次待处理目录后落盘，已有记录的 completed 状态保持不变"""
        with self._lock:
            if self.file_path.exists():
                self._rows = json.loads(self.file_path.read_text(encoding="utf-8"))
            for entry in entries:
                if self._find(entry["dir_name"]) is None:
                    self._rows.append(dict(entry))
            self._save()

    def is_completed(self, dir_name: str) -> bool:
        with self._lock:
            row = self._find(dir_name)
            return row is not None and row.get("completed") == 1

    def mark_completed(self, dir_name: str):
        with self._lock:
            row = self._find(dir_name)
            if row is None:
                return
            row["completed"] = 1
            self._save()

    def _find(self, dir_name: str) -> dict | None:
        for row in self._rows:
            if row.get("dir_name") == dir_name:
                return row
        return None

    def _save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(self._rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _servant_id_of(dir_name: str) -> int | None:
    """从目录名解析从者 id，分组形式取 _ 前的数字段"""
    try:
        return int(dir_name.split("_")[0])
    except ValueError:
        return None


def _group_of(card_rarity: str | None) -> str:
    """稀有度归一化为配置键，空串或未配置的值按铁卡处理"""
    return card_rarity if card_rarity in FRAME_CONFIG else "铁卡"


def _mask_ratio(mask_img: Path) -> float:
    with Image.open(mask_img) as img:
        return img.size[0] / img.size[1]


def _load_avatars(conn: sqlite3.Connection) -> dict[int, dict[str, tuple[int, int, int, int]]]:
    avatars: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    for row in conn.execute("SELECT servant_id, file, left, top, right, bottom FROM avatar"):
        avatars.setdefault(int(row["servant_id"]), {})[row["file"]] = (
            row["left"], row["top"], row["right"], row["bottom"])
    return avatars


def _process_work_dir(
    servant: Servant,
    work_dir: Path,
    avatars: dict[str, tuple[int, int, int, int]],
    output_dir: Path,
    ratio: float,
) -> list[ChainNode]:
    """处理单个从者目录，返回最终训练素材节点"""
    dir_output = output_dir / servant.dir_name

    expand_chain: list[ChainNode] = []
    for node in CharaClassifyChain(work_dir).output():
        rel_file = node.image_path.relative_to(work_dir).as_posix()
        rect = avatars.get(rel_file)
        if rect is None:
            raise ValueError(f"avatar 表缺少裁剪区域: {servant.dir_name}/{rel_file}")
        expand_chain.extend(ExpandClipChain(
            target_img=node.image_path,
            rect=rect,
            ratio=ratio,
            min_scale=MIN_SCALE,
            max_scale=MAX_SCALE,
            steps=STEPS,
            output_dir=dir_output,
        ).output())

    group = _group_of(servant.card_rarity)
    class_chains = []
    for branch in FRAME_CONFIG[group]:
        branch_name = f"{group}_{branch['name']}"
        frame_chain = FrameChain(
            name=branch_name,
            frame=branch["frame"],
            expand_chain=expand_chain,
            output_dir=dir_output,
            mask=MASK_IMG,
        )
        label_chain = LabelChain(
            frame_chain=frame_chain.output(),
            label_list=LABEL_CONFIG[group][branch["name"]],
            output_dir=dir_output,
        )
        status_chain = StatusChain(
            label_chain=label_chain.output(),
            status_list=STATUS_LIST,
            output_dir=dir_output,
        )
        stars_chain = StarsChain(
            status_chain=status_chain.output(),
            output_dir=dir_output,
            stars_list=STARS_CONFIG[group][branch["name"]],
            rank_name=branch_name,
        )
        class_chains.append(ClassChain(
            stars_chain=stars_chain.output(),
            class_list=[FG_DIR / CLASS_CONFIG[group][branch["name"]].format(servant.servant_class)],
            class_name=branch_name,
            output_dir=dir_output,
        ))

    nodes = PostChain(class_chains).output()

    status_nodes = StatusClassifyChain(work_dir).output()
    if status_nodes:
        nodes.extend(FaceMixChain(
            status_nodes=status_nodes,
            bg_nodes=BackgroundClipChain(BG_DIR).output(),
            output_dir=dir_output,
        ).output())
    else:
        print(f"{YELLOW}[{servant.dir_name}] 无 status 图片，跳过背景混合{RESET}")

    # 头像原图与合成图同目录存放，供数据集按 file 直接取用
    nodes.extend(CopyChain(FaceClassifyChain(work_dir).output(), output_dir=dir_output).output())
    return nodes


def composite(
    input_dir: Path = RES_DIR / "assets",
    sub_dir_filter: Callable[[str], bool] = lambda dir_name: True,
    output_dir: Path = TEMP_DIR,
    record_file: Path | None = None,
    max_workers: int | None = None,
    db_path: Path | None = None,
) -> list[ChainNode]:
    """遍历 input_dir 子目录执行合成流水线，返回全部最终素材节点"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    record_file = Path(record_file) if record_file is not None else TEMP_DIR / "composite.json"
    db_path = Path(db_path) if db_path is not None else DB_PATH

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        servants_by_id = {row["id"]: Servant(**dict(row)) for row in conn.execute("SELECT * FROM servants")}
        avatars_by_servant = _load_avatars(conn)
    finally:
        conn.close()

    # 目录名 -> 带 dir_name 的从者副本，同一从者的多个分组目录互不干扰
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

    ratio = _mask_ratio(MASK_IMG)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes: list[ChainNode] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_work_dir, servant, input_dir / servant.dir_name,
                avatars_by_servant.get(servant.id, {}), output_dir, ratio,
            ): servant.dir_name
            for servant in pending
        }
        for future in as_completed(futures):
            dir_name = futures[future]
            produced = future.result()
            record.mark_completed(dir_name)
            nodes.extend(produced)
            print(f"  [{dir_name}] 完成: {len(produced)} 张合成图")

    print(f"\n总计: {len(nodes)} 张合成图已输出到 {output_dir}")
    return nodes
