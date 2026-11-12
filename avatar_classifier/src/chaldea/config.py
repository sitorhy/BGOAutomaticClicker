"""chaldea 的路径配置，一处指向全部资源与产物。

远端机器（autoDL 等）上资源盘和代码盘不在一起，用环境变量覆盖，不设就按代码位置推导，
本地开发的行为与以前完全一致：
    CHALDEA_ROOT_DIR   代码包根目录，servants.db 的默认父目录
    CHALDEA_RES_DIR    资源目录，下面是 assets / background / foreground / foreground_support / mask
    CHALDEA_TEMP_DIR   合成产物与训练输出的落盘目录，全量合成有几十 GB，要指到大盘
    CHALDEA_DB_PATH    从者数据库
    CHALDEA_BG_DIR     随机背景切片来源
"""
import os
from pathlib import Path


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ[env_name]) if os.environ.get(env_name) else default


ROOT_DIR = _path("CHALDEA_ROOT_DIR", Path(__file__).resolve().parents[2])
RES_DIR = _path("CHALDEA_RES_DIR", ROOT_DIR / "res")
TEMP_DIR = _path("CHALDEA_TEMP_DIR", ROOT_DIR / "temp")
DB_PATH = _path("CHALDEA_DB_PATH", ROOT_DIR / "servants.db")
BG_DIR = _path("CHALDEA_BG_DIR", RES_DIR / "background")
