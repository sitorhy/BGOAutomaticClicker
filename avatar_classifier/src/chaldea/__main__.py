"""chaldea 命令行入口：python -m chaldea [dir_name ...]

不带参数时只处理目录 "1"，用于流水线的最小验证；传 all 则处理全部从者目录。
首个参数为 support 时走助战立绘流水线，其后的参数含义与头像流水线一致。

命令示例（在项目根目录 avatar_classifier 下执行）：
    uv run python -m chaldea              # 只合成 res/assets/1，等价于显式传 1
    uv run python -m chaldea 1            # 只合成 res/assets/1
    uv run python -m chaldea 1 17         # 合成 res/assets/1 与 res/assets/17
    uv run python -m chaldea all          # 合成全部从者目录
    uv run python -m chaldea support 1    # 助战立绘流水线，support 之后的参数含义同上
不用 uv 时改用虚拟环境解释器，参数一致：
    .venv/Scripts/python.exe -m chaldea 1
"""
import sys

from chaldea.composite import composite
from chaldea.composite_support import composite_support


def main():
    args = sys.argv[1:]
    runner = composite
    if args and args[0] == "support":
        runner = composite_support
        args = args[1:]

    dir_names = args or ["1"]
    if dir_names == ["all"]:
        runner()
        return

    wanted = set(dir_names)
    runner(sub_dir_filter=lambda dir_name: dir_name in wanted)

# 执行目录 1 的合成：uv run python -m chaldea 1（或 .venv/Scripts/python.exe -m chaldea 1）
if __name__ == "__main__":
    main()
