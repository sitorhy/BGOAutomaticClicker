"""chaldea 命令行入口：python -m chaldea [dir_name ...]

不带参数时只处理目录 "1"，用于流水线的最小验证；传 all 则处理全部从者目录。
首个参数为 support 时走助战立绘流水线，其后的参数含义与头像流水线一致。
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


if __name__ == "__main__":
    main()
