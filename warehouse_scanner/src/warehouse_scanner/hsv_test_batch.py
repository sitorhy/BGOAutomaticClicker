import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsv_test import ROOT, hsv_test

print("文件, 覆盖, 碎片, cols, row_tops, pitch_x, pitch_y")
for p in sorted((ROOT / "res" / "input_chain_01").glob("*.png")):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hsv_test(p)
    out = buf.getvalue()
    m = re.search(
        r"网格: cols=\[([^\]]+)\] row_tops=\[([^\]]+)\] "
        r"间距x=([\d.]+) 间距y=([\d.]+) 覆盖=(\d+)/18 碎片=(\d+)",
        out,
    )
    if m:
        print(
            f"{p.stem}, {m.group(5)}/18, {m.group(6)}, "
            f"{m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}"
        )
    else:
        print(f"{p.stem}, 解析失败")
