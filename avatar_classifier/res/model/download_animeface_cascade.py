"""下载 lbpcascade_animeface.xml 动漫人脸检测级联分类器

来源：nagadomi/lbpcascade_animeface
https://github.com/nagadomi/lbpcascade_animeface

用法：
    python download_animeface_cascade.py              # 依次尝试镜像源 → GitHub 源
    python download_animeface_cascade.py --mirror     # 仅使用 HuggingFace 镜像
    python download_animeface_cascade.py --force      # 强制重新下载
"""
import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "lbpcascade_animeface.xml"

# 多个下载源，按优先级依次尝试
GITHUB_RAW = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface"
    "/master/lbpcascade_animeface.xml"
)
GITHUB_BLOB = (
    "https://github.com/nagadomi/lbpcascade_animeface"
    "/raw/master/lbpcascade_animeface.xml"
)
GITEE_MIRROR = (
    "https://gitee.com/mirrors_nagadomi/lbpcascade_animeface"
    "/raw/master/lbpcascade_animeface.xml"
)

SOURCES = [
    ("Gitee 镜像", GITEE_MIRROR),
    ("GitHub raw", GITHUB_RAW),
    ("GitHub blob", GITHUB_BLOB),
]

TIMEOUT = 30  # 秒


def _download(url: str, dest: Path, proxy: str | None = None) -> bool:
    """尝试从 url 下载，成功返回 True，失败返回 False。"""
    print(f"  来源: {url}")

    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    opener = urllib.request.build_opener(*handlers)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {percent:.1f}%  ({downloaded // 1024}/{total_size // 1024} KB)", end="", flush=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            data = bytearray()
            block_num = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                data.extend(chunk)
                block_num += 1
                _progress(block_num, 8192, total)
        print()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"\n  ✗ 下载失败: {e}")
        return False


def download(force: bool = False, proxy: str | None = None, sources: list | None = None) -> Path:
    """下载级联分类器 XML 到当前目录。

    Args:
        force: 为 True 时即使文件已存在也重新下载
        proxy: HTTPS 代理地址，如 "http://127.0.0.1:7890"；None 时自动读取环境变量
        sources: 自定义下载源列表 [(名称, URL), ...]；None 时使用默认源

    Returns:
        模型文件的本地路径

    Raises:
        SystemExit: 所有下载源均失败时退出
    """
    dest = Path(__file__).resolve().parent / MODEL_NAME
    if dest.exists() and not force:
        print(f"文件已存在，跳过下载: {dest}")
        return dest

    if proxy is None:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    src_list = sources or SOURCES
    print(f"正在下载 {MODEL_NAME} ...")
    if proxy:
        print(f"  代理: {proxy}")

    for name, url in src_list:
        print(f"\n[{name}]")
        if _download(url, dest, proxy=proxy):
            size_kb = dest.stat().st_size // 1024
            print(f"✓ 下载完成: {dest}  ({size_kb} KB)")
            return dest

    print("\n所有下载源均失败，请检查网络连接或设置代理（--proxy 或 HTTPS_PROXY 环境变量）")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 lbpcascade_animeface.xml 动漫人脸检测级联分类器")
    parser.add_argument("-f", "--force", action="store_true", help="强制重新下载")
    parser.add_argument("-p", "--proxy", type=str, default=None, help="HTTPS 代理，如 http://127.0.0.1:7890")
    parser.add_argument("-m", "--mirror", action="store_true", help="仅使用 Gitee 镜像源")
    args = parser.parse_args()

    src = None
    if args.mirror:
        src = [("Gitee 镜像", GITEE_MIRROR)]

    download(force=args.force, proxy=args.proxy, sources=src)
