import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from chaldea.chain import (
    ChaldeaJsonMerger,
    CharaClassifyChain,
    FaceClassifyChain,
    CharaFacePairer,
    AvatarDetectChain,
)
from chaldea.config import RES_DIR, TEMP_DIR


def _process_servant(servant_dir: Path, merged: dict, assets_dir: Path, min_scale: float, max_scale: float) -> tuple[int, list[dict], bool]:
    servant_id = int(servant_dir.name)

    chara_chain = CharaClassifyChain(servant_dir)
    face_chain = FaceClassifyChain(servant_dir)
    pairer = CharaFacePairer(chara_chain, face_chain, merged)

    detect_chain = AvatarDetectChain(pairer, min_scale=min_scale, max_scale=max_scale)
    detections = detect_chain.output()

    results = []
    hit_max_scale = False
    for result in detections:
        rel_path = result.chara_path.relative_to(assets_dir)
        results.append({
            "servant_id": servant_id,
            "file": str(rel_path),
            "left": result.rect[0],
            "top": result.rect[1],
            "right": result.rect[2],
            "bottom": result.rect[3],
            "scale": result.scale,
        })
        if result.scale >= max_scale:
            hit_max_scale = True

    return servant_id, results, hit_max_scale


YELLOW = "\033[93m"
RESET = "\033[0m"


def auto_detect(
    min_servant_id: int = 1,
    max_servant_id: int | None = None,
    max_workers: int | None = None,
    min_scale: float = 0.85,
    max_scale: float = 3.25,
) -> list[dict]:
    json_files = sorted((RES_DIR / "chaldea").glob("servants.*.json"))
    merger = ChaldeaJsonMerger(json_files)
    merged = merger.merge()

    assets_dir = RES_DIR / "assets"
    servant_dirs = []
    for servant_dir in assets_dir.iterdir():
        if not servant_dir.is_dir():
            continue
        try:
            servant_id = int(servant_dir.name)
        except ValueError:
            continue
        if servant_id < min_servant_id:
            continue
        if max_servant_id is not None and servant_id > max_servant_id:
            continue
        servant_dirs.append((servant_id, servant_dir))

    servant_dirs.sort(key=lambda x: x[0])
    servant_dirs = [d for _, d in servant_dirs]

    print(f"待处理: {len(servant_dirs)} 个从者 (ID {min_servant_id}~{max_servant_id or '末尾'})")
    print(f"线程数: {max_workers or '自动'}, min_scale: {min_scale}, max_scale: {max_scale}")

    all_results = []
    suspicious_servants = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_servant, d, merged, assets_dir, min_scale, max_scale): d
            for d in servant_dirs
        }
        for future in as_completed(futures):
            servant_id, results, hit_max_scale = future.result()
            all_results.extend(results)
            if hit_max_scale:
                suspicious_servants.append(servant_id)
            print(f"  [{servant_id}] 完成: {len(results)} 条检测")

    all_results.sort(key=lambda r: (r["servant_id"], r["file"]))

    scales = [r["scale"] for r in all_results]
    if scales:
        print(f"\n最佳缩放范围统计: min={min(scales):.3f}, max={max(scales):.3f}")

    if suspicious_servants:
        print()
        for sid in sorted(suspicious_servants):
            print(f"{YELLOW}从者{sid}疑似使用了独立的头像图片，请使用网页工具手动标记或移除该立绘的标记{RESET}")

    TEMP_DIR.mkdir(exist_ok=True)
    output_path = TEMP_DIR / "avatar_detect_results.json"
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"总计: {len(all_results)} 条记录已保存到 {output_path}")
    return all_results


if __name__ == "__main__":
    import sys

    min_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.25
    auto_detect(min_id, max_id, workers, scale)
