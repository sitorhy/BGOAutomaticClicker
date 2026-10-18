import sys
from pathlib import Path
from typing import Annotated, Optional

import uvicorn

# 确保同目录下的 database 模块可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 确保项目根目录可被导入，以便引用 src.main.detect_cv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from database import ServantDB
from src.main.detect_cv import detect_anime_face, detect_avatar

RES_DIR = Path(__file__).resolve().parent.parent / "res"

app = FastAPI()
db = ServantDB()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/dir/{path:path}")
async def serve_res(path: str, type: str = 'character'):
    target = (RES_DIR / path).resolve()

    entries = []
    for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
        if not entry.suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
            continue
        img = Image.open(entry)
        if type == 'avatar':
            if img.size[0] == 132 and img.size[1] == 144:
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                })
        else:
            if img.size[0] == 512 and img.size[1] == 724:
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                })
    return JSONResponse(content={"path": path, "entries": entries})


@app.get("/file/{path:path}")
async def serve_file(path: str, request: Request):
    target = (RES_DIR / path).resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")

    return FileResponse(target)


# ── Servants HTTP 接口 ───────────────────────────────────


@app.post("/api/servants/import/json")
async def import_servants_json(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
):
    """从上传的 JSON 文件或请求体中的 JSON 数组批量导入从者数据。"""
    if file:
        text = (await file.read()).decode("utf-8")
    else:
        text = (await request.body()).decode("utf-8")
    try:
        n = db.import_from_json_str(text)
        return {"imported": n, "total": db.count()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/servants/import/csv")
async def import_servants_csv(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
):
    """从上传的 CSV 文件或请求体中的 CSV 文本批量导入从者数据。"""
    """utf-8-sig 是带有 BOM (Byte Order Mark，字节顺序标记) 的 UTF-8 编码变体"""
    if file:
        text = (await file.read()).decode("utf-8-sig")
    else:
        text = (await request.body()).decode("utf-8-sig")
    try:
        n = db.import_from_csv_str(text)
        return {"imported": n, "total": db.count()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/servants")
async def get_all_servants():
    """获取全部从者记录。"""
    return db.get_all()


@app.get("/api/servants/count")
async def get_servants_count():
    """返回从者记录总数。"""
    return {"count": db.count()}


@app.get("/api/servants/{servant_id}")
async def get_servant(servant_id: int):
    """按 id 查询单条从者记录。"""
    row = db.get_by_id(servant_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Servant {servant_id} not found")
    return row


@app.delete("/api/servants/{servant_id}")
async def delete_servant(servant_id: int):
    """按 id 删除单条从者记录。"""
    deleted = db.delete_by_id(servant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Servant {servant_id} not found")
    return {"deleted": 1}


@app.delete("/api/servants")
async def delete_servants(request: Request):
    """批量删除从者记录。请求体格式: {"ids": [1, 2, 3]}。
    若请求体为空则清空全部记录。"""
    try:
        body = await request.json()
    except Exception:
        body = {}

    ids = body.get("ids")
    if ids:
        n = db.delete_by_ids(ids)
        return {"deleted": n}
    else:
        n = db.delete_all()
        return {"deleted": n, "message": "all records deleted"}


# ── Avatar HTTP 接口 ─────────────────────────────────────


@app.post("/api/avatars")
async def add_avatar(request: Request):
    """新增 avatar 记录。请求体: {"servant_id": int, "file": str, "left": int, "top": int, "right": int, "bottom": int}"""
    body = await request.json()
    required = ("servant_id", "file", "left", "top", "right", "bottom")
    for key in required:
        if key not in body:
            raise HTTPException(status_code=400, detail=f"Missing field: {key}")
    avatar_id = db.add_avatar(
        servant_id=body["servant_id"],
        file=body["file"],
        left=body["left"],
        top=body["top"],
        right=body["right"],
        bottom=body["bottom"],
    )
    return {"id": avatar_id}


@app.get("/api/avatars")
async def get_all_avatars():
    """获取全部 avatar 记录。"""
    return db.get_all_avatars()


@app.get("/api/avatars/{avatar_id}")
async def get_avatar(avatar_id: int):
    """按 id 查询单条 avatar 记录。"""
    row = db.get_avatar_by_id(avatar_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")
    return row


@app.get("/api/servants/{servant_id}/avatars")
async def get_avatars_by_servant(servant_id: int):
    """获取某个从者的全部 avatar 记录。"""
    return db.get_avatars_by_servant(servant_id)


@app.delete("/api/avatars/{avatar_id}")
async def delete_avatar(avatar_id: int):
    """按 id 删除单条 avatar 记录。"""
    deleted = db.delete_avatar(avatar_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")
    return {"deleted": 1}


@app.delete("/api/servants/{servant_id}/avatars")
async def delete_avatars_by_servant(servant_id: int):
    """删除某个从者的全部 avatar 记录。"""
    n = db.delete_avatars_by_servant(servant_id)
    return {"deleted": n}


# ── 头像范围探测 ─────────────────────────────────────────


@app.get("/api/detect/avatar-range/{servant_id}")
async def detect_avatar_range(
    servant_id: int,
    illustration: str,
    avatar: Optional[str] = None,
):
    """探测头像在立绘中的位置范围。

    Args:
        servant_id: 从者 ID，用于定位 res/assets/{id} 目录
        illustration: 立绘图片文件名（如 "玛修·基列莱特灵衣15卡面.png"）
        avatar: 头像图片文件名（如 "Servant001.jpg"），可选；不传时使用动漫人脸检测

    Returns:
        最佳匹配结果 dict，匹配失败返回 null
    """
    asset_dir = RES_DIR / "assets" / str(servant_id)
    if not asset_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"从者资源目录不存在: assets/{servant_id}")

    illustration_path = asset_dir / illustration
    if not illustration_path.exists():
        raise HTTPException(status_code=404, detail=f"立绘文件不存在: {illustration}")

    mask_path = RES_DIR / "mask" / "mooncell头像探测掩码.jpg"

    try:
        if avatar:
            avatar_path = asset_dir / avatar
            if not avatar_path.exists():
                raise HTTPException(status_code=404, detail=f"头像文件不存在: {avatar}")
            results = detect_avatar(target=illustration_path, template=avatar_path, mask=mask_path)
        else:
            results = detect_anime_face(image=illustration_path)

        if not results:
            return None
        return results[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
