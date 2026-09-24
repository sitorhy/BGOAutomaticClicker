"""HTTP 服务入口：FastAPI 应用与路由。"""

from pathlib import Path
from urllib.parse import quote
from pydantic import BaseModel
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="avatar-classifier-py-server", version="0.1.0")

# 跨域设置：允许所有来源（注意 "*" 与 allow_credentials=True 不能同时使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# res 根目录：py_server/res（app.py 位于 py_server/src/avatar_classifier_py_server/ 下）
RES_ROOT = (Path(__file__).resolve().parents[2] / "res").resolve()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _safe_res_path(*parts: str) -> Path:
    """拼接 res 下的路径并防止目录穿越（如 ../ 或绝对路径）。"""
    candidate = (RES_ROOT.joinpath(*parts)).resolve()
    if candidate != RES_ROOT and RES_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="非法路径")
    return candidate


@app.get("/")
def hello_world() -> dict:
    return {"message": "Hello World"}


class DetectAvatarRequest(BaseModel):
    mask_pic_ath: str
    template_pic_path: str
    target_pic_path: str

@app.post("/detect/avatar")
def detect_avatar_rect(req: DetectAvatarRequest) -> list[dict]:
    from .detect_cv import detect_avatar
    from PIL import Image

    """
    读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习 模型训练来说暂时用不到，因此使用convert('RGB')进行通道转换。
    """
    # img_mask = Image.open(req.mask_pic_ath).convert("RGB")
    # img_tmpl = Image.open(req.template_pic_path).convert("RGB")
    # img_target = Image.open(req.target_pic_path).convert("RGB")

    # template = np.array(img_tmpl)
    # target = np.array(img_target)
    # mask = np.array(img_mask)

    return detect_avatar(target = req.target_pic_path, template = req.template_pic_path, mask = req.mask_pic_ath)


@app.get("/res/list/{dir:path}")
def res_list(request: Request, dir: str) -> dict:
    """列出 res/<dir> 下的图片文件：文件名、文件路径、http url，并附带子目录便于导航。

    dir 支持多级子目录，如 assets/481。
    """
    target = _safe_res_path(dir)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir}")

    files = []
    subdirs = []
    for entry in sorted(target.iterdir()):
        if entry.is_dir():
            subdirs.append(entry.name)
        elif entry.suffix.lower() in IMAGE_SUFFIXES:
            rel = "/".join(p for p in (dir.strip("/"), entry.name) if p)
            files.append({
                "name": entry.name,
                "path": str(entry),
                "url": f"{request.base_url}res/file/{quote(rel)}",
            })
    return {"dir": dir, "files": files, "subdirs": subdirs}


@app.get("/res/file/{dir:path}/{file}", name="res_file")
def res_file(dir: str, file: str) -> FileResponse:
    """读取 res/<dir>/<file> 下的具体图片文件，dir 支持多级子目录。"""
    target = _safe_res_path(dir, file)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {dir}/{file}")
    return FileResponse(target, media_type=None)
