# avatar-classifier-py-server

基于 FastAPI 的基础 HTTP 服务框架，包管理使用 [uv](https://docs.astral.sh/uv/)。

## 结构

- `src/avatar_classifier_py_server/app.py` — FastAPI 应用与路由（目前仅 `GET /` 返回 Hello World）
- `src/avatar_classifier_py_server/__main__.py` — 启动入口（uvicorn，监听 `0.0.0.0:8000`）

## 运行

```bash
uv sync                # 安装依赖
uv run avatar-classifier-py-server   # 启动服务
# 或开发模式（热重载）
uv run uvicorn avatar_classifier_py_server.app:app --reload
```

```bash
curl http://127.0.0.1:8000/
# {"message":"Hello World"}
```

交互式文档：http://127.0.0.1:8000/docs
