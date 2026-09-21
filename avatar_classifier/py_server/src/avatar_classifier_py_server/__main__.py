"""命令行入口：启动本地 HTTP 服务。"""

import uvicorn


def main() -> None:
    uvicorn.run(
        "avatar_classifier_py_server.app:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
