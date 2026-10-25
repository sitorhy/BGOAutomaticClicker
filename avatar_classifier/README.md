## 徽标 PNG 转图层
生成徽标前景图层命令
```shell
uv run python -m main.png_to_classes res/徽标 --out-dir res/foreground
uv run python -m main.png_to_classes res/冠位徽标 --out-dir res/foreground
```

## 前置工作简述（已执行）
+ 执行 fetch_servant_assets.mjs 和 scrape_fgo_servants.js 下载图片和从者的基本信息
> 注意 scrape_fgo_servants.js 是复制到浏览器中执行的，执行后会弹出下载提示，保存文件为 servants.csv 和 servants.json
> fetch_servant_assets.mjs 会下载图片到 res/assets目录下，图片根据从者的 id 分类存放
> 全部图片爬取下拉接近 1GB，仓库只上存几个作为目录结构参考
+ fetch_servant_info.mjs 爬取的是从者的数值数据，跟训练头像识别库没有关系
+ 新建 servants.db 数据库
+ 运行 fact-detect-server ，项目运行后会队数据库进行初始化，创建表结构
+ 将 servants.json 或 servants.csv 导入 servants.db，可通过 fact-detect-web 项目操作
+ 执行 png_to_classes 脚本生成 徽标 / 冠位徽标 前景图层
+ 执行 auto_detect 自动匹配头像范围，范围会写入 servants.db，有小概率是错的，需要人工校验
+ 运行 fact-detect-server / fact-detect-web 查看/校验/补充头像范围
+ 执行 uv run pyton -m main 生成训练图