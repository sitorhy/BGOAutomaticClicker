## 徽标 PNG 转图层
生成徽标前景图层命令
```shell
uv run python -m mooncell.png_to_classes res/徽标 --out-dir res/foreground
uv run python -m mooncell.png_to_classes res/冠位徽标 --out-dir res/foreground
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
+ 执行 uv run pyton -m mooncell 生成训练图

## 疑问和备忘
+ 为什么要将头像范围录入数据库
头像图片可能不是从立绘中截取的，双人卡片的头像可能落差极大，游戏直接使用处理过的独立图片作为头像

## 训练
```shell
# 生成素材：头像卡 + 助战卡（混放在 temp/{dir_name} 下），不带目录参数默认只跑 "1"
uv run python -m chaldea all
uv run python -m chaldea support all
uv run python -m chaldea support 154 175 214 215 268 284 351 357 444

# 全量微调：类别表取自合成记录，一个目录一类，label_map.json 的 id 字段用于识别时归并回从者
uv run python -m chaldea.train_finetune_full
uv run python -m chaldea.train_finetune_full --record temp/composite_support.json temp/composite.json

# 评测：res/test/grp01、grp02 是真实游戏截图（头像与助战卡混合），配 expected.json
uv run python -m mooncell.test_model_finetune --test-input res/test/grp01 --model-root temp/finetune_full
```
+ 训练脚本在 chaldea 侧，因为它的类别表和目录结构都来自 chaldea 的合成流水线；ResNet 模型与数据管线仍复用 mooncell.train_finetune。
+ 默认 `--images-dir temp`、`--record temp/composite_support.json`、`--output-path temp/finetune_full`。
+ `--max-per-dir` / `--max-dirs` 用来控样本规模与冒烟测试，其余超参都在命令行上。
+ 已有 label_map.json 时按它复用类别下标（旧权重继续通用）；要重排类别就加 `--overwrite-label-map`。

## 微调通用步骤
1. 加载在大数据集（ImageNet）上预训练好的权重作为起点，而不是随机初始化。
2. 换掉分类头为 `nn.Linear(in_features, 你的类别数)`。
   ResNet18 原始分类头固定输出 1000 类（ImageNet 类别数），只要你的任务类别数和它不同，就必须替换分类头；
   新头是随机初始化的，需要反向传播学习。替换与否取决于**类别数是否匹配**，与"输出概率像不像"无关。
3. 根据数据量决定微调范围（backbone 冻结多少）：
   - 数据/类别都很少：冻结 backbone，只训练新分类头（或只解冻最深的 layer4），防止过拟合。
   - 数据中等：解冻全部层，用小学习率全量微调。
   - 数据充足、或任务领域与 ImageNet 差异大（如动漫插画）：全量微调收益最大。
   注意：**数据越少越要保守（少解冻），数据越多才适合全量微调**，不是数据少就全量微调。
4. 设置小学习率（通常比从零训练小 10~100 倍，如 1e-4）、数据增强、较少 epoch。
   可对主干和新头使用分层学习率（主干小 lr、分类头大 lr）。
5. 用验证集监控，防过拟合（小数据集几乎必然过拟合，靠增强 + 早停压制）。
6. 保存验证指标最佳的那版权重（最佳模型，而非最后一轮）。
