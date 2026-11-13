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

## (AI)助战卡的坑：礼装画面会带偏识别
真实助战卡整卡喂进模型时，左下角的概念礼装美术会主导预测：grp03 三张误判全部指向同一个从者
（257 巴沙洛缪·罗伯茨），而同一从者的头像裁剪仍然 0.98+ 正确；`215_CBA助战` 甚至让 215 掉出 top-3。

+ 礼装数量比从者多几个数量级，不可能纳入合成池当噪声，所以模型从没见过真实礼装美术。
+ 合成链的礼装位填的是随机背景切片（`res/mask/助战礼装位置掩码.png` 在 170x400 画布上圈出的区域），
  推理时想事后遮盖也不精准：掩码按画布线性映射到截图，边界对不齐，实测只能救回 3 张里的 2 张。
+ 可行的做法是不给模型看下半部分：立绘约占卡面高度 60%，下面 40% 是礼装画面和等级/生命/攻击/消耗值，
  对身份识别没有信息量。**截图与识别输入按只取上半部分规范执行**，需要整卡时再单独裁。
+ res/test/grp03 的 12 张整卡测试图已就地按 `round(h*0.6)` 裁掉下部（原始图在 git 历史里）；
  裁前 14/17 = 82.35% @0.8457，裁后 17/17 = 100% @0.9973。
+ 名字带"助战"但高度明显偏小（如 230x362）的本来就是上半部分，不要再裁。
+ 训练侧暂未同步裁剪，合成卡仍是完整 170x400，实测无碍：礼装位在训练里就是随机噪声，裁掉不亏。
  若以后构图不一致暴露出问题，再考虑把助战链产物也只保留上半部分。
+ 补一句时效：上面那组 82.35% 是 30 类模型（助战合成只有 9 个 id）的实测。换成 471 类 + 全量助战合成的
  `release/全量v2` 后，没裁下部的整卡也能过（`284_助战` 173x427 → 0.9963）。取上半部仍是默认规范，
  但已经不至于像当初那样"整卡必翻车"。

## (AI)截图规范：置信度反常先查输入，别急着怪模型
识别**对了但置信度很低**，第一反应应该是这张截图本身是不是残缺，而不是模型分不清相似从者。
实测案例 `284_C呆2.png`：只截到半张脸时（143x126）全量v2 给 0.5704、全量v1 给 0.3293，次选都是
386 阿尔托莉雅·卡斯特（狂阶，与 284 同名不同职阶）；把同一张重截成完整头部（206x162）后
v2 → 0.9992、v1 → 0.9793，次选消失。两个模型同时跳升，说明瓶颈在输入不在权重。

+ **头部要完整**：整张脸都要在框内，宁可多带一点肩颈，不要贴着眉眼裁。头像类素材尤其吃这个亏。
+ **助战卡只取上半部分**：见上一节，下半部的礼装美术和数值标注没有信息量。
+ 同名不同职阶（卡米拉 46/263、阿尔托莉雅·卡斯特 284/386、玉藻前 62/128 这类，数据库里共 74 组）
  靠职阶徽章 + 立绘差异区分，徽章在卡面左上角，取上半部不会把它裁掉；但截图残缺时最容易让这类
  次选冒头。真要收紧阈值，应该按"低置信度 → 先复核截图 → 再考虑重训"的顺序处理。
+ 全量v2 在 grp01/02/03/04 上 72/72 全对，四组合并的置信度下限 0.9924；回归明细见 `release/全量v2/README.md`。

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
   本项目不照此执行：合成样本高度重复，验证集又小又很快饱和到 1.0000，`val_acc > best_acc` 只会存下
   最早饱和的那一轮，反而丢掉后续收敛的权重；`train_finetune_full` 改为训完直接存末轮。
