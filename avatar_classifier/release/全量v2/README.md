# 全量训练v2版本
+ 轮数 20
+ 整合了助战和头像的合成图训练

# 日志摘要
```log
总计: 93804 张礼装合成图已输出到 /root/autodl-tmp/temp
图片目录: /root/autodl-tmp/temp
输出目录: /root/autodl-tmp/temp/finetune_full
记录文件: [PosixPath('/root/autodl-tmp/temp/composite.json'), PosixPath('/root/autodl-tmp/temp/composite_support.json')]
✓ 已读取记录 /root/autodl-tmp/temp/composite.json（471 行）
✓ 已读取记录 /root/autodl-tmp/temp/composite_support.json（471 行）
候选类别: 471 个目录
✓ 标签映射表已保存到 /root/autodl-tmp/temp/finetune_full/label_map.json（471 类）
分类头类别数: 471（归并到 471 个从者 id），本轮有样本 471 类
单类样本数: 最少 814, 最多 4422, 合计 658698
训练集: 527022 张 / 验证集: 131676 张

开始微调（device=cuda, epochs=20, batch=32, lr=0.0001）...
Epoch   1/20  loss=0.1992  train_acc=0.9825  val_acc=0.9883  用时 688s
Epoch   2/20  loss=0.0085  train_acc=1.0000  val_acc=0.9788  用时 1379s
Epoch   3/20  loss=0.0053  train_acc=1.0000  val_acc=0.9711  用时 2069s
Epoch   4/20  loss=0.0045  train_acc=1.0000  val_acc=0.9708  用时 2755s
Epoch   5/20  loss=0.0035  train_acc=1.0000  val_acc=0.9764  用时 3446s
Epoch   6/20  loss=0.0031  train_acc=1.0000  val_acc=0.9648  用时 4133s
Epoch   7/20  loss=0.0030  train_acc=1.0000  val_acc=0.9681  用时 4819s
Epoch   8/20  loss=0.0026  train_acc=1.0000  val_acc=0.9592  用时 5506s
Epoch   9/20  loss=0.0021  train_acc=1.0000  val_acc=0.9703  用时 6195s
Epoch  10/20  loss=0.0022  train_acc=1.0000  val_acc=0.9544  用时 6881s
Epoch  11/20  loss=0.0017  train_acc=1.0000  val_acc=0.9735  用时 7577s
Epoch  12/20  loss=0.0019  train_acc=1.0000  val_acc=0.9637  用时 8266s
Epoch  13/20  loss=0.0016  train_acc=1.0000  val_acc=0.9584  用时 8963s
Epoch  14/20  loss=0.0015  train_acc=1.0000  val_acc=0.9425  用时 9650s
Epoch  15/20  loss=0.0014  train_acc=1.0000  val_acc=0.9398  用时 10344s
Epoch  16/20  loss=0.0014  train_acc=1.0000  val_acc=0.9349  用时 11033s
Epoch  17/20  loss=0.0012  train_acc=1.0000  val_acc=0.9620  用时 11725s
Epoch  18/20  loss=0.0013  train_acc=1.0000  val_acc=0.9515  用时 12414s
Epoch  19/20  loss=0.0011  train_acc=1.0000  val_acc=0.9657  用时 13103s
Epoch  20/20  loss=0.0012  train_acc=1.0000  val_acc=0.9753  用时 13794s

✓ 最后一轮权重已存到 /root/autodl-tmp/temp/finetune_full/finetuned_resnet18.pth
验证准确率: 0.9753
验证准确率最低的 20 类:
  label= 289 dir=296        id=296   尼莫: 231/334 = 0.692
  label=  14 dir=15         id=15    尤瑞艾莉: 154/220 = 0.700
  label= 393 dir=401        id=401   尼莫〔圣诞〕: 186/265 = 0.702
  label=  61 dir=62         id=62    玉藻前: 194/265 = 0.732
  label= 116 dir=118        id=118   奥斯曼狄斯: 198/265 = 0.747
  label= 424 dir=434        id=434   黑姬: 211/265 = 0.796
  label= 265 dir=272        id=272   超人俄里翁: 212/265 = 0.800
  label= 107 dir=109        id=109   卫宫〔Assassin〕: 216/265 = 0.815
  label= 151 dir=156        id=156   新宿的Archer: 274/334 = 0.820
  label=  39 dir=40         id=40    咒腕哈桑: 262/316 = 0.829
  label= 453 dir=466        id=466   雅克·德·莫莱: 220/265 = 0.830
  label= 290 dir=297        id=297   芦屋道满: 283/334 = 0.847
  label= 387 dir=395        id=395   杉谷善住坊: 149/174 = 0.856
  label=  33 dir=34         id=34    威廉·莎士比亚: 271/316 = 0.858
  label= 256 dir=263        id=263   卡米拉: 228/265 = 0.860
  label=  41 dir=42         id=42    荆轲: 150/174 = 0.862
  label= 232 dir=238        id=238   Kingprotea: 349/403 = 0.866
  label= 399 dir=407        id=407   玛丽·安托瓦内特〔Alter〕: 230/265 = 0.868
  label= 432 dir=444        id=444   U－奥尔加玛丽: 291/334 = 0.871
  label= 269 dir=276        id=276   清少纳言: 231/265 = 0.872
权重与标签表: /root/autodl-tmp/temp/finetune_full
```