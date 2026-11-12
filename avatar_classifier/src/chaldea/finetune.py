"""chaldea 自用的微调模型与数据管线：只依赖 torch / torchvision / Pillow。

之前从 mooncell.train_finetune 复用这三个符号，但那条 import 链会连带拉起 mooncell.train，
把 xgboost、scikit-learn 一起拖进来；远端机器跑全量合成+训练用不到它们，
打包时只想装四个依赖，所以这里留一份最小实现。
mooncell 侧的原脚本继续给自己的头像流水线用，两份代码各自演化。
"""
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
LR = 1e-4
WEIGHT_DECAY = 1e-4

# ImageNet 全局均值/标准差，主干权重就是按这组归一化训出来的，换掉等于喂错分布
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AvatarDataset(Dataset):
    def __init__(self, img_paths: list[Path], labels: list[int], transform: transforms.Compose):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # convert("RGB") 会把 PNG 的透明通道直接丢弃，不是与白底/黑底合成
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class FineTuneModel:
    """ResNet18 主干 + 换掉的线性分类头，AdamW 全量微调（不冻结任何层）。"""

    def __init__(self, num_classes: int, lr: float = LR, weight_decay: float = WEIGHT_DECAY):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.eval()
        self.model.to(DEVICE)

        base = transforms.Compose([
            # 直接拉伸到 224x224：全连接层要求批内尺寸一致，助战卡的竖构图也靠它保留下来
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_eval = base
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            base,
        ])

    def train_epoch(self, loader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            # 再前向一次只为统计 train_acc：与 mooncell 侧原实现逐步对齐，
            # 复用上面那份 logits 会少更新一次 BatchNorm 滑动统计，两边结果就没法比了
            correct += (self.model(x).argmax(1) == y).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, dict[int, tuple[int, int]]]:
        self.model.eval()
        correct = 0
        # label -> (该类别样本数, 预测正确数)
        per_class: dict[int, tuple[int, int]] = {}
        for x, y in loader:
            x = x.to(DEVICE)
            pred = self.model(x).argmax(1)
            correct += (pred == y.to(DEVICE)).sum().item()
            for label, p in zip(y.tolist(), pred.tolist()):
                cnt, ok = per_class.get(label, (0, 0))
                per_class[label] = (cnt + 1, ok + (p == label))
        return correct / len(loader.dataset), per_class
