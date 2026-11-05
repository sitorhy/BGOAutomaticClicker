import argparse
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from mooncell.config import TEMP_DIR, ROOT_DIR
from mooncell.train import Servant, RecordFile, create_label_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DB_PATH = ROOT_DIR / "servants.db"
RANDOM_STATE = 42
BATCH_SIZE = 32  # 训练批次大小，每轮次抽取 N 张图片进行训练
EPOCHS = 10  # 训练 10 轮

LR = 1e-4  # 学习率，超参数之一，简单说就是权重更新的激进程度


class AvatarDataset(Dataset):
    def __init__(self, img_paths: list[Path], labels: list[int], transform: transforms.Compose):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


class FineTuneModel:
    # 这个无法用单例模式了
    def __init__(self, num_classes: int):
        # 从 torchvision.models 加载官方已经训练好的 ResNet18 模型
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 替换分类头, 带分类头的模型输出类别概率，所以不需要外部分类器
        # 主干 → 512维特征 → fc头 → logits → softmax → 类别概率
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 损失函数，简单说就是用来反向传播的
        # 损失函数负责“打分”，反向传播负责“对答案并改错”。
        self.criterion = nn.CrossEntropyLoss()
        # 大致概念：
        # 学习率LR：权重更新的激进程度
        # weight_decay：权重衰减，AdamW优化器的参数，限制权重大小，权重太大微小的特征（噪音）会放大
        # 优化器收的是 model.parameters() —— 没有任何 requires_grad = False 冻结操作
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

        # 设为评估模式
        # 关闭 Dropout 和 BatchNorm 的训练行为，保证推理稳定
        self.model.eval()
        self.model.to(DEVICE)

        # 训练集流水线：带数据增强
        self.transform_train = transforms.Compose([
            # 数据增强，仅训练集使用，测试/验证图必须走 self.transform_eval
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            self._base_transforms(),
        ])

        # 测试集（验证集）流水线：不带数据增强，保证特征提取可复现
        self.transform_eval = transforms.Compose([
            self._base_transforms(),
        ])

    @staticmethod
    def _base_transforms() -> transforms.Compose:
        """两条流水线共用的基础预处理"""
        return transforms.Compose([
            # 图像缩放到 224x224（ResNet18 的标准输入尺寸）
            # 全连接层要求输入的特征图尺寸必须是固定的。为了能够成批次（Batch）处理不同分辨率的原始图片，必须在输入模型前将它们裁剪或缩放到完全相同的尺寸。
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # 数据格式转换，将原本的 PIL 图像或 NumPy 数组
            transforms.ToTensor(),
            # 归一化
            # 在 ImageNet 数据集上训练时，全部都使用了这组均值和标准差进行预处理。
            # 核心作用是将输入图像的像素值转换成符合标准正态分布（均值为 0，方差为 1）的数据。这组特定的数字（0.485, 0.456, 0.406 等）是 ImageNet 数据集上百万张图像的 RGB 三通道的全局均值和标准差。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def train_epoch(self, loader):
        # 将模型设置为训练模式。
        # 因为某些层（如 Dropout 和 BatchNorm）在训练和测试（评估）时的行为是不同的，这一行确保它们以训练模式运行。
        self.model.train()
        # 初始化一个变量，用于累加整个 Epoch 的总损失。
        total_loss = 0.0
        # 初始化一个变量，用于累加整个 Epoch 中预测正确的样本总数。
        correct = 0

        # 从数据加载器（loader）中按批次（Batch）读取数据。x 是输入特征（例如图片张量），y 是对应的真实标签。
        for x, y in loader:
            # 将数据移动到指定的计算设备上（通常是 GPU，即 cuda，或者 CPU）。PyTorch 要求参与计算的模型和数据必须在同一个设备上。
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 清空过往梯度。在默认情况下，PyTorch 的梯度是累加的，如果不清空，上一步的梯度会影响到这一步的参数更新。
            self.optimizer.zero_grad()
            # model(x) 前向传播，将输入数据输入模型，得到预测输出（logits）。
            loss = self.criterion(self.model(x), y)
            # 反向传播，根据损失自动计算模型中每个参数的梯度。
            loss.backward()
            # 参数更新，优化器（如 Adam、SGD）根据刚刚计算出的梯度，按照既定的学习率更新模型的权重。
            self.optimizer.step()
            # loss.item() 获取的是当前 Batch 的平均损失。
            # 乘以 x.size(0)（当前 Batch 的样本数）是为了还原出当前 Batch 的总损失，以便后续计算整个数据集的精确平均损失。
            total_loss += loss.item() * x.size(0)
            # 获取模型预测概率最大的类别索引（即预测结果）。
            # == y：将预测结果与真实标签进行对比，得到一个布尔张量（预测对为 True，错为 False）。
            # .sum().item()：统计当前 Batch中预测正确的样本数量并累加到correct中。
            # 注：这里再次调用了 model(x) 获取预测值。虽然结果准确，但在性能敏感的场景下，通常会把前面计算Loss时得到的模型输出存入变量复用，以避免重复前向传播。
            correct += (self.model(x).argmax(1) == y).sum().item()

        # 循环结束后，计算整个 Epoch 的平均指标并返回：
        # 平均损失：总损失除以数据集的总样本数（len(loader.dataset)）。
        # 准确率：预测正确的总样本数除以数据集的总样本数。
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    # 关闭 PyTorch 的梯度计算（自动求导）。在评估阶段，我们不需要更新模型参数，关闭梯度可以节省显存并加快计算速度。
    @torch.no_grad()
    def evaluate(self, loader):
        # 将模型设置为评估模式。这会关闭训练专用的层（如 Dropout 抑制随机失活、BatchNorm 锁定运行均值与方差），确保测试结果的稳定和正确。
        self.model.eval()
        # 初始化一个变量，用于累加整个验证/测试集中预测正确的样本总数。
        correct = 0
        # 初始化一个字典，用于记录分类别的统计信息。键（Key）是类别标签（例如 0, 1, 2...），值（Value）是一个元组 (cnt, ok)
        #       cnt：该类别的总样本数。
        #       ok：该类别中被预测正确的样本数。
        #
        per_class = {}
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # self.model(x)：前向传播得到模型的输出（Logits）。
            # .argmax(1)：获取每个样本预测概率最大的类别索引作为预测结果（pred）。
            pred = self.model(x).argmax(1)
            # 对比预测结果 pred 和真实标签 y，计算当前 Batch 中预测正确的数量并累加到总数中。
            correct += (pred == y).sum().item()

            # 这几行代码用于精细化统计每一个类别的准确情况（方便后续计算每个类别的准确率，如 Recall / Class-wise Accuracy）：
            # zip 将当前 Batch 的真实标签张量和预测结果张量转换为 Python 列表，并把它们一一对应地打包遍历。
            for label, p in zip(y.tolist(), pred.tolist()):
                # ：从字典中获取当前类别标签的统计数据。如果该类别第一次出现，则默认返回 (0, 0)（代表总数0，正确数0）。
                cnt, ok = per_class.get(label, (0, 0))
                # per_class[label] = (cnt + 1, ok + (p == label))：
                #       cnt + 1：该类别的总样本数加 1。
                #       ok + (p == label)：如果预测值 p 等于真实标签 label（即 p == label 为 True，数值为 1），则正确数加 1；否则加 0。
                per_class[label] = (cnt + 1, ok + (p == label))

        # 获取整个数据集的总样本数。
        total = len(loader.dataset)
        # 返回两个核心指标：
        #   总体准确率：预测正确的总数除以总样本数
        #   per_class 字典：包含每个类别的详细统计数据，便于后续分析模型在哪些类别上表现好、在哪些类别容易混淆（例如计算每个类别的准确率：ok / cnt）。
        return correct / total, per_class

def split_dir_images(dir_path: Path, dir_labels_map: list[Servant]):
    files = [f for f in dir_path.iterdir() if f.suffix in {".png", ".jpg", ".jpeg", ".gif"}]
    random.shuffle(files)
    label_index = next((i for i, l in enumerate(dir_labels_map) if l.dir_name == dir_path.name), -1)
    if label_index == -1:
        raise ValueError(f"从者ID {dir_path.name} 未找到")
    labels = [label_index] * len(files)
    train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=0.2,
                                                                        random_state=RANDOM_STATE)
    return train_files, val_files, train_labels, val_labels


def train(images_dir: Path, output_path: Path, record_file_path: Path):
    if not record_file_path.parent.exists():
        record_file_path.parent.mkdir(parents=True)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    # 小规模测试
    servant_ids = [154, 175, 214, 215, 268, 284, 351, 357, 444]

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if servant_ids is None or len(servant_ids) == 0:
        rows = conn.execute("select * from servants").fetchall()
    else:
        rows = []
        for id in servant_ids:
            rows.append(conn.execute("select * from servants where id = ?", (id,)).fetchone())

    servants = [Servant(**dict(r)) for r in rows]
    servants = [s for s in servants if s.revoke == 0]

    conn.close()

    # train.json 全从者，用于记录哪些子目录已提取特征（以目录名为键）
    record = RecordFile(record_file_path)
    dir_names = [d.name for d in images_dir.iterdir() if d.is_dir()] if images_dir.exists() else []
    if servant_ids is not None and len(servant_ids) > 0:
        dir_names = [d for d in dir_names if any(d.find(str(svt_id)) == 0 for svt_id in servant_ids)]

    # 过滤掉无效的从者的文件夹
    next_dir_names = []
    for dir_name in dir_names:
        if dir_name.find("_") > 0:
            svt_id = int(dir_name.split("_")[0])
        else:
            svt_id = int(dir_name)
        svt = next((s for s in servants if s.id == svt_id), None)
        if svt is not None and svt.revoke == 0:
            next_dir_names.append(dir_name)
    dir_names = next_dir_names

    dir_names = sorted(dir_names)
    record.load(dir_names)

    dir_labels_map: list[Servant] = create_label_map(dir_names, servants, output_path)

    dir_paths = [images_dir / d for d in dir_names]

    train_img_paths = []
    val_img_paths = []
    train_labels = []
    val_labels = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        promises = [pool.submit(split_dir_images, dir_path, dir_labels_map) for dir_path in dir_paths]

    for future in promises:
        try:
            dir_train_files, dir_val_files, dir_train_labels, dir_val_labels = future.result()
            train_img_paths.extend(dir_train_files)
            val_img_paths.extend(dir_val_files)
            train_labels.extend(dir_train_labels)
            val_labels.extend(dir_val_labels)
        except Exception as e:
            print(f"特征提取异常: {e}")

    print(f"训练集图片数: {len(train_img_paths)}")
    print(f"验证集图片数: {len(val_img_paths)}")
    print(f"训练集标签数: {len(train_labels)}")
    print(f"验证集标签数: {len(val_labels)}")

    model = FineTuneModel(num_classes=len(dir_labels_map))

    train_loader = DataLoader(
        AvatarDataset(train_img_paths, train_labels, model.transform_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        AvatarDataset(val_img_paths, val_labels, model.transform_eval),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=DEVICE == "cuda",
    )

    print(f"\n开始微调（device={DEVICE}, epochs={EPOCHS}, batch={BATCH_SIZE}）...")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = model.train_epoch(train_loader)
        val_acc, _ = model.evaluate(val_loader)
        if val_acc > best_acc:
            # 早停
            best_acc = val_acc
            torch.save(model.model.state_dict(), output_path / "finetuned_resnet18.pth")
        print(f"Epoch {epoch:>2}/{EPOCHS}  loss={loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    print(f"\n✓ 最佳验证准确率: {best_acc:.4f}")

    model.model.load_state_dict(torch.load(output_path / "finetuned_resnet18.pth", weights_only=True))
    val_acc, per_class = model.evaluate(val_loader)
    print("\n每类验证准确率:")
    for label_idx in sorted(per_class):
        cnt, ok = per_class[label_idx]
        print(f"  id={label_idx} {servants[label_idx].name_cn}: {ok}/{cnt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="头像分类器特征提取与训练")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="训练集图片根目录（每个从者一个子目录，如 1_1）",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="npz 特征缓存及模型输出目录",
    )
    args = parser.parse_args()

    # Ubuntu 环境
    # IMAGES_DIR = Path("/mnt/DBC9-B465/temp")
    # Windows 环境
    IMAGES_DIR = args.images_dir if args.images_dir is not None else Path("G:\\temp")
    OUT_PUT_PATH = args.output_path if args.output_path is not None else TEMP_DIR / "grp01_finetune"
    RECORD_FILE_PATH = OUT_PUT_PATH / "train.json"
    print(f"图片目录: {IMAGES_DIR}")
    print(f"输出目录: {OUT_PUT_PATH}")
    train(IMAGES_DIR, OUT_PUT_PATH, RECORD_FILE_PATH)
