import json
import sqlite3
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from pathlib import Path
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split

from main.config import TEMP_DIR, ROOT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DB_PATH = ROOT_DIR / "servants.db"


@dataclass
class Servant:
    """从者实体类"""
    id: int
    name_cn: str
    name_jp: str
    name_en: str
    page_url: str | None = None
    avatar: str | None = None
    card_rarity: str | None = None
    servant_class: str | None = None
    np_color: str | None = None
    np_type: str | None = None
    cards: str | None = None
    faction: str | None = None
    get: str | None = None
    atk_max: int | None = None
    hp_max: int | None = None
    revoke: int = 0

class RecordFile:
    """线程安全的提取记录文件"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()
        self._data: dict = {}

    def load(self, servant_ids: list[int]):
        """加载已有记录，缺失的从者标记为 0"""
        with self._lock:
            if self.file_path.exists():
                self._data = json.loads(self.file_path.read_text())
            for servant_id in servant_ids:
                self._data.setdefault(str(servant_id), 0)
            self._save()

    def get(self, servant_id: int) -> int:
        with self._lock:
            return self._data.get(str(servant_id), 0)

    def set(self, servant_id: int, value: int):
        """更新标记并持久化到磁盘（线程安全）"""
        with self._lock:
            self._data[str(servant_id)] = value
            self._save()

    def _save(self):
        self.file_path.write_text(json.dumps(self._data, indent=4))

class FeatureExtractor:

    def __init__(self):
        # 从 torchvision.models 加载官方已经训练好的 ResNet18 模型
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 去掉最后的全连接层（FC 层）
        # 原始模型输出：1000 类 ImageNet 分类结果
        # 去掉后输出：512 维特征向量
        # list(model.children())[:-1] = 除了最后一层的所有层
        # torch.nn.Sequential 是 PyTorch 中的一个顺序容器（Container）。它用于将多个神经网络层或模块按顺序连接并组合在一起。
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        # 设为评估模式
        # 关闭 Dropout 和 BatchNorm 的训练行为，保证推理稳定
        self.model.eval()
        self.model.to(DEVICE)

        self.transform = transforms.Compose([
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

    # 提取特征（属于模型推理/预测阶段，而不是训练阶段），不需要反向传播
    @torch.no_grad()
    def extract(self, image_path: Path):
        img = Image.open(image_path).convert("RGB")

        # 图片输入流水线转换为 Tensor 张量
        x = self.transform(img)

        # unsqueeze(0)：在最前面增加一个维度（把形状从 [通道, 高, 宽] 变成 [1, 通道, 高, 宽]）。
        # 深度学习模型通常要求输入是一个“批次（Batch）”，哪怕只有一张图，也要伪装成“包含 1 张图片的批次”。
        # .to(DEVICE)：将数据转移到指定的设备上（如 GPU cuda 或 CPU）
        x = x.unsqueeze(0).to(DEVICE)

        # 将处理好的图片数据 x 送入神经网络模型 self.model 中进行前向传播。
        feat = self.model(x)

        # 把前面为了迎合模型而特意添加的、大小为 1 的维度（Batch 维度）去掉，让输出的数据结构变得更扁平、更紧凑
        feat = feat.squeeze()

        # 将最终的特征数据转换为通用的 NumPy 数组并返回
        # （scikit-learn 只认 numpy）
        # 如果模型是在 GPU 上运行的，必须先通过 .cpu() 把数据复制回内存（CPU），然后才能调用 .numpy() 转换
        return feat.cpu().numpy()

    def extract_batch(self, image_paths, show_progress=True):
        features = []

        total = len(image_paths)

        for index, image_path in enumerate(image_paths, 1):
            if show_progress:
                print(f"  [{index:>3}/{total}] {image_path.name}")
            feature = self.extract(image_path)
            features.append(feature)

        return np.array(features)

# output_dir 特征数组的缓存目录
def extract_dir(class_dir: Path, output_dir: Path):
    paths = []
    labels = []
    # x 为特征数据，y为标签，有些命名约定也会将 y 命名为 labels
    # x[i] 必须严格对应 y[i]

    dir_name = class_dir.name

    for img_file in class_dir.iterdir():
        if img_file.is_file():
            labels.append(dir_name)
            paths.append(img_file)

    # 预处理（还没到特征提取阶段） 划分训练集和验证集
    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2, random_state=42)

    print(f"训练集: {len(train_paths)} 张")
    print(f"验证集: {len(test_paths)} 张")

    extractor = FeatureExtractor()

    train_values = extractor.extract_batch(train_paths, show_progress=True)
    test_values = extractor.extract_batch(test_paths, show_progress=True)

    # 缓存特征到磁盘（下次训练可以直接加载，跳过特征提取）
    from main.config import TEMP_DIR
    np.savez(
        output_dir / f"{dir_name}.npz",
        train_values=train_values,  # 训练集特征
        train_labels=train_labels,  # 训练集标签
        test_values=test_values,  # 测试集特征
        test_labels=test_labels,  # 测试集标签
    )
    print(f"\n✓ {dir_name} 特征已缓存")

def prepare_one(servant: Servant, record: RecordFile, images_dir: Path, output_path: Path):
    if servant.revoke == 1:
        print(f"\033[33m跳过 {servant.name_cn}\033[0m")
    else:
        if record.get(servant.id) == 0:
            dir_path = images_dir / f"{servant.id}"
            if dir_path.exists():
                print(f"正在处理 {servant.name_cn}")
                extract_dir(dir_path, output_path)
                record.set(servant.id, 1)
            else:
                warnings.warn(f"不存在 {servant.id} {servant.name_cn}")
        else:
            print(f"\033[33m跳过 已标记提取 {servant.name_cn}\033[0m")

def train(images_dir: Path, output_path: Path, record_file_path: Path):
    if not record_file_path.parent.exists():
        record_file_path.parent.mkdir(parents=True)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("select * from servants").fetchall()
    servants = [Servant(**dict(r)) for r in rows]
    conn.close()

    record = RecordFile(record_file_path)
    record.load([s.id for s in servants])

    with ThreadPoolExecutor(max_workers=20) as pool:
        promises = [
            pool.submit(
                prepare_one,
                servant,
                record,
                images_dir,
                output_path,
            ) for servant in servants[:5]
        ]

        for future in promises:
            try:
                future.result()
            except Exception as e:
                print(f"特征提取异常: {e}")


if __name__ == '__main__':
    IMAGES_DIR = Path("G:/temp")
    OUT_PUT_PATH = Path("G:/features")
    RECORD_FILE_PATH = TEMP_DIR / "train.json"
    train(IMAGES_DIR, OUT_PUT_PATH, RECORD_FILE_PATH)