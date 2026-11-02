import json
import sqlite3
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import joblib
import numpy as np
import torch

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from torchvision import models, transforms
from PIL import Image
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from main.config import TEMP_DIR, ROOT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DB_PATH = ROOT_DIR / "servants.db"
RANDOM_STATE = 42


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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name_cn": self.name_cn,
            "name_jp": self.name_jp,
            "name_en": self.name_en,
            "page_url": self.page_url,
            "avatar": self.avatar,
            "card_rarity": self.card_rarity,
            "servant_class": self.servant_class,
            "np_color": self.np_color,
            "np_type": self.np_type,
            "cards": self.cards,
            "faction": self.faction,
            "get": self.get,
            "atk_max": self.atk_max,
            "hp_max": self.hp_max,
            "revoke": self.revoke,
        }

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

    # 单例：显存里只保留一份模型权重，避免多线程各自实例化导致爆显存
    _instance: "FeatureExtractor | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "FeatureExtractor":
        """获取全局唯一的 FeatureExtractor 实例（线程安全，双重检查锁定）"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = FeatureExtractor()
        return cls._instance

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
def extract_dir(class_dir: Path, output_dir: Path, npz_labels_map: list[Servant]):
    paths = []
    labels = []
    # x 为特征数据，y为标签，有些命名约定也会将 y 命名为 labels
    # x[i] 必须严格对应 y[i]

    dir_name = class_dir.name
    # 根据 dir_name (从者ID) 在 npz_labels_map 中查找索引
    servant_id = int(dir_name)
    label_idx = next((i for i, s in enumerate(npz_labels_map) if s.id == servant_id), -1)
    if label_idx == -1:
        raise ValueError(f"从者ID {dir_name} 未找到")

    for img_file in class_dir.iterdir():
        if img_file.is_file():
            labels.append(label_idx)  # 使用索引作为标签
            paths.append(img_file)

    # 预处理（还没到特征提取阶段） 划分训练集和验证集
    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2, random_state=RANDOM_STATE)

    print(f"训练集: {len(train_paths)} 张")
    print(f"验证集: {len(test_paths)} 张")

    # 共享单例，不再每次新建模型实例
    extractor = FeatureExtractor.get()

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


def prepare_one(servant: Servant, record: RecordFile, images_dir: Path, output_path: Path, npz_labels_map: list[Servant]):
    if servant.revoke == 1:
        raise ValueError(f"从者ID {servant.id} 已标记为排除")
    else:
        if record.get(servant.id) == 0:
            dir_path = images_dir / f"{servant.id}"
            if dir_path.exists():
                print(f"正在处理 {servant.name_cn}")
                extract_dir(dir_path, output_path, npz_labels_map)
                record.set(servant.id, 1)
            else:
                warnings.warn(f"不存在 {servant.id} {servant.name_cn}")
        else:
            print(f"\033[33m跳过 已标记提取 {servant.name_cn}\033[0m")


def load_cached_features(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """解析缓存的 npz 特征文件，合并为训练集和验证集（内存优化版）

    Returns:
        (train_X, train_y, test_X, test_y)
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"缓存目录不存在: {output_dir}")

    npz_files = sorted(output_dir.glob("*.npz"))
    if not npz_files:
        all_files = list(output_dir.iterdir())
        raise FileNotFoundError(
            f"在 {output_dir} 下未找到任何 .npz 缓存文件"
            f"\n  目录内共有 {len(all_files)} 个文件/目录:"
            f"\n  {[f.name for f in all_files[:10]]}{'...' if len(all_files) > 10 else ''}"
        )

    # 第一遍：统计总样本数和特征维度，预分配内存
    total_train = 0
    total_test = 0
    feature_dim = None
    all_labels = set()

    print("正在扫描缓存文件...")
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=False)
        n_train = len(data["train_values"])
        n_test = len(data["test_values"])
        total_train += n_train
        total_test += n_test
        if feature_dim is None:
            feature_dim = data["train_values"].shape[1]
        all_labels.update(data["train_labels"].tolist())
        all_labels.update(data["test_labels"].tolist())
        data.close()

    print(f"✓ 扫描完成: {len(npz_files)} 个文件，训练集 {total_train} 条，验证集 {total_test} 条")
    print(f"  特征维度: {feature_dim}，类别数: {len(all_labels)}")

    # 预分配数组（避免多次 concatenate 导致的内存复制）
    train_X = np.empty((total_train, feature_dim), dtype=np.float32)
    train_y = np.empty(total_train, dtype=np.int32)  # 标签为整数索引
    test_X = np.empty((total_test, feature_dim), dtype=np.float32)
    test_y = np.empty(total_test, dtype=np.int32)  # 标签为整数索引

    # 第二遍：填充数据
    train_offset = 0
    test_offset = 0

    print("正在加载特征数据...")
    for i, npz_file in enumerate(npz_files, 1):
        data = np.load(npz_file, allow_pickle=False)

        n_train = len(data["train_values"])
        train_X[train_offset:train_offset + n_train] = data["train_values"]
        train_y[train_offset:train_offset + n_train] = data["train_labels"]
        train_offset += n_train

        n_test = len(data["test_values"])
        test_X[test_offset:test_offset + n_test] = data["test_values"]
        test_y[test_offset:test_offset + n_test] = data["test_labels"]
        test_offset += n_test

        data.close()

        if i % 50 == 0 or i == len(npz_files):
            print(f"  [{i}/{len(npz_files)}] 已加载")

    print(f"✓ 加载完成，内存占用约 {(train_X.nbytes + test_X.nbytes) / 1024 / 1024:.1f} MB")

    return train_X, train_y, test_X, test_y

# 过滤掉无效从者后的映射表，使索引连续
def create_label_map(servants: list[Servant], output_dir: Path):
    file_path = output_dir / "label_map.json"
    if not file_path.exists():
        active_servants = [s for s in servants if s.revoke == 0]
        text = json.dumps([s.to_dict() for s in active_servants], indent=4)
        file_path.write_text(text)

    list_from_file = json.loads(file_path.read_text())
    return [Servant(**s) for s in list_from_file]

def train_model(output_path: Path, npz_labels_map: list[Servant]):
    # 从缓存的 npz 文件加载训练集
    train_X, train_y, test_X, test_y = load_cached_features(output_path)

    # 使用 XGBoost 训练分类器
    # xgboost 3.x 推荐方式：tree_method='hist' + device='cuda'/'cpu'
    device = 'cuda' if DEVICE == 'cuda' else 'cpu'
    print(f"\n使用 XGBoost 训练，device={device}")

    clf = XGBClassifier(
        n_estimators=800,  # 决策树轮数（多分类实际树数 = 轮数 × 类别数）
        max_depth=30,  # 限制树深度，防止过拟合 + 控制 ONNX 体积
        learning_rate=0.1,  # 学习率（步长收缩系数）
        subsample=0.8,  # 每棵树使用的样本比例（防止过拟合）
        colsample_bytree=0.8,  # 每棵树使用的特征比例
        tree_method='hist',  # 直方图算法，CPU/GPU 通用
        device=device,  # 训练设备（cuda/cpu）
        random_state=RANDOM_STATE,  # 固定随机种子
        n_jobs=-1,  # 使用所有 CPU 核心（CPU 模式下生效）
        eval_metric='mlogloss',  # xgboost 3.x 要求在构造函数中指定评估指标
        early_stopping_rounds=50,  # xgboost 3.x 要求在构造函数中指定早停轮数
    )

    print("\n开始训练分类器（带早停）...")
    clf.fit(
        train_X, train_y,
        eval_set=[(test_X, test_y)],
        verbose=100,  # 每 100 轮打印一次训练进度
    )

    print(f"\n✓ 早停触发，最佳轮数: {clf.best_iteration}")

    accuracy = clf.score(test_X, test_y)
    print(f"✓ 验证集准确率: {accuracy:.4f}")

    print("\n详细分类报告 (验证集):")
    class_names = [s.name_cn for s in npz_labels_map]

    # 输入：特征数组 test_X，形状 (n_samples, n_features)
    # 输出：预测标签数组，形状 (n_samples,)，每个值是整数（0, 1, 2...）
    y_pred = clf.predict(test_X)
    print(classification_report(
        test_y,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # 保存模型
    model_path = output_path / "xgboost_model.json"
    clf.save_model(model_path)
    print(f"✓ 模型已保存到 {model_path}")

    return clf


def export_to_onnx(clf: XGBClassifier, feature_dim: int, output_path: Path, model_name: str = "xgboost_model") -> Path:
    """将训练好的 XGBoost 模型转换为 ONNX 格式

    Args:
        clf: 已 fit 的 XGBClassifier（需要 classes_ 属性）
        feature_dim: 输入特征维度（ResNet18 提出来是 512）
        output_path: 输出目录
        model_name: 文件名前缀
    """
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # 使用 onnxmltools 的 xgboost 专用转换器
    onx = convert_xgboost(
        clf,
        initial_types=[("features", FloatTensorType([None, feature_dim]))],
    )

    onnx_path = output_path / f"{model_name}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✓ ONNX 模型已保存到 {onnx_path}（{onnx_path.stat().st_size / 1024 / 1024:.1f} MB）")
    return onnx_path


def convert_saved_model_to_onnx(model_path: Path, output_path: Path, feature_dim: int = 512) -> Path:
    """独立转换入口：把已有的 xgboost_model.json 转成 ONNX，无需重新训练"""
    clf = XGBClassifier()
    clf.load_model(model_path)
    return export_to_onnx(clf, feature_dim, output_path)


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

    npz_labels_map: list[Servant] = create_label_map(servants, output_path)

    # train.json 全从者，用于记录哪些从者已提取特征
    record = RecordFile(record_file_path)
    record.load([s.id for s in servants])

    use_gpu = True

    print(f"特征提取设备: {DEVICE}")

    # npz_labels_map 已经是排除无效从者的列表， train.json 记录不会命中，直接跳过
    if use_gpu:
        for servant in npz_labels_map:
            prepare_one(servant, record, images_dir, output_path, npz_labels_map)
    else:
        # npz_labels_map[:n] 进行分批次提取
        with ThreadPoolExecutor(max_workers=24 if DEVICE == 'cuda' else 24) as pool:
            promises = [
                pool.submit(
                    prepare_one,
                    servant,
                    record,
                    images_dir,
                    output_path,
                    npz_labels_map
                ) for servant in npz_labels_map
            ]

            for future in promises:
                try:
                    future.result()
                except Exception as e:
                    print(f"特征提取异常: {e}")


    train_model(output_path, npz_labels_map)

if __name__ == '__main__':
    IMAGES_DIR = Path("/mnt/DBC9-B465/temp")
    # IMAGES_DIR = Path = Path("G:\\temp")
    OUT_PUT_PATH = TEMP_DIR
    RECORD_FILE_PATH = TEMP_DIR / "train.json"
    train(IMAGES_DIR, OUT_PUT_PATH, RECORD_FILE_PATH)
