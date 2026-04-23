#!/usr/bin/env python3
"""
CTCF 结合位点预测模型 - 基于 Enformer 微调 (CUDA优化版)
用于牛（Bos taurus）基因组 CTCF 结合位点的二分类预测

设备配置: 80GB 内存, 32GB 显存
数据: 8个组织的npz文件，每个约6GB，每个组织2000个正负样本
"""

import os
import gc
import glob
import random
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    precision_recall_curve,
    f1_score,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enformer 相关导入
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import (
    freeze_all_layers_,
    freeze_batchnorms_,
    set_module_requires_grad_
)


# ==================== CUDA 优化设置 ====================

def setup_cuda_optimizations():
    """设置 CUDA 优化选项"""
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        return False
    
    # 启用 cuDNN 自动调优
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cudnn.deterministic = False  # 关闭确定性以提高性能
    
    # 启用 TF32 (Ampere架构及以上)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 打印 CUDA 信息
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU 计算能力: {torch.cuda.get_device_capability(0)}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    return True


def clear_cuda_cache():
    """清理 CUDA 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def print_gpu_memory(prefix=""):
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"{prefix} GPU显存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB, 峰值: {max_allocated:.2f}GB")


# ==================== 配置参数 ====================

class Config:
    """训练配置类"""
    # 路径配置
    DATA_DIR = "/root/autodl-tmp/enformer_data"
    PRETRAINED_PATH = "/root/enformer_pretrained"
    OUTPUT_DIR = "./output"
    
    # 数据键名
    SEQUENCE_KEY = "X"
    LABEL_KEY = "y"
    
    # 模型配置
    SEQUENCE_LENGTH = 196_608
    TARGET_LENGTH = 896
    EMBEDDING_DIM = 3072
    
    # 训练配置 - CUDA优化
    BATCH_SIZE = 8  
    GRADIENT_ACCUMULATION_STEPS = 8  
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # 冻结配置
    FREEZE_CONV_LAYERS = True
    FREEZE_TRANSFORMER_LAYERS = 10  # 冻结前10层
    
    # 数据配置
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    NUM_WORKERS = 4
    PIN_MEMORY = True  # CUDA优化
    PREFETCH_FACTOR = 2  # 数据预取
    
    # CUDA优化配置
    SEED = 42
    USE_AMP = True  # 混合精度
    USE_CHECKPOINTING = True  # 梯度检查点
    USE_COMPILE = False  # torch.compile (PyTorch 2.0+)
    NON_BLOCKING = True  # 异步数据传输
    
    # 保存配置
    SAVE_EVERY_N_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 10


# ==================== 工具函数 ====================

def set_seed(seed: int, use_cuda: bool = True):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 清除现有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """获取设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("警告: CUDA不可用，使用CPU训练将非常缓慢")
    return device


# ==================== 数据集类 (CUDA优化) ====================

class CTCFDatasetCUDA(Dataset):
    """
    CTCF 结合位点数据集 - CUDA优化版
    """
    
    def __init__(
        self, 
        npz_files: List[str], 
        indices: Optional[np.ndarray] = None,
        seq_length: int = 196_608,
        seq_key: str = "X",
        label_key: str = "y",
        cache_data: bool = False  # 是否缓存数据到内存
    ):
        self.npz_files = npz_files
        self.seq_length = seq_length
        self.seq_key = seq_key
        self.label_key = label_key
        self.cache_data = cache_data
        
        # 建立文件索引映射
        self.file_sample_counts = []
        self.cumulative_counts = [0]
        
        for npz_file in npz_files:
            with np.load(npz_file, mmap_mode='r') as data:
                n_samples = data[self.seq_key].shape[0]
                self.file_sample_counts.append(n_samples)
                self.cumulative_counts.append(self.cumulative_counts[-1] + n_samples)
        
        self.total_samples = self.cumulative_counts[-1]
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.total_samples)
        
        # 数据缓存
        self.data_cache = {}
        if cache_data:
            self._preload_data()
    
    def _preload_data(self):
        """预加载数据到内存"""
        print("预加载数据到内存...")
        for file_idx, npz_file in enumerate(tqdm(self.npz_files)):
            with np.load(npz_file) as data:
                self.data_cache[file_idx] = {
                    'X': data[self.seq_key].astype(np.float32),
                    'y': data[self.label_key].astype(np.float32)
                }
    
    def _get_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """全局索引 -> (文件索引, 文件内索引)"""
        for file_idx, (start, end) in enumerate(zip(
            self.cumulative_counts[:-1], 
            self.cumulative_counts[1:]
        )):
            if start <= global_idx < end:
                return file_idx, global_idx - start
        raise IndexError(f"索引 {global_idx} 超出范围")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_idx = self.indices[idx]
        file_idx, local_idx = self._get_file_and_local_idx(global_idx)
        
        if self.cache_data and file_idx in self.data_cache:
            sequence = self.data_cache[file_idx]['X'][local_idx]
            label = self.data_cache[file_idx]['y'][local_idx]
        else:
            with np.load(self.npz_files[file_idx], mmap_mode='r') as data:
                sequence = np.array(data[self.seq_key][local_idx], dtype=np.float32)
                label = np.array(data[self.label_key][local_idx], dtype=np.float32)
        
        # 转换为连续内存的tensor (优化CUDA传输)
        sequence = torch.from_numpy(np.ascontiguousarray(sequence))
        label = torch.tensor(label, dtype=torch.float32)
        
        return sequence, label


# ==================== 模型定义 (CUDA优化) ====================

class CTCFClassifierCUDA(nn.Module):
    """
    基于 Enformer 的 CTCF 二分类器 - CUDA优化版
    """
    
    def __init__(
        self,
        enformer_path: str,
        embedding_dim: int = 3072,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        print(f"加载 Enformer 预训练模型: {enformer_path}")
        
        # 加载Enformer
        if os.path.exists(enformer_path):
            self.enformer = from_pretrained(
                enformer_path,
                use_checkpointing=use_checkpointing
            )
        else:
            print("本地路径不存在，从HuggingFace Hub加载...")
            self.enformer = from_pretrained(
                'EleutherAI/enformer-official-rough',
                use_checkpointing=use_checkpointing
            )
        
        # 分类头 - 使用更高效的结构
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Xavier初始化"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def freeze_layers(
        self,
        freeze_conv: bool = True,
        freeze_n_transformer_layers: int = 8
    ):
        """冻结层"""
        freeze_all_layers_(self.enformer)
        freeze_batchnorms_(self.enformer)
        
        # 解冻后面的transformer层
        if freeze_n_transformer_layers < 11:
            transformer_blocks = self.enformer.transformer
            for i, block in enumerate(transformer_blocks):
                if i >= freeze_n_transformer_layers:
                    set_module_requires_grad_(block, True)
        
        set_module_requires_grad_(self.enformer.final_pointwise, True)
        set_module_requires_grad_(self.classifier, True)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"冻结参数: {total_params - trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # Enformer embeddings
        embeddings = self.enformer(x, return_only_embeddings=True)
        
        # 全局平均池化
        pooled = embeddings.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits


# ==================== 训练器 (CUDA优化) ====================

class TrainerCUDA:
    """CUDA优化的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
        device: torch.device,
        logger: logging.Logger
    ):
        self.config = config
        self.device = device
        self.logger = logger
        
        # 模型移到GPU
        self.model = model.to(device)
        
        # 可选: torch.compile 加速 (PyTorch 2.0+)
        if config.USE_COMPILE and hasattr(torch, 'compile'):
            self.logger.info("使用 torch.compile 编译模型...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 优化器
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            fused=True if torch.cuda.is_available() else False  # CUDA fused优化
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=config.WARMUP_RATIO,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 混合精度 - 使用新API
        self.scaler = GradScaler('cuda') if config.USE_AMP else None
        
        # 训练状态
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': [],
            'learning_rate': []
        }
        
        # CUDA事件用于计时
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch - CUDA优化"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
            # 异步数据传输
            sequences = sequences.to(self.device, non_blocking=self.config.NON_BLOCKING)
            labels = labels.to(self.device, non_blocking=self.config.NON_BLOCKING).unsqueeze(1)
            
            # 混合精度前向传播
            with autocast('cuda', enabled=self.config.USE_AMP):
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # 同步CUDA
        torch.cuda.synchronize()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict:
        """评估 - CUDA优化"""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0.0
        
        for sequences, labels in tqdm(dataloader, desc=desc):
            sequences = sequences.to(self.device, non_blocking=self.config.NON_BLOCKING)
            labels = labels.to(self.device, non_blocking=self.config.NON_BLOCKING).unsqueeze(1)
            
            with autocast('cuda', enabled=self.config.USE_AMP):
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        
        # 同步
        torch.cuda.synchronize()
        
        # 合并结果
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        probs = torch.sigmoid(all_logits).numpy().flatten()
        labels_np = all_labels.numpy().flatten()
        preds = (probs > 0.5).astype(int)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'auc': roc_auc_score(labels_np, probs),
            'ap': average_precision_score(labels_np, probs),
            'accuracy': accuracy_score(labels_np, preds),
            'f1': f1_score(labels_np, preds),
            'probs': probs,
            'labels': labels_np,
            'preds': preds
        }
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        self.logger.info("=" * 60)
        self.logger.info("开始训练 (CUDA优化)")
        self.logger.info("=" * 60)
        
        print_gpu_memory("训练开始前")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 记录开始时间
            self.start_event.record()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.evaluate(self.val_loader, "Validating")
            
            # 记录结束时间
            self.end_event.record()
            torch.cuda.synchronize()
            epoch_time = self.start_event.elapsed_time(self.end_event) / 1000  # 转换为秒
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_ap'].append(val_metrics['ap'])
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val AP: {val_metrics['ap']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # 保存最佳模型
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"★ 新最佳模型! AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % self.config.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(epoch)
            
            # 早停
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"早停触发! Epoch {epoch+1}")
                break
            
            # 定期清理缓存
            if (epoch + 1) % 5 == 0:
                clear_cuda_cache()
        
        self.logger.info("训练完成!")
        print_gpu_memory("训练结束后")
        
        # 最终测试
        self.final_evaluation()
    
    def final_evaluation(self):
        """最终评估"""
        self.logger.info("加载最佳模型进行测试...")
        
        best_path = os.path.join(self.config.OUTPUT_DIR, "best_model.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, "Final Testing")
        
        self.logger.info("=" * 60)
        self.logger.info("最终测试结果:")
        self.logger.info(f"  Test Loss:     {test_metrics['loss']:.4f}")
        self.logger.info(f"  Test AUC:      {test_metrics['auc']:.4f}")
        self.logger.info(f"  Test AP:       {test_metrics['ap']:.4f}")
        self.logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  Test F1:       {test_metrics['f1']:.4f}")
        self.logger.info("=" * 60)
        
        self.plot_curves(test_metrics)
        self.save_results(test_metrics)
    
    def plot_curves(self, test_metrics: Dict):
        """绘制曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Loss曲线
        ax = axes[0, 0]
        ax.plot(self.history['val_loss'], label='Val', color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. AUC/AP曲线
        ax = axes[0, 1]
        ax.plot(self.history['val_auc'], label='AUC', color='green', linewidth=2)
        ax.plot(self.history['val_ap'], label='AP', color='orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation AUC & AP')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 学习率
        ax = axes[0, 2]
        ax.plot(self.history['learning_rate'], color='purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. ROC曲线
        ax = axes[1, 0]
        fpr, tpr, _ = roc_curve(test_metrics['labels'], test_metrics['probs'])
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC (AUC = {test_metrics["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # 5. PR曲线
        ax = axes[1, 1]
        precision, recall, _ = precision_recall_curve(test_metrics['labels'], test_metrics['probs'])
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR (AP = {test_metrics["ap"]:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        # 6. 混淆矩阵
        ax = axes[1, 2]
        cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"曲线已保存: {save_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.config.OUTPUT_DIR, 'best_model.pt')
        else:
            path = os.path.join(self.config.OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, path)
    
    def save_results(self, test_metrics: Dict):
        """保存结果"""
        results = {
            'test_loss': float(test_metrics['loss']),
            'test_auc': float(test_metrics['auc']),
            'test_ap': float(test_metrics['ap']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1': float(test_metrics['f1']),
            'best_val_auc': float(self.best_val_auc),
            'history': {k: [float(v) for v in vals] for k, vals in self.history.items()}
        }
        
        with open(os.path.join(self.config.OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        np.savez(
            os.path.join(self.config.OUTPUT_DIR, 'predictions.npz'),
            probs=test_metrics['probs'],
            labels=test_metrics['labels'],
            preds=test_metrics['preds']
        )


# ==================== 数据准备 ====================

def prepare_data(config: Config, logger: logging.Logger):
    """准备数据"""
    npz_files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.npz")))
    
    if not npz_files:
        raise FileNotFoundError(f"未找到npz文件: {config.DATA_DIR}")
    
    logger.info(f"找到 {len(npz_files)} 个npz文件")
    
    # 统计样本数
    total_samples = 0
    for npz_file in npz_files:
        with np.load(npz_file, mmap_mode='r') as data:
            total_samples += data[config.SEQUENCE_KEY].shape[0]
    
    logger.info(f"总样本数: {total_samples}")
    
    # 划分数据集
    all_indices = np.arange(total_samples)
    
    train_indices, temp_indices = train_test_split(
        all_indices, train_size=config.TRAIN_RATIO,
        random_state=config.SEED, shuffle=True
    )
    
    val_ratio_adj = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_ratio_adj,
        random_state=config.SEED, shuffle=True
    )
    
    logger.info(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")
    
    # 创建数据集
    train_dataset = CTCFDatasetCUDA(
        npz_files, indices=train_indices,
        seq_length=config.SEQUENCE_LENGTH,
        seq_key=config.SEQUENCE_KEY,
        label_key=config.LABEL_KEY
    )
    
    val_dataset = CTCFDatasetCUDA(
        npz_files, indices=val_indices,
        seq_length=config.SEQUENCE_LENGTH,
        seq_key=config.SEQUENCE_KEY,
        label_key=config.LABEL_KEY
    )
    
    test_dataset = CTCFDatasetCUDA(
        npz_files, indices=test_indices,
        seq_length=config.SEQUENCE_LENGTH,
        seq_key=config.SEQUENCE_KEY,
        label_key=config.LABEL_KEY
    )
    
    # 创建DataLoader - CUDA优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 配置
    config = Config()
    
    # CUDA优化设置
    cuda_available = setup_cuda_optimizations()
    
    # 设置种子
    set_seed(config.SEED, use_cuda=cuda_available)
    
    # 日志
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.OUTPUT_DIR)
    
    logger.info("=" * 60)
    logger.info("CTCF 结合位点预测 - Enformer微调 (CUDA优化版)")
    logger.info("=" * 60)
    
    # 设备
    device = get_device()
    
    # 数据
    logger.info("准备数据...")
    train_loader, val_loader, test_loader = prepare_data(config, logger)
    
    # 清理缓存
    clear_cuda_cache()
    print_gpu_memory("数据加载后")
    
    # 模型
    logger.info("构建模型...")
    model = CTCFClassifierCUDA(
        enformer_path=config.PRETRAINED_PATH,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=512,
        dropout_rate=0.3,
        use_checkpointing=config.USE_CHECKPOINTING
    )
    
    # 冻结层
    logger.info("冻结层...")
    model.freeze_layers(
        freeze_conv=config.FREEZE_CONV_LAYERS,
        freeze_n_transformer_layers=config.FREEZE_TRANSFORMER_LAYERS
    )
    
    print_gpu_memory("模型加载后")
    
    # 训练器
    trainer = TrainerCUDA(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # 训练
    trainer.train()
    
    logger.info("全部完成!")


if __name__ == "__main__":
    main()