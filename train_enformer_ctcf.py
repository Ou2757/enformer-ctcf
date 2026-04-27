r"""
使用本地 Enformer 预训练权重微调牛 CTCF 结合位点二分类模型。

默认数据来源：
    D:\enformer_data1\*_data.npz

默认染色体划分：
    训练集：染色体 1-25
    验证集：染色体 28-29
    测试集：染色体 26-27

注意：
    当前 npz 中必须包含 X、y、chrom 三个键。脚本会检查每个 split 是否同时包含
    正负样本；如果验证集或测试集只有单一类别，会停止训练，避免输出无意义的 AUC。
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset


# 这个脚本尽量把“数据划分”和“真正加载 X 序列”分开：
# 1. 先只读取 y/chrom，建立轻量 manifest，确认 split 是否合理；
# 2. 训练时再按 manifest 分文件读取 X，避免一次性把所有 npz 的巨大矩阵读进内存。

# ARS-UCD1.2 FASTA 中前 30 条 CM 编号与牛染色体编号一一对应：
# chr1 -> CM008168.2, chr2 -> CM008169.2, ..., chr30 -> CM008197.2。
CM_CHROM_OFFSET = 8167


def set_seed(seed):
    """固定随机种子，便于复现实验结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_chroms(spec):
    """
    解析染色体参数。

    支持格式：
        1-25
        26,27
        chr28,chr29
        CM008168.2,CM008169.2
    """
    chroms = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        # 先处理范围写法，例如 1-25 或 chr1-chr25。
        range_match = re.fullmatch(r"(?:chr)?(\d+)\s*-\s*(?:chr)?(\d+)", token, re.I)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            chroms.extend(range(start, end + step, step))
            continue

        # 再处理单个数字染色体；如果不是数字，就当作已经是 FASTA key。
        chr_match = re.fullmatch(r"(?:chr)?(\d+)", token, re.I)
        if chr_match:
            chroms.append(int(chr_match.group(1)))
        else:
            chroms.append(token)

    return chroms


def chrom_to_fasta_key(chrom):
    """将 chr 数字转换为当前 npz 中保存的 FASTA 染色体 key。"""
    if isinstance(chrom, int):
        return f"CM{CM_CHROM_OFFSET + chrom:06d}.2"
    return str(chrom)


def center_crop_sequence(x, crop_length):
    """从 one-hot 序列中心裁剪固定长度，降低 Enformer 训练显存开销。"""
    if crop_length is None or crop_length <= 0 or crop_length >= x.shape[0]:
        return x
    start = (x.shape[0] - crop_length) // 2
    end = start + crop_length
    return x[start:end]


def discover_npz_files(data_dir):
    """扫描数据目录下所有组织的 npz 文件。"""
    paths = sorted(Path(data_dir).glob("*_data.npz"))
    if not paths:
        raise FileNotFoundError(f"未找到 npz 文件：{data_dir}")
    return paths


def build_manifest(data_dir, split_chroms, max_samples_per_split=None, seed=42):
    """
    读取每个 npz 的 y/chrom，建立按染色体划分后的样本索引。

    manifest 只保存文件路径、样本下标、标签和染色体，不会在这里加载巨大的 X。
    """
    split_key_sets = {
        split: {chrom_to_fasta_key(chrom) for chrom in chroms}
        for split, chroms in split_chroms.items()
    }

    # 同一条染色体如果同时进入 train/val/test，会造成数据泄漏，所以这里先检查。
    all_keys = []
    for split, keys in split_key_sets.items():
        for key in keys:
            all_keys.append((key, split))
    duplicated = [key for key, count in Counter(key for key, _ in all_keys).items() if count > 1]
    if duplicated:
        raise ValueError(f"同一染色体不能同时出现在多个 split 中：{duplicated}")

    manifest = {split: [] for split in split_chroms}
    npz_files = discover_npz_files(data_dir)

    for path in npz_files:
        tissue = path.name.replace("_data.npz", "")
        with np.load(path, allow_pickle=False) as data:
            # 每个 npz 都应来自预处理脚本，至少包含 one-hot 序列、标签、染色体来源。
            required_keys = {"X", "y", "chrom"}
            missing = required_keys.difference(data.files)
            if missing:
                raise KeyError(f"{path} 缺少键：{sorted(missing)}")

            y = np.asarray(data["y"])
            chrom = np.asarray(data["chrom"]).astype(str)
            if len(y) != len(chrom):
                raise ValueError(f"{path} 中 y 和 chrom 长度不一致：{len(y)} vs {len(chrom)}")

            # 这里只记录样本下标，不读取 X 的具体内容；X 会在 Dataset.__iter__ 中按需读取。
            for split, keys in split_key_sets.items():
                idx = np.flatnonzero(np.isin(chrom, list(keys)))
                for i in idx:
                    manifest[split].append(
                        {
                            "path": str(path),
                            "index": int(i),
                            "label": int(y[i]),
                            "chrom": str(chrom[i]),
                            "tissue": tissue,
                        }
                    )

    # 可选小样本模式，方便先做 smoke test。
    if max_samples_per_split is not None and max_samples_per_split > 0:
        rng = random.Random(seed)
        for split, rows in manifest.items():
            rng.shuffle(rows)
            manifest[split] = rows[:max_samples_per_split]

    return manifest, split_key_sets


def summarize_manifest(manifest):
    """统计每个 split 的样本数、标签分布和染色体分布。"""
    summary = {}
    for split, rows in manifest.items():
        labels = Counter(row["label"] for row in rows)
        chroms = Counter(row["chrom"] for row in rows)
        tissues = Counter(row["tissue"] for row in rows)
        summary[split] = {
            "num_samples": len(rows),
            "label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
            "chrom_counts": {str(k): int(v) for k, v in sorted(chroms.items())},
            "tissue_counts": {str(k): int(v) for k, v in sorted(tissues.items())},
        }
    return summary


def validate_splits(manifest):
    """
    确认每个 split 至少有样本，且同时包含正负样本。

    ROC AUC 和 PR AUC 都需要验证/测试集中同时存在 0 和 1。
    """
    errors = []
    for split, rows in manifest.items():
        # 即使训练能跑，单一类别的验证/测试集也无法计算 ROC AUC，因此必须提前阻止。
        labels = Counter(row["label"] for row in rows)
        if len(rows) == 0:
            errors.append(f"{split} 没有样本")
        if labels.get(0, 0) == 0 or labels.get(1, 0) == 0:
            errors.append(f"{split} 只有单一类别，label_counts={dict(labels)}")
    if errors:
        message = "\n".join(f"- {item}" for item in errors)
        raise ValueError(
            "染色体划分后数据集不满足训练/评估要求：\n"
            f"{message}\n"
            "请重新选择染色体划分，或回到预处理阶段检查 peak 染色体映射。"
        )


class NpzChromIterableDataset(IterableDataset):
    """
    按 manifest 从 npz 中读取样本。

    np.savez 生成的 npz 不适合真正的 mmap 随机读取；这里按文件分组，一次加载一个组织的 X，
    再产出该文件中属于当前 split 的样本，避免同时把所有组织数据读进内存。
    """

    def __init__(self, rows, crop_length=None, shuffle=False, seed=42):
        self.rows = list(rows)
        self.crop_length = crop_length
        self.shuffle = shuffle
        self.seed = seed
        # 按文件分组后，可以一次打开一个 npz，读取其中属于当前 split 的样本。
        self.rows_by_path = defaultdict(list)
        for row in self.rows:
            self.rows_by_path[row["path"]].append(row)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("npz 大文件读取默认只支持 num_workers=0，请不要开启多 worker。")

        rng = random.Random(self.seed + int(time.time()) if self.shuffle else self.seed)
        paths = list(self.rows_by_path.keys())
        if self.shuffle:
            # shuffle 文件顺序和文件内样本顺序，让不同组织的数据在训练中尽量混合。
            rng.shuffle(paths)

        for path in paths:
            rows = list(self.rows_by_path[path])
            if self.shuffle:
                rng.shuffle(rows)

            with np.load(path, allow_pickle=False) as data:
                # 注意：npz 是压缩归档格式，读取某个数组时通常会解压整个数组；
                # 所以这里每次只处理一个组织文件，控制峰值内存。
                x_array = data["X"]
                for row in rows:
                    x = np.asarray(x_array[row["index"]], dtype=np.float32)
                    x = center_crop_sequence(x, self.crop_length)

                    # Enformer PyTorch 实现通常接收形状为 [batch, length, 4] 的 one-hot 输入。
                    yield {
                        "x": torch.from_numpy(x),
                        "y": torch.tensor(float(row["label"]), dtype=torch.float32),
                        "chrom": row["chrom"],
                        "tissue": row["tissue"],
                        "source_index": row["index"],
                        "source_file": Path(row["path"]).name,
                    }


class EnformerBinaryClassifier(nn.Module):
    """Enformer backbone + 一个新的全连接二分类输出层。"""

    def __init__(self, enformer, embedding_dim=1536, dropout=0.1):
        super().__init__()
        self.enformer = enformer
        # 原模型的 human/mouse 多任务头不再使用；这里接一个单输出二分类头。
        # 训练时输出 logits，损失函数用 BCEWithLogitsLoss；评估时再 sigmoid 成概率。
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):
        # lucidrains/enformer-pytorch 支持 return_embeddings=True，可直接拿到 Transformer 后的表征。
        try:
            embeddings = self.enformer(x, return_embeddings=True)
        except TypeError as exc:
            raise TypeError(
                "当前 Enformer 实现不支持 return_embeddings=True；"
                "请安装/使用 enformer-pytorch，或修改 forward 提取 embedding 的方式。"
            ) from exc

        if isinstance(embeddings, (tuple, list)):
            # 有些实现会返回多个中间结果，默认取最后一个作为最终表征。
            embeddings = embeddings[-1]
        if isinstance(embeddings, dict):
            raise TypeError("Enformer 返回了 dict，而不是 embedding tensor，无法接入二分类头。")

        # 期望 embedding 为 [B, T, C]；对序列维度做平均池化，得到 [B, C]。
        if embeddings.ndim != 3:
            raise ValueError(f"Enformer embedding 维度应为 3，实际为 {embeddings.shape}")
        pooled = embeddings.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def load_enformer(pretrained_dir):
    """从本地目录加载 Enformer 预训练权重。"""
    try:
        # 本地目录里的 config.json/pytorch_model.bin 与 enformer-pytorch 的加载方式匹配。
        from enformer_pytorch import Enformer
    except ImportError as exc:
        raise ImportError(
            "缺少 enformer_pytorch。请先安装与本地权重匹配的 enformer-pytorch，"
            "例如：pip install enformer-pytorch"
        ) from exc

    if not Path(pretrained_dir).exists():
        raise FileNotFoundError(f"预训练目录不存在：{pretrained_dir}")

    if hasattr(Enformer, "from_pretrained"):
        return Enformer.from_pretrained(pretrained_dir)

    raise AttributeError("当前 enformer_pytorch.Enformer 没有 from_pretrained 方法，无法加载本地权重。")


def freeze_enformer_layers(model, freeze_conv=True, freeze_transformer_layers=6):
    """
    冻结 Enformer 前段层。

    不同 Enformer 实现的参数命名略有差异，这里按常见名称 stem/conv_tower/transformer.layers.N
    进行匹配；未匹配到的参数保持可训练。
    """
    frozen = 0
    trainable = 0

    for name, param in model.enformer.named_parameters():
        lower = name.lower()
        should_freeze = False

        # 卷积层学习的是相对底层的 motif/局部序列特征，迁移学习时通常先冻结。
        if freeze_conv and ("stem" in lower or "conv_tower" in lower):
            should_freeze = True

        # Transformer 前几层也偏通用，冻结后可以显著降低显存和训练不稳定性。
        if freeze_transformer_layers > 0 and "transformer" in lower:
            numbers = [int(n) for n in re.findall(r"\.(\d+)\.", f".{lower}.")]
            if numbers and min(numbers) < freeze_transformer_layers:
                should_freeze = True

        param.requires_grad = not should_freeze
        if should_freeze:
            frozen += param.numel()
        else:
            trainable += param.numel()

    return {"frozen_params": frozen, "trainable_backbone_params": trainable}


def make_optimizer_and_scheduler(model, args, steps_per_epoch):
    """创建 AdamW 优化器和 warm-up + cosine decay 学习率调度器。"""
    # 只把 requires_grad=True 的参数交给优化器，冻结参数不会被更新。
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        # warm-up 阶段线性升高学习率，避免一开始破坏预训练权重。
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # warm-up 之后使用 cosine decay 平滑降低学习率。
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def compute_pos_weight(rows, device):
    """根据训练集标签比例计算 BCEWithLogitsLoss 的 pos_weight。"""
    labels = Counter(row["label"] for row in rows)
    neg = labels.get(0, 0)
    pos = labels.get(1, 0)
    if pos == 0:
        raise ValueError("训练集没有正样本，无法计算 pos_weight。")
    # 正样本较少时，pos_weight 会提高正样本损失权重，缓解类别不平衡。
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, args):
    """训练一个 epoch，支持 AMP 和梯度累积。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        # AMP 只在 CUDA 下真正生效；CPU 训练时 autocast 会保持关闭。
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
            # 梯度累积时，每个小 batch 的 loss 先除以累积步数，保持等效学习率稳定。
            loss = loss / args.grad_accum_steps

        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            # 先 unscale 再裁剪梯度，是 AMP 下推荐的顺序。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_size = y.numel()
        total_loss += loss.item() * args.grad_accum_steps * batch_size
        total_items += batch_size

    # 如果最后剩余 batch 数不足 grad_accum_steps，也要把已累计的梯度更新掉。
    if total_items > 0 and step % args.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型并返回 loss、ROC AUC、PR AUC 和逐样本预测。"""
    model.eval()
    total_loss = 0.0
    total_items = 0
    labels = []
    probs = []
    rows = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        # 训练时用 logits 计算损失；这里转成 0-1 概率，便于计算 AUC 和导出结果。
        prob = torch.sigmoid(logits)

        batch_size = y.numel()
        total_loss += loss.item() * batch_size
        total_items += batch_size
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        probs.extend(prob.detach().cpu().numpy().tolist())

        for i in range(batch_size):
            rows.append(
                {
                    "label": int(labels[-batch_size + i]),
                    "prob": float(probs[-batch_size + i]),
                    "chrom": batch["chrom"][i],
                    "tissue": batch["tissue"][i],
                    "source_file": batch["source_file"][i],
                    "source_index": int(batch["source_index"][i]),
                }
            )

    if len(set(labels)) < 2:
        raise ValueError(f"评估集只有单一类别，无法计算 AUC：labels={Counter(labels)}")

    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    return {
        "loss": total_loss / max(1, total_items),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "labels": labels,
        "probs": probs,
        "rows": rows,
    }


def save_curves(labels, probs, output_dir, prefix):
    """保存 ROC 和 PR 曲线图片；如果 matplotlib 不可用，则跳过绘图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过 ROC/PR 曲线绘图。")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    # ROC 曲线更关注整体排序能力；类别较不平衡时也要重点看 PR 曲线。
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_roc.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} PR")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_pr.png", dpi=200)
    plt.close()


def save_predictions(rows, output_dir, prefix, threshold):
    """保存逐样本预测结果和高置信度位点。"""
    pred_path = Path(output_dir) / f"{prefix}_predictions.csv"
    high_path = Path(output_dir) / f"{prefix}_high_confidence.csv"
    fieldnames = ["label", "prob", "chrom", "tissue", "source_file", "source_index"]

    # 完整预测结果可用于后续错误分析、按组织统计、按染色体统计。
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 高置信度结果可作为候选新调控位点，再接后续功能富集或可视化分析。
    high_rows = [row for row in rows if row["prob"] >= threshold]
    with high_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(high_rows)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    """保存训练检查点。"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def write_metrics_header(path):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_roc_auc", "val_pr_auc", "lr"])


def append_metrics(path, row):
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Enformer for cattle CTCF binary prediction.")
    # 路径参数保持默认值即可复用当前项目生成的数据和已下载的预训练权重。
    parser.add_argument("--data-dir", default=r"D:\enformer_data1", help="包含 *_data.npz 的目录。")
    parser.add_argument("--pretrained-dir", default=r"E:\shiyang\enformer_pretrained", help="本地 Enformer 权重目录。")
    parser.add_argument("--output-dir", default=r"runs\enformer_ctcf", help="训练输出目录。")

    parser.add_argument("--train-chroms", default="1-25", help="训练集染色体。")
    parser.add_argument("--val-chroms", default="28-29", help="验证集染色体。")
    parser.add_argument("--test-chroms", default="26-27", help="测试集染色体。")

    # crop-length 越小，显存占用越低，但会损失远距离调控信息。
    parser.add_argument("--crop-length", type=int, default=49152, help="中心裁剪后的序列长度；设为 0 表示不裁剪。")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze-conv", action="store_true", default=True, help="冻结前段卷积层。")
    parser.add_argument("--no-freeze-conv", action="store_false", dest="freeze_conv", help="不冻结卷积层。")
    parser.add_argument("--freeze-transformer-layers", type=int, default=10, help="冻结前 N 个 Transformer 层。")

    parser.add_argument("--num-workers", type=int, default=0, help="npz 大文件建议保持为 0。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True, help="CUDA 下启用混合精度。")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="关闭混合精度。")
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="调试用，每个 split 最多保留多少样本。")
    parser.add_argument("--prepare-only", action="store_true", help="只检查数据划分并输出 summary，不启动训练。")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用户输入的是 1-25 这样的染色体编号，后续会转换成 npz 中的 CM008xxx.2 key。
    split_chroms = {
        "train": parse_chroms(args.train_chroms),
        "val": parse_chroms(args.val_chroms),
        "test": parse_chroms(args.test_chroms),
    }

    manifest, split_key_sets = build_manifest(
        args.data_dir,
        split_chroms,
        max_samples_per_split=args.max_samples_per_split or None,
        seed=args.seed,
    )
    summary = summarize_manifest(manifest)
    summary["split_fasta_keys"] = {k: sorted(v) for k, v in split_key_sets.items()}

    # 先把 split 摘要写到磁盘，即使后续类别检查失败，也能查看每个集合的样本分布。
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # 如果某个集合没有正/负样本，这里会提前停止。
    validate_splits(manifest)
    if args.prepare_only:
        print("数据划分检查完成，prepare-only 模式不启动训练。")
        return

    if args.num_workers != 0:
        raise ValueError("当前 npz 读取方式要求 --num-workers 0。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    train_dataset = NpzChromIterableDataset(
        manifest["train"], crop_length=args.crop_length, shuffle=True, seed=args.seed
    )
    val_dataset = NpzChromIterableDataset(manifest["val"], crop_length=args.crop_length, shuffle=False)
    test_dataset = NpzChromIterableDataset(manifest["test"], crop_length=args.crop_length, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # 加载预训练 Enformer，然后接入新的二分类头。
    enformer = load_enformer(args.pretrained_dir)
    model = EnformerBinaryClassifier(enformer, embedding_dim=1536, dropout=args.dropout)
    freeze_info = freeze_enformer_layers(
        model,
        freeze_conv=args.freeze_conv,
        freeze_transformer_layers=args.freeze_transformer_layers,
    )
    print(f"冻结参数统计：{freeze_info}")
    model.to(device)

    # 类别不均衡时，pos_weight 会让正样本错误受到更高惩罚。
    pos_weight = compute_pos_weight(manifest["train"], device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # scheduler 的步数按“优化器真实更新次数”估计，而不是原始 batch 次数。
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch_size / args.grad_accum_steps))
    optimizer, scheduler = make_optimizer_and_scheduler(model, args, steps_per_epoch)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    metrics_path = output_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        # 每轮先训练，再在验证集上选择 best checkpoint。
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, args
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        lr = scheduler.get_last_lr()[0]

        print(
            f"epoch={epoch} train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_roc_auc={val_metrics['roc_auc']:.5f} "
            f"val_pr_auc={val_metrics['pr_auc']:.5f} lr={lr:.6g}"
        )

        append_metrics(
            metrics_path,
            [epoch, train_loss, val_metrics["loss"], val_metrics["roc_auc"], val_metrics["pr_auc"], lr],
        )

        save_checkpoint(
            output_dir / "last_model.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            val_metrics,
            args,
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            # 以验证集 ROC AUC 作为保存最佳模型的标准。
            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                args,
            )

    # 使用最后一轮模型进行测试；如需严格使用 best checkpoint，可在这里 load best_model.pt。
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"test_loss={test_metrics['loss']:.5f} "
        f"test_roc_auc={test_metrics['roc_auc']:.5f} "
        f"test_pr_auc={test_metrics['pr_auc']:.5f}"
    )

    save_curves(test_metrics["labels"], test_metrics["probs"], output_dir, "test")
    save_predictions(
        test_metrics["rows"],
        output_dir,
        "test",
        threshold=args.high_confidence_threshold,
    )


if __name__ == "__main__":
    main()
