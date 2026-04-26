import os
import re
import numpy as np
import random
from pyfaidx import Fasta

# 输入输出配置：FASTA 基因组、各组织 narrowPeak 文件，以及生成 npz 的目录。
GENOME_FA = r"E:\shiyang\ARS-UCD1.2\ncbi_dataset\data\GCA_002263795.2\GCA_002263795.2_ARS-UCD1.2_genomic.fna"
PEAK_FILES = [
    r"E:\shiyang\peak\Adipose\Adipose_peaks.narrowPeak",
    r"E:\shiyang\peak\Cerebellum\Cerebellum_peaks.narrowPeak",
    r"E:\shiyang\peak\Cortex\Cortex_peaks.narrowPeak",
    r"E:\shiyang\peak\Hypothalamus\Hypothalamus_peaks.narrowPeak",
    r"E:\shiyang\peak\Liver\Liver_peaks.narrowPeak",
    r"E:\shiyang\peak\Lung\Lung_peaks.narrowPeak",
    r"E:\shiyang\peak\Muscle\Muscle_peaks.narrowPeak",
    r"E:\shiyang\peak\Spleen\Spleen_peaks.narrowPeak"
]
OUTPUT_DIR = r"D:\enformer_data1"

# Enformer 常用输入长度，后续会以 peak 中心点向两侧各取一半长度。
SEQ_LENGTH = 196608

def build_chrom_mapping(genome):
    # 将 narrowPeak 中可能出现的染色体写法映射到 FASTA 文件中的真实 key。
    # 例如 chr1、1、NC_/NW_/CM 编号等，最终都尽量转换成 genome.keys() 中的名称。
    chrom_map = {}
    genome_keys = list(genome.keys())
    
    for key in genome_keys:
        if 'NC_' in key or 'NW_' in key or 'CM' in key:
            match = re.search(r'(NC_|NW_|CM)(\d+)', key)
            if match:
                prefix = match.group(1)
                num = match.group(2)
                if prefix == 'NC_':
                    chrom_num = str(int(num) - 1)
                elif prefix == 'NW_':
                    chrom_num = f'Un_{num}'
                else:
                    chrom_num = f'chr{num}'
                chrom_map[chrom_num] = key
                chrom_map[f'chr{chrom_num}'] = key
        
        if 'chromosome' in key.lower() or 'chr' in key.lower():
            match = re.search(r'chr?(\d+)', key, re.IGNORECASE)
            if match:
                num = match.group(1)
                chrom_map[num] = key
                chrom_map[f'chr{num}'] = key
        
        if 'X' in key.upper() and 'chrX' not in chrom_map:
            chrom_map['X'] = key
            chrom_map['chrX'] = key
        if 'Y' in key.upper() and 'chrY' not in chrom_map:
            chrom_map['Y'] = key
            chrom_map['chrY'] = key
        if 'MT' in key.upper() or 'mitochond' in key.lower():
            chrom_map['MT'] = key
            chrom_map['chrM'] = key
    
    return chrom_map

def load_genome(fasta_path):
    import re
    # pyfaidx 会为 FASTA 建索引；rebuild=True 用于确保索引和当前 FASTA 一致。
    print("正在加载基因组...")
    genome = Fasta(fasta_path, rebuild=True)
    print(f"基因组加载完成，染色体数量: {len(genome.keys())}")
    
    chrom_map = build_chrom_mapping(genome)
    print(f"染色体映射表已创建，包含 {len(chrom_map)} 个映射")
    
    return genome, chrom_map

def parse_narrowPeak(peak_file):
    # narrowPeak 前三列通常是 chrom、start、end；这里使用区间中心点作为取序列中心。
    peaks = []
    with open(peak_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            center = (start + end) // 2
            peaks.append((chrom, center))
    return peaks

def extract_sequence(genome, chrom, center, half_len, chrom_map=None, return_chrom=False):
    # 根据 chrom_map 找到 FASTA 中实际存在的染色体名称，再围绕中心点提取固定长度序列。
    # return_chrom=True 时额外返回实际使用的染色体 key，方便保存到 npz 的 chrom 字段。
    try:
        actual_chrom = chrom
        if chrom_map and chrom in chrom_map:
            actual_chrom = chrom_map[chrom]
        elif chrom not in genome:
            if chrom_map:
                for key in genome.keys():
                    if chrom in key or key.endswith(chrom) or key.endswith(f'.{chrom}'):
                        actual_chrom = key
                        break
            
            if actual_chrom not in genome:
                return (None, None) if return_chrom else None
        
        if actual_chrom not in genome:
            return (None, None) if return_chrom else None
        
        chrom_len = len(genome[actual_chrom])
        start = max(0, center - half_len)
        end = min(chrom_len, center + half_len)
        
        seq = str(genome[actual_chrom][start:end]).upper()
        
        # 当中心点靠近染色体边界时，提取长度不足，用 N 在缺失的一侧补齐到固定长度。
        if len(seq) < 2 * half_len:
            if start == 0:
                seq = 'N' * (2 * half_len - len(seq)) + seq
            else:
                seq = seq + 'N' * (2 * half_len - len(seq))
        
        if return_chrom:
            return seq, actual_chrom
        return seq
    except Exception as e:
        print(f"提取序列失败 {chrom}:{center} - {e}")
        return (None, None) if return_chrom else None

def one_hot_encode(sequence):
    # 按 A/C/G/T 四通道编码；N 或未知碱基编码为全 0。
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    
    one_hot = []
    for base in sequence:
        if base in encoding:
            one_hot.append(encoding[base])
        else:
            one_hot.append([0, 0, 0, 0])
    
    return np.array(one_hot, dtype=np.float32)

def calculate_gc_content(sequence):
    # GC 含量计算时忽略 N，避免边界补齐或未知碱基影响比例。
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence) - sequence.count('N')
    if total == 0:
        return 0.5
    return gc_count / total

def get_gc_distribution(sequences):
    return [calculate_gc_content(seq) for seq in sequences]

def generate_background_regions(genome, peak_regions, num_samples, half_len, chrom_map=None):
    # 随机生成负样本候选区域，并粗略避开正样本 peak 附近的位置。
    background_regions = []
    chroms = list(genome.keys())
    
    # 用稀疏采样的坐标集合近似表示 peak 覆盖区域，降低负样本与正样本重叠的概率。
    peak_set = set()
    for chrom, center in peak_regions:
        for offset in range(-half_len, half_len + 1, 100):
            peak_set.add((chrom, center + offset))
    
    attempts = 0
    max_attempts = num_samples * 100
    
    while len(background_regions) < num_samples and attempts < max_attempts:
        actual_chrom = random.choice(chroms)
        chrom_len = len(genome[actual_chrom])
        
        if chrom_len < 2 * half_len:
            attempts += 1
            continue
        
        center = random.randint(half_len, chrom_len - half_len)
        
        is_valid = True
        for offset in range(-half_len, half_len + 1, 100):
            if (actual_chrom, center + offset) in peak_set:
                is_valid = False
                break
        
        if is_valid:
            background_regions.append((actual_chrom, center))
        
        attempts += 1
    
    return background_regions

def gc_matching(positive_seqs, background_regions, genome, half_len, tolerance=0.02, chrom_map=None):
    # 从背景候选区域中筛选 GC 含量接近正样本均值的负样本，减少 GC 偏差。
    pos_gc = get_gc_distribution(positive_seqs)
    pos_gc_mean = np.mean(pos_gc)
    pos_gc_std = np.std(pos_gc)
    
    print(f"    正样本GC含量: 均值={pos_gc_mean:.4f}, 标准差={pos_gc_std:.4f}")
    
    matched_background = []
    neg_gc_values = []
    
    for chrom, center in background_regions:
        seq = extract_sequence(genome, chrom, center, half_len, chrom_map)
        if seq:
            gc = calculate_gc_content(seq)
            if abs(gc - pos_gc_mean) <= tolerance:
                matched_background.append((chrom, center, seq))
                neg_gc_values.append(gc)
    
    if len(neg_gc_values) > 0:
        neg_gc_mean = np.mean(neg_gc_values)
        neg_gc_std = np.std(neg_gc_values)
        gc_diff = abs(neg_gc_mean - pos_gc_mean)
        print(f"    负样本GC含量: 均值={neg_gc_mean:.4f}, 标准差={neg_gc_std:.4f}")
        print(f"    GC含量差异: {gc_diff:.4f} ({gc_diff*100:.2f}%)")
        
        if gc_diff > 0.02:
            print(f"    警告: GC含量差异超过2%，需要调整容差参数")
    
    return matched_background

def process_peaks(genome, peak_file, output_dir, tissue_name, max_samples=1000, chrom_map=None):
    # 处理单个组织：提取正样本、生成并筛选负样本、one-hot 编码后保存为 npz。
    print(f"\n{'='*60}")
    print(f"处理组织: {tissue_name}")
    print(f"{'='*60}")
    
    peaks = parse_narrowPeak(peak_file)
    print(f"  Peak总数: {len(peaks)}")
    
    half_len = SEQ_LENGTH // 2
    
    positive_seqs = []
    positive_coords = []
    # 保存正样本实际对应的 FASTA 染色体 key，而不是 peak 文件里的原始写法。
    positive_chroms = []
    
    print(f"  正在提取正样本序列...")
    for i, (chrom, center) in enumerate(peaks[:max_samples]):
        if i % 100 == 0:
            print(f"    进度: {i}/{min(len(peaks), max_samples)}")
        seq, actual_chrom = extract_sequence(
            genome, chrom, center, half_len, chrom_map, return_chrom=True
        )
        if seq and len(seq) == SEQ_LENGTH:
            positive_seqs.append(seq)
            positive_coords.append((chrom, center))
            positive_chroms.append(actual_chrom)
    
    print(f"  有效正样本数: {len(positive_seqs)}")
    
    if len(positive_seqs) == 0:
        print(f"  警告: 没有有效的正样本，跳过该组织")
        return None, None
    
    num_negatives = len(positive_seqs)
    print(f"  正在生成背景区域...")
    background_regions = generate_background_regions(
        genome, positive_coords, num_negatives * 3, half_len, chrom_map
    )
    
    print(f"  生成背景区域数: {len(background_regions)}")
    
    print(f"  正在进行GC-matching (容差±2%)...")
    matched_negatives = gc_matching(
        positive_seqs, background_regions, genome, half_len, tolerance=0.02, chrom_map=chrom_map
    )
    
    print(f"  GC-matching后负样本数: {len(matched_negatives)}")
    
    negative_seqs = [item[2] for item in matched_negatives[:num_negatives]]
    # 负样本由 genome.keys() 随机生成，item[0] 已经是实际 FASTA 染色体 key。
    negative_chroms = [item[0] for item in matched_negatives[:num_negatives]]
    
    if len(negative_seqs) < len(positive_seqs):
        print(f"  警告: 负样本不足，调整正样本数量")
        positive_seqs = positive_seqs[:len(negative_seqs)]
        positive_chroms = positive_chroms[:len(negative_seqs)]
    
    print(f"  正在转换为独热编码...")
    X_pos = np.array([one_hot_encode(seq) for seq in positive_seqs])
    X_neg = np.array([one_hot_encode(seq) for seq in negative_seqs])
    
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    # chrom 与 X/y 一一对应；后面必须使用同一组 indices 一起打乱。
    chrom = np.asarray(positive_chroms + negative_chroms, dtype=str)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    chrom = chrom[indices]
    
    output_file = os.path.join(output_dir, f"{tissue_name}_data.npz")
    np.savez(output_file, X=X, y=y, chrom=chrom)
    
    print(f"\n  数据已保存: {output_file}")
    print(f"  最终数据维度: X={X.shape}, y={y.shape}")
    print(f"  正样本数: {np.sum(y == 1)}, 负样本数: {np.sum(y == 0)}")
    
    return X, y

def main():
    # 主流程：加载基因组和染色体映射表，然后按组织依次生成训练数据。
    print("="*60)
    print("Enformer数据预处理流程")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    genome, chrom_map = load_genome(GENOME_FA)
    
    print("\n染色体列表:")
    for i, chrom in enumerate(list(genome.keys())[:10]):
        print(f"  {chrom}: {len(genome[chrom]):,} bp")
    if len(genome.keys()) > 10:
        print(f"  ... 还有 {len(genome.keys()) - 10} 条染色体")
    
    print("\n染色体映射示例:")
    for i, (k, v) in enumerate(list(chrom_map.items())[:10]):
        print(f"  {k} -> {v}")
    
    results = {}
    for peak_file in PEAK_FILES:
        tissue_name = os.path.basename(os.path.dirname(peak_file))
        try:
            X, y = process_peaks(genome, peak_file, OUTPUT_DIR, tissue_name, chrom_map=chrom_map)
            if X is not None:
                results[tissue_name] = {'X_shape': X.shape, 'y_shape': y.shape}
        except Exception as e:
            print(f"\n处理 {tissue_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("处理完成汇总")
    print("="*60)
    for tissue, info in results.items():
        print(f"{tissue}: X={info['X_shape']}, y={info['y_shape']}")
    
    print("\n所有数据处理完成!")

if __name__ == "__main__":
    main()
