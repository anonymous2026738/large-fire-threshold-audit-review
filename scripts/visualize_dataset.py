"""FireTracks数据集可视化脚本
用于分析正负样本并生成可视化图表
"""
import os
from typing import Optional

# 避免 OpenMP 多重加载报错（libomp.dll vs libiomp5md.dll）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import sys
from datetime import datetime

# 全局字体设置：论文友好的 Times New Roman，并统一大字号，适合缩放
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 22      # 子图标题
plt.rcParams['axes.labelsize'] = 20      # 坐标轴标签
plt.rcParams['xtick.labelsize'] = 16     # 刻度标签
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16     # 图例
plt.rcParams['figure.titlesize'] = 16    # 总标题（下方不再主动使用）

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 如果当前在data_visualization目录，需要回到项目根目录
if os.path.basename(current_dir) == 'data_visualization':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir
code_dir = os.path.join(project_root, 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from fire_equality.datamodules import FireTracksDataset
from torch.utils.data import DataLoader

def load_dataset(filepath='dataset/processed_firetracks_pixel_binary.pth'):
    """加载像素级二分类数据集"""
    print("📊 加载数据集...")
    data = torch.load(filepath, weights_only=False)
    
    if 'spatiotemporal_samples' in data:
        print("   检测到像素级二分类数据集格式")
        spatiotemporal_samples = data['spatiotemporal_samples']
        config = data['config']
        
        dataset = FireTracksDataset(
            spatiotemporal_samples,
            target_type='binary_classification'
        )
        
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
        
        data['dataset'] = dataset
        data['dataloader'] = dataloader
        
        # 尝试加载events和components数据用于FRP查找
        try:
            import os
            import pandas as pd
            data_dir = 'dataset/firetracks_data'
            if os.path.exists(data_dir):
                # 尝试加载events数据（只加载少量用于查找）
                events_file = os.path.join(data_dir, 'v.h5')
                if os.path.exists(events_file):
                    print("   正在加载events数据用于FRP查找...")
                    # 只加载有maxFRP字段的数据
                    try:
                        events = pd.read_hdf(events_file, columns=['lat', 'lon', 'dtime', 'maxFRP'], start=0, stop=1000000)
                        events['dtime'] = pd.to_datetime(events['dtime'])
                        if 'preprocessed_data' not in data:
                            data['preprocessed_data'] = {}
                        data['preprocessed_data']['events'] = events
                        print(f"   ✅ 已加载 {len(events):,} 条events记录")
                    except Exception as e:
                        print(f"   ⚠️  加载events数据失败: {e}")
                
                # 尝试加载components数据
                components_file = os.path.join(data_dir, 'cp.h5')
                if os.path.exists(components_file):
                    print("   正在加载components数据用于FRP查找...")
                    try:
                        # 先读取索引和FRP相关列
                        components = pd.read_hdf(components_file, start=0, stop=100000)
                        # 只保留需要的列
                        frp_cols = [col for col in ['maxFRP_sum', 'maxFRP_mean', 'cp'] if col in components.columns]
                        if frp_cols:
                            components = components[frp_cols]
                        if 'preprocessed_data' not in data:
                            data['preprocessed_data'] = {}
                        data['preprocessed_data']['components'] = components
                        print(f"   ✅ 已加载 {len(components):,} 条components记录")
                        if 'cp' in components.columns:
                            print(f"   ✅ components数据包含cp列，可用于查找")
                    except Exception as e:
                        print(f"   ⚠️  加载components数据失败: {e}")
        except Exception as e:
            print(f"   ⚠️  加载辅助数据时出错: {e}")
    elif 'dataset' in data:
        print("   检测到旧格式，直接使用")
        dataset = data['dataset']
        dataloader = data.get('dataloader', None)
    else:
        raise ValueError("数据文件格式不正确")
    
    return data

def analyze_positive_negative_samples(dataset, spatiotemporal_samples=None):
    """分析正负样本的统计信息"""
    print("\n" + "="*60)
    print("=== 正负样本分析 ===")
    print("="*60)
    
    positive_samples = []
    negative_samples = []
    
    for i in range(len(dataset)):
        features, target = dataset[i]
        sample_info = {
            'idx': i,
            'features': features.numpy(),
            'target': target.item()
        }
        
        # 如果提供了原始样本数据，保存metadata以便后续访问FRP等信息
        if spatiotemporal_samples is not None and i < len(spatiotemporal_samples):
            sample_info['original_sample'] = spatiotemporal_samples[i]
        
        if target.item() == 1:
            positive_samples.append(sample_info)
        else:
            negative_samples.append(sample_info)
    
    print(f"✅ 正样本数量: {len(positive_samples):,}")
    print(f"✅ 负样本数量: {len(negative_samples):,}")
    print(f"✅ 正负样本比例: 1:{len(negative_samples)/len(positive_samples):.2f}" if len(positive_samples) > 0 else "")
    
    # 分析特征统计
    if len(positive_samples) > 0 and len(negative_samples) > 0:
        pos_features = np.stack([s['features'] for s in positive_samples], axis=0)
        neg_features = np.stack([s['features'] for s in negative_samples], axis=0)
        
        print(f"\n特征统计对比（使用所有样本）:")
        # 8通道方案
        num_channels = pos_features.shape[1]
        channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
        if num_channels != 8:
            print(f"⚠️ 当前数据通道数为 {num_channels}，与期望的8不一致。请使用包含8通道特征的数据文件。")
            return positive_samples, negative_samples
        
        for ch_idx, ch_name in enumerate(channel_names):
            pos_ch = pos_features[:, ch_idx, :, :, :]
            neg_ch = neg_features[:, ch_idx, :, :, :]
            
            print(f"\n通道 {ch_idx}: {ch_name}")
            print(f"  正样本 - 均值: {pos_ch.mean():.4f}, 非零比例: {(pos_ch != 0).sum()/pos_ch.size*100:.2f}%")
            print(f"  负样本 - 均值: {neg_ch.mean():.4f}, 非零比例: {(neg_ch != 0).sum()/neg_ch.size*100:.2f}%")
    
    return positive_samples, negative_samples

def visualize_class_distribution(dataset, save_path='data_visualization/class_distribution.png'):
    """可视化类别分布（正负样本，饼图）"""
    print("\n📊 生成类别分布图...")
    
    all_targets = [dataset[i][1].item() for i in range(len(dataset))]
    counter = Counter(all_targets)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # 饼图（配色：Negative 天蓝，Positive 浅粉）
    labels = ['Negative (0)', 'Positive (1)']
    sizes = [counter.get(0, 0), counter.get(1, 0)]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0.05)

    # 自定义 autopct，在百分比下方标注「全数据集」总样本数量（固定为 19,000 / 37,970）
    def make_autopct():
        # 使用整数方便在 f-string 中加千分位逗号
        counts = [37970, 19000]  # 顺序与 labels / sizes 对应: [Negative, Positive]
        state = {'idx': 0}

        def autopct(pct):
            i = state['idx']
            state['idx'] += 1
            # 不再显示换行符，改为同一行标注
            return f'{pct:.1f}% ({counts[i]:,})'

        return autopct
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct=make_autopct(), shadow=True, startangle=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {save_path}")
    plt.close()

def visualize_feature_distribution(dataset, positive_samples, negative_samples, save_path='data_visualization/feature_distribution.png'):
    """可视化正负样本的特征分布对比"""
    print("\n📈 生成特征分布对比图...")
    
    # 使用所有样本（不再限制为100个）
    pos_features_list = []
    neg_features_list = []
    
    for i in range(len(positive_samples)):
        pos_features_list.append(positive_samples[i]['features'])
    
    for i in range(len(negative_samples)):
        neg_features_list.append(negative_samples[i]['features'])
    
    pos_features = np.stack(pos_features_list, axis=0) if pos_features_list else None
    neg_features = np.stack(neg_features_list, axis=0) if neg_features_list else None
    
    # 8通道名称
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    # 如果当前数据非8通道，直接给出提示并提前保存一个空图
    if (pos_features is not None and pos_features.shape[1] != 8) or (neg_features is not None and neg_features.shape[1] != 8):
        print(f"⚠️ 当前数据通道数与8不一致，跳过特征分布图。")
        fig, _ = plt.subplots(1, 1, figsize=(6, 4))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # 子图网格：2行4列显示所有8个通道
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    # fig.suptitle('Feature distribution: positive vs negative samples', fontsize=16, fontweight='bold')
    
    for ch_idx, (ax, ch_name) in enumerate(zip(axes.flat, channel_names)):
        if pos_features is not None:
            pos_ch_data = pos_features[:, ch_idx, :, :, :].flatten()
            pos_ch_nonzero = pos_ch_data[pos_ch_data != 0]
        else:
            pos_ch_nonzero = np.array([])
        
        if neg_features is not None:
            neg_ch_data = neg_features[:, ch_idx, :, :, :].flatten()
            neg_ch_nonzero = neg_ch_data[neg_ch_data != 0]
        else:
            neg_ch_nonzero = np.array([])
        
        if len(pos_ch_nonzero) > 0 or len(neg_ch_nonzero) > 0:
            # 计算统一的数据范围，确保正负样本使用相同的bin边界
            all_values = []
            if len(pos_ch_nonzero) > 0:
                all_values.extend(pos_ch_nonzero)
            if len(neg_ch_nonzero) > 0:
                all_values.extend(neg_ch_nonzero)
            
            if len(all_values) > 0:
                # 对于population通道，使用分位数来设置范围，避免极端值影响可视化
                if ch_name == 'population':
                    # 使用5%和95%分位数来设置范围，这样可以更好地展示主要分布
                    # 同时避免极端值（如30000）占据大部分画布空间
                    all_values_array = np.array(all_values)
                    p5 = np.percentile(all_values_array, 5)
                    p95 = np.percentile(all_values_array, 95)
                    
                    # 如果分位数范围太小（说明数据集中在很小范围），使用更宽的范围
                    if p95 - p5 < 0.1:
                        # 使用中位数和四分位距（IQR）来设置范围
                        median = np.median(all_values_array)
                        q25 = np.percentile(all_values_array, 25)
                        q75 = np.percentile(all_values_array, 75)
                        iqr = q75 - q25
                        # 使用 IQR 的倍数来设置范围，但不超过实际的最小最大值
                        data_min = max(0, median - 2 * iqr)  # population不应该小于0
                        data_max = median + 2 * iqr
                        # 如果范围仍然太小，使用实际的最小最大值
                        if data_max - data_min < 0.01:
                            data_min = np.min(all_values_array)
                            data_max = np.max(all_values_array)
                    else:
                        # 使用5%和95%分位数
                        data_min = max(0, p5)  # population不应该小于0
                        data_max = p95
                    
                    # 确保范围不为0，避免除零错误
                    if data_max == data_min:
                        data_max = data_min + 1.0
                    
                    # 添加一些边距以便更好地显示（5%的边距）
                    range_margin = (data_max - data_min) * 0.05
                    data_min = max(0, data_min - range_margin)
                    data_max = data_max + range_margin
                else:
                    # 其他通道使用最小最大值
                    data_min = np.min(all_values)
                    data_max = np.max(all_values)
                    # 确保范围不为0，避免除零错误
                    if data_max == data_min:
                        data_max = data_min + 1.0
                
                # 使用统一的范围和bin数量
                bins = 50
                bin_range = (data_min, data_max)
                
                # 绘制直方图，使用统一的范围（配色与类别分布图保持一致：Negative 天蓝，Positive 浅粉）
                if len(pos_ch_nonzero) > 0:
                    ax.hist(pos_ch_nonzero, bins=bins, range=bin_range, alpha=0.6, 
                           label='Positive', color='#ff9999', edgecolor='black', density=True)
                if len(neg_ch_nonzero) > 0:
                    ax.hist(neg_ch_nonzero, bins=bins, range=bin_range, alpha=0.6, 
                           label='Negative', color='#66b3ff', edgecolor='black', density=True)
            else:
                # 如果没有数据，显示提示
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
            
            ax.set_title(f'Channel {ch_idx}: {ch_name}', fontsize=22)
            ax.set_xlabel('Feature value')
            ax.set_ylabel('Density')
            # 图例仅在第一个通道显示，保持默认字号
            if ch_idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = []
            if len(pos_ch_nonzero) > 0:
                stats_text.append(f'Positive:\n  mean: {pos_ch_nonzero.mean():.4f}\n  std: {pos_ch_nonzero.std():.4f}\n  non-zero: {len(pos_ch_nonzero)/len(pos_ch_data)*100:.2f}%')
            if len(neg_ch_nonzero) > 0:
                stats_text.append(f'Negative:\n  mean: {neg_ch_nonzero.mean():.4f}\n  std: {neg_ch_nonzero.std():.4f}\n  non-zero: {len(neg_ch_nonzero)/len(neg_ch_data)*100:.2f}%')
            
            stats_x = 0.02 if ch_idx == 6 else 0.98
            stats_ha = 'left' if ch_idx == 6 else 'right'
            ax.text(stats_x, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment=stats_ha,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Channel {ch_idx}: {ch_name}\n(all zeros)', fontsize=22)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {save_path}")
    plt.close()

def visualize_land_cover_distribution(data, save_path='data_visualization/land_cover_distribution.png'):
    """可视化土地覆盖类型分布"""
    print("\n🌍 生成土地覆盖类型分布图...")
    
    if 'positive_samples' not in data or 'negative_samples' not in data:
        print("  ⚠️  数据中不包含正负样本列表，跳过土地覆盖分析")
        return
    
    positive_samples = data['positive_samples']
    negative_samples = data['negative_samples']
    
    # 统计正负样本的土地覆盖类型
    pos_lc = [s.get('land_cover') for s in positive_samples if s.get('land_cover') is not None]
    neg_lc = [s.get('land_cover') for s in negative_samples if s.get('land_cover') is not None]
    
    pos_lc_counter = Counter(pos_lc)
    neg_lc_counter = Counter(neg_lc)
    
    # 土地覆盖类型名称映射
    lc_names = {
        0: "Water", 1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest", 4: "Deciduous Broadleaf Forest",
        5: "Mixed Forests", 6: "Closed Shrublands", 7: "Open Shrublands",
        8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
        11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-up Lands",
        14: "Cropland/Natural Vegetation Mosaics", 15: "Snow and Ice", 16: "Barren"
    }
    
    all_lc_types = sorted(set(list(pos_lc_counter.keys()) + list(neg_lc_counter.keys())))
    
    if len(all_lc_types) == 0:
        print("  ⚠️  没有土地覆盖类型数据，跳过")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 正样本土地覆盖分布（Positive 浅粉）
    pos_labels = [f"LC{lc}\n({lc_names.get(lc, 'Unknown')})" for lc in all_lc_types]
    pos_sizes = [pos_lc_counter.get(lc, 0) for lc in all_lc_types]
    
    ax1.bar(range(len(all_lc_types)), pos_sizes, color='#ff9999', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(all_lc_types)))
    ax1.set_xticklabels([f"LC{lc}" for lc in all_lc_types], rotation=45, ha='right')
    ax1.set_ylabel('Sample count')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, size in enumerate(pos_sizes):
        if size > 0:
            ax1.text(i, size, f'{size}', ha='center', va='bottom', fontsize=11)
    
    # 负样本土地覆盖分布（Negative 天蓝）
    neg_sizes = [neg_lc_counter.get(lc, 0) for lc in all_lc_types]
    
    ax2.bar(range(len(all_lc_types)), neg_sizes, color='#66b3ff', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(all_lc_types)))
    ax2.set_xticklabels([f"LC{lc}" for lc in all_lc_types], rotation=45, ha='right')
    ax2.set_ylabel('Sample count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, size in enumerate(neg_sizes):
        if size > 0:
            ax2.text(i, size, f'{size}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {save_path}")
    plt.close()

def visualize_spatiotemporal_sample(dataset, positive_samples, negative_samples, 
                                   save_path='data_visualization/spatiotemporal_sample.png', data=None):
    """可视化正负样本的时空特征对比"""
    print("\n🎬 生成时空样本对比图...")
    
    # 随机选择正负样本各一个（为保证论文图可复现，这里使用固定随机种子）
    # 若希望重新采样，可手动改动下面的种子值
    import random
    # 固定随机种子，保证 spatiotemporal_sample.png 在多次运行之间可复现
    random.seed(45)
    
    # 对于正样本：按component_id分组，确保选择不同的component
    if len(positive_samples) > 0:
        # 收集所有唯一的component_id
        component_ids = set()
        component_to_samples = {}
        for idx, sample_info in enumerate(positive_samples):
            if 'original_sample' in sample_info:
                orig_sample = sample_info['original_sample']
                if 'metadata' in orig_sample and 'component_id' in orig_sample['metadata']:
                    comp_id = orig_sample['metadata']['component_id']
                    component_ids.add(comp_id)
                    if comp_id not in component_to_samples:
                        component_to_samples[comp_id] = []
                    component_to_samples[comp_id].append(idx)
        
        # 打印component统计信息
        print(f"   发现 {len(component_ids)} 个不同的component，共 {len(positive_samples)} 个正样本")
        if len(component_ids) <= 5:
            for comp_id in component_ids:
                print(f"     Component {comp_id}: {len(component_to_samples[comp_id])} 个样本")
        
        # 如果有多个不同的component，随机选择一个component，然后从该component中随机选择一个样本
        if len(component_ids) > 1:
            selected_comp_id = random.choice(list(component_ids))
            pos_sample_idx = random.choice(component_to_samples[selected_comp_id])
            print(f"   从component {selected_comp_id} 中选择样本（该component有 {len(component_to_samples[selected_comp_id])} 个样本）")
        else:
            # 如果只有一个component，随机选择样本
            pos_sample_idx = random.randint(0, len(positive_samples) - 1)
            if len(component_ids) == 1:
                comp_id = list(component_ids)[0]
                print(f"   所有样本来自同一个component {comp_id}，随机选择样本索引 {pos_sample_idx}")
    else:
        pos_sample_idx = None
    
    # 负样本直接随机选择（负样本没有component_id）
    neg_sample_idx = random.randint(0, len(negative_samples) - 1) if len(negative_samples) > 0 else None
    
    # #region agent log
    import json
    import os
    log_path = r'e:\FireEqual\.cursor\debug.log'
    try:
        log_entry = {
            'sessionId': 'debug-session',
            'runId': 'run1',
            'hypothesisId': 'A',
            'location': 'visualize_dataset.py:428',
            'message': 'Random sample selection for visualization',
            'data': {
                'pos_sample_idx': pos_sample_idx,
                'neg_sample_idx': neg_sample_idx,
                'total_pos_samples': len(positive_samples),
                'total_neg_samples': len(negative_samples),
                'random_seed_used': int(time.time() * 1000) % (2**32)
            },
            'timestamp': int(time.time() * 1000)
        }
        if pos_sample_idx is not None:
            pos_sample_info = positive_samples[pos_sample_idx]
            log_entry['data']['pos_sample_idx_in_dataset'] = pos_sample_info['idx']
            if 'original_sample' in pos_sample_info:
                orig_sample = pos_sample_info['original_sample']
                if 'metadata' in orig_sample and 'component_id' in orig_sample['metadata']:
                    log_entry['data']['pos_component_id'] = orig_sample['metadata']['component_id']
                if 'pixel_date' in orig_sample:
                    log_entry['data']['pos_pixel_date'] = str(orig_sample['pixel_date'])
        if neg_sample_idx is not None:
            neg_sample_info = negative_samples[neg_sample_idx]
            log_entry['data']['neg_sample_idx_in_dataset'] = neg_sample_info['idx']
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        pass
    # #endregion
    
    print(f"   随机选择样本: 正样本索引={pos_sample_idx}, 负样本索引={neg_sample_idx}")
    
    if pos_sample_idx is None and neg_sample_idx is None:
        print("  ⚠️  没有可用的样本，跳过")
        return
    
    # 固定8通道名称
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    # 检查通道数
    sample_idx = (positive_samples[pos_sample_idx]['idx'] if pos_sample_idx is not None else negative_samples[neg_sample_idx]['idx'])
    features, _ = dataset[sample_idx]
    if features.shape[0] != 8:
        print(f"⚠️ 当前数据通道数为 {features.shape[0]}，与期望8不一致，跳过该可视化。")
        return
    num_channels = 8
    
    # 确定子图数量：仅展示8个通道（正负各一列），不再单独展示FRP行
    # 行数：ceil(8/2) = 4行，列数：2列（正负样本各一列）
    n_samples = sum([pos_sample_idx is not None, neg_sample_idx is not None])
    rows = 4  # 8个通道分4行
    cols = 2 * n_samples
    # 适度缩小画布并压缩留白，让子图与文字在同一版面中更大、更紧凑
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows))

    # 根据子图数量自适应放大字体：子图越多，字号越大，保证在整页上的可读性
    n_subplots = rows * cols
    if n_subplots <= 4:
        font_scale = 1.0
    elif n_subplots <= 8:
        font_scale = 1.1
    elif n_subplots <= 16:
        font_scale = 1.2
    else:
        font_scale = 1.3

    title_fs = 20 * font_scale
    axis_label_fs = 18 * font_scale
    tick_fs = 16 * font_scale
    cbar_fs = 16 * font_scale

    # 不再使用总标题，由论文排版统一控制
    # 预先计算每个通道的全局颜色轴范围（正负样本统一）
    vmin_list = []
    vmax_list = []
    
    for ch_idx in range(len(channel_names)):
        channel_values = []
        
        # 收集正样本的值
        if pos_sample_idx is not None:
            pos_idx = positive_samples[pos_sample_idx]['idx']
            features, _ = dataset[pos_idx]
            features_np = features.numpy()
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            channel_values.extend(channel_data.flatten())
        
        # 收集负样本的值
        if neg_sample_idx is not None:
            neg_idx = negative_samples[neg_sample_idx]['idx']
            features, _ = dataset[neg_idx]
            features_np = features.numpy()
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            channel_values.extend(channel_data.flatten())
        
        if len(channel_values) > 0:
            # NDVI固定[0,1]，其他使用数据范围；类别型land_cover直接取范围
            if channel_names[ch_idx] == 'NDVI':
                vmin_list.append(0.0)
                vmax_list.append(1.0)
            else:
                v = np.asarray(channel_values, dtype=float)
                vmin_list.append(np.nanmin(v))
                vmax_list.append(np.nanmax(v))
        else:
            vmin_list.append(0.0)
            vmax_list.append(1.0)
    
    def subplot_ax(r, c):
        # axes 既可能是2D也可能是1D（当rows或cols为1时）
        if rows == 1 and cols == 1:
            return axes
        if rows == 1:
            return axes[c]
        if cols == 1:
            return axes[r]
        return axes[r, c]

    col_offset = 0
    
    # 正样本可视化
    if pos_sample_idx is not None:
        pos_idx = positive_samples[pos_sample_idx]['idx']
        features, target = dataset[pos_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        for ch_idx, ch_name in enumerate(channel_names):
            ax = subplot_ax(ch_idx // 2, (ch_idx % 2) + col_offset)
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            time_with_max = np.argmax(channel_data.sum(axis=(1, 2)))
            
            # 使用统一的颜色轴范围
            im = ax.imshow(channel_data[time_with_max], cmap='hot', interpolation='nearest',
                          vmin=vmin_list[ch_idx], vmax=vmax_list[ch_idx])
            ax.set_title(f'Positive - channel {ch_idx}: {ch_name}\nTime step {time_with_max}',
                         fontsize=title_fs)
            # 第一列第一个子图标注坐标轴，其余子图只保留刻度
            if ch_idx == 0:
                ax.set_xlabel('Longitude (grid)', fontsize=axis_label_fs)
                ax.set_ylabel('Latitude (grid)', fontsize=axis_label_fs)
            ax.tick_params(axis='both', which='both', labelsize=tick_fs)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Feature value', fontsize=axis_label_fs)
            cbar.ax.tick_params(labelsize=cbar_fs)
        
        col_offset += 2
    
    # 负样本可视化
    if neg_sample_idx is not None:
        neg_idx = negative_samples[neg_sample_idx]['idx']
        features, target = dataset[neg_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        for ch_idx, ch_name in enumerate(channel_names):
            ax = subplot_ax(ch_idx // 2, (ch_idx % 2) + col_offset)
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            time_with_max = np.argmax(channel_data.sum(axis=(1, 2)))
            
            # 使用与正样本相同的颜色轴范围
            im = ax.imshow(channel_data[time_with_max], cmap='hot', interpolation='nearest',
                          vmin=vmin_list[ch_idx], vmax=vmax_list[ch_idx])
            ax.set_title(f'Negative - channel {ch_idx}: {ch_name}\nTime step {time_with_max}',
                         fontsize=title_fs)
            if ch_idx == 0 and col_offset > 0:
                ax.set_xlabel('Longitude (grid)', fontsize=axis_label_fs)
                ax.set_ylabel('Latitude (grid)', fontsize=axis_label_fs)
            ax.tick_params(axis='both', which='both', labelsize=tick_fs)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Feature value', fontsize=axis_label_fs)
            cbar.ax.tick_params(labelsize=cbar_fs)
    
    # # 添加FRP可视化（第5行，索引4）
    # # 正样本FRP
    # if pos_sample_idx is not None:
    #     pos_sample_info = positive_samples[pos_sample_idx]
    #     pos_idx = pos_sample_info['idx']
    #     pos_features, _ = dataset[pos_idx]
    #     pos_features_np = pos_features.numpy()  # [C, T, H, W]
    #     grid_size = pos_features_np.shape[2]  # 假设是正方形
        
    #     ax_frp_pos = subplot_ax(4, 0)  # 第5行，第1列
        
    #     # 尝试从原始样本中获取FRP信息
    #     frp_data = None
    #     frp_value = None
    #     frp_text = "No FRP data"
        
    #     # #region agent log
    #     import json
    #     import os
    #     log_path = r'e:\FireEqual\.cursor\debug.log'
    #     try:
    #         log_entry = {
    #             'sessionId': 'debug-session',
    #             'runId': 'run1',
    #             'hypothesisId': 'A',
    #             'location': 'visualize_dataset.py:502',
    #             'message': 'Checking positive sample structure for FRP data',
    #             'data': {
    #                 'pos_sample_keys': list(pos_sample_info.keys()),
    #                 'has_original_sample': 'original_sample' in pos_sample_info
    #             },
    #             'timestamp': int(__import__('time').time() * 1000)
    #         }
    #         if 'original_sample' in pos_sample_info:
    #             orig_sample = pos_sample_info['original_sample']
    #             log_entry['data']['original_sample_keys'] = list(orig_sample.keys())
    #             if 'metadata' in orig_sample:
    #                 log_entry['data']['metadata_keys'] = list(orig_sample['metadata'].keys())
    #                 log_entry['data']['component_id'] = orig_sample['metadata'].get('component_id', None)
    #             if 'pixel_date' in orig_sample:
    #                 log_entry['data']['pixel_date'] = str(orig_sample['pixel_date'])
    #             if 'pixel_lat' in orig_sample:
    #                 log_entry['data']['pixel_lat'] = float(orig_sample['pixel_lat'])
    #             if 'pixel_lon' in orig_sample:
    #                 log_entry['data']['pixel_lon'] = float(orig_sample['pixel_lon'])
    #         with open(log_path, 'a', encoding='utf-8') as f:
    #             f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    #     except Exception as e:
    #         pass
    #     # #endregion
        
    #     if 'original_sample' in pos_sample_info:
    #         orig_sample = pos_sample_info['original_sample']
    #         # 像素级二分类数据集：从events数据中查找FRP
    #         # 需要pixel_date, pixel_lat, pixel_lon来查找events中的FRP值
    #         if 'pixel_date' in orig_sample and 'pixel_lat' in orig_sample and 'pixel_lon' in orig_sample:
    #             pixel_date = orig_sample['pixel_date']
    #             pixel_lat = orig_sample['pixel_lat']
    #             pixel_lon = orig_sample['pixel_lon']
                
    #             # 首先尝试从component_id查找FRP（从components数据）
    #             if 'metadata' in orig_sample and 'component_id' in orig_sample['metadata']:
    #                 component_id = orig_sample['metadata']['component_id']
    #                 if component_id is not None:
    #                     # 尝试从保存的数据中获取components数据
    #                     if data and 'preprocessed_data' in data and 'components' in data['preprocessed_data']:
    #                         components = data['preprocessed_data']['components']
    #                         # 尝试通过索引查找
    #                         try:
    #                             # 首先尝试通过索引查找
    #                             if component_id in components.index:
    #                                 component_row = components.loc[component_id]
    #                                 if 'maxFRP_sum' in component_row:
    #                                     frp_value = float(component_row['maxFRP_sum'])
    #                                     frp_text = f"Component FRP: {frp_value:.2f} MW"
    #                                 elif 'maxFRP_mean' in component_row:
    #                                     frp_value = float(component_row['maxFRP_mean'])
    #                                     frp_text = f"Component mean FRP: {frp_value:.2f} MW"
    #                                 else:
    #                                     frp_text = f"Component ID: {component_id}\n(no FRP field)"
    #                             # 如果索引不匹配，尝试通过cp列查找
    #                             elif 'cp' in components.columns:
    #                                 component_row = components[components['cp'] == component_id]
    #                                 if len(component_row) > 0:
    #                                     if 'maxFRP_sum' in component_row.columns:
    #                                         frp_value = float(component_row['maxFRP_sum'].iloc[0])
    #                                         frp_text = f"Component FRP: {frp_value:.2f} MW"
    #                                     elif 'maxFRP_mean' in component_row.columns:
    #                                         frp_value = float(component_row['maxFRP_mean'].iloc[0])
    #                                         frp_text = f"Component mean FRP: {frp_value:.2f} MW"
    #                                     else:
    #                                         frp_text = f"Component ID: {component_id}\n(no FRP field)"
    #                                 else:
    #                                     frp_text = f"Component ID: {component_id}\n(not found in components)"
    #                             else:
    #                                 # 尝试将component_id作为整数索引
    #                                 try:
    #                                     component_row = components.iloc[component_id] if component_id < len(components) else None
    #                                     if component_row is not None:
    #                                         if 'maxFRP_sum' in component_row:
    #                                             frp_value = float(component_row['maxFRP_sum'])
    #                                             frp_text = f"Component FRP: {frp_value:.2f} MW"
    #                                         elif 'maxFRP_mean' in component_row:
    #                                             frp_value = float(component_row['maxFRP_mean'])
    #                                             frp_text = f"Component mean FRP: {frp_value:.2f} MW"
    #                                 except:
    #                                     frp_text = f"Component ID: {component_id}\n(lookup failed)"
    #                         except Exception as e:
    #                             frp_text = f"Component ID: {component_id}\n(error: {str(e)[:50]})"
    #                     else:
    #                         # 尝试从events数据中查找
    #                         if 'preprocessed_data' in data and 'events' in data['preprocessed_data']:
    #                             import pandas as pd
    #                             events = data['preprocessed_data']['events']
    #                             # 查找匹配的事件（同一天，相近位置）
    #                             tolerance = 0.01  # 约1km的容差
    #                             date_match = pd.to_datetime(events['dtime']).dt.date == pd.to_datetime(pixel_date).date()
    #                             lat_match = (events['lat'] >= pixel_lat - tolerance) & (events['lat'] <= pixel_lat + tolerance)
    #                             lon_match = (events['lon'] >= pixel_lon - tolerance) & (events['lon'] <= pixel_lon + tolerance)
    #                             matching_events = events[date_match & lat_match & lon_match]
                                
    #                             if len(matching_events) > 0:
    #                                 # 获取最大FRP值（如果有多个匹配事件）
    #                                 if 'maxFRP' in matching_events.columns:
    #                                     frp_value = float(matching_events['maxFRP'].max())
    #                                     frp_text = f"Event FRP: {frp_value:.2f} MW"
    #                                 else:
    #                                     frp_text = f"Matched event but no FRP field\nComponent ID: {component_id}"
    #                             else:
    #                                 frp_text = f"Component ID: {component_id}\nNo matching event"
    #                         else:
    #                             frp_text = f"Component ID: {component_id}\n(load data for FRP)"
    #             # 如果没有component_id，尝试直接从events数据中查找
    #             elif 'preprocessed_data' in data and 'events' in data['preprocessed_data']:
    #                 import pandas as pd
    #                 events = data['preprocessed_data']['events']
    #                 # 查找匹配的事件（同一天，相近位置）
    #                 tolerance = 0.01  # 约1km的容差
    #                 date_match = pd.to_datetime(events['dtime']).dt.date == pd.to_datetime(pixel_date).date()
    #                 lat_match = (events['lat'] >= pixel_lat - tolerance) & (events['lat'] <= pixel_lat + tolerance)
    #                 lon_match = (events['lon'] >= pixel_lon - tolerance) & (events['lon'] <= pixel_lon + tolerance)
    #                 matching_events = events[date_match & lat_match & lon_match]
                    
    #                 if len(matching_events) > 0:
    #                     # 获取最大FRP值（如果有多个匹配事件）
    #                     if 'maxFRP' in matching_events.columns:
    #                         frp_value = float(matching_events['maxFRP'].max())
    #                         frp_text = f"Event FRP: {frp_value:.2f} MW"
    #                     else:
    #                         frp_text = f"Matched event but no FRP field\nDate: {pixel_date}"
    #                 else:
    #                     frp_text = f"No matching event\nDate: {pixel_date}\nLocation: ({pixel_lat:.4f}, {pixel_lon:.4f})"
    #         # 尝试从targets中获取FRP（如果存在）
    #         elif 'targets' in orig_sample and 'total_frp' in orig_sample['targets']:
    #             frp_value = orig_sample['targets']['total_frp']
    #             frp_text = f"Total FRP: {frp_value:.2f} MW"
        
    #     # #region agent log
    #     try:
    #         log_entry_frp = {
    #             'sessionId': 'debug-session',
    #             'runId': 'run1',
    #             'hypothesisId': 'B',
    #             'location': 'visualize_dataset.py:702',
    #             'message': 'FRP value for positive sample',
    #             'data': {
    #                 'pos_sample_idx': pos_sample_idx,
    #                 'frp_value': float(frp_value) if frp_value is not None else None,
    #                 'frp_text': frp_text,
    #                 'has_frp_data': frp_value is not None and frp_value > 0
    #             },
    #             'timestamp': int(__import__('time').time() * 1000)
    #         }
    #         if 'original_sample' in pos_sample_info:
    #             orig_sample = pos_sample_info['original_sample']
    #             if 'metadata' in orig_sample and 'component_id' in orig_sample['metadata']:
    #                 log_entry_frp['data']['component_id'] = orig_sample['metadata']['component_id']
    #         with open(log_path, 'a', encoding='utf-8') as f:
    #             f.write(json.dumps(log_entry_frp, ensure_ascii=False) + '\n')
    #     except Exception as e:
    #         pass
    #     # #endregion
        
    #     # 创建FRP热力图（如果有FRP数据，显示在中心位置；否则显示0）
    #     if frp_value is not None and frp_value > 0:
    #         # 创建一个简单的热力图，在中心位置显示FRP强度（使用实际FRP值）
    #         frp_data = np.zeros((grid_size, grid_size))
    #         center = grid_size // 2
    #         # 在中心区域设置实际FRP值
    #         frp_data[center-2:center+3, center-2:center+3] = frp_value
    #         # 设置colorbar的范围为0到FRP值（或稍大一点以便显示）
    #         vmax_frp = frp_value * 1.1 if frp_value > 0 else 1.0
    #     else:
    #         # 无FRP数据，显示全零
    #         frp_data = np.zeros((grid_size, grid_size))
    #         vmax_frp = 1.0  # 默认最大值
        
    #     im_frp = ax_frp_pos.imshow(frp_data, cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax_frp)
    #     ax_frp_pos.set_title(f'Positive - FRP (fire radiative power)\n{frp_text}', fontsize=12)
    #     ax_frp_pos.set_xlabel('Longitude (grid)')
    #     ax_frp_pos.set_ylabel('Latitude (grid)')
    #     plt.colorbar(im_frp, ax=ax_frp_pos, label='FRP (MW)')
    
    # # 负样本FRP（应该为0或无火灾）
    # if neg_sample_idx is not None:
    #     neg_sample_info = negative_samples[neg_sample_idx]
    #     neg_idx = neg_sample_info['idx']
    #     neg_features, _ = dataset[neg_idx]
    #     neg_features_np = neg_features.numpy()  # [C, T, H, W]
    #     grid_size_neg = neg_features_np.shape[2]
        
    #     ax_frp_neg = subplot_ax(4, cols - 1)  # 第5行，最后一列
        
    #     # 负样本没有火灾，FRP应该为0
    #     frp_data_neg = np.zeros((grid_size_neg, grid_size_neg))
    #     # 使用一个合理的最大值范围（例如1000 MW）以便colorbar显示有意义
    #     vmax_frp_neg = 1000.0  # 默认最大值，用于显示colorbar
        
    #     im_frp_neg = ax_frp_neg.imshow(frp_data_neg, cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax_frp_neg)
    #     ax_frp_neg.set_title('Negative - FRP (fire radiative power)\nNo fire (FRP = 0)', fontsize=12)
    #     ax_frp_neg.set_xlabel('Longitude (grid)')
    #     ax_frp_neg.set_ylabel('Latitude (grid)')
    #     plt.colorbar(im_frp_neg, ax=ax_frp_neg, label='FRP (MW)')
    
    plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {save_path}")
    plt.close()

def visualize_time_evolution(dataset, positive_samples, negative_samples,
                            channel_idx=1, save_path='data_visualization/time_evolution.png'):
    """可视化正负样本的时间演化对比"""
    print(f"\n⏱️  生成时间演化对比图 (通道 {channel_idx})...")
    
    # 固定8通道
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    sample_idx = (positive_samples[0]['idx'] if len(positive_samples) > 0 else (negative_samples[0]['idx'] if len(negative_samples) > 0 else 0))
    features, _ = dataset[sample_idx]
    if features.shape[0] != 8:
        print(f"⚠️ 当前数据通道数为 {features.shape[0]}，与期望8不一致，跳过时间演化可视化。")
        return
    
    # 选择正负样本各一个
    pos_sample_idx = 0 if len(positive_samples) > 0 else None
    neg_sample_idx = 0 if len(negative_samples) > 0 else None
    
    if pos_sample_idx is None and neg_sample_idx is None:
        print("  ⚠️  没有可用的样本，跳过")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 正样本时间演化
    if pos_sample_idx is not None:
        pos_idx = positive_samples[pos_sample_idx]['idx']
        features, target = dataset[pos_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        channel_data = features_np[channel_idx, :, :, :]  # [T, H, W]
        
        time_stats = {
            'total': [channel_data[t, :, :].sum() for t in range(channel_data.shape[0])],
            'mean': [channel_data[t, :, :].mean() for t in range(channel_data.shape[0])],
            'nonzero_count': [(channel_data[t, :, :] != 0).sum() for t in range(channel_data.shape[0])]
        }
        
        # 总特征值随时间变化
        axes[0, 0].plot(range(len(time_stats['total'])), time_stats['total'], 
                       marker='o', linewidth=2, markersize=6, label='Positive', color='#ff9999')
        # axes[0, 0].set_title('Total feature value over time')
        axes[0, 0].set_xlabel('Time step (days)')
        axes[0, 0].set_ylabel('Total feature value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 平均特征值随时间变化
        axes[0, 1].plot(range(len(time_stats['mean'])), time_stats['mean'],
                       marker='s', linewidth=2, markersize=6, label='Positive', color='#ff9999')
        # axes[0, 1].set_title('Mean feature value over time')
        axes[0, 1].set_xlabel('Time step (days)')
        axes[0, 1].set_ylabel('Mean feature value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 负样本时间演化
    if neg_sample_idx is not None:
        neg_idx = negative_samples[neg_sample_idx]['idx']
        features, target = dataset[neg_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        channel_data = features_np[channel_idx, :, :, :]  # [T, H, W]
        
        time_stats = {
            'total': [channel_data[t, :, :].sum() for t in range(channel_data.shape[0])],
            'mean': [channel_data[t, :, :].mean() for t in range(channel_data.shape[0])],
            'nonzero_count': [(channel_data[t, :, :] != 0).sum() for t in range(channel_data.shape[0])]
        }
        
        # 总特征值随时间变化（叠加）
        if pos_sample_idx is not None:
            axes[0, 0].plot(range(len(time_stats['total'])), time_stats['total'], 
                           marker='o', linewidth=2, markersize=6, label='Negative', color='#66b3ff', linestyle='--')
            axes[0, 0].legend()
        else:
            axes[0, 0].plot(range(len(time_stats['total'])), time_stats['total'], 
                           marker='o', linewidth=2, markersize=6, label='Negative', color='#66b3ff')
            axes[0, 0].set_title('Total feature value over time')
            axes[0, 0].set_xlabel('Time step (days)')
            axes[0, 0].set_ylabel('Total feature value')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 平均特征值随时间变化（叠加）
        if pos_sample_idx is not None:
            axes[0, 1].plot(range(len(time_stats['mean'])), time_stats['mean'],
                           marker='s', linewidth=2, markersize=6, label='Negative', color='#66b3ff', linestyle='--')
            axes[0, 1].legend()
        else:
            axes[0, 1].plot(range(len(time_stats['mean'])), time_stats['mean'],
                           marker='s', linewidth=2, markersize=6, label='Negative', color='#66b3ff')
            axes[0, 1].set_title('Mean feature value over time')
            axes[0, 1].set_xlabel('Time step (days)')
            axes[0, 1].set_ylabel('Mean feature value')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
    
    # 非零网格数量对比
    if pos_sample_idx is not None and neg_sample_idx is not None:
        pos_idx = positive_samples[pos_sample_idx]['idx']
        neg_idx = negative_samples[neg_sample_idx]['idx']
        
        pos_features, _ = dataset[pos_idx]
        neg_features, _ = dataset[neg_idx]
        
        pos_channel_data = pos_features.numpy()[channel_idx, :, :, :]
        neg_channel_data = neg_features.numpy()[channel_idx, :, :, :]
        
        pos_nonzero = [(pos_channel_data[t, :, :] != 0).sum() for t in range(pos_channel_data.shape[0])]
        neg_nonzero = [(neg_channel_data[t, :, :] != 0).sum() for t in range(neg_channel_data.shape[0])]
        
        x = range(len(pos_nonzero))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x], pos_nonzero, width, 
                      label='Positive', color='#ff9999', alpha=0.7, edgecolor='black')
        axes[1, 0].bar([i + width/2 for i in x], neg_nonzero, width,
                      label='Negative', color='#66b3ff', alpha=0.7, edgecolor='black')
        # axes[1, 0].set_title('Non-zero grid count over time')
        axes[1, 0].set_xlabel('Time step (days)')
        axes[1, 0].set_ylabel('Non-zero grid count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 时间切片热力图对比
    if pos_sample_idx is not None:
        pos_idx = positive_samples[pos_sample_idx]['idx']
        features, _ = dataset[pos_idx]
        features_np = features.numpy()
        channel_data = features_np[channel_idx, :, :, :]
        
        nonzero_times = [t for t in range(channel_data.shape[0]) 
                        if (channel_data[t, :, :] != 0).sum() > 0]
        if len(nonzero_times) > 0:
            top_time = sorted(nonzero_times, 
                            key=lambda t: (channel_data[t, :, :] != 0).sum(), 
                            reverse=True)[0]
            
            im = axes[1, 1].imshow(channel_data[top_time], cmap='hot', 
                                 interpolation='nearest', aspect='auto')
            axes[1, 1].set_title(f'Positive - time step {top_time}')
            plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {save_path}")
    plt.close()

def _load_full_period_dataset(max_samples_per_shard: Optional[int] = 2000):
    """加载 2002-2020 全部年份的数据（内存安全版）。

    优先使用合并文件 processed_firetracks_pixel_binary_2002-2020.pth；
    若不存在，则从 7 个分片中**按年份均匀抽样**，避免一次性把所有样本都放进内存。

    参数
    ----
    max_samples_per_shard : int
        每个分片最多抽取多少样本参与可视化。比如 2000 则最多约 7*2000 ≈ 1.4 万样本。
        这样既覆盖 2002-2020 所有年份，又不会导致内存爆炸。
    """
    merged_path = os.path.join(project_root, 'dataset', 'processed_firetracks_pixel_binary_2002-2020.pth')
    if os.path.exists(merged_path):
        print("📊 检测到合并文件 2002-2020，使用该文件进行可视化。")
        return load_dataset(merged_path)

    print("📊 未找到合并文件，改为从 2002-2020 七个分片中按年份均匀抽样进行可视化。")
    shard_files = [
        'dataset/processed_firetracks_pixel_binary_2002-2004.pth',
        'dataset/processed_firetracks_pixel_binary_2005-2007.pth',
        'dataset/processed_firetracks_pixel_binary_2008-2009.pth',
        'dataset/processed_firetracks_pixel_binary_2010-2013.pth',
        'dataset/processed_firetracks_pixel_binary_2014-2015.pth',
        'dataset/processed_firetracks_pixel_binary_2016-2018.pth',
        'dataset/processed_firetracks_pixel_binary_2019-2020.pth',
    ]

    all_spatiotemporal = []
    base_data = None
    for idx, rel_path in enumerate(shard_files):
        full_path = os.path.join(project_root, rel_path)
        if not os.path.exists(full_path):
            print(f"  ⚠️ 分片不存在，跳过: {rel_path}")
            continue
        print(f"  🔄 加载分片 {idx+1}/7: {rel_path}")
        if base_data is None:
            # 第一份用 load_dataset，这样可以自动构建 FireTracksDataset 并加载辅助数据
            base_data = load_dataset(full_path)
            st_samples = base_data.get('spatiotemporal_samples', [])
        else:
            shard_data = torch.load(full_path, weights_only=False)
            st_samples = shard_data.get('spatiotemporal_samples', [])
        if not st_samples:
            continue

        # 为了避免内存溢出，每个分片只抽取最多 max_samples_per_shard 条样本；
        # 若 max_samples_per_shard 为 None，则不做抽样，使用全部样本（可能占用较多内存）。
        if max_samples_per_shard is not None and len(st_samples) > max_samples_per_shard:
            # 简单下采样：均匀间隔抽样而不是只取前几条
            step = len(st_samples) / max_samples_per_shard
            sampled_indices = [int(i * step) for i in range(max_samples_per_shard)]
            sampled = [st_samples[i] for i in sampled_indices]
            print(f"    - 分片样本数 {len(st_samples):,}，下采样到 {len(sampled):,} 条参与可视化。")
        else:
            sampled = st_samples
            print(f"    - 分片样本数 {len(st_samples):,}，全部用于可视化。")

        all_spatiotemporal.extend(sampled)

    if base_data is None or not all_spatiotemporal:
        raise RuntimeError("未能成功加载任何分片，请检查 dataset 下的 .pth 文件是否存在。")

    print(f"  ✅ 共收集样本数: {len(all_spatiotemporal):,}（2002-2020 全部年份的抽样集合）")

    # 使用所有年份的样本重新构建一个大的 FireTracksDataset
    full_dataset = FireTracksDataset(
        all_spatiotemporal,
        target_type='binary_classification'
    )
    base_data['spatiotemporal_samples'] = all_spatiotemporal
    base_data['dataset'] = full_dataset
    # dataloader 在可视化中用不到，这里不强制重建
    return base_data


USE_FULL_DATA: bool = False  # 若为 True，则不抽样、使用全部 2002-2020 数据（需较大内存）


def main():
    """主函数"""
    print("="*60)
    print("FireTracks像素级二分类数据集可视化工具")
    print("="*60)
    
    # 确保输出目录存在
    output_dir = 'data_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 2002-2020 全部数据：
    # - 若 USE_FULL_DATA=True，则不抽样（max_samples_per_shard=None），可能占用较大内存；
    # - 否则每个分片最多抽样 2000 条，兼顾全时期覆盖与内存安全。
    max_per_shard = None if USE_FULL_DATA else 2000
    data = _load_full_period_dataset(max_samples_per_shard=max_per_shard)
    dataset = data['dataset']
    
    # 获取原始样本数据（用于访问FRP等信息）
    spatiotemporal_samples = data.get('spatiotemporal_samples', None)
    
    # 分析正负样本
    positive_samples, negative_samples = analyze_positive_negative_samples(dataset, spatiotemporal_samples)
    
    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    visualize_class_distribution(dataset)
    visualize_feature_distribution(dataset, positive_samples, negative_samples)
    visualize_land_cover_distribution(data)
    visualize_spatiotemporal_sample(dataset, positive_samples, negative_samples, data=data)
    visualize_time_evolution(dataset, positive_samples, negative_samples, channel_idx=1)
    
    print("\n" + "="*60)
    print("✅ 可视化完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - data_visualization/class_distribution.png: 正负样本分布图")
    print("  - data_visualization/feature_distribution.png: 正负样本特征分布对比图")
    print("  - data_visualization/land_cover_distribution.png: 土地覆盖类型分布图")
    print("  - data_visualization/spatiotemporal_sample.png: 正负样本时空特征对比图")
    print("  - data_visualization/time_evolution.png: 正负样本时间演化对比图")
    print("\n分析要点:")
    print("  1. ✅ 正负样本分布: 检查类别平衡性")
    print("  2. ✅ 特征分布对比: 正负样本特征差异")
    print("  3. ✅ 土地覆盖分布: 正负样本土地覆盖类型匹配度")
    print("  4. ✅ 时空特征对比: 正负样本的空间和时间模式差异")
    print("  5. ✅ 时间演化对比: 正负样本的时间序列特征差异")

if __name__ == '__main__':
    main()
