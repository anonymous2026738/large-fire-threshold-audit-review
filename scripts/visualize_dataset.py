"""FireTracks
for
"""
import os
from typing import Optional

#  OpenMP load(libomp.dll vs libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import sys
from datetime import datetime

# : Times New Roman,
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 22      # 
plt.rcParams['axes.labelsize'] = 20      # 
plt.rcParams['xtick.labelsize'] = 16     # 
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16     # 
plt.rcParams['figure.titlesize'] = 16    # ()

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
# data_visualization,
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
    """load"""
    print("📊 load...")
    data = torch.load(filepath, weights_only=False)
    
    if 'spatiotemporal_samples' in data:
        print("   ")
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
        
        # loadeventscomponentsforFRP
        try:
            import os
            import pandas as pd
            data_dir = 'dataset/firetracks_data'
            if os.path.exists(data_dir):
                # loadevents(loadfor)
                events_file = os.path.join(data_dir, 'v.h5')
                if os.path.exists(events_file):
                    print("   loadeventsforFRP...")
                    # loadmaxFRP
                    try:
                        events = pd.read_hdf(events_file, columns=['lat', 'lon', 'dtime', 'maxFRP'], start=0, stop=1000000)
                        events['dtime'] = pd.to_datetime(events['dtime'])
                        if 'preprocessed_data' not in data:
                            data['preprocessed_data'] = {}
                        data['preprocessed_data']['events'] = events
                        print(f"   ✅ load {len(events):,} events")
                    except Exception as e:
                        print(f"   ⚠️  loadevents: {e}")
                
                # loadcomponents
                components_file = os.path.join(data_dir, 'cp.h5')
                if os.path.exists(components_file):
                    print("   loadcomponentsforFRP...")
                    try:
                        # FRP
                        components = pd.read_hdf(components_file, start=0, stop=100000)
                        # 
                        frp_cols = [col for col in ['maxFRP_sum', 'maxFRP_mean', 'cp'] if col in components.columns]
                        if frp_cols:
                            components = components[frp_cols]
                        if 'preprocessed_data' not in data:
                            data['preprocessed_data'] = {}
                        data['preprocessed_data']['components'] = components
                        print(f"   ✅ load {len(components):,} components")
                        if 'cp' in components.columns:
                            print(f"   ✅ componentscp,for")
                    except Exception as e:
                        print(f"   ⚠️  loadcomponents: {e}")
        except Exception as e:
            print(f"   ⚠️  load: {e}")
    elif 'dataset' in data:
        print("  ,")
        dataset = data['dataset']
        dataloader = data.get('dataloader', None)
    else:
        raise ValueError("")
    
    return data

def analyze_positive_negative_samples(dataset, spatiotemporal_samples=None):
    """"""
    print("\n" + "="*60)
    print("===  ===")
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
        
        #,metadataFRP
        if spatiotemporal_samples is not None and i < len(spatiotemporal_samples):
            sample_info['original_sample'] = spatiotemporal_samples[i]
        
        if target.item() == 1:
            positive_samples.append(sample_info)
        else:
            negative_samples.append(sample_info)
    
    print(f"✅ : {len(positive_samples):,}")
    print(f"✅ : {len(negative_samples):,}")
    print(f"✅ : 1:{len(negative_samples)/len(positive_samples):.2f}" if len(positive_samples) > 0 else "")
    
    # 
    if len(positive_samples) > 0 and len(negative_samples) > 0:
        pos_features = np.stack([s['features'] for s in positive_samples], axis=0)
        neg_features = np.stack([s['features'] for s in negative_samples], axis=0)
        
        print(f"\n():")
        # 8
        num_channels = pos_features.shape[1]
        channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
        if num_channels != 8:
            print(f"⚠️  {num_channels},8.8.")
            return positive_samples, negative_samples
        
        for ch_idx, ch_name in enumerate(channel_names):
            pos_ch = pos_features[:, ch_idx, :, :, :]
            neg_ch = neg_features[:, ch_idx, :, :, :]
            
            print(f"\n {ch_idx}: {ch_name}")
            print(f"   - : {pos_ch.mean():.4f}, : {(pos_ch != 0).sum()/pos_ch.size*100:.2f}%")
            print(f"   - : {neg_ch.mean():.4f}, : {(neg_ch != 0).sum()/neg_ch.size*100:.2f}%")
    
    return positive_samples, negative_samples

def visualize_class_distribution(dataset, save_path='data_visualization/class_distribution.png'):
    """"""
    print("\n📊 ...")
    
    all_targets = [dataset[i][1].item() for i in range(len(dataset))]
    counter = Counter(all_targets)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # (:Negative,Positive )
    labels = ['Negative (0)', 'Positive (1)']
    sizes = [counter.get(0, 0), counter.get(1, 0)]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0.05)

    #  autopct,「」( 19,000 / 37,970)
    def make_autopct():
        #  f-string 
        counts = [37970, 19000]  #  labels / sizes : [Negative, Positive]
        state = {'idx': 0}

        def autopct(pct):
            i = state['idx']
            state['idx'] += 1
            #,
            return f'{pct:.1f}% ({counts[i]:,})'

        return autopct
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct=make_autopct(), shadow=True, startangle=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ : {save_path}")
    plt.close()

def visualize_feature_distribution(dataset, positive_samples, negative_samples, save_path='data_visualization/feature_distribution.png'):
    """"""
    print("\n📈 ...")
    
    # (100)
    pos_features_list = []
    neg_features_list = []
    
    for i in range(len(positive_samples)):
        pos_features_list.append(positive_samples[i]['features'])
    
    for i in range(len(negative_samples)):
        neg_features_list.append(negative_samples[i]['features'])
    
    pos_features = np.stack(pos_features_list, axis=0) if pos_features_list else None
    neg_features = np.stack(neg_features_list, axis=0) if neg_features_list else None
    
    # 8
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    # 8,
    if (pos_features is not None and pos_features.shape[1] != 8) or (neg_features is not None and neg_features.shape[1] != 8):
        print(f"⚠️ 8,.")
        fig, _ = plt.subplots(1, 1, figsize=(6, 4))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # :248
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
            #,bin
            all_values = []
            if len(pos_ch_nonzero) > 0:
                all_values.extend(pos_ch_nonzero)
            if len(neg_ch_nonzero) > 0:
                all_values.extend(neg_ch_nonzero)
            
            if len(all_values) > 0:
                # population,
                if ch_name == 'population':
                    # 5%95%,
                    # (30000)
                    all_values_array = np.array(all_values)
                    p5 = np.percentile(all_values_array, 5)
                    p95 = np.percentile(all_values_array, 95)
                    
                    # (),
                    if p95 - p5 < 0.1:
                        # (IQR)
                        median = np.median(all_values_array)
                        q25 = np.percentile(all_values_array, 25)
                        q75 = np.percentile(all_values_array, 75)
                        iqr = q75 - q25
                        #  IQR,
                        data_min = max(0, median - 2 * iqr)  # population0
                        data_max = median + 2 * iqr
                        #,
                        if data_max - data_min < 0.01:
                            data_min = np.min(all_values_array)
                            data_max = np.max(all_values_array)
                    else:
                        # 5%95%
                        data_min = max(0, p5)  # population0
                        data_max = p95
                    
                    # 0,
                    if data_max == data_min:
                        data_max = data_min + 1.0
                    
                    # (5%)
                    range_margin = (data_max - data_min) * 0.05
                    data_min = max(0, data_min - range_margin)
                    data_max = data_max + range_margin
                else:
                    # 
                    data_min = np.min(all_values)
                    data_max = np.max(all_values)
                    # 0,
                    if data_max == data_min:
                        data_max = data_min + 1.0
                
                # bin
                bins = 50
                bin_range = (data_min, data_max)
                
                #,(:Negative,Positive )
                if len(pos_ch_nonzero) > 0:
                    ax.hist(pos_ch_nonzero, bins=bins, range=bin_range, alpha=0.6, 
                           label='Positive', color='#ff9999', edgecolor='black', density=True)
                if len(neg_ch_nonzero) > 0:
                    ax.hist(neg_ch_nonzero, bins=bins, range=bin_range, alpha=0.6, 
                           label='Negative', color='#66b3ff', edgecolor='black', density=True)
            else:
                #,
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
            
            ax.set_title(f'Channel {ch_idx}: {ch_name}', fontsize=22)
            ax.set_xlabel('Feature value')
            ax.set_ylabel('Density')
            #,
            if ch_idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 
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
    print(f"✅ : {save_path}")
    plt.close()

def visualize_land_cover_distribution(data, save_path='data_visualization/land_cover_distribution.png'):
    """"""
    print("\n🌍 ...")
    
    if 'positive_samples' not in data or 'negative_samples' not in data:
        print("  ⚠️ ,")
        return
    
    positive_samples = data['positive_samples']
    negative_samples = data['negative_samples']
    
    # 
    pos_lc = [s.get('land_cover') for s in positive_samples if s.get('land_cover') is not None]
    neg_lc = [s.get('land_cover') for s in negative_samples if s.get('land_cover') is not None]
    
    pos_lc_counter = Counter(pos_lc)
    neg_lc_counter = Counter(neg_lc)
    
    # 
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
        print("  ⚠️ ,")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # (Positive )
    pos_labels = [f"LC{lc}\n({lc_names.get(lc, 'Unknown')})" for lc in all_lc_types]
    pos_sizes = [pos_lc_counter.get(lc, 0) for lc in all_lc_types]
    
    ax1.bar(range(len(all_lc_types)), pos_sizes, color='#ff9999', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(all_lc_types)))
    ax1.set_xticklabels([f"LC{lc}" for lc in all_lc_types], rotation=45, ha='right')
    ax1.set_ylabel('Sample count')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 
    for i, size in enumerate(pos_sizes):
        if size > 0:
            ax1.text(i, size, f'{size}', ha='center', va='bottom', fontsize=11)
    
    # (Negative )
    neg_sizes = [neg_lc_counter.get(lc, 0) for lc in all_lc_types]
    
    ax2.bar(range(len(all_lc_types)), neg_sizes, color='#66b3ff', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(all_lc_types)))
    ax2.set_xticklabels([f"LC{lc}" for lc in all_lc_types], rotation=45, ha='right')
    ax2.set_ylabel('Sample count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 
    for i, size in enumerate(neg_sizes):
        if size > 0:
            ax2.text(i, size, f'{size}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ : {save_path}")
    plt.close()

def visualize_spatiotemporal_sample(dataset, positive_samples, negative_samples, 
                                   save_path='data_visualization/spatiotemporal_sample.png', data=None):
    """"""
    print("\n🎬 ...")
    
    # 
    #,
    import random
    #, spatiotemporal_sample.png 
    random.seed(45)
    
    # :component_idgrouping,component
    if len(positive_samples) > 0:
        # component_id
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
        
        # component
        print(f"    {len(component_ids)} component, {len(positive_samples)} ")
        if len(component_ids) <= 5:
            for comp_id in component_ids:
                print(f"     Component {comp_id}: {len(component_to_samples[comp_id])} ")
        
        # component,component,component
        if len(component_ids) > 1:
            selected_comp_id = random.choice(list(component_ids))
            pos_sample_idx = random.choice(component_to_samples[selected_comp_id])
            print(f"   component {selected_comp_id} (component {len(component_to_samples[selected_comp_id])} )")
        else:
            # component,
            pos_sample_idx = random.randint(0, len(positive_samples) - 1)
            if len(component_ids) == 1:
                comp_id = list(component_ids)[0]
                print(f"   component {comp_id}, {pos_sample_idx}")
    else:
        pos_sample_idx = None
    
    # (component_id)
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
    
    print(f"   : ={pos_sample_idx}, ={neg_sample_idx}")
    
    if pos_sample_idx is None and neg_sample_idx is None:
        print("  ⚠️ ,")
        return
    
    # 8
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    # 
    sample_idx = (positive_samples[pos_sample_idx]['idx'] if pos_sample_idx is not None else negative_samples[neg_sample_idx]['idx'])
    features, _ = dataset[sample_idx]
    if features.shape[0] != 8:
        print(f"⚠️  {features.shape[0]},8,.")
        return
    num_channels = 8
    
    # :8(),FRP
    # :ceil(8/2) = 4,:2()
    n_samples = sum([pos_sample_idx is not None, neg_sample_idx is not None])
    rows = 4  # 84
    cols = 2 * n_samples
    #, 
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows))

    # :,
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

    #,
    # ()
    vmin_list = []
    vmax_list = []
    
    for ch_idx in range(len(channel_names)):
        channel_values = []
        
        # 
        if pos_sample_idx is not None:
            pos_idx = positive_samples[pos_sample_idx]['idx']
            features, _ = dataset[pos_idx]
            features_np = features.numpy()
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            channel_values.extend(channel_data.flatten())
        
        # 
        if neg_sample_idx is not None:
            neg_idx = negative_samples[neg_sample_idx]['idx']
            features, _ = dataset[neg_idx]
            features_np = features.numpy()
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            channel_values.extend(channel_data.flatten())
        
        if len(channel_values) > 0:
            # NDVI[0,1],;land_cover
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
        # axes 2D1D(rowscols1)
        if rows == 1 and cols == 1:
            return axes
        if rows == 1:
            return axes[c]
        if cols == 1:
            return axes[r]
        return axes[r, c]

    col_offset = 0
    
    # 
    if pos_sample_idx is not None:
        pos_idx = positive_samples[pos_sample_idx]['idx']
        features, target = dataset[pos_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        for ch_idx, ch_name in enumerate(channel_names):
            ax = subplot_ax(ch_idx // 2, (ch_idx % 2) + col_offset)
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            time_with_max = np.argmax(channel_data.sum(axis=(1, 2)))
            
            # 
            im = ax.imshow(channel_data[time_with_max], cmap='hot', interpolation='nearest',
                          vmin=vmin_list[ch_idx], vmax=vmax_list[ch_idx])
            ax.set_title(f'Positive - channel {ch_idx}: {ch_name}\nTime step {time_with_max}',
                         fontsize=title_fs)
            #,
            if ch_idx == 0:
                ax.set_xlabel('Longitude (grid)', fontsize=axis_label_fs)
                ax.set_ylabel('Latitude (grid)', fontsize=axis_label_fs)
            ax.tick_params(axis='both', which='both', labelsize=tick_fs)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Feature value', fontsize=axis_label_fs)
            cbar.ax.tick_params(labelsize=cbar_fs)
        
        col_offset += 2
    
    # 
    if neg_sample_idx is not None:
        neg_idx = negative_samples[neg_sample_idx]['idx']
        features, target = dataset[neg_idx]
        features_np = features.numpy()  # [C, T, H, W]
        
        for ch_idx, ch_name in enumerate(channel_names):
            ax = subplot_ax(ch_idx // 2, (ch_idx % 2) + col_offset)
            channel_data = features_np[ch_idx, :, :, :]  # [T, H, W]
            time_with_max = np.argmax(channel_data.sum(axis=(1, 2)))
            
            # 
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
    
    # # FRP(5,4)
    # # FRP
    # if pos_sample_idx is not None:
    #     pos_sample_info = positive_samples[pos_sample_idx]
    #     pos_idx = pos_sample_info['idx']
    #     pos_features, _ = dataset[pos_idx]
    #     pos_features_np = pos_features.numpy()  # [C, T, H, W]
    #     grid_size = pos_features_np.shape[2]  # 
        
    #     ax_frp_pos = subplot_ax(4, 0)  # 5,1
        
    #     # FRP
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
    #         # :eventsFRP
    #         # pixel_date, pixel_lat, pixel_loneventsFRP
    #         if 'pixel_date' in orig_sample and 'pixel_lat' in orig_sample and 'pixel_lon' in orig_sample:
    #             pixel_date = orig_sample['pixel_date']
    #             pixel_lat = orig_sample['pixel_lat']
    #             pixel_lon = orig_sample['pixel_lon']
                
    #             # component_idFRP(components)
    #             if 'metadata' in orig_sample and 'component_id' in orig_sample['metadata']:
    #                 component_id = orig_sample['metadata']['component_id']
    #                 if component_id is not None:
    #                     # components
    #                     if data and 'preprocessed_data' in data and 'components' in data['preprocessed_data']:
    #                         components = data['preprocessed_data']['components']
    #                         # 
    #                         try:
    #                             # 
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
    #                             #,cp
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
    #                                 # component_id
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
    #                         # events
    #                         if 'preprocessed_data' in data and 'events' in data['preprocessed_data']:
    #                             import pandas as pd
    #                             events = data['preprocessed_data']['events']
    #                             # 
    #                             tolerance = 0.01  # 1km
    #                             date_match = pd.to_datetime(events['dtime']).dt.date == pd.to_datetime(pixel_date).date()
    #                             lat_match = (events['lat'] >= pixel_lat - tolerance) & (events['lat'] <= pixel_lat + tolerance)
    #                             lon_match = (events['lon'] >= pixel_lon - tolerance) & (events['lon'] <= pixel_lon + tolerance)
    #                             matching_events = events[date_match & lat_match & lon_match]
                                
    #                             if len(matching_events) > 0:
    #                                 # FRP()
    #                                 if 'maxFRP' in matching_events.columns:
    #                                     frp_value = float(matching_events['maxFRP'].max())
    #                                     frp_text = f"Event FRP: {frp_value:.2f} MW"
    #                                 else:
    #                                     frp_text = f"Matched event but no FRP field\nComponent ID: {component_id}"
    #                             else:
    #                                 frp_text = f"Component ID: {component_id}\nNo matching event"
    #                         else:
    #                             frp_text = f"Component ID: {component_id}\n(load data for FRP)"
    #             # component_id,events
    #             elif 'preprocessed_data' in data and 'events' in data['preprocessed_data']:
    #                 import pandas as pd
    #                 events = data['preprocessed_data']['events']
    #                 # 
    #                 tolerance = 0.01  # 1km
    #                 date_match = pd.to_datetime(events['dtime']).dt.date == pd.to_datetime(pixel_date).date()
    #                 lat_match = (events['lat'] >= pixel_lat - tolerance) & (events['lat'] <= pixel_lat + tolerance)
    #                 lon_match = (events['lon'] >= pixel_lon - tolerance) & (events['lon'] <= pixel_lon + tolerance)
    #                 matching_events = events[date_match & lat_match & lon_match]
                    
    #                 if len(matching_events) > 0:
    #                     # FRP()
    #                     if 'maxFRP' in matching_events.columns:
    #                         frp_value = float(matching_events['maxFRP'].max())
    #                         frp_text = f"Event FRP: {frp_value:.2f} MW"
    #                     else:
    #                         frp_text = f"Matched event but no FRP field\nDate: {pixel_date}"
    #                 else:
    #                     frp_text = f"No matching event\nDate: {pixel_date}\nLocation: ({pixel_lat:.4f}, {pixel_lon:.4f})"
    #         # targetsFRP()
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
        
    #     # FRP(FRP,;0)
    #     if frp_value is not None and frp_value > 0:
    #         #,FRP(FRP)
    #         frp_data = np.zeros((grid_size, grid_size))
    #         center = grid_size // 2
    #         # FRP
    #         frp_data[center-2:center+3, center-2:center+3] = frp_value
    #         # colorbar0FRP()
    #         vmax_frp = frp_value * 1.1 if frp_value > 0 else 1.0
    #     else:
    #         # FRP,
    #         frp_data = np.zeros((grid_size, grid_size))
    #         vmax_frp = 1.0  # 
        
    #     im_frp = ax_frp_pos.imshow(frp_data, cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax_frp)
    #     ax_frp_pos.set_title(f'Positive - FRP (fire radiative power)\n{frp_text}', fontsize=12)
    #     ax_frp_pos.set_xlabel('Longitude (grid)')
    #     ax_frp_pos.set_ylabel('Latitude (grid)')
    #     plt.colorbar(im_frp, ax=ax_frp_pos, label='FRP (MW)')
    
    # # FRP(0)
    # if neg_sample_idx is not None:
    #     neg_sample_info = negative_samples[neg_sample_idx]
    #     neg_idx = neg_sample_info['idx']
    #     neg_features, _ = dataset[neg_idx]
    #     neg_features_np = neg_features.numpy()  # [C, T, H, W]
    #     grid_size_neg = neg_features_np.shape[2]
        
    #     ax_frp_neg = subplot_ax(4, cols - 1)  # 5,
        
    #     #,FRP0
    #     frp_data_neg = np.zeros((grid_size_neg, grid_size_neg))
    #     # (1000 MW)colorbar
    #     vmax_frp_neg = 1000.0  #,forcolorbar
        
    #     im_frp_neg = ax_frp_neg.imshow(frp_data_neg, cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax_frp_neg)
    #     ax_frp_neg.set_title('Negative - FRP (fire radiative power)\nNo fire (FRP = 0)', fontsize=12)
    #     ax_frp_neg.set_xlabel('Longitude (grid)')
    #     ax_frp_neg.set_ylabel('Latitude (grid)')
    #     plt.colorbar(im_frp_neg, ax=ax_frp_neg, label='FRP (MW)')
    
    plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ : {save_path}")
    plt.close()

def visualize_time_evolution(dataset, positive_samples, negative_samples,
                            channel_idx=1, save_path='data_visualization/time_evolution.png'):
    """"""
    print(f"\n⏱️   ( {channel_idx})...")
    
    # 8
    channel_names = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
    sample_idx = (positive_samples[0]['idx'] if len(positive_samples) > 0 else (negative_samples[0]['idx'] if len(negative_samples) > 0 else 0))
    features, _ = dataset[sample_idx]
    if features.shape[0] != 8:
        print(f"⚠️  {features.shape[0]},8,.")
        return
    
    # 
    pos_sample_idx = 0 if len(positive_samples) > 0 else None
    neg_sample_idx = 0 if len(negative_samples) > 0 else None
    
    if pos_sample_idx is None and neg_sample_idx is None:
        print("  ⚠️ ,")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 
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
        
        # 
        axes[0, 0].plot(range(len(time_stats['total'])), time_stats['total'], 
                       marker='o', linewidth=2, markersize=6, label='Positive', color='#ff9999')
        # axes[0, 0].set_title('Total feature value over time')
        axes[0, 0].set_xlabel('Time step (days)')
        axes[0, 0].set_ylabel('Total feature value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 
        axes[0, 1].plot(range(len(time_stats['mean'])), time_stats['mean'],
                       marker='s', linewidth=2, markersize=6, label='Positive', color='#ff9999')
        # axes[0, 1].set_title('Mean feature value over time')
        axes[0, 1].set_xlabel('Time step (days)')
        axes[0, 1].set_ylabel('Mean feature value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 
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
        
        # ()
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
        
        # ()
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
    
    # 
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
    
    # 
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
    print(f"✅ : {save_path}")
    plt.close()

def _load_full_period_dataset(max_samples_per_shard: Optional[int] = 2000):
    """load 2002-2020 ().

     processed_firetracks_pixel_binary_2002-2020.pth;
   , 7 ****,.

    
    ----
    max_samples_per_shard : int
        . 2000  7*2000 ≈ 1.4 .
         2002-2020,.
    """
    merged_path = os.path.join(project_root, 'dataset', 'processed_firetracks_pixel_binary_2002-2020.pth')
    if os.path.exists(merged_path):
        print("📊  2002-2020,.")
        return load_dataset(merged_path)

    print("📊, 2002-2020 .")
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
            print(f"  ⚠️,: {rel_path}")
            continue
        print(f"  🔄 load {idx+1}/7: {rel_path}")
        if base_data is None:
            #  load_dataset, FireTracksDataset load
            base_data = load_dataset(full_path)
            st_samples = base_data.get('spatiotemporal_samples', [])
        else:
            shard_data = torch.load(full_path, weights_only=False)
            st_samples = shard_data.get('spatiotemporal_samples', [])
        if not st_samples:
            continue

        #, max_samples_per_shard ;
        #  max_samples_per_shard  None,().
        if max_samples_per_shard is not None and len(st_samples) > max_samples_per_shard:
            # :
            step = len(st_samples) / max_samples_per_shard
            sampled_indices = [int(i * step) for i in range(max_samples_per_shard)]
            sampled = [st_samples[i] for i in sampled_indices]
            print(f"    -  {len(st_samples):,}, {len(sampled):,} .")
        else:
            sampled = st_samples
            print(f"    -  {len(st_samples):,},for.")

        all_spatiotemporal.extend(sampled)

    if base_data is None or not all_spatiotemporal:
        raise RuntimeError("loading, dataset  .pth .")

    print(f"  ✅ : {len(all_spatiotemporal):,}(2002-2020 )")

    #  FireTracksDataset
    full_dataset = FireTracksDataset(
        all_spatiotemporal,
        target_type='binary_classification'
    )
    base_data['spatiotemporal_samples'] = all_spatiotemporal
    base_data['dataset'] = full_dataset
    # dataloader,
    return base_data


USE_FULL_DATA: bool = False  #  True,  2002-2020 ()


def main():
    """"""
    print("="*60)
    print("FireTracks")
    print("="*60)
    
    # 
    output_dir = 'data_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # load 2002-2020 :
    # -  USE_FULL_DATA=True,(max_samples_per_shard=None),;
    # -  2000,.
    max_per_shard = None if USE_FULL_DATA else 2000
    data = _load_full_period_dataset(max_samples_per_shard=max_per_shard)
    dataset = data['dataset']
    
    # (forFRP)
    spatiotemporal_samples = data.get('spatiotemporal_samples', None)
    
    # 
    positive_samples, negative_samples = analyze_positive_negative_samples(dataset, spatiotemporal_samples)
    
    # 
    print("\n" + "="*60)
    print("...")
    print("="*60)
    
    visualize_class_distribution(dataset)
    visualize_feature_distribution(dataset, positive_samples, negative_samples)
    visualize_land_cover_distribution(data)
    visualize_spatiotemporal_sample(dataset, positive_samples, negative_samples, data=data)
    visualize_time_evolution(dataset, positive_samples, negative_samples, channel_idx=1)
    
    print("\n" + "="*60)
    print("✅ !")
    print("="*60)
    print("\n:")
    print("  - data_visualization/class_distribution.png: ")
    print("  - data_visualization/feature_distribution.png: ")
    print("  - data_visualization/land_cover_distribution.png: ")
    print("  - data_visualization/spatiotemporal_sample.png: ")
    print("  - data_visualization/time_evolution.png: ")
    print("\n:")
    print("  1. ✅ : ")
    print("  2. ✅ : ")
    print("  3. ✅ : ")
    print("  4. ✅ : ")
    print("  5. ✅ : ")

if __name__ == '__main__':
    main()
