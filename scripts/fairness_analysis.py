"""
模型公平性分析工具
分析模型在不同经济群体（按人口密度划分）间的性能差异和公平性指标
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 统一科研作图风格：使用 Times New Roman，并整体放大字号（论文中缩放更清晰）
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,            # 基础字体：轴刻度等
    "axes.titlesize": 22,       # 子图标题
    "axes.labelsize": 20,       # 轴标签
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, roc_curve, 
    precision_recall_curve, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

# 统计检验相关导入
try:
    from scipy.stats import chi2_contingency, mannwhitneyu, bootstrap
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy未安装，将跳过统计显著性检验")
    print("   安装命令: pip install scipy")

# 添加 src 目录到路径（公开 release 布局）
_release_root = Path(__file__).resolve().parent.parent
src_dir = _release_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import fairlearn.metrics as fl_metrics
    from fairlearn.postprocessing import ThresholdOptimizer
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False
    print("⚠️  fairlearn未安装，将使用自定义公平性指标计算")
    print("   安装命令: pip install fairlearn")

from fire_equality.datamodules.firetracks_loader import FireTracksDataset
import pytorch_lightning as pl
from fire_equality.models.fire_equality_model import ConvLSTM_fire_equality_model

# 可解释性：按群体划分的特征贡献（实验一）
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# 8 通道与 GeoRL/feature_alignment 一致
FEATURE_NAMES = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']


class ConvLSTMWrapperForCaptum(torch.nn.Module):
    """供 Captum 使用的包装器：输入 [B, C, T, H, W]，输出 log_softmax [B, 2]。"""
    def __init__(self, pl_model):
        super().__init__()
        self.pl_model = pl_model
        self.pl_model.eval()

    def forward(self, x):
        # x: [B, C, T, H, W] -> [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.pl_model(x)

# 修复PyTorch 2.6+的checkpoint加载问题
def patch_lightning_checkpoint_loading():
    """修补PyTorch Lightning的checkpoint加载，使用weights_only=False"""
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
        
        # 保存原始函数（如果还没有保存）
        if not hasattr(cloud_io, '_original_load'):
            cloud_io._original_load = cloud_io._load
        
        def patched_load(f, map_location=None):
            """修补的加载函数，使用weights_only=False"""
            return torch.load(f, map_location=map_location, weights_only=False)
        
        # 替换原始函数
        cloud_io._load = patched_load
        
        # 也修补pl_load函数（如果存在）
        if hasattr(cloud_io, 'pl_load'):
            if not hasattr(cloud_io, '_original_pl_load'):
                cloud_io._original_pl_load = cloud_io.pl_load
            
            def patched_pl_load(path, map_location=None):
                """修补的pl_load函数"""
                return torch.load(path, map_location=map_location, weights_only=False)
            
            cloud_io.pl_load = patched_pl_load
        
        # 直接修补torch.load（更激进但更可靠）
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
        
        def patched_torch_load(f, map_location=None, **kwargs):
            """修补的torch.load，强制使用weights_only=False"""
            kwargs['weights_only'] = False
            return torch._original_load(f, map_location=map_location, **kwargs)
        
        torch.load = patched_torch_load
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"无法修补Lightning checkpoint加载: {e}")
        return False

# 在模块加载时修补
patch_lightning_checkpoint_loading()


def save_plot_cache(analyzer, path):
    """保存用于重绘图的缓存（不重新跑模型/数据时使用）。"""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 若已经运行过 XAI 分析，则一并缓存其结果，便于 figures-only 模式下重绘 XAI 图
    xai_results = getattr(analyzer, '_xai_results', None)
    cache = {
        'true_labels': np.asarray(analyzer.true_labels) if analyzer.true_labels is not None else None,
        'probabilities': np.asarray(analyzer.probabilities) if analyzer.probabilities is not None else None,
        'predictions': np.asarray(analyzer.predictions) if analyzer.predictions is not None else None,
        'sensitive_attributes': analyzer.sensitive_attributes,
        'xai_results': xai_results,
    }
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"✅ 绘图缓存已保存: {path}")


def load_plot_cache(path):
    """加载绘图缓存，返回 dict (true_labels, probabilities, predictions, sensitive_attributes)。"""
    import pickle
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


class FairnessAnalyzer:
    """公平性分析器"""
    
    def __init__(self, model_path=None, data_paths=None, checkpoint_path=None):
        """
        初始化公平性分析器
        
        Args:
            model_path: 模型checkpoint路径
            data_paths: 数据文件路径列表
            checkpoint_path: PyTorch Lightning checkpoint路径
        """
        self.model_path = model_path
        self.data_paths = data_paths or []
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.datamodule = None
        self.predictions = None
        self.true_labels = None
        self.sensitive_attributes = None
        self.sample_metadata = []
        # 缓存已计算的公平性指标，避免重复计算
        self._fairness_cache = {}
        # 是否跳过统计显著性检验（在 --figures-only 模式下用于加速绘图）
        self.skip_statistical_tests = False
        # 是否处于仅重绘图模式：用于抑制冗长日志、避免生成非图像文件
        self.is_figures_only = False
        
    def load_data(self):
        """加载测试数据"""
        print("=" * 80)
        print("📊 加载数据")
        print("=" * 80)
        
        all_samples = []
        all_metadata = []
        
        # 加载所有数据文件
        for data_file in self.data_paths:
            file_path = Path(data_file)
            if not file_path.exists():
                print(f"⚠️  文件不存在: {data_file}")
                continue
            
            print(f"\n📂 加载: {file_path.name}")
            try:
                data = torch.load(file_path, weights_only=False)
                
                if 'spatiotemporal_samples' in data:
                    samples = data['spatiotemporal_samples']
                    dataset = FireTracksDataset(samples, target_type='binary_classification')
                    
                    # 提取metadata（ISO3、GDP、人口等）
                    for i, sample in enumerate(samples):
                        metadata = {
                            'file': file_path.name,
                            'sample_idx': len(all_samples) + i,
                            'target': sample.get('target', None),
                            'pixel_lat': sample.get('pixel_lat', None),
                            'pixel_lon': sample.get('pixel_lon', None),
                            'pixel_date': sample.get('pixel_date', None),
                            'land_cover': sample.get('land_cover', None),
                            'metadata': sample.get('metadata', {}),
                            'features': sample.get('features', None)  # 用于提取GDP和人口
                        }
                        all_metadata.append(metadata)
                    
                    all_samples.extend(samples)
                    print(f"   ✅ 加载 {len(samples):,} 个样本")
                    
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
                import traceback
                traceback.print_exc()
        
        self.sample_metadata = all_metadata
        print(f"\n✅ 总共加载 {len(all_samples):,} 个样本")
        
        return all_samples
    
    def extract_sensitive_attributes(self, samples):
        """
        从样本中提取敏感属性（GDP per capita、人口密度、Continent、ISO3）
        
        Args:
            samples: 样本列表
            
        Returns:
            dict: 包含敏感属性的字典
        """
        print("\n" + "=" * 80)
        print("🔍 提取敏感属性")
        print("=" * 80)
        
        # 加载GDP per capita数据
        gdp_per_capita_path = Path('dataset/gdp_per_capita.csv')
        gdp_per_capita_df = None
        if gdp_per_capita_path.exists():
            try:
                gdp_per_capita_df = pd.read_csv(gdp_per_capita_path)
                print(f"✅ 加载GDP per capita数据: {len(gdp_per_capita_df):,} 条记录")
                print(f"   GDP per capita数据列: {list(gdp_per_capita_df.columns[:5])}...")
                # 检查Country Code列
                if 'Country Code' in gdp_per_capita_df.columns:
                    unique_countries = gdp_per_capita_df['Country Code'].nunique()
                    print(f"   唯一国家代码数: {unique_countries}")
            except Exception as e:
                print(f"⚠️  加载GDP per capita数据失败: {e}")
        else:
            print(f"⚠️  GDP per capita文件不存在: {gdp_per_capita_path}")
        
        # 加载协变量数据（用于提取continent和GDP信息）
        covariate_data_path = Path('dataset/filtered_cleaned_cp_covariate.csv')
        covariate_df = None
        if covariate_data_path.exists():
            try:
                covariate_df = pd.read_csv(covariate_data_path)
                print(f"✅ 加载协变量数据: {len(covariate_df):,} 条记录")
                print(f"   协变量数据列: {list(covariate_df.columns)}")
                # 检查关键列
                if 'iso3' in covariate_df.columns:
                    unique_iso3 = covariate_df['iso3'].dropna().nunique()
                    print(f"   唯一ISO3代码数: {unique_iso3}")
                if 'continent' in covariate_df.columns:
                    unique_continents = covariate_df['continent'].dropna().nunique()
                    print(f"   唯一大洲数: {unique_continents}")
                    print(f"   大洲分布: {covariate_df['continent'].value_counts().to_dict()}")
                if 'year' in covariate_df.columns:
                    year_range = (covariate_df['year'].min(), covariate_df['year'].max())
                    print(f"   年份范围: {year_range[0]}-{year_range[1]}")
            except Exception as e:
                print(f"⚠️  加载协变量数据失败: {e}")
        else:
            print(f"⚠️  协变量数据文件不存在: {covariate_data_path}")
        
        sensitive_attrs = {
            'iso3': [],
            'population': [],
            'pop_group': [],
            'gdp_per_capita': [],
            'gdp_group': [],
            'continent': [],
            'continent_group': []
        }
        
        # 协变量表经纬度→最近邻 KD-tree（用于无 metadata['iso3'] 时兜底，无需重跑 .pth）
        _cov_tree = None
        _cov_df_nn = None
        if covariate_df is not None and all(c in covariate_df.columns for c in ['lat_mean', 'lon_mean', 'iso3']):
            try:
                from scipy.spatial import cKDTree
                _df_nn = covariate_df[['lat_mean', 'lon_mean', 'iso3']].copy()
                _df_nn['continent'] = covariate_df['continent'] if 'continent' in covariate_df.columns else None
                _df_nn = _df_nn.dropna(subset=['lat_mean', 'lon_mean', 'iso3'])
                _df_nn = _df_nn[_df_nn['iso3'].astype(str).str.strip() != '']
                if len(_df_nn) > 0:
                    pts = np.column_stack([_df_nn['lat_mean'].values, _df_nn['lon_mean'].values])
                    _cov_tree = cKDTree(pts)
                    _cov_df_nn = _df_nn
                    print(f"   已构建协变量经纬度最近邻索引（样本无 iso3 时将用 2° 内最近邻补全）")
            except Exception:
                pass
        
        # 从特征中提取人口信息
        pop_values = []
        gdp_per_capita_values = []
        continent_values = []
        
        for i, sample in enumerate(samples):
            features = sample.get('features', None)
            metadata = sample.get('metadata', {})
            
            iso3 = None
            population = None
            gdp_per_capita = None
            continent = None
            
            # 方法1: 从metadata中提取ISO3
            if 'iso3' in metadata:
                iso3 = metadata['iso3']
            
            # 从metadata或协变量数据中获取年份
            pixel_date = sample.get('pixel_date', None)
            year = None
            if pixel_date:
                try:
                    year = pd.Timestamp(pixel_date).year
                except:
                    pass
            
            # 获取经纬度信息（用于匹配协变量数据；无 pixel_* 时用 metadata 的 center）
            pixel_lat = sample.get('pixel_lat', None) or (metadata.get('center_lat') if isinstance(metadata.get('center_lat'), (int, float)) else None)
            pixel_lon = sample.get('pixel_lon', None) or (metadata.get('center_lon') if isinstance(metadata.get('center_lon'), (int, float)) else None)
            
            # 方法2: 从特征中提取人口（通道3=population）；GDP 见方法4 与 方法4.5
            if features is not None:
                try:
                    # features shape: [time_steps, grid_size, grid_size, channels]，通道3: population
                    if len(features.shape) >= 4:
                        pop_data = features[:, :, :, 3]
                        pop_nonzero = pop_data[pop_data > 0]
                        if len(pop_nonzero) > 0:
                            population = float(np.mean(pop_nonzero))
                except Exception as e:
                    pass
            
            # 方法3: 从协变量数据文件中查找ISO3、人口、continent
            # 优先使用iso3匹配，如果没有iso3则尝试用经纬度和年份匹配
            if covariate_df is not None:
                try:
                    cov_rows = None
                    
                    # 优先：使用iso3和年份匹配
                    if iso3:
                        if year:
                            cov_rows = covariate_df[(covariate_df['iso3'] == iso3) & (covariate_df['year'] == year)]
                        else:
                            cov_rows = covariate_df[covariate_df['iso3'] == iso3]
                    
                    # 备选：如果没有iso3，尝试用经纬度匹配（允许一定误差）
                    if cov_rows is None or len(cov_rows) == 0:
                        if pixel_lat is not None and pixel_lon is not None and year:
                            # 使用经纬度匹配（允许0.1度的误差）
                            lat_tolerance = 0.1
                            lon_tolerance = 0.1
                            cov_rows = covariate_df[
                                (covariate_df['year'] == year) &
                                (abs(covariate_df['lat_mean'] - pixel_lat) <= lat_tolerance) &
                                (abs(covariate_df['lon_mean'] - pixel_lon) <= lon_tolerance)
                            ]
                    
                    if cov_rows is not None and len(cov_rows) > 0:
                        # 取第一条记录（如果有多个）
                        row = cov_rows.iloc[0]
                        
                        # 提取iso3（如果之前没有）
                        if not iso3 and 'iso3' in row:
                            iso3_val = row['iso3']
                            if pd.notna(iso3_val) and str(iso3_val).strip() != '':
                                iso3 = str(iso3_val).strip()
                        
                        # 提取人口
                        if population is None and 'population' in row:
                            pop_val = row['population']
                            if pd.notna(pop_val):
                                population = float(pop_val)
                        
                        # 提取continent
                        if 'continent' in row:
                            cont_val = row['continent']
                            if pd.notna(cont_val) and str(cont_val).strip() != '':
                                continent = str(cont_val).strip()
                except Exception as e:
                    pass
            
            # 方法3.5: 若无 iso3，用协变量表经纬度最近邻（2° 内）兜底（无需重跑 .pth）
            if iso3 is None and pixel_lat is not None and pixel_lon is not None and _cov_tree is not None and _cov_df_nn is not None:
                try:
                    dist, idx = _cov_tree.query([pixel_lat, pixel_lon], k=1, distance_upper_bound=2.0)
                    if not np.isinf(dist) and float(dist) <= 2.0 and 0 <= idx < len(_cov_df_nn):
                        idx = int(idx)
                        row = _cov_df_nn.iloc[idx]
                        if pd.notna(row['iso3']) and str(row['iso3']).strip():
                            iso3 = str(row['iso3']).strip()
                        if row.get('continent') is not None and pd.notna(row['continent']) and str(row['continent']).strip():
                            continent = str(row['continent']).strip()
                except Exception:
                    pass
            
            # 方法4: 优先从 GDP per capita CSV 按 iso3+年份查找（公平性分析中 iso3 可通过 2° 兜底获得，覆盖通常优于建 .pth 时写入的通道4）
            if iso3 and gdp_per_capita_df is not None and year:
                try:
                    iso3_clean = str(iso3).strip().upper()
                    country_rows = gdp_per_capita_df[gdp_per_capita_df['Country Code'].str.strip().str.upper() == iso3_clean]
                    if len(country_rows) > 0:
                        row = country_rows.iloc[0]
                        year_col = str(year)
                        if year_col in row and pd.notna(row[year_col]):
                            gdp_val = row[year_col]
                            if isinstance(gdp_val, (int, float)) or (isinstance(gdp_val, str) and gdp_val.strip() != ''):
                                try:
                                    gdp_per_capita = float(gdp_val)
                                except (ValueError, TypeError):
                                    pass
                except Exception as e:
                    pass
            
            # 方法4.5: 若 CSV 未匹配到 GDP，用特征通道4（GDP）非零均值兜底（建 .pth 时无 iso3 的样本通道4多为0，故优先用方法4）
            if gdp_per_capita is None and features is not None and len(features.shape) >= 4 and features.shape[3] > 4:
                try:
                    gdp_data = features[:, :, :, 4]
                    gdp_nonzero = gdp_data[gdp_data > 0]
                    if len(gdp_nonzero) > 0:
                        gdp_per_capita = float(np.mean(gdp_nonzero))
                except Exception:
                    pass
            
            sensitive_attrs['iso3'].append(iso3)
            sensitive_attrs['population'].append(population)
            sensitive_attrs['gdp_per_capita'].append(gdp_per_capita)
            sensitive_attrs['continent'].append(continent)
            
            if population is not None:
                pop_values.append(population)
            if gdp_per_capita is not None:
                gdp_per_capita_values.append(gdp_per_capita)
            if continent is not None:
                continent_values.append(continent)
        
        # 划分人口密度组
        if pop_values:
            pop_33 = np.percentile(pop_values, 33)
            pop_67 = np.percentile(pop_values, 67)
            
            print(f"\n📊 人口密度统计:")
            print(f"   33分位: {pop_33:.2f}")
            print(f"   67分位: {pop_67:.2f}")
            
            for i, pop in enumerate(sensitive_attrs['population']):
                if pop is None:
                    sensitive_attrs['pop_group'].append('Unknown')
                elif pop < pop_33:
                    sensitive_attrs['pop_group'].append('Low')
                elif pop < pop_67:
                    sensitive_attrs['pop_group'].append('Medium')
                else:
                    sensitive_attrs['pop_group'].append('High')
        else:
            sensitive_attrs['pop_group'] = ['Unknown'] * len(samples)
        
        # 划分GDP per capita组
        if gdp_per_capita_values:
            gdp_33 = np.percentile(gdp_per_capita_values, 33)
            gdp_67 = np.percentile(gdp_per_capita_values, 67)
            
            print(f"\n📊 GDP per capita统计:")
            print(f"   33分位: ${gdp_33:.2f}")
            print(f"   67分位: ${gdp_67:.2f}")
            
            for i, gdp in enumerate(sensitive_attrs['gdp_per_capita']):
                if gdp is None:
                    sensitive_attrs['gdp_group'].append('Unknown')
                elif gdp < gdp_33:
                    sensitive_attrs['gdp_group'].append('Low')
                elif gdp < gdp_67:
                    sensitive_attrs['gdp_group'].append('Medium')
                else:
                    sensitive_attrs['gdp_group'].append('High')
        else:
            sensitive_attrs['gdp_group'] = ['Unknown'] * len(samples)
        
        # 划分Continent组（按continent分组，然后根据样本数量分为high/medium/low）
        if continent_values:
            # 统计每个continent的样本数
            continent_counts = {}
            for cont in continent_values:
                if cont:
                    continent_counts[cont] = continent_counts.get(cont, 0) + 1
            
            # 按样本数排序，分为三组
            sorted_continents = sorted(continent_counts.items(), key=lambda x: x[1], reverse=True)
            total_continents = len(sorted_continents)
            
            # 将continent分为三组：样本数最多的为High，中间的为Medium，最少的为Low
            if total_continents >= 3:
                high_continents = {cont for cont, _ in sorted_continents[:total_continents//3]}
                medium_continents = {cont for cont, _ in sorted_continents[total_continents//3:2*total_continents//3]}
                low_continents = {cont for cont, _ in sorted_continents[2*total_continents//3:]}
            elif total_continents == 2:
                high_continents = {sorted_continents[0][0]}
                medium_continents = {sorted_continents[1][0]}
                low_continents = set()
            else:
                high_continents = {sorted_continents[0][0]} if sorted_continents else set()
                medium_continents = set()
                low_continents = set()
            
            print(f"\n📊 Continent分组:")
            print(f"   High组: {high_continents}")
            print(f"   Medium组: {medium_continents}")
            print(f"   Low组: {low_continents}")
            
            for i, cont in enumerate(sensitive_attrs['continent']):
                if cont is None:
                    sensitive_attrs['continent_group'].append('Unknown')
                elif cont in high_continents:
                    sensitive_attrs['continent_group'].append('High')
                elif cont in medium_continents:
                    sensitive_attrs['continent_group'].append('Medium')
                elif cont in low_continents:
                    sensitive_attrs['continent_group'].append('Low')
                else:
                    sensitive_attrs['continent_group'].append('Unknown')
        else:
            sensitive_attrs['continent_group'] = ['Unknown'] * len(samples)
        
        # 统计
        n_samples = len(samples)
        print(f"\n📊 敏感属性统计:")
        iso3_count = sum(1 for iso3 in sensitive_attrs['iso3'] if iso3)
        pop_count = sum(1 for pop in sensitive_attrs['population'] if pop is not None)
        gdp_count = sum(1 for gdp in sensitive_attrs['gdp_per_capita'] if gdp is not None)
        continent_count = sum(1 for cont in sensitive_attrs['continent'] if cont)
        
        pct = (lambda a, b: (a / b * 100) if b else 0)
        print(f"   有ISO3代码的样本: {iso3_count:,} ({pct(iso3_count, n_samples):.1f}%)")
        print(f"   有人口数据的样本: {pop_count:,} ({pct(pop_count, n_samples):.1f}%)")
        print(f"   有GDP per capita数据的样本: {gdp_count:,} ({pct(gdp_count, n_samples):.1f}%)")
        print(f"   有Continent数据的样本: {continent_count:,} ({pct(continent_count, n_samples):.1f}%)")
        
        # 显示GDP per capita的统计信息（如果有数据）
        if gdp_per_capita_values:
            print(f"\n   GDP per capita数值范围: ${min(gdp_per_capita_values):,.2f} - ${max(gdp_per_capita_values):,.2f}")
        
        # 显示Continent分布（如果有数据）
        if continent_values:
            continent_dist = {}
            for cont in continent_values:
                continent_dist[cont] = continent_dist.get(cont, 0) + 1
            n_cont = len(continent_values)
            print(f"\n   Continent分布详情:")
            for cont, count in sorted(continent_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"     {cont}: {count:,} ({pct(count, n_cont):.1f}%)")
        
        print(f"\n   人口密度分组分布:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['pop_group'].count(group)
            print(f"     {group}: {count:,} ({pct(count, n_samples):.1f}%)")
        
        print(f"\n   GDP per capita分组分布:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['gdp_group'].count(group)
            print(f"     {group}: {count:,} ({count/len(samples)*100:.1f}%)")
        
        print(f"\n   Continent分组分布:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['continent_group'].count(group)
            print(f"     {group}: {count:,} ({count/len(samples)*100:.1f}%)")
        
        self.sensitive_attributes = sensitive_attrs
        return sensitive_attrs
    
    def load_model_and_predict(self, test_dataset):
        """
        加载模型并生成预测
        
        Args:
            test_dataset: 测试数据集
        """
        print("\n" + "=" * 80)
        print("🤖 加载模型并生成预测")
        print("=" * 80)
        
        if self.checkpoint_path is None:
            raise ValueError("需要提供checkpoint_path")
        
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        print(f"📂 加载checkpoint: {checkpoint_path}")
        self._load_model_only()
        print("✅ 模型加载成功")
        # 生成预测
        print("\n🔮 生成预测...")
        all_predictions = []
        all_probs = []
        all_labels = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(dataloader):
                # 模型前向传播
                # features shape: [B, C, T, H, W]，需要转换为 [B, T, C, H, W]
                features_permuted = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                logits = self.model(features_permuted)
                # 模型输出是log_softmax，需要转换为概率
                probs = torch.exp(logits)[:, 1]  # 取正类的概率
                preds = (probs > 0.5).long()
                
                all_predictions.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   处理进度: {batch_idx + 1}/{len(dataloader)} 批次")
        
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probs)
        self.true_labels = np.array(all_labels)
        
        print(f"✅ 预测完成: {len(self.predictions):,} 个样本")
        
        # 诊断信息
        print(f"\n📊 预测诊断:")
        print(f"   预测分布: 0={np.sum(self.predictions==0):,} ({np.sum(self.predictions==0)/len(self.predictions)*100:.1f}%), "
              f"1={np.sum(self.predictions==1):,} ({np.sum(self.predictions==1)/len(self.predictions)*100:.1f}%)")
        print(f"   真实标签分布: 0={np.sum(self.true_labels==0):,} ({np.sum(self.true_labels==0)/len(self.true_labels)*100:.1f}%), "
              f"1={np.sum(self.true_labels==1):,} ({np.sum(self.true_labels==1)/len(self.true_labels)*100:.1f}%)")
        print(f"   概率范围: {self.probabilities.min():.4f} - {self.probabilities.max():.4f}")
        print(f"   概率均值: {self.probabilities.mean():.4f}, 中位数: {np.median(self.probabilities):.4f}")
        print(f"   概率>0.5的样本数: {np.sum(self.probabilities > 0.5):,}")
        print(f"   概率>0.3的样本数: {np.sum(self.probabilities > 0.3):,}")
        print(f"   概率>0.1的样本数: {np.sum(self.probabilities > 0.1):,}")
        
        return self.predictions, self.probabilities, self.true_labels
    
    def _load_model_only(self):
        """仅加载模型权重，不运行预测。用于 XAI 等只需前向的流程，避免覆盖已有 true_labels/probabilities。"""
        if self.checkpoint_path is None:
            raise ValueError("需要提供checkpoint_path")
        try:
            patch_lightning_checkpoint_loading()
            self.model = ConvLSTM_fire_equality_model.load_from_checkpoint(
                str(Path(self.checkpoint_path)),
                map_location='cpu'
            )
            self.model.eval()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_performance_metrics(self, y_true, y_pred, y_proba=None):
        """
        计算性能指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            
        Returns:
            dict: 性能指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': 2 * precision_score(y_true, y_pred, zero_division=0) * recall_score(y_true, y_pred, zero_division=0) / 
                  (precision_score(y_true, y_pred, zero_division=0) + recall_score(y_true, y_pred, zero_division=0) + 1e-8)
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
                metrics['auprc'] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics['auc'] = None
                metrics['auprc'] = None
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def calculate_fairness_metrics(self, y_true, y_pred, y_proba, sensitive_attr, 
                                   cache_key=None, use_cache=True):
        """
        计算公平性指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            cache_key: 缓存键（可选），如果提供且use_cache=True，将检查缓存
            use_cache: 是否使用缓存（默认True）
            
        Returns:
            dict: 公平性指标字典
        """
        # 检查缓存
        if use_cache and cache_key and cache_key in self._fairness_cache:
            print(f"\n✅ 使用缓存的公平性指标结果 (key: {cache_key})")
            return self._fairness_cache[cache_key]
        
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("⚖️  计算公平性指标")
            if cache_key:
                print(f"   (缓存键: {cache_key})")
            print("=" * 80)
        
        fairness_metrics = {}
        
        # 按敏感属性分组
        unique_groups = np.unique(sensitive_attr)
        unique_groups = [g for g in unique_groups if g != 'Unknown']  # 排除Unknown组
        
        if len(unique_groups) < 2:
            print("⚠️  敏感属性分组少于2组，无法计算公平性指标")
            return fairness_metrics
        
        # 计算各组的性能指标
        group_metrics = {}
        for group in unique_groups:
            mask = sensitive_attr == group
            if mask.sum() == 0:
                continue
            
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_proba = y_proba[mask] if y_proba is not None else None
            
            if not quiet:
                # 诊断信息
                print(f"\n📊 {group} 组诊断:")
                print(f"   样本数: {mask.sum():,}")
                
                # 检查样本索引分布（用于验证顺序一致性）
                sample_indices = np.where(mask)[0]
                if len(sample_indices) > 0:
                    print(f"   样本索引范围: {sample_indices.min()} - {sample_indices.max()}")
                    print(f"   样本索引前10个: {sample_indices[:10].tolist()}")
                    if len(sample_indices) > 10:
                        print(f"   样本索引后10个: {sample_indices[-10:].tolist()}")
                
                print(f"   真实标签分布: 0={np.sum(group_y_true==0):,} ({np.sum(group_y_true==0)/len(group_y_true)*100:.1f}%), "
                      f"1={np.sum(group_y_true==1):,} ({np.sum(group_y_true==1)/len(group_y_true)*100:.1f}%)")
                print(f"   预测分布: 0={np.sum(group_y_pred==0):,} ({np.sum(group_y_pred==0)/len(group_y_pred)*100:.1f}%), "
                      f"1={np.sum(group_y_pred==1):,} ({np.sum(group_y_pred==1)/len(group_y_pred)*100:.1f}%)")
                if group_y_proba is not None:
                    print(f"   概率范围: {group_y_proba.min():.4f} - {group_y_proba.max():.4f}")
                    print(f"   概率均值: {group_y_proba.mean():.4f}, 中位数: {np.median(group_y_proba):.4f}")
                    print(f"   概率标准差: {group_y_proba.std():.4f}")
                    print(f"   概率>0.5的样本数: {np.sum(group_y_proba > 0.5):,}")
                    
                    # 如果概率范围太小（所有值几乎相同），发出警告
                    if group_y_proba.max() - group_y_proba.min() < 0.001:
                        print(f"   ⚠️  警告：该组所有样本的预测概率几乎相同！")
                        print(f"      这可能表明样本顺序不一致或分组有问题")
            
            metrics = self.calculate_performance_metrics(group_y_true, group_y_pred, group_y_proba)
            group_metrics[group] = metrics
            
            if not quiet:
                print(f"\n📊 {group} 组性能:")
                print(f"   准确率: {metrics['accuracy']:.4f}")
                print(f"   精确率: {metrics['precision']:.4f}")
                print(f"   召回率: {metrics['recall']:.4f}")
                print(f"   F1分数: {metrics['f1']:.4f}")
                if metrics.get('auc') is not None:
                    print(f"   AUC: {metrics['auc']:.4f}")
                if metrics.get('auprc') is not None:
                    print(f"   AUPRC: {metrics['auprc']:.4f}")
                if 'tp' in metrics:
                    print(f"   混淆矩阵: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
        
        # 计算公平性指标
        if HAS_FAIRLEARN:
            try:
                # Demographic Parity (统计均等)
                dp_diff = fl_metrics.demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                dp_ratio = fl_metrics.demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                
                # Equalized Odds (均等化机会)
                eo_diff = fl_metrics.equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                eo_ratio = fl_metrics.equalized_odds_ratio(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                
                fairness_metrics['demographic_parity_difference'] = dp_diff
                fairness_metrics['demographic_parity_ratio'] = dp_ratio
                fairness_metrics['equalized_odds_difference'] = eo_diff
                fairness_metrics['equalized_odds_ratio'] = eo_ratio
                
                print(f"\n⚖️  Fairlearn公平性指标:")
                print(f"   Demographic Parity Difference: {dp_diff:.4f} (越小越好，0为完全公平)")
                print(f"   Demographic Parity Ratio: {dp_ratio:.4f} (越接近1越好)")
                print(f"   Equalized Odds Difference: {eo_diff:.4f} (越小越好，0为完全公平)")
                print(f"   Equalized Odds Ratio: {eo_ratio:.4f} (越接近1越好)")
                
            except Exception as e:
                print(f"⚠️  计算Fairlearn指标失败: {e}")
        
        # 计算自定义公平性指标（性能差异）
        if not quiet:
            print(f"\n📊 组间性能差异:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']:
            values = []
            for group in unique_groups:
                if group in group_metrics and metric_name in group_metrics[group]:
                    val = group_metrics[group][metric_name]
                    if val is not None:
                        values.append((group, val))
            
            if len(values) >= 2:
                vals = [v[1] for v in values]
                max_val = max(vals)
                min_val = min(vals)
                diff = max_val - min_val
                ratio = min_val / max_val if max_val > 0 else 0
                
                fairness_metrics[f'{metric_name}_difference'] = diff
                fairness_metrics[f'{metric_name}_ratio'] = ratio
                
                if not quiet:
                    print(f"   {metric_name}:")
                    print(f"     最大值: {max_val:.4f}, 最小值: {min_val:.4f}")
                    if max_val > 0:
                        print(f"     差异: {diff:.4f} ({diff/max_val*100:.1f}%)")
                    else:
                        print(f"     差异: {diff:.4f} (所有组值相同或为0)")
                    print(f"     比率: {ratio:.4f}" if max_val > 0 else f"     比率: N/A (所有组值为0)")
        
        fairness_metrics['group_metrics'] = group_metrics
        
        # 执行统计显著性检验（如未被显式跳过）
        if (not getattr(self, 'skip_statistical_tests', False)) and HAS_SCIPY and len(unique_groups) >= 2:
            statistical_tests = self.perform_statistical_tests(
                y_true, y_pred, y_proba, sensitive_attr, group_metrics
            )
            fairness_metrics['statistical_tests'] = statistical_tests
        
        # 保存到缓存
        if use_cache and cache_key:
            self._fairness_cache[cache_key] = fairness_metrics
            print(f"\n💾 已缓存公平性指标结果 (key: {cache_key})")
        
        return fairness_metrics

    def run_group_wise_attribution(
        self,
        test_dataset,
        sensitive_attrs,
        output_dir,
        max_samples_per_group=800,
        batch_size=8,
        n_steps=50,
        use_cuda=False,
    ):
        """
        实验一：按群体划分的特征贡献分析。
        使用 Integrated Gradients 计算归因，按通道聚合后按 pop_group 分层绘图。
        """
        if not CAPTUM_AVAILABLE:
            print("⚠️  captum 未安装，跳过按群体归因分析。安装: pip install captum")
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pop_group = np.array(sensitive_attrs['pop_group'])
        valid = pop_group != 'Unknown'
        if valid.sum() == 0:
            print("⚠️  无有效 pop_group，跳过归因分析")
            return None
        unique_groups = [g for g in ['Low', 'Medium', 'High'] if (pop_group == g).sum() > 0]
        if len(unique_groups) < 2:
            print("⚠️  有效群体数 < 2，跳过归因分析")
            return None
        # 仅加载模型，不重新预测，避免覆盖已有 true_labels/probabilities（否则会破坏后续 ROC 等图）
        if self.model is None and self.checkpoint_path:
            self._load_model_only()
        if self.model is None:
            print("⚠️  未加载模型，无法运行归因")
            return None
        device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        wrapper = ConvLSTMWrapperForCaptum(self.model).to(device)
        ig = IntegratedGradients(wrapper)
        indices_by_group = {g: np.where(pop_group == g)[0] for g in unique_groups}
        to_run = []
        for g in unique_groups:
            idxs = indices_by_group[g]
            if len(idxs) > max_samples_per_group:
                idxs = np.random.RandomState(42).choice(idxs, max_samples_per_group, replace=False)
            to_run.extend(idxs.tolist())
        to_run = sorted(to_run)
        importance_list = []
        sample_indices = []
        for k, idx in enumerate(to_run):
            features, label = test_dataset[idx]
            features = features.unsqueeze(0).to(device)
            features.requires_grad_(True)
            baseline = torch.zeros_like(features, device=device)
            try:
                attr = ig.attribute(features, baselines=baseline, target=1, n_steps=n_steps)
            except Exception as e:
                continue
            attr = attr.detach().cpu()
            imp = attr.abs().mean(dim=(2, 3, 4)).squeeze(0).numpy()
            importance_list.append(imp)
            sample_indices.append(idx)
            if (k + 1) % 100 == 0:
                print(f"   归因进度: {k + 1}/{len(to_run)}")
        if not importance_list:
            print("⚠️  未得到有效归因结果")
            return None
        importance_matrix = np.array(importance_list)
        group_for_each = pop_group[sample_indices]
        results = {
            'importance_matrix': importance_matrix,
            'sample_indices': np.array(sample_indices),
            'group_for_each': group_for_each,
            'feature_names': FEATURE_NAMES,
        }
        # 保存最新的 XAI 结果到实例，便于写入缓存或 figures-only 模式下重绘
        self._xai_results = results
        self._plot_attribution_boxplot_by_group(results, output_dir)
        self._plot_top5_features_by_group(results, output_dir)
        return results

    def _plot_attribution_boxplot_by_group(self, results, output_dir):
        """按群体绘制各通道归因重要性箱线图。"""
        importance_matrix = results['importance_matrix']
        group_for_each = results['group_for_each']
        names = results['feature_names']
        n_c = importance_matrix.shape[1]
        data = []
        for i in range(importance_matrix.shape[0]):
            for c in range(n_c):
                data.append({
                    'group': group_for_each[i],
                    'feature': names[c],
                    'importance': importance_matrix[i, c],
                })
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        order = [g for g in ['Low', 'Medium', 'High'] if g in df['group'].values]
        # 统一 Low/Medium/High 颜色，与其他图保持一致
        group_palette = {'Low': 'C0', 'Medium': 'C1', 'High': 'C2'}
        sns.boxplot(
            data=df,
            x='feature',
            y='importance',
            hue='group',
            hue_order=order,
            palette=group_palette,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Mean |attribution|')
        ax.set_xlabel('Feature')
        # 主标题加粗
        ax.set_title(
            'Feature attribution by population-density group (Integrated Gradients)')
        # Legend 标题首字母大写，并保持顺序和颜色
        legend = ax.legend(title='Group', loc='best')
        plt.tight_layout()
        out_path = Path(output_dir) / 'xai_attribution_boxplot_by_pop_group.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 箱线图已保存: {out_path}")

    def _plot_top5_features_by_group(self, results, output_dir):
        """三组各自 Top-5 特征重要性条形图对比。"""
        importance_matrix = results['importance_matrix']
        group_for_each = results['group_for_each']
        names = results['feature_names']
        order = [g for g in ['Low', 'Medium', 'High'] if (group_for_each == g).any()]
        fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
        if len(order) == 1:
            axes = [axes]

        # 先按 High 组的量级确定统一的 y 轴上限（若没有 High，则用所有组的最大值）
        global_max = 0.0
        for group in order:
            mask = group_for_each == group
            mean_imp = importance_matrix[mask].mean(axis=0)
            top5 = np.sort(mean_imp)[-5:]
            if group == 'High':
                global_max = max(global_max, top5.max())
        if global_max == 0.0:
            for group in order:
                mask = group_for_each == group
                mean_imp = importance_matrix[mask].mean(axis=0)
                top5 = np.sort(mean_imp)[-5:]
                global_max = max(global_max, top5.max())
        ylim_max = global_max * 1.1 if global_max > 0 else None

        for i, (ax, group) in enumerate(zip(axes, order)):
            mask = group_for_each == group
            mean_imp = importance_matrix[mask].mean(axis=0)
            idx = np.argsort(mean_imp)[::-1][:5]
            # 横纵坐标对调：x 为特征，y 为重要性
            ax.bar([names[i] for i in idx], mean_imp[idx],
                   color={'Low': 'C0', 'Medium': 'C1', 'High': 'C2'}.get(group, 'C0'))
            if i == 0:
                ax.set_ylabel('Mean |attribution|')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Feature')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            # 统一 y 轴范围，按照 High 组的最大值放大 10%
            if ylim_max is not None:
                ax.set_ylim(0, ylim_max)
            # 子图标题字号调大，便于论文阅读
            ax.set_title(f'{group} (n={mask.sum()})', fontsize=18)
        # 调整 suptitle：论文中一般不用加粗
        plt.suptitle('Top-5 feature importance by group', y=0.98)
        plt.tight_layout()
        out_path = Path(output_dir) / 'xai_top5_features_by_pop_group.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Top-5 图已保存: {out_path}")

    def perform_statistical_tests(self, y_true, y_pred, y_proba, sensitive_attr, group_metrics):
        """
        执行统计显著性检验，多方证实不公平性
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            group_metrics: 各组性能指标
            
        Returns:
            dict: 统计检验结果
        """
        print("\n" + "=" * 80)
        print("📊 统计显著性检验")
        print("=" * 80)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        if len(unique_groups) < 2:
            return {}
        
        test_results = {}
        
        # 1. 卡方检验：各组预测分布是否显著不同
        print("\n1️⃣  卡方检验（Chi-square Test）")
        print("   检验各组预测分布是否显著不同")
        contingency_tables = {}
        
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                mask1 = sensitive_attr == group1
                mask2 = sensitive_attr == group2
                
                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue
                
                pred1 = y_pred[mask1]
                pred2 = y_pred[mask2]
                
                # 构建列联表
                table = np.array([
                    [np.sum(pred1 == 0), np.sum(pred1 == 1)],
                    [np.sum(pred2 == 0), np.sum(pred2 == 1)]
                ])
                
                try:
                    chi2, p_value, dof, expected = chi2_contingency(table)
                    
                    key = f"{group1}_vs_{group2}"
                    contingency_tables[key] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'dof': dof,
                        'table': table.tolist(),
                        'significant': p_value < 0.05
                    }
                    
                    significance = "***显著***" if p_value < 0.001 else "**显著**" if p_value < 0.01 else "*显著*" if p_value < 0.05 else "不显著"
                    print(f"   {group1} vs {group2}:")
                    print(f"     卡方统计量: {chi2:.4f}")
                    print(f"     p值: {p_value:.6f} {significance}")
                    print(f"     自由度: {dof}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}: 计算失败 - {e}")
        
        test_results['chi_square'] = contingency_tables
        
        # 2. 召回率差异的Bootstrap置信区间
        print("\n2️⃣  Bootstrap置信区间（Bootstrap Confidence Intervals）")
        print("   估计召回率差异的95%置信区间")
        
        bootstrap_results = {}
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                mask1 = sensitive_attr == group1
                mask2 = sensitive_attr == group2
                
                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue
                
                y_true1 = y_true[mask1]
                y_pred1 = y_pred[mask1]
                y_true2 = y_true[mask2]
                y_pred2 = y_pred[mask2]
                
                def recall_diff_statistic(data1, data2):
                    """计算召回率差异"""
                    y_t1, y_p1 = data1
                    y_t2, y_p2 = data2
                    recall1 = recall_score(y_t1, y_p1, zero_division=0)
                    recall2 = recall_score(y_t2, y_p2, zero_division=0)
                    return recall2 - recall1
                
                try:
                    # Bootstrap置信区间（手动实现，确保兼容性）
                    n_resamples = 10000
                    differences = []
                    
                    for _ in range(n_resamples):
                        # 有放回抽样
                        indices1 = np.random.choice(len(y_true1), len(y_true1), replace=True)
                        indices2 = np.random.choice(len(y_true2), len(y_true2), replace=True)
                        
                        y_t1_resample = y_true1[indices1]
                        y_p1_resample = y_pred1[indices1]
                        y_t2_resample = y_true2[indices2]
                        y_p2_resample = y_pred2[indices2]
                        
                        recall1 = recall_score(y_t1_resample, y_p1_resample, zero_division=0)
                        recall2 = recall_score(y_t2_resample, y_p2_resample, zero_division=0)
                        differences.append(recall2 - recall1)
                    
                    differences = np.array(differences)
                    ci_lower = np.percentile(differences, 2.5)
                    ci_upper = np.percentile(differences, 97.5)
                    
                    # 计算实际差异
                    actual_diff = recall_score(y_true2, y_pred2, zero_division=0) - recall_score(y_true1, y_pred1, zero_division=0)
                    
                    key = f"{group1}_vs_{group2}"
                    bootstrap_results[key] = {
                        'actual_difference': actual_diff,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'excludes_zero': ci_lower > 0 or ci_upper < 0
                    }
                    
                    excludes_zero = "***不包含0***" if bootstrap_results[key]['excludes_zero'] else "包含0"
                    print(f"   {group1} vs {group2} 召回率差异:")
                    print(f"     实际差异: {actual_diff:.4f}")
                    print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] {excludes_zero}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}: Bootstrap计算失败 - {e}")
        
        test_results['bootstrap'] = bootstrap_results
        
        # 3. Cohen's h效应量（用于比例差异）
        print("\n3️⃣  Cohen's h效应量（Effect Size）")
        print("   衡量召回率差异的效应大小")
        
        def cohens_h(p1, p2):
            """计算两个比例的Cohen's h"""
            return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        cohens_h_results = {}
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                if group1 not in group_metrics or group2 not in group_metrics:
                    continue
                
                recall1 = group_metrics[group1].get('recall', 0)
                recall2 = group_metrics[group2].get('recall', 0)
                
                if recall1 == 0 and recall2 == 0:
                    continue
                
                h = cohens_h(recall2, recall1)
                
                # 解释效应量
                if abs(h) < 0.2:
                    effect_size = "小"
                elif abs(h) < 0.5:
                    effect_size = "小到中"
                elif abs(h) < 0.8:
                    effect_size = "中到大"
                else:
                    effect_size = "大"
                
                key = f"{group1}_vs_{group2}"
                cohens_h_results[key] = {
                    'h': h,
                    'effect_size': effect_size,
                    'recall1': recall1,
                    'recall2': recall2
                }
                
                print(f"   {group1} vs {group2}:")
                print(f"     Cohen's h: {h:.4f} ({effect_size}效应)")
                print(f"     召回率: {group1}={recall1:.4f}, {group2}={recall2:.4f}")
        
        test_results['cohens_h'] = cohens_h_results
        
        # 4. Mann-Whitney U检验：各组概率分布是否显著不同
        print("\n4️⃣  Mann-Whitney U检验（Mann-Whitney U Test）")
        print("   检验各组预测概率分布是否显著不同")
        
        mw_results = {}
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                mask1 = sensitive_attr == group1
                mask2 = sensitive_attr == group2
                
                if mask1.sum() == 0 or mask2.sum() == 0 or y_proba is None:
                    continue
                
                prob1 = y_proba[mask1]
                prob2 = y_proba[mask2]
                
                try:
                    statistic, p_value = mannwhitneyu(prob1, prob2, alternative='two-sided')
                    
                    significance = "***显著***" if p_value < 0.001 else "**显著**" if p_value < 0.01 else "*显著*" if p_value < 0.05 else "不显著"
                    
                    key = f"{group1}_vs_{group2}"
                    mw_results[key] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_prob1': np.mean(prob1),
                        'mean_prob2': np.mean(prob2)
                    }
                    
                    print(f"   {group1} vs {group2}:")
                    print(f"     U统计量: {statistic:.2f}")
                    print(f"     p值: {p_value:.6f} {significance}")
                    print(f"     概率均值: {group1}={np.mean(prob1):.4f}, {group2}={np.mean(prob2):.4f}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}: 计算失败 - {e}")
        
        test_results['mann_whitney'] = mw_results
        
        # 5. 混淆矩阵详细分析
        print("\n5️⃣  混淆矩阵详细分析（Confusion Matrix Analysis）")
        print("   分析各组误分类模式")
        
        cm_analysis = {}
        for group in unique_groups:
            if group not in group_metrics:
                continue
            
            metrics = group_metrics[group]
            if 'tp' not in metrics:
                continue
            
            tp = metrics['tp']
            fp = metrics['fp']
            tn = metrics['tn']
            fn = metrics['fn']
            
            total = tp + fp + tn + fn
            total_positive = tp + fn  # 真实正类数
            total_negative = tn + fp  # 真实负类数
            
            # 计算漏检率和误报率
            false_negative_rate = fn / total_positive if total_positive > 0 else 0
            false_positive_rate = fp / total_negative if total_negative > 0 else 0
            
            cm_analysis[group] = {
                'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
                'false_negative_rate': false_negative_rate,
                'false_positive_rate': false_positive_rate,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'missed_fires': fn,  # 漏检的火灾数
                'false_alarms': fp   # 误报数
            }
            
            print(f"   {group}组:")
            print(f"     混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"     漏检率 (FNR): {false_negative_rate:.4f} ({fn}/{total_positive})")
            print(f"     误报率 (FPR): {false_positive_rate:.4f} ({fp}/{total_negative})")
            print(f"     漏检火灾数: {fn:,}")
            print(f"     误报数: {fp:,}")
        
        test_results['confusion_matrix_analysis'] = cm_analysis
        
        # 6. 总结统计显著性
        print("\n" + "=" * 80)
        print("📊 统计显著性总结")
        print("=" * 80)
        
        significant_tests = []
        if test_results.get('chi_square'):
            for key, result in test_results['chi_square'].items():
                if result['p_value'] < 0.05:
                    significant_tests.append(f"卡方检验 ({key}): p={result['p_value']:.6f}")
        
        if test_results.get('mann_whitney'):
            for key, result in test_results['mann_whitney'].items():
                if result['p_value'] < 0.05:
                    significant_tests.append(f"Mann-Whitney U ({key}): p={result['p_value']:.6f}")
        
        if significant_tests:
            print("✅ 发现显著差异的检验:")
            for test in significant_tests:
                print(f"   - {test}")
        else:
            print("⚠️  未发现显著差异（可能需要更大样本量）")
        
        return test_results
    
    def optimize_group_specific_thresholds(self, y_true, y_proba, sensitive_attr, 
                                          metric='f1', threshold_range=(0.1, 0.9), 
                                          num_thresholds=200, min_precision=0.3, 
                                          min_recall=0.3, use_validation_split=False,
                                          validation_ratio=0.2):
        """
        为每个组优化特定阈值
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            metric: 优化的指标 ('f1', 'recall', 'balanced_accuracy', 'balanced_f1')
            threshold_range: 阈值搜索范围
            num_thresholds: 搜索的阈值数量
            min_precision: 最小精确率要求（用于balanced_f1）
            min_recall: 最小召回率要求（用于balanced_f1）
            
        Returns:
            dict: 包含每个组的最优阈值和优化后的性能指标
        """
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("🔧 组特定阈值优化")
            print("=" * 80)
        
        # 如果使用验证集分割，先分割数据
        if use_validation_split:
            if not quiet:
                print(f"📊 使用验证集分割 (验证集比例: {validation_ratio})")
            indices = np.arange(len(y_true))
            train_idx, val_idx = train_test_split(
                indices, test_size=validation_ratio, 
                stratify=y_true, random_state=42
            )
            y_true_train = y_true[train_idx]
            y_proba_train = y_proba[train_idx]
            sensitive_train = sensitive_attr[train_idx]
            y_true_val = y_true[val_idx]
            y_proba_val = y_proba[val_idx]
            sensitive_val = sensitive_attr[val_idx]
        else:
            if not quiet:
                print("⚠️  在全部数据上优化（可能过拟合测试集）")
            y_true_train = y_true
            y_proba_train = y_proba
            sensitive_train = sensitive_attr
            y_true_val = y_true
            y_proba_val = y_proba
            sensitive_val = sensitive_attr
        
        unique_groups = np.unique(sensitive_train)
        unique_groups = [g for g in unique_groups if g != 'Unknown']
        
        if len(unique_groups) < 2:
            print("⚠️  分组少于2组，无法进行阈值优化")
            return {}
        
        optimal_thresholds = {}
        optimized_metrics = {}
        
        # 对于特殊方法，使用曲线优化
        use_curve_optimization = metric in ['youden', 'pr_optimal', 'f2']
        
        if not use_curve_optimization:
            thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        for group in unique_groups:
            mask_train = sensitive_train == group
            mask_val = sensitive_val == group
            
            if mask_train.sum() == 0:
                continue
            
            group_y_true_train = y_true_train[mask_train]
            group_y_proba_train = y_proba_train[mask_train]
            group_y_true_val = y_true_val[mask_val]
            group_y_proba_val = y_proba_val[mask_val]
            
            best_score = -1
            best_threshold = 0.5
            best_metrics = None
            
            if not quiet:
                print(f"\n🔍 优化 {group} 组阈值 (优化指标: {metric})...")
            
            # 特殊优化方法（基于曲线）
            if metric == 'youden':
                fpr, tpr, thresholds_curve = roc_curve(group_y_true_train, group_y_proba_train, pos_label=1)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                best_threshold = thresholds_curve[optimal_idx] if optimal_idx < len(thresholds_curve) else 0.5
            elif metric == 'pr_optimal':
                precision, recall, thresholds_curve = precision_recall_curve(group_y_true_train, group_y_proba_train)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                best_threshold = thresholds_curve[optimal_idx] if optimal_idx < len(thresholds_curve) else 0.5
            elif metric == 'f2':
                precision, recall, thresholds_curve = precision_recall_curve(group_y_true_train, group_y_proba_train)
                f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-10)
                optimal_idx = np.argmax(f2_scores)
                best_threshold = thresholds_curve[optimal_idx] if optimal_idx < len(thresholds_curve) else 0.5
            else:
                # 网格搜索（在训练集上优化）
                for threshold in thresholds:
                    y_pred_thresh = (group_y_proba_train > threshold).astype(int)
                    
                    # 计算当前阈值下的指标（在训练集上）
                    current_metrics_train = self.calculate_performance_metrics(
                        group_y_true_train, y_pred_thresh, group_y_proba_train
                    )
                    current_precision = current_metrics_train['precision']
                    current_recall = current_metrics_train['recall']
                    
                    # 根据metric选择评分函数
                    if metric == 'f1':
                        score = current_metrics_train['f1']
                    elif metric == 'recall':
                        score = current_recall
                    elif metric == 'balanced_accuracy':
                        # 平衡准确率：考虑正负类比例
                        # 同时应用约束：精确率和召回率都要在合理范围内
                        cm = confusion_matrix(group_y_true_train, y_pred_thresh)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            balanced_acc = (sensitivity + specificity) / 2
                            
                            # 应用约束：如果精确率或召回率超出合理范围，惩罚分数
                            # 合理范围：精确率 [0.4, 0.95]，召回率 [0.3, 0.8]
                            precision_penalty = 0
                            recall_penalty = 0
                            
                            if current_precision < min_precision:
                                precision_penalty = (min_precision - current_precision) * 0.5
                            elif current_precision > 0.95:
                                precision_penalty = (current_precision - 0.95) * 0.3
                            
                            if current_recall < min_recall:
                                recall_penalty = (min_recall - current_recall) * 0.5
                            elif current_recall > 0.8:
                                recall_penalty = (current_recall - 0.8) * 0.3
                            
                            score = balanced_acc - precision_penalty - recall_penalty
                        else:
                            score = current_metrics_train['accuracy']
                    elif metric == 'balanced_f1':
                        # 平衡F1：要求精确率和召回率都达到最小值
                        if current_precision >= min_precision and current_recall >= min_recall:
                            # 如果满足约束，使用F1分数
                            score = current_metrics_train['f1']
                        else:
                            # 如果不满足约束，惩罚分数
                            precision_penalty = max(0, min_precision - current_precision) * 2
                            recall_penalty = max(0, min_recall - current_recall) * 2
                            score = current_metrics_train['f1'] - precision_penalty - recall_penalty
                    else:
                        score = current_metrics_train['f1']
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            # 在验证集上评估最优阈值
            y_pred_val = (group_y_proba_val > best_threshold).astype(int)
            best_metrics = self.calculate_performance_metrics(
                group_y_true_val, y_pred_val, group_y_proba_val
            )
            
            optimal_thresholds[group] = best_threshold
            optimized_metrics[group] = best_metrics
            
            if not quiet:
                print(f"   ✅ 最优阈值: {best_threshold:.4f}")
                print(f"   📊 {'验证集' if use_validation_split else '优化后'}性能:")
                print(f"      准确率: {best_metrics['accuracy']:.4f}")
                print(f"      精确率: {best_metrics['precision']:.4f}")
                print(f"      召回率: {best_metrics['recall']:.4f}")
                print(f"      F1分数: {best_metrics['f1']:.4f}")
                if best_metrics.get('auc') is not None:
                    print(f"      AUC: {best_metrics['auc']:.4f}")
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'optimized_metrics': optimized_metrics,
            'use_validation_split': use_validation_split
        }
    
    def optimize_multi_objective(self, y_true, y_proba, sensitive_attr,
                                 performance_weight=0.7, fairness_weight=0.3,
                                 performance_metric='f1', threshold_range=(0.1, 0.9),
                                 num_thresholds=200, use_validation_split=False,
                                 validation_ratio=0.2):
        """
        多目标优化：同时优化性能和公平性
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            performance_weight: 性能权重
            fairness_weight: 公平性权重
            performance_metric: 性能指标 ('f1', 'balanced_accuracy')
            threshold_range: 阈值搜索范围
            num_thresholds: 搜索的阈值数量
            use_validation_split: 是否使用验证集分割
            validation_ratio: 验证集比例
            
        Returns:
            dict: 包含每个组的最优阈值和优化后的性能指标
        """
        print("\n" + "=" * 80)
        print("🎯 多目标优化（性能 + 公平性）")
        print("=" * 80)
        print(f"   性能权重: {performance_weight}, 公平性权重: {fairness_weight}")
        
        # 数据分割
        if use_validation_split:
            indices = np.arange(len(y_true))
            train_idx, val_idx = train_test_split(
                indices, test_size=validation_ratio, 
                stratify=y_true, random_state=42
            )
            y_true_train = y_true[train_idx]
            y_proba_train = y_proba[train_idx]
            sensitive_train = sensitive_attr[train_idx]
            y_true_val = y_true[val_idx]
            y_proba_val = y_proba[val_idx]
            sensitive_val = sensitive_attr[val_idx]
        else:
            y_true_train = y_true
            y_proba_train = y_proba
            sensitive_train = sensitive_attr
            y_true_val = y_true
            y_proba_val = y_proba
            sensitive_val = sensitive_attr
        
        unique_groups = np.unique(sensitive_train)
        unique_groups = [g for g in unique_groups if g != 'Unknown']
        
        if len(unique_groups) < 2:
            print("⚠️  分组少于2组，无法进行多目标优化")
            return {}
        
        optimal_thresholds = {}
        optimized_metrics = {}
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        for group in unique_groups:
            mask_train = sensitive_train == group
            mask_val = sensitive_val == group
            
            if mask_train.sum() == 0:
                continue
            
            group_y_true_train = y_true_train[mask_train]
            group_y_proba_train = y_proba_train[mask_train]
            group_y_true_val = y_true_val[mask_val]
            group_y_proba_val = y_proba_val[mask_val]
            
            best_score = -1
            best_threshold = 0.5
            
            print(f"\n🔍 优化 {group} 组阈值 (多目标优化)...")
            
            for threshold in thresholds:
                y_pred_train = (group_y_proba_train > threshold).astype(int)
                
                # 性能指标
                if performance_metric == 'f1':
                    perf_score = f1_score(group_y_true_train, y_pred_train)
                elif performance_metric == 'balanced_accuracy':
                    perf_score = balanced_accuracy_score(group_y_true_train, y_pred_train)
                else:
                    perf_score = f1_score(group_y_true_train, y_pred_train)
                
                # 公平性指标（需要所有组的数据）
                # 这里简化处理：只考虑当前组的预测分布
                # 实际应用中需要计算组间公平性
                fairness_score = 1.0  # 占位符，实际需要计算组间差异
                
                # 组合分数
                combined_score = (performance_weight * perf_score + 
                                fairness_weight * fairness_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
            
            # 在验证集上评估
            y_pred_val = (group_y_proba_val > best_threshold).astype(int)
            best_metrics = self.calculate_performance_metrics(
                group_y_true_val, y_pred_val, group_y_proba_val
            )
            
            optimal_thresholds[group] = best_threshold
            optimized_metrics[group] = best_metrics
            
            print(f"   ✅ 最优阈值: {best_threshold:.4f}")
            print(f"   📊 {'验证集' if use_validation_split else '优化后'}性能:")
            print(f"      F1分数: {best_metrics['f1']:.4f}")
            print(f"      精确率: {best_metrics['precision']:.4f}")
            print(f"      召回率: {best_metrics['recall']:.4f}")
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'optimized_metrics': optimized_metrics,
            'use_validation_split': use_validation_split
        }
    
    def compare_threshold_methods(self, y_true, y_proba, sensitive_attr,
                                  methods=['f1', 'balanced_accuracy', 'youden', 'f2'],
                                  threshold_range=(0.1, 0.9), num_thresholds=200,
                                  use_validation_split=True, validation_ratio=0.2):
        """
        对比不同阈值优化方法
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            methods: 要对比的方法列表
            threshold_range: 阈值搜索范围
            num_thresholds: 搜索的阈值数量
            use_validation_split: 是否使用验证集分割
            validation_ratio: 验证集比例
            
        Returns:
            dict: 各方法的优化结果对比
        """
        print("\n" + "=" * 80)
        print("📊 阈值优化方法对比")
        print("=" * 80)
        
        results = {}
        
        for method in methods:
            print(f"\n{'='*80}")
            print(f"🔍 方法: {method}")
            print(f"{'='*80}")
            
            if method in ['youden', 'pr_optimal', 'f2']:
                # 这些方法需要特殊处理
                result = self.optimize_group_specific_thresholds(
                    y_true, y_proba, sensitive_attr,
                    metric=method,
                    threshold_range=threshold_range,
                    num_thresholds=num_thresholds,
                    use_validation_split=use_validation_split,
                    validation_ratio=validation_ratio
                )
            else:
                result = self.optimize_group_specific_thresholds(
                    y_true, y_proba, sensitive_attr,
                    metric=method,
                    threshold_range=threshold_range,
                    num_thresholds=num_thresholds,
                    use_validation_split=use_validation_split,
                    validation_ratio=validation_ratio
                )
            
            results[method] = result
        
        # 打印对比结果
        print("\n" + "=" * 80)
        print("📊 方法对比总结")
        print("=" * 80)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        
        for group in unique_groups:
            print(f"\n{group} 组:")
            print(f"{'方法':<20} {'阈值':<10} {'F1':<10} {'精确率':<10} {'召回率':<10}")
            print("-" * 60)
            
            for method, result in results.items():
                if result and 'optimal_thresholds' in result and group in result['optimal_thresholds']:
                    threshold = result['optimal_thresholds'][group]
                    metrics = result['optimized_metrics'].get(group, {})
                    f1 = metrics.get('f1', 0)
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    
                    print(f"{method:<20} {threshold:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")
        
        return results
    
    def apply_group_specific_thresholds(self, y_proba, sensitive_attr, optimal_thresholds):
        """
        应用组特定阈值生成预测
        
        Args:
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            optimal_thresholds: 每个组的最优阈值字典
            
        Returns:
            np.array: 优化后的预测标签
        """
        y_pred_optimized = np.zeros_like(y_proba, dtype=int)
        
        for group, threshold in optimal_thresholds.items():
            mask = sensitive_attr == group
            y_pred_optimized[mask] = (y_proba[mask] > threshold).astype(int)
        
        # 处理Unknown组（使用默认阈值0.5）
        unknown_mask = sensitive_attr == 'Unknown'
        if unknown_mask.sum() > 0:
            y_pred_optimized[unknown_mask] = (y_proba[unknown_mask] > 0.5).astype(int)
        
        return y_pred_optimized
    
    def plot_threshold_tradeoff_curves(self, y_true, y_proba, sensitive_attr,
                                      output_dir='fairness_results',
                                      threshold_range=(0.1, 0.9), num_thresholds=50,
                                      optimal_thresholds=None):
        """
        绘制阈值-性能-公平性权衡曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            sensitive_attr: 敏感属性数组
            output_dir: 输出目录
            threshold_range: 阈值范围
            num_thresholds: 阈值数量
            optimal_thresholds: 每个组的最优阈值字典（如 {'Low': 0.40, 'Medium': 0.42, 'High': 0.47}）
        """
        print("\n" + "=" * 80)
        print("📊 绘制阈值权衡曲线")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        if len(unique_groups) < 2:
            print("⚠️  分组少于2组，无法绘制权衡曲线")
            return
        ordered_groups = [g for g in ['Low', 'Medium', 'High'] if g in unique_groups] + [
            g for g in unique_groups if g not in ['Low', 'Medium', 'High']
        ]
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        # 存储每个阈值下的指标
        metrics_by_threshold = {
            'thresholds': thresholds,
            'overall': {
                'f1': [], 'precision': [], 'recall': [],
                'accuracy': [], 'equalized_odds_diff': []
            },
            'groups': {group: {
                'f1': [], 'precision': [], 'recall': [], 'accuracy': []
            } for group in unique_groups}
        }
        
        print("计算不同阈值下的指标...")
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            # 整体指标
            overall_f1 = f1_score(y_true, y_pred)
            overall_precision = precision_score(y_true, y_pred, zero_division=0)
            overall_recall = recall_score(y_true, y_pred, zero_division=0)
            overall_accuracy = accuracy_score(y_true, y_pred)
            
            metrics_by_threshold['overall']['f1'].append(overall_f1)
            metrics_by_threshold['overall']['precision'].append(overall_precision)
            metrics_by_threshold['overall']['recall'].append(overall_recall)
            metrics_by_threshold['overall']['accuracy'].append(overall_accuracy)
            
            # 公平性指标
            if HAS_FAIRLEARN:
                try:
                    eo_diff = fl_metrics.equalized_odds_difference(
                        y_true, y_pred, sensitive_features=sensitive_attr
                    )
                    metrics_by_threshold['overall']['equalized_odds_diff'].append(eo_diff)
                except:
                    metrics_by_threshold['overall']['equalized_odds_diff'].append(0)
            else:
                metrics_by_threshold['overall']['equalized_odds_diff'].append(0)
            
            # 各组指标
            for group in unique_groups:
                mask = sensitive_attr == group
                if mask.sum() > 0:
                    group_y_true = y_true[mask]
                    group_y_pred = y_pred[mask]
                    
                    group_f1 = f1_score(group_y_true, group_y_pred, zero_division=0)
                    group_precision = precision_score(group_y_true, group_y_pred, zero_division=0)
                    group_recall = recall_score(group_y_true, group_y_pred, zero_division=0)
                    group_accuracy = accuracy_score(group_y_true, group_y_pred)
                    
                    metrics_by_threshold['groups'][group]['f1'].append(group_f1)
                    metrics_by_threshold['groups'][group]['precision'].append(group_precision)
                    metrics_by_threshold['groups'][group]['recall'].append(group_recall)
                    metrics_by_threshold['groups'][group]['accuracy'].append(group_accuracy)
        
        # 绘制图表（去掉全局大标题，论文中由外层排版控制总标题）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1 vs 阈值
        ax1 = axes[0, 0]
        ax1.plot(
            thresholds,
            metrics_by_threshold['overall']['f1'],
            'k-',
            linewidth=2.0,
            label='Overall',
            marker='o',
            markersize=3,
        )
        group_colors = {'Low': 'C0', 'Medium': 'C1', 'High': 'C2'}
        for group in ordered_groups:
            ax1.plot(
                thresholds,
                metrics_by_threshold['groups'][group]['f1'],
                linewidth=2.0,
                label=f'{group} Group',
                alpha=0.7,
                color=group_colors.get(group),
                marker='o',
                markersize=3,
            )
        ax1.set_xlabel('Threshold', fontsize=20)
        ax1.set_ylabel('F1 score', fontsize=20)
        # 子图标题使用 sentence case
        ax1.set_title('F1 score vs threshold', fontsize=24)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label=r'$\tau_0$ = 0.5')
        # 标注组特定最优阈值 tau_g（颜色与 ROC 图一致：Low/Medium/High -> C0/C1/C2）
        if optimal_thresholds:
            for group in ['Low', 'Medium', 'High']:
                if group in optimal_thresholds:
                    tau_g = optimal_thresholds[group]
                    ax1.axvline(
                        x=tau_g,
                        color=group_colors.get(group),
                        linestyle=':',
                        linewidth=2.0,
                        alpha=0.9,
                        label=rf'{group} $\tau_g$={tau_g:.4f}'
                    )
        ax1.legend(fontsize=12, loc='lower right')
        
        # 2. 精确率 vs 召回率（PR曲线）
        ax2 = axes[0, 1]
        ax2.plot(
            metrics_by_threshold['overall']['recall'],
            metrics_by_threshold['overall']['precision'],
            'k-',
            linewidth=2.0,
            label='Overall',
            marker='o',
            markersize=3,
        )
        for group in ordered_groups:
            ax2.plot(
                metrics_by_threshold['groups'][group]['recall'],
                metrics_by_threshold['groups'][group]['precision'],
                linewidth=2.0,
                label=f'{group} Group',
                alpha=0.7,
                color=group_colors.get(group),
                marker='o',
                markersize=3,
            )
        ax2.set_xlabel('Recall', fontsize=20)
        ax2.set_ylabel('Precision', fontsize=20)
        ax2.set_title('Precision–recall trade-off curve', fontsize=24)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. 公平性指标 vs 阈值
        ax3 = axes[1, 0]
        if HAS_FAIRLEARN and len(metrics_by_threshold['overall']['equalized_odds_diff']) > 0:
            ax3.plot(thresholds, metrics_by_threshold['overall']['equalized_odds_diff'],
                    'b-', linewidth=2.0, label='Equalized Odds Difference', marker='o', markersize=3)
            ax3.axhline(
                y=0.05,
                color='purple',
                linestyle='-.',
                alpha=0.8,
                linewidth=1.8,
                label='Fairness threshold (0.05)'
            )
        ax3.set_xlabel('Threshold', fontsize=20)
        ax3.set_ylabel('Equalized odds difference', fontsize=20)
        ax3.set_title('Fairness metric vs threshold', fontsize=24)
        ax3.tick_params(axis='both', labelsize=18)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label=r'$\tau_0$ = 0.5')
        if optimal_thresholds:
            for group in ['Low', 'Medium', 'High']:
                if group in optimal_thresholds:
                    tau_g = optimal_thresholds[group]
                    ax3.axvline(
                        x=tau_g,
                        color=group_colors.get(group),
                        linestyle=':',
                        linewidth=2.0,
                        alpha=0.9,
                        label=rf'{group} $\tau_g$={tau_g:.4f}'
                    )
        ax3.legend(fontsize=12)
        
        # 4. 性能 vs 公平性权衡
        ax4 = axes[1, 1]
        if HAS_FAIRLEARN and len(metrics_by_threshold['overall']['equalized_odds_diff']) > 0:
            scatter = ax4.scatter(metrics_by_threshold['overall']['equalized_odds_diff'],
                                 metrics_by_threshold['overall']['f1'],
                                 c=thresholds, cmap='viridis', 
                                 s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Equalized odds difference (lower is better)', fontsize=20)
            ax4.set_ylabel('F1 score (higher is better)', fontsize=20)
            ax4.set_title('Performance–fairness trade-off scatter plot', fontsize=24)
            ax4.tick_params(axis='both', labelsize=18)
            ax4.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax4, label='Threshold')
            cbar.ax.tick_params(labelsize=12)
            
            # 标记默认阈值0.5的点
            default_idx = np.argmin(np.abs(thresholds - 0.5))
            ax4.scatter(metrics_by_threshold['overall']['equalized_odds_diff'][default_idx],
                       metrics_by_threshold['overall']['f1'][default_idx],
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidth=1, label=r'$\tau_0$ = 0.5', zorder=5)
            # 标记组特定最优阈值组合（optimised tau_g）对应的整体性能-公平性点
            if optimal_thresholds:
                y_pred_opt = (y_proba > 0.5).astype(int)
                for group, tau_g in optimal_thresholds.items():
                    g_mask = sensitive_attr == group
                    if g_mask.sum() > 0:
                        y_pred_opt[g_mask] = (y_proba[g_mask] > tau_g).astype(int)
                try:
                    eo_opt = fl_metrics.equalized_odds_difference(
                        y_true, y_pred_opt, sensitive_features=sensitive_attr
                    )
                except Exception:
                    eo_opt = None
                f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
                if eo_opt is not None:
                    ax4.scatter(
                        eo_opt,
                        f1_opt,
                        c='#B07AA1',
                        s=120,
                        marker='D',
                        edgecolors='black',
                        linewidth=1.0,
                        label=r'Optimised $\tau_g$',
                        zorder=6
                    )
            ax4.legend()
        
        plt.tight_layout()
        
        output_file = output_path / 'threshold_tradeoff_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 阈值权衡曲线已保存至: {output_file}")
        plt.close()
        
        return metrics_by_threshold
    
    def plot_roc_curves(self, output_dir='fairness_results', title=None):
        """
        绘制 ROC 曲线对比图：整体 + 不同分组方式。

        - 人口密度分组 (Low/Medium/High) -> roc_curve_overall_and_pop_group.png
        - 人均GDP分组 (Low/Medium/High) -> roc_curve_overall_and_gdp_group.png
        - 大洲分组 (Low/Medium/High)    -> roc_curve_overall_and_continent_group.png

        用于论文 \"Overall performance and initial unfairness\" 小节及扩展结果。
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        if self.sensitive_attributes is None or self.true_labels is None or self.probabilities is None:
            print("⚠️  缺少预测结果或敏感属性，跳过 ROC 曲线绘制")
            return

        # 统一 ROC 图文字大小（论文场景：清晰但不过大）
        roc_fs = {
            'title': 16,
            'label': 14,
            'tick': 12,
            'legend': 11,
        }

        def _draw_roc_on_ax(ax, y_true, y_proba, groups, attr_key, verbose=False):
            """在给定 ax 上绘制 Overall + 分组的 ROC，不设置标题（由调用方设置）。"""
            fpr_all, tpr_all, _ = roc_curve(y_true, y_proba, pos_label=1, drop_intermediate=False)
            auc_all = roc_auc_score(y_true, y_proba)
            ax.plot(fpr_all, tpr_all, color='black', lw=2, linestyle='--',
                    label=f'Overall (AUC = {auc_all:.3f})')
            if verbose and attr_key == 'pop_group':
                n_unique_all = len(np.unique(y_proba))
                print(f"   [ROC 诊断] Overall: {len(fpr_all)} 个阈值点, {n_unique_all} 个不同概率值")

            colors = {'Low': 'C0', 'Medium': 'C1', 'High': 'C2'}
            unique_groups = [g for g in ['Low', 'Medium', 'High'] if g in groups]
            if not unique_groups:
                unique_groups = sorted(np.unique(groups))
            for group in unique_groups:
                g_mask = groups == group
                if g_mask.sum() == 0:
                    continue
                y_g = y_true[g_mask]
                p_g = y_proba[g_mask]
                if np.unique(y_g).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_g, p_g, pos_label=1, drop_intermediate=False)
                auc_g = roc_auc_score(y_g, p_g)
                color = colors.get(group)
                ax.plot(fpr, tpr, lw=2, color=color, label=f'{group} (AUC = {auc_g:.3f})')
                if verbose and attr_key == 'pop_group':
                    print(f"   [ROC 诊断] {group}: {len(fpr)} 个阈值点, {len(np.unique(p_g))} 个不同概率值")

            ax.plot([0, 1], [0, 1], 'k-', lw=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('False Positive Rate', fontsize=roc_fs['label'])
            ax.set_ylabel('True Positive Rate', fontsize=roc_fs['label'])
            ax.tick_params(axis='both', labelsize=roc_fs['tick'])
            ax.legend(loc='lower right', fontsize=roc_fs['legend'])
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        def _get_roc_data(attr_key):
            """返回 (y_true, y_proba, groups) 或 None（无效时）。"""
            if attr_key not in self.sensitive_attributes:
                return None
            attr_values = np.array(self.sensitive_attributes[attr_key])
            mask = attr_values != 'Unknown'
            if mask.sum() == 0:
                return None
            y_true = np.asarray(self.true_labels)[mask]
            y_proba = np.asarray(self.probabilities)[mask]
            groups = attr_values[mask]
            if np.unique(y_true).size < 2:
                return None
            return (y_true, y_proba, groups)

        def _plot_for_group(attr_key, filename, default_title):
            """单张 ROC 图：标题去掉 'ROC curves:' 且不加粗（供论文子图风格一致）。"""
            if attr_key not in self.sensitive_attributes:
                return
            attr_values = np.array(self.sensitive_attributes[attr_key])
            mask = attr_values != 'Unknown'
            if mask.sum() == 0:
                print(f"⚠️  无有效分组 ({attr_key})，跳过 {filename} 绘制")
                return
            y_true = np.asarray(self.true_labels)[mask]
            y_proba = np.asarray(self.probabilities)[mask]
            groups = attr_values[mask]
            if np.unique(y_true).size < 2:
                print(f"⚠️  {attr_key} 下整体标签不够两类，跳过 ROC 绘制")
                return

            n_unique_all = len(np.unique(y_proba))
            if n_unique_all <= 2:
                print(f"⚠️  [ROC] {attr_key} 概率仅有 {n_unique_all} 个不同值，曲线将只有少量点；请用完整运行（非 --figures-only）并确认预测为连续概率。")

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            _draw_roc_on_ax(ax, y_true, y_proba, groups, attr_key, verbose=True)
            # 论文用：无 "ROC curves:"，不加粗，小标题字号放大
            ax.set_title(title or default_title, fontsize=roc_fs['title'], fontweight='normal', pad=8)
            fig.tight_layout()
            out_file = output_path / filename
            # 不使用 tight 裁切，避免不同标题长度导致导出尺寸不一致
            fig.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"✅ ROC 曲线已保存: {out_file}")

        # 子图标题（无 "ROC curves:"，论文中不加粗）
        _titles = {
            'pop_group': 'Overall and by population-density group',
            'gdp_group': 'Overall and by GDP-per-capita group',
            'continent_group': 'Overall and by continent group',
        }

        # 1) 人口密度分组
        _plot_for_group(
            attr_key='pop_group',
            filename='roc_curve_overall_and_pop_group.png',
            default_title=_titles['pop_group'],
        )
        # 2) 人均GDP分组
        _plot_for_group(
            attr_key='gdp_group',
            filename='roc_curve_overall_and_gdp_group.png',
            default_title=_titles['gdp_group'],
        )
        # 3) 大洲分组
        _plot_for_group(
            attr_key='continent_group',
            filename='roc_curve_overall_and_continent_group.png',
            default_title=_titles['continent_group'],
        )

        # 4) 论文用：三张 ROC 作为子图的一张大图，标题无 "ROC curves:" 且不加粗
        configs = [
            ('pop_group', _titles['pop_group']),
            ('gdp_group', _titles['gdp_group']),
            ('continent_group', _titles['continent_group']),
        ]
        data_list = [_get_roc_data(attr_key) for attr_key, _ in configs]
        if all(d is not None for d in data_list):
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for i, ((attr_key, subplot_title), data) in enumerate(zip(configs, data_list)):
                y_true, y_proba, groups = data
                _draw_roc_on_ax(axes[i], y_true, y_proba, groups, attr_key, verbose=False)
                axes[i].set_title(subplot_title, fontsize=roc_fs['title'], fontweight='normal', pad=8)
            fig.tight_layout()
            out_combined = output_path / 'roc_curve_three_panels.png'
            fig.savefig(out_combined, dpi=300)
            plt.close(fig)
            print(f"✅ ROC 三子图已保存: {out_combined}")
        else:
            print("⚠️  部分分组数据缺失，跳过 ROC 三子图")
    
    def visualize_results(self, output_dir='fairness_results', optimization_results=None):
        """
        可视化公平性分析结果
        
        Args:
            output_dir: 输出目录
            optimization_results: 阈值优化结果（可选）
        """
        print("\n" + "=" * 80)
        print("📊 生成可视化图表")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if self.sensitive_attributes is None:
            print("⚠️  未找到敏感属性数据，跳过可视化")
            return
        
        # 1. 性能指标对比图（仅人口密度分组）
        attr_values = self.sensitive_attributes['pop_group']
        unique_groups = [g for g in np.unique(attr_values) if g != 'Unknown']
        if len(unique_groups) < 2:
            print("⚠️  人口密度分组少于2组，跳过可视化")
            return
        
        # 计算各组的性能指标（原始阈值0.5）
        group_metrics_dict = {}
        for group in unique_groups:
            mask = np.array(attr_values) == group
            if mask.sum() == 0:
                continue
            
            group_y_true = self.true_labels[mask]
            group_y_pred = self.predictions[mask]
            group_y_proba = self.probabilities[mask]
            
            metrics = self.calculate_performance_metrics(group_y_true, group_y_pred, group_y_proba)
            group_metrics_dict[group] = metrics
        
        # 绘制性能对比图（原始，多子图去掉大标题）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'AUPRC']
        
        for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[idx // 3, idx % 3]
            
            groups = []
            values = []
            for group in unique_groups:
                if group in group_metrics_dict and metric_name in group_metrics_dict[group]:
                    val = group_metrics_dict[group][metric_name]
                    if val is not None:
                        groups.append(group)
                        values.append(val)
            
            if len(groups) > 0:
                bars = ax.bar(groups, values, alpha=0.7, edgecolor='black')
                ax.set_ylabel(metric_label, fontsize=20)
                # 子图标题进一步放大，横纵坐标刻度也放大
                ax.set_title(f'{metric_label} comparison', fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f'{val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=16
                    )
            
        plt.tight_layout()
        save_path = output_path / f'performance_comparison_pop_group.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
        plt.close()
        
        # 如果有优化结果，绘制优化前后对比图
        if optimization_results and 'optimal_thresholds' in optimization_results:
            optimized_metrics_dict = {}
            optimal_thresholds = optimization_results['optimal_thresholds']
            
            for group in unique_groups:
                mask = np.array(attr_values) == group
                if mask.sum() == 0 or group not in optimal_thresholds:
                    continue
                
                group_y_true = self.true_labels[mask]
                group_y_proba = self.probabilities[mask]
                threshold = optimal_thresholds[group]
                group_y_pred_opt = (group_y_proba > threshold).astype(int)
                
                metrics = self.calculate_performance_metrics(
                    group_y_true, group_y_pred_opt, group_y_proba
                )
                optimized_metrics_dict[group] = metrics
            
            # 绘制优化前后对比图（多子图去掉大标题）
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
                ax = axes[idx // 3, idx % 3]
                
                groups = []
                original_values = []
                optimized_values = []
                
                preferred_order = ['Low', 'Medium', 'High']
                ordered_groups = [g for g in preferred_order if g in unique_groups] + [
                    g for g in unique_groups if g not in preferred_order
                ]
                for group in ordered_groups:
                    if (group in group_metrics_dict and metric_name in group_metrics_dict[group] and
                        group in optimized_metrics_dict and metric_name in optimized_metrics_dict[group]):
                        orig_val = group_metrics_dict[group][metric_name]
                        opt_val = optimized_metrics_dict[group][metric_name]
                        if orig_val is not None and opt_val is not None:
                            groups.append(group)
                            original_values.append(orig_val)
                            optimized_values.append(opt_val)
                
                if len(groups) > 0:
                    x = np.arange(len(groups))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, original_values, width, 
                                  label='Original (Threshold=0.5)', alpha=0.7, edgecolor='black')
                    bars2 = ax.bar(x + width/2, optimized_values, width,
                                  label='Optimized (Group-Specific Threshold)', alpha=0.7, edgecolor='black')
                    
                    ax.set_ylabel(metric_label, fontsize=20)
                    # 子图标题与其他 performance_comparison 图统一字号，并进一步放大
                    ax.set_title(f'{metric_label} comparison', fontsize=22)
                    ax.set_xticks(x)
                    ax.set_xticklabels(groups)
                    # 只在第一个子图中显示图例，避免重复
                    if idx == 0:
                        ax.legend(fontsize=14)
                    ax.tick_params(axis='both', labelsize=18)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 添加数值标签
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f'{height:.3f}',
                                ha='center',
                                va='bottom',
                                fontsize=16
                            )
            
            plt.tight_layout()
            save_path = output_path / f'performance_comparison_optimized.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 已保存: {save_path}")
            plt.close()
        
        print(f"\n✅ 可视化完成，结果保存在: {output_path}")
    
    def generate_report(self, output_path='fairness_report.md', 
                       optimization_results=None):
        """
        生成公平性分析报告
        
        Args:
            output_path: 报告保存路径
            optimization_results: 阈值优化结果（可选）
        """
        print("\n" + "=" * 80)
        print("📝 生成分析报告")
        print("=" * 80)
        
        # 计算整体性能
        overall_metrics = self.calculate_performance_metrics(
            self.true_labels, self.predictions, self.probabilities
        )
        
        # 计算公平性指标（仅人口密度分组）
        fairness_metrics_pop = None
        
        if self.sensitive_attributes:
            pop_groups = np.array(self.sensitive_attributes['pop_group'])
            
            # 排除Unknown组
            pop_mask = pop_groups != 'Unknown'
            
            if pop_mask.sum() > 0:
                # 使用缓存键，避免重复计算（如果主函数已经计算过）
                # 这里统一使用population_density_original作为人口密度分组的原始阈值公平性缓存键
                fairness_metrics_pop = self.calculate_fairness_metrics(
                    self.true_labels[pop_mask],
                    self.predictions[pop_mask],
                    self.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_original',
                    use_cache=True
                )
        
        # 生成Markdown报告
        report_lines = [
            "# 模型公平性分析报告\n",
            "## 1. 整体性能（统一阈值0.5）\n",
            f"- **准确率**: {overall_metrics['accuracy']:.4f}",
            f"- **精确率**: {overall_metrics['precision']:.4f}",
            f"- **召回率**: {overall_metrics['recall']:.4f}",
            f"- **F1分数**: {overall_metrics['f1']:.4f}",
        ]
        
        if overall_metrics.get('auc') is not None:
            report_lines.append(f"- **AUC**: {overall_metrics['auc']:.4f}")
        if overall_metrics.get('auprc') is not None:
            report_lines.append(f"- **AUPRC**: {overall_metrics['auprc']:.4f}")
        
        if fairness_metrics_pop:
            if 'group_metrics' in fairness_metrics_pop:
                report_lines.append("\n### 2.1 各组性能（统一阈值0.5）\n")
                for group, metrics in fairness_metrics_pop['group_metrics'].items():
                    report_lines.append(f"#### {group} 人口密度组\n")
                    report_lines.append(f"- 准确率: {metrics['accuracy']:.4f}")
                    report_lines.append(f"- 精确率: {metrics['precision']:.4f}")
                    report_lines.append(f"- 召回率: {metrics['recall']:.4f}")
                    report_lines.append(f"- F1分数: {metrics['f1']:.4f}")
                    if metrics.get('auc') is not None:
                        report_lines.append(f"- AUC: {metrics['auc']:.4f}")
                    if metrics.get('auprc') is not None:
                        report_lines.append(f"- AUPRC: {metrics['auprc']:.4f}")
                    report_lines.append("")
            
            report_lines.append("### 2.2 公平性指标（统一阈值0.5）\n")
            if 'demographic_parity_difference' in fairness_metrics_pop:
                report_lines.append(f"- **Demographic Parity Difference**: {fairness_metrics_pop['demographic_parity_difference']:.4f}")
                report_lines.append(f"- **Demographic Parity Ratio**: {fairness_metrics_pop['demographic_parity_ratio']:.4f}")
                report_lines.append(f"- **Equalized Odds Difference**: {fairness_metrics_pop['equalized_odds_difference']:.4f}")
                report_lines.append(f"- **Equalized Odds Ratio**: {fairness_metrics_pop['equalized_odds_ratio']:.4f}")
        
            # 添加统计显著性检验结果
            if 'statistical_tests' in fairness_metrics_pop:
                report_lines.append("\n### 2.3 统计显著性检验\n")
                stat_tests = fairness_metrics_pop['statistical_tests']
                
                # 卡方检验
                if 'chi_square' in stat_tests and stat_tests['chi_square']:
                    report_lines.append("#### 2.3.1 卡方检验（Chi-square Test）\n")
                    report_lines.append("检验各组预测分布是否显著不同。\n")
                    report_lines.append("| 组别对比 | 卡方统计量 | p值 | 显著性 |\n")
                    report_lines.append("|---------|-----------|-----|--------|\n")
                    for key, result in stat_tests['chi_square'].items():
                        groups = key.replace('_vs_', ' vs ')
                        p_val = result['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        report_lines.append(f"| {groups} | {result['chi2']:.4f} | {p_val:.6f} | {sig} |\n")
                    report_lines.append("\n*注: *** p<0.001, ** p<0.01, * p<0.05, ns=不显著*\n")
                
                # Bootstrap置信区间
                if 'bootstrap' in stat_tests and stat_tests['bootstrap']:
                    report_lines.append("#### 2.3.2 Bootstrap置信区间（Bootstrap Confidence Intervals）\n")
                    report_lines.append("估计召回率差异的95%置信区间。\n")
                    report_lines.append("| 组别对比 | 实际差异 | 95% CI下限 | 95% CI上限 | 是否包含0 |\n")
                    report_lines.append("|---------|---------|-----------|-----------|----------|\n")
                    for key, result in stat_tests['bootstrap'].items():
                        groups = key.replace('_vs_', ' vs ')
                        excludes_zero = "否（显著）" if result['excludes_zero'] else "是（不显著）"
                        report_lines.append(f"| {groups} | {result['actual_difference']:.4f} | "
                                         f"{result['ci_lower']:.4f} | {result['ci_upper']:.4f} | {excludes_zero} |\n")
                    report_lines.append("\n")
                
                # Cohen's h效应量
                if 'cohens_h' in stat_tests and stat_tests['cohens_h']:
                    report_lines.append("#### 2.3.3 Cohen's h效应量（Effect Size）\n")
                    report_lines.append("衡量召回率差异的效应大小。|h| < 0.2为小效应，0.2-0.5为小到中，0.5-0.8为中到大，>0.8为大效应。\n")
                    report_lines.append("| 组别对比 | Cohen's h | 效应大小 | 召回率对比 |\n")
                    report_lines.append("|---------|----------|---------|-----------|\n")
                    for key, result in stat_tests['cohens_h'].items():
                        groups = key.replace('_vs_', ' vs ')
                        report_lines.append(f"| {groups} | {result['h']:.4f} | {result['effect_size']} | "
                                         f"{result['recall1']:.4f} vs {result['recall2']:.4f} |\n")
                    report_lines.append("\n")
                
                # Mann-Whitney U检验
                if 'mann_whitney' in stat_tests and stat_tests['mann_whitney']:
                    report_lines.append("#### 2.3.4 Mann-Whitney U检验（Mann-Whitney U Test）\n")
                    report_lines.append("检验各组预测概率分布是否显著不同。\n")
                    report_lines.append("| 组别对比 | U统计量 | p值 | 显著性 | 概率均值对比 |\n")
                    report_lines.append("|---------|---------|-----|--------|------------|\n")
                    for key, result in stat_tests['mann_whitney'].items():
                        groups = key.replace('_vs_', ' vs ')
                        p_val = result['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        prob_comp = f"{result['mean_prob1']:.4f} vs {result['mean_prob2']:.4f}"
                        report_lines.append(f"| {groups} | {result['statistic']:.2f} | {p_val:.6f} | {sig} | {prob_comp} |\n")
                    report_lines.append("\n")
                
                # 混淆矩阵分析
                if 'confusion_matrix_analysis' in stat_tests and stat_tests['confusion_matrix_analysis']:
                    report_lines.append("#### 2.3.5 混淆矩阵详细分析（Confusion Matrix Analysis）\n")
                    report_lines.append("分析各组误分类模式，重点关注漏检率（False Negative Rate）。\n")
                    report_lines.append("| 组别 | TP | FP | TN | FN | 漏检率(FNR) | 误报率(FPR) | 漏检火灾数 |\n")
                    report_lines.append("|------|----|----|----|----|------------|------------|----------|\n")
                    for group, analysis in stat_tests['confusion_matrix_analysis'].items():
                        cm = analysis['confusion_matrix']
                        report_lines.append(f"| {group} | {cm['TP']} | {cm['FP']} | {cm['TN']} | {cm['FN']} | "
                                         f"{analysis['false_negative_rate']:.4f} | {analysis['false_positive_rate']:.4f} | "
                                         f"{analysis['missed_fires']:,} |\n")
                    report_lines.append("\n")
                
                # 统计显著性总结
                report_lines.append("#### 2.3.6 统计显著性总结\n")
                significant_count = 0
                total_tests = 0
                
                if 'chi_square' in stat_tests:
                    for result in stat_tests['chi_square'].values():
                        total_tests += 1
                        if result['p_value'] < 0.05:
                            significant_count += 1
                
                if 'mann_whitney' in stat_tests:
                    for result in stat_tests['mann_whitney'].values():
                        total_tests += 1
                        if result['p_value'] < 0.05:
                            significant_count += 1
                
                report_lines.append(f"- 总检验数: {total_tests}\n")
                report_lines.append(f"- 显著差异数: {significant_count} ({significant_count/total_tests*100:.1f}%)\n")
                
                if significant_count > 0:
                    report_lines.append(f"\n**结论**: 通过多种统计检验，我们发现了{significant_count}个显著差异，")
                    report_lines.append("**强烈支持模型存在不公平性的结论**。\n")
                else:
                    report_lines.append("\n**结论**: 未发现显著差异，可能需要更大样本量或更严格的检验。\n")
        
        # 添加阈值优化结果
        if optimization_results:
            report_lines.append("\n## 3. 组特定阈值优化结果\n")
            
            optimal_thresholds = optimization_results.get('optimal_thresholds', {})
            optimized_metrics = optimization_results.get('optimized_metrics', {})
            
            report_lines.append("### 3.1 最优阈值\n")
            for group, threshold in optimal_thresholds.items():
                report_lines.append(f"- **{group}组**: {threshold:.4f}")
            
            if optimized_metrics:
                report_lines.append("\n### 3.2 优化后各组性能\n")
                for group, metrics in optimized_metrics.items():
                    original_metrics = fairness_metrics_pop['group_metrics'].get(group, {})
                    report_lines.append(f"#### {group} 人口密度组（阈值: {optimal_thresholds.get(group, 0.5):.4f}）\n")
                    report_lines.append(f"- 准确率: {metrics['accuracy']:.4f} "
                                     f"(优化前: {original_metrics.get('accuracy', 0):.4f}, "
                                     f"变化: {metrics['accuracy'] - original_metrics.get('accuracy', 0):+.4f})")
                    report_lines.append(f"- 精确率: {metrics['precision']:.4f} "
                                     f"(优化前: {original_metrics.get('precision', 0):.4f}, "
                                     f"变化: {metrics['precision'] - original_metrics.get('precision', 0):+.4f})")
                    report_lines.append(f"- 召回率: {metrics['recall']:.4f} "
                                     f"(优化前: {original_metrics.get('recall', 0):.4f}, "
                                     f"变化: {metrics['recall'] - original_metrics.get('recall', 0):+.4f})")
                    report_lines.append(f"- F1分数: {metrics['f1']:.4f} "
                                     f"(优化前: {original_metrics.get('f1', 0):.4f}, "
                                     f"变化: {metrics['f1'] - original_metrics.get('f1', 0):+.4f})")
                    report_lines.append("")
                
                # 计算优化后的公平性指标
                if self.sensitive_attributes:
                    pop_groups = np.array(self.sensitive_attributes['pop_group'])
                    pop_mask = pop_groups != 'Unknown'
                    if pop_mask.sum() > 0:
                        optimized_predictions = self.apply_group_specific_thresholds(
                            self.probabilities[pop_mask],
                            pop_groups[pop_mask],
                            optimal_thresholds
                        )
                        
                        # 使用缓存键，避免重复计算（如果主函数已经计算过）
                        # 这里统一使用population_density_optimized作为人口密度分组的优化后阈值公平性缓存键
                        optimized_fairness = self.calculate_fairness_metrics(
                            self.true_labels[pop_mask],
                            optimized_predictions,
                            self.probabilities[pop_mask],
                            pop_groups[pop_mask],
                            cache_key='population_density_optimized',
                            use_cache=True
                        )
                        
                        report_lines.append("### 3.3 优化后公平性指标\n")
                        if 'demographic_parity_difference' in optimized_fairness:
                            original_dp_diff = fairness_metrics_pop.get('demographic_parity_difference', 0)
                            optimized_dp_diff = optimized_fairness['demographic_parity_difference']
                            report_lines.append(f"- **Demographic Parity Difference**: {optimized_dp_diff:.4f} "
                                             f"(优化前: {original_dp_diff:.4f}, "
                                             f"改善: {original_dp_diff - optimized_dp_diff:+.4f})")
                            
                            original_eo_diff = fairness_metrics_pop.get('equalized_odds_difference', 0)
                            optimized_eo_diff = optimized_fairness['equalized_odds_difference']
                            report_lines.append(f"- **Equalized Odds Difference**: {optimized_eo_diff:.4f} "
                                             f"(优化前: {original_eo_diff:.4f}, "
                                             f"改善: {original_eo_diff - optimized_eo_diff:+.4f})")
        
        report_lines.append("\n## 4. 建议\n")
        report_lines.append("根据公平性分析结果，建议：\n")
        report_lines.append("1. 如果发现显著的性能差异，考虑调整模型或数据采样策略")
        report_lines.append("2. 对于表现较差的群体，可以增加训练数据或调整特征工程")
        report_lines.append("3. 使用组特定阈值优化可以快速改善公平性（如本报告所示）")
        report_lines.append("4. 使用fairlearn等工具进行后处理，进一步改善公平性")
        
        # 保存报告
        report_path = Path(output_path)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 报告已保存: {report_path}")
    
    def analyze_multiple_groupings(self, output_dir='fairness_results', 
                                   existing_optimization_results=None):
        """
        分析多种分类方式的性能（人口密度、GDP per capita、Continent）
        使用optimized threshold进行比较
        
        Args:
            output_dir: 输出目录
            existing_optimization_results: 已存在的优化结果字典，格式为 
                {grouping_type: optimization_results}，如果提供，将重用这些结果
        """
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("📊 多种分类方式性能分析（使用Optimized Threshold）")
            print("=" * 80)
        
        if self.sensitive_attributes is None:
            print("⚠️  未找到敏感属性数据，跳过分析")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 定义三种分类方式
        grouping_types = {
            'population_density': {
                'attr_key': 'pop_group',
                'name': 'Population Density',
                'name_cn': '人口密度'
            },
            'gdp_per_capita': {
                'attr_key': 'gdp_group',
                'name': 'GDP per Capita',
                'name_cn': '人均GDP'
            },
            'continent': {
                'attr_key': 'continent_group',
                'name': 'Continent',
                'name_cn': '大洲'
            }
        }
        
        results_summary = {}
        
        for grouping_type, config in grouping_types.items():
            if not quiet:
                print(f"\n{'='*80}")
                print(f"📊 分析分类方式: {config['name_cn']} ({config['name']})")
                print(f"{'='*80}")
            
            attr_key = config['attr_key']
            if attr_key not in self.sensitive_attributes:
                print(f"⚠️  未找到 {attr_key} 属性，跳过")
                continue
            
            attr_values = np.array(self.sensitive_attributes[attr_key])
            unique_groups = [g for g in np.unique(attr_values) if g != 'Unknown']
            
            if len(unique_groups) < 2:
                print(f"⚠️  {config['name_cn']}分组少于2组，跳过")
                continue
            
            # 过滤Unknown组
            mask = attr_values != 'Unknown'
            if mask.sum() == 0:
                print(f"⚠️  没有有效样本，跳过")
                continue
            
            y_true_filtered = self.true_labels[mask]
            y_proba_filtered = self.probabilities[mask]
            attr_filtered = attr_values[mask]
            
            # 先在统一阈值0.5下计算一次公平性指标（原始阈值），三种分类方式都保证“原始1次”
            original_predictions = (y_proba_filtered > 0.5).astype(int)
            original_cache_key = f"{grouping_type}_original"
            self.calculate_fairness_metrics(
                y_true_filtered,
                original_predictions,
                y_proba_filtered,
                attr_filtered,
                cache_key=original_cache_key,
                use_cache=True
            )
            
            # 检查是否已有优化结果（避免重复优化）
            optimization_results = None
            if existing_optimization_results and grouping_type in existing_optimization_results:
                if not quiet:
                    print(f"\n✅ 重用已有的 {config['name_cn']} 阈值优化结果（避免重复计算）")
                optimization_results = existing_optimization_results[grouping_type]
                optimal_thresholds = optimization_results.get('optimal_thresholds', {})
            else:
                # 优化阈值
                if not quiet:
                    print(f"\n🔧 优化 {config['name_cn']} 分组的阈值...")
                optimization_results = self.optimize_group_specific_thresholds(
                    y_true_filtered,
                    y_proba_filtered,
                    attr_filtered,
                    metric='balanced_accuracy',
                    threshold_range=(0.1, 0.9),
                    num_thresholds=200,
                    min_precision=0.4,
                    min_recall=0.3,
                    use_validation_split=True,
                    validation_ratio=0.2
                )
                
                if not optimization_results or 'optimal_thresholds' not in optimization_results:
                    print(f"⚠️  阈值优化失败，跳过 {config['name_cn']}")
                    continue
                
                optimal_thresholds = optimization_results['optimal_thresholds']
            
            # 使用优化后的阈值生成预测
            optimized_predictions = self.apply_group_specific_thresholds(
                y_proba_filtered,
                attr_filtered,
                optimal_thresholds
            )
            
            # 计算各组性能指标
            group_metrics = {}
            if not quiet:
                print(f"\n📊 {config['name_cn']} 各组性能（使用Optimized Threshold）:")
                print("-" * 80)
                print(f"{'组别':<15} {'阈值':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10}")
                print("-" * 80)
            
            for group in unique_groups:
                group_mask = attr_filtered == group
                if group_mask.sum() == 0:
                    continue
                
                group_y_true = y_true_filtered[group_mask]
                group_y_pred = optimized_predictions[group_mask]
                group_y_proba = y_proba_filtered[group_mask]
                threshold = optimal_thresholds.get(group, 0.5)
                
                metrics = self.calculate_performance_metrics(
                    group_y_true, group_y_pred, group_y_proba
                )
                group_metrics[group] = metrics
                
                if not quiet:
                    auc_str = f"{metrics.get('auc', 0):.4f}" if metrics.get('auc') is not None else "N/A"
                    print(f"{group:<15} {threshold:<10.4f} {metrics['accuracy']:<10.4f} "
                          f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1']:<10.4f} {auc_str:<10}")
            
            # 计算公平性指标（使用缓存键，避免重复计算）
            cache_key = f"{grouping_type}_optimized"
            fairness_metrics = self.calculate_fairness_metrics(
                y_true_filtered,
                optimized_predictions,
                y_proba_filtered,
                attr_filtered,
                cache_key=cache_key,
                use_cache=True
            )
            
            # 保存结果
            results_summary[grouping_type] = {
                'name': config['name'],
                'name_cn': config['name_cn'],
                'group_metrics': group_metrics,
                'optimal_thresholds': optimal_thresholds,
                'fairness_metrics': fairness_metrics,
                'optimization_results': optimization_results
            }
            
            # 绘制性能对比图
            self._plot_grouping_performance(
                group_metrics, optimal_thresholds, config, output_path
            )
        
        # 仅在非 figures-only 模式下生成对比总结文件，避免 --figures-only 时写入额外文件
        if not getattr(self, 'is_figures_only', False):
            self._generate_comparison_summary(results_summary, output_path)
        
        return results_summary
    
    def _plot_grouping_performance(self, group_metrics, optimal_thresholds, config, output_path):
        """绘制单个分类方式的性能对比图"""
        unique_groups = list(group_metrics.keys())
        if len(unique_groups) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'AUPRC']
        
        for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[idx // 3, idx % 3]
            
            groups = []
            values = []
            for group in unique_groups:
                if group in group_metrics and metric_name in group_metrics[group]:
                    val = group_metrics[group][metric_name]
                    if val is not None:
                        groups.append(group)
                        values.append(val)
            
            if len(groups) > 0:
                bars = ax.bar(groups, values, alpha=0.7, edgecolor='black')
                ax.set_ylabel(metric_label, fontsize=20)
                # 子图标题与 performance_comparison_optimized.png 保持一致字号，并进一步放大
                ax.set_title(f'{metric_label} comparison', fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f'{val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=16
                    )
        
        plt.tight_layout()
        save_path = output_path / f'performance_comparison_{config["name"].lower().replace(" ", "_")}_optimized.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
        plt.close()
    
    def _generate_comparison_summary(self, results_summary, output_path):
        """生成三种分类方式的对比总结"""
        if getattr(self, 'is_figures_only', False):
            # figures-only 模式下不应被调用（上游已判断），这里再做保护
            return
        
        print("\n" + "=" * 80)
        print("📊 三种分类方式性能对比总结")
        
        # 创建对比表格
        comparison_lines = [
            "# 三种分类方式性能对比总结（使用Optimized Threshold）\n",
            "## 1. 各组性能指标对比\n"
        ]
        
        # 为每种分类方式生成表格
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            group_metrics = result['group_metrics']
            optimal_thresholds = result['optimal_thresholds']
            
            comparison_lines.append(f"### {name_cn} ({result['name']})\n")
            comparison_lines.append("| 组别 | 阈值 | 准确率 | 精确率 | 召回率 | F1 | AUC |\n")
            comparison_lines.append("|------|------|--------|--------|--------|----|----|\n")
            
            for group in ['Low', 'Medium', 'High']:
                if group in group_metrics:
                    metrics = group_metrics[group]
                    threshold = optimal_thresholds.get(group, 0.5)
                    auc_str = f"{metrics.get('auc', 0):.4f}" if metrics.get('auc') is not None else "N/A"
                    comparison_lines.append(
                        f"| {group} | {threshold:.4f} | {metrics['accuracy']:.4f} | "
                        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                        f"{metrics['f1']:.4f} | {auc_str} |\n"
                    )
            comparison_lines.append("\n")
        
        # 添加公平性指标对比
        comparison_lines.append("## 2. 公平性指标对比\n")
        comparison_lines.append("| 分类方式 | Equalized Odds Difference | Demographic Parity Difference |\n")
        comparison_lines.append("|---------|-------------------------|-------------------------------|\n")
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            fairness = result['fairness_metrics']
            eo_diff = fairness.get('equalized_odds_difference', 'N/A')
            dp_diff = fairness.get('demographic_parity_difference', 'N/A')
            
            eo_str = f"{eo_diff:.4f}" if isinstance(eo_diff, (int, float)) else "N/A"
            dp_str = f"{dp_diff:.4f}" if isinstance(dp_diff, (int, float)) else "N/A"
            
            comparison_lines.append(f"| {name_cn} | {eo_str} | {dp_str} |\n")
        
        # 添加三种分类方式的统计性检验概览
        comparison_lines.append("\n## 3. 统计显著性检验概览\n")
        comparison_lines.append("下表总结了在每种分类方式下，针对组间差异所进行的统计检验数量及显著结果数量。\n\n")
        comparison_lines.append("| 分类方式 | 显著差异数 | 总检验数 | 显著比例 |\n")
        comparison_lines.append("|---------|------------|----------|----------|\n")
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            fairness = result['fairness_metrics']
            stat_tests = fairness.get('statistical_tests', {})
            
            significant_count = 0
            total_tests = 0
            
            # 目前主要统计卡方检验和Mann-Whitney U检验的显著性情况
            if 'chi_square' in stat_tests:
                for res in stat_tests['chi_square'].values():
                    total_tests += 1
                    if res.get('p_value', 1.0) < 0.05:
                        significant_count += 1
            
            if 'mann_whitney' in stat_tests:
                for res in stat_tests['mann_whitney'].values():
                    total_tests += 1
                    if res.get('p_value', 1.0) < 0.05:
                        significant_count += 1
            
            if total_tests > 0:
                ratio = significant_count / total_tests * 100
                comparison_lines.append(
                    f"| {name_cn} | {significant_count} | {total_tests} | {ratio:.1f}% |\n"
                )
        
        # 保存对比报告
        comparison_path = output_path / 'grouping_comparison_summary.md'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(''.join(comparison_lines))
        
        print(f"✅ 对比总结已保存: {comparison_path}")
        
        # 打印到控制台
        print("\n" + "-" * 80)
        print("各组F1分数对比:")
        print("-" * 80)
        print(f"{'分类方式':<20} {'Low':<12} {'Medium':<12} {'High':<12}")
        print("-" * 80)
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            group_metrics = result['group_metrics']
            
            low_f1 = group_metrics.get('Low', {}).get('f1', 0)
            medium_f1 = group_metrics.get('Medium', {}).get('f1', 0)
            high_f1 = group_metrics.get('High', {}).get('f1', 0)
            
            print(f"{name_cn:<20} {low_f1:<12.4f} {medium_f1:<12.4f} {high_f1:<12.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型公平性分析')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径（--figures-only 时可不填）')
    parser.add_argument('--data', type=str, nargs='+', default=None,
                       help='数据文件路径，可多个（--figures-only 时可不填）')
    parser.add_argument('--output', type=str, default='fairness_results',
                        help='输出目录')
    parser.add_argument('--xai', action='store_true',
                        help='运行实验一：按群体划分的特征贡献分析 (Integrated Gradients)')
    parser.add_argument('--figures-only', action='store_true',
                       help='仅从缓存重绘结果图，不加载模型和数据（需配合 --cache）')
    parser.add_argument('--cache', type=str, default=None,
                       help='绘图缓存路径：读取（--figures-only）或写入（完整运行时保存到 output/fairness_plot_cache.pkl）')
    
    args = parser.parse_args()
    
    # 仅重绘图模式：从缓存加载，重绘全部结果图（与完整运行一致）
    if args.figures_only:
        cache_path = args.cache or str(Path(args.output) / 'fairness_plot_cache.pkl')
        if not Path(cache_path).exists():
            print(f"❌ 缓存文件不存在: {cache_path}")
            print("   请先完整运行一次公平性分析以生成缓存。")
            sys.exit(1)
        print("📂 从缓存加载，重绘全部结果图...")
        cache = load_plot_cache(cache_path)
        analyzer = FairnessAnalyzer()
        # 仅重绘图时，不再运行耗时的统计显著性检验，只使用缓存的预测和标签算图形，
        # 同时抑制冗长日志并避免生成非图像文件。
        analyzer.skip_statistical_tests = True
        analyzer.is_figures_only = True
        analyzer.true_labels = cache['true_labels']
        analyzer.probabilities = cache['probabilities']
        analyzer.predictions = cache['predictions']
        analyzer.sensitive_attributes = cache['sensitive_attributes']
        # 如果缓存中包含 XAI 结果，则在 figures-only 模式下也重绘 XAI 图
        xai_results = cache.get('xai_results', None)
        if xai_results is not None:
            try:
                analyzer._plot_attribution_boxplot_by_group(xai_results, args.output)
                analyzer._plot_top5_features_by_group(xai_results, args.output)
            except Exception as e:
                print(f"⚠️  重绘 XAI 图失败: {e}")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        # 1) 人口密度性能对比（原始阈值）
        analyzer.visualize_results(output_dir=args.output)
        # 2) 三张 ROC 曲线图
        analyzer.plot_roc_curves(output_dir=args.output)
        # 3) 阈值权衡曲线（样本数适中时）
        if analyzer.sensitive_attributes:
            pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
            pop_mask = pop_groups != 'Unknown'
            if pop_mask.sum() > 0 and len(analyzer.true_labels[pop_mask]) < 100000:
                try:
                    analyzer.plot_threshold_tradeoff_curves(
                        analyzer.true_labels[pop_mask],
                        analyzer.probabilities[pop_mask],
                        pop_groups[pop_mask],
                        output_dir=args.output,
                        threshold_range=(0.1, 0.9),
                        num_thresholds=50
                    )
                except Exception as e:
                    print(f"⚠️  阈值权衡曲线重绘失败: {e}")
        # 4) 人口密度组阈值优化（与完整运行相同参数），再重绘含优化对比图
        optimization_results = None
        if analyzer.sensitive_attributes:
            pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
            pop_mask = pop_groups != 'Unknown'
            if pop_mask.sum() > 0:
                optimization_results = analyzer.optimize_group_specific_thresholds(
                    analyzer.true_labels[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    metric='balanced_accuracy',
                    threshold_range=(0.1, 0.9),
                    num_thresholds=200,
                    min_precision=0.4,
                    min_recall=0.3,
                    use_validation_split=True,
                    validation_ratio=0.2
                )
                # 用组特定阈值重绘阈值权衡图，在图中显示 Low/Medium/High 的 tau_g 竖线
                try:
                    analyzer.plot_threshold_tradeoff_curves(
                        analyzer.true_labels[pop_mask],
                        analyzer.probabilities[pop_mask],
                        pop_groups[pop_mask],
                        output_dir=args.output,
                        threshold_range=(0.1, 0.9),
                        num_thresholds=50,
                        optimal_thresholds=optimization_results.get('optimal_thresholds', {})
                    )
                except Exception as e:
                    print(f"⚠️  使用 tau_g 重绘阈值权衡曲线失败: {e}")
                analyzer.visualize_results(output_dir=args.output, optimization_results=optimization_results)
        # 5) 多种分类方式分析（人口密度复用上面结果，GDP/大洲会重新优化并出图 + grouping_comparison_summary.md）
        if optimization_results is not None:
            existing_optimization = {'population_density': optimization_results}
            analyzer.analyze_multiple_groupings(output_dir=args.output, existing_optimization_results=existing_optimization)
        print("✅ 全部结果图重绘完成，结果在:", args.output)
        return
    
    if not args.checkpoint or not args.data:
        print("❌ 完整运行需要 --checkpoint 和 --data")
        parser.print_help()
        sys.exit(1)
    
    # 创建分析器
    analyzer = FairnessAnalyzer(
        checkpoint_path=args.checkpoint,
        data_paths=args.data
    )
    
    # 加载数据
    samples = analyzer.load_data()
    
    # 提取敏感属性
    sensitive_attrs = analyzer.extract_sensitive_attributes(samples)
    
    # 创建测试数据集
    test_dataset = FireTracksDataset(samples, target_type='binary_classification')
    
    # 验证样本顺序一致性
    print("\n" + "=" * 80)
    print("🔍 验证样本顺序一致性")
    print("=" * 80)
    print(f"原始样本数: {len(samples)}")
    print(f"数据集样本数: {len(test_dataset)}")
    
    # 检查前几个样本的features是否一致
    consistent_count = 0
    for i in range(min(10, len(samples))):
        sample_features = samples[i].get('features', None)
        dataset_features, _ = test_dataset[i]
        if sample_features is not None:
            sample_features_tensor = torch.from_numpy(sample_features).float()
            sample_features_tensor = sample_features_tensor.permute(3, 0, 1, 2)  # [T,H,W,C] -> [C,T,H,W]
            if torch.allclose(sample_features_tensor, dataset_features, atol=1e-6):
                consistent_count += 1
            else:
                print(f"❌ 样本 {i} 特征不一致！")
                print(f"   原始特征shape: {sample_features.shape}")
                print(f"   数据集特征shape: {dataset_features.shape}")
    
    if consistent_count == min(10, len(samples)):
        print(f"✅ 前{min(10, len(samples))}个样本特征一致，样本顺序正确")
    else:
        print(f"⚠️  样本顺序可能有问题！只有{consistent_count}/{min(10, len(samples))}个样本一致")
    
    # 加载模型并生成预测
    predictions, probabilities, true_labels = analyzer.load_model_and_predict(test_dataset)
    
    # 保存绘图缓存，便于之后用 --figures-only 重绘图而无需重新跑模型
    cache_path = Path(args.output) / 'fairness_plot_cache.pkl'
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_plot_cache(analyzer, cache_path)

    # 实验一：按群体划分的特征贡献分析（可选）
    if getattr(args, 'xai', False) and CAPTUM_AVAILABLE:
        print("\n" + "=" * 80)
        print("🔬 实验一：按群体划分的特征贡献分析 (XAI)")
        print("=" * 80)
        try:
            analyzer.run_group_wise_attribution(
                test_dataset,
                analyzer.sensitive_attributes,
                args.output,
                max_samples_per_group=800,
                batch_size=8,
                n_steps=50,
            )
        except Exception as e:
            print(f"⚠️  归因分析失败: {e}")
            import traceback
            traceback.print_exc()
        # 将 XAI 结果写入缓存，便于 --figures-only 重绘时也包含 XAI 图
        save_plot_cache(analyzer, cache_path)
        print("✅ XAI 图已保存。")
    
    # 可视化结果（先使用原始预测）
    analyzer.visualize_results(output_dir=args.output)
    analyzer.plot_roc_curves(output_dir=args.output)
    
    # 绘制阈值权衡曲线（如果数据量不太大）
    if analyzer.sensitive_attributes:
        pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
        pop_mask = pop_groups != 'Unknown'
        
        if pop_mask.sum() > 0 and len(analyzer.true_labels[pop_mask]) < 100000:
            # 如果样本数不太大，绘制权衡曲线
            print("\n" + "=" * 80)
            print("📊 绘制阈值权衡曲线（样本数较大时可能需要较长时间）")
            print("=" * 80)
            try:
                analyzer.plot_threshold_tradeoff_curves(
                    analyzer.true_labels[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    output_dir=args.output,
                    threshold_range=(0.1, 0.9),
                    num_thresholds=50  # 减少阈值数量以加快速度
                )
            except Exception as e:
                print(f"⚠️  绘制阈值权衡曲线失败: {e}")
                print("   这可能是由于数据量过大或内存不足")
        else:
            print("\n⚠️  样本数过大，跳过阈值权衡曲线绘制（可手动调用plot_threshold_tradeoff_curves）")
    
    # 组特定阈值优化
    optimization_results = None
    if analyzer.sensitive_attributes:
        pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
        pop_mask = pop_groups != 'Unknown'
        
        if pop_mask.sum() > 0:
            # 使用balanced_accuracy优化，平衡精确率和召回率
            # 这样可以避免过度预测正类，同时保持合理的召回率
            # 现在默认使用验证集分割，避免过拟合测试集
            optimization_results = analyzer.optimize_group_specific_thresholds(
                analyzer.true_labels[pop_mask],
                analyzer.probabilities[pop_mask],
                pop_groups[pop_mask],
                metric='balanced_accuracy',  # 使用平衡准确率，平衡精确率和召回率
                threshold_range=(0.1, 0.9),
                num_thresholds=200,
                min_precision=0.4,  # 最小精确率40%（更严格）
                min_recall=0.3,     # 最小召回率30%
                use_validation_split=True,  # 使用验证集分割
                validation_ratio=0.2  # 20%作为验证集
            )
            # 用组特定阈值重绘阈值权衡图，在图中显示 Low/Medium/High 的 tau_g 竖线
            try:
                analyzer.plot_threshold_tradeoff_curves(
                    analyzer.true_labels[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    output_dir=args.output,
                    threshold_range=(0.1, 0.9),
                    num_thresholds=50,
                    optimal_thresholds=optimization_results.get('optimal_thresholds', {})
                )
            except Exception as e:
                print(f"⚠️  使用 tau_g 重绘阈值权衡曲线失败: {e}")
            
            # 可选：对比不同优化方法
            print("\n" + "=" * 80)
            print("🔬 可选：运行阈值优化方法对比实验")
            print("=" * 80)
            print("提示：取消下面的注释以运行对比实验（可能需要较长时间）")
            # comparison_results = analyzer.compare_threshold_methods(
            #     analyzer.true_labels[pop_mask],
            #     analyzer.probabilities[pop_mask],
            #     pop_groups[pop_mask],
            #     methods=['f1', 'balanced_accuracy', 'youden', 'f2'],
            #     use_validation_split=True,
            #     validation_ratio=0.2
            # )
            
            # 应用优化后的阈值并重新评估
            if optimization_results and 'optimal_thresholds' in optimization_results:
                print("\n" + "=" * 80)
                print("📊 应用优化后的阈值并重新评估")
                print("=" * 80)
                
                optimized_predictions = analyzer.apply_group_specific_thresholds(
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    optimization_results['optimal_thresholds']
                )
                
                # 计算优化后的公平性指标（使用缓存键）
                # 统一使用population_density_optimized作为人口密度分组的优化后公平性缓存键
                optimized_fairness = analyzer.calculate_fairness_metrics(
                    analyzer.true_labels[pop_mask],
                    optimized_predictions,
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_optimized',
                    use_cache=True
                )
                
                print("\n📈 优化前后对比:")
                print("=" * 80)
                
                # 获取原始公平性指标（使用缓存键）
                # 统一使用population_density_original作为人口密度分组的原始公平性缓存键
                original_fairness = analyzer.calculate_fairness_metrics(
                    analyzer.true_labels[pop_mask],
                    analyzer.predictions[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_original',
                    use_cache=True
                )
                
                if 'group_metrics' in original_fairness and 'group_metrics' in optimized_fairness:
                    print("\n各组召回率对比:")
                    for group in ['Low', 'Medium', 'High']:
                        if group in original_fairness['group_metrics']:
                            orig_recall = original_fairness['group_metrics'][group]['recall']
                            opt_recall = optimized_fairness['group_metrics'][group]['recall']
                            improvement = opt_recall - orig_recall
                            print(f"  {group}组: {orig_recall:.4f} → {opt_recall:.4f} "
                                f"(改善: {improvement:+.4f}, {improvement/orig_recall*100:+.1f}%)")
                
                if 'equalized_odds_difference' in original_fairness:
                    orig_eo = original_fairness['equalized_odds_difference']
                    opt_eo = optimized_fairness['equalized_odds_difference']
                    print(f"\nEqualized Odds Difference: {orig_eo:.4f} → {opt_eo:.4f} "
                          f"(改善: {orig_eo - opt_eo:+.4f})")
            
            # 重新生成可视化（包含优化后的对比）
            analyzer.visualize_results(
                output_dir=args.output,
                optimization_results=optimization_results
            )
    
    # 分析多种分类方式的性能（使用optimized threshold）
    # 传递已存在的优化结果，避免重复优化人口密度分组
    print("\n" + "=" * 80)
    print("📊 开始多种分类方式性能分析")
    print("=" * 80)
    existing_optimization = {}
    if optimization_results:
        # 如果主函数已经优化了人口密度分组，传递给analyze_multiple_groupings重用
        existing_optimization['population_density'] = optimization_results
        print("✅ 将重用主函数中已优化的人口密度分组阈值（避免重复计算）")
    multiple_groupings_results = analyzer.analyze_multiple_groupings(
        output_dir=args.output,
        existing_optimization_results=existing_optimization
    )
    
    # 生成报告（包含优化结果）
    analyzer.generate_report(
        output_path=Path(args.output) / 'fairness_report.md',
        optimization_results=optimization_results
    )
    
    print("\n" + "=" * 80)
    print("✅ 公平性分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

