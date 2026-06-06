"""
modelfairness analysis
model(population density)
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

# : Times New Roman,()
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,            # :
    "axes.titlesize": 22,       # 
    "axes.labelsize": 20,       # 
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

# 
try:
    from scipy.stats import chi2_contingency, mannwhitneyu, bootstrap
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy,")
    print("   : pip install scipy")

#  src ( release )
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
    print("⚠️  fairlearn,")
    print("   : pip install fairlearn")

from fire_equality.datamodules.firetracks_loader import FireTracksDataset
import pytorch_lightning as pl
from fire_equality.models.fire_equality_model import ConvLSTM_fire_equality_model

# :()
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# 8  GeoRL/feature_alignment 
FEATURE_NAMES = ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']


class ConvLSTMWrapperForCaptum(torch.nn.Module):
    """ Captum : [B, C, T, H, W], log_softmax [B, 2]."""
    def __init__(self, pl_model):
        super().__init__()
        self.pl_model = pl_model
        self.pl_model.eval()

    def forward(self, x):
        # x: [B, C, T, H, W] -> [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.pl_model(x)

# FixPyTorch 2.6+checkpoint loading
def patch_lightning_checkpoint_loading():
    """patch PyTorch Lightningcheckpoint loading,weights_only=False"""
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
        
        # save the original function()
        if not hasattr(cloud_io, '_original_load'):
            cloud_io._original_load = cloud_io._load
        
        def patched_load(f, map_location=None):
            """patchloading,weights_only=False"""
            return torch.load(f, map_location=map_location, weights_only=False)
        
        # replace the original function
        cloud_io._load = patched_load
        
        # patch pl_load()
        if hasattr(cloud_io, 'pl_load'):
            if not hasattr(cloud_io, '_original_pl_load'):
                cloud_io._original_pl_load = cloud_io.pl_load
            
            def patched_pl_load(path, map_location=None):
                """patch pl_load"""
                return torch.load(path, map_location=map_location, weights_only=False)
            
            cloud_io.pl_load = patched_pl_load
        
        # patch torch.load(more aggressive but more reliable)
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
        
        def patched_torch_load(f, map_location=None, **kwargs):
            """patch torch.loading,weights_only=False"""
            kwargs['weights_only'] = False
            return torch._original_load(f, map_location=map_location, **kwargs)
        
        torch.load = patched_torch_load
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"unable to patch Lightning checkpoint loading: {e}")
        return False

# loadpatch
patch_lightning_checkpoint_loading()


def save_plot_cache(analyzer, path):
    """for(model/)."""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    #  XAI, figures-only  XAI 
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
    print(f"✅ : {path}")


def load_plot_cache(path):
    """loading, dict (true_labels, probabilities, predictions, sensitive_attributes)."""
    import pickle
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


class FairnessAnalyzer:
    """fairness analysis"""
    
    def __init__(self, model_path=None, data_paths=None, checkpoint_path=None):
        """
        initializefairness analysis
        
        Args:
            model_path: modelcheckpoint
            data_paths: data file path
            checkpoint_path: PyTorch Lightning checkpoint
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
        #,
        self._fairness_cache = {}
        # ( --figures-only for)
        self.skip_statistical_tests = False
        # :for, 
        self.is_figures_only = False
        
    def load_data(self):
        """load"""
        print("=" * 80)
        print("📊 load")
        print("=" * 80)
        
        all_samples = []
        all_metadata = []
        
        # load
        for data_file in self.data_paths:
            file_path = Path(data_file)
            if not file_path.exists():
                print(f"⚠️  : {data_file}")
                continue
            
            print(f"\n📂 load: {file_path.name}")
            try:
                data = torch.load(file_path, weights_only=False)
                
                if 'spatiotemporal_samples' in data:
                    samples = data['spatiotemporal_samples']
                    dataset = FireTracksDataset(samples, target_type='binary_classification')
                    
                    # metadata(ISO3, GDP, )
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
                            'features': sample.get('features', None)  # forGDP
                        }
                        all_metadata.append(metadata)
                    
                    all_samples.extend(samples)
                    print(f"   ✅ load {len(samples):,} ")
                    
            except Exception as e:
                print(f"   ❌ load: {e}")
                import traceback
                traceback.print_exc()
        
        self.sample_metadata = all_metadata
        print(f"\n✅ load {len(all_samples):,} ")
        
        return all_samples
    
    def extract_sensitive_attributes(self, samples):
        """
        (GDP per capita, population density, Continent, ISO3)
        
        Args:
            samples: 
            
        Returns:
            dict: 
        """
        print("\n" + "=" * 80)
        print("🔍 ")
        print("=" * 80)
        
        # loadGDP per capita
        gdp_per_capita_path = Path('dataset/gdp_per_capita.csv')
        gdp_per_capita_df = None
        if gdp_per_capita_path.exists():
            try:
                gdp_per_capita_df = pd.read_csv(gdp_per_capita_path)
                print(f"✅ loadGDP per capita: {len(gdp_per_capita_df):,} ")
                print(f"   GDP per capita: {list(gdp_per_capita_df.columns[:5])}...")
                # Country Code
                if 'Country Code' in gdp_per_capita_df.columns:
                    unique_countries = gdp_per_capita_df['Country Code'].nunique()
                    print(f"   : {unique_countries}")
            except Exception as e:
                print(f"⚠️  loadGDP per capita: {e}")
        else:
            print(f"⚠️  GDP per capita: {gdp_per_capita_path}")
        
        # load(forcontinentGDP)
        covariate_data_path = Path('dataset/filtered_cleaned_cp_covariate.csv')
        covariate_df = None
        if covariate_data_path.exists():
            try:
                covariate_df = pd.read_csv(covariate_data_path)
                print(f"✅ load: {len(covariate_df):,} ")
                print(f"   : {list(covariate_df.columns)}")
                # 
                if 'iso3' in covariate_df.columns:
                    unique_iso3 = covariate_df['iso3'].dropna().nunique()
                    print(f"   ISO3: {unique_iso3}")
                if 'continent' in covariate_df.columns:
                    unique_continents = covariate_df['continent'].dropna().nunique()
                    print(f"   continent: {unique_continents}")
                    print(f"   continent: {covariate_df['continent'].value_counts().to_dict()}")
                if 'year' in covariate_df.columns:
                    year_range = (covariate_df['year'].min(), covariate_df['year'].max())
                    print(f"   : {year_range[0]}-{year_range[1]}")
            except Exception as e:
                print(f"⚠️  load: {e}")
        else:
            print(f"⚠️  : {covariate_data_path}")
        
        sensitive_attrs = {
            'iso3': [],
            'population': [],
            'pop_group': [],
            'gdp_per_capita': [],
            'gdp_group': [],
            'continent': [],
            'continent_group': []
        }
        
        # → KD-tree(for metadata['iso3'], .pth)
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
                    print(f"   ( iso3  2° )")
            except Exception:
                pass
        
        # 
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
            
            # 1: metadataISO3
            if 'iso3' in metadata:
                iso3 = metadata['iso3']
            
            # metadata
            pixel_date = sample.get('pixel_date', None)
            year = None
            if pixel_date:
                try:
                    year = pd.Timestamp(pixel_date).year
                except:
                    pass
            
            # (for; pixel_*  metadata  center)
            pixel_lat = sample.get('pixel_lat', None) or (metadata.get('center_lat') if isinstance(metadata.get('center_lat'), (int, float)) else None)
            pixel_lon = sample.get('pixel_lon', None) or (metadata.get('center_lon') if isinstance(metadata.get('center_lon'), (int, float)) else None)
            
            # 2: (3=population);GDP 4  4.5
            if features is not None:
                try:
                    # features shape: [time_steps, grid_size, grid_size, channels],3: population
                    if len(features.shape) >= 4:
                        pop_data = features[:, :, :, 3]
                        pop_nonzero = pop_data[pop_data > 0]
                        if len(pop_nonzero) > 0:
                            population = float(np.mean(pop_nonzero))
                except Exception as e:
                    pass
            
            # 3: ISO3,, continent
            # iso3,iso3
            if covariate_df is not None:
                try:
                    cov_rows = None
                    
                    # :iso3
                    if iso3:
                        if year:
                            cov_rows = covariate_df[(covariate_df['iso3'] == iso3) & (covariate_df['year'] == year)]
                        else:
                            cov_rows = covariate_df[covariate_df['iso3'] == iso3]
                    
                    # :iso3,()
                    if cov_rows is None or len(cov_rows) == 0:
                        if pixel_lat is not None and pixel_lon is not None and year:
                            # (0.1)
                            lat_tolerance = 0.1
                            lon_tolerance = 0.1
                            cov_rows = covariate_df[
                                (covariate_df['year'] == year) &
                                (abs(covariate_df['lat_mean'] - pixel_lat) <= lat_tolerance) &
                                (abs(covariate_df['lon_mean'] - pixel_lon) <= lon_tolerance)
                            ]
                    
                    if cov_rows is not None and len(cov_rows) > 0:
                        # ()
                        row = cov_rows.iloc[0]
                        
                        # iso3()
                        if not iso3 and 'iso3' in row:
                            iso3_val = row['iso3']
                            if pd.notna(iso3_val) and str(iso3_val).strip() != '':
                                iso3 = str(iso3_val).strip()
                        
                        # 
                        if population is None and 'population' in row:
                            pop_val = row['population']
                            if pd.notna(pop_val):
                                population = float(pop_val)
                        
                        # continent
                        if 'continent' in row:
                            cont_val = row['continent']
                            if pd.notna(cont_val) and str(cont_val).strip() != '':
                                continent = str(cont_val).strip()
                except Exception as e:
                    pass
            
            # 3.5:  iso3,(2° )( .pth)
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
            
            # 4:  GDP per capita CSV  iso3+(fairness analysis iso3  2°, .pth 4)
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
            
            # 4.5:  CSV  GDP,4(GDP)( .pth  iso3 40,4)
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
        
        # population density
        if pop_values:
            pop_33 = np.percentile(pop_values, 33)
            pop_67 = np.percentile(pop_values, 67)
            
            print(f"\n📊 population density:")
            print(f"   33: {pop_33:.2f}")
            print(f"   67: {pop_67:.2f}")
            
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
        
        # GDP per capita
        if gdp_per_capita_values:
            gdp_33 = np.percentile(gdp_per_capita_values, 33)
            gdp_67 = np.percentile(gdp_per_capita_values, 67)
            
            print(f"\n📊 GDP per capita:")
            print(f"   33: ${gdp_33:.2f}")
            print(f"   67: ${gdp_67:.2f}")
            
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
        
        # Continent(continentgrouping,high/medium/low)
        if continent_values:
            # continent
            continent_counts = {}
            for cont in continent_values:
                if cont:
                    continent_counts[cont] = continent_counts.get(cont, 0) + 1
            
            #,
            sorted_continents = sorted(continent_counts.items(), key=lambda x: x[1], reverse=True)
            total_continents = len(sorted_continents)
            
            # continent:High,Medium,Low
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
            
            print(f"\n📊 Continentgrouping:")
            print(f"   High: {high_continents}")
            print(f"   Medium: {medium_continents}")
            print(f"   Low: {low_continents}")
            
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
        
        # 
        n_samples = len(samples)
        print(f"\n📊 :")
        iso3_count = sum(1 for iso3 in sensitive_attrs['iso3'] if iso3)
        pop_count = sum(1 for pop in sensitive_attrs['population'] if pop is not None)
        gdp_count = sum(1 for gdp in sensitive_attrs['gdp_per_capita'] if gdp is not None)
        continent_count = sum(1 for cont in sensitive_attrs['continent'] if cont)
        
        pct = (lambda a, b: (a / b * 100) if b else 0)
        print(f"   ISO3: {iso3_count:,} ({pct(iso3_count, n_samples):.1f}%)")
        print(f"   : {pop_count:,} ({pct(pop_count, n_samples):.1f}%)")
        print(f"   GDP per capita: {gdp_count:,} ({pct(gdp_count, n_samples):.1f}%)")
        print(f"   Continent: {continent_count:,} ({pct(continent_count, n_samples):.1f}%)")
        
        # GDP per capita()
        if gdp_per_capita_values:
            print(f"\n   GDP per capita: ${min(gdp_per_capita_values):,.2f} - ${max(gdp_per_capita_values):,.2f}")
        
        # Continent()
        if continent_values:
            continent_dist = {}
            for cont in continent_values:
                continent_dist[cont] = continent_dist.get(cont, 0) + 1
            n_cont = len(continent_values)
            print(f"\n   Continent:")
            for cont, count in sorted(continent_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"     {cont}: {count:,} ({pct(count, n_cont):.1f}%)")
        
        print(f"\n   population densitygrouping:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['pop_group'].count(group)
            print(f"     {group}: {count:,} ({pct(count, n_samples):.1f}%)")
        
        print(f"\n   GDP per capitagrouping:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['gdp_group'].count(group)
            print(f"     {group}: {count:,} ({count/len(samples)*100:.1f}%)")
        
        print(f"\n   Continentgrouping:")
        for group in ['Low', 'Medium', 'High', 'Unknown']:
            count = sensitive_attrs['continent_group'].count(group)
            print(f"     {group}: {count:,} ({count/len(samples)*100:.1f}%)")
        
        self.sensitive_attributes = sensitive_attrs
        return sensitive_attrs
    
    def load_model_and_predict(self, test_dataset):
        """
        loadmodel
        
        Args:
            test_dataset: 
        """
        print("\n" + "=" * 80)
        print("🤖 loadmodel")
        print("=" * 80)
        
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path")
        
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint: {checkpoint_path}")
        
        print(f"📂 loadcheckpoint: {checkpoint_path}")
        self._load_model_only()
        print("✅ modelload")
        # 
        print("\n🔮 ...")
        all_predictions = []
        all_probs = []
        all_labels = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(dataloader):
                # model
                # features shape: [B, C, T, H, W], [B, T, C, H, W]
                features_permuted = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                logits = self.model(features_permuted)
                # modellog_softmax,
                probs = torch.exp(logits)[:, 1]  # 
                preds = (probs > 0.5).long()
                
                all_predictions.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   : {batch_idx + 1}/{len(dataloader)} ")
        
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probs)
        self.true_labels = np.array(all_labels)
        
        print(f"✅ : {len(self.predictions):,} ")
        
        # 
        print(f"\n📊 :")
        print(f"   : 0={np.sum(self.predictions==0):,} ({np.sum(self.predictions==0)/len(self.predictions)*100:.1f}%), "
              f"1={np.sum(self.predictions==1):,} ({np.sum(self.predictions==1)/len(self.predictions)*100:.1f}%)")
        print(f"   : 0={np.sum(self.true_labels==0):,} ({np.sum(self.true_labels==0)/len(self.true_labels)*100:.1f}%), "
              f"1={np.sum(self.true_labels==1):,} ({np.sum(self.true_labels==1)/len(self.true_labels)*100:.1f}%)")
        print(f"   : {self.probabilities.min():.4f} - {self.probabilities.max():.4f}")
        print(f"   : {self.probabilities.mean():.4f}, : {np.median(self.probabilities):.4f}")
        print(f"   >0.5: {np.sum(self.probabilities > 0.5):,}")
        print(f"   >0.3: {np.sum(self.probabilities > 0.3):,}")
        print(f"   >0.1: {np.sum(self.probabilities > 0.1):,}")
        
        return self.predictions, self.probabilities, self.true_labels
    
    def _load_model_only(self):
        """loadmodel,.for XAI, true_labels/probabilities."""
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path")
        try:
            patch_lightning_checkpoint_loading()
            self.model = ConvLSTM_fire_equality_model.load_from_checkpoint(
                str(Path(self.checkpoint_path)),
                map_location='cpu'
            )
            self.model.eval()
        except Exception as e:
            print(f"❌ modelload: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_performance_metrics(self, y_true, y_pred, y_proba=None):
        """
        
        
        Args:
            y_true: 
            y_pred: 
            y_proba: ()
            
        Returns:
            dict: 
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
        
        # 
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
        
        
        Args:
            y_true: 
            y_pred: 
            y_proba: 
            sensitive_attr: 
            cache_key: (),use_cache=True,
            use_cache: (True)
            
        Returns:
            dict: 
        """
        # 
        if use_cache and cache_key and cache_key in self._fairness_cache:
            print(f"\n✅  (key: {cache_key})")
            return self._fairness_cache[cache_key]
        
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("⚖️  ")
            if cache_key:
                print(f"   (: {cache_key})")
            print("=" * 80)
        
        fairness_metrics = {}
        
        # grouping
        unique_groups = np.unique(sensitive_attr)
        unique_groups = [g for g in unique_groups if g != 'Unknown']  # Unknown
        
        if len(unique_groups) < 2:
            print("⚠️  grouping2,")
            return fairness_metrics
        
        # 
        group_metrics = {}
        for group in unique_groups:
            mask = sensitive_attr == group
            if mask.sum() == 0:
                continue
            
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_proba = y_proba[mask] if y_proba is not None else None
            
            if not quiet:
                # 
                print(f"\n📊 {group} :")
                print(f"   : {mask.sum():,}")
                
                # (for)
                sample_indices = np.where(mask)[0]
                if len(sample_indices) > 0:
                    print(f"   : {sample_indices.min()} - {sample_indices.max()}")
                    print(f"   10: {sample_indices[:10].tolist()}")
                    if len(sample_indices) > 10:
                        print(f"   10: {sample_indices[-10:].tolist()}")
                
                print(f"   : 0={np.sum(group_y_true==0):,} ({np.sum(group_y_true==0)/len(group_y_true)*100:.1f}%), "
                      f"1={np.sum(group_y_true==1):,} ({np.sum(group_y_true==1)/len(group_y_true)*100:.1f}%)")
                print(f"   : 0={np.sum(group_y_pred==0):,} ({np.sum(group_y_pred==0)/len(group_y_pred)*100:.1f}%), "
                      f"1={np.sum(group_y_pred==1):,} ({np.sum(group_y_pred==1)/len(group_y_pred)*100:.1f}%)")
                if group_y_proba is not None:
                    print(f"   : {group_y_proba.min():.4f} - {group_y_proba.max():.4f}")
                    print(f"   : {group_y_proba.mean():.4f}, : {np.median(group_y_proba):.4f}")
                    print(f"   : {group_y_proba.std():.4f}")
                    print(f"   >0.5: {np.sum(group_y_proba > 0.5):,}")
                    
                    # (),
                    if group_y_proba.max() - group_y_proba.min() < 0.001:
                        print(f"   ⚠️  :!")
                        print(f"      grouping")
            
            metrics = self.calculate_performance_metrics(group_y_true, group_y_pred, group_y_proba)
            group_metrics[group] = metrics
            
            if not quiet:
                print(f"\n📊 {group} :")
                print(f"   accuracy: {metrics['accuracy']:.4f}")
                print(f"   precision: {metrics['precision']:.4f}")
                print(f"   recall: {metrics['recall']:.4f}")
                print(f"   F1: {metrics['f1']:.4f}")
                if metrics.get('auc') is not None:
                    print(f"   AUC: {metrics['auc']:.4f}")
                if metrics.get('auprc') is not None:
                    print(f"   AUPRC: {metrics['auprc']:.4f}")
                if 'tp' in metrics:
                    print(f"   : TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
        
        # 
        if HAS_FAIRLEARN:
            try:
                # Demographic Parity ()
                dp_diff = fl_metrics.demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                dp_ratio = fl_metrics.demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                
                # Equalized Odds ()
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
                
                print(f"\n⚖️  Fairlearn:")
                print(f"   Demographic Parity Difference: {dp_diff:.4f} (,0)")
                print(f"   Demographic Parity Ratio: {dp_ratio:.4f} (1)")
                print(f"   Equalized Odds Difference: {eo_diff:.4f} (,0)")
                print(f"   Equalized Odds Ratio: {eo_ratio:.4f} (1)")
                
            except Exception as e:
                print(f"⚠️  Fairlearn: {e}")
        
        # ()
        if not quiet:
            print(f"\n📊 :")
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
                    print(f"     : {max_val:.4f}, : {min_val:.4f}")
                    if max_val > 0:
                        print(f"     : {diff:.4f} ({diff/max_val*100:.1f}%)")
                    else:
                        print(f"     : {diff:.4f} (0)")
                    print(f"     : {ratio:.4f}" if max_val > 0 else f"     : N/A (0)")
        
        fairness_metrics['group_metrics'] = group_metrics
        
        # ()
        if (not getattr(self, 'skip_statistical_tests', False)) and HAS_SCIPY and len(unique_groups) >= 2:
            statistical_tests = self.perform_statistical_tests(
                y_true, y_pred, y_proba, sensitive_attr, group_metrics
            )
            fairness_metrics['statistical_tests'] = statistical_tests
        
        # 
        if use_cache and cache_key:
            self._fairness_cache[cache_key] = fairness_metrics
            print(f"\n💾  (key: {cache_key})")
        
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
        :.
         Integrated Gradients, pop_group .
        """
        if not CAPTUM_AVAILABLE:
            print("⚠️  captum,.: pip install captum")
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pop_group = np.array(sensitive_attrs['pop_group'])
        valid = pop_group != 'Unknown'
        if valid.sum() == 0:
            print("⚠️   pop_group,")
            return None
        unique_groups = [g for g in ['Low', 'Medium', 'High'] if (pop_group == g).sum() > 0]
        if len(unique_groups) < 2:
            print("⚠️   < 2,")
            return None
        # loadmodel, true_labels/probabilities( ROC )
        if self.model is None and self.checkpoint_path:
            self._load_model_only()
        if self.model is None:
            print("⚠️  loadmodel,")
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
                print(f"   : {k + 1}/{len(to_run)}")
        if not importance_list:
            print("⚠️  ")
            return None
        importance_matrix = np.array(importance_list)
        group_for_each = pop_group[sample_indices]
        results = {
            'importance_matrix': importance_matrix,
            'sample_indices': np.array(sample_indices),
            'group_for_each': group_for_each,
            'feature_names': FEATURE_NAMES,
        }
        #  XAI, figures-only 
        self._xai_results = results
        self._plot_attribution_boxplot_by_group(results, output_dir)
        self._plot_top5_features_by_group(results, output_dir)
        return results

    def _plot_attribution_boxplot_by_group(self, results, output_dir):
        """."""
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
        #  Low/Medium/High,
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
        # 
        ax.set_title(
            'Feature attribution by population-density group (Integrated Gradients)')
        # Legend,
        legend = ax.legend(title='Group', loc='best')
        plt.tight_layout()
        out_path = Path(output_dir) / 'xai_attribution_boxplot_by_pop_group.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ : {out_path}")

    def _plot_top5_features_by_group(self, results, output_dir):
        """ Top-5 ."""
        importance_matrix = results['importance_matrix']
        group_for_each = results['group_for_each']
        names = results['feature_names']
        order = [g for g in ['Low', 'Medium', 'High'] if (group_for_each == g).any()]
        fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
        if len(order) == 1:
            axes = [axes]

        #  High  y ( High,)
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
            # :x,y 
            ax.bar([names[i] for i in idx], mean_imp[idx],
                   color={'Low': 'C0', 'Medium': 'C1', 'High': 'C2'}.get(group, 'C0'))
            if i == 0:
                ax.set_ylabel('Mean |attribution|')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Feature')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            #  y, High  10%
            if ylim_max is not None:
                ax.set_ylim(0, ylim_max)
            #,
            ax.set_title(f'{group} (n={mask.sum()})', fontsize=18)
        #  suptitle:
        plt.suptitle('Top-5 feature importance by group', y=0.98)
        plt.tight_layout()
        out_path = Path(output_dir) / 'xai_top5_features_by_pop_group.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Top-5 : {out_path}")

    def perform_statistical_tests(self, y_true, y_pred, y_proba, sensitive_attr, group_metrics):
        """
       ,
        
        Args:
            y_true: 
            y_pred: 
            y_proba: 
            sensitive_attr: 
            group_metrics: 
            
        Returns:
            dict: 
        """
        print("\n" + "=" * 80)
        print("📊 ")
        print("=" * 80)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        if len(unique_groups) < 2:
            return {}
        
        test_results = {}
        
        # 1. :
        print("\n1️⃣  (Chi-square Test)")
        print("   ")
        contingency_tables = {}
        
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                mask1 = sensitive_attr == group1
                mask2 = sensitive_attr == group2
                
                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue
                
                pred1 = y_pred[mask1]
                pred2 = y_pred[mask2]
                
                # 
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
                    
                    significance = "******" if p_value < 0.001 else "****" if p_value < 0.01 else "**" if p_value < 0.05 else ""
                    print(f"   {group1} vs {group2}:")
                    print(f"     : {chi2:.4f}")
                    print(f"     p: {p_value:.6f} {significance}")
                    print(f"     : {dof}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}:  - {e}")
        
        test_results['chi_square'] = contingency_tables
        
        # 2. recallBootstrap
        print("\n2️⃣  Bootstrap(Bootstrap Confidence Intervals)")
        print("   recall95%")
        
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
                    """recall"""
                    y_t1, y_p1 = data1
                    y_t2, y_p2 = data2
                    recall1 = recall_score(y_t1, y_p1, zero_division=0)
                    recall2 = recall_score(y_t2, y_p2, zero_division=0)
                    return recall2 - recall1
                
                try:
                    # Bootstrap
                    n_resamples = 10000
                    differences = []
                    
                    for _ in range(n_resamples):
                        # 
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
                    
                    # 
                    actual_diff = recall_score(y_true2, y_pred2, zero_division=0) - recall_score(y_true1, y_pred1, zero_division=0)
                    
                    key = f"{group1}_vs_{group2}"
                    bootstrap_results[key] = {
                        'actual_difference': actual_diff,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'excludes_zero': ci_lower > 0 or ci_upper < 0
                    }
                    
                    excludes_zero = "***0***" if bootstrap_results[key]['excludes_zero'] else "0"
                    print(f"   {group1} vs {group2} recall:")
                    print(f"     : {actual_diff:.4f}")
                    print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] {excludes_zero}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}: Bootstrap - {e}")
        
        test_results['bootstrap'] = bootstrap_results
        
        # 3. Cohen's h(for)
        print("\n3️⃣  Cohen's h(Effect Size)")
        print("   recall")
        
        def cohens_h(p1, p2):
            """Cohen's h"""
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
                
                # 
                if abs(h) < 0.2:
                    effect_size = ""
                elif abs(h) < 0.5:
                    effect_size = ""
                elif abs(h) < 0.8:
                    effect_size = ""
                else:
                    effect_size = ""
                
                key = f"{group1}_vs_{group2}"
                cohens_h_results[key] = {
                    'h': h,
                    'effect_size': effect_size,
                    'recall1': recall1,
                    'recall2': recall2
                }
                
                print(f"   {group1} vs {group2}:")
                print(f"     Cohen's h: {h:.4f} ({effect_size})")
                print(f"     recall: {group1}={recall1:.4f}, {group2}={recall2:.4f}")
        
        test_results['cohens_h'] = cohens_h_results
        
        # 4. Mann-Whitney U:
        print("\n4️⃣  Mann-Whitney U(Mann-Whitney U Test)")
        print("   ")
        
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
                    
                    significance = "******" if p_value < 0.001 else "****" if p_value < 0.01 else "**" if p_value < 0.05 else ""
                    
                    key = f"{group1}_vs_{group2}"
                    mw_results[key] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_prob1': np.mean(prob1),
                        'mean_prob2': np.mean(prob2)
                    }
                    
                    print(f"   {group1} vs {group2}:")
                    print(f"     U: {statistic:.2f}")
                    print(f"     p: {p_value:.6f} {significance}")
                    print(f"     : {group1}={np.mean(prob1):.4f}, {group2}={np.mean(prob2):.4f}")
                    
                except Exception as e:
                    print(f"   {group1} vs {group2}:  - {e}")
        
        test_results['mann_whitney'] = mw_results
        
        # 5. 
        print("\n5️⃣  (Confusion Matrix Analysis)")
        print("   ")
        
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
            total_positive = tp + fn  # 
            total_negative = tn + fp  # 
            
            # 
            false_negative_rate = fn / total_positive if total_positive > 0 else 0
            false_positive_rate = fp / total_negative if total_negative > 0 else 0
            
            cm_analysis[group] = {
                'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
                'false_negative_rate': false_negative_rate,
                'false_positive_rate': false_positive_rate,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'missed_fires': fn,  # 
                'false_alarms': fp   # 
            }
            
            print(f"   {group}:")
            print(f"     : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"      (FNR): {false_negative_rate:.4f} ({fn}/{total_positive})")
            print(f"      (FPR): {false_positive_rate:.4f} ({fp}/{total_negative})")
            print(f"     : {fn:,}")
            print(f"     : {fp:,}")
        
        test_results['confusion_matrix_analysis'] = cm_analysis
        
        # 6. 
        print("\n" + "=" * 80)
        print("📊 ")
        print("=" * 80)
        
        significant_tests = []
        if test_results.get('chi_square'):
            for key, result in test_results['chi_square'].items():
                if result['p_value'] < 0.05:
                    significant_tests.append(f" ({key}): p={result['p_value']:.6f}")
        
        if test_results.get('mann_whitney'):
            for key, result in test_results['mann_whitney'].items():
                if result['p_value'] < 0.05:
                    significant_tests.append(f"Mann-Whitney U ({key}): p={result['p_value']:.6f}")
        
        if significant_tests:
            print("✅ :")
            for test in significant_tests:
                print(f"   - {test}")
        else:
            print("⚠️  ()")
        
        return test_results
    
    def optimize_group_specific_thresholds(self, y_true, y_proba, sensitive_attr, 
                                          metric='f1', threshold_range=(0.1, 0.9), 
                                          num_thresholds=200, min_precision=0.3, 
                                          min_recall=0.3, use_validation_split=False,
                                          validation_ratio=0.2):
        """
        threshold
        
        Args:
            y_true: 
            y_proba: 
            sensitive_attr: 
            metric:  ('f1', 'recall', 'balanced_accuracy', 'balanced_f1')
            threshold_range: threshold
            num_thresholds: threshold
            min_precision: precision(forbalanced_f1)
            min_recall: recall(forbalanced_f1)
            
        Returns:
            dict: threshold
        """
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("🔧 threshold")
            print("=" * 80)
        
        #,
        if use_validation_split:
            if not quiet:
                print(f"📊  (: {validation_ratio})")
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
                print("⚠️  ()")
            y_true_train = y_true
            y_proba_train = y_proba
            sensitive_train = sensitive_attr
            y_true_val = y_true
            y_proba_val = y_proba
            sensitive_val = sensitive_attr
        
        unique_groups = np.unique(sensitive_train)
        unique_groups = [g for g in unique_groups if g != 'Unknown']
        
        if len(unique_groups) < 2:
            print("⚠️  grouping2,threshold")
            return {}
        
        optimal_thresholds = {}
        optimized_metrics = {}
        
        #,
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
                print(f"\n🔍  {group} threshold (: {metric})...")
            
            # ()
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
                # ()
                for threshold in thresholds:
                    y_pred_thresh = (group_y_proba_train > threshold).astype(int)
                    
                    # threshold()
                    current_metrics_train = self.calculate_performance_metrics(
                        group_y_true_train, y_pred_thresh, group_y_proba_train
                    )
                    current_precision = current_metrics_train['precision']
                    current_recall = current_metrics_train['recall']
                    
                    # metric
                    if metric == 'f1':
                        score = current_metrics_train['f1']
                    elif metric == 'recall':
                        score = current_recall
                    elif metric == 'balanced_accuracy':
                        # accuracy:
                        # :precisionrecall
                        cm = confusion_matrix(group_y_true_train, y_pred_thresh)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            balanced_acc = (sensitivity + specificity) / 2
                            
                            # :precisionrecall,
                            # :precision [0.4, 0.95],recall [0.3, 0.8]
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
                        # F1:precisionrecall
                        if current_precision >= min_precision and current_recall >= min_recall:
                            #,F1
                            score = current_metrics_train['f1']
                        else:
                            #,
                            precision_penalty = max(0, min_precision - current_precision) * 2
                            recall_penalty = max(0, min_recall - current_recall) * 2
                            score = current_metrics_train['f1'] - precision_penalty - recall_penalty
                    else:
                        score = current_metrics_train['f1']
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            # threshold
            y_pred_val = (group_y_proba_val > best_threshold).astype(int)
            best_metrics = self.calculate_performance_metrics(
                group_y_true_val, y_pred_val, group_y_proba_val
            )
            
            optimal_thresholds[group] = best_threshold
            optimized_metrics[group] = best_metrics
            
            if not quiet:
                print(f"   ✅ threshold: {best_threshold:.4f}")
                print(f"   📊 {'' if use_validation_split else ''}:")
                print(f"      accuracy: {best_metrics['accuracy']:.4f}")
                print(f"      precision: {best_metrics['precision']:.4f}")
                print(f"      recall: {best_metrics['recall']:.4f}")
                print(f"      F1: {best_metrics['f1']:.4f}")
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
        :
        
        Args:
            y_true: 
            y_proba: 
            sensitive_attr: 
            performance_weight: 
            fairness_weight: 
            performance_metric:  ('f1', 'balanced_accuracy')
            threshold_range: threshold
            num_thresholds: threshold
            use_validation_split: 
            validation_ratio: 
            
        Returns:
            dict: threshold
        """
        print("\n" + "=" * 80)
        print("🎯 ( + )")
        print("=" * 80)
        print(f"   : {performance_weight}, : {fairness_weight}")
        
        # 
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
            print("⚠️  grouping2,")
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
            
            print(f"\n🔍  {group} threshold ()...")
            
            for threshold in thresholds:
                y_pred_train = (group_y_proba_train > threshold).astype(int)
                
                # 
                if performance_metric == 'f1':
                    perf_score = f1_score(group_y_true_train, y_pred_train)
                elif performance_metric == 'balanced_accuracy':
                    perf_score = balanced_accuracy_score(group_y_true_train, y_pred_train)
                else:
                    perf_score = f1_score(group_y_true_train, y_pred_train)
                
                # ()
                # :
                # 
                fairness_score = 1.0  #,
                
                # 
                combined_score = (performance_weight * perf_score + 
                                fairness_weight * fairness_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
            
            # 
            y_pred_val = (group_y_proba_val > best_threshold).astype(int)
            best_metrics = self.calculate_performance_metrics(
                group_y_true_val, y_pred_val, group_y_proba_val
            )
            
            optimal_thresholds[group] = best_threshold
            optimized_metrics[group] = best_metrics
            
            print(f"   ✅ threshold: {best_threshold:.4f}")
            print(f"   📊 {'' if use_validation_split else ''}:")
            print(f"      F1: {best_metrics['f1']:.4f}")
            print(f"      precision: {best_metrics['precision']:.4f}")
            print(f"      recall: {best_metrics['recall']:.4f}")
        
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
        threshold
        
        Args:
            y_true: 
            y_proba: 
            sensitive_attr: 
            methods: 
            threshold_range: threshold
            num_thresholds: threshold
            use_validation_split: 
            validation_ratio: 
            
        Returns:
            dict: 
        """
        print("\n" + "=" * 80)
        print("📊 threshold")
        print("=" * 80)
        
        results = {}
        
        for method in methods:
            print(f"\n{'='*80}")
            print(f"🔍 : {method}")
            print(f"{'='*80}")
            
            if method in ['youden', 'pr_optimal', 'f2']:
                # 
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
        
        # 
        print("\n" + "=" * 80)
        print("📊 ")
        print("=" * 80)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        
        for group in unique_groups:
            print(f"\n{group} :")
            print(f"{'':<20} {'threshold':<10} {'F1':<10} {'precision':<10} {'recall':<10}")
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
        threshold
        
        Args:
            y_proba: 
            sensitive_attr: 
            optimal_thresholds: threshold
            
        Returns:
            np.array: 
        """
        y_pred_optimized = np.zeros_like(y_proba, dtype=int)
        
        for group, threshold in optimal_thresholds.items():
            mask = sensitive_attr == group
            y_pred_optimized[mask] = (y_proba[mask] > threshold).astype(int)
        
        # Unknown(threshold0.5)
        unknown_mask = sensitive_attr == 'Unknown'
        if unknown_mask.sum() > 0:
            y_pred_optimized[unknown_mask] = (y_proba[unknown_mask] > 0.5).astype(int)
        
        return y_pred_optimized
    
    def plot_threshold_tradeoff_curves(self, y_true, y_proba, sensitive_attr,
                                      output_dir='fairness_results',
                                      threshold_range=(0.1, 0.9), num_thresholds=50,
                                      optimal_thresholds=None):
        """
        threshold--
        
        Args:
            y_true: 
            y_proba: 
            sensitive_attr: 
            output_dir: 
            threshold_range: threshold
            num_thresholds: threshold
            optimal_thresholds: threshold( {'Low': 0.40, 'Medium': 0.42, 'High': 0.47})
        """
        print("\n" + "=" * 80)
        print("📊 threshold")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        unique_groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
        if len(unique_groups) < 2:
            print("⚠️  grouping2,")
            return
        ordered_groups = [g for g in ['Low', 'Medium', 'High'] if g in unique_groups] + [
            g for g in unique_groups if g not in ['Low', 'Medium', 'High']
        ]
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        # threshold
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
        
        print("threshold...")
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            # 
            overall_f1 = f1_score(y_true, y_pred)
            overall_precision = precision_score(y_true, y_pred, zero_division=0)
            overall_recall = recall_score(y_true, y_pred, zero_division=0)
            overall_accuracy = accuracy_score(y_true, y_pred)
            
            metrics_by_threshold['overall']['f1'].append(overall_f1)
            metrics_by_threshold['overall']['precision'].append(overall_precision)
            metrics_by_threshold['overall']['recall'].append(overall_recall)
            metrics_by_threshold['overall']['accuracy'].append(overall_accuracy)
            
            # 
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
            
            # 
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
        
        # 
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1 vs threshold
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
        #  sentence case
        ax1.set_title('F1 score vs threshold', fontsize=24)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label=r'$\tau_0$ = 0.5')
        # threshold tau_g( ROC :Low/Medium/High -> C0/C1/C2)
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
        
        # 2. precision vs recall(PR)
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
        
        # 3.  vs threshold
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
        
        # 4.  vs 
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
            
            # threshold0.5
            default_idx = np.argmin(np.abs(thresholds - 0.5))
            ax4.scatter(metrics_by_threshold['overall']['equalized_odds_diff'][default_idx],
                       metrics_by_threshold['overall']['f1'][default_idx],
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidth=1, label=r'$\tau_0$ = 0.5', zorder=5)
            # threshold(optimised tau_g)-
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
        print(f"✅ threshold: {output_file}")
        plt.close()
        
        return metrics_by_threshold
    
    def plot_roc_curves(self, output_dir='fairness_results', title=None):
        """
         ROC : + grouping.

        - population densitygrouping (Low/Medium/High) -> roc_curve_overall_and_pop_group.png
        - GDPgrouping (Low/Medium/High) -> roc_curve_overall_and_gdp_group.png
        - continentgrouping (Low/Medium/High)    -> roc_curve_overall_and_continent_group.png

        for \"Overall performance and initial unfairness\" .
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        if self.sensitive_attributes is None or self.true_labels is None or self.probabilities is None:
            print("⚠️ , ROC ")
            return

        #  ROC (:)
        roc_fs = {
            'title': 16,
            'label': 14,
            'tick': 12,
            'legend': 11,
        }

        def _draw_roc_on_ax(ax, y_true, y_proba, groups, attr_key, verbose=False):
            """ ax  Overall + grouping ROC,()."""
            fpr_all, tpr_all, _ = roc_curve(y_true, y_proba, pos_label=1, drop_intermediate=False)
            auc_all = roc_auc_score(y_true, y_proba)
            ax.plot(fpr_all, tpr_all, color='black', lw=2, linestyle='--',
                    label=f'Overall (AUC = {auc_all:.3f})')
            if verbose and attr_key == 'pop_group':
                n_unique_all = len(np.unique(y_proba))
                print(f"   [ROC ] Overall: {len(fpr_all)} threshold, {n_unique_all} ")

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
                    print(f"   [ROC ] {group}: {len(fpr)} threshold, {len(np.unique(p_g))} ")

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
            """ (y_true, y_proba, groups)  None()."""
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
            """ ROC : 'ROC curves:' ()."""
            if attr_key not in self.sensitive_attributes:
                return
            attr_values = np.array(self.sensitive_attributes[attr_key])
            mask = attr_values != 'Unknown'
            if mask.sum() == 0:
                print(f"⚠️  grouping ({attr_key}), {filename} ")
                return
            y_true = np.asarray(self.true_labels)[mask]
            y_proba = np.asarray(self.probabilities)[mask]
            groups = attr_values[mask]
            if np.unique(y_true).size < 2:
                print(f"⚠️  {attr_key}, ROC ")
                return

            n_unique_all = len(np.unique(y_proba))
            if n_unique_all <= 2:
                print(f"⚠️  [ROC] {attr_key}  {n_unique_all},;( --figures-only).")

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            _draw_roc_on_ax(ax, y_true, y_proba, groups, attr_key, verbose=True)
            # : "ROC curves:",
            ax.set_title(title or default_title, fontsize=roc_fs['title'], fontweight='normal', pad=8)
            fig.tight_layout()
            out_file = output_path / filename
            #  tight,
            fig.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"✅ ROC : {out_file}")

        # ( "ROC curves:",)
        _titles = {
            'pop_group': 'Overall and by population-density group',
            'gdp_group': 'Overall and by GDP-per-capita group',
            'continent_group': 'Overall and by continent group',
        }

        # 1) population densitygrouping
        _plot_for_group(
            attr_key='pop_group',
            filename='roc_curve_overall_and_pop_group.png',
            default_title=_titles['pop_group'],
        )
        # 2) GDPgrouping
        _plot_for_group(
            attr_key='gdp_group',
            filename='roc_curve_overall_and_gdp_group.png',
            default_title=_titles['gdp_group'],
        )
        # 3) continentgrouping
        _plot_for_group(
            attr_key='continent_group',
            filename='roc_curve_overall_and_continent_group.png',
            default_title=_titles['continent_group'],
        )

        # 4) : ROC, "ROC curves:" 
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
            print(f"✅ ROC : {out_combined}")
        else:
            print("⚠️  grouping, ROC ")
    
    def visualize_results(self, output_dir='fairness_results', optimization_results=None):
        """
        fairness analysis
        
        Args:
            output_dir: 
            optimization_results: threshold()
        """
        print("\n" + "=" * 80)
        print("📊 ")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if self.sensitive_attributes is None:
            print("⚠️ ,")
            return
        
        # 1. (population densitygrouping)
        attr_values = self.sensitive_attributes['pop_group']
        unique_groups = [g for g in np.unique(attr_values) if g != 'Unknown']
        if len(unique_groups) < 2:
            print("⚠️  population densitygrouping2,")
            return
        
        # (threshold0.5)
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
        
        # 
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
                #,
                ax.set_title(f'{metric_label} comparison', fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 
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
        print(f"✅ : {save_path}")
        plt.close()
        
        #,
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
            
            # ()
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
                    #  performance_comparison,
                    ax.set_title(f'{metric_label} comparison', fontsize=22)
                    ax.set_xticks(x)
                    ax.set_xticklabels(groups)
                    #,
                    if idx == 0:
                        ax.legend(fontsize=14)
                    ax.tick_params(axis='both', labelsize=18)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 
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
            print(f"✅ : {save_path}")
            plt.close()
        
        print(f"\n✅,: {output_path}")
    
    def generate_report(self, output_path='fairness_report.md', 
                       optimization_results=None):
        """
        fairness analysis
        
        Args:
            output_path: 
            optimization_results: threshold()
        """
        print("\n" + "=" * 80)
        print("📝 ")
        print("=" * 80)
        
        # 
        overall_metrics = self.calculate_performance_metrics(
            self.true_labels, self.predictions, self.probabilities
        )
        
        # (population densitygrouping)
        fairness_metrics_pop = None
        
        if self.sensitive_attributes:
            pop_groups = np.array(self.sensitive_attributes['pop_group'])
            
            # Unknown
            pop_mask = pop_groups != 'Unknown'
            
            if pop_mask.sum() > 0:
                #,()
                # population_density_originalpopulation densitygroupingthreshold
                fairness_metrics_pop = self.calculate_fairness_metrics(
                    self.true_labels[pop_mask],
                    self.predictions[pop_mask],
                    self.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_original',
                    use_cache=True
                )
        
        # Markdown
        report_lines = [
            "# modelfairness analysis\n",
            "## 1. (threshold0.5)\n",
            f"- **accuracy**: {overall_metrics['accuracy']:.4f}",
            f"- **precision**: {overall_metrics['precision']:.4f}",
            f"- **recall**: {overall_metrics['recall']:.4f}",
            f"- **F1**: {overall_metrics['f1']:.4f}",
        ]
        
        if overall_metrics.get('auc') is not None:
            report_lines.append(f"- **AUC**: {overall_metrics['auc']:.4f}")
        if overall_metrics.get('auprc') is not None:
            report_lines.append(f"- **AUPRC**: {overall_metrics['auprc']:.4f}")
        
        if fairness_metrics_pop:
            if 'group_metrics' in fairness_metrics_pop:
                report_lines.append("\n### 2.1 (threshold0.5)\n")
                for group, metrics in fairness_metrics_pop['group_metrics'].items():
                    report_lines.append(f"#### {group} population density\n")
                    report_lines.append(f"- accuracy: {metrics['accuracy']:.4f}")
                    report_lines.append(f"- precision: {metrics['precision']:.4f}")
                    report_lines.append(f"- recall: {metrics['recall']:.4f}")
                    report_lines.append(f"- F1: {metrics['f1']:.4f}")
                    if metrics.get('auc') is not None:
                        report_lines.append(f"- AUC: {metrics['auc']:.4f}")
                    if metrics.get('auprc') is not None:
                        report_lines.append(f"- AUPRC: {metrics['auprc']:.4f}")
                    report_lines.append("")
            
            report_lines.append("### 2.2 (threshold0.5)\n")
            if 'demographic_parity_difference' in fairness_metrics_pop:
                report_lines.append(f"- **Demographic Parity Difference**: {fairness_metrics_pop['demographic_parity_difference']:.4f}")
                report_lines.append(f"- **Demographic Parity Ratio**: {fairness_metrics_pop['demographic_parity_ratio']:.4f}")
                report_lines.append(f"- **Equalized Odds Difference**: {fairness_metrics_pop['equalized_odds_difference']:.4f}")
                report_lines.append(f"- **Equalized Odds Ratio**: {fairness_metrics_pop['equalized_odds_ratio']:.4f}")
        
            # 
            if 'statistical_tests' in fairness_metrics_pop:
                report_lines.append("\n### 2.3 \n")
                stat_tests = fairness_metrics_pop['statistical_tests']
                
                # 
                if 'chi_square' in stat_tests and stat_tests['chi_square']:
                    report_lines.append("#### 2.3.1 (Chi-square Test)\n")
                    report_lines.append(".\n")
                    report_lines.append("|  |  | p |  |\n")
                    report_lines.append("|---------|-----------|-----|--------|\n")
                    for key, result in stat_tests['chi_square'].items():
                        groups = key.replace('_vs_', ' vs ')
                        p_val = result['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        report_lines.append(f"| {groups} | {result['chi2']:.4f} | {p_val:.6f} | {sig} |\n")
                    report_lines.append("\n*: *** p<0.001, ** p<0.01, * p<0.05, ns=*\n")
                
                # Bootstrap
                if 'bootstrap' in stat_tests and stat_tests['bootstrap']:
                    report_lines.append("#### 2.3.2 Bootstrap(Bootstrap Confidence Intervals)\n")
                    report_lines.append("recall95%.\n")
                    report_lines.append("|  |  | 95% CI | 95% CI | 0 |\n")
                    report_lines.append("|---------|---------|-----------|-----------|----------|\n")
                    for key, result in stat_tests['bootstrap'].items():
                        groups = key.replace('_vs_', ' vs ')
                        excludes_zero = "()" if result['excludes_zero'] else "()"
                        report_lines.append(f"| {groups} | {result['actual_difference']:.4f} | "
                                         f"{result['ci_lower']:.4f} | {result['ci_upper']:.4f} | {excludes_zero} |\n")
                    report_lines.append("\n")
                
                # Cohen's h
                if 'cohens_h' in stat_tests and stat_tests['cohens_h']:
                    report_lines.append("#### 2.3.3 Cohen's h(Effect Size)\n")
                    report_lines.append("recall.|h| < 0.2,0.2-0.5,0.5-0.8,>0.8.\n")
                    report_lines.append("|  | Cohen's h |  | recall |\n")
                    report_lines.append("|---------|----------|---------|-----------|\n")
                    for key, result in stat_tests['cohens_h'].items():
                        groups = key.replace('_vs_', ' vs ')
                        report_lines.append(f"| {groups} | {result['h']:.4f} | {result['effect_size']} | "
                                         f"{result['recall1']:.4f} vs {result['recall2']:.4f} |\n")
                    report_lines.append("\n")
                
                # Mann-Whitney U
                if 'mann_whitney' in stat_tests and stat_tests['mann_whitney']:
                    report_lines.append("#### 2.3.4 Mann-Whitney U(Mann-Whitney U Test)\n")
                    report_lines.append(".\n")
                    report_lines.append("|  | U | p |  |  |\n")
                    report_lines.append("|---------|---------|-----|--------|------------|\n")
                    for key, result in stat_tests['mann_whitney'].items():
                        groups = key.replace('_vs_', ' vs ')
                        p_val = result['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        prob_comp = f"{result['mean_prob1']:.4f} vs {result['mean_prob2']:.4f}"
                        report_lines.append(f"| {groups} | {result['statistic']:.2f} | {p_val:.6f} | {sig} | {prob_comp} |\n")
                    report_lines.append("\n")
                
                # 
                if 'confusion_matrix_analysis' in stat_tests and stat_tests['confusion_matrix_analysis']:
                    report_lines.append("#### 2.3.5 (Confusion Matrix Analysis)\n")
                    report_lines.append(",(False Negative Rate).\n")
                    report_lines.append("|  | TP | FP | TN | FN | (FNR) | (FPR) |  |\n")
                    report_lines.append("|------|----|----|----|----|------------|------------|----------|\n")
                    for group, analysis in stat_tests['confusion_matrix_analysis'].items():
                        cm = analysis['confusion_matrix']
                        report_lines.append(f"| {group} | {cm['TP']} | {cm['FP']} | {cm['TN']} | {cm['FN']} | "
                                         f"{analysis['false_negative_rate']:.4f} | {analysis['false_positive_rate']:.4f} | "
                                         f"{analysis['missed_fires']:,} |\n")
                    report_lines.append("\n")
                
                # 
                report_lines.append("#### 2.3.6 \n")
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
                
                report_lines.append(f"- : {total_tests}\n")
                report_lines.append(f"- : {significant_count} ({significant_count/total_tests*100:.1f}%)\n")
                
                if significant_count > 0:
                    report_lines.append(f"\n****:,{significant_count},")
                    report_lines.append("**model**.\n")
                else:
                    report_lines.append("\n****:,.\n")
        
        # threshold
        if optimization_results:
            report_lines.append("\n## 3. threshold\n")
            
            optimal_thresholds = optimization_results.get('optimal_thresholds', {})
            optimized_metrics = optimization_results.get('optimized_metrics', {})
            
            report_lines.append("### 3.1 threshold\n")
            for group, threshold in optimal_thresholds.items():
                report_lines.append(f"- **{group}**: {threshold:.4f}")
            
            if optimized_metrics:
                report_lines.append("\n### 3.2 \n")
                for group, metrics in optimized_metrics.items():
                    original_metrics = fairness_metrics_pop['group_metrics'].get(group, {})
                    report_lines.append(f"#### {group} population density(threshold: {optimal_thresholds.get(group, 0.5):.4f})\n")
                    report_lines.append(f"- accuracy: {metrics['accuracy']:.4f} "
                                     f"(: {original_metrics.get('accuracy', 0):.4f}, "
                                     f": {metrics['accuracy'] - original_metrics.get('accuracy', 0):+.4f})")
                    report_lines.append(f"- precision: {metrics['precision']:.4f} "
                                     f"(: {original_metrics.get('precision', 0):.4f}, "
                                     f": {metrics['precision'] - original_metrics.get('precision', 0):+.4f})")
                    report_lines.append(f"- recall: {metrics['recall']:.4f} "
                                     f"(: {original_metrics.get('recall', 0):.4f}, "
                                     f": {metrics['recall'] - original_metrics.get('recall', 0):+.4f})")
                    report_lines.append(f"- F1: {metrics['f1']:.4f} "
                                     f"(: {original_metrics.get('f1', 0):.4f}, "
                                     f": {metrics['f1'] - original_metrics.get('f1', 0):+.4f})")
                    report_lines.append("")
                
                # 
                if self.sensitive_attributes:
                    pop_groups = np.array(self.sensitive_attributes['pop_group'])
                    pop_mask = pop_groups != 'Unknown'
                    if pop_mask.sum() > 0:
                        optimized_predictions = self.apply_group_specific_thresholds(
                            self.probabilities[pop_mask],
                            pop_groups[pop_mask],
                            optimal_thresholds
                        )
                        
                        #,()
                        # population_density_optimizedpopulation densitygroupingthreshold
                        optimized_fairness = self.calculate_fairness_metrics(
                            self.true_labels[pop_mask],
                            optimized_predictions,
                            self.probabilities[pop_mask],
                            pop_groups[pop_mask],
                            cache_key='population_density_optimized',
                            use_cache=True
                        )
                        
                        report_lines.append("### 3.3 \n")
                        if 'demographic_parity_difference' in optimized_fairness:
                            original_dp_diff = fairness_metrics_pop.get('demographic_parity_difference', 0)
                            optimized_dp_diff = optimized_fairness['demographic_parity_difference']
                            report_lines.append(f"- **Demographic Parity Difference**: {optimized_dp_diff:.4f} "
                                             f"(: {original_dp_diff:.4f}, "
                                             f": {original_dp_diff - optimized_dp_diff:+.4f})")
                            
                            original_eo_diff = fairness_metrics_pop.get('equalized_odds_difference', 0)
                            optimized_eo_diff = optimized_fairness['equalized_odds_difference']
                            report_lines.append(f"- **Equalized Odds Difference**: {optimized_eo_diff:.4f} "
                                             f"(: {original_eo_diff:.4f}, "
                                             f": {original_eo_diff - optimized_eo_diff:+.4f})")
        
        report_lines.append("\n## 4. \n")
        report_lines.append("fairness analysis,:\n")
        report_lines.append("1.,model")
        report_lines.append("2.,")
        report_lines.append("3. threshold()")
        report_lines.append("4. fairlearn,")
        
        # 
        report_path = Path(output_path)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ : {report_path}")
    
    def analyze_multiple_groupings(self, output_dir='fairness_results', 
                                   existing_optimization_results=None):
        """
        (population density, GDP per capita, Continent)
        optimized threshold
        
        Args:
            output_dir: 
            existing_optimization_results:, 
                {grouping_type: optimization_results},
        """
        quiet = getattr(self, 'is_figures_only', False)
        
        if not quiet:
            print("\n" + "=" * 80)
            print("📊 (Optimized Threshold)")
            print("=" * 80)
        
        if self.sensitive_attributes is None:
            print("⚠️ ,")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 
        grouping_types = {
            'population_density': {
                'attr_key': 'pop_group',
                'name': 'Population Density',
                'name_cn': 'population density'
            },
            'gdp_per_capita': {
                'attr_key': 'gdp_group',
                'name': 'GDP per Capita',
                'name_cn': 'GDP'
            },
            'continent': {
                'attr_key': 'continent_group',
                'name': 'Continent',
                'name_cn': 'continent'
            }
        }
        
        results_summary = {}
        
        for grouping_type, config in grouping_types.items():
            if not quiet:
                print(f"\n{'='*80}")
                print(f"📊 : {config['name_cn']} ({config['name']})")
                print(f"{'='*80}")
            
            attr_key = config['attr_key']
            if attr_key not in self.sensitive_attributes:
                print(f"⚠️   {attr_key},")
                continue
            
            attr_values = np.array(self.sensitive_attributes[attr_key])
            unique_groups = [g for g in np.unique(attr_values) if g != 'Unknown']
            
            if len(unique_groups) < 2:
                print(f"⚠️  {config['name_cn']}grouping2,")
                continue
            
            # Unknown
            mask = attr_values != 'Unknown'
            if mask.sum() == 0:
                print(f"⚠️ ,")
                continue
            
            y_true_filtered = self.true_labels[mask]
            y_proba_filtered = self.probabilities[mask]
            attr_filtered = attr_values[mask]
            
            # threshold0.5(threshold),“1”
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
            
            # ()
            optimization_results = None
            if existing_optimization_results and grouping_type in existing_optimization_results:
                if not quiet:
                    print(f"\n✅  {config['name_cn']} threshold()")
                optimization_results = existing_optimization_results[grouping_type]
                optimal_thresholds = optimization_results.get('optimal_thresholds', {})
            else:
                # optimized threshold
                if not quiet:
                    print(f"\n🔧  {config['name_cn']} groupingthreshold...")
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
                    print(f"⚠️  threshold, {config['name_cn']}")
                    continue
                
                optimal_thresholds = optimization_results['optimal_thresholds']
            
            # threshold
            optimized_predictions = self.apply_group_specific_thresholds(
                y_proba_filtered,
                attr_filtered,
                optimal_thresholds
            )
            
            # 
            group_metrics = {}
            if not quiet:
                print(f"\n📊 {config['name_cn']} (Optimized Threshold):")
                print("-" * 80)
                print(f"{'':<15} {'threshold':<10} {'accuracy':<10} {'precision':<10} {'recall':<10} {'F1':<10} {'AUC':<10}")
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
            
            # 
            cache_key = f"{grouping_type}_optimized"
            fairness_metrics = self.calculate_fairness_metrics(
                y_true_filtered,
                optimized_predictions,
                y_proba_filtered,
                attr_filtered,
                cache_key=cache_key,
                use_cache=True
            )
            
            # 
            results_summary[grouping_type] = {
                'name': config['name'],
                'name_cn': config['name_cn'],
                'group_metrics': group_metrics,
                'optimal_thresholds': optimal_thresholds,
                'fairness_metrics': fairness_metrics,
                'optimization_results': optimization_results
            }
            
            # 
            self._plot_grouping_performance(
                group_metrics, optimal_thresholds, config, output_path
            )
        
        #  figures-only, --figures-only 
        if not getattr(self, 'is_figures_only', False):
            self._generate_comparison_summary(results_summary, output_path)
        
        return results_summary
    
    def _plot_grouping_performance(self, group_metrics, optimal_thresholds, config, output_path):
        """"""
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
                #  performance_comparison_optimized.png,
                ax.set_title(f'{metric_label} comparison', fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 
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
        print(f"✅ : {save_path}")
        plt.close()
    
    def _generate_comparison_summary(self, results_summary, output_path):
        """"""
        if getattr(self, 'is_figures_only', False):
            # figures-only (),
            return
        
        print("\n" + "=" * 80)
        print("📊 ")
        
        # 
        comparison_lines = [
            "# (Optimized Threshold)\n",
            "## 1. \n"
        ]
        
        # 
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            group_metrics = result['group_metrics']
            optimal_thresholds = result['optimal_thresholds']
            
            comparison_lines.append(f"### {name_cn} ({result['name']})\n")
            comparison_lines.append("|  | threshold | accuracy | precision | recall | F1 | AUC |\n")
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
        
        # 
        comparison_lines.append("## 2. \n")
        comparison_lines.append("|  | Equalized Odds Difference | Demographic Parity Difference |\n")
        comparison_lines.append("|---------|-------------------------|-------------------------------|\n")
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            fairness = result['fairness_metrics']
            eo_diff = fairness.get('equalized_odds_difference', 'N/A')
            dp_diff = fairness.get('demographic_parity_difference', 'N/A')
            
            eo_str = f"{eo_diff:.4f}" if isinstance(eo_diff, (int, float)) else "N/A"
            dp_str = f"{dp_diff:.4f}" if isinstance(dp_diff, (int, float)) else "N/A"
            
            comparison_lines.append(f"| {name_cn} | {eo_str} | {dp_str} |\n")
        
        # 
        comparison_lines.append("\n## 3. \n")
        comparison_lines.append(",.\n\n")
        comparison_lines.append("|  |  |  |  |\n")
        comparison_lines.append("|---------|------------|----------|----------|\n")
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            fairness = result['fairness_metrics']
            stat_tests = fairness.get('statistical_tests', {})
            
            significant_count = 0
            total_tests = 0
            
            # Mann-Whitney U
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
        
        # 
        comparison_path = output_path / 'grouping_comparison_summary.md'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(''.join(comparison_lines))
        
        print(f"✅ : {comparison_path}")
        
        # 
        print("\n" + "-" * 80)
        print("F1:")
        print("-" * 80)
        print(f"{'':<20} {'Low':<12} {'Medium':<12} {'High':<12}")
        print("-" * 80)
        
        for grouping_type, result in results_summary.items():
            name_cn = result['name_cn']
            group_metrics = result['group_metrics']
            
            low_f1 = group_metrics.get('Low', {}).get('f1', 0)
            medium_f1 = group_metrics.get('Medium', {}).get('f1', 0)
            high_f1 = group_metrics.get('High', {}).get('f1', 0)
            
            print(f"{name_cn:<20} {low_f1:<12.4f} {medium_f1:<12.4f} {high_f1:<12.4f}")


def main():
    """"""
    import argparse
    
    parser = argparse.ArgumentParser(description='modelfairness analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='modelcheckpoint(--figures-only )')
    parser.add_argument('--data', type=str, nargs='+', default=None,
                       help='data file path,(--figures-only )')
    parser.add_argument('--output', type=str, default='fairness_results',
                        help='')
    parser.add_argument('--xai', action='store_true',
                        help=': (Integrated Gradients)')
    parser.add_argument('--figures-only', action='store_true',
                       help=',loadmodel( --cache)')
    parser.add_argument('--cache', type=str, default=None,
                       help=':(--figures-only)( output/fairness_plot_cache.pkl)')
    
    args = parser.parse_args()
    
    # :loading,()
    if args.figures_only:
        cache_path = args.cache or str(Path(args.output) / 'fairness_plot_cache.pkl')
        if not Path(cache_path).exists():
            print(f"❌ : {cache_path}")
            print("   fairness analysis.")
            sys.exit(1)
        print("📂 loading,...")
        cache = load_plot_cache(cache_path)
        analyzer = FairnessAnalyzer()
        #,,
        # .
        analyzer.skip_statistical_tests = True
        analyzer.is_figures_only = True
        analyzer.true_labels = cache['true_labels']
        analyzer.probabilities = cache['probabilities']
        analyzer.predictions = cache['predictions']
        analyzer.sensitive_attributes = cache['sensitive_attributes']
        #  XAI, figures-only  XAI 
        xai_results = cache.get('xai_results', None)
        if xai_results is not None:
            try:
                analyzer._plot_attribution_boxplot_by_group(xai_results, args.output)
                analyzer._plot_top5_features_by_group(xai_results, args.output)
            except Exception as e:
                print(f"⚠️   XAI : {e}")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        # 1) population density(threshold)
        analyzer.visualize_results(output_dir=args.output)
        # 2)  ROC 
        analyzer.plot_roc_curves(output_dir=args.output)
        # 3) threshold()
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
                    print(f"⚠️  threshold: {e}")
        # 4) population densitythreshold(),
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
                # thresholdthreshold, Low/Medium/High  tau_g 
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
                    print(f"⚠️   tau_g threshold: {e}")
                analyzer.visualize_results(output_dir=args.output, optimization_results=optimization_results)
        # 5) (population density,GDP/continent + grouping_comparison_summary.md)
        if optimization_results is not None:
            existing_optimization = {'population_density': optimization_results}
            analyzer.analyze_multiple_groupings(output_dir=args.output, existing_optimization_results=existing_optimization)
        print("✅,:", args.output)
        return
    
    if not args.checkpoint or not args.data:
        print("❌  --checkpoint  --data")
        parser.print_help()
        sys.exit(1)
    
    # 
    analyzer = FairnessAnalyzer(
        checkpoint_path=args.checkpoint,
        data_paths=args.data
    )
    
    # load
    samples = analyzer.load_data()
    
    # 
    sensitive_attrs = analyzer.extract_sensitive_attributes(samples)
    
    # 
    test_dataset = FireTracksDataset(samples, target_type='binary_classification')
    
    # 
    print("\n" + "=" * 80)
    print("🔍 ")
    print("=" * 80)
    print(f": {len(samples)}")
    print(f": {len(test_dataset)}")
    
    # features
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
                print(f"❌  {i} !")
                print(f"   shape: {sample_features.shape}")
                print(f"   shape: {dataset_features.shape}")
    
    if consistent_count == min(10, len(samples)):
        print(f"✅ {min(10, len(samples))},")
    else:
        print(f"⚠️  !{consistent_count}/{min(10, len(samples))}")
    
    # loadmodel
    predictions, probabilities, true_labels = analyzer.load_model_and_predict(test_dataset)
    
    #, --figures-only model
    cache_path = Path(args.output) / 'fairness_plot_cache.pkl'
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_plot_cache(analyzer, cache_path)

    # :()
    if getattr(args, 'xai', False) and CAPTUM_AVAILABLE:
        print("\n" + "=" * 80)
        print("🔬 : (XAI)")
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
            print(f"⚠️  : {e}")
            import traceback
            traceback.print_exc()
        #  XAI, --figures-only  XAI 
        save_plot_cache(analyzer, cache_path)
        print("✅ XAI .")
    
    # ()
    analyzer.visualize_results(output_dir=args.output)
    analyzer.plot_roc_curves(output_dir=args.output)
    
    # threshold()
    if analyzer.sensitive_attributes:
        pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
        pop_mask = pop_groups != 'Unknown'
        
        if pop_mask.sum() > 0 and len(analyzer.true_labels[pop_mask]) < 100000:
            #,
            print("\n" + "=" * 80)
            print("📊 threshold()")
            print("=" * 80)
            try:
                analyzer.plot_threshold_tradeoff_curves(
                    analyzer.true_labels[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    output_dir=args.output,
                    threshold_range=(0.1, 0.9),
                    num_thresholds=50  # threshold
                )
            except Exception as e:
                print(f"⚠️  threshold: {e}")
                print("   ")
        else:
            print("\n⚠️ ,threshold(plot_threshold_tradeoff_curves)")
    
    # threshold
    optimization_results = None
    if analyzer.sensitive_attributes:
        pop_groups = np.array(analyzer.sensitive_attributes['pop_group'])
        pop_mask = pop_groups != 'Unknown'
        
        if pop_mask.sum() > 0:
            # balanced_accuracy,precisionrecall
            #,recall
            # uses by default,
            optimization_results = analyzer.optimize_group_specific_thresholds(
                analyzer.true_labels[pop_mask],
                analyzer.probabilities[pop_mask],
                pop_groups[pop_mask],
                metric='balanced_accuracy',  # accuracy,precisionrecall
                threshold_range=(0.1, 0.9),
                num_thresholds=200,
                min_precision=0.4,  # precision40%()
                min_recall=0.3,     # recall30%
                use_validation_split=True,  # 
                validation_ratio=0.2  # 20%
            )
            # thresholdthreshold, Low/Medium/High  tau_g 
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
                print(f"⚠️   tau_g threshold: {e}")
            
            # :
            print("\n" + "=" * 80)
            print("🔬 :threshold")
            print("=" * 80)
            print(":()")
            # comparison_results = analyzer.compare_threshold_methods(
            #     analyzer.true_labels[pop_mask],
            #     analyzer.probabilities[pop_mask],
            #     pop_groups[pop_mask],
            #     methods=['f1', 'balanced_accuracy', 'youden', 'f2'],
            #     use_validation_split=True,
            #     validation_ratio=0.2
            # )
            
            # threshold
            if optimization_results and 'optimal_thresholds' in optimization_results:
                print("\n" + "=" * 80)
                print("📊 threshold")
                print("=" * 80)
                
                optimized_predictions = analyzer.apply_group_specific_thresholds(
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    optimization_results['optimal_thresholds']
                )
                
                # ()
                # population_density_optimizedpopulation densitygrouping
                optimized_fairness = analyzer.calculate_fairness_metrics(
                    analyzer.true_labels[pop_mask],
                    optimized_predictions,
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_optimized',
                    use_cache=True
                )
                
                print("\n📈 :")
                print("=" * 80)
                
                # ()
                # population_density_originalpopulation densitygrouping
                original_fairness = analyzer.calculate_fairness_metrics(
                    analyzer.true_labels[pop_mask],
                    analyzer.predictions[pop_mask],
                    analyzer.probabilities[pop_mask],
                    pop_groups[pop_mask],
                    cache_key='population_density_original',
                    use_cache=True
                )
                
                if 'group_metrics' in original_fairness and 'group_metrics' in optimized_fairness:
                    print("\nrecall:")
                    for group in ['Low', 'Medium', 'High']:
                        if group in original_fairness['group_metrics']:
                            orig_recall = original_fairness['group_metrics'][group]['recall']
                            opt_recall = optimized_fairness['group_metrics'][group]['recall']
                            improvement = opt_recall - orig_recall
                            print(f"  {group}: {orig_recall:.4f} → {opt_recall:.4f} "
                                f"(: {improvement:+.4f}, {improvement/orig_recall*100:+.1f}%)")
                
                if 'equalized_odds_difference' in original_fairness:
                    orig_eo = original_fairness['equalized_odds_difference']
                    opt_eo = optimized_fairness['equalized_odds_difference']
                    print(f"\nEqualized Odds Difference: {orig_eo:.4f} → {opt_eo:.4f} "
                          f"(: {orig_eo - opt_eo:+.4f})")
            
            # ()
            analyzer.visualize_results(
                output_dir=args.output,
                optimization_results=optimization_results
            )
    
    # (optimized threshold)
    #,population densitygrouping
    print("\n" + "=" * 80)
    print("📊 ")
    print("=" * 80)
    existing_optimization = {}
    if optimization_results:
        # population densitygrouping,passed toanalyze_multiple_groupings
        existing_optimization['population_density'] = optimization_results
        print("✅ population densitygroupingthreshold()")
    multiple_groupings_results = analyzer.analyze_multiple_groupings(
        output_dir=args.output,
        existing_optimization_results=existing_optimization
    )
    
    # ()
    analyzer.generate_report(
        output_path=Path(args.output) / 'fairness_report.md',
        optimization_results=optimization_results
    )
    
    print("\n" + "=" * 80)
    print("✅ fairness analysis!")
    print("=" * 80)


if __name__ == '__main__':
    main()

