"""
MODIS MCD12Q1 土地覆盖数据获取模块（仅使用 Google Earth Engine）

使用方法：
    from code.fire_equality.datamodules.get_modis_landcover import get_landcover_from_gee, LandCoverCache

    # 单点查询
    lc = get_landcover_from_gee(lat=34.05, lon=-118.25, year=2017, project='your-project')

    # 批量查询（推荐用于负样本创建）
    cache = LandCoverCache(method='gee', year=2017, project='your-project')
    lcs = cache.get_landcover_batch([(lat1, lon1), (lat2, lon2), ...], year=2017)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from pathlib import Path
import warnings
import sys

# 抑制Google API的Python版本警告（不影响功能，只是建议升级Python）
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    warnings.warn("Google Earth Engine (earthengine-api) 未安装。请运行: pip install earthengine-api")


# MCD12Q1 LC_Type1 土地覆盖类型编码
LC_TYPE1_CODES = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Snow and Ice",
    16: "Barren",
    255: "Unclassified"
}


def initialize_gee(project: Optional[str] = None, credentials_path: Optional[str] = None, silent: bool = False):
    """
    初始化Google Earth Engine
    
    Args:
        project: 可选，GEE项目名称（例如 'ee-tpan2203-wildfire'）
        credentials_path: 可选，服务账户密钥文件路径
        silent: 如果为True，失败时不抛出异常，只返回False
    
    Returns:
        bool: 如果silent=True，返回是否初始化成功；否则成功返回None，失败抛出异常
    """
    if not HAS_EE:
        if silent:
            return False
        raise ImportError("请先安装 earthengine-api: pip install earthengine-api")
    
    try:
        # 检查是否已经初始化
        ee.Number(1).getInfo()
        if not silent:
            print("✅ Google Earth Engine 已初始化")
        return True if silent else None
    except:
        pass
    
    # 检查是否已有凭据文件（避免不必要的认证流程）
    import platform
    import os
    from pathlib import Path
    
    # GEE凭据文件位置
    if platform.system() == 'Windows':
        creds_dir = Path.home() / '.config' / 'earthengine'
    else:
        creds_dir = Path.home() / '.config' / 'earthengine'
    
    creds_file = creds_dir / 'credentials'
    has_credentials = creds_file.exists()
    
    try:
        if project:
            # 使用项目名称初始化
            ee.Initialize(project=project)
            if not silent:
                print(f"✅ Google Earth Engine 初始化成功（项目: {project}）")
            return True if silent else None
        elif credentials_path:
            # 使用服务账户
            credentials = ee.ServiceAccountCredentials(None, credentials_path)
            ee.Initialize(credentials)
            if not silent:
                print("✅ Google Earth Engine 初始化成功（使用服务账户）")
            return True if silent else None
        else:
            # 尝试使用默认认证
            ee.Initialize()
            if not silent:
                print("✅ Google Earth Engine 初始化成功（使用默认认证）")
            return True if silent else None
    except Exception as e:
        # 如果初始化失败，检查是否需要认证
        error_msg = str(e).lower()
        needs_auth = 'credentials' in error_msg or 'authentication' in error_msg or 'not authenticated' in error_msg
        
        if not needs_auth and has_credentials:
            # 有凭据文件但初始化失败，可能是凭据过期或其他问题
            if not silent:
                print(f"⚠️  Google Earth Engine 初始化失败: {str(e)}")
                print("   检测到已有凭据文件，但可能已过期")
                print("   请运行: earthengine authenticate")
            if silent:
                return False
            raise
        
        # 如果没有凭据文件或需要认证
        if not silent:
            if has_credentials:
                print("⚠️  Google Earth Engine 凭据可能已过期，需要重新认证...")
            else:
                print("⚠️  Google Earth Engine 未认证，开始认证流程...")
        
        try:
            # 在Windows上，优先使用notebook模式（避免浏览器跳转）
            import subprocess
            
            # 检查是否有gcloud命令
            has_gcloud = False
            try:
                subprocess.run(['gcloud', '--version'], 
                             capture_output=True, check=True, timeout=5)
                has_gcloud = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                has_gcloud = False
            
            if has_gcloud:
                # 如果有gcloud，使用标准认证
                ee.Authenticate()
            else:
                # Windows上使用notebook模式（会显示授权URL和代码输入提示，不需要浏览器）
                if platform.system() == 'Windows':
                    if not silent:
                        print("   💡 使用notebook认证模式（会显示授权URL，复制到浏览器后粘贴授权码）")
                    try:
                        ee.Authenticate(auth_mode='notebook')
                    except Exception as notebook_error:
                        # 如果notebook模式失败，尝试默认方式
                        if not silent:
                            print(f"   ⚠️  notebook模式失败: {notebook_error}")
                            print("   尝试使用默认认证模式...")
                        ee.Authenticate()
                else:
                    # Linux/Mac使用gcloud模式
                    ee.Authenticate()
            
            # 认证成功后初始化
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            if not silent:
                print("✅ Google Earth Engine 认证并初始化成功")
                print("   💡 提示：凭据已保存，下次使用时无需再次认证")
            return True if silent else None
        except Exception as e2:
            error_msg = str(e2)
            if not silent:
                print(f"❌ Google Earth Engine 认证失败: {error_msg}")
                print("\n   手动认证步骤：")
                print("   1. 运行命令: earthengine authenticate")
                print("   2. 或者运行: earthengine authenticate --auth_mode=notebook")
                print("   3. 按照提示完成认证流程")
                print("\n   认证成功后，凭据会保存在本地，之后无需再次认证。")
            
            if silent:
                return False
            raise


def get_landcover_from_gee(lat: float, lon: float, year: int, 
                           lc_type: str = 'LC_Type1',
                           project: Optional[str] = None) -> Optional[int]:
    """
    从Google Earth Engine获取MODIS MCD12Q1土地覆盖类型
    
    Args:
        lat: 纬度
        lon: 经度
        year: 年份（MCD12Q1是年度产品）
        lc_type: 土地覆盖类型，可选 'LC_Type1', 'LC_Type2', 'LC_Type3', 'LC_Type4', 'LC_Type5'
        project: 可选，GEE项目名称
    
    Returns:
        土地覆盖类型编码（整数），如果失败返回None
    """
    if not HAS_EE:
        raise ImportError("请先安装 earthengine-api: pip install earthengine-api")
    
    try:
        # 初始化（如果尚未初始化）
        try:
            ee.Number(1).getInfo()  # 检查是否已初始化
        except:
            # 使用silent模式，失败时不抛出异常
            if not initialize_gee(project=project, silent=True):
                return None
        
        # 加载MCD12Q1数据集（使用新版本 MODIS/061/MCD12Q1）
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        
        # 筛选指定年份
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        
        # 选择土地覆盖类型波段
        lc_band = image.select(lc_type)
        
        # 创建点
        point = ee.Geometry.Point([lon, lat])
        
        # 提取值
        value = lc_band.sample(point, scale=500).first().get(lc_type)
        
        # 获取结果
        result = value.getInfo()
        
        if result is not None:
            return int(result)
        else:
            return None
            
    except Exception as e:
        print(f"⚠️  获取土地覆盖类型失败 ({lat}, {lon}, {year}): {e}")
        return None


def get_landcover_batch_gee(coords: List[Tuple[float, float]], 
                             year: int,
                             lc_type: str = 'LC_Type1',
                             batch_size: int = 1000,
                             project: Optional[str] = None) -> List[Optional[int]]:
    """
    批量从Google Earth Engine获取土地覆盖类型（优化版本）
    
    Args:
        coords: 坐标列表，每个元素为(lat, lon)
        year: 年份
        lc_type: 土地覆盖类型
        batch_size: 批处理大小
        project: 可选，GEE项目名称
    
    Returns:
        土地覆盖类型编码列表
    """
    if not HAS_EE:
        raise ImportError("请先安装 earthengine-api: pip install earthengine-api")
    
    try:
        # 初始化（如果尚未初始化）
        try:
            ee.Number(1).getInfo()  # 检查是否已初始化
        except:
            # 使用silent模式，失败时不抛出异常
            if not initialize_gee(project=project, silent=True):
                return [None] * len(coords)
        
        # 加载MCD12Q1数据集（使用新版本 MODIS/061/MCD12Q1）
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        lc_band = image.select(lc_type)
        
        results = []
        
        # 分批处理
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_start_idx = i  # 当前批次在原始列表中的起始索引
            
            # 验证坐标范围并过滤无效坐标
            valid_coords = []
            valid_orig_indices = []  # 在原始coords列表中的索引
            
            for local_idx, (lat, lon) in enumerate(batch_coords):
                orig_idx = batch_start_idx + local_idx
                # 验证坐标范围：纬度 -90 到 90，经度 -180 到 180
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    valid_coords.append((lat, lon))
                    valid_orig_indices.append(orig_idx)
            
            # 为无效坐标预先添加 None（确保结果列表长度正确）
            # 先扩展结果列表到当前批次结束的位置
            while len(results) < batch_start_idx + len(batch_coords):
                results.append(None)
            
            if len(valid_coords) == 0:
                # 如果整批都是无效坐标，所有结果都是 None（已预先添加）
                continue
            
            try:
                batch_num = i // batch_size + 1
                total_batches = (len(coords) + batch_size - 1) // batch_size
                print(f"      [批次 {batch_num}/{total_batches}] 正在查询 {len(valid_coords)} 个坐标的土地覆盖类型...", end='\r', flush=True)
                
                # 创建点集合（仅使用有效坐标）
                # 使用 valid_orig_indices 作为 id，以便正确映射回原始位置
                points = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([lon, lat]), {'id': orig_idx})
                    for orig_idx, (lat, lon) in zip(valid_orig_indices, valid_coords)
                ])
                
                # 采样（使用 scale=500 对应 MODIS 500m 分辨率）
                # 注意：sampleRegions 不支持 crs 参数，会自动使用图像的投影
                samples = lc_band.sampleRegions(
                    collection=points,
                    scale=500,
                    geometries=False
                )
                
                # 获取结果（这里可能会卡住，如果网络或API有问题）
                print(f"      [批次 {batch_num}/{total_batches}] 正在从GEE API获取结果（可能需要几秒到几分钟）...", end='\r', flush=True)
                sample_list = samples.getInfo()['features']
                print(f"      [批次 {batch_num}/{total_batches}] ✅ 成功获取 {len(sample_list)} 个结果", flush=True)
                
                # 构建结果字典（键为原始索引）
                batch_results = {}
                for feature in sample_list:
                    orig_idx = feature['properties']['id']
                    lc_value = feature['properties'].get(lc_type)
                    batch_results[orig_idx] = int(lc_value) if lc_value is not None else None
                
                # 更新结果列表（仅更新有效坐标的结果）
                for orig_idx in valid_orig_indices:
                    results[orig_idx] = batch_results.get(orig_idx, None)
                
            except Exception as batch_error:
                # 如果批量处理失败，为这批有效坐标返回 None
                print(f"⚠️  批量 {i//batch_size + 1} 处理失败: {batch_error}")
                for orig_idx in valid_orig_indices:
                    results[orig_idx] = None
        
        # 确保结果列表长度与输入坐标列表长度一致
        while len(results) < len(coords):
            results.append(None)
        
        return results
        
    except Exception as e:
        print(f"⚠️  批量获取土地覆盖类型失败: {e}")
        return [None] * len(coords)


def find_locations_by_landcover(positive_lc_types: List[int],
                                spatial_bounds: dict,
                                year: int,
                                num_samples_per_type: int = 1000,
                                project: Optional[str] = None) -> List[dict]:
    """
    直接在 MODIS 数据中搜索特定土地覆盖类型的区域位置
    
    这个方法通过 GEE 的栅格操作，直接在 MODIS 数据中筛选出指定土地覆盖类型的区域，
    然后采样这些区域的像素位置，返回坐标和对应的土地覆盖类型。
    
    Args:
        positive_lc_types: 正样本需要的土地覆盖类型列表，例如 [7, 9, 10]
        spatial_bounds: 研究区域的空间边界，格式为 {
            'lat_min': float, 'lat_max': float,
            'lon_min': float, 'lon_max': float
        }
        year: 年份
        num_samples_per_type: 每种土地覆盖类型采样的像素数量，默认1000
        project: 可选，GEE项目名称
    
    Returns:
        位置列表，每个元素为 {
            'lat': float,
            'lon': float,
            'land_cover': int
        }
    """
    if not HAS_EE:
        raise ImportError("请先安装 earthengine-api: pip install earthengine-api")
    
    try:
        # 初始化（如果尚未初始化）
        try:
            ee.Number(1).getInfo()  # 检查是否已初始化
        except:
            # 使用silent模式，失败时不抛出异常
            if not initialize_gee(project=project, silent=True):
                return []
        
        # 加载MCD12Q1数据集
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        lc_band = image.select('LC_Type1')
        
        # 创建研究区域
        region = ee.Geometry.Rectangle([
            spatial_bounds['lon_min'],
            spatial_bounds['lat_min'],
            spatial_bounds['lon_max'],
            spatial_bounds['lat_max']
        ])
        
        all_locations = []
        
        # 为每种土地覆盖类型分别采样
        for lc_type in positive_lc_types:
            # 创建掩膜：只保留当前土地覆盖类型
            mask = lc_band.eq(lc_type)
            
            # 应用掩膜并采样
            masked_lc = lc_band.updateMask(mask)
            
            # 尝试采样，如果找不到足够的样本，逐步增加采样数量
            current_num_samples = num_samples_per_type
            max_attempts = 3
            found_count = 0
            
            for attempt in range(max_attempts):
                try:
                    # 采样（只采样掩膜为 True 的像素）
                    # 使用 sample 方法，限制采样数量
                    samples = masked_lc.sample(
                        region=region,
                        scale=500,  # MODIS 分辨率 500m
                        numPixels=current_num_samples,
                        geometries=True  # 返回坐标
                    )
                    
                    # 获取采样结果
                    sample_list = samples.getInfo()['features']
                    
                    type_locations = []
                    for feature in sample_list:
                        coords = feature['geometry']['coordinates']
                        lc_value = feature['properties'].get('LC_Type1')
                        
                        # 验证土地覆盖类型
                        if lc_value == lc_type:
                            type_locations.append({
                                'lat': coords[1],  # GEE 返回的是 [lon, lat]
                                'lon': coords[0],
                                'land_cover': int(lc_value)
                            })
                    
                    found_count = len(type_locations)
                    
                    # 如果找到了位置，添加到总列表并跳出循环
                    if found_count > 0:
                        all_locations.extend(type_locations)
                        print(f"    类型 {lc_type}: 找到 {found_count} 个位置（采样数量: {current_num_samples}）")
                        break
                    else:
                        # 如果没找到，增加采样数量重试
                        if attempt < max_attempts - 1:
                            current_num_samples = current_num_samples * 3  # 增加3倍
                            print(f"    类型 {lc_type}: 未找到位置，增加采样数量到 {current_num_samples} 重试...")
                        else:
                            print(f"    ⚠️  类型 {lc_type}: 尝试 {max_attempts} 次后仍未找到位置（可能该类型在研究区域内不存在）")
                
                except Exception as e:
                    if attempt < max_attempts - 1:
                        current_num_samples = current_num_samples * 3
                        print(f"    ⚠️  类型 {lc_type} 采样失败（尝试 {attempt + 1}/{max_attempts}）: {e}")
                        print(f"       增加采样数量到 {current_num_samples} 重试...")
                    else:
                        print(f"    ⚠️  类型 {lc_type} 采样失败（已尝试 {max_attempts} 次）: {e}")
                        break
        
        print(f"  ✅ 总共找到 {len(all_locations):,} 个位置（包含 {len(set(l['land_cover'] for l in all_locations))} 种土地覆盖类型）")
        
        return all_locations
        
    except Exception as e:
        print(f"⚠️  搜索土地覆盖类型位置失败: {e}")
        import traceback
        traceback.print_exc()
        return []


class LandCoverCache:
    """
    土地覆盖类型缓存类，用于批量高效获取土地覆盖信息
    """
    
    def __init__(self, method: str = 'gee', 
                 year: Optional[int] = None,
                 project: Optional[str] = None):
        """
        初始化土地覆盖缓存
        
        Args:
            method: 方法，'gee' 或 'hdf'
            year: 年份（用于缓存）
            project: 可选，GEE项目名称（例如 'ee-tpan2203-wildfire'）
        """
        self.method = method
        self.year = year
        self.project = project
        self.cache = {}  # 缓存已查询的结果
        
        if method == 'gee':
            if not HAS_EE:
                raise ImportError("请先安装 earthengine-api: pip install earthengine-api")
            # 使用silent模式，失败时不抛出异常，只记录警告
            if not initialize_gee(project=project, silent=True):
                print(f"⚠️  GEE初始化失败，LandCoverCache将无法使用GEE功能")
                print("   请先运行: earthengine authenticate")
        else:
            raise ValueError("仅支持 method='gee'")
    
    def get_landcover(self, lat: float, lon: float, 
                     year: Optional[int] = None,
                     lc_type: str = 'LC_Type1') -> Optional[int]:
        """
        获取单个坐标的土地覆盖类型
        
        Args:
            lat: 纬度
            lon: 经度
            year: 年份（如果未提供，使用初始化时的year）
            lc_type: 土地覆盖类型
        
        Returns:
            土地覆盖类型编码
        """
        if year is None:
            year = self.year
        if year is None:
            raise ValueError("必须提供year参数")
        
        # 检查缓存（使用四舍五入到0.001度的坐标作为键，约100m精度）
        cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 获取土地覆盖类型（仅 GEE）
        lc = get_landcover_from_gee(lat, lon, year, lc_type, project=self.project)
        
        # 缓存结果
        self.cache[cache_key] = lc
        
        return lc
    
    def get_landcover_batch(self, coords: List[Tuple[float, float]],
                           year: Optional[int] = None,
                           lc_type: str = 'LC_Type1',
                           use_cache: bool = True) -> List[Optional[int]]:
        """
        批量获取土地覆盖类型（推荐用于负样本创建）
        
        Args:
            coords: 坐标列表，每个元素为(lat, lon)
            year: 年份
            lc_type: 土地覆盖类型
            use_cache: 是否使用缓存
        
        Returns:
            土地覆盖类型编码列表
        """
        if year is None:
            year = self.year
        if year is None:
            raise ValueError("必须提供year参数")
        
        # 初始化结果列表，确保长度与coords相同
        results = [None] * len(coords)
        
        # 使用批量API（仅 GEE）
        if use_cache:
            # 先检查缓存
            uncached_coords = []
            uncached_indices = []
            for idx, (lat, lon) in enumerate(coords):
                cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
                if cache_key in self.cache:
                    results[idx] = self.cache[cache_key]
                else:
                    uncached_coords.append((lat, lon))
                    uncached_indices.append(idx)
            
            # 批量获取未缓存的
            if uncached_coords:
                batch_results = get_landcover_batch_gee(uncached_coords, year, lc_type, 
                                                      project=self.project)
                # 更新缓存和结果
                for orig_idx, lc in zip(uncached_indices, batch_results):
                    lat, lon = coords[orig_idx]
                    cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
                    self.cache[cache_key] = lc
                    results[orig_idx] = lc
        else:
            results = get_landcover_batch_gee(coords, year, lc_type, project=self.project)
            # 确保结果列表长度正确
            if not results or len(results) != len(coords):
                results = [None] * len(coords)
        
        return results

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


# 便捷函数
def get_landcover(lat: float, lon: float, year: int, 
                 method: str = 'gee',
                 hdf_file: Optional[str] = None) -> Optional[int]:
    """
    便捷函数：获取土地覆盖类型
    
    Args:
        lat: 纬度
        lon: 经度
        year: 年份
        method: 方法，仅支持 'gee'
        hdf_file: 已废弃（无效参数）
    
    Returns:
        土地覆盖类型编码
    """
    if method != 'gee':
        raise ValueError("仅支持 method='gee'")
    return get_landcover_from_gee(lat, lon, year)


if __name__ == '__main__':
    # 抑制Python版本警告（不影响功能）
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')
    
    # 测试代码
    print("MODIS MCD12Q1 土地覆盖数据获取模块")
    print("=" * 50)
    
    # 测试坐标（多个不同位置）
    test_coords = [
        (34.05, -118.25, "洛杉矶"),
        (40.71, -74.01, "纽约"),
        (51.51, -0.13, "伦敦"),
        (39.90, 116.41, "北京"),
        (-33.87, 151.21, "悉尼"),
    ]
    test_year = 2017
    project = 'ee-tpan2203-wildfire'
    
    print(f"\n测试年份: {test_year}")
    print(f"GEE项目: {project}")
    print(f"测试坐标数量: {len(test_coords)}")
    
    # 保存结果的列表
    results_data = []
    
    # 测试Google Earth Engine方法（单个查询）
    if HAS_EE:
        print("\n1. 测试Google Earth Engine单个查询...")
        for lat, lon, name in test_coords[:3]:  # 只测试前3个
            try:
                lc = get_landcover_from_gee(lat, lon, test_year, project=project)
                if lc is not None:
                    lc_name = LC_TYPE1_CODES.get(lc, 'Unknown')
                    print(f"   ✅ {name} ({lat}, {lon}): LC={lc} ({lc_name})")
                    results_data.append({
                        'name': name,
                        'lat': lat,
                        'lon': lon,
                        'year': test_year,
                        'lc_code': lc,
                        'lc_name': lc_name
                    })
                else:
                    print(f"   ⚠️  {name}: 未获取到土地覆盖类型")
            except Exception as e:
                print(f"   ❌ {name} 错误: {e}")
    else:
        print("\n1. Google Earth Engine未安装，跳过测试")
    
    # 测试批量获取
    if HAS_EE:
        print("\n2. 测试批量获取...")
        try:
            cache = LandCoverCache(method='gee', year=test_year, project=project)
            batch_coords = [(lat, lon) for lat, lon, _ in test_coords]
            batch_results = cache.get_landcover_batch(batch_coords, test_year)
            
            print(f"   ✅ 批量获取完成:")
            for (lat, lon, name), lc in zip(test_coords, batch_results):
                if lc is not None:
                    lc_name = LC_TYPE1_CODES.get(lc, 'Unknown')
                    print(f"      {name} ({lat}, {lon}): LC={lc} ({lc_name})")
                    results_data.append({
                        'name': name,
                        'lat': lat,
                        'lon': lon,
                        'year': test_year,
                        'lc_code': lc,
                        'lc_name': lc_name
                    })
                else:
                    print(f"      {name}: 未获取到土地覆盖类型")
            
            print(f"   ✅ 缓存大小: {cache.get_cache_size()}")
        except Exception as e:
            print(f"   ❌ 批量获取错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果到CSV文件
    if results_data:
        print("\n3. 保存结果到文件...")
        try:
            import pandas as pd
            df = pd.DataFrame(results_data)
            # 去重（保留最后一个）
            df = df.drop_duplicates(subset=['lat', 'lon', 'year'], keep='last')
            
            output_file = 'dataset/modis_landcover_test_results.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ 结果已保存到: {output_file}")
            print(f"   ✅ 共保存 {len(df)} 条记录")
            print("\n   数据预览:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"   ⚠️  保存失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("测试完成！")

