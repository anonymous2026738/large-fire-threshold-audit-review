"""
特征时空对齐模块
用于将多种数据源的特征对齐到统一的时空网格

特征通道映射：
0: FWI (火灾天气指数)
1: VPD (蒸汽压差)
2: NDVI (归一化植被指数)
3: population (人口数，每像素1km²，单位：人/km²)
4: GDP (经济指标)
5: land_cover (土地覆盖)
6: max_temp (ERA5 2米最大温度)
7: max_wind (ERA5 10米最大风速，单位：m/s)
"""

import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import os
import sys
import threading
import logging

logger = logging.getLogger(__name__)

# 导入特征缓存模块
try:
    from .feature_cache import get_cached_feature, cache_feature, get_cache_stats
except ImportError:
    try:
        from code.fire_equality.datamodules.feature_cache import get_cached_feature, cache_feature, get_cache_stats
    except ImportError:
        # 如果导入失败，使用空函数（不启用缓存）
        def get_cached_feature(*args, **kwargs):
            return None
        def cache_feature(*args, **kwargs):
            pass
        def get_cache_stats():
            return {'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'cache_size': 0}

# 导入NetCDF数据集缓存模块
try:
    from .netcdf_cache import get_netcdf_dataset, clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
    HAS_NETCDF_CACHE = True
except ImportError:
    try:
        from code.fire_equality.datamodules.netcdf_cache import get_netcdf_dataset, clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
        HAS_NETCDF_CACHE = True
    except ImportError:
        # 如果导入失败，不使用预加载（回退到原来的方式）
        HAS_NETCDF_CACHE = False
        def get_netcdf_dataset(filepath, engine=None):
            # 回退到直接打开
            engines_to_try = [engine] if engine else ['netcdf4', 'h5netcdf', None]
            for eng in engines_to_try:
                try:
                    if eng:
                        return xr.open_dataset(filepath, engine=eng)
                    else:
                        return xr.open_dataset(filepath)
                except:
                    continue
            raise RuntimeError(f"无法打开NetCDF文件 {filepath}")
        def clear_netcdf_cache():
            pass
        def get_netcdf_cache_stats():
            return {'cached_datasets': 0, 'max_cache_size': 0}

# 导入同目录下的get_modis_landcover模块
try:
    from .get_modis_landcover import initialize_gee, HAS_EE
    HAS_GEE = HAS_EE
except ImportError:
    try:
        # 如果相对导入失败，尝试绝对导入
        from code.fire_equality.datamodules.get_modis_landcover import initialize_gee, HAS_EE
        HAS_GEE = HAS_EE
    except ImportError:
        # 最终回退：将当前目录加入sys.path并按普通模块名导入；若仍失败，直接使用 ee.Initialize
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from get_modis_landcover import initialize_gee, HAS_EE  # type: ignore
            HAS_GEE = HAS_EE
        except Exception:
            try:
                import ee  # type: ignore
                HAS_GEE = True
                def initialize_gee(project: str = None, silent: bool = False):  # type: ignore
                    try:
                        ee.Number(1).getInfo()
                        return True if silent else None
                    except Exception:
                        try:
                            if project:
                                ee.Initialize(project=project)
                            else:
                                ee.Initialize()
                            return True if silent else None
                        except Exception as e:
                            if silent:
                                return False
                            raise
            except Exception:
                HAS_GEE = False
                print("⚠️  GEE模块未找到，land_cover特征将不可用")
                def initialize_gee(project: str = None, silent: bool = False):  # type: ignore
                    if silent:
                        return False
                    raise ImportError("GEE模块不可用")


# ============================================================================
# 全局缓存：用于缓存dNBR计算结果，避免重复计算
# ============================================================================

# dNBR缓存：key = (fire_date_str, bounds_hash), value = dNBR数据数组
_dnbr_cache = {}
_dnbr_cache_max_size = 100  # 最大缓存条目数，避免内存溢出


def _get_bounds_hash(spatial_bounds, grid_size):
    """生成空间边界的哈希值，用于缓存"""
    import hashlib
    bounds_str = f"{spatial_bounds['lat_min']:.4f},{spatial_bounds['lat_max']:.4f},{spatial_bounds['lon_min']:.4f},{spatial_bounds['lon_max']:.4f},{grid_size}"
    return hashlib.md5(bounds_str.encode()).hexdigest()


def _clear_dnbr_cache():
    """清空dNBR缓存"""
    global _dnbr_cache
    _dnbr_cache.clear()


# ============================================================================
# 辅助函数：检测和映射NetCDF坐标名称
# ============================================================================

def get_coord_names(ds):
    """
    检测NetCDF数据集的坐标名称
    
    Args:
        ds: xarray Dataset
    
    Returns:
        dict: 包含 'lat', 'lon', 'time' 的映射字典
    """
    coords = {}
    dims = list(ds.dims.keys())
    coord_vars = list(ds.coords.keys())
    
    # 检测纬度坐标
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'lat' in name_lower:
            coords['lat'] = name
            break
    
    # 检测经度坐标
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'lon' in name_lower:
            coords['lon'] = name
            break
    
    # 检测时间坐标
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'time' in name_lower:
            coords['time'] = name
            break
    
    # 如果没找到，尝试默认名称
    if 'lat' not in coords:
        # 尝试常见的变体
        for candidate in ['Latitude', 'latitude', 'lat', 'y']:
            if candidate in ds.coords or candidate in ds.dims:
                coords['lat'] = candidate
                break
    
    if 'lon' not in coords:
        for candidate in ['Longitude', 'longitude', 'lon', 'x']:
            if candidate in ds.coords or candidate in ds.dims:
                coords['lon'] = candidate
                break
    
    if 'time' not in coords:
        for candidate in ['Time', 'time', 't']:
            if candidate in ds.coords or candidate in ds.dims:
                coords['time'] = candidate
                break
    
    return coords


def get_time_index(time_coords, current_date, filepath, feature_name="特征"):
    """
    获取时间坐标中最接近当前日期的索引
    
    Args:
        time_coords: 时间坐标数组
        current_date: 当前日期（datetime对象）
        filepath: 文件路径（用于错误信息）
        feature_name: 特征名称（用于错误信息）
    
    Returns:
        int: 时间索引
    
    Raises:
        ValueError: 如果时间坐标为空或无法计算索引
    """
    if len(time_coords) == 0:
        raise ValueError(f"{feature_name}文件 {filepath} 的时间坐标为空，无法提取数据")
    
    first_time_value = time_coords[0]
    
    try:
        if isinstance(first_time_value, (int, np.integer)) or np.issubdtype(type(first_time_value), np.integer):
            # 时间坐标是整数（day of year）
            day_of_year = current_date.timetuple().tm_yday
            time_idx = np.argmin(np.abs(time_coords - day_of_year))
        else:
            # 时间坐标是datetime
            time_diffs = [pd.Timestamp(ts).to_pydatetime() - current_date for ts in time_coords]
            if len(time_diffs) == 0:
                raise ValueError(f"{feature_name}文件 {filepath} 无法计算时间差异")
            time_idx = np.argmin(np.abs(time_diffs))
        
        # 确保索引有效
        if time_idx >= len(time_coords):
            time_idx = len(time_coords) - 1
        
        return time_idx
    except (IndexError, ValueError) as e:
        raise ValueError(f"{feature_name}文件 {filepath} 时间索引计算失败: {e}")


# ============================================================================
# 主函数：提取所有对齐后的特征
# ============================================================================

def extract_aligned_features(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    fire_date: datetime,
    fire_year: int,
    country_code: Optional[str] = None,
    iso3: Optional[str] = None,
    data_dir: str = 'dataset',
    project: str = 'ee-tpan2203-wildfire',
    max_retries: int = 3
) -> np.ndarray:
    """
    提取对齐后的特征立方体
    
    注意：NDVI特征仅支持GIMMS-3G+本地数据，不支持MOD13A2或GEE API。
    如果找不到GIMMS-3G+数据，将抛出FileNotFoundError。
    
    Args:
        spatial_bounds: 空间边界 {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        time_window_start: 时间窗口开始日期
        time_window_end: 时间窗口结束日期
        grid_size: 目标网格大小（例如25，对应25km×25km区域，1km分辨率）
        fire_date: 火灾发生日期（保留参数以兼容现有代码，但不再用于dNBR计算）
        fire_year: 火灾发生年份
        country_code: 国家代码（用于GDP匹配）
        iso3: ISO3代码（用于GDP匹配，优先使用）
        data_dir: 数据目录路径
        project: GEE项目名称
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size, 8]
            通道: 0-FWI, 1-VPD, 2-NDVI, 3-population(人口数，人/km²), 4-GDP, 5-land_cover, 6-max_temp, 7-max_wind
    """
    time_steps = (time_window_end - time_window_start).days + 1
    global _pop_zero_warning_count

    # 初始化特征立方体（8通道：FWI, VPD, NDVI, population, GDP, land_cover, max_temp, max_wind）
    # 使用NaN初始化，以便区分缺失数据和真实0值
    feature_cube = np.full((time_steps, grid_size, grid_size, 8), np.nan, dtype=np.float32)
    
    # 不打印详细信息，减少输出（进度条会显示）
    # print(f"  提取特征立方体: {time_steps} 时间步 × {grid_size}×{grid_size} 网格 × 8 通道")
    
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 辅助函数：带重试的特征提取
    def extract_with_retry(extract_func, feature_name, *args, **kwargs):
        """
        带重试机制的特征提取
        
        Args:
            extract_func: 提取函数
            feature_name: 特征名称（用于日志）
            *args, **kwargs: 传递给提取函数的参数
        
        Returns:
            提取的数据
        
        Raises:
            RuntimeError: 如果重试max_retries次后仍然失败
        """
        retry_delay = 2  # 初始重试延迟（秒）
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = extract_func(*args, **kwargs)
                # 如果成功，且之前有失败尝试，打印成功信息
                if attempt > 0:
                    print(f"    ✅ {feature_name}提取成功（第 {attempt + 1} 次尝试成功）")
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e)
                error_type = type(e).__name__
                
                # 判断是否应该重试
                # 文件不存在、配置错误等不应该重试
                should_retry = True
                if isinstance(e, (FileNotFoundError, ValueError, KeyError)):
                    # 文件不存在或配置错误，不重试
                    should_retry = False
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    should_retry = False
                
                if not should_retry:
                    # 不应该重试的错误，直接抛出
                    print(f"    ❌ {feature_name}提取失败（不可重试的错误）")
                    print(f"       错误类型: {error_type}")
                    print(f"       错误信息: {error_msg[:300]}")
                    raise
                
                # 应该重试的错误
                if attempt < max_retries - 1:
                    print(f"    ⚠️  {feature_name}提取失败 (尝试 {attempt + 1}/{max_retries})")
                    print(f"       错误类型: {error_type}")
                    print(f"       错误信息: {error_msg[:300]}")
                    print(f"       等待 {retry_delay} 秒后重试...")
                    time_module.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    # 最后一次尝试失败
                    print(f"    ❌ {feature_name}提取彻底失败（已重试 {max_retries} 次，全部失败）")
                    print(f"       错误类型: {error_type}")
                    print(f"       最后错误信息: {error_msg[:300]}")
                    raise RuntimeError(f"{feature_name}提取失败：已重试 {max_retries} 次，仍然失败。错误类型: {error_type}, 错误信息: {error_msg[:200]}")
        
        # 理论上不会到达这里
        raise RuntimeError(f"{feature_name}提取失败：未知错误")
    
    # 优化：并行提取FWI和VPD（它们都是NetCDF，可以并行读取）
    def extract_fwi_wrapper():
        return extract_with_retry(
            extract_fwi_feature, "FWI",
            spatial_bounds, time_window_start, time_window_end, 
            grid_size, fire_year, data_dir
        )
    
    def extract_vpd_wrapper():
        return extract_with_retry(
            extract_vpd_feature, "VPD",
            spatial_bounds, time_window_start, time_window_end,
            grid_size, fire_year, data_dir
        )
    
    def extract_era5_temp_wrapper():
        return extract_with_retry(
            extract_era5_temp_feature, "max_temp",
            spatial_bounds, time_window_start, time_window_end,
            grid_size, fire_year, data_dir
        )
    
    def extract_era5_wind_wrapper():
        return extract_with_retry(
            extract_era5_wind_feature, "max_wind",
            spatial_bounds, time_window_start, time_window_end,
            grid_size, fire_year, data_dir
        )
    
    # 提取FWI、VPD、ERA5温度和风速
    # 注意：NetCDF文件在Windows上可能不支持多线程同时读取，改为顺序读取以避免文件句柄冲突
    import platform
    use_parallel = platform.system() != 'Windows'
    
    try:
        if use_parallel:
            # Linux/Mac: 可以并行读取
            with ThreadPoolExecutor(max_workers=4) as executor:
                fwi_future = executor.submit(extract_fwi_wrapper)
                vpd_future = executor.submit(extract_vpd_wrapper)
                era5_temp_future = executor.submit(extract_era5_temp_wrapper)
                era5_wind_future = executor.submit(extract_era5_wind_wrapper)
                
                # 等待完成
                fwi_data = fwi_future.result()
                vpd_data = vpd_future.result()
                era5_temp_data = era5_temp_future.result()
                era5_wind_data = era5_wind_future.result()
        else:
            # Windows: 顺序读取以避免文件句柄冲突
            fwi_data = extract_fwi_wrapper()
            vpd_data = extract_vpd_wrapper()
            era5_temp_data = extract_era5_temp_wrapper()
            era5_wind_data = extract_era5_wind_wrapper()
        
        feature_cube[:, :, :, 0] = fwi_data
        feature_cube[:, :, :, 1] = vpd_data
        feature_cube[:, :, :, 6] = era5_temp_data
        feature_cube[:, :, :, 7] = era5_wind_data
    except RuntimeError as e:
        print(f"    ❌ 终止流水线: {e}")
        raise
    except Exception as e:
        print(f"    ⚠️  FWI/VPD/max_temp/max_wind提取失败（非重试错误）: {e}")
        raise
    
    # 提取NDVI特征（只支持GIMMS-3G+本地数据）
    gimms_ndvi_paths = [
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', f'gimms3g_ndvi_{fire_year}.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', 'gimms3g_ndvi.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', f'ndvi_{fire_year}.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', 'ndvi.nc'),
    ]
    
    # 检查GIMMS-3G+数据（支持新的文件命名格式）
    use_gimms_ndvi = any(os.path.exists(p) for p in gimms_ndvi_paths)
    if not use_gimms_ndvi:
        # 检查新格式：ndvi3g_geo_v1_X_YYYY_0106.nc4
        gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
        if os.path.exists(gimms_dir):
            # 检查是否有该年份的文件
            patterns = [
                f'ndvi3g_geo_v1_1_{fire_year}_*.nc4',
                f'ndvi3g_geo_v1_2_{fire_year}_*.nc4',
            ]
            for pattern in patterns:
                import glob
                matches = glob.glob(os.path.join(gimms_dir, pattern))
                if matches:
                    use_gimms_ndvi = True
                    break
    
    # 如果找不到GIMMS-3G+数据，抛出错误
    if not use_gimms_ndvi:
        gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
        error_msg = (
            f"❌ 未找到GIMMS-3G+ NDVI数据（年份: {fire_year}）\n"
            f"   流水线只支持GIMMS-3G+数据源\n"
            f"   请确保以下目录存在且包含数据文件：\n"
            f"   {gimms_dir}\n"
            f"   支持的文件命名格式：\n"
            f"   - ndvi3g_geo_v1_1_{fire_year}_0106.nc4 和 ndvi3g_geo_v1_1_{fire_year}_0712.nc4 (2002-2014)\n"
            f"   - ndvi3g_geo_v1_2_{fire_year}_0106.nc4 和 ndvi3g_geo_v1_2_{fire_year}_0712.nc4 (2015-2020)\n"
            f"   - gimms3g_ndvi_{fire_year}.nc 或 gimms3g_ndvi.nc (旧格式)"
        )
        raise FileNotFoundError(error_msg)
    
    # 提取NDVI（只使用GIMMS-3G+）
    try:
        ndvi_data = extract_with_retry(
            extract_ndvi_feature_gimms, "NDVI (GIMMS-3G+)",
            spatial_bounds, time_window_start, time_window_end,
            grid_size, fire_year, data_dir
        )
        feature_cube[:, :, :, 2] = ndvi_data
    except RuntimeError as e:
        print(f"    ❌ 终止流水线: {e}")
        raise
    except Exception as e:
        print(f"    ⚠️  NDVI提取失败（非重试错误）: {e}")
        raise
    
    # 优化：并行提取静态特征（Population、GDP、Land Cover）
    def extract_pop_wrapper():
        try:
            return extract_with_retry(
                extract_population_feature, "Population",
                spatial_bounds, grid_size, fire_year, data_dir
            )
        except Exception as e:
            # 确保错误信息被打印
            error_msg = str(e)
            filepath = os.path.join(data_dir, 'worldpop', f'ppp_{fire_year}_1km_Aggregated.tif')
            print(f"    ⚠️  Population提取失败: {error_msg[:200]}")
            print(f"       文件路径: {filepath}")
            print(f"       文件存在: {os.path.exists(filepath)}")
            raise  # 重新抛出异常，让外层处理
    
    def extract_gdp_wrapper():
        return extract_with_retry(
            extract_gdp_feature, "GDP",
            spatial_bounds, grid_size, fire_year, country_code, iso3, data_dir
        )
    
    def extract_lc_wrapper():
        return extract_with_retry(
            extract_landcover_feature_1km, "Land Cover",
            spatial_bounds, grid_size, fire_year, project, data_dir
        )
    
    # 并行提取静态特征（Population、GDP、Land Cover）
    # 初始化默认值（全零），如果提取失败则使用默认值
    pop_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    gdp_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    lc_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            pop_future = executor.submit(extract_pop_wrapper)
            gdp_future = executor.submit(extract_gdp_wrapper)
            lc_future = executor.submit(extract_lc_wrapper)
            
            # 等待完成，如果某个特征提取失败，使用默认值（全零）
            try:
                pop_data = pop_future.result()
                # 提取成功但全为 0：属正常（无人区、稀疏区或 WorldPop 无覆盖），仅记入日志
                if (pop_data == 0).all():
                    with _pop_zero_warning_lock:
                        _pop_zero_warning_count += 1
                        current_count = _pop_zero_warning_count
                    if current_count == 1 or current_count % 100 == 0:
                        logger.debug(
                            "Population 提取成功但数据全为 0（正常：无人区或 WorldPop 无覆盖），已出现 %d 次；"
                            "lat=[%.4f, %.4f], lon=[%.4f, %.4f], year=%s",
                            current_count,
                            spatial_bounds['lat_min'], spatial_bounds['lat_max'],
                            spatial_bounds['lon_min'], spatial_bounds['lon_max'],
                            fire_year,
                        )
            except Exception as e:
                # 错误信息已经在extract_pop_wrapper中打印
                error_msg = str(e)
                print(f"    ⚠️  Population提取最终失败，使用默认值0")
                print(f"       错误: {error_msg[:200]}")
                print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                print(f"       年份: {fire_year}, data_dir: {data_dir}")
                pop_data = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            try:
                gdp_data = gdp_future.result()
            except Exception as e:
                print(f"    ⚠️  GDP提取失败，使用默认值0: {str(e)[:200]}")
                gdp_data = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            try:
                lc_data = lc_future.result()
            except Exception as e:
                print(f"    ⚠️  Land Cover提取失败，使用默认值0: {str(e)[:200]}")
                lc_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # 静态特征：所有时间步使用相同值（通道索引已更新：移除dNBR后，索引前移）
        feature_cube[:, :, :, 3] = pop_data[np.newaxis, :, :]
        feature_cube[:, :, :, 4] = gdp_data[np.newaxis, :, :]
        feature_cube[:, :, :, 5] = lc_data[np.newaxis, :, :]
    except Exception as e:
        # 如果整个并行提取过程失败，使用默认值
        print(f"    ⚠️  静态特征提取过程失败，使用默认值0: {str(e)[:200]}")
        # 默认值已在上面初始化，直接使用
    
    # 将NaN值转换为0（PyTorch不支持NaN，但保留NaN标记以便后续过滤）
    # 注意：这里我们保留NaN，但在数据加载时会将NaN转换为0
    # 如果需要在训练时处理缺失值，可以在DataLoader中处理
    feature_cube = np.nan_to_num(feature_cube, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_cube


# ============================================================================
# 通道 0: FWI (火灾天气指数) - NetCDF数据
# ============================================================================

def extract_fwi_feature(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从NetCDF文件提取FWI特征（使用直接索引方法，参考merge_weather_indices.ipynb）
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 尝试从缓存获取
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'FWI', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'FWI', f'fire_weather_index_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FWI文件不存在: {filepath}")
    
    # 使用NetCDF缓存管理器（避免重复打开文件）
    # 注意：不要关闭数据集，由缓存管理器管理
    ds_from_cache = False
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
            ds_from_cache = True
        except Exception as e:
            # 如果缓存失败，回退到直接打开
            engines_to_try = ['netcdf4', 'h5netcdf', None]
            ds = None
            for engine in engines_to_try:
                try:
                    if engine:
                        ds = xr.open_dataset(filepath, engine=engine)
                    else:
                        ds = xr.open_dataset(filepath)
                    break
                except:
                    continue
            if ds is None:
                raise RuntimeError(f"无法打开FWI NetCDF文件 {filepath}: {e}")
    else:
        # 回退到直接打开（如果缓存不可用）
        engines_to_try = ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        for engine in engines_to_try:
            try:
                if engine:
                    ds = xr.open_dataset(filepath, engine=engine)
                else:
                    ds = xr.open_dataset(filepath)
                break
            except Exception as e:
                last_error = e
                continue
        if ds is None:
            error_msg = f"无法打开FWI NetCDF文件 {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # 确定变量名（可能需要根据实际文件调整）
    # 检查数据集是否有效（如果来自缓存，应该已经验证过了）
    # 注意：在多线程环境下，数据集可能在验证后立即失效，因此需要在使用时再次验证
    try:
        # 先尝试访问dims（这会触发延迟加载）
        _ = ds.dims
        # 然后访问data_vars和coords
        # 注意：在多线程环境下，这些操作可能失败，因为数据集可能被另一个线程关闭
        data_vars_keys = list(ds.data_vars.keys())
        coords_keys = list(ds.coords.keys())
        dims_keys = list(ds.dims.keys())
        
        # 如果所有列表都为空，数据集可能无效
        if len(data_vars_keys) == 0 and len(coords_keys) == 0 and len(dims_keys) == 0:
            raise ValueError("数据集的所有属性都为空，可能已失效")
            
    except (AttributeError, RuntimeError, OSError, ValueError) as e:
        # 数据集可能已经关闭或无效
        time_steps = (time_window_end - time_window_start).days + 1
        print(f"    ⚠️  FWI文件 {filepath} 数据集访问失败: {e}")
        print(f"       数据集可能已关闭或损坏，尝试重新打开...")
        
        # 如果数据集来自缓存，尝试重新打开
        if ds_from_cache and HAS_NETCDF_CACHE:
            try:
                from .netcdf_cache import clear_netcdf_cache
                # 清除缓存并重新打开
                clear_netcdf_cache()
                ds = xr.open_dataset(filepath, engine='netcdf4')
                ds_from_cache = False
                # 重新检查
                data_vars_keys = list(ds.data_vars.keys())
                coords_keys = list(ds.coords.keys())
                dims_keys = list(ds.dims.keys())
                print(f"       ✅ 重新打开成功，找到 {len(data_vars_keys)} 个数据变量: {data_vars_keys}")
            except Exception as e2:
                print(f"       ❌ 重新打开失败: {e2}")
                print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        else:
            # 即使不是来自缓存，也尝试重新打开
            try:
                print(f"       尝试重新打开文件...")
                ds = xr.open_dataset(filepath, engine='netcdf4')
                ds_from_cache = False
                # 重新检查
                data_vars_keys = list(ds.data_vars.keys())
                coords_keys = list(ds.coords.keys())
                dims_keys = list(ds.dims.keys())
                print(f"       ✅ 重新打开成功，找到 {len(data_vars_keys)} 个数据变量: {data_vars_keys}")
            except Exception as e2:
                print(f"       ❌ 重新打开失败: {e2}")
                print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
    
    # 再次检查数据变量列表（可能在验证后数据集失效了）
    if len(data_vars_keys) == 0:
        # 数据集可能在验证后失效了，尝试最后一次重新打开
        time_steps = (time_window_end - time_window_start).days + 1
        print(f"    ⚠️  FWI文件 {filepath} 数据集在验证后失效（data_vars为空）")
        print(f"       尝试最后一次重新打开...")
        try:
            if ds_from_cache and HAS_NETCDF_CACHE:
                from .netcdf_cache import clear_netcdf_cache
                clear_netcdf_cache()
            ds = xr.open_dataset(filepath, engine='netcdf4')
            ds_from_cache = False
            data_vars_keys = list(ds.data_vars.keys())
            coords_keys = list(ds.coords.keys())
            dims_keys = list(ds.dims.keys())
            if len(data_vars_keys) > 0:
                print(f"       ✅ 最后一次重新打开成功，找到 {len(data_vars_keys)} 个数据变量: {data_vars_keys}")
            else:
                print(f"       ❌ 重新打开后仍然没有数据变量")
                print(f"       可用坐标: {coords_keys}")
                print(f"       可用维度: {dims_keys}")
                print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        except Exception as e3:
            print(f"       ❌ 最后一次重新打开失败: {e3}")
            print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
    
    if 'fwi' in ds.data_vars:
        var_name = 'fwi'
    elif 'FWI' in ds.data_vars:
        var_name = 'FWI'
    else:
        # 尝试第一个数据变量
        if len(data_vars_keys) == 0:
            # 文件没有数据变量，可能是文件损坏或格式不正确
            # 返回零填充数组而不是抛出异常，允许流水线继续
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  FWI文件 {filepath} 没有FWI数据变量（可能是文件损坏或格式不正确）")
            print(f"       可用坐标: {coords_keys}")
            print(f"       可用维度: {dims_keys}")
            print(f"       数据变量列表: {data_vars_keys}")
            print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
            # 注意：不要关闭数据集，因为可能来自缓存
            # 返回零填充数组
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars_keys[0]
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 使用直接索引方法（参考merge_weather_indices.ipynb）
    # 创建目标网格的经纬度坐标
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    # 获取原始数据的坐标
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 检查坐标是否为空
    if len(longitude) == 0:
        raise ValueError(f"FWI文件 {filepath} 的经度坐标为空（len={len(longitude)}）。"
                        f"坐标名称: {lon_name}, 数据集维度: {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"FWI文件 {filepath} 的纬度坐标为空（len={len(latitude)}）。"
                        f"坐标名称: {lat_name}, 数据集维度: {dict(ds.dims)}")
    
    # 处理经度范围（0-360 vs -180-180）
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"FWI文件 {filepath} 无法获取经度范围: {e}。"
                        f"经度坐标形状: {longitude.shape}, 类型: {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 经度范围是0-360，但查询范围包含负值，需要转换
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 计算像素大小和原点（参考notebook的方法）
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.25
        if len(longitude) == 0:
            raise ValueError(f"FWI文件 {filepath} 的经度坐标为空，无法计算像素大小")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.25
        if len(latitude) == 0:
            raise ValueError(f"FWI文件 {filepath} 的纬度坐标为空，无法计算像素大小")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # 创建Affine变换（用于将经纬度转换为行列索引）
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 时间步数
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 获取时间坐标
    time_coords = ds.coords[time_name].values
    
    try:
        # 为每个时间步提取数据
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 使用辅助函数获取时间索引
            time_idx = get_time_index(time_coords, current_date, filepath, "max_temp")
            
            # 为网格的每个点提取数据
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 将经纬度转换为行列索引
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 确保索引在有效范围内
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # 使用xarray的高级索引直接提取值
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                result[t, i, j] = float(value)
                    except Exception as e:
                        # 如果索引失败，保持为0
                        pass
    finally:
        # 确保文件总是被关闭（即使在异常情况下）
        # 注意：如果数据集来自缓存，不要关闭它，由缓存管理器管理
        if ds is not None and not ds_from_cache:
            try:
                ds.close()
            except:
                pass
    
    return result


# ============================================================================
# 通道 1: VPD (蒸汽压差) - NetCDF数据
# ============================================================================

def extract_vpd_feature(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从NetCDF文件提取VPD特征（使用直接索引方法，参考merge_weather_indices.ipynb）
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 尝试从缓存获取
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'VPD', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'VPD', f'vapor_pressure_deficit_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VPD文件不存在: {filepath}")
    
    # 使用NetCDF缓存管理器（避免重复打开文件）
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            # 如果缓存失败，回退到直接打开
            engines_to_try = ['netcdf4', 'h5netcdf', None]
            ds = None
            for engine in engines_to_try:
                try:
                    if engine:
                        ds = xr.open_dataset(filepath, engine=engine)
                    else:
                        ds = xr.open_dataset(filepath)
                    break
                except:
                    continue
            if ds is None:
                raise RuntimeError(f"无法打开VPD NetCDF文件 {filepath}: {e}")
    else:
        # 回退到直接打开
        engines_to_try = ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        for engine in engines_to_try:
            try:
                if engine:
                    ds = xr.open_dataset(filepath, engine=engine)
                else:
                    ds = xr.open_dataset(filepath)
                break
            except Exception as e:
                last_error = e
                continue
        if ds is None:
            error_msg = f"无法打开VPD NetCDF文件 {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # 确定变量名
    if 'vpd' in ds.data_vars:
        var_name = 'vpd'
    elif 'VPD' in ds.data_vars:
        var_name = 'VPD'
    else:
        data_vars_list = list(ds.data_vars.keys())
        if len(data_vars_list) == 0:
            # 文件没有数据变量，可能是文件损坏或格式不正确
            # 返回零填充数组而不是抛出异常，允许流水线继续
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  VPD文件 {filepath} 没有数据变量（可能是文件损坏或格式不正确）")
            print(f"       可用坐标: {list(ds.coords.keys())}")
            print(f"       可用维度: {list(ds.dims.keys())}")
            print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
            # 关闭数据集
            try:
                ds.close()
            except:
                pass
            # 返回零填充数组
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars_list[0]
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 使用直接索引方法（参考merge_weather_indices.ipynb）
    # 创建目标网格的经纬度坐标
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    # 获取原始数据的坐标
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 检查坐标是否为空
    if len(longitude) == 0:
        raise ValueError(f"VPD文件 {filepath} 的经度坐标为空（len={len(longitude)}）。"
                        f"坐标名称: {lon_name}, 数据集维度: {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"VPD文件 {filepath} 的纬度坐标为空（len={len(latitude)}）。"
                        f"坐标名称: {lat_name}, 数据集维度: {dict(ds.dims)}")
    
    # 处理经度范围（0-360 vs -180-180）
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"VPD文件 {filepath} 无法获取经度范围: {e}。"
                        f"经度坐标形状: {longitude.shape}, 类型: {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 经度范围是0-360，但查询范围包含负值，需要转换
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 计算像素大小和原点（参考notebook的方法）
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.25
        if len(longitude) == 0:
            raise ValueError(f"VPD文件 {filepath} 的经度坐标为空，无法计算像素大小")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.25
        if len(latitude) == 0:
            raise ValueError(f"VPD文件 {filepath} 的纬度坐标为空，无法计算像素大小")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # 创建Affine变换（用于将经纬度转换为行列索引）
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 时间步数
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 获取时间坐标
    time_coords = ds.coords[time_name].values
    
    try:
        # 为每个时间步提取数据
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 使用辅助函数获取时间索引
            time_idx = get_time_index(time_coords, current_date, filepath, "VPD")
            
            # 为网格的每个点提取数据
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 将经纬度转换为行列索引
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 确保索引在有效范围内
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # 使用xarray的高级索引直接提取值
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                result[t, i, j] = float(value)
                    except Exception as e:
                        # 如果索引失败，保持为0
                        pass
    finally:
        # 注意：如果使用netcdf_cache，不要关闭数据集（由缓存管理器管理）
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 存入缓存
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'VPD', year, result)
    
    return result


# ============================================================================
# 通道 6: max_temp (ERA5 2米最大温度) - NetCDF数据
# ============================================================================

def extract_era5_temp_feature(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从NetCDF文件提取ERA5 2米最大温度特征（使用直接索引方法，参考extract_fwi_feature）
    
    Args:
        spatial_bounds: 空间边界字典
        time_window_start: 时间窗口开始日期
        time_window_end: 时间窗口结束日期
        grid_size: 目标网格大小
        year: 年份
        data_dir: 数据目录路径
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 尝试从缓存获取
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'max_temp', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'ERA5', f'era5_land_temp_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5温度文件不存在: {filepath}")
    
    # 使用NetCDF缓存管理器（避免重复打开文件）
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            # 如果缓存失败，回退到直接打开
            engines_to_try = ['netcdf4', 'h5netcdf', None]
            ds = None
            for engine in engines_to_try:
                try:
                    if engine:
                        ds = xr.open_dataset(filepath, engine=engine)
                    else:
                        ds = xr.open_dataset(filepath)
                    break
                except:
                    continue
            if ds is None:
                raise RuntimeError(f"无法打开ERA5温度 NetCDF文件 {filepath}: {e}")
    else:
        # 回退到直接打开
        engines_to_try = ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        for engine in engines_to_try:
            try:
                if engine:
                    ds = xr.open_dataset(filepath, engine=engine)
                else:
                    ds = xr.open_dataset(filepath)
                break
            except Exception as e:
                last_error = e
                continue
        if ds is None:
            error_msg = f"无法打开ERA5温度 NetCDF文件 {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # 确定变量名（ERA5温度数据使用t2m_max）
    if 't2m_max' in ds.data_vars:
        var_name = 't2m_max'
    elif 'temp' in ds.data_vars:
        var_name = 'temp'
    elif 'temperature' in ds.data_vars:
        var_name = 'temperature'
    else:
        # 尝试第一个数据变量（排除spatial_ref）
        data_vars = [v for v in ds.data_vars.keys() if v != 'spatial_ref']
        if len(data_vars) == 0:
            # 文件没有数据变量，可能是文件损坏或格式不正确
            # 返回零填充数组而不是抛出异常，允许流水线继续
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  max_temp文件 {filepath} 没有数据变量（可能是文件损坏或格式不正确）")
            print(f"       可用坐标: {list(ds.coords.keys())}")
            print(f"       可用维度: {list(ds.dims.keys())}")
            print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
            # 关闭数据集
            try:
                ds.close()
            except:
                pass
            # 返回零填充数组
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars[0]
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 使用直接索引方法（参考extract_fwi_feature）
    # 创建目标网格的经纬度坐标
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    # 获取原始数据的坐标
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 检查坐标是否为空
    if len(longitude) == 0:
        raise ValueError(f"max_temp文件 {filepath} 的经度坐标为空（len={len(longitude)}）。"
                        f"坐标名称: {lon_name}, 数据集维度: {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"max_temp文件 {filepath} 的纬度坐标为空（len={len(latitude)}）。"
                        f"坐标名称: {lat_name}, 数据集维度: {dict(ds.dims)}")
    
    # 处理经度范围（0-360 vs -180-180）
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"max_temp文件 {filepath} 无法获取经度范围: {e}。"
                        f"经度坐标形状: {longitude.shape}, 类型: {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 经度范围是0-360，但查询范围包含负值，需要转换
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 计算像素大小和原点（参考notebook的方法）
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.1
        if len(longitude) == 0:
            raise ValueError(f"max_temp文件 {filepath} 的经度坐标为空，无法计算像素大小")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.1
        if len(latitude) == 0:
            raise ValueError(f"max_temp文件 {filepath} 的纬度坐标为空，无法计算像素大小")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # 创建Affine变换（用于将经纬度转换为行列索引）
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 时间步数
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 获取时间坐标
    time_coords = ds.coords[time_name].values
    
    try:
        # 为每个时间步提取数据
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 使用辅助函数获取时间索引
            time_idx = get_time_index(time_coords, current_date, filepath, "max_temp")
            
            # 为网格的每个点提取数据
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 将经纬度转换为行列索引
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 确保索引在有效范围内
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # 使用xarray的高级索引直接提取值
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                # ERA5温度单位通常是开尔文，转换为摄氏度
                                # 如果值大于200，可能是开尔文，需要减去273.15
                                if value > 200:
                                    value = value - 273.15
                                result[t, i, j] = float(value)
                    except Exception as e:
                        # 如果索引失败，保持为0
                        pass
    finally:
        # 注意：如果使用netcdf_cache，不要关闭数据集（由缓存管理器管理）
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 存入缓存
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'max_temp', year, result)
    
    return result


# ============================================================================
# 通道 7: max_wind (ERA5 10米最大风速) - NetCDF数据
# ============================================================================

def extract_era5_wind_feature(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从NetCDF文件提取ERA5 10米最大风速特征（优化版本：使用批量切片和缓存）
    
    Args:
        spatial_bounds: 空间边界字典
        time_window_start: 时间窗口开始日期
        time_window_end: 时间窗口结束日期
        grid_size: 目标网格大小
        year: 年份
        data_dir: 数据目录路径
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size] 风速（m/s）
    """
    # 尝试从缓存获取
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'max_wind', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'ERA5', f'era5_land_wind_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5风速文件不存在: {filepath}")
    
    # 使用NetCDF缓存管理器（避免重复打开文件）
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            # 如果缓存失败，回退到直接打开
            engines_to_try = ['netcdf4', 'h5netcdf', None]
            ds = None
            for engine in engines_to_try:
                try:
                    if engine:
                        ds = xr.open_dataset(filepath, engine=engine)
                    else:
                        ds = xr.open_dataset(filepath)
                    break
                except:
                    continue
            if ds is None:
                raise RuntimeError(f"无法打开ERA5风速 NetCDF文件 {filepath}: {e}")
    else:
        # 回退到直接打开
        engines_to_try = ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        for engine in engines_to_try:
            try:
                if engine:
                    ds = xr.open_dataset(filepath, engine=engine)
                else:
                    ds = xr.open_dataset(filepath)
                break
            except Exception as e:
                last_error = e
                continue
        if ds is None:
            error_msg = f"无法打开ERA5风速 NetCDF文件 {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # 确定变量名（ERA5风速数据使用wind_speed_max）
    if 'wind_speed_max' in ds.data_vars:
        var_name = 'wind_speed_max'
    elif 'wind' in ds.data_vars:
        var_name = 'wind'
    elif 'wind_speed' in ds.data_vars:
        var_name = 'wind_speed'
    else:
        # 尝试第一个数据变量（排除spatial_ref）
        data_vars = [v for v in ds.data_vars.keys() if v != 'spatial_ref']
        if len(data_vars) == 0:
            # 文件没有数据变量，可能是文件损坏或格式不正确
            # 返回零填充数组而不是抛出异常，允许流水线继续
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  max_wind文件 {filepath} 没有数据变量（可能是文件损坏或格式不正确）")
            print(f"       可用坐标: {list(ds.coords.keys())}")
            print(f"       可用维度: {list(ds.dims.keys())}")
            print(f"       将返回零填充数组（{time_steps} 时间步 × {grid_size}×{grid_size} 网格）")
            # 关闭数据集
            try:
                ds.close()
            except:
                pass
            # 返回零填充数组
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars[0]
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 使用直接索引方法（参考extract_era5_temp_feature）
    # 创建目标网格的经纬度坐标
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    # 获取原始数据的坐标
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 检查坐标是否为空
    if len(longitude) == 0:
        raise ValueError(f"max_wind文件 {filepath} 的经度坐标为空（len={len(longitude)}）。"
                        f"坐标名称: {lon_name}, 数据集维度: {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"max_wind文件 {filepath} 的纬度坐标为空（len={len(latitude)}）。"
                        f"坐标名称: {lat_name}, 数据集维度: {dict(ds.dims)}")
    
    # 处理经度范围（0-360 vs -180-180）
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"max_wind文件 {filepath} 无法获取经度范围: {e}。"
                        f"经度坐标形状: {longitude.shape}, 类型: {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 经度范围是0-360，但查询范围包含负值，需要转换
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 计算像素大小和原点
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.1
        if len(longitude) == 0:
            raise ValueError(f"max_wind文件 {filepath} 的经度坐标为空，无法计算像素大小")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.1
        if len(latitude) == 0:
            raise ValueError(f"max_wind文件 {filepath} 的纬度坐标为空，无法计算像素大小")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # 创建Affine变换（用于将经纬度转换为行列索引）
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 时间步数
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 获取时间坐标
    time_coords = ds.coords[time_name].values
    
    try:
        # 优化：批量计算所有目标点的行列索引
        rows = []
        cols = []
        valid_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        for i, lat in enumerate(lat_target):
            for j, lon in enumerate(lon_target):
                try:
                    row, col = transformer.rowcol(lon, lat)
                    row = int(row)
                    col = int(col)
                    if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                        rows.append(row)
                        cols.append(col)
                        valid_mask[i, j] = True
                    else:
                        rows.append(-1)
                        cols.append(-1)
                except:
                    rows.append(-1)
                    cols.append(-1)
        
        rows = np.array(rows)
        cols = np.array(cols)
        
        # 为每个时间步批量提取数据
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 使用辅助函数获取时间索引
            time_idx = get_time_index(time_coords, current_date, filepath, "max_wind")
            
            # 批量提取：只提取有效索引的点
            valid_indices = (rows >= 0) & (cols >= 0)
            if np.any(valid_indices):
                valid_rows = rows[valid_indices]
                valid_cols = cols[valid_indices]
                
                # 使用xarray的批量索引（比逐点提取快得多）
                try:
                    # 创建索引数组
                    row_indices = xr.DataArray(valid_rows, dims='points')
                    col_indices = xr.DataArray(valid_cols, dims='points')
                    
                    # 批量提取数据
                    values = ds[var_name].isel(**{
                        time_name: time_idx,
                        lat_name: row_indices,
                        lon_name: col_indices
                    }).values
                    
                    # 将结果填充到result数组中
                    valid_positions = np.where(valid_mask.flatten())[0]
                    for idx, pos in enumerate(valid_positions):
                        if idx < len(values):
                            value = values[idx]
                            if not np.isnan(value):
                                i_pos = pos // grid_size
                                j_pos = pos % grid_size
                                result[t, i_pos, j_pos] = float(value)
                except Exception as e:
                    # 如果批量提取失败，回退到逐点提取
                    for idx, (i, j) in enumerate(np.ndindex(grid_size, grid_size)):
                        if valid_mask[i, j]:
                            row = rows[idx]
                            col = cols[idx]
                            try:
                                value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                                if not np.isnan(value):
                                    result[t, i, j] = float(value)
                            except:
                                pass
    finally:
        # 注意：如果使用netcdf_cache，不要关闭数据集（由缓存管理器管理）
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 存入缓存
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'max_wind', year, result)
    
    return result


# ============================================================================
# 通道 2: NDVI (归一化植被指数) - 支持GIMMS-3G+本地数据
# ============================================================================

def extract_ndvi_feature_gimms(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从GIMMS-3G+ NDVI数据集提取NDVI特征
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    GIMMS-3G+特点：
    - 空间分辨率：0.0833度（约8.3km）
    - 时间分辨率：每半月一次（每月2个值）
    - 时间范围：1982-2022
    - 格式：NetCDF
    
    数据文件路径：
    - 单文件：data_dir/NDVI/GIMMS3G+/gimms3g_ndvi.nc
    - 或按年：data_dir/NDVI/GIMMS3G+/gimms3g_ndvi_{year}.nc
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # GIMMS-3G+文件命名格式：ndvi3g_geo_v1_1_2002_0106.nc4 或 ndvi3g_geo_v1_2_2015_0106.nc4
    # 每个年份有2个文件：0106（1-6月）和0712（7-12月），需要合并
    gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
    
    # 查找该年份的两个文件
    file_patterns = [
        f'ndvi3g_geo_v1_1_{year}_0106.nc4',  # v1.1格式（2002-2014）
        f'ndvi3g_geo_v1_1_{year}_0712.nc4',
        f'ndvi3g_geo_v1_2_{year}_0106.nc4',  # v1.2格式（2015-2020）
        f'ndvi3g_geo_v1_2_{year}_0712.nc4',
    ]
    
    # 也尝试旧的命名格式（向后兼容）
    old_patterns = [
        os.path.join(gimms_dir, f'gimms3g_ndvi_{year}.nc'),
        os.path.join(gimms_dir, 'gimms3g_ndvi.nc'),
        os.path.join(gimms_dir, f'ndvi_{year}.nc'),
        os.path.join(gimms_dir, 'ndvi.nc'),
    ]
    
    # 查找文件
    found_files = []
    for pattern in file_patterns:
        filepath = os.path.join(gimms_dir, pattern)
        if os.path.exists(filepath):
            found_files.append(filepath)
    
    # 如果没找到新格式，尝试旧格式
    if not found_files:
        for path in old_patterns:
            if os.path.exists(path):
                found_files = [path]
                break
    
    if not found_files:
        raise FileNotFoundError(
            f"GIMMS-3G+ NDVI文件不存在（年份: {year}）。\n"
            f"期望的文件格式：ndvi3g_geo_v1_X_{year}_0106.nc4 和 ndvi3g_geo_v1_X_{year}_0712.nc4\n"
            f"目录：{gimms_dir}"
        )
    
    # 如果找到多个文件（0106和0712），需要合并
    if len(found_files) == 2:
        # 合并两个文件
        filepath = found_files  # 保存为列表，后续处理
    else:
        # 只有一个文件
        filepath = found_files[0]
    
    # 加载NetCDF（如果是列表，需要合并）
    if isinstance(filepath, list):
        # 合并同一年的两个文件（0106和0712）
        datasets = []
        for fp in sorted(filepath):
            try:
                ds_temp = xr.open_dataset(fp, engine='netcdf4')
                datasets.append(ds_temp)
            except:
                try:
                    ds_temp = xr.open_dataset(fp, engine='h5netcdf')
                    datasets.append(ds_temp)
                except:
                    ds_temp = xr.open_dataset(fp)
                    datasets.append(ds_temp)
        
        # 按时间维度合并
        ds = xr.concat(datasets, dim='time')
        # 按时间排序
        if 'time' in ds.coords:
            ds = ds.sortby('time')
    else:
        # 单个文件 - 使用NetCDF缓存管理器（如果可用）
        if HAS_NETCDF_CACHE:
            try:
                ds = get_netcdf_dataset(filepath)
            except Exception:
                # 如果缓存失败，回退到直接打开
                try:
                    ds = xr.open_dataset(filepath, engine='netcdf4')
                except:
                    try:
                        ds = xr.open_dataset(filepath, engine='h5netcdf')
                    except:
                        ds = xr.open_dataset(filepath)
        else:
            # 回退到直接打开
            try:
                ds = xr.open_dataset(filepath, engine='netcdf4')
            except:
                try:
                    ds = xr.open_dataset(filepath, engine='h5netcdf')
                except:
                    ds = xr.open_dataset(filepath)
    
    # 确定变量名（GIMMS-3G+可能使用的变量名）
    possible_var_names = ['NDVI', 'ndvi', 'NDVI_max', 'ndvi_max', 'gimms_ndvi']
    var_name = None
    for vn in possible_var_names:
        if vn in ds.data_vars:
            var_name = vn
            break
    
    if var_name is None:
        # 如果找不到，使用第一个数据变量
        if len(ds.data_vars) > 0:
            var_name = list(ds.data_vars.keys())[0]
        else:
            raise ValueError(f"无法找到NDVI变量。可用变量：{list(ds.data_vars.keys())}")
    
    # 获取fill_value（nodata值）和scale_factor如果存在
    fill_value = None
    scale_factor = None
    valid_range = None
    
    if var_name in ds.data_vars:
        var_attrs = ds[var_name].attrs
        # 检查常见的fill_value属性名
        for attr_name in ['_FillValue', 'fill_value', 'missing_value', 'nodata']:
            if attr_name in var_attrs:
                fill_value = float(var_attrs[attr_name])
                break
        
        # 检查scale属性（GIMMS-3G+通常使用scale: x 10000）
        if 'scale' in var_attrs:
            scale_str = str(var_attrs['scale']).lower()
            if 'x' in scale_str or '*' in scale_str:
                # 提取数字，例如 "x 10000" -> 10000
                import re
                numbers = re.findall(r'\d+', scale_str)
                if numbers:
                    scale_factor = float(numbers[0])
        
        # 检查valid_range属性
        if 'valid_range' in var_attrs:
            valid_range = var_attrs['valid_range']
            if hasattr(valid_range, '__len__') and len(valid_range) == 2:
                valid_range = [float(valid_range[0]), float(valid_range[1])]
        
        # 根据description检查nodata值（GIMMS-3G+通常-5000是nodata）
        if 'description' in var_attrs:
            desc = str(var_attrs['description']).lower()
            if 'ndvi = -5000' in desc or '-5000' in desc:
                if fill_value is None:
                    fill_value = -5000.0
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 获取坐标值
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    time_coords = ds.coords[time_name].values
    
    # 处理经度范围（GIMMS-3G+可能使用0-360或-180-180）
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 如果数据使用0-360经度，需要转换
    if longitude.max() > 180 and lon_min < 0:
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360
        lon_target_360 = np.linspace(lon_min_360, lon_max_360, grid_size)
    else:
        lon_target_360 = lon_target
    
    # 计算像素大小和原点
    pixel_size_x = abs(longitude[1] - longitude[0]) if len(longitude) > 1 else 0.0833
    pixel_size_y = abs(latitude[1] - latitude[0]) if len(latitude) > 1 else 0.0833
    
    origin_x = longitude[0] - (pixel_size_x / 2)
    origin_y = latitude[0] + (pixel_size_y / 2) if latitude[0] > latitude[-1] else latitude[-1] + (pixel_size_y / 2)
    
    # 创建Affine变换
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 时间步数
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 将时间坐标转换为datetime对象（如果还不是）
    time_datetimes = []
    for tc in time_coords:
        if isinstance(tc, (int, np.integer)) or np.issubdtype(type(tc), np.integer):
            # 可能是天数或年份天数
            if tc < 1000:  # 可能是天数（1-365）
                # 假设是年份天数，需要结合年份
                try:
                    time_datetimes.append(datetime(year, 1, 1) + timedelta(days=int(tc) - 1))
                except:
                    time_datetimes.append(pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=int(tc) - 1))
            else:
                # 可能是日期数字（YYYYMMDD格式）
                try:
                    time_str = str(tc)
                    if len(time_str) == 8:
                        time_datetimes.append(datetime(int(time_str[:4]), int(time_str[4:6]), int(time_str[6:8])))
                    else:
                        time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
                except:
                    time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
        else:
            # 已经是datetime或类似对象
            try:
                time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
            except:
                time_datetimes.append(tc)
    
    # 为每个时间步提取数据（需要处理半月合成到每日的插值）
    for t in range(time_steps):
        current_date = time_window_start + timedelta(days=t)
        
        # 找到最接近的时间索引（GIMMS-3G+是半月合成，需要找到最接近的）
        if len(time_datetimes) > 0:
            time_diffs = [abs((td - current_date).days) for td in time_datetimes]
            time_idx = np.argmin(time_diffs)
            
            # 如果时间差太大（超过30天），可能需要插值
            min_diff = time_diffs[time_idx]
            if min_diff > 15:  # 如果距离最近的半月数据超过15天，使用插值
                # 找到前后两个时间点进行插值
                if time_idx > 0 and time_idx < len(time_datetimes) - 1:
                    prev_idx = time_idx - 1
                    next_idx = time_idx + 1
                    prev_date = time_datetimes[prev_idx]
                    next_date = time_datetimes[next_idx]
                    
                    # 计算插值权重
                    total_diff = (next_date - prev_date).days
                    if total_diff > 0:
                        weight = (current_date - prev_date).days / total_diff
                        weight = max(0, min(1, weight))  # 限制在[0,1]
                    else:
                        weight = 0.5
                else:
                    # 只有一个时间点可用，直接使用
                    prev_idx = time_idx
                    next_idx = time_idx
                    weight = 0.5
            else:
                # 时间差足够小，直接使用最近的时间点
                prev_idx = time_idx
                next_idx = time_idx
                weight = 0.5
        else:
            prev_idx = 0
            next_idx = 0
            weight = 0.5
        
        # 为网格的每个点提取数据
        for i, lat in enumerate(np.linspace(spatial_bounds['lat_min'], spatial_bounds['lat_max'], grid_size)):
            for j, lon in enumerate(lon_target):
                try:
                    # 处理经度（如果需要转换到0-360）
                    lon_for_lookup = lon
                    if longitude.max() > 180 and lon < 0:
                        lon_for_lookup = lon % 360
                    
                    row, col = transformer.rowcol(lon_for_lookup, lat)
                    row = int(row)
                    col = int(col)
                    
                    if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                        # 如果使用插值，需要获取前后两个时间点的值
                        if prev_idx != next_idx and weight != 0.5:
                            value_prev = ds[var_name].isel(**{time_name: prev_idx, lat_name: row, lon_name: col}).values
                            value_next = ds[var_name].isel(**{time_name: next_idx, lat_name: row, lon_name: col}).values
                            
                            # 线性插值
                            if not np.isnan(value_prev) and not np.isnan(value_next):
                                value = value_prev * (1 - weight) + value_next * weight
                            elif not np.isnan(value_prev):
                                value = value_prev
                            elif not np.isnan(value_next):
                                value = value_next
                            else:
                                value = np.nan
                        else:
                            # 直接使用最近时间点的值
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                        
                        # GIMMS-3G+数据处理
                        # GIMMS-3G+ NDVI数据格式（根据实际文件属性）：
                        # - scale: x 10000（数据需要除以10000）
                        # - valid_range: [-0.3, 1.0]（对应原始值-3000到10000）
                        # - nodata值：-5000（根据description）
                        # - 有效值：-3000到10000（对应NDVI -0.3到1.0）
                        if not np.isnan(value):
                            value = float(value)
                            
                            # 检查是否为fill_value（nodata值）
                            if fill_value is not None and abs(value - fill_value) < 1e-6:
                                # 这是nodata值，跳过（保持result中的0值）
                                continue
                            
                            # 检查是否为常见的nodata值
                            # GIMMS-3G+的nodata值通常是-5000
                            if abs(value - (-5000)) < 1e-6:
                                continue
                            
                            # 应用scale_factor（如果存在）
                            if scale_factor is not None and scale_factor > 0:
                                value = value / scale_factor
                            
                            # 检查valid_range（如果存在）
                            # 注意：valid_range通常是[-0.3, 1.0]，-0.3对应冰覆盖区域，应该保留
                            if valid_range is not None:
                                if value < valid_range[0] or value > valid_range[1]:
                                    # 超出有效范围，视为nodata
                                    continue
                                # 如果值在valid_range内，继续处理（包括-0.3）
                            
                            # 如果没有scale_factor，尝试自动检测
                            # GIMMS-3G+通常使用10000作为scale factor
                            if scale_factor is None:
                                # 如果值在合理范围内（-3000到10000），除以10000
                                if -5000 < value < 15000:
                                    # 排除nodata值-5000
                                    if abs(value - (-5000)) > 1e-6:
                                        value = value / 10000.0
                                    else:
                                        continue
                                # 如果值已经在-1到1之间，直接使用
                                elif -1.0 <= value <= 1.0:
                                    pass  # 已经正确缩放
                                else:
                                    # 其他情况视为nodata
                                    continue
                            
                            # 最终检查：确保值在NDVI的理论范围内
                            # 注意：-0.3是冰覆盖区域的有效值，应该保留
                            # NDVI理论范围是-1到1，但实际有效值通常在-0.3到1.0之间
                            if -1.0 <= value <= 1.0:
                                result[t, i, j] = value
                            # 超出范围的值视为nodata，跳过
                except Exception as e:
                    pass
    
    # 注意：如果使用netcdf_cache（单个文件），不要关闭数据集（由缓存管理器管理）
    # 如果是合并的文件（列表），需要关闭
    if isinstance(filepath, list) or not HAS_NETCDF_CACHE:
        try:
            ds.close()
        except:
            pass
    
    # 存入缓存
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'NDVI', year, result)
    
    return result


# ============================================================================
# 通道 3: dNBR (差分归一化燃烧比) - 本地NetCDF数据（优先）或GEE API
# ============================================================================

def extract_dnbr_feature_local(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    fire_date: datetime,
    year: int,
    data_dir: str
) -> np.ndarray:
    """
    从本地NetCDF文件提取dNBR特征
    
    dNBR数据应该是预处理好的（火灾前后的NBR差值）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    filepath = os.path.join(data_dir, 'dNBR', f'dnbr_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"dNBR文件不存在: {filepath}")
    
    # 加载NetCDF
    try:
        ds = xr.open_dataset(filepath, engine='netcdf4')
    except:
        try:
            ds = xr.open_dataset(filepath, engine='h5netcdf')
        except:
            ds = xr.open_dataset(filepath)
    
    # 确定变量名
    if 'dNBR' in ds.data_vars:
        var_name = 'dNBR'
    elif 'dnbr' in ds.data_vars:
        var_name = 'dnbr'
    else:
        var_name = list(ds.data_vars.keys())[0]
    
    # 检测坐标名称
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 使用直接索引方法
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 处理经度范围
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    if longitude.max() > 180 and lon_min < 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 计算像素大小和原点
    pixel_size_x = longitude[1] - longitude[0] if len(longitude) > 1 else 0.01
    pixel_size_y = latitude[0] - latitude[1] if len(latitude) > 1 else 0.01
    
    origin_x = longitude[0] - (pixel_size_x / 2)
    origin_y = latitude[0] + (pixel_size_y / 2)
    
    # 创建Affine变换
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # dNBR是相对于火灾日期的，所有时间步使用相同的dNBR值
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 找到最接近火灾日期的时间索引
    time_coords = ds.coords[time_name].values
    if len(time_coords) > 0:
        if isinstance(time_coords[0], (int, np.integer)) or np.issubdtype(type(time_coords[0]), np.integer):
            fire_day = fire_date.timetuple().tm_yday
            time_idx = np.argmin(np.abs(time_coords - fire_day))
        else:
            time_diffs = [abs((pd.Timestamp(ts).to_pydatetime() - fire_date).days) for ts in time_coords]
            time_idx = np.argmin(time_diffs)
    else:
        time_idx = 0
    
    # 为网格的每个点提取数据
    for i, lat in enumerate(lat_target):
        for j, lon in enumerate(lon_target):
            try:
                row, col = transformer.rowcol(lon, lat)
                row = int(row)
                col = int(col)
                
                if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                    value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                    if not np.isnan(value):
                        # 所有时间步使用相同的dNBR值
                        result[:, i, j] = float(value)
            except Exception as e:
                pass
    
    ds.close()
    return result


def extract_dnbr_feature(
    spatial_bounds: dict,
    time_window_start: datetime,
    time_window_end: datetime,
    grid_size: int,
    fire_date: datetime,
    year: int,
    project: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从GEE API计算dNBR特征
    
    注意：此函数从Google Earth Engine API获取MODIS数据并计算dNBR。
    dNBR值只有在有火灾的区域才会非零，对于没有火灾的区域会返回0。
    
    dNBR = NBR_pre - NBR_post
    其中NBR = (B2 - B7) / (B2 + B7)
    
    计算时间窗口：
    - 火灾前：火灾发生前32-8天（24天窗口）
    - 火灾后：火灾发生后8-32天（24天窗口）
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
        注意：如果区域没有火灾或MODIS数据不足，可能返回全0数组
    """
    global _dnbr_cache
    
    # 检查缓存
    if use_cache:
        fire_date_str = fire_date.strftime('%Y-%m-%d')
        bounds_hash = _get_bounds_hash(spatial_bounds, grid_size)
        cache_key = (fire_date_str, bounds_hash)
        
        if cache_key in _dnbr_cache:
            # 从缓存返回（需要复制，避免修改缓存）
            cached_result = _dnbr_cache[cache_key]
            time_steps = (time_window_end - time_window_start).days + 1
            # 确保时间步数匹配
            if cached_result.shape[0] == time_steps:
                return cached_result.copy()
    
    if not HAS_GEE:
        raise ImportError("GEE模块不可用")
    
    # 初始化GEE
    try:
        import ee
        ee.Number(1).getInfo()
    except:
        # 使用silent模式，失败时返回零数组
        if not initialize_gee(project=project, silent=True):
            time_steps = (time_window_end - time_window_start).days + 1
            return np.zeros((time_steps, grid_size, grid_size))
    
    # 创建1km网格采样点
    lat_grid = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_grid = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 加载MODIS表面反射率数据集
    import ee
    collection = ee.ImageCollection('MODIS/061/MOD09GA')
    
    # dNBR是相对于火灾发生日期计算的，所有时间步使用相同的dNBR值
    # 火灾前：火灾发生前32-8天
    # 火灾后：火灾发生后8-32天
    
    fire_date_ee = ee.Date(fire_date.strftime('%Y-%m-%d'))
    pre_start = fire_date_ee.advance(-32, 'day')
    pre_end = fire_date_ee.advance(-8, 'day')
    post_start = fire_date_ee.advance(8, 'day')
    post_end = fire_date_ee.advance(32, 'day')
    
    # 获取火灾前后影像
    pre_fire = collection.filterDate(pre_start, pre_end).median()
    post_fire = collection.filterDate(post_start, post_end).median()
    
    # 计算NBR
    nbr_pre = pre_fire.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
    nbr_post = post_fire.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
    
    # 计算dNBR
    dnbr = nbr_pre.subtract(nbr_post)
    
    # 优化：一次性获取整个区域的数据，而不是逐个点采样
    # 创建覆盖整个区域的矩形
    region = ee.Geometry.Rectangle([
        spatial_bounds['lon_min'],
        spatial_bounds['lat_min'],
        spatial_bounds['lon_max'],
        spatial_bounds['lat_max']
    ])
    
    # 创建网格采样点（只采样网格中心点，而不是所有点）
    # 这样可以减少采样点数量，同时保持足够的空间分辨率
    sample_points = []
    point_indices = []
    
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            sample_points.append((lat, lon))
            point_indices.append((i, j))
    
    # 创建采样点集合
    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {'idx': idx, 'i': i, 'j': j})
        for idx, ((i, j), (lat, lon)) in enumerate(zip(point_indices, sample_points))
    ])
    
    # 采样（所有时间步使用相同的dNBR值）
    # 添加重试机制处理网络错误
    import time
    max_retries = 3
    retry_delay = 2  # 秒
    
    dnbr_values = {}
    for attempt in range(max_retries):
        try:
            # 优化：使用sampleRegions一次性采样所有点
            # 虽然还是采样625个点，但这是一次API调用，比625次调用快得多
            samples = dnbr.sampleRegions(
                collection=points_fc,
                scale=1000,  # 1km采样
                geometries=False
            )
            
            # 获取结果（这里可能发生SSL错误）
            sample_list = samples.getInfo()['features']
            
            # 成功获取数据，解析结果
            for feature in sample_list:
                idx = feature['properties']['idx']
                dnbr_value = feature['properties'].get('nd')
                if dnbr_value is not None:
                    dnbr_values[idx] = float(dnbr_value)
            
            # 如果成功获取数据，跳出重试循环
            break
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"    ⚠️  dNBR采样失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}...")
                print(f"       等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                # 最后一次尝试失败，使用降级方案：使用区域平均值
                print(f"    ⚠️  dNBR采样失败 (已重试 {max_retries} 次): {error_msg[:100]}...")
                print(f"       尝试使用区域平均值作为降级方案...")
                try:
                    stats = dnbr.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=1000,
                        maxPixels=1e9
                    )
                    mean_value = stats.getInfo().get('nd', 0)
                    if mean_value is not None:
                        # 使用平均值填充所有网格点
                        mean_dnbr = float(mean_value)
                        for idx in range(len(point_indices)):
                            dnbr_values[idx] = mean_dnbr
                        print(f"       ✅ 使用区域平均值: {mean_dnbr:.4f}")
                    else:
                        print(f"       ⚠️  无法获取区域平均值，使用默认值0")
                except Exception as e2:
                    print(f"       ⚠️  降级方案也失败: {str(e2)[:100]}...")
                    # 如果都失败，保持为0（已在初始化时设置）
    
    # 填充到所有时间步
    for t in range(time_steps):
        for idx, (i, j) in enumerate(point_indices):
            if idx in dnbr_values:
                result[t, i, j] = dnbr_values[idx]
    
    # 保存到缓存
    if use_cache:
        fire_date_str = fire_date.strftime('%Y-%m-%d')
        bounds_hash = _get_bounds_hash(spatial_bounds, grid_size)
        cache_key = (fire_date_str, bounds_hash)
        
        # 如果缓存已满，删除最旧的条目（FIFO）
        if len(_dnbr_cache) >= _dnbr_cache_max_size:
            # 删除第一个条目
            oldest_key = next(iter(_dnbr_cache))
            del _dnbr_cache[oldest_key]
        
        _dnbr_cache[cache_key] = result.copy()
    
    return result


# ============================================================================
# 通道 3: Population (人口数，每像素1km²，单位：人/km²) - WorldPop栅格数据
# ============================================================================

def extract_population_feature(
    spatial_bounds: dict,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    从WorldPop GeoTIFF文件提取Population特征
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    数据来源：WorldPop ppp (persons per pixel) 数据集
    分辨率：1km×1km
    单位：每像素人口数（在1km分辨率下，数值等于人口密度，单位：人/km²）
    
    Returns:
        np.ndarray: [grid_size, grid_size] 人口数（人/km²）
    """
    # 尝试从缓存获取（静态特征，不需要时间窗口）
    if use_cache:
        cached = get_cached_feature(spatial_bounds, None, None, grid_size, 'population', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'worldpop', f'ppp_{year}_1km_Aggregated.tif')
    
    # 如果文件不存在，尝试使用绝对路径
    if not os.path.exists(filepath):
        abs_data_dir = os.path.abspath(data_dir)
        abs_filepath = os.path.join(abs_data_dir, 'worldpop', f'ppp_{year}_1km_Aggregated.tif')
        if os.path.exists(abs_filepath):
            filepath = abs_filepath
        else:
            # 提供详细的错误信息
            error_msg = f"Population文件不存在: {filepath}"
            error_msg += f"\n       尝试的绝对路径: {abs_filepath} (存在: {os.path.exists(abs_filepath)})"
            error_msg += f"\n       当前工作目录: {os.getcwd()}"
            error_msg += f"\n       data_dir: {data_dir} (绝对路径: {abs_data_dir})"
            # 列出worldpop目录中的文件（如果目录存在）
            worldpop_dir = os.path.join(data_dir, 'worldpop')
            abs_worldpop_dir = os.path.abspath(worldpop_dir)
            if os.path.exists(abs_worldpop_dir):
                try:
                    files = [f for f in os.listdir(abs_worldpop_dir) if f.endswith('.tif')]
                    error_msg += f"\n       worldpop目录中的文件: {files[:5]}..." if len(files) > 5 else f"\n       worldpop目录中的文件: {files}"
                except:
                    pass
            raise FileNotFoundError(error_msg)
    
    # 使用rasterio读取
    with rasterio.open(filepath) as src:
        # 获取NoData值
        nodata_value = src.nodata
        
        # 计算目标变换矩阵
        from rasterio.transform import from_bounds
        
        transform = from_bounds(
            spatial_bounds['lon_min'],
            spatial_bounds['lat_min'],
            spatial_bounds['lon_max'],
            spatial_bounds['lat_max'],
            grid_size,
            grid_size
        )
        
        # 创建输出数组（使用float32避免整数溢出问题）
        output_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # 计算源数据和目标数据的像素面积（用于单位转换）
        # 源数据：1km分辨率，每个像素约1km²
        src_pixel_size_deg = abs(src.transform[0])  # 经度方向像素大小（度）
        src_pixel_area_km2 = (src_pixel_size_deg * 111.32) ** 2  # 约1km²
        
        # 目标数据：每个像素的面积
        dst_pixel_size_lon_deg = abs(transform[0])  # 经度方向像素大小（度）
        dst_pixel_size_lat_deg = abs(transform[4])  # 纬度方向像素大小（度）
        # 考虑纬度对经度距离的影响
        center_lat = (spatial_bounds['lat_min'] + spatial_bounds['lat_max']) / 2
        dst_pixel_area_km2 = (dst_pixel_size_lon_deg * 111.32 * np.cos(np.radians(center_lat))) * (dst_pixel_size_lat_deg * 111.32)
        
        # 重采样：WorldPop数据是"persons per pixel"，在1km分辨率下每个像素代表1km²的人口数
        # 当重采样到更大的像素时，应该：
        # 1. 使用sum重采样累加总人口数
        # 2. 然后除以目标像素面积得到人口密度（人/km²）
        temp_output = np.zeros((grid_size, grid_size), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=temp_output,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.sum,  # 使用sum重采样，累加总人口数
            src_nodata=nodata_value,  # 指定源NoData值
            dst_nodata=np.nan  # 目标NoData值使用NaN
        )
        
        # 将总人口数转换为人口密度（人/km²）
        # 注意：WorldPop数据是"persons per pixel"，在1km分辨率下每个像素代表1km²的人口数
        # 当我们重采样到目标网格时，rasterio的sum重采样会累加所有源像素的人口数
        # 
        # 关键理解：
        # - WorldPop原始数据：每个像素 = 1km²，值 = 人/km²
        # - 重采样后：每个目标像素可能包含多个源像素
        # - sum重采样会累加总人口数（不是密度）
        # - 要得到人口密度（人/km²），需要除以目标像素面积
        #
        # 但是，如果目标像素面积计算错误（比如太小），除以面积会导致值异常大
        # 检查：如果计算出的面积异常小（<0.01 km²），可能是计算错误
        if dst_pixel_area_km2 > 0:
            # 检查面积是否合理（目标像素面积应该在0.1到100 km²之间，取决于grid_size和空间范围）
            expected_min_area = 0.1  # 最小合理面积（0.1 km²）
            expected_max_area = 100.0  # 最大合理面积（100 km²，对应10km×10km的像素）
            
            if dst_pixel_area_km2 < expected_min_area or dst_pixel_area_km2 > expected_max_area:
                # 面积计算可能有问题，使用源像素面积作为参考
                print(f"    ⚠️  Population目标像素面积异常: {dst_pixel_area_km2:.6f} km²")
                print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                print(f"       grid_size: {grid_size}, 源像素面积: {src_pixel_area_km2:.6f} km²")
                # 如果目标像素面积异常小，直接使用temp_output（不除以面积）
                # 因为sum重采样已经给出了正确的人口数，如果目标像素和源像素大小相近
                if dst_pixel_area_km2 < expected_min_area:
                    print(f"       使用源像素面积作为参考，不除以异常小的目标面积")
                    output_data = temp_output / src_pixel_area_km2 if src_pixel_area_km2 > 0 else temp_output
                else:
                    # 面积异常大，仍然除以面积，但记录警告
                    output_data = temp_output / dst_pixel_area_km2
            else:
                # 面积合理，正常除以面积
                output_data = temp_output / dst_pixel_area_km2
        else:
            # 面积计算失败，使用源像素面积
            print(f"    ⚠️  Population目标像素面积计算失败，使用源像素面积")
            output_data = temp_output / src_pixel_area_km2 if src_pixel_area_km2 > 0 else temp_output
        
        # 处理NoData值：将NaN和异常值（负数或极大值）转换为0
        # WorldPop数据应该是非负的，负数通常是NoData值的错误表示
        # 注意：WorldPop的NoData值可能是-3.4028234663852886e+38（float32的最大负值）
        valid_mask = ~np.isnan(output_data)
        
        # 检查是否有异常值（负数或极大值）
        if valid_mask.any():
            # 计算合理的数据范围（WorldPop通常在0到几万之间）
            valid_data = output_data[valid_mask]
            # 如果数据中有负数或异常大的值，可能是NoData值的错误表示
            # WorldPop的NoData值通常是-3.4028234663852886e+38（float32的最大负值）
            if np.any(valid_data < 0) or np.any(valid_data < -1e10):
                # 将负数或异常大的负值视为NoData
                output_data[output_data < 0] = np.nan
                valid_mask = ~np.isnan(output_data)
        
        # 将NaN和无效值设为0
        output_data[~valid_mask] = 0.0
        
        # 处理异常大的值和负值（可能是数据错误或计算错误）
        # WorldPop ppp数据：每像素人口数（1km²），最大合理值约为100,000人/km²（非常密集的城市）
        max_reasonable_value = 100000.0
        
        # 检查异常值
        if np.any(output_data < 0):
            negative_count = np.sum(output_data < 0)
            print(f"    ⚠️  Population发现 {negative_count} 个负值，将设为0")
            output_data[output_data < 0] = 0.0
        
        if np.any(output_data > max_reasonable_value):
            large_count = np.sum(output_data > max_reasonable_value)
            max_value = np.max(output_data)
            print(f"    ⚠️  Population发现 {large_count} 个异常大的值（最大: {max_value:.2f}），将设为0")
            print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
            print(f"       目标像素面积: {dst_pixel_area_km2:.6f} km², 源像素面积: {src_pixel_area_km2:.6f} km²")
            # 将异常大的值设为0
            output_data[output_data > max_reasonable_value] = 0.0
        
        # 注意：保留浮点数精度，因为人口密度可能是小数
        # 但如果值太小（<1），可能是重采样问题，需要检查
    
    # 存入缓存（静态特征，不需要时间窗口）
    if use_cache:
        cache_feature(spatial_bounds, None, None, grid_size, 'population', year, output_data)
    
    return output_data


# ============================================================================
# 通道 5: GDP (经济指标) - 优先 gdp_per_capita.csv，按火灾位置→国家→年份查找
# ============================================================================

# GDP警告计数器（用于减少警告输出频率）
_gdp_warning_count = 0
_gdp_warning_lock = threading.Lock()
# Population 全为 0 的警告计数器（属正常：无人区或 WorldPop 无覆盖）
_pop_zero_warning_count = 0
_pop_zero_warning_lock = threading.Lock()

# gdp_per_capita.csv 内存缓存（按火灾位置→国家→年份查 GDP）
_gdp_per_capita_df = None
_gdp_per_capita_lock = threading.Lock()

def _load_gdp_per_capita_df(data_dir: str):
    """懒加载 gdp_per_capita.csv，列含 Country Code 及 2002,2003,... 年份列"""
    global _gdp_per_capita_df
    with _gdp_per_capita_lock:
        if _gdp_per_capita_df is not None:
            return _gdp_per_capita_df
        path = os.path.join(data_dir, 'gdp_per_capita.csv')
        if not os.path.exists(path):
            return None
        try:
            _gdp_per_capita_df = pd.read_csv(path, engine='python')
        except Exception:
            _gdp_per_capita_df = pd.read_csv(path, engine='c')
        return _gdp_per_capita_df


def _get_gdp_from_per_capita_csv(data_dir: str, iso3_clean: str, year: int) -> Optional[float]:
    """
    在 gdp_per_capita.csv 中按 (Country Code, 年份) 查找 GDP per capita。
    列格式：Country Code，以及 2002, 2003, ... 2023 等年份列。
    """
    df = _load_gdp_per_capita_df(data_dir)
    if df is None or 'Country Code' not in df.columns:
        return None
    year_str = str(year)
    if year_str not in df.columns:
        # 用最接近的年份列
        year_cols = [c for c in df.columns if c.strip().isdigit() and 1990 <= int(c) <= 2030]
        if not year_cols:
            return None
        year_cols_sorted = sorted(year_cols, key=lambda x: abs(int(x) - year))
        year_str = year_cols_sorted[0]
    row = df[df['Country Code'].astype(str).str.strip().str.upper() == iso3_clean.upper()]
    if len(row) == 0:
        return None
    val = row[year_str].iloc[0]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def reset_gdp_warning_count():
    """重置GDP警告计数器（用于测试或重新开始）"""
    global _gdp_warning_count
    with _gdp_warning_lock:
        _gdp_warning_count = 0


def _lat_lon_to_iso3_point_in_country(lat: float, lon: float) -> Optional[str]:
    """
    用「点所在国家」逻辑得到 iso3：reverse_geocoder（离线）返回 ISO2，
    再用 pycountry 转为 ISO3。起火点/样本点理应落在某国境内，此方法可显著减少 GDP=0。
    若未安装 reverse_geocoder 或 pycountry、或 Python<3.10 导致 importlib.metadata 报错，返回 None。
    """
    try:
        import reverse_geocoder as rg
    except (ImportError, AttributeError):
        # AttributeError: importlib.metadata 在 Python<3.10 无 packages_distributions
        return None
    try:
        res = rg.search((float(lat), float(lon)))
        if not res or not isinstance(res, (list, tuple)):
            return None
        rec = res[0] if isinstance(res, (list, tuple)) else res
        cc2 = getattr(rec, 'cc', None) or (rec.get('cc') if isinstance(rec, dict) else None)
        if not cc2 or len(str(cc2).strip()) != 2:
            return None
        cc2 = str(cc2).strip().upper()
    except (Exception, AttributeError):
        return None
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=cc2)
        if c is not None and getattr(c, 'alpha_3', None):
            return str(c.alpha_3).strip().upper()
    except ImportError:
        pass
    except Exception:
        pass
    # 无 pycountry 时用常见 ISO2->ISO3 映射（仅部分，保证主要国家）
    _iso2_to_iso3 = {
        'AU': 'AUS', 'US': 'USA', 'CN': 'CHN', 'BR': 'BRA', 'RU': 'RUS', 'IN': 'IND',
        'CA': 'CAN', 'ID': 'IDN', 'ZA': 'ZAF', 'MX': 'MEX', 'GB': 'GBR', 'FR': 'FRA',
        'DE': 'DEU', 'ES': 'ESP', 'IT': 'ITA', 'JP': 'JPN', 'KR': 'KOR', 'PT': 'PRT',
        'GR': 'GRC', 'PL': 'POL', 'TR': 'TUR', 'AR': 'ARG', 'CL': 'CHL', 'CO': 'COL',
        'PE': 'PER', 'NG': 'NGA', 'EG': 'EGY', 'MA': 'MAR', 'KE': 'KEN', 'TH': 'THA',
        'VN': 'VNM', 'MY': 'MYS', 'PH': 'PHL', 'PK': 'PAK', 'BD': 'BGD', 'IR': 'IRN',
        'IQ': 'IRQ', 'SA': 'SAU', 'IL': 'ISR', 'NZ': 'NZL', 'NL': 'NLD', 'SE': 'SWE',
        'NO': 'NOR', 'FI': 'FIN', 'AT': 'AUT', 'CH': 'CHE', 'BE': 'BEL', 'IE': 'IRL',
    }
    return _iso2_to_iso3.get(cc2)


def _lat_lon_to_iso3_from_covariate_df(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    max_dist_deg: float = 2.0
) -> Optional[str]:
    """
    用 (lat, lon) 在协变量表 DataFrame 的 (lat_mean, lon_mean) 上做最近邻，返回 2° 内的 iso3。
    用于 extract_gdp_feature 在无 iso3 时根据 patch 中心推断国家（如负样本）。
    """
    if df is None or len(df) == 0:
        return None
    need = ['lat_mean', 'lon_mean', 'iso3']
    if not all(c in df.columns for c in need):
        return None
    try:
        sub = df[need].dropna(subset=need)
        sub = sub[sub['iso3'].astype(str).str.strip() != '']
        if len(sub) == 0:
            return None
        pts = np.column_stack([sub['lat_mean'].values, sub['lon_mean'].values])
    except Exception:
        return None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dist, idx = tree.query([lat, lon], k=1, distance_upper_bound=max_dist_deg)
        if np.isinf(dist) or float(dist) > max_dist_deg or idx >= len(sub):
            return None
        iso3_val = sub.iloc[int(idx)]['iso3']
        if pd.isna(iso3_val) or str(iso3_val).strip() == '':
            return None
        return str(iso3_val).strip().upper()
    except ImportError:
        # 无 scipy 时退化为简单距离搜索
        d = np.sqrt((sub['lat_mean'].values - lat) ** 2 + (sub['lon_mean'].values - lon) ** 2)
        idx = np.argmin(d)
        if d[idx] > max_dist_deg:
            return None
        iso3_val = sub.iloc[idx]['iso3']
        if pd.isna(iso3_val) or str(iso3_val).strip() == '':
            return None
        return str(iso3_val).strip().upper()


def extract_gdp_feature(
    spatial_bounds: dict,
    grid_size: int,
    year: int,
    country_code: Optional[str] = None,
    iso3: Optional[str] = None,
    data_dir: str = 'dataset',
    use_cache: bool = True
) -> np.ndarray:
    """
    从 gdp_per_capita.csv 按「火灾位置→国家→年份」提取GDP特征。
    优先用 patch 中心 (lat, lon) 在协变量表中得到国家 iso3，再在 gdp_per_capita.csv 中
    按 (Country Code, year) 查找；若无该文件或未匹配则回退到 filtered_cleaned_cp_covariate.csv。
    
    Returns:
        np.ndarray: [grid_size, grid_size] (所有网格使用相同的GDP值)
    """
    global _gdp_warning_count
    
    center_lat = (spatial_bounds['lat_min'] + spatial_bounds['lat_max']) / 2
    center_lon = (spatial_bounds['lon_min'] + spatial_bounds['lon_max']) / 2

    # 缓存键用「位置+年份」以保证同一位置同年份复用
    if use_cache:
        extra_params = {'lat': center_lat, 'lon': center_lon}
        cached = get_cached_feature(spatial_bounds, None, None, grid_size, 'GDP', year, extra_params)
        if cached is not None:
            return cached

    csv_path = os.path.join(data_dir, 'filtered_cleaned_cp_covariate.csv')
    covariate_df = None
    if os.path.exists(csv_path):
        try:
            covariate_df = pd.read_csv(csv_path, engine='c')
        except Exception:
            covariate_df = pd.read_csv(csv_path, engine='python')

    # 1) 优先「点所在国家」（境内点必属某国，reverse_geocoder）；再协变量表、再传入 iso3/country
    iso3_clean = _lat_lon_to_iso3_point_in_country(center_lat, center_lon)
    if iso3_clean is None and covariate_df is not None and len(covariate_df) > 0:
        iso3_clean = _lat_lon_to_iso3_from_covariate_df(
            covariate_df, center_lat, center_lon, max_dist_deg=2.0
        )
    if iso3_clean is None and iso3 is not None and str(iso3).strip() != '':
        iso3_clean = str(iso3).strip().upper()
    if iso3_clean is None and country_code is not None and str(country_code).strip() != '':
        if covariate_df is not None and 'country' in covariate_df.columns and 'iso3' in covariate_df.columns:
            sub = covariate_df[
                covariate_df['country'].astype(str).str.strip() == str(country_code).strip()
            ]
            if len(sub) > 0:
                iso3_val = sub['iso3'].iloc[0]
                if pd.notna(iso3_val) and str(iso3_val).strip() != '':
                    iso3_clean = str(iso3_val).strip().upper()

    if iso3_clean is None:
        with _gdp_warning_lock:
            _gdp_warning_count += 1
            current_count = _gdp_warning_count
            if current_count == 1 or current_count % 100 == 0:
                print(f"    ⚠️  未从位置(lat,lon)或传入参数得到 iso3，GDP 使用默认值0（已出现 {current_count} 次）")
        return np.zeros((grid_size, grid_size))

    # 3) 优先在 gdp_per_capita.csv 中按 (Country Code, year) 查找
    gdp_value = _get_gdp_from_per_capita_csv(data_dir, iso3_clean, year)
    if gdp_value is not None:
        result = np.full((grid_size, grid_size), float(gdp_value))
        if use_cache:
            cache_feature(
                spatial_bounds, None, None, grid_size, 'GDP', year, result,
                extra_params={'lat': center_lat, 'lon': center_lon}
            )
        return result

    # 4) 回退：从 filtered_cleaned_cp_covariate.csv 按 (iso3, year) 查 gdp 列
    if covariate_df is None:
        with _gdp_warning_lock:
            _gdp_warning_count += 1
            current_count = _gdp_warning_count
            if current_count == 1 or current_count % 100 == 0:
                print(f"    ⚠️  未找到匹配的GDP数据（iso3='{iso3_clean}', year={year}），使用默认值0（已出现 {current_count} 次）")
        return np.zeros((grid_size, grid_size))

    if 'iso3' in covariate_df.columns and 'year' in covariate_df.columns and 'gdp' in covariate_df.columns:
        df_clean = covariate_df[covariate_df['iso3'].notna()].copy()
        df_clean['iso3_clean'] = df_clean['iso3'].astype(str).str.strip().str.upper()
        match_df = df_clean[(df_clean['iso3_clean'] == iso3_clean) & (df_clean['year'] == year)]
        if len(match_df) > 0:
            gdp_value = match_df['gdp'].iloc[0]
            if not pd.isna(gdp_value):
                result = np.full((grid_size, grid_size), float(gdp_value))
                if use_cache:
                    cache_feature(
                        spatial_bounds, None, None, grid_size, 'GDP', year, result,
                        extra_params={'lat': center_lat, 'lon': center_lon}
                    )
                return result

    with _gdp_warning_lock:
        _gdp_warning_count += 1
        current_count = _gdp_warning_count
        if current_count == 1 or current_count % 100 == 0:
            print(f"    ⚠️  未找到匹配的GDP数据（iso3='{iso3_clean}', year={year}），使用默认值0（已出现 {current_count} 次）")
    return np.zeros((grid_size, grid_size))


# ============================================================================
# 通道 6: Land Cover (土地覆盖) - 本地GeoTIFF文件（优先）或GEE API
# ============================================================================

def extract_landcover_feature_1km(
    spatial_bounds: dict,
    grid_size: int,
    year: int,
    project: str,
    data_dir: str = 'dataset',
    use_cache: bool = True
) -> np.ndarray:
    """
    提取Land Cover特征（优先使用本地文件，如果不存在则使用GEE API）
    
    数据源优先级：
    1. 本地GeoTIFF文件：dataset/LandCover/modis_mcd12q1_lc_type1_{year}.tif
    2. GEE API：MODIS/061/MCD12Q1（如果本地文件不存在）
    
    Args:
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        np.ndarray: [grid_size, grid_size]
    """
    # 尝试从缓存获取（静态特征，不需要时间窗口）
    if use_cache:
        cached = get_cached_feature(spatial_bounds, None, None, grid_size, 'land_cover', year)
        if cached is not None:
            return cached
    
    # 首先尝试从本地文件读取
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if os.path.exists(local_file):
        try:
            # 从本地GeoTIFF文件读取
            with rasterio.open(local_file) as src:
                # 计算目标变换矩阵
                from rasterio.transform import from_bounds
                
                transform = from_bounds(
                    spatial_bounds['lon_min'],
                    spatial_bounds['lat_min'],
                    spatial_bounds['lon_max'],
                    spatial_bounds['lat_max'],
                    grid_size,
                    grid_size
                )
                
                # 创建输出数组
                output_data = np.zeros((grid_size, grid_size), dtype=np.float32)
                
                # 重采样到目标网格
                reproject(
                    source=rasterio.band(src, 1),
                    destination=output_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,  # 土地覆盖使用最近邻重采样
                    src_nodata=src.nodata,
                    dst_nodata=255  # 255是未分类
                )
                
                # 处理NoData值
                if src.nodata is not None:
                    output_data[output_data == src.nodata] = 255
                
                # 将NaN值设为255（未分类）
                output_data[np.isnan(output_data)] = 255
                
                # 转换为整数
                result = output_data.astype(np.int32)
                
                # 检查结果是否合理
                # 注意：一个25km×25km的区域可能主要由单一土地覆盖类型组成，这是正常的
                # 只有当所有值都是0（数据缺失）时才需要警告
                unique_values = np.unique(result[result != 0])
                if len(unique_values) == 0:
                    # 所有值都是0或255（未分类），可能是数据缺失
                    print(f"    ⚠️  警告: Land Cover（本地文件）所有值都是0或未分类，可能数据缺失")
                    print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
                          f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                    print(f"       文件: {local_file}")
                # 注意：如果只有一个唯一值，这是正常的（区域主要由单一类型组成），不需要警告
                
                # 存入缓存（静态特征，不需要时间窗口）
                if use_cache:
                    cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
                
                return result
        except Exception as e:
            print(f"    ⚠️  读取本地Land Cover文件失败: {e}")
            print(f"       将回退到GEE API...")
    
    # 如果本地文件不存在或读取失败，使用GEE API
    if not HAS_GEE:
        print(f"    ⚠️  GEE模块不可用，且本地文件不存在: {local_file}")
        print(f"       返回零数组")
        result = np.zeros((grid_size, grid_size))
        # 即使返回零数组，也存入缓存（避免重复尝试）
        if use_cache:
            cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
        return result
    
    # 初始化GEE
    try:
        import ee
        ee.Number(1).getInfo()
    except:
        # 使用silent模式，失败时返回零数组
        if not initialize_gee(project=project, silent=True):
            return np.zeros((grid_size, grid_size))
    
    # 创建1km网格采样点
    lat_grid = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_grid = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    
    # 创建采样点集合
    sample_points = []
    point_indices = []
    
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            sample_points.append((lat, lon))
            point_indices.append((i, j))
    
    # 加载MODIS土地覆盖数据集
    import ee
    dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
    image = dataset.filter(
        ee.Filter.eq('system:time_start', 
                    ee.Date.fromYMD(year, 1, 1).millis())
    ).first()
    lc_band = image.select('LC_Type1')
    
    # 批量采样（使用1km scale，GEE自动聚合500m数据）
    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {'idx': idx})
        for idx, (lat, lon) in enumerate(sample_points)
    ])
    
    samples = lc_band.sampleRegions(
        collection=points_fc,
        scale=1000,  # 关键：使用1000m scale，GEE自动聚合500m数据到1km
        geometries=False
    )
    
    # 获取结果并填充到网格
    result = np.zeros((grid_size, grid_size), dtype=np.int32)
    sample_list = samples.getInfo()['features']
    
    # 检查采样结果数量
    if len(sample_list) != len(sample_points):
        print(f"    ⚠️  Land Cover采样点数量不匹配: 期望 {len(sample_points)}, 实际 {len(sample_list)}")
    
    # 统计采样结果
    valid_samples = 0
    invalid_samples = 0
    lc_value_counts = {}
    
    for feature in sample_list:
        idx = feature['properties'].get('idx')
        if idx is None:
            invalid_samples += 1
            continue
        
        lc_value = feature['properties'].get('LC_Type1')
        if lc_value is None:
            invalid_samples += 1
            continue
        
        # 检查索引是否有效
        if idx >= len(point_indices):
            print(f"    ⚠️  索引超出范围: idx={idx}, point_indices长度={len(point_indices)}")
            invalid_samples += 1
            continue
        
        i, j = point_indices[idx]
        
        # 检查i, j是否在有效范围内
        if 0 <= i < grid_size and 0 <= j < grid_size:
            result[i, j] = int(lc_value)
            valid_samples += 1
            lc_value_counts[int(lc_value)] = lc_value_counts.get(int(lc_value), 0) + 1
        else:
            print(f"    ⚠️  索引超出网格范围: i={i}, j={j}, grid_size={grid_size}")
            invalid_samples += 1
    
    # 检查结果是否合理
    unique_values = np.unique(result[result != 0])
    if len(unique_values) == 1:
        print(f"    ⚠️  警告: Land Cover只有一个唯一值 {unique_values[0]}，可能有问题")
        print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
              f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
        print(f"       有效采样数: {valid_samples}/{len(sample_points)}")
        print(f"       值分布: {lc_value_counts}")
    elif len(unique_values) == 0:
        print(f"    ⚠️  警告: Land Cover所有值都是0，可能采样失败")
        print(f"       空间边界: lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
              f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
        print(f"       有效采样数: {valid_samples}/{len(sample_points)}")
    
    # 存入缓存（静态特征，不需要时间窗口）
    if use_cache:
        cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
    
    return result


# ============================================================================
# 辅助函数：空间重采样
# ============================================================================

def resample_to_grid(
    data: np.ndarray,
    lat_orig: np.ndarray,
    lon_orig: np.ndarray,
    spatial_bounds: dict,
    grid_size: int,
    time_steps: int
) -> np.ndarray:
    """
    将数据重采样到目标网格
    
    Args:
        data: 原始数据 [time, lat, lon] 或 [lat, lon]
        lat_orig: 原始纬度坐标
        lon_orig: 原始经度坐标
        spatial_bounds: 目标空间边界
        grid_size: 目标网格大小
        time_steps: 时间步数
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    from scipy.interpolate import griddata
    
    # 创建目标网格
    lat_target = np.linspace(
        spatial_bounds['lat_min'],
        spatial_bounds['lat_max'],
        grid_size
    )
    lon_target = np.linspace(
        spatial_bounds['lon_min'],
        spatial_bounds['lon_max'],
        grid_size
    )
    lon_mesh, lat_mesh = np.meshgrid(lon_target, lat_target)
    
    # 创建原始坐标网格
    lon_orig_mesh, lat_orig_mesh = np.meshgrid(lon_orig, lat_orig)
    points_orig = np.column_stack([
        lat_orig_mesh.ravel(),
        lon_orig_mesh.ravel()
    ])
    
    # 创建目标点
    points_target = np.column_stack([
        lat_mesh.ravel(),
        lon_mesh.ravel()
    ])
    
    # 处理时间维度
    if data.ndim == 2:
        # 2D数据：扩展到3D
        data = data[np.newaxis, :, :]
        time_steps = 1
    
    result = np.zeros((time_steps, grid_size, grid_size))
    
    for t in range(min(time_steps, data.shape[0])):
        values_orig = data[t].ravel()
        
        # 检查有效数据点（非NaN和非零）
        valid_mask = ~np.isnan(values_orig)
        if len(values_orig) > 0:
            # 如果所有值都是NaN，使用fill_value
            if not valid_mask.any():
                result[t] = np.full((grid_size, grid_size), 0.0)
                continue
            
            # 获取有效点
            valid_points = points_orig[valid_mask]
            valid_values = values_orig[valid_mask]
            
            # 根据有效点数量选择插值方法
            num_valid_points = len(valid_points)
            
            if num_valid_points == 0:
                # 没有有效点，使用fill_value
                result[t] = np.full((grid_size, grid_size), 0.0)
            elif num_valid_points == 1:
                # 只有一个点，直接填充整个网格
                result[t] = np.full((grid_size, grid_size), float(valid_values[0]))
            elif num_valid_points < 4:
                # 少于4个点，使用nearest方法（只需要1个点）
                values_target = griddata(
                    valid_points,
                    valid_values,
                    points_target,
                    method='nearest',
                    fill_value=0.0
                )
                result[t] = values_target.reshape(grid_size, grid_size)
            else:
                # 4个或更多点，使用linear方法
                try:
                    values_target = griddata(
                        valid_points,
                        valid_values,
                        points_target,
                        method='linear',
                        fill_value=0.0
                    )
                    result[t] = values_target.reshape(grid_size, grid_size)
                except Exception as e:
                    # 如果linear方法失败（例如共线点），回退到nearest
                    values_target = griddata(
                        valid_points,
                        valid_values,
                        points_target,
                        method='nearest',
                        fill_value=0.0
                    )
                    result[t] = values_target.reshape(grid_size, grid_size)
        else:
            # 没有数据，使用fill_value
            result[t] = np.full((grid_size, grid_size), 0.0)
    
    return result


# ============================================================================
# 测试函数
# ============================================================================

if __name__ == '__main__':
    # 测试代码
    spatial_bounds = {
        'lat_min': 34.0,
        'lat_max': 35.0,
        'lon_min': -119.0,
        'lon_max': -118.0
    }
    
    time_window_start = datetime(2017, 1, 1)
    time_window_end = datetime(2017, 1, 10)
    fire_date = datetime(2017, 1, 11)
    
    features = extract_aligned_features(
        spatial_bounds=spatial_bounds,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        grid_size=25,
        fire_date=fire_date,
        fire_year=2017,
        iso3='USA',
        data_dir='dataset'
    )
    
    print(f"特征立方体形状: {features.shape}")
    print(f"特征统计:")
    for ch in range(7):
        ch_data = features[:, :, :, ch]
        print(f"  通道 {ch}: 均值={ch_data.mean():.4f}, 非零比例={(ch_data != 0).sum() / ch_data.size * 100:.2f}%")

