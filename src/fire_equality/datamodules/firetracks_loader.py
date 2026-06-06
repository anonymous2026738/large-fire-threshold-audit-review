"""
FireTracks数据加载模块
用于加载和处理FireTracks科学数据集
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import warnings
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple
warnings.filterwarnings('ignore')

# 导入国家名称到ISO3代码的转换函数
try:
    from .country_to_iso3 import add_iso3_to_dataframe
except ImportError:
    try:
        from code.fire_equality.datamodules.country_to_iso3 import add_iso3_to_dataframe
    except ImportError:
        # 最终回退：将当前目录加入sys.path并按普通模块名导入
        try:
            current_dir = Path(__file__).resolve().parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from country_to_iso3 import add_iso3_to_dataframe
        except Exception:
            # 如果导入失败，定义一个空函数（不会报错，但不会添加iso3列）
            def add_iso3_to_dataframe(df, country_column='country'):
                print("⚠️  无法导入country_to_iso3模块，无法添加iso3列")
                return df

# 导入特征对齐函数（从同目录模块）
# 注意：此函数仅在生成训练数据时需要，加载已生成的数据时不需要
extract_aligned_features = None  # type: ignore
try:
    from .feature_alignment import extract_aligned_features  # type: ignore
except ImportError:
    try:
        from code.fire_equality.datamodules.feature_alignment import extract_aligned_features  # type: ignore
    except ImportError:
        # 最终回退：将当前目录加入sys.path并按普通模块名导入
        try:
            current_dir = Path(__file__).resolve().parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from feature_alignment import extract_aligned_features  # type: ignore
        except Exception:
            # 导入失败，但不立即警告（只在真正需要使用时才警告）
            # 这样可以避免在加载已生成数据时显示误导性警告
            pass

# 本地 landcover 读取函数（使用本地 GeoTIFF 文件）
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except (ImportError, OSError, Exception) as e:
    # ImportError: 模块未安装
    # OSError: DLL加载失败（Windows常见问题）
    # Exception: 其他错误
    HAS_RASTERIO = False
    # 只在真正需要使用时才警告，避免导入时的误报
    # 注意：DLL错误通常不影响功能，因为可能只是某些功能不可用

# 标记：现在使用本地 landcover 数据，不再需要 GEE API
HAS_MODIS_LC = True  # 总是为 True，因为使用本地文件
MODIS_LC_ERROR = None

def get_landcover_from_local(lat: float, lon: float, year: int, data_dir: str = 'dataset') -> int:
    """
    从本地 GeoTIFF 文件获取单个坐标的土地覆盖类型
    
    Args:
        lat: 纬度
        lon: 经度
        year: 年份
        data_dir: 数据目录，默认 'dataset'
    
    Returns:
        土地覆盖类型编码（整数），如果失败返回 255（未分类）
    """
    if not HAS_RASTERIO:
        # rasterio未安装或DLL加载失败，返回未分类值
        return 255
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️  本地 landcover 文件不存在: {local_file}")
        return 255
    
    try:
        with rasterio.open(local_file) as src:
            # 使用 sample 方法获取单个点的值
            values = list(src.sample([(lon, lat)]))
            if values and len(values) > 0:
                lc_value = int(values[0][0])
                # 处理 NoData 值
                if lc_value == src.nodata or np.isnan(lc_value):
                    return 255
                return lc_value
            else:
                return 255
    except Exception as e:
        print(f"⚠️  读取本地 landcover 文件失败 ({lat}, {lon}, {year}): {e}")
        return 255


def get_landcover_batch_local(coords, year: int, data_dir: str = 'dataset', batch_size: int = 1000):
    """
    批量从本地 GeoTIFF 文件获取土地覆盖类型
    
    Args:
        coords: 坐标列表，每个元素为 (lat, lon) 或索引
        year: 年份
        data_dir: 数据目录，默认 'dataset'
        batch_size: 批处理大小（保留参数以兼容接口，实际不使用）
    
    Returns:
        土地覆盖类型编码列表
    """
    if not HAS_RASTERIO:
        print("⚠️  rasterio 未安装，无法读取 landcover 数据")
        return [255] * len(coords)
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️  本地 landcover 文件不存在: {local_file}")
        return [255] * len(coords)
    
    results = []
    try:
        with rasterio.open(local_file) as src:
            # 提取所有坐标的经度、纬度
            lon_lat_pairs = [(lon, lat) for lat, lon in coords]
            
            # 批量采样
            for i, (lon, lat) in enumerate(lon_lat_pairs):
                try:
                    values = list(src.sample([(lon, lat)]))
                    if values and len(values) > 0:
                        lc_value = int(values[0][0])
                        # 处理 NoData 值
                        if lc_value == src.nodata or np.isnan(lc_value):
                            results.append(255)
                        else:
                            results.append(lc_value)
                    else:
                        results.append(255)
                except Exception as e:
                    # 单个点失败，返回 255
                    results.append(255)
            
            return results
    except Exception as e:
        print(f"⚠️  批量读取本地 landcover 文件失败: {e}")
        return [255] * len(coords)


def find_locations_by_landcover_local(positive_lc_types, spatial_bounds: dict, year: int,
                                     num_samples_per_type: int = 1000, data_dir: str = 'dataset'):
    """
    从本地 GeoTIFF 文件中搜索特定土地覆盖类型的区域位置
    
    Args:
        positive_lc_types: 正样本需要的土地覆盖类型列表，例如 [7, 9, 10]
        spatial_bounds: 研究区域的空间边界
        year: 年份
        num_samples_per_type: 每种土地覆盖类型采样的像素数量
        data_dir: 数据目录，默认 'dataset'
    
    Returns:
        位置列表，每个元素为 {'lat': float, 'lon': float, 'land_cover': int}
    """
    if not HAS_RASTERIO:
        print("⚠️  rasterio 未安装，无法读取 landcover 数据")
        return []
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️  本地 landcover 文件不存在: {local_file}")
        return []
    
    all_locations = []
    
    try:
        with rasterio.open(local_file) as src:
            # 读取整个研究区域的数据
            # 计算边界在图像中的像素坐标
            from rasterio.warp import transform
            
            # 将地理坐标转换为像素坐标
            lon_min, lat_min = spatial_bounds['lon_min'], spatial_bounds['lat_min']
            lon_max, lat_max = spatial_bounds['lon_max'], spatial_bounds['lat_max']
            
            # 读取边界框内的数据
            # 使用 window 读取指定区域
            from rasterio.windows import from_bounds
            
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            
            # 读取数据
            data = src.read(1, window=window)
            
            # 获取窗口的变换矩阵
            window_transform = src.window_transform(window)
            
            # 为每种土地覆盖类型采样
            for lc_type in positive_lc_types:
                # 找到所有匹配的像素
                mask = (data == lc_type)
                matching_pixels = np.where(mask)
                
                if len(matching_pixels[0]) == 0:
                    print(f"    类型 {lc_type}: 在研究区域内未找到")
                    continue
                
                # 随机采样
                num_found = len(matching_pixels[0])
                num_to_sample = min(num_samples_per_type, num_found)
                
                if num_found > num_samples_per_type:
                    # 随机选择
                    indices = np.random.choice(num_found, num_to_sample, replace=False)
                    sampled_rows = matching_pixels[0][indices]
                    sampled_cols = matching_pixels[1][indices]
                else:
                    # 使用所有匹配的像素
                    sampled_rows = matching_pixels[0]
                    sampled_cols = matching_pixels[1]
                
                # 将像素坐标转换为地理坐标
                type_locations = []
                for row, col in zip(sampled_rows, sampled_cols):
                    # 像素坐标（相对于窗口）
                    pixel_x = col
                    pixel_y = row
                    
                    # 转换为地理坐标
                    lon, lat = rasterio.transform.xy(window_transform, pixel_y, pixel_x)
                    
                    type_locations.append({
                        'lat': lat,
                        'lon': lon,
                        'land_cover': int(lc_type)
                    })
                
                all_locations.extend(type_locations)
                print(f"    类型 {lc_type}: 找到 {len(type_locations)} 个位置（总共 {num_found} 个匹配像素）")
        
        print(f"  ✅ 总共找到 {len(all_locations):,} 个位置（包含 {len(set(l['land_cover'] for l in all_locations))} 种土地覆盖类型）")
        
        return all_locations
        
    except Exception as e:
        print(f"⚠️  搜索土地覆盖类型位置失败: {e}")
        import traceback
        traceback.print_exc()
        return []


# 为了兼容性，创建别名函数（替换原来的 GEE API 函数）
def get_landcover_batch_gee(coords, year: int, lc_type: str = 'LC_Type1', 
                            batch_size: int = 1000, project: Optional[str] = None):
    """
    批量获取土地覆盖类型（本地版本，替换 GEE API）
    
    注意：project 参数保留以兼容接口，但实际不使用
    """
    return get_landcover_batch_local(coords, year, data_dir='dataset', batch_size=batch_size)


def find_locations_by_landcover(positive_lc_types, spatial_bounds: dict, year: int,
                                num_samples_per_type: int = 1000, project: Optional[str] = None):
    """
    搜索特定土地覆盖类型的区域位置（本地版本，替换 GEE API）
    
    注意：project 参数保留以兼容接口，但实际不使用
    """
    return find_locations_by_landcover_local(
        positive_lc_types, spatial_bounds, year, 
        num_samples_per_type, data_dir='dataset'
    )


def load_firetracks_dataset(data_directory, max_rows=None, year_range=None, use_chunks=False, chunk_size=1000000):
    """
    加载FireTracks数据集
    
    Args:
        data_directory: 包含FireTracks数据文件的目录路径
        max_rows: 最大读取行数（用于测试，None表示读取全部）
        year_range: 年份范围过滤，例如 (2002, 2020)，None表示不过滤
        use_chunks: 是否使用分块读取（适用于大文件）
        chunk_size: 分块大小（当use_chunks=True时使用）
    
    Returns:
        dict: 包含所有FireTracks数据集的字典
            - 'events': 活跃火事件表 (v.h5)
            - 'events_lc': 活跃火事件土地覆盖表 (v_LC_Type1.h5)
            - 'components': 时空火灾组件表 (cp.h5)
            - 'components_lc': 时空火灾组件土地覆盖表 (cp_LC_Type1.h5)
    """
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:292","message":"load_firetracks_dataset entry","data":{"data_directory":data_directory,"max_rows":max_rows,"year_range":year_range},"timestamp":int(time_module.time()*1000)}) + '\n')
    except Exception as log_err:
        pass  # 忽略日志错误，不影响主流程
    # #endregion
    print(f"从 {data_directory} 加载FireTracks数据...")
    if max_rows:
        print(f"⚠️  限制读取: 最多 {max_rows:,} 行")
    if year_range:
        print(f"⚠️  年份过滤: {year_range[0]}-{year_range[1]}")
    
    datasets = {}
    
    import os
    
    try:
        # 加载核心数据表（必需文件）
        if os.path.exists(f'{data_directory}/v.h5'):
            # #region agent log
            import json
            import os as os_module
            file_path_events = f'{data_directory}/v.h5'
            file_size_gb_events = os_module.path.getsize(file_path_events) / (1024**3) if os_module.path.exists(file_path_events) else 0
            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:322","message":"Events file size check","data":{"file_size_gb":round(file_size_gb_events,2),"file_exists":os_module.path.exists(file_path_events)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            # 优先使用where条件过滤（如果支持且提供了year_range）
            if year_range:
                try:
                    start_date = f"{year_range[0]}-01-01"
                    end_date = f"{year_range[1]+1}-01-01"
                    print(f"尝试使用where条件过滤 events (dtime >= {start_date} & dtime < {end_date})...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:329","message":"Attempting where clause filter","data":{"start_date":start_date,"end_date":end_date,"year_range":year_range},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    datasets['events'] = pd.read_hdf(
                        f'{data_directory}/v.h5',
                        where=f'dtime >= "{start_date}" & dtime < "{end_date}"'
                    )
                    print(f"✅ 使用where条件成功加载 {len(datasets['events']):,} 行")
                    # #region agent log
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:333","message":"Where clause succeeded","data":{"rows_loaded":len(datasets['events'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                except Exception as e:
                    print(f"⚠️  where条件失败 ({e})，尝试其他方法...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:335","message":"Where clause failed","data":{"error":str(e)[:200],"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    # 如果where条件失败，尝试限制读取行数或使用分块读取
                    if max_rows:
                        print(f"使用迭代器读取前 {max_rows:,} 行...")
                        store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                        try:
                            datasets['events'] = store.select('table', start=0, stop=max_rows)
                        finally:
                            store.close()
                    else:
                        # where条件失败时，直接使用分块读取（避免内存不足）
                        # 检查文件大小作为参考
                        # #region agent log
                        import os as os_check
                        file_path = f'{data_directory}/v.h5'
                        file_size_gb = os_check.path.getsize(file_path) / (1024**3) if os_check.path.exists(file_path) else 0
                        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"firetracks_loader.py:375","message":"Where clause failed, using chunked reading","data":{"file_size_gb":round(file_size_gb,2),"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # where条件失败时，直接使用分块读取（更安全，避免内存不足）
                        if True:  # 总是使用分块读取当where条件失败时
                            reason = f"where条件失败，文件大小 ({file_size_gb:.1f} GB)"
                            print(f"⚠️  {reason}，使用分块读取...")
                            # #region agent log
                            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:373","message":"Using chunked reading","data":{"chunk_size":10000000,"reason":reason},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                            # #endregion
                            # 使用分块读取，每块1000万行
                            chunk_size = 10000000
                            chunks = []
                            store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                            try:
                                total_rows = store.get_storer('table').nrows
                                print(f"   总行数: {total_rows:,}")
                                for start in range(0, total_rows, chunk_size):
                                    stop = min(start + chunk_size, total_rows)
                                    print(f"   读取行 {start:,} 到 {stop:,}...", end='\r')
                                    chunk = store.select('table', start=start, stop=stop)
                                    # 如果提供了year_range，在内存中过滤
                                    if year_range:
                                        chunk['dtime'] = pd.to_datetime(chunk['dtime'])
                                        start_date_pd = pd.Timestamp(f"{year_range[0]}-01-01")
                                        end_date_pd = pd.Timestamp(f"{year_range[1]+1}-01-01")
                                        chunk = chunk[(chunk['dtime'] >= start_date_pd) & (chunk['dtime'] < end_date_pd)]
                                    if len(chunk) > 0:
                                        chunks.append(chunk)
                                print()  # 换行
                            finally:
                                store.close()
                            
                            if chunks:
                                datasets['events'] = pd.concat(chunks, ignore_index=True)
                                print(f"✅ 分块读取完成，共加载 {len(datasets['events']):,} 行")
                                # #region agent log
                                with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:417","message":"Chunked reading completed","data":{"rows_loaded":len(datasets['events'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                            else:
                                raise ValueError("分块读取后未找到匹配的数据")
                        # 注意：当where条件失败时，我们总是使用分块读取，不再尝试直接读取全部文件
                        # 这样可以避免内存不足的问题
            elif max_rows:
                # 只限制行数
                print(f"使用迭代器读取前 {max_rows:,} 行 events...")
                store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                try:
                    datasets['events'] = store.select('table', start=0, stop=max_rows)
                finally:
                    store.close()
            else:
                # 读取全部数据
                datasets['events'] = pd.read_hdf(f'{data_directory}/v.h5')
        
        # 添加iso3列（如果events有country列但没有iso3列）
        if 'events' in datasets and len(datasets['events']) > 0:
            if 'country' in datasets['events'].columns:
                if 'iso3' not in datasets['events'].columns:
                    print("正在从country列添加iso3列...")
                    add_iso3_to_dataframe(datasets['events'], country_column='country')
                else:
                    # 检查iso3列是否大部分为空
                    iso3_valid = datasets['events']['iso3'].dropna()
                    iso3_valid = iso3_valid[iso3_valid != '']
                    if len(iso3_valid) < len(datasets['events']) * 0.1:
                        print("iso3列大部分为空，尝试从country列补充...")
                        # 只更新iso3为空的行
                        mask = datasets['events']['iso3'].isna() | (datasets['events']['iso3'] == '')
                        if mask.sum() > 0:
                            # 导入country_to_iso3函数
                            try:
                                from .country_to_iso3 import country_to_iso3
                            except ImportError:
                                try:
                                    from code.fire_equality.datamodules.country_to_iso3 import country_to_iso3
                                except ImportError:
                                    current_dir = Path(__file__).resolve().parent
                                    if str(current_dir) not in sys.path:
                                        sys.path.insert(0, str(current_dir))
                                    from country_to_iso3 import country_to_iso3
                            datasets['events'].loc[mask, 'iso3'] = datasets['events'].loc[mask, 'country'].apply(country_to_iso3)
                            new_iso3_count = datasets['events'].loc[mask, 'iso3'].dropna()
                            new_iso3_count = new_iso3_count[new_iso3_count != '']
                            print(f"✅ 补充了 {len(new_iso3_count)} 个ISO3代码")
        else:
            raise FileNotFoundError(f"未找到文件: {data_directory}/v.h5")
        
        if os.path.exists(f'{data_directory}/cp.h5'):
            # 优先使用where条件过滤（如果支持且提供了year_range）
            if year_range:
                try:
                    start_date = f"{year_range[0]}-01-01"
                    end_date = f"{year_range[1]+1}-01-01"
                    print(f"尝试使用where条件过滤 components (dtime_min >= {start_date} & dtime_min < {end_date})...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:398","message":"Attempting where clause filter for components","data":{"start_date":start_date,"end_date":end_date},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    datasets['components'] = pd.read_hdf(
                        f'{data_directory}/cp.h5',
                        where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
                    )
                    print(f"✅ 使用where条件成功加载 {len(datasets['components']):,} 行")
                except Exception as e:
                    print(f"⚠️  where条件失败 ({e})，尝试其他方法...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:410","message":"Components where clause failed","data":{"error":str(e)[:200],"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    # 如果where条件失败，尝试限制读取行数或使用分块读取
                    if max_rows:
                        print(f"使用迭代器读取前 {max_rows:,} 行...")
                        store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                        try:
                            datasets['components'] = store.select('table', start=0, stop=max_rows)
                        finally:
                            store.close()
                    else:
                        # where条件失败时，直接使用分块读取（避免内存不足）
                        # 检查文件大小作为参考
                        # #region agent log
                        import os as os_check_comp
                        file_path = f'{data_directory}/cp.h5'
                        file_size_gb = os_check_comp.path.getsize(file_path) / (1024**3) if os_check_comp.path.exists(file_path) else 0
                        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"firetracks_loader.py:545","message":"Components where clause failed, using chunked reading","data":{"file_size_gb":round(file_size_gb,2)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # where条件失败时，直接使用分块读取（更安全，避免内存不足）
                        if True:  # 总是使用分块读取当where条件失败时
                            reason_comp = f"where条件失败，文件大小 ({file_size_gb:.1f} GB)"
                            print(f"⚠️  {reason_comp}，使用分块读取...")
                            chunk_size = 1000000  # components通常比events小，使用较小的块
                            chunks = []
                            store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                            try:
                                total_rows = store.get_storer('table').nrows
                                print(f"   总行数: {total_rows:,}")
                                for start in range(0, total_rows, chunk_size):
                                    stop = min(start + chunk_size, total_rows)
                                    print(f"   读取行 {start:,} 到 {stop:,}...", end='\r')
                                    chunk = store.select('table', start=start, stop=stop)
                                    # 如果提供了year_range，在内存中过滤
                                    if year_range:
                                        chunk['dtime_min'] = pd.to_datetime(chunk['dtime_min'])
                                        start_date_pd = pd.Timestamp(f"{year_range[0]}-01-01")
                                        end_date_pd = pd.Timestamp(f"{year_range[1]+1}-01-01")
                                        chunk = chunk[(chunk['dtime_min'] >= start_date_pd) & (chunk['dtime_min'] < end_date_pd)]
                                    if len(chunk) > 0:
                                        chunks.append(chunk)
                                print()  # 换行
                            finally:
                                store.close()
                            
                            if chunks:
                                datasets['components'] = pd.concat(chunks, ignore_index=True)
                                print(f"✅ 分块读取完成，共加载 {len(datasets['components']):,} 行")
                                # #region agent log
                                with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:548","message":"Components chunked reading completed","data":{"rows_loaded":len(datasets['components'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                            else:
                                raise ValueError("分块读取后未找到匹配的数据")
                        # 注意：当where条件失败时，我们总是使用分块读取，不再尝试直接读取全部文件
                        # 这样可以避免内存不足的问题
            elif max_rows:
                # 只限制行数
                print(f"使用迭代器读取前 {max_rows:,} 行 components...")
                store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                try:
                    datasets['components'] = store.select('table', start=0, stop=max_rows)
                finally:
                    store.close()
            else:
                # 读取全部数据
                datasets['components'] = pd.read_hdf(f'{data_directory}/cp.h5')
        else:
            raise FileNotFoundError(f"未找到文件: {data_directory}/cp.h5")
        
        # 加载可选数据表（如果存在且内存允许）
        # 注意：land cover 数据是可选的，如果内存不足可以跳过
        if os.path.exists(f'{data_directory}/v_LC_Type1.h5') and max_rows is None:
            try:
                # 如果提供了年份范围，尝试使用where条件过滤
                if year_range:
                    try:
                        start_date = f"{year_range[0]}-01-01"
                        end_date = f"{year_range[1]+1}-01-01"
                        datasets['events_lc'] = pd.read_hdf(
                            f'{data_directory}/v_LC_Type1.h5',
                            where=f'dtime >= "{start_date}" & dtime < "{end_date}"'
                        )
                        print("✅ 成功加载 events_lc (已过滤年份)")
                    except:
                        # 如果where条件失败，尝试加载全部（可能失败）
                        datasets['events_lc'] = pd.read_hdf(f'{data_directory}/v_LC_Type1.h5')
                        print("✅ 成功加载 events_lc")
                else:
                    datasets['events_lc'] = pd.read_hdf(f'{data_directory}/v_LC_Type1.h5')
                    print("✅ 成功加载 events_lc")
            except (MemoryError, OSError, Exception) as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"⚠️  跳过 events_lc（内存不足或文件损坏）")
        
        if os.path.exists(f'{data_directory}/cp_LC_Type1.h5') and max_rows is None:
            try:
                # 如果提供了年份范围，尝试使用where条件过滤
                if year_range:
                    try:
                        start_date = f"{year_range[0]}-01-01"
                        end_date = f"{year_range[1]+1}-01-01"
                        datasets['components_lc'] = pd.read_hdf(
                            f'{data_directory}/cp_LC_Type1.h5',
                            where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
                        )
                        print("✅ 成功加载 components_lc (已过滤年份)")
                    except:
                        # 如果where条件失败，尝试加载全部（可能失败）
                        datasets['components_lc'] = pd.read_hdf(f'{data_directory}/cp_LC_Type1.h5')
                        print("✅ 成功加载 components_lc")
                else:
                    datasets['components_lc'] = pd.read_hdf(f'{data_directory}/cp_LC_Type1.h5')
                    print("✅ 成功加载 components_lc")
            except (MemoryError, OSError, Exception) as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"⚠️  跳过 components_lc（内存不足或文件损坏）")
        
        # 添加iso3列（如果events有country列但没有iso3列）
        if 'events' in datasets and len(datasets['events']) > 0:
            if 'country' in datasets['events'].columns:
                if 'iso3' not in datasets['events'].columns:
                    print("正在从country列添加iso3列...")
                    add_iso3_to_dataframe(datasets['events'], country_column='country')
                else:
                    # 检查iso3列是否大部分为空
                    iso3_valid = datasets['events']['iso3'].dropna()
                    iso3_valid = iso3_valid[iso3_valid != '']
                    if len(iso3_valid) < len(datasets['events']) * 0.1:
                        print("iso3列大部分为空，尝试从country列补充...")
                        # 只更新iso3为空的行
                        mask = datasets['events']['iso3'].isna() | (datasets['events']['iso3'] == '')
                        if mask.sum() > 0:
                            # 导入country_to_iso3函数
                            try:
                                from .country_to_iso3 import country_to_iso3
                            except ImportError:
                                try:
                                    from code.fire_equality.datamodules.country_to_iso3 import country_to_iso3
                                except ImportError:
                                    current_dir = Path(__file__).resolve().parent
                                    if str(current_dir) not in sys.path:
                                        sys.path.insert(0, str(current_dir))
                                    from country_to_iso3 import country_to_iso3
                            datasets['events'].loc[mask, 'iso3'] = datasets['events'].loc[mask, 'country'].apply(country_to_iso3)
                            new_iso3_count = datasets['events'].loc[mask, 'iso3'].dropna()
                            new_iso3_count = new_iso3_count[new_iso3_count != '']
                            print(f"✅ 补充了 {len(new_iso3_count)} 个ISO3代码")
        
        print("✅ FireTracks数据加载成功!")
        print(f"   - 活跃火事件: {len(datasets['events']):,} 条")
        print(f"   - 时空火灾组件: {len(datasets['components']):,} 个")
        
    except MemoryError as me:
        # #region agent log
        import json
        import time as time_module
        try:
            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"firetracks_loader.py:701","message":"MemoryError in load_firetracks_dataset","data":{"error":str(me)[:300]},"timestamp":int(time_module.time()*1000)}) + '\n')
        except:
            pass
        # #endregion
        print(f"❌ 数据加载失败（内存不足）: {me}")
        print(f"   建议：使用分块读取或减少年份范围")
        return None
    except Exception as e:
        # #region agent log
        import json
        import time as time_module
        try:
            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:710","message":"Exception in load_firetracks_dataset","data":{"error":str(e)[:300]},"timestamp":int(time_module.time()*1000)}) + '\n')
        except:
            pass
        # #endregion
        print(f"❌ 数据加载失败: {e}")
        return None
    
    return datasets


# 数据预处理和特征构建

def preprocess_firetracks_data(datasets, target_year_range=(2002, 2020)):
    """
    预处理FireTracks数据，构建ConvLSTM可用的特征
    
    Args:
        datasets: 加载的数据集字典
        target_year_range: 目标年份范围
    
    Returns:
        dict: 预处理后的数据集字典
            - 'components': 过滤和特征增强后的火灾组件表
            - 'events': 原始事件表
            - 'events_lc': 事件土地覆盖表
            - 'components_lc': 组件土地覆盖表
    """
    print("开始数据预处理...")
    
    # 1. 过滤目标年份
    components = datasets['components'].copy()
    events = datasets['events'].copy()
    
    # 转换日期格式
    components['dtime_min'] = pd.to_datetime(components['dtime_min'])
    events['dtime'] = pd.to_datetime(events['dtime'])
    
    # 过滤年份
    start_year, end_year = target_year_range
    mask = (components['dtime_min'].dt.year >= start_year) & (components['dtime_min'].dt.year <= end_year)
    filtered_components = components[mask].copy()
    
    print(f"过滤后保留 {len(filtered_components):,} 个火灾组件 ({start_year}-{end_year})")
    
    # 2. 计算额外特征
    filtered_components['frp_intensity'] = filtered_components['maxFRP_sum'] / filtered_components['area']
    filtered_components['expansion_rate'] = filtered_components['area'] / filtered_components['duration']
    
    # 3. 创建火灾严重度分类
    # 使用95%分位数作为阈值，标记前5%（top 5%）的火灾为严重火灾
    # 标准：maxFRP_sum（总辐射功率）
    frp_threshold = filtered_components['maxFRP_sum'].quantile(0.95)
    filtered_components['fire_severity'] = (filtered_components['maxFRP_sum'] >= frp_threshold).astype(int)
    
    print(f"严重火灾阈值: {frp_threshold:,.0f} MW (95%分位数)")
    print(f"严重火灾数量: {filtered_components['fire_severity'].sum():,} 个 (约前5%)")
    print(f"非严重火灾数量: {(filtered_components['fire_severity'] == 0).sum():,} 个 (约95%)")
    
    # 4. 只保留严重火灾组件（用于像素级二分类任务）
    severe_components = filtered_components[filtered_components['fire_severity'] == 1].copy()
    print(f"保留严重火灾组件: {len(severe_components):,} 个（用于正样本生成）")
    
    # 构建返回字典，只包含存在的数据表
    result = {
        'components': filtered_components,  # 所有组件（包含严重和非严重）
        'severe_components': severe_components,  # 只包含严重火灾组件
        'events': events,
    }
    
    # 可选数据表
    if 'events_lc' in datasets:
        result['events_lc'] = datasets['events_lc']
    
    if 'components_lc' in datasets:
        result['components_lc'] = datasets['components_lc']
    
    return result


# 构建ConvLSTM时空数据块

def create_spatiotemporal_patches(preprocessed_data, patch_size_km=25, time_steps=10, spatial_resolution_km=1, max_samples=None):
    """
    为每个火灾组件创建时空数据块
    
    Args:
        preprocessed_data: 预处理后的数据
        patch_size_km: 空间块大小 (km)
        time_steps: 时间步长 (天)
        spatial_resolution_km: 空间分辨率 (km)
        max_samples: 最大处理样本数（用于测试，None表示处理全部）
    
    Returns:
        list: 时空数据块列表，每个元素包含：
            - component_id: 组件ID
            - features: 特征立方体 [time_steps, grid_size, grid_size, channels]
            - targets: 目标值字典
            - metadata: 元数据字典
    """
    print("构建时空数据块...")
    
    components = preprocessed_data['components']
    events = preprocessed_data['events']
    
    total_components = len(components)
    if max_samples:
        total_components = min(total_components, max_samples)
        print(f"   ⚠️  限制处理数量: 最多 {max_samples:,} 个组件")
    
    print(f"   - 待处理组件数量: {total_components:,} 个")
    print(f"   - 事件数据量: {len(events):,} 条")
    
    spatiotemporal_samples = []
    
    # 添加进度输出
    progress_interval = max(1, total_components // 100)  # 每1%输出一次
    
    processed_count = 0
    for idx, component in components.iterrows():
        if max_samples and processed_count >= max_samples:
            break
        processed_count += 1
        # 进度输出
        if processed_count % progress_interval == 0 or processed_count == total_components:
            progress = processed_count / total_components * 100
            print(f"   处理进度: {processed_count:,}/{total_components:,} ({progress:.1f}%) - 已创建 {len(spatiotemporal_samples):,} 个样本", end='\r')
        component_id = component['cp']
        center_lat = component['lat_mean']
        center_lon = component['lon_mean']
        start_date = component['dtime_min']
        
        # 计算空间边界 (近似经纬度转换)
        lat_per_km = 1 / 110.574
        lon_per_km = 1 / (111.320 * np.cos(np.radians(center_lat)))
        
        patch_radius_deg = (patch_size_km / 2) * lat_per_km
        
        spatial_bounds = {
            'lat_min': center_lat - patch_radius_deg,
            'lat_max': center_lat + patch_radius_deg,
            'lon_min': center_lon - patch_radius_deg, 
            'lon_max': center_lon + patch_radius_deg
        }
        
        # 计算时间窗口 (火灾发生前的时间)
        time_window_start = start_date - pd.Timedelta(days=time_steps)
        time_window_end = start_date - pd.Timedelta(days=1)
        
        # 提取该时空窗口内的事件
        spatial_mask = (
            (events['lat'] >= spatial_bounds['lat_min']) & 
            (events['lat'] <= spatial_bounds['lat_max']) &
            (events['lon'] >= spatial_bounds['lon_min']) & 
            (events['lon'] <= spatial_bounds['lon_max'])
        )
        
        temporal_mask = (
            (events['dtime'] >= time_window_start) & 
            (events['dtime'] <= time_window_end)
        )
        
        window_events = events[spatial_mask & temporal_mask].copy()
        
        # 调试信息：检查是否有事件
        if processed_count <= 5:  # 只对前5个样本输出调试信息
            print(f"\n   调试样本 {processed_count}: 组件ID={component_id}, 时间窗口={time_window_start.date()} 到 {time_window_end.date()}")
            print(f"      空间范围: lat[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
            print(f"      找到 {len(window_events)} 个事件")
            if len(window_events) > 0:
                print(f"      事件日期范围: {window_events['dtime'].min()} 到 {window_events['dtime'].max()}")
        
        # 创建特征网格
        grid_size = int(patch_size_km / spatial_resolution_km)  # 25x25网格
        if extract_aligned_features is None:
            error_msg = (
                "❌ extract_aligned_features 未导入成功，无法生成特征。\n"
                "   这通常是由于DLL加载失败（Windows常见问题）或缺少依赖库。\n"
                "   请检查：\n"
                "   1. 是否安装了所有必需的依赖（numpy, scipy, xarray等）\n"
                "   2. 是否安装了Visual C++运行时库\n"
                "   3. 如果只是加载已生成的数据，可以忽略此错误\n"
                "   文件位置: code/fire_equality/datamodules/feature_alignment.py"
            )
            raise ImportError(error_msg)
        # 提取ISO3国家代码（如果可用）
        iso3_code = extract_iso3_from_events(window_events, spatial_bounds, preprocessed_data.get('events'))
        # 使用正确的时间窗口：火灾发生前time_steps天到火灾发生前1天
        feature_cube = extract_aligned_features(
            spatial_bounds=spatial_bounds,
            time_window_start=time_window_start,  # 火灾前time_steps天
            time_window_end=time_window_end,  # 火灾前1天（使用第580行计算好的正确值）
            grid_size=grid_size,
            fire_date=pd.Timestamp(start_date).to_pydatetime(),  # 火灾发生日期
            fire_year=pd.Timestamp(start_date).year,
            iso3=iso3_code,
            data_dir='dataset',
            project='ee-tpan2203-wildfire'
        )
        
        if feature_cube is not None:
            sample = {
                'component_id': component_id,
                'features': feature_cube,  # [time_steps, grid_size, grid_size, channels]
                'targets': {
                    'total_frp': component['maxFRP_sum'],
                    'fire_severity': component['fire_severity'],
                    'duration': component['duration'],
                    'area': component['area']
                },
                'metadata': {
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'start_date': start_date,
                    'spatial_bounds': spatial_bounds,
                    'iso3': iso3_code
                }
            }
            spatiotemporal_samples.append(sample)
    
    print()  # 换行
    
    # 统计特征填充情况
    samples_with_features = sum(1 for s in spatiotemporal_samples if s['features'].sum() > 0)
    samples_without_features = len(spatiotemporal_samples) - samples_with_features
    
    print(f"✅ 成功创建 {len(spatiotemporal_samples):,} 个时空数据块")
    if samples_without_features > 0:
        print(f"   ⚠️  警告: {samples_without_features} 个样本的特征全为0（可能是时间窗口内无事件）")
    print(f"   ✅ {samples_with_features} 个样本包含有效特征")
    
    return spatiotemporal_samples


def create_feature_cube(*args, **kwargs):
    """
    该函数已废弃。请使用 extract_aligned_features 生成8通道特征。
    """
    raise NotImplementedError("create_feature_cube 已废弃。请改用 extract_aligned_features。")


# 创建PyTorch数据集

class FireTracksDataset(Dataset):
    """
    FireTracks PyTorch数据集
    
    用于将时空数据块转换为PyTorch可用的数据集格式
    """
    
    def __init__(self, spatiotemporal_samples, target_type='severity'):
        """
        FireTracks PyTorch数据集
        
        Args:
            spatiotemporal_samples: 时空数据块列表
            target_type: 目标类型 ('severity', 'frp', 'duration')
        """
        self.samples = spatiotemporal_samples
        self.target_type = target_type
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            tuple: (features, target)
                - features: torch.Tensor, shape [C, T, H, W]
                - target: torch.Tensor, 目标值（根据target_type确定类型）
        """
        sample = self.samples[idx]
        
        # 特征: [channels, timesteps, height, width]
        features = torch.from_numpy(sample['features']).float()
        features = features.permute(3, 0, 1, 2)  # 从 [T,H,W,C] 到 [C,T,H,W]
        
        # 目标
        if self.target_type == 'binary_classification':
            # 像素级二分类任务：直接从sample中获取target
            target = torch.tensor(sample['target'], dtype=torch.long)
        elif self.target_type == 'severity':
            target = torch.tensor(sample['targets']['fire_severity'], dtype=torch.long)
        elif self.target_type == 'frp':
            target = torch.tensor(sample['targets']['total_frp'], dtype=torch.float)
        elif self.target_type == 'duration':
            target = torch.tensor(sample['targets']['duration'], dtype=torch.float)
        else:
            target = torch.tensor(sample['targets']['fire_severity'], dtype=torch.long)
        
        return features, target
    
    def get_sample_metadata(self, idx):
        """
        获取样本的元数据
        
        Args:
            idx: 样本索引
        
        Returns:
            dict: 样本元数据字典
        """
        return self.samples[idx]['metadata']


# 完整的数据处理流水线

def create_convLSTM_ready_dataset(data_directory, output_path=None, config=None):
    """
    端到端的FireTracks数据处理流水线
    
    整合数据加载、预处理、时空数据块创建和PyTorch数据集构建的完整流程
    
    Args:
        data_directory: FireTracks数据目录
        output_path: 处理后数据的保存路径 (可选)
        config: 处理配置字典，包含以下键：
            - patch_size_km: 空间块大小 (km)，默认25
            - time_steps: 时间步长 (天)，默认10
            - target_years: 目标年份范围，默认(2002, 2020)
            - target_type: 目标类型 ('severity', 'frp', 'duration')，默认'severity'
            - batch_size: 批次大小，默认32
    
    Returns:
        dict: 包含以下键的字典：
            - 'dataset': FireTracksDataset实例
            - 'dataloader': DataLoader实例
            - 'config': 使用的配置字典
            - 'preprocessed_data': 预处理后的数据字典
        如果处理失败，返回None
    """
    if config is None:
        config = {
            'patch_size_km': 25,
            'time_steps': 10,
            'target_years': (2002, 2020),
            'target_type': 'severity',
            'batch_size': 32,
            'max_samples': None  # 限制处理样本数（用于测试，None表示处理全部）
        }
    
    print("🚀 启动FireTracks数据处理流水线...")
    
    # 1. 加载数据（使用年份过滤以减少内存使用）
    # 注意：由于丢弃了需要补零的样本，events 数据只需要目标年份的数据
    print("\n[步骤1/5] 加载FireTracks数据...")
    start_year, end_year = config['target_years']
    # events 只需要目标年份的数据（不再需要前一年，因为会丢弃时间窗口不完整的样本）
    events_year_range = config['target_years']
    print(f"   Events 年份范围: {events_year_range} (只加载目标年份，节省内存)")
    
    datasets = load_firetracks_dataset(
        data_directory, 
        year_range=events_year_range
    )
    
    # 但是 components 只需要目标年份
    if datasets and 'components' in datasets:
        # 重新加载 components，只包含目标年份
        print(f"   重新加载 components，仅包含目标年份: {config['target_years']}")
        import os
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year+1}-01-01"
        try:
            datasets['components'] = pd.read_hdf(
                f'{data_directory}/cp.h5',
                where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
            )
            print(f"   ✅ Components: {len(datasets['components']):,} 行")
        except Exception as e:
            print(f"   ⚠️  重新加载 components 失败，使用已加载的数据")
    if datasets is None:
        return None
    
    # 2. 数据预处理
    print("\n[步骤2/5] 数据预处理...")
    import time
    start_time = time.time()
    preprocessed_data = preprocess_firetracks_data(
        datasets, 
        target_year_range=config['target_years']
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  预处理耗时: {elapsed_time:.1f} 秒")
    
    # 3. 创建时空数据块
    print("\n[步骤3/5] 创建时空数据块...")
    # 如果数据量太大，可以限制处理数量（用于测试）
    max_samples = config.get('max_samples', None)
    if max_samples:
        print(f"   ⚠️  限制处理: 最多 {max_samples:,} 个样本")
    
    start_time = time.time()
    spatiotemporal_samples = create_spatiotemporal_patches(
        preprocessed_data,
        patch_size_km=config['patch_size_km'],
        time_steps=config['time_steps'],
        max_samples=max_samples
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  创建数据块耗时: {elapsed_time:.1f} 秒")
    
    if not spatiotemporal_samples:
        print("❌ 未能创建时空数据块")
        return None
    
    # 4. 创建PyTorch数据集
    print("\n[步骤4/5] 创建PyTorch数据集...")
    dataset = FireTracksDataset(
        spatiotemporal_samples, 
        target_type=config['target_type']
    )
    
    # 5. 创建数据加载器
    print("\n[步骤5/5] 创建DataLoader...")
    # Windows系统上num_workers>0可能导致多进程问题，设为0避免
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print("\n✅ FireTracks数据处理完成!")
    print(f"   - 数据集大小: {len(dataset):,} 样本")
    if len(dataset) > 0:
        sample_features, sample_target = dataset[0]
        print(f"   - 输入维度: {sample_features.shape}")
        print(f"   - 目标类型: {config['target_type']}")
        print(f"   - 批次数量: {len(dataloader):,} 批次")
    
    # 7. 保存处理后的数据 (可选)
    if output_path:
        print(f"\n💾 保存数据至: {output_path}...")
        try:
            # 只保存原始数据，不保存dataset和dataloader（避免序列化问题）
            # 使用时可以根据config重新创建
            torch.save({
                'spatiotemporal_samples': spatiotemporal_samples,  # 原始数据
                'config': config,
                'metadata': {
                    'total_samples': len(dataset),
                    'input_shape': dataset[0][0].shape if len(dataset) > 0 else None,
                    'feature_channels': ['FRP强度', '火灾存在', '检测置信度', '邻域质量']
                }
            }, output_path)
            print(f"✅ 数据已保存至: {output_path}")
            print(f"   注意: 保存的是原始数据，使用时需根据config重新创建dataset和dataloader")
        except Exception as e:
            print(f"⚠️  数据保存失败: {e}")
    
    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'config': config,
        'preprocessed_data': preprocessed_data
    }


# 使用示例和测试

def test_firetracks_pipeline():
    """
    测试FireTracks数据处理流水线（像素级二分类任务）
    
    这是一个完整的使用示例，展示如何使用create_pixel_level_binary_classification_dataset函数
    处理FireTracks数据并创建可用于ConvLSTM训练的像素级二分类数据集。
    
    Returns:
        dict: 处理结果字典，包含dataset、dataloader等，如果失败返回None
    """
    # 配置参数
    config = {
        'use_modis_api': True,  # 使用MODIS API（默认True）
        'gee_project': 'ee-tpan2203-wildfire',  # GEE项目名称
        'cache_dir': 'dataset',  # 缓存目录（默认dataset）
        'patch_size_km': 25,
        'time_steps': 10,
        'target_years': (2017, 2018),  # 训练年份
        'batch_size': 32,
        'neg_pos_ratio': 2.0,  # 负样本是正样本的2倍
        'max_samples': 10  # 限制处理10个严重火灾组件用于测试（设为None处理全部）
    }
    
    # 运行处理流水线
    # 主要处理函数：像素级二分类任务
    result = create_pixel_level_binary_classification_dataset(
        data_directory='dataset/firetracks_data',  # 你的FireTracks数据路径
        output_path='dataset/processed_firetracks_pixel_binary.pth',  # 处理后的数据保存路径.pth
        config=config
    )
    
    if result is not None:
        # 测试数据加载器
        dataloader = result['dataloader']
        features, targets = next(iter(dataloader))
        
        print(f"\n📊 数据加载器测试:")
        print(f"   - Batch大小: {features.shape[0]}")
        print(f"   - 输入维度: {features.shape}")  # [batch, channels, timesteps, height, width]
        print(f"   - 目标维度: {targets.shape}")
        print(f"   - 目标类型: 二分类 (0=负样本, 1=正样本)")
        print(f"   - 数据类型: {features.dtype}")
        
        # 统计批次中的正负样本数量
        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()
        print(f"   - 批次中正样本数: {positive_count}")
        print(f"   - 批次中负样本数: {negative_count}")
        
        return result
    else:
        print("❌ 数据处理失败")
        return None


# ============================================================================
# 像素级二分类任务：预测某像素点是否会在未来一天发生大火
# ============================================================================

def _process_single_pixel(pixel_event, ignition_date, component_id, time_window_start, 
                          time_window_end, patch_radius_deg, grid_size, events, 
                          pixel_lc_dict, use_modis_api, use_events_lc, events_lc,
                          use_index_matching, data_dir, project):
    """
    处理单个像素的特征提取（用于并行处理）
    
    Returns:
        dict: 样本字典，如果失败返回None
    """
    try:
        pixel_lat = pixel_event['lat']
        pixel_lon = pixel_event['lon']
        pixel_idx = pixel_event.name
        
        # 以该像素为中心计算空间边界（25km×25km区块）
        pixel_spatial_bounds = {
            'lat_min': pixel_lat - patch_radius_deg,
            'lat_max': pixel_lat + patch_radius_deg,
            'lon_min': pixel_lon - patch_radius_deg,
            'lon_max': pixel_lon + patch_radius_deg
        }
        
        # 提取该像素周围时间窗口内的事件（用于构建特征）
        spatial_mask = (
            (events['lat'] >= pixel_spatial_bounds['lat_min']) & 
            (events['lat'] <= pixel_spatial_bounds['lat_max']) &
            (events['lon'] >= pixel_spatial_bounds['lon_min']) & 
            (events['lon'] <= pixel_spatial_bounds['lon_max'])
        )
        temporal_mask = (
            (events['dtime'] >= time_window_start) & 
            (events['dtime'] <= time_window_end)
        )
        window_events = events[spatial_mask & temporal_mask].copy()
        
        # 为该像素创建特征立方体（以该像素为中心）
        if extract_aligned_features is None:
            error_msg = (
                "❌ extract_aligned_features 未导入成功，无法生成特征。\n"
                "   这通常是由于DLL加载失败（Windows常见问题）或缺少依赖库。\n"
                "   请检查：\n"
                "   1. 是否安装了所有必需的依赖（numpy, scipy, xarray等）\n"
                "   2. 是否安装了Visual C++运行时库\n"
                "   3. 如果只是加载已生成的数据，可以忽略此错误\n"
                "   文件位置: code/fire_equality/datamodules/feature_alignment.py"
            )
            raise ImportError(error_msg)
        # 提取ISO3国家代码（如果可用）
        iso3_code = extract_iso3_from_events(window_events, pixel_spatial_bounds, events)
        feature_cube = extract_aligned_features(
            spatial_bounds=pixel_spatial_bounds,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            grid_size=grid_size,
            fire_date=pd.Timestamp(ignition_date).to_pydatetime(),
            fire_year=pd.Timestamp(ignition_date).year,
            iso3=iso3_code,
            data_dir=data_dir,
            project=project
        )
        
        # 获取土地覆盖类型
        land_cover = None
        
        # 优先使用MODIS API获取的结果
        if use_modis_api:
            if pixel_idx in pixel_lc_dict:
                lc = pixel_lc_dict[pixel_idx]
                if lc is not None and lc != 255:  # 255是未分类
                    land_cover = lc
        # 如果use_modis_api=False，使用events_lc方法
        elif use_events_lc and events_lc is not None:
            if use_index_matching:
                if pixel_idx in events_lc.index:
                    lc_row = events_lc.loc[pixel_idx]
                    lc_values = [lc_row.get(f'lc{i}', None) for i in range(1, 5)]
                    lc_values = [lc for lc in lc_values if lc is not None and lc > 0]
                    if lc_values:
                        from collections import Counter
                        land_cover = Counter(lc_values).most_common(1)[0][0]
            else:
                tolerance = 0.005
                lc_match = events_lc[
                    (events_lc['dtime'].dt.date == ignition_date) &
                    (np.abs(events_lc['lat'] - pixel_lat) < tolerance) &
                    (np.abs(events_lc['lon'] - pixel_lon) < tolerance)
                ]
                if len(lc_match) > 0:
                    if len(lc_match) > 1:
                        distances = np.sqrt(
                            (lc_match['lat'] - pixel_lat)**2 + 
                            (lc_match['lon'] - pixel_lon)**2
                        )
                        lc_match = lc_match.iloc[[distances.idxmin()]]
                    lc_values = [lc_match.iloc[0].get(f'lc{i}', None) for i in range(1, 5)]
                    lc_values = [lc for lc in lc_values if lc is not None and lc > 0]
                    if lc_values:
                        from collections import Counter
                        land_cover = Counter(lc_values).most_common(1)[0][0]
        
        sample = {
            'pixel_lat': pixel_lat,
            'pixel_lon': pixel_lon,
            'pixel_date': ignition_date,
            'features': feature_cube,  # [time_steps, grid_size, grid_size, channels]
            'target': 1,  # 正样本
            'land_cover': land_cover,
                'metadata': {
                    'component_id': component_id,
                    'center_lat': pixel_lat,
                    'center_lon': pixel_lon,
                    'spatial_bounds': pixel_spatial_bounds,
                    'time_window': (time_window_start, time_window_end),
                    'iso3': iso3_code
                }
            }
        return sample
        
    except Exception as e:
        # 返回错误信息，而不是抛出异常
        return {'error': str(e), 'pixel_idx': pixel_idx}


def create_pixel_level_positive_samples(preprocessed_data, patch_size_km=25, time_steps=10, 
                                        spatial_resolution_km=1, max_samples=None, target_year_range=None,
                                        use_modis_api=True, project='ee-tpan2203-wildfire',
                                        max_consecutive_errors=5, parallel_workers=2):
    """
    创建像素级正样本：从严重火灾组件中提取起火当天的像素点
    
    任务定义：预测某像素点是否会在未来一天发生大火
    正样本（y_t = 1）：所有在当天起火并最终形成大火的像素
    
    Args:
        preprocessed_data: 预处理后的数据（必须包含severe_components和events）
        patch_size_km: 空间块大小 (km)
        time_steps: 时间步长 (天)，火灾发生前的时间窗口
        spatial_resolution_km: 空间分辨率 (km)
        max_samples: 最大处理样本数（用于测试，None表示处理全部）
        target_year_range: 目标年份范围 (start_year, end_year)，用于限制时间窗口不早于开始年份
        use_modis_api: 是否使用MODIS API获取土地覆盖类型（默认True，更快）
        project: GEE项目名称（默认'ee-tpan2203-wildfire'）
        max_consecutive_errors: 最大连续错误数，超过此数量将停止流水线（默认5）
    
    Returns:
        list: 像素级正样本列表，每个元素包含：
            - pixel_lat: 像素纬度
            - pixel_lon: 像素经度
            - pixel_date: 起火日期
            - features: 特征立方体 [time_steps, grid_size, grid_size, channels]
            - target: 1 (正样本)
            - land_cover: 土地覆盖类型（如果有）
            - metadata: 元数据字典
    """
    print("\n" + "="*60)
    print("创建像素级正样本...")
    print("="*60)
    
    if 'severe_components' not in preprocessed_data:
        raise ValueError("预处理数据中必须包含 'severe_components'")
    
    severe_components = preprocessed_data['severe_components']
    events = preprocessed_data['events']
    events_lc = preprocessed_data.get('events_lc', None)
    
    # 检查events是否有iso3列（用于GDP匹配）
    has_iso3_column = 'iso3' in events.columns if events is not None and len(events) > 0 else False
    if not has_iso3_column:
        # 尝试从country列添加iso3列
        if events is not None and len(events) > 0 and 'country' in events.columns:
            print(f"   💡 events数据没有iso3列，尝试从country列添加...")
            add_iso3_to_dataframe(events, country_column='country')
            has_iso3_column = 'iso3' in events.columns
    
    if not has_iso3_column:
        print(f"   ⚠️  注意：events数据没有iso3列，GDP特征将使用默认值0")
        print(f"      建议：如果FireTracks数据包含国家信息，可以添加iso3列以启用GDP匹配")
    else:
        # 检查iso3列的有效性
        iso3_valid = events['iso3'].dropna()
        iso3_valid = iso3_valid[iso3_valid != '']
        if len(iso3_valid) > 0:
            print(f"   ✅ events数据有iso3列，包含 {len(iso3_valid)} / {len(events)} 个有效值")
        else:
            print(f"   ⚠️  events数据有iso3列，但所有值都为空，GDP特征将使用默认值0")
    
    # 转换日期格式
    severe_components['dtime_min'] = pd.to_datetime(severe_components['dtime_min'])
    events['dtime'] = pd.to_datetime(events['dtime'])
    if events_lc is not None:
        events_lc['dtime'] = pd.to_datetime(events_lc['dtime'])
    
    # 确定年份（用于MODIS查询）
    if target_year_range:
        year = target_year_range[0]  # 使用起始年份
    else:
        # 从组件日期推断年份
        year = severe_components['dtime_min'].dt.year.iloc[0] if len(severe_components) > 0 else 2017
    
    # 检查是否使用本地 landcover 文件
    use_index_matching = False  # 初始化变量
    if use_modis_api and HAS_MODIS_LC:
        print(f"   🚀 使用本地 landcover 文件获取土地覆盖类型（dataset/LandCover/）")
        use_events_lc = False
    else:
        print(f"   🔄 使用events_lc获取土地覆盖类型")
        use_events_lc = True
        # 检查events和events_lc的行索引是否一致（用于精确匹配）
        if events_lc is not None:
            if len(events) == len(events_lc):
                # 检查索引是否一致（至少前1000行作为样本检查）
                sample_size = min(1000, len(events))
                if events.index[:sample_size].equals(events_lc.index[:sample_size]):
                    use_index_matching = True
                    print(f"   ✅ 检测到events和events_lc行索引一致，将使用索引匹配")
                else:
                    print(f"   ⚠️  events和events_lc行索引不一致，将使用位置匹配")
            else:
                print(f"   ⚠️  events和events_lc行数不一致 ({len(events)} vs {len(events_lc)})，将使用位置匹配")
    
    total_components = len(severe_components)
    # 如果设置了max_samples，限制的是成功创建的正样本数量，而不是处理的组件数量
    # 这样即使前几个组件被跳过，也会继续处理后续组件
    target_samples = max_samples if max_samples else None
    
    # 如果设置了target_year_range和max_samples，按年份均匀采样
    if target_samples and target_year_range:
        start_year, end_year = target_year_range
        num_years = end_year - start_year + 1
        samples_per_year = max(1, target_samples // num_years)  # 每个年份至少1个样本
        print(f"⚠️  限制正样本数量: 最多 {target_samples:,} 个正样本")
        print(f"   按年份均匀采样: {num_years} 个年份，每个年份约 {samples_per_year:,} 个样本")
        
        # 按年份分组组件
        severe_components['year'] = severe_components['dtime_min'].dt.year
        components_by_year = {}
        for year in range(start_year, end_year + 1):
            year_components = severe_components[severe_components['year'] == year]
            if len(year_components) > 0:
                components_by_year[year] = year_components
                print(f"   年份 {year}: {len(year_components):,} 个组件")
        
        # 计算每个年份的目标样本数（均匀分配）
        year_targets = {}
        remaining_samples = target_samples
        for year in sorted(components_by_year.keys()):
            if year == sorted(components_by_year.keys())[-1]:
                # 最后一个年份分配剩余的所有样本
                year_targets[year] = remaining_samples
            else:
                year_targets[year] = samples_per_year
                remaining_samples -= samples_per_year
        
        print(f"   各年份目标样本数: {year_targets}")
    else:
        components_by_year = None
        year_targets = None
    
    if target_samples and not year_targets:
        print(f"⚠️  限制正样本数量: 最多 {target_samples:,} 个正样本（会处理足够多的组件以达到目标）")
    
    print(f"待处理严重火灾组件: {total_components:,} 个")
    if parallel_workers > 1:
        print(f"   🚀 启用并行处理: {parallel_workers} 个工作线程")
    
    positive_samples = []
    grid_size = int(patch_size_km / spatial_resolution_km)
    
    progress_interval = max(1, total_components // 100)
    processed_count = 0
    skipped_count = 0
    skipped_reasons = {}  # 统计跳过原因
    consecutive_errors = 0  # 连续错误计数
    
    # 导入并行处理模块
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # 线程安全的锁，用于保护共享变量
    samples_lock = threading.Lock()
    error_lock = threading.Lock()
    
    # 如果按年份均匀采样，先按年份迭代
    if components_by_year and year_targets:
        for year in sorted(components_by_year.keys()):
            year_components = components_by_year[year]
            year_target = year_targets[year]
            
            if len(positive_samples) >= target_samples:
                break
            
            print(f"\n处理年份 {year} (目标: {year_target:,} 个样本)...")
            
            for idx, component in year_components.iterrows():
                # 如果已经达到该年份的目标样本数，停止处理该年份
                year_current_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
                if year_current_count >= year_target:
                    break
                
                # 如果已经达到总目标样本数，停止处理
                if target_samples and len(positive_samples) >= target_samples:
                    break
                
                # 继续处理组件（使用原来的逻辑）
                processed_count += 1
                
                if processed_count % progress_interval == 0 or processed_count == 1:
                    progress = processed_count / total_components * 100
                    target_info = f"（总目标: {target_samples}, 年份 {year} 目标: {year_target}）" if target_samples else ""
                    print(f"\r   处理进度: {processed_count:,}/{total_components:,} ({progress:.1f}%) - 已创建 {len(positive_samples):,} 个正样本{target_info} (跳过 {skipped_count:,} 个)", end='', flush=True)
                
                component_id = component['cp']
                start_date = component['dtime_min']
                center_lat = component['lat_mean']
                center_lon = component['lon_mean']
                
                # 时间窗口：火灾发生前的时间
                time_window_start = start_date - pd.Timedelta(days=time_steps)
                skipped_reason = None
                if target_year_range is not None:
                    start_year_check, _ = target_year_range
                    year_start_date = pd.Timestamp(f'{start_year_check}-01-01')
                    if time_window_start < year_start_date:
                        skipped_reason = "时间窗口不完整"
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                        continue
                
                time_window_end = start_date - pd.Timedelta(days=1)
                ignition_date = start_date.date()
                
                # 提取起火当天的事件
                search_radius_deg = 0.5
                search_bounds = {
                    'lat_min': center_lat - search_radius_deg,
                    'lat_max': center_lat + search_radius_deg,
                    'lon_min': center_lon - search_radius_deg,
                    'lon_max': center_lon + search_radius_deg
                }
                
                ignition_events = events[
                    (events['lat'] >= search_bounds['lat_min']) & 
                    (events['lat'] <= search_bounds['lat_max']) &
                    (events['lon'] >= search_bounds['lon_min']) & 
                    (events['lon'] <= search_bounds['lon_max']) &
                    (events['dtime'].dt.date == ignition_date) &
                    (events['cp'] == component_id)
                ].copy()
                
                if len(ignition_events) == 0:
                    skipped_reason = "未找到起火当天的事件"
                    skipped_count += 1
                    if skipped_reason not in skipped_reasons:
                        skipped_reasons[skipped_reason] = 0
                    skipped_reasons[skipped_reason] += 1
                    continue
                
                # 为每个起火像素创建正样本
                lat_per_km = 1 / 110.574
                patch_radius_deg = (patch_size_km / 2) * lat_per_km
                
                # 获取土地覆盖类型
                pixel_coords = []
                pixel_indices = []
                if use_modis_api:
                    if not HAS_RASTERIO:
                        error_msg = f"❌ 无法读取本地 landcover 数据！\n"
                        error_msg += f"   原因: rasterio 未正确安装或DLL加载失败\n"
                        error_msg += f"   请检查: pip install rasterio 或检查系统依赖\n"
                        error_msg += f"   流水线已终止，不会使用events_lc作为回退方案"
                        raise ImportError(error_msg)
                    
                    for idx_pixel, pixel_event in ignition_events.iterrows():
                        pixel_coords.append((pixel_event['lat'], pixel_event['lon']))
                        pixel_indices.append(idx_pixel)
                    
                    if len(pixel_coords) > 0:
                        lc_results = get_landcover_batch_gee(pixel_coords, year, lc_type='LC_Type1', 
                                                             batch_size=1000, project=project)
                        pixel_lc_dict = {pixel_indices[i]: lc for i, lc in enumerate(lc_results)}
                    else:
                        pixel_lc_dict = {}
                else:
                    pixel_lc_dict = {}
                
                # 处理每个像素（简化版，使用顺序处理）
                for idx_pixel, pixel_event in ignition_events.iterrows():
                    # 检查是否达到年份目标
                    year_current_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
                    if year_current_count >= year_target:
                        break
                    
                    # 检查是否达到总目标
                    if target_samples and len(positive_samples) >= target_samples:
                        break
                    
                    result = _process_single_pixel(
                        pixel_event, ignition_date, component_id,
                        time_window_start, time_window_end, patch_radius_deg,
                        grid_size, events, pixel_lc_dict, use_modis_api,
                        use_events_lc, events_lc, use_index_matching,
                        'dataset', project
                    )
                    
                    if 'error' in result:
                        consecutive_errors += 1
                        skipped_reason = "特征提取失败"
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                    else:
                        positive_samples.append(result)
                        consecutive_errors = 0
            
            year_final_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
            print(f"\n   年份 {year} 完成: {year_final_count:,}/{year_target:,} 个样本")
    
    # 如果不按年份均匀采样，使用原来的逻辑
    else:
        for idx, component in severe_components.iterrows():
            # 如果已经达到目标样本数，停止处理
            if target_samples and len(positive_samples) >= target_samples:
                break
            
            processed_count += 1
        
            if processed_count % progress_interval == 0 or processed_count == 1:
                progress = processed_count / total_components * 100
                target_info = f"（目标: {target_samples}）" if target_samples else ""
                # 使用\r清除当前行，避免被其他输出打断
                print(f"\r   处理进度: {processed_count:,}/{total_components:,} ({progress:.1f}%) - 已创建 {len(positive_samples):,} 个正样本{target_info} (跳过 {skipped_count:,} 个)", end='', flush=True)
            
            component_id = component['cp']
            start_date = component['dtime_min']
            center_lat = component['lat_mean']
            center_lon = component['lon_mean']
            
            # 时间窗口：火灾发生前的时间
            # 如果提供了target_year_range，确保时间窗口不早于目标年份的开始日期
            time_window_start = start_date - pd.Timedelta(days=time_steps)
            skipped_reason = None
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                # 如果计算出的时间窗口开始日期早于目标年份开始日期，则跳过这个组件
                # 因为无法提供完整的时间窗口数据
                if time_window_start < year_start_date:
                    # 跳过需要补零的样本（时间窗口不完整）
                    skipped_reason = "时间窗口不完整"
                    skipped_count += 1
                    if skipped_reason not in skipped_reasons:
                        skipped_reasons[skipped_reason] = 0
                    skipped_reasons[skipped_reason] += 1
                    continue
            else:
                # 如果没有年份限制，使用完整的时间窗口
                pass
            
            time_window_end = start_date - pd.Timedelta(days=1)
            ignition_date = start_date.date()
            
            # 提取起火当天的事件（这些是正样本像素）
            # 使用组件的空间范围作为初始筛选（但后续会为每个像素单独创建特征立方体）
            # 为了找到属于该组件的所有起火像素，我们需要一个较大的搜索范围
            # 使用组件的中心点加上一个较大的半径来搜索
            search_radius_deg = 0.5  # 约55km的搜索半径，确保能找到所有相关像素
            search_bounds = {
                'lat_min': center_lat - search_radius_deg,
                'lat_max': center_lat + search_radius_deg,
                'lon_min': center_lon - search_radius_deg,
                'lon_max': center_lon + search_radius_deg
            }
            
            ignition_events = events[
                (events['lat'] >= search_bounds['lat_min']) & 
                (events['lat'] <= search_bounds['lat_max']) &
                (events['lon'] >= search_bounds['lon_min']) & 
                (events['lon'] <= search_bounds['lon_max']) &
                (events['dtime'].dt.date == ignition_date) &
                (events['cp'] == component_id)  # 确保属于这个组件
            ].copy()
            
            if len(ignition_events) == 0:
                # 如果没有找到起火当天的事件，跳过这个组件
                skipped_reason = "未找到起火当天的事件"
                skipped_count += 1
                if skipped_reason not in skipped_reasons:
                    skipped_reasons[skipped_reason] = 0
                skipped_reasons[skipped_reason] += 1
                continue
            
            # 为每个起火像素创建正样本（每个像素单独创建特征立方体）
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            # 如果使用MODIS API，批量获取所有像素的土地覆盖类型
            pixel_coords = []
            pixel_indices = []
            if use_modis_api:
                if not HAS_RASTERIO:
                    error_msg = f"❌ 无法读取本地 landcover 数据！\n"
                    error_msg += f"   原因: rasterio 未正确安装或DLL加载失败\n"
                    error_msg += f"   请检查: pip install rasterio 或检查系统依赖\n"
                    error_msg += f"   流水线已终止，不会使用events_lc作为回退方案"
                    raise ImportError(error_msg)
                
                # 获取年份用于landcover查询
                year = start_date.year
                
                for idx, pixel_event in ignition_events.iterrows():
                    pixel_coords.append((pixel_event['lat'], pixel_event['lon']))
                    pixel_indices.append(idx)
                
                if len(pixel_coords) > 0:
                    print(f"      批量查询 {len(pixel_coords)} 个像素的土地覆盖类型（从本地文件）...", end='\r')
                    lc_results = get_landcover_batch_gee(pixel_coords, year, lc_type='LC_Type1', 
                                                         batch_size=1000, project=project)
                    pixel_lc_dict = {pixel_indices[i]: lc for i, lc in enumerate(lc_results)}
                    success_count = sum(1 for lc in lc_results if lc is not None and lc != 255)
                    print(f"      批量查询完成，成功获取 {success_count}/{len(pixel_coords)} 个像素的土地覆盖类型")
                    
                    if success_count == 0:
                        local_file = os.path.join('dataset', 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
                        error_msg = f"❌ 未能从本地文件获取到任何土地覆盖类型！\n"
                        error_msg += f"   查询了 {len(pixel_coords)} 个像素，但未获取到有效结果\n"
                        error_msg += f"   请检查:\n"
                        error_msg += f"   1. 本地文件是否存在: {local_file}\n"
                        error_msg += f"   2. 文件是否损坏\n"
                        error_msg += f"   3. 坐标是否在研究区域内\n"
                        error_msg += f"   流水线已终止"
                        raise RuntimeError(error_msg)
                else:
                    pixel_lc_dict = {}
            else:
                pixel_lc_dict = {}
            
            # 使用并行处理或顺序处理
            if parallel_workers > 1 and len(ignition_events) > 1:
                
                # 并行处理多个像素
                import time
                pixel_start_time = time.time()
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    futures = []
                    for idx, pixel_event in ignition_events.iterrows():
                        # 在提交任务前检查是否已经达到目标
                        with samples_lock:
                            if target_samples and len(positive_samples) >= target_samples:
                                # 已经达到目标，取消剩余任务
                                break
                        future = executor.submit(
                            _process_single_pixel,
                            pixel_event, ignition_date, component_id,
                            time_window_start, time_window_end, patch_radius_deg,
                            grid_size, events, pixel_lc_dict, use_modis_api,
                            use_events_lc, events_lc, use_index_matching,
                            'dataset', project
                        )
                        futures.append(future)
                    
                    # 收集结果
                    completed_count = 0
                    for future in as_completed(futures):
                        # 在收集结果前再次检查是否已经达到目标
                        with samples_lock:
                            if target_samples and len(positive_samples) >= target_samples:
                                # 已经达到目标，取消剩余任务
                                for remaining_future in futures:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                        
                        result = future.result()
                        completed_count += 1
                        
                        with samples_lock:
                            if 'error' in result:
                                # 处理失败
                                error_msg = result['error']
                                pixel_idx = result.get('pixel_idx', 'unknown')
                                with error_lock:
                                    consecutive_errors += 1
                                    skipped_reason = "特征提取失败"
                                    skipped_count += 1
                                    if skipped_reason not in skipped_reasons:
                                        skipped_reasons[skipped_reason] = 0
                                    skipped_reasons[skipped_reason] += 1
                                
                                # 打印错误信息（限制频率）
                                if len(positive_samples) < 3 or consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                                    print(f"    ⚠️  像素 {pixel_idx} 特征提取失败 ({consecutive_errors}/{max_consecutive_errors}): {error_msg[:150]}")
                            else:
                                # 成功处理
                                # 检查是否超过目标样本数（并行处理时可能出现超限）
                                if target_samples and len(positive_samples) >= target_samples:
                                    # 已经达到目标，跳过这个结果
                                    continue
                                positive_samples.append(result)
                                # 再次检查是否达到目标，如果达到则取消剩余任务
                                if target_samples and len(positive_samples) >= target_samples:
                                    for remaining_future in futures:
                                        if not remaining_future.done():
                                            remaining_future.cancel()
                        with error_lock:
                            consecutive_errors = 0
                        
                        # 性能监控：每10个样本显示一次统计
                        if completed_count % 10 == 0 or completed_count == len(futures):
                                    pixel_elapsed = time.time() - pixel_start_time
                                    avg_time_per_pixel = pixel_elapsed / completed_count if completed_count > 0 else 0
                                    try:
                                        from .feature_cache import get_cache_stats
                                        from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
                                        cache_stats = get_cache_stats()
                                        netcdf_stats = get_netcdf_cache_stats()
                                        if cache_stats['hits'] + cache_stats['misses'] > 0:
                                            cache_info = f", 缓存命中率: {cache_stats['hit_rate']:.1%}"
                                        else:
                                            cache_info = ""
                                        if netcdf_stats['cached_datasets'] > 0:
                                            netcdf_info = f", NetCDF缓存: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']}"
                                        else:
                                            netcdf_info = ""
                                    except:
                                        cache_info = ""
                                        netcdf_info = ""
                                    
                                    # 只在进度更新时显示（避免干扰进度条）
                                    if processed_count % progress_interval == 0:
                                        print(f"\r   处理进度: {processed_count:,}/{total_components:,} - 已创建 {len(positive_samples):,} 个正样本 - 平均 {avg_time_per_pixel:.1f}秒/样本{cache_info}{netcdf_info}", end='', flush=True)
            else:
                # 顺序处理（原始方式，用于调试或单线程）
                import time
                pixel_start_time = time.time()
                pixel_count = 0
                for idx, pixel_event in ignition_events.iterrows():
                    pixel_count += 1
                    result = _process_single_pixel(
                        pixel_event, ignition_date, component_id,
                        time_window_start, time_window_end, patch_radius_deg,
                        grid_size, events, pixel_lc_dict, use_modis_api,
                        use_events_lc, events_lc, use_index_matching,
                        'dataset', project
                    )
                    
                    if 'error' in result:
                        # 处理失败
                        error_msg = result['error']
                        pixel_idx = result.get('pixel_idx', idx)
                        consecutive_errors += 1
                        
                        try:
                            coord_str = f"({pixel_event['lat']:.4f}, {pixel_event['lon']:.4f})"
                        except:
                            coord_str = f"(索引: {pixel_idx})"
                        
                        if len(positive_samples) < 3 or consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                            print(f"    ⚠️  像素 {coord_str} 特征提取失败 ({consecutive_errors}/{max_consecutive_errors}): {error_msg[:150]}")
                        
                        skipped_reason = "特征提取失败"
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                        
                        # 检查是否应该停止
                        if consecutive_errors >= max_consecutive_errors and len(positive_samples) > 0:
                            error_summary = f"\n❌ 连续 {consecutive_errors} 次特征提取失败，停止流水线以便调试\n"
                            error_summary += f"   已成功创建 {len(positive_samples):,} 个正样本\n"
                            error_summary += f"   总跳过数: {skipped_count:,} 个\n"
                            error_summary += f"   最后失败的像素: {coord_str}\n"
                            error_summary += f"   最后错误信息: {error_msg[:200]}"
                            raise RuntimeError(error_summary)
                        elif consecutive_errors >= max_consecutive_errors * 2 and len(positive_samples) == 0:
                            error_summary = f"\n❌ 连续 {consecutive_errors} 次特征提取失败，且未创建任何正样本\n"
                            error_summary += f"   总跳过数: {skipped_count:,} 个\n"
                            error_summary += f"   最后失败的像素: {coord_str}\n"
                            error_summary += f"   最后错误信息: {error_msg[:200]}\n"
                            error_summary += f"   请检查数据文件和网络连接"
                            raise RuntimeError(error_summary)
                    else:
                        # 成功处理
                        # 检查是否超过目标样本数
                        if target_samples and len(positive_samples) >= target_samples:
                            # 已经达到目标，跳过这个结果
                            break
                        positive_samples.append(result)
                        consecutive_errors = 0
                        
                        # 性能监控：每10个样本显示一次统计
                        if pixel_count % 10 == 0 or pixel_count == len(ignition_events):
                            pixel_elapsed = time.time() - pixel_start_time
                            avg_time_per_pixel = pixel_elapsed / pixel_count if pixel_count > 0 else 0
                            try:
                                from .feature_cache import get_cache_stats
                                from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
                                cache_stats = get_cache_stats()
                                netcdf_stats = get_netcdf_cache_stats()
                                if cache_stats['hits'] + cache_stats['misses'] > 0:
                                    cache_info = f", 缓存命中率: {cache_stats['hit_rate']:.1%}"
                                else:
                                    cache_info = ""
                                if netcdf_stats['cached_datasets'] > 0:
                                    netcdf_info = f", NetCDF缓存: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']}"
                                else:
                                    netcdf_info = ""
                            except:
                                cache_info = ""
                                netcdf_info = ""
                            
                            # 只在进度更新时显示
                            if processed_count % progress_interval == 0:
                                print(f"\r   处理进度: {processed_count:,}/{total_components:,} - 已创建 {len(positive_samples):,} 个正样本 - 平均 {avg_time_per_pixel:.1f}秒/样本{cache_info}{netcdf_info}", end='', flush=True)
    
    # 换行并清除进度行
    print("\r" + " " * 100 + "\r", end='', flush=True)  # 清除进度行
    print(f"✅ 成功创建 {len(positive_samples):,} 个像素级正样本")
    
    # 显示年份分布统计
    if target_year_range and len(positive_samples) > 0:
        from collections import Counter
        year_counts = Counter()
        for sample in positive_samples:
            if 'pixel_date' in sample:
                try:
                    year = pd.Timestamp(sample['pixel_date']).year
                    year_counts[year] += 1
                except:
                    pass
        
        if year_counts:
            print(f"\n📊 各年份样本分布:")
            total_shown = 0
            for year in sorted(year_counts.keys()):
                count = year_counts[year]
                total_shown += count
                percentage = (count / len(positive_samples) * 100) if len(positive_samples) > 0 else 0
                print(f"   {year}: {count:,} 个 ({percentage:.1f}%)")
            if total_shown < len(positive_samples):
                print(f"   其他: {len(positive_samples) - total_shown:,} 个")
    
    if skipped_count > 0:
        print(f"\n⚠️  跳过了 {skipped_count:,} 个组件，原因统计:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {reason}: {count:,} 个")
    
    return positive_samples


def save_positive_samples(positive_samples, config, filepath):
    """
    保存正样本到文件
    
    Args:
        positive_samples: 正样本列表
        config: 配置字典（用于验证）
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 保存数据（不保存numpy数组，因为torch.save可以处理）
    # 注意：将 datetime.date 对象转换为字符串，避免 PyTorch 2.6+ 的加载问题
    import datetime
    positive_samples_serializable = []
    for sample in positive_samples:
        sample_copy = sample.copy()
        # 转换 datetime.date 对象为字符串
        if 'date' in sample_copy and isinstance(sample_copy['date'], datetime.date):
            sample_copy['date'] = sample_copy['date'].isoformat()
        positive_samples_serializable.append(sample_copy)
    
    save_data = {
        'positive_samples': positive_samples_serializable,
        'config': config,
        'metadata': {
            'total_samples': len(positive_samples),
            'samples_with_lc': sum(1 for s in positive_samples if s.get('land_cover') is not None),
            'saved_time': pd.Timestamp.now().isoformat()
        }
    }
    
    torch.save(save_data, filepath)
    print(f"✅ 正样本已保存到: {filepath}")
    print(f"   - 总样本数: {len(positive_samples):,}")
    print(f"   - 有土地覆盖信息的样本: {save_data['metadata']['samples_with_lc']:,}")


def load_positive_samples(filepath, verify_config=None):
    """
    从文件加载正样本
    
    Args:
        filepath: 文件路径
        verify_config: 可选的配置字典，用于验证（如果提供，会检查配置是否匹配）
    
    Returns:
        tuple: (positive_samples, config) 如果成功，否则返回 (None, None)
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {filepath}")
            return None, None
        
        print(f"📂 从文件加载正样本: {filepath}")
        # PyTorch 2.6+ 默认 weights_only=True，需要允许 datetime.date
        import datetime
        try:
            # 尝试使用安全模式加载（PyTorch 2.6+）
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            # 如果失败，尝试添加安全全局变量
            try:
                torch.serialization.add_safe_globals([datetime.date])
                save_data = torch.load(filepath, map_location='cpu')
            except:
                # 最后回退到 weights_only=False
                save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        positive_samples = save_data.get('positive_samples', [])
        config = save_data.get('config', {})
        metadata = save_data.get('metadata', {})
        
        # 恢复 datetime.date 对象（如果被保存为字符串）
        import datetime
        for sample in positive_samples:
            if 'date' in sample and isinstance(sample['date'], str):
                try:
                    sample['date'] = datetime.date.fromisoformat(sample['date'])
                except:
                    pass  # 如果转换失败，保持原样
        
        print(f"✅ 成功加载正样本:")
        print(f"   - 总样本数: {len(positive_samples):,}")
        print(f"   - 有土地覆盖信息的样本: {metadata.get('samples_with_lc', 'N/A')}")
        print(f"   - 保存时间: {metadata.get('saved_time', 'N/A')}")
        
        # 验证配置（如果提供）
        if verify_config is not None:
            key_params = ['patch_size_km', 'time_steps', 'target_years']
            mismatches = []
            for key in key_params:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        mismatches.append(f"{key}: {verify_config[key]} != {config[key]}")
            
            # 检查max_samples：如果当前请求设置了max_samples，但缓存中的样本数量超过了限制
            # 应该重新生成或截断缓存
            if 'max_samples' in verify_config and verify_config['max_samples'] is not None:
                max_samples = verify_config['max_samples']
                if len(positive_samples) > max_samples:
                    print(f"⚠️  缓存中的样本数量 ({len(positive_samples):,}) 超过了请求的限制 ({max_samples:,})")
                    print(f"   将截断缓存样本到 {max_samples:,} 个")
                    positive_samples = positive_samples[:max_samples]
                    print(f"   ✅ 已截断到 {len(positive_samples):,} 个样本")
            
            if mismatches:
                print(f"⚠️  配置不匹配:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                print(f"   建议：删除缓存文件并重新生成")
                return None, None
            else:
                print(f"   ✅ 配置验证通过")
        
        return positive_samples, config
        
    except Exception as e:
        print(f"❌ 加载正样本失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_negative_pool(negative_pool, config, filepath):
    """
    保存负样本池到文件
    
    Args:
        negative_pool: 负样本池字典
        config: 配置字典（用于验证）
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 准备可序列化的负样本池
    # 注意：events数据可能很大，但我们保存它以便完整恢复
    # 将 datetime.date 对象转换为字符串
    import datetime
    negative_pool_serializable = negative_pool.copy()
    
    # 处理 patch_candidates 中的日期
    if 'patch_candidates' in negative_pool_serializable:
        patch_candidates = negative_pool_serializable['patch_candidates']
        for patch_key, patch_info in patch_candidates.items():
            if 'date' in patch_info and isinstance(patch_info['date'], datetime.date):
                patch_info['date'] = patch_info['date'].isoformat()
    
    # 处理 all_fire_dates 中的日期
    if 'all_fire_dates' in negative_pool_serializable:
        all_fire_dates = negative_pool_serializable['all_fire_dates']
        if isinstance(all_fire_dates, set):
            # 将 set 转换为列表以便序列化
            negative_pool_serializable['all_fire_dates'] = [
                d.isoformat() if isinstance(d, datetime.date) else d 
                for d in all_fire_dates
            ]
    
    save_data = {
        'negative_pool': negative_pool_serializable,
        'config': config,
        'metadata': {
            'total_candidates': len(negative_pool.get('patch_candidates', {})),
            'saved_time': pd.Timestamp.now().isoformat()
        }
    }
    
    torch.save(save_data, filepath)
    print(f"✅ 负样本池已保存到: {filepath}")
    print(f"   - 候选区块数: {save_data['metadata']['total_candidates']:,}")


def load_negative_pool(filepath, verify_config=None):
    """
    从文件加载负样本池
    
    Args:
        filepath: 文件路径
        verify_config: 可选的配置字典，用于验证（如果提供，会检查配置是否匹配）
    
    Returns:
        tuple: (negative_pool, config) 如果成功，否则返回 (None, None)
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {filepath}")
            return None, None
        
        print(f"📂 从文件加载负样本池: {filepath}")
        # PyTorch 2.6+ 默认 weights_only=True，需要允许 datetime.date
        import datetime
        try:
            # 尝试使用安全模式加载（PyTorch 2.6+）
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            # 如果失败，尝试添加安全全局变量
            try:
                torch.serialization.add_safe_globals([datetime.date])
                save_data = torch.load(filepath, map_location='cpu')
            except:
                # 最后回退到 weights_only=False
                save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        negative_pool = save_data.get('negative_pool', {})
        config = save_data.get('config', {})
        metadata = save_data.get('metadata', {})
        
        # 恢复 datetime.date 对象（如果被保存为字符串）
        import datetime
        # 恢复 patch_candidates 中的日期
        if 'patch_candidates' in negative_pool:
            patch_candidates = negative_pool['patch_candidates']
            for patch_key, patch_info in patch_candidates.items():
                if 'date' in patch_info and isinstance(patch_info['date'], str):
                    try:
                        patch_info['date'] = datetime.date.fromisoformat(patch_info['date'])
                    except:
                        pass  # 如果转换失败，保持原样
        
        # 恢复 all_fire_dates
        if 'all_fire_dates' in negative_pool:
            all_fire_dates = negative_pool['all_fire_dates']
            if isinstance(all_fire_dates, list):
                # 尝试将字符串转换回 datetime.date
                restored_dates = set()
                for d in all_fire_dates:
                    if isinstance(d, str):
                        try:
                            restored_dates.add(datetime.date.fromisoformat(d))
                        except:
                            restored_dates.add(d)
                    else:
                        restored_dates.add(d)
                negative_pool['all_fire_dates'] = restored_dates
        
        print(f"✅ 成功加载负样本池:")
        print(f"   - 候选区块数: {metadata.get('total_candidates', 'N/A'):,}")
        print(f"   - 保存时间: {metadata.get('saved_time', 'N/A')}")
        
        # 验证配置（如果提供）
        if verify_config is not None:
            key_params = ['patch_size_km', 'target_years', 'fast_mode']
            mismatches = []
            for key in key_params:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        mismatches.append(f"{key}: {verify_config[key]} != {config[key]}")
            
            if mismatches:
                print(f"⚠️  配置不匹配:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                print(f"   建议：删除缓存文件并重新生成")
                return None, None
            else:
                print(f"   ✅ 配置验证通过")
        
        return negative_pool, config
        
    except Exception as e:
        print(f"❌ 加载负样本池失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_negative_samples(negative_samples, config, filepath):
    """
    保存负样本到文件（用于断点恢复）
    
    Args:
        negative_samples: 负样本列表
        config: 配置字典（用于验证）
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 保存数据（将 datetime.date 对象转换为字符串）
    import datetime
    negative_samples_serializable = []
    for sample in negative_samples:
        sample_copy = sample.copy()
        # 转换 datetime.date 对象为字符串
        if 'pixel_date' in sample_copy and isinstance(sample_copy['pixel_date'], datetime.date):
            sample_copy['pixel_date'] = sample_copy['pixel_date'].isoformat()
        negative_samples_serializable.append(sample_copy)
    
    save_data = {
        'negative_samples': negative_samples_serializable,
        'config': config,
        'metadata': {
            'total_samples': len(negative_samples),
            'samples_by_lc': {lc: sum(1 for s in negative_samples if s.get('land_cover') == lc) 
                             for lc in set(s.get('land_cover') for s in negative_samples if s.get('land_cover') is not None)},
            'saved_time': pd.Timestamp.now().isoformat()
        }
    }
    
    torch.save(save_data, filepath)
    print(f"✅ 负样本已保存到: {filepath}")
    print(f"   - 总样本数: {len(negative_samples):,}")


def load_negative_samples(filepath, verify_config=None):
    """
    从文件加载负样本（用于断点恢复）
    
    Args:
        filepath: 文件路径
        verify_config: 可选的配置字典，用于验证
    
    Returns:
        tuple: (negative_samples, config) 如果成功，否则返回 (None, None)
    """
    try:
        if not os.path.exists(filepath):
            return None, None
        
        print(f"📂 从文件加载负样本: {filepath}")
        import datetime
        try:
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            try:
                torch.serialization.add_safe_globals([datetime.date])
                save_data = torch.load(filepath, map_location='cpu')
            except:
                save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        negative_samples = save_data.get('negative_samples', [])
        config = save_data.get('config', {})
        metadata = save_data.get('metadata', {})
        
        # 恢复 datetime.date 对象
        for sample in negative_samples:
            if 'pixel_date' in sample and isinstance(sample['pixel_date'], str):
                try:
                    sample['pixel_date'] = datetime.date.fromisoformat(sample['pixel_date'])
                except:
                    pass
        
        print(f"✅ 成功加载负样本:")
        print(f"   - 总样本数: {len(negative_samples):,}")
        if metadata.get('samples_by_lc'):
            print(f"   - 按土地覆盖类型分布:")
            for lc, count in sorted(metadata['samples_by_lc'].items()):
                print(f"     类型 {lc}: {count:,} 个")
        
        # 验证配置（如果提供）
        if verify_config:
            # 检查关键配置是否匹配
            key_configs = ['patch_size_km', 'time_steps', 'target_years']
            for key in key_configs:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        print(f"   ⚠️  配置不匹配: {key} (缓存: {config[key]}, 期望: {verify_config[key]})")
                        return None, None
        
        return negative_samples, config
    except Exception as e:
        print(f"❌ 加载负样本失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_positive_sample_land_cover_distribution(positive_samples):
    """
    分析正样本的土地覆盖分布
    
    Args:
        positive_samples: 正样本列表
    
    Returns:
        dict: 土地覆盖类型分布字典 {land_cover_type: count}
    """
    land_cover_counts = {}
    samples_without_lc = 0
    
    for sample in positive_samples:
        lc = sample.get('land_cover', None)
        if lc is not None:
            land_cover_counts[lc] = land_cover_counts.get(lc, 0) + 1
        else:
            samples_without_lc += 1
    
    total = len(positive_samples)
    if total > 0:
        land_cover_proportions = {lc: count / total for lc, count in land_cover_counts.items()}
    else:
        land_cover_proportions = {}
    
    print(f"\n正样本土地覆盖分布分析:")
    print(f"  总正样本数: {total:,}")
    print(f"  有土地覆盖信息的样本: {total - samples_without_lc:,}")
    print(f"  无土地覆盖信息的样本: {samples_without_lc:,}")
    
    if land_cover_counts:
        print(f"\n  土地覆盖类型分布:")
        for lc, count in sorted(land_cover_counts.items(), key=lambda x: x[1], reverse=True):
            prop = land_cover_proportions[lc]
            print(f"    类型 {lc}: {count:,} ({prop*100:.1f}%)")
    
    return {
        'counts': land_cover_counts,
        'proportions': land_cover_proportions,
        'samples_without_lc': samples_without_lc
    }


def extract_iso3_from_events(window_events, spatial_bounds=None, events=None):
    """
    从window_events或events中提取ISO3国家代码
    
    Args:
        window_events: 时间窗口内的事件数据（DataFrame，可能为空）
        spatial_bounds: 空间边界（用于从events中查找）
        events: 完整事件数据（当window_events为空时使用）
    
    Returns:
        str or None: ISO3国家代码，如果找不到则返回None
    """
    # 方法1: 从window_events中提取（优先）
    if window_events is not None and len(window_events) > 0:
        # 检查是否有iso3列
        if 'iso3' in window_events.columns:
            # 获取最常见的iso3值（排除NaN和空字符串）
            iso3_counts = window_events['iso3'].dropna()
            iso3_counts = iso3_counts[iso3_counts != '']
            if len(iso3_counts) > 0:
                most_common_iso3 = iso3_counts.value_counts().index[0]
                if pd.notna(most_common_iso3) and str(most_common_iso3).strip() != '':
                    return str(most_common_iso3).strip()
        
        # 如果window_events没有iso3列，但有country列，尝试转换
        if 'iso3' not in window_events.columns and 'country' in window_events.columns:
            try:
                # 导入country_to_iso3函数
                try:
                    from .country_to_iso3 import country_to_iso3
                except ImportError:
                    try:
                        from code.fire_equality.datamodules.country_to_iso3 import country_to_iso3
                    except ImportError:
                        current_dir = Path(__file__).resolve().parent
                        if str(current_dir) not in sys.path:
                            sys.path.insert(0, str(current_dir))
                        from country_to_iso3 import country_to_iso3
                country_counts = window_events['country'].dropna()
                country_counts = country_counts[country_counts != '']
                if len(country_counts) > 0:
                    most_common_country = country_counts.value_counts().index[0]
                    iso3_code = country_to_iso3(most_common_country)
                    if iso3_code is not None:
                        return iso3_code
            except Exception:
                pass
    
    # 方法2: 如果window_events为空或没有iso3，从events中查找该区域最常见的iso3
    if events is not None and spatial_bounds is not None:
        try:
            # 优先使用iso3列
            if 'iso3' in events.columns:
                # 查找该空间区域内的所有事件（不限制时间，因为GDP是年度数据）
                spatial_mask = (
                    (events['lat'] >= spatial_bounds['lat_min']) & 
                    (events['lat'] <= spatial_bounds['lat_max']) &
                    (events['lon'] >= spatial_bounds['lon_min']) & 
                    (events['lon'] <= spatial_bounds['lon_max'])
                )
                region_events = events[spatial_mask]
                
                if len(region_events) > 0:
                    # 获取最常见的iso3值（排除NaN和空字符串）
                    iso3_counts = region_events['iso3'].dropna()
                    iso3_counts = iso3_counts[iso3_counts != '']
                    if len(iso3_counts) > 0:
                        most_common_iso3 = iso3_counts.value_counts().index[0]
                        if pd.notna(most_common_iso3) and str(most_common_iso3).strip() != '':
                            return str(most_common_iso3).strip()
            
            # 如果没有iso3列，尝试从country列转换
            if 'iso3' not in events.columns and 'country' in events.columns:
                spatial_mask = (
                    (events['lat'] >= spatial_bounds['lat_min']) & 
                    (events['lat'] <= spatial_bounds['lat_max']) &
                    (events['lon'] >= spatial_bounds['lon_min']) & 
                    (events['lon'] <= spatial_bounds['lon_max'])
                )
                region_events = events[spatial_mask]
                
                if len(region_events) > 0:
                    try:
                        # 导入country_to_iso3函数
                        try:
                            from .country_to_iso3 import country_to_iso3
                        except ImportError:
                            try:
                                from code.fire_equality.datamodules.country_to_iso3 import country_to_iso3
                            except ImportError:
                                current_dir = Path(__file__).resolve().parent
                                if str(current_dir) not in sys.path:
                                    sys.path.insert(0, str(current_dir))
                                from country_to_iso3 import country_to_iso3
                        country_counts = region_events['country'].dropna()
                        country_counts = country_counts[country_counts != '']
                        if len(country_counts) > 0:
                            most_common_country = country_counts.value_counts().index[0]
                            iso3_code = country_to_iso3(most_common_country)
                            if iso3_code is not None:
                                return iso3_code
                    except Exception:
                        pass
        except Exception as e:
            # 静默失败，不打印错误（避免输出过多）
            pass
    
    return None


# 缓存：协变量表构建的 (lat,lon)->iso3 查找，供建样本时写入 metadata['iso3']
_covariate_iso3_tree = None
_covariate_iso3_list = None


def lat_lon_to_iso3_from_covariate(lat, lon, max_dist_deg=2.0, data_dir=None):
    """
    根据经纬度从协变量表 dataset/filtered_cleaned_cp_covariate.csv 最近邻匹配 iso3。
    用于建 .pth 时给无 events 匹配的样本（如负样本）补上 iso3，便于公平性分析中 GDP/continent 匹配。
    """
    global _covariate_iso3_tree, _covariate_iso3_list
    if lat is None or lon is None:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if _covariate_iso3_tree is None:
        base = Path(__file__).resolve().parent
        for _ in range(3):
            base = base.parent
        cov_path = (Path(data_dir) if data_dir else base / 'dataset') / 'filtered_cleaned_cp_covariate.csv'
        if not cov_path.exists():
            cov_path = Path('dataset/filtered_cleaned_cp_covariate.csv')
        if not cov_path.exists():
            return None
        try:
            df = pd.read_csv(cov_path, usecols=['lat_mean', 'lon_mean', 'iso3'])
        except Exception:
            return None
        df = df.dropna(subset=['lat_mean', 'lon_mean', 'iso3'])
        df = df[df['iso3'].astype(str).str.strip() != '']
        if len(df) == 0:
            return None
        pts = np.column_stack([df['lat_mean'].values, df['lon_mean'].values])
        _covariate_iso3_tree = cKDTree(pts)
        _covariate_iso3_list = df['iso3'].astype(str).str.strip().str.upper().tolist()
    dist, idx = _covariate_iso3_tree.query([lat, lon], k=1, distance_upper_bound=max_dist_deg)
    if np.isinf(dist) or idx >= len(_covariate_iso3_list) or float(dist) > max_dist_deg:
        return None
    return _covariate_iso3_list[int(idx)]


def _sample_iso3(iso3_from_events, pixel_lat, pixel_lon, data_dir=None):
    """确定样本的 iso3：优先用 events 提取的，否则用协变量表最近邻。用于写入 metadata['iso3']。"""
    if iso3_from_events is not None and str(iso3_from_events).strip() != '':
        return str(iso3_from_events).strip()
    return lat_lon_to_iso3_from_covariate(pixel_lat, pixel_lon, data_dir=data_dir)


def analyze_fire_seasonality(events, target_year_range):
    """
    分析火灾的季节性分布
    
    Args:
        events: 事件数据
        target_year_range: 目标年份范围 (start_year, end_year)
    
    Returns:
        dict: 季节性分析结果，包含：
            - 'monthly_counts': 每个月的火灾事件数量
            - 'monthly_proportions': 每个月的火灾事件比例
            - 'high_fire_seasons': 高火灾季节（火灾事件数超过平均值的月份）
            - 'low_fire_seasons': 低火灾季节（火灾事件数低于平均值的月份）
    """
    print("\n分析火灾季节性分布...")
    
    start_year, end_year = target_year_range
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    
    # 筛选目标年份范围内的事件
    events_in_range = events[
        (events['dtime'] >= start_date) & (events['dtime'] <= end_date)
    ].copy()
    
    # 按月份统计
    events_in_range['month'] = events_in_range['dtime'].dt.month
    monthly_counts = events_in_range['month'].value_counts().sort_index()
    
    # 计算比例
    total_events = len(events_in_range)
    monthly_proportions = (monthly_counts / total_events).to_dict()
    
    # 计算平均值
    avg_count = monthly_counts.mean()
    
    # 识别高/低火灾季节
    high_fire_seasons = set(monthly_counts[monthly_counts > avg_count].index)
    low_fire_seasons = set(monthly_counts[monthly_counts <= avg_count].index)
    
    print(f"  总火灾事件数: {total_events:,}")
    print(f"  平均每月事件数: {avg_count:.1f}")
    print(f"  高火灾季节（月份）: {sorted(high_fire_seasons)}")
    print(f"  低火灾季节（月份）: {sorted(low_fire_seasons)}")
    
    return {
        'monthly_counts': monthly_counts.to_dict(),
        'monthly_proportions': monthly_proportions,
        'high_fire_seasons': high_fire_seasons,
        'low_fire_seasons': low_fire_seasons,
        'avg_count': avg_count
    }


def create_negative_sample_pool(preprocessed_data, target_year_range, patch_size_km=25, 
                                 events_lc=None, spatial_resolution_km=1, fast_mode=False,
                                 positive_lc_types=None, use_modis_api=True, project='ee-tpan2203-wildfire'):
    """
    创建区块级别的负样本池
    
    负样本定义：
    - 空间标准：整个25×25km ConvLSTM输入区块内没有任何活跃火像素
    - 时间标准：从所有可用日期中筛选
    
    Args:
        preprocessed_data: 预处理后的数据
        target_year_range: 目标年份范围 (start_year, end_year)
        patch_size_km: 空间块大小（km），默认25km
        events_lc: 事件土地覆盖数据（可选，用于获取区块中心点的土地覆盖类型）
        spatial_resolution_km: 空间分辨率（km），默认1km
        fast_mode: 快速模式，使用更粗的网格和更少的日期（用于测试），默认False
    
    Returns:
        dict: 负样本池，包含：
            - 'patch_candidates': dict，键为 (center_lat, center_lon, date) 元组，值为区块信息
            - 'all_fire_dates': 所有有火灾事件的日期集合
            - 'events': 事件数据（用于后续采样）
            - 'seasonality': 季节性分析结果
            - 'target_year_range': 目标年份范围
            - 'patch_size_km': 区块大小
    """
    print("\n" + "="*60)
    print("创建区块级别负样本池...")
    print("="*60)
    if fast_mode:
        print("   ⚡ 快速模式：使用更粗的网格和更少的日期（用于测试）")
    print(f"  空间标准：整个{patch_size_km}×{patch_size_km}km区块内没有任何活跃火像素")
    print(f"  时间标准：从所有可用日期中筛选")
    
    events = preprocessed_data['events']
    components = preprocessed_data.get('components', pd.DataFrame())
    
    # 转换日期格式
    events['dtime'] = pd.to_datetime(events['dtime'])
    if not components.empty:
        components['dtime_min'] = pd.to_datetime(components['dtime_min'])
    
    # 分析季节性
    seasonality = analyze_fire_seasonality(events, target_year_range)
    
    # 生成目标年份范围内的所有日期
    start_year, end_year = target_year_range
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 快速模式：只检查每月的第1、11、21天（约每10天一次）
    if fast_mode:
        all_dates_list = sorted([d.date() for d in all_dates if d.day in [1, 11, 21]])
        print(f"   ⚡ 快速模式：只检查每月的第1、11、21天（共 {len(all_dates_list):,} 天）")
    else:
        all_dates_list = sorted(all_dates.date)
    
    # 获取目标年份范围内有火灾事件的日期
    event_dates_in_range = set(events[
        (events['dtime'] >= start_date) & (events['dtime'] <= end_date)
    ]['dtime'].dt.date.unique())
    
    if not components.empty:
        component_dates_in_range = set(components[
            (components['dtime_min'] >= start_date) & (components['dtime_min'] <= end_date)
        ]['dtime_min'].dt.date.unique())
        all_fire_dates = event_dates_in_range | component_dates_in_range
    else:
        all_fire_dates = event_dates_in_range
    
    # 确定研究区域的空间边界
    # 限制纬度范围，排除高纬度地区（>65°N或<-65°S），这些区域数据覆盖可能不完整
    lat_min = max(events['lat'].min(), -65.0)
    lat_max = min(events['lat'].max(), 65.0)
    spatial_bounds = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': events['lon'].min(),
        'lon_max': events['lon'].max()
    }
    
    print(f"\n研究区域边界:")
    print(f"  纬度: [{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}] (限制在±65°以内)")
    print(f"  经度: [{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
    print(f"  总日期数: {len(all_dates_list):,}")
    print(f"  有火灾事件的日期: {len(all_fire_dates):,}")
    
    # 计算区块半径（度）
    lat_per_km = 1 / 110.574
    patch_radius_deg = (patch_size_km / 2) * lat_per_km
    
    # 方法1：如果提供了正样本土地覆盖类型，直接使用本地 MODIS 文件搜索这些类型的区域
    if positive_lc_types is not None and len(positive_lc_types) > 0 and use_modis_api and HAS_MODIS_LC:
        print(f"\n🚀 使用本地 MODIS 文件直接搜索正样本土地覆盖类型: {sorted(positive_lc_types)}")
        print(f"   策略：直接在本地 MODIS 数据中搜索这些类型的区域，确保负样本池包含所有需要的类型")
        
        # 导入搜索函数（已在文件开头导入）
        
        # 确定采样数量（快速模式下也使用较大的数量，确保能找到所有类型）
        # 如果某种类型找不到，函数内部会自动增加采样数量重试
        num_samples_per_type = 2000 if fast_mode else 5000
        
        # 搜索正样本土地覆盖类型的位置
        lc_locations = find_locations_by_landcover(
            positive_lc_types=positive_lc_types,
            spatial_bounds=spatial_bounds,
            year=start_year,
            num_samples_per_type=num_samples_per_type,
            project=project
        )
        
        if len(lc_locations) == 0:
            print("  ⚠️  未能找到任何正样本土地覆盖类型的区域，回退到网格方法")
            # 回退到原来的网格方法
            use_modis_search = False
        else:
            use_modis_search = True
            print(f"  ✅ 找到 {len(lc_locations):,} 个正样本土地覆盖类型的区域位置")
    else:
        use_modis_search = False
        lc_locations = []
    
    # 方法2：使用空间网格（原来的方法，或作为回退）
    if not use_modis_search:
        # 创建空间网格，确保地理分布覆盖
        # 快速模式：使用更大的网格步长（500倍区块半径），进一步减少网格点数以加速迭代
        if fast_mode:
            grid_step_deg = patch_radius_deg * 500  # 快速模式：500倍步长（更稀疏，加速迭代）
            print(f"   ⚡ 快速模式：使用500倍网格步长（{grid_step_deg:.4f}度，约{grid_step_deg * 110.574:.1f}km）")
        else:
            grid_step_deg = patch_radius_deg  # 正常模式：区块半径作为步长
        
        lat_grid = np.arange(
            spatial_bounds['lat_min'] + patch_radius_deg,
            spatial_bounds['lat_max'] - patch_radius_deg + grid_step_deg,
            grid_step_deg
        )
        lon_grid = np.arange(
            spatial_bounds['lon_min'] + patch_radius_deg,
            spatial_bounds['lon_max'] - patch_radius_deg + grid_step_deg,
            grid_step_deg
        )
        
        print(f"\n创建空间网格（确保地理分布覆盖）...")
        print(f"  纬度网格点数: {len(lat_grid):,}")
        print(f"  经度网格点数: {len(lon_grid):,}")
        print(f"  总候选区块中心点数: {len(lat_grid) * len(lon_grid):,}")
        if fast_mode:
            normal_grid_count = int((spatial_bounds['lat_max'] - spatial_bounds['lat_min'] - 2*patch_radius_deg) / patch_radius_deg) * int((spatial_bounds['lon_max'] - spatial_bounds['lon_min'] - 2*patch_radius_deg) / patch_radius_deg)
            fast_grid_count = len(lat_grid) * len(lon_grid)
            reduction_factor = normal_grid_count / fast_grid_count if fast_grid_count > 0 else 1
            print(f"   ⚡ 快速模式：网格点数减少到 {fast_grid_count:,} 个（正常模式约 {normal_grid_count:,} 个，减少约 {reduction_factor:.1f} 倍）")
        
        # 将网格点转换为位置列表
        lc_locations = []
        for center_lat in lat_grid:
            for center_lon in lon_grid:
                lc_locations.append({
                    'lat': center_lat,
                    'lon': center_lon,
                    'land_cover': None  # 网格方法中，土地覆盖类型稍后获取
                })
    
    # 对于每个位置和日期，检查该区块内是否有活跃火像素
    print(f"\n检查区块内是否有活跃火像素...")
    patch_candidates = {}  # 键为 (center_lat, center_lon, date)，值为区块信息
    
    events_in_range = events[
        (events['dtime'] >= start_date) & (events['dtime'] <= end_date)
    ].copy()
    
    # 为了加速，按日期分组事件
    events_by_date = {}
    for date in all_dates_list:
        date_events = events_in_range[events_in_range['dtime'].dt.date == date]
        if len(date_events) > 0:
            events_by_date[date] = date_events
    
    total_checks = len(lc_locations) * len(all_dates_list)
    checked = 0
    print(f"  需要检查 {total_checks:,} 个位置×日期组合...")
    
    for location in lc_locations:
        center_lat = location['lat']
        center_lon = location['lon']
        
        # 计算区块边界
        patch_bounds = {
            'lat_min': center_lat - patch_radius_deg,
            'lat_max': center_lat + patch_radius_deg,
            'lon_min': center_lon - patch_radius_deg,
            'lon_max': center_lon + patch_radius_deg
        }
        
        # 检查该区块在每个日期是否有活跃火像素
        for date in all_dates_list:
            checked += 1
            if checked % 10000 == 0:
                print(f"    进度: {checked:,}/{total_checks:,} ({checked*100/total_checks:.1f}%)")
            
            # 如果该日期有火灾事件，检查是否在区块内
            if date in events_by_date:
                date_events = events_by_date[date]
                # 检查是否有事件在区块内
                in_patch = (
                    (date_events['lat'] >= patch_bounds['lat_min']) &
                    (date_events['lat'] <= patch_bounds['lat_max']) &
                    (date_events['lon'] >= patch_bounds['lon_min']) &
                    (date_events['lon'] <= patch_bounds['lon_max'])
                )
                if in_patch.any():
                    continue  # 区块内有活跃火像素，跳过
            
            # 区块内没有活跃火像素，添加到候选池
            patch_key = (center_lat, center_lon, date)
            patch_candidates[patch_key] = {
                'center_lat': center_lat,
                'center_lon': center_lon,
                'date': date,
                'patch_bounds': patch_bounds,
                'land_cover': location.get('land_cover')  # 如果使用 MODIS 搜索，已经有土地覆盖类型
            }
    
    print(f"\n✅ 负样本池创建完成:")
    print(f"  候选负样本数（区块×日期）: {len(patch_candidates):,}")
    
    return {
        'patch_candidates': patch_candidates,
        'all_fire_dates': all_fire_dates,
        'events': events,
        'events_lc': events_lc,
        'seasonality': seasonality,
        'target_year_range': target_year_range,
        'patch_size_km': patch_size_km,
        'spatial_bounds': spatial_bounds,
        'project': project  # 保存project参数，供后续采样使用
    }


def sample_negative_samples_by_land_cover(negative_pool, positive_lc_distribution, 
                                         num_negative_samples,
                                         patch_size_km=25, time_steps=10, 
                                         spatial_resolution_km=1, spatial_bounds=None,
                                         target_year_range=None,
                                         use_modis_api=True, project='ee-tpan2203-wildfire',
                                         checkpoint_file=None, checkpoint_interval=100):
    """
    按土地覆盖类型分层采样负样本
    
    按照正样本的土地覆盖分布比例来分配负样本的名额
    
    Args:
        negative_pool: 负样本池（来自create_negative_sample_pool，包含区块级别候选）
        positive_lc_distribution: 正样本土地覆盖分布（来自analyze_positive_sample_land_cover_distribution）
        num_negative_samples: 需要采样的负样本数量（通常是正样本的2倍）
        patch_size_km: 空间块大小
        time_steps: 时间步长
        spatial_resolution_km: 空间分辨率
        spatial_bounds: 研究区域的空间边界（如果为None，则从events数据推断）
        target_year_range: 目标年份范围
    
    Returns:
        list: 负样本列表，格式与正样本相同
    """
    print("\n" + "="*60)
    print("按土地覆盖类型分层采样负样本...")
    print("="*60)
    print("  土地覆盖分层：按正样本的土地覆盖分布进行分层采样")
    print("  地理分布：确保负样本覆盖不同地理区域")
    
    patch_candidates = negative_pool['patch_candidates']
    events = negative_pool['events']
    events_lc = negative_pool.get('events_lc', None)
    seasonality = negative_pool.get('seasonality', None)
    target_year_range = target_year_range or negative_pool.get('target_year_range')
    spatial_bounds = spatial_bounds or negative_pool.get('spatial_bounds')
    
    if len(patch_candidates) == 0:
        print("⚠️  警告：没有找到候选负样本区块，无法生成负样本")
        return []
    
    # 如果没有正样本分布，则随机采样
    if positive_lc_distribution['counts'] == {}:
        print("⚠️  没有正样本土地覆盖分布，使用随机采样策略")
        return sample_negative_samples_random(negative_pool, num_negative_samples, 
                                            patch_size_km, time_steps, spatial_resolution_km,
                                            target_year_range)
    
    # 计算每个土地覆盖类型需要采样的数量（按正样本分布）
    lc_proportions = positive_lc_distribution['proportions']
    lc_sample_counts = {}
    
    for lc, proportion in lc_proportions.items():
        lc_sample_counts[lc] = int(num_negative_samples * proportion)
    
    # 确保总数不超过num_negative_samples
    total_allocated = sum(lc_sample_counts.values())
    if total_allocated < num_negative_samples:
        # 将剩余数量分配给最常见的土地覆盖类型
        remaining = num_negative_samples - total_allocated
        if lc_proportions:
            most_common_lc = max(lc_proportions.items(), key=lambda x: x[1])[0]
            lc_sample_counts[most_common_lc] += remaining
    
    print(f"负样本分配策略（共 {num_negative_samples:,} 个，按正样本土地覆盖分布）:")
    for lc, count in sorted(lc_sample_counts.items(), key=lambda x: x[1], reverse=True):
        prop = lc_proportions.get(lc, 0)
        print(f"  土地覆盖类型 {lc}: {count:,} 个 ({prop*100:.1f}%)")
    
    # 转换日期格式
    events['dtime'] = pd.to_datetime(events['dtime'])
    
    # 检查是否使用本地 landcover 文件
    if use_modis_api:
        if not HAS_RASTERIO:
            error_msg = f"❌ 无法读取本地 landcover 数据！\n"
            error_msg += f"   原因: rasterio 未安装\n"
            error_msg += f"   请运行: pip install rasterio\n"
            error_msg += f"   流水线已终止，不会使用events_lc作为回退方案"
            raise ImportError(error_msg)
        
        print(f"   🚀 使用本地 landcover 文件获取土地覆盖类型（dataset/LandCover/）")
        use_events_lc = False
    else:
        print(f"   🔄 使用events_lc获取土地覆盖类型（use_modis_api=False）")
        use_events_lc = True
        if events_lc is not None:
            events_lc['dtime'] = pd.to_datetime(events_lc['dtime'])
            # 检查events和events_lc的行索引是否一致（用于精确匹配）
            if len(events) == len(events_lc):
                # 检查索引是否一致（至少前1000行作为样本检查）
                sample_size = min(1000, len(events))
                if events.index[:sample_size].equals(events_lc.index[:sample_size]):
                    use_index_matching = True
                    print("   ✅ 检测到events和events_lc行索引一致，将使用索引匹配")
                else:
                    print("   ⚠️  events和events_lc行索引不一致，将使用位置匹配")
            else:
                print(f"   ⚠️  events和events_lc行数不一致 ({len(events)} vs {len(events_lc)})，将使用位置匹配")
    
    # 获取区块中心点的土地覆盖类型
    # 优先使用MODIS模块（更快），否则回退到events_lc方法
    print("\n获取区块中心点的土地覆盖类型...")
    patch_candidates_with_lc = {}  # 键为 (center_lat, center_lon, date)，值为 (patch_info, lc)
    
    # 确定年份（用于MODIS查询）
    if target_year_range:
        year = target_year_range[0]  # 使用起始年份
    else:
        # 从候选区块的日期推断年份
        sample_date = next(iter(patch_candidates.values()))['date']
        year = sample_date.year if hasattr(sample_date, 'year') else pd.Timestamp(sample_date).year
    
    # 方法1：使用本地 MODIS 文件（更快）
    if use_modis_api:
        if not HAS_RASTERIO:
            error_msg = f"❌ 无法读取本地 landcover 数据！\n"
            error_msg += f"   原因: rasterio 未安装\n"
            error_msg += f"   请运行: pip install rasterio\n"
            error_msg += f"   流水线已终止，不会使用events_lc作为回退方案"
            raise ImportError(error_msg)
        
        print("  🚀 使用本地 MODIS 文件获取土地覆盖类型...")
        
        # 获取正样本需要的土地覆盖类型列表
        positive_lc_types = set(positive_lc_distribution['counts'].keys())
        print(f"    正样本包含的土地覆盖类型: {sorted(positive_lc_types)}")
        
        # 检查负样本池中的候选是否已经有土地覆盖类型（如果使用了 MODIS 直接搜索方法）
        has_precomputed_lc = any(
            patch_info.get('land_cover') is not None 
            for patch_info in patch_candidates.values()
        )
        
        if has_precomputed_lc:
            print(f"    ✅ 负样本池中已有土地覆盖类型信息（来自 MODIS 直接搜索）")
            # 直接使用已有的土地覆盖类型
            for patch_key, patch_info in patch_candidates.items():
                lc = patch_info.get('land_cover')
                if lc is not None and lc != 255:  # 255是未分类
                    patch_candidates_with_lc[patch_key] = (patch_info, lc)
        else:
            # 需要批量获取土地覆盖类型
            print(f"    批量查询 {len(patch_candidates):,} 个坐标的土地覆盖类型...")
            coords = [(patch_info['center_lat'], patch_info['center_lon']) 
                     for patch_info in patch_candidates.values()]
            patch_keys = list(patch_candidates.keys())
            
            lc_results = get_landcover_batch_gee(coords, year, lc_type='LC_Type1', 
                                                 batch_size=1000, project=project)
            
            # 将结果映射回patch_candidates
            for patch_key, lc in zip(patch_keys, lc_results):
                if lc is not None and lc != 255:  # 255是未分类
                    patch_candidates_with_lc[patch_key] = (patch_candidates[patch_key], lc)
        
        positive_found_count = sum(
            1 for (patch_info, lc) in patch_candidates_with_lc.values()
            if lc in positive_lc_types
        )
        
        print(f"  ✅ 成功获取 {len(patch_candidates_with_lc):,}/{len(patch_candidates):,} 个区块的土地覆盖类型")
        print(f"    其中正样本需要的类型: {positive_found_count:,} 个")
        
        # 检查是否获取到所有正样本需要的土地覆盖类型
        found_lc_types = set()
        for patch_key, (patch_info, lc) in patch_candidates_with_lc.items():
            found_lc_types.add(lc)
        
        missing_positive_lc_types = positive_lc_types - found_lc_types
        if missing_positive_lc_types:
            print(f"  ⚠️  警告：以下正样本土地覆盖类型在负样本池中未找到: {sorted(missing_positive_lc_types)}")
            print(f"     将尝试从其他类型补充")
        else:
            print(f"  ✅ 已获取到所有正样本需要的土地覆盖类型: {sorted(positive_lc_types)}")
        
        if len(patch_candidates_with_lc) == 0:
            local_file = os.path.join('dataset', 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
            error_msg = f"❌ 未能从本地文件获取到任何土地覆盖类型！\n"
            error_msg += f"   查询了 {len(patch_candidates):,} 个候选区块，但未获取到有效结果\n"
            error_msg += f"   请检查:\n"
            error_msg += f"   1. 本地文件是否存在: {local_file}\n"
            error_msg += f"   2. 文件是否损坏\n"
            error_msg += f"   3. 坐标是否在研究区域内\n"
            error_msg += f"   流水线已终止"
            raise RuntimeError(error_msg)
    
    # 方法2：使用events_lc方法（仅当use_modis_api=False时）
    elif not use_modis_api:
        if use_events_lc and events_lc is not None and len(events_lc) > 0:
            print("  🔄 使用events_lc方法获取土地覆盖类型（扩大日期容差：前后7天）...")
            
            lat_per_km = 1 / 110.574
            tolerance = 0.005  # 容差（度）
            
            # 预先按日期分组events，避免重复筛选
            all_dates = set(patch_info['date'] for patch_info in patch_candidates.values())
            # 扩展日期范围：前后7天
            extended_dates = set()
            for date in all_dates:
                date_pd = pd.Timestamp(date)
                for days_offset in range(-7, 8):  # 前后7天
                    extended_dates.add((date_pd + pd.Timedelta(days=days_offset)).date())
            
            # 优化：一次性筛选所有扩展日期范围内的事件，然后按日期分组
            min_extended_date = min(extended_dates)
            max_extended_date = max(extended_dates)
            events_extended = events[
                (events['dtime'].dt.date >= min_extended_date) & 
                (events['dtime'].dt.date <= max_extended_date)
            ]
            
            events_by_date = {}
            for date in extended_dates:
                date_events = events_extended[events_extended['dtime'].dt.date == date]
                if len(date_events) > 0:
                    events_by_date[date] = date_events
            print(f"    已为 {len(extended_dates):,} 个日期（包含前后7天）预分组events数据，共 {len(events_extended):,} 个事件")
            
            total_patches = len(patch_candidates)
            processed = 0
            progress_interval = max(1, total_patches // 20)  # 每5%显示一次进度
            
            for patch_key, patch_info in patch_candidates.items():
                processed += 1
                if processed % progress_interval == 0:
                    progress = processed / total_patches * 100
                    print(f"    进度: {processed:,}/{total_patches:,} ({progress:.1f}%) - 已匹配 {len(patch_candidates_with_lc):,} 个", end='\r')
                
                center_lat, center_lon, date = patch_key
                
                # 从events_lc中查找匹配的土地覆盖类型（使用索引匹配，扩大日期容差）
                lc = None
                try:
                    # 扩大日期容差：在该日期前后7天内查找
                    date_pd = pd.Timestamp(date)
                    best_match = None
                    best_distance = float('inf')
                    
                    # 优先查找目标日期，然后查找前后7天（按距离优先排序）
                    date_range = [date] + [(date_pd + pd.Timedelta(days=offset)).date() 
                                          for offset in range(-7, 8) if offset != 0]
                    
                    # 使用更大的空间容差（0.5度，约55km），因为负样本区块可能距离最近的events事件较远
                    max_tolerance = 0.5  # 约55km
                    
                    # 优化：一旦找到足够近的匹配（<0.1度），就停止搜索
                    early_stop_threshold = 0.1  # 约11km，足够近的匹配
                    
                    for search_date in date_range:
                        if search_date in events_by_date:
                            date_events = events_by_date[search_date]
                            # 计算距离
                            distances = np.sqrt(
                                (date_events['lat'] - center_lat)**2 + 
                                (date_events['lon'] - center_lon)**2
                            )
                            closest_idx = distances.idxmin()
                            min_distance = distances.loc[closest_idx]
                            
                            if min_distance < max_tolerance and min_distance < best_distance:
                                if closest_idx in events_lc.index:
                                    lc_row = events_lc.loc[closest_idx]
                                    # 检查日期是否在合理范围内（前后7天）
                                    lc_date = lc_row['dtime'].date()
                                    if abs((lc_date - date).days) <= 7:
                                        best_match = lc_row
                                        best_distance = min_distance
                                        
                                        # 早期停止：如果找到足够近的匹配，停止搜索
                                        if min_distance < early_stop_threshold:
                                            break
                    
                    # 如果找到匹配，提取土地覆盖类型
                    if best_match is not None:
                        lc_values = []
                        for i in range(1, 5):
                            lc_val = best_match.get(f'lc{i}', None)
                            if lc_val is not None and lc_val > 0:
                                lc_values.append(lc_val)
                        if lc_values:
                            from collections import Counter
                            lc = Counter(lc_values).most_common(1)[0][0]
                except Exception as e:
                    # 如果匹配失败，跳过这个区块的土地覆盖匹配
                    continue
                
                if lc is not None:
                    patch_candidates_with_lc[patch_key] = (patch_info, lc)
            
            print()  # 换行
            print(f"    有土地覆盖信息的候选区块: {len(patch_candidates_with_lc):,}/{len(patch_candidates):,}")
        else:
            print("  ⚠️  无法使用events_lc方法（数据不可用或不一致）")
    
    print(f"  总计有土地覆盖信息的候选区块: {len(patch_candidates_with_lc):,}/{len(patch_candidates):,}")
    
    # 按土地覆盖类型分组候选区块
    patches_by_lc = {}
    for patch_key, (patch_info, lc) in patch_candidates_with_lc.items():
        if lc not in patches_by_lc:
            patches_by_lc[lc] = []
        patches_by_lc[lc].append((patch_key, patch_info))
    
    # 显示每个土地覆盖类型的候选区块数量
    print(f"\n  各土地覆盖类型的候选区块数量:")
    for lc in sorted(patches_by_lc.keys()):
        print(f"    类型 {lc}: {len(patches_by_lc[lc]):,} 个候选区块")
    
    # 检查哪些目标土地覆盖类型没有候选区块
    missing_lc_types = []
    for lc, target_count in lc_sample_counts.items():
        if target_count > 0 and lc not in patches_by_lc:
            missing_lc_types.append((lc, target_count))
    
    if missing_lc_types:
        print(f"\n  ⚠️  以下土地覆盖类型没有候选区块:")
        for lc, target_count in missing_lc_types:
            print(f"     类型 {lc}: 目标 {target_count:,} 个，但可用候选为 0")
        print(f"     可能原因：快速模式下网格过稀疏，这些类型的位置未被采样到")
        print(f"     解决方案：将从其他类型补充，或降低快速模式的稀疏度")
    
    negative_samples = []
    grid_size = int(patch_size_km / spatial_resolution_km)
    
    # 记录需要补充的样本数量（用于后续从其他类型补充）
    total_missing_samples = sum(count for lc, count in missing_lc_types)
    
    # 尝试加载已保存的负样本（断点恢复）
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n💾 检测到检查点文件: {checkpoint_file}")
        loaded_samples, loaded_config = load_negative_samples(checkpoint_file)
        if loaded_samples:
            negative_samples = loaded_samples
            print(f"   ✅ 已加载 {len(negative_samples):,} 个负样本，将继续采样...")
            # 计算还需要采样的数量
            remaining_needed = num_negative_samples - len(negative_samples)
            if remaining_needed <= 0:
                print(f"   ✅ 已采样足够的负样本，无需继续")
                return negative_samples
            print(f"   📊 还需要采样 {remaining_needed:,} 个负样本")
            # 更新每个土地覆盖类型的目标数量（减去已采样的）
            for lc in lc_sample_counts:
                existing_count = sum(1 for s in negative_samples if s.get('land_cover') == lc)
                lc_sample_counts[lc] = max(0, lc_sample_counts[lc] - existing_count)
    
    # 为每个土地覆盖类型采样
    for lc, target_count in lc_sample_counts.items():
        if target_count == 0:
            continue
        
        if lc not in patches_by_lc:
            print(f"\n  ⚠️  土地覆盖类型 {lc} 没有候选区块（目标: {target_count:,} 个）")
            print(f"     将尝试从其他类型补充")
            continue
        
        available_count = len(patches_by_lc[lc])
        print(f"\n  采样土地覆盖类型 {lc} 的负样本: {target_count:,} 个（可用候选: {available_count:,} 个）...")
        sampled_count = 0
        skipped_time_window = 0
        skipped_data_missing = 0  # 数据缺失区域（关键特征全为0）
        skipped_high_lat = 0  # 高纬度地区（>65°）
        max_attempts = target_count * 10
        attempts = 0
        
        # 随机打乱候选区块（确保地理分布）
        np.random.seed(42)
        candidate_patches = patches_by_lc[lc].copy()
        np.random.shuffle(candidate_patches)
        
        # 添加进度条
        from tqdm import tqdm
        progress_bar = tqdm(
            total=target_count,
            desc=f"    类型 {lc}",
            unit="样本",
            leave=False,
            ncols=100
        )
        
        for patch_key, patch_info in candidate_patches:
            if sampled_count >= target_count:
                break
            if attempts >= max_attempts:
                print(f"    ⚠️  达到最大尝试次数 ({max_attempts:,})，停止采样")
                break
            
            attempts += 1
            
            center_lat = patch_info['center_lat']
            center_lon = patch_info['center_lon']
            sampled_date = patch_info['date']
            patch_bounds = patch_info['patch_bounds']
            
            # 创建特征立方体（时间窗口：该日期前time_steps天）
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    skipped_time_window += 1
                    continue  # 跳过需要补零的样本（时间窗口不完整）
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            # 提取时间窗口内的事件（用于构建特征）
            spatial_mask = (
                (events['lat'] >= patch_bounds['lat_min']) & 
                (events['lat'] <= patch_bounds['lat_max']) &
                (events['lon'] >= patch_bounds['lon_min']) & 
                (events['lon'] <= patch_bounds['lon_max'])
            )
            temporal_mask = (
                (events['dtime'] >= time_window_start) & 
                (events['dtime'] <= time_window_end)
            )
            window_events = events[spatial_mask & temporal_mask].copy()
            
            # 创建特征立方体
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features 未导入成功，无法生成6通道特征。请检查 code/fire_equality/datamodules/feature_alignment.py")
            # 提取ISO3国家代码（如果可用）
            iso3_code = extract_iso3_from_events(window_events, patch_bounds, events)
            # 调试信息（仅在第一次采样时打印）
            if sampled_count == 0 and iso3_code is None:
                # 检查events是否有iso3列
                if events is not None and 'iso3' in events.columns:
                    print(f"    💡 提示：events数据有iso3列，但该区域未找到有效的iso3值")
                elif events is not None:
                    print(f"    💡 提示：events数据没有iso3列，无法提取国家代码")
            feature_cube = extract_aligned_features(
                spatial_bounds=patch_bounds,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                grid_size=grid_size,
                fire_date=pd.Timestamp(sampled_date).to_pydatetime(),
                fire_year=pd.Timestamp(sampled_date).year,
                iso3=iso3_code,
                data_dir='dataset',
                project=negative_pool.get('project', 'ee-tpan2203-wildfire') if isinstance(negative_pool, dict) else 'ee-tpan2203-wildfire'
            )
            
            # 检查特征是否有效（排除数据缺失区域）
            # 检查关键特征通道（FWI, NDVI, max_temp, max_wind）的数据完整性
            # 策略：
            # 1. 如果所有关键特征都全为0 → 跳过（数据缺失区域，如海洋）
            # 2. 如果3个或以上关键特征全为0 → 跳过（数据严重缺失）
            # 3. 如果只有1-2个特征为0 → 可能是正常的（如NDVI在沙漠为0），保留
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 检查每个特征是否全为0
            fwi_all_zero = (fwi_data == 0).all()
            ndvi_all_zero = (ndvi_data == 0).all()
            temp_all_zero = (temp_data == 0).all()
            wind_all_zero = (wind_data == 0).all()
            
            zero_count = sum([fwi_all_zero, ndvi_all_zero, temp_all_zero, wind_all_zero])
            
            # 如果所有关键特征都全为0，跳过这个样本（数据缺失区域）
            if zero_count == 4:
                skipped_data_missing += 1
                continue
            
            # 如果3个或以上关键特征全为0，也跳过（数据严重缺失）
            if zero_count >= 3:
                skipped_data_missing += 1
                continue
            
            # 检查地理范围：排除高纬度地区（>65°N或<-65°S）和明显超出数据覆盖范围的区域
            # 这些区域通常数据覆盖不完整
            if abs(center_lat) > 65:
                skipped_high_lat += 1
                continue
            
            iso3_final = _sample_iso3(iso3_code, center_lat, center_lon, data_dir='dataset')
            sample = {
                'pixel_lat': center_lat,
                'pixel_lon': center_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 负样本
                'land_cover': lc,
                'metadata': {
                    'component_id': None,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'spatial_bounds': patch_bounds,
                    'time_window': (time_window_start, time_window_end),
                    'iso3': iso3_final
                }
            }
            negative_samples.append(sample)
            sampled_count += 1
            progress_bar.update(1)
            
            # 定期保存检查点（每checkpoint_interval个样本保存一次）
            if checkpoint_file and len(negative_samples) % checkpoint_interval == 0:
                try:
                    # 准备配置用于保存
                    checkpoint_config = {
                        'patch_size_km': patch_size_km,
                        'time_steps': time_steps,
                        'target_years': target_year_range
                    }
                    save_negative_samples(negative_samples, checkpoint_config, checkpoint_file)
                except Exception as e:
                    print(f"    ⚠️  保存检查点失败: {e}")
        
        progress_bar.close()
        
        # 完成该土地覆盖类型后保存检查点
        if checkpoint_file:
            try:
                checkpoint_config = {
                    'patch_size_km': patch_size_km,
                    'time_steps': time_steps,
                    'target_years': target_year_range
                }
                save_negative_samples(negative_samples, checkpoint_config, checkpoint_file)
            except Exception as e:
                print(f"    ⚠️  保存检查点失败: {e}")
        
        # 打印跳过统计信息
        if skipped_time_window > 0:
            print(f"    ⚠️  跳过了 {skipped_time_window:,} 个候选区块（时间窗口不完整）")
        if skipped_data_missing > 0:
            print(f"    ⚠️  跳过了 {skipped_data_missing:,} 个候选区块（数据缺失区域：3个或以上关键特征全为0）")
        if skipped_high_lat > 0:
            print(f"    ⚠️  跳过了 {skipped_high_lat:,} 个候选区块（高纬度地区：|lat| > 65°）")
        if sampled_count < target_count:
            print(f"    ⚠️  采样不足：成功 {sampled_count:,} 个，目标 {target_count:,} 个（缺少 {target_count - sampled_count:,} 个）")
        print(f"    ✅ 成功采样 {sampled_count:,} 个负样本（目标: {target_count:,}）")
    
    # 如果某些土地覆盖类型没有候选区块，尝试从其他类型补充
    if missing_lc_types and len(negative_samples) < num_negative_samples:
        print(f"\n  🔄 尝试从其他类型补充缺失的负样本...")
        remaining_needed = num_negative_samples - len(negative_samples)
        print(f"     需要补充: {remaining_needed:,} 个负样本")
        
        # 从有剩余候选的类型中补充
        available_lc_types = []
        for lc in sorted(patches_by_lc.keys()):
            if lc not in [s['land_cover'] for s in negative_samples]:  # 还没有采样过的类型
                available_lc_types.append((lc, len(patches_by_lc[lc])))
            else:
                # 检查是否还有未使用的候选
                used_count = sum(1 for s in negative_samples if s['land_cover'] == lc)
                remaining = len(patches_by_lc[lc]) - used_count
                if remaining > 0:
                    available_lc_types.append((lc, remaining))
        
        if available_lc_types:
            # 按可用数量排序，优先使用候选多的类型
            available_lc_types.sort(key=lambda x: x[1], reverse=True)
            print(f"     可用类型: {', '.join(f'类型{lc}({count}个候选)' for lc, count in available_lc_types[:5])}")
            
            # 从可用类型中补充
            for lc, available_count in available_lc_types:
                if len(negative_samples) >= num_negative_samples:
                    break
                
                # 计算需要补充的数量
                to_sample = min(remaining_needed, available_count, num_negative_samples - len(negative_samples))
                if to_sample == 0:
                    continue
                
                print(f"     从类型 {lc} 补充 {to_sample:,} 个负样本...")
                
                # 获取该类型的未使用候选
                np.random.seed(42)
                candidate_patches = patches_by_lc[lc].copy()
                np.random.shuffle(candidate_patches)
                
                # 过滤掉已经采样过的候选（通过坐标和日期判断）
                used_keys = {(s['pixel_lat'], s['pixel_lon'], s['pixel_date']) for s in negative_samples}
                unused_candidates = [
                    (k, p) for k, p in candidate_patches 
                    if (p['center_lat'], p['center_lon'], p['date']) not in used_keys
                ]
                
                # 添加进度条
                from tqdm import tqdm
                supplement_progress = tqdm(
                    total=to_sample,
                    desc=f"      补充类型 {lc}",
                    unit="样本",
                    leave=False,
                    ncols=100
                )
                
                sampled_supplement = 0
                for patch_key, patch_info in unused_candidates:
                    if sampled_supplement >= to_sample or len(negative_samples) >= num_negative_samples:
                        break
                    
                    center_lat = patch_info['center_lat']
                    center_lon = patch_info['center_lon']
                    sampled_date = patch_info['date']
                    patch_bounds = patch_info['patch_bounds']
                    
                    # 检查时间窗口
                    time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
                    if target_year_range is not None:
                        start_year, _ = target_year_range
                        year_start_date = pd.Timestamp(f'{start_year}-01-01')
                        if time_window_start < year_start_date:
                            continue
                    
                    time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
                    
                    # 提取时间窗口内的事件
                    spatial_mask = (
                        (events['lat'] >= patch_bounds['lat_min']) & 
                        (events['lat'] <= patch_bounds['lat_max']) &
                        (events['lon'] >= patch_bounds['lon_min']) & 
                        (events['lon'] <= patch_bounds['lon_max'])
                    )
                    temporal_mask = (
                        (events['dtime'] >= time_window_start) & 
                        (events['dtime'] <= time_window_end)
                    )
                    window_events = events[spatial_mask & temporal_mask].copy()
                    
                # 创建特征立方体（8通道）
                if extract_aligned_features is None:
                    raise ImportError("extract_aligned_features 未导入成功，无法生成8通道特征。请检查 code/fire_equality/datamodules/feature_alignment.py")
                # 提取ISO3国家代码（如果可用）
                iso3_code = extract_iso3_from_events(window_events, patch_bounds, events)
                feature_cube = extract_aligned_features(
                    spatial_bounds=patch_bounds,
                    time_window_start=time_window_start,
                    time_window_end=time_window_end,
                    grid_size=grid_size,
                    fire_date=pd.Timestamp(sampled_date).to_pydatetime(),
                    fire_year=pd.Timestamp(sampled_date).year,
                    iso3=iso3_code,
                    data_dir='dataset',
                    project=negative_pool.get('project', 'ee-tpan2203-wildfire') if isinstance(negative_pool, dict) else 'ee-tpan2203-wildfire'
                )
                
                # 检查特征是否有效（排除数据缺失区域）
                # 策略：如果3个或以上关键特征全为0，跳过（数据严重缺失）
                fwi_data = feature_cube[:, :, :, 0]  # FWI
                ndvi_data = feature_cube[:, :, :, 2]  # NDVI
                temp_data = feature_cube[:, :, :, 6]  # max_temp
                wind_data = feature_cube[:, :, :, 7]  # max_wind
                
                zero_count = sum([
                    (fwi_data == 0).all(),
                    (ndvi_data == 0).all(),
                    (temp_data == 0).all(),
                    (wind_data == 0).all()
                ])
                
                # 如果3个或以上关键特征全为0，跳过（数据严重缺失）
                if zero_count >= 3:
                    continue
                
                # 检查地理范围：排除高纬度地区
                if abs(center_lat) > 65:
                    continue
                
                iso3_final = _sample_iso3(None, center_lat, center_lon, data_dir='dataset')
                sample = {
                    'pixel_lat': center_lat,
                    'pixel_lon': center_lon,
                    'pixel_date': sampled_date,
                    'features': feature_cube,
                    'target': 0,
                    'land_cover': lc,  # 保持原始土地覆盖类型
                    'metadata': {
                        'component_id': None,
                        'center_lat': center_lat,
                        'center_lon': center_lon,
                        'spatial_bounds': patch_bounds,
                        'time_window': (time_window_start, time_window_end),
                        'supplemented': True,  # 标记为补充样本
                        'iso3': iso3_final
                    }
                }
                negative_samples.append(sample)
                sampled_supplement += 1
                supplement_progress.update(1)
            
            supplement_progress.close()
            print(f"     ✅ 从类型 {lc} 补充了 {sampled_supplement:,} 个负样本")
        else:
            print(f"     ⚠️  没有可用的其他类型来补充")
    
    print(f"\n✅ 总共创建 {len(negative_samples):,} 个负样本（目标: {num_negative_samples:,}）")
    if len(negative_samples) < num_negative_samples:
        print(f"   ⚠️  实际采样数量 ({len(negative_samples):,}) 少于目标数量 ({num_negative_samples:,})")
        print(f"      缺少: {num_negative_samples - len(negative_samples):,} 个")
        print(f"      建议：减少快速模式的稀疏度，或增加候选区块数量")
    
    return negative_samples


def sample_negative_samples_random(negative_pool, num_negative_samples,
                                  patch_size_km=25, time_steps=10, spatial_resolution_km=1,
                                  target_year_range=None):
    """
    随机采样负样本（当没有土地覆盖数据时使用，使用区块级别策略）
    
    Args:
        negative_pool: 负样本池（包含区块级别候选）
        num_negative_samples: 需要采样的负样本数量
        patch_size_km: 空间块大小
        time_steps: 时间步长
        spatial_resolution_km: 空间分辨率
        target_year_range: 目标年份范围 (start_year, end_year)，用于限制时间窗口不早于开始年份
    
    Returns:
        list: 负样本列表
    """
    print("使用随机采样策略（无土地覆盖数据，区块级别）...")
    
    # 检查是否使用新的区块级别负样本池
    if 'patch_candidates' in negative_pool:
        patch_candidates = negative_pool['patch_candidates']
        events = negative_pool['events']
        target_year_range = target_year_range or negative_pool.get('target_year_range')
        
        if len(patch_candidates) == 0:
            return []
        
        # 随机采样区块和日期
        np.random.seed(42)
        all_patch_keys = list(patch_candidates.keys())
        np.random.shuffle(all_patch_keys)
        
        negative_samples = []
        grid_size = int(patch_size_km / spatial_resolution_km)
        sampled_count = 0
        
        for patch_key in all_patch_keys:
            if sampled_count >= num_negative_samples:
                break
            
            patch_info = patch_candidates[patch_key]
            center_lat = patch_info['center_lat']
            center_lon = patch_info['center_lon']
            sampled_date = patch_info['date']
            patch_bounds = patch_info['patch_bounds']
            
            # 创建特征立方体
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 跳过需要补零的样本
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            # 提取时间窗口内的事件
            events['dtime'] = pd.to_datetime(events['dtime'])
            spatial_mask = (
                (events['lat'] >= patch_bounds['lat_min']) & 
                (events['lat'] <= patch_bounds['lat_max']) &
                (events['lon'] >= patch_bounds['lon_min']) & 
                (events['lon'] <= patch_bounds['lon_max'])
            )
            temporal_mask = (
                (events['dtime'] >= time_window_start) & 
                (events['dtime'] <= time_window_end)
            )
            window_events = events[spatial_mask & temporal_mask].copy()
            
            # 使用新的特征对齐函数
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features 未导入成功，无法生成6通道特征。请检查 code/fire_equality/datamodules/feature_alignment.py")
            # 提取ISO3国家代码（如果可用）
            iso3_code = extract_iso3_from_events(window_events, patch_bounds, events)
            feature_cube = extract_aligned_features(
                spatial_bounds=patch_bounds,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                grid_size=grid_size,
                fire_date=pd.Timestamp(sampled_date).to_pydatetime(),
                fire_year=pd.Timestamp(sampled_date).year,
                iso3=iso3_code,
                data_dir='dataset',
                project=negative_pool.get('project', 'ee-tpan2203-wildfire') if isinstance(negative_pool, dict) else 'ee-tpan2203-wildfire'
            )
            
            # 检查特征是否有效（排除数据缺失区域）
            # 策略：如果3个或以上关键特征全为0，跳过（数据严重缺失）
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            zero_count = sum([
                (fwi_data == 0).all(),
                (ndvi_data == 0).all(),
                (temp_data == 0).all(),
                (wind_data == 0).all()
            ])
            
            # 如果3个或以上关键特征全为0，跳过（数据严重缺失）
            if zero_count >= 3:
                continue
            
            # 检查地理范围：排除高纬度地区（>65°N或<-65°S）
            if abs(center_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(iso3_code, center_lat, center_lon, data_dir='dataset')
            sample = {
                'pixel_lat': center_lat,
                'pixel_lon': center_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 负样本
                'land_cover': None,
                'metadata': {
                    'component_id': None,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'spatial_bounds': patch_bounds,
                    'time_window': (time_window_start, time_window_end),
                    'iso3': iso3_final
                }
            }
            negative_samples.append(sample)
            sampled_count += 1
        
        return negative_samples
    elif 'grid_candidates' in negative_pool:
        grid_candidates = negative_pool['grid_candidates']
        events = negative_pool['events']
        target_year_range = target_year_range or negative_pool.get('target_year_range')
        
        if len(grid_candidates) == 0:
            return []
        
        # 确定研究区域的空间边界
        spatial_bounds = {
            'lat_min': events['lat'].min(),
            'lat_max': events['lat'].max(),
            'lon_min': events['lon'].min(),
            'lon_max': events['lon'].max()
        }
        
        # 随机采样网格和日期
        np.random.seed(42)
        all_grid_keys = list(grid_candidates.keys())
        np.random.shuffle(all_grid_keys)
        
        negative_samples = []
        grid_size = int(patch_size_km / spatial_resolution_km)
        sampled_count = 0
        
        for grid_key in all_grid_keys:
            if sampled_count >= num_negative_samples:
                break
            
            grid_lat, grid_lon = grid_key
            grid_safe_dates = grid_candidates[grid_key]
            
            if len(grid_safe_dates) == 0:
                continue
            
            # 随机选择一个日期
            sampled_date = np.random.choice(grid_safe_dates)
            
            # 检查空间边界
            if not (spatial_bounds['lat_min'] <= grid_lat <= spatial_bounds['lat_max'] and
                   spatial_bounds['lon_min'] <= grid_lon <= spatial_bounds['lon_max']):
                continue
            
            # 创建特征立方体
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 跳过需要补零的样本
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            pixel_spatial_bounds = {
                'lat_min': grid_lat - patch_radius_deg,
                'lat_max': grid_lat + patch_radius_deg,
                'lon_min': grid_lon - patch_radius_deg,
                'lon_max': grid_lon + patch_radius_deg
            }
            
            # 提取时间窗口内的事件
            events['dtime'] = pd.to_datetime(events['dtime'])
            spatial_mask = (
                (events['lat'] >= pixel_spatial_bounds['lat_min']) & 
                (events['lat'] <= pixel_spatial_bounds['lat_max']) &
                (events['lon'] >= pixel_spatial_bounds['lon_min']) & 
                (events['lon'] <= pixel_spatial_bounds['lon_max'])
            )
            temporal_mask = (
                (events['dtime'] >= time_window_start) & 
                (events['dtime'] <= time_window_end)
            )
            window_events = events[spatial_mask & temporal_mask].copy()
            
            # 使用新的特征对齐函数
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features 未导入成功，无法生成6通道特征。请检查 code/fire_equality/datamodules/feature_alignment.py")
            # 提取ISO3国家代码（如果可用）
            iso3_code = extract_iso3_from_events(window_events, pixel_spatial_bounds, events)
            feature_cube = extract_aligned_features(
                spatial_bounds=pixel_spatial_bounds,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                grid_size=grid_size,
                fire_date=pd.Timestamp(sampled_date).to_pydatetime(),
                fire_year=pd.Timestamp(sampled_date).year,
                iso3=iso3_code,
                data_dir='dataset',
                project='ee-tpan2203-wildfire'
            )
            
            # 检查特征是否有效（排除数据缺失区域）
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 如果所有关键特征都全为0，跳过这个样本（数据缺失区域）
            if (fwi_data == 0).all() and (ndvi_data == 0).all() and (temp_data == 0).all() and (wind_data == 0).all():
                continue
            
            # 检查地理范围：排除高纬度地区
            if abs(grid_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(None, grid_lat, grid_lon, data_dir='dataset')
            sample = {
                'pixel_lat': grid_lat,
                'pixel_lon': grid_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 负样本
                'land_cover': None,
                'metadata': {
                    'component_id': None,
                    'center_lat': grid_lat,
                    'center_lon': grid_lon,
                    'spatial_bounds': pixel_spatial_bounds,
                    'time_window': (time_window_start, time_window_end),
                    'iso3': iso3_final
                }
            }
            negative_samples.append(sample)
            sampled_count += 1
        
        return negative_samples
    else:
        # 兼容旧的负样本池格式（使用safe_dates）
        safe_dates = negative_pool.get('safe_dates', [])
        events = negative_pool['events']
        
        if len(safe_dates) == 0:
            return []
        
        # 确定研究区域的空间边界
        spatial_bounds = {
            'lat_min': events['lat'].min(),
            'lat_max': events['lat'].max(),
            'lon_min': events['lon'].min(),
            'lon_max': events['lon'].max()
        }
        
        # 随机采样日期和位置
        np.random.seed(42)
        sampled_dates = np.random.choice(safe_dates, size=min(num_negative_samples, len(safe_dates)), replace=False)
        
        negative_samples = []
        grid_size = int(patch_size_km / spatial_resolution_km)
        
        for date in sampled_dates:
            # 随机选择一个位置（在研究区域内）
            pixel_lat = np.random.uniform(spatial_bounds['lat_min'], spatial_bounds['lat_max'])
            pixel_lon = np.random.uniform(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
            
            # 创建特征立方体
            time_window_start = pd.Timestamp(date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 跳过需要补零的样本
            
            time_window_end = pd.Timestamp(date) - pd.Timedelta(days=1)
            
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            pixel_spatial_bounds = {
                'lat_min': pixel_lat - patch_radius_deg,
                'lat_max': pixel_lat + patch_radius_deg,
                'lon_min': pixel_lon - patch_radius_deg,
                'lon_max': pixel_lon + patch_radius_deg
            }
            
            # 提取时间窗口内的事件
            events['dtime'] = pd.to_datetime(events['dtime'])
            spatial_mask = (
                (events['lat'] >= pixel_spatial_bounds['lat_min']) & 
                (events['lat'] <= pixel_spatial_bounds['lat_max']) &
                (events['lon'] >= pixel_spatial_bounds['lon_min']) & 
                (events['lon'] <= pixel_spatial_bounds['lon_max'])
            )
            temporal_mask = (
                (events['dtime'] >= time_window_start) & 
                (events['dtime'] <= time_window_end)
            )
            window_events = events[spatial_mask & temporal_mask].copy()
            
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features 未导入成功，无法生成6通道特征。请检查 code/fire_equality/datamodules/feature_alignment.py")
            # 提取ISO3国家代码（如果可用）
            iso3_code = extract_iso3_from_events(window_events, pixel_spatial_bounds, events)
            feature_cube = extract_aligned_features(
                spatial_bounds=pixel_spatial_bounds,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                grid_size=grid_size,
                fire_date=pd.Timestamp(date).to_pydatetime() if 'date' in locals() else pd.Timestamp(time_window_end + pd.Timedelta(days=1)).to_pydatetime(),
                fire_year=(pd.Timestamp(date).year if 'date' in locals() else pd.Timestamp(time_window_end + pd.Timedelta(days=1)).year),
                iso3=iso3_code,
                data_dir='dataset',
                project=negative_pool.get('project', 'ee-tpan2203-wildfire') if isinstance(negative_pool, dict) else 'ee-tpan2203-wildfire'
            )
            
            # 检查特征是否有效（排除数据缺失区域）
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 如果所有关键特征都全为0，跳过这个样本（数据缺失区域）
            if (fwi_data == 0).all() and (ndvi_data == 0).all() and (temp_data == 0).all() and (wind_data == 0).all():
                continue
            
            # 检查地理范围：排除高纬度地区
            if abs(pixel_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(iso3_code, pixel_lat, pixel_lon, data_dir='dataset')
            sample = {
                'pixel_lat': pixel_lat,
                'pixel_lon': pixel_lon,
                'pixel_date': date,
                'features': feature_cube,
                'target': 0,  # 负样本
                'land_cover': None,
                'metadata': {
                    'component_id': None,
                    'center_lat': pixel_lat,
                    'center_lon': pixel_lon,
                    'spatial_bounds': pixel_spatial_bounds,
                    'time_window': (time_window_start, time_window_end),
                    'iso3': iso3_final
                }
            }
            negative_samples.append(sample)
        
        return negative_samples


def create_pixel_level_binary_classification_dataset(data_directory, output_path=None, config=None):
    """
    创建像素级二分类数据集：预测某像素点是否会在未来一天发生大火
    
    任务定义：
    - 正样本（y_t = 1）：所有在当天起火并最终形成大火的像素（严重火灾）
    - 负样本（y_t = 0）：从无火灾发生的日期中抽取，按土地覆盖类型分层采样
    
    Args:
        data_directory: FireTracks数据目录
        output_path: 处理后数据的保存路径 (可选)
        config: 处理配置字典，包含：
            - patch_size_km: 空间块大小 (km)，默认25
            - time_steps: 时间步长 (天)，默认10
            - target_years: 目标年份范围，默认(2002, 2020)
            - batch_size: 批次大小，默认32
            - neg_pos_ratio: 负样本与正样本的比例，默认2.0
            - max_samples: 最大处理样本数（用于测试，None表示处理全部）
    
    Returns:
        dict: 包含以下键的字典：
            - 'dataset': FireTracksDataset实例
            - 'dataloader': DataLoader实例
            - 'config': 使用的配置字典
            - 'preprocessed_data': 预处理后的数据字典
            - 'positive_samples': 正样本列表
            - 'negative_samples': 负样本列表
    """
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:3688","message":"create_pixel_level_binary_classification_dataset entry","data":{"data_directory":data_directory,"config":str(config)[:200] if config else None},"timestamp":int(time_module.time()*1000)}) + '\n')
    except:
        pass
    # #endregion
    if config is None:
        config = {
            'patch_size_km': 25,
            'time_steps': 10,
            'target_years': (2002, 2020),
            'batch_size': 32,
            'neg_pos_ratio': 2.0,  # 负样本是正样本的2倍
            'max_samples': None
        }
    
    print("="*60)
    print("🚀 启动像素级二分类数据集创建流水线...")
    print("="*60)
    
    # 1. 加载数据
    print("\n[步骤1/6] 加载FireTracks数据...")
    start_year, end_year = config['target_years']
    # 只加载目标年份的数据，跳过时间窗口不完整的组件
    events_year_range = config['target_years']
    print(f"   Events 年份范围: {events_year_range}")
    
    # #region agent log
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:3730","message":"Calling load_firetracks_dataset","data":{"data_directory":data_directory,"year_range":events_year_range},"timestamp":int(time_module.time()*1000)}) + '\n')
    except:
        pass
    # #endregion
    
    datasets = load_firetracks_dataset(
        data_directory,
        year_range=events_year_range
    )
    
    if datasets is None:
        return None
    
    # 重新加载components，只包含目标年份
    if 'components' in datasets:
        print(f"   重新加载 components，仅包含目标年份: {config['target_years']}")
        import os
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year+1}-01-01"
        try:
            datasets['components'] = pd.read_hdf(
                f'{data_directory}/cp.h5',
                where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
            )
            print(f"   ✅ Components: {len(datasets['components']):,} 行")
        except Exception as e:
            print(f"   ⚠️  重新加载 components 失败，使用已加载的数据")
    
    print(f"   ✅ FireTracks数据加载成功!")
    print(f"   - 活跃火事件: {len(datasets.get('events', [])):,} 条")
    print(f"   - 时空火灾组件: {len(datasets.get('components', [])):,} 个")
    
    # 2. 数据预处理
    print("\n[步骤2/6] 数据预处理...")
    import time
    start_time = time.time()
    preprocessed_data = preprocess_firetracks_data(
        datasets,
        target_year_range=config['target_years']
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  预处理耗时: {elapsed_time:.1f} 秒")
    
    # 3. 创建或加载像素级正样本
    print("\n[步骤3/7] 创建或加载像素级正样本...")
    
    # 重置GDP警告计数器（确保每次运行都从0开始）
    try:
        from .feature_alignment import reset_gdp_warning_count
        reset_gdp_warning_count()
    except (ImportError, AttributeError):
        try:
            from code.fire_equality.datamodules.feature_alignment import reset_gdp_warning_count
            reset_gdp_warning_count()
        except (ImportError, AttributeError):
            pass  # 如果函数不存在，忽略
    
    # 重置性能统计（缓存命中率等）
    try:
        from .feature_cache import clear_cache, get_cache_stats
        from .netcdf_cache import clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
        # 在开始处理前清除NetCDF缓存，避免之前运行留下的无效数据集
        netcdf_stats_before = get_netcdf_cache_stats()
        if netcdf_stats_before['cached_datasets'] > 0:
            print(f"   🧹 清除之前的NetCDF缓存 ({netcdf_stats_before['cached_datasets']} 个数据集)...")
            clear_netcdf_cache()
            print(f"   ✅ NetCDF缓存已清除")
        # 不清空特征缓存，保留之前计算的特征以提高性能
    except (ImportError, AttributeError):
        pass
    
    # 生成缓存文件路径（保存到dataset目录）
    cache_dir = config.get('cache_dir', 'dataset')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件名（基于配置参数）
    cache_filename = f"positive_samples_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}_t{config['time_steps']}"
    if config.get('max_samples'):
        cache_filename += f"_max{config['max_samples']}"
    cache_filename += ".pth"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    # 尝试加载缓存
    use_modis_api = config.get('use_modis_api', True)
    project = config.get('gee_project', 'ee-tpan2203-wildfire')
    
    positive_samples, cached_config = load_positive_samples(cache_filepath, verify_config=config)
    
    if positive_samples is None:
        # 缓存不存在或配置不匹配，重新创建
        print("   缓存不存在或配置不匹配，重新创建正样本...")
        start_time = time.time()
        positive_samples = create_pixel_level_positive_samples(
            preprocessed_data,
            patch_size_km=config['patch_size_km'],
            time_steps=config['time_steps'],
            max_samples=config.get('max_samples', None),
            target_year_range=config['target_years'],  # 传递目标年份范围
            use_modis_api=use_modis_api,
            project=project,
            parallel_workers=config.get('parallel_workers', 2)  # 并行工作线程数，默认2
        )
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  创建正样本耗时: {elapsed_time:.1f} 秒")
        print(f"   ✅ 正样本数量: {len(positive_samples):,}")
        
        # 显示性能统计
        try:
            from .feature_cache import get_cache_stats
            from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
            cache_stats = get_cache_stats()
            netcdf_stats = get_netcdf_cache_stats()
            if cache_stats['hits'] + cache_stats['misses'] > 0:
                print(f"   📊 特征缓存统计: 命中率 {cache_stats['hit_rate']:.1%} ({cache_stats['hits']} 命中 / {cache_stats['misses']} 未命中), 缓存大小: {cache_stats['cache_size']} 个特征")
            if netcdf_stats['cached_datasets'] > 0:
                print(f"   📊 NetCDF缓存: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']} 个数据集已缓存")
        except (ImportError, AttributeError):
            pass
        
        # 保存到缓存
        if len(positive_samples) > 0:
            print(f"\n💾 保存正样本到缓存: {cache_filepath}")
            save_positive_samples(positive_samples, config, cache_filepath)
    else:
        print(f"   ✅ 从缓存加载正样本，跳过创建步骤")
        # 注意：如果缓存中的样本数量超过了max_samples限制，load_positive_samples已经截断了
        print(f"   ✅ 正样本数量: {len(positive_samples):,}")
        if config.get('max_samples') and len(positive_samples) > config['max_samples']:
            print(f"   ⚠️  警告: 缓存中的样本数量 ({len(positive_samples):,}) 超过了max_samples限制 ({config['max_samples']})")
            print(f"   这不应该发生，因为load_positive_samples应该已经截断了")
    
    if len(positive_samples) == 0:
        print("❌ 未能创建正样本")
        return None
    
    # 4. 分析正样本土地覆盖分布（用于负样本池创建和采样）
    print("\n[步骤4/7] 分析正样本土地覆盖分布...")
    positive_lc_distribution = analyze_positive_sample_land_cover_distribution(positive_samples)
    
    # 获取正样本的土地覆盖类型（用于直接搜索 MODIS）
    positive_lc_types = None
    if positive_lc_distribution and positive_lc_distribution.get('counts'):
        positive_lc_types = list(positive_lc_distribution['counts'].keys())
        print(f"   🎯 将使用正样本土地覆盖类型 {sorted(positive_lc_types)} 直接搜索 MODIS 数据")
    
    # 5. 创建或加载负样本池（区块级别）
    print("\n[步骤5/7] 创建或加载区块级别负样本池...")
    
    # 生成负样本池缓存文件路径
    negative_pool_cache_filename = f"negative_pool_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}"
    fast_mode = config.get('fast_mode', False) or (config.get('max_samples') is not None)
    if fast_mode:
        negative_pool_cache_filename += "_fast"
    negative_pool_cache_filename += ".pth"
    negative_pool_cache_filepath = os.path.join(cache_dir, negative_pool_cache_filename)
    
    # 准备用于验证的配置（包含fast_mode）
    pool_config = {
        'patch_size_km': config['patch_size_km'],
        'target_years': config['target_years'],
        'fast_mode': fast_mode
    }
    
    # 尝试加载缓存
    negative_pool, cached_pool_config = load_negative_pool(negative_pool_cache_filepath, verify_config=pool_config)
    
    if negative_pool is None:
        # 缓存不存在或配置不匹配，重新创建
        print("   缓存不存在或配置不匹配，重新创建负样本池...")
        start_time = time.time()
        # 快速模式：如果设置了max_samples（测试模式）或显式设置了fast_mode，启用快速模式
        if fast_mode:
            if config.get('max_samples') is not None:
                print("   ⚡ 检测到max_samples设置，启用快速模式（用于测试）")
            else:
                print("   ⚡ 快速模式已启用")
        negative_pool = create_negative_sample_pool(
            preprocessed_data,
            config['target_years'],
            patch_size_km=config['patch_size_km'],
            events_lc=preprocessed_data.get('events_lc', None),
            spatial_resolution_km=1,
            fast_mode=fast_mode,
            positive_lc_types=positive_lc_types,  # 传递正样本土地覆盖类型
            use_modis_api=use_modis_api,  # 使用 MODIS API
            project=project  # GEE 项目名称
        )
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  创建负样本池耗时: {elapsed_time:.1f} 秒")
        
        # 保存到缓存
        if len(negative_pool.get('patch_candidates', {})) > 0:
            print(f"\n💾 保存负样本池到缓存: {negative_pool_cache_filepath}")
            save_negative_pool(negative_pool, pool_config, negative_pool_cache_filepath)
    else:
        print(f"   ✅ 从缓存加载负样本池，跳过创建步骤")
        print(f"   ✅ 候选区块数: {len(negative_pool.get('patch_candidates', {})):,}")
        # 确保负样本池中有events数据（如果缓存中没有，从preprocessed_data中获取）
        if 'events' not in negative_pool or negative_pool['events'] is None:
            negative_pool['events'] = preprocessed_data.get('events')
            print(f"   💡 从preprocessed_data补充events数据")
    
    # 6. 采样负样本（按正样本的土地覆盖分布进行分层采样）
    print("\n[步骤6/7] 采样负样本...")
    start_time = time.time()
    num_negative_samples = int(len(positive_samples) * config['neg_pos_ratio'])
    print(f"   目标负样本数量: {num_negative_samples:,} (正样本的 {config['neg_pos_ratio']:.1f} 倍)")
    
    # 生成检查点文件路径（用于断点恢复）
    checkpoint_filename = f"negative_samples_checkpoint_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}.pth"
    checkpoint_filepath = os.path.join(cache_dir, checkpoint_filename)
    
    negative_samples = sample_negative_samples_by_land_cover(
        negative_pool,
        positive_lc_distribution,  # 传递正样本土地覆盖分布
        num_negative_samples,
        patch_size_km=config['patch_size_km'],
        time_steps=config['time_steps'],
        spatial_resolution_km=1,
        target_year_range=config['target_years'],
        use_modis_api=use_modis_api,  # 使用MODIS API
        project=project,  # GEE项目名称
        checkpoint_file=checkpoint_filepath,  # 检查点文件路径
        checkpoint_interval=100  # 每100个样本保存一次检查点
    )
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  创建负样本耗时: {elapsed_time:.1f} 秒")
    print(f"   ✅ 负样本数量: {len(negative_samples):,}")
    
    # 显示性能统计
    try:
        from .feature_cache import get_cache_stats
        from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
        cache_stats = get_cache_stats()
        netcdf_stats = get_netcdf_cache_stats()
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            print(f"   📊 特征缓存统计: 命中率 {cache_stats['hit_rate']:.1%} ({cache_stats['hits']} 命中 / {cache_stats['misses']} 未命中), 缓存大小: {cache_stats['cache_size']} 个特征")
        if netcdf_stats['cached_datasets'] > 0:
            print(f"   📊 NetCDF缓存: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']} 个数据集已缓存")
    except (ImportError, AttributeError):
        pass
    
    # 7. 合并正负样本并创建数据集
    print("\n[步骤7/7] 创建PyTorch数据集...")
    all_samples = positive_samples + negative_samples
    
    # 更新配置
    config['target_type'] = 'binary_classification'  # 二分类任务
    
    # 创建数据集（需要修改FireTracksDataset以支持二分类）
    dataset = FireTracksDataset(
        all_samples,
        target_type='binary_classification'
    )
    
    # 创建数据加载器
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print("\n✅ 像素级二分类数据集创建完成!")
    print(f"   - 总样本数: {len(dataset):,}")
    print(f"   - 正样本数: {len(positive_samples):,}")
    print(f"   - 负样本数: {len(negative_samples):,}")
    print(f"   - 正负样本比例: 1:{len(negative_samples)/len(positive_samples):.2f}")
    if len(dataset) > 0:
        sample_features, sample_target = dataset[0]
        print(f"   - 输入维度: {sample_features.shape}")
        print(f"   - 目标类型: 二分类 (0/1)")
        print(f"   - 批次数量: {len(dataloader):,} 批次")
    
    # 保存数据
    if output_path:
        try:
            torch.save({
                'spatiotemporal_samples': all_samples,
                'config': config,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples,
                'positive_lc_distribution': positive_lc_distribution,
                'metadata': {
                    'total_samples': len(dataset),
                    'positive_samples': len(positive_samples),
                    'negative_samples': len(negative_samples),
                    'input_shape': dataset[0][0].shape if len(dataset) > 0 else None,
                    'feature_channels': ['FWI', 'VPD', 'NDVI', 'population', 'GDP', 'land_cover', 'max_temp', 'max_wind']
                }
            }, output_path)
            print(f"✅ 数据已保存至: {output_path}")
        except Exception as e:
            print(f"⚠️  数据保存失败: {e}")
    
    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'config': config,
        'preprocessed_data': preprocessed_data,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'positive_lc_distribution': positive_lc_distribution
    }


# 执行测试

if __name__ == "__main__":
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:4038","message":"Script entry point","data":{},"timestamp":int(time_module.time()*1000)}) + '\n')
    except Exception as log_err:
        pass  # 忽略日志错误
    # #endregion
    import argparse
    
    parser = argparse.ArgumentParser(description='FireTracks数据处理流水线（像素级二分类任务）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数（用于测试，None表示处理全部）')
    parser.add_argument('--target_years', type=int, nargs=2, default=[2017, 2018],
                        metavar=('START', 'END'),
                        help='目标年份范围，例如: --target_years 2017 2018')
    parser.add_argument('--patch_size_km', type=int, default=25,
                        help='空间块大小 (km)，默认25')
    parser.add_argument('--time_steps', type=int, default=10,
                        help='时间步长 (天)，默认10')
    parser.add_argument('--neg_pos_ratio', type=float, default=2.0,
                        help='负样本与正样本的比例，默认2.0')
    parser.add_argument('--data_directory', type=str, default='dataset/firetracks_data',
                        help='FireTracks数据目录路径，默认dataset/firetracks_data')
    parser.add_argument('--output_path', type=str, default='dataset/processed_firetracks_pixel_binary.pth',
                        help='处理后数据的保存路径，默认dataset/processed_firetracks_pixel_binary.pth')
    parser.add_argument('--cache_dir', type=str, default='dataset',
                        help='缓存目录，默认dataset')
    parser.add_argument('--parallel_workers', type=int, default=2,
                        help='并行工作线程数，默认2')
    
    args = parser.parse_args()
    
    # 构建配置字典
    config = {
        'use_modis_api': True,
        'gee_project': 'ee-tpan2203-wildfire',
        'cache_dir': args.cache_dir,
        'patch_size_km': args.patch_size_km,
        'time_steps': args.time_steps,
        'target_years': tuple(args.target_years),
        'batch_size': 32,
        'neg_pos_ratio': args.neg_pos_ratio,
        'max_samples': args.max_samples,
        'parallel_workers': args.parallel_workers
    }
    
    print("="*60)
    print("FireTracks数据处理流水线")
    print("="*60)
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:4094","message":"Calling create_pixel_level_binary_classification_dataset","data":{"data_directory":args.data_directory,"max_samples":args.max_samples},"timestamp":int(time_module.time()*1000)}) + '\n')
    except:
        pass
    # #endregion
    
    # 运行处理流水线
    result = create_pixel_level_binary_classification_dataset(
        data_directory=args.data_directory,
        output_path=args.output_path,
        config=config
    )
    
    if result is not None:
        # 测试数据加载器
        dataloader = result['dataloader']
        features, targets = next(iter(dataloader))
        
        print(f"\n📊 数据加载器测试:")
        print(f"   - Batch大小: {features.shape[0]}")
        print(f"   - 输入维度: {features.shape}")
        print(f"   - 目标维度: {targets.shape}")
        print(f"   - 目标类型: 二分类 (0=负样本, 1=正样本)")
        print(f"   - 数据类型: {features.dtype}")
        
        # 统计批次中的正负样本数量
        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()
        print(f"   - 批次中正样本数: {positive_count}")
        print(f"   - 批次中负样本数: {negative_count}")
        
        print(f"\n✅ 数据处理完成！")
    else:
        print(f"\n❌ 数据处理失败")
