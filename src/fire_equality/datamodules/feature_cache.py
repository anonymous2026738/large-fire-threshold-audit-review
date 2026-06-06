"""
特征缓存模块
用于缓存已提取的特征，避免重复计算，大幅提升样本生成速度
"""
import hashlib
import pickle
import os
from typing import Optional, Dict, Tuple, Union
import numpy as np
from datetime import datetime
import threading

# 全局缓存字典（线程安全）
_cache: Dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()
_cache_stats = {'hits': 0, 'misses': 0}


def _make_cache_key(spatial_bounds: dict, time_window_start: Optional[datetime], 
                    time_window_end: Optional[datetime], grid_size: int, 
                    feature_type: str, year: int, 
                    extra_params: Optional[dict] = None) -> str:
    """
    生成缓存键
    
    Args:
        spatial_bounds: 空间边界字典
        time_window_start: 时间窗口开始日期（对于静态特征可以为None）
        time_window_end: 时间窗口结束日期（对于静态特征可以为None）
        grid_size: 网格大小
        feature_type: 特征类型（'FWI', 'VPD', 'max_temp', 'max_wind', 'NDVI', 'population', 'GDP', 'land_cover'等）
        year: 年份
        extra_params: 额外的参数（例如GDP的iso3，用于区分不同国家的GDP）
    
    Returns:
        缓存键字符串
    """
    # 将空间边界转换为可哈希的字符串
    key_parts = [
        f"{feature_type}",
        f"{year}",
        f"{grid_size}",
        f"{spatial_bounds['lat_min']:.6f}_{spatial_bounds['lat_max']:.6f}",
        f"{spatial_bounds['lon_min']:.6f}_{spatial_bounds['lon_max']:.6f}",
    ]
    
    # 对于时间序列特征，添加时间窗口
    if time_window_start is not None and time_window_end is not None:
        key_parts.append(f"{time_window_start.strftime('%Y%m%d')}_{time_window_end.strftime('%Y%m%d')}")
    
    # 对于有额外参数的特征（如GDP的iso3），添加额外参数
    if extra_params is not None:
        for key, value in sorted(extra_params.items()):
            if value is not None:
                key_parts.append(f"{key}_{str(value)}")
    
    key_str = "_".join(key_parts)
    # 使用MD5生成固定长度的键
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_feature(spatial_bounds: dict, time_window_start: Optional[datetime],
                      time_window_end: Optional[datetime], grid_size: int,
                      feature_type: str, year: int,
                      extra_params: Optional[dict] = None) -> Optional[np.ndarray]:
    """
    从缓存中获取特征
    
    Args:
        spatial_bounds: 空间边界字典
        time_window_start: 时间窗口开始日期（对于静态特征可以为None）
        time_window_end: 时间窗口结束日期（对于静态特征可以为None）
        grid_size: 网格大小
        feature_type: 特征类型
        year: 年份
        extra_params: 额外的参数（例如GDP的iso3）
    
    Returns:
        如果缓存命中，返回特征数组；否则返回None
    """
    cache_key = _make_cache_key(spatial_bounds, time_window_start, 
                                time_window_end, grid_size, feature_type, year, extra_params)
    
    with _cache_lock:
        if cache_key in _cache:
            _cache_stats['hits'] += 1
            return _cache[cache_key].copy()  # 返回副本避免意外修改
        else:
            _cache_stats['misses'] += 1
            return None


def cache_feature(spatial_bounds: dict, time_window_start: Optional[datetime],
                 time_window_end: Optional[datetime], grid_size: int,
                 feature_type: str, year: int, feature_data: np.ndarray,
                 extra_params: Optional[dict] = None):
    """
    将特征存入缓存
    
    Args:
        spatial_bounds: 空间边界字典
        time_window_start: 时间窗口开始日期（对于静态特征可以为None）
        time_window_end: 时间窗口结束日期（对于静态特征可以为None）
        grid_size: 网格大小
        feature_type: 特征类型
        year: 年份
        feature_data: 特征数据数组
        extra_params: 额外的参数（例如GDP的iso3）
    """
    cache_key = _make_cache_key(spatial_bounds, time_window_start,
                                time_window_end, grid_size, feature_type, year, extra_params)
    
    with _cache_lock:
        _cache[cache_key] = feature_data.copy()  # 存储副本


def clear_cache():
    """清空缓存"""
    with _cache_lock:
        _cache.clear()
        _cache_stats['hits'] = 0
        _cache_stats['misses'] = 0


def get_cache_stats() -> Dict[str, int]:
    """获取缓存统计信息"""
    with _cache_lock:
        total = _cache_stats['hits'] + _cache_stats['misses']
        hit_rate = _cache_stats['hits'] / total if total > 0 else 0.0
        return {
            'hits': _cache_stats['hits'],
            'misses': _cache_stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(_cache)
        }


def save_cache_to_disk(cache_path: str):
    """将缓存保存到磁盘"""
    with _cache_lock:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'cache': _cache,
                'stats': _cache_stats
            }, f)


def load_cache_from_disk(cache_path: str):
    """从磁盘加载缓存"""
    if not os.path.exists(cache_path):
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            with _cache_lock:
                _cache.update(data.get('cache', {}))
                _cache_stats.update(data.get('stats', {'hits': 0, 'misses': 0}))
        return True
    except Exception as e:
        print(f"⚠️  加载缓存失败: {e}")
        return False

