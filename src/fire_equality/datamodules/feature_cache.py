"""

for,
"""
import hashlib
import pickle
import os
from typing import Optional, Dict, Tuple, Union
import numpy as np
from datetime import datetime
import threading

# ()
_cache: Dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()
_cache_stats = {'hits': 0, 'misses': 0}


def _make_cache_key(spatial_bounds: dict, time_window_start: Optional[datetime], 
                    time_window_end: Optional[datetime], grid_size: int, 
                    feature_type: str, year: int, 
                    extra_params: Optional[dict] = None) -> str:
    """
    
    
    Args:
        spatial_bounds: 
        time_window_start: (None)
        time_window_end: (None)
        grid_size: 
        feature_type: ('FWI', 'VPD', 'max_temp', 'max_wind', 'NDVI', 'population', 'GDP', 'land_cover')
        year: 
        extra_params: (GDPiso3,forGDP)
    
    Returns:
        
    """
    # 
    key_parts = [
        f"{feature_type}",
        f"{year}",
        f"{grid_size}",
        f"{spatial_bounds['lat_min']:.6f}_{spatial_bounds['lat_max']:.6f}",
        f"{spatial_bounds['lon_min']:.6f}_{spatial_bounds['lon_max']:.6f}",
    ]
    
    #,
    if time_window_start is not None and time_window_end is not None:
        key_parts.append(f"{time_window_start.strftime('%Y%m%d')}_{time_window_end.strftime('%Y%m%d')}")
    
    # (GDPiso3),
    if extra_params is not None:
        for key, value in sorted(extra_params.items()):
            if value is not None:
                key_parts.append(f"{key}_{str(value)}")
    
    key_str = "_".join(key_parts)
    # MD5
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_feature(spatial_bounds: dict, time_window_start: Optional[datetime],
                      time_window_end: Optional[datetime], grid_size: int,
                      feature_type: str, year: int,
                      extra_params: Optional[dict] = None) -> Optional[np.ndarray]:
    """
    
    
    Args:
        spatial_bounds: 
        time_window_start: (None)
        time_window_end: (None)
        grid_size: 
        feature_type: 
        year: 
        extra_params: (GDPiso3)
    
    Returns:
       ,;None
    """
    cache_key = _make_cache_key(spatial_bounds, time_window_start, 
                                time_window_end, grid_size, feature_type, year, extra_params)
    
    with _cache_lock:
        if cache_key in _cache:
            _cache_stats['hits'] += 1
            return _cache[cache_key].copy()  # 
        else:
            _cache_stats['misses'] += 1
            return None


def cache_feature(spatial_bounds: dict, time_window_start: Optional[datetime],
                 time_window_end: Optional[datetime], grid_size: int,
                 feature_type: str, year: int, feature_data: np.ndarray,
                 extra_params: Optional[dict] = None):
    """
    
    
    Args:
        spatial_bounds: 
        time_window_start: (None)
        time_window_end: (None)
        grid_size: 
        feature_type: 
        year: 
        feature_data: 
        extra_params: (GDPiso3)
    """
    cache_key = _make_cache_key(spatial_bounds, time_window_start,
                                time_window_end, grid_size, feature_type, year, extra_params)
    
    with _cache_lock:
        _cache[cache_key] = feature_data.copy()  # 


def clear_cache():
    """"""
    with _cache_lock:
        _cache.clear()
        _cache_stats['hits'] = 0
        _cache_stats['misses'] = 0


def get_cache_stats() -> Dict[str, int]:
    """"""
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
    """"""
    with _cache_lock:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'cache': _cache,
                'stats': _cache_stats
            }, f)


def load_cache_from_disk(cache_path: str):
    """load"""
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
        print(f"⚠️  load: {e}")
        return False

