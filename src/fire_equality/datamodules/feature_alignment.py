"""

for

:
0: FWI ()
1: VPD ()
2: NDVI ()
3: population (,1km²,:/km²)
4: GDP ()
5: land_cover ()
6: max_temp (ERA5 2)
7: max_wind (ERA5 10,:m/s)
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

# 
try:
    from .feature_cache import get_cached_feature, cache_feature, get_cache_stats
except ImportError:
    try:
        from code.fire_equality.datamodules.feature_cache import get_cached_feature, cache_feature, get_cache_stats
    except ImportError:
        #,()
        def get_cached_feature(*args, **kwargs):
            return None
        def cache_feature(*args, **kwargs):
            pass
        def get_cache_stats():
            return {'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'cache_size': 0}

# NetCDF
try:
    from .netcdf_cache import get_netcdf_dataset, clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
    HAS_NETCDF_CACHE = True
except ImportError:
    try:
        from code.fire_equality.datamodules.netcdf_cache import get_netcdf_dataset, clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
        HAS_NETCDF_CACHE = True
    except ImportError:
        #,load()
        HAS_NETCDF_CACHE = False
        def get_netcdf_dataset(filepath, engine=None):
            # 
            engines_to_try = [engine] if engine else ['netcdf4', 'h5netcdf', None]
            for eng in engines_to_try:
                try:
                    if eng:
                        return xr.open_dataset(filepath, engine=eng)
                    else:
                        return xr.open_dataset(filepath)
                except:
                    continue
            raise RuntimeError(f"NetCDF {filepath}")
        def clear_netcdf_cache():
            pass
        def get_netcdf_cache_stats():
            return {'cached_datasets': 0, 'max_cache_size': 0}

# get_modis_landcover
try:
    from .get_modis_landcover import initialize_gee, HAS_EE
    HAS_GEE = HAS_EE
except ImportError:
    try:
        #,
        from code.fire_equality.datamodules.get_modis_landcover import initialize_gee, HAS_EE
        HAS_GEE = HAS_EE
    except ImportError:
        # :sys.path;, ee.Initialize
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
                print("⚠️  GEE,land_cover")
                def initialize_gee(project: str = None, silent: bool = False):  # type: ignore
                    if silent:
                        return False
                    raise ImportError("GEE")


# ============================================================================
# :fordNBR,
# ============================================================================

# dNBR:key = (fire_date_str, bounds_hash), value = dNBR
_dnbr_cache = {}
_dnbr_cache_max_size = 100  #,


def _get_bounds_hash(spatial_bounds, grid_size):
    """,for"""
    import hashlib
    bounds_str = f"{spatial_bounds['lat_min']:.4f},{spatial_bounds['lat_max']:.4f},{spatial_bounds['lon_min']:.4f},{spatial_bounds['lon_max']:.4f},{grid_size}"
    return hashlib.md5(bounds_str.encode()).hexdigest()


def _clear_dnbr_cache():
    """dNBR"""
    global _dnbr_cache
    _dnbr_cache.clear()


# ============================================================================
# :NetCDF
# ============================================================================

def get_coord_names(ds):
    """
    NetCDF
    
    Args:
        ds: xarray Dataset
    
    Returns:
        dict:  'lat', 'lon', 'time' 
    """
    coords = {}
    dims = list(ds.dims.keys())
    coord_vars = list(ds.coords.keys())
    
    # 
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'lat' in name_lower:
            coords['lat'] = name
            break
    
    # 
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'lon' in name_lower:
            coords['lon'] = name
            break
    
    # 
    for name in coord_vars + dims:
        name_lower = name.lower()
        if 'time' in name_lower:
            coords['time'] = name
            break
    
    #,
    if 'lat' not in coords:
        # 
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


def get_time_index(time_coords, current_date, filepath, feature_name=""):
    """
    
    
    Args:
        time_coords: 
        current_date: (datetime)
        filepath: (for)
        feature_name: (for)
    
    Returns:
        int: 
    
    Raises:
        ValueError: 
    """
    if len(time_coords) == 0:
        raise ValueError(f"{feature_name} {filepath},")
    
    first_time_value = time_coords[0]
    
    try:
        if isinstance(first_time_value, (int, np.integer)) or np.issubdtype(type(first_time_value), np.integer):
            # (day of year)
            day_of_year = current_date.timetuple().tm_yday
            time_idx = np.argmin(np.abs(time_coords - day_of_year))
        else:
            # datetime
            time_diffs = [pd.Timestamp(ts).to_pydatetime() - current_date for ts in time_coords]
            if len(time_diffs) == 0:
                raise ValueError(f"{feature_name} {filepath} ")
            time_idx = np.argmin(np.abs(time_diffs))
        
        # 
        if time_idx >= len(time_coords):
            time_idx = len(time_coords) - 1
        
        return time_idx
    except (IndexError, ValueError) as e:
        raise ValueError(f"{feature_name} {filepath} : {e}")


# ============================================================================
# :
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
    project: str = 'ee-your-gee-project',
    max_retries: int = 3
) -> np.ndarray:
    """
    
    
    :NDVIGIMMS-3G+,MOD13A2GEE API.
    GIMMS-3G+,FileNotFoundError.
    
    Args:
        spatial_bounds:  {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        time_window_start: 
        time_window_end: 
        grid_size: (25,25km×25km,1km)
        fire_date: (,fordNBR)
        fire_year: 
        country_code: (forGDP)
        iso3: ISO3(forGDP,)
        data_dir: 
        project: GEE
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size, 8]
            : 0-FWI, 1-VPD, 2-NDVI, 3-population(,/km²), 4-GDP, 5-land_cover, 6-max_temp, 7-max_wind
    """
    time_steps = (time_window_end - time_window_start).days + 1
    global _pop_zero_warning_count

    # initialize(8:FWI, VPD, NDVI, population, GDP, land_cover, max_temp, max_wind)
    # NaNinitialize,0
    feature_cube = np.full((time_steps, grid_size, grid_size, 8), np.nan, dtype=np.float32)
    
    #,()
    # print(f"  : {time_steps}  × {grid_size}×{grid_size}  × 8 ")
    
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # :
    def extract_with_retry(extract_func, feature_name, *args, **kwargs):
        """
        
        
        Args:
            extract_func: 
            feature_name: (for)
            *args, **kwargs: passed to
        
        Returns:
            
        
        Raises:
            RuntimeError: max_retries
        """
        retry_delay = 2  # ()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = extract_func(*args, **kwargs)
                #,
                if attempt > 0:
                    print(f"    ✅ {feature_name}( {attempt + 1} )")
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e)
                error_type = type(e).__name__
                
                # 
                #, configuration
                should_retry = True
                if isinstance(e, (FileNotFoundError, ValueError, KeyError)):
                    # configuration,
                    should_retry = False
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    should_retry = False
                
                if not should_retry:
                    #,
                    print(f"    ❌ {feature_name}()")
                    print(f"       : {error_type}")
                    print(f"       : {error_msg[:300]}")
                    raise
                
                # 
                if attempt < max_retries - 1:
                    print(f"    ⚠️  {feature_name} ( {attempt + 1}/{max_retries})")
                    print(f"       : {error_type}")
                    print(f"       : {error_msg[:300]}")
                    print(f"        {retry_delay} ...")
                    time_module.sleep(retry_delay)
                    retry_delay *= 2  # 
                else:
                    # 
                    print(f"    ❌ {feature_name}( {max_retries},)")
                    print(f"       : {error_type}")
                    print(f"       : {error_msg[:300]}")
                    raise RuntimeError(f"{feature_name}: {max_retries},.: {error_type}, : {error_msg[:200]}")
        
        # 
        raise RuntimeError(f"{feature_name}:")
    
    # :FWIVPD(NetCDF,)
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
    
    # FWI, VPD, ERA5
    # :NetCDFWindows,
    import platform
    use_parallel = platform.system() != 'Windows'
    
    try:
        if use_parallel:
            # Linux/Mac: 
            with ThreadPoolExecutor(max_workers=4) as executor:
                fwi_future = executor.submit(extract_fwi_wrapper)
                vpd_future = executor.submit(extract_vpd_wrapper)
                era5_temp_future = executor.submit(extract_era5_temp_wrapper)
                era5_wind_future = executor.submit(extract_era5_wind_wrapper)
                
                # 
                fwi_data = fwi_future.result()
                vpd_data = vpd_future.result()
                era5_temp_data = era5_temp_future.result()
                era5_wind_data = era5_wind_future.result()
        else:
            # Windows: 
            fwi_data = extract_fwi_wrapper()
            vpd_data = extract_vpd_wrapper()
            era5_temp_data = extract_era5_temp_wrapper()
            era5_wind_data = extract_era5_wind_wrapper()
        
        feature_cube[:, :, :, 0] = fwi_data
        feature_cube[:, :, :, 1] = vpd_data
        feature_cube[:, :, :, 6] = era5_temp_data
        feature_cube[:, :, :, 7] = era5_wind_data
    except RuntimeError as e:
        print(f"    ❌ : {e}")
        raise
    except Exception as e:
        print(f"    ⚠️  FWI/VPD/max_temp/max_wind(): {e}")
        raise
    
    # NDVI(GIMMS-3G+)
    gimms_ndvi_paths = [
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', f'gimms3g_ndvi_{fire_year}.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', 'gimms3g_ndvi.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', f'ndvi_{fire_year}.nc'),
        os.path.join(data_dir, 'NDVI', 'GIMMS3G+', 'ndvi.nc'),
    ]
    
    # GIMMS-3G+()
    use_gimms_ndvi = any(os.path.exists(p) for p in gimms_ndvi_paths)
    if not use_gimms_ndvi:
        # :ndvi3g_geo_v1_X_YYYY_0106.nc4
        gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
        if os.path.exists(gimms_dir):
            # 
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
    
    # GIMMS-3G+,
    if not use_gimms_ndvi:
        gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
        error_msg = (
            f"❌ GIMMS-3G+ NDVI(: {fire_year})\n"
            f"   GIMMS-3G+\n"
            f"   :\n"
            f"   {gimms_dir}\n"
            f"   :\n"
            f"   - ndvi3g_geo_v1_1_{fire_year}_0106.nc4  ndvi3g_geo_v1_1_{fire_year}_0712.nc4 (2002-2014)\n"
            f"   - ndvi3g_geo_v1_2_{fire_year}_0106.nc4  ndvi3g_geo_v1_2_{fire_year}_0712.nc4 (2015-2020)\n"
            f"   - gimms3g_ndvi_{fire_year}.nc  gimms3g_ndvi.nc ()"
        )
        raise FileNotFoundError(error_msg)
    
    # NDVI(GIMMS-3G+)
    try:
        ndvi_data = extract_with_retry(
            extract_ndvi_feature_gimms, "NDVI (GIMMS-3G+)",
            spatial_bounds, time_window_start, time_window_end,
            grid_size, fire_year, data_dir
        )
        feature_cube[:, :, :, 2] = ndvi_data
    except RuntimeError as e:
        print(f"    ❌ : {e}")
        raise
    except Exception as e:
        print(f"    ⚠️  NDVI(): {e}")
        raise
    
    # :(Population, GDP, Land Cover)
    def extract_pop_wrapper():
        try:
            return extract_with_retry(
                extract_population_feature, "Population",
                spatial_bounds, grid_size, fire_year, data_dir
            )
        except Exception as e:
            # 
            error_msg = str(e)
            filepath = os.path.join(data_dir, 'worldpop', f'ppp_{fire_year}_1km_Aggregated.tif')
            print(f"    ⚠️  Population: {error_msg[:200]}")
            print(f"       : {filepath}")
            print(f"       : {os.path.exists(filepath)}")
            raise  #,
    
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
    
    # (Population, GDP, Land Cover)
    # initialize(),
    pop_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    gdp_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    lc_data = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            pop_future = executor.submit(extract_pop_wrapper)
            gdp_future = executor.submit(extract_gdp_wrapper)
            lc_future = executor.submit(extract_lc_wrapper)
            
            #,()
            try:
                pop_data = pop_future.result()
                #  0:(,  WorldPop ),
                if (pop_data == 0).all():
                    with _pop_zero_warning_lock:
                        _pop_zero_warning_count += 1
                        current_count = _pop_zero_warning_count
                    if current_count == 1 or current_count % 100 == 0:
                        logger.debug(
                            "Population  0(: WorldPop ), %d ;"
                            "lat=[%.4f, %.4f], lon=[%.4f, %.4f], year=%s",
                            current_count,
                            spatial_bounds['lat_min'], spatial_bounds['lat_max'],
                            spatial_bounds['lon_min'], spatial_bounds['lon_max'],
                            fire_year,
                        )
            except Exception as e:
                # extract_pop_wrapper
                error_msg = str(e)
                print(f"    ⚠️  Population,0")
                print(f"       : {error_msg[:200]}")
                print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                print(f"       : {fire_year}, data_dir: {data_dir}")
                pop_data = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            try:
                gdp_data = gdp_future.result()
            except Exception as e:
                print(f"    ⚠️  GDP,0: {str(e)[:200]}")
                gdp_data = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            try:
                lc_data = lc_future.result()
            except Exception as e:
                print(f"    ⚠️  Land Cover,0: {str(e)[:200]}")
                lc_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # :(:dNBR,)
        feature_cube[:, :, :, 3] = pop_data[np.newaxis, :, :]
        feature_cube[:, :, :, 4] = gdp_data[np.newaxis, :, :]
        feature_cube[:, :, :, 5] = lc_data[np.newaxis, :, :]
    except Exception as e:
        #,
        print(f"    ⚠️ ,0: {str(e)[:200]}")
        # initialize,
    
    # NaN0(PyTorchNaN,NaN)
    # :NaN,loadNaN0
    #,DataLoader
    feature_cube = np.nan_to_num(feature_cube, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_cube


# ============================================================================
#  0: FWI () - NetCDF
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
    NetCDFFWI(,merge_weather_indices.ipynb)
    
    Args:
        use_cache: (True)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'FWI', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'FWI', f'fire_weather_index_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FWI: {filepath}")
    
    # NetCDF()
    # :,
    ds_from_cache = False
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
            ds_from_cache = True
        except Exception as e:
            #,
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
                raise RuntimeError(f"FWI NetCDF {filepath}: {e}")
    else:
        # ()
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
            error_msg = f"FWI NetCDF {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # ()
    # 
    # :,
    try:
        # dims(load)
        _ = ds.dims
        # data_varscoords
        # :,
        data_vars_keys = list(ds.data_vars.keys())
        coords_keys = list(ds.coords.keys())
        dims_keys = list(ds.dims.keys())
        
        #,
        if len(data_vars_keys) == 0 and len(coords_keys) == 0 and len(dims_keys) == 0:
            raise ValueError(",")
            
    except (AttributeError, RuntimeError, OSError, ValueError) as e:
        # 
        time_steps = (time_window_end - time_window_start).days + 1
        print(f"    ⚠️  FWI {filepath} : {e}")
        print(f"      ,...")
        
        #,
        if ds_from_cache and HAS_NETCDF_CACHE:
            try:
                from .netcdf_cache import clear_netcdf_cache
                # 
                clear_netcdf_cache()
                ds = xr.open_dataset(filepath, engine='netcdf4')
                ds_from_cache = False
                # 
                data_vars_keys = list(ds.data_vars.keys())
                coords_keys = list(ds.coords.keys())
                dims_keys = list(ds.dims.keys())
                print(f"       ✅, {len(data_vars_keys)} : {data_vars_keys}")
            except Exception as e2:
                print(f"       ❌ : {e2}")
                print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        else:
            #,
            try:
                print(f"       ...")
                ds = xr.open_dataset(filepath, engine='netcdf4')
                ds_from_cache = False
                # 
                data_vars_keys = list(ds.data_vars.keys())
                coords_keys = list(ds.coords.keys())
                dims_keys = list(ds.dims.keys())
                print(f"       ✅, {len(data_vars_keys)} : {data_vars_keys}")
            except Exception as e2:
                print(f"       ❌ : {e2}")
                print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
    
    # ()
    if len(data_vars_keys) == 0:
        #,
        time_steps = (time_window_end - time_window_start).days + 1
        print(f"    ⚠️  FWI {filepath} (data_vars)")
        print(f"       ...")
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
                print(f"       ✅, {len(data_vars_keys)} : {data_vars_keys}")
            else:
                print(f"       ❌ ")
                print(f"       : {coords_keys}")
                print(f"       : {dims_keys}")
                print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
                return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        except Exception as e3:
            print(f"       ❌ : {e3}")
            print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
    
    if 'fwi' in ds.data_vars:
        var_name = 'fwi'
    elif 'FWI' in ds.data_vars:
        var_name = 'FWI'
    else:
        # 
        if len(data_vars_keys) == 0:
            #,
            #,
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  FWI {filepath} FWI()")
            print(f"       : {coords_keys}")
            print(f"       : {dims_keys}")
            print(f"       : {data_vars_keys}")
            print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
            # :,
            # 
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars_keys[0]
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # (merge_weather_indices.ipynb)
    # 
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
    
    # 
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 
    if len(longitude) == 0:
        raise ValueError(f"FWI {filepath} (len={len(longitude)})."
                        f": {lon_name}, : {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"FWI {filepath} (len={len(latitude)})."
                        f": {lat_name}, : {dict(ds.dims)}")
    
    # (0-360 vs -180-180)
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"FWI {filepath} : {e}."
                        f": {longitude.shape}, : {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 0-360,
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # (notebook)
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.25
        if len(longitude) == 0:
            raise ValueError(f"FWI {filepath},")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.25
        if len(latitude) == 0:
            raise ValueError(f"FWI {filepath},")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # Affine(for)
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 
    time_coords = ds.coords[time_name].values
    
    try:
        # 
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 
            time_idx = get_time_index(time_coords, current_date, filepath, "max_temp")
            
            # 
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # xarray
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                result[t, i, j] = float(value)
                    except Exception as e:
                        #,0
                        pass
    finally:
        # ()
        # :,
        if ds is not None and not ds_from_cache:
            try:
                ds.close()
            except:
                pass
    
    return result


# ============================================================================
#  1: VPD () - NetCDF
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
    NetCDFVPD(,merge_weather_indices.ipynb)
    
    Args:
        use_cache: (True)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'VPD', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'VPD', f'vapor_pressure_deficit_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VPD: {filepath}")
    
    # NetCDF()
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            #,
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
                raise RuntimeError(f"VPD NetCDF {filepath}: {e}")
    else:
        # 
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
            error_msg = f"VPD NetCDF {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # 
    if 'vpd' in ds.data_vars:
        var_name = 'vpd'
    elif 'VPD' in ds.data_vars:
        var_name = 'VPD'
    else:
        data_vars_list = list(ds.data_vars.keys())
        if len(data_vars_list) == 0:
            #,
            #,
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  VPD {filepath} ()")
            print(f"       : {list(ds.coords.keys())}")
            print(f"       : {list(ds.dims.keys())}")
            print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
            # 
            try:
                ds.close()
            except:
                pass
            # 
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars_list[0]
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # (merge_weather_indices.ipynb)
    # 
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
    
    # 
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 
    if len(longitude) == 0:
        raise ValueError(f"VPD {filepath} (len={len(longitude)})."
                        f": {lon_name}, : {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"VPD {filepath} (len={len(latitude)})."
                        f": {lat_name}, : {dict(ds.dims)}")
    
    # (0-360 vs -180-180)
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"VPD {filepath} : {e}."
                        f": {longitude.shape}, : {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 0-360,
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # (notebook)
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.25
        if len(longitude) == 0:
            raise ValueError(f"VPD {filepath},")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.25
        if len(latitude) == 0:
            raise ValueError(f"VPD {filepath},")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # Affine(for)
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 
    time_coords = ds.coords[time_name].values
    
    try:
        # 
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 
            time_idx = get_time_index(time_coords, current_date, filepath, "VPD")
            
            # 
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # xarray
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                result[t, i, j] = float(value)
                    except Exception as e:
                        #,0
                        pass
    finally:
        # :netcdf_cache,()
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'VPD', year, result)
    
    return result


# ============================================================================
#  6: max_temp (ERA5 2) - NetCDF
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
    NetCDFERA5 2(,extract_fwi_feature)
    
    Args:
        spatial_bounds: 
        time_window_start: 
        time_window_end: 
        grid_size: 
        year: 
        data_dir: 
        use_cache: (True)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'max_temp', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'ERA5', f'era5_land_temp_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5: {filepath}")
    
    # NetCDF()
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            #,
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
                raise RuntimeError(f"ERA5 NetCDF {filepath}: {e}")
    else:
        # 
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
            error_msg = f"ERA5 NetCDF {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # (ERA5t2m_max)
    if 't2m_max' in ds.data_vars:
        var_name = 't2m_max'
    elif 'temp' in ds.data_vars:
        var_name = 'temp'
    elif 'temperature' in ds.data_vars:
        var_name = 'temperature'
    else:
        # (spatial_ref)
        data_vars = [v for v in ds.data_vars.keys() if v != 'spatial_ref']
        if len(data_vars) == 0:
            #,
            #,
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  max_temp {filepath} ()")
            print(f"       : {list(ds.coords.keys())}")
            print(f"       : {list(ds.dims.keys())}")
            print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
            # 
            try:
                ds.close()
            except:
                pass
            # 
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars[0]
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # (extract_fwi_feature)
    # 
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
    
    # 
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 
    if len(longitude) == 0:
        raise ValueError(f"max_temp {filepath} (len={len(longitude)})."
                        f": {lon_name}, : {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"max_temp {filepath} (len={len(latitude)})."
                        f": {lat_name}, : {dict(ds.dims)}")
    
    # (0-360 vs -180-180)
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"max_temp {filepath} : {e}."
                        f": {longitude.shape}, : {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 0-360,
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # (notebook)
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.1
        if len(longitude) == 0:
            raise ValueError(f"max_temp {filepath},")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.1
        if len(latitude) == 0:
            raise ValueError(f"max_temp {filepath},")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # Affine(for)
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 
    time_coords = ds.coords[time_name].values
    
    try:
        # 
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 
            time_idx = get_time_index(time_coords, current_date, filepath, "max_temp")
            
            # 
            for i, lat in enumerate(lat_target):
                for j, lon in enumerate(lon_target):
                    try:
                        # 
                        row, col = transformer.rowcol(lon, lat)
                        row = int(row)
                        col = int(col)
                        
                        # 
                        if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                            # xarray
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                            if not np.isnan(value):
                                # ERA5,
                                # 200,273.15
                                if value > 200:
                                    value = value - 273.15
                                result[t, i, j] = float(value)
                    except Exception as e:
                        #,0
                        pass
    finally:
        # :netcdf_cache,()
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'max_temp', year, result)
    
    return result


# ============================================================================
#  7: max_wind (ERA5 10) - NetCDF
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
    NetCDFERA5 10(:)
    
    Args:
        spatial_bounds: 
        time_window_start: 
        time_window_end: 
        grid_size: 
        year: 
        data_dir: 
        use_cache: (True)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size] (m/s)
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, time_window_start, 
                                   time_window_end, grid_size, 'max_wind', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'ERA5', f'era5_land_wind_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5: {filepath}")
    
    # NetCDF()
    if HAS_NETCDF_CACHE:
        try:
            ds = get_netcdf_dataset(filepath)
        except Exception as e:
            #,
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
                raise RuntimeError(f"ERA5 NetCDF {filepath}: {e}")
    else:
        # 
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
            error_msg = f"ERA5 NetCDF {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
    
    # (ERA5wind_speed_max)
    if 'wind_speed_max' in ds.data_vars:
        var_name = 'wind_speed_max'
    elif 'wind' in ds.data_vars:
        var_name = 'wind'
    elif 'wind_speed' in ds.data_vars:
        var_name = 'wind_speed'
    else:
        # (spatial_ref)
        data_vars = [v for v in ds.data_vars.keys() if v != 'spatial_ref']
        if len(data_vars) == 0:
            #,
            #,
            time_steps = (time_window_end - time_window_start).days + 1
            print(f"    ⚠️  max_wind {filepath} ()")
            print(f"       : {list(ds.coords.keys())}")
            print(f"       : {list(ds.dims.keys())}")
            print(f"       ({time_steps}  × {grid_size}×{grid_size} )")
            # 
            try:
                ds.close()
            except:
                pass
            # 
            return np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
        var_name = data_vars[0]
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # (extract_era5_temp_feature)
    # 
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
    
    # 
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    
    # 
    if len(longitude) == 0:
        raise ValueError(f"max_wind {filepath} (len={len(longitude)})."
                        f": {lon_name}, : {dict(ds.dims)}")
    if len(latitude) == 0:
        raise ValueError(f"max_wind {filepath} (len={len(latitude)})."
                        f": {lat_name}, : {dict(ds.dims)}")
    
    # (0-360 vs -180-180)
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    try:
        lon_min_orig = longitude.min()
        lon_max_orig = longitude.max()
    except Exception as e:
        raise ValueError(f"max_wind {filepath} : {e}."
                        f": {longitude.shape}, : {type(longitude)}")
    
    if lon_max_orig > 180 and lon_min < 0:
        # 0-360,
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 
    if len(longitude) > 1:
        pixel_size_x = float(longitude[1] - longitude[0])
    else:
        pixel_size_x = 0.1
        if len(longitude) == 0:
            raise ValueError(f"max_wind {filepath},")
    
    if len(latitude) > 1:
        pixel_size_y = float(latitude[0] - latitude[1])
    else:
        pixel_size_y = 0.1
        if len(latitude) == 0:
            raise ValueError(f"max_wind {filepath},")
    
    origin_x = float(longitude[0]) - (pixel_size_x / 2)
    origin_y = float(latitude[0]) + (pixel_size_y / 2)
    
    # Affine(for)
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 
    time_coords = ds.coords[time_name].values
    
    try:
        # :
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
        
        # 
        for t in range(time_steps):
            current_date = time_window_start + timedelta(days=t)
            
            # 
            time_idx = get_time_index(time_coords, current_date, filepath, "max_wind")
            
            # :
            valid_indices = (rows >= 0) & (cols >= 0)
            if np.any(valid_indices):
                valid_rows = rows[valid_indices]
                valid_cols = cols[valid_indices]
                
                # xarray()
                try:
                    # 
                    row_indices = xr.DataArray(valid_rows, dims='points')
                    col_indices = xr.DataArray(valid_cols, dims='points')
                    
                    # 
                    values = ds[var_name].isel(**{
                        time_name: time_idx,
                        lat_name: row_indices,
                        lon_name: col_indices
                    }).values
                    
                    # result
                    valid_positions = np.where(valid_mask.flatten())[0]
                    for idx, pos in enumerate(valid_positions):
                        if idx < len(values):
                            value = values[idx]
                            if not np.isnan(value):
                                i_pos = pos // grid_size
                                j_pos = pos % grid_size
                                result[t, i_pos, j_pos] = float(value)
                except Exception as e:
                    #,
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
        # :netcdf_cache,()
        if ds is not None and not HAS_NETCDF_CACHE:
            try:
                ds.close()
            except:
                pass
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'max_wind', year, result)
    
    return result


# ============================================================================
#  2: NDVI () - GIMMS-3G+
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
    GIMMS-3G+ NDVINDVI
    
    Args:
        use_cache: (True)
    
    GIMMS-3G+:
    - :0.0833(8.3km)
    - :(2)
    - :1982-2022
    - :NetCDF
    
    data file path:
    - :data_dir/NDVI/GIMMS3G+/gimms3g_ndvi.nc
    - :data_dir/NDVI/GIMMS3G+/gimms3g_ndvi_{year}.nc
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    # GIMMS-3G+:ndvi3g_geo_v1_1_2002_0106.nc4  ndvi3g_geo_v1_2_2015_0106.nc4
    # 2:0106(1-6)0712(7-12),
    gimms_dir = os.path.join(data_dir, 'NDVI', 'GIMMS3G+')
    
    # 
    file_patterns = [
        f'ndvi3g_geo_v1_1_{year}_0106.nc4',  # v1.1(2002-2014)
        f'ndvi3g_geo_v1_1_{year}_0712.nc4',
        f'ndvi3g_geo_v1_2_{year}_0106.nc4',  # v1.2(2015-2020)
        f'ndvi3g_geo_v1_2_{year}_0712.nc4',
    ]
    
    # ()
    old_patterns = [
        os.path.join(gimms_dir, f'gimms3g_ndvi_{year}.nc'),
        os.path.join(gimms_dir, 'gimms3g_ndvi.nc'),
        os.path.join(gimms_dir, f'ndvi_{year}.nc'),
        os.path.join(gimms_dir, 'ndvi.nc'),
    ]
    
    # 
    found_files = []
    for pattern in file_patterns:
        filepath = os.path.join(gimms_dir, pattern)
        if os.path.exists(filepath):
            found_files.append(filepath)
    
    #,
    if not found_files:
        for path in old_patterns:
            if os.path.exists(path):
                found_files = [path]
                break
    
    if not found_files:
        raise FileNotFoundError(
            f"GIMMS-3G+ NDVI(: {year}).\n"
            f":ndvi3g_geo_v1_X_{year}_0106.nc4  ndvi3g_geo_v1_X_{year}_0712.nc4\n"
            f":{gimms_dir}"
        )
    
    # (01060712),
    if len(found_files) == 2:
        # 
        filepath = found_files  #,
    else:
        # 
        filepath = found_files[0]
    
    # loadNetCDF
    if isinstance(filepath, list):
        # (01060712)
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
        
        # 
        ds = xr.concat(datasets, dim='time')
        # 
        if 'time' in ds.coords:
            ds = ds.sortby('time')
    else:
        #  - NetCDF()
        if HAS_NETCDF_CACHE:
            try:
                ds = get_netcdf_dataset(filepath)
            except Exception:
                #,
                try:
                    ds = xr.open_dataset(filepath, engine='netcdf4')
                except:
                    try:
                        ds = xr.open_dataset(filepath, engine='h5netcdf')
                    except:
                        ds = xr.open_dataset(filepath)
        else:
            # 
            try:
                ds = xr.open_dataset(filepath, engine='netcdf4')
            except:
                try:
                    ds = xr.open_dataset(filepath, engine='h5netcdf')
                except:
                    ds = xr.open_dataset(filepath)
    
    # (GIMMS-3G+)
    possible_var_names = ['NDVI', 'ndvi', 'NDVI_max', 'ndvi_max', 'gimms_ndvi']
    var_name = None
    for vn in possible_var_names:
        if vn in ds.data_vars:
            var_name = vn
            break
    
    if var_name is None:
        #,
        if len(ds.data_vars) > 0:
            var_name = list(ds.data_vars.keys())[0]
        else:
            raise ValueError(f"NDVI.:{list(ds.data_vars.keys())}")
    
    # fill_value(nodata)scale_factor
    fill_value = None
    scale_factor = None
    valid_range = None
    
    if var_name in ds.data_vars:
        var_attrs = ds[var_name].attrs
        # fill_value
        for attr_name in ['_FillValue', 'fill_value', 'missing_value', 'nodata']:
            if attr_name in var_attrs:
                fill_value = float(var_attrs[attr_name])
                break
        
        # scale(GIMMS-3G+scale: x 10000)
        if 'scale' in var_attrs:
            scale_str = str(var_attrs['scale']).lower()
            if 'x' in scale_str or '*' in scale_str:
                #, "x 10000" -> 10000
                import re
                numbers = re.findall(r'\d+', scale_str)
                if numbers:
                    scale_factor = float(numbers[0])
        
        # valid_range
        if 'valid_range' in var_attrs:
            valid_range = var_attrs['valid_range']
            if hasattr(valid_range, '__len__') and len(valid_range) == 2:
                valid_range = [float(valid_range[0]), float(valid_range[1])]
        
        # descriptionnodata(GIMMS-3G+-5000nodata)
        if 'description' in var_attrs:
            desc = str(var_attrs['description']).lower()
            if 'ndvi = -5000' in desc or '-5000' in desc:
                if fill_value is None:
                    fill_value = -5000.0
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 
    longitude = ds.coords[lon_name].values
    latitude = ds.coords[lat_name].values
    time_coords = ds.coords[time_name].values
    
    # (GIMMS-3G+0-360-180-180)
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 0-360,
    if longitude.max() > 180 and lon_min < 0:
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360
        lon_target_360 = np.linspace(lon_min_360, lon_max_360, grid_size)
    else:
        lon_target_360 = lon_target
    
    # 
    pixel_size_x = abs(longitude[1] - longitude[0]) if len(longitude) > 1 else 0.0833
    pixel_size_y = abs(latitude[1] - latitude[0]) if len(latitude) > 1 else 0.0833
    
    origin_x = longitude[0] - (pixel_size_x / 2)
    origin_y = latitude[0] + (pixel_size_y / 2) if latitude[0] > latitude[-1] else latitude[-1] + (pixel_size_y / 2)
    
    # Affine
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # 
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # datetime()
    time_datetimes = []
    for tc in time_coords:
        if isinstance(tc, (int, np.integer)) or np.issubdtype(type(tc), np.integer):
            # 
            if tc < 1000:  # (1-365)
                #,
                try:
                    time_datetimes.append(datetime(year, 1, 1) + timedelta(days=int(tc) - 1))
                except:
                    time_datetimes.append(pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=int(tc) - 1))
            else:
                # (YYYYMMDD)
                try:
                    time_str = str(tc)
                    if len(time_str) == 8:
                        time_datetimes.append(datetime(int(time_str[:4]), int(time_str[4:6]), int(time_str[6:8])))
                    else:
                        time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
                except:
                    time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
        else:
            # datetime
            try:
                time_datetimes.append(pd.Timestamp(tc).to_pydatetime())
            except:
                time_datetimes.append(tc)
    
    # ()
    for t in range(time_steps):
        current_date = time_window_start + timedelta(days=t)
        
        # (GIMMS-3G+,)
        if len(time_datetimes) > 0:
            time_diffs = [abs((td - current_date).days) for td in time_datetimes]
            time_idx = np.argmin(time_diffs)
            
            # (30),
            min_diff = time_diffs[time_idx]
            if min_diff > 15:  # 15,
                # 
                if time_idx > 0 and time_idx < len(time_datetimes) - 1:
                    prev_idx = time_idx - 1
                    next_idx = time_idx + 1
                    prev_date = time_datetimes[prev_idx]
                    next_date = time_datetimes[next_idx]
                    
                    # 
                    total_diff = (next_date - prev_date).days
                    if total_diff > 0:
                        weight = (current_date - prev_date).days / total_diff
                        weight = max(0, min(1, weight))  # [0,1]
                    else:
                        weight = 0.5
                else:
                    #,
                    prev_idx = time_idx
                    next_idx = time_idx
                    weight = 0.5
            else:
                #,
                prev_idx = time_idx
                next_idx = time_idx
                weight = 0.5
        else:
            prev_idx = 0
            next_idx = 0
            weight = 0.5
        
        # 
        for i, lat in enumerate(np.linspace(spatial_bounds['lat_min'], spatial_bounds['lat_max'], grid_size)):
            for j, lon in enumerate(lon_target):
                try:
                    # (0-360)
                    lon_for_lookup = lon
                    if longitude.max() > 180 and lon < 0:
                        lon_for_lookup = lon % 360
                    
                    row, col = transformer.rowcol(lon_for_lookup, lat)
                    row = int(row)
                    col = int(col)
                    
                    if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                        #,
                        if prev_idx != next_idx and weight != 0.5:
                            value_prev = ds[var_name].isel(**{time_name: prev_idx, lat_name: row, lon_name: col}).values
                            value_next = ds[var_name].isel(**{time_name: next_idx, lat_name: row, lon_name: col}).values
                            
                            # 
                            if not np.isnan(value_prev) and not np.isnan(value_next):
                                value = value_prev * (1 - weight) + value_next * weight
                            elif not np.isnan(value_prev):
                                value = value_prev
                            elif not np.isnan(value_next):
                                value = value_next
                            else:
                                value = np.nan
                        else:
                            # 
                            value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                        
                        # GIMMS-3G+
                        # GIMMS-3G+ NDVI():
                        # - scale: x 10000(10000)
                        # - valid_range: [-0.3, 1.0](-300010000)
                        # - nodata:-5000(description)
                        # - :-300010000(NDVI -0.31.0)
                        if not np.isnan(value):
                            value = float(value)
                            
                            # fill_value(nodata)
                            if fill_value is not None and abs(value - fill_value) < 1e-6:
                                # nodata,(result0)
                                continue
                            
                            # nodata
                            # GIMMS-3G+nodata-5000
                            if abs(value - (-5000)) < 1e-6:
                                continue
                            
                            # scale_factor()
                            if scale_factor is not None and scale_factor > 0:
                                value = value / scale_factor
                            
                            # valid_range()
                            # :valid_range[-0.3, 1.0],-0.3,
                            if valid_range is not None:
                                if value < valid_range[0] or value > valid_range[1]:
                                    #,nodata
                                    continue
                                # valid_range,(-0.3)
                            
                            # scale_factor,
                            # GIMMS-3G+10000scale factor
                            if scale_factor is None:
                                # (-300010000),10000
                                if -5000 < value < 15000:
                                    # nodata-5000
                                    if abs(value - (-5000)) > 1e-6:
                                        value = value / 10000.0
                                    else:
                                        continue
                                # -11,
                                elif -1.0 <= value <= 1.0:
                                    pass  # 
                                else:
                                    # nodata
                                    continue
                            
                            # :NDVI
                            # :-0.3,
                            # NDVI-11,-0.31.0
                            if -1.0 <= value <= 1.0:
                                result[t, i, j] = value
                            # nodata,
                except Exception as e:
                    pass
    
    # :netcdf_cache(),()
    # (),
    if isinstance(filepath, list) or not HAS_NETCDF_CACHE:
        try:
            ds.close()
        except:
            pass
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, time_window_start, time_window_end,
                     grid_size, 'NDVI', year, result)
    
    return result


# ============================================================================
#  3: dNBR () - NetCDF()GEE API
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
    NetCDFdNBR
    
    dNBR(NBR)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    filepath = os.path.join(data_dir, 'dNBR', f'dnbr_{year}.nc')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"dNBR: {filepath}")
    
    # loadNetCDF
    try:
        ds = xr.open_dataset(filepath, engine='netcdf4')
    except:
        try:
            ds = xr.open_dataset(filepath, engine='h5netcdf')
        except:
            ds = xr.open_dataset(filepath)
    
    # 
    if 'dNBR' in ds.data_vars:
        var_name = 'dNBR'
    elif 'dnbr' in ds.data_vars:
        var_name = 'dnbr'
    else:
        var_name = list(ds.data_vars.keys())[0]
    
    # 
    coord_names = get_coord_names(ds)
    lat_name = coord_names.get('lat', 'lat')
    lon_name = coord_names.get('lon', 'lon')
    time_name = coord_names.get('time', 'time')
    
    # 
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
    
    # 
    lon_min = spatial_bounds['lon_min']
    lon_max = spatial_bounds['lon_max']
    if longitude.max() > 180 and lon_min < 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        lon_target = np.linspace(lon_min, lon_max, grid_size)
    
    # 
    pixel_size_x = longitude[1] - longitude[0] if len(longitude) > 1 else 0.01
    pixel_size_y = latitude[0] - latitude[1] if len(latitude) > 1 else 0.01
    
    origin_x = longitude[0] - (pixel_size_x / 2)
    origin_y = latitude[0] + (pixel_size_y / 2)
    
    # Affine
    from rasterio.transform import Affine
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(
        pixel_size_x, -pixel_size_y
    )
    transformer = rasterio.transform.AffineTransformer(transform)
    
    # dNBR,dNBR
    time_steps = (time_window_end - time_window_start).days + 1
    result = np.zeros((time_steps, grid_size, grid_size))
    
    # 
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
    
    # 
    for i, lat in enumerate(lat_target):
        for j, lon in enumerate(lon_target):
            try:
                row, col = transformer.rowcol(lon, lat)
                row = int(row)
                col = int(col)
                
                if 0 <= row < len(latitude) and 0 <= col < len(longitude):
                    value = ds[var_name].isel(**{time_name: time_idx, lat_name: row, lon_name: col}).values
                    if not np.isnan(value):
                        # dNBR
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
    GEE APIdNBR
    
    :Google Earth Engine APIMODISdNBR.
    dNBR,0.
    
    dNBR = NBR_pre - NBR_post
    NBR = (B2 - B7) / (B2 + B7)
    
    :
    - :32-8(24)
    - :8-32(24)
    
    Args:
        use_cache: (True)
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
        :MODIS,0
    """
    global _dnbr_cache
    
    # 
    if use_cache:
        fire_date_str = fire_date.strftime('%Y-%m-%d')
        bounds_hash = _get_bounds_hash(spatial_bounds, grid_size)
        cache_key = (fire_date_str, bounds_hash)
        
        if cache_key in _dnbr_cache:
            # 
            cached_result = _dnbr_cache[cache_key]
            time_steps = (time_window_end - time_window_start).days + 1
            # 
            if cached_result.shape[0] == time_steps:
                return cached_result.copy()
    
    if not HAS_GEE:
        raise ImportError("GEE")
    
    # initializeGEE
    try:
        import ee
        ee.Number(1).getInfo()
    except:
        # silent,
        if not initialize_gee(project=project, silent=True):
            time_steps = (time_window_end - time_window_start).days + 1
            return np.zeros((time_steps, grid_size, grid_size))
    
    # 1km
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
    
    # loadMODIS
    import ee
    collection = ee.ImageCollection('MODIS/061/MOD09GA')
    
    # dNBR,dNBR
    # :32-8
    # :8-32
    
    fire_date_ee = ee.Date(fire_date.strftime('%Y-%m-%d'))
    pre_start = fire_date_ee.advance(-32, 'day')
    pre_end = fire_date_ee.advance(-8, 'day')
    post_start = fire_date_ee.advance(8, 'day')
    post_end = fire_date_ee.advance(32, 'day')
    
    # 
    pre_fire = collection.filterDate(pre_start, pre_end).median()
    post_fire = collection.filterDate(post_start, post_end).median()
    
    # NBR
    nbr_pre = pre_fire.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
    nbr_post = post_fire.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
    
    # dNBR
    dnbr = nbr_pre.subtract(nbr_post)
    
    # :,
    # 
    region = ee.Geometry.Rectangle([
        spatial_bounds['lon_min'],
        spatial_bounds['lat_min'],
        spatial_bounds['lon_max'],
        spatial_bounds['lat_max']
    ])
    
    # 
    #,
    sample_points = []
    point_indices = []
    
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            sample_points.append((lat, lon))
            point_indices.append((i, j))
    
    # 
    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {'idx': idx, 'i': i, 'j': j})
        for idx, ((i, j), (lat, lon)) in enumerate(zip(point_indices, sample_points))
    ])
    
    # (dNBR)
    # 
    import time
    max_retries = 3
    retry_delay = 2  # 
    
    dnbr_values = {}
    for attempt in range(max_retries):
        try:
            # :sampleRegions
            # 625,API,625
            samples = dnbr.sampleRegions(
                collection=points_fc,
                scale=1000,  # 1km
                geometries=False
            )
            
            # (SSL)
            sample_list = samples.getInfo()['features']
            
            #,
            for feature in sample_list:
                idx = feature['properties']['idx']
                dnbr_value = feature['properties'].get('nd')
                if dnbr_value is not None:
                    dnbr_values[idx] = float(dnbr_value)
            
            #,
            break
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"    ⚠️  dNBR ( {attempt + 1}/{max_retries}): {error_msg[:100]}...")
                print(f"        {retry_delay} ...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 
            else:
                #,:
                print(f"    ⚠️  dNBR ( {max_retries} ): {error_msg[:100]}...")
                print(f"       ...")
                try:
                    stats = dnbr.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=1000,
                        maxPixels=1e9
                    )
                    mean_value = stats.getInfo().get('nd', 0)
                    if mean_value is not None:
                        # 
                        mean_dnbr = float(mean_value)
                        for idx in range(len(point_indices)):
                            dnbr_values[idx] = mean_dnbr
                        print(f"       ✅ : {mean_dnbr:.4f}")
                    else:
                        print(f"       ⚠️ ,0")
                except Exception as e2:
                    print(f"       ⚠️  : {str(e2)[:100]}...")
                    #,0(initialize)
    
    # 
    for t in range(time_steps):
        for idx, (i, j) in enumerate(point_indices):
            if idx in dnbr_values:
                result[t, i, j] = dnbr_values[idx]
    
    # 
    if use_cache:
        fire_date_str = fire_date.strftime('%Y-%m-%d')
        bounds_hash = _get_bounds_hash(spatial_bounds, grid_size)
        cache_key = (fire_date_str, bounds_hash)
        
        #,(FIFO)
        if len(_dnbr_cache) >= _dnbr_cache_max_size:
            # 
            oldest_key = next(iter(_dnbr_cache))
            del _dnbr_cache[oldest_key]
        
        _dnbr_cache[cache_key] = result.copy()
    
    return result


# ============================================================================
#  3: Population (,1km²,:/km²) - WorldPop
# ============================================================================

def extract_population_feature(
    spatial_bounds: dict,
    grid_size: int,
    year: int,
    data_dir: str,
    use_cache: bool = True
) -> np.ndarray:
    """
    WorldPop GeoTIFFPopulation
    
    Args:
        use_cache: (True)
    
    :WorldPop ppp (persons per pixel) 
    :1km×1km
    :(1km,population density,:/km²)
    
    Returns:
        np.ndarray: [grid_size, grid_size] (/km²)
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, None, None, grid_size, 'population', year)
        if cached is not None:
            return cached
    
    filepath = os.path.join(data_dir, 'worldpop', f'ppp_{year}_1km_Aggregated.tif')
    
    #,
    if not os.path.exists(filepath):
        abs_data_dir = os.path.abspath(data_dir)
        abs_filepath = os.path.join(abs_data_dir, 'worldpop', f'ppp_{year}_1km_Aggregated.tif')
        if os.path.exists(abs_filepath):
            filepath = abs_filepath
        else:
            # 
            error_msg = f"Population: {filepath}"
            error_msg += f"\n       : {abs_filepath} (: {os.path.exists(abs_filepath)})"
            error_msg += f"\n       : {os.getcwd()}"
            error_msg += f"\n       data_dir: {data_dir} (: {abs_data_dir})"
            # worldpop()
            worldpop_dir = os.path.join(data_dir, 'worldpop')
            abs_worldpop_dir = os.path.abspath(worldpop_dir)
            if os.path.exists(abs_worldpop_dir):
                try:
                    files = [f for f in os.listdir(abs_worldpop_dir) if f.endswith('.tif')]
                    error_msg += f"\n       worldpop: {files[:5]}..." if len(files) > 5 else f"\n       worldpop: {files}"
                except:
                    pass
            raise FileNotFoundError(error_msg)
    
    # rasterio
    with rasterio.open(filepath) as src:
        # NoData
        nodata_value = src.nodata
        
        # 
        from rasterio.transform import from_bounds
        
        transform = from_bounds(
            spatial_bounds['lon_min'],
            spatial_bounds['lat_min'],
            spatial_bounds['lon_max'],
            spatial_bounds['lat_max'],
            grid_size,
            grid_size
        )
        
        # (float32)
        output_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # (for)
        # :1km,1km²
        src_pixel_size_deg = abs(src.transform[0])  # ()
        src_pixel_area_km2 = (src_pixel_size_deg * 111.32) ** 2  # 1km²
        
        # :
        dst_pixel_size_lon_deg = abs(transform[0])  # ()
        dst_pixel_size_lat_deg = abs(transform[4])  # ()
        # 
        center_lat = (spatial_bounds['lat_min'] + spatial_bounds['lat_max']) / 2
        dst_pixel_area_km2 = (dst_pixel_size_lon_deg * 111.32 * np.cos(np.radians(center_lat))) * (dst_pixel_size_lat_deg * 111.32)
        
        # :WorldPop"persons per pixel",1km1km²
        #,:
        # 1. sum
        # 2. population density(/km²)
        temp_output = np.zeros((grid_size, grid_size), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=temp_output,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.sum,  # sum,
            src_nodata=nodata_value,  # NoData
            dst_nodata=np.nan  # NoDataNaN
        )
        
        # population density(/km²)
        # :WorldPop"persons per pixel",1km1km²
        #,rasteriosum
        # 
        # :
        # - WorldPop: = 1km², = /km²
        # - :
        # - sum()
        # - population density(/km²),
        #
        #,(),
        # :(<0.01 km²),
        if dst_pixel_area_km2 > 0:
            # (0.1100 km²,grid_size)
            expected_min_area = 0.1  # (0.1 km²)
            expected_max_area = 100.0  # (100 km²,10km×10km)
            
            if dst_pixel_area_km2 < expected_min_area or dst_pixel_area_km2 > expected_max_area:
                #,
                print(f"    ⚠️  Population: {dst_pixel_area_km2:.6f} km²")
                print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                print(f"       grid_size: {grid_size}, : {src_pixel_area_km2:.6f} km²")
                #,temp_output()
                # sum,
                if dst_pixel_area_km2 < expected_min_area:
                    print(f"      ,")
                    output_data = temp_output / src_pixel_area_km2 if src_pixel_area_km2 > 0 else temp_output
                else:
                    #,
                    output_data = temp_output / dst_pixel_area_km2
            else:
                #,
                output_data = temp_output / dst_pixel_area_km2
        else:
            #,
            print(f"    ⚠️  Population,")
            output_data = temp_output / src_pixel_area_km2 if src_pixel_area_km2 > 0 else temp_output
        
        # NoData:NaN()0
        # WorldPop,NoData
        # :WorldPopNoData-3.4028234663852886e+38(float32)
        valid_mask = ~np.isnan(output_data)
        
        # ()
        if valid_mask.any():
            # (WorldPop0)
            valid_data = output_data[valid_mask]
            #,NoData
            # WorldPopNoData-3.4028234663852886e+38(float32)
            if np.any(valid_data < 0) or np.any(valid_data < -1e10):
                # NoData
                output_data[output_data < 0] = np.nan
                valid_mask = ~np.isnan(output_data)
        
        # NaN0
        output_data[~valid_mask] = 0.0
        
        # ()
        # WorldPop ppp:(1km²),100,000/km²()
        max_reasonable_value = 100000.0
        
        # 
        if np.any(output_data < 0):
            negative_count = np.sum(output_data < 0)
            print(f"    ⚠️  Population {negative_count},0")
            output_data[output_data < 0] = 0.0
        
        if np.any(output_data > max_reasonable_value):
            large_count = np.sum(output_data > max_reasonable_value)
            max_value = np.max(output_data)
            print(f"    ⚠️  Population {large_count} (: {max_value:.2f}),0")
            print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
            print(f"       : {dst_pixel_area_km2:.6f} km², : {src_pixel_area_km2:.6f} km²")
            # 0
            output_data[output_data > max_reasonable_value] = 0.0
        
        # :,population density
        # (<1),
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, None, None, grid_size, 'population', year, output_data)
    
    return output_data


# ============================================================================
#  5: GDP () -  gdp_per_capita.csv,→→
# ============================================================================

# GDP(for)
_gdp_warning_count = 0
_gdp_warning_lock = threading.Lock()
# Population  0 (: WorldPop )
_pop_zero_warning_count = 0
_pop_zero_warning_lock = threading.Lock()

# gdp_per_capita.csv (→→ GDP)
_gdp_per_capita_df = None
_gdp_per_capita_lock = threading.Lock()

def _load_gdp_per_capita_df(data_dir: str):
    """load gdp_per_capita.csv, Country Code  2002,2003,... """
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
     gdp_per_capita.csv  (Country Code, )  GDP per capita.
    :Country Code, 2002, 2003, ... 2023 .
    """
    df = _load_gdp_per_capita_df(data_dir)
    if df is None or 'Country Code' not in df.columns:
        return None
    year_str = str(year)
    if year_str not in df.columns:
        # 
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
    """GDP(for)"""
    global _gdp_warning_count
    with _gdp_warning_lock:
        _gdp_warning_count = 0


def _lat_lon_to_iso3_point_in_country(lat: float, lon: float) -> Optional[str]:
    """
    「」 iso3:reverse_geocoder() ISO2,
     pycountry  ISO3./, GDP=0.
     reverse_geocoder  pycountry,  Python<3.10  importlib.metadata, None.
    """
    try:
        import reverse_geocoder as rg
    except (ImportError, AttributeError):
        # AttributeError: importlib.metadata  Python<3.10  packages_distributions
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
    #  pycountry  ISO2->ISO3 
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
     (lat, lon)  DataFrame  (lat_mean, lon_mean), 2°  iso3.
    for extract_gdp_feature  iso3  patch ().
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
        #  scipy 
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
     gdp_per_capita.csv 「→→」GDP.
     patch  (lat, lon)  iso3, gdp_per_capita.csv 
     (Country Code, year) ; filtered_cleaned_cp_covariate.csv.
    
    Returns:
        np.ndarray: [grid_size, grid_size] (GDP)
    """
    global _gdp_warning_count
    
    center_lat = (spatial_bounds['lat_min'] + spatial_bounds['lat_max']) / 2
    center_lon = (spatial_bounds['lon_min'] + spatial_bounds['lon_max']) / 2

    # 「+」
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

    # 1) 「」(,reverse_geocoder);,  iso3/country
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
                print(f"    ⚠️  (lat,lon) iso3,GDP 0( {current_count} )")
        return np.zeros((grid_size, grid_size))

    # 3)  gdp_per_capita.csv  (Country Code, year) 
    gdp_value = _get_gdp_from_per_capita_csv(data_dir, iso3_clean, year)
    if gdp_value is not None:
        result = np.full((grid_size, grid_size), float(gdp_value))
        if use_cache:
            cache_feature(
                spatial_bounds, None, None, grid_size, 'GDP', year, result,
                extra_params={'lat': center_lat, 'lon': center_lon}
            )
        return result

    # 4) : filtered_cleaned_cp_covariate.csv  (iso3, year)  gdp 
    if covariate_df is None:
        with _gdp_warning_lock:
            _gdp_warning_count += 1
            current_count = _gdp_warning_count
            if current_count == 1 or current_count % 100 == 0:
                print(f"    ⚠️  GDP(iso3='{iso3_clean}', year={year}),0( {current_count} )")
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
            print(f"    ⚠️  GDP(iso3='{iso3_clean}', year={year}),0( {current_count} )")
    return np.zeros((grid_size, grid_size))


# ============================================================================
#  6: Land Cover () - GeoTIFF()GEE API
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
    Land Cover(,GEE API)
    
    :
    1. GeoTIFF:dataset/LandCover/modis_mcd12q1_lc_type1_{year}.tif
    2. GEE API:MODIS/061/MCD12Q1()
    
    Args:
        use_cache: (True)
    
    Returns:
        np.ndarray: [grid_size, grid_size]
    """
    # 
    if use_cache:
        cached = get_cached_feature(spatial_bounds, None, None, grid_size, 'land_cover', year)
        if cached is not None:
            return cached
    
    # 
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if os.path.exists(local_file):
        try:
            # GeoTIFF
            with rasterio.open(local_file) as src:
                # 
                from rasterio.transform import from_bounds
                
                transform = from_bounds(
                    spatial_bounds['lon_min'],
                    spatial_bounds['lat_min'],
                    spatial_bounds['lon_max'],
                    spatial_bounds['lat_max'],
                    grid_size,
                    grid_size
                )
                
                # 
                output_data = np.zeros((grid_size, grid_size), dtype=np.float32)
                
                # 
                reproject(
                    source=rasterio.band(src, 1),
                    destination=output_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,  # 
                    src_nodata=src.nodata,
                    dst_nodata=255  # 255
                )
                
                # NoData
                if src.nodata is not None:
                    output_data[output_data == src.nodata] = 255
                
                # NaN255()
                output_data[np.isnan(output_data)] = 255
                
                # 
                result = output_data.astype(np.int32)
                
                # 
                # :25km×25km,
                # 0()
                unique_values = np.unique(result[result != 0])
                if len(unique_values) == 0:
                    # 0255(),
                    print(f"    ⚠️  : Land Cover()0,")
                    print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
                          f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
                    print(f"       : {local_file}")
                # :,(),
                
                # 
                if use_cache:
                    cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
                
                return result
        except Exception as e:
            print(f"    ⚠️  Land Cover: {e}")
            print(f"       GEE API...")
    
    #,GEE API
    if not HAS_GEE:
        print(f"    ⚠️  GEE,: {local_file}")
        print(f"       ")
        result = np.zeros((grid_size, grid_size))
        #,()
        if use_cache:
            cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
        return result
    
    # initializeGEE
    try:
        import ee
        ee.Number(1).getInfo()
    except:
        # silent,
        if not initialize_gee(project=project, silent=True):
            return np.zeros((grid_size, grid_size))
    
    # 1km
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
    
    # 
    sample_points = []
    point_indices = []
    
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            sample_points.append((lat, lon))
            point_indices.append((i, j))
    
    # loadMODIS
    import ee
    dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
    image = dataset.filter(
        ee.Filter.eq('system:time_start', 
                    ee.Date.fromYMD(year, 1, 1).millis())
    ).first()
    lc_band = image.select('LC_Type1')
    
    # (1km scale,GEE500m)
    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {'idx': idx})
        for idx, (lat, lon) in enumerate(sample_points)
    ])
    
    samples = lc_band.sampleRegions(
        collection=points_fc,
        scale=1000,  # :1000m scale,GEE500m1km
        geometries=False
    )
    
    # 
    result = np.zeros((grid_size, grid_size), dtype=np.int32)
    sample_list = samples.getInfo()['features']
    
    # 
    if len(sample_list) != len(sample_points):
        print(f"    ⚠️  Land Cover:  {len(sample_points)},  {len(sample_list)}")
    
    # 
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
        
        # 
        if idx >= len(point_indices):
            print(f"    ⚠️  : idx={idx}, point_indices={len(point_indices)}")
            invalid_samples += 1
            continue
        
        i, j = point_indices[idx]
        
        # i, j
        if 0 <= i < grid_size and 0 <= j < grid_size:
            result[i, j] = int(lc_value)
            valid_samples += 1
            lc_value_counts[int(lc_value)] = lc_value_counts.get(int(lc_value), 0) + 1
        else:
            print(f"    ⚠️  : i={i}, j={j}, grid_size={grid_size}")
            invalid_samples += 1
    
    # 
    unique_values = np.unique(result[result != 0])
    if len(unique_values) == 1:
        print(f"    ⚠️  : Land Cover {unique_values[0]},")
        print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
              f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
        print(f"       : {valid_samples}/{len(sample_points)}")
        print(f"       : {lc_value_counts}")
    elif len(unique_values) == 0:
        print(f"    ⚠️  : Land Cover0,")
        print(f"       : lat=[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], "
              f"lon=[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
        print(f"       : {valid_samples}/{len(sample_points)}")
    
    # 
    if use_cache:
        cache_feature(spatial_bounds, None, None, grid_size, 'land_cover', year, result)
    
    return result


# ============================================================================
# :
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
    
    
    Args:
        data:  [time, lat, lon]  [lat, lon]
        lat_orig: 
        lon_orig: 
        spatial_bounds: 
        grid_size: 
        time_steps: 
    
    Returns:
        np.ndarray: [time_steps, grid_size, grid_size]
    """
    from scipy.interpolate import griddata
    
    # 
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
    
    # 
    lon_orig_mesh, lat_orig_mesh = np.meshgrid(lon_orig, lat_orig)
    points_orig = np.column_stack([
        lat_orig_mesh.ravel(),
        lon_orig_mesh.ravel()
    ])
    
    # 
    points_target = np.column_stack([
        lat_mesh.ravel(),
        lon_mesh.ravel()
    ])
    
    # 
    if data.ndim == 2:
        # 2D:3D
        data = data[np.newaxis, :, :]
        time_steps = 1
    
    result = np.zeros((time_steps, grid_size, grid_size))
    
    for t in range(min(time_steps, data.shape[0])):
        values_orig = data[t].ravel()
        
        # (NaN)
        valid_mask = ~np.isnan(values_orig)
        if len(values_orig) > 0:
            # NaN,fill_value
            if not valid_mask.any():
                result[t] = np.full((grid_size, grid_size), 0.0)
                continue
            
            # 
            valid_points = points_orig[valid_mask]
            valid_values = values_orig[valid_mask]
            
            # 
            num_valid_points = len(valid_points)
            
            if num_valid_points == 0:
                #,fill_value
                result[t] = np.full((grid_size, grid_size), 0.0)
            elif num_valid_points == 1:
                #,
                result[t] = np.full((grid_size, grid_size), float(valid_values[0]))
            elif num_valid_points < 4:
                # 4,nearest(1)
                values_target = griddata(
                    valid_points,
                    valid_values,
                    points_target,
                    method='nearest',
                    fill_value=0.0
                )
                result[t] = values_target.reshape(grid_size, grid_size)
            else:
                # 4,linear
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
                    # linear(),nearest
                    values_target = griddata(
                        valid_points,
                        valid_values,
                        points_target,
                        method='nearest',
                        fill_value=0.0
                    )
                    result[t] = values_target.reshape(grid_size, grid_size)
        else:
            #,fill_value
            result[t] = np.full((grid_size, grid_size), 0.0)
    
    return result


# ============================================================================
# 
# ============================================================================

if __name__ == '__main__':
    # 
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
    
    print(f": {features.shape}")
    print(f":")
    for ch in range(7):
        ch_data = features[:, :, :, ch]
        print(f"   {ch}: ={ch_data.mean():.4f}, ={(ch_data != 0).sum() / ch_data.size * 100:.2f}%")

