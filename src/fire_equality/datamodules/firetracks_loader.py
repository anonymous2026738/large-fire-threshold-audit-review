"""
FireTracksload
forloadFireTracks
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

# ISO3
try:
    from .country_to_iso3 import add_iso3_to_dataframe
except ImportError:
    try:
        from code.fire_equality.datamodules.country_to_iso3 import add_iso3_to_dataframe
    except ImportError:
        # :sys.path
        try:
            current_dir = Path(__file__).resolve().parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from country_to_iso3 import add_iso3_to_dataframe
        except Exception:
            #,(,iso3)
            def add_iso3_to_dataframe(df, country_column='country'):
                print("⚠️  country_to_iso3,iso3")
                return df

# ()
# :,load
extract_aligned_features = None  # type: ignore
try:
    from .feature_alignment import extract_aligned_features  # type: ignore
except ImportError:
    try:
        from code.fire_equality.datamodules.feature_alignment import extract_aligned_features  # type: ignore
    except ImportError:
        # :sys.path
        try:
            current_dir = Path(__file__).resolve().parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from feature_alignment import extract_aligned_features  # type: ignore
        except Exception:
            #,()
            # load
            pass

#  landcover ( GeoTIFF )
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except (ImportError, OSError, Exception) as e:
    # ImportError: 
    # OSError: DLLload(Windows)
    # Exception: 
    HAS_RASTERIO = False
    #,
    # :DLL,

# : landcover, GEE API
HAS_MODIS_LC = True  #  True,
MODIS_LC_ERROR = None

def get_landcover_from_local(lat: float, lon: float, year: int, data_dir: str = 'dataset') -> int:
    """
     GeoTIFF 
    
    Args:
        lat: 
        lon: 
        year: 
        data_dir:, 'dataset'
    
    Returns:
        (), 255()
    """
    if not HAS_RASTERIO:
        # rasterioDLLloading,
        return 255
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️   landcover : {local_file}")
        return 255
    
    try:
        with rasterio.open(local_file) as src:
            #  sample 
            values = list(src.sample([(lon, lat)]))
            if values and len(values) > 0:
                lc_value = int(values[0][0])
                #  NoData 
                if lc_value == src.nodata or np.isnan(lc_value):
                    return 255
                return lc_value
            else:
                return 255
    except Exception as e:
        print(f"⚠️   landcover  ({lat}, {lon}, {year}): {e}")
        return 255


def get_landcover_batch_local(coords, year: int, data_dir: str = 'dataset', batch_size: int = 1000):
    """
     GeoTIFF 
    
    Args:
        coords:, (lat, lon) 
        year: 
        data_dir:, 'dataset'
        batch_size: 
    
    Returns:
        
    """
    if not HAS_RASTERIO:
        print("⚠️  rasterio, landcover ")
        return [255] * len(coords)
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️   landcover : {local_file}")
        return [255] * len(coords)
    
    results = []
    try:
        with rasterio.open(local_file) as src:
            #, 
            lon_lat_pairs = [(lon, lat) for lat, lon in coords]
            
            # 
            for i, (lon, lat) in enumerate(lon_lat_pairs):
                try:
                    values = list(src.sample([(lon, lat)]))
                    if values and len(values) > 0:
                        lc_value = int(values[0][0])
                        #  NoData 
                        if lc_value == src.nodata or np.isnan(lc_value):
                            results.append(255)
                        else:
                            results.append(lc_value)
                    else:
                        results.append(255)
                except Exception as e:
                    #, 255
                    results.append(255)
            
            return results
    except Exception as e:
        print(f"⚠️   landcover : {e}")
        return [255] * len(coords)


def find_locations_by_landcover_local(positive_lc_types, spatial_bounds: dict, year: int,
                                     num_samples_per_type: int = 1000, data_dir: str = 'dataset'):
    """
     GeoTIFF 
    
    Args:
        positive_lc_types:, [7, 9, 10]
        spatial_bounds: 
        year: 
        num_samples_per_type: 
        data_dir:, 'dataset'
    
    Returns:
       , {'lat': float, 'lon': float, 'land_cover': int}
    """
    if not HAS_RASTERIO:
        print("⚠️  rasterio, landcover ")
        return []
    
    local_file = os.path.join(data_dir, 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
    
    if not os.path.exists(local_file):
        print(f"⚠️   landcover : {local_file}")
        return []
    
    all_locations = []
    
    try:
        with rasterio.open(local_file) as src:
            # 
            # 
            from rasterio.warp import transform
            
            # 
            lon_min, lat_min = spatial_bounds['lon_min'], spatial_bounds['lat_min']
            lon_max, lat_max = spatial_bounds['lon_max'], spatial_bounds['lat_max']
            
            # 
            #  window 
            from rasterio.windows import from_bounds
            
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            
            # 
            data = src.read(1, window=window)
            
            # 
            window_transform = src.window_transform(window)
            
            # 
            for lc_type in positive_lc_types:
                # 
                mask = (data == lc_type)
                matching_pixels = np.where(mask)
                
                if len(matching_pixels[0]) == 0:
                    print(f"     {lc_type}: ")
                    continue
                
                # 
                num_found = len(matching_pixels[0])
                num_to_sample = min(num_samples_per_type, num_found)
                
                if num_found > num_samples_per_type:
                    # 
                    indices = np.random.choice(num_found, num_to_sample, replace=False)
                    sampled_rows = matching_pixels[0][indices]
                    sampled_cols = matching_pixels[1][indices]
                else:
                    # 
                    sampled_rows = matching_pixels[0]
                    sampled_cols = matching_pixels[1]
                
                # 
                type_locations = []
                for row, col in zip(sampled_rows, sampled_cols):
                    # ()
                    pixel_x = col
                    pixel_y = row
                    
                    # 
                    lon, lat = rasterio.transform.xy(window_transform, pixel_y, pixel_x)
                    
                    type_locations.append({
                        'lat': lat,
                        'lon': lon,
                        'land_cover': int(lc_type)
                    })
                
                all_locations.extend(type_locations)
                print(f"     {lc_type}:  {len(type_locations)} ( {num_found} )")
        
        print(f"  ✅  {len(all_locations):,} ( {len(set(l['land_cover'] for l in all_locations))} )")
        
        return all_locations
        
    except Exception as e:
        print(f"⚠️  : {e}")
        import traceback
        traceback.print_exc()
        return []


#,( GEE API )
def get_landcover_batch_gee(coords, year: int, lc_type: str = 'LC_Type1', 
                            batch_size: int = 1000, project: Optional[str] = None):
    """
    (, GEE API)
    
    :project,
    """
    return get_landcover_batch_local(coords, year, data_dir='dataset', batch_size=batch_size)


def find_locations_by_landcover(positive_lc_types, spatial_bounds: dict, year: int,
                                num_samples_per_type: int = 1000, project: Optional[str] = None):
    """
    (, GEE API)
    
    :project,
    """
    return find_locations_by_landcover_local(
        positive_lc_types, spatial_bounds, year, 
        num_samples_per_type, data_dir='dataset'
    )


def load_firetracks_dataset(data_directory, max_rows=None, year_range=None, use_chunks=False, chunk_size=1000000):
    """
    loadFireTracks
    
    Args:
        data_directory: FireTracks
        max_rows: (for,None)
        year_range:, (2002, 2020),None
        use_chunks: (for)
        chunk_size: (use_chunks=True)
    
    Returns:
        dict: FireTracks
            - 'events':  (v.h5)
            - 'events_lc':  (v_LC_Type1.h5)
            - 'components':  (cp.h5)
            - 'components_lc':  (cp_LC_Type1.h5)
    """
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:292","message":"load_firetracks_dataset entry","data":{"data_directory":data_directory,"max_rows":max_rows,"year_range":year_range},"timestamp":int(time_module.time()*1000)}) + '\n')
    except Exception as log_err:
        pass  #,
    # #endregion
    print(f" {data_directory} loadFireTracks...")
    if max_rows:
        print(f"⚠️  :  {max_rows:,} ")
    if year_range:
        print(f"⚠️  : {year_range[0]}-{year_range[1]}")
    
    datasets = {}
    
    import os
    
    try:
        # load()
        if os.path.exists(f'{data_directory}/v.h5'):
            # #region agent log
            import json
            import os as os_module
            file_path_events = f'{data_directory}/v.h5'
            file_size_gb_events = os_module.path.getsize(file_path_events) / (1024**3) if os_module.path.exists(file_path_events) else 0
            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:322","message":"Events file size check","data":{"file_size_gb":round(file_size_gb_events,2),"file_exists":os_module.path.exists(file_path_events)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            # where(year_range)
            if year_range:
                try:
                    start_date = f"{year_range[0]}-01-01"
                    end_date = f"{year_range[1]+1}-01-01"
                    print(f"where events (dtime >= {start_date} & dtime < {end_date})...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:329","message":"Attempting where clause filter","data":{"start_date":start_date,"end_date":end_date,"year_range":year_range},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    datasets['events'] = pd.read_hdf(
                        f'{data_directory}/v.h5',
                        where=f'dtime >= "{start_date}" & dtime < "{end_date}"'
                    )
                    print(f"✅ whereload {len(datasets['events']):,} ")
                    # #region agent log
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:333","message":"Where clause succeeded","data":{"rows_loaded":len(datasets['events'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                except Exception as e:
                    print(f"⚠️  where ({e}),...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:335","message":"Where clause failed","data":{"error":str(e)[:200],"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    # where,
                    if max_rows:
                        print(f" {max_rows:,} ...")
                        store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                        try:
                            datasets['events'] = store.select('table', start=0, stop=max_rows)
                        finally:
                            store.close()
                    else:
                        # where,()
                        # 
                        # #region agent log
                        import os as os_check
                        file_path = f'{data_directory}/v.h5'
                        file_size_gb = os_check.path.getsize(file_path) / (1024**3) if os_check.path.exists(file_path) else 0
                        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"firetracks_loader.py:375","message":"Where clause failed, using chunked reading","data":{"file_size_gb":round(file_size_gb,2),"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # where,
                        if True:  # where
                            reason = f"where, ({file_size_gb:.1f} GB)"
                            print(f"⚠️  {reason},...")
                            # #region agent log
                            with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:373","message":"Using chunked reading","data":{"chunk_size":10000000,"reason":reason},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                            # #endregion
                            #,1000
                            chunk_size = 10000000
                            chunks = []
                            store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                            try:
                                total_rows = store.get_storer('table').nrows
                                print(f"   : {total_rows:,}")
                                for start in range(0, total_rows, chunk_size):
                                    stop = min(start + chunk_size, total_rows)
                                    print(f"    {start:,}  {stop:,}...", end='\r')
                                    chunk = store.select('table', start=start, stop=stop)
                                    # year_range,
                                    if year_range:
                                        chunk['dtime'] = pd.to_datetime(chunk['dtime'])
                                        start_date_pd = pd.Timestamp(f"{year_range[0]}-01-01")
                                        end_date_pd = pd.Timestamp(f"{year_range[1]+1}-01-01")
                                        chunk = chunk[(chunk['dtime'] >= start_date_pd) & (chunk['dtime'] < end_date_pd)]
                                    if len(chunk) > 0:
                                        chunks.append(chunk)
                                print()  # 
                            finally:
                                store.close()
                            
                            if chunks:
                                datasets['events'] = pd.concat(chunks, ignore_index=True)
                                print(f"✅,load {len(datasets['events']):,} ")
                                # #region agent log
                                with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:417","message":"Chunked reading completed","data":{"rows_loaded":len(datasets['events'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                            else:
                                raise ValueError("")
                        # :where,
                        # 
            elif max_rows:
                # 
                print(f" {max_rows:,}  events...")
                store = pd.HDFStore(f'{data_directory}/v.h5', mode='r')
                try:
                    datasets['events'] = store.select('table', start=0, stop=max_rows)
                finally:
                    store.close()
            else:
                # 
                datasets['events'] = pd.read_hdf(f'{data_directory}/v.h5')
        
        # iso3(eventscountryiso3)
        if 'events' in datasets and len(datasets['events']) > 0:
            if 'country' in datasets['events'].columns:
                if 'iso3' not in datasets['events'].columns:
                    print("countryiso3...")
                    add_iso3_to_dataframe(datasets['events'], country_column='country')
                else:
                    # iso3
                    iso3_valid = datasets['events']['iso3'].dropna()
                    iso3_valid = iso3_valid[iso3_valid != '']
                    if len(iso3_valid) < len(datasets['events']) * 0.1:
                        print("iso3,country...")
                        # iso3
                        mask = datasets['events']['iso3'].isna() | (datasets['events']['iso3'] == '')
                        if mask.sum() > 0:
                            # country_to_iso3
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
                            print(f"✅  {len(new_iso3_count)} ISO3")
        else:
            raise FileNotFoundError(f": {data_directory}/v.h5")
        
        if os.path.exists(f'{data_directory}/cp.h5'):
            # where(year_range)
            if year_range:
                try:
                    start_date = f"{year_range[0]}-01-01"
                    end_date = f"{year_range[1]+1}-01-01"
                    print(f"where components (dtime_min >= {start_date} & dtime_min < {end_date})...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:398","message":"Attempting where clause filter for components","data":{"start_date":start_date,"end_date":end_date},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    datasets['components'] = pd.read_hdf(
                        f'{data_directory}/cp.h5',
                        where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
                    )
                    print(f"✅ whereload {len(datasets['components']):,} ")
                except Exception as e:
                    print(f"⚠️  where ({e}),...")
                    # #region agent log
                    import json
                    with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"firetracks_loader.py:410","message":"Components where clause failed","data":{"error":str(e)[:200],"max_rows":max_rows},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    # where,
                    if max_rows:
                        print(f" {max_rows:,} ...")
                        store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                        try:
                            datasets['components'] = store.select('table', start=0, stop=max_rows)
                        finally:
                            store.close()
                    else:
                        # where,()
                        # 
                        # #region agent log
                        import os as os_check_comp
                        file_path = f'{data_directory}/cp.h5'
                        file_size_gb = os_check_comp.path.getsize(file_path) / (1024**3) if os_check_comp.path.exists(file_path) else 0
                        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"firetracks_loader.py:545","message":"Components where clause failed, using chunked reading","data":{"file_size_gb":round(file_size_gb,2)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # where,
                        if True:  # where
                            reason_comp = f"where, ({file_size_gb:.1f} GB)"
                            print(f"⚠️  {reason_comp},...")
                            chunk_size = 1000000  # componentsevents,
                            chunks = []
                            store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                            try:
                                total_rows = store.get_storer('table').nrows
                                print(f"   : {total_rows:,}")
                                for start in range(0, total_rows, chunk_size):
                                    stop = min(start + chunk_size, total_rows)
                                    print(f"    {start:,}  {stop:,}...", end='\r')
                                    chunk = store.select('table', start=start, stop=stop)
                                    # year_range,
                                    if year_range:
                                        chunk['dtime_min'] = pd.to_datetime(chunk['dtime_min'])
                                        start_date_pd = pd.Timestamp(f"{year_range[0]}-01-01")
                                        end_date_pd = pd.Timestamp(f"{year_range[1]+1}-01-01")
                                        chunk = chunk[(chunk['dtime_min'] >= start_date_pd) & (chunk['dtime_min'] < end_date_pd)]
                                    if len(chunk) > 0:
                                        chunks.append(chunk)
                                print()  # 
                            finally:
                                store.close()
                            
                            if chunks:
                                datasets['components'] = pd.concat(chunks, ignore_index=True)
                                print(f"✅,load {len(datasets['components']):,} ")
                                # #region agent log
                                with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"firetracks_loader.py:548","message":"Components chunked reading completed","data":{"rows_loaded":len(datasets['components'])},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                            else:
                                raise ValueError("")
                        # :where,
                        # 
            elif max_rows:
                # 
                print(f" {max_rows:,}  components...")
                store = pd.HDFStore(f'{data_directory}/cp.h5', mode='r')
                try:
                    datasets['components'] = store.select('table', start=0, stop=max_rows)
                finally:
                    store.close()
            else:
                # 
                datasets['components'] = pd.read_hdf(f'{data_directory}/cp.h5')
        else:
            raise FileNotFoundError(f": {data_directory}/cp.h5")
        
        # load()
        # :land cover,
        if os.path.exists(f'{data_directory}/v_LC_Type1.h5') and max_rows is None:
            try:
                #,where
                if year_range:
                    try:
                        start_date = f"{year_range[0]}-01-01"
                        end_date = f"{year_range[1]+1}-01-01"
                        datasets['events_lc'] = pd.read_hdf(
                            f'{data_directory}/v_LC_Type1.h5',
                            where=f'dtime >= "{start_date}" & dtime < "{end_date}"'
                        )
                        print("✅ load events_lc ()")
                    except:
                        # where,load()
                        datasets['events_lc'] = pd.read_hdf(f'{data_directory}/v_LC_Type1.h5')
                        print("✅ load events_lc")
                else:
                    datasets['events_lc'] = pd.read_hdf(f'{data_directory}/v_LC_Type1.h5')
                    print("✅ load events_lc")
            except (MemoryError, OSError, Exception) as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"⚠️   events_lc()")
        
        if os.path.exists(f'{data_directory}/cp_LC_Type1.h5') and max_rows is None:
            try:
                #,where
                if year_range:
                    try:
                        start_date = f"{year_range[0]}-01-01"
                        end_date = f"{year_range[1]+1}-01-01"
                        datasets['components_lc'] = pd.read_hdf(
                            f'{data_directory}/cp_LC_Type1.h5',
                            where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
                        )
                        print("✅ load components_lc ()")
                    except:
                        # where,load()
                        datasets['components_lc'] = pd.read_hdf(f'{data_directory}/cp_LC_Type1.h5')
                        print("✅ load components_lc")
                else:
                    datasets['components_lc'] = pd.read_hdf(f'{data_directory}/cp_LC_Type1.h5')
                    print("✅ load components_lc")
            except (MemoryError, OSError, Exception) as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"⚠️   components_lc()")
        
        # iso3(eventscountryiso3)
        if 'events' in datasets and len(datasets['events']) > 0:
            if 'country' in datasets['events'].columns:
                if 'iso3' not in datasets['events'].columns:
                    print("countryiso3...")
                    add_iso3_to_dataframe(datasets['events'], country_column='country')
                else:
                    # iso3
                    iso3_valid = datasets['events']['iso3'].dropna()
                    iso3_valid = iso3_valid[iso3_valid != '']
                    if len(iso3_valid) < len(datasets['events']) * 0.1:
                        print("iso3,country...")
                        # iso3
                        mask = datasets['events']['iso3'].isna() | (datasets['events']['iso3'] == '')
                        if mask.sum() > 0:
                            # country_to_iso3
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
                            print(f"✅  {len(new_iso3_count)} ISO3")
        
        print("✅ FireTracksload!")
        print(f"   - : {len(datasets['events']):,} ")
        print(f"   - : {len(datasets['components']):,} ")
        
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
        print(f"❌ load(): {me}")
        print(f"   :")
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
        print(f"❌ load: {e}")
        return None
    
    return datasets


# 

def preprocess_firetracks_data(datasets, target_year_range=(2002, 2020)):
    """
    FireTracks,ConvLSTM
    
    Args:
        datasets: load
        target_year_range: 
    
    Returns:
        dict: 
            - 'components': 
            - 'events': 
            - 'events_lc': 
            - 'components_lc': 
    """
    print("...")
    
    # 1. 
    components = datasets['components'].copy()
    events = datasets['events'].copy()
    
    # 
    components['dtime_min'] = pd.to_datetime(components['dtime_min'])
    events['dtime'] = pd.to_datetime(events['dtime'])
    
    # 
    start_year, end_year = target_year_range
    mask = (components['dtime_min'].dt.year >= start_year) & (components['dtime_min'].dt.year <= end_year)
    filtered_components = components[mask].copy()
    
    print(f" {len(filtered_components):,}  ({start_year}-{end_year})")
    
    # 2. 
    filtered_components['frp_intensity'] = filtered_components['maxFRP_sum'] / filtered_components['area']
    filtered_components['expansion_rate'] = filtered_components['area'] / filtered_components['duration']
    
    # 3. 
    # 95%threshold,5%(top 5%)
    # :maxFRP_sum()
    frp_threshold = filtered_components['maxFRP_sum'].quantile(0.95)
    filtered_components['fire_severity'] = (filtered_components['maxFRP_sum'] >= frp_threshold).astype(int)
    
    print(f"threshold: {frp_threshold:,.0f} MW (95%)")
    print(f": {filtered_components['fire_severity'].sum():,}  (5%)")
    print(f": {(filtered_components['fire_severity'] == 0).sum():,}  (95%)")
    
    # 4. (for)
    severe_components = filtered_components[filtered_components['fire_severity'] == 1].copy()
    print(f": {len(severe_components):,} (for)")
    
    #,
    result = {
        'components': filtered_components,  # ()
        'severe_components': severe_components,  # 
        'events': events,
    }
    
    # 
    if 'events_lc' in datasets:
        result['events_lc'] = datasets['events_lc']
    
    if 'components_lc' in datasets:
        result['components_lc'] = datasets['components_lc']
    
    return result


# ConvLSTM

def create_spatiotemporal_patches(preprocessed_data, patch_size_km=25, time_steps=10, spatial_resolution_km=1, max_samples=None):
    """
    
    
    Args:
        preprocessed_data: 
        patch_size_km:  (km)
        time_steps: stride ()
        spatial_resolution_km:  (km)
        max_samples: (for,None)
    
    Returns:
        list:,:
            - component_id: ID
            - features:  [time_steps, grid_size, grid_size, channels]
            - targets: 
            - metadata: 
    """
    print("...")
    
    components = preprocessed_data['components']
    events = preprocessed_data['events']
    
    total_components = len(components)
    if max_samples:
        total_components = min(total_components, max_samples)
        print(f"   ⚠️  :  {max_samples:,} ")
    
    print(f"   - : {total_components:,} ")
    print(f"   - : {len(events):,} ")
    
    spatiotemporal_samples = []
    
    # 
    progress_interval = max(1, total_components // 100)  # 1%
    
    processed_count = 0
    for idx, component in components.iterrows():
        if max_samples and processed_count >= max_samples:
            break
        processed_count += 1
        # 
        if processed_count % progress_interval == 0 or processed_count == total_components:
            progress = processed_count / total_components * 100
            print(f"   : {processed_count:,}/{total_components:,} ({progress:.1f}%) -  {len(spatiotemporal_samples):,} ", end='\r')
        component_id = component['cp']
        center_lat = component['lat_mean']
        center_lon = component['lon_mean']
        start_date = component['dtime_min']
        
        #  ()
        lat_per_km = 1 / 110.574
        lon_per_km = 1 / (111.320 * np.cos(np.radians(center_lat)))
        
        patch_radius_deg = (patch_size_km / 2) * lat_per_km
        
        spatial_bounds = {
            'lat_min': center_lat - patch_radius_deg,
            'lat_max': center_lat + patch_radius_deg,
            'lon_min': center_lon - patch_radius_deg, 
            'lon_max': center_lon + patch_radius_deg
        }
        
        #  ()
        time_window_start = start_date - pd.Timedelta(days=time_steps)
        time_window_end = start_date - pd.Timedelta(days=1)
        
        # 
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
        
        # :
        if processed_count <= 5:  # 5
            print(f"\n    {processed_count}: ID={component_id}, ={time_window_start.date()}  {time_window_end.date()}")
            print(f"      : lat[{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}], lon[{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
            print(f"       {len(window_events)} ")
            if len(window_events) > 0:
                print(f"      : {window_events['dtime'].min()}  {window_events['dtime'].max()}")
        
        # 
        grid_size = int(patch_size_km / spatial_resolution_km)  # 25x25
        if extract_aligned_features is None:
            error_msg = (
                "❌ extract_aligned_features,.\n"
                "   DLLload(Windows).\n"
                "   :\n"
                "   1. (numpy, scipy, xarray)\n"
                "   2. Visual C++\n"
                "   3. loading,\n"
                "   : code/fire_equality/datamodules/feature_alignment.py"
            )
            raise ImportError(error_msg)
        # ISO3()
        iso3_code = extract_iso3_from_events(window_events, spatial_bounds, preprocessed_data.get('events'))
        # :time_steps1
        feature_cube = extract_aligned_features(
            spatial_bounds=spatial_bounds,
            time_window_start=time_window_start,  # time_steps
            time_window_end=time_window_end,  # 1(580)
            grid_size=grid_size,
            fire_date=pd.Timestamp(start_date).to_pydatetime(),  # 
            fire_year=pd.Timestamp(start_date).year,
            iso3=iso3_code,
            data_dir='dataset',
            project='ee-your-gee-project'
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
    
    print()  # 
    
    # 
    samples_with_features = sum(1 for s in spatiotemporal_samples if s['features'].sum() > 0)
    samples_without_features = len(spatiotemporal_samples) - samples_with_features
    
    print(f"✅  {len(spatiotemporal_samples):,} ")
    if samples_without_features > 0:
        print(f"   ⚠️  : {samples_without_features} 0()")
    print(f"   ✅ {samples_with_features} ")
    
    return spatiotemporal_samples


def create_feature_cube(*args, **kwargs):
    """
    . extract_aligned_features 8.
    """
    raise NotImplementedError("create_feature_cube . extract_aligned_features.")


# PyTorch

class FireTracksDataset(Dataset):
    """
    FireTracks PyTorch
    
    forPyTorch
    """
    
    def __init__(self, spatiotemporal_samples, target_type='severity'):
        """
        FireTracks PyTorch
        
        Args:
            spatiotemporal_samples: 
            target_type:  ('severity', 'frp', 'duration')
        """
        self.samples = spatiotemporal_samples
        self.target_type = target_type
        
    def __len__(self):
        """"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        
        
        Args:
            idx: 
        
        Returns:
            tuple: (features, target)
                - features: torch.Tensor, shape [C, T, H, W]
                - target: torch.Tensor, (target_type)
        """
        sample = self.samples[idx]
        
        # : [channels, timesteps, height, width]
        features = torch.from_numpy(sample['features']).float()
        features = features.permute(3, 0, 1, 2)  #  [T,H,W,C]  [C,T,H,W]
        
        # 
        if self.target_type == 'binary_classification':
            # :sampletarget
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
        
        
        Args:
            idx: 
        
        Returns:
            dict: 
        """
        return self.samples[idx]['metadata']


# 

def create_convLSTM_ready_dataset(data_directory, output_path=None, config=None):
    """
    FireTracks
    
    loading,, PyTorch
    
    Args:
        data_directory: FireTracks
        output_path:  ()
        config: configuration,:
            - patch_size_km:  (km),25
            - time_steps: stride (),10
            - target_years:,(2002, 2020)
            - target_type:  ('severity', 'frp', 'duration'),'severity'
            - batch_size:,32
    
    Returns:
        dict: :
            - 'dataset': FireTracksDataset
            - 'dataloader': DataLoader
            - 'config': configuration
            - 'preprocessed_data': 
       ,None
    """
    if config is None:
        config = {
            'patch_size_km': 25,
            'time_steps': 10,
            'target_years': (2002, 2020),
            'target_type': 'severity',
            'batch_size': 32,
            'max_samples': None  # (for,None)
        }
    
    print("🚀 FireTracks...")
    
    # 1. load()
    # :,events 
    print("\n[1/5] loadFireTracks...")
    start_year, end_year = config['target_years']
    # events 
    events_year_range = config['target_years']
    print(f"   Events : {events_year_range} (loading,)")
    
    datasets = load_firetracks_dataset(
        data_directory, 
        year_range=events_year_range
    )
    
    #  components 
    if datasets and 'components' in datasets:
        # load components,
        print(f"   load components,: {config['target_years']}")
        import os
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year+1}-01-01"
        try:
            datasets['components'] = pd.read_hdf(
                f'{data_directory}/cp.h5',
                where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
            )
            print(f"   ✅ Components: {len(datasets['components']):,} ")
        except Exception as e:
            print(f"   ⚠️  load components,load")
    if datasets is None:
        return None
    
    # 2. 
    print("\n[2/5] ...")
    import time
    start_time = time.time()
    preprocessed_data = preprocess_firetracks_data(
        datasets, 
        target_year_range=config['target_years']
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  : {elapsed_time:.1f} ")
    
    # 3. 
    print("\n[3/5] ...")
    #,(for)
    max_samples = config.get('max_samples', None)
    if max_samples:
        print(f"   ⚠️  :  {max_samples:,} ")
    
    start_time = time.time()
    spatiotemporal_samples = create_spatiotemporal_patches(
        preprocessed_data,
        patch_size_km=config['patch_size_km'],
        time_steps=config['time_steps'],
        max_samples=max_samples
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  : {elapsed_time:.1f} ")
    
    if not spatiotemporal_samples:
        print("❌ ")
        return None
    
    # 4. PyTorch
    print("\n[4/5] PyTorch...")
    dataset = FireTracksDataset(
        spatiotemporal_samples, 
        target_type=config['target_type']
    )
    
    # 5. load
    print("\n[5/5] DataLoader...")
    # Windowsnum_workers>0,0
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print("\n✅ FireTracks!")
    print(f"   - : {len(dataset):,} ")
    if len(dataset) > 0:
        sample_features, sample_target = dataset[0]
        print(f"   - : {sample_features.shape}")
        print(f"   - : {config['target_type']}")
        print(f"   - : {len(dataloader):,} ")
    
    # 7.  ()
    if output_path:
        print(f"\n💾 : {output_path}...")
        try:
            #,datasetdataloader()
            # config
            torch.save({
                'spatiotemporal_samples': spatiotemporal_samples,  # 
                'config': config,
                'metadata': {
                    'total_samples': len(dataset),
                    'input_shape': dataset[0][0].shape if len(dataset) > 0 else None,
                    'feature_channels': ['FRP', '', '', '']
                }
            }, output_path)
            print(f"✅ : {output_path}")
            print(f"   :,configdatasetdataloader")
        except Exception as e:
            print(f"⚠️  : {e}")
    
    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'config': config,
        'preprocessed_data': preprocessed_data
    }


# 

def test_firetracks_pipeline():
    """
    FireTracks()
    
   ,create_pixel_level_binary_classification_dataset
    FireTracksforConvLSTM.
    
    Returns:
        dict:,dataset, dataloader,None
    """
    # configuration
    config = {
        'use_modis_api': True,  # MODIS API(True)
        'gee_project': 'ee-your-gee-project',  # GEE
        'cache_dir': 'dataset',  # (dataset)
        'patch_size_km': 25,
        'time_steps': 10,
        'target_years': (2017, 2018),  # 
        'batch_size': 32,
        'neg_pos_ratio': 2.0,  # 2
        'max_samples': 10  # 10for(None)
    }
    
    # 
    # :
    result = create_pixel_level_binary_classification_dataset(
        data_directory='dataset/firetracks_data',  # FireTracks
        output_path='dataset/processed_firetracks_pixel_binary.pth',  # .pth
        config=config
    )
    
    if result is not None:
        # load
        dataloader = result['dataloader']
        features, targets = next(iter(dataloader))
        
        print(f"\n📊 load:")
        print(f"   - Batch: {features.shape[0]}")
        print(f"   - : {features.shape}")  # [batch, channels, timesteps, height, width]
        print(f"   - : {targets.shape}")
        print(f"   - :  (0=, 1=)")
        print(f"   - : {features.dtype}")
        
        # 
        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()
        print(f"   - : {positive_count}")
        print(f"   - : {negative_count}")
        
        return result
    else:
        print("❌ ")
        return None


# ============================================================================
# :
# ============================================================================

def _process_single_pixel(pixel_event, ignition_date, component_id, time_window_start, 
                          time_window_end, patch_radius_deg, grid_size, events, 
                          pixel_lc_dict, use_modis_api, use_events_lc, events_lc,
                          use_index_matching, data_dir, project):
    """
    (for)
    
    Returns:
        dict:,None
    """
    try:
        pixel_lat = pixel_event['lat']
        pixel_lon = pixel_event['lon']
        pixel_idx = pixel_event.name
        
        # (25km×25km)
        pixel_spatial_bounds = {
            'lat_min': pixel_lat - patch_radius_deg,
            'lat_max': pixel_lat + patch_radius_deg,
            'lon_min': pixel_lon - patch_radius_deg,
            'lon_max': pixel_lon + patch_radius_deg
        }
        
        # (for)
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
        
        # ()
        if extract_aligned_features is None:
            error_msg = (
                "❌ extract_aligned_features,.\n"
                "   DLLload(Windows).\n"
                "   :\n"
                "   1. (numpy, scipy, xarray)\n"
                "   2. Visual C++\n"
                "   3. loading,\n"
                "   : code/fire_equality/datamodules/feature_alignment.py"
            )
            raise ImportError(error_msg)
        # ISO3()
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
        
        # 
        land_cover = None
        
        # MODIS API
        if use_modis_api:
            if pixel_idx in pixel_lc_dict:
                lc = pixel_lc_dict[pixel_idx]
                if lc is not None and lc != 255:  # 255
                    land_cover = lc
        # use_modis_api=False,events_lc
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
            'target': 1,  # 
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
        #,
        return {'error': str(e), 'pixel_idx': pixel_idx}


def create_pixel_level_positive_samples(preprocessed_data, patch_size_km=25, time_steps=10, 
                                        spatial_resolution_km=1, max_samples=None, target_year_range=None,
                                        use_modis_api=True, project='ee-your-gee-project',
                                        max_consecutive_errors=5, parallel_workers=2):
    """
    :
    
    :
    (y_t = 1):
    
    Args:
        preprocessed_data: (severe_componentsevents)
        patch_size_km:  (km)
        time_steps: stride (),
        spatial_resolution_km:  (km)
        max_samples: (for,None)
        target_year_range:  (start_year, end_year),for
        use_modis_api: MODIS API(True,)
        project: GEE('ee-your-gee-project')
        max_consecutive_errors:,(5)
    
    Returns:
        list:,:
            - pixel_lat: 
            - pixel_lon: 
            - pixel_date: 
            - features:  [time_steps, grid_size, grid_size, channels]
            - target: 1 ()
            - land_cover: ()
            - metadata: 
    """
    print("\n" + "="*60)
    print("...")
    print("="*60)
    
    if 'severe_components' not in preprocessed_data:
        raise ValueError(" 'severe_components'")
    
    severe_components = preprocessed_data['severe_components']
    events = preprocessed_data['events']
    events_lc = preprocessed_data.get('events_lc', None)
    
    # eventsiso3(forGDP)
    has_iso3_column = 'iso3' in events.columns if events is not None and len(events) > 0 else False
    if not has_iso3_column:
        # countryiso3
        if events is not None and len(events) > 0 and 'country' in events.columns:
            print(f"   💡 eventsiso3,country...")
            add_iso3_to_dataframe(events, country_column='country')
            has_iso3_column = 'iso3' in events.columns
    
    if not has_iso3_column:
        print(f"   ⚠️  :eventsiso3,GDP0")
        print(f"      :FireTracks,iso3GDP")
    else:
        # iso3
        iso3_valid = events['iso3'].dropna()
        iso3_valid = iso3_valid[iso3_valid != '']
        if len(iso3_valid) > 0:
            print(f"   ✅ eventsiso3, {len(iso3_valid)} / {len(events)} ")
        else:
            print(f"   ⚠️  eventsiso3,GDP0")
    
    # 
    severe_components['dtime_min'] = pd.to_datetime(severe_components['dtime_min'])
    events['dtime'] = pd.to_datetime(events['dtime'])
    if events_lc is not None:
        events_lc['dtime'] = pd.to_datetime(events_lc['dtime'])
    
    # (forMODIS)
    if target_year_range:
        year = target_year_range[0]  # 
    else:
        # 
        year = severe_components['dtime_min'].dt.year.iloc[0] if len(severe_components) > 0 else 2017
    
    #  landcover 
    use_index_matching = False  # initialize
    if use_modis_api and HAS_MODIS_LC:
        print(f"   🚀  landcover (dataset/LandCover/)")
        use_events_lc = False
    else:
        print(f"   🔄 events_lc")
        use_events_lc = True
        # eventsevents_lc(for)
        if events_lc is not None:
            if len(events) == len(events_lc):
                # (1000)
                sample_size = min(1000, len(events))
                if events.index[:sample_size].equals(events_lc.index[:sample_size]):
                    use_index_matching = True
                    print(f"   ✅ eventsevents_lc,")
                else:
                    print(f"   ⚠️  eventsevents_lc,")
            else:
                print(f"   ⚠️  eventsevents_lc ({len(events)} vs {len(events_lc)}),")
    
    total_components = len(severe_components)
    # max_samples,
    #,
    target_samples = max_samples if max_samples else None
    
    # target_year_rangemax_samples,
    if target_samples and target_year_range:
        start_year, end_year = target_year_range
        num_years = end_year - start_year + 1
        samples_per_year = max(1, target_samples // num_years)  # 1
        print(f"⚠️  :  {target_samples:,} ")
        print(f"   : {num_years}, {samples_per_year:,} ")
        
        # grouping
        severe_components['year'] = severe_components['dtime_min'].dt.year
        components_by_year = {}
        for year in range(start_year, end_year + 1):
            year_components = severe_components[severe_components['year'] == year]
            if len(year_components) > 0:
                components_by_year[year] = year_components
                print(f"    {year}: {len(year_components):,} ")
        
        # ()
        year_targets = {}
        remaining_samples = target_samples
        for year in sorted(components_by_year.keys()):
            if year == sorted(components_by_year.keys())[-1]:
                # 
                year_targets[year] = remaining_samples
            else:
                year_targets[year] = samples_per_year
                remaining_samples -= samples_per_year
        
        print(f"   : {year_targets}")
    else:
        components_by_year = None
        year_targets = None
    
    if target_samples and not year_targets:
        print(f"⚠️  :  {target_samples:,} ()")
    
    print(f": {total_components:,} ")
    if parallel_workers > 1:
        print(f"   🚀 : {parallel_workers} ")
    
    positive_samples = []
    grid_size = int(patch_size_km / spatial_resolution_km)
    
    progress_interval = max(1, total_components // 100)
    processed_count = 0
    skipped_count = 0
    skipped_reasons = {}  # 
    consecutive_errors = 0  # 
    
    # 
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    #,for
    samples_lock = threading.Lock()
    error_lock = threading.Lock()
    
    #,
    if components_by_year and year_targets:
        for year in sorted(components_by_year.keys()):
            year_components = components_by_year[year]
            year_target = year_targets[year]
            
            if len(positive_samples) >= target_samples:
                break
            
            print(f"\n {year} (: {year_target:,} )...")
            
            for idx, component in year_components.iterrows():
                #,
                year_current_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
                if year_current_count >= year_target:
                    break
                
                #,
                if target_samples and len(positive_samples) >= target_samples:
                    break
                
                # ()
                processed_count += 1
                
                if processed_count % progress_interval == 0 or processed_count == 1:
                    progress = processed_count / total_components * 100
                    target_info = f"(: {target_samples},  {year} : {year_target})" if target_samples else ""
                    print(f"\r   : {processed_count:,}/{total_components:,} ({progress:.1f}%) -  {len(positive_samples):,} {target_info} ( {skipped_count:,} )", end='', flush=True)
                
                component_id = component['cp']
                start_date = component['dtime_min']
                center_lat = component['lat_mean']
                center_lon = component['lon_mean']
                
                # :
                time_window_start = start_date - pd.Timedelta(days=time_steps)
                skipped_reason = None
                if target_year_range is not None:
                    start_year_check, _ = target_year_range
                    year_start_date = pd.Timestamp(f'{start_year_check}-01-01')
                    if time_window_start < year_start_date:
                        skipped_reason = ""
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                        continue
                
                time_window_end = start_date - pd.Timedelta(days=1)
                ignition_date = start_date.date()
                
                # 
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
                    skipped_reason = ""
                    skipped_count += 1
                    if skipped_reason not in skipped_reasons:
                        skipped_reasons[skipped_reason] = 0
                    skipped_reasons[skipped_reason] += 1
                    continue
                
                # 
                lat_per_km = 1 / 110.574
                patch_radius_deg = (patch_size_km / 2) * lat_per_km
                
                # 
                pixel_coords = []
                pixel_indices = []
                if use_modis_api:
                    if not HAS_RASTERIO:
                        error_msg = f"❌  landcover !\n"
                        error_msg += f"   : rasterio DLLload\n"
                        error_msg += f"   : pip install rasterio \n"
                        error_msg += f"  ,events_lc"
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
                
                # 
                for idx_pixel, pixel_event in ignition_events.iterrows():
                    # 
                    year_current_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
                    if year_current_count >= year_target:
                        break
                    
                    # 
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
                        skipped_reason = ""
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                    else:
                        positive_samples.append(result)
                        consecutive_errors = 0
            
            year_final_count = sum(1 for s in positive_samples if s.get('pixel_date') and pd.Timestamp(s['pixel_date']).year == year)
            print(f"\n    {year} : {year_final_count:,}/{year_target:,} ")
    
    #,
    else:
        for idx, component in severe_components.iterrows():
            #,
            if target_samples and len(positive_samples) >= target_samples:
                break
            
            processed_count += 1
        
            if processed_count % progress_interval == 0 or processed_count == 1:
                progress = processed_count / total_components * 100
                target_info = f"(: {target_samples})" if target_samples else ""
                # \r,
                print(f"\r   : {processed_count:,}/{total_components:,} ({progress:.1f}%) -  {len(positive_samples):,} {target_info} ( {skipped_count:,} )", end='', flush=True)
            
            component_id = component['cp']
            start_date = component['dtime_min']
            center_lat = component['lat_mean']
            center_lon = component['lon_mean']
            
            # :
            # target_year_range,
            time_window_start = start_date - pd.Timedelta(days=time_steps)
            skipped_reason = None
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                #,
                # 
                if time_window_start < year_start_date:
                    # ()
                    skipped_reason = ""
                    skipped_count += 1
                    if skipped_reason not in skipped_reasons:
                        skipped_reasons[skipped_reason] = 0
                    skipped_reasons[skipped_reason] += 1
                    continue
            else:
                #,
                pass
            
            time_window_end = start_date - pd.Timedelta(days=1)
            ignition_date = start_date.date()
            
            # ()
            # ()
            #,
            # 
            search_radius_deg = 0.5  # 55km,
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
                (events['cp'] == component_id)  # 
            ].copy()
            
            if len(ignition_events) == 0:
                #,
                skipped_reason = ""
                skipped_count += 1
                if skipped_reason not in skipped_reasons:
                    skipped_reasons[skipped_reason] = 0
                skipped_reasons[skipped_reason] += 1
                continue
            
            # ()
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            # MODIS API,
            pixel_coords = []
            pixel_indices = []
            if use_modis_api:
                if not HAS_RASTERIO:
                    error_msg = f"❌  landcover !\n"
                    error_msg += f"   : rasterio DLLload\n"
                    error_msg += f"   : pip install rasterio \n"
                    error_msg += f"  ,events_lc"
                    raise ImportError(error_msg)
                
                # forlandcover
                year = start_date.year
                
                for idx, pixel_event in ignition_events.iterrows():
                    pixel_coords.append((pixel_event['lat'], pixel_event['lon']))
                    pixel_indices.append(idx)
                
                if len(pixel_coords) > 0:
                    print(f"       {len(pixel_coords)} ()...", end='\r')
                    lc_results = get_landcover_batch_gee(pixel_coords, year, lc_type='LC_Type1', 
                                                         batch_size=1000, project=project)
                    pixel_lc_dict = {pixel_indices[i]: lc for i, lc in enumerate(lc_results)}
                    success_count = sum(1 for lc in lc_results if lc is not None and lc != 255)
                    print(f"     , {success_count}/{len(pixel_coords)} ")
                    
                    if success_count == 0:
                        local_file = os.path.join('dataset', 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
                        error_msg = f"❌ !\n"
                        error_msg += f"    {len(pixel_coords)},\n"
                        error_msg += f"   :\n"
                        error_msg += f"   1. : {local_file}\n"
                        error_msg += f"   2. \n"
                        error_msg += f"   3. \n"
                        error_msg += f"   "
                        raise RuntimeError(error_msg)
                else:
                    pixel_lc_dict = {}
            else:
                pixel_lc_dict = {}
            
            # 
            if parallel_workers > 1 and len(ignition_events) > 1:
                
                # 
                import time
                pixel_start_time = time.time()
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    futures = []
                    for idx, pixel_event in ignition_events.iterrows():
                        # 
                        with samples_lock:
                            if target_samples and len(positive_samples) >= target_samples:
                                #,
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
                    
                    # 
                    completed_count = 0
                    for future in as_completed(futures):
                        # 
                        with samples_lock:
                            if target_samples and len(positive_samples) >= target_samples:
                                #,
                                for remaining_future in futures:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                        
                        result = future.result()
                        completed_count += 1
                        
                        with samples_lock:
                            if 'error' in result:
                                # 
                                error_msg = result['error']
                                pixel_idx = result.get('pixel_idx', 'unknown')
                                with error_lock:
                                    consecutive_errors += 1
                                    skipped_reason = ""
                                    skipped_count += 1
                                    if skipped_reason not in skipped_reasons:
                                        skipped_reasons[skipped_reason] = 0
                                    skipped_reasons[skipped_reason] += 1
                                
                                # ()
                                if len(positive_samples) < 3 or consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                                    print(f"    ⚠️   {pixel_idx}  ({consecutive_errors}/{max_consecutive_errors}): {error_msg[:150]}")
                            else:
                                # 
                                # ()
                                if target_samples and len(positive_samples) >= target_samples:
                                    #,
                                    continue
                                positive_samples.append(result)
                                #,
                                if target_samples and len(positive_samples) >= target_samples:
                                    for remaining_future in futures:
                                        if not remaining_future.done():
                                            remaining_future.cancel()
                        with error_lock:
                            consecutive_errors = 0
                        
                        # :10
                        if completed_count % 10 == 0 or completed_count == len(futures):
                                    pixel_elapsed = time.time() - pixel_start_time
                                    avg_time_per_pixel = pixel_elapsed / completed_count if completed_count > 0 else 0
                                    try:
                                        from .feature_cache import get_cache_stats
                                        from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
                                        cache_stats = get_cache_stats()
                                        netcdf_stats = get_netcdf_cache_stats()
                                        if cache_stats['hits'] + cache_stats['misses'] > 0:
                                            cache_info = f", : {cache_stats['hit_rate']:.1%}"
                                        else:
                                            cache_info = ""
                                        if netcdf_stats['cached_datasets'] > 0:
                                            netcdf_info = f", NetCDF: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']}"
                                        else:
                                            netcdf_info = ""
                                    except:
                                        cache_info = ""
                                        netcdf_info = ""
                                    
                                    # ()
                                    if processed_count % progress_interval == 0:
                                        print(f"\r   : {processed_count:,}/{total_components:,} -  {len(positive_samples):,}  -  {avg_time_per_pixel:.1f}/{cache_info}{netcdf_info}", end='', flush=True)
            else:
                # (,for)
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
                        # 
                        error_msg = result['error']
                        pixel_idx = result.get('pixel_idx', idx)
                        consecutive_errors += 1
                        
                        try:
                            coord_str = f"({pixel_event['lat']:.4f}, {pixel_event['lon']:.4f})"
                        except:
                            coord_str = f"(: {pixel_idx})"
                        
                        if len(positive_samples) < 3 or consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                            print(f"    ⚠️   {coord_str}  ({consecutive_errors}/{max_consecutive_errors}): {error_msg[:150]}")
                        
                        skipped_reason = ""
                        skipped_count += 1
                        if skipped_reason not in skipped_reasons:
                            skipped_reasons[skipped_reason] = 0
                        skipped_reasons[skipped_reason] += 1
                        
                        # 
                        if consecutive_errors >= max_consecutive_errors and len(positive_samples) > 0:
                            error_summary = f"\n❌  {consecutive_errors},\n"
                            error_summary += f"    {len(positive_samples):,} \n"
                            error_summary += f"   : {skipped_count:,} \n"
                            error_summary += f"   : {coord_str}\n"
                            error_summary += f"   : {error_msg[:200]}"
                            raise RuntimeError(error_summary)
                        elif consecutive_errors >= max_consecutive_errors * 2 and len(positive_samples) == 0:
                            error_summary = f"\n❌  {consecutive_errors},\n"
                            error_summary += f"   : {skipped_count:,} \n"
                            error_summary += f"   : {coord_str}\n"
                            error_summary += f"   : {error_msg[:200]}\n"
                            error_summary += f"   "
                            raise RuntimeError(error_summary)
                    else:
                        # 
                        # 
                        if target_samples and len(positive_samples) >= target_samples:
                            #,
                            break
                        positive_samples.append(result)
                        consecutive_errors = 0
                        
                        # :10
                        if pixel_count % 10 == 0 or pixel_count == len(ignition_events):
                            pixel_elapsed = time.time() - pixel_start_time
                            avg_time_per_pixel = pixel_elapsed / pixel_count if pixel_count > 0 else 0
                            try:
                                from .feature_cache import get_cache_stats
                                from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
                                cache_stats = get_cache_stats()
                                netcdf_stats = get_netcdf_cache_stats()
                                if cache_stats['hits'] + cache_stats['misses'] > 0:
                                    cache_info = f", : {cache_stats['hit_rate']:.1%}"
                                else:
                                    cache_info = ""
                                if netcdf_stats['cached_datasets'] > 0:
                                    netcdf_info = f", NetCDF: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']}"
                                else:
                                    netcdf_info = ""
                            except:
                                cache_info = ""
                                netcdf_info = ""
                            
                            # 
                            if processed_count % progress_interval == 0:
                                print(f"\r   : {processed_count:,}/{total_components:,} -  {len(positive_samples):,}  -  {avg_time_per_pixel:.1f}/{cache_info}{netcdf_info}", end='', flush=True)
    
    # 
    print("\r" + " " * 100 + "\r", end='', flush=True)  # 
    print(f"✅  {len(positive_samples):,} ")
    
    # 
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
            print(f"\n📊 :")
            total_shown = 0
            for year in sorted(year_counts.keys()):
                count = year_counts[year]
                total_shown += count
                percentage = (count / len(positive_samples) * 100) if len(positive_samples) > 0 else 0
                print(f"   {year}: {count:,}  ({percentage:.1f}%)")
            if total_shown < len(positive_samples):
                print(f"   : {len(positive_samples) - total_shown:,} ")
    
    if skipped_count > 0:
        print(f"\n⚠️   {skipped_count:,},:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {reason}: {count:,} ")
    
    return positive_samples


def save_positive_samples(positive_samples, config, filepath):
    """
    
    
    Args:
        positive_samples: 
        config: configuration(for)
        filepath: 
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # (numpy,torch.save)
    # : datetime.date convert to string, PyTorch 2.6+ load
    import datetime
    positive_samples_serializable = []
    for sample in positive_samples:
        sample_copy = sample.copy()
        #  datetime.date 
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
    print(f"✅ : {filepath}")
    print(f"   - : {len(positive_samples):,}")
    print(f"   - : {save_data['metadata']['samples_with_lc']:,}")


def load_positive_samples(filepath, verify_config=None):
    """
    load
    
    Args:
        filepath: 
        verify_config: configuration,for(,configuration)
    
    Returns:
        tuple: (positive_samples, config), (None, None)
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️  : {filepath}")
            return None, None
        
        print(f"📂 load: {filepath}")
        # PyTorch 2.6+  weights_only=True, datetime.date
        import datetime
        try:
            # load(PyTorch 2.6+)
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            #,
            try:
                torch.serialization.add_safe_globals([datetime.date])
                save_data = torch.load(filepath, map_location='cpu')
            except:
                #  weights_only=False
                save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        positive_samples = save_data.get('positive_samples', [])
        config = save_data.get('config', {})
        metadata = save_data.get('metadata', {})
        
        #  datetime.date ()
        import datetime
        for sample in positive_samples:
            if 'date' in sample and isinstance(sample['date'], str):
                try:
                    sample['date'] = datetime.date.fromisoformat(sample['date'])
                except:
                    pass  #,
        
        print(f"✅ load:")
        print(f"   - : {len(positive_samples):,}")
        print(f"   - : {metadata.get('samples_with_lc', 'N/A')}")
        print(f"   - : {metadata.get('saved_time', 'N/A')}")
        
        # configuration()
        if verify_config is not None:
            key_params = ['patch_size_km', 'time_steps', 'target_years']
            mismatches = []
            for key in key_params:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        mismatches.append(f"{key}: {verify_config[key]} != {config[key]}")
            
            # max_samples:max_samples,
            # 
            if 'max_samples' in verify_config and verify_config['max_samples'] is not None:
                max_samples = verify_config['max_samples']
                if len(positive_samples) > max_samples:
                    print(f"⚠️   ({len(positive_samples):,})  ({max_samples:,})")
                    print(f"    {max_samples:,} ")
                    positive_samples = positive_samples[:max_samples]
                    print(f"   ✅  {len(positive_samples):,} ")
            
            if mismatches:
                print(f"⚠️  configuration:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                print(f"   :")
                return None, None
            else:
                print(f"   ✅ configuration")
        
        return positive_samples, config
        
    except Exception as e:
        print(f"❌ load: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_negative_pool(negative_pool, config, filepath):
    """
    
    
    Args:
        negative_pool: 
        config: configuration(for)
        filepath: 
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 
    # :events,
    #  datetime.date convert to string
    import datetime
    negative_pool_serializable = negative_pool.copy()
    
    #  patch_candidates 
    if 'patch_candidates' in negative_pool_serializable:
        patch_candidates = negative_pool_serializable['patch_candidates']
        for patch_key, patch_info in patch_candidates.items():
            if 'date' in patch_info and isinstance(patch_info['date'], datetime.date):
                patch_info['date'] = patch_info['date'].isoformat()
    
    #  all_fire_dates 
    if 'all_fire_dates' in negative_pool_serializable:
        all_fire_dates = negative_pool_serializable['all_fire_dates']
        if isinstance(all_fire_dates, set):
            #  set 
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
    print(f"✅ : {filepath}")
    print(f"   - : {save_data['metadata']['total_candidates']:,}")


def load_negative_pool(filepath, verify_config=None):
    """
    load
    
    Args:
        filepath: 
        verify_config: configuration,for(,configuration)
    
    Returns:
        tuple: (negative_pool, config), (None, None)
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️  : {filepath}")
            return None, None
        
        print(f"📂 load: {filepath}")
        # PyTorch 2.6+  weights_only=True, datetime.date
        import datetime
        try:
            # load(PyTorch 2.6+)
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            #,
            try:
                torch.serialization.add_safe_globals([datetime.date])
                save_data = torch.load(filepath, map_location='cpu')
            except:
                #  weights_only=False
                save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        negative_pool = save_data.get('negative_pool', {})
        config = save_data.get('config', {})
        metadata = save_data.get('metadata', {})
        
        #  datetime.date ()
        import datetime
        #  patch_candidates 
        if 'patch_candidates' in negative_pool:
            patch_candidates = negative_pool['patch_candidates']
            for patch_key, patch_info in patch_candidates.items():
                if 'date' in patch_info and isinstance(patch_info['date'], str):
                    try:
                        patch_info['date'] = datetime.date.fromisoformat(patch_info['date'])
                    except:
                        pass  #,
        
        #  all_fire_dates
        if 'all_fire_dates' in negative_pool:
            all_fire_dates = negative_pool['all_fire_dates']
            if isinstance(all_fire_dates, list):
                #  datetime.date
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
        
        print(f"✅ load:")
        print(f"   - : {metadata.get('total_candidates', 'N/A'):,}")
        print(f"   - : {metadata.get('saved_time', 'N/A')}")
        
        # configuration()
        if verify_config is not None:
            key_params = ['patch_size_km', 'target_years', 'fast_mode']
            mismatches = []
            for key in key_params:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        mismatches.append(f"{key}: {verify_config[key]} != {config[key]}")
            
            if mismatches:
                print(f"⚠️  configuration:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                print(f"   :")
                return None, None
            else:
                print(f"   ✅ configuration")
        
        return negative_pool, config
        
    except Exception as e:
        print(f"❌ load: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_negative_samples(negative_samples, config, filepath):
    """
    (for)
    
    Args:
        negative_samples: 
        config: configuration(for)
        filepath: 
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # ( datetime.date convert to string)
    import datetime
    negative_samples_serializable = []
    for sample in negative_samples:
        sample_copy = sample.copy()
        #  datetime.date 
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
    print(f"✅ : {filepath}")
    print(f"   - : {len(negative_samples):,}")


def load_negative_samples(filepath, verify_config=None):
    """
    load(for)
    
    Args:
        filepath: 
        verify_config: configuration,for
    
    Returns:
        tuple: (negative_samples, config), (None, None)
    """
    try:
        if not os.path.exists(filepath):
            return None, None
        
        print(f"📂 load: {filepath}")
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
        
        #  datetime.date 
        for sample in negative_samples:
            if 'pixel_date' in sample and isinstance(sample['pixel_date'], str):
                try:
                    sample['pixel_date'] = datetime.date.fromisoformat(sample['pixel_date'])
                except:
                    pass
        
        print(f"✅ load:")
        print(f"   - : {len(negative_samples):,}")
        if metadata.get('samples_by_lc'):
            print(f"   - :")
            for lc, count in sorted(metadata['samples_by_lc'].items()):
                print(f"      {lc}: {count:,} ")
        
        # configuration()
        if verify_config:
            # configuration
            key_configs = ['patch_size_km', 'time_steps', 'target_years']
            for key in key_configs:
                if key in verify_config and key in config:
                    if verify_config[key] != config[key]:
                        print(f"   ⚠️  configuration: {key} (: {config[key]}, : {verify_config[key]})")
                        return None, None
        
        return negative_samples, config
    except Exception as e:
        print(f"❌ load: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_positive_sample_land_cover_distribution(positive_samples):
    """
    
    
    Args:
        positive_samples: 
    
    Returns:
        dict:  {land_cover_type: count}
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
    
    print(f"\n:")
    print(f"  : {total:,}")
    print(f"  : {total - samples_without_lc:,}")
    print(f"  : {samples_without_lc:,}")
    
    if land_cover_counts:
        print(f"\n  :")
        for lc, count in sorted(land_cover_counts.items(), key=lambda x: x[1], reverse=True):
            prop = land_cover_proportions[lc]
            print(f"     {lc}: {count:,} ({prop*100:.1f}%)")
    
    return {
        'counts': land_cover_counts,
        'proportions': land_cover_proportions,
        'samples_without_lc': samples_without_lc
    }


def extract_iso3_from_events(window_events, spatial_bounds=None, events=None):
    """
    window_eventseventsISO3
    
    Args:
        window_events: (DataFrame,)
        spatial_bounds: (forevents)
        events: (window_events)
    
    Returns:
        str or None: ISO3,None
    """
    # 1: window_events()
    if window_events is not None and len(window_events) > 0:
        # iso3
        if 'iso3' in window_events.columns:
            # iso3(NaN)
            iso3_counts = window_events['iso3'].dropna()
            iso3_counts = iso3_counts[iso3_counts != '']
            if len(iso3_counts) > 0:
                most_common_iso3 = iso3_counts.value_counts().index[0]
                if pd.notna(most_common_iso3) and str(most_common_iso3).strip() != '':
                    return str(most_common_iso3).strip()
        
        # window_eventsiso3,country,
        if 'iso3' not in window_events.columns and 'country' in window_events.columns:
            try:
                # country_to_iso3
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
    
    # 2: window_eventsiso3,eventsiso3
    if events is not None and spatial_bounds is not None:
        try:
            # iso3
            if 'iso3' in events.columns:
                # (,GDP)
                spatial_mask = (
                    (events['lat'] >= spatial_bounds['lat_min']) & 
                    (events['lat'] <= spatial_bounds['lat_max']) &
                    (events['lon'] >= spatial_bounds['lon_min']) & 
                    (events['lon'] <= spatial_bounds['lon_max'])
                )
                region_events = events[spatial_mask]
                
                if len(region_events) > 0:
                    # iso3(NaN)
                    iso3_counts = region_events['iso3'].dropna()
                    iso3_counts = iso3_counts[iso3_counts != '']
                    if len(iso3_counts) > 0:
                        most_common_iso3 = iso3_counts.value_counts().index[0]
                        if pd.notna(most_common_iso3) and str(most_common_iso3).strip() != '':
                            return str(most_common_iso3).strip()
            
            # iso3,country
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
                        # country_to_iso3
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
            #,()
            pass
    
    return None


# : (lat,lon)->iso3, metadata['iso3']
_covariate_iso3_tree = None
_covariate_iso3_list = None


def lat_lon_to_iso3_from_covariate(lat, lon, max_dist_deg=2.0, data_dir=None):
    """
     dataset/filtered_cleaned_cp_covariate.csv  iso3.
    for .pth  events () iso3,fairness analysis GDP/continent .
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
    """ iso3: events,.for metadata['iso3']."""
    if iso3_from_events is not None and str(iso3_from_events).strip() != '':
        return str(iso3_from_events).strip()
    return lat_lon_to_iso3_from_covariate(pixel_lat, pixel_lon, data_dir=data_dir)


def analyze_fire_seasonality(events, target_year_range):
    """
    
    
    Args:
        events: 
        target_year_range:  (start_year, end_year)
    
    Returns:
        dict:,:
            - 'monthly_counts': 
            - 'monthly_proportions': 
            - 'high_fire_seasons': ()
            - 'low_fire_seasons': ()
    """
    print("\n...")
    
    start_year, end_year = target_year_range
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    
    # 
    events_in_range = events[
        (events['dtime'] >= start_date) & (events['dtime'] <= end_date)
    ].copy()
    
    # 
    events_in_range['month'] = events_in_range['dtime'].dt.month
    monthly_counts = events_in_range['month'].value_counts().sort_index()
    
    # 
    total_events = len(events_in_range)
    monthly_proportions = (monthly_counts / total_events).to_dict()
    
    # 
    avg_count = monthly_counts.mean()
    
    # /
    high_fire_seasons = set(monthly_counts[monthly_counts > avg_count].index)
    low_fire_seasons = set(monthly_counts[monthly_counts <= avg_count].index)
    
    print(f"  : {total_events:,}")
    print(f"  : {avg_count:.1f}")
    print(f"  (): {sorted(high_fire_seasons)}")
    print(f"  (): {sorted(low_fire_seasons)}")
    
    return {
        'monthly_counts': monthly_counts.to_dict(),
        'monthly_proportions': monthly_proportions,
        'high_fire_seasons': high_fire_seasons,
        'low_fire_seasons': low_fire_seasons,
        'avg_count': avg_count
    }


def create_negative_sample_pool(preprocessed_data, target_year_range, patch_size_km=25, 
                                 events_lc=None, spatial_resolution_km=1, fast_mode=False,
                                 positive_lc_types=None, use_modis_api=True, project='ee-your-gee-project'):
    """
    
    
    :
    - :25×25km ConvLSTM
    - :
    
    Args:
        preprocessed_data: 
        target_year_range:  (start_year, end_year)
        patch_size_km: (km),25km
        events_lc: (,for)
        spatial_resolution_km: (km),1km
        fast_mode:,(for),False
    
    Returns:
        dict:,:
            - 'patch_candidates': dict, (center_lat, center_lon, date),
            - 'all_fire_dates': 
            - 'events': (for)
            - 'seasonality': 
            - 'target_year_range': 
            - 'patch_size_km': 
    """
    print("\n" + "="*60)
    print("...")
    print("="*60)
    if fast_mode:
        print("   ⚡ :(for)")
    print(f"  :{patch_size_km}×{patch_size_km}km")
    print(f"  :")
    
    events = preprocessed_data['events']
    components = preprocessed_data.get('components', pd.DataFrame())
    
    # 
    events['dtime'] = pd.to_datetime(events['dtime'])
    if not components.empty:
        components['dtime_min'] = pd.to_datetime(components['dtime_min'])
    
    # 
    seasonality = analyze_fire_seasonality(events, target_year_range)
    
    # 
    start_year, end_year = target_year_range
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # :1, 11, 21(10)
    if fast_mode:
        all_dates_list = sorted([d.date() for d in all_dates if d.day in [1, 11, 21]])
        print(f"   ⚡ :1, 11, 21( {len(all_dates_list):,} )")
    else:
        all_dates_list = sorted(all_dates.date)
    
    # 
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
    
    # 
    #,(>65°N<-65°S),
    lat_min = max(events['lat'].min(), -65.0)
    lat_max = min(events['lat'].max(), 65.0)
    spatial_bounds = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': events['lon'].min(),
        'lon_max': events['lon'].max()
    }
    
    print(f"\n:")
    print(f"  : [{spatial_bounds['lat_min']:.4f}, {spatial_bounds['lat_max']:.4f}] (±65°)")
    print(f"  : [{spatial_bounds['lon_min']:.4f}, {spatial_bounds['lon_max']:.4f}]")
    print(f"  : {len(all_dates_list):,}")
    print(f"  : {len(all_fire_dates):,}")
    
    # ()
    lat_per_km = 1 / 110.574
    patch_radius_deg = (patch_size_km / 2) * lat_per_km
    
    # 1:, MODIS 
    if positive_lc_types is not None and len(positive_lc_types) > 0 and use_modis_api and HAS_MODIS_LC:
        print(f"\n🚀  MODIS : {sorted(positive_lc_types)}")
        print(f"   : MODIS,")
        
        # ()
        
        # 
        #,
        num_samples_per_type = 2000 if fast_mode else 5000
        
        # 
        lc_locations = find_locations_by_landcover(
            positive_lc_types=positive_lc_types,
            spatial_bounds=spatial_bounds,
            year=start_year,
            num_samples_per_type=num_samples_per_type,
            project=project
        )
        
        if len(lc_locations) == 0:
            print("  ⚠️ ,")
            # 
            use_modis_search = False
        else:
            use_modis_search = True
            print(f"  ✅  {len(lc_locations):,} ")
    else:
        use_modis_search = False
        lc_locations = []
    
    # 2:
    if not use_modis_search:
        #,
        # :stride(500),
        if fast_mode:
            grid_step_deg = patch_radius_deg * 500  # :500stride
            print(f"   ⚡ :500stride({grid_step_deg:.4f},{grid_step_deg * 110.574:.1f}km)")
        else:
            grid_step_deg = patch_radius_deg  # :stride
        
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
        
        print(f"\n()...")
        print(f"  : {len(lat_grid):,}")
        print(f"  : {len(lon_grid):,}")
        print(f"  : {len(lat_grid) * len(lon_grid):,}")
        if fast_mode:
            normal_grid_count = int((spatial_bounds['lat_max'] - spatial_bounds['lat_min'] - 2*patch_radius_deg) / patch_radius_deg) * int((spatial_bounds['lon_max'] - spatial_bounds['lon_min'] - 2*patch_radius_deg) / patch_radius_deg)
            fast_grid_count = len(lat_grid) * len(lon_grid)
            reduction_factor = normal_grid_count / fast_grid_count if fast_grid_count > 0 else 1
            print(f"   ⚡ : {fast_grid_count:,} ( {normal_grid_count:,}, {reduction_factor:.1f} )")
        
        # 
        lc_locations = []
        for center_lat in lat_grid:
            for center_lon in lon_grid:
                lc_locations.append({
                    'lat': center_lat,
                    'lon': center_lon,
                    'land_cover': None  #,
                })
    
    #,
    print(f"\n...")
    patch_candidates = {}  #  (center_lat, center_lon, date),
    
    events_in_range = events[
        (events['dtime'] >= start_date) & (events['dtime'] <= end_date)
    ].copy()
    
    #,grouping
    events_by_date = {}
    for date in all_dates_list:
        date_events = events_in_range[events_in_range['dtime'].dt.date == date]
        if len(date_events) > 0:
            events_by_date[date] = date_events
    
    total_checks = len(lc_locations) * len(all_dates_list)
    checked = 0
    print(f"   {total_checks:,} ×...")
    
    for location in lc_locations:
        center_lat = location['lat']
        center_lon = location['lon']
        
        # 
        patch_bounds = {
            'lat_min': center_lat - patch_radius_deg,
            'lat_max': center_lat + patch_radius_deg,
            'lon_min': center_lon - patch_radius_deg,
            'lon_max': center_lon + patch_radius_deg
        }
        
        # 
        for date in all_dates_list:
            checked += 1
            if checked % 10000 == 0:
                print(f"    : {checked:,}/{total_checks:,} ({checked*100/total_checks:.1f}%)")
            
            #,
            if date in events_by_date:
                date_events = events_by_date[date]
                # 
                in_patch = (
                    (date_events['lat'] >= patch_bounds['lat_min']) &
                    (date_events['lat'] <= patch_bounds['lat_max']) &
                    (date_events['lon'] >= patch_bounds['lon_min']) &
                    (date_events['lon'] <= patch_bounds['lon_max'])
                )
                if in_patch.any():
                    continue  #,
            
            #,
            patch_key = (center_lat, center_lon, date)
            patch_candidates[patch_key] = {
                'center_lat': center_lat,
                'center_lon': center_lon,
                'date': date,
                'patch_bounds': patch_bounds,
                'land_cover': location.get('land_cover')  #  MODIS,
            }
    
    print(f"\n✅ :")
    print(f"  (×): {len(patch_candidates):,}")
    
    return {
        'patch_candidates': patch_candidates,
        'all_fire_dates': all_fire_dates,
        'events': events,
        'events_lc': events_lc,
        'seasonality': seasonality,
        'target_year_range': target_year_range,
        'patch_size_km': patch_size_km,
        'spatial_bounds': spatial_bounds,
        'project': project  # project,
    }


def sample_negative_samples_by_land_cover(negative_pool, positive_lc_distribution, 
                                         num_negative_samples,
                                         patch_size_km=25, time_steps=10, 
                                         spatial_resolution_km=1, spatial_bounds=None,
                                         target_year_range=None,
                                         use_modis_api=True, project='ee-your-gee-project',
                                         checkpoint_file=None, checkpoint_interval=100):
    """
    
    
    
    
    Args:
        negative_pool: (create_negative_sample_pool,)
        positive_lc_distribution: (analyze_positive_sample_land_cover_distribution)
        num_negative_samples: (2)
        patch_size_km: 
        time_steps: stride
        spatial_resolution_km: 
        spatial_bounds: (None,events)
        target_year_range: 
    
    Returns:
        list:,
    """
    print("\n" + "="*60)
    print("...")
    print("="*60)
    print("  :")
    print("  :")
    
    patch_candidates = negative_pool['patch_candidates']
    events = negative_pool['events']
    events_lc = negative_pool.get('events_lc', None)
    seasonality = negative_pool.get('seasonality', None)
    target_year_range = target_year_range or negative_pool.get('target_year_range')
    spatial_bounds = spatial_bounds or negative_pool.get('spatial_bounds')
    
    if len(patch_candidates) == 0:
        print("⚠️  :,")
        return []
    
    #,
    if positive_lc_distribution['counts'] == {}:
        print("⚠️ ,")
        return sample_negative_samples_random(negative_pool, num_negative_samples, 
                                            patch_size_km, time_steps, spatial_resolution_km,
                                            target_year_range)
    
    # ()
    lc_proportions = positive_lc_distribution['proportions']
    lc_sample_counts = {}
    
    for lc, proportion in lc_proportions.items():
        lc_sample_counts[lc] = int(num_negative_samples * proportion)
    
    # num_negative_samples
    total_allocated = sum(lc_sample_counts.values())
    if total_allocated < num_negative_samples:
        # 
        remaining = num_negative_samples - total_allocated
        if lc_proportions:
            most_common_lc = max(lc_proportions.items(), key=lambda x: x[1])[0]
            lc_sample_counts[most_common_lc] += remaining
    
    print(f"( {num_negative_samples:,},):")
    for lc, count in sorted(lc_sample_counts.items(), key=lambda x: x[1], reverse=True):
        prop = lc_proportions.get(lc, 0)
        print(f"   {lc}: {count:,}  ({prop*100:.1f}%)")
    
    # 
    events['dtime'] = pd.to_datetime(events['dtime'])
    
    #  landcover 
    if use_modis_api:
        if not HAS_RASTERIO:
            error_msg = f"❌  landcover !\n"
            error_msg += f"   : rasterio \n"
            error_msg += f"   : pip install rasterio\n"
            error_msg += f"  ,events_lc"
            raise ImportError(error_msg)
        
        print(f"   🚀  landcover (dataset/LandCover/)")
        use_events_lc = False
    else:
        print(f"   🔄 events_lc(use_modis_api=False)")
        use_events_lc = True
        if events_lc is not None:
            events_lc['dtime'] = pd.to_datetime(events_lc['dtime'])
            # eventsevents_lc(for)
            if len(events) == len(events_lc):
                # (1000)
                sample_size = min(1000, len(events))
                if events.index[:sample_size].equals(events_lc.index[:sample_size]):
                    use_index_matching = True
                    print("   ✅ eventsevents_lc,")
                else:
                    print("   ⚠️  eventsevents_lc,")
            else:
                print(f"   ⚠️  eventsevents_lc ({len(events)} vs {len(events_lc)}),")
    
    # 
    # MODIS(),events_lc
    print("\n...")
    patch_candidates_with_lc = {}  #  (center_lat, center_lon, date), (patch_info, lc)
    
    # (forMODIS)
    if target_year_range:
        year = target_year_range[0]  # 
    else:
        # 
        sample_date = next(iter(patch_candidates.values()))['date']
        year = sample_date.year if hasattr(sample_date, 'year') else pd.Timestamp(sample_date).year
    
    # 1: MODIS ()
    if use_modis_api:
        if not HAS_RASTERIO:
            error_msg = f"❌  landcover !\n"
            error_msg += f"   : rasterio \n"
            error_msg += f"   : pip install rasterio\n"
            error_msg += f"  ,events_lc"
            raise ImportError(error_msg)
        
        print("  🚀  MODIS ...")
        
        # 
        positive_lc_types = set(positive_lc_distribution['counts'].keys())
        print(f"    : {sorted(positive_lc_types)}")
        
        # ( MODIS )
        has_precomputed_lc = any(
            patch_info.get('land_cover') is not None 
            for patch_info in patch_candidates.values()
        )
        
        if has_precomputed_lc:
            print(f"    ✅ ( MODIS )")
            # 
            for patch_key, patch_info in patch_candidates.items():
                lc = patch_info.get('land_cover')
                if lc is not None and lc != 255:  # 255
                    patch_candidates_with_lc[patch_key] = (patch_info, lc)
        else:
            # 
            print(f"     {len(patch_candidates):,} ...")
            coords = [(patch_info['center_lat'], patch_info['center_lon']) 
                     for patch_info in patch_candidates.values()]
            patch_keys = list(patch_candidates.keys())
            
            lc_results = get_landcover_batch_gee(coords, year, lc_type='LC_Type1', 
                                                 batch_size=1000, project=project)
            
            # patch_candidates
            for patch_key, lc in zip(patch_keys, lc_results):
                if lc is not None and lc != 255:  # 255
                    patch_candidates_with_lc[patch_key] = (patch_candidates[patch_key], lc)
        
        positive_found_count = sum(
            1 for (patch_info, lc) in patch_candidates_with_lc.values()
            if lc in positive_lc_types
        )
        
        print(f"  ✅  {len(patch_candidates_with_lc):,}/{len(patch_candidates):,} ")
        print(f"    : {positive_found_count:,} ")
        
        # 
        found_lc_types = set()
        for patch_key, (patch_info, lc) in patch_candidates_with_lc.items():
            found_lc_types.add(lc)
        
        missing_positive_lc_types = positive_lc_types - found_lc_types
        if missing_positive_lc_types:
            print(f"  ⚠️  :: {sorted(missing_positive_lc_types)}")
            print(f"     ")
        else:
            print(f"  ✅ : {sorted(positive_lc_types)}")
        
        if len(patch_candidates_with_lc) == 0:
            local_file = os.path.join('dataset', 'LandCover', f'modis_mcd12q1_lc_type1_{year}.tif')
            error_msg = f"❌ !\n"
            error_msg += f"    {len(patch_candidates):,},\n"
            error_msg += f"   :\n"
            error_msg += f"   1. : {local_file}\n"
            error_msg += f"   2. \n"
            error_msg += f"   3. \n"
            error_msg += f"   "
            raise RuntimeError(error_msg)
    
    # 2:events_lc(use_modis_api=False)
    elif not use_modis_api:
        if use_events_lc and events_lc is not None and len(events_lc) > 0:
            print("  🔄 events_lc(:7)...")
            
            lat_per_km = 1 / 110.574
            tolerance = 0.005  # ()
            
            # groupingevents,
            all_dates = set(patch_info['date'] for patch_info in patch_candidates.values())
            # :7
            extended_dates = set()
            for date in all_dates:
                date_pd = pd.Timestamp(date)
                for days_offset in range(-7, 8):  # 7
                    extended_dates.add((date_pd + pd.Timedelta(days=days_offset)).date())
            
            # :,grouping
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
            print(f"     {len(extended_dates):,} (7)groupingevents, {len(events_extended):,} ")
            
            total_patches = len(patch_candidates)
            processed = 0
            progress_interval = max(1, total_patches // 20)  # 5%
            
            for patch_key, patch_info in patch_candidates.items():
                processed += 1
                if processed % progress_interval == 0:
                    progress = processed / total_patches * 100
                    print(f"    : {processed:,}/{total_patches:,} ({progress:.1f}%) -  {len(patch_candidates_with_lc):,} ", end='\r')
                
                center_lat, center_lon, date = patch_key
                
                # events_lc
                lc = None
                try:
                    # :7
                    date_pd = pd.Timestamp(date)
                    best_match = None
                    best_distance = float('inf')
                    
                    #,7()
                    date_range = [date] + [(date_pd + pd.Timedelta(days=offset)).date() 
                                          for offset in range(-7, 8) if offset != 0]
                    
                    # (0.5,55km),events
                    max_tolerance = 0.5  # 55km
                    
                    # :(<0.1),
                    early_stop_threshold = 0.1  # 11km,
                    
                    for search_date in date_range:
                        if search_date in events_by_date:
                            date_events = events_by_date[search_date]
                            # 
                            distances = np.sqrt(
                                (date_events['lat'] - center_lat)**2 + 
                                (date_events['lon'] - center_lon)**2
                            )
                            closest_idx = distances.idxmin()
                            min_distance = distances.loc[closest_idx]
                            
                            if min_distance < max_tolerance and min_distance < best_distance:
                                if closest_idx in events_lc.index:
                                    lc_row = events_lc.loc[closest_idx]
                                    # (7)
                                    lc_date = lc_row['dtime'].date()
                                    if abs((lc_date - date).days) <= 7:
                                        best_match = lc_row
                                        best_distance = min_distance
                                        
                                        # :,
                                        if min_distance < early_stop_threshold:
                                            break
                    
                    #,
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
                    #,
                    continue
                
                if lc is not None:
                    patch_candidates_with_lc[patch_key] = (patch_info, lc)
            
            print()  # 
            print(f"    : {len(patch_candidates_with_lc):,}/{len(patch_candidates):,}")
        else:
            print("  ⚠️  events_lc()")
    
    print(f"  : {len(patch_candidates_with_lc):,}/{len(patch_candidates):,}")
    
    # grouping
    patches_by_lc = {}
    for patch_key, (patch_info, lc) in patch_candidates_with_lc.items():
        if lc not in patches_by_lc:
            patches_by_lc[lc] = []
        patches_by_lc[lc].append((patch_key, patch_info))
    
    # 
    print(f"\n  :")
    for lc in sorted(patches_by_lc.keys()):
        print(f"     {lc}: {len(patches_by_lc[lc]):,} ")
    
    # 
    missing_lc_types = []
    for lc, target_count in lc_sample_counts.items():
        if target_count > 0 and lc not in patches_by_lc:
            missing_lc_types.append((lc, target_count))
    
    if missing_lc_types:
        print(f"\n  ⚠️  :")
        for lc, target_count in missing_lc_types:
            print(f"      {lc}:  {target_count:,}, 0")
        print(f"     :,")
        print(f"     Solution:,")
    
    negative_samples = []
    grid_size = int(patch_size_km / spatial_resolution_km)
    
    # (for)
    total_missing_samples = sum(count for lc, count in missing_lc_types)
    
    # load()
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n💾 : {checkpoint_file}")
        loaded_samples, loaded_config = load_negative_samples(checkpoint_file)
        if loaded_samples:
            negative_samples = loaded_samples
            print(f"   ✅ load {len(negative_samples):,},...")
            # 
            remaining_needed = num_negative_samples - len(negative_samples)
            if remaining_needed <= 0:
                print(f"   ✅,")
                return negative_samples
            print(f"   📊  {remaining_needed:,} ")
            # ()
            for lc in lc_sample_counts:
                existing_count = sum(1 for s in negative_samples if s.get('land_cover') == lc)
                lc_sample_counts[lc] = max(0, lc_sample_counts[lc] - existing_count)
    
    # 
    for lc, target_count in lc_sample_counts.items():
        if target_count == 0:
            continue
        
        if lc not in patches_by_lc:
            print(f"\n  ⚠️   {lc} (: {target_count:,} )")
            print(f"     ")
            continue
        
        available_count = len(patches_by_lc[lc])
        print(f"\n   {lc} : {target_count:,} (: {available_count:,} )...")
        sampled_count = 0
        skipped_time_window = 0
        skipped_data_missing = 0  # (0)
        skipped_high_lat = 0  # (>65°)
        max_attempts = target_count * 10
        attempts = 0
        
        # ()
        np.random.seed(42)
        candidate_patches = patches_by_lc[lc].copy()
        np.random.shuffle(candidate_patches)
        
        # 
        from tqdm import tqdm
        progress_bar = tqdm(
            total=target_count,
            desc=f"     {lc}",
            unit="",
            leave=False,
            ncols=100
        )
        
        for patch_key, patch_info in candidate_patches:
            if sampled_count >= target_count:
                break
            if attempts >= max_attempts:
                print(f"    ⚠️   ({max_attempts:,}),")
                break
            
            attempts += 1
            
            center_lat = patch_info['center_lat']
            center_lon = patch_info['center_lon']
            sampled_date = patch_info['date']
            patch_bounds = patch_info['patch_bounds']
            
            # (:time_steps)
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    skipped_time_window += 1
                    continue  # ()
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            # (for)
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
            
            # 
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features,6. code/fire_equality/datamodules/feature_alignment.py")
            # ISO3()
            iso3_code = extract_iso3_from_events(window_events, patch_bounds, events)
            # ()
            if sampled_count == 0 and iso3_code is None:
                # eventsiso3
                if events is not None and 'iso3' in events.columns:
                    print(f"    💡 :eventsiso3,iso3")
                elif events is not None:
                    print(f"    💡 :eventsiso3,")
            feature_cube = extract_aligned_features(
                spatial_bounds=patch_bounds,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                grid_size=grid_size,
                fire_date=pd.Timestamp(sampled_date).to_pydatetime(),
                fire_year=pd.Timestamp(sampled_date).year,
                iso3=iso3_code,
                data_dir='dataset',
                project=negative_pool.get('project', 'ee-your-gee-project') if isinstance(negative_pool, dict) else 'ee-your-gee-project'
            )
            
            # ()
            # (FWI, NDVI, max_temp, max_wind)
            # :
            # 1. 0 → 
            # 2. 30 → ()
            # 3. 1-20 → (NDVI0),
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 0
            fwi_all_zero = (fwi_data == 0).all()
            ndvi_all_zero = (ndvi_data == 0).all()
            temp_all_zero = (temp_data == 0).all()
            wind_all_zero = (wind_data == 0).all()
            
            zero_count = sum([fwi_all_zero, ndvi_all_zero, temp_all_zero, wind_all_zero])
            
            # 0,()
            if zero_count == 4:
                skipped_data_missing += 1
                continue
            
            # 30,()
            if zero_count >= 3:
                skipped_data_missing += 1
                continue
            
            # :(>65°N<-65°S)
            # 
            if abs(center_lat) > 65:
                skipped_high_lat += 1
                continue
            
            iso3_final = _sample_iso3(iso3_code, center_lat, center_lon, data_dir='dataset')
            sample = {
                'pixel_lat': center_lat,
                'pixel_lon': center_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 
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
            
            # (checkpoint_interval)
            if checkpoint_file and len(negative_samples) % checkpoint_interval == 0:
                try:
                    # configurationfor
                    checkpoint_config = {
                        'patch_size_km': patch_size_km,
                        'time_steps': time_steps,
                        'target_years': target_year_range
                    }
                    save_negative_samples(negative_samples, checkpoint_config, checkpoint_file)
                except Exception as e:
                    print(f"    ⚠️  : {e}")
        
        progress_bar.close()
        
        # 
        if checkpoint_file:
            try:
                checkpoint_config = {
                    'patch_size_km': patch_size_km,
                    'time_steps': time_steps,
                    'target_years': target_year_range
                }
                save_negative_samples(negative_samples, checkpoint_config, checkpoint_file)
            except Exception as e:
                print(f"    ⚠️  : {e}")
        
        # 
        if skipped_time_window > 0:
            print(f"    ⚠️   {skipped_time_window:,} ()")
        if skipped_data_missing > 0:
            print(f"    ⚠️   {skipped_data_missing:,} (:30)")
        if skipped_high_lat > 0:
            print(f"    ⚠️   {skipped_high_lat:,} (:|lat| > 65°)")
        if sampled_count < target_count:
            print(f"    ⚠️  : {sampled_count:,}, {target_count:,} ( {target_count - sampled_count:,} )")
        print(f"    ✅  {sampled_count:,} (: {target_count:,})")
    
    #,
    if missing_lc_types and len(negative_samples) < num_negative_samples:
        print(f"\n  🔄 ...")
        remaining_needed = num_negative_samples - len(negative_samples)
        print(f"     : {remaining_needed:,} ")
        
        # 
        available_lc_types = []
        for lc in sorted(patches_by_lc.keys()):
            if lc not in [s['land_cover'] for s in negative_samples]:  # 
                available_lc_types.append((lc, len(patches_by_lc[lc])))
            else:
                # 
                used_count = sum(1 for s in negative_samples if s['land_cover'] == lc)
                remaining = len(patches_by_lc[lc]) - used_count
                if remaining > 0:
                    available_lc_types.append((lc, remaining))
        
        if available_lc_types:
            #,
            available_lc_types.sort(key=lambda x: x[1], reverse=True)
            print(f"     : {', '.join(f'{lc}({count})' for lc, count in available_lc_types[:5])}")
            
            # 
            for lc, available_count in available_lc_types:
                if len(negative_samples) >= num_negative_samples:
                    break
                
                # 
                to_sample = min(remaining_needed, available_count, num_negative_samples - len(negative_samples))
                if to_sample == 0:
                    continue
                
                print(f"      {lc}  {to_sample:,} ...")
                
                # 
                np.random.seed(42)
                candidate_patches = patches_by_lc[lc].copy()
                np.random.shuffle(candidate_patches)
                
                # ()
                used_keys = {(s['pixel_lat'], s['pixel_lon'], s['pixel_date']) for s in negative_samples}
                unused_candidates = [
                    (k, p) for k, p in candidate_patches 
                    if (p['center_lat'], p['center_lon'], p['date']) not in used_keys
                ]
                
                # 
                from tqdm import tqdm
                supplement_progress = tqdm(
                    total=to_sample,
                    desc=f"       {lc}",
                    unit="",
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
                    
                    # 
                    time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
                    if target_year_range is not None:
                        start_year, _ = target_year_range
                        year_start_date = pd.Timestamp(f'{start_year}-01-01')
                        if time_window_start < year_start_date:
                            continue
                    
                    time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
                    
                    # 
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
                    
                # (8)
                if extract_aligned_features is None:
                    raise ImportError("extract_aligned_features,8. code/fire_equality/datamodules/feature_alignment.py")
                # ISO3()
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
                    project=negative_pool.get('project', 'ee-your-gee-project') if isinstance(negative_pool, dict) else 'ee-your-gee-project'
                )
                
                # ()
                # :30,()
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
                
                # 30,()
                if zero_count >= 3:
                    continue
                
                # :
                if abs(center_lat) > 65:
                    continue
                
                iso3_final = _sample_iso3(None, center_lat, center_lon, data_dir='dataset')
                sample = {
                    'pixel_lat': center_lat,
                    'pixel_lon': center_lon,
                    'pixel_date': sampled_date,
                    'features': feature_cube,
                    'target': 0,
                    'land_cover': lc,  # 
                    'metadata': {
                        'component_id': None,
                        'center_lat': center_lat,
                        'center_lon': center_lon,
                        'spatial_bounds': patch_bounds,
                        'time_window': (time_window_start, time_window_end),
                        'supplemented': True,  # 
                        'iso3': iso3_final
                    }
                }
                negative_samples.append(sample)
                sampled_supplement += 1
                supplement_progress.update(1)
            
            supplement_progress.close()
            print(f"     ✅  {lc}  {sampled_supplement:,} ")
        else:
            print(f"     ⚠️  ")
    
    print(f"\n✅  {len(negative_samples):,} (: {num_negative_samples:,})")
    if len(negative_samples) < num_negative_samples:
        print(f"   ⚠️   ({len(negative_samples):,})  ({num_negative_samples:,})")
        print(f"      : {num_negative_samples - len(negative_samples):,} ")
        print(f"      :,")
    
    return negative_samples


def sample_negative_samples_random(negative_pool, num_negative_samples,
                                  patch_size_km=25, time_steps=10, spatial_resolution_km=1,
                                  target_year_range=None):
    """
    
    
    Args:
        negative_pool: ()
        num_negative_samples: 
        patch_size_km: 
        time_steps: stride
        spatial_resolution_km: 
        target_year_range:  (start_year, end_year),for
    
    Returns:
        list: 
    """
    print("...")
    
    # 
    if 'patch_candidates' in negative_pool:
        patch_candidates = negative_pool['patch_candidates']
        events = negative_pool['events']
        target_year_range = target_year_range or negative_pool.get('target_year_range')
        
        if len(patch_candidates) == 0:
            return []
        
        # 
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
            
            # 
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            # 
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
            
            # 
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features,6. code/fire_equality/datamodules/feature_alignment.py")
            # ISO3()
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
                project=negative_pool.get('project', 'ee-your-gee-project') if isinstance(negative_pool, dict) else 'ee-your-gee-project'
            )
            
            # ()
            # :30,()
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
            
            # 30,()
            if zero_count >= 3:
                continue
            
            # :(>65°N<-65°S)
            if abs(center_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(iso3_code, center_lat, center_lon, data_dir='dataset')
            sample = {
                'pixel_lat': center_lat,
                'pixel_lon': center_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 
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
        
        # 
        spatial_bounds = {
            'lat_min': events['lat'].min(),
            'lat_max': events['lat'].max(),
            'lon_min': events['lon'].min(),
            'lon_max': events['lon'].max()
        }
        
        # 
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
            
            # 
            sampled_date = np.random.choice(grid_safe_dates)
            
            # 
            if not (spatial_bounds['lat_min'] <= grid_lat <= spatial_bounds['lat_max'] and
                   spatial_bounds['lon_min'] <= grid_lon <= spatial_bounds['lon_max']):
                continue
            
            # 
            time_window_start = pd.Timestamp(sampled_date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 
            
            time_window_end = pd.Timestamp(sampled_date) - pd.Timedelta(days=1)
            
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            pixel_spatial_bounds = {
                'lat_min': grid_lat - patch_radius_deg,
                'lat_max': grid_lat + patch_radius_deg,
                'lon_min': grid_lon - patch_radius_deg,
                'lon_max': grid_lon + patch_radius_deg
            }
            
            # 
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
            
            # 
            if extract_aligned_features is None:
                raise ImportError("extract_aligned_features,6. code/fire_equality/datamodules/feature_alignment.py")
            # ISO3()
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
                project='ee-your-gee-project'
            )
            
            # ()
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 0,()
            if (fwi_data == 0).all() and (ndvi_data == 0).all() and (temp_data == 0).all() and (wind_data == 0).all():
                continue
            
            # :
            if abs(grid_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(None, grid_lat, grid_lon, data_dir='dataset')
            sample = {
                'pixel_lat': grid_lat,
                'pixel_lon': grid_lon,
                'pixel_date': sampled_date,
                'features': feature_cube,
                'target': 0,  # 
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
        # (safe_dates)
        safe_dates = negative_pool.get('safe_dates', [])
        events = negative_pool['events']
        
        if len(safe_dates) == 0:
            return []
        
        # 
        spatial_bounds = {
            'lat_min': events['lat'].min(),
            'lat_max': events['lat'].max(),
            'lon_min': events['lon'].min(),
            'lon_max': events['lon'].max()
        }
        
        # 
        np.random.seed(42)
        sampled_dates = np.random.choice(safe_dates, size=min(num_negative_samples, len(safe_dates)), replace=False)
        
        negative_samples = []
        grid_size = int(patch_size_km / spatial_resolution_km)
        
        for date in sampled_dates:
            # ()
            pixel_lat = np.random.uniform(spatial_bounds['lat_min'], spatial_bounds['lat_max'])
            pixel_lon = np.random.uniform(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
            
            # 
            time_window_start = pd.Timestamp(date) - pd.Timedelta(days=time_steps)
            if target_year_range is not None:
                start_year, _ = target_year_range
                year_start_date = pd.Timestamp(f'{start_year}-01-01')
                if time_window_start < year_start_date:
                    continue  # 
            
            time_window_end = pd.Timestamp(date) - pd.Timedelta(days=1)
            
            lat_per_km = 1 / 110.574
            patch_radius_deg = (patch_size_km / 2) * lat_per_km
            
            pixel_spatial_bounds = {
                'lat_min': pixel_lat - patch_radius_deg,
                'lat_max': pixel_lat + patch_radius_deg,
                'lon_min': pixel_lon - patch_radius_deg,
                'lon_max': pixel_lon + patch_radius_deg
            }
            
            # 
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
                raise ImportError("extract_aligned_features,6. code/fire_equality/datamodules/feature_alignment.py")
            # ISO3()
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
                project=negative_pool.get('project', 'ee-your-gee-project') if isinstance(negative_pool, dict) else 'ee-your-gee-project'
            )
            
            # ()
            fwi_data = feature_cube[:, :, :, 0]  # FWI
            ndvi_data = feature_cube[:, :, :, 2]  # NDVI
            temp_data = feature_cube[:, :, :, 6]  # max_temp
            wind_data = feature_cube[:, :, :, 7]  # max_wind
            
            # 0,()
            if (fwi_data == 0).all() and (ndvi_data == 0).all() and (temp_data == 0).all() and (wind_data == 0).all():
                continue
            
            # :
            if abs(pixel_lat) > 65:
                continue
            
            iso3_final = _sample_iso3(iso3_code, pixel_lat, pixel_lon, data_dir='dataset')
            sample = {
                'pixel_lat': pixel_lat,
                'pixel_lon': pixel_lon,
                'pixel_date': date,
                'features': feature_cube,
                'target': 0,  # 
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
    :
    
    :
    - (y_t = 1):()
    - (y_t = 0):,
    
    Args:
        data_directory: FireTracks
        output_path:  ()
        config: configuration,:
            - patch_size_km:  (km),25
            - time_steps: stride (),10
            - target_years:,(2002, 2020)
            - batch_size:,32
            - neg_pos_ratio:,2.0
            - max_samples: (for,None)
    
    Returns:
        dict: :
            - 'dataset': FireTracksDataset
            - 'dataloader': DataLoader
            - 'config': configuration
            - 'preprocessed_data': 
            - 'positive_samples': 
            - 'negative_samples': 
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
            'neg_pos_ratio': 2.0,  # 2
            'max_samples': None
        }
    
    print("="*60)
    print("🚀 ...")
    print("="*60)
    
    # 1. load
    print("\n[1/6] loadFireTracks...")
    start_year, end_year = config['target_years']
    # loading,
    events_year_range = config['target_years']
    print(f"   Events : {events_year_range}")
    
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
    
    # loadcomponents,
    if 'components' in datasets:
        print(f"   load components,: {config['target_years']}")
        import os
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year+1}-01-01"
        try:
            datasets['components'] = pd.read_hdf(
                f'{data_directory}/cp.h5',
                where=f'dtime_min >= "{start_date}" & dtime_min < "{end_date}"'
            )
            print(f"   ✅ Components: {len(datasets['components']):,} ")
        except Exception as e:
            print(f"   ⚠️  load components,load")
    
    print(f"   ✅ FireTracksload!")
    print(f"   - : {len(datasets.get('events', [])):,} ")
    print(f"   - : {len(datasets.get('components', [])):,} ")
    
    # 2. 
    print("\n[2/6] ...")
    import time
    start_time = time.time()
    preprocessed_data = preprocess_firetracks_data(
        datasets,
        target_year_range=config['target_years']
    )
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  : {elapsed_time:.1f} ")
    
    # 3. load
    print("\n[3/7] load...")
    
    # GDP(0)
    try:
        from .feature_alignment import reset_gdp_warning_count
        reset_gdp_warning_count()
    except (ImportError, AttributeError):
        try:
            from code.fire_equality.datamodules.feature_alignment import reset_gdp_warning_count
            reset_gdp_warning_count()
        except (ImportError, AttributeError):
            pass  #,
    
    # ()
    try:
        from .feature_cache import clear_cache, get_cache_stats
        from .netcdf_cache import clear_netcdf_cache, get_cache_stats as get_netcdf_cache_stats
        # NetCDF,
        netcdf_stats_before = get_netcdf_cache_stats()
        if netcdf_stats_before['cached_datasets'] > 0:
            print(f"   🧹 NetCDF ({netcdf_stats_before['cached_datasets']} )...")
            clear_netcdf_cache()
            print(f"   ✅ NetCDF")
        #,
    except (ImportError, AttributeError):
        pass
    
    # (dataset)
    cache_dir = config.get('cache_dir', 'dataset')
    os.makedirs(cache_dir, exist_ok=True)
    
    # (configuration)
    cache_filename = f"positive_samples_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}_t{config['time_steps']}"
    if config.get('max_samples'):
        cache_filename += f"_max{config['max_samples']}"
    cache_filename += ".pth"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    # load
    use_modis_api = config.get('use_modis_api', True)
    project = config.get('gee_project', 'ee-your-gee-project')
    
    positive_samples, cached_config = load_positive_samples(cache_filepath, verify_config=config)
    
    if positive_samples is None:
        # configuration,
        print("   configuration,...")
        start_time = time.time()
        positive_samples = create_pixel_level_positive_samples(
            preprocessed_data,
            patch_size_km=config['patch_size_km'],
            time_steps=config['time_steps'],
            max_samples=config.get('max_samples', None),
            target_year_range=config['target_years'],  # 
            use_modis_api=use_modis_api,
            project=project,
            parallel_workers=config.get('parallel_workers', 2)  #,2
        )
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  : {elapsed_time:.1f} ")
        print(f"   ✅ : {len(positive_samples):,}")
        
        # 
        try:
            from .feature_cache import get_cache_stats
            from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
            cache_stats = get_cache_stats()
            netcdf_stats = get_netcdf_cache_stats()
            if cache_stats['hits'] + cache_stats['misses'] > 0:
                print(f"   📊 :  {cache_stats['hit_rate']:.1%} ({cache_stats['hits']}  / {cache_stats['misses']} ), : {cache_stats['cache_size']} ")
            if netcdf_stats['cached_datasets'] > 0:
                print(f"   📊 NetCDF: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']} ")
        except (ImportError, AttributeError):
            pass
        
        # 
        if len(positive_samples) > 0:
            print(f"\n💾 : {cache_filepath}")
            save_positive_samples(positive_samples, config, cache_filepath)
    else:
        print(f"   ✅ loading,")
        # :max_samples,load_positive_samples
        print(f"   ✅ : {len(positive_samples):,}")
        if config.get('max_samples') and len(positive_samples) > config['max_samples']:
            print(f"   ⚠️  :  ({len(positive_samples):,}) max_samples ({config['max_samples']})")
            print(f"  ,load_positive_samples")
    
    if len(positive_samples) == 0:
        print("❌ ")
        return None
    
    # 4. (for)
    print("\n[4/7] ...")
    positive_lc_distribution = analyze_positive_sample_land_cover_distribution(positive_samples)
    
    # (for MODIS)
    positive_lc_types = None
    if positive_lc_distribution and positive_lc_distribution.get('counts'):
        positive_lc_types = list(positive_lc_distribution['counts'].keys())
        print(f"   🎯  {sorted(positive_lc_types)}  MODIS ")
    
    # 5. load()
    print("\n[5/7] load...")
    
    # 
    negative_pool_cache_filename = f"negative_pool_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}"
    fast_mode = config.get('fast_mode', False) or (config.get('max_samples') is not None)
    if fast_mode:
        negative_pool_cache_filename += "_fast"
    negative_pool_cache_filename += ".pth"
    negative_pool_cache_filepath = os.path.join(cache_dir, negative_pool_cache_filename)
    
    # forconfiguration(fast_mode)
    pool_config = {
        'patch_size_km': config['patch_size_km'],
        'target_years': config['target_years'],
        'fast_mode': fast_mode
    }
    
    # load
    negative_pool, cached_pool_config = load_negative_pool(negative_pool_cache_filepath, verify_config=pool_config)
    
    if negative_pool is None:
        # configuration,
        print("   configuration,...")
        start_time = time.time()
        # :max_samples()fast_mode,
        if fast_mode:
            if config.get('max_samples') is not None:
                print("   ⚡ max_samples,(for)")
            else:
                print("   ⚡ ")
        negative_pool = create_negative_sample_pool(
            preprocessed_data,
            config['target_years'],
            patch_size_km=config['patch_size_km'],
            events_lc=preprocessed_data.get('events_lc', None),
            spatial_resolution_km=1,
            fast_mode=fast_mode,
            positive_lc_types=positive_lc_types,  # 
            use_modis_api=use_modis_api,  #  MODIS API
            project=project  # GEE 
        )
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  : {elapsed_time:.1f} ")
        
        # 
        if len(negative_pool.get('patch_candidates', {})) > 0:
            print(f"\n💾 : {negative_pool_cache_filepath}")
            save_negative_pool(negative_pool, pool_config, negative_pool_cache_filepath)
    else:
        print(f"   ✅ loading,")
        print(f"   ✅ : {len(negative_pool.get('patch_candidates', {})):,}")
        # events(,preprocessed_data)
        if 'events' not in negative_pool or negative_pool['events'] is None:
            negative_pool['events'] = preprocessed_data.get('events')
            print(f"   💡 preprocessed_dataevents")
    
    # 6. ()
    print("\n[6/7] ...")
    start_time = time.time()
    num_negative_samples = int(len(positive_samples) * config['neg_pos_ratio'])
    print(f"   : {num_negative_samples:,} ( {config['neg_pos_ratio']:.1f} )")
    
    # (for)
    checkpoint_filename = f"negative_samples_checkpoint_{config['target_years'][0]}_{config['target_years'][1]}_p{config['patch_size_km']}.pth"
    checkpoint_filepath = os.path.join(cache_dir, checkpoint_filename)
    
    negative_samples = sample_negative_samples_by_land_cover(
        negative_pool,
        positive_lc_distribution,  # 
        num_negative_samples,
        patch_size_km=config['patch_size_km'],
        time_steps=config['time_steps'],
        spatial_resolution_km=1,
        target_year_range=config['target_years'],
        use_modis_api=use_modis_api,  # MODIS API
        project=project,  # GEE
        checkpoint_file=checkpoint_filepath,  # 
        checkpoint_interval=100  # 100
    )
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️  : {elapsed_time:.1f} ")
    print(f"   ✅ : {len(negative_samples):,}")
    
    # 
    try:
        from .feature_cache import get_cache_stats
        from .netcdf_cache import get_cache_stats as get_netcdf_cache_stats
        cache_stats = get_cache_stats()
        netcdf_stats = get_netcdf_cache_stats()
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            print(f"   📊 :  {cache_stats['hit_rate']:.1%} ({cache_stats['hits']}  / {cache_stats['misses']} ), : {cache_stats['cache_size']} ")
        if netcdf_stats['cached_datasets'] > 0:
            print(f"   📊 NetCDF: {netcdf_stats['cached_datasets']}/{netcdf_stats['max_cache_size']} ")
    except (ImportError, AttributeError):
        pass
    
    # 7. 
    print("\n[7/7] PyTorch...")
    all_samples = positive_samples + negative_samples
    
    # configuration
    config['target_type'] = 'binary_classification'  # 
    
    # (FireTracksDataset)
    dataset = FireTracksDataset(
        all_samples,
        target_type='binary_classification'
    )
    
    # load
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print("\n✅ !")
    print(f"   - : {len(dataset):,}")
    print(f"   - : {len(positive_samples):,}")
    print(f"   - : {len(negative_samples):,}")
    print(f"   - : 1:{len(negative_samples)/len(positive_samples):.2f}")
    if len(dataset) > 0:
        sample_features, sample_target = dataset[0]
        print(f"   - : {sample_features.shape}")
        print(f"   - :  (0/1)")
        print(f"   - : {len(dataloader):,} ")
    
    # 
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
            print(f"✅ : {output_path}")
        except Exception as e:
            print(f"⚠️  : {e}")
    
    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'config': config,
        'preprocessed_data': preprocessed_data,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'positive_lc_distribution': positive_lc_distribution
    }


# 

if __name__ == "__main__":
    # #region agent log
    import json
    import time as time_module
    try:
        with open(r'e:\FireEqual\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"firetracks_loader.py:4038","message":"Script entry point","data":{},"timestamp":int(time_module.time()*1000)}) + '\n')
    except Exception as log_err:
        pass  # 
    # #endregion
    import argparse
    
    parser = argparse.ArgumentParser(description='FireTracks()')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='(for,None)')
    parser.add_argument('--target_years', type=int, nargs=2, default=[2017, 2018],
                        metavar=('START', 'END'),
                        help=',: --target_years 2017 2018')
    parser.add_argument('--patch_size_km', type=int, default=25,
                        help=' (km),25')
    parser.add_argument('--time_steps', type=int, default=10,
                        help='stride (),10')
    parser.add_argument('--neg_pos_ratio', type=float, default=2.0,
                        help=',2.0')
    parser.add_argument('--data_directory', type=str, default='dataset/firetracks_data',
                        help='FireTracks,dataset/firetracks_data')
    parser.add_argument('--output_path', type=str, default='dataset/processed_firetracks_pixel_binary.pth',
                        help=',dataset/processed_firetracks_pixel_binary.pth')
    parser.add_argument('--cache_dir', type=str, default='dataset',
                        help=',dataset')
    parser.add_argument('--parallel_workers', type=int, default=2,
                        help=',2')
    
    args = parser.parse_args()
    
    # configuration
    config = {
        'use_modis_api': True,
        'gee_project': 'ee-your-gee-project',
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
    print("FireTracks")
    print("="*60)
    print(f"configuration:")
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
    
    # 
    result = create_pixel_level_binary_classification_dataset(
        data_directory=args.data_directory,
        output_path=args.output_path,
        config=config
    )
    
    if result is not None:
        # load
        dataloader = result['dataloader']
        features, targets = next(iter(dataloader))
        
        print(f"\n📊 load:")
        print(f"   - Batch: {features.shape[0]}")
        print(f"   - : {features.shape}")
        print(f"   - : {targets.shape}")
        print(f"   - :  (0=, 1=)")
        print(f"   - : {features.dtype}")
        
        # 
        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()
        print(f"   - : {positive_count}")
        print(f"   - : {negative_count}")
        
        print(f"\n✅ !")
    else:
        print(f"\n❌ ")
