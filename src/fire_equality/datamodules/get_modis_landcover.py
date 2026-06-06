"""
MODIS MCD12Q1 ( Google Earth Engine)

:
    from code.fire_equality.datamodules.get_modis_landcover import get_landcover_from_gee, LandCoverCache

    # 
    lc = get_landcover_from_gee(lat=34.05, lon=-118.25, year=2017, project='your-project')

    # (for)
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

# Google APIPython(,Python)
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    warnings.warn("Google Earth Engine (earthengine-api) .: pip install earthengine-api")


# MCD12Q1 LC_Type1 
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
    initializeGoogle Earth Engine
    
    Args:
        project:,GEE( 'ee-your-gee-project')
        credentials_path:,
        silent: True,False
    
    Returns:
        bool: silent=True,initialize;None,
    """
    if not HAS_EE:
        if silent:
            return False
        raise ImportError(" earthengine-api: pip install earthengine-api")
    
    try:
        # initialize
        ee.Number(1).getInfo()
        if not silent:
            print("✅ Google Earth Engine initialize")
        return True if silent else None
    except:
        pass
    
    # ()
    import platform
    import os
    from pathlib import Path
    
    # GEE
    if platform.system() == 'Windows':
        creds_dir = Path.home() / '.config' / 'earthengine'
    else:
        creds_dir = Path.home() / '.config' / 'earthengine'
    
    creds_file = creds_dir / 'credentials'
    has_credentials = creds_file.exists()
    
    try:
        if project:
            # initialize
            ee.Initialize(project=project)
            if not silent:
                print(f"✅ Google Earth Engine initialize(: {project})")
            return True if silent else None
        elif credentials_path:
            # 
            credentials = ee.ServiceAccountCredentials(None, credentials_path)
            ee.Initialize(credentials)
            if not silent:
                print("✅ Google Earth Engine initialize()")
            return True if silent else None
        else:
            # 
            ee.Initialize()
            if not silent:
                print("✅ Google Earth Engine initialize()")
            return True if silent else None
    except Exception as e:
        # initialize,
        error_msg = str(e).lower()
        needs_auth = 'credentials' in error_msg or 'authentication' in error_msg or 'not authenticated' in error_msg
        
        if not needs_auth and has_credentials:
            # initialize,
            if not silent:
                print(f"⚠️  Google Earth Engine initialize: {str(e)}")
                print("  ,")
                print("   : earthengine authenticate")
            if silent:
                return False
            raise
        
        # 
        if not silent:
            if has_credentials:
                print("⚠️  Google Earth Engine,...")
            else:
                print("⚠️  Google Earth Engine,...")
        
        try:
            # Windows,notebook()
            import subprocess
            
            # gcloud
            has_gcloud = False
            try:
                subprocess.run(['gcloud', '--version'], 
                             capture_output=True, check=True, timeout=5)
                has_gcloud = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                has_gcloud = False
            
            if has_gcloud:
                # gcloud,
                ee.Authenticate()
            else:
                # Windowsnotebook(URL,)
                if platform.system() == 'Windows':
                    if not silent:
                        print("   💡 notebook(URL,)")
                    try:
                        ee.Authenticate(auth_mode='notebook')
                    except Exception as notebook_error:
                        # notebook,
                        if not silent:
                            print(f"   ⚠️  notebook: {notebook_error}")
                            print("   ...")
                        ee.Authenticate()
                else:
                    # Linux/Macgcloud
                    ee.Authenticate()
            
            # initialize
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            if not silent:
                print("✅ Google Earth Engine initialize")
                print("   💡 :,")
            return True if silent else None
        except Exception as e2:
            error_msg = str(e2)
            if not silent:
                print(f"❌ Google Earth Engine : {error_msg}")
                print("\n   :")
                print("   1. : earthengine authenticate")
                print("   2. : earthengine authenticate --auth_mode=notebook")
                print("   3. ")
                print("\n  ,.")
            
            if silent:
                return False
            raise


def get_landcover_from_gee(lat: float, lon: float, year: int, 
                           lc_type: str = 'LC_Type1',
                           project: Optional[str] = None) -> Optional[int]:
    """
    Google Earth EngineMODIS MCD12Q1
    
    Args:
        lat: 
        lon: 
        year: (MCD12Q1)
        lc_type:, 'LC_Type1', 'LC_Type2', 'LC_Type3', 'LC_Type4', 'LC_Type5'
        project:,GEE
    
    Returns:
        (),None
    """
    if not HAS_EE:
        raise ImportError(" earthengine-api: pip install earthengine-api")
    
    try:
        # initialize(initialize)
        try:
            ee.Number(1).getInfo()  # initialize
        except:
            # silent,
            if not initialize_gee(project=project, silent=True):
                return None
        
        # loadMCD12Q1( MODIS/061/MCD12Q1)
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        
        # 
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        
        # 
        lc_band = image.select(lc_type)
        
        # 
        point = ee.Geometry.Point([lon, lat])
        
        # 
        value = lc_band.sample(point, scale=500).first().get(lc_type)
        
        # 
        result = value.getInfo()
        
        if result is not None:
            return int(result)
        else:
            return None
            
    except Exception as e:
        print(f"⚠️   ({lat}, {lon}, {year}): {e}")
        return None


def get_landcover_batch_gee(coords: List[Tuple[float, float]], 
                             year: int,
                             lc_type: str = 'LC_Type1',
                             batch_size: int = 1000,
                             project: Optional[str] = None) -> List[Optional[int]]:
    """
    Google Earth Engine()
    
    Args:
        coords:,(lat, lon)
        year: 
        lc_type: 
        batch_size: 
        project:,GEE
    
    Returns:
        
    """
    if not HAS_EE:
        raise ImportError(" earthengine-api: pip install earthengine-api")
    
    try:
        # initialize(initialize)
        try:
            ee.Number(1).getInfo()  # initialize
        except:
            # silent,
            if not initialize_gee(project=project, silent=True):
                return [None] * len(coords)
        
        # loadMCD12Q1( MODIS/061/MCD12Q1)
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        lc_band = image.select(lc_type)
        
        results = []
        
        # 
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_start_idx = i  # 
            
            # 
            valid_coords = []
            valid_orig_indices = []  # coords
            
            for local_idx, (lat, lon) in enumerate(batch_coords):
                orig_idx = batch_start_idx + local_idx
                # : -90  90, -180  180
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    valid_coords.append((lat, lon))
                    valid_orig_indices.append(orig_idx)
            
            #  None()
            # 
            while len(results) < batch_start_idx + len(batch_coords):
                results.append(None)
            
            if len(valid_coords) == 0:
                #, None()
                continue
            
            try:
                batch_num = i // batch_size + 1
                total_batches = (len(coords) + batch_size - 1) // batch_size
                print(f"      [ {batch_num}/{total_batches}]  {len(valid_coords)} ...", end='\r', flush=True)
                
                # ()
                #  valid_orig_indices  id,
                points = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([lon, lat]), {'id': orig_idx})
                    for orig_idx, (lat, lon) in zip(valid_orig_indices, valid_coords)
                ])
                
                # ( scale=500  MODIS 500m )
                # :sampleRegions  crs,
                samples = lc_band.sampleRegions(
                    collection=points,
                    scale=500,
                    geometries=False
                )
                
                # (,API)
                print(f"      [ {batch_num}/{total_batches}] GEE API()...", end='\r', flush=True)
                sample_list = samples.getInfo()['features']
                print(f"      [ {batch_num}/{total_batches}] ✅  {len(sample_list)} ", flush=True)
                
                # ()
                batch_results = {}
                for feature in sample_list:
                    orig_idx = feature['properties']['id']
                    lc_value = feature['properties'].get(lc_type)
                    batch_results[orig_idx] = int(lc_value) if lc_value is not None else None
                
                # ()
                for orig_idx in valid_orig_indices:
                    results[orig_idx] = batch_results.get(orig_idx, None)
                
            except Exception as batch_error:
                #, None
                print(f"⚠️   {i//batch_size + 1} : {batch_error}")
                for orig_idx in valid_orig_indices:
                    results[orig_idx] = None
        
        # 
        while len(results) < len(coords):
            results.append(None)
        
        return results
        
    except Exception as e:
        print(f"⚠️  : {e}")
        return [None] * len(coords)


def find_locations_by_landcover(positive_lc_types: List[int],
                                spatial_bounds: dict,
                                year: int,
                                num_samples_per_type: int = 1000,
                                project: Optional[str] = None) -> List[dict]:
    """
     MODIS 
    
     GEE, MODIS,
   ,.
    
    Args:
        positive_lc_types:, [7, 9, 10]
        spatial_bounds:, {
            'lat_min': float, 'lat_max': float,
            'lon_min': float, 'lon_max': float
        }
        year: 
        num_samples_per_type:,1000
        project:,GEE
    
    Returns:
       , {
            'lat': float,
            'lon': float,
            'land_cover': int
        }
    """
    if not HAS_EE:
        raise ImportError(" earthengine-api: pip install earthengine-api")
    
    try:
        # initialize(initialize)
        try:
            ee.Number(1).getInfo()  # initialize
        except:
            # silent,
            if not initialize_gee(project=project, silent=True):
                return []
        
        # loadMCD12Q1
        dataset = ee.ImageCollection('MODIS/061/MCD12Q1')
        image = dataset.filter(ee.Filter.eq('system:time_start', 
                                           ee.Date.fromYMD(year, 1, 1).millis())).first()
        lc_band = image.select('LC_Type1')
        
        # 
        region = ee.Geometry.Rectangle([
            spatial_bounds['lon_min'],
            spatial_bounds['lat_min'],
            spatial_bounds['lon_max'],
            spatial_bounds['lat_max']
        ])
        
        all_locations = []
        
        # 
        for lc_type in positive_lc_types:
            # :
            mask = lc_band.eq(lc_type)
            
            # 
            masked_lc = lc_band.updateMask(mask)
            
            #,
            current_num_samples = num_samples_per_type
            max_attempts = 3
            found_count = 0
            
            for attempt in range(max_attempts):
                try:
                    # ( True )
                    #  sample,
                    samples = masked_lc.sample(
                        region=region,
                        scale=500,  # MODIS  500m
                        numPixels=current_num_samples,
                        geometries=True  # 
                    )
                    
                    # 
                    sample_list = samples.getInfo()['features']
                    
                    type_locations = []
                    for feature in sample_list:
                        coords = feature['geometry']['coordinates']
                        lc_value = feature['properties'].get('LC_Type1')
                        
                        # 
                        if lc_value == lc_type:
                            type_locations.append({
                                'lat': coords[1],  # GEE  [lon, lat]
                                'lon': coords[0],
                                'land_cover': int(lc_value)
                            })
                    
                    found_count = len(type_locations)
                    
                    #,
                    if found_count > 0:
                        all_locations.extend(type_locations)
                        print(f"     {lc_type}:  {found_count} (: {current_num_samples})")
                        break
                    else:
                        #,
                        if attempt < max_attempts - 1:
                            current_num_samples = current_num_samples * 3  # 3
                            print(f"     {lc_type}:, {current_num_samples} ...")
                        else:
                            print(f"    ⚠️   {lc_type}:  {max_attempts} ()")
                
                except Exception as e:
                    if attempt < max_attempts - 1:
                        current_num_samples = current_num_samples * 3
                        print(f"    ⚠️   {lc_type} ( {attempt + 1}/{max_attempts}): {e}")
                        print(f"        {current_num_samples} ...")
                    else:
                        print(f"    ⚠️   {lc_type} ( {max_attempts} ): {e}")
                        break
        
        print(f"  ✅  {len(all_locations):,} ( {len(set(l['land_cover'] for l in all_locations))} )")
        
        return all_locations
        
    except Exception as e:
        print(f"⚠️  : {e}")
        import traceback
        traceback.print_exc()
        return []


class LandCoverCache:
    """
   ,for
    """
    
    def __init__(self, method: str = 'gee', 
                 year: Optional[int] = None,
                 project: Optional[str] = None):
        """
        initialize
        
        Args:
            method:,'gee'  'hdf'
            year: (for)
            project:,GEE( 'ee-your-gee-project')
        """
        self.method = method
        self.year = year
        self.project = project
        self.cache = {}  # 
        
        if method == 'gee':
            if not HAS_EE:
                raise ImportError(" earthengine-api: pip install earthengine-api")
            # silent,
            if not initialize_gee(project=project, silent=True):
                print(f"⚠️  GEEinitialize,LandCoverCacheGEE")
                print("   : earthengine authenticate")
        else:
            raise ValueError(" method='gee'")
    
    def get_landcover(self, lat: float, lon: float, 
                     year: Optional[int] = None,
                     lc_type: str = 'LC_Type1') -> Optional[int]:
        """
        
        
        Args:
            lat: 
            lon: 
            year: (,initializeyear)
            lc_type: 
        
        Returns:
            
        """
        if year is None:
            year = self.year
        if year is None:
            raise ValueError("year")
        
        # (0.001,100m)
        cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # ( GEE)
        lc = get_landcover_from_gee(lat, lon, year, lc_type, project=self.project)
        
        # 
        self.cache[cache_key] = lc
        
        return lc
    
    def get_landcover_batch(self, coords: List[Tuple[float, float]],
                           year: Optional[int] = None,
                           lc_type: str = 'LC_Type1',
                           use_cache: bool = True) -> List[Optional[int]]:
        """
        (for)
        
        Args:
            coords:,(lat, lon)
            year: 
            lc_type: 
            use_cache: 
        
        Returns:
            
        """
        if year is None:
            year = self.year
        if year is None:
            raise ValueError("year")
        
        # initialize,coords
        results = [None] * len(coords)
        
        # API( GEE)
        if use_cache:
            # 
            uncached_coords = []
            uncached_indices = []
            for idx, (lat, lon) in enumerate(coords):
                cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
                if cache_key in self.cache:
                    results[idx] = self.cache[cache_key]
                else:
                    uncached_coords.append((lat, lon))
                    uncached_indices.append(idx)
            
            # 
            if uncached_coords:
                batch_results = get_landcover_batch_gee(uncached_coords, year, lc_type, 
                                                      project=self.project)
                # 
                for orig_idx, lc in zip(uncached_indices, batch_results):
                    lat, lon = coords[orig_idx]
                    cache_key = (round(lat, 3), round(lon, 3), year, lc_type)
                    self.cache[cache_key] = lc
                    results[orig_idx] = lc
        else:
            results = get_landcover_batch_gee(coords, year, lc_type, project=self.project)
            # 
            if not results or len(results) != len(coords):
                results = [None] * len(coords)
        
        return results

    def clear_cache(self):
        """"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """"""
        return len(self.cache)


# 
def get_landcover(lat: float, lon: float, year: int, 
                 method: str = 'gee',
                 hdf_file: Optional[str] = None) -> Optional[int]:
    """
    :
    
    Args:
        lat: 
        lon: 
        year: 
        method:, 'gee'
        hdf_file: ()
    
    Returns:
        
    """
    if method != 'gee':
        raise ValueError(" method='gee'")
    return get_landcover_from_gee(lat, lon, year)


if __name__ == '__main__':
    # Python()
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')
    
    # 
    print("MODIS MCD12Q1 ")
    print("=" * 50)
    
    # ()
    test_coords = [
        (34.05, -118.25, ""),
        (40.71, -74.01, ""),
        (51.51, -0.13, ""),
        (39.90, 116.41, ""),
        (-33.87, 151.21, ""),
    ]
    test_year = 2017
    project = 'ee-your-gee-project'
    
    print(f"\n: {test_year}")
    print(f"GEE: {project}")
    print(f": {len(test_coords)}")
    
    # 
    results_data = []
    
    # Google Earth Engine()
    if HAS_EE:
        print("\n1. Google Earth Engine...")
        for lat, lon, name in test_coords[:3]:  # 3
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
                    print(f"   ⚠️  {name}: ")
            except Exception as e:
                print(f"   ❌ {name} : {e}")
    else:
        print("\n1. Google Earth Engine,")
    
    # 
    if HAS_EE:
        print("\n2. ...")
        try:
            cache = LandCoverCache(method='gee', year=test_year, project=project)
            batch_coords = [(lat, lon) for lat, lon, _ in test_coords]
            batch_results = cache.get_landcover_batch(batch_coords, test_year)
            
            print(f"   ✅ :")
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
                    print(f"      {name}: ")
            
            print(f"   ✅ : {cache.get_cache_size()}")
        except Exception as e:
            print(f"   ❌ : {e}")
            import traceback
            traceback.print_exc()
    
    # CSV
    if results_data:
        print("\n3. ...")
        try:
            import pandas as pd
            df = pd.DataFrame(results_data)
            # ()
            df = df.drop_duplicates(subset=['lat', 'lon', 'year'], keep='last')
            
            output_file = 'dataset/modis_landcover_test_results.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ : {output_file}")
            print(f"   ✅  {len(df)} ")
            print("\n   :")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"   ⚠️  : {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("!")

