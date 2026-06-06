"""
NetCDF
forNetCDF,/,
"""
import os
import threading
from typing import Optional, Dict, Tuple
import xarray as xr
import warnings
import sys

# HDF5()
# xarrayCachingFileManager__del__

# 1:HDF5(C)
# netCDF4
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
# :HDF5stderr,Pythonsys.stderr
# 

_original_stderr = sys.stderr

class HDF5ErrorFilter:
    """HDF5"""
    def __init__(self):
        self.enabled = True
        self._buffer = ""
        self._in_hdf5_block = False  # HDF5
    
    def write(self, text):
        if not self.enabled:
            _original_stderr.write(text)
            return
        
        # 
        self._buffer += text
        
        # HDF5
        if not self._in_hdf5_block:
            if self._is_hdf5_error_start(self._buffer):
                self._in_hdf5_block = True
        
        #,
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            # ()
            self._buffer = lines[-1]
            
            # 
            output_lines = []
            for line in lines[:-1]:
                if self._in_hdf5_block:
                    # HDF5,
                    if self._is_hdf5_error_end(line):
                        # 
                        self._in_hdf5_block = False
                        # (,HDF5)
                    #,
                else:
                    #,
                    if self._is_hdf5_error_start(line):
                        self._in_hdf5_block = True
                    else:
                        #,
                        output_lines.append(line)
            
            # HDF5
            if output_lines:
                _original_stderr.write('\n'.join(output_lines) + '\n')
        
        #,HDF5,
        elif not self._in_hdf5_block and not self._is_hdf5_error_start(self._buffer):
            #,
            #,
            pass
    
    def _is_hdf5_error_start(self, text):
        """HDF5"""
        start_patterns = [
            'HDF5-DIAG:',
            'Exception ignored in:',
            'Traceback (most recent call last):',
            'CachingFileManager.__del__'
        ]
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in start_patterns)
    
    def _is_hdf5_error_end(self, line):
        """HDF5"""
        # HDF5,HDF5
        line_stripped = line.strip()
        if not line_stripped:
            return True  # 
        
        # HDF5
        hdf5_keywords = [
            'hdf5', 'h5d', 'h5a', 'h5vl', 'netcdf', 'file_manager',
            'invalid', 'identifier', 'dataspace', 'major:', 'minor:',
            'runtimeerror', 'traceback', 'cachingfilemanager'
        ]
        line_lower = line_stripped.lower()
        has_hdf5_keyword = any(keyword in line_lower for keyword in hdf5_keywords)
        
        # HDF5:
        # 1. #(#000, #001)
        # 2. "major:""minor:"
        # 3. ("File \"...file_manager.py\" line 250")
        is_hdf5_format = (
            line_stripped.startswith('#') or
            'major:' in line_lower or
            'minor:' in line_lower or
            ('file "' in line_lower and ('line ' in line_lower or '.py' in line_lower))
        )
        
        # HDF5,HDF5,
        if not has_hdf5_keyword and not is_hdf5_format:
            return True
        
        return False
    
    def flush(self):
        # 
        if self._buffer and not self._in_hdf5_block and not self._is_hdf5_error_start(self._buffer):
            _original_stderr.write(self._buffer)
            self._buffer = ""
        _original_stderr.flush()
        self._in_hdf5_block = False  # 

# 
_hdf5_error_filter = HDF5ErrorFilter()

def enable_hdf5_error_filter():
    """HDF5"""
    _hdf5_error_filter.enabled = True
    sys.stderr = _hdf5_error_filter

def disable_hdf5_error_filter():
    """HDF5"""
    _hdf5_error_filter.enabled = False
    sys.stderr = _original_stderr

# 
enable_hdf5_error_filter()

# ()
_datasets: Dict[str, xr.Dataset] = {}
_cache_lock = threading.Lock()
_max_cache_size = 10  # ()


def _safe_close_dataset(ds: xr.Dataset):
    """
    NetCDF,
    
    Args:
        ds: xarray Dataset
    """
    if ds is None:
        return
    
    try:
        # 
        # xarray Dataset  _file_obj 
        if hasattr(ds, '_file_obj') and ds._file_obj is not None:
            # 
            if hasattr(ds._file_obj, 'isopen'):
                if not ds._file_obj.isopen():
                    return  #,
        
        # 
        ds.close()
    except (RuntimeError, AttributeError, OSError, ValueError) as e:
        # :
        # - RuntimeError: NetCDF: Not a valid ID ()
        # - AttributeError: 
        # - OSError: 
        # - ValueError: 
        pass
    except Exception:
        #,
        pass


def _validate_dataset(ds: xr.Dataset) -> bool:
    """
    
    
    Args:
        ds: xarray Dataset
    
    Returns:
        bool: True,False
    """
    if ds is None:
        return False
    try:
        # 
        dims = ds.dims  # load
        data_vars = ds.data_vars if hasattr(ds, 'data_vars') else {}
        coords = ds.coords if hasattr(ds, 'coords') else {}
        
        # ()
        if len(dims) == 0 and len(data_vars) == 0 and len(coords) == 0:
            return False
        
        # ()
        if len(coords) > 0:
            _ = list(coords.keys())[0]
        
        # ()
        if len(data_vars) > 0:
            _ = list(data_vars.keys())[0]
        
        return True
    except (AttributeError, RuntimeError, OSError, ValueError):
        # 
        return False
    except Exception:
        # 
        return False


def get_netcdf_dataset(filepath: str, engine: Optional[str] = None) -> xr.Dataset:
    """
    NetCDF
    
    :,.
    :,.
    
    Args:
        filepath: NetCDF
        engine: ('netcdf4', 'h5netcdf', None)
    
    Returns:
        xr.Dataset: (:,)
    """
    # 
    abs_path = os.path.abspath(filepath)
    
    #,
    with _cache_lock:
        # 
        if abs_path in _datasets:
            ds = _datasets[abs_path]
            # 
            # :,
            # 
            try:
                if _validate_dataset(ds):
                    # data_vars()
                    try:
                        data_vars_count = len(ds.data_vars)
                        if data_vars_count > 0:
                            return ds
                        else:
                            # data_vars,
                            print(f"   ⚠️  data_vars,: {os.path.basename(filepath)}")
                            _safe_close_dataset(ds)
                            del _datasets[abs_path]
                    except Exception:
                        # data_vars,
                        print(f"   ⚠️ ,: {os.path.basename(filepath)}")
                        _safe_close_dataset(ds)
                        del _datasets[abs_path]
                else:
                    #,
                    print(f"   ⚠️ ,: {os.path.basename(filepath)}")
                    _safe_close_dataset(ds)
                    del _datasets[abs_path]
            except Exception as e:
                #,
                print(f"   ⚠️  : {e},: {os.path.basename(filepath)}")
                try:
                    _safe_close_dataset(ds)
                except:
                    pass
                if abs_path in _datasets:
                    del _datasets[abs_path]
        
        #,(FIFO)
        if len(_datasets) >= _max_cache_size:
            # 
            oldest_key = next(iter(_datasets))
            _safe_close_dataset(_datasets[oldest_key])
            del _datasets[oldest_key]
        
        # 
        engines_to_try = [engine] if engine else ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        
        for eng in engines_to_try:
            try:
                if eng:
                    ds = xr.open_dataset(filepath, engine=eng)
                else:
                    ds = xr.open_dataset(filepath)
                # 
                if _validate_dataset(ds) and len(ds.data_vars) > 0:
                    break
                else:
                    #,
                    _safe_close_dataset(ds)
                    ds = None
                    continue
            except Exception as e:
                last_error = e
                if ds is not None:
                    _safe_close_dataset(ds)
                    ds = None
                continue
        
        if ds is None:
            error_msg = f"NetCDF {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
        
        # 
        _datasets[abs_path] = ds
        return ds


def clear_netcdf_cache():
    """NetCDF"""
    with _cache_lock:
        for ds in _datasets.values():
            _safe_close_dataset(ds)
        _datasets.clear()


def get_cache_stats() -> Dict[str, int]:
    """"""
    with _cache_lock:
        return {
            'cached_datasets': len(_datasets),
            'max_cache_size': _max_cache_size
        }


