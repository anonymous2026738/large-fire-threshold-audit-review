"""
NetCDF数据集缓存管理器
用于在内存中保持NetCDF文件打开，避免重复打开/关闭文件，提升性能
"""
import os
import threading
from typing import Optional, Dict, Tuple
import xarray as xr
import warnings
import sys

# 抑制HDF5垃圾回收时的诊断错误（这些错误通常是无害的）
# 这些错误来自xarray的CachingFileManager在__del__时访问已关闭的文件句柄

# 方法1：设置环境变量来抑制HDF5诊断输出（C级别）
# 这需要在导入netCDF4之前设置
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
# 注意：HDF5库可能直接写入stderr，Python的sys.stderr重定向可能无法捕获
# 但我们可以尝试通过环境变量来抑制

_original_stderr = sys.stderr

class HDF5ErrorFilter:
    """过滤HDF5垃圾回收时的诊断错误"""
    def __init__(self):
        self.enabled = True
        self._buffer = ""
        self._in_hdf5_block = False  # 标记是否在处理HDF5错误块中
    
    def write(self, text):
        if not self.enabled:
            _original_stderr.write(text)
            return
        
        # 累积文本到缓冲区
        self._buffer += text
        
        # 检查是否开始HDF5错误块
        if not self._in_hdf5_block:
            if self._is_hdf5_error_start(self._buffer):
                self._in_hdf5_block = True
        
        # 如果缓冲区中有换行符，处理完整的行
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            # 保留最后一行（可能不完整）
            self._buffer = lines[-1]
            
            # 处理完整的行
            output_lines = []
            for line in lines[:-1]:
                if self._in_hdf5_block:
                    # 在HDF5错误块中，检查是否结束
                    if self._is_hdf5_error_end(line):
                        # 错误块结束
                        self._in_hdf5_block = False
                        # 不输出这一行（它是错误块的结束标记，通常是空行或非HDF5行）
                    # 否则继续在错误块中，不输出
                else:
                    # 不在错误块中，检查是否开始新的错误块
                    if self._is_hdf5_error_start(line):
                        self._in_hdf5_block = True
                    else:
                        # 正常行，输出
                        output_lines.append(line)
            
            # 输出非HDF5错误的行
            if output_lines:
                _original_stderr.write('\n'.join(output_lines) + '\n')
        
        # 如果缓冲区中没有换行符，且不在HDF5错误块中，检查是否需要输出
        elif not self._in_hdf5_block and not self._is_hdf5_error_start(self._buffer):
            # 不在错误块中，且不是错误开始，可能需要输出
            # 但为了安全，我们等待换行符
            pass
    
    def _is_hdf5_error_start(self, text):
        """检查文本是否是HDF5错误的开始"""
        start_patterns = [
            'HDF5-DIAG:',
            'Exception ignored in:',
            'Traceback (most recent call last):',
            'CachingFileManager.__del__'
        ]
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in start_patterns)
    
    def _is_hdf5_error_end(self, line):
        """检查是否是HDF5错误块的结束"""
        # HDF5错误块通常以空行结束，或者遇到非HDF5相关的行
        line_stripped = line.strip()
        if not line_stripped:
            return True  # 空行通常表示错误块结束
        
        # 检查是否包含HDF5相关的关键词或格式
        hdf5_keywords = [
            'hdf5', 'h5d', 'h5a', 'h5vl', 'netcdf', 'file_manager',
            'invalid', 'identifier', 'dataspace', 'major:', 'minor:',
            'runtimeerror', 'traceback', 'cachingfilemanager'
        ]
        line_lower = line_stripped.lower()
        has_hdf5_keyword = any(keyword in line_lower for keyword in hdf5_keywords)
        
        # HDF5错误行的特征：
        # 1. 以#开头的行（如#000, #001）
        # 2. 包含"major:"或"minor:"的行
        # 3. 包含文件路径和行号的行（如"File \"...file_manager.py\" line 250"）
        is_hdf5_format = (
            line_stripped.startswith('#') or
            'major:' in line_lower or
            'minor:' in line_lower or
            ('file "' in line_lower and ('line ' in line_lower or '.py' in line_lower))
        )
        
        # 如果行不包含HDF5关键词，且不符合HDF5错误格式，可能是错误块结束
        if not has_hdf5_keyword and not is_hdf5_format:
            return True
        
        return False
    
    def flush(self):
        # 刷新缓冲区
        if self._buffer and not self._in_hdf5_block and not self._is_hdf5_error_start(self._buffer):
            _original_stderr.write(self._buffer)
            self._buffer = ""
        _original_stderr.flush()
        self._in_hdf5_block = False  # 重置状态

# 创建错误过滤器（可选，默认启用）
_hdf5_error_filter = HDF5ErrorFilter()

def enable_hdf5_error_filter():
    """启用HDF5错误过滤"""
    _hdf5_error_filter.enabled = True
    sys.stderr = _hdf5_error_filter

def disable_hdf5_error_filter():
    """禁用HDF5错误过滤"""
    _hdf5_error_filter.enabled = False
    sys.stderr = _original_stderr

# 默认启用错误过滤
enable_hdf5_error_filter()

# 全局数据集缓存（线程安全）
_datasets: Dict[str, xr.Dataset] = {}
_cache_lock = threading.Lock()
_max_cache_size = 10  # 最大缓存数据集数量（避免内存溢出）


def _safe_close_dataset(ds: xr.Dataset):
    """
    安全地关闭NetCDF数据集，避免重复关闭导致的错误
    
    Args:
        ds: xarray Dataset对象
    """
    if ds is None:
        return
    
    try:
        # 检查数据集是否已经关闭
        # xarray Dataset 的 _file_obj 属性可以告诉我们文件是否打开
        if hasattr(ds, '_file_obj') and ds._file_obj is not None:
            # 检查底层文件对象是否仍然有效
            if hasattr(ds._file_obj, 'isopen'):
                if not ds._file_obj.isopen():
                    return  # 文件已经关闭，不需要再次关闭
        
        # 尝试关闭数据集
        ds.close()
    except (RuntimeError, AttributeError, OSError, ValueError) as e:
        # 忽略所有可能的关闭错误：
        # - RuntimeError: NetCDF: Not a valid ID (文件已经关闭)
        # - AttributeError: 对象没有某些属性
        # - OSError: 文件系统相关错误
        # - ValueError: 其他值错误
        pass
    except Exception:
        # 捕获所有其他异常，确保不会影响程序运行
        pass


def _validate_dataset(ds: xr.Dataset) -> bool:
    """
    验证数据集是否有效
    
    Args:
        ds: xarray Dataset对象
    
    Returns:
        bool: 如果数据集有效返回True，否则返回False
    """
    if ds is None:
        return False
    try:
        # 尝试访问数据集的基本属性
        dims = ds.dims  # 这会触发数据集的延迟加载
        data_vars = ds.data_vars if hasattr(ds, 'data_vars') else {}
        coords = ds.coords if hasattr(ds, 'coords') else {}
        
        # 检查是否所有属性都为空（这通常表示数据集已失效）
        if len(dims) == 0 and len(data_vars) == 0 and len(coords) == 0:
            return False
        
        # 尝试访问一个坐标（如果存在）
        if len(coords) > 0:
            _ = list(coords.keys())[0]
        
        # 尝试访问一个数据变量（如果存在）
        if len(data_vars) > 0:
            _ = list(data_vars.keys())[0]
        
        return True
    except (AttributeError, RuntimeError, OSError, ValueError):
        # 数据集无效（已关闭、损坏或无法访问）
        return False
    except Exception:
        # 其他异常也视为无效
        return False


def get_netcdf_dataset(filepath: str, engine: Optional[str] = None) -> xr.Dataset:
    """
    获取NetCDF数据集（如果已缓存则返回缓存的，否则打开并缓存）
    
    注意：此函数是线程安全的，但在多线程环境下，返回的数据集对象本身可能不是线程安全的。
    建议：每个线程在使用数据集时，应该只读取数据，不要修改数据集状态。
    
    Args:
        filepath: NetCDF文件路径
        engine: 引擎名称（'netcdf4', 'h5netcdf', None）
    
    Returns:
        xr.Dataset: 数据集对象（注意：不要关闭，由缓存管理器管理）
    """
    # 使用绝对路径作为键
    abs_path = os.path.abspath(filepath)
    
    # 使用锁保护整个操作，确保线程安全
    with _cache_lock:
        # 检查是否已缓存
        if abs_path in _datasets:
            ds = _datasets[abs_path]
            # 验证缓存的数据集是否仍然有效
            # 注意：在多线程环境下，数据集可能在验证后立即失效
            # 因此我们需要在使用时再次验证
            try:
                if _validate_dataset(ds):
                    # 再次快速检查data_vars是否为空（防止在验证和使用之间失效）
                    try:
                        data_vars_count = len(ds.data_vars)
                        if data_vars_count > 0:
                            return ds
                        else:
                            # data_vars为空，数据集可能已失效
                            print(f"   ⚠️  缓存的数据集data_vars为空，移除并重新打开: {os.path.basename(filepath)}")
                            _safe_close_dataset(ds)
                            del _datasets[abs_path]
                    except Exception:
                        # 访问data_vars时出错，数据集已失效
                        print(f"   ⚠️  缓存的数据集访问失败，移除并重新打开: {os.path.basename(filepath)}")
                        _safe_close_dataset(ds)
                        del _datasets[abs_path]
                else:
                    # 数据集无效，从缓存中移除
                    print(f"   ⚠️  缓存的数据集无效，移除并重新打开: {os.path.basename(filepath)}")
                    _safe_close_dataset(ds)
                    del _datasets[abs_path]
            except Exception as e:
                # 验证过程中出错，移除并重新打开
                print(f"   ⚠️  验证缓存数据集时出错: {e}，移除并重新打开: {os.path.basename(filepath)}")
                try:
                    _safe_close_dataset(ds)
                except:
                    pass
                if abs_path in _datasets:
                    del _datasets[abs_path]
        
        # 如果缓存已满，删除最旧的（FIFO）
        if len(_datasets) >= _max_cache_size:
            # 删除第一个条目
            oldest_key = next(iter(_datasets))
            _safe_close_dataset(_datasets[oldest_key])
            del _datasets[oldest_key]
        
        # 打开并缓存
        engines_to_try = [engine] if engine else ['netcdf4', 'h5netcdf', None]
        ds = None
        last_error = None
        
        for eng in engines_to_try:
            try:
                if eng:
                    ds = xr.open_dataset(filepath, engine=eng)
                else:
                    ds = xr.open_dataset(filepath)
                # 验证新打开的数据集
                if _validate_dataset(ds) and len(ds.data_vars) > 0:
                    break
                else:
                    # 数据集无效，关闭并尝试下一个引擎
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
            error_msg = f"无法打开NetCDF文件 {filepath}"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise RuntimeError(error_msg)
        
        # 缓存数据集
        _datasets[abs_path] = ds
        return ds


def clear_netcdf_cache():
    """清空NetCDF数据集缓存"""
    with _cache_lock:
        for ds in _datasets.values():
            _safe_close_dataset(ds)
        _datasets.clear()


def get_cache_stats() -> Dict[str, int]:
    """获取缓存统计信息"""
    with _cache_lock:
        return {
            'cached_datasets': len(_datasets),
            'max_cache_size': _max_cache_size
        }


