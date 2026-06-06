# 特征重采样逻辑说明

## 概述

本文档说明所有特征通道的重采样逻辑，确保重采样方法正确，避免值被错误缩放。

## 各特征通道的重采样方法

### 1. Population (通道3) - GeoTIFF数据 ✅ 已修复

**数据源**: WorldPop ppp (persons per pixel) 数据集
**分辨率**: 1km×1km
**单位**: 每像素人口数（在1km分辨率下，数值等于人口密度，单位：人/km²）

**重采样方法**:
- 使用 `Resampling.sum` 累加总人口数
- 然后除以目标像素面积（km²）得到人口密度（人/km²）

**原因**:
- WorldPop数据是"persons per pixel"，当重采样到更大的像素时，需要累加总人口数
- 然后除以目标像素面积，保持"人/km²"的单位一致性

**代码位置**: `extract_population_feature` (line 2072-2101)

### 2. Land Cover (通道5) - GeoTIFF数据 ✅ 正确

**数据源**: MODIS MCD12Q1 土地覆盖数据
**分辨率**: 500m（本地文件可能已重采样到1km）
**单位**: 分类值（整数，1-17）

**重采样方法**:
- 使用 `Resampling.nearest`（最近邻重采样）

**原因**:
- 土地覆盖是分类数据，不能使用平均或求和
- 最近邻重采样确保每个目标像素获得最接近的源像素的分类值
- 保持分类值的完整性

**代码位置**: `extract_landcover_feature_1km` (line 2343-2353)

### 3. FWI (通道0) - NetCDF数据 ✅ 正确

**数据源**: 本地NetCDF文件
**分辨率**: 通常0.25度（约25km）
**单位**: 火灾天气指数

**提取方法**:
- 使用逐点提取（`isel`），不是重采样
- 将目标网格的经纬度转换为源数据的行列索引
- 直接提取对应位置的值

**原因**:
- FWI是连续变量，不是累积量
- 逐点提取不会导致值被缩放
- 如果像素大小不匹配，最多是丢失一些空间分辨率，但值本身是正确的

**代码位置**: `extract_fwi_feature` (line 669-686)

### 4. VPD (通道1) - NetCDF数据 ✅ 正确

**数据源**: 本地NetCDF文件
**分辨率**: 通常0.25度（约25km）
**单位**: 蒸汽压差

**提取方法**:
- 使用逐点提取（`isel`），不是重采样
- 与FWI相同的逻辑

**代码位置**: `extract_vpd_feature` (line 838-865)

### 5. max_temp (通道6) - NetCDF数据 ✅ 正确

**数据源**: ERA5-Land NetCDF文件
**分辨率**: 通常0.1度（约11km）
**单位**: 摄氏度

**提取方法**:
- 使用逐点提取（`isel`），不是重采样
- 与FWI相同的逻辑

**代码位置**: `extract_era5_temp_feature` (line 1030-1049)

### 6. max_wind (通道7) - NetCDF数据 ✅ 正确（已优化）

**数据源**: ERA5-Land NetCDF文件
**分辨率**: 通常0.1度（约11km）
**单位**: m/s

**提取方法**:
- 使用批量提取（优化版本），不是重采样
- 批量计算所有目标点的行列索引
- 使用xarray的批量索引一次性提取所有值
- 如果批量提取失败，回退到逐点提取

**原因**:
- 批量提取比逐点提取快得多
- 仍然是直接索引，不会导致值被缩放

**代码位置**: `extract_era5_wind_feature` (line 1282-1321)

### 7. NDVI (通道2) - NetCDF数据 ✅ 正确

**数据源**: GIMMS-3G+ NetCDF文件
**分辨率**: 通常0.0833度（约8km）
**单位**: 归一化植被指数（-1到1）

**提取方法**:
- 使用逐点提取（`isel`）+ 时间插值
- 对于半月合成的数据，使用线性插值到每日
- 空间上使用逐点提取

**原因**:
- NDVI是连续变量，不是累积量
- 时间插值是为了处理半月合成数据
- 空间上逐点提取不会导致值被缩放

**代码位置**: `extract_ndvi_feature_gimms` (line 1590-1620)

### 8. GDP (通道4) - CSV数据 ✅ 正确

**数据源**: 本地CSV文件
**单位**: 经济指标

**提取方法**:
- 不需要重采样
- 根据ISO3代码和年份匹配
- 将匹配的值填充到整个网格

**原因**:
- GDP是国家级别的数据，不是空间栅格
- 直接填充，不需要重采样

**代码位置**: `extract_gdp_feature` (line 2278-2279)

## 重采样方法选择指南

### 累积量（需要累加）
- **Population**: 使用 `Resampling.sum` + 除以面积
- **其他累积量**: 如果将来有，也应该使用 `Resampling.sum` + 除以面积

### 分类数据
- **Land Cover**: 使用 `Resampling.nearest`
- **其他分类数据**: 应该使用 `Resampling.nearest` 或 `Resampling.mode`

### 连续变量（不需要累加）
- **FWI, VPD, max_temp, max_wind, NDVI**: 使用逐点提取（`isel`）或插值
- **其他连续变量**: 可以使用 `Resampling.bilinear` 或 `Resampling.cubic` 进行插值

## 常见错误

### ❌ 错误1: 对累积量使用average重采样
```python
# 错误：Population使用average会导致值被稀释
resampling=Resampling.average  # ❌ 错误
```

### ✅ 正确: 对累积量使用sum重采样
```python
# 正确：Population使用sum累加总人口数
resampling=Resampling.sum  # ✅ 正确
# 然后除以目标像素面积
output_data = temp_output / dst_pixel_area_km2
```

### ❌ 错误2: 对分类数据使用average重采样
```python
# 错误：Land Cover使用average会产生无意义的中间值
resampling=Resampling.average  # ❌ 错误
```

### ✅ 正确: 对分类数据使用nearest重采样
```python
# 正确：Land Cover使用nearest保持分类值
resampling=Resampling.nearest  # ✅ 正确
```

## 验证方法

运行 `check_resampling_logic.py` 可以验证所有特征的重采样逻辑是否正确。

## 更新历史

- 2024-XX-XX: 修复Population重采样逻辑，从`Resampling.average`改为`Resampling.sum` + 除以面积

