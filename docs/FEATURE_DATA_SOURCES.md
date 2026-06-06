# 特征数据源说明

## 8个特征通道的数据来源

### 通道 0: FWI (火灾天气指数)
- **数据源**: 本地NetCDF文件
- **文件路径**: `dataset/FWI/fire_weather_index_{year}.nc`
- **获取方式**: 本地文件读取
- **是否需要API**: ❌ 否

### 通道 1: VPD (蒸汽压差)
- **数据源**: 本地NetCDF文件
- **文件路径**: `dataset/VPD/vapor_pressure_deficit_{year}.nc`
- **获取方式**: 本地文件读取
- **是否需要API**: ❌ 否

### 通道 2: NDVI (归一化植被指数)
- **数据源**: 本地NetCDF文件（GIMMS-3G+）
- **文件路径**: `dataset/NDVI/GIMMS3G+/ndvi3g_geo_v1_X_{year}_*.nc4`
- **获取方式**: 本地文件读取
- **是否需要API**: ❌ 否
- **注意**: 仅支持GIMMS-3G+本地数据，不支持MOD13A2或GEE API

### 通道 3: Population (人口数)
- **数据源**: 本地GeoTIFF文件（WorldPop）
- **文件路径**: `dataset/worldpop/ppp_{year}_1km_Aggregated.tif`
- **获取方式**: 本地文件读取（rasterio）
- **是否需要API**: ❌ 否
- **单位**: 人/km²（每像素1km²的人口数）

### 通道 4: GDP (经济指标)
- **数据源**: 本地CSV文件
- **文件路径**: `dataset/filtered_cleaned_cp_covariate.csv`
- **获取方式**: 本地文件读取（pandas）
- **是否需要API**: ❌ 否
- **匹配方式**: 根据ISO3代码和年份匹配

### 通道 5: Land Cover (土地覆盖)
- **数据源**: **Google Earth Engine API** (MODIS MCD12Q1)
- **数据集**: `MODIS/061/MCD12Q1`
- **获取方式**: **GEE API调用**
- **是否需要API**: ✅ **是**（唯一使用API的特征）
- **采样方式**: 批量采样，使用1km scale（GEE自动聚合500m数据）
- **认证要求**: 需要GEE认证（`earthengine authenticate`）

### 通道 6: max_temp (ERA5 2米最大温度)
- **数据源**: 本地NetCDF文件（ERA5-Land）
- **文件路径**: `dataset/ERA5/era5_land_temp_{year}.nc`
- **获取方式**: 本地文件读取
- **是否需要API**: ❌ 否

### 通道 7: max_wind (ERA5 10米最大风速)
- **数据源**: 本地NetCDF文件（ERA5-Land）
- **文件路径**: `dataset/ERA5/era5_land_wind_{year}.nc`
- **获取方式**: 本地文件读取
- **是否需要API**: ❌ 否

## 总结

| 通道 | 特征名称 | 数据源 | 是否需要API |
|------|---------|--------|------------|
| 0 | FWI | 本地NetCDF | ❌ |
| 1 | VPD | 本地NetCDF | ❌ |
| 2 | NDVI | 本地NetCDF (GIMMS-3G+) | ❌ |
| 3 | Population | 本地GeoTIFF (WorldPop) | ❌ |
| 4 | GDP | 本地CSV | ❌ |
| 5 | Land Cover | **GEE API** | ✅ **是** |
| 6 | max_temp | 本地NetCDF (ERA5) | ❌ |
| 7 | max_wind | 本地NetCDF (ERA5) | ❌ |

## 重要说明

1. **只有Land Cover（通道5）使用API获取**，其他7个特征都从本地文件读取。

2. **GEE认证**：
   - 首次使用前需要运行：`earthengine authenticate`
   - 认证成功后，凭据会保存在本地，之后无需再次认证
   - 如果GEE认证失败，Land Cover特征会返回零数组

3. **性能影响**：
   - Land Cover是唯一需要网络请求的特征，可能成为性能瓶颈
   - 建议：如果可能，考虑将Land Cover数据预先下载到本地

4. **离线使用**：
   - 除了Land Cover外，其他7个特征都可以完全离线使用
   - 如果GEE不可用，Land Cover会返回零数组，但不会影响其他特征


