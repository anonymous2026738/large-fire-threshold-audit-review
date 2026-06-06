# 公平性分析工具使用指南

## 安装依赖

### 1. 安装 fairlearn（推荐）

```bash
conda activate fire_py310
pip install fairlearn
```

### 2. 其他依赖

脚本会自动使用以下已安装的库：
- torch
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

## 使用方法

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 模型 checkpoint 路径（完整运行时必填；`--figures-only` 时可省略） |
| `--data` | 数据文件路径，可多个 .pth（完整运行时必填；`--figures-only` 时可省略） |
| `--output` | 输出目录，默认 `fairness_results` |
| `--figures-only` | 仅从缓存重绘结果图，不加载模型和数据 |
| `--cache` | 绘图缓存路径：完整运行时写入 `output/fairness_plot_cache.pkl`；`--figures-only` 时从此文件读取（不指定则用 `output/fairness_plot_cache.pkl`） |

### 基本用法（完整运行）

**Linux/Mac (Bash/Zsh):**
```bash
python fairness_analysis.py \
    --checkpoint outputs/2026-01-10/20-35-51/lightning_logs/fire_equality_convlstm_2002-2020/version_0/checkpoints/fire-equality-epoch=26-val_auc=0.0000.ckpt \
    --data dataset/processed_firetracks_pixel_binary_2002-2004.pth \
           dataset/processed_firetracks_pixel_binary_2005-2007.pth \
           dataset/processed_firetracks_pixel_binary_2008-2009.pth \
           dataset/processed_firetracks_pixel_binary_2010-2013.pth \
           dataset/processed_firetracks_pixel_binary_2014-2015.pth \
           dataset/processed_firetracks_pixel_binary_2016-2018.pth \
           dataset/processed_firetracks_pixel_binary_2019-2020.pth \
    --output fairness_results
```

**Windows PowerShell:**
```powershell
python fairness_analysis.py `
    --checkpoint outputs/2026-01-10/20-35-51/lightning_logs/fire_equality_convlstm_2002-2020/version_0/checkpoints/fire-equality-epoch=26-val_auc=0.0000.ckpt `
    --data dataset/processed_firetracks_pixel_binary_2002-2004.pth `
           dataset/processed_firetracks_pixel_binary_2005-2007.pth `
           dataset/processed_firetracks_pixel_binary_2008-2009.pth `
           dataset/processed_firetracks_pixel_binary_2010-2013.pth `
           dataset/processed_firetracks_pixel_binary_2014-2015.pth `
           dataset/processed_firetracks_pixel_binary_2016-2018.pth `
           dataset/processed_firetracks_pixel_binary_2019-2020.pth `
    --output fairness_results
```

**注意：** 
- Bash/Zsh 使用反斜杠 `\` 作为行续接符
- PowerShell 使用反引号 `` ` ``（在键盘上通常是 `~` 键）作为行续接符
- 行续接符必须在行尾，后面不能有空格

### 使用训练好的最佳模型

```bash
# python fairness_analysis.py --checkpoint "multirun\2026-01-30\14-55-17\2\lightning_logs\fire_equality_convlstm_2002-2020\version_0\checkpoints\fire-equality-epoch=34-val_auc=0.0000.ckpt" --data dataset/processed_firetracks_pixel_binary_2002-2004.pth dataset/processed_firetracks_pixel_binary_2005-2007.pth dataset/processed_firetracks_pixel_binary_2008-2009.pth dataset/processed_firetracks_pixel_binary_2010-2013.pth dataset/processed_firetracks_pixel_binary_2014-2015.pth dataset/processed_firetracks_pixel_binary_2016-2018.pth dataset/processed_firetracks_pixel_binary_2019-2020.pth --output fairness_results_best

python fairness_analysis.py --checkpoint E:\FireEqual\multirun\2026-02-27\21-33-26\0\lightning_logs\fire_equality_convlstm_2002-2020\version_0\checkpoints\fire-equality-epoch=48-val_auc=0.0000.ckpt --data E:\FireEqual\dataset\processed_firetracks_pixel_binary_2002-2004.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2005-2007.pth  E:\FireEqual\dataset\processed_firetracks_pixel_binary_2008-2009.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2010-2013.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2014-2015.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2016-2018.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2019-2020.pth --output fairness_results_best
```

### 使用最佳模型 + backup 数据 + 实验一（XAI）

```bash
# python fairness_analysis.py --checkpoint E:\FireEqual\multirun\2026-01-30\14-55-17\2\lightning_logs\fire_equality_convlstm_2002-2020\version_0\checkpoints\fire-equality-epoch=34-val_auc=0.0000.ckpt --data E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2002-2004.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2005-2007.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2008-2009.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2010-2013.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2014-2015.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2016-2018.pth E:\FireEqual\dataset\backup\processed_firetracks_pixel_binary_2019-2020.pth --output fairness_results_best --xai

python fairness_analysis.py --checkpoint E:\FireEqual\multirun\2026-02-27\21-33-26\0\lightning_logs\fire_equality_convlstm_2002-2020\version_0\checkpoints\fire-equality-epoch=48-val_auc=0.0000.ckpt --data E:\FireEqual\dataset\processed_firetracks_pixel_binary_2002-2004.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2005-2007.pth  E:\FireEqual\dataset\processed_firetracks_pixel_binary_2008-2009.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2010-2013.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2014-2015.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2016-2018.pth E:\FireEqual\dataset\processed_firetracks_pixel_binary_2019-2020.pth --output fairness_results_best --xai
```

### 仅重绘结果图（有缓存时，不加载模型和数据）

完整运行一次后，会在输出目录下生成 `fairness_plot_cache.pkl`。之后使用 `--figures-only` 会**重绘全部结果图**，与完整运行生成的图表一致，且无需再次加载 checkpoint 和 .pth 数据（仅基于缓存的预测与敏感属性，会重新跑阈值优化以出优化相关图）。

**命令（缓存默认在输出目录下）：**

```bash
python fairness_analysis.py --figures-only --output fairness_results
```

即：从 `fairness_results/fairness_plot_cache.pkl` 读取缓存，在 `fairness_results/` 下**全部重绘**以下内容：

| 类型 | 文件 |
|------|------|
| 性能对比 | `performance_comparison_pop_group.png` |
| 性能对比（优化后） | `performance_comparison_optimized.png` |
| ROC 曲线 | `roc_curve_overall_and_pop_group.png` |
| ROC 曲线 | `roc_curve_overall_and_gdp_group.png` |
| ROC 曲线 | `roc_curve_overall_and_continent_group.png` |
| 阈值权衡曲线 | `threshold_tradeoff_curves.png`（样本数 &lt; 10 万时） |
| 分组优化对比 | `performance_comparison_population_density_optimized.png` |
| 分组优化对比 | `performance_comparison_gdp_per_capita_optimized.png` |
| 分组优化对比 | `performance_comparison_continent_optimized.png` |
| 对比总结 | `grouping_comparison_summary.md` |

**指定缓存路径（缓存不在输出目录时）：**

```bash
python fairness_analysis.py --figures-only --cache path/to/fairness_plot_cache.pkl --output fairness_results
```

**Windows PowerShell 示例：**

```powershell
python fairness_analysis.py --figures-only --output fairness_results
```

## 输出结果

分析完成后，会在 `--output` 指定目录（默认 `fairness_results`）下生成（**使用 `--figures-only` 时会全部重绘**）：

1. **fairness_report.md**：详细文本分析报告
2. **performance_comparison_pop_group.png**：按人口密度分组的性能对比图（原始阈值）
3. **performance_comparison_optimized.png**：人口密度组阈值优化前后对比图
4. **roc_curve_overall_and_pop_group.png**：总体 + 人口密度分组的 ROC 曲线
5. **roc_curve_overall_and_gdp_group.png**：总体 + 人均 GDP 分组的 ROC 曲线
6. **roc_curve_overall_and_continent_group.png**：总体 + 大洲分组的 ROC 曲线
7. **threshold_tradeoff_curves.png**：阈值权衡曲线（样本数 &lt; 10 万时）
8. **performance_comparison_population_density_optimized.png**：人口密度分组（优化阈值）性能对比
9. **performance_comparison_gdp_per_capita_optimized.png**：人均 GDP 分组（优化阈值）性能对比
10. **performance_comparison_continent_optimized.png**：大洲分组（优化阈值）性能对比
11. **grouping_comparison_summary.md**：三种分类方式对比总结
12. **fairness_plot_cache.pkl**：绘图用缓存，供 `--figures-only` 重绘全部结果图时使用

## 敏感属性与 ISO3 / GDP / 大洲 提取（新增）

脚本会为每个样本解析 **iso3**、**人均 GDP**、**大洲** 等，用于分组与公平性分析；**无需重跑 .pth** 即可提高 GDP/大洲 匹配率。

### ISO3 获取顺序
1. **方法1**：从样本 `metadata['iso3']` 读取（若建 .pth 时已写入）。
2. **方法3**：从协变量表 `dataset/filtered_cleaned_cp_covariate.csv` 用 **iso3+年份** 或 **经纬度+年份** 做 **0.1° 内** 精确匹配。
3. **方法3.5（兜底）**：若仍无 iso3，用协变量表 **经纬度最近邻**（**2° 内**）匹配：  
   加载协变量后构建 **KD-tree**（`scipy.spatial.cKDTree`），对样本的 `pixel_lat`/`pixel_lon`（或 `metadata['center_lat']`/`metadata['center_lon']`）查最近邻，若距离 ≤ 2° 则取该点的 iso3 与 continent。  
   这样旧 .pth 没有 metadata iso3 时也能补全，降低 GDP/continent 的 Unknown 比例。

### 经纬度来源
- 优先：`sample['pixel_lat']` / `sample['pixel_lon']`
- 若无：`metadata['center_lat']` / `metadata['center_lon']`

### 依赖
- 协变量表：`dataset/filtered_cleaned_cp_covariate.csv`（需含列 `lat_mean`, `lon_mean`, `iso3`，可选 `continent`）
- 人均 GDP：`dataset/gdp_per_capita.csv`（需含 `Country Code` 与年份列）
- 方法3.5 需安装 `scipy`（脚本中已有 scipy 统计检验等依赖）

---

## 分析内容

### 1. 敏感属性提取
- 从特征中提取人口密度信息（通道3）
- 按人口密度分为：Low / Medium / High 三组
- 分组基于33%和67%分位数
- iso3 / GDP / 大洲 见上文「敏感属性与 ISO3 / GDP / 大洲 提取」

### 2. 性能指标
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- AUC
- AUPRC

### 3. 公平性指标
- **Demographic Parity Difference**: 统计均等差异（越小越好）
- **Demographic Parity Ratio**: 统计均等比率（越接近1越好）
- **Equalized Odds Difference**: 均等化机会差异（越小越好）
- **Equalized Odds Ratio**: 均等化机会比率（越接近1越好）

## 注意事项

1. **如果没有安装fairlearn**：脚本仍会运行，但只计算自定义的公平性指标（性能差异和比率）

2. **数据要求**：
   - 数据文件必须包含 `spatiotemporal_samples`
   - 样本的 `features` 中应包含人口密度信息（通道3）
   - 如果metadata中有ISO3代码，会优先使用
   - 如果特征中没有人口数据，会尝试从 `dataset/filtered_cleaned_cp_covariate.csv` 中查找

3. **内存使用**：
   - 如果数据文件很大，可能需要较多内存
   - 建议先在小数据集上测试

## 示例输出解读

### 性能差异
- **差异 < 5%**: 基本公平，可以接受
- **差异 5-10%**: 中等差异，需要关注
- **差异 > 10%**: 显著差异，需要改进

### 公平性指标
- **Demographic Parity Difference < 0.05**: 统计均等性良好
- **Equalized Odds Difference < 0.05**: 均等化机会良好

## 后续改进建议

如果发现不公平问题：

1. **数据层面**：
   - 增加弱势群体的训练数据
   - 平衡不同群体的样本数量

2. **模型层面**：
   - 使用fairlearn进行后处理
   - 调整类别权重
   - 使用对抗训练减少偏差

3. **特征层面**：
   - 检查特征在不同群体间的分布
   - 移除可能导致偏差的特征

