# 按年份分段生成 .pth（分终端运行）

**运行环境**：运行任何 Python 命令前请先激活 conda 环境：`conda activate fire_py310`。环境创建与依赖安装见项目根目录下 `environment_setup.md`。

在**项目根目录** `E:\FireEqual` 下打开多个终端，每个终端执行下面**其中一条**命令（一条对应一个年份段）。  
每条命令会生成一个 `dataset/processed_firetracks_pixel_binary_YYYY-YYYY.pth` 文件。

**样本量**：下面命令已加 `--max_samples`，按「每年约 1000 个正样本」设置（总正样本数 = 1000 × 该段年数），负样本仍按 `neg_pos_ratio=2` 约 2 倍。若要不限样本量，去掉 `--max_samples ...` 即可。

---

## 运行前准备（建议）

1. **删掉该年份段的负样本检查点（若存在）**  
   若之前用「不限样本量」或其它 `--max_samples` 跑过同一年份段，建议先删掉对应检查点，否则会沿用旧检查点、负样本数量可能不对。  
   在 `dataset/` 下删除与本次要跑的年份段匹配的文件即可，例如跑 2002–2004 前执行：
   ```bash
   # 在项目根目录执行，按需把 2002、2004 换成你当前要跑的起止年
   del dataset\negative_samples_checkpoint_2002_2004_p*.pth
   ```
   （Linux/macOS 用 `rm dataset/negative_samples_checkpoint_2002_2004_p*.pth`）

2. **备份已有 .pth 再跑**  
   若已有同名的 `processed_firetracks_pixel_binary_YYYY-YYYY.pth`，建议先重命名或挪到备份目录，再执行下面的命令，避免被覆盖后无法恢复。例如：
   ```powershell
   ren dataset\processed_firetracks_pixel_binary_2002-2004.pth processed_firetracks_pixel_binary_2002-2004.pth.bak
   ```
   （或新建 `dataset/backup/` 后把旧 .pth 移进去。）

---

## 终端 1：2002-2004（3 年 → 约 3000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2002 2004 --output_path dataset/processed_firetracks_pixel_binary_2002-2004.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 3000
```

**Windows PowerShell / CMD：**
```powershell
python code\fire_equality\datamodules\firetracks_loader.py --target_years 2002 2004 --output_path dataset/processed_firetracks_pixel_binary_2002-2004.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 3000
```

---

## 终端 2：2005-2007（3 年 → 约 3000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2005 2007 --output_path dataset/processed_firetracks_pixel_binary_2005-2007.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 3000
```

---

## 终端 3：2008-2009（2 年 → 约 2000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2008 2009 --output_path dataset/processed_firetracks_pixel_binary_2008-2009.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 2000
```

---

## 终端 4：2010-2013（4 年 → 约 4000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2010 2013 --output_path dataset/processed_firetracks_pixel_binary_2010-2013.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 4000
```

---

## 终端 5：2014-2015（2 年 → 约 2000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2014 2015 --output_path dataset/processed_firetracks_pixel_binary_2014-2015.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 2000
```

---

## 终端 6：2016-2018（3 年 → 约 3000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2016 2018 --output_path dataset/processed_firetracks_pixel_binary_2016-2018.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 3000
```

---

## 终端 7：2019-2020（2 年 → 约 2000 正样本）

```bash
python code/fire_equality/datamodules/firetracks_loader.py --target_years 2019 2020 --output_path dataset/processed_firetracks_pixel_binary_2019-2020.pth --data_directory dataset/firetracks_data --cache_dir dataset --max_samples 2000
```

---

## 说明

- **工作目录**：请先 `cd` 到项目根目录再执行（即包含 `code`、`dataset` 的目录）。
- **样本量**：`--max_samples` 限制的是**正样本总数**；流水线会按年份均分（每年约 1000 个正样本）。负样本数 ≈ 正样本数 × 2（`neg_pos_ratio=2`）。若要不限量，删掉 `--max_samples ...` 即可。
- **并行**：7 个年份段可同时在 7 个终端里跑，互不依赖。
- **缓存**：`--cache_dir dataset` 表示正/负样本池等缓存放在 `dataset/` 下；同一年份段重复跑会复用已有缓存。正样本/负样本池的缓存文件名会随 `max_samples` 变化，但**负样本检查点** `negative_samples_checkpoint_*` 不会，故切换样本量时建议按上面「运行前准备」删除对应检查点。
- **数据目录**：若 FireTracks 数据不在 `dataset/firetracks_data`，把 `--data_directory` 改成实际路径即可。
