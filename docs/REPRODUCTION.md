# 完整复现流程：训练 → 网格搜索 → 公平性分析

本文档对应 **paper_draft.tex** 中的实验流程：在**更新后的数据**（`dataset/processed_firetracks_pixel_binary_*.pth`，7 个分片，2002–2020）上重新训练、做超参数网格搜索、选出最优 checkpoint、再运行公平性分析，结果写入 **fairness_results_best**。

---

## 已有文档说明（便于整理）

- **environment_setup.md**：conda 环境与依赖  
- **dataset/BUILD_PTH_COMMANDS.md**：从原始数据构建 .pth  
- **fairness_analysis_README.md**：公平性分析脚本参数说明  
- **multirun_analysis_summary.md**：历史 multirun 结果与 Run 18 说明  
- **paper_summary.md**：数据/模型/公平性方法摘要  

本文件 **REPRODUCTION_WORKFLOW.md** 只负责**按顺序给出可复现命令**，不重复上述细节。

---

## 环境与工作目录

所有命令均在**本仓库根目录**下执行，且先激活环境：

```powershell
conda activate fire-danger-audit
cd path/to/large-fire-threshold-audit-review
```

数据使用 **data/processed/** 下 7 个分片（需自行构建，见 `docs/BUILD_PTH_COMMANDS.md`）：

- `data/processed/processed_firetracks_pixel_binary_2002-2004.pth`
- `data/processed/processed_firetracks_pixel_binary_2005-2007.pth`
- `data/processed/processed_firetracks_pixel_binary_2008-2009.pth`
- `data/processed/processed_firetracks_pixel_binary_2010-2013.pth`
- `data/processed/processed_firetracks_pixel_binary_2014-2015.pth`
- `data/processed/processed_firetracks_pixel_binary_2016-2018.pth`
- `data/processed/processed_firetracks_pixel_binary_2019-2020.pth`

---

## 第一步：主网格搜索（16 组，与论文一致）

**训练入口**：**`scripts/train.py`**。若存在 `data/processed/processed_firetracks_pixel_binary_2002-2020.pth` 则用合并文件，否则用上述 7 个分片（多文件流式加载）。

使用 Hydra 的 `-m` multirun，对 **lr、hidden_size、positive_weight、seed** 做网格搜索（论文 Section 2.3）：

- `model.lr` ∈ {3e-4, 1e-3}
- `model.hidden_size` ∈ {32, 64}
- `model.positive_weight` ∈ {0.5, 1.0}
- `seed` ∈ {42, 123}

```powershell
python scripts/train.py -m model.lr=0.0003,0.001 model.hidden_size=32,64 model.positive_weight=0.5,1.0 seed=42,123
```

**说明**：在项目根运行，multirun 输出在 **`multirun/YYYY-MM-DD/HH-MM-SS/0`** … **`15`**。每个子目录对应一个 run，每个 run 内按 **val/auc** 保存 best checkpoint。

---

## 第二步（可选）：细网格搜索

在主网格中选出表现较好的区域后，可在其附近做细网格（例如论文中的 Run 16–19）：微调 lr、hidden_size、seed。下面示例是 4 组 scenario（run16–run19）。

**方式 A：一次 multirun 4 个 scenario（推荐）**

```powershell
python scripts/train.py -m scenario=run16,run17,run18,run19
```

**方式 B：只跑当前最优配置（例如 Run 18 对应 scenario=run18）**

```powershell
python scripts/train.py scenario=run18
```

细网格的 lr/hidden_size/seed 定义在 `configs/scenario/run16.yaml` … `run19.yaml`。

---

## 第三步：确定最优 checkpoint

1. 在 multirun 输出目录 **`multirun/YYYY-MM-DD/HH-MM-SS/`** 下，根据各子目录（0…15 或 0…3）中 **val/auc** 选最高的一次 run。  
2. 该 run 目录下 best checkpoint 路径形如：  
   `multirun/YYYY-MM-DD/HH-MM-SS/<N>/lightning_logs/fire_equality_convlstm_2002-2020/version_0/checkpoints/fire-equality-epoch=XX-val_auc=0.0000.ckpt`  
   （文件名里 val_auc 可能显示为 0.0000，是 callback 记录方式问题，以 CSV/日志里的 val/auc 为准。）  
3. 记下该 **完整路径**（相对项目根或绝对路径），用于第四步。

---

## 第四步：公平性分析并写入 fairness_results_best

将 checkpoint 路径代入下面的 `--checkpoint`（本 release 已包含 `results/checkpoints/fire-equality-run18-epoch49.ckpt`），用 **data/processed/** 下 7 个 .pth，输出到 **results/audit_full**：

```powershell
python scripts/fairness_analysis.py --checkpoint results/checkpoints/fire-equality-run18-epoch49.ckpt --data data/processed/processed_firetracks_pixel_binary_2002-2004.pth data/processed/processed_firetracks_pixel_binary_2005-2007.pth data/processed/processed_firetracks_pixel_binary_2008-2009.pth data/processed/processed_firetracks_pixel_binary_2010-2013.pth data/processed/processed_firetracks_pixel_binary_2014-2015.pth data/processed/processed_firetracks_pixel_binary_2016-2018.pth data/processed/processed_firetracks_pixel_binary_2019-2020.pth --output results/audit_full --xai
```

仅重绘已发布图（无需数据/checkpoint）：

```powershell
python scripts/reproduce_figures.py
```

---

## 流程小结

| 步骤 | 内容 | 输出 |
|------|------|------|
| 1 | 主网格 multirun（16 组） | `multirun/YYYY-MM-DD/HH-MM-SS/0` … `15`，每 run 一个 best checkpoint |
| 2（可选） | 细网格 run16–run19 | 同上，4 个 run |
| 3 | 按 val/auc 选最优 run，记下 checkpoint 路径 | 一个 .ckpt 路径 |
| 4 | `fairness_analysis.py`，7 个 .pth，`--output fairness_results_best` | **fairness_results_best/** 下报告与图 |

按上述顺序执行即可在**更新后的数据**上完整复现论文中的训练与公平性分析流程；仅数据更新，流程与 paper_draft.tex 描述一致。
