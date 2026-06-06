# Fire Quality 代码迁移

本文件夹包含从 `Orion-AI-Lab-wildfire_forecasting-ee4f168` 迁移的代码结构，包括：

## 📁 目录结构

```
fire_quality/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── fire_quality_model.py  # ConvLSTM模型类（PyTorch Lightning）
│   └── modules/
│       ├── __init__.py
│       ├── convlstm.py         # ConvLSTM核心模块
│       └── fire_modules.py     # SimpleConvLSTM模块
├── train.py                    # 训练循环结构
├── utils/                      # 工具函数
│   ├── __init__.py
│   └── utils.py                # 训练辅助函数
├── xai/                        # 可解释性AI工具
│   ├── __init__.py
│   ├── shap_explainer.py       # SHAP解释器
│   ├── captum_explainer.py     # Captum解释器
│   └── lime_explainer.py       # LIME解释器
└── README.md                   # 本文件
```

## 🔧 已迁移的组件

### 1. ConvLSTM模型类

- **`models/modules/convlstm.py`**: ConvLSTM核心实现
  - `ConvLSTMCell`: ConvLSTM单元
  - `ConvLSTM`: 多层ConvLSTM网络

- **`models/modules/fire_modules.py`**: SimpleConvLSTM模型
  - 整合ConvLSTM + CNN + 全连接层

- **`models/fire_quality_model.py`**: PyTorch Lightning模型封装
  - `ConvLSTM_fire_quality_model`: 完整的训练/验证/测试循环

### 2. 训练循环结构

- **`train.py`**: 完整的训练流程
  - 数据模块初始化
  - 模型初始化
  - 回调函数配置
  - 日志记录器配置
  - 训练器配置和执行

### 3. 损失函数和优化器

在 `models/fire_quality_model.py` 的 `configure_optimizers()` 方法中：

- **损失函数**: `torch.nn.NLLLoss` (负对数似然损失)
  - 支持类别权重平衡
  - 权重: `[1 - positive_weight, positive_weight]`

- **优化器**: `torch.optim.Adam`
  - 学习率: `lr` (默认 0.001)
  - 权重衰减: `weight_decay` (默认 0.0005)

- **学习率调度器**: `torch.optim.lr_scheduler.StepLR`
  - 步长: `lr_scheduler_step` (默认 10)
  - 衰减率: `lr_scheduler_gamma` (默认 0.1)

### 4. xAI分析工具

#### SHAP解释器 (`xai/shap_explainer.py`)
- `SHAPExplainer`: SHAP值计算和可视化
- 支持 `DeepExplainer` 和 `KernelExplainer`
- 提供摘要图和瀑布图

#### Captum解释器 (`xai/captum_explainer.py`)
- `CaptumExplainer`: 多种归因方法
- 支持的方法：
  - Integrated Gradients (积分梯度)
  - Gradient SHAP
  - Saliency (显著性)
  - Guided Backprop (引导反向传播)
  - DeepLift
  - Occlusion (遮挡)
  - Feature Ablation (特征消融)

#### LIME解释器 (`xai/lime_explainer.py`)
- `LIMEExplainer`: 局部可解释性分析
- 支持图像和表格数据模式
- 提供单个预测的局部解释

## 📦 使用方法

### 导入模型

```python
from fire_quality.models import ConvLSTM_fire_quality_model
from fire_quality.models.modules import SimpleConvLSTM, ConvLSTM
```

### 使用训练循环

```python
from fire_quality.train import train
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)
```

### 使用xAI工具

```python
# SHAP
from fire_quality.xai import SHAPExplainer
explainer = SHAPExplainer(model, background_data)
shap_values = explainer.explain(test_data)

# Captum
from fire_quality.xai import CaptumExplainer
explainer = CaptumExplainer(model)
attributions = explainer.integrated_gradients(inputs, target=1)

# LIME
from fire_quality.xai import LIMEExplainer
explainer = LIMEExplainer(model, mode="image")
explanation = explainer.explain_instance_image(image)
```

## 🔄 与原代码的差异

1. **模块路径**: 所有导入路径已更新为 `fire_quality` 命名空间
2. **模型名称**: `ConvLSTM_fire_model` → `ConvLSTM_fire_quality_model`
3. **独立结构**: 代码已独立，不依赖原始项目的其他部分

## 📝 依赖项

确保安装以下依赖：

```bash
pip install torch pytorch-lightning torchmetrics
pip install shap captum lime  # xAI工具
pip install hydra-core omegaconf  # 配置管理
pip install rich  # 日志美化
```

## ⚠️ 注意事项

1. 数据加载模块需要单独配置，本迁移不包含数据加载代码
2. 配置文件需要根据实际需求调整
3. xAI工具需要模型和数据适配，可能需要根据具体场景调整

## 🚀 下一步

1. 配置数据加载模块
2. 创建配置文件（config.yaml等）
3. 根据实际数据调整模型输入维度
4. 测试训练流程
5. 集成xAI分析工具到训练/评估流程中

