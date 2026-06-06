"""
SHAP (SHapley Additive exPlanations) 解释器
用于模型特征重要性分析
"""

import numpy as np
import torch
from typing import Optional, Callable, Any
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP解释器类
    
    使用SHAP值来解释模型预测，提供特征重要性分析
    """
    
    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor):
        """
        初始化SHAP解释器
        
        Args:
            model: 训练好的PyTorch模型
            background_data: 背景数据集，用于计算SHAP值
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def create_explainer(self, explainer_type: str = "DeepExplainer"):
        """
        创建SHAP解释器
        
        Args:
            explainer_type: 解释器类型 ("DeepExplainer", "KernelExplainer", "LinearExplainer")
        """
        if explainer_type == "DeepExplainer":
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        elif explainer_type == "KernelExplainer":
            # 包装模型函数
            def model_wrapper(x):
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                with torch.no_grad():
                    return self.model(x).cpu().numpy()
            
            self.explainer = shap.KernelExplainer(
                model_wrapper, 
                self.background_data.cpu().numpy()
            )
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    def explain(self, data: torch.Tensor, max_evals: Optional[int] = None):
        """
        计算SHAP值
        
        Args:
            data: 要解释的数据
            max_evals: 最大评估次数（仅用于KernelExplainer）
        
        Returns:
            SHAP值
        """
        if self.explainer is None:
            self.create_explainer()
        
        if isinstance(self.explainer, shap.DeepExplainer):
            shap_values = self.explainer.shap_values(data)
        else:
            shap_values = self.explainer.shap_values(
                data.cpu().numpy(), 
                nsamples=max_evals or 100
            )
        
        return shap_values
    
    def plot_summary(self, shap_values, feature_names: Optional[list] = None, **kwargs):
        """
        绘制SHAP摘要图
        
        Args:
            shap_values: SHAP值
            feature_names: 特征名称列表
            **kwargs: 传递给shap.summary_plot的其他参数
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        shap.summary_plot(shap_values, feature_names=feature_names, **kwargs)
    
    def plot_waterfall(self, shap_values, feature_names: Optional[list] = None, **kwargs):
        """
        绘制SHAP瀑布图
        
        Args:
            shap_values: SHAP值
            feature_names: 特征名称列表
            **kwargs: 传递给shap.waterfall_plot的其他参数
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        shap.waterfall_plot(shap_values, feature_names=feature_names, **kwargs)

