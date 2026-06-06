"""
Captum解释器
用于PyTorch模型的可解释性分析
"""

import torch
import numpy as np
from typing import Optional, Callable, Any, List
import warnings

try:
    from captum.attr import (
        IntegratedGradients,
        GradientShap,
        Saliency,
        GuidedBackprop,
        DeepLift,
        InputXGradient,
        Occlusion,
        FeatureAblation
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. Install with: pip install captum")


class CaptumExplainer:
    """
    Captum解释器类
    
    提供多种归因方法用于模型可解释性分析
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        初始化Captum解释器
        
        Args:
            model: 训练好的PyTorch模型
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is not installed. Install with: pip install captum")
        
        self.model = model
        self.model.eval()  # 设置为评估模式
        
    def integrated_gradients(
        self, 
        inputs: torch.Tensor, 
        target: Optional[int] = None,
        baselines: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ):
        """
        积分梯度归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
            baselines: 基线输入
            n_steps: 积分步数
        
        Returns:
            归因值
        """
        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(
            inputs, 
            baselines=baselines, 
            target=target,
            n_steps=n_steps
        )
        return attributions
    
    def gradient_shap(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baselines: Optional[torch.Tensor] = None,
        n_samples: int = 50
    ):
        """
        梯度SHAP归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
            baselines: 基线输入
            n_samples: 采样次数
        
        Returns:
            归因值
        """
        gs = GradientShap(self.model)
        attributions = gs.attribute(
            inputs,
            baselines=baselines,
            target=target,
            n_samples=n_samples
        )
        return attributions
    
    def saliency(self, inputs: torch.Tensor, target: Optional[int] = None):
        """
        显著性归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
        
        Returns:
            归因值
        """
        saliency = Saliency(self.model)
        attributions = saliency.attribute(inputs, target=target)
        return attributions
    
    def guided_backprop(self, inputs: torch.Tensor, target: Optional[int] = None):
        """
        引导反向传播
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
        
        Returns:
            归因值
        """
        gbp = GuidedBackprop(self.model)
        attributions = gbp.attribute(inputs, target=target)
        return attributions
    
    def deeplift(self, inputs: torch.Tensor, target: Optional[int] = None, baselines: Optional[torch.Tensor] = None):
        """
        DeepLift归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
            baselines: 基线输入
        
        Returns:
            归因值
        """
        dl = DeepLift(self.model)
        attributions = dl.attribute(inputs, baselines=baselines, target=target)
        return attributions
    
    def occlusion(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        sliding_window_shapes: tuple = (1, 1, 1),
        strides: Optional[tuple] = None
    ):
        """
        遮挡归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
            sliding_window_shapes: 滑动窗口形状
            strides: 步长
        
        Returns:
            归因值
        """
        occlusion_attr = Occlusion(self.model)
        attributions = occlusion_attr.attribute(
            inputs,
            target=target,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides
        )
        return attributions
    
    def feature_ablation(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        feature_mask: Optional[torch.Tensor] = None
    ):
        """
        特征消融归因
        
        Args:
            inputs: 输入张量
            target: 目标类别索引
            feature_mask: 特征掩码
        
        Returns:
            归因值
        """
        fa = FeatureAblation(self.model)
        attributions = fa.attribute(inputs, target=target, feature_mask=feature_mask)
        return attributions
    
    def visualize_attributions(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor,
        method_name: str = "attribution"
    ):
        """
        可视化归因结果
        
        Args:
            attributions: 归因值
            inputs: 原始输入
            method_name: 方法名称
        """
        try:
            from captum.attr import visualization as viz
            
            # 转换为numpy
            if isinstance(attributions, torch.Tensor):
                attributions = attributions.detach().cpu().numpy()
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            
            # 可视化
            viz.visualize_image_attributions(
                attributions,
                inputs,
                method=method_name,
                sign="all",
                show_colorbar=True
            )
        except Exception as e:
            print(f"Visualization error: {e}")
            print(f"Attributions shape: {attributions.shape}")
            print(f"Inputs shape: {inputs.shape}")

