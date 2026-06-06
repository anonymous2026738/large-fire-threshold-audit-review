"""
Captum explainer
for PyTorch models
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
    Captum explainer
    
    for model
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        initializeCaptum explainer
        
        Args:
            model: trained PyTorch model
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is not installed. Install with: pip install captum")
        
        self.model = model
        self.model.eval()  # 
        
    def integrated_gradients(
        self, 
        inputs: torch.Tensor, 
        target: Optional[int] = None,
        baselines: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ):
        """
        Integrated Gradients attribution
        
        Args:
            inputs: input tensor
            target: target class index
            baselines: baseline input
            n_steps: number of integration steps
        
        Returns:
            attribution values
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
        Gradient SHAP attribution
        
        Args:
            inputs: input tensor
            target: target class index
            baselines: baseline input
            n_samples: number of samples
        
        Returns:
            attribution values
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
        Saliency attribution
        
        Args:
            inputs: input tensor
            target: target class index
        
        Returns:
            attribution values
        """
        saliency = Saliency(self.model)
        attributions = saliency.attribute(inputs, target=target)
        return attributions
    
    def guided_backprop(self, inputs: torch.Tensor, target: Optional[int] = None):
        """
        Guided Backpropagation
        
        Args:
            inputs: input tensor
            target: target class index
        
        Returns:
            attribution values
        """
        gbp = GuidedBackprop(self.model)
        attributions = gbp.attribute(inputs, target=target)
        return attributions
    
    def deeplift(self, inputs: torch.Tensor, target: Optional[int] = None, baselines: Optional[torch.Tensor] = None):
        """
        DeepLift attribution
        
        Args:
            inputs: input tensor
            target: target class index
            baselines: baseline input
        
        Returns:
            attribution values
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
        Occlusion attribution
        
        Args:
            inputs: input tensor
            target: target class index
            sliding_window_shapes: sliding window shapes
            strides: stride
        
        Returns:
            attribution values
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
        Feature Ablation attribution
        
        Args:
            inputs: input tensor
            target: target class index
            feature_mask: feature mask
        
        Returns:
            attribution values
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
        visualize attribution results
        
        Args:
            attributions: attribution values
            inputs: 
            method_name: 
        """
        try:
            from captum.attr import visualization as viz
            
            # convert to NumPy
            if isinstance(attributions, torch.Tensor):
                attributions = attributions.detach().cpu().numpy()
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            
            # 
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

