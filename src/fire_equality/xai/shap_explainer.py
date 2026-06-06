"""
SHAP (SHapley Additive exPlanations) explainer
for model
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
    SHAP explainer
    
    SHAPmodel,
    """
    
    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor):
        """
        initializeSHAP explainer
        
        Args:
            model: trained PyTorch model
            background_data: background dataset,forcompute SHAP values
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def create_explainer(self, explainer_type: str = "DeepExplainer"):
        """
        SHAP explainer
        
        Args:
            explainer_type: explainer ("DeepExplainer", "KernelExplainer", "LinearExplainer")
        """
        if explainer_type == "DeepExplainer":
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        elif explainer_type == "KernelExplainer":
            # model
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
        compute SHAP values
        
        Args:
            data: 
            max_evals: (forKernelExplainer)
        
        Returns:
            SHAP
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
        plot the SHAP summary plot
        
        Args:
            shap_values: SHAP
            feature_names: feature-name list
            **kwargs: passed to shap.summary_plotadditional arguments
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        shap.summary_plot(shap_values, feature_names=feature_names, **kwargs)
    
    def plot_waterfall(self, shap_values, feature_names: Optional[list] = None, **kwargs):
        """
        plot the SHAP waterfall plot
        
        Args:
            shap_values: SHAP
            feature_names: feature-name list
            **kwargs: passed to shap.waterfall_plotadditional arguments
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        shap.waterfall_plot(shap_values, feature_names=feature_names, **kwargs)

