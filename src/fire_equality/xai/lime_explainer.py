"""
LIME (Local Interpretable Model-agnostic Explanations) explainer
forlocal model interpretability analysis
"""

import torch
import numpy as np
from typing import Optional, Callable, Any, List
import warnings

try:
    import lime
    from lime import lime_image, lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")


class LIMEExplainer:
    """
    LIME explainer
    
   ,
    """
    
    def __init__(self, model: torch.nn.Module, mode: str = "image"):
        """
        initializeLIME explainer
        
        Args:
            model: trained PyTorch model
            mode:  ("image"  "tabular")
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        self.model = model
        self.mode = mode
        self.explainer = None
        
    def create_image_explainer(self, **kwargs):
        """
        create an image LIME explainer
        
        Args:
            **kwargs: passed to lime_image.LimeImageExplainer
        """
        if self.mode != "image":
            raise ValueError("Mode must be 'image' to use image explainer")
        
        self.explainer = lime_image.LimeImageExplainer(**kwargs)
    
    def create_tabular_explainer(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        create a tabular LIME explainer
        
        Args:
            training_data: 
            feature_names: feature-name list
            **kwargs: passed to lime_tabular.LimeTabularExplainer
        """
        if self.mode != "tabular":
            raise ValueError("Mode must be 'tabular' to use tabular explainer")
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode="classification",
            **kwargs
        )
    
    def explain_instance_image(
        self,
        image: np.ndarray,
        top_labels: int = 5,
        hide_color: int = 0,
        num_features: int = 100000,
        num_samples: int = 1000
    ):
        """
        explain an image instance
        
        Args:
            image: 
            top_labels: 
            hide_color: 
            num_features: 
            num_samples: 
        
        Returns:
            LIME
        """
        if self.explainer is None:
            self.create_image_explainer()
        
        def model_predict(images):
            """model prediction function"""
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
                if isinstance(outputs, torch.Tensor):
                    outputs = torch.softmax(outputs, dim=1)
                return outputs.cpu().numpy()
        
        explanation = self.explainer.explain_instance(
            image,
            model_predict,
            top_labels=top_labels,
            hide_color=hide_color,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def explain_instance_tabular(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ):
        """
        explain a tabular instance
        
        Args:
            instance: 
            num_features: 
            num_samples: 
        
        Returns:
            LIME
        """
        if self.explainer is None:
            raise ValueError("Tabular explainer not created. Call create_tabular_explainer first.")
        
        def model_predict(instances):
            """model prediction function"""
            if isinstance(instances, np.ndarray):
                instances = torch.from_numpy(instances).float()
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(instances)
                if isinstance(outputs, torch.Tensor):
                    outputs = torch.softmax(outputs, dim=1)
                return outputs.cpu().numpy()
        
        explanation = self.explainer.explain_instance(
            instance,
            model_predict,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def get_explanation_as_list(self, explanation, label: int = 1):
        """
        return the explanation as a list
        
        Args:
            explanation: LIME
            label: 
        
        Returns:
            
        """
        if hasattr(explanation, 'as_list'):
            return explanation.as_list(label=label)
        else:
            return explanation.as_list()
    
    def visualize_explanation(self, explanation, image: Optional[np.ndarray] = None, label: int = 1):
        """
        visualize explanation results
        
        Args:
            explanation: LIME
            image: (for)
            label: 
        """
        try:
            if self.mode == "image" and image is not None:
                from skimage.segmentation import mark_boundaries
                import matplotlib.pyplot as plt
                
                temp, mask = explanation.get_image_and_mask(
                    label,
                    positive_only=True,
                    num_features=5,
                    hide_rest=True
                )
                img_boundry = mark_boundaries(temp / 2 + 0.5, mask)
                plt.imshow(img_boundry)
                plt.axis('off')
                plt.show()
            else:
                # tabular mode: print feature importance
                exp_list = self.get_explanation_as_list(explanation, label)
                print("Feature Importance:")
                for feature, importance in exp_list:
                    print(f"  {feature}: {importance:.4f}")
        except Exception as e:
            print(f"Visualization error: {e}")

