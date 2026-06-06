"""
LIME (Local Interpretable Model-agnostic Explanations) 解释器
用于局部模型可解释性分析
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
    LIME解释器类
    
    提供局部可解释性分析，解释单个预测
    """
    
    def __init__(self, model: torch.nn.Module, mode: str = "image"):
        """
        初始化LIME解释器
        
        Args:
            model: 训练好的PyTorch模型
            mode: 模式 ("image" 或 "tabular")
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        self.model = model
        self.mode = mode
        self.explainer = None
        
    def create_image_explainer(self, **kwargs):
        """
        创建图像LIME解释器
        
        Args:
            **kwargs: 传递给lime_image.LimeImageExplainer的参数
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
        创建表格LIME解释器
        
        Args:
            training_data: 训练数据
            feature_names: 特征名称列表
            **kwargs: 传递给lime_tabular.LimeTabularExplainer的参数
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
        解释图像实例
        
        Args:
            image: 输入图像
            top_labels: 要解释的顶级标签数量
            hide_color: 隐藏颜色值
            num_features: 特征数量
            num_samples: 采样数量
        
        Returns:
            LIME解释结果
        """
        if self.explainer is None:
            self.create_image_explainer()
        
        def model_predict(images):
            """模型预测函数"""
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
        解释表格实例
        
        Args:
            instance: 输入实例
            num_features: 要解释的特征数量
            num_samples: 采样数量
        
        Returns:
            LIME解释结果
        """
        if self.explainer is None:
            raise ValueError("Tabular explainer not created. Call create_tabular_explainer first.")
        
        def model_predict(instances):
            """模型预测函数"""
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
        获取解释结果作为列表
        
        Args:
            explanation: LIME解释结果
            label: 要解释的标签
        
        Returns:
            特征重要性列表
        """
        if hasattr(explanation, 'as_list'):
            return explanation.as_list(label=label)
        else:
            return explanation.as_list()
    
    def visualize_explanation(self, explanation, image: Optional[np.ndarray] = None, label: int = 1):
        """
        可视化解释结果
        
        Args:
            explanation: LIME解释结果
            image: 原始图像（用于图像模式）
            label: 要可视化的标签
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
                # 表格模式：打印特征重要性
                exp_list = self.get_explanation_as_list(explanation, label)
                print("Feature Importance:")
                for feature, importance in exp_list:
                    print(f"  {feature}: {importance:.4f}")
        except Exception as e:
            print(f"Visualization error: {e}")

