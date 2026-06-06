"""
训练循环结构
基于PyTorch Lightning的训练流程
"""

from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
try:
    from pytorch_lightning.loggers import LightningLoggerBase
except ImportError:
    # 新版本的PyTorch Lightning
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase

# 修复PyTorch 2.6+的checkpoint加载问题
# PyTorch 2.6默认使用weights_only=True，但checkpoint中包含omegaconf和typing对象
# 解决方案：monkey patch PyTorch Lightning的checkpoint加载函数，使用weights_only=False

def patch_lightning_checkpoint_loading():
    """修补PyTorch Lightning的checkpoint加载，使用weights_only=False"""
    try:
        # 方法1: 修补lightning_fabric的_load函数
        import lightning_fabric.utilities.cloud_io as cloud_io
        
        # 保存原始函数（如果还没有保存）
        if not hasattr(cloud_io, '_original_load'):
            cloud_io._original_load = cloud_io._load
        
        def patched_load(f, map_location=None):
            """修补的加载函数，使用weights_only=False"""
            return torch.load(f, map_location=map_location, weights_only=False)
        
        # 替换原始函数
        cloud_io._load = patched_load
        
        # 方法2: 也修补pl_load函数（如果存在）
        if hasattr(cloud_io, 'pl_load'):
            if not hasattr(cloud_io, '_original_pl_load'):
                cloud_io._original_pl_load = cloud_io.pl_load
            
            def patched_pl_load(path, map_location=None):
                """修补的pl_load函数"""
                return torch.load(path, map_location=map_location, weights_only=False)
            
            cloud_io.pl_load = patched_pl_load
        
        # 方法3: 直接修补torch.load（更激进但更可靠）
        # 保存原始torch.load
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
        
        def patched_torch_load(f, map_location=None, **kwargs):
            """修补的torch.load，强制使用weights_only=False"""
            kwargs['weights_only'] = False
            return torch._original_load(f, map_location=map_location, **kwargs)
        
        torch.load = patched_torch_load
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"无法修补Lightning checkpoint加载: {e}")
        return False

# 在模块加载时修补
patch_lightning_checkpoint_loading()

try:
    from .utils import utils
except ImportError:
    # 如果作为独立模块运行
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fire_equality.utils import utils

log = utils.get_logger(__name__)


def extract_years_from_data_path(data_path) -> Optional[str]:
    """
    从数据文件路径中提取年份信息
    
    Args:
        data_path: 数据文件路径（字符串）或文件路径列表，例如：
                   'processed_firetracks_pixel_binary_2002-2004.pth' 或
                   ['processed_firetracks_pixel_binary_2002-2004.pth', ...]
    
    Returns:
        年份字符串，例如 '2002-2020'，如果无法提取则返回None
    """
    import re
    import os
    from omegaconf import ListConfig
    
    # 处理列表类型的数据路径
    if isinstance(data_path, (list, tuple, ListConfig)):
        if len(data_path) == 0:
            return None
        
        # 从所有文件中提取年份
        years = []
        for path in data_path:
            path_str = str(path)  # 转换为字符串
            filename = os.path.basename(path_str)
            # 提取年份范围（格式：2002-2004）
            match = re.search(r'(\d{4})-(\d{4})', filename)
            if match:
                years.append((int(match.group(1)), int(match.group(2))))
            else:
                # 尝试提取单个年份（格式：2002）
                match = re.search(r'_(\d{4})\.', filename)
                if match:
                    year = int(match.group(1))
                    years.append((year, year))
        
        if years:
            # 找到最小和最大年份
            min_year = min(y[0] for y in years)
            max_year = max(y[1] for y in years)
            return f"{min_year}-{max_year}"
        return None
    
    # 处理字符串类型的数据路径
    data_path_str = str(data_path)  # 确保是字符串
    filename = os.path.basename(data_path_str)
    
    # 从文件名中提取年份范围（格式：2002-2020）
    match = re.search(r'(\d{4})-(\d{4})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # 尝试提取单个年份（格式：2002）
    match = re.search(r'_(\d{4})\.', filename)
    if match:
        return match.group(1)
    
    return None


def train(config: DictConfig) -> Optional[float]:
    """
    训练流程
    
    从配置实例化所有PyTorch Lightning对象并执行训练。
    
    Args:
        config (DictConfig): 由Hydra组成的配置
    
    Returns:
        Optional[float]: 用于超参数优化的指标分数
    """

    # 设置随机数生成器的种子（pytorch, numpy, python.random）
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # 初始化lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # 初始化lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # 初始化lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 从数据文件路径中提取年份信息（在初始化logger之前）
    years = None
    if hasattr(datamodule, 'data_path'):
        years = extract_years_from_data_path(datamodule.data_path)
        if years:
            log.info(f"从数据文件路径提取到年份信息: {years}")
    
    # 初始化lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                # 如果提取到年份信息，更新logger配置中的name
                if years and 'name' in lg_conf:
                    try:
                        original_name = lg_conf['name']
                        # 尝试直接修改配置（Hydra配置通常是可写的）
                        lg_conf['name'] = f"{original_name}_{years}"
                        log.info(f"Logger名称已更新为: {lg_conf['name']} (年份: {years})")
                    except Exception as e:
                        log.warning(f"无法更新logger配置中的name，将使用原始名称: {e}")
                        # 如果修改配置失败，在实例化后手动更新logger
                        temp_logger = hydra.utils.instantiate(lg_conf)
                        if hasattr(temp_logger, 'name'):
                            temp_logger.name = f"{original_name}_{years}"
                        elif hasattr(temp_logger, '_name'):
                            temp_logger._name = f"{original_name}_{years}"
                        logger.append(temp_logger)
                        continue
                logger.append(hydra.utils.instantiate(lg_conf))

    # 初始化lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # 将一些参数从config发送到所有lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 训练模型
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # 在测试集上评估模型，使用训练期间获得的最佳模型
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing with best model!")
        # 确保checkpoint加载已修补
        patch_lightning_checkpoint_loading()
        trainer.test(datamodule=datamodule, ckpt_path="best")

    # 确保所有内容正确关闭
    log.info("Finalizing!")

    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 打印最佳检查点路径
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # 返回用于超参数优化的指标分数
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> Optional[float]:
    """主函数，使用Hydra装饰器"""
    return train(config)


if __name__ == "__main__":
    main()

