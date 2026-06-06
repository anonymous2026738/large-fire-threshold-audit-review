"""
Training loop structure
PyTorch Lightning training workflow
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
    # newer PyTorch Lightning version
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase

# FixPyTorch 2.6+checkpoint loading
# PyTorch 2.6 uses weights_only=True by default,checkpointomegaconftyping
# Solution:monkey patch PyTorch Lightningcheckpoint loading,weights_only=False

def patch_lightning_checkpoint_loading():
    """patch PyTorch Lightningcheckpoint loading,weights_only=False"""
    try:
        # 1: patch lightning_fabric_load
        import lightning_fabric.utilities.cloud_io as cloud_io
        
        # save the original function()
        if not hasattr(cloud_io, '_original_load'):
            cloud_io._original_load = cloud_io._load
        
        def patched_load(f, map_location=None):
            """patchloading,weights_only=False"""
            return torch.load(f, map_location=map_location, weights_only=False)
        
        # replace the original function
        cloud_io._load = patched_load
        
        # 2: patch pl_load()
        if hasattr(cloud_io, 'pl_load'):
            if not hasattr(cloud_io, '_original_pl_load'):
                cloud_io._original_pl_load = cloud_io.pl_load
            
            def patched_pl_load(path, map_location=None):
                """patch pl_load"""
                return torch.load(path, map_location=map_location, weights_only=False)
            
            cloud_io.pl_load = patched_pl_load
        
        # 3: patch torch.load(more aggressive but more reliable)
        # torch.load
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
        
        def patched_torch_load(f, map_location=None, **kwargs):
            """patch torch.loading,weights_only=False"""
            kwargs['weights_only'] = False
            return torch._original_load(f, map_location=map_location, **kwargs)
        
        torch.load = patched_torch_load
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"unable to patch Lightning checkpoint loading: {e}")
        return False

# loadpatch
patch_lightning_checkpoint_loading()

try:
    from .utils import utils
except ImportError:
    # 
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fire_equality.utils import utils

log = utils.get_logger(__name__)


def extract_years_from_data_path(data_path) -> Optional[str]:
    """
    Extract year information from data file paths
    
    Args:
        data_path: data file path(),:
                   'processed_firetracks_pixel_binary_2002-2004.pth' 
                   ['processed_firetracks_pixel_binary_2002-2004.pth', ...]
    
    Returns:
        year string, '2002-2020',return None if extraction fails
    """
    import re
    import os
    from omegaconf import ListConfig
    
    # handle list-valued data paths
    if isinstance(data_path, (list, tuple, ListConfig)):
        if len(data_path) == 0:
            return None
        
        # extract years from all files
        years = []
        for path in data_path:
            path_str = str(path)  # convert to string
            filename = os.path.basename(path_str)
            # extract year range(:2002-2004)
            match = re.search(r'(\d{4})-(\d{4})', filename)
            if match:
                years.append((int(match.group(1)), int(match.group(2))))
            else:
                # try extracting a single year(:2002)
                match = re.search(r'_(\d{4})\.', filename)
                if match:
                    year = int(match.group(1))
                    years.append((year, year))
        
        if years:
            # find the minimum and maximum years
            min_year = min(y[0] for y in years)
            max_year = max(y[1] for y in years)
            return f"{min_year}-{max_year}"
        return None
    
    # handle string-valued data path
    data_path_str = str(data_path)  # ensure this is a string
    filename = os.path.basename(data_path_str)
    
    # extract year range(:2002-2020)
    match = re.search(r'(\d{4})-(\d{4})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # try extracting a single year(:2002)
    match = re.search(r'_(\d{4})\.', filename)
    if match:
        return match.group(1)
    
    return None


def train(config: DictConfig) -> Optional[float]:
    """
    Training workflow
    
    configurationPyTorch Lightning.
    
    Args:
        config (DictConfig): Hydra configuration
    
    Returns:
        Optional[float]: for
    """

    # set random seeds(pytorch, numpy, python.random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # initializelightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # initializelightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # initializelightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Extract year information from data file paths(initializelogger)
    years = None
    if hasattr(datamodule, 'data_path'):
        years = extract_years_from_data_path(datamodule.data_path)
        if years:
            log.info(f"data file path: {years}")
    
    # initializelightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                #,logger configurationname
                if years and 'name' in lg_conf:
                    try:
                        original_name = lg_conf['name']
                        # configuration (Hydra configuration)
                        lg_conf['name'] = f"{original_name}_{years}"
                        log.info(f"Logger: {lg_conf['name']} (: {years})")
                    except Exception as e:
                        log.warning(f"logger configurationname,: {e}")
                        # configuration,logger
                        temp_logger = hydra.utils.instantiate(lg_conf)
                        if hasattr(temp_logger, 'name'):
                            temp_logger.name = f"{original_name}_{years}"
                        elif hasattr(temp_logger, '_name'):
                            temp_logger._name = f"{original_name}_{years}"
                        logger.append(temp_logger)
                        continue
                logger.append(hydra.utils.instantiate(lg_conf))

    # initializelightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # configlightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # model,model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing with best model!")
        # checkpoint loadingpatch
        patch_lightning_checkpoint_loading()
        trainer.test(datamodule=datamodule, ckpt_path="best")

    # 
    log.info("Finalizing!")

    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # for
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> Optional[float]:
    """,Hydra"""
    return train(config)


if __name__ == "__main__":
    main()

