"""

forTraining workflow
"""

import logging
import os
import gc
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

try:
    from pytorch_lightning.loggers import LightningLoggerBase
except ImportError:
    # newer PyTorch Lightning version
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """initializeGPUpython logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # rank_zero
    # GPU,GPU
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """,configuration:
    - 
    - 
    - configuration

    DictConfig.

    Args:
        config (DictConfig): Hydra configuration.
    """

    log = get_logger()

    # config
    OmegaConf.set_struct(config, False)

    # <config.ignore_warnings=True>python
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # <config.debug=True><config.trainer.fast_dev_run=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # <config.trainer.fast_dev_run=True>configuration
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # GPU
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """RichDictConfig.

    Args:
        config (DictConfig): Hydra configuration.
        fields (Sequence[str], optional): config.
        resolve (bool, optional): DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List,
) -> None:
    """Hydra configurationLightning loggers.

    :
        - model
    """

    hparams = {}

    # loggershydra config
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # model
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # hparamsloggers
    trainer.logger.log_hyperparams(hparams)

    # loggers
    # trainermodelhparams,
    # 
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List,
) -> None:
    """."""

    #,wandb loggersweeps!
    del datamodule
    del trainer
    del model
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
    gc.collect()

