"""
工具函数
用于训练流程的辅助函数
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
    # 新版本的PyTorch Lightning
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """初始化多GPU友好的python logger。"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 这确保所有日志级别都标记了rank_zero装饰器
    # 否则在多GPU设置中，每个GPU进程的日志会成倍增加
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """一些可选工具，由主配置文件控制：
    - 禁用警告
    - 更容易访问调试模式
    - 强制调试友好配置

    就地修改DictConfig。

    Args:
        config (DictConfig): 由Hydra组成的配置。
    """

    log = get_logger()

    # 启用向config添加新键
    OmegaConf.set_struct(config, False)

    # 如果<config.ignore_warnings=True>则禁用python警告
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # 如果<config.debug=True>则设置<config.trainer.fast_dev_run=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # 如果<config.trainer.fast_dev_run=True>则强制调试器友好配置
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # 调试器不喜欢GPU或多处理
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # 禁用向config添加新键
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
    """使用Rich库及其树结构打印DictConfig的内容。

    Args:
        config (DictConfig): 由Hydra组成的配置。
        fields (Sequence[str], optional): 确定将从config打印哪些主字段以及顺序。
        resolve (bool, optional): 是否解析DictConfig的引用字段。
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
    """此方法控制哪些Hydra配置参数由Lightning loggers保存。

    另外保存：
        - 可训练模型参数的数量
    """

    hparams = {}

    # 选择将保存到loggers的hydra config部分
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # 保存模型参数数量
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # 将hparams发送到所有loggers
    trainer.logger.log_hyperparams(hparams)

    # 禁用为所有loggers记录更多超参数
    # 这只是防止trainer记录模型hparams的技巧，
    # 因为我们已经在上面的做了
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List,
) -> None:
    """确保所有内容正确关闭。"""

    # 没有这个，使用wandb logger的sweeps可能会崩溃！
    del datamodule
    del trainer
    del model
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
    gc.collect()

