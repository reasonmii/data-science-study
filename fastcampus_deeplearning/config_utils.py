import os
from omegaconf import DictConfig
import torch
from torch import optim
from torch_optimizer import RAdam
from torch_optimizer import AdamP
import pytorch_lightning as pl
import hydra
from hydra.core.config_store import ConfigStore
import wandb


def flatten_dict(
    input_dict: dict, separator='_', prefix=''
):
    """flattening dict,
    used in wandb log.
    """
    return {
        prefix + separator + k if prefix else k : v
        for kk, vv in input_dict.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(input_dict, dict) else {prefix: input_dict}


def register_config(configs_dict: dict) -> None:
    """hydra register configuration"""
    cs = ConfigStore.instance()
    for k, merged_cfg in configs_dict.items():
        cs.store(name=k, node=merged_cfg)


def configure_optimizers_from_cfg(cfg: DictConfig, model):
    optimizers = []
    schedulers = []
        
    # schedulers are optional but, if they're given, the below should be same.
    if len(cfg.opt.lr_schedulers) > 0:
        assert len(cfg.opt.lr_schedulers) == len(cfg.opt.optimizers)

    # setup optimizer
    for opt_cfg in cfg.opt.optimizers:
        if opt_cfg.name == "RAdam":
            optimizers.append(
                RAdam(model.parameters(), **opt_cfg.kwargs)
            )
        elif opt_cfg.name == "SGD":
            optimizers.append(
                SGD(model.parameters(), **opt_cfg.kwargs)
            )
        elif opt_cfg.name == "AdamP":
            optimizers.append(
                AdamP(model.parameters(), **opt_cfg.kwargs)
            )
        elif opt_cfg.name == "Adam":
            optimizers.append(
                Adam(model.parameters(), **opt_cfg.kwargs)
            )
        else:
            raise NotImplementedError(f"Not supported optimizer: {opt_cfg.name}")
    
    # setup lr scheduler
    for idx, lr_sch_cfg in enumerate(cfg.opt.lr_schedulers):
        if lr_sch_cfg.name is None or lr_sch_cfg.name == "":
            pass
        elif lr_sch_cfg.name == "LinearWarmupLR":
            schedulers.append(
                LinearWarmupLR(optimizers[idx], **lr_sch_cfg.kwargs)
            )
        else:
            raise NotImplementedError(f"Not supported lr_scheduler: {lr_sch_cfg.name}")
    
    return optimizers, schedulers


def configure_optimizer_element(
    opt_cfg: DictConfig, lr_sch_cfg: DictConfig, model
):
    optimizer = None
    scheduler = None
    # setup optimizer
    if opt_cfg.name == "RAdam":
        optimizer = RAdam(model.parameters(), **opt_cfg.kwargs)
    elif opt_cfg.name == "SGD":
        optimizer = SGD(model.parameters(), **opt_cfg.kwargs)
    elif opt_cfg.name == "AdamP":
        optimizer = AdamP(model.parameters(), **opt_cfg.kwargs)
    elif opt_cfg.name == "Adam":
        optimizer = Adam(model.parameters(), **opt_cfg.kwargs)
    else:
        raise NotImplementedError(f"Not supported optimizer: {opt_cfg.name}")

    # setup lr scheduler
    if lr_sch_cfg.name is None or lr_sch_cfg.name == "":
        pass
    elif lr_sch_cfg.name == "LinearWarmupLR":
        scheduler = LinearWarmupLR(optimizer, **lr_sch_cfg.kwargs)
    else:
        raise NotImplementedError(f"Not supported lr_scheduler: {lr_sch_cfg.name}")
    
    return optimizer, scheduler


# loggers
def get_loggers(cfg: DictConfig):
    logger = []
    loggers_cfg = cfg.log.loggers
    for name, kwargs_dict in loggers_cfg.items():
        if name == "WandbLogger":
            wandb.finish()
            os.makedirs(kwargs_dict.save_dir, exist_ok=True)
            logger.append(pl.loggers.WandbLogger(
                config=flatten_dict(cfg),
                # reinit=True,
                settings=wandb.Settings(start_method="thread"),
                **kwargs_dict,
            ))
        elif name == "TensorBoardLogger":
            logger.append(pl.loggers.TensorBoardLogger(
                **kwargs_dict
            ))
        else:
            raise NotImplementedError(f"invalid loggers_cfg name {name}")

    return logger

# callbacks
def get_callbacks(cfg: DictConfig):
    callbacks = []
    callbacks_cfg = cfg.log.callbacks

    for name, kwargs_dict in callbacks_cfg.items():
        if name == "ModelCheckpoint":
            callbacks.append(pl.callbacks.ModelCheckpoint(
                **kwargs_dict,
            ))
        elif name == "EarlyStopping":
            callbacks.append(pl.callbacks.EarlyStopping(
                **kwargs_dict
            ))
        else:
            raise NotImplementedError(f"invalid callbacks_cfg name {name}")

    return callbacks
