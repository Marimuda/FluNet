from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def validate_callbacks_cfg(callbacks_cfg: DictConfig) -> None:
    """Validates that the callbacks config is not None and is a DictConfig."""
    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")


def instantiate_callback(cb_conf: DictConfig) -> Optional[Callback]:
    """Instantiates a callback from its config if possible."""
    if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
        log.info(f"Instantiating callback <{cb_conf._target_}>")
        return hydra.utils.instantiate(cb_conf)

    return None


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    validate_callbacks_cfg(callbacks_cfg)

    callbacks: List[Callback] = []
    for _, cb_conf in callbacks_cfg.items():
        callback = instantiate_callback(cb_conf)
        if callback is not None:
            callbacks.append(callback)

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
