import os
import logging
import pathlib

from lightning.pytorch.loggers import WandbLogger

util_logger = logging.getLogger('perk.utils.wandb_utils')

try:
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())
except NotImplementedError:
    pathlib.PosixPath = pathlib.WindowsPath
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())


def create_wandb_vars(config):
    """Creates special environment variables for trainers and other utilities
    to use if such configuration values are provided

    :param config: the global configuration values
    :raises: ValueError
    """
    if config.wandb_name:
        os.environ["WANDB_NAME"] = config.wandb_name
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity

    if config.wandb_name or config.wandb_project or config.wandb_entity:
        util_logger.info(
            'WANDB settings (options), name=%s, project=%s, entity=%s' %
            (config.wandb_name, config.wandb_project, config.wandb_entity))

def init_wandb_logger(config):
    """Initializes the wandb logger

    :param config: the global configuration
    """
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name
    )
    return wandb_logger