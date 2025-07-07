from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import hydra
import random
import torch
import numpy as np

from typing import Optional
from omegaconf import OmegaConf, DictConfig, open_dict
from perk.runner import run

import logging
from rich.logging import RichHandler

from rich.traceback import install
install(show_locals=False)

import transformers
transformers.logging.set_verbosity_info()
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()

RICH_HANDLER_OPTIONS = {
    "level": "INFO",
    "show_time": True,
    "show_level": True,
    "show_path": True,
    "markup": True,
    "rich_tracebacks": True,
    "tracebacks_show_locals": True,
    "keywords": ["critical", "error", "warning", "info", "debug", "trace"]
}

def setup_logging(file_handler):
    """Configures logging to use RichHandler."""
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            file_handler,
            RichHandler(**RICH_HANDLER_OPTIONS)
        ],
        force=True
    )

    logging.captureWarnings(True)
    pywarnings_logger = logging.getLogger("py.warnings")
    pywarnings_logger.handlers = [RichHandler(**RICH_HANDLER_OPTIONS)]

@hydra.main(version_base="1.3", config_path="./configs/", config_name="run.yaml")
def main(args: DictConfig) -> Optional[float]:
    config_dict = OmegaConf.to_container(args, resolve=True)

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")
    log_file_handler = logging.FileHandler(
        os.path.join(args.output_dir, log_filename))
    setup_logging(log_file_handler)

    logger = logging.getLogger("perk.run_maml")
    logger.info("***** Running training/evaluation *****")
    logger.info(config_dict)

    install(show_locals=False)

    logger.info(f"Set output directory to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{args.output_dir}/{args.wandb_name}_{timestr}"
    gen_dir = f"{run_dir}/outputs"
    eval_dir = f"{run_dir}/evals"

    with open_dict(args):
        args.run_dir = run_dir
        args.gen_dir = gen_dir
        args.eval_dir = eval_dir

    args.wandb_name = f"{args.dataset_name}-{args.wandb_name}"

    if args.do_eval and not args.do_train:
        args.num_train_epochs = 1

    run(args)

if __name__ == "__main__":
    main()
