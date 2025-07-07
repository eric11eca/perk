import os
import wandb
import logging

from omegaconf import OmegaConf

from perk.train import setup_trainer
from perk.utils.wandb_utils import setup_wandb
from perk.modules.meta_module import MetaLMModule

util_logger = logging.getLogger("perk.runner")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(args):
    util_logger.info("Setting up configuration for model runner...")

    setup_wandb(args)
    wandb.config = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True)

    module_class = MetaLMModule
    util_logger.info("Running MAML model")

    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint:
            assert args.checkpoint is not None
            util_logger.info(f"Loading model from checkpoint: {args.checkpoint}")
            model = module_class.load_from_checkpoint(
                args.checkpoint, config=args, map_location="cuda:0")
        else:
            util_logger.info("Creating new model")
            model = module_class(args)

        if args.resume_from_checkpoint:
            assert args.checkpoint is not None
            util_logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
            trainer.fit(model, ckpt_path=args.checkpoint)
        else:
            util_logger.info("Start training from scratch")
            trainer.fit(model)

    if args.do_eval:
        try:
            assert args.checkpoint is not None
        except AssertionError:
            util_logger.error("Checkpoint path is not provided for evaluation")

        # train on 1 datapoint such that gradient is enabled in validation for test-time learning
        model = module_class.load_from_checkpoint(
            args.checkpoint, config=args, map_location="cuda:0")
        trainer.fit(model)
