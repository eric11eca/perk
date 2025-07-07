import os
import logging
import lightning as pl

from pprint import pformat

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from perk.utils.wandb_utils import init_wandb_logger

util_logger = logging.getLogger("perk.trainer")
util_logger.setLevel(logging.INFO)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar

def setup_trainer(args) -> pl.Trainer:
    """Sets up the trainer and associated call backs from configuration

    :param configuration: the target configuration
    :rtype: a trainer instance
    """
    if "loss" in args.callback_monitor:
        mode = "min"
    else:
        mode = "max"
    util_logger.info("mode=%s via %s" % (mode, args.callback_monitor))

    if not os.path.isdir(args.output_dir):
        util_logger.info("making target directory: %s" % args.output_dir)
        os.mkdir(args.output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        monitor=args.callback_monitor,
        mode=mode,
        save_top_k=10,
        verbose=True,
        save_last=True,
        auto_insert_metric_name=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor=args.callback_monitor,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=mode,
    )

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="bold magenta",
            processing_speed="grey82",
            metrics="bold blue",
        )
    )

    progress_bar = LitProgressBar()
    callbacks = [lr_monitor, progress_bar, checkpoint_callback]
    if args.do_train:
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

    train_params = dict(
        accelerator="gpu",
        devices=args.n_gpu,
        max_epochs=args.num_train_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=4 if args.do_train else 0,
    )

    train_params["gradient_clip_val"] = args.max_grad_norm
    train_params["accumulate_grad_batches"] = args.gradient_accumulation_steps

    if args.bf16:
        train_params["precision"] = "bf16-mixed"
    else:
        train_params["precision"] = 32

    if args.n_gpu > 1:
        util_logger.info("using DDP mode for inner loop")
        train_params["strategy"] = "ddp"
        train_params["use_distributed_sampler"] = True

    if args.wandb_project:
        train_params["logger"] = init_wandb_logger(args)
        train_params["logger"].log_hyperparams(args)

    util_logger.info("\n===========\n" + pformat(train_params) + "\n===========")

    trainer = pl.Trainer(**train_params)

    return trainer
