import os
import yaml
import torch
import warnings
import lightning as pl

from typing import Dict
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup
from lightning.pytorch.utilities import grad_norm

from perk.evaluate import evaluate
from perk.utils.io_utils import write_generations, write_metrics
from perk.dataset import MetaKnowledgeDataset, create_dataloader

warnings.filterwarnings("ignore")

class BaseModule(pl.LightningModule):
    def __init__(self, config, logger):
        """Creates  runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model_logger = logger
        self.hparams.update(
            OmegaConf.to_container(config))
        self.global_trainin_step = 0
        self.global_epoch_counter = 0
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def load_dataset(self):
        self.model_logger.info("Loading train, dev, and test dataset")
        self.train_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="train",
            is_training=True)
        self.dev_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="validation",
            is_training=False)
        self.test_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="test",
            is_training=False)
        self.model_logger.info("Train, Dev, and Test datasets loaded")

    def train_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.train_data, is_training=True)
        self.model_logger.info("Train dataloader length: %d" % len(dataloader))
        return dataloader

    def val_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.dev_data, is_training=False)
        self.model_logger.info("Validation dataloader length: %d" % len(dataloader))
        return dataloader

    def test_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.test_data, is_training=False)
        self.model_logger.info("Test dataloader length: %d" % len(dataloader))
        return dataloader

    def on_train_epoch_end(self):
        self.global_epoch_counter += 1

    def validation_epoch_callback(self, outputs):
        raise NotImplementedError

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler"""
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.gradient_accumulation_steps
            * num_devices
        )

        total_steps = (
            len(self.train_dataloader().dataset) / effective_batch_size
        ) * self.hparams.num_train_epochs
        self.hparams.warmup_steps = (
            total_steps / effective_batch_size
        ) * self.hparams.warmup_proportion

        self.hparams.warmup_steps = min(self.hparams.warmup_steps, 8000)

        self.model_logger.info(
            "total_steps computed for scheduler: %s, warmup step: %s"
            % (total_steps, str(self.hparams.warmup_steps))
        )

        scheduler = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch"""
        outputs = self.validation_step_outputs

        if len(outputs) == 0:
            if self.hparams.multi_task:
                self.log(f"val_acc_label", 0.50, on_epoch=True, prog_bar=False)
            else:
                self.log(f"val_acc", 0.50, on_epoch=True, prog_bar=False)
            return

        val_loss, generations = self.validation_epoch_callback(outputs)

        self.log(
            "val_loss", val_loss.to(self.device),
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True)

        generation_rows = []
        for record in generations:
            generation_rows.append([
                record["guid"],
                record["prompt"],
                record["answer"],
                record["output"]
            ])

        columns = ["guid", "prompt", "answer", "output"]
        step = self.global_trainin_step
        epoch = self.global_epoch_counter
        self.logger.log_table(
            f"epoch={epoch}-step={step}-generations",
            columns=columns,
            data=generation_rows)

        metrics = {}
        metrics = self.validation_eval(generations)
        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        for key, val in metrics.items():
            self.log(
                f"val_{key}",
                torch.tensor(val).to(self.device),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(test_out)
        return test_out

    def test_epoch_callback(self, outputs):
        raise NotImplementedError

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        val_loss, generations = self.test_epoch_logic(outputs)
        self.log("test_loss", val_loss, on_epoch=True, prog_bar=True)

        metrics = {}
        metrics = self.validation_eval(generations)
        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        self.validation_step_outputs.clear()
        for key, val in metrics.items():
            self.log(
                f"test_{key}", val,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)

    def validation_eval(self, results):
        metrics, _ = evaluate(results)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            os.makedirs(self.hparams.run_dir, exist_ok=True)
            os.makedirs(self.hparams.gen_dir, exist_ok=True)
            os.makedirs(self.hparams.eval_dir, exist_ok=True)

            config_name = os.path.join(self.hparams.run_dir, "config.yaml")
            if not os.path.exists(config_name):
                with open(config_name, "w") as f:
                    yaml.dump(self.hparams, f)
            log_name = f"outputs_{self.global_trainin_step}.jsonl"
            write_generations(results, self.hparams.gen_dir, log_name)
            log_name = f"eval_{self.global_trainin_step}.jsonl"
            write_metrics(metrics, self.hparams.eval_dir, log_name)
        elif not torch.distributed.is_initialized():
            os.makedirs(self.hparams.run_dir, exist_ok=True)
            os.makedirs(self.hparams.gen_dir, exist_ok=True)
            os.makedirs(self.hparams.eval_dir, exist_ok=True)

            config_name = os.path.join(self.hparams.run_dir, "config.yaml")
            if not os.path.exists(config_name):
                with open(config_name, "w") as f:
                    yaml.dump(self.hparams, f)
            log_name = f"outputs_{self.global_trainin_step}.jsonl"
            write_generations(results, self.hparams.gen_dir, log_name)
            log_name = f"eval_{self.global_trainin_step}.jsonl"
            write_metrics(metrics, self.hparams.eval_dir, log_name)
        return metrics
