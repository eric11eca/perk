import gc
import torch
import higher
import logging
import warnings

from typing import Any, Dict
from torch.optim import AdamW
from transformers import PreTrainedModel
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2
)

from perk.model import CausalLM
from perk.modules.base_module import BaseModule
from perk.optimizer import LSLRSchedular
from perk.inference import LLM_Generator
from perk.utils.data_utils import get_features

from transformers import logging as hf_logging
hf_logging.set_verbosity_info()

logger = logging.getLogger("perk.meta_module")

def print_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(f"{prefix} => allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")

def print_gpu_stats(prefix: str, init_gpu_memory: int):
    if torch.cuda.is_available():
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        print(f"=================={prefix}=====================")
        print(f"Peak memory usage: {peak_memory / 1024**3:.2f} GB")
        print(f"Total memory usage: {total_gpu_memory / 1024**3:.2f} GB")
        print(f"Free memory: {free_gpu_memory / 1024**3:.2f} GB")

def grad_callback(grads):
    all_grads = []
    for g in grads:
        if g is not None:
            all_grads.append(g.detach())
        else:
            all_grads.append(g)
    return all_grads

class MetaLMModule(BaseModule):
    def __init__(self, config):
        super().__init__(config, logger)
        self.model = CausalLM.from_config(config)
        self.tokenizer = self.model.tokenizer
        self.load_dataset()
        self.inner_lr_schedular_config(
            config.n_inner_iter, config.inner_lr)

        self.inner_funct_types = {
            "truncated": self.truncated_inner_loop_step,
            "default": self.inner_loop_step
        }
        self.inner_funct = self.inner_funct_types[config.inner_funct]
        if not config.do_train and config.do_eval:
            self.inner_funct = self.inner_loop_step

        if config.use_liger_kernel:
            self.apply_liger_kernel()

        self.init_gpu_memory = None
        if torch.cuda.is_available():
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]

    def apply_liger_kernel(self):
        if "qwen" in self.hparams.model_name_or_path.lower():
            lm_model = self.model.lm_model
            if isinstance(lm_model, PreTrainedModel):
                apply_liger_kernel_to_qwen2(
                    rope=True,
                    swiglu=True,
                    cross_entropy=False,
                    fused_linear_cross_entropy=False,
                    rms_norm=True,
                    model=lm_model
                )
            elif hasattr(lm_model, "get_base_model") and isinstance(lm_model.get_base_model(), PreTrainedModel):
                apply_liger_kernel_to_qwen2(
                    rope=True,
                    swiglu=True,
                    cross_entropy=False,
                    fused_linear_cross_entropy=False,
                    rms_norm=True,
                    model=lm_model.get_base_model()
                )
            else:
                raise ValueError("Model is not a PreTrainedModel or does not have a base model")

    def inner_lr_schedular_config(self, n_inner_iter, inner_lr):
        self.inner_schedular = LSLRSchedular(
            num_inner_iter=n_inner_iter, init_lr=inner_lr)
        if self.hparams.peft_lora:
            params_opt = list(filter(
                lambda p: "lora" in p[0] and p[1].requires_grad,
                self.model.named_parameters()))
        else:
            params_opt = list(filter(
                lambda p: p[1].requires_grad,
                self.model.named_parameters()))
        self.inner_schedular.initialization(params_opt)

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        no_decay = ["bias", "LayerNorm.weight"]

        if self.hparams.peft_lora:
            trainable_parameters = [
                (n, p)
                for n, p in self.model.named_parameters()
                if p.requires_grad and "lora" in n
            ]
        else:
            trainable_parameters = [
                (n, p)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            ]

        for name, param in trainable_parameters:
            if not any(nd in name for nd in no_decay):
                model_params.append(
                    {
                        "name": name,
                        "params": param,
                        "lr": self.hparams.inner_lr,
                        "weight_decay": self.hparams.weight_decay
                    }
                )
            else:
                model_params.append(
                    {
                        "name": name,
                        "params": param,
                        "lr": self.hparams.inner_lr,
                        "weight_decay": 0.0
                    }
                )

        inner_opt = torch.optim.AdamW(
            model_params, betas=(0.9, 0.95))
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :rtype: torch.optim
        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p
            for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]

        parameters_sec = [
            p
            for n, p in self.model.named_parameters()
            if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {"params": parameters_first, "weight_decay": self.hparams.weight_decay},
            {"params": parameters_sec, "weight_decay": 0.0},
            {"params": self.inner_schedular.parameters(), "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            betas=(0.9, 0.95)
        )

        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def inner_loop_step(self, features, fmodel, diffopt):
        """Runs a single inner loop step

        :param features: the list of inner loop batch
        :param fmodel: the fast model for adaptation
        :param diffopt: the optimizer for adaptation
        :rtype: torch.Tensor
        :returns: the inner loss
        """
        call_back = grad_callback if self.hparams.first_order else None
        for iter in range(self.hparams.n_inner_iter):
            for idx, batch in enumerate(features):
                inner_loss = fmodel(batch, is_inner=True, inner_iter=iter)["loss"]
                inner_loss /= len(features)

                if idx == len(features) - 1 and self.hparams.dyna_lr:
                    if self.hparams.peft_lora:
                        named_params = list(filter(
                            lambda p: p[1].requires_grad and "lora" in p[0],
                            fmodel.named_parameters()))
                    else:
                        named_params = list(filter(
                            lambda p: p[1].requires_grad,
                            fmodel.named_parameters()))
                    self.inner_schedular.step(diffopt, named_params, iter)

                diffopt.step(inner_loss, grad_callback=call_back)
                torch.cuda.empty_cache()
        return inner_loss

    def truncated_inner_loop_step(self, features, fmodel, diffopt):
        """Runs a single inner loop step

        :param features: the list of inner loop batch
        :param fmodel: the fast model for adaptation
        :param diffopt: the optimizer for adaptation
        :rtype: torch.Tensor
        :returns: the inner loss
        """
        unroll_start = self.hparams.unroll_start
        for iter in range(self.hparams.n_inner_iter):
            for batch in features:
                inner_loss = fmodel(batch, is_inner=True, inner_iter=iter)["loss"]
                inner_loss /= len(features)

                if self.hparams.dyna_lr:
                    if self.hparams.peft_lora:
                        named_params = list(filter(
                            lambda p: p[1].requires_grad and "lora" in p[0],
                            fmodel.named_parameters()))
                    else:
                        named_params = list(filter(
                        lambda p: p[1].requires_grad,
                        fmodel.named_parameters()))
                    self.inner_schedular.step(diffopt, named_params, iter)

                if iter < unroll_start:
                    diffopt.step(inner_loss, grad_callback=grad_callback)
                else:
                    diffopt.step(inner_loss)
                torch.cuda.empty_cache()
        return inner_loss

    def step(self, batch):
        """Runs a single meta-training step

        :param batch: the target batch
        :rtype: dict
        :returns: dictionary that includes loss
        """
        inner_opt = self.config_inner_optimizer()
        inner_batches, outer_batches, _ = get_features(
            batch, packing=self.hparams.packing,
            accumulation_steps=1)

        self.model.train()
        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            accumulation_steps=1
        ) as (fmodel, diffopt):
            inner_loss = self.inner_funct(
                inner_batches,
                fmodel, diffopt)
            outer_loss = fmodel(outer_batches)["loss"]

        output_dict = {
            "loss": outer_loss,
            "inner_loss": inner_loss.detach(),
            "outer_loss": outer_loss.detach()
        }
        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        if not self.hparams.do_train:
            return None

        try:
            output_dict = self.step(batch)
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[warn] GPU ran OOM: {oom}")
            torch.cuda.empty_cache()
            gc.collect()
            return None

        for mkey in ["inner_loss", "outer_loss"]:
            self.log(
                f"batch_{mkey}",
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=False
            )
        self.global_trainin_step += 1
        return output_dict

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        torch.set_grad_enabled(True)
        inner_opt = self.config_inner_optimizer()

        packing = self.hparams.packing
        inner_grad_accum = self.hparams.inner_grad_accum

        inner_batches, outer_batches, print_out = get_features(
            batch, packing=packing,
            accumulation_steps=inner_grad_accum)

        self.model.train()
        if self.hparams.peft_lora:
            for n, p in self.model.lm_model.named_parameters():
                if "lora" not in n:
                    p.requires_grad = False

        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=False,
            accumulation_steps=inner_grad_accum
        ) as (fmodel, diffopt):
            inner_loss = self.inner_loop_step(
                inner_batches, fmodel, diffopt)
            records = []
            with torch.no_grad():
                fmodel.eval()
                outer_loss = fmodel(outer_batches)["loss"]
                for prompt, response in zip(
                    print_out["prompt"],
                    print_out["response"]):
                    records.append({
                        "guid": print_out["guid"],
                        "prompt": prompt,
                        "answer": response
                    })
                self.validation_inference(records, fmodel)

        output_dict = {
            "inner_loss": inner_loss.detach(),
            "outer_loss": outer_loss.detach(),
            "records": records
        }

        if self.local_rank == 0:
            print()
            print(f"Validation Loss: {output_dict['outer_loss'].item()} || Inner Loss: {output_dict['inner_loss'].item()}")

        self.validation_step_outputs.append(output_dict)
        return output_dict

    def validation_inference(self, records, model):
        generator = LLM_Generator(
            model=model.lm_model,
            tokenizer=model.tokenizer,
            device=self.device)

        print("\n\n")
        print("GUID: ", records[0]["guid"])
        print("PROMPT: ", records[0]["prompt"])
        print("ANSWER: ", records[0]["answer"])
        print("\n\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generator.generate(
                records,
                pad_token_id=model.tokenizer.pad_token_id,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=512
            )
            print("\n")
            print("PRED: ", records[0]["output"])
            print("\n")

    def validation_epoch_callback(self, outputs):
        val_loss = torch.stack(
            [x["outer_loss"].cpu() for x in outputs]
        ).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return val_loss, print_out_flatten

    def test_epoch_callback(self, outputs):
        test_loss = torch.stack(
            [x["outer_loss"].cpu() for x in outputs]
        ).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return test_loss, print_out_flatten

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).
        """
        if self.hparams.do_train:
            super().backward(loss, *args, **kwargs)
        else:
            pass