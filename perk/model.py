import torch
import logging
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

logger = logging.getLogger("perk.model")

class WeightingModel(nn.Module):
    def __init__(self, normalize=False, non_linearity="gelu", embed_dim=768):
        super().__init__()
        self.normalize = normalize
        self.non_linearities = {
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'softplus': nn.Softplus(beta=3),
            'gelu': nn.GELU()
        }
        self.non_linearity = non_linearity
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        with torch.no_grad():
            self.fc2.weight.fill_(0)
            self.fc2.bias.fill_(0)

    def trainable_params(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the weighting model

        :param x: the 'last_hidden_state' from a language model
        :param attention_mask: the attention mask
        :return: the weighted sum of the input tensor
        """
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[1], device = x.device)
        nl = self.non_linearities[self.non_linearity]
        gelu = nn.GELU()
        x = gelu(self.fc1(x))
        x = self.fc2(x).squeeze()
        x = nl(x)
        x = (x * attention_mask)
        if self.normalize:
            x = x / (
                x.sum(1).unsqueeze(1)
            ) * (
                attention_mask.sum(1).unsqueeze(1)
            )
        return x

class CausalLM(nn.Module):
    """
    Generic transformer-based autoregressive language model (e.g., GPT-2, GPT-J, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation.
    """

    def __init__(self, model, tokenizer, weighting_model, model_config, global_config):
        super().__init__()
        self.lm_model = model
        self.weighting_model = weighting_model
        self.tokenizer = tokenizer
        self.config = model_config
        self.global_config = global_config

    @staticmethod
    def freeze_params(model, range: List[int]):
        for name, param in model.named_parameters():
            if np.any([x in name for x in range]):
                print(f"Freezing {name}")
                param.requires_grad = False

    @classmethod
    def from_config(cls, config):
        """
        Loads a pretrained decoder-only causal LLM from configuration

        :param config: the global configuration
        :rtype CausalLM
        """
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            padding_side="left")
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype="auto",
            attn_implementation=config.attn_implementation)
        model.train()

        if model_config.model_type == "gpt2":
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
        elif config.model_type == "llama":
            tokenizer.pad_token_id = 128004
            model.config.pad_token_id = 128004

        if config.peft_lora:
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj",
                "gate_proj", "up_proj", "down_proj",
                "c_attn", "c_proj", "c_fc"]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                use_rslora=True,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1)
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=True)

            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info("PEFT model loaded")
            logger.info(f"trainable parameters: {trainable_params:,d} || all params: {all_param:,d}")
            logger.info(f"trainable%: {100 * trainable_params / all_param:.4f}")

        if config.weighted_loss:
            weighting_model = WeightingModel(
                normalize=config.weight_normalize,
                non_linearity=config.weight_non_linearity,
                embed_dim=model.config.hidden_size)
        else:
            weighting_model = None

        return cls(
            model,
            tokenizer,
            weighting_model,
            model_config,
            config,
        )

    def weighted_loss(self, outputs, features):
        weights = self.weighting_model(
            outputs["hidden_states"][-1],
            features.get(
                "attention_mask",
                torch.ones_like(features["input_ids"]))
        )

        logits = outputs["logits"]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100, reduction='none')
        batch_size = len(features["input_ids"])
        reshaped_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        reshaped_labels = features["labels"][:, 1:].reshape(-1)

        loss = loss_fn(reshaped_logits, reshaped_labels)
        loss = (loss.reshape(batch_size, -1) * weights[:, 1:]).mean()
        return loss

    def forward(self, features, is_inner=False, inner_iter=0):
        """LM forward pass with optional weighted cross-entropy loss
        :param features: the target inputs
        :param is_inner: whether the forward is called in the inner loop
        :param inner_iter: the current iteration in the inner loop
        """
        outputs = self.lm_model(
            **features,
            return_dict=True,
            output_hidden_states=True)

        if self.global_config.weighted_loss and is_inner:
            outputs["loss"] = self.weighted_loss(outputs, features)
        return outputs