import torch
import logging

from typing import Optional
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)
from perk.utils.io_utils import *

from transformers import logging as hf_logging
hf_logging.set_verbosity_info()

util_logger = logging.getLogger("perk.inference")
util_logger.setLevel(logging.INFO)

class LLM_Generator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Optional[AutoTokenizer],
        device: str = "cuda:0"
    ) -> None:
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, queries, **kwargs):
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device)

        for query in queries:
            response = generator(
                query["prompt"],
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=kwargs["pad_token_id"],
                do_sample=kwargs["do_sample"],
                top_p=kwargs["top_p"],
                max_new_tokens=kwargs["max_new_tokens"],
            )
            query["output"] = response[0]["generated_text"]