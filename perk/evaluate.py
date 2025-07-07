import re
import string

from typing import Dict
from dataclasses import dataclass

from perk.utils.io_utils import *

ANSWER_PREFIX = "Answer:"
REASON_PREFIX = "Support"
REASON_SEP = "**\n"

ANSWER_PATTERN = re.compile(r'<output>(.*?)</output>', re.DOTALL)
REASON_PATTERN = re.compile(r'<support>(.*?)</support>', re.DOTALL)

def normalize_text(text):
    """
    Removing articles and punctuation, and
    standardizing whitespace are all typical
    text processing steps.

    :param text: text to normalize
    """
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

@dataclass
class GenerationOutput:
    """Data class for a unified format of generation output"""
    guid: str
    prompt: str
    response: str
    answer: str
    metrics: Dict
    metadata: Dict

    @classmethod
    def from_output(cls, output: Dict):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        :rtype: GenerationOutput
        """
        guid = output["guid"]
        prompt = output["prompt"]
        response = output["output"]
        answer = output["answer"]
        metrics = {}
        metadata = {}

        return cls(
            guid=guid,
            prompt=prompt,
            response=response,
            answer=answer,
            metrics=metrics,
            metadata=metadata,
        )

    @property
    def to_dict(self):
        """Converts to a dictionary"""
        return {
            "guid": self.guid,
            "prompt": self.prompt,
            "response": self.response,
            "answer": self.answer,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def compute_exact_match(self, response, target):
        """
        Compute exact match between response and target

        :param response: response from the model
        :param target: target from the dataset
        :rtype: int
        """
        norm_response = normalize_text(response)
        norm_target = normalize_text(target)
        return int(norm_response == norm_target)

    def compute_f1(self, response, target):
        """
        Compute F1 score between response and target

        :param response: response from the model
        :param target: target from the dataset
        :rtype: float
        """
        pred_tokens = normalize_text(response).split()
        truth_tokens = normalize_text(target).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    def reason_ratio(self, reason_steps_gen, reason_steps_gold):
        """
        Compute the ratio of the reason steps generated

        :param reason_steps_gen: generated reasoning steps from the model
        :param reason_steps_gold: gold reasoning steps from the dataset
        :rtype: float
        """
        gen_set = set(reason_steps_gen)
        gold_set = set(reason_steps_gold)
        coverage = len(gen_set & gold_set)
        ratio = coverage / len(gold_set)
        return ratio, coverage

    def extract_reason_steps(self, response, target):
        """Extract the reason steps from the response & target"""
        reason_gold = target.split(REASON_PREFIX)[1].strip()
        reason_gen = response.split(REASON_PREFIX)[1].strip()
        reason_steps_gen = reason_gen.split(REASON_SEP)
        reason_steps_gold = reason_gold.split(REASON_SEP)
        return reason_steps_gen, reason_steps_gold

    def compute_metrics(self):
        """Returns an exact match accuracy for generation"""
        response = self.response.lower()
        target = self.answer.lower()
        try:
            pred = ANSWER_PATTERN.findall(response)[0].strip()
            gold = ANSWER_PATTERN.findall(target)[0].strip()
        except IndexError:
            pred = response
            gold = target

        self.metrics["em"] = self.compute_exact_match(pred, gold)
        self.metrics["f1"] = self.compute_f1(response, target)

def evaluate(records):
    generations = [GenerationOutput.from_output(record) for record in records]
    for generation in generations:
        generation.compute_metrics()
    accuracy = sum([gen.metrics["em"] for gen in generations]) / len(generations)
    f1 = sum([gen.metrics["f1"] for gen in generations]) / len(generations)
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
    }

    if "reason_ratio" in generations[0].metrics:
        reason_ratio = sum([gen.metrics["reason_ratio"] for gen in generations]) / len(generations)
        reason_coverage = sum([gen.metrics["reason_coverage"] for gen in generations]) / sum([gen.metrics["num_gen_stpes"] for gen in generations])
        metrics["reason_ratio"] = reason_ratio
        metrics["reason_coverage"] = reason_coverage
    return metrics, generations