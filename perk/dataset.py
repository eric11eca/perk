import os
import torch
import random

from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from datasets import load_dataset

from perk.utils.io_utils import read_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dict_to_sequence(record):
    return f"Student <{record['Student Id']}>, Name: {record['Student Name']}, Year: {record['Year']}, School: {record['School']}, Major: {record['Major']}, Grade: {record['Grade']}"

class DataReader:
    """Custom dataset loader for prompt-response pairs."""
    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        NotImplemented

    @classmethod
    def jsonl_file_reader(cls, path, config, tokenizer=None, split="train"):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param config: the configuration
        :param tokenizer: the model tokenizer
        """
        total_data = read_jsonl(path)
        if "context" in total_data[0]:
            all_facts = [
                fact for data in total_data for fact in data["context"]]
        elif "facts" in total_data[0]:
            all_facts = [
                fact for data in total_data for fact in data["facts"]]
        else:
            raise ValueError("No context or facts found in the data")

        dataset = []
        for instance in total_data:
            data = cls._read(instance, config)

            if config.random_facts:
                num_facts = len(data[0]["context"])
                data["context"] = random.choices(all_facts, k=num_facts)
                num_facts = len(data[0]["context"])
                data["context"] = random.choices(all_facts, k=num_facts)

            dataset.append(data)
        return dataset

class StudentRecordDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args, tokenizer):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :param tokenizer: the model tokenizer
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        problems = instance["questions"]
        context = [
            dict_to_sequence(record)
            for record in instance["database"]]

        examples = []
        for p in problems:
            q, a, s = p['question'], p['answer'], p.get('support', "")
            examples.append({
                "guid": guid,
                "context": context,
                "qa_pairs": [[q, str(a), s]]
            })
        return examples

    @classmethod
    def jsonl_file_reader(cls, path, args, tokenizer=None, split="Train"):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param args: the configuration of the experiment
        :param tokenizer: the model tokenizer
        """
        dir_path = path.replace(".jsonl", "")
        context_path = os.path.join(dir_path, "context.jsonl")
        context_data = read_jsonl(context_path)

        recall_path = os.path.join(dir_path, "recall.jsonl")
        relation_path = os.path.join(dir_path, "relation.jsonl")
        count_path = os.path.join(dir_path, "count.jsonl")
        aggregate_path = os.path.join(dir_path, "aggregate.jsonl")
        sort_path = os.path.join(dir_path, "sort.jsonl")

        if args.student_record_query == "recall":
            dataset = read_jsonl(recall_path)
        elif args.student_record_query == "relation":
            dataset = read_jsonl(relation_path)
        elif args.student_record_query == "count":
            dataset = read_jsonl(count_path)
        elif args.student_record_query == "aggregate":
            dataset = read_jsonl(aggregate_path)
        elif args.student_record_query == "sort":
            dataset = read_jsonl(sort_path)
        else:
            raise ValueError("Invalid student record query")

        total_data = [{
            "guid": db["database_id"],
            "database": db['database'],
            "questions": data["questions"]
        } for db, data in zip(context_data, dataset)]

        dataset = []
        for instance in tqdm(total_data):
            dataset.extend(cls._read(instance, args, tokenizer))

        return dataset

class BabiLongDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(idx, instance, args, tokenizer=None):
        guid = f"{args.dataset_subset}_{idx}"
        context = instance["context"]
        qa_pairs = [[
            instance["question"],
            instance["answer"],
            "\n".join(instance["support"])
        ]]
        return {"guid": guid, "qa_pairs": qa_pairs, "context": context}

    @classmethod
    def jsonl_file_reader(cls, path, args, tokenizer=None, split="train"):
        data_rows = read_jsonl(path)
        dataset = []
        for idx, instance in enumerate(data_rows):
            dataset.append(cls._read(idx, instance, args, tokenizer))
        return dataset

class LongGorillaDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(idx, instance, args):
        guid = f"{args.dataset_subset}_{idx}"
        context = instance["context"]
        problems = [{
            "question": instance["prompt"],
            "answer": instance["api_call"],
            "support": instance.get("support", "")
        }]

        qa_pairs = []
        for p in problems:
            q, a = p['question'], p['answer']
            qa_pairs.append([q, a, p.get("support", "")])

        return {"guid": guid, "qa_pairs": qa_pairs, "context": context}

    @classmethod
    def jsonl_file_reader(cls, path, args, tokenizer=None, split="train"):
        dataset = []
        subset = args.dataset_subset
        data_rows = load_dataset(
            "<DATASET_NAME_OR_PATH>", subset,
            split="train" if "train" in path else "test")
        for idx, instance in enumerate(data_rows):
            dataset.append(cls._read(idx, instance, args))
        return dataset

class MetaKnowledgeDataset(Dataset):
    def __init__(self, args, tokenizer, data_path, data_type, is_training):
        self.args = args

        if args.dataset_subset != "none":
            self.task = f"{args.dataset_name}/{args.dataset_subset}"
        else:
            self.task = args.dataset_name

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.data_path = data_path
        self.data_type = data_type

        reader_classes = {
            "student_records": StudentRecordDataReader,
            "babilong": BabiLongDataReader,
            "longgorilla": LongGorillaDataReader,
        }

        self.reader = reader_classes[args.dataset_name]
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.data_rows = self.read_data_from_file()

        if not self.is_training and args.max_eval_data > 0:
            num_samples = min(args.max_eval_data, len(self.data_rows))
            self.data_rows = random.choices(self.data_rows, k=num_samples)

        if self.is_training and args.do_eval:
            try:
                self.data_rows = self.data_rows.select(range(1))
            except:
                self.data_rows = self.data_rows[:1]

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, index):
        batch = self.data_rows[index]
        return self.causal_lm_collator(batch)

    def read_data_from_file(self) -> list:
        file_path = f"{self.data_path}/{self.task}/{self.data_type}.jsonl"
        file_data = self.reader.jsonl_file_reader(
            file_path, self.args, self.tokenizer, self.data_type)
        return file_data

    def compute_seq_length(self, prompt, response):
        input_text = f"{prompt} {response}{self.tokenizer.eos_token}"
        tokenized_input = self.tokenizer.encode(input_text, add_special_tokens=True)
        return len(tokenized_input)

    def encode_with_prompt_completion_format(self, prompt, completion, max_seq_length):
        """
        Here we assume each example has 'prompt' and 'completion' fields.
        We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
        and it doesn't make sense to follow directly with the completion.
        """
        example_text = f"{prompt} {completion}{self.tokenizer.eos_token}"
        tokenized_example = self.tokenizer(
            example_text,
            return_tensors='pt',
            max_length=max_seq_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = tokenized_example.input_ids
        attention_mask = tokenized_example.attention_mask
        labels = input_ids.clone()
        tokenized_prompt = self.tokenizer(
            prompt, return_tensors='pt',
            max_length=max_seq_length,
            truncation=True
        )
        num_padded_tokens = torch.sum(attention_mask == 0).item()
        prompt_start_idx = 0
        prompt_end_idx = num_padded_tokens + len(tokenized_prompt.input_ids[0])

        labels[:, prompt_start_idx:prompt_end_idx] = -100
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }

    def causal_lm_collator(self, qa_data):
        """Batch collator for this custom class

        :param qa_data: the dict of a reasoning problem with context, question, and answer
        """
        train_input_ids_batch = []
        train_attention_mask_batch = []
        train_labels_batch = []
        train_inputs, train_outputs = [], []
        max_seq_length = 0
        sequences = []
        context = qa_data["context"]

        doc_ids = [i for i in range(len(context))]
        if self.data_type == "train":
            random.shuffle(doc_ids)
        for idx, fact in zip(doc_ids, context):
            sequences.append((f"doc_{idx}:", fact))
            max_seq_length = max(
                max_seq_length, self.compute_seq_length(
                    f"doc_{idx}:", fact))

        max_seq_length = min(max_seq_length, self.tokenizer.model_max_length)
        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt, response,
                max_seq_length=max_seq_length)
            train_input_ids_batch.append(encoded['input_ids'])
            train_attention_mask_batch.append(encoded['attention_mask'])
            train_labels_batch.append(encoded['labels'])
            train_inputs.append(prompt)
            train_outputs.append(response)

        dev_input_ids_batch = []
        dev_attention_mask_batch = []
        dev_labels_batch = []
        dev_inputs, dev_outputs = [], []
        sequences = []
        max_seq_length = 0

        prefix = f"You have memorized {len(context)} documents."
        for qa_pair in qa_data["qa_pairs"]:
            prompt = f"{prefix}\n{qa_pair[0]}\nAnswer:"
            if qa_pair[2]:
                response = f"<support>{qa_pair[2]}</support>\n<output>{qa_pair[1]}</output>"
            else:
                response = f"{qa_pair[1]}"
            sequences.append((prompt, response))
            max_seq_length = max(
                max_seq_length,
                self.compute_seq_length(
                    prompt, response))

        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt, response,
                max_seq_length=max_seq_length)
            dev_input_ids_batch.append(encoded['input_ids'])
            dev_attention_mask_batch.append(encoded['attention_mask'])
            dev_labels_batch.append(encoded['labels'])
            dev_inputs.append(prompt)
            dev_outputs.append(str(response))

        feature = {
            "input_ids": torch.stack(dev_input_ids_batch, dim=0),
            "attention_mask": torch.stack(dev_attention_mask_batch, dim=0),
            "labels": torch.stack(dev_labels_batch, dim=0),
            "train_input_ids": torch.stack(train_input_ids_batch, dim=0),
            "train_attention_mask": torch.stack(train_attention_mask_batch, dim=0),
            "train_labels": torch.stack(train_labels_batch, dim=0),
            "print_out": {"guid": qa_data["guid"]},
        }

        if not self.is_training:
            feature["print_out"].update({
                    "prompt": dev_inputs,
                    "response": dev_outputs
                }
            )
            feature["inner_print_out"] = {
                "guid": qa_data["guid"],
                "prompt": train_inputs,
                "response": train_outputs
            }
        return feature

def create_dataloader(args, dataset, is_training):
    if is_training:
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=unroll
    )

def unroll(batch):
   return batch[0]
