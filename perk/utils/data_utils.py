import torch
import torch.distributed as dist

from torch.utils.data import DataLoader

def batch_split(row):
    """Split a batch into multiple inner-loop batches

    :param row: a batch of data
    :rtype: list
    """
    inner_loader = DataLoader(row, batch_size=4, shuffle=False)
    splited = [inner_batch for inner_batch in inner_loader]
    return splited

def batch_aggregate(rb):
    """Aggregate a batch of data

    :param rb: a batch of data
    :rtype: dict
    """
    inputs, masks, types = rb[0], rb[1], rb[2]
    train_feature = {
        "input_ids": inputs,
        "attention_mask": masks,
        "token_type_ids": types,
        "evaluate": False,
    }
    return train_feature


def get_features_dist(batch, is_train=True):
    """Get features from batch

    :param batch: the target batch
    :param accumulate: whether to accumulate gradient
    :rtype: dict
    """
    print_out = batch["print_out"]
    dev_features = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"]
    }

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = 1, 0

    if world_size > 1 and is_train:
        micro_input_ids = batch["train_input_ids"].chunk(world_size, dim=0)
        micro_attention_mask = batch["train_attention_mask"].chunk(world_size, dim=0)
        micro_labels = batch["train_labels"].chunk(world_size, dim=0)

        local_input_ids = micro_input_ids[rank]
        local_attention_mask = micro_attention_mask[rank]
        local_labels = micro_labels
    else:
        local_input_ids = batch["train_input_ids"]
        local_attention_mask = batch["train_attention_mask"]
        local_labels = batch["train_labels"]

    local_input_ids = batch["train_input_ids"]
    local_attention_mask = batch["train_attention_mask"]
    local_labels = batch["train_labels"]

    train_features = {
        "input_ids": local_input_ids,
        "attention_mask": local_attention_mask,
        "labels": local_labels
    }

    return train_features, dev_features, print_out

def get_features(batch, packing=False, accumulation_steps=1):
    """Get features from batch

    :param batch: the target batch.
    :param packing: flag to indicate whether to pack the input tensors.
    :param accumulation_steps: number of groups to split the training tensors into.
           When accumulation_steps > 1, the training features will be divided
           into that many chunks.
    :rtype: tuple (train_features, dev_features, print_out)
    """
    print_out = batch["print_out"]
    dev_features = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"]
    }

    local_input_ids = batch["train_input_ids"]
    local_attention_mask = batch["train_attention_mask"]
    local_labels = batch["train_labels"]

    # If accumulation_steps > 1, split the training tensors into chunks.
    if accumulation_steps > 1:
        input_ids_chunks = torch.chunk(local_input_ids, accumulation_steps)
        attention_mask_chunks = torch.chunk(local_attention_mask, accumulation_steps)
        labels_chunks = torch.chunk(local_labels, accumulation_steps)
        train_features = [
            {"input_ids": ids, "attention_mask": mask, "labels": lbl}
            for ids, mask, lbl in zip(
                input_ids_chunks,
                attention_mask_chunks,
                labels_chunks)
        ]
    else:
        train_features = [{
            "input_ids": local_input_ids,
            "attention_mask": local_attention_mask,
            "labels": local_labels
        }]

    if packing:
        train_features_packed = []
        for train_batch in train_features:
            attn_mask = train_batch.pop("attention_mask")
            train_batch["input_ids"] = train_batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            train_batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            train_batch["labels"] = train_batch["labels"][attn_mask.bool()].unsqueeze(0)
            train_batch["labels"][train_batch["position_ids"] == 0] = -100
            flattened_position_ids = train_batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            train_batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(),
                        device=flattened_position_ids.device,
                        dtype=torch.int32
                    ),
                )
            )
            train_batch["cu_seq_lens_k"] = train_batch["cu_seq_lens_q"]
            train_batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            train_batch["max_length_q"] = train_batch["max_length_k"]
            train_features_packed.append(train_batch)
            train_features = train_features_packed

    return train_features, dev_features, print_out
