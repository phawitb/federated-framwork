import torch
from typing import Optional, Tuple
from datasets import load_dataset


def load_sst2_datasets(
    tokenizer,
    max_length: int = 256,
    avoid_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[object, object]:
    ds = load_dataset("glue", "sst2", cache_dir=cache_dir)

    def tok_fn(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

    map_kwargs = {
        "remove_columns": ["sentence", "idx"],
        "num_proc": None,
        "desc": "tokenize",
    }
    if avoid_cache:
        map_kwargs.update({
            "load_from_cache_file": False,
            "keep_in_memory": True,
            "writer_batch_size": None,
        })

    ds = ds.map(tok_fn, **map_kwargs)

    train_cols = set(ds["train"].column_names)
    if "labels" in train_cols and "label" in train_cols:
        ds = ds.remove_columns(["label"])
    elif "label" in train_cols and "labels" not in train_cols:
        ds = ds.rename_column("label", "labels")

    return ds["train"], ds["validation"]


def to_tensor_batch(tokenizer, features):
    keys = ["input_ids", "attention_mask", "labels"]
    batch = {k: [f[k] for f in features] for k in keys}
    batch_enc = tokenizer.pad(
        {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        padding=True,
        return_tensors="pt",
    )
    batch_enc["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
    return batch_enc


def collate_fn(tokenizer, batch):
    return to_tensor_batch(tokenizer, batch)
