import itertools

import datasets
import torch
from torch.utils.data import DataLoader

from myllm import config


def _pack_sequence_batch(batch: dict, doc_indices, seq_len: int, eos_token_id: int):
    """packs the sequence into fixed length
    - tokens from single document may get split into different sequence
    - tokens from multiple document may belong to same sequence
    - at the end of a document, eos_token_id is appended
    - each output sequence exactly has seq_len tokens.
        if any sequence has less than that, then it is dropped
    - doc_ids indicate which document the tokens belong to
    """
    token_ids_list = batch["token_ids"]
    token_ids_list = [(token_ids + [eos_token_id]) for token_ids in token_ids_list]

    all_input_ids = list(itertools.chain.from_iterable(token_ids_list))
    all_doc_ids = list(
        itertools.chain.from_iterable(
            itertools.repeat(i, len(ids)) for i, ids in zip(doc_indices, token_ids_list)
        )
    )

    def has_exact_seq_len(batch: list):
        return len(batch) == seq_len

    return {
        "token_ids": list(
            filter(has_exact_seq_len, itertools.batched(all_input_ids, seq_len))  # type: ignore
        ),
        "doc_ids": list(
            filter(has_exact_seq_len, itertools.batched(all_doc_ids, seq_len))  # type: ignore
        ),
    }


def pack_dataset(ds, seq_len: int, eos_token_id: int):
    return ds.map(
        _pack_sequence_batch,
        batched=True,
        batch_size=1000,
        with_indices=True,
        fn_kwargs={"seq_len": seq_len, "eos_token_id": eos_token_id},
    )


def pretrain_collator(batch: list[dict]):
    token_ids_list = []
    doc_ids_list = []
    for row in batch:
        token_ids_list.append(torch.tensor(row["token_ids"], dtype=torch.long))
        doc_ids_list.append(torch.tensor(row["doc_ids"], dtype=torch.long))
    output = {
        "token_ids": torch.stack(token_ids_list, dim=0),
        "doc_ids": torch.stack(doc_ids_list, dim=0),
    }

    return output


def get_pretraining_dataset(seq_len: int, eos_token_id: int, dev=True):
    data_path = config.PRETRAIN_DATA_DIR / ("dev" if dev else "")
    data_files = str(data_path / "*.parquet")
    ds = datasets.load_dataset(
        "parquet", data_files=data_files, split="train", streaming=True
    )
    packed_ds = pack_dataset(ds=ds, seq_len=seq_len, eos_token_id=eos_token_id)
    return packed_ds


def get_pretraining_data_loader(packed_ds, bs: int, num_workers: int, pin_memory=True):
    dl = DataLoader(
        packed_ds,
        batch_size=bs,
        collate_fn=pretrain_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dl
