from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from myllm import config

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
SPLIT = "train"

ROWS_PER_SHARD = 30_000  # no. of documents. thse could have thousands of tokens per doc
MAX_SHARDS_FOR_DEV = 10


def tokenize(output_dir: str | Path, dev=False):
    output_dir = Path(output_dir)
    final_output_dir = output_dir / ("dev" if dev else "full")
    final_output_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL_ID, fast=True)

    ds = datasets.load_dataset(
        DATASET,
        name=SUBSET,
        split=SPLIT,
        streaming=True,
        columns=["text"],
    )

    def _tokenize(batch):
        return {"token_ids": tokenizer.encode(batch["text"], add_special_tokens=False)}

    tokenized_ds = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)

    total_tokens = 0
    batches = tokenized_ds.iter(batch_size=ROWS_PER_SHARD)

    try:
        for shard_idx, batch in enumerate(batches):
            num_tokens_in_batch = sum(map(len, batch["token_ids"]))
            total_tokens += num_tokens_in_batch

            table = pa.Table.from_pydict(batch)
            file_path = final_output_dir / f"shard-{shard_idx:05d}.parquet"

            print(
                f"Writing shard number: {shard_idx:05d} with {num_tokens_in_batch:,} tokens to {file_path}"
            )
            pq.write_table(table, file_path, compression="zstd")

            if dev and shard_idx + 1 >= MAX_SHARDS_FOR_DEV:
                print("Max shards for dev processed. Stopping")
                break
    finally:
        batches.close()

    print("=" * 50)
    print(f"Finished tokenizing. Total tokens: {total_tokens:,}")
    print(f"{shard_idx + 1:,} shards written to {final_output_dir}")
