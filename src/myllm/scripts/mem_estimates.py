from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd
import torch

from myllm.scaling import (
    Allocation,
    AttnImpl,
    Precision,
    compute_naive_training_mem_usage,
)
from myllm.v1 import MyLLMV1

DEFAULT_SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_MICROBATCH_SIZES = [1, 2, 4, 8]


@dataclass(frozen=True)
class ModelMemoryConfig:
    num_params: int = 122_827_776
    embed_dim: int = 768
    vocab_size: int = 49_152
    num_heads: int = 8
    ff_dim: int = 1_024
    num_layers: int = 12
    compute_precision: Precision = 16
    storage_precision: Precision = 32


def _child_memory(allocation: Allocation, name: str) -> float:
    for child in allocation.sub_allocations or []:
        if child.name == name:
            return child.total_mem_usage
    raise ValueError(f"Allocation does not contain a {name!r} branch")


def sweep_memory_estimates(
    model: ModelMemoryConfig,
    sequence_lengths: list[int],
    microbatch_sizes: list[int],
    gpu_memory_gib: float,
    headroom_percent: float = 10.0,
    world_size: int = 1,
    target_batch_tokens: int | None = None,
) -> pd.DataFrame:
    """Estimate memory for a sequence-length by microbatch-size grid.

    The returned memory values are in GiB. ``fits`` reserves the requested
    percentage of the GPU's capacity rather than treating all VRAM as usable.
    """
    model_dimensions = {
        "num_params": model.num_params,
        "embed_dim": model.embed_dim,
        "vocab_size": model.vocab_size,
        "num_heads": model.num_heads,
        "ff_dim": model.ff_dim,
        "num_layers": model.num_layers,
    }
    if any(value <= 0 for value in model_dimensions.values()):
        raise ValueError("All model dimensions must be positive")
    if model.embed_dim % model.num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by the number of heads")
    if not sequence_lengths or not microbatch_sizes:
        raise ValueError("At least one sequence length and microbatch size is required")
    if any(value <= 0 for value in sequence_lengths):
        raise ValueError("Sequence lengths must be positive")
    if any(value <= 0 for value in microbatch_sizes):
        raise ValueError("Microbatch sizes must be positive")
    if gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    if not 0 <= headroom_percent < 100:
        raise ValueError("Headroom percent must be between 0 (inclusive) and 100")
    if world_size <= 0:
        raise ValueError("World size must be positive")
    if target_batch_tokens is not None and target_batch_tokens <= 0:
        raise ValueError("Target batch tokens must be positive")

    usable_memory_gib = gpu_memory_gib * (1 - headroom_percent / 100)
    rows = []

    for sequence_length in sorted(set(sequence_lengths)):
        for microbatch_size in sorted(set(microbatch_sizes)):
            allocation = compute_naive_training_mem_usage(
                num_params=model.num_params,
                bs=microbatch_size,
                seq_len=sequence_length,
                embed_dim=model.embed_dim,
                vocab_size=model.vocab_size,
                nheads=model.num_heads,
                dim_ff=model.ff_dim,
                num_layers=model.num_layers,
                compute_dtype_precision=model.compute_precision,
                storage_dtype_precision=model.storage_precision,
            )
            estimated_memory_gib = allocation.total_mem_usage / 1024
            tokens_per_microbatch = sequence_length * microbatch_size
            accumulation_steps = (
                ceil(target_batch_tokens / (tokens_per_microbatch * world_size))
                if target_batch_tokens is not None
                else None
            )
            rows.append(
                {
                    "sequence_length": sequence_length,
                    "microbatch_size": microbatch_size,
                    "tokens_per_microbatch": tokens_per_microbatch,
                    "gradient_accumulation_steps": accumulation_steps,
                    "effective_batch_tokens": (
                        tokens_per_microbatch * world_size * accumulation_steps
                        if accumulation_steps is not None
                        else None
                    ),
                    "estimated_memory_gib": estimated_memory_gib,
                    "static_memory_gib": _child_memory(allocation, "static") / 1024,
                    "dynamic_memory_gib": _child_memory(allocation, "dynamic") / 1024,
                    "gpu_memory_gib": gpu_memory_gib,
                    "usable_memory_gib": usable_memory_gib,
                    "remaining_memory_gib": usable_memory_gib - estimated_memory_gib,
                    "fits": estimated_memory_gib <= usable_memory_gib,
                }
            )

    return pd.DataFrame(rows)


def memory_frontier(sweep: pd.DataFrame) -> pd.DataFrame:
    """Select the largest tested feasible microbatch for each sequence length.

    When no tested microbatch fits, retain the smallest one so the caller can
    report both the failed sequence length and its minimum tested requirement.
    """
    rows = []
    for _, group in sweep.groupby("sequence_length", sort=True):
        feasible = group[group["fits"]]
        candidates = feasible if not feasible.empty else group
        index = (
            candidates["microbatch_size"].idxmax()
            if not feasible.empty
            else candidates["microbatch_size"].idxmin()
        )
        rows.append(sweep.loc[index])
    return pd.DataFrame(rows).reset_index(drop=True)


def save_memory_sweep(sweep: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_path, index=False)


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def print_cuda_memory(prefix=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    print(
        f"{prefix} {allocated=:,.2f} MB, {reserved=:,.2f}, {peak_allocated=:,.2f} MB, {peak_reserved=:,.2f} MB"
    )


def estimate_vs_real(
    bs: int,
    seq_len: int,
    embed_dim: int,
    vocab_size: int,
    nheads: int,
    dim_ff: int,
    num_layers: int,
    compute_dtype_precision: Precision = 32,
    storage_dtype_precision: Precision = 32,
    attn_impl: AttnImpl = "vanilla",
):
    dtype = torch.float32
    if compute_dtype_precision == 16:
        dtype = torch.bfloat16
    elif compute_dtype_precision != 32:
        raise ValueError("compute dtype precision must be 16 or 32")

    torch.set_float32_matmul_precision("high")

    model = MyLLMV1(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        nlayers=num_layers,
        nheads=nheads,
        dim_ff=dim_ff,
        padding_idx=None,
        rope_theta=10000,
    )

    allocation = compute_naive_training_mem_usage(
        num_params=model.num_params,
        bs=bs,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        nheads=nheads,
        dim_ff=dim_ff,
        num_layers=num_layers,
        compute_dtype_precision=compute_dtype_precision,
        storage_dtype_precision=storage_dtype_precision,
        attn_impl=attn_impl,
    )

    print(f"Total model params: {model.num_params:,}, {bs=:,}, {seq_len=:,}")
    print(f"Estimated memory: {allocation.total_mem_usage:,.2f} MB")

    reset_peak()
    print_cuda_memory()

    if storage_dtype_precision == 16:
        model.bfloat16()

    model.cuda()
    print_cuda_memory("after model to cuda")

    reset_peak()
    token_ids = torch.zeros((bs, seq_len), dtype=torch.long).cuda()
    print_cuda_memory("after input to cuda")

    optim = torch.optim.AdamW(model.parameters(), fused=True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100).cuda()
    loss_fn.compile()

    for i in range(2):
        with torch.autocast(device_type="cuda", dtype=dtype):
            reset_peak()
            out = model(token_ids)
            labels = torch.full_like(token_ids, -100)
            labels[:, :-1] = token_ids[:, 1:]
            labels = labels.view(-1)
            logits = out.view(-1, vocab_size)
            loss = loss_fn(logits, labels)
        loss.backward()
        optim.step()
    print_cuda_memory("after optim step")

    peak_memory_allocated = torch.cuda.max_memory_reserved() / 1024**2

    delta = peak_memory_allocated - allocation.total_mem_usage
    pct_delta = delta / peak_memory_allocated * 100
    print()
    print(
        f"estimated: {allocation.total_mem_usage:,.2f} MB, actual peak: {peak_memory_allocated:,} MB, delta: {delta:,.2f} MB, pct_delta: {pct_delta:,.2f}%"
    )
