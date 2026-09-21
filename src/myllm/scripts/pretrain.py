import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoConfig

from myllm import config, data
from myllm.scaling import compute_naive_training_mem_usage
from myllm.trainer import (
    CheckpointCallback,
    MetricsCallback,
    ResolvedTrainerConfig,
    Trainer,
    TrainingConfig,
)
from myllm.v1 import MyLLMV1

REFERENCE_DEPTH = 12
REFERENCE_WIDTH = 1024
HEAD_DIM = 128
SEQ_LEN = 1024
TARGET_TOKENS_PER_UPDATE = 24 * (SEQ_LEN - 1) * 6
TOKENS_PER_PARAMETER = 20
MEMORY_HEADROOM = 0.01
MICRO_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 64, 128)


@dataclass(frozen=True)
class PretrainingPlan:
    depth: int
    embed_dim: int
    num_heads: int
    ff_dim: int
    num_params: int
    micro_batch_size: int
    max_train_tokens: int


def _model(
    *,
    vocab_size: int,
    depth: int,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
):
    return MyLLMV1(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        nlayers=depth,
        nheads=num_heads,
        dim_ff=ff_dim,
        padding_idx=None,
        rope_theta=10_000,
    )


# Used this as a reference but I went my own way :D
# https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py
def resolve_pretraining_plan(
    depth: int,
    vocab_size: int,
    seq_len: int,
    gpu_memory_gib: float | None = None,
    user_supplied_batch_size: int | None = None,
) -> PretrainingPlan:
    if depth <= 0:
        raise ValueError("depth must be positive")

    unrounded_width = depth * REFERENCE_WIDTH / REFERENCE_DEPTH
    embed_dim = max(HEAD_DIM, round(unrounded_width / HEAD_DIM) * HEAD_DIM)
    num_heads = embed_dim // HEAD_DIM
    ff_dim = embed_dim * 4

    with torch.device("meta"):
        model = _model(
            vocab_size=vocab_size,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
    num_params = model.num_params

    if user_supplied_batch_size is None:
        if gpu_memory_gib is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required to resolve the micro-batch size")
            free_memory, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            gpu_memory_gib = free_memory / 1024**3

        usable_memory_gib = gpu_memory_gib * (1 - MEMORY_HEADROOM)
        micro_batch_size = None
        minimum_estimate_gib = None
        for candidate in MICRO_BATCH_SIZES:
            allocation = compute_naive_training_mem_usage(
                num_params=num_params,
                bs=candidate,
                seq_len=seq_len,
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                nheads=num_heads,
                dim_ff=ff_dim,
                num_layers=depth,
                compute_dtype_precision=16,
                storage_dtype_precision=32,
                attn_impl="flash",
            )
            estimated_memory_gib = allocation.total_mem_usage / 1024
            minimum_estimate_gib = minimum_estimate_gib or estimated_memory_gib
            if estimated_memory_gib <= usable_memory_gib:
                micro_batch_size = candidate

        if micro_batch_size is None:
            print(
                f"Depth {depth} needs an estimated {minimum_estimate_gib:.2f} GiB "
                f"for micro-batch size 1; only {usable_memory_gib:.2f} GiB is usable. "
                "\nSetting it to 1. Check VRAM usage and explicitly pass --bs if under utilized"
            )
            micro_batch_size = 1
    else:
        micro_batch_size = user_supplied_batch_size

    return PretrainingPlan(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_params=num_params,
        micro_batch_size=micro_batch_size,
        max_train_tokens=math.ceil(TOKENS_PER_PARAMETER * num_params),
    )


def resolve_training_config(
    plan: PretrainingPlan,
    *,
    seq_len: int,
    dev: bool = False,
    micro_batch_size: int | None = None,
    max_train_minutes: float | None = None,
    max_train_tokens: int | None = None,
    max_train_steps: int | None = None,
    grad_accum_steps: int | None = None,
    target_tokens_per_update: int | None = None,
    lr: float | None = None,
    lr_schedule: Literal["constant", "cosine"] | None = None,
    warmup_ratio: float | None = None,
    min_lr_ratio: float | None = None,
    max_grad_norm: float | None = None,
    compute_dtype: torch.dtype | None = None,
) -> TrainingConfig:
    if not any(
        value is not None
        for value in (max_train_minutes, max_train_tokens, max_train_steps)
    ):
        if dev:
            max_train_steps = 10
        else:
            max_train_tokens = plan.max_train_tokens

    if grad_accum_steps is None and target_tokens_per_update is None:
        target_tokens_per_update = 24 * (seq_len - 1) * 6

    optional_overrides = {
        name: value
        for name, value in (
            ("lr", lr),
            ("lr_schedule", lr_schedule),
            ("warmup_ratio", warmup_ratio),
            ("min_lr_ratio", min_lr_ratio),
            ("max_grad_norm", max_grad_norm),
            ("compute_dtype", compute_dtype),
        )
        if value is not None
    }
    return TrainingConfig(
        micro_batch_size=(
            plan.micro_batch_size if micro_batch_size is None else micro_batch_size
        ),
        seq_len=seq_len,
        max_train_minutes=max_train_minutes,
        max_train_tokens=max_train_tokens,
        max_train_steps=max_train_steps,
        grad_accum_steps=grad_accum_steps,
        target_tokens_per_update=target_tokens_per_update,
        **optional_overrides,
    )


def resolve_run_dir(*, dev: bool, tag: str) -> Path:
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be a single folder name")

    parent_dir = config.PRETRAIN_CKPT_DIR / ("dev" if dev else "full")
    parent_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    while True:
        folder_name = tag if index == 0 else f"{tag}-{index:03d}"
        run_dir = parent_dir / folder_name
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            index += 1
        else:
            return run_dir


def save_run_metadata(
    run_dir: Path,
    *,
    dev: bool,
    plan: PretrainingPlan,
    training_config: ResolvedTrainerConfig,
    model: torch.nn.Module,
    vocab_size: int,
    eos_token_id: int,
):
    resolved_training = asdict(training_config)
    resolved_training["compute_dtype"] = str(training_config.compute_dtype)
    resolved_training.update(
        effective_batch_size=training_config.effective_batch_size,
        tokens_per_micro_batch=training_config.tokens_per_micro_batch,
        tokens_per_update=training_config.tokens_per_update,
        total_train_tokens=training_config.total_train_tokens,
    )

    metadata = {
        "run": {
            "name": run_dir.name,
            "mode": "dev" if dev else "full",
        },
        "model": {
            "class": type(model).__name__,
            "parameters": {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
            "vocab_size": vocab_size,
            "depth": plan.depth,
            "embed_dim": plan.embed_dim,
            "num_heads": plan.num_heads,
            "head_dim": plan.embed_dim // plan.num_heads,
            "ff_dim": plan.ff_dim,
            "rope_theta": 10_000,
        },
        "tokenizer": {
            "model_id": config.TOKENIZER_MODEL_ID,
            "eos_token_id": eos_token_id,
        },
        "training": resolved_training,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def launch_training(
    depth: int = REFERENCE_DEPTH,
    tag: str = "latest",
    seq_len: int = SEQ_LEN,
    micro_batch_size: int | None = None,
    max_train_minutes: float | None = None,
    max_train_tokens: int | None = None,
    max_train_steps: int | None = None,
    grad_accum_steps: int | None = None,
    target_tokens_per_update: int | None = None,
    lr: float | None = None,
    lr_schedule: Literal["constant", "cosine"] | None = None,
    warmup_ratio: float | None = None,
    min_lr_ratio: float | None = None,
    max_grad_norm: float | None = None,
    compute_dtype: torch.dtype | None = None,
    dev: bool = False,
):
    tokenizer_config = AutoConfig.from_pretrained(config.TOKENIZER_MODEL_ID)
    plan = resolve_pretraining_plan(
        depth=depth,
        seq_len=seq_len,
        vocab_size=tokenizer_config.vocab_size,
        user_supplied_batch_size=micro_batch_size,
    )

    training_config = resolve_training_config(
        plan,
        seq_len=seq_len,
        dev=dev,
        micro_batch_size=micro_batch_size,
        max_train_minutes=max_train_minutes,
        max_train_tokens=max_train_tokens,
        max_train_steps=max_train_steps,
        grad_accum_steps=grad_accum_steps,
        target_tokens_per_update=target_tokens_per_update,
        lr=lr,
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        max_grad_norm=max_grad_norm,
        compute_dtype=compute_dtype,
    ).resolve()

    dataset = data.get_pretraining_dataset(
        seq_len=seq_len,
        eos_token_id=tokenizer_config.eos_token_id,
        dev=dev,
    )
    train_dl = data.get_pretraining_data_loader(
        dataset,
        bs=training_config.micro_batch_size,
        num_workers=4,
    )

    model = _model(
        vocab_size=tokenizer_config.vocab_size,
        depth=plan.depth,
        embed_dim=plan.embed_dim,
        num_heads=plan.num_heads,
        ff_dim=plan.ff_dim,
    )
    run_dir = resolve_run_dir(dev=dev, tag=tag)
    save_run_metadata(
        run_dir,
        dev=dev,
        plan=plan,
        training_config=training_config,
        model=model,
        vocab_size=tokenizer_config.vocab_size,
        eos_token_id=tokenizer_config.eos_token_id,
    )
    model.cuda().compile()
    trainer = Trainer(
        config=training_config,
        model=model,
        callbacks=[
            CheckpointCallback(run_dir / "checkpoints" / "checkpoint.pt"),
            MetricsCallback(run_dir / "training_metrics.jsonl"),
        ],
    )
    print(f"Artifacts are available at: {run_dir.resolve()}")
    trainer.train(train_dl=train_dl)
