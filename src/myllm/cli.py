from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from myllm import config

banner = """
+========================================================================+
|                                                                        |
|  ██████   ██████               █████       █████       ██████   ██████ |
| ░░██████ ██████               ░░███       ░░███       ░░██████ ██████  |
|  ░███░█████░███  █████ ████    ░███        ░███        ░███░█████░███  |
|  ░███░░███ ░███ ░░███ ░███     ░███        ░███        ░███░░███ ░███  |
|  ░███ ░░░  ░███  ░███ ░███     ░███        ░███        ░███ ░░░  ░███  |
|  ░███      ░███  ░███ ░███     ░███      █ ░███      █ ░███      ░███  |
|  █████     █████ ░░███████     ███████████ ███████████ █████     █████ |
| ░░░░░     ░░░░░   ░░░░░███    ░░░░░░░░░░░ ░░░░░░░░░░░ ░░░░░     ░░░░░  |
|                   ███ ░███                                             |
|                  ░░██████                                              |
|                   ░░░░░░                                               |
|                                                                        |
+========================================================================+ 
"""
cli = typer.Typer(no_args_is_help=True, epilog=banner)

data_cli = typer.Typer(no_args_is_help=True, help="Data tokenization")
pretrain_cli = typer.Typer(no_args_is_help=True, help="Pretraining")
sft_cli = typer.Typer(no_args_is_help=True, help="Supervised Fine Tuning")
memory_cli = typer.Typer(
    no_args_is_help=True,
    help="Estimate and explore memory requirements.",
)

cli.add_typer(data_cli, name="data")
cli.add_typer(pretrain_cli, name="pretrain")
cli.add_typer(sft_cli, name="sft")
cli.add_typer(memory_cli, name="memory")


@data_cli.command()
def tokenize(dev: bool = False):
    typer.echo(f"Tokenizing data {dev=}")
    from myllm.scripts.tokenize import tokenize

    tokenize(output_dir=config.PRETRAIN_DATA_DIR, dev=dev)


@pretrain_cli.command()
def pretrain(dev: bool = False):
    typer.echo(f"Starting pretraining {dev=}")


@memory_cli.command("sweep")
def memory_sweep(
    sequence_lengths: Annotated[
        list[int] | None,
        typer.Option(
            "--sequence-length",
            "-s",
            help=(
                "Sequence length to test. Repeat for multiple values. "
                "Defaults to 128, 256, 512, 1024, 2048, 4096."
            ),
        ),
    ] = None,
    microbatch_sizes: Annotated[
        list[int] | None,
        typer.Option(
            "--microbatch-size",
            "-b",
            help=(
                "Per-GPU microbatch size to test. Repeat for multiple values. "
                "Defaults to 1, 2, 4, 8"
            ),
        ),
    ] = None,
    gpu_memory_gib: Annotated[
        float,
        typer.Option("--gpu-memory-gib", min=0, help="GPU VRAM capacity in GiB."),
    ] = 24.0,
    headroom_percent: Annotated[
        float,
        typer.Option(
            "--headroom-percent",
            min=0,
            max=99.999,
            help="Percentage of VRAM reserved for overhead and uncertainty.",
        ),
    ] = 10.0,
    num_params: Annotated[int, typer.Option(min=1)] = 122_827_776,
    embed_dim: Annotated[int, typer.Option(min=1)] = 768,
    vocab_size: Annotated[int, typer.Option(min=1)] = 49_152,
    num_heads: Annotated[int, typer.Option("--num-heads", min=1)] = 8,
    ff_dim: Annotated[int, typer.Option("--ff-dim", min=1)] = 1_024,
    num_layers: Annotated[int, typer.Option(min=1)] = 12,
    compute_precision: Annotated[
        int, typer.Option(help="Compute precision in bits (4, 8, 16, 32, or 64).")
    ] = 32,
    storage_precision: Annotated[
        int,
        typer.Option(help="Parameter/storage precision in bits (4, 8, 16, 32, or 64)."),
    ] = 32,
    world_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of data-parallel workers used for effective batch tokens.",
        ),
    ] = 1,
    target_batch_tokens: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Calculate accumulation steps needed to reach this global batch size.",
        ),
    ] = None,
    output_csv: Annotated[
        Path | None,
        typer.Option("--output-csv", help="Write the complete sweep grid to CSV."),
    ] = None,
    show_all: Annotated[
        bool,
        typer.Option(
            help="Print every tested combination instead of only the frontier."
        ),
    ] = False,
):
    """Find the largest tested microbatch that fits at each sequence length."""
    from myllm.scaling import Precision
    from myllm.scripts.mem_estimates import (
        DEFAULT_MICROBATCH_SIZES,
        DEFAULT_SEQUENCE_LENGTHS,
        ModelMemoryConfig,
        memory_frontier,
        save_memory_sweep,
        sweep_memory_estimates,
    )

    valid_precisions = {4, 8, 16, 32, 64}
    if compute_precision not in valid_precisions:
        raise typer.BadParameter(
            "must be one of 4, 8, 16, 32, or 64", param_hint="compute-precision"
        )
    if storage_precision not in valid_precisions:
        raise typer.BadParameter(
            "must be one of 4, 8, 16, 32, or 64", param_hint="storage-precision"
        )

    model = ModelMemoryConfig(
        num_params=num_params,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        compute_precision=cast(Precision, compute_precision),
        storage_precision=cast(Precision, storage_precision),
    )
    try:
        sweep = sweep_memory_estimates(
            model=model,
            sequence_lengths=sequence_lengths or DEFAULT_SEQUENCE_LENGTHS,
            microbatch_sizes=microbatch_sizes or DEFAULT_MICROBATCH_SIZES,
            gpu_memory_gib=gpu_memory_gib,
            headroom_percent=headroom_percent,
            world_size=world_size,
            target_batch_tokens=target_batch_tokens,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    result = sweep if show_all else memory_frontier(sweep)
    usable_memory_gib = gpu_memory_gib * (1 - headroom_percent / 100)
    typer.echo(
        f"GPU capacity: {gpu_memory_gib:.2f} GiB; usable after "
        f"{headroom_percent:g}% headroom: {usable_memory_gib:.2f} GiB"
    )
    typer.echo(
        f"Model: {num_params:,} params; {num_layers} layers; width {embed_dim}; "
        f"FFN {ff_dim}; vocab {vocab_size:,}; {compute_precision}-bit compute / "
        f"{storage_precision}-bit storage"
    )

    if not result["fits"].any():
        minimum = sweep.loc[sweep["estimated_memory_gib"].idxmin()]
        typer.echo("No tested configuration fits.")
        typer.echo(
            "Smallest estimate: "
            f"sequence={int(minimum['sequence_length'])}, "
            f"microbatch={int(minimum['microbatch_size'])}, "
            f"memory={minimum['estimated_memory_gib']:.2f} GiB"
        )
    if not result.empty:
        columns = [
            "sequence_length",
            "microbatch_size",
            "tokens_per_microbatch",
            "estimated_memory_gib",
            "static_memory_gib",
            "dynamic_memory_gib",
            "remaining_memory_gib",
        ]
        if target_batch_tokens is not None:
            columns.extend(["gradient_accumulation_steps", "effective_batch_tokens"])
        columns.append("fits")
        display = result[columns].rename(
            columns={
                "sequence_length": "sequence",
                "microbatch_size": "microbatch",
                "tokens_per_microbatch": "tokens",
                "estimated_memory_gib": "memory_gib",
                "static_memory_gib": "static_gib",
                "dynamic_memory_gib": "dynamic_gib",
                "remaining_memory_gib": "remaining_gib",
                "gradient_accumulation_steps": "accumulation",
                "effective_batch_tokens": "effective_tokens",
            }
        )
        typer.echo(
            display.to_string(index=False, float_format=lambda value: f"{value:.2f}")
        )

    if output_csv is not None:
        save_memory_sweep(sweep, output_csv)
        typer.echo(f"Wrote {len(sweep)} estimates to {output_csv}")


@memory_cli.command("estimate-vs-real")
def estimate_vs_real(
    bs: int = 8,
    seq_len: int = 1024,
    embed_dim: int = 1024,
    vocab_size: int = 49512,
    nheads: int = 8,
    dim_ff: int = 1024,
    num_layers: int = 12,
    compute_dtype_precision: int = 32,
    storage_dtype_precision: int = 32,
    attn_impl: Literal["vanilla", "flash"] = "flash",
):
    from myllm.scripts import mem_estimates

    mem_estimates.estimate_vs_real(
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


if __name__ == "__main__":
    cli()
