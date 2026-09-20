from dataclasses import dataclass
from typing import Literal

import pandas as pd
from lets_plot import *

Precision = Literal[64, 32, 16, 8, 4]


@dataclass
class AllocationMeta:
    num_params: int
    bs: int
    seq_len: int
    embed_dim: int
    vocab_size: int
    nheads: int
    dim_ff: int
    num_layers: int


@dataclass
class Allocation:
    name: str
    precision: Precision
    num_items: int | None = None
    sub_allocations: list["Allocation"] | None = None
    allocation_meta: AllocationMeta | None = None

    def __post_init__(self):
        if self.num_items is None and self.sub_allocations is None:
            raise ValueError("both num_items and sub_allocations cannot be None")
        if self.num_items is not None and self.sub_allocations:
            raise ValueError("both num_items and sub_allocations cannot be set")

        for child in self.sub_allocations or []:
            if child.allocation_meta is None:
                child.allocation_meta = self.allocation_meta

    @property
    def total_mem_usage(self):
        if self.num_items is None:
            return sum(a.total_mem_usage for a in self.sub_allocations or [])
        else:
            return get_memory_usage_mb(self.num_items, precision=self.precision)

    @property
    def direct_mem_usage(self):
        if self.num_items is None:
            return 0.0
        return get_memory_usage_mb(self.num_items, precision=self.precision)

    @property
    def total_num_items(self):
        if not self.sub_allocations:
            return self.num_items
        else:
            return sum(a.total_num_items for a in self.sub_allocations)  # type: ignore

    def to_df(self, max_level: int | None = None) -> pd.DataFrame:
        rows = []

        def visit(
            allocation: "Allocation",
            path: tuple[str, ...],
            level: int,
        ):
            current_path = path + (allocation.name,)

            rows.append(
                {
                    "path": current_path,
                    "precision": allocation.precision,
                    "num_items": allocation.num_items,
                    "total_num_items": allocation.total_num_items,
                    "direct_mem_mb": allocation.direct_mem_usage,
                    "total_mem_mb": allocation.total_mem_usage,
                }
            )

            # Stop expanding once we reach max_level
            if max_level is not None and level >= max_level:
                return

            for child in allocation.sub_allocations or []:
                visit(
                    child,
                    current_path,
                    level + 1,
                )

        visit(self, (), 0)

        max_depth = max(len(row["path"]) for row in rows)

        records = []

        for row in rows:
            padded_path = row["path"] + ("",) * (max_depth - len(row["path"]))

            records.append(
                {
                    **{f"level_{i}": name for i, name in enumerate(padded_path)},
                    "precision": row["precision"],
                    "total_num_items": row["total_num_items"],
                    "total_mem_mb": row["total_mem_mb"],
                    "num_items": row["num_items"],
                    "direct_mem_mb": row["direct_mem_mb"],
                }
            )

        level_columns = [f"level_{i}" for i in range(max_depth)]

        return pd.DataFrame(records).set_index(level_columns)


def get_memory_usage_mb(num_params, precision=32):
    return (num_params * precision) / (8 * 1024 * 1024)


def compute_naive_inference_mem_usage(
    bs: int, seq_len: int, embed_dim: int, vocab_size: int
):
    """
    approximate memory usage in forward pass (excluding model weights)
    memory used by intermediate tensors are re-used by torch in other places
    so even if we have 12 tranformer layers, the same memory space used by
    the first layer's output will be used to store the output of last transformer
    layer (very simplified explanation)

    NOTE: this assumes an un-optimized forward pass which is not really the case for
    production level inference which uses pre-fill stage, KV cache and
    prediction from the last token. the lm_head here predicts the next token
    for each token but during inference we only want the prediction from the
    last token in the sequence.

    we need to sum these up
    (bs * seq_len * 2),          # input token ids, since they are torch.long we multiply by 2
    (bs * seq_len * embed_dim),  # output of embedding layer
    (bs * seq_len * embed_dim),  # output of transformer layer
    (bs * seq_len * vocab_size), # output of lm head

    lets simplify the equation
    a=bs, b=seq_len, c=embed_dim, d=vocab_size
    then we can rewrite the above expressions as
        mem_usage = 2ab + abc + abc + abd
        mem_usage = ab(2 + c + c + d)
        mem_usage = ab(2 + 2c + d)
        mem_usage = bs * seq_len * (2 + 2*embed_dim + vocab_size)
    """
    return get_memory_usage_mb(
        bs * seq_len * (2 + 2 * embed_dim + vocab_size), precision=32
    )


AttnImpl = Literal["vanilla", "flash"]


def get_attention_allocation(
    bs: int,
    seq_len: int,
    embed_dim: int,
    nheads: int,
    compute_dtype_precision: Precision,
    storage_dtype_precision: Precision,
    attn_impl: AttnImpl = "vanilla",
):
    total_tokens = bs * seq_len
    # vanilla as O(N^2), flash attn has O(n)
    seq_len_exponent = 2 if attn_impl == "vanilla" else 1

    return Allocation(
        "attn",
        precision=compute_dtype_precision,
        sub_allocations=[
            Allocation(
                "qproj",
                num_items=total_tokens * embed_dim,
                precision=compute_dtype_precision,
            ),
            Allocation(
                "kproj",
                num_items=total_tokens * embed_dim,
                precision=compute_dtype_precision,
            ),
            Allocation(
                "vproj",
                num_items=total_tokens * embed_dim,
                precision=compute_dtype_precision,
            ),
            Allocation(
                "rope_cos",
                num_items=total_tokens * embed_dim // (2 * nheads),
                precision=32,
            ),
            Allocation(
                "rope_sin",
                num_items=total_tokens * embed_dim // (2 * nheads),
                precision=32,
            ),
            Allocation(
                "qrope",
                num_items=total_tokens * embed_dim,
                precision=32,
            ),
            Allocation(
                "krope",
                num_items=total_tokens * embed_dim,
                precision=32,
            ),
            Allocation(
                "q@k.T",
                num_items=bs * nheads * (seq_len**seq_len_exponent),
                precision=compute_dtype_precision,
            ),
            # softmax also uses 32 bit even when using amp
            Allocation(
                "softmax",
                # valilla has to compute softmax for all items in the qK.T matrix
                # but flash attn optimizes it
                num_items=bs * nheads * (seq_len**seq_len_exponent),
                # num_items=bs * nheads * seq_len * seq_len,
                precision=32,
            ),
            Allocation(
                "scores@V",
                num_items=total_tokens
                * embed_dim,  # same as bs*seq_len*nheads*head_dim
                precision=compute_dtype_precision,
            ),
            Allocation(
                "out_proj",
                num_items=total_tokens * embed_dim,
                precision=compute_dtype_precision,
            ),
        ],
    )


def compute_naive_training_mem_usage(
    num_params: int,
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
    uses_fp16_autocast = compute_dtype_precision == 16 and storage_dtype_precision == 32
    allocation_meta = AllocationMeta(
        num_params=num_params,
        bs=bs,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        nheads=nheads,
        dim_ff=dim_ff,
        num_layers=num_layers,
    )
    total_tokens = bs * seq_len
    input_module = Allocation(
        "input", num_items=total_tokens, precision=64
    )  # torch.long so 64 bit allocation
    position_ids = Allocation("position_ids", num_items=total_tokens, precision=64)
    causal_mask = Allocation("causal_mask", num_items=seq_len * seq_len, precision=8)

    # embedding output are in fp32 even if amp is used
    embedding_module = Allocation(
        "embedding",
        num_items=total_tokens * embed_dim,
        precision=32,
    )
    transformers = Allocation(
        "transformers",
        precision=compute_dtype_precision,
        sub_allocations=[
            Allocation(
                f"tf_{i}",
                precision=compute_dtype_precision,
                sub_allocations=[
                    # layer norm also uses 32 bit even when using amp
                    Allocation(
                        "norm1",
                        precision=32,
                        num_items=total_tokens * embed_dim,
                    ),
                    Allocation(
                        "norm1_stats",
                        precision=32,
                        num_items=total_tokens * 2,
                    ),
                    get_attention_allocation(
                        bs=bs,
                        seq_len=seq_len,
                        embed_dim=embed_dim,
                        nheads=nheads,
                        compute_dtype_precision=compute_dtype_precision,
                        storage_dtype_precision=storage_dtype_precision,
                        attn_impl=attn_impl,
                    ),
                    Allocation(
                        "attn_residual",
                        precision=32,
                        num_items=total_tokens * embed_dim,
                    ),
                    Allocation(
                        "ff",
                        precision=compute_dtype_precision,
                        sub_allocations=[
                            Allocation(
                                "linear1",
                                num_items=total_tokens * dim_ff,
                                precision=compute_dtype_precision,
                            ),
                            Allocation(
                                "gelu",
                                num_items=total_tokens * dim_ff,
                                precision=compute_dtype_precision,
                            ),
                            Allocation(
                                "linear2",
                                num_items=total_tokens * embed_dim,
                                precision=compute_dtype_precision,
                            ),
                        ],
                    ),
                    # layer norm also uses 32 bit even when using amp
                    Allocation(
                        "norm2",
                        precision=32,
                        num_items=total_tokens * embed_dim,
                    ),
                    Allocation(
                        "norm2_stats",
                        precision=32,
                        num_items=total_tokens * 2,
                    ),
                    Allocation(
                        "ff_residual",
                        precision=32,
                        num_items=total_tokens * embed_dim,
                    ),
                ],
            )
            for i in range(num_layers)
        ],
    )
    lm_head = Allocation(
        "lm_head",
        num_items=total_tokens * vocab_size,
        precision=compute_dtype_precision,
    )

    loss = Allocation(
        "loss", num_items=total_tokens * vocab_size, precision=storage_dtype_precision
    )

    dynamic_allocations = Allocation(
        "dynamic",
        precision=compute_dtype_precision,
        sub_allocations=[
            input_module,
            position_ids,
            causal_mask,
            embedding_module,
            transformers,
            lm_head,
            loss,
        ],
    )

    static_allocations = Allocation(
        "static",
        precision=32,
        sub_allocations=[
            Allocation(
                "params", num_items=num_params, precision=storage_dtype_precision
            ),
            Allocation(
                "grads", num_items=num_params, precision=storage_dtype_precision
            ),
            Allocation(
                "adamw", num_items=num_params * 2, precision=storage_dtype_precision
            ),
            *(
                [
                    Allocation(
                        "autocast_weight_cache",
                        num_items=(
                            num_layers
                            * (
                                4 * embed_dim * embed_dim
                                + 2 * embed_dim * dim_ff
                                + 5 * embed_dim
                                + dim_ff
                            )
                            + embed_dim * vocab_size
                            + vocab_size
                        ),
                        precision=compute_dtype_precision,
                    )
                ]
                if uses_fp16_autocast
                else []
            ),
        ],
    )

    return Allocation(
        "training",
        precision=compute_dtype_precision,
        sub_allocations=[static_allocations, dynamic_allocations],
        allocation_meta=allocation_meta,
    )


# Plot allocation
def prepare_icicle_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the MultiIndex allocation dataframe into coordinates suitable
    for an icicle chart.

    Assumes:
        - level_0, level_1, ... are index levels
        - total_mem_mb contains memory including all descendants
        - dataframe rows are in tree traversal order
    """
    data = df.reset_index().copy()

    level_cols = [c for c in data.columns if c.startswith("level_")]

    # Convert NaN / "" hierarchy values into a path.
    def get_path(row):
        return tuple(
            str(row[c]) for c in level_cols if pd.notna(row[c]) and str(row[c]) != ""
        )

    data["path"] = data.apply(get_path, axis=1)
    data["depth"] = data["path"].map(len) - 1
    data["name"] = data["path"].map(lambda x: x[-1])

    # Map path -> total memory.
    mem = dict(zip(data["path"], data["total_mem_mb"]))

    # Children of every node, preserving dataframe order.
    children = {path: [] for path in data["path"]}

    for path in data["path"]:
        if len(path) > 1:
            parent = path[:-1]
            if parent in children:
                children[parent].append(path)

    coordinates = {}

    def layout(path, x0):
        """
        Parent occupies [x0, x0 + total_mem].
        Children are packed inside that interval.
        """
        width = mem[path]

        coordinates[path] = {
            "xmin": x0,
            "xmax": x0 + width,
        }

        child_x = x0

        for child in children[path]:
            layout(child, child_x)
            child_x += mem[child]

    root = data.iloc[0]["path"]
    layout(root, 0)

    data["xmin"] = data["path"].map(lambda p: coordinates[p]["xmin"])
    data["xmax"] = data["path"].map(lambda p: coordinates[p]["xmax"])

    # Reverse depth so root appears at the top.
    max_depth = data["depth"].max()

    data["ymin"] = max_depth - data["depth"]
    data["ymax"] = data["ymin"] + 0.9
    data["xmid"] = (data["xmin"] + data["xmax"]) / 2
    data["ymid"] = (data["ymin"] + data["ymax"]) / 2

    # Useful labels/tooltips
    data["memory"] = data["total_mem_mb"].map(lambda x: f"{x:,.1f} MB")

    data["path_str"] = data["path"].map(lambda p: " → ".join(p))

    # Use level-1 branch as the colour category.
    def top_group(path):
        if len(path) >= 2:
            return path[1]
        return path[0]

    data["group"] = data["path"].map(top_group)

    return data


def plot_training_allocation(
    df: pd.DataFrame | Allocation,
    max_level: int = 3,
    title: str | None = None,
    subtitle: str | None = None,
):
    if isinstance(df, Allocation):
        title = (
            title or f"Estimated Training Memory Usage: {df.total_mem_usage:,.0f} MB"
        )
        am: AllocationMeta = df.allocation_meta  # type: ignore
        subtitle = (
            subtitle
            or f"{am.num_params=:,}, {am.bs=:,}, {am.seq_len=:,}, {am.embed_dim=:,}, {am.vocab_size=:,}, {am.num_layers=}, {am.nheads=}, {am.dim_ff=}"
        )
    df = df if isinstance(df, pd.DataFrame) else df.to_df(max_level=max_level)

    plot_df = prepare_icicle_df(df)
    # Don't write text into very narrow boxes.
    # Tooltip will still show the information.
    label_df = plot_df[plot_df["total_mem_mb"] >= plot_df["total_mem_mb"].max() * 0.025]

    p = (
        ggplot()
        + geom_rect(
            aes(
                xmin="xmin",
                xmax="xmax",
                ymin="ymin",
                ymax="ymax",
                fill="group",
            ),
            data=plot_df,
            color="white",
            size=0.5,
            tooltips=layer_tooltips()
            .line("@path_str")
            .line("Memory|@memory")
            .line("Precision|@precision bit")
            .line("Items|@total_num_items"),
        )
        + geom_text(
            aes(
                x="xmid",
                y="ymid",
                label="name",
            ),
            data=label_df,
            size=9,
        )
        + labs(
            title=title or "Training Memory Allocation",
            subtitle=subtitle or "Rectangle width represents total memory usage",
            x="Memory (MB)",
            y="",
            fill="Allocation",
        )
        + scale_x_continuous(format=",.0f")
        + theme(
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_title_y=element_blank(),
            panel_grid=element_blank(),
            legend_position="bottom",
            plot_title=element_text(size=20, face="bold"),
        )
        + ggsize(1600, 500)
    )

    return p
