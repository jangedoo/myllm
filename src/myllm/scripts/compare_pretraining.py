import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from myllm import config

METADATA_FILE = "metadata.json"
METRICS_FILE = "training_metrics.jsonl"


def resolve_run_dirs(run_names: list[str], *, dev: bool) -> list[Path]:
    """Resolve run folder names below the configured pretraining directory."""
    if len(run_names) < 2:
        raise ValueError("provide at least two run names")
    if len(set(run_names)) != len(run_names):
        raise ValueError("run names must be unique")

    runs_root = config.PRETRAIN_CKPT_DIR / ("dev" if dev else "full")
    run_dirs = []
    for run_name in run_names:
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError(f"invalid run name: {run_name!r}")

        run_dir = runs_root / run_name
        if not run_dir.is_dir():
            raise ValueError(f"run does not exist: {run_dir}")

        missing = [
            filename
            for filename in (METADATA_FILE, METRICS_FILE)
            if not (run_dir / filename).is_file()
        ]
        if missing:
            raise ValueError(f"run {run_name!r} is missing: {', '.join(missing)}")
        run_dirs.append(run_dir)

    return run_dirs


def _nested_value(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_comparison_data(
    run_dirs: list[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load and normalize metadata and training metrics for multiple runs."""
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    required_columns = {
        "split",
        "step",
        "loss",
        "learning_rate",
        "tokens_processed",
        "elapsed_seconds",
        "tokens_per_second",
    }

    for run_dir in run_dirs:
        try:
            run_metadata = json.loads(
                (run_dir / METADATA_FILE).read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"could not read metadata for {run_dir.name}: {exc}"
            ) from exc

        try:
            run_metrics = pd.read_json(run_dir / METRICS_FILE, lines=True)
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"could not read metrics for {run_dir.name}: {exc}"
            ) from exc

        missing_columns = sorted(required_columns - set(run_metrics.columns))
        if missing_columns:
            raise ValueError(
                f"metrics for {run_dir.name!r} are missing columns: "
                f"{', '.join(missing_columns)}"
            )

        run_metrics = run_metrics.loc[run_metrics["split"] == "train"].copy()
        if run_metrics.empty:
            raise ValueError(f"run {run_dir.name!r} has no training metrics")

        run_metrics = run_metrics.sort_values("step")
        smoothing_window = min(20, max(1, len(run_metrics) // 10))
        run_metrics["loss_smoothed"] = (
            run_metrics["loss"].rolling(smoothing_window, min_periods=1).mean()
        )
        run_metrics["run"] = run_dir.name
        run_metrics["tokens_processed_m"] = run_metrics["tokens_processed"] / 1_000_000
        run_metrics["elapsed_minutes"] = run_metrics["elapsed_seconds"] / 60
        run_metrics["tokens_per_second_k"] = run_metrics["tokens_per_second"] / 1_000
        run_metrics["learning_rate_micro"] = run_metrics["learning_rate"] * 1_000_000
        frames.append(run_metrics)
        metadata.append(run_metadata)

    return pd.concat(frames, ignore_index=True), metadata


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _summary_table(metrics: pd.DataFrame, metadata: list[dict[str, Any]]) -> str:
    metadata_by_name = {
        _nested_value(item, "run", "name", default=""): item for item in metadata
    }
    columns = (
        "Run",
        "Parameters",
        "Depth",
        "Width",
        "Seq len",
        "Micro batch",
        "Accum",
        "Peak LR",
        "Final loss",
        "Steps",
        "Tokens",
        "Elapsed",
        "Final tok/s",
        "Dtype",
    )
    rows = []
    for run_name, run_metrics in metrics.groupby("run", sort=False):
        meta = metadata_by_name.get(run_name, {})
        final = run_metrics.iloc[-1]
        values = (
            run_name,
            _nested_value(meta, "model", "parameters", "total"),
            _nested_value(meta, "model", "depth"),
            _nested_value(meta, "model", "embed_dim"),
            _nested_value(meta, "training", "seq_len"),
            _nested_value(meta, "training", "micro_batch_size"),
            _nested_value(meta, "training", "grad_accum_steps"),
            float(run_metrics["learning_rate"].max()),
            float(final["loss"]),
            int(final["step"]),
            int(final["tokens_processed"]),
            f"{float(final['elapsed_seconds']) / 60:,.2f} min",
            f"{float(final['tokens_per_second']):,.0f}",
            _nested_value(meta, "training", "compute_dtype"),
        )
        cells = "".join(
            f"<td>{html.escape(_format_number(value))}</td>" for value in values
        )
        rows.append(f"<tr>{cells}</tr>")

    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    return f"""
<section class="run-summary">
  <h1>Pretraining run comparison</h1>
  <p>Hover over the plots for exact values. Loss is smoothed independently for
     each run with a rolling window of up to 20 steps.</p>
  <div class="table-wrap">
    <table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>
  </div>
</section>
"""


def _inject_summary(output_path: Path, summary: str) -> None:
    document = output_path.read_text(encoding="utf-8")
    styles = """
<style>
  body { margin: 0; background: #f8fafc; color: #0f172a;
         font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  .run-summary { box-sizing: border-box; max-width: 1400px; margin: 0 auto;
                 padding: 28px 28px 4px; }
  .run-summary h1 { font-size: 28px; margin: 0 0 6px; }
  .run-summary p { color: #475569; margin: 0 0 18px; }
  body > div[id] { max-width: 1400px; margin-left: auto; margin-right: auto;
                   overflow-x: auto; }
  .table-wrap { overflow-x: auto; border: 1px solid #e2e8f0;
                border-radius: 10px; background: white; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { padding: 9px 11px; border-bottom: 1px solid #e2e8f0;
           text-align: right; white-space: nowrap; }
  th { color: #475569; background: #f1f5f9; font-weight: 600; }
  th:first-child, td:first-child { text-align: left; font-weight: 600; }
  tbody tr:last-child td { border-bottom: 0; }
</style>
"""
    document = document.replace("</head>", f"{styles}</head>", 1)
    document = document.replace("<body>", f"<body>{summary}", 1)
    output_path.write_text(document, encoding="utf-8")


def save_comparison_dashboard(
    run_dirs: list[Path], *, output_path: Path | None = None
) -> Path:
    """Build an interactive Lets-Plot HTML dashboard comparing training runs."""
    from lets_plot import (
        aes,
        geom_line,
        gggrid,
        ggplot,
        ggsave,
        ggsize,
        labs,
        layer_tooltips,
        scale_color_brewer,
        theme,
        theme_minimal,
    )

    metrics, metadata = load_comparison_data(run_dirs)
    if output_path is None:
        output_path = run_dirs[0].parent / "comparison.html"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common_theme = theme_minimal() + theme(
        plot_title={"face": "bold", "size": 16},
        axis_title={"size": 12},
        axis_text={"size": 10},
        panel_grid_minor="blank",
        legend_position="bottom",
        legend_title="blank",
    )
    colors = scale_color_brewer(type="qual", palette="Dark2")
    loss_tooltip = (
        layer_tooltips().format("@loss_smoothed", ".3f").line("@loss_smoothed")
    )
    throughput_tooltip = (
        layer_tooltips()
        .format("@tokens_per_second_k", ".1f")
        .line("@tokens_per_second_k")
    )
    learning_rate_tooltip = (
        layer_tooltips()
        .format("@learning_rate_micro", ".2f")
        .line("@learning_rate_micro")
    )
    progress_tooltip = (
        layer_tooltips()
        .format("@tokens_processed_m", ".2f")
        .line("@tokens_processed_m")
    )

    loss_plot = (
        ggplot(
            metrics,
            aes(x="tokens_processed_m", y="loss_smoothed", color="run"),
        )
        + geom_line(size=1.3, tooltips=loss_tooltip)
        + colors
        + labs(
            title="Training loss",
            x="Tokens processed (millions)",
            y="Smoothed cross-entropy loss",
        )
        + common_theme
    )
    throughput_plot = (
        ggplot(
            metrics,
            aes(x="tokens_processed_m", y="tokens_per_second_k", color="run"),
        )
        + geom_line(size=1.2, tooltips=throughput_tooltip)
        + colors
        + labs(
            title="Average training throughput",
            x="Tokens processed (millions)",
            y="Tokens / second (thousands)",
        )
        + common_theme
    )
    learning_rate_plot = (
        ggplot(
            metrics,
            aes(x="step", y="learning_rate_micro", color="run"),
        )
        + geom_line(size=1.2, tooltips=learning_rate_tooltip)
        + colors
        + labs(
            title="Learning-rate schedule",
            x="Optimizer step",
            y="Learning rate (×10⁻⁶)",
        )
        + common_theme
    )
    progress_plot = (
        ggplot(
            metrics,
            aes(x="elapsed_minutes", y="tokens_processed_m", color="run"),
        )
        + geom_line(size=1.2, tooltips=progress_tooltip)
        + colors
        + labs(
            title="Training progress",
            x="Elapsed time (minutes)",
            y="Tokens processed (millions)",
        )
        + common_theme
    )
    dashboard = gggrid(
        [loss_plot, throughput_plot, learning_rate_plot, progress_plot],
        ncol=2,
        widths=[1, 1],
        heights=[1, 1],
        hspace=0.08,
        vspace=0.12,
        fit=False,
    ) + ggsize(1400, 1000)

    ggsave(
        dashboard,
        output_path.name,
        path=str(output_path.parent),
        iframe=False,
    )
    _inject_summary(output_path, _summary_table(metrics, metadata))
    return output_path
