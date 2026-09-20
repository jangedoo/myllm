import os
import platform
import subprocess
from importlib import metadata
from io import StringIO
from typing import Any

import torch
from rich import box
from rich.console import Console, Group
from rich.rule import Rule
from rich.table import Table

GIB = 1024**3


def _version_or_unavailable(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _format_cudnn_version(version: int | None) -> str:
    if version is None:
        return "unavailable"
    if version >= 10_000:
        return f"{version // 10_000}.{version % 10_000 // 100}.{version % 100}"
    return f"{version // 1_000}.{version % 1_000 // 100}.{version % 100}"


def _nccl_version() -> str:
    try:
        version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        return "unavailable"
    if isinstance(version, tuple):
        return ".".join(map(str, version))
    return str(version)


def _system_memory() -> tuple[float, float] | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES") / GIB
        available = page_size * os.sysconf("SC_AVPHYS_PAGES") / GIB
    except (AttributeError, OSError, ValueError):
        return None
    return total, available


def _nvidia_smi_rows() -> list[tuple[str, ...]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [
        tuple(part.strip() for part in line.split(","))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _cuda_stack(device: torch.device) -> str:
    versions = [
        f"CUDA {torch.version.cuda or 'unavailable'}",
        f"cuDNN {_format_cudnn_version(torch.backends.cudnn.version())}",
        f"NCCL {_nccl_version()}",
        f"Triton {_version_or_unavailable('triton')}",
    ]
    if device.type == "cuda":
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        smi = next(
            (row for row in _nvidia_smi_rows() if row[0] == str(device_index)), None
        )
        if smi is not None:
            versions.insert(0, f"driver {smi[2]}")
    return " · ".join(versions[:2]) + "\n" + " · ".join(versions[2:])


def _details_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=False,
        title_style="bold cyan",
    )
    table.add_column("Setting", style="bold", no_wrap=True, width=23)
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, value)
    return table


def system_report() -> list[Any]:
    system_rows = [
        ("OS", f"{platform.system()} {platform.release()} ({platform.machine()})"),
        ("Python", platform.python_version()),
        ("CPU", platform.processor() or platform.machine()),
        ("Logical CPU cores", str(os.cpu_count() or "unknown")),
    ]
    memory = _system_memory()
    if memory is not None:
        total, available = memory
        system_rows.append(
            ("System memory", f"{available:.2f} GiB free / {total:.2f} GiB total")
        )

    software_rows = [
        ("PyTorch", str(torch.__version__)),
        ("Debug build", str(torch.version.debug)),
        ("CUDA available", "yes" if torch.cuda.is_available() else "no"),
        ("CUDA build", torch.version.cuda or "unavailable"),
        ("cuDNN", _format_cudnn_version(torch.backends.cudnn.version())),
        ("NCCL", _nccl_version()),
        ("Triton", _version_or_unavailable("triton")),
    ]

    smi_rows = _nvidia_smi_rows()
    smi_by_index = {row[0]: row for row in smi_rows}
    gpu_tables: list[Any] = [Rule("[bold cyan]GPUs")]
    if torch.cuda.is_available():
        software_rows.append(("Current device", f"cuda:{torch.cuda.current_device()}"))
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            allocated = torch.cuda.memory_allocated(index)
            reserved = torch.cuda.memory_reserved(index)
            smi = smi_by_index.get(str(index))
            driver = smi[2] if smi is not None else "unavailable"
            gpu_tables.append(
                _details_table(
                    f"GPU {index} — {properties.name}",
                    [
                        ("Driver", driver),
                        (
                            "Compute capability",
                            f"{properties.major}.{properties.minor}",
                        ),
                        ("Multiprocessors", str(properties.multi_processor_count)),
                        (
                            "VRAM",
                            (
                                f"{free_bytes / GIB:.2f} GiB free / "
                                f"{total_bytes / GIB:.2f} GiB total"
                            ),
                        ),
                        (
                            "PyTorch memory",
                            (
                                f"{allocated / GIB:.2f} GiB allocated / "
                                f"{reserved / GIB:.2f} GiB reserved"
                            ),
                        ),
                    ],
                )
            )
    elif smi_rows:
        for row in smi_rows:
            index, name, driver, total_mib, free_mib = row
            gpu_tables.append(
                _details_table(
                    f"GPU {index} — {name}",
                    [
                        ("Driver", driver),
                        ("VRAM", f"{free_mib} MiB free / {total_mib} MiB total"),
                        ("PyTorch", "CUDA device unavailable"),
                    ],
                )
            )
    else:
        gpu_tables.append(
            _details_table("GPU status", [("Status", "No CUDA GPU is available")])
        )

    return [
        _details_table("System", system_rows),
        _details_table("Software & CUDA", software_rows),
        *gpu_tables,
    ]


def training_report(model: torch.nn.Module, config: Any) -> Table:
    parameters = list(model.parameters())
    num_parameters = sum(parameter.numel() for parameter in parameters)
    num_trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    device = _model_device(model)

    if config.max_duration_sec is not None:
        stopping = f"{config.max_duration_sec / 60:g} minutes"
    elif config.max_train_tokens is not None:
        stopping = f"{config.max_train_tokens:,} tokens"
    else:
        stopping = f"{config.max_train_steps:,} optimizer steps"

    rows = [
        ("Model", type(model).__name__),
        ("Parameters", f"{num_parameters:,} total; {num_trainable:,} trainable"),
        ("Parameter memory", f"{parameter_bytes / GIB:.2f} GiB"),
        ("Parameter dtype(s)", ", ".join(parameter_dtypes) or "none"),
        ("Device", str(device)),
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        rows.append(("Device name", torch.cuda.get_device_name(device)))

    rows.extend(
        [
            ("CUDA stack", _cuda_stack(device)),
            ("Compute dtype", str(config.compute_dtype)),
            ("Micro-batch size", str(config.micro_batch_size)),
            ("Sequence length", str(config.seq_len)),
            ("Gradient accumulation", str(config.grad_accum_steps)),
            ("Effective batch size", str(config.effective_batch_size)),
            ("Tokens / update", f"{config.tokens_per_update:,}"),
            ("Stopping condition", stopping),
            ("Optimizer", "fused AdamW"),
            ("Learning rate", f"{config.lr:g}"),
            ("LR schedule", config.lr_schedule),
        ]
    )
    if config.max_grad_norm is not None:
        rows.append(("Maximum gradient norm", f"{config.max_grad_norm:g}"))
    return _details_table("Training", rows)


def _report_group(
    model: torch.nn.Module | None = None,
    trainer_config: Any | None = None,
) -> Group:
    renderables: list[Any] = [Rule("[bold cyan]MyLLM Doctor")]
    renderables.extend(system_report())
    if model is not None and trainer_config is not None:
        renderables.append(training_report(model, trainer_config))
    return Group(*renderables)


def doctor_report(
    model: torch.nn.Module | None = None,
    trainer_config: Any | None = None,
) -> str:
    output = StringIO()
    console = Console(file=output, color_system=None, width=120)
    console.print(_report_group(model=model, trainer_config=trainer_config))
    return output.getvalue().rstrip()


def print_doctor_report(
    model: torch.nn.Module | None = None,
    trainer_config: Any | None = None,
) -> None:
    Console().print(_report_group(model=model, trainer_config=trainer_config))


def print_training_report(model: torch.nn.Module, trainer_config: Any) -> None:
    Console().print(training_report(model, trainer_config))
