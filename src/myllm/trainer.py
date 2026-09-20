import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from tqdm.auto import tqdm

from myllm.doctor import print_training_report


def format_token_count(num_tokens: float) -> str:
    value = float(num_tokens)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1_000 or suffix == "T":
            if not suffix:
                return f"{value:,.0f}"
            if abs(value) >= 100:
                return f"{value:.0f}{suffix}"
            if abs(value) >= 10:
                return f"{value:.1f}{suffix}"
            return f"{value:.2f}{suffix}"
        value /= 1_000

    raise RuntimeError("Unreachable")


@dataclass
class StepDetail:
    step_num: int
    start_time: float
    end_time: float
    num_tokens: float
    loss: float


@dataclass
class TrainerState:
    start_time: float = 0
    current_time: float = 0
    current_step: int = 0
    tokens_processed: int = 0

    step_history: list[StepDetail] = field(default_factory=list)

    @property
    def elapsed_sec(self):
        return self.current_time - self.start_time

    @property
    def tokens_per_second(self):
        if self.elapsed_sec <= 0:
            return 0.0
        return self.tokens_processed / self.elapsed_sec

    def update_start_time(self):
        self.start_time = time.perf_counter()

    def update_current_time(self):
        self.current_time = time.perf_counter()

    def step_complete(self):
        self.update_current_time()
        self.current_step += 1

    def add_tokens_processed(self, new_tokens_processed: int):
        self.tokens_processed += new_tokens_processed


class BaseInterrupt:
    def should_stop(self, state: TrainerState):
        raise NotImplementedError()

    def get_progress(self, state: TrainerState):
        raise NotImplementedError()


@dataclass
class TimedInterrupt(BaseInterrupt):
    max_duration_sec: float

    def should_stop(self, state: TrainerState, **kwargs):
        return state.elapsed_sec >= self.max_duration_sec

    def get_progress(self, state: TrainerState):
        if state.elapsed_sec == 0:
            return 0.0
        return min(state.elapsed_sec / self.max_duration_sec, 1.0)


@dataclass
class StepInterrupt(BaseInterrupt):
    max_steps: int

    def should_stop(self, state: TrainerState, **kwargs):
        return state.current_step >= self.max_steps

    def get_progress(self, state: TrainerState):
        if state.current_step == 0:
            return 0.0
        return min(state.current_step / self.max_steps, 1.0)


@dataclass
class TokenInterrupt(BaseInterrupt):
    max_tokens: int

    def should_stop(self, state: TrainerState, **kwargs):
        return state.tokens_processed >= self.max_tokens

    def get_progress(self, state: TrainerState):
        if state.tokens_processed == 0:
            return 0.0
        return min(state.tokens_processed / self.max_tokens, 1.0)


class Callback:
    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass

    def on_train_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass

    def on_batch_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass

    def on_batch_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass

    def on_micro_batch_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass

    def on_micro_batch_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        pass


@dataclass
class TrainingConfig:
    # data
    micro_batch_size: int
    seq_len: int

    # stopping criterias
    max_train_minutes: float | None = None
    max_train_tokens: int | None = None
    max_train_steps: int | None = None

    # optimization
    grad_accum_steps: int | None = None
    target_tokens_per_update: int | None = None
    lr: float = 1e-4
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    max_grad_norm: float | None = None
    compute_dtype: torch.dtype = torch.bfloat16

    def resolve(self) -> "ResolvedTrainerConfig":
        return ResolvedTrainerConfig.from_training_config(self)


@dataclass
class ResolvedTrainerConfig:
    micro_batch_size: int
    seq_len: int
    grad_accum_steps: int

    max_duration_sec: float | None = None
    max_train_tokens: int | None = None
    max_train_steps: int | None = None

    lr: float = 1e-4
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    max_grad_norm: float | None = None
    compute_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_training_config(cls, config: TrainingConfig):
        stop_values = (
            config.max_train_minutes,
            config.max_train_tokens,
            config.max_train_steps,
        )
        if sum(value is not None for value in stop_values) != 1:
            raise ValueError(
                "Set exactly one of max_train_minutes, max_train_tokens, or "
                "max_train_steps"
            )

        if any(value is not None and value <= 0 for value in stop_values):
            raise ValueError("The selected training limit must be positive")

        if config.micro_batch_size <= 0 or config.seq_len <= 1:
            raise ValueError(
                "micro_batch_size must be positive and seq_len must exceed 1"
            )

        tokens_per_micro_batch = config.micro_batch_size * (config.seq_len - 1)

        if config.grad_accum_steps is not None:
            if config.grad_accum_steps <= 0:
                raise ValueError("grad_accum_steps must be positive")
            grad_accum_steps = config.grad_accum_steps
        elif config.target_tokens_per_update is not None:
            if config.target_tokens_per_update <= 0:
                raise ValueError("target_tokens_per_update must be positive")
            grad_accum_steps = math.ceil(
                config.target_tokens_per_update / tokens_per_micro_batch
            )
        else:
            raise ValueError("Set grad_accum_steps or target_tokens_per_update")

        if not 0 <= config.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= config.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if config.lr <= 0:
            raise ValueError("lr must be positive")
        if config.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")

        tokens_per_update = tokens_per_micro_batch * grad_accum_steps
        max_train_steps = config.max_train_steps
        if config.max_train_tokens is not None:
            max_train_steps = math.ceil(config.max_train_tokens / tokens_per_update)

        return cls(
            micro_batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            grad_accum_steps=grad_accum_steps,
            max_duration_sec=(
                config.max_train_minutes * 60
                if config.max_train_minutes is not None
                else None
            ),
            max_train_tokens=config.max_train_tokens,
            max_train_steps=max_train_steps,
            lr=config.lr,
            lr_schedule=config.lr_schedule,
            warmup_ratio=config.warmup_ratio,
            min_lr_ratio=config.min_lr_ratio,
            max_grad_norm=config.max_grad_norm,
            compute_dtype=config.compute_dtype,
        )

    def make_interrupt(self) -> BaseInterrupt:
        if self.max_duration_sec is not None:
            return TimedInterrupt(max_duration_sec=self.max_duration_sec)
        if self.max_train_tokens is not None:
            return TokenInterrupt(max_tokens=self.max_train_tokens)
        if self.max_train_steps is not None:
            return StepInterrupt(max_steps=self.max_train_steps)
        raise RuntimeError("Resolved config has no stopping criterion")

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def tokens_per_micro_batch(self):
        return self.micro_batch_size * (self.seq_len - 1)

    @property
    def tokens_per_update(self):
        return self.tokens_per_micro_batch * self.grad_accum_steps

    @property
    def total_train_tokens(self):
        if self.max_train_tokens is not None:
            return self.max_train_tokens
        if self.max_train_steps is not None:
            return self.max_train_steps * self.tokens_per_update
        return None


class TqdmCallback(Callback):
    def update_progress(self, trainer: "Trainer", loss=None):
        progress = round(trainer.progress, 2)
        self.progress_pbar.update(progress - self.progress_pbar.n)

        state = trainer.state
        processed = format_token_count(state.tokens_processed)
        total = trainer.config.total_train_tokens
        token_progress = (
            f"{processed}/{format_token_count(total)}"
            if total is not None
            else processed
        )
        loss_text = f"{loss:.4f}" if loss is not None else "--"
        self.progress_pbar.set_postfix_str(
            f"loss={loss_text}, "
            f"step={state.current_step}, "
            f"tokens={token_progress}, "
            f"avg tokens/s={format_token_count(state.tokens_per_second)}"
        )

    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.progress_pbar = tqdm(total=1.0, desc="Training Progress", position=0)
        self.update_progress(trainer)

    def on_batch_end(
        self, state: TrainerState, trainer: "Trainer", loss, num_tokens, **kwargs
    ):
        self.update_progress(trainer, loss=loss)

    def on_train_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.progress_pbar.close()


class StateUpdateCallback(Callback):
    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        state.update_start_time()
        state.update_current_time()

    def on_batch_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        state.update_current_time()

    def on_batch_end(
        self, state: TrainerState, trainer: "Trainer", loss, num_tokens: int, **kwargs
    ):
        state.step_complete()

    def on_micro_batch_end(
        self, state: TrainerState, trainer: "Trainer", num_tokens: int, **kwargs
    ):
        state.update_current_time()
        state.add_tokens_processed(num_tokens)


def save_training_metrics_dashboard(metrics_path: Path):
    import pandas as pd
    from lets_plot import (
        aes,
        geom_line,
        gggrid,
        ggplot,
        ggsave,
        ggsize,
        labs,
        theme,
        theme_minimal,
    )

    metrics = pd.read_json(metrics_path, lines=True)
    metrics = metrics.loc[metrics["split"] == "train"].copy()
    if metrics.empty:
        return

    smoothing_window = min(20, max(1, len(metrics) // 10))
    metrics["loss_smoothed"] = metrics["loss"].rolling(
        smoothing_window, min_periods=1
    ).mean()
    metrics["tokens_per_second_k"] = metrics["tokens_per_second"] / 1_000
    metrics["tokens_processed_m"] = metrics["tokens_processed"] / 1_000_000
    metrics["learning_rate_micro"] = metrics["learning_rate"] * 1_000_000

    common_theme = theme_minimal() + theme(
        plot_title={"face": "bold", "size": 16},
        axis_title={"size": 12},
        axis_text={"size": 10},
        panel_grid_minor="blank",
    )
    loss_plot = (
        ggplot(metrics, aes(x="step"))
        + geom_line(aes(y="loss"), color="#94a3b8", alpha=0.5, size=0.7)
        + geom_line(aes(y="loss_smoothed"), color="#2563eb", size=1.4)
        + labs(
            title="Training loss",
            subtitle=f"Raw loss and {smoothing_window}-step rolling average",
            x="Optimizer step",
            y="Cross-entropy loss",
        )
        + common_theme
    )
    throughput_plot = (
        ggplot(metrics, aes(x="step", y="tokens_per_second_k"))
        + geom_line(color="#059669", size=1.2)
        + labs(
            title="Average training throughput",
            subtitle="Cumulative average since training began",
            x="Optimizer step",
            y="Tokens / second (thousands)",
        )
        + common_theme
    )
    tokens_plot = (
        ggplot(metrics, aes(x="step", y="tokens_processed_m"))
        + geom_line(color="#7c3aed", size=1.2)
        + labs(
            title="Total tokens consumed",
            x="Optimizer step",
            y="Tokens (millions)",
        )
        + common_theme
    )
    learning_rate_plot = (
        ggplot(metrics, aes(x="step", y="learning_rate_micro"))
        + geom_line(color="#d97706", size=1.2)
        + labs(
            title="Learning-rate schedule",
            x="Optimizer step",
            y="Learning rate (×10⁻⁶)",
        )
        + common_theme
    )
    dashboard = gggrid(
        [loss_plot, throughput_plot, tokens_plot, learning_rate_plot],
        ncol=2,
        widths=[1, 1],
        heights=[1, 1],
        hspace=0.08,
        vspace=0.12,
        fit=False,
    ) + ggsize(1400, 1000)

    ggsave(
        dashboard,
        metrics_path.with_suffix(".html").name,
        path=str(metrics_path.parent),
        iframe=False,
    )


class MetricsCallback(Callback):
    def __init__(self, metrics_path: Path):
        self.metrics_path = metrics_path
        self.metrics_file = None

    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_path.open("w", encoding="utf-8")

    def on_batch_end(
        self, state: TrainerState, trainer: "Trainer", loss, num_tokens: int, **kwargs
    ):
        loss_value = loss.detach().item() if isinstance(loss, torch.Tensor) else loss
        record = {
            "split": "train",
            "step": state.current_step,
            "loss": float(loss_value),
            "learning_rate": trainer.optim.param_groups[0]["lr"],
            "num_tokens": num_tokens,
            "tokens_processed": state.tokens_processed,
            "elapsed_seconds": state.elapsed_sec,
            "tokens_per_second": state.tokens_per_second,
        }
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

    def on_train_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.metrics_file.close()
        save_training_metrics_dashboard(self.metrics_path)


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def on_train_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optim.state_dict(),
                "lr_scheduler": trainer.lr_scheduler.state_dict(),
                "trainer_state": {
                    "step": state.current_step,
                    "tokens_processed": state.tokens_processed,
                },
                "training_config": asdict(trainer.config),
            },
            self.checkpoint_path,
        )


def cycle_data_loader(train_dl: torch.utils.data.DataLoader):
    while True:
        yielded_batch = False
        for batch in train_dl:
            yielded_batch = True
            yield batch
        if not yielded_batch:
            raise RuntimeError("The training data loader is empty")


class Trainer:
    def __init__(
        self,
        config: TrainingConfig | ResolvedTrainerConfig,
        model: torch.nn.Module,
        callbacks: list[Callback] | None = None,
    ):
        self.state = TrainerState()
        if isinstance(config, TrainingConfig):
            config = config.resolve()
        self.config = config

        self.training_interrupt = self.config.make_interrupt()
        self.callbacks = [
            StateUpdateCallback(),
            TqdmCallback(),
            *(callbacks or []),
        ]
        self.model = model
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, fused=True
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lambda _: self._lr_multiplier(self.progress),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.loss_fn.cuda().compile()

    @property
    def progress(self):
        return self.training_interrupt.get_progress(state=self.state)

    def _lr_multiplier(self, progress: float):
        if self.config.lr_schedule == "constant":
            return 1.0

        if progress < self.config.warmup_ratio:
            return progress / self.config.warmup_ratio

        decay_progress = (progress - self.config.warmup_ratio) / (
            1 - self.config.warmup_ratio
        )
        cosine_multiplier = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return (
            self.config.min_lr_ratio
            + (1 - self.config.min_lr_ratio) * cosine_multiplier
        )

    def compute_loss_micro_batch(self, batch):
        token_ids = batch["token_ids"].cuda(non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=self.config.compute_dtype):
            model_out = self.model(token_ids)
            bs, seq_len, vocab_size = model_out.shape

            labels = torch.full_like(token_ids, -100, device=model_out.device)
            labels[:, :-1] = token_ids[:, 1:]
            labels = labels.view(-1)
            logits = model_out.view(-1, vocab_size)

            loss = self.loss_fn(logits, labels)

        loss.backward()
        # since there are no padded items in the batch, we can simply
        # say num_tokens = bs * (seq_len-1). -1 since we shift the labels
        num_tokens = bs * (seq_len - 1)
        return loss, num_tokens

    def accumulate_gradients(self, data_iter):
        self.call_callbacks("on_batch_begin")

        self.optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0

        for _ in range(self.config.grad_accum_steps):
            batch = next(data_iter)
            self.call_callbacks("on_micro_batch_begin")

            loss_sum, num_tokens = self.compute_loss_micro_batch(batch)
            total_loss += loss_sum
            total_tokens += num_tokens

            self.call_callbacks(
                "on_micro_batch_end",
                num_tokens=num_tokens,
            )

            if self.training_interrupt.should_stop(self.state):
                break

        return total_loss, total_tokens

    def optimizer_step(self, num_tokens: int):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(num_tokens)

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        self.optim.step()

    def train(self, train_dl: torch.utils.data.DataLoader):
        torch.set_float32_matmul_precision("high")

        print_training_report(model=self.model, trainer_config=self.config)
        self.call_callbacks("on_train_begin")

        self.model.train()

        train_data_iter = cycle_data_loader(train_dl)

        while not self.training_interrupt.should_stop(state=self.state):
            # start a new gradient accumulation cycle
            loss_sum, num_tokens = self.accumulate_gradients(train_data_iter)
            # update optimizer progress
            self.optimizer_step(num_tokens=num_tokens)

            self.call_callbacks(
                "on_batch_end",
                loss=loss_sum / num_tokens,
                num_tokens=num_tokens,
            )
            self.lr_scheduler.step()

        self.call_callbacks("on_train_end")

    def call_callbacks(self, event: str, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                state=self.state,
                trainer=self,
                **kwargs,
            )
