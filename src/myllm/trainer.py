import time
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm


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
    max_duration_sec: int

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
    # stopping criterias
    max_train_minutes: int | None = None
    max_train_tokens: int | None = None
    max_train_steps: int | None = None

    # optimization
    target_tokens_per_update: int | None = None
    lr: float = 1e-4


@dataclass
class ResolvedTrainerConfig:
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int
    max_grad_norm: float | None = None
    compute_dtype = torch.bfloat16

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.grad_accum_steps


class TqdmCallback(Callback):
    def update_progress(self, trainer: "Trainer"):
        progress = trainer.progress
        self.progress_pbar.update(progress - self.progress_pbar.n)

    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.progress_pbar = tqdm(total=1.0, desc="Training Progress", position=0)
        self.batch_pbar = tqdm(desc="Optimizer steps", unit="step", position=1)
        self.micro_batch_pbar = tqdm(
            desc="Micro Batch",
            unit="micro batch",
            total=trainer.config.grad_accum_steps,
            position=2,
            leave=False,
        )

    def on_micro_batch_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.micro_batch_pbar.update(1)
        self.update_progress(trainer)

    def on_batch_end(
        self, state: TrainerState, trainer: "Trainer", loss, num_tokens, **kwargs
    ):
        self.batch_pbar.update(1)
        self.batch_pbar.set_postfix(
            loss=f"{loss:.4f}", tokens=num_tokens, total_tokens=state.tokens_processed
        )
        self.micro_batch_pbar.reset(trainer.config.grad_accum_steps)
        self.update_progress(trainer)

    def on_train_end(self, state: TrainerState, trainer: "Trainer", **kwargs):
        self.progress_pbar.close()
        self.batch_pbar.close()
        self.micro_batch_pbar.close()


class StateUpdateCallback(Callback):
    def on_train_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        state.update_start_time()
        state.update_current_time()

    def on_batch_begin(self, state: TrainerState, trainer: "Trainer", **kwargs):
        state.update_current_time()

    def on_batch_end(
        self, state: TrainerState, trainer: "Trainer", loss, num_tokens: int, **kwargs
    ):
        state.update_current_time()
        state.step_complete()

    def on_micro_batch_end(
        self, state: TrainerState, trainer: "Trainer", num_tokens: int, **kwargs
    ):
        state.update_current_time()
        state.add_tokens_processed(num_tokens)


class Trainer:
    def __init__(self, config: ResolvedTrainerConfig, model: torch.nn.Module):
        self.state = TrainerState()
        self.config = config

        self.training_interrupt: BaseInterrupt = StepInterrupt(max_steps=100)
        self.callbacks = [
            StateUpdateCallback(),
            TqdmCallback(),
        ]
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.001, fused=True)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.loss_fn.cuda().compile()

    @property
    def progress(self):
        return self.training_interrupt.get_progress(state=self.state)

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

        self.call_callbacks(
            "on_batch_end",
            loss=total_loss / total_tokens,
            num_tokens=total_tokens,
        )

        return total_loss, total_tokens

    def optimizer_step(self, num_tokens: int):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(num_tokens)

        if self.config.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        # TODO: use learning rate scheduler here
        # to figure out what lr to use
        self.optim.step()

    def train(self, train_dl: torch.utils.data.DataLoader):
        self.call_callbacks("on_train_begin")

        self.model.train()

        train_data_iter = iter(train_dl)

        while not self.training_interrupt.should_stop(state=self.state):
            # start a new gradient accumulation cycle
            loss_sum, num_tokens = self.accumulate_gradients(train_data_iter)
            # update optimizer progress
            self.optimizer_step(num_tokens=num_tokens)

        self.call_callbacks("on_train_end")

    def call_callbacks(self, event: str, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                state=self.state,
                trainer=self,
                **kwargs,
            )
