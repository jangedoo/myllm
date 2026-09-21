from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from myllm import config
from myllm.v1 import MyLLMV1

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 256


class TopKSampling:
    def __init__(self, k: int):
        if k < 1:
            raise ValueError("top-k must be at least 1")
        self.k = k

    def next_token(self, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.k, logits.shape[-1])
        values, indices = torch.topk(logits, k)
        probabilities = torch.nn.functional.softmax(values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return indices.gather(-1, sampled_index).view(1, 1)


@dataclass(frozen=True)
class LoadedModel:
    model: MyLLMV1
    tokenizer: object
    device: torch.device
    eos_token_id: int | None
    context_window: int


def resolve_run_dir(*, dev: bool, tag: str) -> Path:
    """Return the artifact directory for an exact training run tag."""
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be a single folder name")

    run_dir = config.PRETRAIN_CKPT_DIR / ("dev" if dev else "full") / tag
    if not run_dir.is_dir():
        raise FileNotFoundError(f"training run not found: {run_dir}")
    return run_dir


def load_model_and_tokenizer(*, dev: bool, tag: str) -> LoadedModel:
    """Reconstruct a trained model and its tokenizer from a run directory."""
    # Keep transformers out of CLI startup; it is only imported after the first
    # actual chat message.
    from transformers import AutoTokenizer

    run_dir = resolve_run_dir(dev=dev, tag=tag)
    metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    model_config = metadata["model"]
    tokenizer_config = metadata["tokenizer"]
    model = MyLLMV1(
        vocab_size=model_config["vocab_size"],
        embed_dim=model_config["embed_dim"],
        nlayers=model_config["depth"],
        nheads=model_config["num_heads"],
        dim_ff=model_config["ff_dim"],
        padding_idx=None,
        rope_theta=model_config["rope_theta"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_id"], use_fast=True
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eos_token_id=tokenizer_config.get("eos_token_id"),
        context_window=metadata.get("training", {}).get("seq_len", 1024),
    )


@torch.inference_mode()
def generate(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Generate and decode only the tokens added after ``prompt``."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    sampler = TopKSampling(k=top_k)
    token_ids = loaded.tokenizer.encode(prompt, return_tensors="pt")
    max_new_tokens = min(max_new_tokens, max(1, loaded.context_window - 1))
    prompt_budget = max(1, loaded.context_window - max_new_tokens)
    token_ids = token_ids[:, -prompt_budget:].to(loaded.device)
    prompt_length = token_ids.shape[1]

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if loaded.device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        for _ in range(max_new_tokens):
            logits = loaded.model(token_ids)[0, -1].float() / temperature
            next_token = sampler.next_token(logits=logits).to(loaded.device)
            token_ids = torch.cat([token_ids, next_token], dim=1)
            if (
                loaded.eos_token_id is not None
                and next_token.item() == loaded.eos_token_id
            ):
                break

    generated_ids = token_ids[0, prompt_length:].cpu()
    return loaded.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _render_prompt(tokenizer: object, messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)
    # try:
    #     return tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    # except (AttributeError, TypeError, ValueError):
    #     transcript = "".join(
    #         f"{message['role'].title()}: {message['content']}\n" for message in messages
    #     )
    #     return f"{transcript}Assistant:"


def _setting_command(
    command: str,
    argument: str,
    *,
    current_temperature: float,
    current_top_k: int,
) -> tuple[float, int, str]:
    if command == "/temp":
        if not argument:
            return (
                current_temperature,
                current_top_k,
                f"temperature={current_temperature:g}",
            )
        try:
            value = float(argument)
        except ValueError:
            return current_temperature, current_top_k, "usage: /temp NUMBER"
        if value <= 0:
            return (
                current_temperature,
                current_top_k,
                "temperature must be greater than 0",
            )
        return value, current_top_k, f"temperature={value:g}"

    if not argument:
        return current_temperature, current_top_k, f"top_k={current_top_k}"
    try:
        value = int(argument)
    except ValueError:
        return current_temperature, current_top_k, "usage: /topk INTEGER"
    if value < 1:
        return current_temperature, current_top_k, "top-k must be at least 1"
    return current_temperature, value, f"top_k={value}"


def run_chat(
    *,
    dev: bool = False,
    tag: str = "latest",
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run an interactive terminal chat, loading artifacts on first message."""
    if input_fn is None:
        input_fn = input

    mode = "dev" if dev else "full"
    output_fn(f"Chatting with {mode}/{tag}.")
    output_fn("Commands: /new, /temp [NUMBER], /topk [INTEGER], /quit")

    loaded: LoadedModel | None = None
    messages: list[dict[str, str]] = []
    temperature = DEFAULT_TEMPERATURE
    top_k = DEFAULT_TOP_K

    while True:
        try:
            user_input = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0

        if not user_input:
            continue

        command, _, argument = user_input.partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command == "/quit":
            return 0
        if command == "/new":
            messages.clear()
            output_fn("Started a new conversation.")
            continue
        if command in {"/temp", "/topk"}:
            temperature, top_k, status = _setting_command(
                command,
                argument,
                current_temperature=temperature,
                current_top_k=top_k,
            )
            output_fn(status)
            continue
        if command.startswith("/"):
            output_fn(f"Unknown command: {command}")
            continue

        if loaded is None:
            output_fn(f"Loading {mode}/{tag} ...")
            try:
                loaded = load_model_and_tokenizer(dev=dev, tag=tag)
            except (
                FileNotFoundError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                output_fn(f"Could not load model: {exc}")
                return 1
            output_fn(f"Loaded on {loaded.device}.")

        messages.append({"role": "user", "content": user_input})
        prompt = _render_prompt(loaded.tokenizer, messages)
        response = generate(
            loaded,
            prompt,
            temperature=temperature,
            top_k=top_k,
        )
        output_fn(f"model> {response}")
        messages.append({"role": "assistant", "content": response})
