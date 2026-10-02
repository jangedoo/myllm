# myllm

Inspired by [nanochat](https://github.com/karpathy/nanochat), I wanted to train LLM from scratch (sort of).
The goal is not necessarily to create an intelligent model, but to play with different model architectures, datasets, training loops etc.

I've tested this code on RTX Pro 4000 Blackwell (24GB) and RTX 2070 Super (8GB) GPUs.

I will share my experience and findings in my [blog](https://sanjayasubedi.com.np/)

Only pretraining is supported till now.

## Installation

Run: `uv sync`

## Usage
After installation run `myllm doctor` to print out installed libraries and check if gpu(s) are detected.


## Data Preparation

`myllm data tokenize`

It will stream FineWebEdu dataset and tokenize and save it as parquet files. Approximately 3B tokens will be in the processed dataset.
Pass `--dev` to only tokenize subset (about 300 million tokens)

Max shards is hard coded in `scripts/tokenize.py`, so update them if you want more or less.

## Pre training
Commands available for pretraining are available under `myllm pretrain`.
Below is one example to train approx 200 million parameter model on 3B token using fp8.

```bash
myllm pretrain train \
    --depth 10 \
    --max-train-tokens 3000000000 \
    --bs 32 \
    --fp8 \
    --torch-compile-mode max-autotune-no-cudagraphs
```

There are several options you can change. Not all of them needs to be passed and many of them are derived based on the values passed.
e.g. `target-tokens-per-update` is derived based on `bs, seq-len and grad-accum-steps` if they are given.
if `target-tokens-per-update, bs, seq-len` are given, `grad-accum-steps` will be derived automatically.

- `seq-len`: default 1024
- `max-train-minutes`: if specified the training will run for this many minutes and stop
- `max-train-tokens`: if specified the training will stop after processing this many tokens
- `max-train-steps`: if specified the training will stop after this many optimizer steps
- `grad-accum-steps`: number of micro batches after which one optimizer step is run
- `target-tokens-per-update`: number of tokens to process after which the optimizer will step
- `torch-compile-mode`: compile model for `torch.compile`
- `fp8`: enable fp8 training or not
- `tag`: a name for this run. default is "latest". if a folder with tag name already exists, `<tag>-00X` folder will be used

Run `myllm pretrain train --help` for all available options

### Chatting
A basic interface for chatting is available via `myllm chat --tag <tag>`

