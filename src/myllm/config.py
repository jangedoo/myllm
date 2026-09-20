import os
from pathlib import Path

ROOT_DIR = Path.home() / "myllm"
ROOT_DIR.mkdir(exist_ok=True)

PRETRAIN_DIR = ROOT_DIR / "pretrain"
PRETRAIN_DIR.mkdir(exist_ok=True)

PRETRAIN_DATA_DIR = PRETRAIN_DIR / "data"
PRETRAIN_DATA_DIR.mkdir(exist_ok=True)

PRETRAIN_CKPT_DIR = PRETRAIN_DIR / "models"
PRETRAIN_CKPT_DIR.mkdir(exist_ok=True)
