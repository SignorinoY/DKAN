from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

# ## get root path ## #
this_file = Path(__file__)
this_project_idx = [
    i for i, j in enumerate(this_file.parents) if j.name.endswith("TrActv")
][0]
this_project = this_file.parents[this_project_idx]

@dataclass
class Config:
    cache_dir: str = os.path.join(this_project, "data")
    log_dir: str = os.path.join(this_project, "logs")
    ckpt_dir: str = os.path.join(this_project, "checkpoints")
    prof_dir: str = os.path.join(this_project, "logs", "profiler")
    seed: int = 0

@dataclass
class ModuleConfig:
    model_name: str = "KAN"
    hidden_dim: int = 1
    lr: float = 5e-2

@dataclass
class DataModuleConfig:
    dataset_name: str = "type2"
    input_dim: int = 2
    num_classes: int = 10
    batch_size: int = 64
    train_split: str = "train"
    test_split: str = "test"
    train_size: float = 0.9
    num_workers: int = min(16, cpu_count())

@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | None = "16-mixed"
    max_epochs: int = 1000
