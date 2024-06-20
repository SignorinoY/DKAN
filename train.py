from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler

from src.config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from src.datamodule import ImageDataModule
from src.module import Classifier
from src.utils import create_dirs

# constants
dataset_name = DataModuleConfig.dataset_name

# paths
cache_dir = Config.cache_dir
log_dir = Config.log_dir
ckpt_dir = Config.ckpt_dir
prof_dir = Config.prof_dir
# creates dirs to avoid failure if empty dir has been deleted
create_dirs([cache_dir, log_dir, ckpt_dir, prof_dir])

# set matmul precision
# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    accelerator: str = TrainerConfig.accelerator,
    devices: int | str = TrainerConfig.devices,
    strategy: str = TrainerConfig.strategy,
    precision: str | None = TrainerConfig.precision,
    max_epochs: int = TrainerConfig.max_epochs,
    model_name: str = ModuleConfig.model_name,
    hidden_dim: int = ModuleConfig.hidden_dim,
    lr: float = ModuleConfig.lr,
    batch_size: int = DataModuleConfig.batch_size,
    profile: bool = False,
    seed: int = Config.seed
) -> None:
    pl.seed_everything(seed)

    # ## LightningDataModule ## #
    lit_datamodule = ImageDataModule(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=batch_size
    )

    # ## LightningModule ## #
    lit_module = Classifier(
        model_name=model_name, hidden_dim=hidden_dim, lr=lr
    )

    # ## Lightning Trainer callbacks, loggers, plugins ## #

    # set logger
    logger = CSVLogger(save_dir=log_dir, name="csv-logs")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{model_name}" + "-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
        )
    ]

    # set profiler
    if profile:
        profiler = PyTorchProfiler(dirpath=prof_dir)
    else:
        profiler = None

    # ## create Trainer and call .fit ## #
    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=50
    )
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    lit_trainer.test(ckpt_path='best', datamodule=lit_datamodule)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(train, as_positional=False)
