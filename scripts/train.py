import sys
from pathlib import Path

import lightning as L
import torch
from torch import nn

sys.path.append(str(Path.cwd()))

from mcs_prac.rpp_dock.data.datamodule import DataModule
from mcs_prac.rpp_dock.model.fake_model import DummyDenoisingModel
from mcs_prac.rpp_dock.model.lightning import Denoiser

if __name__ == "__main__":
    datamodule = DataModule(
        pdbdir="structures/",
        train_csv="splits_test.csv",
        val_csv="splits_test.csv",
        batch_size=4
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    model = Denoiser(
        model=DummyDenoisingModel(),
        optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
    )

    trainer = L.Trainer(max_epochs=10, logger=True, accelerator="cpu")
    trainer.fit(model, datamodule)
