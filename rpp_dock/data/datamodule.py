from pathlib import Path
from typing import cast

import lightning as L
import os
import sys

import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from pathlib import Path

sys.path.append(str(Path.cwd()))

from rpp_dock.data.dataset import ReceptorLigandDataset, ReceptorLigandPair


def positions_to_file(step: int, positions: torch.Tensor, file_path: Path):
    with open(file_path, 'w') as file:
        file.write(f'STEP: {step}' + '\n')
        for row in positions:
            file.write(' '.join(map(str, row)) + '\n')


class DataModule(L.LightningDataModule):
    train_dataset: ReceptorLigandDataset
    val_dataset: ReceptorLigandDataset
    batch_size: int

    def __init__(
            self, pdbdir, train_csv, val_csv, batch_size: int
    ) -> None:
        super().__init__()
        self.pdbdir = pdbdir
        self.train_csv = train_csv
        self.batch_size = batch_size
        self.val_csv = val_csv

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ReceptorLigandDataset(self.train_csv, self.pdbdir)
            ...

        elif stage == "validate":
            self.val_dataset = ReceptorLigandDataset(self.val_csv, self.pdbdir)
            ...

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, data_list: list[ReceptorLigandPair]) -> tuple[Batch, Batch]:

        receptors, ligands = map(list, zip(*data_list))
        receptors = cast(list[Data], receptors)
        ligands = cast(list[Data], ligands)

        receptor_batch = Batch.from_data_list(receptors)
        ligand_batch = Batch.from_data_list(ligands)

        return receptor_batch, ligand_batch
