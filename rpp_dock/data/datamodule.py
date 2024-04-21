from pathlib import Path
from typing import cast

import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from rpp_dock.diff.transform import NoiseTransform
from rpp_dock.data.dataset import ReceptorLigandDataset, ReceptorLigandPair


class DataModule(L.LightningDataModule):
    train_dataset: ReceptorLigandDataset
    val_dataset: ReceptorLigandDataset
    batch_size: int

    def __init__(
        self, pdbdir: Path, train_csv: Path, val_csv: Path, batch_size: int, transform_steps: int
    ) -> None:
        super().__init__()
        self.transform = NoiseTransform(transform_steps)
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
    
    def collate_fn(self, data_list: list[ReceptorLigandPair]) -> tuple[Data, Data]:
        # задача функции: запаковать список ReceptorLigandPair (который, по сути, просто пара объктов класса
        # torch_geometric.data.Data) в пару объектов класса torch_geometric.data.Data, где первый содержит все графы-рецепторы,
        # второй - все графы-лиганды.
        #
        receptors, ligands = map(list, zip(*data_list))
        receptors = cast(list[Data], receptors)
        ligands = cast(list[Data], ligands)

        #

        noised_ligands, steps = self.transform(ligands)
        receptor_batch = Batch.from_data_list(receptors)
        ligand_batch = Batch.from_data_list(noised_ligands)

        # батчи получены, но в них оригинальные структуры, их ещё требуется зашумить и добавить номера шагов.
        # Можно это сделать здесь

        return receptor_batch, ligand_batch
    
