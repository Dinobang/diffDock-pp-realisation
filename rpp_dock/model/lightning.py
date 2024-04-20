from typing import Any, Mapping, Protocol

import lightning as L
from torch import Tensor
from torch_geometric.data import Data


class DenoisingModel(Protocol):
    # интерфейс для нашей диффузионной модели.
    # Архитектуры могут быть разными, но будем считать, что все будут принимать на вход
    # пару (receptor, ligand) и возвращать что-то, что позволит обучать диффузионную модель
    def forward(self, batch: tuple[Data, Data]) -> Any:
        receptors, ligands = batch
        ...


class Denoiser(L.LightningModule):
    def __init__(self, model: DenoisingModel) -> None:
        super().__init__()
        self.model = model

    def training_step(
        self, batch: tuple[Data, Data], batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:
        # здесь мы описываем, что нужно сделать с пришедшим батчем в процессе обучения
        # т.е. реализуем алгоритм обучения из статьи про диффузионные модели
        #
        ...

    def sample(self, batch: tuple[Data, Data]) -> Any:
        # будем использовать этот метод,чтобы получать правильные положения белка-рецептора,
        # т.е. реализуем алгоритм алгоритм семплинга из диффузионной модели
        ...
