from typing import Any, Callable, Iterator, Mapping, Protocol

import lightning as L
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Data
from tqdm import tqdm

from rpp_dock.diff.transform import Noise, NoiseTransform


class DenoisingModel(Protocol):
    # интерфейс для нашей диффузионной модели.
    # Архитектуры могут быть разными, но будем считать, что все будут принимать на вход
    # пару (receptor, ligand) и возвращать что-то, что позволит обучать диффузионную модель
    def forward(self, batch: tuple[Data, Data]) -> Tensor:
        receptors, ligands = batch
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    def compute_loss(self, pred: Tensor, true: Tensor) -> Tensor:
        return torch.sqrt(torch.mean(torch.sum((pred - true) ** 2, axis=1)))


class Denoiser(L.LightningModule):
    def __init__(
        self,
        model: DenoisingModel,
        optimizer_fn: Callable[[Iterator[torch.nn.Parameter]], Optimizer],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer_fn(self.model.parameters())

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer

    def training_step(
        self, batch: tuple[Data, Data], batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:
        # здесь мы описываем, что нужно сделать с пришедшим батчем в процессе обучения
        # т.е. реализуем алгоритм обучения из статьи про диффузионные модели

        # подсчет функции потерь
        denoised_data = self.model.forward(batch)
        
        # получение правильных значений из батча
        _, _, _, true = batch
        loss = self.model.compute_loss(denoised_data, true)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return loss

    def sample(self, batch: tuple[Data, Data], num_steps: int, args) -> Any:
        # будем использовать этот метод,чтобы получать правильные положения белка-рецептора,
        # т.е. реализуем алгоритм алгоритм семплинга из диффузионной модели
        # NOTE: убрала итерацию по loader и по батчу

        self.model.eval()
        transformer = NoiseTransform(steps=num_steps)
        time_steps = torch.linspace(1, 0, num_steps + 1)[:-1]

        for t_step in range(num_steps):
                if torch.cuda.is_available():
                    batch = batch.cuda()

                noiser = Noise()
                tr_s, rot_s = noiser(t_step)

                with torch.no_grad():
                    output = self.model(batch)

                tr_score = output["tr_pred"].cpu()
                rot_score = output["rot_pred"].cpu()

                # градиенты
                tr_scale = torch.sqrt(
                    2 * torch.log(torch.tensor(args.tr_s_max / args.tr_s_min))
                )
                tr_g = tr_s * tr_scale

                rot_scale = torch.sqrt(
                    torch.log(torch.tensor(args.rot_s_max / args.rot_s_min))
                )
                rot_g = 2 * rot_s * rot_scale

                # пересчет
                cur_t = time_steps[t_step]
                tr_update = 0.5 * tr_g**2 * cur_t * tr_score
                rot_update = 0.5 * rot_score * cur_t * rot_g**2

                # обновление
                new_batch = transformer.apply_updates(
                        batch, tr_update, rot_update
                    )
                
                batch = new_batch

        return batch
