from typing import Any, Mapping, Protocol

import lightning as L
import torch
from tqdm import tqdm
from torch import Tensor
from torch_geometric.data import Data
from rpp_dock.diff.transform import NoiseTransform, Noise



class DenoisingModel(Protocol):
    # интерфейс для нашей диффузионной модели.
    # Архитектуры могут быть разными, но будем считать, что все будут принимать на вход
    # пару (receptor, ligand) и возвращать что-то, что позволит обучать диффузионную модель
    def forward(self, batch: tuple[Data, Data]) -> Any:
        receptors, ligands = batch
        ...
    
    def compute_loss(self, pred, true):
        return torch.sqrt(torch.mean(torch.sum((pred - true) ** 2, axis=1)))


class Denoiser(L.LightningModule):
    def __init__(self, model: DenoisingModel, optimizer, train_loader, test_loader, val_loader, log_freq=1000, lr=0.01, patience=200) -> None:
        super().__init__()
        self.model = model
        self.optimizer = ...
        self.log_freq = log_freq
        self.loader = train_loader

    def train(self, num_epochs):

        for epoch in tqdm(range(num_epochs)): 
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.training_step(batch, batch_idx, epoch)
        
        return loss


    def training_step(
        self, batch: tuple[Data, Data], batch_idx: int, epoch: int
    ) -> Tensor | Mapping[str, Any] | None:
        # здесь мы описываем, что нужно сделать с пришедшим батчем в процессе обучения
        # т.е. реализуем алгоритм обучения из статьи про диффузионные модели
        if torch.cuda.is_available():
            batch = batch.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        # подсчет функции потерь
        denoised_data = self.model(batch)
        loss = self.model.compute_loss(batch, denoised_data)

        # шаг оптимайзера
        loss.backward()
        self.optimizer.step()

        if batch_idx % self.log_freq == 0: 
            print(f'Ep: {epoch} Batch idx: {batch_idx} Loss: {loss}')
        
        return loss


    def sample(self, batch: tuple[Data, Data], num_steps: int, args) -> Any:
        # будем использовать этот метод,чтобы получать правильные положения белка-рецептора,
        # т.е. реализуем алгоритм алгоритм семплинга из диффузионной модели

        self.model.eval()
        transformer = NoiseTransform(steps=num_steps)
        time_steps = torch.linspace(1, 0, num_steps + 1)[:-1]

        for t_step in range(num_steps):
            for pair_idx, pair_graph in enumerate(self.loader):
                if torch.cuda.is_available():
                    pair_graph = pair_graph.cuda()

                noiser = Noise()
                tr_s, rot_s, tor_s = noiser(t_step)

                with torch.no_grad():
                    output = self.model(pair_graph)

                tr_score = output["tr_pred"].cpu()
                rot_score = output["rot_pred"].cpu()


                # градиенты
                tr_scale = torch.sqrt(
                2 * torch.log(torch.tensor(args.tr_s_max /
                                           args.tr_s_min)))
                tr_g = tr_s * tr_scale


                rot_scale = torch.sqrt(
                    torch.log(torch.tensor(args.rot_s_max /
                                           args.rot_s_min)))
                rot_g = 2 * rot_s * rot_scale

                # пересчет
                cur_t = time_steps[t_step]
                tr_update = (0.5 * tr_g**2 * cur_t * tr_score)
                rot_update = (0.5 * rot_score * cur_t * rot_g**2)

                # обновление
                updates = []
                for protein in pair_graph:
                    new_protein = transformer.apply_updates(protein, tr_update, rot_update)
                    updates.append(new_protein)

        return updates

