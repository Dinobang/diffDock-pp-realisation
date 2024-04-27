import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN


class DummyDenoisingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=16)
        self.gcn = GCN(
            in_channels=16, hidden_channels=16, out_channels=16, num_layers=2
        )
        self.fc = nn.Linear(
            in_features=16 * 2, out_features=3 + 3
        )  # translation + rotation concatenated

    def forward(self, batch: tuple[Data, Data]) -> Tensor:
        receptors, ligands = batch
        receptor_embeds = self.get_protein_embeds(receptors)
        ligand_embeds = self.get_protein_embeds(ligands)
        h = torch.cat([receptor_embeds, ligand_embeds], dim=1)
        return self.fc.forward(h)

    def get_protein_embeds(self, protein: Data) -> Tensor:
        protein_aa_embeds = self.embedding.forward(protein.x)
        protein_aa_embeds = self.gcn.forward(
            x=protein_aa_embeds, edge_index=protein.edge_index, batch=protein.batch
        )
        protein_embeds = global_mean_pool(protein_aa_embeds, batch=protein.batch)
        return protein_embeds

    def compute_loss(self, pred: Tensor, true: Tensor) -> Tensor:
        return torch.sqrt(torch.mean(torch.sum((pred - true) ** 2, axis=1)))
