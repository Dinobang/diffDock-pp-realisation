import torch
from torch_geometric.data import Data
import random
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import BatchNorm
from rpp_dock.diff.transform import NoiseTransform, Noise, DenoiseTransform

class TPCL(nn.Module):
     def __init__(self, args,
                 in_irreps, sh_irreps, out_irreps, hidden_features,
                 residual=True, is_last_layer=False):
        super(TPCL, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.hidden_features = hidden_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_features, self.tensor_prod.weight_numel),
        )

        self.batch_norm = BatchNorm(out_irreps)

     def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        """
        edge_src, edge_dst = edge_index
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded

        out = self.batch_norm(out)
        return out
    
