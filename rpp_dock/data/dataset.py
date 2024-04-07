from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from biopandas.constants import protein_letters_3to1_extended
from biopandas.pdb import PandasPdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


class ReceptorLigandPair(NamedTuple):
    receptor: Data
    ligand: Data
    # NOTE: do we assume that ligand protein position corresponds to docked state?
    # or do we need to store correct rigid body motion (R, t) separately?


class ReceptorLigandDataset(Dataset):
    def __init__(self, data_csv: Path, pdbdir: Path) -> None:
        # NOTE: data_csv contains description of receptor and ligand pairs,
        # e.g. a pair of PDB file names

        # TODO: parse all examples from
        self._data: list[ReceptorLigandPair] = []

    def __getitem__(self, index: int) -> ReceptorLigandPair:
        ...

    def __len__(self) -> int:
        return len(self._data)


def parse_pdb(pdb: Path) -> Data:
    pandas_pdb = PandasPdb().read_pdb(pdb)
    atoms_df = pandas_pdb.df["ATOM"]

    atoms_df = atoms_df[
        atoms_df["residue_name"].isin(protein_letters_3to1_extended.keys())
    ].reset_index(drop=True)

    # extract CA atoms positions

    coordinates = torch.tensor(
        atoms_df[["x_coord", "y_coord", "z_coord"]][
            atoms_df["atom_name"].isin(["CA"])
        ].values
    )
    # TODO: extract residue names and encode to integer indices (e.g. ALA = 0, CYS = 1 etc)
    # TODO: extract residue orientations from CA-C-N atoms

    # create KNN graph from euclidean distances
    edge_index = knn_graph(x=coordinates, k=10)
    # NOTE: edge index tensor describes directed edges, so we need to add
    # reverse edges to ensure our graph is treated as undirected
    edge_index = to_undirected(edge_index)

    return Data(pos=coordinates, edge_index=edge_index)
