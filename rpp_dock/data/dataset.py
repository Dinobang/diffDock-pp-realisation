from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch

from biopandas.pdb import PandasPdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

import os
import sys

sys.path.append(os.getcwd())

from rpp_dock.diff.transform import NoiseTransform
from rpp_dock.utils.geom_utils import compute_orientation_vectors
import rpp_dock.data.constants as constants


class ReceptorLigandPair(NamedTuple):
    receptor: Data
    ligand: Data
    # NOTE: do we assume that ligand protein position corresponds to docked state?
    # or do we need to store correct rigid body motion (R, t) separately?


class ReceptorLigandDataset(Dataset):
    def __init__(self, data_csv: str, pdbdir: str) -> None:
        # NOTE: data_csv contains description of receptor and ligand pairs,
        # e.g. a pair of PDB file names

        self._data: list[ReceptorLigandPair] = []
        df = pd.read_csv(data_csv)
        self.transform = NoiseTransform()

        for path in df['path']:
            ligand = parse_pdb(pdbdir + path + '_l_b.pdb')
            receptor = parse_pdb(pdbdir + path + '_r_b.pdb')
            self._data.append(ReceptorLigandPair(receptor, ligand))

    def __getitem__(self, index: int) -> ReceptorLigandPair:
        if index < len(self._data):
            pair = self._data[index]
            receptor, ligand = pair
            return ReceptorLigandPair(receptor, self.transform(ligand))
        else:
            raise IndexError('Index out of range')

    def __len__(self) -> int:
        return len(self._data)


def parse_pdb(pdb: Path) -> Data:
    pandas_pdb = PandasPdb().read_pdb(pdb)
    atoms_df = pandas_pdb.df["ATOM"]

    atoms_df = atoms_df[
        atoms_df["residue_name"].isin(constants.protein_letters_3to1_extended.keys())
    ].reset_index(drop=True)

    # extract CA atoms positions

    coordinates = torch.tensor(
        atoms_df[["x_coord", "y_coord", "z_coord"]][
            atoms_df["atom_name"].isin(["CA"])
        ].values
    )

    # extract coordinates from CA-C-N atoms

    n_coordinates = torch.tensor(atoms_df[["x_coord", "y_coord", "z_coord"]][
                                     atoms_df["atom_name"].isin(["N"])
                                 ].values)

    ca_coordinates = torch.tensor(atoms_df[["x_coord", "y_coord", "z_coord"]][
                                      atoms_df["atom_name"].isin(["CA"])
                                  ].values)

    c_coordinates = torch.tensor(atoms_df[["x_coord", "y_coord", "z_coord"]][
                                     atoms_df["atom_name"].isin(["C"])
                                 ].values)

    assert len(n_coordinates) == len(ca_coordinates) == len(c_coordinates)

    # create KNN graph from euclidean distances
    edge_index = knn_graph(x=coordinates, k=10)

    # NOTE: edge index tensor describes directed edges, so we need to add
    # reverse edges to ensure our graph is treated as undirected
    edge_index = to_undirected(edge_index)

    edge_attr = compute_orientation_vectors(n_coordinates, ca_coordinates, c_coordinates, edge_index)

    residue_names = atoms_df["residue_name"][atoms_df["atom_name"].isin(["CA"])]
    residue_list = list(set(constants.protein_letters_3to1_extended.keys()))
    residue_names = torch.tensor([residue_list.index(res_name) for res_name in residue_names])

    return Data(pos=coordinates, edge_index=edge_index, x=residue_names, edge_attr=edge_attr)
