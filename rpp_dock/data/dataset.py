from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from rpp_dock.data.constants import *
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

        self._data: list[ReceptorLigandPair] = []
        df = pd.read_csv(data_csv)

        for path in df['path']:
            ligand = parse_pdb(pdbdir + path + '_l_b.pdb')
            receptor = parse_pdb(pdbdir + path + '_r_b.pdb')
            self._data.append(ReceptorLigandPair(receptor, ligand))

    def __getitem__(self, index: int) -> ReceptorLigandPair:
        if index < len(self._data):
            return self._data[index]
        else:
            raise IndexError('Index out of range')

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
            atoms_df["atom_name"].isin(["CA", "C", "N"])
        ].values
    )

    residue_names = atoms_df["residue_name"]
    residue_set = list(set(residue_names))
    residue_names = [residue_set.index(res_name) for res_name in residue_names]
    
    # TODO: extract residue orientations from CA-C-N atoms
    # n_c, ca_c, c_c = 0, 0, 0

    # n_coordinates = atoms_df[["x_coord", "y_coord", "z_coord"]][
    #         atoms_df["atom_name"].isin(["N"])
    #     ].values
    
    # ca_coordinates = atoms_df[["x_coord", "y_coord", "z_coord"]][
    #         atoms_df["atom_name"].isin(["CA"])
    #     ].values
    
    # c_coordinates = 


    # u_i = (n_c - ca_c) / np.linalg.norm(n_c - ca_c)
    # t_i = (c_c - ca_c) / np.linalg.norm(c_c - ca_c)
    # n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
    # v_i = np.cross(n_i, u_i)

    
    # edge_attr = ...

    # create KNN graph from euclidean distances
    edge_index = knn_graph(x=coordinates, k=10)
    # NOTE: edge index tensor describes directed edges, so we need to add
    # reverse edges to ensure our graph is treated as undirected
    edge_index = to_undirected(edge_index)

    return Data(pos=coordinates, edge_index=edge_index, x=residue_names)
