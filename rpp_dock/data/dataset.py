from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from rpp_dock.utils.geom_utils import compute_orientation_vectors
from rpp_dock import constants


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
        atoms_df["residue_name"].isin(constants.protein_letters_3to1_extended.keys())
    ].reset_index(drop=True)

    # extract CA atoms positions

    coordinates = torch.tensor(
        atoms_df[["x_coord", "y_coord", "z_coord"]][
            atoms_df["atom_name"].isin(["CA"])
        ].values
    )

    # NOTE: здесь всё ещё аминокислотных остатков столько же,
    # сколько атомов, есть избыточность.
    # см выборку координат ниже, достаточно взять один атом внутри аминокислоты
    # (например CA), как-то так:
    # atoms_df["residue_name"][atoms_df["atom_name"].isin(["CA"])]
    residue_names = atoms_df["residue_name"][atoms_df["atom_name"].isin(["CA"])]

    # то, что ниже, рискованно: при разных запусках у аминокислот могут быть разные индексы,
    # некоторые аминокислоты могут вообще в белке отсутствовать. лучше использовать какую-нибудь
    # константу, где точно все аминокислоты перечислены
    residue_dict = constants.protein_letters_3to1_extended

    # NOTE: residue_names должен быть тензором с shape = (n_residues,)
    # NOTE: у меня почему-то не получается сделать здесь тензор
    residue_names = [residue_dict[name] for name in residue_names]

    
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
    print(edge_index)

    # NOTE: edge index tensor describes directed edges, so we need to add
    # reverse edges to ensure our graph is treated as undirected
    edge_index = to_undirected(edge_index)

    edge_attr = compute_orientation_vectors(n_coordinates, ca_coordinates, c_coordinates, edge_index)

    # # FIXME: здесь я временно частично исправлю ошибку, чтобы код обучения запускался
    # residue_names = atoms_df["residue_name"][atoms_df["atom_name"].isin(["CA"])]
    # residue_set = list(set(residue_names))
    # residue_names = torch.tensor([residue_set.index(res_name) for res_name in residue_names])

    return Data(pos=coordinates, edge_index=edge_index, x=residue_names, edge_attr=edge_attr)


