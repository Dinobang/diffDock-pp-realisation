import torch
from torch_cluster import radius, radius_graph


def cross_graph_creation(ligand, receptor, cross_dist_cutoff):
        
        if torch.is_tensor(cross_dist_cutoff):
            edge_index = radius(
                receptor.pos / cross_dist_cutoff[receptor.batch],
                ligand.pos / cross_dist_cutoff[ligand.batch],
                1,
                receptor.batch,
                ligand.batch,
                max_num_neighbors=10000,
            )
        else:
            edge_index = radius(
                receptor.pos,
                ligand.pos,
                cross_dist_cutoff,
                receptor.batch,
                ligand.batch,
                max_num_neighbors=10000,
            )
        
        # TODO
        return edge_index
    