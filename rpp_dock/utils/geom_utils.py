import torch
from tqdm import tqdm

def compute_orientation_vectors(n_coordinates, ca_coordinates, c_coordinates, edge_index):

    # T = [SO3 p]
    #     [ 0  1] 

    u_i = (n_coordinates - ca_coordinates) / torch.linalg.norm(n_coordinates - ca_coordinates)
    t_i = (c_coordinates - ca_coordinates) / torch.linalg.norm(c_coordinates - ca_coordinates)
    n_i = torch.cross(u_i, t_i) / torch.linalg.norm(torch.cross(u_i, t_i))
    v_i = torch.cross(n_i, u_i)


    edge_attr = []

    for i in tqdm(range(len(edge_index[0]))):

        src, dst = edge_index[0][i], edge_index[1][i]
        src_u_i, dst_u_i = u_i[src, :], u_i[dst, :]
        src_v_i, dst_v_i = v_i[src, :], v_i[dst, :]
        src_n_i, dst_n_i = n_i[src, :], n_i[dst, :]

        T1 = torch.stack(
                          (torch.cat((src_n_i, ca_coordinates[src][0].unsqueeze(dim=0))), 
                          torch.cat((src_u_i, ca_coordinates[src][1].unsqueeze(dim=0))),
                          torch.cat((src_v_i, ca_coordinates[src][2].unsqueeze(dim=0))), 
                          torch.tensor([0, 0, 0, 1]))
                        )
        
        T2 = torch.stack(
                          (torch.cat((dst_n_i, ca_coordinates[dst][0].unsqueeze(dim=0))), 
                          torch.cat((dst_u_i, ca_coordinates[dst][1].unsqueeze(dim=0))),
                          torch.cat((dst_v_i, ca_coordinates[dst][2].unsqueeze(dim=0))), 
                          torch.tensor([0, 0, 0, 1]))
                        )
    
        edge_attr.append(torch.linalg.inv(T1) @ T2)
    
    return edge_attr