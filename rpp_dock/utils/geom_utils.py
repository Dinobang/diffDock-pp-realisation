import torch
from tqdm import tqdm
import math
import numpy as np

MAX_EPS, MIN_EPS, N_EPS = 2, 0.01, 2000

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

def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))




# NOTE: generate axis by https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

def _density(
    expansion, omega, marginal=True
):  # NOTE: if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return (
            expansion / (8 * np.pi**2)
        )  
    

def _expansion(omega, eps, L=2000):
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * np.sin(omega * (l + 1 / 2))
            / np.sin(omega / 2)
        )
    return p

eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), 1000) 
omegas_array = np.linspace(0, np.pi, 2000 + 1)[1:]  
exp_vals = np.asarray([_expansion(omegas_array, eps) for eps in eps_array])
pdf_vals = np.asarray([_density(exp, omegas_array, marginal=True) for exp in exp_vals])
cdf_vals = np.asarray([pdf.cumsum() / N_EPS * np.pi for pdf in pdf_vals])

def generate_axis(eps):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * 1000
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=1000 - 1)
    x = np.random.rand()

    return np.interp(x, cdf_vals[eps_idx], omegas_array)


def generate_angle(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * generate_axis(eps)