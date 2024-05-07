import torch
from tqdm import tqdm
import math
import numpy as np



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




import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 2000
X_N = 2000

omegas = np.linspace(0, np.pi, X_N + 1)[1:]


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * np.sin(omega * (l + 1 / 2))
            / np.sin(omega / 2)
        )
    return p


def _density(
    expansion, omega, marginal=True
):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return (
            expansion / 8 / np.pi**2
        )  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * (lo * dhi - hi * dlo)
            / lo**2
        )
    return dSigma / exp


if os.path.exists(".so3_omegas_array2.npy"):
    _omegas_array = np.load(".so3_omegas_array2.npy")
    _cdf_vals = np.load(".so3_cdf_vals2.npy")
    _score_norms = np.load(".so3_score_norms2.npy")
    _exp_score_norms = np.load(".so3_exp_score_norms2.npy")
else:
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array])
    _pdf_vals = np.asarray(
        [_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]
    )
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray(
        [
            _score(_exp_vals[i], _omegas_array, _eps_array[i])
            for i in range(len(_eps_array))
        ]
    )

    _exp_score_norms = np.sqrt(
        np.sum(_score_norms**2 * _pdf_vals, axis=1)
        / np.sum(_pdf_vals, axis=1)
        / np.pi
    )

    np.save(".so3_omegas_array2.npy", _omegas_array)
    np.save(".so3_cdf_vals2.npy", _cdf_vals)
    np.save(".so3_score_norms2.npy", _score_norms)
    np.save(".so3_exp_score_norms2.npy", _exp_score_norms)


def sample(eps):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def generate_angle(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    if torch.is_tensor(eps):
        device = eps.device
        eps = eps.cpu()
    else:
        device = None
    eps = eps.numpy()
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    norm = torch.from_numpy(_exp_score_norms[eps_idx]).float()
    if device is not None:
        norm = norm.to(device)
    return norm


