# добавление шума к положению белка-рецептора (поворот + перенос)
import numpy as np
import torch
from rpp_dock.utils.geom_utils import axis_angle_to_matrix

class NoiseTransform:
    
    def __init__(self, args):
        self.noise_maker = Noise(args)

    def __call__(self, protein):
        upd_protein = self.apply_noise(protein)
        return upd_protein
    
    def apply_noise(self, protein, t_update=True, rot_update=True, tor_update=False):
        t, rot, tor = self.noise_maker(protein)
        if t_update: 
            tr = torch.normal(mean=0, std=t, size=(1, 3))
        if rot_update: 
            rot = # получение вектор sample_vec(eps=rot)
            rot_update = torch.from_numpy(rot_update).float()
        if tor_update:
            tor = np.random.normal(loc=0.0,
                scale=tor, size=protein.edge_mask.sum())
        
        self.apply_updates(protein, tr, rot, tor)


    def apply_updates(self, data, tr, rot, tor):
        center = torch.mean(data.pos, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot.squeeze())
        rigid_new_pos = (
            (data.pos - center) @ rot_mat.T + tr + center
        )
        data.pos = rigid_new_pos
        return data

        


class Noise:

    def __init__(self, t_min, t_max, rot_min, rot_max, tor_min, tor_max):
        self.t_min, self.t_max = t_min, t_max 
        self.rot_min, self.rot_max = rot_min, rot_max 
        self.tor_min, self.tor_max = tor_min, tor_max

    def __call__(self):
        time = np.random.uniform()
        tor = self.tor_min*(1-time) + self.tor_max*time
        rot = self.rot_min*(1-time) + self.rot_max*time
        t = self.rot_min*(1-time) + self.rot_max*time
        return t, rot, tor
    
