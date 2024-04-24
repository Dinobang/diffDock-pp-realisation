# добавление шума к положению белка-рецептора (поворот + перенос)
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.transforms import BaseTransform
from rpp_dock.utils.geom_utils import axis_angle_to_matrix, generate_angle

class DenoiseTransform(BaseTransform):
    def __init__(self, args):

        self.noise_maker = ...
        self.noise_sceduler = NoiseSchedule()

    def forward(self, protein, t: torch.Tensor):
        predicted_rot, predicted_t = ...
        return protein
        

class NoiseTransform(BaseTransform):
    
    def __init__(self, steps: int, args=None):

        self.noise_maker = Noise(args)
        self.noise_sceduler = NoiseSchedule()
        self.steps = steps

    def __call__(self, batch):
        for idx, protein in enumerate(batch):
            batch[idx] = self.apply_noise(protein, idx)
        return batch, self.noise_sceduler.time_steps
    
    def apply_noise(self, protein, idx_of_ligand: int, t_update=True, rot_update=True):

        for step in range(self.steps):
            t_param, rot_param, time = self.noise_maker()

            self.noise_sceduler.set_time(idx_of_ligand, protein.num_nodes, time)
            
            if t_update: 
                tr_vec = torch.normal(mean=0, std=t_param, size=(1, 3))
            if rot_update: 
                axis_angle = generate_angle(eps=rot_param)
                axis_angle = torch.from_numpy(axis_angle)
            
            upd_protein = self.apply_updates(protein, tr_vec, axis_angle)

            protein = upd_protein

        return protein


    def apply_updates(self, protein, tr_vec, axis_angle):
        center = torch.mean(protein.pos, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(axis_angle.squeeze())
        rigid_new_pos = (
            (protein.pos - center) @ rot_mat.T + tr_vec + center
        )
        protein.pos = rigid_new_pos
        return protein

        
class Noise:

    def __init__(self, args):
        if not args:
            args = {'t_min':0, 't_max':1, 'rot_min': 0, 'rot_max': 1}
        self.t_min, self.t_max = args['t_min'], args['t_max'] 
        self.rot_min, self.rot_max = args['rot_min'], args['rot_max'] 

    def __call__(self, time=None):
        if time:
            time = time
        else:
            time = np.random.uniform()
        rot_param = self.rot_min*(1-time) + self.rot_max*time
        t_param = self.t_min*(1-time) + self.t_max*time
        return t_param, rot_param, time
    

class NoiseSchedule:
    def __init__(self):
        self.time_steps = {}


    def set_time(self, num_nodes, time, idx_of_ligand: int,  device=None):
        lig_size = num_nodes
        if idx_of_ligand in self.time_steps:
            self.time_steps[idx_of_ligand].append(time * torch.ones(lig_size).to(device))
        else:
            self.time_steps[idx_of_ligand] = [time * torch.ones(lig_size).to(device)]

    

    