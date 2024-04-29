# добавление шума к положению белка-рецептора (поворот + перенос)
import numpy as np
import torch
import torch.nn as nn
from  pathlib import Path
from torch_geometric.transforms import BaseTransform
from rpp_dock.utils.geom_utils import axis_angle_to_matrix, generate_angle

def positions_to_file(step: int, positions : torch.Tensor, file_path: Path):
    with open(file_path, 'w') as file:
        file.write(f'STEP: {step}' + '\n')
        for row in positions:
            file.write(' '.join(map(str, row)) + '\n')

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
        self.noise_scheduler = NoiseSchedule()
        self.steps = steps

    def __call__(self, batch):

        noised_batch = self.apply_noise(batch)
        return noised_batch, self.noise_scheduler.time_steps

    
    def apply_noise(self, batch, t_update=True, rot_update=True):

        for step in range(self.steps):
            t_param, rot_param, time = self.noise_maker()
            self.noise_scheduler.set_time(batch.num_nodes, time)
            
            if t_update: 
                tr_vec = torch.normal(mean=0, std=t_param, size=(1, 3))
            if rot_update: 
                axis_angle = generate_angle(eps=rot_param)
                axis_angle = torch.from_numpy(axis_angle)
            
            noised_batch = self.apply_updates(batch, tr_vec, axis_angle)

            batch = noised_batch

            positions_to_file(step, batch.pos, f'test_positions_file_{step}.txt')

        return batch


    def apply_updates(self, batch, tr_vec, axis_angle):
        center = torch.mean(batch.pos, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(axis_angle.squeeze())
        rigid_new_pos = (
            (batch.pos - center) @ rot_mat.T + tr_vec + center
        )
        batch.pos = rigid_new_pos
        return batch

        
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
        self.time_steps = []


    def set_time(self, num_nodes, time, device=None):
        self.time_steps.append(time * torch.ones(num_nodes).to(device))


    

    