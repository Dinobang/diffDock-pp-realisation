# добавление шума к положению белка-рецептора (поворот + перенос)
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.transforms import BaseTransform
from rpp_dock.utils.geom_utils import score_vec, generate_angle, axis_angle_to_matrix


def positions_to_file(step: int, positions: torch.Tensor, file_path: Path):
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

    def __init__(self, args=None):

        self.noise_maker = Noise(args)
        self.noise_scheduler = NoiseSchedule()

    def __call__(self, data):
        time = np.random.uniform()
        noised_data = self.apply_noise(data, time)
        noised_data.time_steps = self.noise_scheduler.time_steps
        return noised_data

    def apply_noise(self, data, time, t_update=True, rot_update=True):
        data.true_pos = data.pos
        t_param, rot_param = self.noise_maker(time)
        self.noise_scheduler.set_time(data.num_nodes, time)

        if t_update:
            tr_vec = torch.normal(mean=0, std=t_param, size=(1, 3))
        if rot_update:
            axis_angle = generate_angle(eps=rot_param)
            axis_angle = torch.from_numpy(axis_angle)

        noised_data = self.apply_updates(data, tr_vec, axis_angle)

        self.get_score(data, tr_vec, t_param, axis_angle, rot_param)

        return noised_data

    def get_score(self, data, tr_vec, t_param, axis_angle, rot_param):

        data.tr_score = - tr_vec / t_param ** 2
        data.rot_score = score_vec(vec=axis_angle, eps=rot_param).unsqueeze(0)

        return data

    def apply_updates(self, data, tr_vec, axis_angle):
        center = torch.mean(data.pos, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(axis_angle.squeeze())
        rigid_new_pos = (
                (data.pos - center) @ rot_mat.T + tr_vec + center
        )
        data.pos = rigid_new_pos
        return data


class Noise:

    def __init__(self, args=None):
        if not args:
            args = {'t_min': 0, 't_max': 1, 'rot_min': 0, 'rot_max': 1}
        self.t_min, self.t_max = args['t_min'], args['t_max']
        self.rot_min, self.rot_max = args['rot_min'], args['rot_max']

    def __call__(self, time=None):
        rot_param = self.rot_min * (1 - time) + self.rot_max * time
        t_param = self.t_min * (1 - time) + self.t_max * time
        return t_param, rot_param


class NoiseSchedule:
    def __init__(self):
        self.time_steps = []

    def set_time(self, num_nodes, time):
        self.time_steps.append(time)
