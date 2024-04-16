import torch
from torch_geometric.data import Data
import random
from rpp_dock.diff.transform import NoiseTransform, Noise, DenoiseTransform

def forward_diffusion(x_t: Data, noiser: NoiseTransform):
    #TODO: добавление шума с помощью NoiseTransform, q_xt, n_steps?
    transformator = NoiseTransform({'t_min':0, 't_max':1, 'rot_min': 0, 'rot_max': 1})

    ligand, t = transformator(x_t)
    transformator.noise_sceduler.set_time(x_t.num_nodes, t)
    return ligand



def backward_diffusion(model, x_t: Data, t):
    transformator = DenoiseTransform()
    #TODO предикт шума с помощью модели и его отмена 
    
