import torch
from tqdm import tqdm
from torch import tensor
from tensorboardX import SummaryWriter

LOG_DIR = './logs'

class Logger:

    def add_line(self, tag: str, value, annotation=None):
        raise Exception("Not implemented for this logger.")
    
class TensorLogger(Logger): 

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_line(self, tag: str, value, annotation=None):
        self.writer.add_scalar(tag, value)
    
    # def add_graph(self, verbose=True):
    #     self.writer.add_graph(model=None, verbose=True)

