import os
import torch
import numpy as np
import random

def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print('log dir already existing - overwrite')
        pass
    return dir_path

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)