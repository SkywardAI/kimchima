from enum import Enum
import torch
import platform

class Devices(Enum):
    Silicon = 'mps'
    CPU = 'cpu'
    # only Nvidia GPU is supported currently
    GPU = 'cuda'


def get_device():
    """
    Only support Single GPU for now
    """
    if platform.system() == 'Darwin':
        return Devices.Silicon
    elif torch.cuda.is_available():
        return Devices.GPU
    return Devices.CPU
