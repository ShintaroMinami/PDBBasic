import numpy as np
import torch

def unsqueeze(data, dim=0):
    return data.unsqueeze(dim) if torch.is_tensor(data) else np.expand_dims(np.array(data), dim)
