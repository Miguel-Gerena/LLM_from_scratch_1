import numpy as np
import torch
from typing import Tuple

def get_batch(x:np.array, batch_size:int, context_length:int, device:str="cpu") -> Tuple[torch.tensor, torch.tensor]:
    length_dataset = len(x)
    max_index = length_dataset - context_length
    indices = np.random.randint(0, max_index, batch_size)
    sampled_seq = np.vstack([x[index: index + context_length] for index in indices])
    next_seq = np.vstack([x[index + 1: index + 1 + context_length] for index in indices])
    return (torch.tensor(sampled_seq, device=device), torch.tensor(next_seq, device=device))