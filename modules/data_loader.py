import numpy as np
import torch
from typing import Tuple, Generator
import numpy.typing as npt

def get_batch(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu") -> Tuple[torch.tensor, torch.tensor]:
    length_dataset = len(x)
    max_index = length_dataset - context_length
    indices = np.random.randint(0, max_index, batch_size)
    sampled_seq = np.vstack([x[index: index + context_length] for index in indices])
    next_seq = np.vstack([x[index + 1: index + 1 + context_length] for index in indices])
    return (torch.as_tensor(sampled_seq, dtype=torch.long).pin_memory(), torch.as_tensor(next_seq, dtype=torch.long).pin_memory())


def train_data_generator(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu") -> Tuple[torch.tensor, torch.tensor]:
    while True:
        yield get_batch(x, batch_size, context_length, device)

def val_data_generator(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu", index:int=0) ->Tuple[torch.tensor, torch.tensor]:
    while True:
        yield get_batch(x, batch_size, context_length, device)
    