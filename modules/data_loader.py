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
    return (torch.tensor(sampled_seq, device=device, dtype=torch.long), torch.tensor(next_seq, device=device, dtype=torch.long))

def get_val_batch(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu", index:int = 0) -> Tuple[torch.tensor, torch.tensor]:
    length_dataset = len(x)
    max_index = length_dataset - context_length
    if index >= max_index:
        raise StopIteration
    indices = range(index, index + batch_size)
    sampled_seq = np.vstack([x[index: index + context_length] for index in indices])
    next_seq = np.vstack([x[index + 1: index + 1 + context_length] for index in indices])
    return (torch.tensor(sampled_seq, device=device), torch.tensor(next_seq, device=device), index)

def train_data_generator(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu") -> Tuple[torch.tensor, torch.tensor]:
    yield get_batch(x, batch_size, context_length, device)

def val_data_generator(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu", index:int=0) ->Tuple[torch.tensor, torch.tensor]:
    x, y, index = get_batch(x, batch_size, context_length, device, index)
    yield (x, y)