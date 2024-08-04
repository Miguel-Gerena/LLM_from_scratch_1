import numpy as np
import torch
from typing import Tuple
import numpy.typing as npt
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

class Data_Iterator(Dataset):
    def __init__(self, numpy_arr:str, context_length:int):
        self.data = np.load(numpy_arr, mmap_mode="r").astype(np.int32)
        self.length_dataset = len(self.data)
        self.context_length = context_length
        self.max_index = self.length_dataset - context_length

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled_seq = self.data[idx: idx + self.context_length]
        next_seq = self.data[idx + 1: idx + 1 + self.context_length]
        return (torch.as_tensor(sampled_seq, dtype=torch.long), torch.as_tensor(next_seq, dtype=torch.long))

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
    print("train generator exited")

def val_data_generator(x:npt.NDArray, batch_size:int, context_length:int, device:str="cpu", index:int=0) ->Tuple[torch.tensor, torch.tensor]:
    while True:
        yield get_batch(x, batch_size, context_length, device)
    print("val  generator exited")
    
    