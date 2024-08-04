from math import cos, pi
import torch
from typing import Iterable, List, Dict, Tuple
import numpy as np

def cosine_learning_warmup(time:int, alpha_max:float, alpha_min:float, time_warm:int, time_anneal:int) -> float:
    if time < time_warm:
        alpha = time/time_warm * alpha_max
    elif time_warm <= time <= time_anneal:
        alpha = alpha_min + 0.5 * ( 1 + cos((time-time_warm)/(time_anneal-time_warm)*pi)) * (alpha_max - alpha_min)
    else:
        assert time > time_anneal, f"cosine learning post annealing condition not correct t:{time} tc:{time_anneal}"
        alpha = alpha_min
    return alpha

def clip_gradient(params: Iterable[torch.nn.Parameter], maxl2:float, eps:float=1e-6):
    grads = torch.cat([p.grad for p in params if p.grad is not None])
    l2 = torch.sqrt(torch.sum(torch.square(grads)))
    if l2.item() > maxl2:
        for p in params:
            if p.requires_grad:
                p.grad[:] = torch.div(maxl2, (l2 + eps)) 

def save_checkpoint(model:torch.nn.Module, path:str, epoch:int, optim:torch.optim.Optimizer, args) -> None:
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            "args":args,
            "numpy_seed_state":np.random.get_state(),
            "torch_rng_state":torch.get_rng_state(),
            }, path)

def load_checkpoint(model:torch.nn.Module, path:str, optim:torch.optim.Optimizer)-> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    np.random.set_state(checkpoint["numpy_seed_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    return epoch

def get_norms(named_parameters:Dict[str, torch.nn.Parameter], device="cuda") -> Tuple[torch.Tensor, Dict[str, float]]:
    norms:torch.Tensor = torch.zeros(0).to(device)
    norms_dict:Dict[str, float] = {}
    for key, p in named_parameters:
        if p.grad is not None:
            norms = torch.cat([norms, torch.unsqueeze(torch.linalg.vector_norm(p), 0)])
            norms_dict[key] = norms[-1].item()
    norm = torch.mean(norms)
    return norm, norms_dict