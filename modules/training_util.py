from math import cos, pi
import torch
from typing import Iterable

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
            }, path)

def load_checkpoint(model:torch.nn.Module, path:str, optim:torch.optim.Optimizer)-> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
