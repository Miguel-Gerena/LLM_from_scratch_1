from math import cos, pi
import torch
from typing import Iterable

def cosine_learning_warmup(t:int, alpha_max:float, alpha_min:float, t_w:int, t_c:int) -> float:
    if t < t_w:
        alpha = t/t_w * alpha_max
    elif t_w <= t <= t_c:
        alpha = alpha_min + 0.5 * ( 1 + cos((t-t_w)/(t_c-t_w)*pi)) * (alpha_max - alpha_min)
    else:
        assert t > t_c, f"cosine learning post annealing condition not correct t:{t} tc:{t_c}"
        alpha = alpha_min
    return alpha

def clip_gradient(params: Iterable[torch.nn.Parameter], maxl2:float, eps:float=1e-6):
    grads = torch.cat([p.grad for p in params if p.grad is not None])
    l2 = torch.sqrt(torch.sum(torch.square(grads)))
    if l2.item() > maxl2:
        for p in params:
            if p.requires_grad:
                p.grad[:] = torch.div(maxl2, (l2 + eps)) 
