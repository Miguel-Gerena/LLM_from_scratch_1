from numpy import dtype
from torch import nn
import torch
import math


class RMSnorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gi = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        return torch.div(x, rms) * self.gi 


class Gelu(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(torch.div(x, torch.sqrt(torch.tensor(2)))))
    

class FF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gelu = Gelu()
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        a = self.gelu(self.ff1(x))
        return  self.ff2(a)
    
def softmax(x, dim):
    max_x = torch.max(x)
    x -= max_x
    max_sum = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return torch.div(torch.exp(x), max_sum)
    

class Attention(nn.Module):
    def __init__(self, p_drop):
        super().__init__()
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, Q, K, V, mask):
        head_dim = torch.tensor(K.shape[-1])
        if len(Q.shape) > 3:
            K = torch.permute(K, (0, 1, 3, 2))     
        else:
            K = torch.permute(K, (0, 2, 1))

        a_scaled = torch.div(torch.matmul(Q, K), torch.sqrt(head_dim))
        if mask is not None:
            a_scaled = torch.masked_fill(a_scaled, mask, -torch.inf)

        a_scaled = softmax(a_scaled, dim=-1)
        if self.p_drop:
            output = self.dropout(torch.matmul(a_scaled, V))
        else:
            output = torch.matmul(a_scaled, V)
        return output

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads, p_drop):
        super().__init__()
        self.p_drop = p_drop
        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.total = nn.Linear(d_model, 3*d_model, bias=False)

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.attention_dot_prod = Attention(p_drop)
    
    def set_weights_from_dict(self, weights: dict):
        Q = torch.empty((self.num_heads, self.head_dim, self.d_model))
        K =  torch.empty((self.num_heads, self.head_dim, self.d_model))
        V =  torch.empty((self.num_heads, self.head_dim, self.d_model))

        for head in range(self.num_heads):
            Q[head] = weights[f"q_heads.{head}.weight"]
            K[head] = weights[f"k_heads.{head}.weight"]
            V[head] = weights[f"v_heads.{head}.weight"]

        self.proj.weight.data[:] =  weights["output_proj.weight"]
        self.Q.weight.data[:] = Q.view(self.d_model, self.d_model)
        self.V.weight.data[:] = V.view(self.d_model, self.d_model)
        self.K.weight.data[:] = K.view(self.d_model, self.d_model)
        self.total.weight.data[0:self.d_model, :] =  Q.view(self.d_model, self.d_model)
        self.total.weight.data[self.d_model:2*self.d_model, :] =  Q.view(self.d_model, self.d_model)
        self.total.weight.data[2*self.d_model:3*self.d_model, :] =  Q.view(self.d_model, self.d_model)


    def forward(self, x):
        B, source_seq_len = x.shape[0:2]
        target_seq_len = self.d_model
        num_head = self.num_heads
        head_dim = self.head_dim
        print(self.Q.weight.data.shape, x.shape)

        Q = self.Q(x).view(B, -1, num_head, head_dim).transpose(1, 2)
        K = self.K(x).view(B, -1, num_head, head_dim).permute(0, 2, 1, 3)
        V = self.V(x).view(B, -1, num_head, head_dim).transpose(1, 2)
        torch.tensor_split(self.total(x), 3, -1)

        mask = torch.triu(torch.ones((source_seq_len, source_seq_len), dtype=torch.bool, device=x.device), diagonal=1).to(x.device)
        output = self.attention_dot_prod( Q, K, V, mask)
        return  self.proj(output.transpose(1,2).reshape(B, source_seq_len, self.d_model))
        
        

    


    
    