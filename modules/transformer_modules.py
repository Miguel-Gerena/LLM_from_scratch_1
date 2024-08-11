from typing import Tuple
from torch import nn
from torch.nn import functional as F
import torch
from math import sqrt



class RMSnorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-2):
        super().__init__()
        self.ln = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x:torch.FloatTensor) -> torch.tensor:
        x_fp32 = x.float()
        rms = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return rms * self.ln

def gelu(x:torch.FloatTensor) -> torch.tensor:
    return 0.5 * x * (1 + torch.erf(torch.div(x, torch.sqrt(torch.tensor(2)))))
    

class FF(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x:torch.FloatTensor) -> torch.FloatTensor:
        a = F.gelu(self.ff1(x))
        return  self.ff2(a)
    
def softmax(x:torch.FloatTensor, dim:int) -> torch.tensor:
    max_x = torch.max(x)
    x = x - max_x
    max_sum = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return torch.div(torch.exp(x), max_sum)
    

class Attention(nn.Module):
    def __init__(self, p_drop:float|None):
        super().__init__()
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop if p_drop else 0)
    
    def forward(self, Q:torch.FloatTensor, K:torch.FloatTensor, V:torch.FloatTensor, mask:torch.FloatTensor) -> torch.FloatTensor:
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
    def __init__(self, d_model:int, num_heads:int, attn_drop:float|None):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.fused_qkv = nn.Linear(d_model, 3*d_model, bias=False)

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.attention_dot_prod = Attention(attn_drop if attn_drop else 0)
    
    def set_weights_from_dict(self, weights: dict):
        Q = torch.empty((self.num_heads, self.head_dim, self.d_model))
        K =  torch.empty((self.num_heads, self.head_dim, self.d_model))
        V =  torch.empty((self.num_heads, self.head_dim, self.d_model))

        for head in range(self.num_heads):
            Q[head] = weights[f"q_heads.{head}.weight"]
            K[head] = weights[f"k_heads.{head}.weight"]
            V[head] = weights[f"v_heads.{head}.weight"]

        self.proj.weight.data[:] =  weights["output_proj.weight"]
        self.fused_qkv.weight.data[0:self.d_model, :] =  Q.view(self.d_model, self.d_model)
        self.fused_qkv.weight.data[self.d_model:2*self.d_model, :] =  K.view(self.d_model, self.d_model)
        self.fused_qkv.weight.data[2*self.d_model:3*self.d_model, :] =  V.view(self.d_model, self.d_model)
    
    def reset_parameters(self):
        for qkv in range(self.fused_qkv.shape[0]):
            for head in range(self.fused_qkv.shape[1]):
                nn.init.kaiming_uniform_(self.fused_qkv[qkv, head], a=sqrt(5))

    def forward(self, x:torch.FloatTensor) -> torch.FloatTensor:
        B, source_seq_len = x.shape[0:2]
        target_seq_len = self.d_model
        num_head = self.num_heads
        head_dim = self.head_dim
        Q, K, V = torch.tensor_split(self.fused_qkv(x), 3, -1)
        Q = Q.view(B, source_seq_len, num_head, head_dim).transpose(1, 2)
        K = K.view(B, source_seq_len, num_head, head_dim).permute(0, 2, 1, 3)
        V = V.view(B, source_seq_len, num_head, head_dim).transpose(1, 2)

        mask = torch.triu(torch.ones((source_seq_len, source_seq_len), dtype=torch.bool, device=x.device), diagonal=1)
        output = self.attention_dot_prod(Q, K, V, mask)
        output =  self.proj(output.transpose(1,2).reshape(B, source_seq_len, self.d_model))
        return output
    
class Transformer_block(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_drop:float|None = None,  res_drop:float|None = None, pre_norm:bool=True ) :
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pre_norm = pre_norm
        self.d_ff = d_ff

        self.mha = Multi_Head_Attention(d_model, num_heads, attn_drop)
        self.ff = FF(d_model, d_ff)
        
        self.rms_norm1 = RMSnorm(d_model)
        self.rms_norm2 = RMSnorm(d_model)
        self.res_drop = nn.Dropout(res_drop if res_drop else 0) 

    def set_weights_from_dict(self, weights: dict):
        self.mha.fused_qkv.weight.data[0:self.d_model, :] =  weights["attn.q_proj.weight"]
        self.mha.fused_qkv.weight.data[self.d_model:2*self.d_model, :] =  weights["attn.k_proj.weight"]
        self.mha.fused_qkv.weight.data[2*self.d_model:3*self.d_model, :] =  weights["attn.v_proj.weight"]
        self.rms_norm1.ln.data[:] = weights["ln1.weight"]
        self.rms_norm2.ln.data[:] = weights["ln2.weight"]
        self.mha.proj.weight.data[:] =  weights["attn.output_proj.weight"]
        self.ff.ff1.weight.data[:] =  weights["ffn.w1.weight"]
        self.ff.ff2.weight.data[:] =  weights["ffn.w2.weight"]

    def forward(self, x:torch.tensor) -> torch.FloatTensor:
        if self.pre_norm:
            x = self.mha(self.rms_norm1(x)) + x
            x = self.res_drop(self.ff(self.rms_norm2(x))) + x
        else:
            x = self.rms_norm1(self.mha(x) + x)
            x = self.rms_norm2(self.res_drop(self.ff(x)) + x)
        return x

class MOE_Transformer_block(Transformer_block):
    def __init__(self, num_experts:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff_dict = {}
        for i in range(num_experts):
            self.ff_dict[f"moe{i}"] = FF(self.d_model, self.d_ff)
        self.ff = nn.ModuleDict(self.ff_dict)
    
    def forward(self, args:Tuple[torch.tensor, int]) :  # nq:override -> Tuple[torch.FloatTensor, int]
        x, expert_index = args
        if self.pre_norm:
            att = self.mha(self.rms_norm1(x)) + x
            ff = self.res_drop(self.ff[f"moe{expert_index}"](self.rms_norm2(att))) + att
        else:
            att = self.rms_norm1(self.mha(x) + x)
            ff = self.rms_norm2(self.res_drop(self.ff[f"moe{expert_index}"](att)) + att)
        return ff, expert_index
    
def transformer_layers(num_layers:int, d_model:int, num_heads:int, d_ff:int, attn_drop:float|None = None,  res_drop:float|None = None, pre_norm:bool=True, num_experts:int=0) -> nn.Module:
    if  num_experts <= 1:
        return nn.Sequential(*[Transformer_block(d_model, num_heads, d_ff, attn_drop, res_drop, pre_norm) for _ in range(num_layers)])
    return nn.Sequential(*[MOE_Transformer_block(num_experts, d_model, num_heads, d_ff, attn_drop, res_drop, pre_norm) for _ in range(num_layers)])


