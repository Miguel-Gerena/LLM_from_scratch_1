from torch import nn
import torch


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
        B = K.shape[0]
        source_sq_len = Q.shape[-2]
        target_seq_len = K.shape[-2]
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
    def __init__(self, p_drop):
        super().__init__()
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, Q, K, V, mask):
        B = K.shape[0]
        source_sq_len = Q.shape[-2]
        target_seq_len = K.shape[-2]
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
        
        

    


    
    