from torch import nn

from transformer_modules import transformer_layers



class Transformer(nn.Module):
    def __init__(self, num_layers:int, d_model:int, num_heads:int, d_ff:int, attn_drop:float|None = None,  res_drop:float|None = None, pre_norm:bool=True):
        super().__init__()
        self.transformer_blocks = transformer_layers(num_layers, d_model, num_heads, d_ff, attn_drop, res_drop)

    
