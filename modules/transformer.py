import torch
from torch import nn, numel

from modules.transformer_modules import RMSnorm, transformer_layers



class Transformer(nn.Module):
    def __init__(self, context_length:int, vocab_size:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, attn_drop:float|None = None,  res_drop:float|None = None, pre_norm:bool=True):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.context_length = context_length
        self.transformer_blocks = transformer_layers(num_layers, d_model, num_heads, d_ff, attn_drop, res_drop)
        self.embed_layer = nn.Embedding(vocab_size, d_model)
        self.pos_embeding = nn.Parameter(torch.zeros(context_length, d_model))
        self.mlp = nn.Linear(d_model, vocab_size, bias=False)
        self.final_norm = RMSnorm(d_model)
        self.embed_drop = nn.Dropout(attn_drop if attn_drop else 0)
       
    def set_weights_from_dict(self, weights: dict):
        self.embed_layer.weight.data[:] = weights["token_embeddings.weight"]
        self.pos_embeding.data[:] = weights["position_embeddings.weight"]
        self.mlp.weight.data[:] = weights["lm_head.weight"]
        
        for i, block in enumerate(self.transformer_blocks.children()):
            new_weights = {}
            for key, value in weights.items():
                if f"layers.{i}" in key:
                    new_key = key.split("layers.")[-1][2:]
                    new_weights[new_key] = value
            block.set_weights_from_dict(new_weights)
        self.final_norm.ln.data[:] = weights["ln_final.weight"]

    def forward(self, x):
        embed = self.embed_drop(self.embed_layer(x) + self.pos_embeding[None, :x.shape[-1], :])
        x = self.transformer_blocks(embed)
        return self.mlp(self.final_norm(x))


model = Transformer(256, 10000, 4, 512, 16, 2048, 0, 0)
model.to("cuda")
trainable = sum([numel(param) for param in list(model.parameters()) if param.requires_grad])
total = sum([numel(param) for param in list(model.parameters()) ])

print("hi")
    
