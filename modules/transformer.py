import torch
from torch import nn
from typing import List, overload

from modules.transformer_modules import RMSnorm, transformer_layers, softmax


class Transformer(nn.Module):
    def __init__(self, context_length:int, vocab_size:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, attn_drop:float|None = None,  res_drop:float|None = None, pre_norm:bool=True):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.context_length = context_length
        self.transformer_blocks = transformer_layers(num_layers, d_model, num_heads, d_ff, attn_drop, res_drop, pre_norm)
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
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embed_drop(self.embed_layer(x) + self.pos_embeding[None, :x.shape[-1], :])
        x = self.transformer_blocks(x)
        x = self.mlp(self.final_norm(x)).permute([0,2,1])
        return x

    @torch.inference_mode
    def generate(self, input_ids: torch.Tensor, stop_tokens: List[int], max_tokens:int, temperature:float, top_p:float = 1) -> List[int]:
        sampled_tokens: List[int] = []
        self.eval()
        while True:
            generated = self.forward(input_ids)
            probs = softmax(generated/temperature, dim=-1)

            highest_prob = torch.argmax(probs, dim=-1)[:,-1]
            if highest_prob.item() in stop_tokens or max_tokens == 0:
                if highest_prob.item() in stop_tokens:
                    sampled_tokens.append(highest_prob.item())
                return sampled_tokens

            sampled_tokens.append(highest_prob.item())
            max_tokens -= 1
            input_ids = highest_prob.unsqueeze(0).to(input_ids.device)

class MOE_Transformer(Transformer):
    def __init__(self, context_length: int, vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, attn_drop: float | None = None, res_drop: float | None = None, pre_norm: bool = True, num_experts: int = 0):
        super().__init__(context_length, vocab_size, num_layers, d_model, num_heads, d_ff, attn_drop, res_drop, pre_norm)
        self.transformer_blocks = transformer_layers(num_layers, d_model, num_heads, d_ff, attn_drop, res_drop, pre_norm, num_experts)
      
    def forward(self, x, class_id):
        embed = self.embed_drop(self.embed_layer(x) + self.pos_embeding[None, :x.shape[-1], :])
        x = self.transformer_blocks((embed, class_id))[0]
        return self.mlp(self.final_norm(x)).permute([0,2,1])
