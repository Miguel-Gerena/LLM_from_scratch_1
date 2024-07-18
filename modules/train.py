import torch
from torch.optim import AdamW
from modules.transformer import Transformer
import argparse
from typing import Tuple
import numpy as np

def save_checkpoint(model:torch.nn.Module, path:str, epoch:int, optim:torch.optim.Optimizer) -> None:
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, path)

def load_checkpoint(model:torch.nn.Module, path:str, optim:torch.optim.Optimizer)-> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Transformer(args.context_length, args.vocab_size, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.attn_drop, args.res_drop)
    optim = AdamW(model.parameters(), args.lr, args.betas, args.weight_decay)
    critereon = torch.nn.CrossEntropyLoss()
    model.to(args.device)
    optim.to(args.device)
    critereon.to(args.device)
    model.train()
    dataloader = [0]

    train(model, optim, args.epochs, dataloader, critereon)



def train(model:torch.nn.Module, optim:torch.optim.Optimizer, epochs:int, dataloader, critereon:torch.nn.Module):
    for _ in range(epochs):
        for x, y in dataloader:
            logits = model(x)
            optim.zero_grad()
            loss = critereon(logits, y)
            loss.backward()
            optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_drop", type=float, default=0.2)
    parser.add_argument("--res_drop", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=Tuple[float, float], default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    main(parser)


