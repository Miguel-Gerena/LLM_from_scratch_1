import torch
from torch.optim import AdamW
from modules.transformer import Transformer, MOE_Transformer
import argparse
from typing import Tuple
import numpy as np
from data_loader import train_data_generator, val_data_generator
from tqdm import tqdm
from training_util import clip_gradient, cosine_learning_warmup

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
    device = args.device
    if args.num_experts > 1:
        model = MOE_Transformer(args.context_length, args.vocab_size, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.attn_drop, args.res_drop, args.pre_norm, args.num_experts)
    else:
        model = Transformer(args.context_length, args.vocab_size, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.attn_drop, args.res_drop, args.pre_norm)
    optim = AdamW(model.parameters(), args.lr, args.betas, args.weight_decay)
    critereon = torch.nn.CrossEntropyLoss()
    model.to(device)
    total_num_parameters = sum([torch.numel(p) if p.requires_grad else 0 for p in model.parameters()])
    total_num_parameters_in_embed_layer = sum([torch.numel(p) if p.requires_grad else 0 for p in model.get_submodule("embed_layer").parameters()])
   
    print(f"Parameters in the embedding layer {total_num_parameters_in_embed_layer}")
    print(f"Total parameters: {total_num_parameters}\nTotal parameters without embedding: {total_num_parameters - total_num_parameters_in_embed_layer}")
      
    torch.compile(model, fullgraph=True)
    critereon.to(device)
    torch.compile(critereon, fullgraph=True)
    
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    args.train_step = len(train_data)//args.batch_size

    dataloader = train_data_generator(train_data, args.batch_size, args.context_length, device)
    dataloader_val = val_data_generator(val_data, args.batch_size_val, args.context_length, device)

    model.train()
    train(args, model, optim, args.epochs, dataloader, dataloader_val, critereon)

@torch.inference_mode
def validate(model:torch.nn.Module, dataloader_val, critereon:torch.nn.Module):
    for x, y in dataloader_val:
        logits = model(x)
        loss = critereon(logits, y)
        # acc loss

def train(args, model:torch.nn.Module, optim:torch.optim.Optimizer, epochs:int, dataloader, dataloader_val, critereon:torch.nn.Module):
    print("training")
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        for _ in range(0, epochs):
            for i, data in enumerate(tqdm(dataloader, total=args.train_step)):
                lr = cosine_learning_warmup(i, args.lr, args.min_lr, args.warm_up_steps, args.train_step * 0.9)
                optim.param_groups[0]["lr"] = lr
                print("forward")
                if args.num_experts > 1:
                    x, y, class_id = data
                    logits = model(x, class_id)
                else:
                    x, y = data
                    logits = model(x)
                optim.zero_grad()
                loss = critereon(logits, y)
                print("back")
                loss.backward()
                clip_gradient(model.parameters(), 1)
                optim.step()

                if args.validate_steps != 0 and i != 0 and i % args.validate_steps == 0:
                    model.eval()
                    validate(model, dataloader_val, critereon)
                    model.train()

            model.eval()
            validate(model, dataloader_val, critereon)
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--batch_size_val", type=int, default=24)
    parser.add_argument("--warm_up_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validate_steps", type=int, default=200)
    parser.add_argument("--train_data_path", type=str, default="data/TinyStoriesV2-GPT4-train.npy")
    parser.add_argument("--val_data_path", type=str, default="data/TinyStoriesV2-GPT4-valid.npy")
    
    # Model args
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--pre_norm", type=bool, default=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_drop", type=float, default=0.2)
    parser.add_argument("--res_drop", type=float, default=0.2)

    # Optim args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=Tuple[float, float], default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Misc args
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    args = parser.parse_args()
    main(args)


