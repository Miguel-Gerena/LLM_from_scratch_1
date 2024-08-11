import time
import torch
from torch.optim import AdamW
from modules.transformer import Transformer, MOE_Transformer
import argparse
from typing import Tuple
import numpy as np
from modules.data_loader import train_data_generator, val_data_generator
from tqdm import tqdm
from modules.training_util import clip_gradient, cosine_learning_warmup, save_checkpoint, load_checkpoint, get_norms
import wandb
import os
import sys
import torch._dynamo
os.environ["TORCH_LOGS"]="+dynamo"
os.environ["WANDB_AGENT_DISABLE_FLAPPING"]="true"
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]="100"


os.environ["TORCHDYNAMO_VERBOSE"]="1"


def main(args):
    torch._dynamo.reset()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.eval:
        with open(f"weights/{args.weights_path.split('_')[0].split('/')[-1]}.txt", "r") as F:
            args_from_file = F.read()
        
        args_text = args_from_file.split("Namespace")[-1][1:-1].split("betas")
        betas = "betas" + args_text[1].split("),")[0] + ")"
        args_text[1] = args_text[1].split("),")[1]
        args_text = "".join(args_text).split(",")


        for argument in args_text:
            key, val = argument.strip().split("=")
            args.__dict__[key] = eval(val)

    device = args.device
    if args.num_experts > 1:
        model = MOE_Transformer(args.context_length, args.vocab_size, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.attn_drop, args.res_drop, args.pre_norm, args.num_experts)
    else:
        model = Transformer(args.context_length, args.vocab_size, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.attn_drop, args.res_drop, args.pre_norm)
    model.to(device)
    optim = AdamW(model.parameters(), args.learning_rate,(args.beta1, args.beta2), args.weight_decay)
    critereon:torch.Tensor = torch.nn.CrossEntropyLoss()
    critereon.to(device)

    if args.compile:
        model = torch.compile(model, fullgraph=True, backend=args.backend, mode="reduce-overhead")
        critereon = torch.compile(critereon, fullgraph=True, backend=args.backend,mode="reduce-overhead")
    
    train_data = np.load(args.train_data_path, mmap_mode="r").astype(np.int32)
    val_data = np.load(args.val_data_path, mmap_mode="r").astype(np.int32)

    dataloader = train_data_generator(train_data, args.batch_size, args.context_length, device)
    dataloader_val = val_data_generator(val_data, args.batch_size_val, args.context_length, device)

    total_num_parameters = sum([torch.numel(p) if p.requires_grad else 0 for p in model.parameters()])
    total_num_parameters_in_embed_layer = sum([torch.numel(p) if p.requires_grad else 0 for p in model.get_submodule("embed_layer").parameters()])
    
    # args.d_model = args.d_model - 1 if args.d_model%2!= 0 else args.d_model
    # args.training_steps = 47844 * 48//args.batch_size //2 # 13.28 it/s
    args.training_steps = len(train_data) # 13.28 it/s
    args.validate_every = args.validate_every * 48//args.batch_size
    if args.wandb:
        args.wandb_instance = wandb.init(
        project="LLM",
        name=args.weights_path.split("_")[0].split("/")[-1] if args.eval else None,
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "learning_rate_min": args.learning_rate_min,
        "dataset": args.train_data_path,
        "training_steps": args.training_steps,
        "max_wall_clock": args.max_wall_clock,
        "num_experts": args.num_experts,
        "pre_norm": args.pre_norm,
        "context_length":args.context_length,
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "num_heads":args.num_heads, 
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "attn_drop": args.attn_drop,
        "res_drop": args.res_drop,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "Total parameters without embedding":total_num_parameters - total_num_parameters_in_embed_layer,
        "Parameters in the embedding layer": total_num_parameters_in_embed_layer,
        "Total parameters":total_num_parameters,
        }
        )

    print(args, "\n")
    print(model, "\n")
    

    print(f"Parameters in the embedding layer {total_num_parameters_in_embed_layer}")
    print(f"Total parameters: {total_num_parameters}\nTotal parameters without embedding: {total_num_parameters - total_num_parameters_in_embed_layer}")
    
    if args.eval:
        i = load_checkpoint(model, args.weights_path, optim)
        i = i * 48 // args.batch_size
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            evaluate(args, model, dataloader_val, critereon, i)
    else:
        model.train()
        train(args, model, optim, args.epochs, dataloader, dataloader_val, critereon)
    wandb.finish(0, False)
    

def validate(args, model:torch.nn.Module, dataloader_val, critereon:torch.nn.Module, step:int=0):
    torch._dynamo.reset()
    model = torch.compile(model, mode="reduce-overhead")
    total_loss = torch.tensor(0.0)
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader_val, total=args.val_steps)):
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            logits = model(x)
            loss = critereon(logits, y)
            total_loss += loss.detach().cpu()
            if i == args.val_steps:
                if args.wandb:
                    wandb.log({"Average Validation loss":total_loss.item()/(i+1), "Average Validation Perplexity":torch.exp(total_loss/(i+1))}, step=step)
                del loss, logits, model
                torch.cuda.empty_cache()
                return total_loss.item()/(i+1)

def validate_and_save(args, model:torch.nn.Module, optim:torch.optim.Optimizer, dataloader_val, critereon:torch.nn.Module, step:int=0, best_val_loss:float=0, postfix:str=""):
    val_loss = validate(args, model, dataloader_val, critereon, step)
    filename = f"weights/{wandb.run.name}{postfix}" if args.wandb else "weights/current_run"
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        with open(f"{filename}.txt", "w") as F:
            F.write(f"{args}")
        save_checkpoint(model, f"{filename}.weights", step, optim, args)

    if args.wandb:
        model_art = wandb.Artifact('model', type='model')   
        model_art.add_file(f'{filename}.weights')
        args.wandb_instance.log_artifact(model_art)

        args_art = wandb.Artifact('args', type='args')
        args_art.add_file(f"{filename}.txt")
        args.wandb_instance.log_artifact(args_art)
    return best_val_loss


def evaluate(args, model:torch.nn.Module, dataloader_val, critereon:torch.nn.Module, step:int=0):
        model.eval()
        _ = validate(args, model, dataloader_val, critereon, step)

def train(args, model:torch.nn.Module, optim:torch.optim.Optimizer, epochs:int, dataloader, dataloader_val, critereon:torch.nn.Module):
    print("training")
    t0 = time.time()
    total_loss = torch.tensor(0.0)
    best_val_loss = float("inf")
    for i, data in enumerate(tqdm(dataloader, total=args.training_steps)):
        step = i * args.batch_size
        # if  time.time() - t0 >= args.max_wall_clock:
        if i == args.training_steps:
            break
        learning_rate = cosine_learning_warmup(i, args.learning_rate, args.learning_rate_min, args.warm_up_steps, args.training_steps * 0.9)
        optim.param_groups[0]["lr"] = learning_rate

        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            if args.num_experts > 1:
                x, y, class_id = data
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                logits = model(x, class_id)
            else:
                x, y = data
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                logits = model(x)

        # change optimizer precison to float32
        optim.zero_grad()
        loss = critereon(logits, y)
        loss.backward()

        if torch.isnan(loss):
            print("Nan in loss.  Exiting...")
            sys.exit(0)
            
        total_loss += loss.detach().cpu()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()

        if i%args.log_iter==0 and i != 0 and args.wandb: 
            norm, norms_dict = get_norms(model.named_parameters(), args.device)
            wandb.log({"weights norm": norm, "learning_rate": learning_rate, "Training loss": loss.detach().item(), "Average Training  loss":total_loss.item()/(i+1), "Average Training Perplexity":torch.exp(total_loss/(i+1)),  "Training Perplexity":torch.exp(loss.detach()), **norms_dict}, step=step)

        if args.validate_every != 0 and i != 0 and i % args.validate_every == 0 or os.path.exists("end"):
            print(f"time:{time.time()-t0} seconds")
            model.eval()
            best_val_loss = validate_and_save(args, model, optim, dataloader_val, critereon, step, best_val_loss)
            torch._dynamo.reset()
            model.train()

    print(f"Total Train time:{time.time()-t0} seconds")
    model.eval()
    best_val_loss = validate_and_save(args, model, optim, dataloader_val, critereon, step, best_val_loss, "_final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Eval args
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--weights_path', type=str, default="weights/confused-sweep-78_final.weights")

    # Training args
    parser.add_argument("--max_wall_clock", type=int, default=3600)
    parser.add_argument("--log_iter", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_val", type=int, default=64)
    parser.add_argument("--warm_up_steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validate_every", type=int, default=10000)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--train_data_path", type=str, default="data/TinyStoriesV2-GPT4-train.npy")
    parser.add_argument("--val_data_path", type=str, default="data/TinyStoriesV2-GPT4-valid.npy")
    
    # Model args
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--pre_norm", type=bool, default=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_drop", type=float, default=0.133468341589087)
    parser.add_argument("--res_drop", type=float, default=0.2)

    # Optim args
    learning_rate = 0.0005
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--learning_rate_min", type=float, default=learning_rate*.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type= float, default= 0.999)

    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Misc args
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--compile', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    args = parser.parse_args()
    
    os.makedirs("weights", exist_ok=True)
    if args.eval:
        for file in os.listdir("weights"):
            if "final" in file:
                args.weights_path = "weights/" + file
                main(args)
    else:
        main(args)



