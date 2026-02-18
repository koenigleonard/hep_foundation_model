from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from dataset import *
from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        # warmup phase
        if step < warmup_steps:
            return step / float(warmup_steps)

        # cosine decay phase
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

#cli
def parse_inputs():
    parser = ArgumentParser()

    #add arguments here
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to training data file",
    )
    parser.add_argument(
        "--num_const", type=int, default=50, help="Number of constituents"
    )
    parser.add_argument(
        "--add_start",
        action="store_true",
        help="Whether to use a start particle (learn first particle as well)",
    )
    parser.set_defaults(add_start = True)

    parser.add_argument(
        "--add_stop",
        action="store_true",
        help="Whether to use a end particle (learn jet length as well)",
    )
    parser.set_defaults(add_stop = True)
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dim of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--n_pt", type= int, default = 40, help = "Number of pt bins")
    parser.add_argument("--n_eta", type= int, default = 30, help = "Number of eta bins")
    parser.add_argument("--n_phi", type= int, default = 30, help = "Number of phi bins")
    parser.add_argument("--causal_mask", action = "store_true", help = "Wether to use a causal mask in the attention layer.")
    parser.set_defaults(causal_mask = True)
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for storing logs and model files",
        default = "output/"
    )
    parser.add_argument("--name", type=str, default = "latest", help = "Name of model")
    parser.add_argument("--contin", "-c", action = "store_true", help = "if selected training is continued with specified file, all args are ignored and taken from original run")
    parser.set_defaults(contin = False )
    parser.add_argument("--batch_size", type=int, default = 100)

    args = parser.parse_args()
    return args

#saves a model to disk
def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))

def load_model(model_path):
    model = torch.load(model_path)

    return model

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, path="output/checkpoints", name = "latest"):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "args":vars(args)
    }

    torch.save(checkpoint, os.path.join(path, name + ".pt"))
