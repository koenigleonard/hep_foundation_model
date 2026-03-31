from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from dataset import *
from torch.optim.lr_scheduler import LambdaLR
import math
import model

def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        # warmup phase
        if step < warmup_steps:
            return step / float(warmup_steps)

        # cosine decay phase
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def constant_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0, total_steps)

def get_scheduler(optimizer, epoch_steps, args):

    if args.scheduler == "warmup_cosine":
        scheduler = warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(0.1*epoch_steps*args.num_epochs),
            total_steps=epoch_steps*args.num_epochs
        )
        print("Using cosine scheduler with warmup.")
    elif args.scheduler == "constant":
        scheduler = constant_scheduler(
            optimizer,
            total_steps=epoch_steps*args.num_epochs
        )
        print("Using constant scheduler.")
    elif args.scheduler == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = args.restart_period * epoch_steps,
            eta_min = 5e-6
        )
        print(f"Using cosine restart scheduler with a T_0 = {args.restart_period} and eta_min = {5e-6}")
    elif args.scheduler == "exp":

        gamma = args.gamma**(1/epoch_steps)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = gamma
        )
        print(f"Using exponential scheduler with gamma = {args.gamma}.")
    elif args.scheduler == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max= epoch_steps*args.num_epochs,
            eta_min= 5e-6)
        
        print(f"Using cosine decay scheduler with eta_min = {5e-6}")

    else:
        scheduler = constant_scheduler(
            optimizer,
            total_steps=epoch_steps*args.num_epochs,
        )
        print("Using constant scheduler.")

    return scheduler
#cli
def parse_inputs():
    parser = ArgumentParser()

    #add arguments here
    parser.add_argument("--data_path", type=str, help="Path to training data file")
    parser.add_argument("--num_const", type=int, default=50, help="Number of constituents")
    parser.add_argument("--add_start",action="store_true",help="Whether to use a start particle (learn first particle as well)",)
    parser.set_defaults(add_start = True)
    parser.add_argument("--add_stop", action="store_true", help="Whether to use a end particle (learn jet length as well)",)
    parser.set_defaults(add_stop = True)
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim of the model")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--n_pt", type= int, default = 40, help = "Number of pt bins")
    parser.add_argument("--n_eta", type= int, default = 30, help = "Number of eta bins")
    parser.add_argument("--n_phi", type= int, default = 30, help = "Number of phi bins")
    parser.add_argument("--causal_mask", action = "store_true", help = "Wether to use a causal mask in the attention layer.")
    parser.set_defaults(causal_mask = True)
    parser.add_argument("--output_path", type=str, help="Path for storing logs and model files", default = "output/")
    parser.add_argument("--name", type=str, default = "latest", help = "Name of model")
    parser.add_argument("--contin", "-c", action = "store_true", help = "if selected training is continued with specified file, all args are ignored and taken from original run")
    parser.set_defaults(contin = False )
    parser.add_argument("--batch_size", type=int, default = 100)
    parser.add_argument("--batch_size_val", type = int, default = 500)
    parser.add_argument("--n_jets", type=int, default = None)
    parser.add_argument("--n_jets_val", type=int, default = None)
    parser.add_argument("--input_key", type = str, default = "discretized", help = "if the key of table in the h5 is different it can be specified here")
    parser.add_argument("--linear_output", action = "store_true", help = "wether to use a linear output head instead of a factorized output head. Default is False")
    parser.set_defaults(linear_output = False)
    parser.add_argument("--checkpoints", type = str, default = "all", help = "sets checkpoint mode. Options: best, all")
    parser.add_argument("--scheduler", type = str, default = "constant", help = "which scheduler is used for training (constant, warmup_cosine, cosine_restarts, cosine)")
    parser.add_argument("--restart_period", type = int, default = 10, help = "Number of epoch until the first restart. cosine_restarts")
    parser.add_argument("--gamma", type = float, default = 0.8, help = "sets the factor by which LR is reduced every epoch, only used with exponential scheduler")
    parser.add_argument("--early_stopping", action = "store_true", help = "if early stopping should be used.")
    parser.set_defaults(early_stopping = False)
    parser.add_argument("--patience", type = int, default = 5, help = "patience period for early stopping")
    parser.add_argument("--delta_min", type = float, default = 1e-4, help = "The tolerance for early stopping.")
    parser.add_argument("--ema_alpha", type = float, default = 0.98, help = "alpha value for the exponential moving average")
    parser.add_argument("--verbose_output", action = "store_true", help = "if every batch loss is supposed to be logged")
    parser.set_defaults(verbose_output = False)
    parser.add_argument("--no_shuffle", action = "store_true", help = "if train data set is shuffled after each epoch")
    parser.set_defaults(no_shuffle = False)
    parser.add_argument("--checkpoint_name", type = str, default = "best.pt", help = "name of model checkpoint if training is continued.")
    parser.add_argument("--new_lr", type=float, default=None)
    parser.add_argument("--reset_scheduler", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.set_defaults(reset_scheduler = False)
    parser.set_defaults(reset_optimizer = False)


    args = parser.parse_args()
    return args

#saves a model to disk
def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))

def load_model(model_path):
    model = torch.load(model_path)

    return model

#just use for testing
def load_model_checkpoint(checkpoint_path):

    num_features = 3

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    train_args = checkpoint["args"]
    print(f"Args used for training: {train_args}")

    newModel = model.JetTransformer(
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        num_features=num_features,
        num_bins=(train_args["n_pt"], train_args["n_eta"], train_args["n_phi"]),
        dropout=train_args["dropout"],
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        causal_mask = train_args["causal_mask"],
        linear_output = train_args["linear_output"]
    )

    newModel.load_state_dict(checkpoint["model_state"])

    return newModel


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

