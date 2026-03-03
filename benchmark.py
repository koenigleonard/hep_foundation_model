import torch
import time
import argparse
import random
import numpy as np

from train import build_model, build_dataloader  # reuse your code
from helpers_train import *

from torch.amp import autocast, GradScaler

#set all seed and deactivate random behavior for comparability
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#simulates training one epochs 
def benchmark(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Benchmark using device: {device}")

    model = build_model(args).to(device)
    model = torch.compile(model)
    model.train()
    
    train_loader = build_dataloader(args, tag = "train")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = GradScaler("cuda")

    max_batches = int(args.n_jets // args.batch_size)
    print(f"Running {max_batches} batches.")

    # Warmup this is being done because the first runs are always slower (caching etc. so we dont want do measure this)
    for i, x in enumerate(train_loader):
        if i >= 50:
            break
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):        
            logits = model(x)
            loss = model.loss(logits, x)

        scaler.scale(loss).backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for i, x in enumerate(train_loader):
        if i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):        
            logits = model(x)
            loss = model.loss(logits, x)

        scaler.scale(loss).backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    end = time.time()

    print("Time per batch:", (end - start) / max_batches)
    print("Peak memory (GB):",
          torch.cuda.max_memory_allocated() / 1024**3)
    print("Final loss:", loss.item())


if __name__ == "__main__":

    args = parse_inputs()

    benchmark(args)