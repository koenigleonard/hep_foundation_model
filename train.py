import torch
import os
from model import JetTransformer
from helpers_train import *
import dataset
from torch.utils.data import DataLoader
import pandas as pd
import time

from torch.amp import autocast, GradScaler

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")

num_features = 3

def build_model(args):
    
    model = JetTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        dropout=args.dropout,
        add_start=args.add_start,
        add_stop=args.add_stop,
        causal_mask = args.causal_mask,
        linear_output = args.linear_output,
    )

    return model

def build_dataloader(args, n_jets, tag = "train"):
    data_loader = DataLoader(JetDataSet(
        data_dir = args.data_path.replace("train", tag),
        tag = tag,
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        add_stop=args.add_stop,
        add_start=args.add_start,
        n_jets=n_jets,
        key = args.input_key,
        ),
        batch_size=args.batch_size_val if tag == "val" else args.batch_size,
        shuffle= not args.no_shuffle, # optimization
        num_workers=4,            
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,)

    return data_loader


def train(model, train_loader, val_loader, optimizer, scheduler, args,
          epochs = 10,
          ):
    
    model.to(device)
    model = torch.compile(model) #pre compiles the model for faster performance
    model.train()

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    
    #counter for early stopping
    counter = 0

    training_step = 0
    epoch_steps = int(len(train_loader))
    #eval_steps = int(epoch_steps/args.val_frequency)

    ema_loss = None
    alpha = args.ema_alpha

    #print(f"Evaluating every {eval_steps} steps")

    if not args.contin and os.path.exists(os.path.join(args.output_path, f"{args.name}_training_log.csv")):
        os.remove(os.path.join(args.output_path, f"{args.name}_training_log.csv"))

    if args.verbose_output:
        if not args.contin and os.path.exists(os.path.join(args.output_path, f"{args.name}_batch_loss.csv")):
            os.remove(os.path.join(args.output_path, f"{args.name}_batch_loss.csv"))

    for epoch in range(epochs):

        stop = False
        #for timing length of epoch
        start_time = time.time()

        avg_val_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{epochs} [Training]",
            leave = True
        )
        for x in progress_bar:

            training_step += 1
            #move batch to gpu if possible
            x = x.to(device)
            optimizer.zero_grad(set_to_none = True)
            #compute one forward pass and the loss on the data passed into the network
            with autocast(device_type = "cuda"):
                logits = model(x)
                loss = model.loss(logits, x) #the target is the data we have trained with
            
            #backward pass
            scaler.scale(loss).backward()
            #clips gradient so they dont explode at the start
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
 
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = alpha * ema_loss + (1 - alpha) * loss.item()

            if args.verbose_output:
                ### logging
                log_data = {
                    "step": training_step,
                    "ema_loss": ema_loss,
                    "batch_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }

                df = pd.DataFrame([log_data])

                df.to_csv(
                    os.path.join(args.output_path, f"{args.name}_batch_loss.csv"),
                    mode="a",
                    header=not os.path.exists(os.path.join(args.output_path, f"{args.name}_batch_loss.csv")),
                    index=False
                )

            #update progress bar
            progress_bar.set_postfix(loss = loss.item())

        #rund validation on validation set
        avg_val_loss = validate(model, val_loader)
        model.train()

        #check for saving and early stopping
        #save only best model
        if args.checkpoints == "best" or args.checkpoints == "all":
            if avg_val_loss < best_val_loss - args.delta_min:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, args, name = args.name + "_best", path=os.path.join(args.output_path, "checkpoints"))
                print(f" Checkpoint saved as: {args.name}_best.pt")

                counter = 0
                stop = False
            else:
                counter += 1

        #save all models after the epochs
        if args.checkpoints == "all":
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, args, name = args.name + f"_epoch_{epoch}", path=os.path.join(args.output_path, "checkpoints"))
            print(f"Checkpoint saved as: {args.name}_epoch_{epoch}.pt")


        if counter >= args.patience:
            stop = True

        ### logging
        log_data = {
            "step": training_step,
             "train_loss": ema_loss, 
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        df = pd.DataFrame([log_data])

        df.to_csv(
            os.path.join(args.output_path, f"{args.name}_training_log.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(args.output_path, f"{args.name}_training_log.csv")),
            index=False
        )

        total_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} [Finished] | "
            f"Train Loss: {ema_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]} | "
            f"Time: {total_time:.2f} s"
        )

        if stop and args.early_stopping:
            print(" Early stopping triggered. Stopping training.")
            break

#for running the validation set
def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc= "Validation", leave = False)

        for x in progress_bar:
            x = x.to(device)

            logits = model(x)
            loss = model.loss(logits, x)

            total_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item())

    avg_loss = total_loss / len(dataloader) #dividing by number of batches
    return avg_loss

if __name__ == "__main__":
    args = parse_inputs()

    print("Running trainings process:")
    print("Trainings parameter:" )
    print("-" * 30)
    for key, value in vars(args).items():
        print(f"{key:<15} | {value}")
    print(f"Running on device: {device}")

    os.makedirs(args.output_path, exist_ok=True)

    #load datasets
    train_loader = build_dataloader(args,n_jets=args.n_jets, tag = "train")
    print(f"Training set number batches: {len(train_loader)}")

    val_loader = build_dataloader(args, n_jets = args.n_jets_val, tag = "val")
    print(f"Validation set number batches: {len(val_loader)}")

    #construct model
    model = build_model(args)

    model.to(device)

    #add optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr,
    )

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.lr,
    # )

    scheduler = get_scheduler(optimizer, total_steps= len(train_loader), args= args)

    train(model, train_loader, val_loader, optimizer, scheduler, args, epochs=args.num_epochs)
