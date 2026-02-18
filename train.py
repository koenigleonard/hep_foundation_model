import torch
import os
from model import JetTransformer
from helpers_train import *
import dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, optimizer, args,
          epochs = 10,
          ):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_train_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{epochs} [Training]",
            leave = True
        )
        for x in progress_bar:
            #move batch to gpu if possible
            x = x.to(device)

            #compute one forward pass and the loss on the data passed into the network
            logits = model(x)
            loss = model.loss(logits, x) #the target is the data we have trained with
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            #update progress bar
            progress_bar.set_postfix(loss = loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        ### run validation after epoc
        avg_val_loss = validate(model, val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} finished | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

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

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    args = parse_inputs()

    print("Running trainings process:")
    print(f"Running on device: {device}")

    num_features = 3
    #load datasets
    print(f"Loading training set")
    train_loader = DataLoader(JetDataSet(
        data_dir = args.data_path,
        tag = "train",
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=100)

    print(f"Loading validation set")
    val_loader = DataLoader(JetDataSet(
        data_dir = args.data_path.replace("train", "val"),
        tag = "val",
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=100)

    #construct model
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
    )

    model.to(device)

    #add optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr,
    )

    #print(train_loader.dataset[:, : , :])

    train(model, train_loader, val_loader, optimizer, args)

    save_model(model, args.output_path, "test2")

