import torch
from helpers_sample import *
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(
        model,
        device,
        args,
        train_args,
):

    model.sample(batch_size = args.batch_size,
                 max_length = args.max_length,
                 topk = args.topk
                 )
    return None


if __name__ == "__main__":
    args = parse_inputs()

    print(f"Running on device: {device}")

    num_features = 3

    ##load model from file
    print(f"Load model state from:{args.model_path}")

    checkpoint = torch.load(args.model_path)

    train_args = checkpoint["args"]
    print(f"Args used for training: {train_args}")

    sampleModel = model.JetTransformer(
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        num_features=num_features,
        num_bins=(train_args["n_pt"], train_args["n_eta"], train_args["n_phi"]),
        dropout=train_args["dropout"],
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        causal_mask = train_args["causal_mask"],
    )
    sampleModel.to(device)

    sampleModel.load_state_dict(checkpoint["model_state"])

    sampleModel.eval()

    ## run sampling
    with torch.no_grad():
        jets = sample(sampleModel, device, args, train_args,)
    