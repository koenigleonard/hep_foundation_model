import torch
import time
from tqdm import tqdm
import numpy as np
import dataset
from torch.utils.data import DataLoader
import csv
import os
from helpers_compute_metrics import *
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_bin_edges(args):
    pt_bins = torch.linspace(args.pt_min, args.pt_max, args.n_pt, device = device) #(pt_min, ..., pt_max)
    eta_bins = torch.linspace(args.eta_min, args.eta_max, args.n_eta, device = device) #(eta_min, ..., eta_max)
    phi_bins = torch.linspace(args.phi_min, args.phi_max, args.n_phi, device = device) #(phi_min, ..., phi_max)
    
    return pt_bins, eta_bins, phi_bins

def compute_bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def metrics(
        dataloader,
        args,
):
    n_jets = args.batch_size * len(dataloader)

    flat_dim = 3 * args.num_const

    #get bin edges
    pt_bin_edges, eta_bin_edges, phi_bin_edges = compute_bin_edges(args)

    #get bin centers
    pt_centers = compute_bin_centers(pt_bin_edges)
    eta_centers = compute_bin_centers(eta_bin_edges)
    phi_centers = compute_bin_centers(phi_bin_edges)

    print(f"Shape pt_centers = {pt_centers.shape[0]}")

    # Create axis0 labels once
    axis0_labels = []
    for i in range(args.num_const):
        axis0_labels.extend([f"PT_{i}", f"ETA_{i}", f"PHI_{i}"])

    with h5py.File(args.output_file, "w") as f:

        #prepare dataset
        progress_bar = tqdm(dataloader, 
                        desc = f"Computing metrics from jet dataset",
                        )
            
        dset = f.create_dataset(
            "metrics",
            shape=(n_jets, 3 * args.num_const),
            maxshape=(None, 3 * args.num_const),
            dtype=np.float32,
            chunks=(min(args.batch_size, n_jets), 3 * args.num_const)
        )
        # Speichere Feature-Namen als Attribut
        dset.attrs["feature_names"] = np.bytes_(axis0_labels)

        start_time = time.time()

        for x in progress_bar:
            pass

    progress_bar.close()

    total_time = time.time() - start_time
    print(f"\nFinished sampling {n_jets} jets")
    print(f"Total time: {total_time:.2f} s")
    print(f"Average speed: {n_jets / total_time:.2f} jets/s")    

if __name__ == "__main__":
    args = parse_inputs()

    num_features = 3

    #load jet data

    hf = h5py.File(args.data_path)

    data = np.array(hf.get("sampled_jets")) 

    dataloader = DataLoader(data, 
                            batch_size= args.batch_size)

    print(f"Dataset size: {len(dataloader)}")

    metrics(dataloader, args)


