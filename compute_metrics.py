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

def metrics(
        dataloader,
        args,
):

    n_jets = args.n_jets

    progress_bar = tqdm(total = n_jets, 
                        desc = f"Computing metrics from jet dataset",
                        )
    
    flat_dim = 3 * args.max_length

    # Create axis0 labels once
    axis0_labels = []
    for i in range(args.max_length):
        axis0_labels.extend([f"PT_{i}", f"ETA_{i}", f"PHI_{i}"])

    with h5py.File(args.output_file, "w") as f:

        #prepare dataset
            
        dset = f.create_dataset(
            "metrics",
            shape=(n_jets, 3 * args.max_length),
            maxshape=(None, 3 * args.max_length),
            dtype=np.float32,
            chunks=(min(args.batch_size, n_jets), 3 * args.max_length)
        )
        # Speichere Feature-Namen als Attribut
        dset.attrs["feature_names"] = np.bytes_(axis0_labels)

        start_time = time.time()

        for x in progress_bar:

            start_time_batch = time.time()




            dt = time.time() - start_time_batch
            speed = len(x) / dt
            progress_bar.set_postfix({"jets/s": f"{speed:.2f}"})

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


