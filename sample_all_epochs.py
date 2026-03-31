import argparse
import subprocess
import os

CLASSES = ["QCD", "TTBar"]

def get_model_path(base, cls, epoch):
    return f"{base}/checkpoints/{cls}_epoch_{epoch}.pt"


def main(args):

    epoch = args.epoch

    out_dir = f"{args.output}/sampled_jets"
    os.makedirs(out_dir, exist_ok=True)

    for cls in CLASSES:

        model_path = get_model_path(args.training_folder, cls, epoch)
        output_file = f"{out_dir}/{cls}_epoch_{epoch}.h5"

        if os.path.exists(output_file):
            print(f"Skipping existing {output_file}")
            continue

        print(f"Sampling {cls} epoch {epoch}")

        subprocess.run([
            "python", "sample.py",
            "--model_path", model_path,
            "--output_file", output_file,
            "--n_jets", str(args.n_jets),
            "--batch_size", str(args.batch_size),
            "--max_length", str(args.max_length),
            "--topk", str(args.topk),
        ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, required=True)

    parser.add_argument("--n_jets", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5000)

    args = parser.parse_args()
    main(args)