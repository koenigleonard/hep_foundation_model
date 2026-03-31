import argparse
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

CLASSES = ["QCD", "TTBar"]
TAGS = ["train", "test", "val", "sampled"]


# -----------------------------
# MODEL PATH (PER CLASS)
# -----------------------------
def get_model_path(base, cls, mode, epoch):

    if mode == "best":
        return f"{base}/checkpoints/{cls}_best.pt"
    elif mode == "untrained":
        return f"{base}/checkpoints/{cls}_untrained.pt"
    elif mode == "epoch":
        return f"{base}/checkpoints/{cls}_epoch_{epoch}.pt"
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# DATA
# -----------------------------
def get_data_path(base, cls, tag):
    return f"{base}/preprocessed_data/{cls}_{tag}_processed.h5"


# -----------------------------
# SUBPROCESS WRAPPERS
# -----------------------------
def run_sampling(model_path, output_file, args):
    subprocess.run([
        "python", "sample.py",
        "--model_path", model_path,
        "--output_file", output_file,
        "--n_jets", str(args.n_jets),
        "--batch_size", str(args.sample_batch_size),
        "--max_length", str(args.max_length),
        "--topk", str(args.topk),
    ], check=True)


def run_prob(model_path, data_path, output_file, args, sampled=False):

    cmd = [
        "python", "compute_probabilities.py",
        "--model_path", model_path,
        "--data_path", data_path,
        "--output_file", output_file,
        "--n_jets", str(args.n_jets),
        "--batch_size", str(args.batch_size),
        "--num_const", str(args.num_const),
    ]

    if sampled:
        cmd += ["--input_key", "sampled_jets", "--h5"]

    subprocess.run(cmd, check=True)


# -----------------------------
# CLASSIFIER
# -----------------------------
def compute_scores(folder, tag):
    scores = []
    labels = []

    for i, cls in enumerate(CLASSES):
        top_vs_i = pd.read_csv(f"{folder}/TTBar_{cls}_{tag}.csv")
        qcd_vs_i = pd.read_csv(f"{folder}/QCD_{cls}_{tag}.csv")

        s_i = top_vs_i["probs"] - qcd_vs_i["probs"]

        scores.append(s_i.values)
        labels.append(np.full(len(s_i), i))

    return np.concatenate(scores), np.concatenate(labels)


# -----------------------------
# MAIN
# -----------------------------
def main(args):

    # Build per-class config
    model_config = {
        "TTBar": (args.ttbar_mode, args.ttbar_epoch),
        "QCD": (args.qcd_mode, args.qcd_epoch),
    }

    # Naming for output
    name = f"TTBar-{args.ttbar_mode}"
    if args.ttbar_mode == "epoch":
        name += f"{args.ttbar_epoch}"

    name += f"_QCD-{args.qcd_mode}"
    if args.qcd_mode == "epoch":
        name += f"{args.qcd_epoch}"

    base_outdir = f"{args.output}/{name}"
    os.makedirs(base_outdir, exist_ok=True)

    sampled_cache_dir = f"{base_outdir}/sampled_jets"
    os.makedirs(sampled_cache_dir, exist_ok=True)

    all_scores, all_labels, all_tags = [], [], []
    roc_data = {}

    print(f"Running evaluation: {name}")

    for tag in TAGS:

        print(f"\n--- TAG: {tag} ---")

        tag_outdir = f"{base_outdir}/{tag}"
        os.makedirs(tag_outdir, exist_ok=True)

        sampled_files = {}

        # -----------------------------
        # SAMPLING (PER CLASS MODEL!)
        # -----------------------------
        if tag == "sampled":

            for cls in CLASSES:

                mode, epoch = model_config[cls]

                model_path = get_model_path(
                    args.training_folder,
                    cls,
                    mode,
                    epoch
                )

                sampled_file = f"{sampled_cache_dir}/{cls}.h5"

                if not os.path.exists(sampled_file):
                    print(f"Sampling {cls} → {sampled_file}")
                    run_sampling(model_path, sampled_file, args)
                else:
                    print(f"Using cached sampled jets: {sampled_file}")

                sampled_files[cls] = sampled_file

        # -----------------------------
        # PROBABILITIES
        # -----------------------------
        for model_cls in CLASSES:
            for data_cls in CLASSES:

                mode, epoch = model_config[model_cls]

                model_path = get_model_path(
                    args.training_folder,
                    model_cls,
                    mode,
                    epoch
                )

                if tag == "sampled":
                    data_path = sampled_files[data_cls]
                    sampled_flag = True
                else:
                    data_path = get_data_path(args.data_folder, data_cls, tag)
                    sampled_flag = False

                output_file = f"{tag_outdir}/{model_cls}_{data_cls}_{tag}.csv"

                print(f"{model_cls}({mode}) on {data_cls} ({tag})")

                run_prob(model_path, data_path, output_file, args, sampled_flag)

        # -----------------------------
        # SCORING
        # -----------------------------
        scores, labels = compute_scores(tag_outdir, tag)

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        roc_data[tag] = (fpr, tpr, roc_auc)

        print(f"[RESULT] {tag} AUC = {roc_auc:.6f}")

        all_scores.append(scores)
        all_labels.append(labels)
        all_tags.append(np.full(len(scores), tag))

    # -----------------------------
    # SAVE SCORES
    # -----------------------------
    df = pd.DataFrame({
        "score": np.concatenate(all_scores),
        "label": np.concatenate(all_labels),
        "tag": np.concatenate(all_tags),
    })
    df.to_csv(f"{base_outdir}/all_scores.csv", index=False)

    # -----------------------------
    # ROC PLOT
    # -----------------------------
    plt.figure()

    for tag in TAGS:
        fpr, tpr, roc_auc = roc_data[tag]
        plt.plot(tpr, 1/fpr, label=f"{tag} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.yscale("log")

    plt.xlabel(r"$\epsilon_{top}$")
    plt.ylabel(r"$1/\epsilon_{QCD}$")

    plt.title(f"ROC Curves ({name})")
    plt.legend()

    plt.savefig(f"{base_outdir}/roc_all_tags.png", dpi=150)
    plt.close()
    # -----------------------------
    # SUMMARY
    # -----------------------------
    with open(f"{base_outdir}/summary.txt", "w") as f:
        for tag in TAGS:
            f.write(f"{tag}: {roc_data[tag][2]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_folder", required=True)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--output", required=True)

    # per-model control
    parser.add_argument("--ttbar_mode", choices=["epoch", "best", "untrained"], required=True)
    parser.add_argument("--qcd_mode", choices=["epoch", "best", "untrained"], required=True)

    parser.add_argument("--ttbar_epoch", type=int, default=None)
    parser.add_argument("--qcd_epoch", type=int, default=None)

    parser.add_argument("--n_jets", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_const", type=int, default=100)

    parser.add_argument("--sample_batch_size", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5000)

    args = parser.parse_args()

    main(args)