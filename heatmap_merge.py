import numpy as np
import matplotlib.pyplot as plt

epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18]
tag = "train"

n_jets = 600000
num_const = 50


folder = f"JETCLASS_{n_jets}_warmup_cosine"
base = f"output/evaluation/{folder}/heatmap_{tag}/points"

heatmap = np.zeros((len(epochs), len(epochs)))

for i, tt in enumerate(epochs):
    for j, qcd in enumerate(epochs):
        f = f"{base}/TTBar_{tt}_QCD_{qcd}.txt"
        heatmap[i, j] = float(open(f).read())

plt.imshow(heatmap, origin="lower")
plt.xticks(range(len(epochs)), epochs)
plt.yticks(range(len(epochs)), epochs)

plt.title(f"N_jets = {n_jets} with {num_const} const. on {tag} data")

plt.xlabel("QCD epoch")
plt.ylabel("TTBar epoch")
plt.colorbar(label="AUC")

plt.savefig(f"{base}/heatmap.png", dpi=150)