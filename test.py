import numpy as np
import pandas as pd
import dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import JetTransformer
from helpers_train import *
import matplotlib.pyplot as plt
import ipykernel

device = "cpu"

testModel = JetTransformer()

testModel = load_model_checkpoint("output/checkpoints/debug_best.pt")

testSet = DataLoader(
    JetDataSet("processed_data/TTBar_5000_test.h5", "test", n_jets=2, num_const = 50),
    batch_size=1,
)

probabilities = np.array([])

testTuples = torch.tensor([[[-1, -1, -1],[-1, -1, -1]]])

print(f"testModel.tuple_to_index(-1, -1, -1, [43, 33, 33])={testModel.tuple_to_index(testTuples[..., 0], testTuples[..., 1], testTuples[..., 2], [43, 33, 33])}")

for x in testSet:
    
    #print(f"targets {x}")

    logits = testModel.forward(x)
    probs = testModel.probability(logits, x, logarithmic= True, topk = 5000)

    probabilities = np.append(probabilities, probs.detach().numpy())
