import numpy as np
import pandas as pd
import dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import model
from helpers_train import *

input_file = "processed_data/debug_small_train.h5"
#input_file = "/net/data_ttk/hreyes/JetClass/JetClass_pt_part/JetClass_pt_part/TTBar_test.h5"

dataloader = DataLoader(
    dataset.JetDataSet(input_file, "train"),
    batch_size=5
)

testModel = load_model("output/model_test.pt")
i = 1
for x in dataloader:
    logits = testModel.forward(x)
    print(x)
    print(testModel.probability(logits, x, logarithmic=True, 
                                #topk=5000
                                ))



