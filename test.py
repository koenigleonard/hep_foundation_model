import numpy as np
import pandas as pd
import dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import model

input_file = "processed_data/debug_small_train.h5"
#input_file = "/net/data_ttk/hreyes/JetClass/JetClass_pt_part/JetClass_pt_part/TTBar_test.h5"

dataloader = DataLoader(
    dataset.JetDataSet(input_file, "train"),
    batch_size=5
)

testModel = model.JetTransformer()
i = 1
for x in dataloader:
    logits = testModel.forward(x)
    testModel.probability(logits, x)



