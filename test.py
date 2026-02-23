import numpy as np
import h5py


hf = h5py.File("output/sampled_jets/jets.h5")

data = np.array(hf.get("sampled_jets"))
print(data)