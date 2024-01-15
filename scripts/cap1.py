# %% [markdown]
# # Introducing deep learning and the PyTorch Library

# %%
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

# %%



