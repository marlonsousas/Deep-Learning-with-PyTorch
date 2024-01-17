# %% [markdown]
# # It starts with a tensor

# %%
import torch

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
a = torch.ones(3)
a

# %%
a[1]

# %%
a[0]

# %%
points = torch.zeros(6)
points

# %%
points[0] = 4.0
points[0]

# %%
points.shape

# %%
points.shape[0]

# %%
points = torch.tensor([[1, 2, 3, 4, 5], 
                       [6, 7, 8, 9, 10], 
                       [11, 12, 13, 14, 15]])
points

# %%
points.shape

# %%
some_list = list(range(6))
some_list

# %%
some_list[:]

# %%
some_list[1:4]

# %%
img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# %%
batch_t = torch.randn(2, 3, 5, 5)

# %%
plt.imshow(img_t[0])

# %%
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
img_gray_naive.shape, batch_gray_naive.shape

# %%
a = torch.tensor([[1, 2, 3],[3, 4, 5],[5, 6, 7]])
#a = torch.ones(3, 2)
a

# %%
a_t = a.transpose(0, 1)
a_t

# %%
b = torch.tensor([[1, 2*4, 3],[3, 4, 5],[5, 6, 7]])
b

# %%
torch.multiply(a, b)

# %%
a.storage()

# %%
a*a_t

# %%
points_gpu = torch.tensor([[1, 2*4, 3],[3, 4, 5],[5, 6, 7]], device="cuda")
points_gpu

# %%
torch.cuda.get_device_name()

# %%
a = a.to(device="cuda")

# %%
a

# %%
torch.save(points, "data/p1ch3/outpoints.t")

# %%
torch.save(points, "data/p1ch4/outpoints.t")

# %%
new_points = torch.load("data/p1ch3/outpoints.t")
new_points

# %%



