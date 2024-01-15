# %% [markdown]
# # Pretrained Networks

# %%
from torchvision import models

# %%
import torch

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dir(models)

# %%
alexnet = models.AlexNet()

# %%
alexnet

# %%
resnet = models.resnet101(pretrained=True)
resnet

# %%
from torchvision import transforms

# %%
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# %%
from PIL import Image

# %%
img = Image.open("images/dog.jpg")

# %%
img.show()

# %%
img_t = preprocess(img)

# %%
plt.imshow(img_t[0], cmap="gray");

# %%
batch_t = torch.unsqueeze(img_t, 0)

# %%
resnet.eval()

# %%
out = resnet(batch_t)

# %%
_, index = torch.max(out, 1)

# %%
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# %%
index, percentage[index[0]]

# %% [markdown]
# ## GANs

# %%
netG = ResNetGenerator()

# %%



