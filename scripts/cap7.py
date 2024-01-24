# %% [markdown]
# # Telling birds from airplanes

# %%
from torchvision import datasets
import torch

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data_path = "data/p1ch7/"

# %%
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

# %%
len(cifar10)

# %%
img, label = cifar10[99]

# %%
plt.imshow(img);

# %%
from torchvision import transforms

# %%
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape

# %%
plt.imshow(img_t[0]);

# %%
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())

# %%
img_t, _ = tensor_cifar10[99]

# %%
plt.imshow(img_t[0]);

# %% [markdown]
# ## Normalizing Data

# %%
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
imgs.shape

# %%
imgs.view(3,-1).mean(dim=1)

# %%
imgs.view(3, -1).std(dim=1)

# %%
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

# %%
transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
]))

# %%
img_t, _ = transformed_cifar10[99]

# %%
plt.imshow(img_t.permute(1, 2, 0));

# %%
label_map = {
    0: 0,
    2: 1
}

# %%
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]

cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

# %%
import torch.nn as nn

# %%
n_out = 2

# %%
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, n_out)
)

# %%
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

# %%
x = torch.tensor([1.0, 2.0, 3.0])
softmax(x)

# %%
softmax = nn.Softmax(dim=1)

# %%
x = torch.tensor([[1.0, 2.0, 3.0], 
                  [1.0, 2.0, 3.0]])
softmax(x)

# %%
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, n_out),
    nn.Softmax(dim=1)
)

# %%
img, _ = cifar2[0]
plt.imshow(img);

# %%
img_batch = imgs.view(-1).unsqueeze(0)

# %%
img, label = cifar2[0]

# %%
from torchvision import datasets, transforms
data_path = 'data/p1ch7/'
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

# %%
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10 
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]

# %%
import torch.nn as nn

n_out = 2

model = nn.Sequential(
            nn.Linear(
                3072,  # <1>
                512,   # <2>
            ),
            nn.Tanh(),
            nn.Linear(
                512,   # <2>
                n_out, # <3>
            )
        )

# %%
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

# %%
softmax = nn.Softmax(dim=1)

x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]])

softmax(x)

# %%
model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1))

# %%
img, _ = cifar2[0]

plt.imshow(img.permute(1, 2, 0))
plt.show()

# %%
img_batch = img.view(-1).unsqueeze(0)


# %%
out = model(img_batch)
out

# %%
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1))

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

n_epochs = 100

for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

# %%



