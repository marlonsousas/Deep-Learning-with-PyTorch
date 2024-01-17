# %% [markdown]
# # Real-world data representation using tensors

# %%
import imageio

# %%
import torch

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
img_arr = imageio.imread("images/dog.jpg")
img_arr

# %%
plt.imshow(img_arr);

# %%
plt.imshow(img_arr/(255*2));

# %%
img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)

# %%
img.shape

# %%
plt.imshow(out[0]);

# %%
df = pd.read_csv("data/wine.csv", sep=";")
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df["quality"].value_counts()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
x = df.drop("quality", axis=1).values
y = df["quality"].values

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
st = StandardScaler()

# %%
st.fit(x_train)

# %%
x_train = st.transform(x_train)
x_test = st.transform(x_test)

# %%
scores = {}
models = [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(), SVC()]

for model in models:
    model.fit(x_train, y_train)
    scores[f"{model}"] = model.score(x_test, y_test)

# %%
scores

# %%
sns.scatterplot(data=df, x="total sulfur dioxide", y="total sulfur dioxide", hue="quality")

# %%
sns.pairplot(data=df, hue="quality")

# %% [markdown]
# ## Converting text to numbers

# %%
with open("data/p1ch4/1342.txt", encoding="utf8") as f:
    text = f.read()

# %%
text

# %% [markdown]
# ## One-hot-encoding characters

# %%
lines = text.split("\n")
line = lines[200]
line

# %%
letter_t = torch.zeros(len(line), 128)
letter_t.shape

# %%
letter_t

# %%
for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

# %%
def clean_words(input_str):
    punctuation = '.,;:"“”!?_-'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

# %%
words_in_line = clean_words(line)
line, words_in_line

# %%
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

# %%
len(word2index_dict), word2index_dict['impossible']

# %%



