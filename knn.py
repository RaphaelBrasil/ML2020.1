import torch
import pandas as pd
import numpy as np
import torch.distributions.uniform as dist
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# Create color maps
cmap_light = ListedColormap(['#b3c5ec', '#ebecb3'])
cmap_bold = ListedColormap(['#1568db', '#ffce3c'])

def pdist2(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def plot(x, y, ran):
    for i in range(ran):
        if model.predict((x[i], y[i])) == 1:
            plt.scatter(x[i], y[i], marker='o', c="green")
        else:
            plt.scatter(x[i], y[i], marker='o', c="blue")


train = pd.read_csv('dados-ex1-1.csv')
train_tensor = torch.tensor(train.values)
data_x = train_tensor[:, :2]
data_y = train_tensor[:, -1]


# print(train_tensor)
train_data, test_data, train_targets, test_targets = train_test_split(data_x, data_y)


class KNN:
    def __init__(self, K, train_data, train_targets):
        super(KNN, self).__init__()
        self.data = train_data
        self.K = K
        self.targets = train_targets

    def predict(self, x):
        dists = pdist2(x, self.data)  # computa dist. ponto a ponto
        idxs = dists.argsort()[:, :self.K]  # orderna e separa os indices dos k mais proximos
        knns = self.targets[idxs]  # seleciona os rotulos dos vizinhos
        return knns.mode(dim=1).values  # retorna os valores da moda nos K vizinhos


model = KNN(10, data_x, data_y)  # usando K=5

zx, zy = torch.meshgrid((torch.arange(-1, 1.7, 0.01)), torch.arange(-1, 1.7, 0.01))

zx = zx.reshape(zx.shape[0] * zx.shape[1])
zy = zy.reshape(zy.shape[0] * zy.shape[1])

ax = np.array(zx)
ay = np.array(zy)
xy = (ax, ay)

zz = torch.tensor(xy)
zz = torch.t(zz)
plt.subplot(221)
plt.title('teste')
plt.scatter(zz[:, 0], zz[:, 1], c=model.predict(zz), cmap=cmap_light)
plt.scatter(data_x[:, 0], data_x[:, 1], c=model.predict(data_x), marker="x", cmap=cmap_bold, linewidths=0.01)

plt.subplot(222)
plt.scatter(zz[:, 0], zz[:, 1], c=model.predict(zz), cmap=cmap_light)
plt.scatter(data_x[:, 0], data_x[:, 1], c=model.predict(data_x), marker="x", cmap=cmap_bold, linewidths=0.01)

plt.subplot(223)
plt.scatter(zz[:, 0], zz[:, 1], c=model.predict(zz), cmap=cmap_light)
plt.scatter(data_x[:, 0], data_x[:, 1], c=model.predict(data_x), marker="x", cmap=cmap_bold, linewidths=0.01)
plt.show()


