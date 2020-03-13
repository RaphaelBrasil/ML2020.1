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
from sklearn.metrics import mean_squared_error


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


# Create color maps
cmap_light = ListedColormap(['#b3c5ec', '#ebecb3'])
cmap_bold = ListedColormap(['#1568db', '#ffce3c'])


def funct(x):
    y = torch.pow((2 * x), 2) * torch.sin(15 * x)
    return y


def pdist2(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def teste1(zx, zy, k, pos):
    model = KNN(k, data_x, data_y)  # usando K=5

    zx = zx.reshape(zx.shape[0] * zx.shape[1])
    zy = zy.reshape(zy.shape[0] * zy.shape[1])

    ax = np.array(zx)
    ay = np.array(zy)
    xy = (ax, ay)

    zz = torch.tensor(xy)
    zz = torch.t(zz)

    plt.subplot(2, 2, pos)
    plt.title('Para k = ' + str(k))
    plt.scatter(zz[:, 0], zz[:, 1], c=model.predict(zz), cmap=cmap_light)
    plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, marker="x", cmap=cmap_bold, linewidths=0.01)


trainSet = pd.read_csv('dados-ex1-1.csv')
trainTensor = torch.tensor(trainSet.values)
data_x = trainTensor[:, :2]
data_y = trainTensor[:, -1]


# print(train_tensor)
train_data, test_data, train_targets, test_targets = train_test_split(data_x, data_y)

#For Classification
zx, zy = torch.meshgrid((torch.arange(-1, 1.7, 0.01)), torch.arange(-1, 1.7, 0.01))

for a in ([1, 1], [3, 2], [9, 3], [27, 4]):
        teste1(zx, zy, a[0], a[1])
#teste1(zx, zy, 3, 2)
#teste1(zx, zy, 9, 3)
#teste1(zx, zy, 27, 4)
plt.show()

# For Regression
xTeste = torch.linspace(-1, 1, steps=1000).reshape(-1, 1)
pos = 1

for k in (1, 3, 5):
    for s in (5, 10, 100):

        x = torch.linspace(-1, 1, steps=s).reshape(-1, 1)
        model = KNN(k, x, funct(x))

        if k == 3:
            plt.subplot(2, 2, pos)
            plt.title('Para k = ' + str(k) + ' e trainSet = ' + str(s))
            plt.scatter(xTeste, model.predict(xTeste), cmap=cmap_light)
            pos = pos + 1

        print('O MSE para k = ' + str(k) + ' é: '
              + str(format(mean_squared_error(funct(xTeste), model.predict(xTeste)), '.2f'))+ ' para trainSet = ' + str(s))

plt.show()
