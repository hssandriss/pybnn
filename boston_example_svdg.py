import random

import numpy as np
import torch

from pybnn import SVGD
from pybnn.util.normalization import zero_mean_unit_var_normalization

np.random.seed(1)

''' load dfalseata file '''
data = np.loadtxt('data/boston_housing')

# Please make sure that the last column is the label and the other columns are features
X_input = data[:, range(data.shape[1] - 1)]
y_input = data[:, data.shape[1] - 1]

''' build the training and testing data set'''
train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data
permutation = np.arange(X_input.shape[0])
random.shuffle(permutation)

size_train = int(np.round(X_input.shape[0] * train_ratio))
index_train = permutation[0: size_train]
index_test = permutation[size_train:]

X, y = X_input[index_train, :], y_input[index_train]
X_test, y_test = X_input[index_test, :], y_input[index_test]

model = SVGD(X, y, rng=np.random.RandomState(12345))
model.train()


x_test_norm = zero_mean_unit_var_normalization(X_test, model.X_mean, model.X_std)[0]

# Get basis functions from the network
basis_funcs = model.network.basis_funcs(torch.Tensor(x_test_norm)).data.numpy()
m, v = model.predict(X_test)
