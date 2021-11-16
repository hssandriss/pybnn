import random

import numpy as np
import scipy
import torch

from pybnn import DNGO
from pybnn.util.normalization import (inverse_z_score, whiten,
                                      whitening_params, z_score,
                                      z_score_params,
                                      zero_mean_unit_var_denormalization,
                                      zero_mean_unit_var_normalization)

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

X_train, y_train = X_input[index_train, :], y_input[index_train]
X_test, y_test = X_input[index_test, :], y_input[index_test]

model = DNGO(do_mcmc=True, batch_size=100, num_epochs=3000, learning_rate=1e-4)
model.train(X_train, y_train, do_optimize=True)


x_test_norm = zero_mean_unit_var_normalization(X_test, model.X_mean, model.X_std)[0]

# Get basis functions from the network
basis_funcs = model.network.basis_funcs(torch.Tensor(x_test_norm)).data.numpy()
m, v = model.predict(X_test)
ll = scipy.stats.norm.logpdf(y_test, loc=m, scale=v)
print(ll.mean())
print(np.mean(np.log(np.sqrt(2 * np.pi * v) * np.exp(-1 * (np.power(y_test - m, 2) / (2 * v))))))
import pdb; pdb.set_trace()


# ll = loglikelihood(m, v, y_test)
