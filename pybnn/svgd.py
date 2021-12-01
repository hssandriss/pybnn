import logging
import time

import emcee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import optimize
from tqdm import tqdm

from pybnn.base_model import BaseModel
from pybnn.bayesian_linear_regression import BayesianLinearRegression, Prior
from pybnn.util.normalization import (zero_mean_unit_var_denormalization,
                                      zero_mean_unit_var_normalization)


def repulsion_force(Phi):
    ans_ = torch.exp(-F.pdist(Phi.t(), 2))
    return ans_.mean()


class TruePosterior:
    """
    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda)
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead."""

    def __init__(self, n, d, a0, b0) -> None:
        self.n, self.d = n, d  # n is batch size and d is dimenstion of w
        self.a0 = a0
        self.b0 = b0

    def log_lik_data(self, y, y_hat, loggamma):
        # Likelihood
        return -0.5 * y_hat.shape[0] * (np.log(2 * np.pi) - loggamma) - (torch.exp(loggamma) / 2) * torch.sum(torch.pow(y_hat - y, 2))

    def log_prior_data(self, loggamma):
        # Prior on log gamma
        return (self.a0 - 1) * loggamma - self.b0 * torch.exp(loggamma) + loggamma

    def log_prior_w(self, loglambda, w):
        # Prior on w
        return -0.5 * (self.d - 2) * (np.log(2 * np.pi) - loglambda) - (torch.exp(loglambda) / 2) * (torch.pow(w, 2).sum()) +\
            (self.a0 - 1) * loglambda - self.b0 * torch.exp(loglambda) + loglambda

    def log_posterior(self, phi, y, w, loglambda, loggamma):
        # Phi === [BasisFct | Linear | Bias]
        # Posterior on w
        return self.log_lik_data(y, phi @ w, loggamma) * (self.n / y.shape[0]) + self.log_prior_data(loggamma) + self.log_prior_w(loglambda, w)

    def grad_log_posterior(self, phi: torch.Tensor, y: torch.Tensor, w: torch.Tensor, loglambda: torch.Tensor, loggamma: torch.Tensor):
        """Computes the gradient of the log posterior 

        Args:
            phi (torch.Tensor): Basis function
            y (torch.Tensor): Ground Truth
            w (torch.Tensor): Weights of the last layer
            log_lambda (torch.Tensor): log-precision of prior parameter
            log_gamma (torch.Tensor): log-precision of observation noise

        Returns:
            [type]: [description]
        """
        w.requires_grad = True; loglambda.requires_grad = True; loggamma.requires_grad = True
        log_posterior = self.log_posterior(phi, y, w, loglambda, loggamma)
        dw, d_log_gamma, d_log_lambda = torch.autograd.grad(log_posterior, [w, loggamma, loglambda])
        w.requires_grad = False; loglambda.requires_grad = False; loggamma.requires_grad = False
        return dw, d_log_gamma, d_log_lambda

    def sample(self):
        w = 1.0 / np.sqrt(self.d + 1) * torch.randn(size=(self.d, 1))
        loggamma = torch.distributions.Gamma(self.a0, self.b0).sample().log()
        loglambda = torch.distributions.Gamma(self.a0, self.b0).sample().log()
        return (w, loggamma, loglambda)


class Net(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50]):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return self.out(x)

    def basis_funcs(self, x, bias=False, linear=False):
        raw_x = x
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        if linear:
            x = torch.cat((x, raw_x), dim=-1)
        if bias:
            x = torch.cat((x, torch.ones(size=(raw_x.shape[0], 1))), dim=-1)
        return x


class SVGD(BaseModel):

    def __init__(self, X_train, y_train, batch_size=10, num_epochs=50, learning_rate=1e-3, svdg_iters=500, M=20, n_units_1=50, n_units_2=50, n_units_3=50, a0=1.0, b0=.1, linear=True, bias=True, normalize_input=True, normalize_output=True, rng=None, beta_1=0.9, beta_2=0.99, svgd_step=0.001):
        """
        Deep Networks for Global Optimization [1]. This module performs
        Bayesian Linear Regression with basis function extracted from a
        feed forward neural network.

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        n_units_1: int
            Number of units in layer 1
        n_units_2: int
            Number of units in layer 2
        n_units_3: int
            Number of units in layer 3
        n_units_4: int
            Number of units in layer 4
        a0: float
            Hyperparameter of the Bayesian linear regression Gamma prior
        b0: float
            Hyperparameter of the Bayesian linear regression Gamma prior
        M:  int
            Number of particles are used to fit the posterior distribution
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X_train)
        else:
            self.X = X_train

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_train)
        else:
            self.y = y_train
        self.y = self.y[:, None]

        self.n, self.d = self.X.shape[0], self.X.shape[1]   # number of data, dimension

        # Number of particles
        self.M = M
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # Network hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_iter = svdg_iters
        self.init_learning_rate = learning_rate
        self.svgd_learning_rate = svgd_step

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3

        self.network = None
        self.models = []

        self.linear = linear
        self.bias = bias

        # Posterior information
        self.dim_w = self.n_units_3
        if self.linear:
            self.dim_w += self.d
        if self.bias:
            self.dim_w += 1
        self.dim_loggamma = 1
        self.dim_loglambda = 1
        self.dim_vars = self.dim_w + self.dim_loggamma + self.dim_loglambda
        self.a0 = a0
        self.b0 = b0
        self.posterior = TruePosterior(self.n, self.dim_w, self.a0, self.b0)
        # Particle initialization
        self.theta = torch.zeros([self.M, self.dim_vars])

    @BaseModel._check_shapes_train
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters are used.

        """
        start_time = time.time()

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X_train.shape[1]

        self.network = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_2, self.n_units_3])

        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.init_learning_rate)

        pbar = tqdm(range(self.num_epochs))
        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in pbar:

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y, batch_size, shuffle=True):
                inputs = torch.Tensor(batch[0])
                targets = torch.Tensor(batch[1])

                optimizer.zero_grad()
                phi = self.network.basis_funcs(inputs)
                output = self.network.out(phi)
                loss = F.mse_loss(output, targets) + 0.1 * repulsion_force(phi)
                loss.backward()
                optimizer.step()
                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            # print(f"epoch/total training time: {epoch_time}/{total_time}")
            pbar.set_description(desc=f"Epoch {epoch+1}/{self.num_epochs} , Trainng loss: {train_err / train_batches:.3f}")

        # Design matrix
        self.Phi = self.network.basis_funcs(torch.Tensor(self.X), bias=self.bias, linear=self.linear).data
        # The basis function is now trained and in the next steps we define SVGD iterrations
        for i in range(self.M):
            w, loggamma, loglambda = self.posterior.sample()
            # use better initialization for gamma
            ridx = np.random.choice(range(self.Phi.shape[0]), np.min([self.Phi.shape[0], 1000]), replace=False)
            y_hat = self.Phi[ridx] @ w
            loggamma = -torch.log(torch.mean(np.power(y_hat - torch.from_numpy(y_train[ridx]), 2)))
            self.theta[i, :] = self.pack_weights(w, loggamma, loglambda)

        grad_theta = torch.zeros(size=(self.M, self.dim_vars))  # gradient
        # adagrad with momentum
        eps = 1e-6
        v = 0
        m = 0
        pbar = tqdm(range(self.max_iter), ncols=150)
        for iter in pbar:
            # sub-sampling
            for batch in self.iterate_minibatches(self.Phi, self.y, batch_size, shuffle=True):
                phi = torch.Tensor(batch[0])
                targets = torch.Tensor(batch[1])
                for i in range(self.M):
                    w, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                    dw, dloggamma, dloglambda = self.posterior.grad_log_posterior(phi, targets, w, loglambda, loggamma)
                    grad_theta[i, :] = self.pack_weights(dw, dloggamma, dloglambda)

                # calculating the kernel matrix
                kxy, dxkxy = self.svgd_kernel(h=-1)
                grad_theta = (torch.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)

                # adagrad
                if iter == 0:
                    m = m + grad_theta
                    v = v + torch.multiply(grad_theta, grad_theta)
                    m_ = m
                    v_ = v
                else:
                    m = self.beta_1 * m + (1 - self.beta_1) * grad_theta
                    v = self.beta_2 * v + (1 - self.beta_2) * torch.multiply(grad_theta, grad_theta)
                    m_ = m / (1 - self.beta_1**iter)
                    v_ = v / (1 - self.beta_2**iter)

                adj_grad = m_.div_(eps + np.sqrt(v_))
                # adj_grad = np.divide(m_, eps + np.sqrt(v_))
                self.theta = self.theta + self.svgd_learning_rate * adj_grad

            # '''
            #     Model selection by using a validation set
            # '''
            # X_dev = self.normalization(X_dev)
            # val = [0 for _ in range(self.M)]
            # for i in range(self.M):
            #     w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            #     pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            #     # likelihood

            #     def f_log_lik(loggamma):
            #         return np.sum(np.log(np.sqrt(np.exp(loggamma)) / np.sqrt(2 * np.pi) * np.exp(-1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma))))
            #     # The higher probability is better
            #     lik1 = f_log_lik(loggamma)
            #     # one heuristic setting
            #     loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            #     lik2 = f_log_lik(loggamma)
            #     if lik2 > lik1:
            #         self.theta[i, -2] = loggamma  # update loggamma

    def svgd_kernel(self, h=-1):
        sq_dist = torch.pdist(self.theta)
        pairwise_dists = squareform(self.theta.shape[0], sq_dist)**2
        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0] + 1))
        # compute the rbf kernel
        Kxy = torch.exp(-pairwise_dists / h**2 / 2)
        dxkxy = -torch.matmul(Kxy, self.theta)
        sumkxy = torch.sum(Kxy, axis=1)

        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + torch.multiply(self.theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)

    def nn_predict(self, phi, w):
        """
        docstring
        """
        return phi @ w

    def pack_weights(self, w, loggamma, loglambda):
        '''
            Pack all parameters in our model
        '''
        params = torch.cat([w.view(-1), loggamma.view(-1), loglambda.view(-1)], dim=0)
        return params

    def unpack_weights(self, z):
        '''
            Unpack all parameters in our model
        '''
        assert z.shape[-1] == self.dim_vars
        w_ = z
        w = np.reshape(w_[:self.dim_w], [self.dim_w, 1])

        # the last two parameters are log variance
        loggamma, loglambda = w_[-2], w_[-1]

        return (w, loggamma, loglambda)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0], \
            "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


def squareform(n: int, dist: torch.Tensor):
    assert dist.shape[-1] == int(0.5 * n * (n - 1))
    sq = torch.zeros(n, n).type(dist.dtype)
    i, j = torch.triu_indices(n, n, 1)
    sq[i, j] = dist; sq[j, i] = dist
    return sq
