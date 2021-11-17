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

    def log_lik_data(self, y, y_hat, log_gamma):
        # Likelihood
        return - 0.5 * self.n * (np.log(2 * np.pi) - log_gamma) - (torch.exp(log_gamma) / 2) * torch.sum(torch.power(y_hat - y, 2))

    def log_prior_data(self, log_gamma):
        # Prior on log gamma
        # return (self.a0 - 1) * log_gamma - self.b0 * torch.exp(log_gamma) + (log_gamma<----???)
        return self.a0 * torch.log(self.b0) - torch.lgamma(self.a0) + (self.a0 - 1) * log_gamma - self.b0 * torch.exp(log_gamma)

    def log_prior_w(self, log_lambda, w):
        # Prior on w
        return -0.5 * (self.d - 2) * (np.log(2 * np.pi) - log_lambda) - (torch.exp(log_lambda) / 2) * ((w**2).sum()) +\
            self.a0 * torch.log(self.b0) - torch.lgamma(self.a0) + (self.a0 - 1) * log_lambda - self.b0 * torch.exp(log_lambda)

    def log_posterior(self, phi, y, w, log_lambda, log_gamma):
        # Phi === [BasisFct | Linear | Bias]
        # Posterior on w

        return self.log_lik_data(log_gamma, y, phi @ w) * y.shape[0] / self.n + self.log_prior_data(log_gamma) + self.log_prior_w(log_lambda, w)

    def grad_log_posterior(self, y: torch.Tensor, phi: torch.Tensor, w: torch.Tensor, log_lambda: torch.Tensor, log_gamma: torch.Tensor):
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
        w = w.requires_grad(True); log_lambda = log_lambda.requires_grad(True); log_gamma = log_gamma.requires_grad(True)
        log_posterior = self.log_posterior(phi, y, w, log_lambda, log_gamma)
        dw, d_log_gamma, d_log_lambda = torch.autograd.grad(log_posterior, [w, log_gamma, log_lambda])
        w = w.requires_grad(False); log_lambda = log_lambda.requires_grad(False); log_gamma = log_gamma.requires_grad(False)
        return dw, d_log_gamma, d_log_lambda

    def sample(self):
        w = 1.0 / np.sqrt(self.d + 1) * torch.randn(size=self.d)
        loggamma = torch.distributions.gamma.Gamma(self.a0, self.b0).sample().log()
        loglambda = torch.distributions.gamma.Gamma(self.a0, self.b0).sample().log()
        return (w, b, loggamma, loglambda)


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

    def __init__(self, X_train, y_train, batch_size=10, num_epochs=500, learning_rate=1e-3, svdg_iters=500, M=20, n_units_1=50, n_units_2=50, n_units_3=50, a0=1.0, b0=.1, linear=True, bias=True, normalize_input=True, normalize_output=True, rng=None):
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

        # Network hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_iter = svdg_iters
        self.init_learning_rate = learning_rate

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
        self.theta = np.zeros([self.M, self.dim_vars])

    @BaseModel._check_shapes_train
    def train(self, X, y):
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
        features = X.shape[1]

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
                loss = torch.nn.functional.mse_loss(output, targets) + 0.1 * repulsion_force(phi)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))
            pbar.set_description(desc=f"Epoch {epoch}, Trainng loss: {train_err / train_batches:.3f}")
        # Design matrix
        self.Phi = self.network.basis_funcs(torch.Tensor(self.X)).data.numpy().astype(np.float64)

        for i in range(self.M):
            w, loggamma, loglambda = self.posterior.sample()
            # use better initialization for gamma
            ridx = np.random.choice(range(self.Phi.shape[0]), np.min([self.Phi.shape[0], 1000]), replace=False)
            y_hat = self.Phi[ridx] @ w
            loggamma = -np.log(np.mean(np.power(y_hat - y[ridx], 2)))
            self.theta[i, :] = self.pack_weights(w, loggamma, loglambda)

        grad_theta = np.zeros([self.M, self.dim_vars])  # gradient
        # adagrad with momentum
        eps = 1e-6
        v = 0
        m = 0

        pbar = tqdm(range(self.max_iter), ncols=150)
        for iter in pbar:
            # sub-sampling

            for batch in self.iterate_minibatches(self.X, self.y, batch_size, shuffle=True):
                inputs = torch.Tensor(batch[0])
                targets = torch.Tensor(batch[1])
                phi = self.network.basis_funcs(torch.Tensor(inputs)).data.numpy().astype(np.float64)
            for i in range(self.M):
                w, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                dw, dloggamma, dloglambda = self.posterior.grad_log_posterior(phi, targets, )
                grad_theta[i, :] = self.pack_weights(dw, dloggamma, dloglambda)

            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(h=-1)
            current_coef = self.annealer.getCoef(iter)
            grad_theta = (current_coef * np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)

            # adagrad
            if iter == 0:
                m = m + grad_theta
                v = v + np.multiply(grad_theta, grad_theta)
                m_ = m
                v_ = v
            else:
                m = beta_1 * m + (1 - beta_1) * grad_theta
                v = beta_2 * v + (1 - beta_2) * np.multiply(grad_theta, grad_theta)
                m_ = m / (1 - beta_1**iter)
                v_ = v / (1 - beta_2**iter)

            adj_grad = np.divide(m_, eps + np.sqrt(v_))
            self.theta = self.theta + master_stepsize * adj_grad
            pbar.set_description(f"Current Coef  {current_coef:.3f}")
            '''
                Model selection by using a development set
            '''
            X_dev = self.normalization(X_dev)
            val = [0 for _ in range(self.M)]
            for i in range(self.M):
                w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
                # likelihood
                def f_log_lik(loggamma): return np.sum(np.log(np.sqrt(np.exp(loggamma)) / np.sqrt(2 * np.pi)
                                                              * np.exp(-1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma))))
                # The higher probability is better
                lik1 = f_log_lik(loggamma)
                # one heuristic setting
                loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
                lik2 = f_log_lik(loggamma)
                if lik2 > lik1:
                    self.theta[i, -2] = loggamma  # update loggamma

    def pack_weights(self, w, loggamma, loglambda):
        '''
            Pack all parameters in our model
        '''
        params = np.concatenate([w.flatten(), loggamma, loglambda])
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

    def marginal_log_likelihood(self, theta):
        """
        Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            lnlikelihood + prior
        """
        if np.any(theta == np.inf):
            return -np.inf

        if np.any((-10 > theta) + (theta > 10)):
            return -np.inf

        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.linalg.LinAlgError:
            print("inversion didn't worked!")
            K_inv = np.linalg.inv(K + np.random.rand(K.shape[0], K.shape[1]) * 1e-8)

        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        # mll -= 0.5 * np.log(np.linalg.det(K) + 1e-8) # instable
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            import pdb; pdb.set_trace()
        mll -= 0.5 * logdet

        if np.any(np.isnan(mll)):
            return -1e25
        return mll

    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll

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

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Get features from the net

        theta = self.network.basis_funcs(torch.Tensor(X_)).data.numpy()

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu ** 2 + var, axis=0) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(DNGO, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
