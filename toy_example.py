import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os
import pathlib
from pybnn import DNGO
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
import scipy
import logging
import sys

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root.addHandler(handler)


def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
#     x = np.expand_dims(x, -1)
    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma)
    return x, y


def my_data_2_intervals(inter_1: tuple, inter_2: tuple, train=True):
    x_min, x_max, n0 = inter_1
    x_0 = np.linspace(x_min, x_max, n0)
    x_min, x_max, n1 = inter_2
    x_1 = np.linspace(x_min, x_max, n1)
    x = np.concatenate((x_0, x_1), 0)
    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma)
    return x, y


x, y = my_data_2_intervals((-4, 1, 1000), (4, 5, 200), train=True)

grid, fvals = my_data(-7, 7, 10000, train=False)

plt.plot(grid, fvals, "k--")
plt.plot(x, y, "ro")
plt.grid()
plt.xlim(-7, 7)

plt.show()

model = DNGO(do_mcmc=True, batch_size=64, num_epochs=1000, learning_rate=1e-4)
model.train(x[:, None], y, do_optimize=True)

model_tag = f'model_cubic_toy_bll_{datetime.now().strftime("%d_%m_%Y-%H_%M")}'
model_dir = os.path.join("experiments", model_tag)
pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
x_test_norm = zero_mean_unit_var_normalization(grid[:, None], model.X_mean, model.X_std)[0]

# Get basis functions from the network
basis_funcs = model.network.basis_funcs(torch.Tensor(x_test_norm)).data.numpy()

for i in range(min(50, model.n_units_3)):
    plt.plot(grid, basis_funcs[:, i])
plt.grid()
plt.gca().set_xlim(-7, 7)
plt.xlabel("Input")
plt.ylabel("Basisfunction")
plt.show()
plt.savefig(os.path.join(model_dir, f"basis_functions.png"))


y_pred_st = model.network(torch.Tensor(x_test_norm)).data.numpy()
y_pred = zero_mean_unit_var_denormalization(y_pred_st, model.y_mean, model.y_std)

plt.plot(grid, y_pred)
plt.scatter(x, y, s=1., c='#463c3c', zorder=0)
plt.grid()
plt.gca().set_ylim(-250, 250)
plt.gca().set_xlim(-7, 7)
plt.xlabel("Input")
plt.ylabel("y_pred")
plt.show()
plt.savefig(os.path.join(model_dir, f"regression.png"))
m, v = model.predict(grid[:, None])
ll = scipy.stats.norm.logpdf(fvals, loc=m, scale=v)
# np.mean(np.log(np.sqrt(2 * np.pi * v) * np.exp(-1 * (np.power(fvals - m, 2) / (2 * v)))))
print(ll.mean())
print(np.mean(np.log(np.sqrt(2 * np.pi * v) * np.exp(-1 * (np.power(fvals - m, 2) / (2 * v))))))


plt.scatter(x, y, s=1., c='#463c3c', zorder=0, label="Train")
plt.grid()
plt.plot(grid, fvals, 'r--', zorder=2, label="True")
plt.plot(grid, m, color='#007cab', zorder=3, label="Pred")
n_stds = 4
for k in np.linspace(0, n_stds, 4):
    plt.fill_between(
        grid, (m - k * np.sqrt(v)), (m + k * np.sqrt(v)),
        alpha=0.3,
        edgecolor=None,
        facecolor='#00aeef',
        linewidth=0,
        zorder=1,
        label="Unc." if k == 0 else None)

plt.gca().set_ylim(-250, 250)
plt.gca().set_xlim(-7, 7)
plt.title("(Gaussian) Bayesian Last Layer NN- MCMC")
plt.legend(loc="upper left")
plt.show()
plt.savefig(os.path.join(model_dir, f"prediction.png"))
