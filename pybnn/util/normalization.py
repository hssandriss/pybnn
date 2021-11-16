import numpy as np


def zero_one_normalization(X, lower=None, upper=None):

    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = np.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean


def z_score_params(X, eps=1e-5):
    """
    Implements z-score data normalization.

    Parameters:
        X (np.array)[*, k]: The un-normalized data.

    Keywords:
        eps (float): Small value for numerical stability.

    Returns:
        mu (np.array)[k]: The mean vector.
        sigma (np.array)[k]: The standard deviations.
    """
    std, mu = np.std(X, axis=0), np.mean(X, axis=0)
    return mu, std + eps


def whitening_params(X, eps=1e-5):
    """
    Implements ZCA data whitening.

    Parameters:
        X (np.array)[*, k]: The un-whitened data.

    Keywords:
        eps (float): Small value for numerical stability.

    Returns:
        ZCA_mean (np.array)[k]: The ZCA whitening mean.
        ZCA_cov (np.array)[k, k]: The ZCA whitening covariance.
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    Sigma = X_centered.T @ X_centered / X.shape[0]
    U, S, _ = np.linalg.svd(Sigma)
    ZCA_mean = X_mean
    ZCA_cov = U @ np.diag(np.sqrt(S + eps) ** -1) @ U.T
    return ZCA_mean, ZCA_cov


def whiten(x, mu, cov):
    """
    Applies ZCA whitening using previously computed mu and Sigma.

    Parameters:
        x (np.array)[n, k]: The un-whitened data.
        mu (np.array)[k]: The previously computed ZCA whitening mean vector.
        Sigma (np.array)[k, k]: The previously computed ZCA whitening covariance matrix.

    Returns:
        x' (np.array)[n, k]: The whitened data.
    """
    return (x - mu) @ cov


def z_score(X, mu, std):
    """
    Applies z_score normalization using previously computed mu and sigma.

    Parameters:
        x (np.array)[n, k]: The un-normalized data.
        mu (np.array)[k]: The previously computed mean vector.
        sigma (np.array)[k]: The previously computed standard deviations.

    Returns:
        x' (np.array)[n, k]: The normalized data.
    """
    return (X - mu) / std


def inverse_z_score(X, std, mu=None):
    """
    Reverses z_score normalization using previously computed mu and sigma.

    Parameters:
        X' (np.array)[*, k]: The normalized data.
        sigma (np.array)[k, k]: The previously computed standard deviations.

    Keywords:
        mu (np.array)[k]: Optional, previously computed mean vector.

    Returns:
        X (np.array)[*, k]: The un-normalized data.
    """
    if mu is None:
        return std ** 2 * X
    else:
        return std * X + mu
