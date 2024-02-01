from scipy import stats
import numpy as np
from utils import *
import random

random.seed(0)
np.random.seed(0)


def sample_one_dataset(
    X,
    s,
    R_y,
    a,
    b,
    A,
    B,
):
    # Initialize R2 using beta distribution
    T, k = X.shape
    R2 = stats.beta(A, B).rvs()
    # Initialize q using beta distribution
    q = stats.beta(a, b).rvs()
    # Initialize z as a random vector of s ones and k-s zeros
    z = np.array([0] * (k - s) + [1] * s)
    np.random.shuffle(z)

    # Initialize beta as a random vector of k values
    beta = (np.random.randn(k) * z).reshape(-1, 1)

    sigma2 = (1 / R_y - 1) / T * np.sum((X @ beta) ** 2)
    eps = np.random.multivariate_normal(np.zeros(T), sigma2 * np.eye(T)).reshape(-1, 1)

    # Standardize X and eps
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    eps = (eps - np.mean(eps)) / np.std(eps)

    # Sanity checks
    assert X.shape == (T, k)
    assert eps.shape == (T, 1)
    assert beta.shape == (k, 1)
    assert z.shape == (k,)
    assert R2 > 0 and R2 < 1
    assert q > 0 and q < 1
    return beta, eps, sigma2, R2, q, z


def posterior_R2_q(X, R2, q, sigma2, beta_tilde_norm2, s, a=1, b=1, A=1, B=1):
    """
    Auxiliary function to compute the posterior on a grid of R2 and q values
    """

    vx = 1  # X is standardized
    k = X.shape[1]
    # Compute the log of the posterior
    log_posterior = (
        -1 / (2 * sigma2) * (k * vx * q * (1 - R2)) / (R2) * beta_tilde_norm2
        + (s + s / 2 + a - 1) * np.log(q)
        + (k - s + b - 1) * np.log(1 - q)
        + (A - 1 - s / 2) * np.log(R2)
        + (B - 1 + s / 2) * np.log(1 - R2)
    )
    return np.exp(log_posterior)


def compute_posterior_grid(X, Rs, qs, z, beta, sigma2):
    """Compute the posterior on a grid of R2 and q values"""
    s = int(np.sum(z))
    beta_tilde = beta[z == 1]
    beta_tilde_norm2 = beta_tilde.T @ beta_tilde
    # Compute posterior
    posterior = posterior_R2_q(X, Rs, qs, sigma2, beta_tilde_norm2, s)
    return posterior


def sample_joint_R2_q(X, Rs, qs, z, beta, sigma2):
    """
    Sample R2 and q jointly using the posterior on a grid of R2 and q values
    """
    posterior = compute_posterior_grid(X, Rs, qs, z, beta, sigma2)
    posterior = posterior / np.sum(posterior)
    random_idx = np.random.choice(posterior.size, p=posterior.flatten())
    R2_idx, q_idx = np.unravel_index(random_idx, posterior.shape)
    R2 = Rs[R2_idx, q_idx]
    q = qs[R2_idx, q_idx]
    return R2, q


def sample_z(X, Y, eps, beta, z, R2, q):
    """Sample z using one gibbs iteration"""
    T = X.shape[0]
    gamma = np.sqrt(compute_gamma2(X, R2, q))
    Y_tilde = Y
    for i in range(z.shape[0]):
        # Compute W_tilde_0 and W_tilde_1 depending on z_i state
        z[i] = 0
        X_tilde_0 = X[:, z == 1]
        W_tilde_0 = X_tilde_0.T @ X_tilde_0 + np.eye(int(np.sum(z))) / gamma**2
        z[i] = 1
        X_tilde_1 = X[:, z == 1]
        W_tilde_1 = X_tilde_1.T @ X_tilde_1 + np.eye(int(np.sum(z))) / gamma**2

        # Fast computation of beta_tilde_0 and beta_tilde_1
        beta_tilde_0 = np.linalg.solve(
            W_tilde_0.astype(float), X_tilde_0.astype(float).T @ Y_tilde.astype(float)
        )
        beta_tilde_1 = np.linalg.solve(
            W_tilde_1.astype(float), X_tilde_1.astype(float).T @ Y_tilde.astype(float)
        )

        # Fast computation of the log-determinant of W_tilde_0 and W_tilde_1
        log_det_W_tilde_0 = np.linalg.slogdet(W_tilde_0)[1]
        log_det_W_tilde_1 = np.linalg.slogdet(W_tilde_1)[1]

        # # Compute the log of the probability ratio
        log_ratio = (
            np.log(gamma * (1 - q) / q)
            - 1 / 2 * log_det_W_tilde_0
            + 1 / 2 * log_det_W_tilde_1
            - T
            / 2
            * np.log((Y_tilde.T @ Y_tilde - beta_tilde_0.T @ W_tilde_0 @ beta_tilde_0))
            + T
            / 2
            * np.log((Y_tilde.T @ Y_tilde - beta_tilde_1.T @ W_tilde_1 @ beta_tilde_1))
        ).item()

        # Compute the probability of state z_i = 1
        prob = 1 / (1 + np.exp(log_ratio))
        # Sample z_i
        if np.random.rand() < prob:
            z[i] = 1
        else:
            z[i] = 0
    return z


def compute_gamma2(X, R2, q):
    """Compute gamma^2 using the formula given in the assignment"""
    vx = 1  # X is standardized
    k = X.shape[1]
    return R2 / ((1 - R2) * k * q * vx)


def sample_sigma2_scale(X, Y, R2, q, z):
    """Sample sigma2 scale"""
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    Y_tilde = Y
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / compute_gamma2(X, R2, q)
    # Fast computation of beta_tilde_hat
    beta_tilde_hat = np.linalg.solve(
        W_tilde.astype(float), X_tilde.astype(float).T @ Y_tilde.astype(float)
    )
    scale = (Y_tilde.T @ Y_tilde - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat) / 2
    return scale


def sample_sigma2(X, Y, eps, beta, R2, q, z):
    """Sample sigma2 using formula 4"""
    T = X.shape[0]
    scale = sample_sigma2_scale(X, Y, R2, q, z)
    return stats.invgamma(T / 2, scale=scale).rvs()


def sample_beta_tilde_param(X, Y, eps, beta, R2, q, sigma2, z):
    """Sample beta_tilde mean and covariance matrix"""
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    Y_tilde = Y
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / compute_gamma2(X, R2, q)
    W_tilde_inv = np.linalg.inv(W_tilde)
    beta_tilde_hat = W_tilde_inv @ X_tilde.T @ Y_tilde
    mean = beta_tilde_hat.reshape(-1)
    cov = sigma2 * W_tilde_inv
    return mean, cov


def sample_beta_tilde(X, Y, eps, beta, R2, q, sigma2, z):
    """Sample beta_tilde using formula 5"""
    mean, cov = sample_beta_tilde_param(X, Y, eps, beta, R2, q, sigma2, z)
    return np.random.multivariate_normal(mean.astype(float), cov.astype(float)).reshape(
        -1, 1
    )


def sample_beta(X, Y, eps, beta, R2, q, sigma2, z):
    """Auxiliary function to sample beta from beta_tilde"""
    k = X.shape[1]
    beta_tilde = sample_beta_tilde(X, Y, eps, beta, R2, q, sigma2, z)
    beta = np.zeros((k, 1))
    beta[z == 1] = beta_tilde
    return beta.reshape(-1, 1)


def one_gibbs_iteration(X, Y, eps, R2, q, z, sigma2, beta, Rs, qs):
    """Run one iteration of the Gibbs sampler"""
    R2, q = sample_joint_R2_q(X, Rs, qs, z, beta, sigma2)
    sampled_z = sample_z(X, Y, eps, beta, z, R2, q)
    if sampled_z.sum() == 0:
        return R2, q, z, sigma2, beta
    else:
        z = sampled_z
    sigma2 = sample_sigma2(X, Y, eps, beta, R2, q, z)
    beta = sample_beta(X, Y, eps, beta, R2, q, sigma2, z)
    return R2, q, z, sigma2, beta


def compute_one_dataset(X, Y, a, b, A, B, R_y, s, N_iter=1000):
    """Sample one dataset and run the Gibbs sampler for N_iter iterations"""
    beta, eps, sigma2, R2, q, z = sample_one_dataset(X, s, R_y, a, b, A, B)
    q_chain = np.zeros(N_iter)
    # Create grid of R2 and q values
    x = np.concatenate(
        [
            np.arange(0.001, 0.1, 0.001),
            np.arange(0.1, 0.9, 0.01),
            np.arange(0.9, 0.999, 0.001),
        ]
    )
    Rs, qs = np.meshgrid(x, x)
    for iter in range(N_iter):
        R2, q, z, sigma2, beta = one_gibbs_iteration(
            X, Y, eps, R2, q, z, sigma2, beta, Rs, qs
        )
        q_chain[iter] = q
    return beta
