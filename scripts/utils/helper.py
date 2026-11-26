from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


# --------------
# Bradley-Terry objective function
# --------------
def bradley_terry_objective(theta_free, W, M):
    """Compute the negative log-likelihood of the Bradley-Terry model.

    Args:
        theta_free: Unconstrained parameters (K-1,), excluding the baseline arm.
        W: Sampling weight matrix (K, K).
        M: Observed win-rate matrix (K, K).

    Returns:
        Scalar negative log-likelihood value.
    """
    theta = np.zeros(W.shape[0])
    theta[1:] = theta_free

    diff = theta[:, None] - theta[None, :]  # calculate pairwise distances
    return -np.sum(W * (M * diff - np.logaddexp(0, diff)))


def bradley_terry_gradient(theta_free, W, M):
    """Compute the gradient of the Bradley-Terry negative log-likelihood.

    Args:
        theta_free: Unconstrained parameters (K-1,), excluding the baseline arm.
        W: Sampling weight matrix (K, K).
        M: Observed win-rate matrix (K, K).

    Returns:
        Gradient vector of shape (K-1,) with respect to theta_free.
    """
    theta = np.zeros(W.shape[0])
    theta[1:] = theta_free

    diff = theta[:, None] - theta[None, :]
    P = expit(diff)

    grad = np.sum(W * (P - M), axis=1)
    return grad[1:]  # drop constrained coordinate from gradient calculation


def add_columns_to_list(arr: List[Dict], col_value_list: List[Tuple[str, Any]]):
    for a in arr:
        for colname, value in col_value_list:
            a[colname] = value
    return arr


class ContextualBradleyTerry:
    """Contextual Bradley-Terry model that learns per-arm linear utility functions."""

    def __init__(self, n_arms, context_dim, l2=1.0):
        """
        Args:
            n_arms: Number of arms (K).
            context_dim: Dimension of the context vector (d).
            l2: L2 regularization strength applied to non-baseline arms.
        """
        self.n_arms = n_arms
        self.d = context_dim
        self.l2 = l2
        self.theta = np.zeros((n_arms, context_dim))  # parameters stored as (K, d), baseline row will be zero

    def _pack(self, theta):
        """Remove baseline arm for optimization.

        Args:
            theta: Full parameter matrix (K, d).

        Returns:
            Flattened array (K-1)*d of non-baseline parameters.
        """
        return theta[1:, :].ravel()

    def _unpack(self, w):
        """Insert baseline arm back as zero.

        Args:
            w: Flattened non-baseline parameters of length (K-1)*d.

        Returns:
            Full parameter matrix (K, d) with the baseline row set to zero.
        """
        theta = np.zeros((self.n_arms, self.d))
        theta[1:, :] = w.reshape(-1, self.d)
        return theta

    def _neg_loglik(self, w, data):
        """Compute the regularized negative log-likelihood over pairwise observations.

        Args:
            w: Flattened non-baseline parameters.
            data: List of (context, arm_a, arm_b, y) tuples.

        Returns:
            Scalar negative log-likelihood plus L2 penalty.
        """
        theta = self._unpack(w)
        loss = 0.0

        for x, a, b, y in data:
            z = (theta[a] - theta[b]) @ x
            p = expit(z)
            loss -= y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)

        # L2 regularization (excluding baseline arm)
        loss += 0.5 * self.l2 * np.sum(w**2)
        return loss

    def _grad(self, w, data):
        """Compute the gradient of the regularized negative log-likelihood.

        Args:
            w: Flattened non-baseline parameters.
            data: List of (context, arm_a, arm_b, y) tuples.

        Returns:
            Gradient array of shape (K-1)*d.
        """
        theta = self._unpack(w)
        grad = np.zeros_like(theta)

        for x, a, b, y in data:
            z = (theta[a] - theta[b]) @ x
            p = expit(z)
            g = (p - y) * x
            grad[a] += g
            grad[b] -= g

        # remove baseline arm and add L2
        g_flat = grad[1:, :].ravel()
        g_flat += self.l2 * w
        return g_flat

    def fit(self, data, maxiter=200):
        """Fit the model via L-BFGS-B optimization over the provided data.

        Args:
            data: List of (context, arm_a, arm_b, y) tuples where y=1 means arm_a won.
            maxiter: Maximum number of L-BFGS-B iterations.

        Returns:
            self, with updated theta parameters.
        """
        w0 = self._pack(self.theta)

        res = minimize(
            fun=self._neg_loglik, x0=w0, jac=self._grad, args=(data,), method="L-BFGS-B", options={"maxiter": maxiter}
        )

        self.theta = self._unpack(res.x)
        return self

    def predict_proba(self, context, arm_a, arm_b):
        """Compute the probability that arm_a is preferred over arm_b given context.

        Args:
            context: Context vector of shape (d,).
            arm_a: Index of the first arm.
            arm_b: Index of the second arm.

        Returns:
            Scalar probability P(arm_a > arm_b | context).
        """
        z = (self.theta[arm_a] - self.theta[arm_b]) @ context
        return expit(z)

    def get_laplace_posterior(self, data):
        """
        Computes the MAP estimate and the Hessian (Precision matrix)
        for the non-baseline parameters using the Laplace approximation.
        """
        w_map = self._pack(self.theta)

        k_minus_1_d = (self.n_arms - 1) * self.d
        H = np.eye(k_minus_1_d) * self.l2

        for x, a, b, y in data:
            z = (self.theta[a] - self.theta[b]) @ x
            p = expit(z)
            variance = p * (1 - p)

            X_ab = np.zeros(k_minus_1_d)
            if a > 0:
                X_ab[(a - 1) * self.d : a * self.d] = x
            if b > 0:
                X_ab[(b - 1) * self.d : b * self.d] = -x

            H += variance * np.outer(X_ab, X_ab)

        # Return the Hessian DIRECTLY instead of the covariance
        return w_map, H
