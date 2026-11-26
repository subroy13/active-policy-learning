"""
Active learning sampling strategies
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as linalg
import pandas as pd

from .pairmodels import PairComparisonModelBase


class SamplingPolicy(ABC):

    def __init__(self, n_arms: int, context_dim: int):
        self.K = n_arms
        self.d = context_dim
        self.initial_round = 4 * self.K**2

    def get_uniform_sampling_weights(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        pair_df["prob"] = 1.0
        pair_df["prob"] = pair_df["prob"] / pair_df["prob"].sum()  # normalize
        return pair_df

    @abstractmethod
    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ) -> pd.DataFrame:
        """
        Appends the 'prob' column to the pairwise dataset
        """
        raise NotImplementedError("Abstract method")

    def pick_sample(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        round: int,
        seed: int = 1234,
        data_history: list = [],
    ):
        if round < self.initial_round:
            pair_df = self.get_uniform_sampling_weights(pair_df)  # run initial rounds with uniformly collected samples
        else:
            pair_df = self.get_sampling_weights(pair_df, theta_hat, pair_model, current_context, data_history)
        sampled_row: dict = pair_df.sample(n=1, weights="prob", random_state=seed).to_dict("records")[0]
        return sampled_row

    def update_state(self, context: np.ndarray, arm_a: int, arm_b: int, Y: float):
        pass


class UniformSamplingPolicy(SamplingPolicy):

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ):
        return self.get_uniform_sampling_weights(pair_df)

class RoundRobinSamplingPolicy(SamplingPolicy):

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ) -> pd.DataFrame:
        n_rounds = len(data_history)
        n_pairs = len(pair_df)
        pair_df["prob"] = 1e-2
        if n_pairs > 0:
            selected_index = pair_df.index[n_rounds % n_pairs]
            pair_df.loc[selected_index, "prob"] = 1.0
        pair_df["prob"] = pair_df['prob'] / pair_df['prob'].sum()  # normalize            
        return pair_df
        
        


class AdaptiveSamplingPolicy(SamplingPolicy):

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ) -> pd.DataFrame:
        pair_df["mhat_var"] = pair_df["mhat"] * (1 - pair_df["mhat"])
        mask = pair_df["arm_a"] != pair_df["arm_b"]
        pair_df["prob"] = pair_df["mhat_var"].copy()
        pair_df.loc[~mask, "prob"] = 0  # avoid sampling same pairs
        pair_df["prob"] = pair_df['prob'] / pair_df['prob'].sum()  # normalize        
        return pair_df


class MaxEntSamplingPolicy(SamplingPolicy):

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ) -> pd.DataFrame:
        dist = (pair_df["arm_a"] != pair_df["arm_b"]) * ((pair_df["mhat"] - 0.5).abs()) + (
            pair_df["arm_a"] == pair_df["arm_b"]
        ) * 1
        pair_df["prob"] = 1e-2  # for IPW, require a nonzero minimum weight
        pair_df.loc[dist == dist.min(), "prob"] = 1.0
        pair_df["prob"] = pair_df["prob"] / pair_df["prob"].sum()  # normalize
        return pair_df


class DoubleThompsonSampling(SamplingPolicy):
    """
    Contextual Double Thompson Sampling (D-TS)
    - P(y = 1 | i, j, theta) = sigmoid(theta_i - theta_j)
    - Prior: theta ~ N(0, lambda^{-1} I )
    - Posterior: P(data | theta) P(theta) => apply Laplace approximation:
    - theta | data ~ N(mu_t, sigma_t)
    - mu_t = argmin [lambda / 2 * ||theta||^2 - \\sum_{s=1}^t log P(y_s | a_s, b_s, theta) ]
    - sigma_t^{-1} = lambda I + \\sum_{s=1}^t sigma(mu_s(a_s) - mu_s(b_s) ) (1 - sigma(...)) x_s x_s'
    - Method:
        - Draw sample theta ~ posterior
        - Select champion (first arm) that maximizes theta
        - Draw another sample theta'
        - Select challenger (second arm) that maximizes (theta' - theta' of champion)
    """

    def __init__(self, n_arms, context_dim):
        super().__init__(n_arms, context_dim)
        self.d_eff = (self.K - 1) * self.d  # effective dimension

    def _unpack(self, w):
        theta = np.zeros((self.K, self.d))
        theta[1:, :] = w.reshape(-1, self.d)  # unpack theta
        return theta

    def select_pair(self, x, posterior_mean, posterior_precision):
        """
        x: Context vector of shape (d,)
        posterior_mean: MAP estimate of shape ((K-1)*d,)
        posterior_precision: Hessian matrix of shape ((K-1)*d, (K-1)*d)
        """
        # 1. Cholesky decomposition of the Precision Matrix (Hessian)
        # lower=True guarantees L * L^T = H
        try:
            L = linalg.cholesky(posterior_precision, lower=True)
        except linalg.LinAlgError:
            # Fallback in case of severe numerical instability (e.g., matrix not positive definite)
            # Add a small ridge to the diagonal
            ridge = np.eye(self.d_eff) * 1e-5
            L = linalg.cholesky(posterior_precision + ridge, lower=True)

        # 2. Draw Sample 1 (Champion)
        z1 = np.random.randn(self.d_eff)
        # Solve L^T y1 = z1.  trans=1 solves A^T x = b
        y1 = linalg.solve_triangular(L, z1, trans=1, lower=True)
        w1 = posterior_mean + y1

        theta1 = self._unpack(w1)
        champion = np.argmax(theta1 @ x)

        # 3. Draw Sample 2 (Challenger)
        z2 = np.random.randn(self.d_eff)
        y2 = linalg.solve_triangular(L, z2, trans=1, lower=True)
        w2 = posterior_mean + y2

        theta2 = self._unpack(w2)
        scores2 = theta2 @ x

        # Challenger is the best arm excluding the Champion
        scores2[champion] = -np.inf
        challenger = np.argmax(scores2)

        return min(champion, challenger), max(champion, challenger)

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ):
        if not hasattr(pair_model, "get_posterior"):
            raise ValueError("pair_model must implement get_posterior() for DoubleThompsonSampling")
        # use the pair model to get MAP and Hessian
        posterior_mean, posterior_precision = pair_model.get_posterior(data_history)  # type: ignore
        a, b = self.select_pair(current_context, posterior_mean, posterior_precision)
        pair_df["prob"] = 1e-2
        matched_idx = ((pair_df["arm_a"] == a) & (pair_df["arm_b"] == b)) | (
            (pair_df["arm_b"] == a) & (pair_df["arm_a"] == b)
        )
        pair_df.loc[matched_idx, "prob"] = 1.0
        pair_df["prob"] = pair_df["prob"] / pair_df["prob"].sum()  # normalize
        return pair_df


class MaxIntPSamplingPolicy(SamplingPolicy):
    """
    It uses the MaxIntP policy.
    Reference: https://proceedings.neurips.cc/paper/2021/hash/fc3cf452d3da8402bebb765225ce8c0e-Abstract.html
    """

    def __init__(self, n_arms, context_dim, lambda_reg=1.0, beta=1.0):
        super().__init__(n_arms, context_dim)
        self.beta = beta

        # Track the INVERSE covariance matrix of the Kd feature space
        self.V_inv = np.eye(self.K * self.d) / lambda_reg

    def _get_Z(self, a, x):
        """Construct the Kd-dimensional feature vector for arm a."""
        Z = np.zeros(self.K * self.d)
        Z[a * self.d : (a + 1) * self.d] = x
        return Z

    def get_pair(self, x, theta_scores):
        """
        x: current context vector (d,)
        theta_scores: (K,) array output from PairComparisonBTModel.estimate_theta(x)
        """
        U = np.zeros(self.K)
        L = np.zeros(self.K)
        Z_list = []

        for a in range(self.K):
            Z_a = self._get_Z(a, x)
            Z_list.append(Z_a)

            # Feature-space uncertainty width
            width = self.beta * np.sqrt(np.dot(Z_a.T, np.dot(self.V_inv, Z_a)))

            # theta_scores[a] is provided directly by your ContextualBradleyTerry class!
            U[a] = theta_scores[a] + width
            L[a] = theta_scores[a] - width

        # Form Candidate Set (Arms that might still be the best)
        max_L = np.max(L)
        candidate_set = [a for a in range(self.K) if U[a] >= max_L]

        if len(candidate_set) < 2:
            candidate_set = np.argsort(U)[-2:]

        # Find Maximum Informative Pair
        max_variance = -1
        best_pair = (candidate_set[0], candidate_set[1])

        for i in range(len(candidate_set)):
            for j in range(i + 1, len(candidate_set)):
                a, b = candidate_set[i], candidate_set[j]
                X_ab = Z_list[a] - Z_list[b]

                var_ab = np.dot(X_ab.T, np.dot(self.V_inv, X_ab))

                if var_ab > max_variance:
                    max_variance = var_ab
                    best_pair = (a, b)

        return best_pair

    def get_sampling_weights(
        self,
        pair_df: pd.DataFrame,
        theta_hat: np.ndarray,
        pair_model: PairComparisonModelBase,
        current_context: np.ndarray,
        data_history: list = [],
    ):
        pair_df["prob"] = 1e-2  # for IPW, require a nonzero minimum weight
        a, b = self.get_pair(current_context, theta_hat)
        matched_idx = ((pair_df["arm_a"] == a) & (pair_df["arm_b"] == b)) | (
            (pair_df["arm_b"] == a) & (pair_df["arm_a"] == b)
        )
        pair_df.loc[matched_idx, "prob"] = 1.0
        pair_df["prob"] = pair_df["prob"] / pair_df["prob"].sum()  # normalize
        return pair_df  # already normalized

    def update_state(self, context, arm_a, arm_b, Y):
        """Update V_inv using Sherman-Morrison after the pair is selected."""
        X_ab = self._get_Z(arm_a, context) - self._get_Z(arm_b, context)

        numerator = np.outer(np.dot(self.V_inv, X_ab), np.dot(X_ab.T, self.V_inv))
        denominator = 1.0 + np.dot(X_ab.T, np.dot(self.V_inv, X_ab))
        self.V_inv -= numerator / denominator
