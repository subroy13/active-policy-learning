from abc import ABC
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.optimize import minimize

from .utils.helper import bradley_terry_objective, bradley_terry_gradient, ContextualBradleyTerry


class PairComparisonModelBase(ABC):
    """
    This is a base abstract class for all pair comparison models
    """

    def __init__(self, n_arms: int, context_dim: int):
        self.K = n_arms
        self.context_dim = context_dim

    def mhat_predict(self, arm_a: int, arm_b: int, context: np.ndarray) -> float:
        """
        Return estimates P(arm_a wins | context) for arm_a < arm_b
        """
        raise NotImplementedError("Abstract method")

    def mhat_fit(self, dataset: list[dict]):
        raise NotImplementedError("Abstract method")

    def get_mhat_a_b(self, arm_a: int, arm_b: int, context: Any) -> float:
        """Return the estimated win probability for any ordered pair (arm_a, arm_b).

        Handles arm ordering by delegating to mhat_predict with the canonical (min, max) order.
        """
        if arm_a < arm_b:
            prob = self.mhat_predict(arm_a, arm_b, context)
        elif arm_a == arm_b:
            prob = 0.5  # mhat for same arms
        else:
            prob = 1 - self.mhat_predict(arm_b, arm_a, context)  # swap to get the prob
        return prob

    def update_mhat_a_b(self, dataset: list[dict]):
        """
        Refit the model, normalizing arm order.
        """
        cleaned_dataset = []
        for row in dataset:
            arm_a, arm_b = row["arm_a"], row["arm_b"]
            if arm_a < arm_b:
                cleaned_dataset.append({"context": row["context"], "arm_a": arm_a, "arm_b": arm_b, "Y": row["Y"]})
            else:
                cleaned_dataset.append({"context": row["context"], "arm_a": arm_b, "arm_b": arm_a, "Y": 1 - row["Y"]})

        # perform the fit
        self.mhat_fit(cleaned_dataset)

    def get_rho(self):
        """
        Return the weights for estimating equation
        """
        rho = np.ones((self.K, self.K)) - np.eye(self.K)
        return rho

    def get_all_pairwise_mhat(self, context: np.ndarray) -> pd.DataFrame:
        rows = []
        for a in range(self.K):
            for b in range(a + 1, self.K):
                # arm_a < arm_b
                prob = self.mhat_predict(a, b, context)
                rows.append({"arm_a": a, "arm_b": b, "mhat": prob})
                rows.append({"arm_a": b, "arm_b": a, "mhat": 1 - prob})
            rows.append({"arm_a": a, "arm_b": a, "mhat": 0.5})

        return pd.DataFrame(rows)

    def estimate_theta(self, context: np.ndarray, pair_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Estimate the Bradley-Terry utility parameters
        """
        if hasattr(self, "get_estimated_theta"):
            return self.get_estimated_theta(context)

        if pair_df is None:
            pair_df = self.get_all_pairwise_mhat(context)

        # based on the sample df, estimate the theta's
        rho = self.get_rho()  # (K, K)
        M = pair_df.pivot(index="arm_a", columns="arm_b", values="mhat").to_numpy()  # (K, K)

        # optimize the bradley terry objective
        result = minimize(
            fun=bradley_terry_objective,
            x0=np.zeros(self.K - 1),  # useful to work with last best guess of theta?
            args=(rho, M),
            jac=bradley_terry_gradient,
            method="L-BFGS-B",
        )
        theta_hat = np.zeros(self.K)
        theta_hat[1:] = result.x
        return theta_hat

    def get_udiff_theta(self, theta_hat: np.ndarray):
        """
        Compute the Jacobian of the BT score equaitions with respect to theta
        """
        theta_hat1 = theta_hat[1:]  # theta_1 = 0
        theta_diff = theta_hat1[:, None] - theta_hat1[None, :]  # (K-1, K-1)
        rho = self.get_rho()  # (K, K)
        udiff_theta = -rho[1:, 1:] * expit(theta_diff) * (1 - expit(theta_diff))
        np.fill_diagonal(udiff_theta, 0)  # set diagonals to 0

        udiff_theta_diag = -udiff_theta.sum(axis=1) + rho[0, 1:] * expit(theta_hat1) * (1 - expit(theta_hat1))
        np.fill_diagonal(udiff_theta, udiff_theta_diag)
        return udiff_theta

    def estimate_gamma(self, theta_hat: np.ndarray, sample_row: dict, Y: float) -> np.ndarray:
        """
        Compute the IPW-corrected influence-function estimate of arm utilities (gamma).
        """
        theta_hat1 = theta_hat[1:]
        udiff_theta = self.get_udiff_theta(theta_hat)  # (K-1, K-1)

        # create error vector
        udiff_m_e_vector = np.zeros(self.K - 1)  # (K-1,) - vector to hold result of (del_m U * e)
        Ydiff = Y - sample_row["mhat"]

        # if Y > mhat(a, b) => Ydiff > 0 => theta_a(hat) is underestimated (need to increase) and theta_b(hat) is overestimated
        W_ab = sample_row["prob"]
        rho = self.get_rho()
        arm_a, arm_b = int(sample_row["arm_a"]), int(sample_row["arm_b"])
        rho_ab = rho[arm_a, arm_b]
        if arm_a > 0:
            udiff_m_e_vector[arm_a - 1] = (Ydiff / W_ab) * rho_ab  # IPW weighting
        if arm_b > 0:
            udiff_m_e_vector[arm_b - 1] = (-Ydiff / W_ab) * rho_ab

        adjustment = np.linalg.solve(udiff_theta + 1e-8 * np.eye(self.K - 1), udiff_m_e_vector)
        gamma_hat = np.zeros(self.K)
        gamma_hat[1:] = theta_hat1 + adjustment
        return gamma_hat


class PairComparisonLogisticModel(PairComparisonModelBase):

    def __init__(self, n_arms: int, context_dim: int, c=0.01):
        super().__init__(n_arms, context_dim)
        self.c = c  # this is used for thresholding to avoid mhat ~ 0 or 1
        self.models_dict: Dict[Tuple[int, int], Union[LogisticRegression, float]] = {}

    def mhat_fit(self, dataset: list[dict]):
        # always get cleaned dataset, so arm_a < arm_b
        for a in range(self.K):
            for b in range(a + 1, self.K):
                subdata = [row for row in dataset if (row["arm_a"] == a and row["arm_b"] == b)]

                if len(subdata) == 0:
                    # no data present yet
                    continue  # don't fit any model

                unique_yvals = list(set([row["Y"] for row in subdata]))
                Xvals = [row["context"] for row in subdata]
                Yvals = [row["Y"] for row in subdata]
                Xvals = np.array(Xvals).reshape(-1, self.context_dim)  # (n, d)
                Yvals = np.array(Yvals).reshape(-1)  # (n, )

                if len(unique_yvals) > 1:
                    self.models_dict[(a, b)] = LogisticRegression(penalty="l2").fit(Xvals, Yvals)
                elif unique_yvals[0] == 0:
                    # all values are 0, if # of samples is less, prob ~ 0.5, but # of samples high, prob ~ c
                    self.models_dict[(a, b)] = self.c + (0.5 - self.c) / Xvals.shape[0]
                else:
                    # all values are 1,
                    self.models_dict[(a, b)] = 1 - (self.c + (0.5 - self.c) / Xvals.shape[0])

    def mhat_predict(self, arm_a: int, arm_b: int, context: np.ndarray) -> float:
        model = self.models_dict.get((arm_a, arm_b))
        if model is None:
            # no data present
            return 0.5
        elif isinstance(model, float) or isinstance(model, int):
            return model
        else:
            # use model to predict
            probs = model.predict_proba(np.array(context).reshape(1, -1))
            return float(
                min(1 - self.c, max(self.c, probs[0][1]))
            )  # ! sklearn sorts classes so (Y = 0 is first index, then Y = 1)


class PairComparisonBTModel(PairComparisonModelBase):
    """
    This class directly estimates the Bradley-Terry parameters for the pair comparison model
    """

    def __init__(self, n_arms, context_dim, c=0.01):
        """Initialize the underlying contextual Bradley-Terry model.

        Args:
            n_arms: Total number of arms K.
            context_dim: Dimension of the context vector.
        """
        super().__init__(n_arms, context_dim)
        self.bt_model = ContextualBradleyTerry(n_arms, context_dim, l2=0.5)
        self.c = c  # this is used for thresholding to avoid mhat ~ 0 or 1

    def mhat_predict(self, arm_a: int, arm_b: int, context: Any) -> float:
        """Predict P(arm_a wins | context) via the fitted Bradley-Terry model.

        Args:
            arm_a: Index of the first arm (must be < arm_b).
            arm_b: Index of the second arm.
            context: Context vector.

        Returns:
            Estimated win probability in [0, 1].
        """
        prob = self.bt_model.predict_proba(context, arm_a, arm_b)
        return float(min(1 - self.c, max(self.c, prob)))

    def mhat_fit(self, dataset: list[dict]):
        dataset_tuples = [(np.array(row["context"]), row["arm_a"], row["arm_b"], row["Y"]) for row in dataset]
        self.bt_model.fit(dataset_tuples)  # fit bt model

    def get_estimated_theta(self, context: np.ndarray) -> np.ndarray:
        x = np.array(context).reshape(-1)
        return self.bt_model.theta @ x

    def get_posterior(self, dataset: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns the mean (w_map) and covariance matrix of the parameters."""
        dataset_tuples = [(np.array(row["context"]), row["arm_a"], row["arm_b"], row["Y"]) for row in dataset]
        return self.bt_model.get_laplace_posterior(dataset_tuples)
