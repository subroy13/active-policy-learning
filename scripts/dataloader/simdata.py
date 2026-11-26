from typing import Any
import numpy as np
from scipy.optimize import minimize

from ..dataloader.base import PairComparisonDataLoader
from ..utils.helper import bradley_terry_gradient, bradley_terry_objective


class DiscreteUniformSetup(PairComparisonDataLoader):
    """
    Generate dataset according to the following rule:
    X ~ DiscreteUniform(1, 2, ..., K)
    theta_a(X) = -(X - a)^2, for a in [K]
    """

    def __init__(self, seed: int = 1234):
        super().__init__(seed)
        self.n_arms = 5

    def generate_context(self, seed: int) -> Any:
        np.random.seed(seed)
        return np.array(np.random.randint(self.n_arms)).reshape(1)

    def generate_theta_given_context(self, context: np.ndarray, seed: int) -> np.ndarray:
        thetas = -((np.arange(self.n_arms) - context) ** 2)
        thetas = thetas - thetas[0]  # so that theta_1 = 0
        return thetas


def li_li_discrete_context(seed):
    """Sample a 5-dimensional discrete context from the Li & Li feature map.

    Draws X1 from {-3,...,3} and X2 from Bernoulli(0.5), then returns
    [X1, X2, X1^2, X2^2, X1*X2].

    Args:
        seed: Random seed.

    Returns:
        numpy array of shape (5,).
    """
    np.random.seed(seed)
    x1 = np.random.choice([-3, -2, -1, 0, 1, 2, 3])
    x2 = np.random.binomial(1, 0.5)
    return np.array([x1, x2, x1**2, x2**2, x1 * x2])


class LiAndLiCorrectDiscreteSetup(PairComparisonDataLoader):
    """
    This is similar to the one taken from Li and Li (Setting II)

    1. K = 5.
    2. X = (X_1, X_2), X_1 ~ N(0, 0.5^2), X_2 ~ Bernoulli(0.5)
    3. theta_vals:
        1. theta_2(x) = x1 * x2
        2. theta_3(x) = x_1^2 + x_2
        3. theta_4(x) = -0.5x1 + x2
        4. theta_5(x) = 0.5x1 + x2
    """

    def __init__(self, seed=1234):
        super().__init__(seed)
        self.n_arms = 5

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_discrete_context(seed)

    def generate_theta_given_context(self, context: np.ndarray, seed: int) -> np.ndarray:
        x1, x2 = context[0], context[1]
        theta = np.array([0, x1 * x2, x1**2 + x2, -0.5 * x1 + x2, 0.5 * x1 + x2])
        return theta


class LiAndLiMiss1DiscreteSetup(PairComparisonDataLoader):
    """
    Adds slight non-linearity in one arm
    """

    def __init__(self, seed=1234):
        super().__init__(seed)
        self.n_arms = 5

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_discrete_context(seed)

    def generate_theta_given_context(self, context: np.ndarray, seed: int) -> np.ndarray:
        x1, x2 = context[0], context[1]
        theta = np.array([0, x1 * x2, x1**2 + x2, -0.5 * x1 + x2, np.sin(1.5 * (0.5 * x1 + x2))])
        return theta


class LiAndLiMiss2DiscreteSetup(PairComparisonDataLoader):
    """
    This is taken from Li and Li (Setting I)
    1. K = 3
    2. X = (X_1, X_2), X_1 ~ N(0, 0.5^2), X_2 ~ Bernoulli(0.5)
    3. Wining prob:
        1. m_12(x) = 0.5 + 0.2 sin(1.5 (x_1 + x_2))
        2. m_13(x) = sigmoid(0.3 x_1 (x_2 - 1))
        3. m_23(x) = sigmoid(0.2x_1^2 - 0.5)
    """

    def __init__(self, seed=1234):
        super().__init__(seed)
        self.n_arms = 3

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_discrete_context(seed)

    def get_pairwise_comparison_prob_matrix(self, context: np.ndarray) -> np.ndarray:
        """Compute the full (K, K) pairwise win-probability matrix for the given context.

        Uses the non-Bradley-Terry misspecified model from Li & Li Setting I.

        Args:
            context: Context array; uses X1=context[0], X2=context[1].

        Returns:
            (K, K) matrix where entry (i, j) = P(arm i beats arm j | context).
        """
        M = np.zeros((self.n_arms, self.n_arms))  # (K, K), generate comparison matrix
        X1, X2 = context[0], context[1]

        # fill the winning prob functions
        M[0, 1] = 0.5 + 0.2 * np.sin(1.5 * (X1 + X2))
        M[0, 2] = 1 / (1 + np.exp(-(0.3 * X1 * (X2 - 1))))
        M[1, 2] = 1 / (1 + np.exp(-(0.2 * X1**2 - 0.5)))
        M[1, 0] = 1 - M[0, 1]
        M[2, 0] = 1 - M[0, 2]
        M[2, 1] = 1 - M[1, 2]
        return M

    def generate_theta_given_context(self, context: np.ndarray, seed: int) -> np.ndarray:
        W = np.ones((self.n_arms, self.n_arms)) - np.eye(
            self.n_arms
        )  # every pair is equally likely to be drawn, except diagonal
        M = self.get_pairwise_comparison_prob_matrix(context)

        result = minimize(
            fun=bradley_terry_objective,
            x0=np.zeros(self.n_arms - 1),  # useful to work with last best guess of theta?
            args=(W, M),
            jac=bradley_terry_gradient,
            method="L-BFGS-B",
        )
        theta_hat = np.zeros(self.n_arms)
        theta_hat[1:] = result.x
        return theta_hat

    def generate_pairwise_comparison(
        self, theta: np.ndarray, arm_a: int, arm_b: int, context: np.ndarray, seed: int = 1234
    ) -> float:
        """
        Sample a pairwise comparison from the true (non-BT) win probabilities
        """
        np.random.seed(seed)
        M = self.get_pairwise_comparison_prob_matrix(context)
        Y = np.random.binomial(1, M[arm_a, arm_b])
        return float(Y)


def li_li_continuous_context(seed):
    """Sample a 5-dimensional continuous context from the Li & Li feature map.

    Draws X1 ~ N(0, 0.5) and X2 ~ Bernoulli(0.5), then returns
    [X1, X2, X1^2, X2^2, X1*X2].

    Args:
        seed: Random seed.

    Returns:
        numpy array of shape (5,).
    """
    np.random.seed(seed)
    x1 = np.random.normal(0, np.sqrt(0.5))  # variance is 0.5
    x2 = np.random.binomial(1, 0.5)
    return np.array([x1, x2, x1**2, x2**2, x1 * x2])


class LiAndLiCorrectContinuousSetup(LiAndLiCorrectDiscreteSetup):
    """
    Li and Li correct model
    """

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_continuous_context(seed)


class LiAndLiMiss1ContinuousSetup(LiAndLiMiss1DiscreteSetup):
    """
    Li and Li misspecified 1 model
    """

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_continuous_context(seed)


class LiAndLiMiss2ContinuousSetup(LiAndLiMiss2DiscreteSetup):
    """
    Li and Li misspecified 2 model
    """

    def generate_context(self, seed: int) -> np.ndarray:
        return li_li_continuous_context(seed)
