from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Any
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from scipy.special import softmax


class RecommendationPolicy(ABC):
    """
    Base class for all recommendation policies
    """

    def __init__(self, n_arms: int, context_dim: int) -> None:
        self.K = n_arms
        self.d = context_dim

    def create_one_hot_vector(self, size: int, one_index: int):
        """
        Create a one-hot vector of given size with a 1 at one_index.
        """
        x = np.zeros(size)
        x[one_index] = 1
        return x


    @abstractmethod
    def recommend_policy(self, utility_list: List[Tuple[np.ndarray, np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a policy
        """
        raise NotImplementedError("Abstract method")

    def get_best_policy(
        self, gen_func: Callable[[int], Tuple[np.ndarray, np.ndarray]], seed: int = 1234, num_repeats: int = 1000
    ):
        """
        Compute the oracle best policy using true utility values sampled from generation_function.
        This is used to track the regret of a policy
        """
        np.random.seed(seed)
        seed_list = np.random.randint(low=0, high=999999, size=num_repeats)
        utility_list = []
        for b in range(num_repeats):
            context, theta = gen_func(seed_list[b])
            utility_list.append((context, theta))

        return self.recommend_policy(utility_list)


class KroneckerPolicy(RecommendationPolicy):

    def context_to_key(self, x: np.ndarray):
        return tuple(x.flatten())

    def recommend_policy(self, utility_list: List[Tuple[np.ndarray, np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
        X_values = list(set([self.context_to_key(x) for x, _ in utility_list]))
        X_values_mapping = {x: i for i, x in enumerate(X_values)}
        m = len(X_values)
        context_value_matrix = np.zeros((m, self.K))  # (m, K), rowwise = support(X)
        for x, v in utility_list:
            context_value_matrix[X_values_mapping[self.context_to_key(x)], :] += v

        best_arm_hat_idx = np.argmax(context_value_matrix, axis=1)  # (m, ), empirical maximizer for each choice of X

        def policy_fun(x):
            x_val = X_values_mapping.get(self.context_to_key(x))
            if x_val is None:
                # no data observed for this x, everything is equally likely
                return np.ones(self.K) / self.K
            else:
                return self.create_one_hot_vector(self.K, best_arm_hat_idx[x_val])

        return policy_fun

class KroneckerRankingPolicy(KroneckerPolicy):
    """Non-parametric policy that assigns softmax-normalized arm probabilities based on cumulative utility."""

    def recommend_policy(self, context_and_value_list: List[Tuple[Any, np.ndarray]]):
        """Return a softmax policy that weights arms by their relative cumulative utility per context.

        Args:
            context_and_value_list: List of (context, utility_vector) tuples.

        Returns:
            Policy function returning softmax arm probabilities for seen contexts, uniform otherwise.
        """
        X_values = list(set([self.context_to_key(x) for x, _ in context_and_value_list]))
        X_values_mapping = {x: i for i, x in enumerate(X_values)}
        m = len(X_values)
        context_value_matrix = np.zeros((m, self.K))  # (m, K), rowwise = support(X)
        for x, v in context_and_value_list:
            context_value_matrix[X_values_mapping[self.context_to_key(x)], :] += v

        def policy_ranking_fun(x):
            x_val = X_values_mapping.get(self.context_to_key(x))
            if x_val is None:
                # no data observed for this x, everything is equally likely
                return np.ones(self.K) / self.K
            else:
                context_value_row = context_value_matrix[x_val, :].copy()
                context_value_row -= context_value_row.min()
                context_value_row /= context_value_row.max()  # normalize
                return np.exp(context_value_row) / np.exp(context_value_row).sum()

        return policy_ranking_fun


class LinearPolicy(RecommendationPolicy):

    def recommend_policy(self, utility_list: List[Tuple[np.ndarray, np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit a linear regression from context to utility 
        return the policy with probability ~ exp(score)
        """

        if not utility_list or len(utility_list) == 0:
            return lambda x: np.ones(self.K) / self.K  # return uniform probs over arms

        # prepare data
        X_train = np.array([x for x, _ in utility_list]).reshape(-1, self.d)  # (N, d)
        Y_train = np.array([v for _, v in utility_list]).reshape(-1, self.K)  # (N, K)
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, Y_train)

        # define the linear policy
        def pi(x: np.ndarray):
            score = regressor.predict(x.reshape(1, -1))[0]
            probs = softmax(score)
            return probs
        
        return pi
