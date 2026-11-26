from abc import ABC, abstractmethod
from typing import Callable, Union
from contextlib import contextmanager
from datasets import load_from_disk, concatenate_datasets, DatasetDict, disable_progress_bar, enable_progress_bar
from scipy.special import expit
import numpy as np


class PairComparisonDataLoader(ABC):
    """
    Implements a base class for all kinds of data loader
    """

    def __init__(self, seed=1234):
        """Initialize the data loader with the number of arms and a base random seed.

        Args:
            n_arms: Number of arms K.
            seed: Base random seed; each round increments from this value.
        """
        self.seed = seed
        self.n_arms = -1

    @abstractmethod
    def generate_context(self, seed: int) -> np.ndarray:
        """Sample and return a context vector for the given seed.

        Args:
            seed: Random seed for this draw.

        Returns:
            Context as a numpy array or scalar.
        """
        raise NotImplementedError("Abstract method")

    def get_context_vector_dimension(self) -> int:
        """Return the dimensionality of the context vector.

        Returns:
            Integer dimension d of the context produced by generate_context.
        """
        dummy_context = self.generate_context(0)
        return dummy_context.shape[0]

    @abstractmethod
    def generate_theta_given_context(self, context: np.ndarray, seed: int) -> np.ndarray:
        """Return the arm utility vector normalized so that theta[0] = 0."""
        raise NotImplementedError("Abstract method")

    def generate_context_and_theta(self, seed: int):
        """
        Generate a context vector and theta vector pair
        """
        context = self.generate_context(seed)
        theta = self.generate_theta_given_context(context, seed + self.seed)
        return context, theta

    def generate_pairwise_comparison(
        self, theta: np.ndarray, arm_a: int, arm_b: int, context: np.ndarray, seed: int = 1234
    ) -> float:
        """
        Sample a response from Bradley-Terry model
        """
        np.random.seed(seed)
        theta_a, theta_b = theta[arm_a], theta[arm_b]
        prob = expit(theta_a - theta_b)  # bradley-terry model
        Y = 1 if np.random.uniform() <= prob else 0
        return Y

    def get_policy_value(
        self, policy_fun: Callable[[np.ndarray], np.ndarray], seed: int = 1234, num_repeats: int = 1000
    ) -> float:
        """
        Estimate the expected value of a policy via Monte Carlo over context-theta draws
        """
        np.random.seed(seed)
        value_repeats = []
        seed_list = np.random.randint(low=0, high=100_000, size=num_repeats)
        for b in range(num_repeats):
            context, theta = self.generate_context_and_theta(seed=seed_list[b])
            pi_hat = policy_fun(context)
            value_hat = np.sum(pi_hat * theta)
            value_repeats.append(value_hat)
        return float(np.mean(value_repeats))
