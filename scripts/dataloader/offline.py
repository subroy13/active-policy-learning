from typing import Any, Callable, Union
from contextlib import contextmanager
from datasets import load_from_disk, concatenate_datasets, DatasetDict, disable_progress_bar, enable_progress_bar
import numpy as np

from .base import PairComparisonDataLoader


@contextmanager
def no_progress_bar():
    """Context manager that suppresses HuggingFace dataset progress bars."""
    disable_progress_bar()
    try:
        yield
    finally:
        enable_progress_bar()


class OffinePairedDataLoader(PairComparisonDataLoader):
    """
    This converts an offline pair data loader
    into an online pair data loader
    """

    def __init__(self, n_arms, seed=1234):
        """Initialize the offline paired data loader.

        :param n_arms: Number of arms.
        :param seed: Random seed.
        """
        super().__init__(seed)
        self.n_arms = n_arms

    def load_arena_data(
        self,
        data: Union[str, DatasetDict],
        context_key: str = "context",
        model_a_key: str = "model_a",
        model_b_key: str = "model_b",
        response_key: str = "Y",  # 0 or 1
    ):
        """Load offline pairwise comparison data from a dataset or disk path.

        :param data: Path to the dataset on disk or a loaded DatasetDict.
        :param context_key: Column name for the context.
        :param model_a_key: Column name for the first model.
        :param model_b_key: Column name for the second model.
        :param response_key: Column name for the response (1 if model_a wins, 0 otherwise).
        """
        if isinstance(data, str):
            self.offline_data = load_from_disk(data)
        else:
            self.offline_data = data

        # check if it has train / test / validation setup
        if isinstance(self.offline_data, DatasetDict):
            train_ds = self.offline_data["train"]
            if "validation" in self.offline_data:
                val_ds = self.offline_data["validation"]
                self.train_ds = concatenate_datasets([train_ds, val_ds])
            else:
                self.train_ds = train_ds
            self.test_ds = self.offline_data["test"]
        else:
            self.train_ds = self.offline_data
            self.test_ds = self.offline_data

        self.context_key = context_key
        self.model_a_key = model_a_key
        self.model_b_key = model_b_key
        self.response_key = response_key
        self.context_variable_list = self.get_context_variables_list()
        self.context_list = self.get_context_support()
        self.model_index_mapping = self.get_model_mapping()
        self.MODEL_LIST = list(self.model_index_mapping.keys())

    def get_model_mapping(self):
        """Extract and map unique model names to integer arm indices.

        :return: A dictionary mapping model names to zero-indexed integers.
        """
        model_list = []
        for row in self.train_ds:
            model_list.append(row[self.model_a_key])  # type: ignore
            model_list.append(row[self.model_b_key])  # type: ignore
        model_list = list(set(model_list))
        model_list.sort()
        return {model: i for i, model in enumerate(model_list)}

    def get_context_variables_list(self):
        """Determine the keys or indices representing the context variables.

        :return: A list of context variable keys or indices.
        """
        context_col = self.train_ds[0][self.context_key]
        if isinstance(context_col, list):
            return list(range(len(context_col)))  # the index mapping for the columns used for context
        elif isinstance(context_col, dict):
            return list(context_col.keys())
        else:
            return [0]

    def get_serializable_context(self, context: Any) -> Any:
        """Convert a context representation into a hashable tuple.

        :param context: The context data.
        :return: A tuple representation of the context.
        """
        if isinstance(context, list):
            return tuple(context)
        elif isinstance(context, dict):
            return tuple([context[v] for v in self.context_variable_list])
        else:
            return (context,)

    def get_context_support(self):
        """Extract all unique contexts present in the training and testing datasets.

        :return: A list of unique hashable context tuples.
        """
        combinations = []
        for row in self.train_ds:
            xrow = row[self.context_key]  # type: ignore
            combinations.append(self.get_serializable_context(xrow))

        for row in self.test_ds:
            xrow = row[self.context_key]  # type: ignore
            combinations.append(self.get_serializable_context(xrow))

        return list(set(combinations))

    def generate_context(self, seed: int) -> Any:
        """Sample a random context from the empirical support of contexts.

        :param seed: Random seed.
        :return: A randomly selected context array.
        """
        np.random.seed(seed)
        context_idx = np.random.randint(0, len(self.context_list))
        return np.array(self.context_list[context_idx])

    def generate_theta_given_context(self, context: Any, seed: int) -> np.ndarray:
        """Return dummy theta values, as true parameters are typically unknown in offline data.

        :param context: The context vector.
        :param seed: Random seed.
        :return: An array of zeros.
        """
        return np.zeros(self.n_arms)  # some dummy theta values, not used

    def generate_pairwise_comparison(self, theta: np.ndarray, arm_a: int, arm_b: int, context: np.ndarray, seed: int = 1234) -> float:
        """Retrieve the pairwise comparison result from the offline dataset for the given context and arms.

        :param theta: The dummy theta array.
        :param context: The context vector.
        :param arm_a: Index of the first arm.
        :param arm_b: Index of the second arm.
        :return: 1 if arm_a wins, 0 if arm_b wins, or a random coin flip if no match is found.
        """
        if isinstance(context, np.ndarray):
            context = tuple(context.tolist())
        else:
            context = tuple(context)

        possible_val = []
        for row in self.train_ds:
            if self.get_serializable_context(row[self.context_key]) == context:
                if row[self.model_a_key] == self.MODEL_LIST[arm_a] and row[self.model_b_key] == self.MODEL_LIST[arm_b]:  # type: ignore
                    possible_val.append(row[self.response_key])  # type: ignore
                elif row[self.model_b_key] == self.MODEL_LIST[arm_a] and row[self.model_a_key] == self.MODEL_LIST[arm_b]:  # type: ignore
                    possible_val.append(1 - row[self.response_key])  # type: ignore

        np.random.seed(seed)
        if len(possible_val) == 0:
            # no match found
            return float(np.random.binomial(1, 0.5))
        else:
            # draw one at random 
            return float(np.random.binomial(1, np.array(possible_val).mean()))

    def compare_alignment(self, rows, recommender_policy: Callable[[Any], np.ndarray]):
        """Calculate the alignment accuracy of a recommendation policy against the empirical pairwise comparisons.

        :param rows: Dataset rows to evaluate against.
        :param recommender_policy: The policy to evaluate.
        :return: The proportion of comparisons where the policy correctly predicts the winner.
        """
        total = 0.0
        correct_pred = 0.0
        for row in rows:
            xrow = row[self.context_key]
            context = np.array(self.get_serializable_context(xrow))
            pi_hat = recommender_policy(context)

            # compare probabilites of arm_a and arm_b
            arm_a = self.model_index_mapping[row[self.model_a_key]]
            arm_b = self.model_index_mapping[row[self.model_b_key]]
            pi_a = pi_hat[arm_a]
            pi_b = pi_hat[arm_b]

            total += 1.0
            if pi_a > pi_b and row[self.response_key] == 1:
                correct_pred += 1.0
            elif pi_a < pi_b and row[self.response_key] == 0:
                correct_pred += 1.0
        return correct_pred / total

    def calculate_performance_metrics(self, recommender_policy: Callable[[Any], np.ndarray]):
        """Evaluate the policy alignment on both train and test datasets.

        :param recommender_policy: The policy to evaluate.
        :return: A tuple of (training accuracy, testing accuracy).
        """
        # look at the testing samples, and count the number of times the prediction ranking matches
        train_acc = self.compare_alignment(self.train_ds, recommender_policy)
        test_acc = self.compare_alignment(self.test_ds, recommender_policy)

        return train_acc, test_acc

    def set_true_theta(self, theta_true_fun: Callable[[Any], np.ndarray]):
        """Set a ground truth theta function for evaluating policy values.

        :param theta_true_fun: A callable returning the true theta parameters given a context.
        """
        self.theta_true_fun = theta_true_fun

    def get_policy_value(self, policy_fun: Callable[[Any], np.ndarray], seed=1234, num_repeats=1000) -> float:
        """Compute the expected policy value over the test dataset using the ground truth theta function.

        :param policy_fun: The policy to evaluate.
        :param seed: Unused (maintained for signature compatibility).
        :param num_repeats: Unused (maintained for signature compatibility).
        :return: The expected value of the policy.
        """
        if not hasattr(self, "theta_true_fun"):
            raise ValueError("true theta function must be set")
        value = 0.0
        for row in self.test_ds:
            xrow = row[self.context_key]  # type: ignore
            context = np.array(self.get_serializable_context(xrow))
            pi_hat = policy_fun(context)
            true_theta = self.theta_true_fun(context)
            value += np.sum(pi_hat * true_theta)
        return value / len(self.test_ds)