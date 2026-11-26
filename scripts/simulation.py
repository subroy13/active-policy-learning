import os
from typing import List, Tuple, Type
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Configure BEFORE importing seaborn/pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from .utils.experiment_tools import SafeExperimentModule
from .utils.helper import add_columns_to_list

from .dataloader import (
    PairComparisonDataLoader,
    DiscreteUniformSetup,
    LiAndLiCorrectDiscreteSetup,
    LiAndLiMiss1DiscreteSetup,
    LiAndLiMiss2DiscreteSetup,
    LiAndLiCorrectContinuousSetup,
    LiAndLiMiss1ContinuousSetup,
    LiAndLiMiss2ContinuousSetup,
)
from .pairmodels import PairComparisonModelBase, PairComparisonLogisticModel, PairComparisonBTModel
from .policies import RecommendationPolicy, KroneckerPolicy, LinearPolicy
from .sampling import (
    SamplingPolicy,
    UniformSamplingPolicy,
    AdaptiveSamplingPolicy,
    MaxEntSamplingPolicy,
    MaxIntPSamplingPolicy,
    DoubleThompsonSampling,
)
from .active_learner import ActiveLearning

DATA_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sim2")
experimenter = SafeExperimentModule(DATA_SAVE_PATH)

# Each entry: (sampling_label, model_label, SamplerClass, ModelClass)
METHODS: List[Tuple[str, str, Type[SamplingPolicy], Type[PairComparisonModelBase]]] = [
    ("uniform", "logistic", UniformSamplingPolicy, PairComparisonLogisticModel),
    ("adaptive", "logistic", AdaptiveSamplingPolicy, PairComparisonLogisticModel),
    ("uniform", "BT", UniformSamplingPolicy, PairComparisonBTModel),
    ("adaptive", "BT", AdaptiveSamplingPolicy, PairComparisonBTModel),
    ("maxent", "BT", MaxEntSamplingPolicy, PairComparisonBTModel),
    ("maxintp", "BT", MaxIntPSamplingPolicy, PairComparisonBTModel),
    ("double_ts", "BT", DoubleThompsonSampling, PairComparisonBTModel),
]


def _make_reco_policy(policy_class: str, n_arms: int, context_dim: int) -> RecommendationPolicy:
    if policy_class == "kronecker":
        return KroneckerPolicy(n_arms, context_dim)
    elif policy_class == "linear":
        return LinearPolicy(n_arms, context_dim)
    else:
        raise NotImplementedError(f"Unknown policy class: {policy_class}")


def _make_dataset(dataset_name: str, seed: int) -> PairComparisonDataLoader:
    SIMULATION_DATA_LOADER_LIST = {
        "discrete": DiscreteUniformSetup,
        "li_li_correct_discrete": LiAndLiCorrectDiscreteSetup,
        "li_li_miss1_discrete": LiAndLiMiss1DiscreteSetup,
        "li_li_miss2_discrete": LiAndLiMiss2DiscreteSetup,
        "li_li_correct_continuous": LiAndLiCorrectContinuousSetup,
        "li_li_miss1_continuous": LiAndLiMiss1ContinuousSetup,
        "li_li_miss2_continuous": LiAndLiMiss2ContinuousSetup,
    }
    cls = SIMULATION_DATA_LOADER_LIST.get(dataset_name)
    if cls is None:
        raise NotImplementedError(f"Invalid dataset name: {dataset_name}")
    else:
        return cls(seed)


def perform_simulation_study(
    n_rounds: int,
    dataset_name: str,
    policy_class: str,
    num_repeats: int = 50,
    n_cores: int = 10,
    seed: int = 1234,
    correct_for_bias: bool = True,
):
    """Run a parallelized simulation study and save per-round regret across repeats.

    For each repeat seed, every method in METHODS is run on an identical data sequence
    (same seed) so results are directly comparable. Regret is recorded per round per method.

    Args:
        n_rounds: Number of active learning rounds per method per repeat.
        dataset_name: Key into SIMULATION_DATA_LOADER_LIST selecting the data-generating process.
        policy_class: Recommendation policy ('kronecker' or 'linear').
        num_repeats: Number of independent simulation repeats.
        n_cores: Number of parallel worker processes.
        seed: Master seed used to generate per-repeat seeds.
        correct_for_bias: Use the bias-corrected utility estimate.

    Returns:
        The path to the saved CSV and the results DataFrame.
    """
    np.random.seed(seed)
    repeat_seeds = np.random.randint(low=1, high=1_000_000, size=num_repeats)

    def simulation_job(s: int) -> List[dict]:
        """Run all methods for a single repeat and return flat list of round-level results."""
        ref_loader = _make_dataset(dataset_name, s)
        n_arms = ref_loader.n_arms
        d = ref_loader.get_context_vector_dimension()

        # oracle best policy
        ref_policy = _make_reco_policy(policy_class, n_arms, d)
        gen_func = lambda x: ref_loader.generate_context_and_theta(x)
        best_policy = ref_policy.get_best_policy(gen_func, num_repeats=500, seed=s + 1)
        best_value = ref_loader.get_policy_value(best_policy, num_repeats=500, seed=s + 1)

        rows = []
        for sampling_name, model_name, SamplerClass, ModelClass in METHODS:
            # load fresh dataloader every time
            dataset = _make_dataset(dataset_name, s)

            learner = ActiveLearning(
                data=dataset,
                pair_model=ModelClass(n_arms, d),
                sampling_policy=SamplerClass(n_arms, d),
                reco_policy=_make_reco_policy(policy_class, n_arms, d),
            )
            data_history, _ = learner.perform_rounds(n_rounds, seed=s, correct_for_bias=correct_for_bias, verbose=True)

            method_rows = [
                {"round": t, "regret": best_value - row["policy_value"], "sampling": sampling_name, "model": model_name}
                for t, row in enumerate(data_history)
            ]
            rows.extend(method_rows)

        return rows

    # get the list of results
    results_list = experimenter.parallelize_simulation_loops(
        repeat_seeds, sim_func=simulation_job, n_jobs=n_cores, stop_on_error=True
    )

    # combine all repeats into a long format data
    all_rows = []
    for repeat_idx, rows in enumerate(results_list):
        rows = add_columns_to_list(rows, [("repeat", repeat_idx)])
        all_rows.extend(rows)

    result_df = pd.DataFrame(all_rows)

    input_vars = {
        "n_rounds": n_rounds,
        "dataset_name": dataset_name,
        "policy_class": policy_class,
        "initial_seed": seed,
        "n_repeats": num_repeats,
        "n_cores": n_cores,
        "correct_for_bias": correct_for_bias,
    }
    fname = experimenter.store_result(result_df, input_vars)
    return fname, result_df


if __name__ == "__main__":
    DATASET_NAME = "li_li_miss1_discrete"
    POLICY_CLASS = "kronecker"

    fname, _ = perform_simulation_study(
        n_rounds=1000,
        dataset_name=DATASET_NAME,
        policy_class=POLICY_CLASS,
        num_repeats=100,
        n_cores=20,
        seed=1234,
        correct_for_bias=True,
    )
    print(fname)
