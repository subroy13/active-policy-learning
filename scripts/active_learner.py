from typing import List, Dict
from tqdm.auto import tqdm

from .dataloader.base import PairComparisonDataLoader
from .pairmodels import PairComparisonModelBase
from .policies import RecommendationPolicy
from .sampling import SamplingPolicy


class ActiveLearning:
    """
    Creates the active learning framework loop
    """

    def __init__(
        self,
        data: PairComparisonDataLoader,  # produces comparison data
        pair_model: PairComparisonModelBase,  # estimates m_hat(a,b)
        sampling_policy: SamplingPolicy,  # policy used for sampling new data
        reco_policy: RecommendationPolicy,  # policy used to recommend arms
    ):
        self.dataloader = data
        self.K = data.n_arms
        assert self.K > 1
        self.modeller = pair_model
        self.sampler = sampling_policy
        self.recommender = reco_policy

    def perform_rounds(
        self,
        n_rounds: int,  # number of rounds to sample
        seed: int = 1234,  # used to sample rows
        correct_for_bias: bool = True,  # whether to correct for bias or not
        verbose: bool = False,  # whther to show progress bar or not
        measure_func=None,
    ):
        if verbose:
            desc = f"Modeller: {self.modeller.__class__.__name__}, Sampler: {self.sampler.__class__.__name__}, Recommender: {self.recommender.__class__.__name__}"
            round_iterator = tqdm(range(n_rounds), desc=desc)
        else:
            round_iterator = range(n_rounds)

        contextualized_utility = []  # list of tuple (X, theta)
        data_history = []  # list of generated data rows
        rec_policy_fun = None

        for t in round_iterator:
            context, theta = self.dataloader.generate_context_and_theta(seed + t)

            # fit current dataset, and get latest estimate of theta
            pair_df = self.modeller.get_all_pairwise_mhat(context)  # predicted m_hats
            theta_hat = self.modeller.estimate_theta(context, pair_df)

            # pass to the sampler to pick a sample
            sampled_row: dict = self.sampler.pick_sample(
                pair_df, theta_hat, self.modeller, context, round=t, seed=2 * seed + t, data_history=data_history
            )
            arm_a, arm_b = sampled_row["arm_a"], sampled_row["arm_b"]  # sample arms
            Y = self.dataloader.generate_pairwise_comparison(
                theta, arm_a, arm_b, context, seed=3 * seed + t
            )  # generate Y

            # correct for bias if any
            if correct_for_bias:
                gamma_hat = self.modeller.estimate_gamma(theta_hat, sampled_row, Y)
                contextualized_utility.append((context, gamma_hat))
            else:
                contextualized_utility.append((context, theta_hat))

            # perform recommendation
            rec_policy_fun = self.recommender.recommend_policy(contextualized_utility)
            if measure_func is not None:
                rec_policy_value = measure_func(rec_policy_fun)
            else:
                rec_policy_value = self.dataloader.get_policy_value(rec_policy_fun, num_repeats=100, seed=4 * seed + t)

            current_data = {
                "context": context,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "Y": Y,
                "theta": theta_hat,
                "policy_value": rec_policy_value,
            }
            # if gamma_hat is not None:
            #     current_data["gamma"] = gamma_hat
            data_history.append(current_data)

            # fit the current dataset
            if t < n_rounds - 1:
                self.modeller.update_mhat_a_b(data_history)
                self.sampler.update_state(context, arm_a, arm_b, Y)

        return data_history, rec_policy_fun
