import os
import json
import traceback
import warnings
from typing import Tuple, Any, Union
from joblib import Parallel, delayed, parallel_backend
from uuid import uuid4
from tqdm.auto import tqdm
import datetime as dt
import pandas as pd

# -------------
# Safe simulation utilities
# -------------


class SafeExperimentModule:
    """Manages parallelized simulation experiments with persistent result storage."""

    def __init__(self, sim_directory: str):
        """Initialize with the root directory for storing experiment outputs.

        Args:
            sim_directory: Path to the directory where results and metadata are saved.
        """
        self.root = sim_directory  # the directory where to store the simulation / experiment
        self.metafile = os.path.join(self.root, "meta.json")
        os.makedirs(self.root, exist_ok=True)

    def write_meta_json(self, input_vars: dict):
        """Append a new experiment entry to the metadata file and return its unique ID.

        Args:
            input_vars: Dictionary of input parameters for the experiment run.

        Returns:
            UUID string identifying this experiment run.
        """
        try:
            with open(self.metafile, "r") as f:
                current_meta = json.load(f)  # this is a list of inputs
                metadata_to_write = {
                    "inputs": input_vars,
                    "_id": str(uuid4()),
                    "timestamp": str(dt.datetime.now()),
                }
                current_meta.append(metadata_to_write)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata_to_write = {
                "inputs": input_vars,
                "_id": str(uuid4()),
                "timestamp": str(dt.datetime.now()),
            }
            current_meta = [metadata_to_write]

        with open(self.metafile, "w") as f:
            json.dump(current_meta, f)

        return metadata_to_write["_id"]  # return the id

    def read_meta_json(self):
        """Read and return all experiment metadata entries.

        Returns:
            List of metadata dicts, each with keys 'inputs', '_id', and 'timestamp'.
        """
        with open(self.metafile, "r") as f:
            return json.load(f)

    def read_experiment_results(self, experiment_id: str):
        """Load the results CSV for a given experiment ID.

        Args:
            experiment_id: UUID string of the experiment run.

        Returns:
            pandas DataFrame with the stored simulation results.
        """
        fname = os.path.join(self.root, experiment_id + ".csv")
        return pd.read_csv(fname)

    def _safe_simulation(self, sim_func, elem) -> Union[Tuple[str, Any], Tuple[str, Any, str, str]]:
        """Wrap sim_func(elem) in a try/except, returning status and result or error info.

        Args:
            sim_func: Callable to execute.
            elem: Argument passed to sim_func.

        Returns:
            Tuple ('ok', result) on success or ('error', elem, repr, traceback) on failure.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ("ok", sim_func(elem))
        except Exception as e:
            return ("error", elem, repr(e), traceback.format_exc())

    def parallelize_simulation_loops(
        self,
        iterator,  # each elem of iterator passed as argument to your simulation function
        sim_func,
        n_jobs=5,
        progress_desc="",
        stop_on_error=False,
    ):
        """Run sim_func over iterator in parallel using joblib, collecting results.

        Args:
            iterator: Iterable of inputs, each passed individually to sim_func.
            sim_func: Callable that takes a single element and returns a result.
            n_jobs: Number of parallel worker processes.
            progress_desc: Label shown on the tqdm progress bar.
            stop_on_error: If True, halt immediately on the first error.

        Returns:
            List of successful results in completion order.

        Raises:
            Exception: Re-raises the first encountered error if any occurred.
        """
        results = []
        errors = []

        with parallel_backend("loky"):
            with Parallel(
                n_jobs=n_jobs,
                prefer="processes",
                return_as="generator",
                max_nbytes=None,  # avoids mmap edge cases
            ) as parallel:
                tasks = (delayed(self._safe_simulation)(sim_func, elem) for elem in iterator)

                try:
                    for out in tqdm(parallel(tasks), total=len(iterator), desc=progress_desc, leave=True):
                        if out[0] == "ok":  # type: ignore
                            results.append(out[1])  # type: ignore
                        else:
                            errors.append(out)
                            if stop_on_error:
                                break
                finally:
                    pass

        if len(errors) > 0:
            _, elem, exc_repr, tb_str = errors[0]
            raise RuntimeError(f"Simulation job failed for input {elem!r}:\n{exc_repr}\n{tb_str}")
        return results

    def store_result(self, result_df: pd.DataFrame, input_vars):
        """Persist a result DataFrame to CSV and register the run in metadata.

        Args:
            result_df: DataFrame containing the experiment results.
            input_vars: Dictionary of input parameters for the experiment run.

        Returns:
            File path of the saved CSV.
        """
        meta_id = self.write_meta_json(input_vars)
        fname = os.path.join(self.root, meta_id + ".csv")
        result_df.to_csv(fname, index=False)
        return fname  # store the csv file
