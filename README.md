# Active Policy Learning from Pairwise Preferences

This repository implements active pair selection and policy learning from contextual preference feedback. The experiments evaluate recommendation regret in simulations and alignment with recorded preferences in three LLM routing benchmarks. The seven simulation figures are in [figures/fig2](figures/fig2), and the six benchmark figures are in [figures](figures).

## Setup

From the repository root, create a Python environment and install the dependencies. The versions below were used to check this release with Python 3.13.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.5 pandas==2.3.2 scipy==1.15.3 scikit-learn==1.7.1 \
  datasets==4.4.2 matplotlib==3.10.3 seaborn==0.13.2 joblib==1.5.1 tqdm==4.67.1 \
  jupyterlab==4.4.6
```

## Simulations

Run `python -m scripts.simulation` from the repository root for the default discrete misspecification setting. To run the seven configurations shown in the supplied figures, use the following from the repository root. The example uses four workers. Adjust `n_cores` to suit your machine.

```sh
python - <<'PY'
from scripts.simulation import perform_simulation_study

settings = [
    ("discrete", "kronecker"),
    ("li_li_correct_discrete", "kronecker"),
    ("li_li_miss1_discrete", "kronecker"),
    ("li_li_miss2_discrete", "kronecker"),
    ("li_li_correct_continuous", "linear"),
    ("li_li_miss1_continuous", "linear"),
    ("li_li_miss2_continuous", "linear"),
]

for dataset_name, policy_class in settings:
    path, _ = perform_simulation_study(
        n_rounds=1000,
        dataset_name=dataset_name,
        policy_class=policy_class,
        num_repeats=100,
        n_cores=4,
        seed=1234,
        correct_for_bias=True,
    )
    print(path)
PY
```

Results are generated in `data/sim2/`. Start JupyterLab with `python -m jupyterlab` and open [notebooks/plots.ipynb](notebooks/plots.ipynb). Select the dataset and policy in the settings cell, then run the notebook. It reads the latest matching simulation by default and writes the PDF to `data/figures/fig2/`. The plotting code adds 0.05 to regret for display on a log scale and applies a 25-round moving average within each repeat and method. The CSV records the original per-round regret.

## LLM routing benchmarks

Open [notebooks/llm_benchmarking.ipynb](notebooks/llm_benchmarking.ipynb) in JupyterLab and run its cells in order. The notebook trains a prompt category classifier from MT-Bench and uses it to create contexts for the two Arena datasets. It compares adaptive sampling with the round-robin baseline labeled Uniform in the figures, using Bradley–Terry–Luce preference estimation. Outputs include train and test alignment accuracy by round and estimated model preferences by prompt category.

The notebook downloads these datasets into `data/raw/` on first use. Each dataset is subject to its upstream terms.

- [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- [Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- [LMArena-140K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)

To access Chatbot Arena Conversations, accept the terms on its Hugging Face dataset page and authenticate locally with `hf auth login`. The benchmark notebook writes regenerated figures to `data/figures/`.

## License

The code and included figures are licensed under the [Apache License, Version 2.0](LICENSE).
