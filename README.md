Investigating the Muon optimizer as a final project for STATS 606


To reproduce our results:

1. Install [uv](https://github.com/astral-sh/uv) and sync packages: `uv sync`
2. Run

```{sh}
uv run scripts/mnist_rs.py
uv run scripts/mnist_imb.py

uv run scripts/cifar10_mlp_rs.py
uv run scripts/cifar10_mlp_imb.py
```

3. Run all cells in `notebooks/plots.ipynb`