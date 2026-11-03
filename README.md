# Dissertation Code — Neural Network Methods for Option Pricing

Code accompanying my dissertation. Each experiment is provided both as a Python script and as a Jupyter notebook (with outputs).

| # | Experiment | Script | Notebook |
|---|-----------|--------|----------|
| 1 | Supervised learning | `1_supervised_experiment.py` | `Experiment_1_supervised.ipynb` |
| 2 | PDE-residual (physics-informed) training | `2_PDE_residual_experiment.py` | `Experiment_2_PDE.ipynb` |
| 3 | Hybrid (data + PDE) training | `3_hybrid_experiment.py` | `Experiment_3_Hybrid.ipynb` |
| 4 | Basket options | `4_basket_options_experiment.py` | `Experiment_4_Basket.ipynb` |
| 5 | Supervised learning on real data | `5_supervised_experiment_rd.py` | `Experiment_5_supervised_rd.ipynb` |

The CSV file contains SPX option quotes (Dec 18 2026 monthly expiry, snapshot from Aug 2026) used in experiment 5.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python 1_supervised_experiment.py
```

or open any notebook in Jupyter.

## Author

Angela Krlevska
