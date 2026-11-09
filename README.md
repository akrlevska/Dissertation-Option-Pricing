# Neural Networks for High-Dimensional Option Pricing

Code for my MSc Financial Mathematics dissertation at Queen Mary University of London (awarded Merit).

Classical option-pricing methods struggle as the number of assets grows: grid-based PDE solvers become infeasible and Monte Carlo simulation converges slowly. This project compares two ways of training a neural network to price options: **supervised learning** on reference prices, and **PDE-residual (physics-informed) training**, where the network learns to satisfy the Black–Scholes equation directly, without any labelled prices. It then tests the approach on real S&P 500 market data and on basket options with three correlated assets.

## Results

| Experiment | Benchmark | Result |
|---|---|---|
| Supervised learning (single asset) | Black–Scholes formula | 0.92% mean relative error |
| PDE-residual training (no labelled prices) | Black–Scholes formula | 0.93% mean relative error on options above $1; MSE reduced about 350× from the first working version |
| Hybrid (PDE-residual + 5% labelled prices) | Black–Scholes formula | A further 30% reduction in MSE |
| Real market data: 224 S&P 500 index call options | Real SPX prices | 0.24% median relative error |
| Basket options on 3 correlated assets | Monte Carlo simulation | 0.67% median relative error |

**What I learnt:**

- **Plausible results can still be wrong.** The first PDE-residual network consistently underestimated prices, and the gap grew for more expensive options. I traced the cause to the network having to output large dollar values through saturating layers, and fixed it by predicting the price ratio C/S instead.
- **Small, tested changes add up.** A log-moneyness change of variables, a smoothed no-arbitrage bound, a learning-rate schedule and repeated L-BFGS refinement together made PDE-residual training about 350 times more accurate. Each technique was tested rather than assumed.
- **Data preparation matters on real data.** For the SPX options, I removed illiquid and crossed quotes, calibrated the risk-free rate, and used a homogeneity-based input transform and a square-root target transform. These let a network trained on a $50–$150 domain price a $7,786 index.

## Experiments

Each experiment is provided both as a Python script and as a Jupyter notebook (with outputs).

| # | Experiment | Script | Notebook |
|---|------------|--------|----------|
| 1 | Supervised learning | `1_supervised_experiment.py` | `Experiment_1_supervised.ipynb` |
| 2 | PDE-residual (physics-informed) training | `2_PDE_residual_experiment.py` | `Experiment_2_PDE.ipynb` |
| 3 | Hybrid (data + PDE) training | `3_hybrid_experiment.py` | `Experiment_3_Hybrid.ipynb` |
| 4 | Basket options | `4_basket_options_experiment.py` | `Experiment_4_Basket.ipynb` |
| 5 | Supervised learning on real data | `5_supervised_experiment_rd.py` | `Experiment_5_supervised_rd.ipynb` |

The CSV file contains SPX option quotes (Dec 18 2026 monthly expiry, snapshot from Aug 2026) used in experiment 5.

## How I used AI

I used Claude to help debug parts of the code. All modelling choices were my own, including the network design, training approach, data cleaning rules and how to validate the results. I checked every result against a Black–Scholes or Monte Carlo benchmark rather than taking any output on trust.

## Setup

```
pip install -r requirements.txt
```

## Running

```
python 1_supervised_experiment.py
```

or open any notebook in Jupyter.

## Tools

Python, PyTorch, NumPy, pandas, Jupyter

## Author

Angela Krlevska

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
