import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# ============================================================================================================
# EXPERIMENT 1: 


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------------------------------------------------------------------------------------------

plt.rcParams.update({"font.size": 14})

# ------------------------------------------------------------------------------------------------------------
# Black-Scholes price for a European call

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ------------------------------------------------------------------------------------------------------------

#
# Domain filtering -
# any combination producing a price at or below $0.01 is rejected
# and resampled, until N accepted samples are collected.


# Generating 50000 samples (Artificial data)
N = 50000
price_floor = 0.01

S_list, K_list, T_list, r_list, sigma_list, price_list = [], [], [], [], [], []
n_rejected = 0
n_accepted = 0

while n_accepted < N:
    n_batch = N - n_accepted
    S_b = np.random.uniform(50, 150, n_batch)
    K_b = np.random.uniform(50, 150, n_batch)
    T_b = np.random.uniform(0.05, 2.0, n_batch)
    r_b = np.random.uniform(0.01, 0.06, n_batch)
    sigma_b = np.random.uniform(0.10, 0.60, n_batch)
    price_b = black_scholes_call(S_b, K_b, T_b, r_b, sigma_b)

    keep = price_b > price_floor
    n_rejected += (~keep).sum()
    n_accepted += keep.sum()

    S_list.append(S_b[keep]); K_list.append(K_b[keep]); T_list.append(T_b[keep])
    r_list.append(r_b[keep]); sigma_list.append(sigma_b[keep]); price_list.append(price_b[keep])

S = np.concatenate(S_list)[:N]
K = np.concatenate(K_list)[:N]
T = np.concatenate(T_list)[:N]
r = np.concatenate(r_list)[:N]
sigma = np.concatenate(sigma_list)[:N]
prices = np.concatenate(price_list)[:N]

print(f"Accepted {N} samples; rejected {n_rejected} combinations "
      f"({n_rejected / (N + n_rejected):.1%} rejection rate).")

data = pd.DataFrame({"S": S, "K": K, "T": T, "r": r,
                      "sigma": sigma, "price": prices})

print("Minimum BS price:", data["price"].min())
print("Number of prices below 1e-3:", (data["price"] < 1e-3).sum())

# ------------------------------------------------------------------------------------------------------------
# Training/testing data

#Standardization
X = data[["S", "K", "T", "r", "sigma"]].values
y = data[["price"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#This script trains on float32.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# ------------------------------------------------------------------------------------------------------------
# Neural Network Architecture
# The neural network has 3 hidden layers, and each hidden layer contains 128 neurons

class OptionPricingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


model = OptionPricingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------------------------------------------------------
# Training loop
# The neural network learns here and repeats the process 1000 times, makes 
# predictions, and checks how wrong they are. Also, it adjusts the network to reduce the 
# error then checks performance on the test data.

epochs = 1000
training_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    training_losses.append(epoch_loss / len(train_ds))

    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
    test_losses.append(test_loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}  "
              f"train={training_losses[-1]:.6f}  test={test_losses[-1]:.6f}")

# ------------------------------------------------------------------------------------------------------------
# Training loss plot

plt.figure(figsize=(8, 5))
plt.plot(training_losses, label="Training loss")
plt.plot(test_losses, label="Test loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.grid(True, which="both")
plt.savefig("training_loss_supervised_nn.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Predictions, with non-negativity enforced because a European call price
# can never be negative!

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred_raw = scaler_y.inverse_transform(y_pred_scaled)
y_pred = np.clip(y_pred_raw, 0, None)
y_true = y_test

n_clipped = np.sum(y_pred_raw < 0)
print("Number of predictions clipped to zero:", n_clipped)

# ------------------------------------------------------------------------------------------------------------
# Predicted vs. benchmark price plot

plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
         color="black")
plt.xlabel("Black--Scholes Price")
plt.ylabel("Neural Network Predicted Price")
plt.grid(True)
plt.savefig("supervised_nn_vs_black_scholes.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Error metrics

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
absolute_error = np.abs(y_pred - y_true)

print("MSE:", mse)
print("MAE:", mae)

# Sensitivity of the filtered relative error to the price threshold,
# reported explicitly rather than fixing a single convenient cutoff.

for price_threshold in [1.0, 0.1, 0.01]:
    mask = y_true.flatten() > price_threshold
    rel_err = absolute_error.flatten()[mask] / y_true.flatten()[mask]
    print(f"threshold={price_threshold}: n={mask.sum()}, "
          f"mean rel. error={rel_err.mean():.4f}")

price_threshold = 0.1  # chosen threshold; justified relative to S, K ~ O(100)

results = pd.DataFrame(X_test, columns=["S", "K", "T", "r", "sigma"])
results["BS_price"] = y_true.flatten()
results["NN_price"] = y_pred.flatten()
results["absolute_error"] = np.abs(results["NN_price"] - results["BS_price"])

results["relative_error"] = np.nan
mask = results["BS_price"] > price_threshold
results.loc[mask, "relative_error"] = (
    results.loc[mask, "absolute_error"] / results.loc[mask, "BS_price"]
)
results["moneyness"] = results["S"] / results["K"]

# ------------------------------------------------------------------------------------------------------------
# Relative error vs. moneyness

plt.figure(figsize=(8, 5))
plt.scatter(results.loc[mask, "moneyness"], results.loc[mask, "relative_error"],
            alpha=0.3)
plt.xlabel("Moneyness S/K")
plt.ylabel("Filtered Relative Pricing Error")
plt.grid(True)
plt.savefig("relative_error_vs_moneyness.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Implied-volatility inversion

def implied_volatility_call(price, S, K, T, r):
    intrinsic_value = max(S - K * np.exp(-r * T), 0)
    if price < intrinsic_value:
        return np.nan
    try:
        return brentq(
            lambda sigma: black_scholes_call(S, K, T, r, sigma) - price,
            1e-6, 5.0
        )
    except ValueError:
        return np.nan


implied_vol_errors = []
predicted_implied_vols = []

for i in range(len(results)):
    price_pred = results.loc[i, "NN_price"]
    S_i, K_i, T_i, r_i, sigma_true = results.loc[
        i, ["S", "K", "T", "r", "sigma"]
    ]
    sigma_imp_pred = implied_volatility_call(price_pred, S_i, K_i, T_i, r_i)
    predicted_implied_vols.append(sigma_imp_pred)
    implied_vol_errors.append(
        np.nan if np.isnan(sigma_imp_pred) else abs(sigma_imp_pred - sigma_true)
    )

results["NN_implied_vol"] = predicted_implied_vols
results["implied_vol_error"] = implied_vol_errors

valid_iv_errors = results["implied_vol_error"].dropna()
print("Number of valid implied volatility errors:", len(valid_iv_errors))
print("Mean implied volatility error:", valid_iv_errors.mean())
print("Median implied volatility error:", valid_iv_errors.median())
print("Maximum implied volatility error:", valid_iv_errors.max())

# ------------------------------------------------------------------------------------------------------------
# Implied-volatility error vs. moneyness

plt.figure(figsize=(8, 5))
plt.scatter(results["moneyness"], results["implied_vol_error"], alpha=0.3)
plt.xlabel("Moneyness S/K")
plt.ylabel("Implied Volatility Error")
plt.grid(True)
plt.savefig("implied_vol_error_vs_moneyness.pdf", bbox_inches="tight")
plt.show()

results.to_csv("black_scholes_supervised_results.csv", index=False)