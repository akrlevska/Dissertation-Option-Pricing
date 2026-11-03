import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================================================================
# EXPERIMENT 3:

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------------------------------------------------

plt.rcParams.update({"font.size": 14})

# ------------------------------------------------------------------------------------------------------------
# Black-Scholes price for a European call

def black_scholes_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
# ------------------------------------------------------------------------------------------------------------
def bs_v_torch(x, tau, r, sigma):
    """Exact transformed Black-Scholes: v = V/K. Used only for the
    smoothed terminal condition target (an exact conditional
    expectation under GBM), following Bayer, Siebenmorgen and
    Tempone (2018)."""
    tau = torch.clamp(tau, min=1e-10)
    d1 = (x + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.exp(x) * normal.cdf(d1) - torch.exp(-r * tau) * normal.cdf(d2)


S_LO, S_HI = 50.0, 150.0
K_LO, K_HI = 50.0, 150.0
TAU_LO, TAU_HI = 0.05, 2.0
R_LO, R_HI = 0.01, 0.06
SIG_LO, SIG_HI = 0.10, 0.60
PRICE_FLOOR = 0.01
EPSILON_SMOOTH = 0.02
X_LO, X_HI = -1.3, 1.3

# Smoothed no-arbitrage bound (Glau and Wunderlich 2022);
# beta = lambda*K = 0.1*100 = 10 in strike-normalised units.
BETA_BOUND = 10.0

def smoothed_bound(x, tau, r):
    z = torch.exp(x) - torch.exp(-r * tau)
    return F.softplus(z, beta=BETA_BOUND)

def normalize(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0

# ------------------------------------------------------------------------------------------------------------
# Same benchmark data as Experiments 1 and 2


# Generating 50000 samples (Artificial data)
N = 50000
S_list, K_list, T_list, r_list, sigma_list, price_list = [], [], [], [], [], []
n_accepted = 0
while n_accepted < N:
    n_batch = N - n_accepted
    S_b = np.random.uniform(S_LO, S_HI, n_batch)
    K_b = np.random.uniform(K_LO, K_HI, n_batch)
    T_b = np.random.uniform(TAU_LO, TAU_HI, n_batch)
    r_b = np.random.uniform(R_LO, R_HI, n_batch)
    sigma_b = np.random.uniform(SIG_LO, SIG_HI, n_batch)
    price_b = black_scholes_call(S_b, K_b, T_b, r_b, sigma_b)
    keep = price_b > PRICE_FLOOR
    n_accepted += keep.sum()
    S_list.append(S_b[keep]); K_list.append(K_b[keep]); T_list.append(T_b[keep])
    r_list.append(r_b[keep]); sigma_list.append(sigma_b[keep]); price_list.append(price_b[keep])

S_all = np.concatenate(S_list)[:N]
K_all = np.concatenate(K_list)[:N]
T_all = np.concatenate(T_list)[:N]
r_all = np.concatenate(r_list)[:N]
sigma_all = np.concatenate(sigma_list)[:N]
price_all = np.concatenate(price_list)[:N]

X_all = np.column_stack([S_all, K_all, T_all, r_all, sigma_all])
X_train_pool, X_test, price_train_pool, price_test = train_test_split(
    X_all, price_all, test_size=0.2, random_state=seed
)
print(f"Benchmark test set: {len(X_test)} options (identical sampling to Experiments 1 and 2).")

idx_test = np.arange(len(X_test))
X_val, X_test_final, price_val, price_test_final, idx_val, idx_test_final = train_test_split(
    X_test, price_test, idx_test, test_size=0.5, random_state=seed
)

def to_xtau(X):
    S, K, T, r, sigma = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    x = np.log(S / K)
    return np.column_stack([x, T, r, sigma])

val_x = to_xtau(X_val)
val_tensor = {
    "x": torch.tensor(val_x[:, 0:1]),
    "tau": torch.tensor(val_x[:, 1:2]),
    "r": torch.tensor(val_x[:, 2:3]),
    "sigma": torch.tensor(val_x[:, 3:4]),
    "K": torch.tensor(X_val[:, 1:2]),
    "price": torch.tensor(price_val.reshape(-1, 1)),
}

# ------------------------------------------------------------------------------------------------------------

N_LABELLED = 2000
rng = np.random.RandomState(seed)

# Select the labelled examples
labelled_idx = rng.choice(len(X_train_pool), N_LABELLED, replace=False)
X_labelled = X_train_pool[labelled_idx]
price_labelled = price_train_pool[labelled_idx]
labelled_x = to_xtau(X_labelled)
data_tensor = {
    "x": torch.tensor(labelled_x[:, 0:1]),
    "tau": torch.tensor(labelled_x[:, 1:2]),
    "r": torch.tensor(labelled_x[:, 2:3]),
    "sigma": torch.tensor(labelled_x[:, 3:4]),
    "v": torch.tensor((price_labelled / X_labelled[:, 1]).reshape(-1, 1)),
}
print(f"Labelled dataset for the hybrid data loss: {N_LABELLED} points "
      f"({100.0 * N_LABELLED / len(X_train_pool):.1f}% of the supervised training set).")

# ------------------------------------------------------------------------------------------------------------
# Network: identical architecture and composite trial function
# v = w_theta + u_hat as in Experiment 2.
# ------------------------------------------------------------------------------------------------------------
class LogTransformedPDENN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, tau, x, r, sigma):
        inp = torch.cat([
            normalize(tau, TAU_LO, TAU_HI),
            normalize(x, X_LO, X_HI),
            normalize(r, R_LO, R_HI),
            normalize(sigma, SIG_LO, SIG_HI),
        ], dim=1)
        return self.network(inp)   # predicts the time value w


model = LogTransformedPDENN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reduce the learning rate during training
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

def v_model(tau, x, r, sigma):
    """Composite trial function v = w_theta + u_hat
    (Glau and Wunderlich, 2022, Section 2.5.2)."""

    # Combine the network output with the smoothed lower bound
    return model(tau, x, r, sigma) + smoothed_bound(x, tau, r)

def sample_interior(n):
    tau = torch.empty(n, 1).uniform_(EPSILON_SMOOTH, TAU_HI)
    x = torch.empty(n, 1).uniform_(X_LO, X_HI)
    r = torch.empty(n, 1).uniform_(R_LO, R_HI)
    sigma = torch.empty(n, 1).uniform_(SIG_LO, SIG_HI)
    return tau, x, r, sigma

def sample_terminal_smoothed(n):
    tau = torch.full((n, 1), EPSILON_SMOOTH)
    x = torch.empty(n, 1).uniform_(X_LO, X_HI)
    r = torch.empty(n, 1).uniform_(R_LO, R_HI)
    sigma = torch.empty(n, 1).uniform_(SIG_LO, SIG_HI)
    smoothed_target = bs_v_torch(x, tau, r, sigma)
    return tau, x, r, sigma, smoothed_target

def pde_residual(tau, x, r, sigma):
    """Constant-coefficient transformed PDE residual on the
    composite trial function v = w + u_hat."""
    tau = tau.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    v = v_model(tau, x, r, sigma)
    dv_dtau = torch.autograd.grad(v, tau, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
    residual = dv_dtau - 0.5 * sigma**2 * d2v_dx2 - (r - 0.5 * sigma**2) * dv_dx + r * v
    return residual

def data_loss_fn():
    """Data loss in strike-normalised units v = V/K, commensurate
    with the PDE and terminal residuals (see header)."""
    v_data = v_model(data_tensor["tau"], data_tensor["x"], data_tensor["r"], data_tensor["sigma"])
    return torch.mean((v_data - data_tensor["v"]) ** 2)

n_interior = 4000
n_terminal = 1000
lambda_terminal = 1.0
lambda_data = 10.0   

max_epochs = 3000
patience = 400   # generous: the LR schedule needs room to act before early stopping
best_val_loss = float("inf")
epochs_without_improvement = 0

# Calculate the PDE error
pde_losses, terminal_losses, data_losses, val_losses = [], [], [], []

print(f"\nTraining HYBRID network (PDE residual + smoothed terminal + "
      f"{N_LABELLED} labelled points, lambda_data={lambda_data})...")
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()

    tau_i, x_i, r_i, sigma_i = sample_interior(n_interior)
    residual = pde_residual(tau_i, x_i, r_i, sigma_i)
    pde_loss = torch.mean(residual ** 2)

    tau_t, x_t, r_t, sigma_t, smoothed_target_t = sample_terminal_smoothed(n_terminal)
    v_terminal = v_model(tau_t, x_t, r_t, sigma_t)

    # Calculate the terminal-condition error
    terminal_loss = torch.mean((v_terminal - smoothed_target_t) ** 2)

    # Calculate the labelled-data error
    data_loss = data_loss_fn()

    # Combine all three errors
    loss = pde_loss + lambda_terminal * terminal_loss + lambda_data * data_loss
    loss.backward()
    optimizer.step()

    pde_losses.append(pde_loss.item())
    terminal_losses.append(terminal_loss.item())
    data_losses.append(data_loss.item())

    model.eval()
    with torch.no_grad():
        val_pred_v = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
        val_pred_price = val_pred_v * val_tensor["K"]
        val_loss = torch.mean((val_pred_price - val_tensor["price"]) ** 2).item()
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1}/{max_epochs}  pde={pde_loss.item():.6f}  "
              f"terminal={terminal_loss.item():.6f}  data={data_loss.item():.6f}  "
              f"val_MSE={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.1e}")

    if epochs_without_improvement >= patience:
        print(f"Stopped early at epoch {epoch+1} -- no val improvement for {patience} epochs.")
        break

model.load_state_dict(best_state)
adam_epochs_completed = len(pde_losses)
print(f"Adam stage complete. Best validation MSE: {best_val_loss:.6f}")

# Improve the model with L-BFGS and keep the best version
def evaluate_val_mse():
    with torch.no_grad():
        val_pred_v_e = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
        val_pred_price_e = val_pred_v_e * val_tensor["K"]
        return torch.mean((val_pred_price_e - val_tensor["price"]) ** 2).item()

print("Fine-tuning with L-BFGS...")
for lbfgs_round in range(4):
    lbfgs_tau_i, lbfgs_x_i, lbfgs_r_i, lbfgs_sigma_i = sample_interior(4000)
    lbfgs_tau_t, lbfgs_x_t, lbfgs_r_t, lbfgs_sigma_t, lbfgs_smoothed_target_t = sample_terminal_smoothed(1000)

    def lbfgs_closure():
        lbfgs_optimizer.zero_grad()
        residual = pde_residual(lbfgs_tau_i, lbfgs_x_i, lbfgs_r_i, lbfgs_sigma_i)
        pde_loss_l = torch.mean(residual ** 2)
        v_term = v_model(lbfgs_tau_t, lbfgs_x_t, lbfgs_r_t, lbfgs_sigma_t)
        terminal_loss_l = torch.mean((v_term - lbfgs_smoothed_target_t) ** 2)
        data_loss_l = data_loss_fn()
        loss = pde_loss_l + lambda_terminal * terminal_loss_l + lambda_data * data_loss_l
        loss.backward()
        pde_losses.append(pde_loss_l.item())
        terminal_losses.append(terminal_loss_l.item())
        data_losses.append(data_loss_l.item())
        with torch.no_grad():
            val_pred_v_l = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
            val_pred_price_l = val_pred_v_l * val_tensor["K"]
            val_losses.append(torch.mean((val_pred_price_l - val_tensor["price"]) ** 2).item())
        return loss

    lbfgs_optimizer = optim.LBFGS(model.parameters(), lr=0.5, max_iter=400,
                                  history_size=50, line_search_fn="strong_wolfe")
    lbfgs_optimizer.step(lbfgs_closure)

    round_val_loss = evaluate_val_mse()
    print(f"  L-BFGS round {lbfgs_round + 1}: validation MSE = {round_val_loss:.6f} "
          f"(best so far {best_val_loss:.6f})")
    if np.isfinite(round_val_loss) and round_val_loss < best_val_loss:
        best_val_loss = round_val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        # Revert before the next round / final evaluation: keeps the model at
        # the best validated weights and guards against NaN divergence.
        model.load_state_dict(best_state)

model.load_state_dict(best_state)
print(f"Best validation MSE after all stages: {best_val_loss:.6f}")

# ------------------------------------------------------------------------------------------------------------
# Evaluation on the held-out test set 

test_x = to_xtau(X_test_final)
test_tensor = {
    "x": torch.tensor(test_x[:, 0:1]),
    "tau": torch.tensor(test_x[:, 1:2]),
    "r": torch.tensor(test_x[:, 2:3]),
    "sigma": torch.tensor(test_x[:, 3:4]),
    "K": torch.tensor(X_test_final[:, 1:2]),
}
model.eval()
with torch.no_grad():
    v_pred = v_model(test_tensor["tau"], test_tensor["x"], test_tensor["r"], test_tensor["sigma"])
    
    # Calculate the final predictions
    y_pred_raw = (v_pred * test_tensor["K"]).numpy()
y_pred = np.clip(y_pred_raw, 0, None)
y_true = price_test_final.reshape(-1, 1)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
absolute_error = np.abs(y_pred - y_true)

print(f"\n=== HYBRID results (PDE + {N_LABELLED} labels) ===")
print("MSE:", mse)
print("MAE:", mae)
for price_threshold in [1.0, 0.1, 0.01]:
    mask = y_true.flatten() > price_threshold
    rel_err = absolute_error.flatten()[mask] / y_true.flatten()[mask]
    print(f"threshold={price_threshold}: n={mask.sum()}, mean rel. error={rel_err.mean():.4f}")

# ------------------------------------------------------------------------------------------------------------
# Matched comparison: re-score Experiments 1 and 2
def scored(name, y_t, y_p):
    y_t = np.asarray(y_t).flatten(); y_p = np.asarray(y_p).flatten()
    m = mean_squared_error(y_t, y_p)
    a = mean_absolute_error(y_t, y_p)
    msk = y_t > 0.1
    rel = (np.abs(y_p - y_t)[msk] / y_t[msk]).mean()
    return name, m, a, rel

rows = []
sup_path = "black_scholes_supervised_results.csv"
if os.path.exists(sup_path):
    sup = pd.read_csv(sup_path)
    if len(sup) == len(X_test) and np.allclose(
            sup[["S", "K", "T", "r", "sigma"]].values[idx_test_final],
            X_test_final, rtol=1e-6, atol=1e-8):
        rows.append(scored("Supervised (Exp 1, 40,000 labels)",
                           price_test_final, sup["NN_price"].values[idx_test_final]))
    else:
        print("WARNING: supervised results file does not align with the "
              "current test set -- re-run 1_supervised_experiment.py.")
else:
    print("NOTE: run 1_supervised_experiment.py first to include the "
          "supervised baseline in the matched comparison.")

pde_path = "logtransform_pde_results.csv"
if os.path.exists(pde_path):
    pde = pd.read_csv(pde_path)
    if len(pde) == len(X_test_final) and np.allclose(
            pde[["S", "K", "T", "r", "sigma"]].values,
            X_test_final, rtol=1e-6, atol=1e-8):
        rows.append(scored("PDE-residual only (Exp 2, 0 labels)",
                           price_test_final, pde["NN_price"].values))
    else:
        print("WARNING: PDE results file does not align with the current "
              "test set -- re-run 2_PDE_residual_experiment.py.")
else:
    print("NOTE: run 2_PDE_residual_experiment.py first to include the "
          "PDE-only baseline in the matched comparison.")

rows.append(scored(f"Hybrid (Exp 3, PDE + {N_LABELLED} labels)", y_true, y_pred))

header = f"{'Method':<42}{'MSE':>12}{'MAE':>10}{'rel.err(V>0.1)':>17}"
lines = [header, "-" * len(header)]
for name, m, a, rel in rows:
    lines.append(f"{name:<42}{m:>12.5f}{a:>10.4f}{rel:>16.2%}")
table = "\n".join(lines)
print("\n=== Matched comparison on the identical 5,000-option test set ===")
print(table)
with open("hybrid_comparison.txt", "w") as f:
    f.write("Matched comparison on the identical 5,000-option test set\n")
    f.write(table + "\n")

if len(rows) == 3:
    hyb = rows[-1]
    if hyb[2] < rows[0][2] and hyb[2] < rows[1][2]:
        print("\nThe hybrid improves on BOTH the fully supervised and the "
              "PDE-only baselines (MAE).")
    elif hyb[2] < rows[1][2]:
        print("\nThe hybrid improves on the PDE-only baseline but not on "
              "the fully supervised one (MAE).")

results = pd.DataFrame(X_test_final, columns=["S", "K", "T", "r", "sigma"])
results["BS_price"] = y_true.flatten()
results["NN_price"] = y_pred.flatten()
results["absolute_error"] = np.abs(results["NN_price"] - results["BS_price"])
mask = results["BS_price"] > 0.1
results["relative_error"] = np.nan
results.loc[mask, "relative_error"] = results.loc[mask, "absolute_error"] / results.loc[mask, "BS_price"]
results["moneyness"] = results["S"] / results["K"]
results.to_csv("hybrid_results.csv", index=False)



plt.figure(figsize=(8, 5))
# ------------------------------------------------------------------------------------------------------------
# PDE error
plt.plot(pde_losses, label="PDE residual loss")

# ------------------------------------------------------------------------------------------------------------
# Terminal error
plt.plot(terminal_losses, label="Terminal (smoothed) loss")
plt.plot(data_losses, label=f"Data loss ({N_LABELLED} labels)")

# ------------------------------------------------------------------------------------------------------------
# Validation error
plt.plot(val_losses, label="Validation MSE vs Black--Scholes (monitoring only)")
plt.axvline(adam_epochs_completed, color="gray", linestyle="--", linewidth=1, label="Adam ends / L-BFGS begins")
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("Loss (log scale)")
plt.legend(); plt.grid(True, which="both")
plt.savefig("hybrid_training_loss.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Predicted prices vs Black-Scholes prices
plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="black")
plt.xlabel("Black--Scholes Price"); plt.ylabel("Hybrid Predicted Price")
plt.grid(True)
plt.savefig("hybrid_nn_vs_black_scholes.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Relative error vs moneyness
plt.figure(figsize=(8, 5))
plt.scatter(results.loc[mask, "moneyness"], results.loc[mask, "relative_error"], alpha=0.3)
plt.xlabel("Moneyness S/K"); plt.ylabel("Filtered Relative Pricing Error")
plt.grid(True)
plt.savefig("hybrid_relative_error_vs_moneyness.pdf", bbox_inches="tight")
plt.show()
