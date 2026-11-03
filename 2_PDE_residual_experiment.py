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

# ==========================================================================================================
# EXPERIMENT 2: 


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#Use high numerical precision, which is useful when calculating derivatives
torch.set_default_dtype(torch.float64) 

plt.rcParams.update({"font.size": 14})

# ------------------------------------------------------------------------------------------------------------
# Black-Scholes price for a European call

def black_scholes_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ------------------------------------------------------------------------------------------------------------

# Black-Scholes formula written in transformed variables.
# Instead of using S and K directly, it uses:
# x = log(S/K) and v = V/K.
# This is used to make the smoothed terminal-condition target.

def bs_v_torch(x, tau, r, sigma):
    """Exact transformed Black-Scholes: v = V/K as a function of
    x=ln(S/K), tau, r, sigma. Used only to build the smoothed
    terminal condition target (an exact conditional expectation
    under GBM, not a fitted/learned quantity), following Bayer,
    Siebenmorgen and Tempone (2018)."""
    tau = torch.clamp(tau, min=1e-10)
    d1 = (x + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.exp(x) * normal.cdf(d1) - torch.exp(-r * tau) * normal.cdf(d2)


#Set the ranges for the option parameters
S_LO, S_HI = 50.0, 150.0
K_LO, K_HI = 50.0, 150.0
TAU_LO, TAU_HI = 0.05, 2.0
R_LO, R_HI = 0.01, 0.06
SIG_LO, SIG_HI = 0.10, 0.60
PRICE_FLOOR = 0.01
EPSILON_SMOOTH = 0.02

X_LO, X_HI = -1.3, 1.3

# ------------------------------------------------------------------------------------------------------------
# Smoothed lower bound (Glau and Wunderlich, 2022)

def smoothed_bound(x, tau, r):
    z = torch.exp(x) - torch.exp(-r * tau)
    return F.softplus(z, beta=BETA_BOUND)

def normalize(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0

# ------------------------------------------------------------------------------------------------------------
# Same benchmark test set as Experiment 1 (S,K,tau,r,sigma), 
# for direct comparison. Converted to (x,tau,r,sigma) only at the point of network evaluation.


# Generating 50000 samples (Artificial data)
N = 50000

S_list, K_list, T_list, r_list, sigma_list, price_list = [], [], [], [], [], []
n_accepted = 0

# Generate random option parameters
while n_accepted < N:
    n_batch = N - n_accepted
    S_b = np.random.uniform(S_LO, S_HI, n_batch)
    K_b = np.random.uniform(K_LO, K_HI, n_batch)
    T_b = np.random.uniform(TAU_LO, TAU_HI, n_batch)
    r_b = np.random.uniform(R_LO, R_HI, n_batch)
    sigma_b = np.random.uniform(SIG_LO, SIG_HI, n_batch)
    price_b = black_scholes_call(S_b, K_b, T_b, r_b, sigma_b)

    # Keep only option prices above the minimum price
    keep = price_b > PRICE_FLOOR
    n_accepted += keep.sum()
    S_list.append(S_b[keep]); K_list.append(K_b[keep]); T_list.append(T_b[keep])
    r_list.append(r_b[keep]); sigma_list.append(sigma_b[keep]); price_list.append(price_b[keep])

# Combine all accepted samples
S_all = np.concatenate(S_list)[:N]
K_all = np.concatenate(K_list)[:N]
T_all = np.concatenate(T_list)[:N]
r_all = np.concatenate(r_list)[:N]
sigma_all = np.concatenate(sigma_list)[:N]
price_all = np.concatenate(price_list)[:N]

X_all = np.column_stack([S_all, K_all, T_all, r_all, sigma_all])

# Split the data into training and test sets
X_train_unused, X_test, price_train_unused, price_test = train_test_split(
    X_all, price_all, test_size=0.2, random_state=seed
)
print(f"Benchmark test set: {len(X_test)} options (identical sampling to Experiment 1).")

# Split the test data equally into validation and final test sets
X_val, X_test_final, price_val, price_test_final = train_test_split(
    X_test, price_test, test_size=0.5, random_state=seed
)

# Transform S and K into x = log(S/K) for the PDE model
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
# Define the neural network

class LogTransformedPDENN(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 hidden layers with 128 neurons each
        self.network = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Setting the starting weights and biases
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, tau, x, r, sigma):

        # Normalizung and combining the four inputs
        inp = torch.cat([
            normalize(tau, TAU_LO, TAU_HI),
            normalize(x, X_LO, X_HI),
            normalize(r, R_LO, R_HI),
            normalize(sigma, SIG_LO, SIG_HI),
        ], dim=1)
        return self.network(inp)   # predicts the time value w


model = LogTransformedPDENN()

# Adam optimizer updates the network weights during training
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------------------------------------------------------
# Learning-rate schedule
# Reduce the learning rate during training
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Combine the neural network output with the smoothed lower bound
def v_model(tau, x, r, sigma):
    """Composite trial function v = w_theta + u_hat, following
    Glau and Wunderlich (2022), Section 2.5.2."""
    return model(tau, x, r, sigma) + smoothed_bound(x, tau, r)

# Generate random points inside the PDE domain
def sample_interior(n):
    # PDE enforced on [epsilon, T]: the smoothed terminal condition
    # replaces the payoff at tau = epsilon (see header).
    tau = torch.empty(n, 1).uniform_(EPSILON_SMOOTH, TAU_HI)
    x = torch.empty(n, 1).uniform_(X_LO, X_HI)
    r = torch.empty(n, 1).uniform_(R_LO, R_HI)
    sigma = torch.empty(n, 1).uniform_(SIG_LO, SIG_HI)
    return tau, x, r, sigma

# Generate points for the smoothed terminal condition
def sample_terminal_smoothed(n):
    tau = torch.full((n, 1), EPSILON_SMOOTH)
    x = torch.empty(n, 1).uniform_(X_LO, X_HI)
    r = torch.empty(n, 1).uniform_(R_LO, R_HI)
    sigma = torch.empty(n, 1).uniform_(SIG_LO, SIG_HI)
    smoothed_target = bs_v_torch(x, tau, r, sigma)
    return tau, x, r, sigma, smoothed_target


# Calculate how well the neural network satisfies the PDE    
def pde_residual(tau, x, r, sigma):
    """Constant-coefficient transformed PDE residual:
    dv/dtau - 0.5*sigma^2*d2v/dx2 - (r-0.5*sigma^2)*dv/dx + r*v,
    evaluated on the composite v = w + u_hat. By linearity of the
    operator this is identical to solving the transformed problem
    for w with the bound's residual moved to the right-hand side."""
    tau = tau.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    v = v_model(tau, x, r, sigma)

    # Calculate the derivatives needed in the Black-Scholes PDE
    dv_dtau = torch.autograd.grad(v, tau, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
    residual = dv_dtau - 0.5 * sigma**2 * d2v_dx2 - (r - 0.5 * sigma**2) * dv_dx + r * v
    return residual



n_interior = 4000
n_terminal = 1000
lambda_terminal = 1.0

max_epochs = 3000
patience = 400  
best_val_loss = float("inf")
epochs_without_improvement = 0
pde_losses, terminal_losses, val_losses = [], [], []

print(f"\nTraining log-transformed PDE-residual network "
      f"(smoothed payoff bound beta={BETA_BOUND}, "
      f"smoothed terminal epsilon={EPSILON_SMOOTH})...")

# ------------------------------------------------------------------------------------------------------------
# Main training loop

for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()

    # Calculate the PDE error
    tau_i, x_i, r_i, sigma_i = sample_interior(n_interior)
    residual = pde_residual(tau_i, x_i, r_i, sigma_i)
    pde_loss = torch.mean(residual ** 2)

    # Calculate the terminal-condition error
    tau_t, x_t, r_t, sigma_t, smoothed_target_t = sample_terminal_smoothed(n_terminal)
    v_terminal = v_model(tau_t, x_t, r_t, sigma_t)
    terminal_loss = torch.mean((v_terminal - smoothed_target_t) ** 2)
    
    loss = pde_loss + lambda_terminal * terminal_loss
    loss.backward()
    optimizer.step()

    pde_losses.append(pde_loss.item())
    terminal_losses.append(terminal_loss.item())

    # Check the model on the validation data
    model.eval()
    with torch.no_grad():
        val_pred_v = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
        val_pred_price = val_pred_v * val_tensor["K"]
        val_loss = torch.mean((val_pred_price - val_tensor["price"]) ** 2).item()
    val_losses.append(val_loss)
    scheduler.step()

    # Save the best version of the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1}/{max_epochs}  pde={pde_loss.item():.6f}  "
              f"terminal={terminal_loss.item():.6f}  val_MSE={val_loss:.6f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

    if epochs_without_improvement >= patience:
        print(f"Stopped early at epoch {epoch+1} -- no val improvement for {patience} epochs.")
        break

model.load_state_dict(best_state)
adam_epochs_completed = len(pde_losses)
print(f"Adam stage complete. Best validation MSE: {best_val_loss:.6f}")

# ------------------------------------------------------------------------------------------------------------
# L-BFGS refinement with restarts.

# Calculate the validation MSE
def evaluate_val_mse():
    with torch.no_grad():
        val_pred_v_e = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
        val_pred_price_e = val_pred_v_e * val_tensor["K"]
        return torch.mean((val_pred_price_e - val_tensor["price"]) ** 2).item()

print("Fine-tuning with L-BFGS...")
for lbfgs_round in range(4):
    lbfgs_tau_i, lbfgs_x_i, lbfgs_r_i, lbfgs_sigma_i = sample_interior(4000)
    lbfgs_tau_t, lbfgs_x_t, lbfgs_r_t, lbfgs_sigma_t, lbfgs_smoothed_target_t = sample_terminal_smoothed(1000)

    # Calculate the loss used by L-BFGS
    def lbfgs_closure():
        lbfgs_optimizer.zero_grad()
        residual = pde_residual(lbfgs_tau_i, lbfgs_x_i, lbfgs_r_i, lbfgs_sigma_i)
        pde_loss_l = torch.mean(residual ** 2)
        v_term = v_model(lbfgs_tau_t, lbfgs_x_t, lbfgs_r_t, lbfgs_sigma_t)
        terminal_loss_l = torch.mean((v_term - lbfgs_smoothed_target_t) ** 2)
        loss = pde_loss_l + lambda_terminal * terminal_loss_l
        loss.backward()
        pde_losses.append(pde_loss_l.item())
        terminal_losses.append(terminal_loss_l.item())
        with torch.no_grad():
            val_pred_v_l = v_model(val_tensor["tau"], val_tensor["x"], val_tensor["r"], val_tensor["sigma"])
            val_pred_price_l = val_pred_v_l * val_tensor["K"]
            val_losses.append(torch.mean((val_pred_price_l - val_tensor["price"]) ** 2).item())
        return loss

    # L-BFGS optimizer used to further improve the trained model
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
    y_pred_raw = (v_pred * test_tensor["K"]).numpy()
y_pred = np.clip(y_pred_raw, 0, None)
y_true = price_test_final.reshape(-1, 1)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
absolute_error = np.abs(y_pred - y_true)

print(f"\n=== Log-transformed PDE-residual results "
      f"(with smoothed no-arbitrage bound) ===")
print("MSE:", mse)
print("MAE:", mae)
for price_threshold in [1.0, 0.1, 0.01]:
    mask = y_true.flatten() > price_threshold
    rel_err = absolute_error.flatten()[mask] / y_true.flatten()[mask]
    print(f"threshold={price_threshold}: n={mask.sum()}, mean rel. error={rel_err.mean():.4f}")

print("\n=== Experiment 2 progression (earlier figures from previous runs) ===")
print("Raw variables (variable scaling only):             MSE=4.08     MAE=$1.21")
print("+ smoothed terminal condition:                     MSE=3.26     MAE=$1.10")
print("+ bagging:                                         MSE=3.07     MAE=$1.03")
print("Log-transform + smoothed terminal:                 MSE=0.478    MAE=$0.384")
print(f"Log-transform + smoothed terminal + smoothed")
print(f"no-arbitrage bound (THIS RUN):                     MSE={mse:.4f}   MAE=${mae:.4f}")

results = pd.DataFrame(X_test_final, columns=["S", "K", "T", "r", "sigma"])
results["BS_price"] = y_true.flatten()
results["NN_price"] = y_pred.flatten()
results["absolute_error"] = np.abs(results["NN_price"] - results["BS_price"])

# Calculate relative error only for Black-Scholes prices above 0.1
mask = results["BS_price"] > 0.1
results["relative_error"] = np.nan
results.loc[mask, "relative_error"] = results.loc[mask, "absolute_error"] / results.loc[mask, "BS_price"]
results["moneyness"] = results["S"] / results["K"]
results.to_csv("logtransform_pde_results.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(pde_losses, label="PDE residual loss")
plt.plot(terminal_losses, label="Terminal (smoothed) loss")
plt.plot(val_losses, label="Validation MSE vs Black--Scholes (monitoring only)")

# Shows where Adam training ends and L-BFGS fine-tuning begins
plt.axvline(adam_epochs_completed, color="gray", linestyle="--", linewidth=1, label="Adam ends / L-BFGS begins")
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("Loss (log scale)")
plt.legend(); plt.grid(True, which="both")
plt.savefig("logtransform_pde_training_loss.pdf", bbox_inches="tight")
plt.show()

# Compares neural-network prices with Black-Scholes prices
plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="black")
plt.xlabel("Black--Scholes Price"); plt.ylabel("Log-Transformed PDE-Residual Predicted Price")
plt.grid(True)
plt.savefig("logtransform_pde_nn_vs_black_scholes.pdf", bbox_inches="tight")
plt.show()

# Shows how relative pricing error changes with moneyness
plt.figure(figsize=(8, 5))
plt.scatter(results.loc[mask, "moneyness"], results.loc[mask, "relative_error"], alpha=0.3)
plt.xlabel("Moneyness S/K"); plt.ylabel("Filtered Relative Pricing Error")
plt.grid(True)
plt.savefig("logtransform_pde_relative_error_vs_moneyness.pdf", bbox_inches="tight")
plt.show()
