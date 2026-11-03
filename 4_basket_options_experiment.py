import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# ============================================================================================================
# Experiment 4:

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------------------------------------------------------------------------------------------

# Simulate the basket option price with Monte Carlo
def simulate_basket_call_price(S0, K, T, r, sigma, corr, M=100_000, seed=None):
    S0 = np.asarray(S0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    d = len(S0)

    rng = np.random.default_rng(seed)

    # Create correlated random values
    L = np.linalg.cholesky(corr)

    n_half = M // 2
    Z = rng.standard_normal((n_half, d))
    Z_antithetic = -Z

    def terminal_prices(Z_batch):
        Z_corr = Z_batch @ L.T
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T) * Z_corr
        return S0 * np.exp(drift + diffusion)

    # Final asset prices
    ST_1 = terminal_prices(Z)
    ST_2 = terminal_prices(Z_antithetic)

    # Calculate the basket value
    basket_1 = ST_1.mean(axis=1)
    basket_2 = ST_2.mean(axis=1)
    payoff_1 = np.maximum(basket_1 - K, 0.0)
    payoff_2 = np.maximum(basket_2 - K, 0.0)
    paired_payoff = 0.5 * (payoff_1 + payoff_2)

    discounted = np.exp(-r * T) * paired_payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, std_error


def random_correlation_matrix(d, rng, k=8):
    A = rng.standard_normal((d, k))
    G = A @ A.T
    diag_sqrt_inv = 1.0 / np.sqrt(np.diag(G))
    corr = G * np.outer(diag_sqrt_inv, diag_sqrt_inv)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr

# Use 3 assets
D = 3

# Generate 50,000 basket option samples
N_SAMPLES = 50_000
M_PATHS = 20_000
PRICE_FLOOR = 0.01
S_LO, S_HI = 50.0, 150.0
K_LO, K_HI = 50.0, 150.0
T_LO, T_HI = 0.05, 2.0
R_LO, R_HI = 0.01, 0.06
SIG_LO, SIG_HI = 0.10, 0.60

print(f"STEP 2: Generating {N_SAMPLES} basket option samples (d={D} assets)...")
rng = np.random.default_rng(seed)

S_all = np.zeros((N_SAMPLES, D))
sigma_all = np.zeros((N_SAMPLES, D))
corr_all = np.zeros((N_SAMPLES, D, D))
K_all = np.zeros(N_SAMPLES)
T_all = np.zeros(N_SAMPLES)
r_all = np.zeros(N_SAMPLES)
price_all = np.zeros(N_SAMPLES)

n_accepted = 0
n_rejected = 0
t_start = time.time()

while n_accepted < N_SAMPLES:

    # Generate random option parameters
    S0 = rng.uniform(S_LO, S_HI, D)
    K = rng.uniform(K_LO, K_HI)
    T = rng.uniform(T_LO, T_HI)
    r = rng.uniform(R_LO, R_HI)
    sigma = rng.uniform(SIG_LO, SIG_HI, D)
    corr = random_correlation_matrix(D, rng)

    # Calculate the Monte Carlo basket option price
    price, se = simulate_basket_call_price(S0, K, T, r, sigma, corr, M=M_PATHS, seed=None)

    if price <= PRICE_FLOOR:
        n_rejected += 1
        continue

    S_all[n_accepted] = S0
    sigma_all[n_accepted] = sigma
    corr_all[n_accepted] = corr
    K_all[n_accepted] = K
    T_all[n_accepted] = T
    r_all[n_accepted] = r
    price_all[n_accepted] = price
    n_accepted += 1

    if n_accepted % 10_000 == 0:
        print(f"  {n_accepted}/{N_SAMPLES} generated ({time.time()-t_start:.1f}s elapsed)")

print(f"STEP 2 done: {N_SAMPLES} samples in {time.time()-t_start:.1f}s "
      f"({n_rejected} rejected)")

np.savez("basket_training_data.npz", S=S_all, sigma=sigma_all, corr=corr_all,
         K=K_all, T=T_all, r=r_all, price=price_all)


S = S_all; sigma = sigma_all; corr = corr_all
K = K_all; T = T_all; r = r_all; price = price_all
N, d = S.shape
n_corr_features = d * (d - 1) // 2

print(f"\nSTEP 3: Training bagging ensemble (d={d} assets)...")

# Use moneyness S/K as input
moneyness = S / K[:, None]
triu_idx = np.triu_indices(d, k=1)
corr_features = corr[:, triu_idx[0], triu_idx[1]]

X = np.concatenate([moneyness, T[:, None], r[:, None], sigma, corr_features], axis=1)
n_features = X.shape[1]
print(f"Input feature dimension: {n_features}")

# Transform the option price target
y = np.sqrt(price / K)[:, None]

# Split into training and test data
X_train, X_test, y_train, y_test, K_train, K_test = train_test_split(
    X, y, K, test_size=0.2, random_state=seed
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# Define the basket-pricing neural network
class BasketPricingNN(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


# Use 5 neural networks in the ensemble
n_bag_models = 5
bag_models = []
rng_bag = np.random.RandomState(seed)
all_train_loss_history = []
all_test_loss_history = []

for bag_i in range(n_bag_models):
    boot_idx = rng_bag.randint(0, len(X_train_scaled), len(X_train_scaled))
    X_boot = torch.tensor(X_train_scaled[boot_idx], dtype=torch.float32)
    y_boot = torch.tensor(y_train_scaled[boot_idx], dtype=torch.float32)
    boot_ds = TensorDataset(X_boot, y_boot)
    boot_loader = DataLoader(boot_ds, batch_size=256, shuffle=True)

    torch.manual_seed(seed + bag_i)
    model = BasketPricingNN(n_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 1000
    patience = 20
    best_test_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    train_loss_history = []
    test_loss_history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in boot_loader:
            optimizer.zero_grad()

            # Calculate the training error
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(xb)
        train_loss_history.append(epoch_train_loss / len(boot_ds))

        model.eval()
        with torch.no_grad():

            # Check the test error
            test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
        test_loss_history.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    bag_models.append(model)
    if bag_i == 0:
        all_train_loss_history = train_loss_history
        all_test_loss_history = test_loss_history
    print(f"  ensemble member {bag_i+1}/{n_bag_models}: "
          f"stopped at epoch {epoch+1}, best test loss={best_test_loss:.6f}")

print("STEP 3 done: bagging ensemble training complete.")

# Collect predictions from all 5 networks
member_preds = []
for model in bag_models:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_tensor).numpy()
    member_preds.append(scaler_y.inverse_transform(pred_scaled).flatten())

pred_sqrt_ratio = np.clip(np.mean(member_preds, axis=0), 0, None)

# Convert back to the basket option price
NN_price = np.clip((pred_sqrt_ratio ** 2) * K_test, 0, None)
true_price = (y_test.flatten() ** 2) * K_test

mse = mean_squared_error(true_price, NN_price)
mae = mean_absolute_error(true_price, NN_price)
abs_error = np.abs(NN_price - true_price)
rel_error = abs_error / np.maximum(true_price, 1e-8)

print(f"\n=== Test set results (d={d} basket options, bagging ensemble) ===")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Mean relative error: {rel_error.mean()*100:.2f}%")
print(f"Median relative error: {np.median(rel_error)*100:.2f}%")
for threshold in [1.0, 0.1, 0.01]:
    mask = true_price > threshold
    print(f"  threshold={threshold}: n={mask.sum()}, "
          f"mean rel. error={rel_error[mask].mean()*100:.2f}%")

np.savez("basket_model_results.npz",
         true_price=true_price, NN_price=NN_price,
         X_test=X_test, K_test=K_test,
         train_loss_history=np.array(all_train_loss_history),
         test_loss_history=np.array(all_test_loss_history),
         n_corr_features=n_corr_features, d=d)



print("\nSTEP 4: Generating plots...")

moneyness_cols = X_test[:, 0:d]
corr_cols = X_test[:, 2 + 2 * d: 2 + 2 * d + n_corr_features]
basket_moneyness = moneyness_cols.mean(axis=1)
mean_pairwise_corr = corr_cols.mean(axis=1)

plt.rcParams.update({"font.size": 14})

# ------------------------------------------------------------------------------------------------------------
# Training and test errors
plt.figure(figsize=(8, 5))
plt.plot(all_train_loss_history, label="Training loss")
plt.plot(all_test_loss_history, label="Test loss")
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("Loss (log scale)")
plt.legend(); plt.grid(True, alpha=0.4, which="both")
plt.savefig("basket_training_loss.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
#  Neural-network prices vs Monte Carlo prices
plt.figure(figsize=(7, 6))
plt.scatter(true_price, NN_price, alpha=0.3, s=10)
lo, hi = true_price.min(), true_price.max()
plt.plot([lo, hi], [lo, hi], color="black", label="Perfect agreement")
plt.xlabel("True Price (Monte Carlo)"); plt.ylabel("Predicted Price (Neural Network)")
plt.legend(); plt.grid(True, alpha=0.4)
plt.savefig("basket_predicted_vs_true.pdf", bbox_inches="tight")
plt.show()

price_threshold = 0.1
mask = true_price > price_threshold

# ------------------------------------------------------------------------------------------------------------
# Relative error vs basket moneyness
plt.figure(figsize=(8, 5))
plt.scatter(basket_moneyness[mask], rel_error[mask], alpha=0.3, s=10)
plt.xlabel("Basket Moneyness  (mean(S_i) / K)"); plt.ylabel("Relative Pricing Error")
plt.grid(True, alpha=0.4)
plt.savefig("basket_relative_error_vs_moneyness.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------------------------------------
# Relative error vs correlation
plt.figure(figsize=(8, 5))
plt.scatter(mean_pairwise_corr[mask], rel_error[mask], alpha=0.3, s=10)
plt.xlabel("Mean Pairwise Correlation"); plt.ylabel("Relative Pricing Error")
plt.grid(True, alpha=0.4)
plt.savefig("basket_relative_error_vs_correlation.pdf", bbox_inches="tight")
plt.show()

print("\nALL DONE. Saved 4 PDF plots and basket_model_results.npz.")