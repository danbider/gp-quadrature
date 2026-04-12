"""Generate a clean GP regression figure for a beamer slide."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -- Style for slides --
rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
    "figure.dpi": 300,
})

rng = np.random.default_rng(42)

# Ground-truth function
def f_true(x):
    return np.sin(1.5 * x) + 0.3 * np.cos(3.5 * x)

# Training data
n_train = 8
X_train = rng.uniform(-3, 3, n_train)
X_train.sort()
sigma_noise = 0.25
y_train = f_true(X_train) + rng.normal(0, sigma_noise, n_train)

# GP posterior (RBF kernel, analytic)
def rbf_kernel(x1, x2, l=1.0, sigma_f=1.0):
    return sigma_f**2 * np.exp(-0.5 * np.subtract.outer(x1, x2)**2 / l**2)

X_test = np.linspace(-3.5, 3.5, 200)
K = rbf_kernel(X_train, X_train) + sigma_noise**2 * np.eye(n_train)
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test)

K_inv = np.linalg.inv(K)
mu = K_s.T @ K_inv @ y_train
cov = K_ss - K_s.T @ K_inv @ K_s
std = np.sqrt(np.clip(np.diag(cov), 0, None))

# -- Plot --
fig, ax = plt.subplots(figsize=(4.2, 3.2))

# Confidence band
ax.fill_between(X_test, mu - 2*std, mu + 2*std, alpha=0.2, color="#4C72B0",
                label=r"$\pm 2\sigma$", edgecolor="none")

# Mean
ax.plot(X_test, mu, color="#2C3E7B", lw=2.5, label="Posterior mean")

# Data points
ax.scatter(X_train, y_train, s=60, zorder=5, color="#C44E52",
           edgecolors="white", linewidths=1.0, label="Observations")

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-2.5, 2.5)

# Clean up for slide: minimal chrome
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$x$", fontsize=13)
ax.set_ylabel(r"$y$", fontsize=13, rotation=0, labelpad=10)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
          edgecolor="none", handlelength=1.5, handletextpad=0.5)

fig.tight_layout(pad=0.3)
fig.savefig("GP.png", dpi=300, bbox_inches="tight", transparent=True)
fig.savefig("GP.pdf", bbox_inches="tight", transparent=True)
print("Saved GP.png and GP.pdf")
