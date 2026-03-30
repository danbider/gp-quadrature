"""
Example showing how to use log marginal likelihood computation options in EFGPND.

This script demonstrates how to:
1. Control the precision vs. speed of log marginal likelihood computation
2. Compare different settings for log marginal likelihood accuracy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from efgpnd import EFGPND

# Generate some toy data
np.random.seed(42)
n_points = 100
x = torch.linspace(0, 10, n_points).reshape(-1, 1)
true_y = torch.sin(x) * torch.exp(-0.1 * x)
noise = 0.1 * torch.randn(n_points, 1)
y = true_y + noise

# Create test points for prediction
x_test = torch.linspace(-1, 11, 200).reshape(-1, 1)

def run_with_settings(probes, steps):
    """Run EFGPND with specific log marginal likelihood settings."""
    start_time = time.time()
    
    # Initialize the model with SE kernel
    model = EFGPND(
        x=x,
        y=y,
        kernel="SE",  # Using string-based kernel specification
        eps=1e-4,
        opts={
            # Log marginal parameters now set in fit() rather than opts
            "log_marginal_probes": probes,
            "log_marginal_steps": steps
        }
    )
    
    # Fit the model and compute predictions with log marginal calculation
    model.fit(compute_log_marginal=True)
    mean, var = model.predict(x_test)
    
    # Get the computed log marginal likelihood
    log_marginal = model._cache_full[2]["log_marginal_likelihood"]
    
    elapsed = time.time() - start_time
    
    return {
        "mean": mean.detach().numpy(),
        "var": var.detach().numpy(),
        "log_marginal": log_marginal.item(),
        "time": elapsed,
        "lengthscale": model.kernel.lengthscale,
        "variance": model.kernel.variance,
        "noise": model._gp_params.sig2.item()
    }

# Compare different probe & step settings
settings_list = [
    {"probes": 10, "steps": 10, "label": "Fast (low accuracy)"},
    {"probes": 100, "steps": 25, "label": "Default"},
    {"probes": 500, "steps": 50, "label": "Precise (slow)"}
]

results = {}
for setting in settings_list:
    print(f"Running with {setting['label']} settings: probes={setting['probes']}, steps={setting['steps']}")
    results[setting['label']] = run_with_settings(setting['probes'], setting['steps'])
    print(f"  Completed in {results[setting['label']]['time']:.3f} seconds")
    print(f"  Log marginal likelihood: {results[setting['label']]['log_marginal']:.5f}")
    print(f"  Hyperparameters: lengthscale={results[setting['label']]['lengthscale']:.3f}, "
          f"variance={results[setting['label']]['variance']:.3f}, "
          f"noise={results[setting['label']]['noise']:.3f}")
    print()

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(x.numpy(), y.numpy(), color='black', alpha=0.5, label='Data')

# Plot the true function
plt.plot(x_test.numpy(), torch.sin(x_test) * torch.exp(-0.1 * x_test), 'k--', label='True function')

# Color map for different settings
colors = ['blue', 'red', 'green']

# Plot the predictions
for i, (setting_name, result) in enumerate(results.items()):
    mean = result['mean']
    var = result['var']
    
    # Plot the mean prediction
    plt.plot(x_test.numpy(), mean, color=colors[i], label=f"{setting_name}")
    
    # Plot the confidence intervals
    plt.fill_between(
        x_test.numpy().flatten(),
        mean.flatten() - 1.96 * np.sqrt(var),
        mean.flatten() + 1.96 * np.sqrt(var),
        color=colors[i], alpha=0.2
    )

plt.title('EFGPND with Different Log Marginal Likelihood Settings')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

# Print comparison table
print("====================== Comparison of Results ======================")
print(f"{'Setting':<20} {'Log Marginal':<15} {'Time (s)':<10} {'Hyperparameters'}")
print("="*75)
for setting_name, result in results.items():
    print(f"{setting_name:<20} {result['log_marginal']:<15.5f} {result['time']:<10.3f} "
          f"l={result['lengthscale']:.3f}, v={result['variance']:.3f}, n={result['noise']:.3f}")

print("\nAs you can see, different probe/step settings affect:")
print("1. Computation time")
print("2. Precision of the log marginal likelihood calculation")
print("3. May slightly impact hyperparameter estimation when combined with optimization")
print("\nFor production use, use higher values for more precise results.")
print("For quick iterations during development, lower values may be sufficient.")

plt.savefig('log_marginal_comparison.png')
plt.show() 