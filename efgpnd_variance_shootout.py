import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd

from vanilla_gp_sampling import sample_gp_fast, squared_exponential_kernel
from kernels.squared_exponential import SquaredExponential
from efgpnd import efgp_nd

def generate_2d_data(n_train: int = 100, 
                    length_scale: float = 0.3, 
                    variance: float = 1.0,
                    noise_variance: float = 0.05,
                    seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 2D training data using vanilla GP sampling
    
    Args:
        n_train: Number of training points
        length_scale: Length scale for the kernel
        variance: Variance for the kernel
        noise_variance: Observation noise variance
        seed: Random seed
        
    Returns:
        x: Input points tensor of shape (n_train, 2)
        y: Output values tensor of shape (n_train,)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random 2D points in [-1, 1] x [-1, 1]
    x = torch.rand(n_train, 2, dtype=torch.float64) * 2 - 1
    
    # Sample from GP with squared exponential kernel
    y = sample_gp_fast(
        x=x,
        length_scale=length_scale,
        variance=variance,
        noise_variance=noise_variance,
        num_samples=1
    )
    
    return x, y

def create_test_grid(grid_size: int) -> torch.Tensor:
    """
    Create a 2D test grid of specified size
    
    Args:
        grid_size: Number of points per dimension
        
    Returns:
        x_test: Grid points tensor of shape (grid_size^2, 2)
    """
    # Create 1D grid points
    grid_1d = torch.linspace(-1, 1, grid_size, dtype=torch.float64)
    
    # Create 2D grid
    xx, yy = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    
    # Reshape to (grid_size^2, 2)
    x_test = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    return x_test

def run_efgpnd_variance_comparison(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    grid_sizes: List[int],
    eps: float = 1e-4,
    sigmasq: float = 0.05,
    length_scale: float = 0.3,
    variance: float = 1.0,
    hutchinson_probes: int = 1000,
    n_trials: int = 3
) -> pd.DataFrame:
    """
    Run comparison between regular and stochastic variance methods
    
    Args:
        x_train: Training input points tensor of shape (n_train, 2)
        y_train: Training output values tensor of shape (n_train,)
        grid_sizes: List of grid sizes to test
        eps: Tolerance parameter for EFGPND
        sigmasq: Noise variance
        length_scale: Length scale for the kernel
        variance: Variance for the kernel
        hutchinson_probes: Number of probes for stochastic method
        n_trials: Number of trials to run for each configuration
        
    Returns:
        results: DataFrame with runtime results
    """
    # Create kernel
    kernel = SquaredExponential(dimension=2, lengthscale=length_scale, variance=variance)
    
    # Initialize results dictionary
    results = {
        'grid_size': [],
        'n_test_points': [],
        'method': [],
        'runtime_seconds': [],
        'trial': []
    }
    
    for grid_size in grid_sizes:
        print(f"Testing grid size: {grid_size}x{grid_size} ({grid_size**2} points)")
        
        # Create test grid
        x_test = create_test_grid(grid_size)
        n_test = x_test.shape[0]
        
        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}")
            
            # Regular method
            opts = {
                "estimate_variance": True,
                "variance_method": "regular",
                "cg_tolerance": 1e-4,
                "max_cg_iter": 1000
            }
            
            start_time = time.time()
            _, _, ytrg_regular, _, _ = efgp_nd(
                x=x_train,
                y=y_train,
                sigmasq=sigmasq,
                kernel=kernel,
                eps=eps,
                x_new=x_test,
                opts=opts
            )
            regular_runtime = time.time() - start_time
            
            results['grid_size'].append(grid_size)
            results['n_test_points'].append(n_test)
            results['method'].append('regular')
            results['runtime_seconds'].append(regular_runtime)
            results['trial'].append(trial)
            
            # Stochastic method
            opts = {
                "estimate_variance": True,
                "variance_method": "stochastic",
                "hutchinson_probes": hutchinson_probes,
                "cg_tolerance": 1e-4,
                "max_cg_iter": 1000
            }
            
            start_time = time.time()
            _, _, ytrg_stochastic, _, _ = efgp_nd(
                x=x_train,
                y=y_train,
                sigmasq=sigmasq,
                kernel=kernel,
                eps=eps,
                x_new=x_test,
                opts=opts
            )
            stochastic_runtime = time.time() - start_time
            
            results['grid_size'].append(grid_size)
            results['n_test_points'].append(n_test)
            results['method'].append('stochastic')
            results['runtime_seconds'].append(stochastic_runtime)
            results['trial'].append(trial)
            
            # Verify that both methods produce similar results
            if trial == 0 and grid_size == grid_sizes[0]:
                var_diff = torch.abs(ytrg_regular['var'] - ytrg_stochastic['var']).mean()
                print(f"  Mean absolute difference in variance: {var_diff:.6f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_results(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot the runtime comparison results
    
    Args:
        results_df: DataFrame with runtime results
        save_path: Path to save the plot (if None, just display)
    """
    # Group by grid_size and method, and calculate mean and std of runtime
    summary = results_df.groupby(['grid_size', 'method'])['runtime_seconds'].agg(['mean', 'std']).reset_index()
    
    # Pivot to get methods as columns
    pivot_df = summary.pivot(index='grid_size', columns='method', values=['mean', 'std'])
    
    # Extract values for plotting
    grid_sizes = pivot_df.index.values
    regular_mean = pivot_df[('mean', 'regular')].values
    regular_std = pivot_df[('std', 'regular')].values
    stochastic_mean = pivot_df[('mean', 'stochastic')].values
    stochastic_std = pivot_df[('std', 'stochastic')].values
    
    # Calculate number of test points for each grid size
    n_test_points = [g**2 for g in grid_sizes]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot runtime vs grid size
    plt.subplot(1, 2, 1)
    plt.errorbar(grid_sizes, regular_mean, yerr=regular_std, marker='o', label='Regular')
    plt.errorbar(grid_sizes, stochastic_mean, yerr=stochastic_std, marker='s', label='Stochastic')
    plt.xlabel('Grid Size (per dimension)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Grid Size')
    plt.legend()
    plt.grid(True)
    
    # Plot runtime vs number of test points
    plt.subplot(1, 2, 2)
    plt.errorbar(n_test_points, regular_mean, yerr=regular_std, marker='o', label='Regular')
    plt.errorbar(n_test_points, stochastic_mean, yerr=stochastic_std, marker='s', label='Stochastic')
    plt.xlabel('Number of Test Points')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Test Points')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    # Parameters
    n_train = 100
    length_scale = 0.3
    variance = 1.0
    noise_variance = 0.05
    eps = 1e-4
    hutchinson_probes = 1000
    n_trials = 3
    grid_sizes = [5, 10, 15, 20, 25, 30]  # Adjust based on your computational resources
    
    print("Generating training data...")
    x_train, y_train = generate_2d_data(
        n_train=n_train,
        length_scale=length_scale,
        variance=variance,
        noise_variance=noise_variance
    )
    
    print("Running variance method comparison...")
    results_df = run_efgpnd_variance_comparison(
        x_train=x_train,
        y_train=y_train,
        grid_sizes=grid_sizes,
        eps=eps,
        sigmasq=noise_variance,
        length_scale=length_scale,
        variance=variance,
        hutchinson_probes=hutchinson_probes,
        n_trials=n_trials
    )
    
    # Save results to CSV
    results_df.to_csv('efgpnd_variance_comparison_results.csv', index=False)
    print("Results saved to efgpnd_variance_comparison_results.csv")
    
    # Plot results
    plot_results(results_df, save_path='efgpnd_variance_comparison_plot.png')
    print("Plot saved to efgpnd_variance_comparison_plot.png")
    
    # Print summary
    print("\nSummary of results:")
    summary = results_df.groupby(['grid_size', 'method'])['runtime_seconds'].agg(['mean', 'std']).reset_index()
    print(summary)
    
    # Calculate speedup
    pivot = summary.pivot(index='grid_size', columns='method', values='mean')
    pivot['speedup'] = pivot['regular'] / pivot['stochastic']
    print("\nSpeedup (regular / stochastic):")
    print(pivot[['speedup']])

if __name__ == "__main__":
    main() 