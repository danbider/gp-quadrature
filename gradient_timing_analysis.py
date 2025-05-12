#!/usr/bin/env python
"""
Analysis of gradient calculation times and xi counts for different
kernel types, epsilon values, and hyperparameter settings.
"""
import torch
import numpy as np
import time
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from kernels.squared_exponential import SquaredExponential
from kernels.matern import Matern
from efgpnd import get_xis, efgpnd_gradient_batched

def load_data(filepath):
    """Load the temperature data."""
    print(f"Loading temperature data from {filepath}")
    data = torch.load(filepath)
    return data

def measure_gradient_time(x, y, kernel, epsilon, sigmasq=1.0, trace_samples=10):
    """
    Measure the time to calculate the gradient using efgpnd_gradient_batched and count the number of xis.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Input data and target values
    kernel : Kernel object
        Kernel instance (SquaredExponential or Matern)
    epsilon : float
        Quadrature accuracy parameter
    sigmasq : float
        Noise variance parameter
    trace_samples : int
        Number of trace samples for Hutchinson estimator
        
    Returns:
    --------
    dict
        Dictionary with timing and xi count information
    """
    # Get current hyperparameter values
    if hasattr(kernel, 'lengthscale'):
        lengthscale = kernel.lengthscale
        variance = kernel.variance
    else:
        lengthscale = kernel.get_hyper('lengthscale')
        variance = kernel.get_hyper('variance')
    
    # Calculate data bounds (needed for get_xis)
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    
    # We need to access the number of xis used, but efgpnd_gradient_batched doesn't return it
    # Let's call get_xis separately to count them, using the same parameters
    # From the efgpnd_gradient_batched implementation
    L = (x1 - x0).max().item()
    print(f"  Calculating xis for {type(kernel).__name__}{' '+kernel.name if hasattr(kernel, 'name') and kernel.name else ''}, ℓ={lengthscale:.4f}, ε={epsilon:.0e}")
    start_time = time.time()
    xis, h_spacing, mtot = get_xis(kernel_obj=kernel, eps=epsilon, L=L, use_integral=True)
    xis_time = time.time() - start_time
    num_xis = len(xis)
    print(f"    - xis calculation: {xis_time:.4f} seconds, generated {num_xis} quadrature points")
    
    # Measure time for gradient calculation using efgpnd_gradient_batched
    print(f"  Computing gradient...")
    start_time = time.time()
    
    # Call the actual gradient function from efgpnd.py
    # It only returns the gradient, not the loss
    grad = efgpnd_gradient_batched(
        x=x, 
        y=y, 
        sigmasq=sigmasq, 
        kernel=kernel, 
        eps=epsilon,
        trace_samples=trace_samples,
        x0=x0, 
        x1=x1
    )
    
    grad_time = time.time() - start_time
    total_time = xis_time + grad_time
    
    # Print gradient values and timing
    print(f"    - gradient calculation: {grad_time:.4f} seconds")
    print(f"    - total time: {total_time:.4f} seconds")
    print(f"    - gradient values: [∂L/∂ℓ={grad[0].item():.4e}, ∂L/∂σ²={grad[1].item():.4e}, ∂L/∂σ²ₙ={grad[2].item():.4e}]")
    
    return {
        'kernel_type': type(kernel).__name__,
        'kernel_name': getattr(kernel, 'name', ''),
        'lengthscale': lengthscale,
        'variance': variance,
        'epsilon': epsilon,
        'trace_samples': trace_samples,
        'num_xis': num_xis,
        'xis_time': xis_time,
        'grad_time': grad_time,
        'total_time': total_time,
        'grad_lengthscale': grad[0].item() if len(grad) > 0 else None,
        'grad_variance': grad[1].item() if len(grad) > 1 else None
    }

def analyze_gradients(x, y, epsilon_values=None, lengthscale_values=None):
    """
    Analyze gradient calculation times for different kernel types,
    epsilon values, and lengthscale settings, with fixed variance and trace samples.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Input data and target values
    epsilon_values : list, optional
        List of epsilon values to test
    lengthscale_values : list, optional
        List of lengthscale values to test
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timing and xi count information
    """
    # Set default values if not provided
    if epsilon_values is None:
        epsilon_values = [1e-2, 1e-3, 1e-4]
    if lengthscale_values is None:
        data_range = (x.max(dim=0).values - x.min(dim=0).values).max().item()
        lengthscale_values = [0.1 * data_range, 0.5 * data_range, data_range]
    
    # Fixed parameters
    variance = 1.0
    sigmasq = 1.0
    trace_samples = 10
    
    results = []
    
    # Define kernel types to test
    kernel_configs = [
        {'class': SquaredExponential, 'name': None},
        {'class': Matern, 'name': 'matern32'}
    ]
    
    total_combinations = len(kernel_configs) * len(epsilon_values) * len(lengthscale_values)
    print(f"Analyzing gradients for {total_combinations} combinations...")
    print(f"Fixed parameters: variance={variance}, noise={sigmasq}, trace_samples={trace_samples}")
    
    overall_start_time = time.time()
    
    # Loop over all combinations
    combination_count = 0
    
    for kernel_config in kernel_configs:
        kernel_class = kernel_config['class']
        kernel_name = kernel_config['name']
        
        kernel_type = kernel_class.__name__
        if kernel_name:
            kernel_display_name = f"{kernel_type} ({kernel_name})"
        else:
            kernel_display_name = kernel_type
            
        kernel_start_time = time.time()
        print(f"\n{'-' * 60}")
        print(f"Analyzing {kernel_display_name}:")
        print(f"{'-' * 60}")
        
        for lengthscale in lengthscale_values:
            # Create kernel instance with current hyperparameters
            if kernel_class == Matern:
                kernel = kernel_class(
                    dimension=x.shape[1],
                    name=kernel_name,
                    lengthscale=lengthscale,
                    variance=variance
                )
            else:
                kernel = kernel_class(
                    dimension=x.shape[1],
                    lengthscale=lengthscale,
                    variance=variance
                )
            
            print(f"\nTesting lengthscale ℓ={lengthscale:.4f}:")
            ls_start_time = time.time()
            
            for epsilon in epsilon_values:
                combination_count += 1
                print(f"\nCombination {combination_count}/{total_combinations}: " +
                     f"{kernel_display_name}, ℓ={lengthscale:.4f}, ε={epsilon:.0e}")
                
                # Measure gradient calculation time and xi count
                result = measure_gradient_time(
                    x, y, kernel, epsilon, sigmasq=sigmasq, trace_samples=trace_samples
                )
                results.append(result)
            
            ls_time = time.time() - ls_start_time
            print(f"Completed lengthscale ℓ={lengthscale:.4f} in {ls_time:.2f} seconds")
        
        kernel_time = time.time() - kernel_start_time
        print(f"\nCompleted {kernel_display_name} analysis in {kernel_time:.2f} seconds")
    
    overall_time = time.time() - overall_start_time
    print(f"\n{'-' * 60}")
    print(f"Analysis complete. Total time: {overall_time:.2f} seconds")
    print(f"{'-' * 60}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

def main():
    # Start timing the overall execution
    overall_start_time = time.time()
    print(f"Starting gradient timing analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load the USA temperature data
    data = load_data('usa_temp_data.pt')
    x = data['x']  # Scaled coordinates
    y = data['y']  # Temperature values
    
    print(f"Data loaded: {len(x)} points with dimensions {x.shape[1]}")
    
    # Check if the data is in the right format
    if x.dim() == 1:
        x = x.unsqueeze(1)  # Add dimension if 1D
    
    # Define epsilon values to test
    epsilon_values = [1e-2, 1e-3, 1e-4]
    
    # Compute minimum distance between data points
    min_dist = 0.5
    
    # Use 10 times the minimum distance as the minimum lengthscale
    min_lengthscale = min_dist
    print(f"Minimum distance between data points: {min_dist:.6f}")
    print(f"Using minimum lengthscale of 10x minimum distance: {min_lengthscale:.6f}")
    
    # Define lengthscale values to test from min_lengthscale to 5.0
    lengthscale_values = np.logspace(np.log10(min_lengthscale), np.log10(5), 10)
    print(f"Lengthscale values: {', '.join([f'{l:.6f}' for l in lengthscale_values])}")
    
    # Analyze gradients
    print(f"\nStarting gradient analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_start_time = time.time()
    results_df = analyze_gradients(
        x, y, 
        epsilon_values=epsilon_values, 
        lengthscale_values=lengthscale_values
    )
    analysis_time = time.time() - analysis_start_time
    print(f"Gradient analysis completed in {analysis_time:.2f} seconds")
    
    # Add kernel display name for better readability
    results_df['kernel_display'] = results_df.apply(
        lambda row: f"{row['kernel_type']}{' '+row['kernel_name'] if row['kernel_name'] else ''}",
        axis=1
    )
    
    # Create a pivot table for the reorganized output
    print("\nGradient Calculation Results:")
    
    # Create section for number of quadrature points
    print("\nNumber of Quadrature Points:")
    xi_table = pd.pivot_table(
        results_df,
        values='num_xis',
        index=['lengthscale'],
        columns=['kernel_display', 'epsilon']
    )
    print(tabulate(xi_table, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create section for total time
    print("\nTotal Gradient Calculation Time (seconds):")
    time_table = pd.pivot_table(
        results_df,
        values='total_time',
        index=['lengthscale'],
        columns=['kernel_display', 'epsilon']
    )
    print(tabulate(time_table, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Save results to CSV
    results_df.to_csv('gradient_timing_results.csv', index=False)
    xi_table.to_csv('xi_count_table.csv')
    time_table.to_csv('timing_table.csv')
    print("\nResults saved to CSV files")
    
    # Print overall execution time
    overall_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {overall_time:.2f} seconds")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 