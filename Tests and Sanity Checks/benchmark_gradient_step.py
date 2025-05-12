import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from efgpnd import ToeplitzND

def benchmark_gradient_step(n_values, device='cpu', num_runs=5):
    """
    Benchmark gradient step times for different problem sizes.
    
    Args:
        n_values: List of problem sizes to benchmark
        device: Device to run on ('cpu' or 'cuda')
        num_runs: Number of runs to average over
        
    Returns:
        Dictionary with timing results for both implementations
    """
    # Hyperparameters from the original benchmark
    lengthscale = 0.1
    variance = 1.0
    batch_size = 1
    
    results = {
        'original': [],
        'optimized': []
    }
    
    for n in n_values:
        print(f"Benchmarking n = {n}")
        
        # Create input data
        x = torch.randn(batch_size, n, device=device)
        
        # Initialize kernels
        original_kernel = ToeplitzND(lengthscale=lengthscale, variance=variance, device=device)
        optimized_kernel = OptimizedToeplitzND(lengthscale=lengthscale, variance=variance, device=device)
        
        # Warm up
        for _ in range(2):
            _ = original_kernel(x)
            _ = optimized_kernel(x)
        
        # Benchmark original implementation
        original_times = []
        for _ in range(num_runs):
            start_time = time.time()
            k_xx = original_kernel(x)
            loss = k_xx.trace()
            loss.backward()
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            original_times.append(end_time - start_time)
        
        # Benchmark optimized implementation
        optimized_times = []
        for _ in range(num_runs):
            start_time = time.time()
            k_xx = optimized_kernel(x)
            loss = k_xx.trace()
            loss.backward()
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            optimized_times.append(end_time - start_time)
        
        # Store average times
        results['original'].append(np.mean(original_times))
        results['optimized'].append(np.mean(optimized_times))
        
        print(f"  Original: {np.mean(original_times):.4f}s")
        print(f"  Optimized: {np.mean(optimized_times):.4f}s")
        print(f"  Speedup: {np.mean(original_times) / np.mean(optimized_times):.2f}x")
    
    return results

def plot_results(n_values, results):
    """Plot the benchmark results."""
    plt.figure(figsize=(10, 6))
    
    # Plot times
    plt.semilogx(n_values, results['original'], 'o-', label='Original ToeplitzND')
    plt.semilogx(n_values, results['optimized'], 's-', label='Optimized ToeplitzND')
    
    # Plot speedup
    speedup = np.array(results['original']) / np.array(results['optimized'])
    plt.semilogx(n_values, speedup, '^-', label='Speedup (Original/Optimized)')
    
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Time (s) / Speedup')
    plt.title('Gradient Step Time Comparison')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add speedup values as annotations
    for i, n in enumerate(n_values):
        plt.annotate(f"{speedup[i]:.2f}x", 
                    xy=(n, speedup[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('gradient_step_benchmark.png')
    plt.show()

if __name__ == "__main__":
    # Problem sizes to benchmark
    n_values = [10000, 100000, 1000000, 10000000]
    
    # Run benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on {device}")
    
    results = benchmark_gradient_step(n_values, device=device)
    
    # Plot results
    plot_results(n_values, results) 