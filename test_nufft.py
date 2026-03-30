import torch
import matplotlib.pyplot as plt
import time
from efgpnd import NUFFT, setup_nufft

# Test parameters
N = 1000  # Number of points
d = 2     # Dimensionality
h = 0.1   # Grid spacing
eps = 1e-6  # NUFFT accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_type1():
    """Test Type 1 NUFFT (nonuniform → uniform)"""
    print(f"\n=== Testing Type 1 NUFFT on {device} ===")
    
    # Create random points and values
    x = torch.rand(N, d, device=device)
    xcen = torch.zeros(d, device=device)
    vals = torch.ones(N, dtype=torch.complex64, device=device)
    
    # Create NUFFT operator
    nufft_op = NUFFT(x, xcen, h, eps)
    
    # Test basic functionality
    result = nufft_op.type1(vals)
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Mean value: {result.abs().mean().item():.6f}")
    
    # Test with batched input
    batch_size = 5
    batch_vals = torch.ones(batch_size, N, dtype=torch.complex64, device=device)
    batch_result = nufft_op.type1(batch_vals)
    print(f"Batched output shape: {batch_result.shape}")
    
    # Test with explicit output shape
    out_shape = (20, 20)
    out_result = nufft_op.type1(vals, out_shape=out_shape)
    print(f"Shaped output shape: {out_result.shape}")
    
    # Compare with legacy function
    phi, old_finufft1, _ = setup_nufft(x, xcen, h, eps, nufft_op.cdtype)
    old_result = old_finufft1(phi, vals, OUT=out_shape)
    max_diff = (old_result - out_result).abs().max().item()
    print(f"Max difference from legacy function: {max_diff:.6e}")
    
    return True

def test_type2():
    """Test Type 2 NUFFT (uniform → nonuniform)"""
    print(f"\n=== Testing Type 2 NUFFT on {device} ===")
    
    # Create random points and grid values
    x = torch.rand(N, d, device=device)
    xcen = torch.zeros(d, device=device)
    
    # Create NUFFT operator
    nufft_op = NUFFT(x, xcen, h, eps)
    
    # Create grid values (M^d points)
    M = 10
    grid_size = M**d
    grid_vals = torch.ones(grid_size, dtype=torch.complex64, device=device)
    
    # Test basic functionality
    result = nufft_op.type2(grid_vals)
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Mean value: {result.abs().mean().item():.6f}")
    
    # Test with batched input
    batch_size = 5
    batch_grid = torch.ones(batch_size, grid_size, dtype=torch.complex64, device=device)
    batch_result = nufft_op.type2(batch_grid)
    print(f"Batched output shape: {batch_result.shape}")
    
    # Test with explicit output shape
    out_shape = (M, M)
    grid_vals_shaped = grid_vals.view(out_shape)
    
    # Compare with legacy function
    phi, _, old_finufft2 = setup_nufft(x, xcen, h, eps, nufft_op.cdtype)
    old_result = old_finufft2(phi, grid_vals, OUT=out_shape)
    
    out_result = nufft_op.type2(grid_vals, out_shape=out_shape)
    max_diff = (old_result - out_result).abs().max().item()
    print(f"Max difference from legacy function: {max_diff:.6e}")
    
    return True

def benchmark():
    """Benchmark the new NUFFT class against the old implementation"""
    print(f"\n=== Benchmarking NUFFT on {device} ===")
    
    # Create data
    x = torch.rand(N, d, device=device)
    xcen = torch.zeros(d, device=device)
    vals = torch.ones(N, dtype=torch.complex64, device=device)
    out_shape = (20, 20)
    
    # Setup for old implementation
    setup_start = time.time()
    phi, old_finufft1, old_finufft2 = setup_nufft(x, xcen, h, eps, torch.complex64)
    setup_time = time.time() - setup_start
    
    # Setup for new implementation
    new_setup_start = time.time()
    nufft_op = NUFFT(x, xcen, h, eps, cdtype=torch.complex64)
    new_setup_time = time.time() - new_setup_start
    
    print(f"Setup time (old): {setup_time:.6f} s")
    print(f"Setup time (new): {new_setup_time:.6f} s")
    
    # Benchmark Type 1
    n_runs = 10
    
    # Old Type 1
    old_start = time.time()
    for _ in range(n_runs):
        old_result1 = old_finufft1(phi, vals, OUT=out_shape)
    old_time1 = (time.time() - old_start) / n_runs
    
    # New Type 1
    new_start = time.time()
    for _ in range(n_runs):
        new_result1 = nufft_op.type1(vals, out_shape=out_shape)
    new_time1 = (time.time() - new_start) / n_runs
    
    print(f"Type 1 avg time (old): {old_time1:.6f} s")
    print(f"Type 1 avg time (new): {new_time1:.6f} s")
    print(f"Type 1 speedup: {old_time1 / new_time1:.2f}x")
    
    # Reshape for Type 2
    grid_vals = torch.ones(out_shape[0] * out_shape[1], dtype=torch.complex64, device=device)
    
    # Old Type 2
    old_start = time.time()
    for _ in range(n_runs):
        old_result2 = old_finufft2(phi, grid_vals, OUT=out_shape)
    old_time2 = (time.time() - old_start) / n_runs
    
    # New Type 2
    new_start = time.time()
    for _ in range(n_runs):
        new_result2 = nufft_op.type2(grid_vals, out_shape=out_shape)
    new_time2 = (time.time() - new_start) / n_runs
    
    print(f"Type 2 avg time (old): {old_time2:.6f} s")
    print(f"Type 2 avg time (new): {new_time2:.6f} s")
    print(f"Type 2 speedup: {old_time2 / new_time2:.2f}x")
    
    return True

if __name__ == "__main__":
    print(f"Running on device: {device}")
    
    try:
        test_type1()
        test_type2()
        benchmark()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Error: {e}") 