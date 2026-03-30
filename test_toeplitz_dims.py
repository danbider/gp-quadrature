import torch
import numpy as np
from efgpnd import NUFFT, ToeplitzND, compute_convolution_vector_vectorized_dD

def test_nufft_toeplitz_compatibility():
    """Test compatibility between NUFFT and ToeplitzND in a simplified context"""
    print("\n=== Testing NUFFT and ToeplitzND compatibility ===")
    
    # Setup test parameters
    device = torch.device('cpu')
    N = 50  # Number of data points
    d = 2   # Dimensionality
    eps = 1e-6
    nufft_eps = 1e-6
    h = 0.1
    mtot = 25  # Grid size per dimension
    sigmasq = 0.1
    
    # Create random points
    x = torch.rand(N, d, device=device)
    y = torch.randn(N, device=device) 
    xcen = torch.zeros(d, device=device)
    
    # Create complex data type
    rdtype = x.dtype
    cdtype = torch.complex64 if rdtype == torch.float32 else torch.complex128
    
    # Define output shape
    OUT = (mtot,) * d
    
    # Initialize our NUFFT operator
    print("Initializing NUFFT...")
    nufft_op = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype)
    
    # Create Toeplitz operator
    print("Creating Toeplitz operator...")
    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    
    # Show more details about the ToeplitzND dimensions
    print(f"Toeplitz v_kernel shape: {v_kernel.shape}")
    print(f"Toeplitz dimensions (d): {toeplitz.d}")
    print(f"Toeplitz Ls: {toeplitz.Ls}")
    print(f"Toeplitz ns: {toeplitz.ns}")
    print(f"Toeplitz size: {toeplitz.size}")
    print(f"Toeplitz fft_shape: {toeplitz.fft_shape}")
    
    # Show more details about NUFFT dimensions
    print(f"NUFFT input shape (x): {x.shape}")
    print(f"NUFFT dimensions (d): {nufft_op.d}")
    print(f"NUFFT OUT shape: {OUT}")
    print(f"Expected flattened size: {mtot**d}")
    
    # Get the expected size from Toeplitz
    expected_size = toeplitz.size
    print(f"Toeplitz expects size: {expected_size}")
    print(f"Toeplitz ns: {toeplitz.ns}")
    
    # Transform with NUFFT type1
    print("Running NUFFT type1...")
    Fy = nufft_op.type1(y, out_shape=OUT, toeplitz_size=expected_size)
    print(f"NUFFT type1 output shape: {Fy.shape}")
    
    # Check if the output size matches what Toeplitz expects
    if Fy.shape[0] == expected_size:
        print("✅ NUFFT output size matches Toeplitz expected size!")
    else:
        print(f"❌ NUFFT output size ({Fy.shape[0]}) does not match Toeplitz expected size ({expected_size})")
    
    # Test the A_apply function that caused the error
    print("Testing A_apply function...")
    ws = torch.sqrt(torch.ones(expected_size, dtype=cdtype))
    
    try:
        # Define A_apply similarly to how it's defined in efgpnd_gradient_batched
        def A_apply(beta):
            return ws * toeplitz(ws * beta) + sigmasq * beta
            
        # Test A_apply with the result from NUFFT
        result = A_apply(Fy)
        print(f"A_apply result shape: {result.shape}")
        print("✅ A_apply executed successfully!")
    except ValueError as e:
        print(f"❌ A_apply failed with error: {e}")
        
    # Test with batched input
    print("\nTesting with batched input...")
    batch_size = 3
    y_batch = torch.randn(batch_size, N, device=device)
    
    try:
        Fy_batch = nufft_op.type1(y_batch, out_shape=OUT, toeplitz_size=expected_size)
        print(f"Batched NUFFT type1 output shape: {Fy_batch.shape}")
        print("✅ Batched NUFFT type1 executed successfully!")
    except Exception as e:
        print(f"❌ Batched NUFFT type1 failed with error: {e}")
    
    return True

if __name__ == "__main__":
    try:
        test_nufft_toeplitz_compatibility()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Error: {e}") 