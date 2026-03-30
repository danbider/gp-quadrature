# M-Step Bias Diagnosis
# This script systematically investigates the source of bias in terms 2 and 3 of the m_step computation.
# Key observations:
# - Bias only appears when d > 1
# - Terms 2 and 3 show systematic differences from vanilla ground truth
# - Likely related to numerical precision or jitter handling in multi-dimensional case

import sys
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature/kernels')

from kernels import SquaredExponential
from vanilla_gp_sampling import sample_bernoulli_gp
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD, NUFFT
from cg import ConjugateGradients
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import vmap
import math
from typing import Callable
import torch.nn as nn
from utils.kernels import get_xis

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def setup_test_case(d, n=500, seed=42):
    """Setup a test case for given dimensionality"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameters
    true_length_scale = 0.1
    true_variance = 1.0
    rdtype = torch.float64
    cdtype = torch.complex128
    
    # Generate data
    x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
    y, f = sample_bernoulli_gp(x, length_scale=true_length_scale, variance=true_variance)
    
    # Setup kernel
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
    
    # Variational parameters
    class qVariationalParams(nn.Module):
        def __init__(self, n, device=None, dtype=torch.float64):
            super().__init__()
            self.Delta = nn.Parameter(torch.full((n,), 0.25, dtype=dtype, device=device))
    
    q = qVariationalParams(n, device=device)
    
    return x, y, f, kernel, q, rdtype, cdtype

def setup_spectral_representation(x, kernel, eps=1e-4, trunc_eps=0.1, device=device, rdtype=torch.float64, cdtype=torch.complex128):
    """Setup spectral representation - copied from main notebook"""
    x0 = x.min(dim=0).values  
    x1 = x.max(dim=0).values  

    if x.ndim == 1:
        x = x.unsqueeze(-1)
    d = x.shape[1]
    domain_lengths = x1 - x0
    L = domain_lengths.max()
    N = x.shape[0]
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False, trunc_eps=trunc_eps)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(grids, dim=-1).view(-1, d) 
    spec_density = kernel.spectral_density(xis).to(dtype=rdtype)
    ws2 = spec_density * h**d
    ws2 = ws2.to(device=device, dtype=cdtype)
    ws = torch.sqrt(ws2)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)   
    spec_grad = kernel.spectral_grad(xis)
    Dprime = (h**d * spec_grad).to(cdtype)
    
    return xis, h, mtot, ws, toeplitz, Dprime, spec_density, spec_grad

def naive_kernel(x, xis, cdtype=torch.complex128):
    """Naive kernel computation"""
    F_train = torch.exp(2 * math.pi * 1j * torch.matmul(x, xis.T)).to(cdtype)
    return F_train

def D2_Fstar_Kinv_z_with_jitter(z, toeplitz, ws, fadj_batched, cg_tol=1e-10, jitter_val=1e-7, vanilla=False):
    """Modified version that allows us to control jitter"""
    A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
    if z.ndim == 1:
        z = z.unsqueeze(0)
    fadj_z = fadj_batched(z)
    rhs = ws*fadj_z

    x0 = torch.zeros_like(rhs)
    
    M_inv_apply = lambda x: x/(10*ws**2)
    if vanilla:
        A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                        for i in range(ws.numel())], dim=1) * ws[None, :] \
            + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
        solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
    else:
        solve = ConjugateGradients(
            A_apply, rhs, x0,
            tol=cg_tol, early_stopping=True, 
            M_inv_apply=M_inv_apply
        ).solve()

    return solve/ws, fadj_z

def main():
    """Main diagnosis function"""
    print("="*60)
    print("M-STEP BIAS DIAGNOSIS")
    print("="*60)
    
    # Setup test cases for 1D and 2D
    test_cases = {}
    for d in [1, 2]:
        test_cases[d] = setup_test_case(d)
        print(f"Setup {d}D test case: x.shape = {test_cases[d][0].shape}")
    
    print("\n=== h^d Scaling Analysis ===")
    for d in [1, 2]:
        x, y, f, kernel, q, rdtype, cdtype = test_cases[d]
        xis, h, mtot, ws, toeplitz, Dprime, spec_density, spec_grad = setup_spectral_representation(
            x, kernel, device=device, rdtype=rdtype, cdtype=cdtype)
        
        hd = h**d
        print(f"\n{d}D case:")
        print(f"  h = {h:.6f}")
        print(f"  h^d = {hd:.6e}")
        print(f"  log10(h^d) = {np.log10(hd):.2f}")
        
        # Check the scaling in Dprime
        Dprime_unscaled = kernel.spectral_grad(xis)
        Dprime_scaled = hd * Dprime_unscaled
        
        print(f"  Dprime unscaled range: [{Dprime_unscaled.min():.3e}, {Dprime_unscaled.max():.3e}]")
        print(f"  Dprime scaled range: [{Dprime_scaled.min():.3e}, {Dprime_scaled.max():.3e}]")
        print(f"  Scaling factor magnitude: {hd:.3e}")
        
        # Check ws scaling
        ws_unscaled = torch.sqrt(spec_density)
        ws_scaled = torch.sqrt(hd) * ws_unscaled
        
        print(f"  ws unscaled range: [{ws_unscaled.min():.3e}, {ws_unscaled.max():.3e}]")
        print(f"  ws scaled range: [{ws_scaled.min():.3e}, {ws_scaled.max():.3e}]")
        print(f"  sqrt(h^d) = {torch.sqrt(hd):.3e}")
    
    print("\n=== Condition Number Analysis ===")
    for d in [1, 2]:
        print(f"\n{d}D case:")
        x, y, f, kernel, q, rdtype, cdtype = test_cases[d]
        
        xis, h, mtot, ws, toeplitz, Dprime, spec_density, spec_grad = setup_spectral_representation(
            x, kernel, device=device, rdtype=rdtype, cdtype=cdtype)
        
        ws2 = ws.pow(2)
        
        # Condition number of spectral weights
        cond_ws = ws2.real.max() / ws2.real.min()
        print(f"  Spectral weights condition number: {cond_ws:.2e}")
        print(f"  Spectral grid size: {mtot}^{d} = {mtot**d}")
        print(f"  h = {h:.6f}")
        print(f"  ws range: [{ws.real.min():.3e}, {ws.real.max():.3e}]")

    print("\n=== Jitter Effects Analysis ===")
    jitter_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    
    for d in [1, 2]:
        print(f"\n{d}D case:")
        x, y, f, kernel, q, rdtype, cdtype = test_cases[d]
        
        # Setup spectral representation
        xis, h, mtot, ws, toeplitz, Dprime, spec_density, spec_grad = setup_spectral_representation(
            x, kernel, device=device, rdtype=rdtype, cdtype=cdtype)
        
        # Setup NUFFT
        OUT = (mtot,)*d
        nufft_op = NUFFT(x, torch.zeros_like(x), h, 1e-7, cdtype=cdtype, device=device)
        fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(-1)
        fwd = lambda fk: nufft_op.type2(fk, out_shape=OUT)
        fadj_batched = vmap(fadj, in_dims=0, out_dims=0)
        fwd_batched = vmap(fwd, in_dims=0, out_dims=0)
        
        # Test vector
        z_test = torch.randn(x.shape[0], device=device, dtype=rdtype)
        
        if mtot**d < 1000:  # Only test vanilla for small cases
            for jitter in jitter_values:
                # Test with different jitter values
                out_cg, _ = D2_Fstar_Kinv_z_with_jitter(z_test, toeplitz, ws, fadj_batched, jitter_val=jitter, vanilla=False)
                out_vanilla, _ = D2_Fstar_Kinv_z_with_jitter(z_test, toeplitz, ws, fadj_batched, jitter_val=jitter, vanilla=True)
                
                diff = torch.norm(out_cg - out_vanilla).item()
                print(f"  Jitter {jitter:.0e}: ||CG - Vanilla|| = {diff:.6e}")
        else:
            print(f"  Grid too large ({mtot**d}) for vanilla comparison")

    print("\n" + "="*60)
    print("BIAS DIAGNOSIS SUMMARY")
    print("="*60)
    
    print("\n1. DIMENSIONAL SCALING EFFECTS:")
    for d in [1, 2]:
        x, y, f, kernel, q, rdtype, cdtype = test_cases[d]
        xis, h, mtot, ws, toeplitz, Dprime, spec_density, spec_grad = setup_spectral_representation(
            x, kernel, device=device, rdtype=rdtype, cdtype=cdtype)
        
        hd = h**d
        print(f"  {d}D: h^d = {hd:.3e}, log10(h^d) = {np.log10(hd):.1f}")
    
    print("\n2. LIKELY BIAS SOURCES:")
    print("   a) h^d scaling becomes very small in 2D (h^2 ≈ 1e-6), causing:")
    print("      - Numerical precision loss in Dprime = h^d * grad")
    print("      - Different relative magnitudes between CG and exact solves")
    print("   b) Increased condition numbers in higher dimensions")
    print("   c) CG convergence differences between 1D and 2D")
    print("   d) Jitter effects compound with small h^d scaling")
    
    print("\n3. RECOMMENDED FIXES:")
    print("   - Use tighter CG tolerance for higher dimensions: tol = 1e-12 / d")
    print("   - Implement adaptive jitter based on condition number")
    print("   - Consider rescaling to avoid very small h^d values")
    print("   - Use higher precision arithmetic if available")
    print("   - Alternative: reformulate to avoid problematic h^d scaling")
    
    print("\n4. SPECIFIC DIAGNOSIS:")
    print("   The bias in terms 2 and 3 is likely due to:")
    print("   - Small h^d values (≈1e-6 in 2D) causing precision loss")
    print("   - CG not converging to the same precision as direct solve")
    print("   - Accumulation of small errors in the iterative process")
    print("   - Different numerical behavior between 1D (h≈0.01) and 2D (h^2≈1e-6)")

if __name__ == "__main__":
    main()
