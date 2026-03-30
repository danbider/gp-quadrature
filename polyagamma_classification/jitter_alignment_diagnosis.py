#!/usr/bin/env python3
"""
Jitter Alignment Diagnosis
Systematically test different jitter values and numerical settings to make
compute_vanilla_gradient and m_step agree (up to Hutchinson probe stochasticity).
"""

import sys
import os
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap

# Import required modules
from kernels import SquaredExponential
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD, NUFFT
from cg import ConjugateGradients
from vanilla_gp_sampling import sample_bernoulli_gp
from utils.kernels import get_xis
from polyagamma_classification.PG_GP import qVariationalParams

def setup_test_case(d, n=50, seed=42):
    """Setup consistent test case"""
    torch.manual_seed(seed)
    device = torch.device('cpu')
    rdtype = torch.float64
    cdtype = torch.complex128
    
    # Generate data
    x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
    y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
    
    # Setup kernel with same parameters
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
    
    # Setup spectral representation
    eps = 1e-7
    trunc_eps = 1e-12
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    domain_lengths = x1 - x0
    L = domain_lengths.max()
    
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False, trunc_eps=trunc_eps)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(grids, dim=-1).view(-1, d)
    
    # Compute spectral weights and gradients
    ws = torch.sqrt(kernel.spectral_density(xis)).to(cdtype)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)
    
    # Setup Toeplitz operator
    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    
    # NUFFT setup
    OUT = (mtot,)*d
    nufft_op = NUFFT(x, torch.zeros_like(x), h, 1e-7, cdtype=cdtype, device=device)
    fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(-1)
    fwd = lambda fk: nufft_op.type2(fk, out_shape=OUT)
    fadj_batched = vmap(fadj, in_dims=0, out_dims=0)
    fwd_batched = vmap(fwd, in_dims=0, out_dims=0)
    
    # Create F_train matrix (n x M) for vanilla method
    F_train = torch.zeros(n, ws.numel(), dtype=cdtype, device=device)
    for i in range(n):
        F_train[i] = fadj(x[i:i+1])
    
    return {
        'x': x, 'y': y, 'f': f, 'kernel': kernel, 'h': h, 'mtot': mtot,
        'ws': ws, 'toeplitz': toeplitz, 'Dprime': Dprime, 'xis': xis,
        'fadj_batched': fadj_batched, 'fwd_batched': fwd_batched,
        'F_train': F_train, 'device': device, 'rdtype': rdtype, 'cdtype': cdtype, 'd': d, 'n': n
    }

def compute_vanilla_gradient_aligned(setup_dict, m, q, jitter=1e-8):
    """
    Aligned version of compute_vanilla_gradient matching your notebook implementation
    """
    x = setup_dict['x']
    y = setup_dict['y']
    kernel = setup_dict['kernel']
    ws = setup_dict['ws']
    F_train = setup_dict['F_train']
    Dprime = setup_dict['Dprime']
    device = setup_dict['device']
    rdtype = setup_dict['rdtype']
    cdtype = setup_dict['cdtype']
    n = setup_dict['n']
    
    print(f"  Vanilla gradient (jitter={jitter:.2e}):")
    
    # Build kernel matrix using spectral representation (same as your notebook)
    Kff = F_train @ torch.diag(ws.pow(2).to(dtype=cdtype)) @ F_train.T.conj()
    Kff = Kff.real
    
    # Add jitter
    K = Kff + jitter * torch.eye(n, device=device, dtype=rdtype)
    
    print(f"    K condition number: {torch.linalg.cond(K):.2e}")
    
    # Cholesky decomposition
    try:
        L = torch.linalg.cholesky(K, upper=False)
    except RuntimeError:
        print(f"    Cholesky failed, adding more jitter")
        K = Kff + (jitter * 10) * torch.eye(n, device=device, dtype=rdtype)
        L = torch.linalg.cholesky(K, upper=False)
    
    # K^{-1}
    I = torch.eye(n, device=device, dtype=rdtype)
    K_inv = torch.cholesky_solve(I, L, upper=False)
    
    # Variational parameters
    omega = q.Delta
    
    # S^{-1} = K^{-1} + diag(ω)
    S_inv = K_inv + torch.diag(omega)
    try:
        LS = torch.linalg.cholesky(S_inv, upper=False)
    except RuntimeError:
        S_inv = S_inv + (jitter * 10) * torch.eye(n, device=device, dtype=rdtype)
        LS = torch.linalg.cholesky(S_inv, upper=False)
    S = torch.cholesky_inverse(LS, upper=False)
    
    # Gradient computation
    m_col = m.unsqueeze(-1)
    v = torch.cholesky_solve(m_col, L, upper=False).squeeze(-1)
    
    # Derivative matrices
    dK_dvar = F_train @ torch.diag(Dprime[:, 1].to(dtype=cdtype)) @ F_train.T.conj()
    dK_dls = F_train @ torch.diag(Dprime[:, 0].to(dtype=cdtype)) @ F_train.T.conj()
    
    dK_dvar_r = dK_dvar.real
    dK_dls_r = dK_dls.real
    
    # Variance gradient
    t1var = v @ (dK_dvar_r @ v)
    KinvS = torch.cholesky_solve(S, L, upper=False)
    t2var = torch.sum(KinvS * (K_inv @ dK_dvar_r))
    t3var = torch.sum(K_inv * dK_dvar_r)
    grad_var = 0.5 * (t1var + t2var - t3var)
    
    # Lengthscale gradient
    t1ls = v @ (dK_dls_r @ v)
    t2ls = torch.sum(KinvS * (K_inv @ dK_dls_r))
    t3ls = torch.sum(K_inv * dK_dls_r)
    grad_ls = 0.5 * (t1ls + t2ls - t3ls)
    
    print(f"    Vanilla terms: t1_var={t1var:.6f}, t2_var={t2var:.6f}, t3_var={t3var:.6f}")
    print(f"    Vanilla terms: t1_ls={t1ls:.6f}, t2_ls={t2ls:.6f}, t3_ls={t3ls:.6f}")
    print(f"    Vanilla gradients: ls={grad_ls:.6f}, var={grad_var:.6f}")
    
    return grad_ls, grad_var, t1ls, t2ls, t3ls, t1var, t2var, t3var

def compute_mstep_gradient_aligned(setup_dict, q, m, jitter_val, cg_tol, vanilla_mode=False):
    """
    Aligned version of m_step gradient computation with configurable parameters
    """
    ws = setup_dict['ws']
    toeplitz = setup_dict['toeplitz']
    Dprime = setup_dict['Dprime']
    fadj_batched = setup_dict['fadj_batched']
    fwd_batched = setup_dict['fwd_batched']
    d = setup_dict['d']
    
    print(f"  M-step gradient (jitter={jitter_val:.2e}, cg_tol={cg_tol:.2e}, vanilla_mode={vanilla_mode}):")
    
    # Condition number for reference
    ws2 = ws.pow(2)
    condition_number = ws2.real.max() / ws2.real.min()
    print(f"    Spectral condition number: {condition_number:.2e}")
    
    def D2_Fstar_Kinv_z_aligned(z):
        A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
        if z.ndim == 1:
            z = z.unsqueeze(0)
        fadj_z = fadj_batched(z)
        rhs = ws*fadj_z

        x0 = torch.zeros_like(rhs)
        M_inv_apply = lambda x: x/(10*ws**2)
        
        if vanilla_mode and ws.numel() < 500:
            # Direct solve (vanilla mode)
            A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                            for i in range(ws.numel())], dim=1) * ws[None, :] \
                + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
            try:
                solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
                print(f"    M-step using direct solve (Cholesky)")
            except:
                solve = torch.linalg.solve(A_dense, rhs.T).T
                print(f"    M-step using direct solve (LU)")
        else:
            # CG solve
            cg_solver = ConjugateGradients(
                A_apply, rhs, x0,
                tol=cg_tol, early_stopping=True, 
                M_inv_apply=M_inv_apply
            )
            solve = cg_solver.solve()
            print(f"    M-step using CG solve (iterations: {cg_solver.iters_completed})")
        
        return solve/ws, fadj_z
    
    def Sigma_z_mm_aligned(z):
        A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
        if z.ndim == 1:
            z = z.unsqueeze(0)
        rhs = ws * z

        x0 = torch.zeros_like(rhs)
        M_inv_apply = lambda x: x/(10*ws**2)
        
        if vanilla_mode and ws.numel() < 500:
            # Direct solve (vanilla mode)
            A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                            for i in range(ws.numel())], dim=1) * ws[None, :] \
                + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
            try:
                solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
            except:
                solve = torch.linalg.solve(A_dense, rhs.T).T
        else:
            # CG solve
            cg_solver = ConjugateGradients(
                A_apply, rhs, x0,
                tol=cg_tol, early_stopping=True, 
                M_inv_apply=M_inv_apply
            )
            solve = cg_solver.solve()
        
        result = fwd_batched(solve/ws)
        return result
    
    # Compute m-step terms
    beta, fadj_delta = D2_Fstar_Kinv_z_aligned(q.Delta)
    
    term1 = torch.sum(beta * Dprime, dim=0)
    
    Sigma_z = Sigma_z_mm_aligned(beta)
    term2 = torch.sum(Sigma_z * q.Delta.unsqueeze(-1) * Dprime.unsqueeze(0), dim=0)
    
    term3 = torch.sum(fadj_delta * Dprime, dim=0)
    
    # Final gradients
    grad_ls = 0.5 * (term1[0] + term2[0] - term3[0]).real
    grad_var = 0.5 * (term1[1] + term2[1] - term3[1]).real
    
    print(f"    M-step terms: term1={term1.real}, term2={term2.real}, term3={term3.real}")
    print(f"    M-step gradients: ls={grad_ls:.6f}, var={grad_var:.6f}")
    
    return grad_ls, grad_var, term1.real, term2.real, term3.real

def test_jitter_alignment():
    """Test different jitter values to find optimal alignment"""
    
    print("="*80)
    print("JITTER ALIGNMENT TEST")
    print("="*80)
    
    results = {}
    
    for d in [1, 2]:
        print(f"\n{'='*30} {d}D CASE {'='*30}")
        
        # Setup test case
        setup_dict = setup_test_case(d, n=30)  # Smaller n for faster testing
        
        # Create consistent variational parameters and posterior mean
        q = qVariationalParams(setup_dict['n'], device=setup_dict['device'])
        q.Delta = torch.ones_like(q.Delta) * 0.1  # Fixed values for consistency
        m = torch.zeros(setup_dict['n'], dtype=setup_dict['rdtype'], device=setup_dict['device'])
        
        print(f"Grid size: {setup_dict['mtot']}^{d} = {setup_dict['mtot']**d}")
        
        # Test different jitter values
        jitter_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        
        results[d] = {}
        
        for jitter in jitter_values:
            print(f"\n--- Testing jitter = {jitter:.1e} ---")
            
            try:
                # Compute vanilla gradient
                grad_ls_v, grad_var_v, t1ls_v, t2ls_v, t3ls_v, t1var_v, t2var_v, t3var_v = \
                    compute_vanilla_gradient_aligned(setup_dict, m, q, jitter=jitter)
                
                # Compute m-step gradient with same jitter and tight CG tolerance
                cg_tol = 1e-14  # Very tight tolerance
                
                # Test both CG and vanilla modes if possible
                if setup_dict['mtot']**d < 500:
                    grad_ls_m_v, grad_var_m_v, term1_m_v, term2_m_v, term3_m_v = \
                        compute_mstep_gradient_aligned(setup_dict, q, m, jitter, cg_tol, vanilla_mode=True)
                else:
                    grad_ls_m_v, grad_var_m_v, term1_m_v, term2_m_v, term3_m_v = None, None, None, None, None
                
                grad_ls_m_cg, grad_var_m_cg, term1_m_cg, term2_m_cg, term3_m_cg = \
                    compute_mstep_gradient_aligned(setup_dict, q, m, jitter, cg_tol, vanilla_mode=False)
                
                # Compute differences
                diff_v_cg_ls = abs(grad_ls_v - grad_ls_m_cg)
                diff_v_cg_var = abs(grad_var_v - grad_var_m_cg)
                
                if grad_ls_m_v is not None:
                    diff_v_mv_ls = abs(grad_ls_v - grad_ls_m_v)
                    diff_v_mv_var = abs(grad_var_v - grad_var_m_v)
                    diff_mv_cg_ls = abs(grad_ls_m_v - grad_ls_m_cg)
                    diff_mv_cg_var = abs(grad_var_m_v - grad_var_m_cg)
                else:
                    diff_v_mv_ls = diff_v_mv_var = diff_mv_cg_ls = diff_mv_cg_var = None
                
                print(f"    Differences:")
                print(f"      |vanilla - m_step_cg|: ls={diff_v_cg_ls:.2e}, var={diff_v_cg_var:.2e}")
                if diff_v_mv_ls is not None:
                    print(f"      |vanilla - m_step_vanilla|: ls={diff_v_mv_ls:.2e}, var={diff_v_mv_var:.2e}")
                    print(f"      |m_step_vanilla - m_step_cg|: ls={diff_mv_cg_ls:.2e}, var={diff_mv_cg_var:.2e}")
                
                results[d][jitter] = {
                    'vanilla': (grad_ls_v, grad_var_v),
                    'm_step_cg': (grad_ls_m_cg, grad_var_m_cg),
                    'm_step_vanilla': (grad_ls_m_v, grad_var_m_v) if grad_ls_m_v is not None else None,
                    'diff_v_cg': (diff_v_cg_ls, diff_v_cg_var),
                    'diff_v_mv': (diff_v_mv_ls, diff_v_mv_var) if diff_v_mv_ls is not None else None,
                    'diff_mv_cg': (diff_mv_cg_ls, diff_mv_cg_var) if diff_mv_cg_ls is not None else None
                }
                
            except Exception as e:
                print(f"    Error with jitter {jitter:.1e}: {e}")
                results[d][jitter] = None
    
    return results

def analyze_results(results):
    """Analyze the jitter alignment results"""
    
    print(f"\n{'='*80}")
    print("JITTER ALIGNMENT ANALYSIS")
    print("="*80)
    
    for d in [1, 2]:
        if d not in results:
            continue
            
        print(f"\n{d}D Case - Optimal Jitter Analysis:")
        
        best_jitter = None
        best_diff = float('inf')
        
        valid_results = [(jitter, res) for jitter, res in results[d].items() if res is not None]
        
        if not valid_results:
            print("  No valid results")
            continue
        
        print(f"  {'Jitter':<10} {'|V-CG| LS':<12} {'|V-CG| Var':<12} {'|V-MV| LS':<12} {'|V-MV| Var':<12}")
        print(f"  {'-'*60}")
        
        for jitter, res in valid_results:
            diff_v_cg = res['diff_v_cg']
            diff_v_mv = res['diff_v_mv']
            
            print(f"  {jitter:<10.1e} {diff_v_cg[0]:<12.2e} {diff_v_cg[1]:<12.2e}", end="")
            
            if diff_v_mv is not None:
                print(f" {diff_v_mv[0]:<12.2e} {diff_v_mv[1]:<12.2e}")
                total_diff = diff_v_mv[0] + diff_v_mv[1]  # Use vanilla mode for best comparison
            else:
                print(f" {'N/A':<12} {'N/A':<12}")
                total_diff = diff_v_cg[0] + diff_v_cg[1]  # Fallback to CG mode
            
            if total_diff < best_diff:
                best_diff = total_diff
                best_jitter = jitter
        
        print(f"\n  Best jitter for {d}D: {best_jitter:.1e} (total diff: {best_diff:.2e})")

def test_hutchinson_probe_effects():
    """Test if remaining differences are due to Hutchinson probe stochasticity"""
    
    print(f"\n{'='*80}")
    print("HUTCHINSON PROBE STOCHASTICITY TEST")
    print("="*80)
    
    # Use 2D case with best jitter
    d = 2
    setup_dict = setup_test_case(d, n=30, seed=42)
    
    q = qVariationalParams(setup_dict['n'], device=setup_dict['device'])
    q.Delta = torch.ones_like(q.Delta) * 0.1
    m = torch.zeros(setup_dict['n'], dtype=setup_dict['rdtype'], device=setup_dict['device'])
    
    # Use optimal jitter (you'll need to determine this from previous test)
    optimal_jitter = 1e-8  # Placeholder - use result from jitter alignment test
    
    print(f"Testing Hutchinson probe variability with jitter={optimal_jitter:.1e}")
    
    # Compute vanilla gradient (deterministic)
    grad_ls_v, grad_var_v, _, _, _, _, _, _ = \
        compute_vanilla_gradient_aligned(setup_dict, m, q, jitter=optimal_jitter)
    
    print(f"Vanilla (deterministic): ls={grad_ls_v:.6f}, var={grad_var_v:.6f}")
    
    # Compute m-step multiple times with different random seeds (stochastic due to probes)
    print(f"\nM-step with different random seeds:")
    
    mstep_results = []
    for seed in [42, 123, 456, 789, 101112]:
        torch.manual_seed(seed)  # This affects Hutchinson probes
        
        grad_ls_m, grad_var_m, _, _, _ = \
            compute_mstep_gradient_aligned(setup_dict, q, m, optimal_jitter, 1e-14, vanilla_mode=False)
        
        mstep_results.append((grad_ls_m, grad_var_m))
        
        diff_ls = abs(grad_ls_v - grad_ls_m)
        diff_var = abs(grad_var_v - grad_var_m)
        
        print(f"  Seed {seed}: ls={grad_ls_m:.6f}, var={grad_var_m:.6f} | diff: ls={diff_ls:.2e}, var={diff_var:.2e}")
    
    # Analyze variability
    ls_values = [res[0] for res in mstep_results]
    var_values = [res[1] for res in mstep_results]
    
    ls_std = np.std(ls_values)
    var_std = np.std(var_values)
    ls_mean = np.mean(ls_values)
    var_mean = np.mean(var_values)
    
    print(f"\nM-step variability analysis:")
    print(f"  Lengthscale: mean={ls_mean:.6f}, std={ls_std:.2e}")
    print(f"  Variance: mean={var_mean:.6f}, std={var_std:.2e}")
    
    # Compare with vanilla
    ls_bias = abs(ls_mean - grad_ls_v)
    var_bias = abs(var_mean - grad_var_v)
    
    print(f"\nBias vs vanilla (after accounting for stochasticity):")
    print(f"  Lengthscale bias: {ls_bias:.2e} (vs std: {ls_std:.2e})")
    print(f"  Variance bias: {var_bias:.2e} (vs std: {var_std:.2e})")
    
    if ls_bias < 3 * ls_std and var_bias < 3 * var_std:
        print(f"\n✅ SUCCESS: Bias is within 3σ of Hutchinson probe variability!")
        print(f"   The remaining differences are likely just due to probe stochasticity.")
    else:
        print(f"\n⚠️  Bias still larger than expected from Hutchinson probe variability.")
        print(f"   There may be additional systematic differences to investigate.")

def main():
    """Main function"""
    print("COMPREHENSIVE JITTER ALIGNMENT AND BIAS DIAGNOSIS")
    print("Goal: Make compute_vanilla_gradient and m_step agree up to Hutchinson probe stochasticity")
    
    # Test jitter alignment
    results = test_jitter_alignment()
    
    # Analyze results
    analyze_results(results)
    
    # Test Hutchinson probe effects
    test_hutchinson_probe_effects()
    
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("""
1. JITTER ALIGNMENT:
   - Use the optimal jitter value identified above in both methods
   - This should be the primary fix for the bias

2. SOLVER CONSISTENCY:
   - For small problems: Use vanilla mode in m_step (direct solve)
   - For large problems: Use very tight CG tolerance (1e-14)

3. REMAINING DIFFERENCES:
   - Should be within 3σ of Hutchinson probe variability
   - If not, investigate spectral approximation accuracy

4. IMPLEMENTATION:
   - Replace adaptive jitter with fixed optimal value
   - Test on your actual problem to verify the fix
    """)

if __name__ == "__main__":
    main()








