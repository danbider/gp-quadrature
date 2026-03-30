"""
Numerical stability fixes for Polyagamma GP with small spectral weights (ws)

Key improvements:
1. Adaptive jitter scaling based on ws magnitude and condition number
2. Improved vanilla path with system scaling and robust fallbacks
3. Better preconditioner that handles small ws values
4. Robust linear system solvers with multiple fallback options
"""

import torch
import numpy as np

def improved_D2_Fstar_Kinv_z(z, toeplitz, ws, fadj_batched, cg_tol=1e-10, vanilla=False):
    """
    Numerically stable version of D2_Fstar_Kinv_z for small ws values
    
    Key improvements:
    - Adaptive jitter scaling based on problem magnitude
    - System scaling in vanilla path to improve conditioning
    - Robust fallbacks for linear system solving
    - Better preconditioner design
    """
    
    # Adaptive jitter and scaling for numerical stability
    ws_magnitude = ws.abs().max()
    ws_min = ws.abs().min()
    
    # Better condition number estimation
    if ws_min > 0:
        condition_estimate = ws_magnitude / ws_min
        # Scale jitter based on problem magnitude and condition number
        jitter_val = max(1e-12, ws_magnitude * 1e-8 * torch.sqrt(condition_estimate))
    else:
        jitter_val = 1e-8
    
    print(f"ws range: [{ws_min:.2e}, {ws_magnitude:.2e}], condition: {condition_estimate:.2e}, jitter: {jitter_val:.2e}")
    
    ws2 = ws.pow(2)  

    A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
    if z.ndim == 1:
        z = z.unsqueeze(0)
    fadj_z = fadj_batched(z)
    rhs = ws*fadj_z

    x0 = torch.zeros_like(rhs)
    
    # Improved preconditioner that handles small ws better
    # Use diagonal scaling based on actual operator magnitude
    diag_estimate = ws2 + jitter_val
    M_inv_apply = lambda x: x / (diag_estimate + 1e-15)  # Add tiny epsilon to prevent division by zero
    
    if vanilla:
        # Improved vanilla path with better numerical conditioning
        # Scale the system to improve conditioning
        ws_scaled = ws / ws_magnitude  # Normalize to prevent extreme scaling
        scale_factor = ws_magnitude**2
        
        # Build the scaled system matrix
        A_scaled = ws_scaled[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                        for i in range(ws.numel())], dim=1) * ws_scaled[None, :] \
            + (jitter_val / scale_factor) * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
        
        # Scale RHS accordingly
        rhs_scaled = rhs / ws_magnitude
        
        try:
            # Try Cholesky first
            L = torch.linalg.cholesky(A_scaled)
            solve_scaled = torch.cholesky_solve(rhs_scaled.T, L).T
            print("✓ Cholesky decomposition succeeded")
        except RuntimeError as e:
            print(f"⚠ Cholesky decomposition failed: {e}")
            print("Falling back to LU decomposition with partial pivoting")
            
            # Fallback to more robust LU decomposition
            try:
                LU, pivots = torch.linalg.lu_factor(A_scaled)
                solve_scaled = torch.linalg.lu_solve(LU, pivots, rhs_scaled.T).T
                print("✓ LU decomposition succeeded")
            except RuntimeError as e2:
                print(f"⚠ LU decomposition also failed: {e2}")
                print("Using pseudo-inverse as last resort")
                solve_scaled = torch.linalg.pinv(A_scaled) @ rhs_scaled.T
                solve_scaled = solve_scaled.T
                print("✓ Pseudo-inverse used")
        
        # Scale solution back
        solve = solve_scaled / ws_magnitude
        
    else:
        # Improved CG with better preconditioning
        from utils.cg import ConjugateGradients  # Assuming this is where CG is imported from
        solve = ConjugateGradients(
            A_apply, rhs, x0,
            tol=cg_tol, early_stopping=True, 
            M_inv_apply=M_inv_apply
        ).solve()

    return solve/ws, fadj_z


def improved_vanilla_path_alternative(ws, toeplitz, rhs, jitter_val):
    """
    Alternative vanilla path implementation with even better numerical stability
    
    Uses:
    1. Symmetric positive definite structure preservation
    2. Regularized Cholesky with iterative refinement
    3. Multiple fallback strategies
    """
    
    # Method 1: Regularized Cholesky with iterative jitter increase
    ws_magnitude = ws.abs().max()
    current_jitter = jitter_val
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            # Build system matrix with current jitter
            A = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                          for i in range(ws.numel())], dim=1) * ws[None, :] \
                + current_jitter * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
            
            # Try Cholesky
            L = torch.linalg.cholesky(A)
            solve = torch.cholesky_solve(rhs.T, L).T
            
            if attempt > 0:
                print(f"✓ Cholesky succeeded with jitter={current_jitter:.2e} (attempt {attempt+1})")
            return solve
            
        except RuntimeError:
            current_jitter *= 10
            if attempt == max_attempts - 1:
                print(f"⚠ Cholesky failed even with jitter={current_jitter:.2e}")
                break
    
    # Method 2: Eigenvalue regularization
    try:
        A = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                      for i in range(ws.numel())], dim=1) * ws[None, :]
        
        # Eigenvalue decomposition and regularization
        eigenvals, eigenvecs = torch.linalg.eigh(A)
        min_eigenval = eigenvals.min()
        reg_eigenvals = torch.clamp(eigenvals, min=max(min_eigenval, jitter_val))
        
        A_reg = eigenvecs @ torch.diag(reg_eigenvals) @ eigenvecs.T
        solve = torch.linalg.solve(A_reg, rhs.T).T
        
        print("✓ Eigenvalue regularization succeeded")
        return solve
        
    except RuntimeError as e:
        print(f"⚠ Eigenvalue regularization failed: {e}")
    
    # Method 3: SVD-based pseudo-inverse (most robust)
    try:
        A = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                      for i in range(ws.numel())], dim=1) * ws[None, :]
        
        U, S, Vh = torch.linalg.svd(A)
        # Regularize singular values
        S_reg = torch.clamp(S, min=jitter_val)
        A_pinv = Vh.T @ torch.diag(1.0 / S_reg) @ U.T
        solve = (A_pinv @ rhs.T).T
        
        print("✓ SVD pseudo-inverse succeeded")
        return solve
        
    except RuntimeError as e:
        print(f"⚠ All methods failed. Final error: {e}")
        raise e


def adaptive_preconditioner(ws, method='diagonal_plus'):
    """
    Create an adaptive preconditioner for small ws values
    
    Methods:
    - 'diagonal_plus': Diagonal + small identity
    - 'scaled_diagonal': Scale-aware diagonal
    - 'approximate_inverse': Approximate operator inverse
    """
    ws2 = ws.pow(2)
    ws_magnitude = ws.abs().max()
    
    if method == 'diagonal_plus':
        # Simple diagonal with regularization
        diag_vals = ws2 + max(1e-15, ws_magnitude * 1e-10)
        return lambda x: x / diag_vals
        
    elif method == 'scaled_diagonal':
        # Scale-aware diagonal preconditioning
        scale = ws_magnitude
        normalized_diag = (ws2 / scale) + 1e-12
        return lambda x: x / (normalized_diag * scale)
        
    elif method == 'approximate_inverse':
        # More sophisticated approximate inverse
        # For operator A = ws * T * ws + jitter*I
        # Approximate A^{-1} ≈ (1/ws²) * T^{-1} where T^{-1} ≈ I for well-conditioned T
        inv_diag = 1.0 / (ws2 + max(1e-15, ws_magnitude * 1e-8))
        return lambda x: x * inv_diag
        
    else:
        raise ValueError(f"Unknown preconditioner method: {method}")


# Example usage and testing functions
def test_numerical_stability():
    """Test the numerical stability improvements"""
    
    # Create test case with very small ws values
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Simulate very small spectral weights
    M = 100
    ws_small = torch.logspace(-8, -4, M, device=device, dtype=dtype)  # Very small values
    
    print(f"Testing with ws range: [{ws_small.min():.2e}, {ws_small.max():.2e}]")
    print(f"Condition number estimate: {(ws_small.max() / ws_small.min()):.2e}")
    
    # Create simple test toeplitz operator
    def simple_toeplitz(x):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(torch.ones_like(x))).real
    
    # Test vector
    z_test = torch.randn(50, device=device, dtype=dtype)
    
    # Simple fadj_batched for testing
    def simple_fadj_batched(z):
        return torch.randn(z.shape[0], M, device=device, dtype=dtype)
    
    try:
        result, _ = improved_D2_Fstar_Kinv_z(
            z_test, simple_toeplitz, ws_small, simple_fadj_batched, 
            vanilla=True, cg_tol=1e-10
        )
        print("✓ Improved function succeeded")
        print(f"Result range: [{result.min():.2e}, {result.max():.2e}]")
        return True
    except Exception as e:
        print(f"✗ Improved function failed: {e}")
        return False


if __name__ == "__main__":
    test_numerical_stability()
