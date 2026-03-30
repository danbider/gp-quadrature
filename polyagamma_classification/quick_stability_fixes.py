"""
Quick numerical stability fixes for small ws values

Apply these changes to your D2_Fstar_Kinv_z function and m_step function
"""

# Fix 1: Replace the jitter calculation in D2_Fstar_Kinv_z
def get_adaptive_jitter(ws):
    """Calculate adaptive jitter based on ws magnitude and condition number"""
    ws_magnitude = ws.abs().max()
    ws_min = ws.abs().min()
    
    if ws_min > 0:
        condition_estimate = ws_magnitude / ws_min
        # Scale jitter based on problem magnitude and condition number
        jitter_val = max(1e-12, ws_magnitude * 1e-8 * torch.sqrt(condition_estimate))
    else:
        jitter_val = 1e-8
    
    print(f"Adaptive jitter: ws_range=[{ws_min:.2e}, {ws_magnitude:.2e}], jitter={jitter_val:.2e}")
    return jitter_val

# Fix 2: Improved preconditioner for small ws
def get_stable_preconditioner(ws, jitter_val):
    """Create a numerically stable preconditioner for small ws"""
    ws2 = ws.pow(2)
    # Use the larger of jitter or a fraction of the diagonal
    diag_estimate = ws2 + jitter_val
    # Add small epsilon to prevent division by zero
    return lambda x: x / (diag_estimate + 1e-15)

# Fix 3: Robust vanilla path solver
def robust_vanilla_solve(A_dense, rhs):
    """Solve linear system with multiple fallback strategies"""
    try:
        # Try Cholesky first
        L = torch.linalg.cholesky(A_dense)
        solve = torch.cholesky_solve(rhs.T, L).T
        return solve
    except RuntimeError as e:
        print(f"Cholesky failed: {e}. Trying LU decomposition...")
        try:
            # Fallback to LU with partial pivoting
            LU, pivots = torch.linalg.lu_factor(A_dense)
            solve = torch.linalg.lu_solve(LU, pivots, rhs.T).T
            return solve
        except RuntimeError as e2:
            print(f"LU failed: {e2}. Using pseudo-inverse...")
            # Last resort: pseudo-inverse
            solve = torch.linalg.pinv(A_dense) @ rhs.T
            return solve.T

# Fix 4: System scaling for vanilla path
def scale_system_for_stability(ws, jitter_val):
    """Scale the system to improve numerical conditioning"""
    ws_magnitude = ws.abs().max()
    ws_scaled = ws / ws_magnitude  # Normalize to prevent extreme scaling
    scale_factor = ws_magnitude**2
    scaled_jitter = jitter_val / scale_factor
    
    return ws_scaled, scale_factor, scaled_jitter

"""
INSTRUCTIONS FOR APPLYING FIXES:

1. In your D2_Fstar_Kinv_z function, replace:
   
   OLD:
   jitter_val = 1e-8
   
   NEW:
   jitter_val = get_adaptive_jitter(ws)

2. Replace the preconditioner:
   
   OLD:
   M_inv_apply = lambda x: x/(10*ws**2)
   
   NEW:
   M_inv_apply = get_stable_preconditioner(ws, jitter_val)

3. In the vanilla path, replace the matrix construction and solve:
   
   OLD:
   A_dense = ws[:, None] * torch.stack([...]) * ws[None, :] + jitter_val * torch.eye(...)
   solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
   
   NEW:
   ws_scaled, scale_factor, scaled_jitter = scale_system_for_stability(ws, jitter_val)
   A_scaled = ws_scaled[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                   for i in range(ws.numel())], dim=1) * ws_scaled[None, :] + scaled_jitter * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
   rhs_scaled = rhs / ws_magnitude  
   solve_scaled = robust_vanilla_solve(A_scaled, rhs_scaled)
   solve = solve_scaled / ws_magnitude  # Scale back

4. For the m_step function, ensure you're using float64:
   
   Make sure you're using:
   dtype = torch.float64  # or torch.complex128 for complex
   
   And when creating matrices, ensure proper dtype:
   A_dense = A_dense.to(dtype=dtype)
"""

print("Quick stability fixes loaded. Apply the changes above to your notebook.")
