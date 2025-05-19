import torch
import numpy as np


class ConjugateGradients:
    """
    Unified Conjugate Gradients solver that handles both single and batched systems.
    For batched systems, per-system early stopping is supported.
    
    Args:
        A_apply_function (callable): Function to apply matrix A to a vector or batch of vectors.
            - For single system: accepts (n,) tensor and returns (n,) tensor
            - For batched systems: accepts (B, n) tensor and returns (B, n) tensor
        b (torch.Tensor): Right-hand side vector(s).
            - For single system: shape (n,)
            - For batched systems: shape (B, n)
        x0 (torch.Tensor): Initial guess.
            - For single system: shape (n,)
            - For batched systems: shape (B, n)
        tol (float): Convergence tolerance.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2*n.
        early_stopping (bool): Whether to stop early if residual is below tolerance.
        M_inv_apply (callable, optional): Preconditioner function. Uses identity if None.

    Returns:
        x (torch.Tensor): Solution to the linear system(s).
            - For single system: shape (n,)
            - For batched systems: shape (B, n)
    """
    def __init__(self, A_apply_function, b, x0, tol=1e-6, max_iter=None, early_stopping=True, M_inv_apply=None):
        # Get device and dtype from input tensors
        self.device = b.device
        
        # Store the matrix-vector product function
        if isinstance(A_apply_function, torch.Tensor):
            self.A_apply_function = lambda x: A_apply_function @ x
        elif callable(A_apply_function):
            self.A_apply_function = A_apply_function
        else:
            raise ValueError("A_apply_function must be a torch.Tensor or a callable")
        
        # Determine dtype
        if isinstance(A_apply_function, torch.Tensor):
            self.dtype = A_apply_function.dtype
        else:
            self.dtype = x0.dtype
        
        # Store inputs with appropriate dtype
        self.b = b.to(dtype=self.dtype, device=self.device)
        self.x0 = x0.to(dtype=self.dtype, device=self.device)
        
        # Determine if we're solving a batch of systems
        self.is_batched = (b.dim() > 1)
        
        # Configuration parameters
        self.tol = tol
        self.div_eps = 1e-16  # Small value to prevent division by zero
        
        # Set max iterations based on problem size
        if max_iter is not None:
            self.max_iter = max_iter
        else:
            # For batched systems, use the second dimension (n)
            system_size = self.b.shape[1] if self.is_batched else len(self.b)
            self.max_iter = 2 * system_size
            
        self.early_stopping = early_stopping
        self.M_inv_apply = M_inv_apply  # Preconditioner function (optional)
        
        # For tracking progress
        self.iters_completed = 0
        
    def solve(self):
        """
        Solve the linear system(s) using Conjugate Gradients.
        Automatically handles single or batched systems.
        
        Returns:
            x (torch.Tensor): Solution to the system(s)
        """
        if self.is_batched:
            return self._solve_batched()
        else:
            return self._solve_single()
    
    def _solve_single(self):
        """
        Solve a single linear system using Conjugate Gradients.
        
        Returns:
            x (torch.Tensor): Solution vector of shape (n,)
        """
        with torch.no_grad():
            x = self.x0.clone()  # Initial guess
            r = self.b - self.A_apply_function(x)  # Initial residual
            
            # Apply preconditioner if provided
            if self.M_inv_apply is not None:
                z = self.M_inv_apply(r)
            else:
                z = r.clone()
                
            p = z.clone()  # Initial search direction
            r_dot_z = self._inner_product(r, z).real  # Inner product for residual
            r0_norm = torch.sqrt(r_dot_z)  # Initial residual norm
            
            # Main CG loop
            for i in range(self.max_iter):
                # Compute matrix-vector product
                Ap = self.A_apply_function(p)
                
                # Compute step size
                pAp = self._inner_product(p, Ap).real + self.div_eps
                alpha = r_dot_z / pAp
                
                # Update solution
                x = x + alpha * p
                
                # Update residual
                r = r - alpha * Ap
                
                # Check convergence
                if self.early_stopping and torch.sqrt(r_dot_z) / (r0_norm + self.div_eps) < self.tol:
                    break
                
                # Apply preconditioner to new residual
                if self.M_inv_apply is not None:
                    z_new = self.M_inv_apply(r)
                else:
                    z_new = r
                
                # Compute new inner product
                r_dot_z_new = self._inner_product(r, z_new).real
                
                # Update search direction
                beta = r_dot_z_new / (r_dot_z + self.div_eps)
                p = z_new + beta * p
                
                # Update for next iteration
                r_dot_z = r_dot_z_new
                z = z_new
            
            self.iters_completed = i + 1
            return x
    
    def _solve_batched(self):
        """
        Solve a batch of linear systems using Conjugate Gradients with per-system early stopping.
        
        Returns:
            x (torch.Tensor): Solution tensor of shape (B, n)
        """
        with torch.no_grad():
            # Initial state
            x = self.x0.clone()  # (B, n)
            r = self.b - self.A_apply_function(x)  # (B, n)
            
            # Apply preconditioner if provided
            if self.M_inv_apply is not None:
                z = self.M_inv_apply(r)
            else:
                z = r.clone()
                
            p = z.clone()  # (B, n)
            
            # Compute initial inner products and norms
            r_dot_z = torch.sum(r.conj() * z, dim=1).real  # (B,)
            r0_norm = torch.sqrt(r_dot_z)  # (B,)
            
            # Mask to track which systems are still active
            active = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
            
            # Main CG loop
            for i in range(self.max_iter):
                # Get indices of active systems
                idx = torch.where(active)[0]
                
                # If all systems have converged, exit
                if idx.numel() == 0:
                    break
                
                # Compute matrix-vector product for active systems
                Ap = self.A_apply_function(p[idx])  # (k, n)
                
                # Compute step sizes
                pAp = torch.sum(p[idx].conj() * Ap, dim=1).real + self.div_eps  # (k,)
                alpha = r_dot_z[idx] / pAp  # (k,)
                
                # Update solutions and residuals for active systems
                x[idx] += alpha.unsqueeze(1) * p[idx]
                r[idx] -= alpha.unsqueeze(1) * Ap
                
                # Apply preconditioner to new residuals
                if self.M_inv_apply is not None:
                    z_new = self.M_inv_apply(r[idx])
                else:
                    z_new = r[idx]
                
                # Compute new inner products
                r_dot_z_new = torch.sum(r[idx].conj() * z_new, dim=1).real  # (k,)
                
                # Update search directions BEFORE updating the mask
                beta = r_dot_z_new / (r_dot_z[idx] + self.div_eps)  # (k,)
                p[idx] = z_new + beta.unsqueeze(1) * p[idx]
                
                # Update inner products for next iteration
                r_dot_z[idx] = r_dot_z_new
                
                # Check convergence based on relative residual
                if self.early_stopping:
                    converged = (torch.sqrt(r_dot_z_new) / (r0_norm[idx] + self.div_eps)) < self.tol
                    # Update the mask of active systems
                    active[idx[converged]] = False
            
            self.iters_completed = i + 1
            return x
    
    def _inner_product(self, a, b):
        """
        Compute inner product based on tensor dimensions.
        
        Args:
            a, b (torch.Tensor): Vectors or matrices
            
        Returns:
            torch.Tensor: Inner product result
        """
        if a.dim() == 1:
            return torch.dot(torch.conj(a), b)
        elif a.dim() == 2:
            return torch.sum(a.conj() * b, dim=1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {a.dim()}")


# # Retain original classes for backward compatibility
# class ConjugateGradients(UnifiedConjugateGradients):
#     """Legacy ConjugateGradients class (single system only)."""
#     def __init__(self, A_apply_function, b, x0, tol=1e-6, early_stopping=True, max_iter=None):
#         super().__init__(A_apply_function, b, x0, tol, max_iter, early_stopping)
        
#     def solve(self):
#         return self._solve_single()


# class BatchConjugateGradients(UnifiedConjugateGradients):
#     """Legacy BatchConjugateGradients class."""
#     def __init__(self, A_apply_function, b, x0, tol=1e-6, early_stopping=True, max_iter=None):
#         super().__init__(A_apply_function, b, x0, tol, max_iter, early_stopping)
        
#     def solve(self):
#         return self._solve_batched()

