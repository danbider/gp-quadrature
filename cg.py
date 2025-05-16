import torch
import numpy as np


class ConjugateGradients:
    """
    Conjugate Gradients solver for linear systems Ax = b with symmetric positive definite A.
    Args:
        A (torch.Tensor): Symmetric positive definite matrix
        b (torch.Tensor): Right-hand side vector
        x0 (torch.Tensor): Initial guess
        tol (float): Tolerance for early stopping
        max_iter (int): Maximum number of iterations
        early_stopping (bool): Whether to stop early if residual is below tolerance

    Returns:
        x (torch.Tensor): Solution to the linear system
    """

    def __init__(self, A_apply_function, b, x0, tol=1e-6, early_stopping=True, max_iter=None):
        # Get device from input tensors
        device = b.device

        if isinstance(A_apply_function, torch.Tensor):
            # if A_apply_function is a matrix, we use it directly
            self.A_apply_function = lambda x: A_apply_function @ x
        elif callable(A_apply_function):
            # if A_apply_function is a function, we use it as is
            self.A_apply_function = A_apply_function
        else:
            raise ValueError("A_apply_function must be a torch.Tensor or a callable")
        if isinstance(A_apply_function, torch.Tensor):
            dtype = A_apply_function.dtype
        else:
            dtype = x0.dtype
            # dtype = torch.complex128 # for the FFTs in the complex case
        
        self.b = b.to(dtype=dtype, device=device)
        self.x0 = x0.to(dtype=dtype, device=device)
        self.tol = tol
        self.div_eps = 1e-16 # small value to prevent division by zero
        # Set max_iter to either the provided value or 2n as a default
        self.max_iter = max_iter if max_iter is not None else 2 * len(self.b)
        self.early_stopping = early_stopping
        self.iters_completed = None
        self.solution_history = []
        # print("="*5 + " CG Solver initialized " + "="*5)
        # print(f"Using {dtype} precision for CG matrix-vector products!")
        # print(f"Using {device} device!")
        # print(f"System size: {len(self.b)}, Max iterations: {self.max_iter}")
        # if self.early_stopping:
            # print(f"Early stopping enabled, tolerance: {self.tol}")
        # else:
            # print("Early stopping disabled")
        # print("="*25)
    def solve(self):
        """Solve the linear system using Conjugate Gradients.
        Returns:
            torch.Tensor: Solution to the linear system
        """
        x = self.x0  # initial guess
        r = self.b - self.A_apply_function(x)  # initial residual
        p = r  # initial search direction
        
        for i in range(self.max_iter):
            # Compute matrix-vector product once to reuse
            Ap = self.A_apply_function(p)
            
            # Compute inner product based on tensor dimensions
            r_norm = self._inner_product(r, r)
            
            # Check convergence using the user-specified tolerance
            # using absolute to avoid complex number issues, also sqrt because dot product is norm squared
            if self.early_stopping and torch.sqrt(abs(r_norm)) < self.tol:
                break

            # Compute step size
            alpha_k = r_norm / (self._inner_product(p, Ap) + self.div_eps)
            
            # Update solution
            x = x + alpha_k * p
            self.solution_history.append(x)
            
            # Update residual
            r_next = r - alpha_k * Ap
            
            # Compute next residual norm
            next_r_norm = self._inner_product(r_next, r_next)
            
            # Update search direction
            beta_k = next_r_norm / (r_norm + self.div_eps)  # magnitude of next residual over current residual
            p = r_next + beta_k * p  # update search direction
            r = r_next

        self.iters_completed = i
        return x
        
    def _inner_product(self, a, b):
        """Helper method to compute inner product based on tensor dimensions."""
        if a.dim() == 1:
            return torch.dot(torch.conj(a), b)
        elif a.dim() == 2:
            return torch.conj(a.T) @ b
        else:
            # For higher dimensional tensors, use mT (matrix transpose)
            return torch.conj(a.mT) @ b
class BatchConjugateGradients:
    """
    Batched Conjugate Gradients solver for a set of linear systems A x = b,
    where A is symmetric positive definite and b has shape (B, n).

    Args:
        A_apply_function (callable): Function that accepts an input tensor of shape (B, n)
                                     and returns A @ x (shape (B, n)).
        b (torch.Tensor): Right-hand side tensor of shape (B, n).
        x0 (torch.Tensor): Initial guess of shape (B, n).
        tol (float): Convergence tolerance.
        early_stopping (bool): Whether to stop early if the residual norm is below tol.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        x (torch.Tensor): Solution tensor of shape (B, n).
    """
    def __init__(self, A_apply_function, b, x0, tol=1e-6, early_stopping=True, max_iter=None):
        self.A_apply_function = A_apply_function
        self.b = b
        self.x0 = x0
        self.tol = tol
        self.early_stopping = early_stopping
        B, n = b.shape
        self.max_iter = max_iter if max_iter is not None else 2 * n
        self.div_eps = 1e-16

    # def solve(self):
    #     x = self.x0.clone()  # (B, n)
    #     r = self.b - self.A_apply_function(x)  # (B, n)
    #     p = r.clone()  # (B, n)
    #     # Compute squared norms for each system: shape (B,)
    #     r_norm = torch.sum(torch.conj(r) * r, dim=1)
        
    #     for i in range(self.max_iter):
    #         Ap = self.A_apply_function(p)  # (B, n)
    #         # Compute inner products: (B,)
    #         pAp = torch.sum(torch.conj(p) * Ap, dim=1) + self.div_eps
    #         alpha = r_norm / pAp  # (B,)
    #         # Update: unsqueeze alpha to multiply elementwise.
    #         x = x + alpha.unsqueeze(1) * p
    #         r_new = r - alpha.unsqueeze(1) * Ap
    #         r_new_norm = torch.sum(torch.conj(r_new) * r_new, dim=1)
            
    #         # Early stopping can be implemented for each system if desired.
    #         if self.early_stopping and torch.all(r_new_norm.real < self.tol**2):
    #             # print(f"Batch CG early stopping at iter {i}") # Optional: Add print
    #             break
            
    #         beta = r_new_norm / (r_norm + self.div_eps)  # (B,)
    #         p = r_new + beta.unsqueeze(1) * p
    #         r = r_new
    #         r_norm = r_new_norm
    #     return x
    def solve(self):
        """
        Batched Conjugate‑Gradient solver with per‑system early stopping
        and an optional left pre‑conditioner `self.M_inv_apply`.

        Returns
        -------
        x : torch.Tensor  # shape (B, n)
        """
        with torch.no_grad():
            # ---------- initial state -----------------------------------------
            x = self.x0.clone()                                # (B, n)
            r = self.b - self.A_apply_function(x)

            # Pre‑conditioner (identity if none supplied)
            if getattr(self, "M_inv_apply", None) is not None:
                z = self.M_inv_apply(r)
            else:
                z = r

            p         = z.clone()
            r_dot_z   = torch.sum(r.conj() * z, dim=1).real    # (B,)
            r0_norm   = torch.sqrt(r_dot_z)                    # (B,)

            active = torch.ones_like(r0_norm, dtype=torch.bool)
            max_iter = self.max_iter if self.max_iter is not None else 100

            # ---------- CG loop -----------------------------------------------
            for _ in range(max_iter):
                idx = torch.where(active)[0]                  # integer indices
                if idx.numel() == 0:
                    break

                Ap   = self.A_apply_function(p[idx])          # (k, n)
                pAp  = torch.sum(p[idx].conj() * Ap, dim=1).real + self.div_eps
                alpha = r_dot_z[idx] / pAp

                x[idx] += alpha.unsqueeze(1) * p[idx]
                r[idx] -= alpha.unsqueeze(1) * Ap

                # Pre‑condition new residual
                if getattr(self, "M_inv_apply", None) is not None:
                    z_new = self.M_inv_apply(r[idx])
                else:
                    z_new = r[idx]

                r_dot_z_new = torch.sum(r[idx].conj() * z_new, dim=1).real

                # ---- update search direction BEFORE shrinking the mask -------
                beta       = r_dot_z_new / (r_dot_z[idx] + self.div_eps)
                p[idx]     = z_new + beta.unsqueeze(1) * p[idx]
                r_dot_z[idx] = r_dot_z_new

                # ---- convergence check (relative residual) -------------------
                converged = (torch.sqrt(r_dot_z_new) /
                            (r0_norm[idx] + self.div_eps)) < self.tol

                active[idx] = ~converged                     # no aliasing problem

        return x

