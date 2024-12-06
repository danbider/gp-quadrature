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
            dtype = torch.complex128 # for the FFTs in the complex case
        
        print(f"Using {dtype} precision for CG matrix-vector products!")
        print(f"Using {device} device!")
        self.b = b.to(dtype=dtype, device=device)
        self.x0 = x0.to(dtype=dtype, device=device)
        self.tol = tol
        self.div_eps = 1e-16 # small value to prevent division by zero
        # Set max_iter to either the provided value or 2n as a default
        self.max_iter = max_iter if max_iter is not None else 2 * len(self.b)
        self.early_stopping = early_stopping
        self.iters_completed = None
        self.solution_history = []

    def solve(self):
        """Solve the linear system using Conjugate Gradients.
        Returns:
            torch.Tensor: Solution to the linear system
        """
        x = self.x0  # initial guess
        r = self.b - self.A_apply_function(x)  # initial residual
        p = r  # initial search direction
        
        for i in range(self.max_iter):
            # TODO: more comments on ops
            Ap = self.A_apply_function(p)  # appears twice, in alpha and next residual
            r_norm = torch.conj(r.T) @ r
            
            # Check convergence using the user-specified tolerance
            # using absolute to avoid complex number issues, also sqrt beause dot product is norm squared
            if self.early_stopping and torch.sqrt(abs(r_norm)) < self.tol:
                print(f"Converged in {i} iterations: Residual norm {torch.sqrt(abs(r_norm))} below tolerance {self.tol}")
                break

            alpha_k = r_norm / (torch.conj(p.T) @ Ap + self.div_eps)
            x = x + alpha_k * p
            self.solution_history.append(x)
            r_next = r - alpha_k * Ap

            # update search direction
            next_r_norm = torch.conj(r_next.T) @ r_next
            beta_k = next_r_norm / (r_norm + self.div_eps)  # magnitude of next residual over current residual
            p = r_next + beta_k * p  # update search direction
            r = r_next

        self.iters_completed = i
        print(f"Completed {i} iterations, final norm {r_norm}, residual norm: {torch.sqrt(abs(r_norm))}")

        return x