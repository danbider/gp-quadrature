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

    def __init__(self, A_apply_function, b, x0, tol=1e-6, max_iter=10000, early_stopping=False):
        if isinstance(A_apply_function, torch.Tensor):
            # if A_apply_function is a matrix, we use it directly
            self.A_apply_function = lambda x: A_apply_function @ x
        elif callable(A_apply_function):
            # if A_apply_function is a function, we use it as is
            self.A_apply_function = A_apply_function
        else:
            raise ValueError("A_apply_function must be a torch.Tensor or a callable")
        self.b = b.to(dtype=torch.complex128)
        self.x0 = x0.to(dtype=torch.complex128)
        self.tol = tol
        self.max_iter = max_iter
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

        # Add small value for numerical stability
        eps = 1e-16
        
        for i in range(self.max_iter):
            # TODO: more comments on ops
            Ap = self.A_apply_function(p)  # appears twice, in alpha and next residual
            r_norm = torch.conj(r.T) @ r

            if abs(r_norm) < eps:
                print(f"Residual norm too small, stopping at iteration {i}")
                break

            alpha_k = r_norm / (torch.conj(p.T) @ Ap + eps)
            x = x + alpha_k * p
            self.solution_history.append(x)
            r_next = r - alpha_k * Ap
            if self.early_stopping and torch.norm(r_next) < self.tol:
                print(f"Converged in {i} iterations")
                break
            # update search direction
            next_r_norm = torch.conj(r_next.T) @ r_next
            beta_k = next_r_norm / (r_norm + eps)  # magnitude of next residual over current residual
            p = r_next + beta_k * p  # update search direction
            r = r_next

        self.iters_completed = i

        return x