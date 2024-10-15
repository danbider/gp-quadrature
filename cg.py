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

    def __init__(self, A, b, x0, tol=1e-6, max_iter=10000, early_stopping=False):
        self.A = A
        self.b = b
        self.x0 = x0
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
        r = self.b - self.A @ x  # initial residual
        p = r  # initial search direction

        for i in range(self.max_iter):
            # TODO: more comments on ops
            Ap = self.A @ p  # appears twice, in alpha and next residual
            alpha_k = (r.T @ r) / (p.T @ Ap)  # step size
            x = x + alpha_k * p  # update solution
            self.solution_history.append(x)
            r_next = r - alpha_k * Ap
            if self.early_stopping and torch.norm(r_next) < self.tol:
                print(f"Converged in {i} iterations")
                break
            # update search direction
            beta_k = (r_next.T @ r_next) / (
                r.T @ r
            )  # magnitude of next residual over current residual
            p = r_next + beta_k * p  # update search direction
            r = r_next

        self.iters_completed = i

        return x