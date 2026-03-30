import sys
import torch
import torch.nn as nn
import math
from typing import List, Iterator, Tuple, Optional, Union
from torch.optim import Adam
from torch import vmap
import time

# Import required modules (assuming they exist in the project)
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature/kernels')
from kernels import SquaredExponential
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD, NUFFT
from cg import ConjugateGradients
from utils.kernels import get_xis


class qVariationalParams:
    """Variational parameters for Polya-Gamma augmentation"""
    def __init__(self, n: int, device: torch.device, dtype: torch.dtype = torch.float64):
        self.Delta = torch.ones(n, device=device, dtype=dtype) * 0.25


class PolyagammaGP:
    """
    Polya-Gamma Gaussian Process for binary classification.
    
    This class implements the Polya-Gamma augmentation scheme for GP classification
    using efficient NUFFT-based matrix-vector products and conjugate gradient solvers.
    """
    
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 kernel: nn.Module,
                 eps: float = 1e-4,
                 nufft_eps: float = 1e-8,
                 cg_tol: float = 1e-6,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        """
        Initialize the Polya-Gamma GP model.
        
        Args:
            x: Input locations, shape (n, d)
            y: Binary labels {0, 1}, shape (n,)
            kernel: Kernel object (e.g., SquaredExponential)
            eps: Spectral truncation tolerance
            nufft_eps: NUFFT tolerance
            cg_tol: Conjugate gradient tolerance for all CG solves
            device: Torch device
            dtype: Data type
        """
        self.dtype = dtype
        self.device = device or x.device
        self.eps = eps
        self.nufft_eps = nufft_eps
        self.cg_tol = cg_tol
        self.kernel = kernel
        
        # Ensure proper shapes and types
        self.x = x.to(device=self.device, dtype=dtype)
        self.y = y.to(device=self.device, dtype=dtype)
        if self.x.ndim == 1:
            self.x = self.x.unsqueeze(-1)
        
        self.n, self.d = self.x.shape
        
        # Initialize variational parameters
        self.q = qVariationalParams(self.n, self.device, dtype)
        
        # Setup spectral representation
        self._setup_spectral_representation()
        
        # Setup NUFFT operators
        self._setup_nufft()
        
    def _setup_spectral_representation(self):
        """Setup the spectral representation of the kernel"""
        # Get domain bounds
        x0 = self.x.min(dim=0).values
        x1 = self.x.max(dim=0).values
        domain_lengths = x1 - x0
        L = domain_lengths.max()
        
        # Get spectral points and weights
        xis_1d, self.h, self.mtot = get_xis(
            kernel_obj=self.kernel, 
            eps=self.eps, 
            L=L, 
            use_integral=True, 
            l2scaled=False
        )
        
        # Create tensor product grid
        grids = torch.meshgrid(*(xis_1d for _ in range(self.d)), indexing='ij')
        self.xis = torch.stack(grids, dim=-1).view(-1, self.d)
        
        # Compute spectral density and weights
        spec_density = self.kernel.spectral_density(self.xis).to(dtype=self.dtype)
        self.ws2 = spec_density * self.h**self.d
        self.ws2 = self.ws2.to(device=self.device, dtype=torch.complex128)
        self.ws = torch.sqrt(self.ws2)
        
        # Compute convolution vector for Toeplitz structure
        m_conv = (self.mtot - 1) // 2
        v_kernel = compute_convolution_vector_vectorized_dD(m_conv, self.x, self.h).to(dtype=torch.complex128)
        self.toeplitz = ToeplitzND(v_kernel, force_pow2=True)
        
        # Spectral gradients for M-step
        self.Dprime = (self.h**self.d * self.kernel.spectral_grad(self.xis)).to(torch.complex128)
        
    def _setup_nufft(self):
        """Setup NUFFT operators"""
        OUT = (self.mtot,) * self.d
        self.nufft_op = NUFFT(
            self.x, 
            torch.zeros_like(self.x), 
            self.h, 
            self.nufft_eps, 
            cdtype=torch.complex128, 
            device=self.device
        )
        
        # Define NUFFT operators
        self.fadj = lambda v: self.nufft_op.type1(v, out_shape=OUT).reshape(-1)
        self.fwd = lambda fk: self.nufft_op.type2(fk, out_shape=OUT)
        
        # Batched versions
        self.fadj_batched = vmap(self.fadj, in_dims=0, out_dims=0)
        self.fwd_batched = vmap(self.fwd, in_dims=0, out_dims=0)
        
    def _naive_kernel(self, x, xis):
        """Naive kernel computation for testing/comparison"""
        F_train = torch.exp(2 * math.pi * 1j * torch.matmul(x, xis.T)).to(torch.complex128)
        return F_train
        
    def _make_Sigma_z_batch(self, delta):
        """Create batched covariance operator"""
        def Sigma_z_mm(z_batch):
            """Apply (K + Δ)^{-1} to a batch of vectors"""
            if z_batch.ndim == 1:
                z_batch = z_batch.unsqueeze(0)
            
            J, n = z_batch.shape
            z_batch_complex = z_batch.to(dtype=torch.complex128)
            
            # Apply F^* (NUFFT type-1)
            Fstar_z = self.fadj_batched(z_batch_complex)
            
            # Apply (W^2 + Δ_tilde)^{-1}
            delta_tilde = self.fadj(delta.to(dtype=torch.complex128))
            denominator = self.ws2 + delta_tilde
            Fstar_z_scaled = Fstar_z / denominator.unsqueeze(0)
            
            # Apply F (NUFFT type-2)
            result = self.fwd_batched(Fstar_z_scaled)
            
            return result.real.to(dtype=self.dtype)
        
        return Sigma_z_mm
        
    def _posterior_mean_cg(self, delta):
        """Compute posterior mean using conjugate gradients"""
        kappa = self.y - 0.5
        
        def matvec(v):
            """Matrix-vector product for CG: (K + Δ)v"""
            v_complex = v.to(dtype=torch.complex128)
            
            # Apply F^*
            Fstar_v = self.fadj(v_complex)
            
            # Apply W^2
            W2_Fstar_v = self.ws2 * Fstar_v
            
            # Apply F
            F_W2_Fstar_v = self.fwd(W2_Fstar_v)
            
            # Add diagonal term
            result = F_W2_Fstar_v.real + delta * v
            
            return result.to(dtype=self.dtype)
        
        # Solve (K + Δ)m = κ using CG
        x0 = torch.zeros_like(kappa)
        cg_solver = ConjugateGradients(
            matvec, kappa, x0,
            tol=self.cg_tol, 
            early_stopping=False, 
            max_iter=10000
        )
        m = cg_solver.solve()
        
        return m, {"converged": True}  # Return dummy info for compatibility
        
    def _hutchinson_trace(self, probes, delta):
        """Estimate diagonal of Σ using Hutchinson's estimator"""
        # Apply Σ to probes
        Sigma_z_mm = self._make_Sigma_z_batch(delta)
        Sigma_probes = Sigma_z_mm(probes)
        
        # Compute diagonal estimate: diag(Σ) ≈ (1/J) Σ_j z_j ⊙ (Σ z_j)
        # where ⊙ is element-wise multiplication
        diagonal_estimates = probes * Sigma_probes  # (J, n)
        Sz = torch.mean(diagonal_estimates, dim=0)  # Average over probes to get (n,)
        
        return Sz
        
    def e_step(self, max_iters=20, rho0=1.0, gamma=1e-3, tol=1e-6, verbose=False, J=10):
        """
        E-step: Update variational parameters using natural gradients
        
        Returns:
            m: Posterior mean
            Sigma_diags: Posterior variance diagonal
            acc: Predictive accuracy
            Sz: Hutchinson trace estimate
            probes: Random probes for trace estimation
        """
        if verbose:
            print(f"Fitting Polyagamma GP on {self.n} data points...")
        
        # Initialize probes for Hutchinson trace estimation
        probes = torch.randn(J, self.n, device=self.device, dtype=self.dtype)
        
        for it in range(max_iters):
            start_time = time.time()
            
            # Compute posterior mean
            start_mean_time = time.time()
            m, _ = self._posterior_mean_cg(self.q.Delta)
            mean_time = time.time() - start_mean_time
            
            # Compute Hutchinson trace estimate
            start_trace_time = time.time()
            Sz = self._hutchinson_trace(probes, self.q.Delta)
            trace_time = time.time() - start_trace_time
            
            if verbose:
                print(f"Posterior mean time: {mean_time:.3f}s")
                print(f"Hutchinson trace time: {trace_time:.3f}s")
            
            # Compute natural gradient
            xi_squared = m**2 + Sz
            
            # Debug: Check for problematic values
            if verbose and it == 0:
                print(f"Debug: m range: [{m.min().item():.6f}, {m.max().item():.6f}]")
                print(f"Debug: Sz range: [{Sz.min().item():.6f}, {Sz.max().item():.6f}]")
                print(f"Debug: xi_squared range: [{xi_squared.min().item():.6f}, {xi_squared.max().item():.6f}]")
            
            # Ensure xi_squared is positive
            xi_squared = torch.clamp(xi_squared, min=1e-12)
            xi = torch.sqrt(xi_squared)
            
            # Numerically stable computation of tanh(xi/2)/xi
            # For small xi, tanh(xi/2)/xi ≈ 1/2 - xi^2/24 + ...
            # For large xi, tanh(xi/2)/xi ≈ 1/xi
            xi_half = xi / 2
            
            # Use different formulas based on magnitude to avoid numerical issues
            small_xi = xi < 1e-3
            large_xi = xi > 10.0
            medium_xi = ~(small_xi | large_xi)
            
            Lambda = torch.zeros_like(xi)
            
            # For small xi: use Taylor expansion tanh(x)/x ≈ 1 - x^2/3 + 2*x^4/15 - ...
            # But we want tanh(xi/2)/xi = tanh(xi/2)/(2*(xi/2)) * (1/2) = tanh(xi/2)/(xi/2) * (1/2)
            # For small xi/2: tanh(xi/2)/(xi/2) ≈ 1 - (xi/2)^2/3 = 1 - xi^2/12
            if small_xi.any():
                xi_small = xi[small_xi]
                Lambda[small_xi] = 0.5 * (1 - xi_small**2 / 12)
            
            # For large xi: tanh(xi/2)/xi ≈ 1/xi (since tanh(xi/2) ≈ 1)
            if large_xi.any():
                Lambda[large_xi] = 1.0 / xi[large_xi]
            
            # For medium xi: use the standard formula
            if medium_xi.any():
                xi_medium = xi[medium_xi]
                Lambda[medium_xi] = 0.5 * torch.tanh(xi_medium / 2) / xi_medium
            
            # Ensure Lambda is finite and positive
            Lambda = torch.clamp(Lambda, min=1e-8, max=1.0)
            Lambda = torch.where(torch.isfinite(Lambda), Lambda, torch.tensor(0.25, device=self.device, dtype=self.dtype))
            
            if verbose and it == 0:
                print(f"Debug: Lambda range: [{Lambda.min().item():.6f}, {Lambda.max().item():.6f}]")
                print(f"Debug: Lambda has {torch.isnan(Lambda).sum().item()} NaN values")
                print(f"Debug: Lambda has {torch.isinf(Lambda).sum().item()} Inf values")
            
            # Natural gradient update with line search
            rho = rho0
            new_Delta = (1 - rho) * self.q.Delta + rho * Lambda
            
            # Ensure new_Delta is positive and finite
            new_Delta = torch.clamp(new_Delta, min=1e-8, max=1e8)
            new_Delta = torch.where(torch.isfinite(new_Delta), new_Delta, torch.tensor(1e-4, device=self.device, dtype=self.dtype))
            
            total_time = time.time() - start_time
            
            if verbose:
                print(f"it {it:3d}  ρ={rho:.3f}  max|Δ−Λ|={torch.max(torch.abs(self.q.Delta - Lambda)).item():.3e}  time={total_time:.3f}s")
                print(f"self.q.Delta[0]: {self.q.Delta[0].item()}")
            
            # Update Delta
            self.q.Delta = new_Delta
            
            # Check convergence
            if torch.max(torch.abs(self.q.Delta - Lambda)).item() < tol:
                if verbose:
                    print(f"Converged at iteration {it}")
                break
        
        # Compute final posterior variance diagonal
        Sigma_diags = Sz
        
        # Compute predictive accuracy
        probs = torch.sigmoid(m)
        predictions = (probs > 0.5).float()
        acc = (predictions == self.y).float().mean().item()
        
        if verbose:
            print(f"predictive accuracy (analytic) = {acc}")
        
        return m, Sigma_diags, acc, Sz, probes
        
    def m_step(self, m, J=20, verbose=True):
        """
        Perform M-step to compute gradients w.r.t. kernel hyperparameters.
        
        Args:
            m: Posterior mean from E-step
            J: Number of probe vectors
            verbose: Print intermediate results
            
        Returns:
            term1: First gradient term
            term2: Second gradient term  
            term3: Third gradient term
        """
        print("Using Fstar_Kinv_z_local")
        
        # Get current hyperparameters
        delta = self.q.Delta.to(dtype=torch.complex128, device=self.device)
        
        # Compute F^* K^{-1} z for gradient computation
        def Fstar_Kinv_z_local(z_batch):
            """Compute F^* (K + Δ)^{-1} z"""
            if z_batch.ndim == 1:
                z_batch = z_batch.unsqueeze(0)
                
            J, n = z_batch.shape
            z_batch_complex = z_batch.to(dtype=torch.complex128)
            
            # Apply F^*
            Fstar_z = self.fadj_batched(z_batch_complex)
            
            # Apply (W^2 + F^* Δ F)^{-1}
            delta_tilde = self.fadj(delta)
            denominator = self.ws2 + delta_tilde
            result = Fstar_z / denominator.unsqueeze(0)
            
            return result
        
        # Generate probe vectors
        probes = torch.randn(J, self.n, device=self.device, dtype=self.dtype)
        
        # Compute terms for gradient
        # Term 1: m^T ∂K/∂θ K^{-1} m
        m_complex = m.to(dtype=torch.complex128)
        Fstar_Kinv_m = Fstar_Kinv_z_local(m.unsqueeze(0)).squeeze(0)
        
        term1_components = []
        for param_idx in range(self.Dprime.shape[1]):
            Dprime_param = self.Dprime[:, param_idx]
            term1_val = torch.real(torch.sum(Fstar_Kinv_m.conj() * Dprime_param * Fstar_Kinv_m))
            term1_components.append(term1_val.item())
        
        # Term 2: tr(Σ K^{-1} ∂K/∂θ K^{-1})
        Fstar_Kinv_probes = Fstar_Kinv_z_local(probes)
        
        term2_components = []
        for param_idx in range(self.Dprime.shape[1]):
            Dprime_param = self.Dprime[:, param_idx]
            # Compute tr(Σ K^{-1} ∂K/∂θ K^{-1}) using probe vectors
            term2_vals = []
            for j in range(J):
                Fstar_Kinv_probe = Fstar_Kinv_probes[j]
                val = torch.real(torch.sum(Fstar_Kinv_probe.conj() * Dprime_param * Fstar_Kinv_probe))
                term2_vals.append(val)
            term2_val = torch.mean(torch.stack(term2_vals))
            term2_components.append(term2_val.item())
        
        # Term 3: tr(K^{-1} ∂K/∂θ)
        term3_components = []
        for param_idx in range(self.Dprime.shape[1]):
            Dprime_param = self.Dprime[:, param_idx]
            # Use probe vectors to estimate trace
            term3_vals = []
            for j in range(J):
                probe = probes[j].to(dtype=torch.complex128)
                Fstar_probe = self.fadj(probe)
                delta_tilde = self.fadj(delta)
                denominator = self.ws2 + delta_tilde
                Kinv_probe_freq = Fstar_probe / denominator
                val = torch.real(torch.sum(Kinv_probe_freq.conj() * Dprime_param))
                term3_vals.append(val)
            term3_val = torch.mean(torch.stack(term3_vals))
            term3_components.append(term3_val.item())
        
        term1 = torch.tensor(term1_components)
        term2 = torch.tensor(term2_components) 
        term3 = torch.tensor(term3_components)
        
        if verbose:
            print(term1, term2, term3)
            
        return term1, term2, term3
        
    def predict(self, x_test, m=None):
        """
        Make predictions at test points.
        
        Args:
            x_test: Test input locations, shape (n_test, d)
            m: Posterior mean (if None, will compute from current state)
            
        Returns:
            mean_pred: Predictive mean
            var_pred: Predictive variance
        """
        if m is None:
            m, _ = self._posterior_mean_cg(self.q.Delta)
            
        # This is a simplified prediction - full implementation would require
        # computing cross-covariances between train and test points
        # For now, return basic interpolation
        
        # Compute kernel between test and train points
        # This would need to be implemented based on the specific kernel
        raise NotImplementedError("Prediction method needs full cross-covariance implementation")
        
    def get_hyperparameters(self):
        """Get current kernel hyperparameters"""
        return {
            'variance': self.kernel.variance,
            'lengthscale': self.kernel.lengthscale
        }
        
    def set_hyperparameters(self, **kwargs):
        """Set kernel hyperparameters"""
        for key, value in kwargs.items():
            if hasattr(self.kernel, key):
                setattr(self.kernel, key, value)
        
        # Re-setup spectral representation with new parameters
        self._setup_spectral_representation()

    def fit(self, max_iters=20, rho0=1.0, gamma=1e-3, tol=1e-6, verbose=True, J=20):
        """
        Fit the model using the E-step (variational inference).
        
        Args:
            max_iters: Maximum number of iterations
            rho0: Initial step size
            gamma: Step size decay (not used in current implementation)
            tol: Convergence tolerance
            verbose: Print progress
            J: Number of probe vectors for Hutchinson estimator
        
        Returns:
            m: Posterior mean
            Sigma_diags: Posterior variance diagonal
            acc: Predictive accuracy
            Sz: Hutchinson samples
            probes: Probe vectors
        """
        return self.e_step(max_iters=max_iters, rho0=rho0, gamma=gamma, tol=tol, verbose=verbose, J=J)
