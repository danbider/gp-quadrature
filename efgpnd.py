from typing import Dict, Optional, Tuple, Union
import math
from math import prod
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.fft import fftn, ifftn
from torch import nn
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
import pytorch_finufft.functional as pff
from utils.kernels import get_xis
from cg import ConjugateGradients, BatchConjugateGradients
import time
from kernels.kernel_params import GPParams

def efgpnd_gradient_batched(
        x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
        *, nufft_eps=6e-8, cg_tol = 1e-4, early_stopping = True, device=None,
        do_profiling=False, compute_log_marginal=False,
        log_marginal_probes=100, log_marginal_steps=25):
    """
    Gradient of the N‑D GP log‑marginal likelihood estimated with
    Hutchinson trace + CG
    
    Parameters
    ----------
    do_profiling : bool
        If True, use torch.profiler to profile the computation and print results
    compute_log_marginal : bool
        If True, also compute and return the log marginal likelihood
    log_marginal_probes : int
        Number of probes for log det estimation when computing log marginal likelihood 
    log_marginal_steps : int
        Number of Lanczos steps for log det when computing log marginal likelihood
    """
    num_hypers = kernel.num_hypers
    
    # Determine profiler activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        
    # Set up profiling context
    if do_profiling:
        profiler_ctx = profile(activities=activities, record_shapes=True, profile_memory=True)
    else:
        # Use a no-op context manager when profiling is disabled
        from contextlib import nullcontext
        profiler_ctx = nullcontext()
        
    with profiler_ctx as prof:
        # 0)  Book‑keeping ------------------------------------------------------
        with record_function("0_book_keeping"):
            device  = device or x.device
            # device = x.device
            dtype = x.dtype
            rdtype = dtype 
            cdtype = _cmplx(rdtype)
            x       = x.to(device, dtype)
            y       = y.to(device, dtype)
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            cmplx   = _cmplx(dtype)
            x0 = x.min(dim=0).values  
            x1 = x.max(dim=0).values  

            if x.ndim == 1:
                x = x.unsqueeze(-1)
            d = x.shape[1]
            domain_lengths = x1 - x0
            L = domain_lengths.max()
            N = x.shape[0]
        
        # Get frequency grid and weights
        with record_function("1_frequency_grid_setup"):
            xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
            grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij') # makes tensor product Jm 
            xis = torch.stack(grids, dim=-1).view(-1, d) 
            ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d) # (mtot**d,1)
            Dprime  = (h**d * kernel.spectral_grad(xis)).to(cmplx)  # (M, 3)

        # 1)  NUFFT adjoint / forward helpers (modeord=False) -------------------
        with record_function("2_nufft_setup"):
            OUT = (mtot,)*d
            
            # nufft_eps = 1e-4
            phi = (2 * math.pi * h * (x - 0.0)).to(dtype).T.contiguous()   # real (N,)
            def finufft1(vals):
                """
                Adjoint NUFFT: nonuniform→uniform.
                vals: (N,) complex
                returns: tensor of shape OUT, then flattened
                """
                if vals.dtype != cdtype:
                    vals = vals.to(cdtype)
                # if vals not contiguous, make it contiguous
                #TODO I don't understand how contiguous works. 
                if not vals.is_contiguous():
                    vals = vals.contiguous()
                arr = pff.finufft_type1(
                    phi, vals, OUT,
                    eps=nufft_eps, isign=-1, modeord=False
                )
                return arr  # (mtot**d,)

            def finufft2(fk_flat):
                """
                Forward NUFFT: uniform→nonuniform.
                fk_flat: (mtot**d,) complex, in CMCL order
                returns: tensor of shape (N,)
                """

                if fk_flat.ndim == 1:
                    fk_nd = fk_flat.reshape(OUT)
                else:
                    OUT_trace = (fk_flat.shape[0],) + OUT
                    fk_nd = fk_flat.reshape(OUT_trace).contiguous() # (T, mtot**d)
                    if fk_nd.dtype != cdtype:
                        fk_nd = fk_nd.to(cdtype)
                # fk_nd = fk_flat
                return pff.finufft_type2(
                    phi, fk_nd, # phi is (d,N), fk_nd is (T, mtot**d)
                    eps=nufft_eps, isign=+1, modeord=False
                )


            fadj = lambda v: finufft1(v)    # NU → U
            fwd  = lambda fk: finufft2(fk)  # U  → NU

        # 2)  Toeplitz operator T (cached FFT) ----------------------------------
        with record_function("3_toeplitz_setup"):
            m_conv       = (mtot - 1) // 2
            v_kernel     = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
            toeplitz   = ToeplitzND(v_kernel, force_pow2=True)               # cached once

            # 3)  Linear map A· = D F*F (D·) + σ² I -------------------------------
            def A_apply(beta):
                return ws * toeplitz(ws * beta) + sigmasq * beta

        # 4)  Solve A β = W F* y ---------------------------------------------
        with record_function("4_solve_cg"):
            Fy = fadj(y).reshape(-1)

            rhs   = ws * Fy
            beta  = ConjugateGradients(A_apply, rhs,
                                    torch.zeros_like(rhs),
                                    tol=cg_tol, early_stopping=early_stopping).solve()
            alpha = (y - fwd(ws * beta)) / sigmasq # (\tilde{K} +sigma^2 I)^{-1} y

        # 5)  Term‑2  (α*D'α, α*α) --------------------------------------------
        with record_function("5_compute_term2"):
            fadj_alpha = (Fy - toeplitz(ws * beta))/sigmasq # this is faster than fadj(alpha)
            term2 = torch.stack((
                torch.vdot(fadj_alpha, Dprime[:, 0] * fadj_alpha),
                torch.vdot(fadj_alpha, Dprime[:, 1] * fadj_alpha),
                torch.vdot(alpha,       alpha)
            ))

        # 6)  Monte‑Carlo trace (Term‑1) ---------------------------------------
        with record_function("6_monte_carlo_trace"):
            T  = trace_samples
            Z  = (2 * torch.randint(0, 2, (T, x.shape[0]), device=device,
                                  dtype=dtype) - 1).to(cmplx) # (T, N)
            fadjZ = fadj(Z)
            ## Redone 
            fadjZ_flat = fadjZ.reshape(T, -1)              # (T, M)
            Hk = num_hypers -1
            # stack: (Hk, T, M)  ->  (Hk*T, M)
            Di_FZ_all  = torch.stack(
                [Dprime[:, i] * fadjZ_flat for i in range(Hk)],
                dim=0
            ).reshape(-1, fadjZ_flat.shape[-1])            # ((Hk*T), M)

            # one batched NUFFT‑2 for every rhs_i
            rhs_all_kernel = fwd(Di_FZ_all).reshape(Hk, T, -1)      # (Hk, T, N)

            # Toeplitz for B_i
            B_all_kernel   = ws * toeplitz(Di_FZ_all).reshape(Hk, T, -1)  # (Hk, T, M)

            # ------------------------------------------------------------------
            #  add the noise‑variance block (hyper‑parameter index == Hk)
            # ------------------------------------------------------------------
            rhs_noise = Z                                   # (T, N)
            B_noise   = ws * fadjZ_flat                     # (T, M)

            # ------------------------------------------------------------------
            #  concatenate into the shapes expected downstream
            # ------------------------------------------------------------------
            R_all = torch.cat((rhs_all_kernel, rhs_noise.unsqueeze(0)), dim=0) \
                        .reshape(num_hypers * T, -1)        # (num_hypers*T, N)

            B_all = torch.cat((B_all_kernel,  B_noise.unsqueeze(0)), dim=0) \
                        .reshape(num_hypers * T, -1)        # (num_hypers*T, M)

            def A_apply_batch(B):
                return ws * toeplitz(ws * B) + sigmasq * B
            # A_apply_batch_compiled = torch.compile(A_apply_batch)
        with record_function("7_batch_cg_solve"):

            Beta_all = BatchConjugateGradients(
                A_apply_batch, B_all, torch.zeros_like(B_all),
                tol=cg_tol, early_stopping=early_stopping).solve()
        with record_function("7.5_compute_alpha"):
            #
            Alpha_all = (R_all - fwd(ws * Beta_all)) / sigmasq
            A_chunks  = Alpha_all.chunk(num_hypers, 0)

            term1 = torch.stack([(Z * a).sum(1).mean() for a in A_chunks])

        # 7)  Gradient ----------------------------------------------------------
        with record_function("8_gradient_calculation"):
            grad = 0.5 * (term1 - term2)
    
        with record_function("9_log_marginal_likelihood"):
            log_marginal = None
            if compute_log_marginal:
                # Use the log_marginal_probes and log_marginal_steps parameters
                det_term = logdet_slq(ws, sigmasq, toeplitz,
                        probes=log_marginal_probes,
                        steps=log_marginal_steps,
                        dtype=rdtype, device=device, n=N)
                log_marginal = (-0.5 * torch.vdot(y.to(cdtype), alpha) - 0.5*det_term -0.5*N*math.log(2*math.pi)).to(dtype=rdtype)



    # Print profiling results if requested
    if do_profiling and prof is not None:
        print("\n===== Profiling Results for efgpnd_gradient_batched =====")
        # Sort the results by total time, self cpu time or self cuda time
        print("\nEvents sorted by total time:")
        prof_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        print(prof_table)
        
        if torch.cuda.is_available():
            print("\nEvents sorted by CUDA time:")
            cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            print(cuda_table)
        
        # Memory stats if available
        # try:
        #     print("\nMemory stats:")
        #     memory_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
        #     print(memory_table)
        # except:
        #     pass
    
    if compute_log_marginal:
        return grad.real, log_marginal
    else:
        return grad.real

### Main class for the EFGPND model ###
# Example usage:
# model = EFGPND(x, y, "SquaredExponential", EPSILON)
class EFGPND(nn.Module):
    """
    Equispaced‑Fourier Gaussian Process (EFGP) regression in d dimensions.

    Parameters
    ----------
    x, y        : training inputs / targets  (stored but *not* copied)
    kernel      : object providing `spectral_density` (passed straight to `efgp_nd`)
                  or a string indicating kernel type (e.g., "SquaredExponential", "SE", "Matern32", "Matern52")
    sigmasq     : (optional) scalar observation noise variance
    eps         : quadrature accuracy parameter (passed to `get_xis`)
    nufft_eps   : NUFFT accuracy parameter
    # TODO some of these opts are getting clunky, maybe see Gpytorch for inspiration
    opts        : dict with CG tolerances, variance method, and other options:
                  - estimate_variance: bool - whether to compute predictive variances
                  - variance_method: str - 'regular' or 'stochastic'
                  - compute_log_marginal: bool - whether to compute log marginal likelihood (default: False)
                  - log_marginal_probes: int - number of probes for log det estimation (default: 100)
                  - log_marginal_steps: int - number of Lanczos steps for log det (default: 25)
                  - hutchinson_probes: int - number of probes for Hutchinson variance estimation
    estimate_params : bool - whether to automatically estimate kernel parameters
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel: object,
        sigmasq: float = None,
        eps: float = 1e-2,
        nufft_eps: float = 1e-4,
        opts: Optional[Dict] = None,
        estimate_params: bool = True,
    ):
        """
        Equispaced‑Fourier Gaussian Process (EFGP) regression in d dimensions.

        Parameters
        ----------
        x, y        : training inputs / targets  (stored but *not* copied)
        kernel      : object providing `spectral_density` (passed straight to `efgp_nd`)
                    or a string indicating kernel type (e.g., "SquaredExponential", "SE", "Matern32", "Matern52")
        sigmasq     : scalar observation noise variance
        eps         : quadrature accuracy parameter (passed to `get_xis`)
        nufft_eps   : NUFFT accuracy parameter
        opts        : dictionary of options for auxiliary methods
                    (e.g., `max_cg_iter` for CG solver)
        estimate_params : whether to initialize lengthscales and noise by MLE
        """
        super().__init__()
        self.x = x
        self.y = y
        self.device = x.device
        self.eps = eps
        self.nufft_eps = nufft_eps
        self.opts = {} if opts is None else opts.copy()
        
        # Get dimension from input data
        if x.ndim == 1:
            dimension = 1
        else:
            dimension = x.shape[1]
        
        # Handle string kernel specification
        if isinstance(kernel, str):
            # Import kernel classes here to avoid circular imports
            from kernels.squared_exponential import SquaredExponential
            from kernels.matern import Matern32, Matern52
            
            # Create kernel object based on string name
            kernel_name = kernel.lower()
            if kernel_name in ["squaredexponential", "se"]:
                kernel = SquaredExponential(dimension=dimension, lengthscale=1.0)
            elif kernel_name == "matern32":
                kernel = Matern32(dimension=dimension, lengthscale=1.0)
            elif kernel_name == "matern52":
                kernel = Matern52(dimension=dimension, lengthscale=1.0)
            else:
                raise ValueError(f"Unknown kernel type: {kernel}")
        
        self.kernel = kernel
        
        # Estimate optimal hyperparameters if passed a string
        # TODO this doesn't need to be great but probably need to see how this works on different datasets.
        if estimate_params:
            try:
                # Compute pairwise distances to estimate lengthscale
                import numpy as np
                from scipy.spatial.distance import pdist
                
                # Subsample data if needed
                x_np = x.detach().cpu().numpy()
                if len(x) > 500:
                    indices = np.random.choice(len(x), 500, replace=False)
                    x_np = x_np[indices]
                
                # Compute median distance
                if x_np.ndim > 1 and x_np.shape[1] > 1:
                    # Multi-dimensional data
                    median_dist = np.median(pdist(x_np))
                else:
                    # 1D data
                    median_dist = np.median(pdist(x_np.reshape(-1, 1)))
                
                # Set lengthscale to median distance / 2
                lengthscale = median_dist / 2.0
                
                # Adjust for Matern kernels
                if hasattr(kernel, 'name') and 'matern' in kernel.name.lower():
                    lengthscale *= 1.5
                    
                # Set kernel lengthscale
                if hasattr(kernel, 'set_hyper'):
                    kernel.set_hyper('lengthscale', lengthscale)
                elif hasattr(kernel, 'lengthscale'):
                    kernel.lengthscale = lengthscale
                    
                # Estimate noise variance (10% of data variance)
                if sigmasq is None:
                    sigmasq = 0.1 * y.var().item()
                    
            except Exception as e:
                print(f"Warning: Failed to estimate hyperparameters: {e}")
                # Use default values (already set during kernel creation)
                if sigmasq is None:
                    sigmasq = 0.1
        
        # Create GP parameters
        self._gp_params = GPParams(kernel=kernel, init_sig2=(sigmasq or 0.1))
        
        # Register parameters
        self.register_parameter("gp_params", self._gp_params.raw)
        
        # Store fit data
        self._beta = None        # Fourier weights
        self._xis = None         # frequency grid
        self._ws = None          # quadrature weights √S(ξ) h^{d/2}
        self._toeplitz = None    # convolution operator object
        self._fitted = False
        self._cached_params = {} # For tracking hyperparameter changes
        self._registered_optimizers = []
        
        # Update parameter cache
        self._update_param_cache()

    def register_optimizer(self, optimizer):
        """
        This method adds hooks to the optimizer's step function to automatically
        call sync_parameters() after each optimization step.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to register for automatic parameter synchronization
            
        Returns
        -------
        optimizer : torch.optim.Optimizer
            The same optimizer, now with hooks for automatic syncing
        """
        # Skip if this optimizer is already registered
        if optimizer in self._registered_optimizers:
            return optimizer
            
        # Store original step function
        original_step = optimizer.step
        
        # Create a new step function that calls sync_parameters after the original step
        def step_with_sync(*args, **kwargs):
            # Call the original step function
            result = original_step(*args, **kwargs)
            
            # Ensure parameters are properly synchronized
            self._gp_params.sync_all_parameters()
            self.sync_parameters()
            
            return result
            
        # Replace the optimizer's step function with our new one
        optimizer.step = step_with_sync
        
        # Store the optimizer to avoid registering it multiple times
        self._registered_optimizers.append(optimizer)
        
        return optimizer

    @property
    def sigmasq(self) -> torch.Tensor:
        """Get the current noise variance tensor from GPParams"""
        return self._gp_params.sig2
    
    def sync_parameters(self):
        """
        Synchronize parameters between GPParams, kernel, and sigmasq.
        Call this after manual parameter updates to ensure consistency.
        """
        # Get the current parameter values from GPParams
        pos_vals = self._gp_params.pos.detach().cpu().tolist()
        
        # Update kernel hyperparameters directly
        for i, name in enumerate(self._gp_params.hypers_names):
            self.kernel.set_hyper(name, pos_vals[i])
        
        # Store the current noise variance
        self._internal_sigmasq = self._gp_params.sig2.detach().clone()
        
        # Update parameter cache
        self._update_param_cache()
        
        return self
    

    def _update_param_cache(self):
        """
        Store current hyperparameter values in the cache to detect changes. 
        Used so that predict can check if hyperparameters have changed and refit if so.
        """
        if isinstance(self.kernel, str):
            # In case kernel is a string
            self._cached_params['kernel_type'] = str(type(self.kernel))
            self._cached_params['sigmasq'] = float(self.sigmasq.item())
        else:
            # Store kernel parameters
            for name, value in self.kernel.iter_hypers():
                self._cached_params[name] = float(value)
            # Store noise variance
            self._cached_params['sigmasq'] = float(self.sigmasq.item())
    
    def _params_changed(self):
        """
        Check if hyperparameters have changed since the last fit.
        
        Returns
        -------
        bool
            True if parameters have changed, False otherwise
        """
        # If no cached parameters, assume changed
        if not self._cached_params:
            return True
        
        # Check if kernel parameters have changed
        if isinstance(self.kernel, str):
            # For string kernels, just check if the type has changed
            if 'kernel_type' not in self._cached_params or self._cached_params['kernel_type'] != str(type(self.kernel)):
                return True
        else:
            # For proper kernel objects with iter_hypers method
            try:
                for name, value in self.kernel.iter_hypers():
                    if name not in self._cached_params or abs(self._cached_params[name] - float(value)) > 1e-8:
                        return True
            except (AttributeError, TypeError):
                # Fallback if iter_hypers isn't available
                if 'kernel_type' not in self._cached_params or self._cached_params['kernel_type'] != str(type(self.kernel)):
                    return True
        
        # Check if noise parameter has changed
        current_sigmasq = float(self.sigmasq.item())
        if 'sigmasq' not in self._cached_params or abs(self._cached_params['sigmasq'] - current_sigmasq) > 1e-8:
            return True
            
        return False

# TODO there's some redundancy here-- this is calling a fcn to get gradients
# then converts to raw space and applies them to the parameters. 
    def compute_gradients(
        self,
        *,
        trace_samples: int = 10,
        do_profiling: bool = False,
        nufft_eps: Optional[float] = None,
        apply_gradients: bool = True,
        compute_log_marginal: bool = False,
        log_marginal_probes: int = 100,
        log_marginal_steps: int = 25,
        verbose: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute gradient of the negative log marginal likelihood with respect to hyperparameters.
        
        This method computes the gradients for manual optimization approaches, where users want
        to use their own optimization routine (like torch optimizers).
        
        Parameters
        ----------
        trace_samples : int
            Number of Hutchinson trace samples to use
        do_profiling : bool
            Whether to profile the gradient computation
        nufft_eps : Optional[float]
            NUFFT accuracy (defaults to self.eps * 0.1)
        apply_gradients : bool
            If True, automatically apply gradients to model parameters (self._gp_params.raw)
        compute_log_marginal : bool
            If True, also compute and return the log marginal likelihood value
        log_marginal_probes : int
            Number of probes for log det estimation when computing log marginal likelihood 
        log_marginal_steps : int
            Number of Lanczos steps for log det when computing log marginal likelihood
        verbose : bool
            If True, print verbose debugging information
            
        Returns
        -------
        If compute_log_marginal=False:
        grads : torch.Tensor
            Gradients for kernel hyperparameters and sigmasq in log space
        If compute_log_marginal=True:
            (grads, log_marginal) : Tuple[torch.Tensor, torch.Tensor]
                Gradients and log marginal likelihood value
        """
        # Ensure parameters are synchronized from GPParams to kernel
        self.sync_parameters()
        
        # Set default nufft_eps if not provided
        if nufft_eps is None:
            nufft_eps = self.eps * 0.1
        
        # Calculate data bounds for gradient computation
        x0 = self.x.min(dim=0).values
        x1 = self.x.max(dim=0).values
        
        log_marginal_kwargs = {
            'compute_log_marginal': compute_log_marginal,
            'log_marginal_probes': log_marginal_probes, 
            'log_marginal_steps': log_marginal_steps
        }
        
        # Compute gradients with respect to the original parameters
        result = efgpnd_gradient_batched(
            self.x, self.y,
            sigmasq=self._gp_params.sig2,
            kernel=self.kernel,
            eps=self.eps,
            trace_samples=trace_samples,
            x0=x0, x1=x1,
            do_profiling=do_profiling,
            nufft_eps=nufft_eps,
            **log_marginal_kwargs,
            **kwargs
        )
        
        # Extract results based on what was returned
        if compute_log_marginal:
            grads, log_marginal = result
        else:
            grads = result
        if not isinstance(grads, torch.Tensor):
            grads = torch.tensor(grads, device=self.device, dtype=self.x.dtype)
        if grads.ndim == 0:
            grads = grads.unsqueeze(0)  # Convert scalar to 1D tensor
        
        
        # Transform to raw space gradient via chain rule
        pos_vec = self._gp_params.pos.to(self.device)
        raw_grad = torch.stack([
            grads[i].detach().to(self.device) * pos_vec[i]
            for i in range(len(grads))
        ], dim=0)
        
        # Automatically apply gradients 
        if apply_gradients:
            # Make sure raw_grad is detached before setting as gradient
            with torch.no_grad():
                # Set the whole gradient at once instead of element by element
                self._gp_params.raw.grad = raw_grad.detach().clone()
         
        if compute_log_marginal:
            return raw_grad, log_marginal
        else:
            return raw_grad

    def _compute_common_parameters(self, force_recompute: bool = False, nufft_eps: Optional[float] = None) -> None:
        """
        Compute and cache common parameters that depend on hyperparameters.
        These parameters can be reused for all predictions as long as the hyperparameters remain unchanged.
        
        Parameters
        ----------
        force_recompute : bool, default=False
            Force recomputation of the parameters even if hyperparameters haven't changed
        nufft_eps : float, optional
            NUFFT accuracy parameter, defaults to self.nufft_eps if not specified
        """
        # Ensure parameters are synchronized
        self.sync_parameters()
        
        # Check if we need to compute parameters
        needs_recompute = (
            not self._fitted or          # Never been fitted
            self._params_changed() or    # Parameters changed
            force_recompute              # User requested recomputation
        )
        
        if not needs_recompute:
            return
            
        # Setup common constants and parameters
        device = self.x.device
        rdtype = torch.float32
        cdtype = _cmplx(rdtype)
        sigmasq_scalar = float(self._gp_params.sig2)
        
        # Use provided nufft_eps or default to class value
        if nufft_eps is None:
            nufft_eps = self.nufft_eps
            
        # Cast training data to appropriate device and dtype
        x = self.x.to(device=device, dtype=rdtype)
        y = self.y.to(device=device, dtype=rdtype)
        N, d = x.shape
        
        # Get domain size (L)
        L = torch.max(torch.max(x, dim=0).values - torch.min(x, dim=0).values)
        if L <= 1e-9:
            L = torch.tensor(1.0, device=device, dtype=rdtype)
        
        # Get frequency grid
        xis_1d, h_float, mtot = get_xis(
            kernel_obj=self.kernel, 
            eps=self.eps, 
            L=L.item(), 
            use_integral=True, 
            l2scaled=False
        )
        h = torch.tensor(h_float, device=device, dtype=rdtype)
        xis_1d = xis_1d.to(device=device, dtype=rdtype)
        
        # Create n-dimensional grid
        grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
        xis = torch.stack(grids, dim=-1).view(-1, d)
        
        # Store h_float as an attribute for later use
        xis.h_float = h_float
        
        # Compute spectral density values
        spectral_vals = self.kernel.spectral_density(xis).to(dtype=cdtype)
        ws = torch.sqrt(spectral_vals * (h ** d))
        ws = ws.to(dtype=cdtype)
    
        # NUFFT Setup for training points
        OUT = (mtot,) * d
        xcen = torch.zeros(d, device=device, dtype=rdtype)
        
        # Setup for training points
        phi_train, finufft1, _ = setup_nufft(x, xcen, h, nufft_eps, cdtype)
    
        # Compute right-hand side for mean
        right_hand_side = ws * finufft1(phi_train, y, OUT=OUT)
    
        # Setup Toeplitz operator
        m_conv = (mtot - 1) // 2
        v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
        toeplitz = ToeplitzND(v_kernel, force_pow2=True)
        
        # Create operator functions
        A_mean, A_var, Gv = setup_operators(ws, toeplitz, sigmasq_scalar, cdtype)
    
        # Solve CG system for mean
        cg_tol = self.opts.get("cg_tolerance", 1e-4)
        beta = ConjugateGradients(
            A_mean, 
            right_hand_side, 
            x0=torch.zeros_like(right_hand_side, dtype=cdtype),
            tol=cg_tol,
            early_stopping=True
        ).solve()
    
        # Cache results for future predictions
        self._beta = beta
        self._xis = xis
        self._ws = ws
        self._toeplitz = toeplitz
        self._fitted = True
        self._update_param_cache()
    
    def predict(
        self,
        x_new: torch.Tensor,
        *,
        return_variance: bool = True,
        variance_method: str = "stochastic",
        hutchinson_probes: int = 1_000,
        compute_log_marginal: bool = False,
        force_recompute: bool = False,
        do_profiling: bool = False,
        nufft_eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict posterior mean (and optionally variance) at `x_new`.
        
        This method will automatically fit the model if it hasn't been
        fitted yet or if hyperparameters have changed since the last fit.

        Parameters
        ----------
        x_new : torch.Tensor
            New input points to predict at
        return_variance : bool, default=True
            Whether to compute and return variance
        variance_method : str, default="stochastic"
            Method to use for variance calculation
        hutchinson_probes : int, default=1000
            Number of probes for stochastic variance estimation
        compute_log_marginal : bool, default=False
            Whether to compute and return log marginal likelihood
        force_recompute : bool, default=False
            Force recomputation of the fit even if parameters haven't changed
        do_profiling : bool, default=False
            Enable profiling for performance analysis
        nufft_eps : float, optional
            NUFFT accuracy parameter, defaults to self.nufft_eps if not specified

        Returns
        -------
        If compute_log_marginal=False:
            mean : torch.Tensor
                Predicted mean values
            var : Optional[torch.Tensor]
                Predicted variance values (if return_variance=True)
        If compute_log_marginal=True:
            mean, var, log_marginal_likelihood
        """
        if x_new is None:
            raise ValueError("x_new must be provided for prediction")
        
        # Compute or retrieve cached parameters
        self._compute_common_parameters(force_recompute=force_recompute, nufft_eps=nufft_eps)
        
        # Setup common constants and parameters
        device = self.x.device
        rdtype = torch.float32
        cdtype = _cmplx(rdtype)
        sigmasq_scalar = float(self._gp_params.sig2)
        
        # Use provided nufft_eps or default to class value
        if nufft_eps is None:
            nufft_eps = self.nufft_eps
            
        # Setup profiling context
        if do_profiling and torch.cuda.is_available():
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        else:
            activities = [ProfilerActivity.CPU]
            
        from contextlib import nullcontext
        profile_ctx = profile(activities=activities, record_shapes=True) if do_profiling else nullcontext()
        
        with profile_ctx as prof:
            # Cast data to appropriate device and dtype
            x_new = x_new.to(device=device, dtype=rdtype)
            B, d = x_new.shape
            
            # Retrieve cached parameters
            beta = self._beta
            xis = self._xis
            ws = self._ws
            toeplitz = self._toeplitz
            h_float = xis.h_float
            h = torch.tensor(h_float, device=device, dtype=rdtype)
            mtot = int(xis.shape[0]**(1/d))
            
            # Create operator functions
            A_mean, A_var, Gv = setup_operators(ws, toeplitz, sigmasq_scalar, cdtype)
        
            # Setup for prediction
            xcen = torch.zeros(d, device=device, dtype=rdtype)
            OUT = (mtot,) * d
            
            # Setup NUFFT for prediction points
            phi_test, _, finufft2 = setup_nufft(x_new, xcen, h, nufft_eps, cdtype)
            
            # Compute predictive mean
            with record_function("predict_mean"):
                yhat = finufft2(phi_test, ws * beta, OUT=OUT).real
            
            # Initialize results
            ytrg = {"mean": yhat}
            
            # Compute variance if requested
            with record_function("compute_variance"):
                if return_variance:
                    variance = compute_prediction_variance(
                        x_new=x_new,
                        xis=xis,
                        ws=ws,
                        A_var=A_var,
                        cg_tol=self.opts.get("cg_tolerance", 1e-4),
                        max_cg_iter=self.opts.get("max_cg_iterations", 1000),
                        variance_method=variance_method,
                        h=h,
                        xcen=xcen,
                        hutchinson_probes=hutchinson_probes,
                        nufft_eps=nufft_eps,
                        device=device,
                        rdtype=rdtype,
                        cdtype=cdtype
                    )
                    ytrg["var"] = variance
                else:
                    ytrg["var"] = torch.full((B,), float("nan"), device=device, dtype=rdtype)
            
            # Compute log marginal likelihood if requested
            with record_function("compute_log_marginal"):
                if compute_log_marginal:
                    log_marginal_likelihood = self._compute_log_marginal(
                        beta=beta,
                        ws=ws,
                        sigmasq=sigmasq_scalar,
                        toeplitz=toeplitz,
                        device=device,
                        rdtype=rdtype,
                        n=self.x.shape[0]
                    )
                    ytrg["log_marginal"] = log_marginal_likelihood
            
        # Show profiling results if requested
        if do_profiling:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            
        # Return results based on what was computed
        if compute_log_marginal:
            return ytrg["mean"], ytrg["var"], ytrg["log_marginal"]
        else:
            return ytrg["mean"], ytrg["var"]
            
    def _compute_log_marginal(self, beta, ws, sigmasq, toeplitz, device, rdtype, n):
        """
        Compute the log marginal likelihood.
        
        Parameters
        ----------
        beta : torch.Tensor
            Fourier weights
        ws : torch.Tensor
            Quadrature weights
        sigmasq : float
            Noise variance
        toeplitz : ToeplitzND
            Toeplitz operator
        device : torch.device
            Device to use
        rdtype : torch.dtype
            Real data type
        n : int
            Number of training points
            
        Returns
        -------
        torch.Tensor
            Log marginal likelihood
        """
        # Use cached hyperparameters for log determinant calculation
        log_det = logdet_slq(
            ws=ws,
            sigma2=sigmasq,
            toeplitz=toeplitz,
            probes=self.opts.get("log_marginal_probes", 100),
            steps=self.opts.get("log_marginal_steps", 25),
            dtype=rdtype,
            device=device,
            n=n
        )
        
        # Compute data fit term (beta^T D beta)
        data_fit = (ws.abs() * (beta.abs() ** 2)).sum().real
        
        # Return the negative log marginal likelihood (for minimization)
        return -0.5 * (log_det + data_fit)

    def optimize_hyperparameters(
        self,
        *,
        # --- optimization options ---------------------------------------
        optimizer              = 'Adam',         # optional optimizer (e.g., 'Adam' or Adam instance)
        lr: Optional[float]    = 0.1,         # learning rate
        max_iters: int         = 50,
        min_lengthscale: float = 5e-3,
        # --- logging options -------------------------------------------
        log_interval: int      = 10,          # Interval for logging progress
        compute_log_marginal: bool = False,   # Whether to compute log marginal likelihood
        verbose: bool = False,                # Print detailed information (changed to default True to help debug)
        trace_samples: int = 10,             # Number of trace samples to use

        **gkwargs,                             # forwarded to compute_gradients
    ):
        """
        Optimize hyperparameters using gradient-based optimization.
        
        Parameters
        ----------
        optimizer : str or torch.optim.Optimizer
            'Adam' or an optimizer instance
        lr : float
            Learning rate for optimization
        max_iters : int
            Maximum number of iterations
        min_lengthscale : float
            Minimum allowed lengthscale value
        log_interval : int
            Interval for logging progress
        compute_log_marginal : bool
            Whether to compute log marginal likelihood during optimization
        verbose : bool
            Print detailed information about the optimization process
        trace_samples : int
            Number of trace samples for gradient estimation
        """
        device = self.x.device
        rdtype = self.x.dtype

        # Create optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                optimizer_instance = Adam(self._gp_params.parameters(), lr=lr)  # Only optimize GPParams
            else:
                raise ValueError(f"Unsupported optimizer string: {optimizer}. Currently supporting: 'adam'")
        else:
            optimizer_instance = optimizer
        
        # Track hyperparameters
        history = {
            'log_marginal': [],
            'gradients': []
        }
        
        # Store initial hyperparameters
        for name, value in self.kernel.iter_hypers():
            if name not in history:
                history[name] = []
            history[name].append(float(value))
        
        history['sigmasq'] = [float(self.sigmasq.item())]
        
        start_time = time.time()
        print(f"Optimizing hyperparameters using {optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__}")

        # Optimization loop
        for it in range(max_iters):
            # Tracking current hyperparameters 
            current_hypers = {}
            for name, value in self.kernel.iter_hypers():
                if name not in history:
                    history[name] = []
                current_val = float(value)
                history[name].append(current_val)
                current_hypers[name] = current_val
            
            current_sigmasq = float(self.sigmasq.item())
            history['sigmasq'].append(current_sigmasq)
            
            # Zero gradients
            optimizer_instance.zero_grad()
            
            # Compute gradients with periodic log marginal likelihood
            if compute_log_marginal and (it % log_interval == 0 or it == max_iters - 1):
                grad, log_marginal = self.compute_gradients(
                    trace_samples=trace_samples,
                    nufft_eps=self.nufft_eps,  # higher accuracy for optimization
                    apply_gradients=True,  # Apply gradients directly
                    compute_log_marginal=True,
                    verbose=verbose,
                    **gkwargs
                )
                history['log_marginal'].append(float(log_marginal.item() if isinstance(log_marginal, torch.Tensor) else log_marginal))
            else:
                grad = self.compute_gradients(
                    trace_samples=trace_samples,
                    nufft_eps=self.nufft_eps,  # higher accuracy for optimization
                    apply_gradients=True,  # Apply gradients directly
                    compute_log_marginal=False,
                    verbose=verbose,
                    **gkwargs
                )
            
            # Store gradients for debugging
            history['gradients'].append([float(g.item() if isinstance(g, torch.Tensor) else g) for g in grad])
            if verbose:
                print(f"  Iter {it}: Gradients = {[float(g.item() if isinstance(g, torch.Tensor) else g) for g in grad]}")
            
            # Step optimizer 
            optimizer_instance.step()
            
            # Apply constraints
            with torch.no_grad():
                # Apply minimum lengthscale constraint
                try:
                    ls_idx = self._gp_params.hypers_names.index('lengthscale')
                    min_val = torch.tensor(min_lengthscale, device=device, dtype=rdtype)
                    if torch.exp(self._gp_params.raw[ls_idx]) < min_val:
                        self._gp_params.raw[ls_idx].copy_(torch.log(min_val))
                except (ValueError, IndexError) as e:
                    if verbose:
                        print(f"Note: Could not apply lengthscale constraint: {e}")
                
                # Sync parameters after optimization step
                self.sync_parameters()
            
            # Log progress
            if it % log_interval == 0 or it == max_iters - 1:
                log_parts = [f"iter {it}/{max_iters}"]
                for name, values in history.items():
                    if values and name not in ['gradients']:  # Skip gradient arrays
                        if name == 'log_marginal' and not compute_log_marginal:
                            continue
                        log_parts.append(f"{name}={values[-1]:.6g}")
                print(", ".join(log_parts))
        
        # Reset the model to ensure it will recompute common parameters on next predict call
        self._fitted = False
        self._cached_params = {}
        
        # If using final hyperparameters, pre-compute common parameters once for efficiency
        if verbose:
            print("Pre-computing model parameters with final hyperparameters...")
        self._compute_common_parameters(force_recompute=True)
        
        print(f"Optimization complete after {time.time() - start_time:.2f} seconds")
        
        # Print final hyperparameters
        print("\nFinal hyperparameters:")
        for name, value in self.kernel.iter_hypers():
            print(f"{name} = {float(value):.6g}")
        print(f"sigmasq = {float(self.sigmasq.item()):.6g}")
        
        self.training_log = history
        # return model , access log via model.training_log
        return self



# # ──────────────────────────────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────────────────────────────
def _cmplx(real_dtype: torch.dtype) -> torch.dtype:
    """Matching complex dtype for a given real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128

# TODO - work out when to do the force_pow2 etc 
class ToeplitzND:
    """
    ToeplitzND class for multidimensional Toeplitz matrix-vector products using FFTs.
    """
    
    def __init__(self, v: torch.Tensor, *, force_pow2: bool = True, precompute_fft: bool = True):
        """
        Initialize the Toeplitz operator with the first column/row vector.
        
        Args:
            v: First column/row of the Toeplitz matrix, zero-padded to full convolution size
            force_pow2: If True, use power of 2 for FFT sizes (default: True)
            precompute_fft: If True, precompute and store the FFT of v (default: True)
        """
        # Ensure complex
        if not torch.is_complex(v):
            v = v.to(torch.complex128 if v.dtype == torch.float64 else torch.complex64)
            
        # Dimensions and block sizes
        self.Ls = list(v.shape)                   # [L1,...,Ld]
        self.ns = [(L+1)//2 for L in self.Ls]     # [n1,...,nd]
        self.size = prod(self.ns)                 # total block elements
        self.d = len(self.Ls)                     # dimensionality
        
        self.device = v.device
        self.dtype = v.dtype
        
        # Determine optimal FFT sizes
        if force_pow2:
            # Use power of 2 for FFT (original approach)
            self.fft_shape = [1 << (L-1).bit_length() for L in self.Ls]
        else:
            # Use optimal FFT sizes (can be more efficient)
            self.fft_shape = [self._next_fast_fft_size(L) for L in self.Ls]
        
        # Prepare padding for the kernel
        pad = []
        for L, F in zip(reversed(self.Ls), reversed(self.fft_shape)):
            pad += [0, F - L]
            
        # Pad and cache kernel FFT
        self.v_pad = nnf.pad(v, pad)
        
        # Precompute FFT of kernel for efficiency
        if precompute_fft:
            self.fft_kernel = fftn(self.v_pad, dim=list(range(-len(self.Ls), 0)))
        else:
            self.fft_kernel = None
        
        # Slice indices for central block extraction
        self.starts = [n-1 for n in self.ns]
        self.ends = [st+n for st, n in zip(self.starts, self.ns)]
        
        # Prepare slices for central block extraction
        self.central_slices = []
        
        # Cache FFT dimensions for repeated use
        self.fft_dims = list(range(-len(self.Ls), 0))
        
        # Preallocate buffers for reuse
        self._cached_batch_shape = None
        self._cached_x_pad = None
        self._cached_x_fft = None
        
    def _next_fast_fft_size(self, n):
        """
        Find the next efficient FFT size.
        For real FFTs, sizes with small prime factors are more efficient.
        """
        # Good FFT sizes are products of small primes (2, 3, 5, 7)
        while True:
            factors = self._prime_factors(n)
            # Check if all factors are "good" for FFT (2, 3, 5, 7)
            if all(f in {2, 3, 5, 7} for f in factors):
                return n
            n += 1
            
    def _prime_factors(self, n):
        """Return the prime factors of n."""
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d*d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors
            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Toeplitz matrix-vector product using FFT-based convolution.
        
        Args:
            x: Input tensor of shape (batch_size, *dimensions) or (*dimensions)
            
        Returns:
            Toeplitz matrix-vector product of shape (batch_size, *output_dimensions) or (*output_dimensions)
        """
        x = x.to(self.device)
        orig_flat = False  # remember how input looked

        # --- Determine input layout and batch shape ---
        if x.shape[-1] == self.size:  # flat batch
            orig_flat = True
            batch_shape = x.shape[:-1]
            x = x.reshape(*batch_shape, *self.ns)
        elif list(x.shape[-self.d:]) == self.ns:  # block batch
            batch_shape = x.shape[:-len(self.ns)]
        else:
            raise ValueError(
                f"Expected trailing dim {self.size} or block {tuple(self.ns)}, "
                f"got {tuple(x.shape)}"
            )

        # --- Convert to complex if necessary ---
        if not x.is_complex():
            x = x.to(dtype=self.dtype)

        # --- Create padding values ---
        pad = []
        for n, F in zip(reversed(self.ns), reversed(self.fft_shape)):
            pad += [0, F - n]

        # --- Zero-pad input ---
        x_pad = nnf.pad(x, pad)

        # --- Perform FFT convolution ---
        x_fft = fftn(x_pad, dim=list(range(-self.d, 0)))
        
        # Multiply in frequency domain
        if self.fft_kernel is not None:
            y_fft = x_fft * self.fft_kernel
        else:
            y_fft = x_fft * fftn(self.v_pad, dim=self.fft_dims)
        
        # Inverse FFT
        y = ifftn(y_fft, dim=list(range(-self.d, 0)))
        
        # --- Extract central block ---
        # Create slices for the batch dimensions
        slices = [slice(None)] * (y.ndim - self.d)
        # Add slices for the data dimensions
        for st, en in zip(self.starts, self.ends):
            slices.append(slice(st, en))
        y = y[tuple(slices)]
        
        # --- Restore original layout ---
        if orig_flat:
            y = y.reshape(*batch_shape, self.size)
            
        return y

def compute_convolution_vector_vectorized_dD(m: int, x: torch.Tensor, h: float) -> torch.Tensor:
    """
    Multi‑D type‑1 NUFFT convolution vector:
      v[k1,...,kd] = sum_n exp(2πi <k, x_n>)
    """
    device      = x.device
    dtype_real  = x.dtype
    dtype_cmplx = torch.complex64 if dtype_real == torch.float32 else torch.complex128
    if x.ndim == 1:
        x = x[:, None]
    N, d        = x.shape
    eps         = 6e-8

    # (d, N) array of phases
    phi = (2 * math.pi * h * x).T.contiguous().to(dtype_real)

    # all weights = 1 + 0i
    c = torch.ones(N, dtype=dtype_cmplx, device=device)

    # output grid size in each of the d dims
    OUT = tuple([4*m + 1] * d)

    v = pff.finufft_type1(
        phi,    # (d, N)
        c,      # (N,)
        OUT,
        eps=eps,
        isign=-1,
        modeord=False
    )
    return v

# Add helper functions before the EFGPND class

def setup_nufft(x, xcen, h, nufft_eps, cdtype):
    """Helper function to set up NUFFT parameters and helper functions."""
    TWO_PI = 2 * math.pi
    
    # Compute phi for the input points
    phi = (TWO_PI * h * (x - xcen)).T.contiguous()
    d = phi.shape[0]  # Get dimensionality
    
    def finufft1(phi_in, vals, OUT=None):
        """
        Adjoint NUFFT: nonuniform→uniform.
        """
        phi_in = phi_in.contiguous()
        vals = vals.contiguous()
        if vals.dtype != cdtype:
            vals = vals.to(dtype=cdtype)
            
        # Use provided OUT shape or default
        if OUT is None:
            # This is an approximation - in practice, OUT should be passed explicitly for finufft1
            n_pts = phi_in.shape[1]
            d_loc = phi_in.shape[0]
            m_per_dim = int(n_pts**(1/d_loc) + 0.5)  # Round to nearest integer
            OUT = (m_per_dim,) * d_loc
        
        fk = pff.finufft_type1(phi_in, vals, OUT, eps=nufft_eps, isign=-1, modeord=False)
        if fk.dtype != cdtype:
            fk = fk.to(dtype=cdtype)
        return fk.view(-1)

    def finufft2(phi_in, fk_flat, OUT=None):
        """
        Forward NUFFT: uniform→nonuniform.
        """
        phi_in = phi_in.contiguous()
        
        # If OUT not specified, infer it
        if OUT is None:
            n_pts = fk_flat.shape[0]
            d_loc = phi_in.shape[0]
            m_per_dim = int(n_pts**(1/d_loc) + 0.5)  # Round to nearest integer
            OUT = (m_per_dim,) * d_loc
            
        fk_nd = fk_flat.view(OUT).to(dtype=cdtype).contiguous()
        if fk_nd.dtype != cdtype:
            fk_nd = fk_nd.to(dtype=cdtype)
        return pff.finufft_type2(phi_in, fk_nd, eps=nufft_eps, isign=+1, modeord=False)
    
    return phi, finufft1, finufft2

def setup_operators(ws, toeplitz, sigmasq_scalar, cdtype):
    """Helper function to set up operator functions."""
    ns_shape = tuple(toeplitz.ns)
    ws_block = ws.view(1, *ns_shape)
    
    # Create the basic weighted Toeplitz function
    def Gv(v):
        return ws * toeplitz(ws * v)
    
    def A_mean(beta):
        beta = beta.to(dtype=cdtype)
        wbeta = ws * beta
        Twbeta = toeplitz(wbeta)
        return ws * Twbeta + sigmasq_scalar * beta

    def A_var(gamma):
        is_batch = gamma.ndim > 1
        gamma = gamma.to(dtype=cdtype)
        shape_in = (gamma.shape[0], *ns_shape) if is_batch else (1, *ns_shape)
        g_block = gamma.view(shape_in)
        Tg = toeplitz(ws_block * g_block)
        out_block = ws_block * Tg / sigmasq_scalar + g_block
        return out_block.view(gamma.shape)
    
    return A_mean, A_var, Gv

# for stochastic variance estimation
def diag_sums_nd(A_apply, J, xis_flat, max_cg_iter, cg_tol,ws):
    """
    # computing c[r] where r is a vector offset....
    # TODO write a summary 
    """
    N, d_loc = xis_flat.shape
    M_loc = round(N ** (1 / d_loc))
    assert M_loc ** d_loc == N, "xis must lie on tensor grid"
    etas  = (torch.randint(0, 2, (J, N), device=xis_flat.device) * 2 - 1).to(xis_flat.dtype)
    rhs   = ws[None, :] * etas
    us    = BatchConjugateGradients(A_apply, rhs, x0=torch.zeros_like(rhs),
                                    tol=cg_tol, max_iter=max_cg_iter, early_stopping=True).solve()
    gammas = ws[None, :] * us
    shape = (J,) + (M_loc,) * d_loc
    gam_nd, eta_nd = gammas.view(shape), etas.view(shape)
    s_size = (2 * M_loc - 1,) * d_loc
    G = fftn(gam_nd, s=s_size, dim=tuple(range(1, d_loc + 1)))
    E = fftn(eta_nd, s=s_size, dim=tuple(range(1, d_loc + 1)))
    R = ifftn(G * torch.conj(E), s=s_size, dim=tuple(range(1, d_loc + 1)))
    return R.mean(dim=0)

def nufft_var_est_nd(est_sums, h_val, x_center, pts, eps_val):
    """
    Computes the posterior variance given estimated sums of the sums of vector-diagonal offsets 
    """
    B_loc, d_loc = pts.shape
    if est_sums.ndim != d_loc:
        raise ValueError("est_sums wrong dimensionality")
    phi = (2 * math.pi * h_val * (pts - x_center[None, :])).T.contiguous()
    return pff.finufft_type2(phi, est_sums, eps=eps_val, isign=+1, modeord=True).real


# ------------------------------------------------------------
# 2.  Hutchinson + Lanczos log‑det approx. 
# Only used for computing log marginal likelihood, which we don't use for gradients, but if you want to check periodically while learning.
# ------------------------------------------------------------
@torch.no_grad()
def logdet_slq(ws, sigma2, toeplitz,
               *, probes=1000, steps=100,
               dtype=torch.float64, device="cpu",
               eps=1e-18,n=None):
    """
    Estimate  log det(I + σ⁻² D T D).

    ws        (m,)            – diagonal weights
    sigma2    float           – noise variance σ²
    toeplitz  callable        – fast Toeplitz mat‑vec
    probes, steps             – Hutchinson & Lanczos params
    eps                       – floor for Ritz eigen‑values
    """
    if n is None:
        n = y.shape[0]
    ws  = ws.to(dtype=dtype, device=device)
    m   = ws.numel()
    σ2  = torch.as_tensor(sigma2, dtype=dtype, device=device)

    # Get the Gv function directly from setup_operators
    cdtype = _cmplx(dtype)
    _, _, Gv = setup_operators(ws, toeplitz, sigma2, cdtype)
    Av = lambda v: v + (1.0 / σ2) * Gv(v)

    logdet_acc = torch.zeros((), dtype=dtype, device=device)

    for _ in range(probes):
        # ---------- probe ------------------------------
        z = torch.empty(m, dtype=dtype, device=device).bernoulli_(0.5)
        z = z.mul_(2).sub_(1)           # ±1  Rademacher
        q = z / z.norm()                # safer than sqrt(m)

        # ---------- Lanczos ----------------------------
        alphas, betas = [], []
        q_prev, beta_prev = torch.zeros_like(q), torch.tensor(
            0.0, dtype=dtype, device=device
        )

        for _ in range(steps):
            v     = Av(q) - beta_prev * q_prev
            # Ensure both vectors have the same dtype before dot product
            q_conj = q.conj().to(dtype=v.dtype)
            alpha = torch.dot(q_conj, v).real
            v    -= alpha * q
            beta  = v.norm()

            alphas.append(alpha)
            betas.append(beta)

            if beta < 1e-12:            # converged early
                break
            q_prev, beta_prev = q, beta
            q = v / beta

        k = len(alphas)
        T = torch.zeros(k, k, dtype=dtype, device=device)
        for i in range(k):
            T[i, i] = alphas[i]
            if i < k - 1:
                T[i, i + 1] = betas[i]      # ← correct β indexing
                T[i + 1, i] = betas[i]

        # ---------- Gauss–Lanczos quadrature ----------
        evals, evecs = torch.linalg.eigh(T)
        evals.clamp_min_(eps)               # avoid log(≤0)
        w1   = evecs[0]
        quad = (w1**2 * torch.log(evals)).sum() * (z.norm() ** 2)
        logdet_acc += quad
    logdet = (logdet_acc / probes).item() + n * math.log(sigma2)
    return logdet

def compute_prediction_variance(x_new, xis, ws, A_var, cg_tol, max_cg_iter, variance_method, h, xcen, hutchinson_probes, nufft_eps, device, rdtype, cdtype):
    """
    Compute the prediction variance.

    Parameters
    ----------
    x_new : torch.Tensor
        New input points to predict at
    xis : torch.Tensor
        Frequency grid
    ws : torch.Tensor
        Quadrature weights
    A_var : callable
        Operator function for variance computation
    cg_tol : float
        Tolerance for conjugate gradient
    max_cg_iter : int
        Maximum iterations for conjugate gradient
    variance_method : str
        Method for variance estimation ('regular' or 'stochastic')
    h : float
        Grid spacing
    xcen : torch.Tensor
        Center of the domain
    hutchinson_probes : int
        Number of probes for Hutchinson estimator
    nufft_eps : float
        NUFFT accuracy
    device : torch.device
        Device to use
    rdtype : torch.dtype
        Real data type
    cdtype : torch.dtype
        Complex data type

    Returns
    -------
    torch.Tensor
        Prediction variance
    """
    TWO_PI = 2 * math.pi
    B, d = x_new.shape
    mtot = int(xis.shape[0]**(1/d))

    if variance_method.lower() == 'regular':
        # Exact CG with microbatching for large x_new
        microbatch = 8192  # Adjust as needed for hardware
        out = []
        for xb in torch.split(x_new, microbatch, dim=0):  # (b, d)
            fx = torch.exp(TWO_PI * 1j * (xb @ xis.T)).to(cdtype)  # (b, m)
            rhs = ws * fx.conj()  # (b, m)
            gamma = BatchConjugateGradients(
                A_var, rhs, x0=torch.zeros_like(rhs, dtype=cdtype),
                tol=cg_tol, max_iter=max_cg_iter, early_stopping=True
            ).solve()  # (b, m)
            s2b = torch.real((fx * (ws * gamma)).sum(dim=-1)).clamp_min(0.0)  # (b,)
            out.append(s2b)
            del fx, rhs, gamma
        return torch.cat(out, dim=0)
        
    elif variance_method.lower() == 'stochastic':
        # Hutchinson variance estimation
        J = hutchinson_probes
        
        # Compute diagonal sums
        est_sums = diag_sums_nd(
            A_var, J, xis, max_cg_iter, cg_tol, ws
        )
        
        # Compute variance
        pvar = nufft_var_est_nd(
            est_sums, h, xcen, x_new, nufft_eps
        )
        
        return pvar

    else:
        raise ValueError(f"Variance method '{variance_method}' not implemented. Choose 'regular' or 'stochastic'.")


## #TODO : kept an old version here for reference, do some more testing to see when the new version is better... 

# Original ToeplitzND implementation (commented out and replaced with optimized version)
# class ToeplitzND:
#     """
#     Fast d‑dimensional block‑Toeplitz convolution via FFT.
#     Caches the FFT of a fixed kernel and then performs
#     conv with any flat-or-block input.

#     Args:
#       v: complex tensor of shape (L1,...,Ld) with Li = 2*ni - 1
#       force_pow2: zero‑pad each dim to next power‑of‑2

#     Call with x either:
#       • flat: 1D tensor of length prod(ni), or
#       • block: tensor whose last d dims are exactly (n1,...,nd)
#     """
#     def __init__(self, v: torch.Tensor, *, force_pow2: bool = True):
#         # ensure complex
#         if not v.is_complex():
#             v = v.to(torch.complex128)
#         self.device = v.device

#         # dims and block‑sizes
#         self.Ls   = list(v.shape)                   # [L1,...,Ld]
#         self.ns   = [(L+1)//2 for L in self.Ls]     # [n1,...,nd]
#         self.size = prod(self.ns)                   # total block elements

#         # fft grid
#         if force_pow2:
#             self.fft_shape = [1 << (L-1).bit_length() for L in self.Ls]
#         else:
#             self.fft_shape = self.Ls.copy()

#         # pad & cache kernel FFT
#         pad = []
#         for L, F in zip(reversed(self.Ls), reversed(self.fft_shape)):
#             pad += [0, F - L]
#         v_pad     = nnf.pad(v.to(self.device), pad)
#         self.fft_v = torch.fft.fftn(v_pad, s=self.fft_shape,
#                                     dim=list(range(-len(self.Ls), 0)))

#         # slice indices for central block
#         self.starts = [n-1 for n in self.ns]
#         self.ends   = [st+n for st, n in zip(self.starts, self.ns)]


#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Accepts x with shape (..., self.size)  – flat batch
#                     or  (..., *self.ns)       – block batch
#         and returns a tensor of the same outer-batch shape.
#         """
#         x = x.to(self.device)
#         orig_flat = False                     # remember how input looked

#         # ---------- detect layout ----------------------------------------
#         if x.shape[-1] == self.size:          # (..., N)  – flat batch
#             orig_flat = True
#             batch_shape = x.shape[:-1]
#             x = x.view(*batch_shape, *self.ns)
#         elif list(x.shape[-len(self.ns):]) == self.ns:
#             batch_shape = x.shape[:-len(self.ns)]
#         else:
#             raise AssertionError(
#                 f"Expected trailing dim {self.size} or block {tuple(self.ns)}, got {tuple(x.shape)}"
#             )

#         # ---------- promote to complex -----------------------------------
#         if not x.is_complex():
#             x = x.to(torch.complex128)

#         # ---------- zero-pad to FFT grid ---------------------------------
#         pad = []
#         for n, F in zip(reversed(self.ns), reversed(self.fft_shape)):
#             pad += [0, F - n]                 # torch.nn.functional.pad wants (dim_k_before, dim_k_after) pairs
#         x_pad = nnf.pad(x, pad)

#         # ---------- FFT convolution --------------------------------------
#         x_fft = torch.fft.fftn(x_pad, s=self.fft_shape,
#                             dim=list(range(-len(self.ns), 0)))
#         y     = torch.fft.ifftn(self.fft_v * x_fft,
#                                 s=self.fft_shape,
#                                 dim=list(range(-len(self.ns), 0)))

#         # ---------- crop central block -----------------------------------
#         slices = [slice(None)] * y.ndim       # keep batch dims
#         for st, en in zip(self.starts, self.ends):
#             slices[-len(self.starts) + len(slices) - y.ndim]  # just to calm linters
#         slices = [slice(None)] * (y.ndim - len(self.ns))
#         for st, en in zip(self.starts, self.ends):
#             slices.append(slice(st, en))
#         y = y[tuple(slices)]

#         # ---------- restore original layout ------------------------------
#         # ---------- restore original layout ------------------------------
#         if orig_flat:
#             # Either option below is fine; reshape is the simpler one.
#             y = y.reshape(*batch_shape, self.size)          # works with non-contiguous tensors
#             # y = y.contiguous().view(*batch_shape, self.size)  # alternative

#         return y
# 
#     