from typing import Dict, Optional, Tuple

import torch.nn.functional as nnf
from torch.fft import fftn, ifftn
import torch
from utils.kernels import get_xis
from cg import ConjugateGradients, BatchConjugateGradients
import time
from torch.profiler import profile, record_function, ProfilerActivity
import finufft
import pytorch_finufft.functional as pff
import numpy as np
import math
import torch.nn.functional as nnf
from torch.optim import Adam
from torch import nn
from math import prod
from typing import Dict, Optional, Callable
from torch.fft import fftn, ifftn


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

## main function

def efgp_nd(x: torch.Tensor, y: torch.Tensor, sigmasq: float,
            kernel: object,  # kernel.spectral_density(xis)
            eps: float,
            x_new: torch.Tensor,
            do_profiling: bool = False,
            nufft_eps: float = 1e-4,
            rdtype: torch.dtype = torch.float32,
            opts: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Equispaced Fourier Gaussian Process Regression for N-D data with optional variance estimation.

    Parameters
    ----------
    estimate_variance : bool  (opts)
        If ``True`` compute predictive variances. Otherwise ``ytrg['var']`` is NaN.
    variance_method   : {'regular', 'stochastic'}  (opts)
        * ``regular``   – exact CG solve per test point.
        * ``stochastic`` – Hutchinson‑FFT diagonal‑sum estimator.
    """

    if opts is None:
        opts = {}
    TWO_PI = 2 * math.pi

    # -------------------- Options ------------------------------------
    cg_tol            = opts.get("cg_tolerance", 1e-4)
    max_cg_iter       = opts.get("max_cg_iter", 5_000)
    estimate_variance = opts.get("estimate_variance", False)
    variance_method   = opts.get("variance_method", "regular").lower()
    hutchinson_probes = opts.get("hutchinson_probes", 8_000)

    x_new_dim_for_var = opts.get("x_new_dim_for_var", 0)
    if x_new_dim_for_var >= x_new.shape[1]:
        raise ValueError(
            f"x_new_dim_for_var ({x_new_dim_for_var}) out of bounds for x_new dim {x_new.shape[1]}")

    # -----------------------------------------------------------------
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if do_profiling and torch.cuda.is_available() else [ProfilerActivity.CPU]
    with profile(activities=activities, record_shapes=True):

        # ---------- Casts & Devices ----------------------------------
        device  = x.device
        rdtype  = rdtype
        cdtype  = _cmplx(rdtype)
        sigmasq_scalar = float(sigmasq)
        x       = x.to(device=device, dtype=rdtype)
        y       = y.to(device=device, dtype=rdtype)
        x_new   = x_new.to(device=device, dtype=rdtype)
        N, d    = x.shape
        B       = x_new.shape[0]
        if x_new.shape[1] != d:
            raise ValueError(f"Dim mismatch: x ({d}) vs x_new ({x_new.shape[1]})")

        # ---------- Geometry & Quadrature ----------------------------
        with record_function("geometry_quadrature"):
            x_all = torch.cat((x, x_new), dim=0)
            L     = torch.max(torch.max(x_all, dim=0).values - torch.min(x_all, dim=0).values)
            if L <= 1e-9:
                L = torch.tensor(1.0, device=device, dtype=rdtype)
            xis_1d, h_float, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L.item(), use_integral=True, l2scaled=False)
            h         = torch.tensor(h_float, device=device, dtype=rdtype)
            xis_1d    = xis_1d.to(device=device, dtype=rdtype)
            grids     = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
            xis       = torch.stack(grids, dim=-1).view(-1, d)            # (M, d)
            M         = xis.shape[0]
            spectral_vals = kernel.spectral_density(xis).to(dtype=cdtype)
            ws        = torch.sqrt(spectral_vals * (h ** d))               # (M,)
            ws = ws.to(dtype=cdtype)
        # ---------- NUFFT Setup --------------------------------------
        with record_function("nufft_setup"):
            OUT        = (mtot,) * d
            xcen       = torch.zeros(d, device=device, dtype=rdtype)
            ## TODO I have no idea how contiguous works 
            phi_train  = (TWO_PI * h * (x      - xcen)).T.contiguous()
            phi_test   = (TWO_PI * h * (x_new  - xcen)).T.contiguous()

            def finufft1(phi_in, vals):
                phi_in = phi_in.contiguous()
                vals = vals.contiguous()
                fk = pff.finufft_type1(phi_in, vals.to(dtype=cdtype), OUT, eps=nufft_eps, isign=-1, modeord=False)
                if fk.dtype != cdtype:
                    fk = fk.to(dtype=cdtype)
                return fk.view(-1)

            def finufft2(phi_in, fk_flat):
                phi_in = phi_in.contiguous()
                fk_flat = fk_flat
                fk_nd = fk_flat.view(OUT).to(dtype=cdtype).contiguous()
                if fk_nd.dtype != cdtype:
                    fk_nd = fk_nd.to(dtype=cdtype)
                return pff.finufft_type2(phi_in, fk_nd, eps=nufft_eps, isign=+1, modeord=False)

        # ---------- RHS for Mean -------------------------------------
        with record_function("rhs_mean"):
            right_hand_side = ws * finufft1(phi_train, y)

        # ---------- Toeplitz Operator --------------------------------
        with record_function("toeplitz_setup"):
            m_conv       = (mtot - 1) // 2
            v_kernel     = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
            toeplitz     = ToeplitzND(v_kernel, force_pow2=True)
            ns_shape     = tuple(toeplitz.ns)
            ws_block     = ws.view(1, *ns_shape)

            def A_mean(beta):
                beta   = beta.to(dtype=cdtype)
                wbeta  = ws * beta
                Twbeta = toeplitz(wbeta)
                return ws * Twbeta + sigmasq_scalar * beta

            def A_var(gamma):
                is_batch   = gamma.ndim > 1
                gamma      = gamma.to(dtype=cdtype)
                shape_in   = (gamma.shape[0], *ns_shape) if is_batch else (1, *ns_shape)
                g_block    = gamma.view(shape_in)
                Tg         = toeplitz(ws_block * g_block)
                out_block  = ws_block * Tg / sigmasq_scalar + g_block
                return out_block.view(gamma.shape)

        # ---------- Solve for Mean -----------------------------------
        with record_function("cg_solve_mean"):
            beta = ConjugateGradients(A_mean, right_hand_side, x0=torch.zeros_like(right_hand_side, dtype=cdtype),
                                      tol=cg_tol, early_stopping=True).solve()

        # ---------- Predictive Mean ----------------------------------
        with record_function("predict_mean"):
            yhat = finufft2(phi_test, ws * beta).real
        ytrg = {"mean": yhat}

        # ---------- Predictive Variance ------------------------------
        with record_function("predict_variance"):
            if not estimate_variance:
                # user didn't request variance → fill with NaNs
                ytrg["var"] = torch.full((B,), float("nan"), device=device, dtype=rdtype)
            else:
                if variance_method == "regular":
                    # ---- exact CG with microbatching for large x_new ----
                    microbatch = 8192  # Adjust as needed for your hardware
                    out = []
                    for xb in torch.split(x_new, microbatch, dim=0):  # (b, d)
                        fx = torch.exp(TWO_PI * 1j * (xb @ xis.T)).to(cdtype)  # (b, m)
                        rhs = ws * fx.conj()  # (b, m)
                        gamma = BatchConjugateGradients(
                            A_var, rhs, x0=torch.zeros_like(rhs, dtype=cdtype),
                            tol=cg_tol, early_stopping=True
                        ).solve()  # (b, m)
                        s2b = torch.real((fx * (ws * gamma)).sum(dim=-1)).clamp_min(0.0)  # (b,)
                        out.append(s2b)
                        del fx, rhs, gamma
                    ytrg["var"] = torch.cat(out, dim=0)

                # if variance_method == "regular":
                #     # ---- exact CG ----------------------------------
                #     f_x   = torch.exp(TWO_PI * 1j * torch.matmul(x_new, xis.T))
                #     rhs_v = ws.unsqueeze(0) * f_x.conj()
                #     gamma = BatchConjugateGradients(A_var, rhs_v, x0=torch.zeros_like(rhs_v, dtype=cdtype),
                #                                     tol=cg_tol, early_stopping=True).solve()
                #     s2    = torch.real(torch.einsum('bm,m,bm->b', f_x, ws, gamma)).clamp_min_(0.0)
                #     ytrg["var"] = s2
                elif variance_method == "stochastic":
                    # ---- Hutchinson / FFT --------------------------
                    J = hutchinson_probes
                    est_grid = diag_sums_nd(A_var, J, xis, max_cg_iter, cg_tol,ws)
                    s_est    = nufft_var_est_nd(est_grid, h, torch.zeros(d, device=device, dtype=rdtype), x_new, nufft_eps)
                    ytrg["var"] = s_est.clamp_min_(0.0)

                else:
                    raise ValueError("variance_method must be 'regular' or 'stochastic'")

        # ---------- Optional Log Marginal -----------------------------
        ## TODO: implement this efficiently
        if opts.get("get_log_marginal_likelihood", False):
            ytrg["log_marginal_likelihood"] = torch.tensor(float("nan"), device=device, dtype=rdtype)

    # ----------------- Return ----------------------------------------
    return beta, xis, ytrg, ws, toeplitz



def efgpnd_gradient_batched(
        x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
        *, nufft_eps=6e-8, cg_tol = 1e-4, early_stopping = True, device=None,
        do_profiling=False):
    """
    Gradient of the N‑D GP log‑marginal likelihood estimated with
    Hutchinson trace + CG
    
    Parameters
    ----------
    do_profiling : bool
        If True, use torch.profiler to profile the computation and print results
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
            toeplitz   = ToeplitzND(v_kernel)               # cached once

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
    
    return grad.real  # returns in the same order as hypers in kernel class 

# -----------------------------------------------------------------------------
# Parameter container that wraps any kernel and a noise variance
# -----------------------------------------------------------------------------
class GPParams(nn.Module):
    def __init__(self, kernel, init_sig2):
        """
        kernel: object exposing
          - iter_hypers() -> yields (name, value)
          - set_hyper(name, new_value)
        init_sig2: initial noise variance (float or tensor)
        """
        super().__init__()
        self.kernel = kernel
        
        # collect kernel hyper‐names & initial values
        hypers = list(kernel.iter_hypers())
        self.hypers_names = [name for name, _ in hypers]
        init_vals = [float(val) for _, val in hypers]
        init_vals.append(float(init_sig2))       # add noise variance last
        
        # raw = log of all positives, packed into one vector
        init_raw = torch.log(torch.tensor(init_vals, dtype=torch.get_default_dtype()))
        self.raw = nn.Parameter(init_raw)        # shape=(D+1,)
    
    @property
    def pos(self):
        "All positive parameters: kernel‐hypers followed by noise variance"
        return self.raw.exp()                    # shape=(D+1,)
    
    @property
    def sig2(self):
        "Noise variance (last entry)"
        return self.pos[-1]
    
    def sync_all_parameters(self):
        """
        Write current positive hypers back into self.kernel and return noise variance.
        This consolidates both kernel hyperparameter and noise variance synchronization.
        
        Returns:
            noise_variance: The current noise variance value
        """
        pos_vals = self.pos.detach().cpu().tolist()
        # Update kernel hyperparameters
        for i, name in enumerate(self.hypers_names):
            self.kernel.set_hyper(name, pos_vals[i])
        # Return noise variance
        return self.sig2.detach().clone()

class EFGPND(nn.Module):
    """
    Equispaced‑Fourier Gaussian Process (EFGP) regression in d dimensions.

    Parameters
    ----------
    x, y        : training inputs / targets  (stored but *not* copied)
    kernel      : object providing `spectral_density` (passed straight to `efgp_nd`)
    sigmasq     : scalar observation noise variance
    eps         : quadrature accuracy parameter (passed to `get_xis`)
    opts        : dict with CG tolerances, variance method … exactly as in the
                  plain `efgp_nd` version (can be edited after construction)
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
        super().__init__()  # Initialize nn.Module parent
        
        self.x        = x
        self.y        = y
        self.rdtype   = x.dtype
        self.device   = x.device
        self.eps      = eps
        self.nufft_eps = nufft_eps
        self.opts     = {} if opts is None else opts
        
        # Handle string kernel specification
        if isinstance(kernel, str):
            # Import kernel classes here to avoid circular imports
            from kernels.squared_exponential import SquaredExponential
            from kernels.matern import Matern32, Matern52
            
            # Get the dimension from the input data
            if x.ndim == 1:
                dimension = 1
            else:
                dimension = x.shape[1]
                
            # Create the kernel based on the string identifier
            kernel_map = {
                "SquaredExponential": SquaredExponential,
                "SE": SquaredExponential,
                "RBF": SquaredExponential,
                "Matern32": Matern32,
                "Matern52": Matern52
            }
            
            if kernel not in kernel_map:
                raise ValueError(f"Unknown kernel string: {kernel}. Available options: {list(kernel_map.keys())}")
            
            # Create kernel with default parameters (will be estimated)
            kernel_class = kernel_map[kernel]
            
            # Start with default values that will be overridden by estimation
            kernel = kernel_class(dimension=dimension, lengthscale=1.0, variance=1.0)
            
            # Force parameter estimation when kernel is specified as string
            estimate_params = True
            
        self.kernel = kernel
        
        # Store data bounds for gradient computation
        self.x0 = self.x.min(dim=0).values
        self.x1 = self.x.max(dim=0).values

        # Estimate hyperparameters if needed
        if estimate_params and (sigmasq is None or hasattr(kernel, 'estimate_hyperparameters')):
            try:
                # Try to estimate hyperparameters using the kernel's method
                estimated_lengthscale, estimated_variance, estimated_sigmasq = kernel.estimate_hyperparameters(x, y)
                
                # Update kernel hyperparameters if they should be estimated
                if hasattr(kernel, 'lengthscale'):
                    kernel.lengthscale = estimated_lengthscale
                if hasattr(kernel, 'variance'):
                    kernel.variance = estimated_variance
                
                # Use estimated noise variance if none was provided
                if sigmasq is None:
                    sigmasq = estimated_sigmasq
            except NotImplementedError:
                # If the kernel doesn't implement the method, use a default value
                if sigmasq is None:
                    # Default to 10% of the variance of y
                    sigmasq = 0.1 * torch.var(y).item()

        # If sigmasq is still None, use a default value
        if sigmasq is None:
            sigmasq = 0.1 * torch.var(y).item()

        # --- placeholders populated by `fit` -------------------------------------------------
        self._beta       = None        # Fourier weights
        self._xis        = None        # frequency grid
        self._ws         = None        # quadrature weights √S(ξ) h^{d/2}
        self._toeplitz   = None        # convolution operator object
        self._fitted     = False
        self.train_mean  = None        # f̂(x_i) on training set
        self._cache_full = None        # (beta, xis, ytrg, ws, toeplitz) for reuse
        
        # --- Set up parameters for optimization using GPParams ---
        # Initialize and register parameters as part of the module
        self._gp_params = GPParams(kernel, init_sig2=sigmasq)
        
        # Register GPParams as a submodule so it's parameters are accessible via parameters()
        self.add_module("gp_params", self._gp_params)
        
        # Initial sync to ensure everything is in sync
        self.sync_parameters()
        
        # Store registered optimizers
        self._registered_optimizers = []
    
    def register_optimizer(self, optimizer):
        """
        Register an optimizer to automatically sync parameters after each step.
        
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
            
            # Sync parameters back to kernel using the consolidated method
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
        return self._gp_params.sig2.detach().clone()
    
    def sync_parameters(self):
        """
        Synchronize parameters between GPParams, kernel, and sigmasq.
        Call this after manual parameter updates to ensure consistency.
        """
        self._internal_sigmasq = self._gp_params.sync_all_parameters()
        return self
        
    def compute_gradients(
        self,
        *,
        trace_samples: int = 10,
        do_profiling: bool = False,
        nufft_eps: Optional[float] = None,
        apply_gradients: bool = True,
        **kwargs
    ) -> torch.Tensor:
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
            
        Returns
        -------
        grads : torch.Tensor
            Gradients for kernel hyperparameters and sigmasq in log space
        """
        # Ensure parameters are synchronized
        self.sync_parameters()
        
        # Set default nufft_eps if not provided
        if nufft_eps is None:
            nufft_eps = self.eps * 0.1
            
        # Compute gradients with respect to the original parameters
        grads = efgpnd_gradient_batched(
            self.x, self.y,
            sigmasq=self._gp_params.sig2,
            kernel=self.kernel,
            eps=self.eps,
            trace_samples=trace_samples,
            x0=self.x0, x1=self.x1,
            do_profiling=do_profiling,
            nufft_eps=nufft_eps,
            **kwargs
        )
        
        # Transform to raw space gradient via chain rule: ∂L/∂raw_i = (∂L/∂param_i) * param_i
        pos_vec = self._gp_params.pos.to(self.device)
        raw_grad = torch.stack([
            grads[i].detach().to(self.device) * pos_vec[i]
            for i in range(len(grads))
        ], dim=0)
        
        # Automatically apply gradients if requested
        if apply_gradients:
            self._gp_params.raw.grad = raw_grad.clone()
        
        return raw_grad

    def fit(self) -> "EFGPND":
        """
        Solves the linear system once and stores everything needed for
        fast predictions at new x*.  Returns `self` for chaining.
        """
        # We only need the mean here; keep variance off for speed
        opts = dict(self.opts)               # shallow copy
        opts.setdefault("estimate_variance", False)
        
        # Ensure parameters are synchronized
        self.sync_parameters()
        
        beta, xis, ytrg, ws, toeplitz = efgp_nd(
            x        = self.x,
            y        = self.y,
            sigmasq  = self._gp_params.sig2,
            kernel   = self.kernel,
            eps      = self.eps,
            x_new    = self.x,               # ← predict on training set
            opts     = opts,
            do_profiling = False,
            nufft_eps = self.nufft_eps,
            rdtype = torch.float32
        )

        self._beta       = beta
        self._xis        = xis
        self._ws         = ws
        self._toeplitz   = toeplitz
        self.train_mean  = ytrg["mean"]
        self._cache_full = (beta, xis, ytrg, ws, toeplitz)
        self._fitted     = True
        return self

    def predict(
        self,
        x_new: torch.Tensor,
        *,
        return_variance: bool = True,
        variance_method: str = "stochastic",
        hutchinson_probes: int = 1_000,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict posterior mean (and optionally variance) at `x_new`.

        Variance is reused from the cached Toeplitz system, so we avoid the
        expensive CG solve when only the mean is desired.
        """
        if not self._fitted:
            raise RuntimeError("Call `.fit()` before `.predict()`.")

        # Ensure parameters are synchronized
        self.sync_parameters()
        
        # Re‑use what we have; toggle variance computation only
        opts = dict(self.opts)
        opts.update(
            estimate_variance = return_variance,
            variance_method   = variance_method,
            hutchinson_probes = hutchinson_probes,
        )

        _, _, ytrg, _ws, _T = efgp_nd(
            x        = self.x,
            y        = self.y,
            sigmasq  = self._gp_params.sig2,
            kernel   = self.kernel,
            eps      = self.eps,
            x_new    = x_new,
            opts     = opts,
            do_profiling = False,
            nufft_eps = self.nufft_eps,
        )
        mean = ytrg["mean"]
        var  = ytrg["var"] if return_variance else None
        return mean, var

    def optimize_hyperparameters(
        self,
        *,
        # --- search grid ------------------------------------------------
        epsilon_values         = (1e-2,),      # quadrature accuracy grid
        trace_samples_values   = (10,),        # Hutchinson probes grid
        # --- optimization options ---------------------------------------
        optimizer              = 'Adam',         # optional optimizer (e.g., 'Adam' or Adam instance)
        lr: Optional[float]    = None,         # learning rate (will be auto-scaled if None)
        base_lr: float         = 0.05,         # base learning rate for auto-scaling
        max_iters: int         = 50,
        min_lengthscale: float = 5e-3,
        # --- profiling options -----------------------------------------
        profile_gradient: bool = False,        # Enable profiling of gradient computation
        profile_first_iter: bool = True,       # Profile only the first iteration
        profile_last_iter: bool = False,       # Profile only the last iteration

        **gkwargs,                             # forwarded to efgpnd_gradient_batched
    ):
        """
        Optimize hyperparameters using either vanilla SGD or a provided optimizer.
        
        
        Parameters
        ----------
        optimizer : optional
            If provided, uses the enhanced optimization scheme with GPParams.
            Can be either:
            - String: 'Adam' for Adam optimizer
            - Optimizer instance: direct optimizer object to use
            - None: Uses vanilla SGD optimization
        """
        device = self.x.device
        rdtype = self.x.dtype

        # Auto-scale learning rate  
        n_samples = self.x.shape[0]
        if lr is None:
            if optimizer == 'Adam':
                lr = 1e-1
            else:
                # Scale learning rate inversely with number of samples
                # Use a non-linear scaling that prevents extremely small learning rates for very large datasets
                # and provides a smoother transition between dataset sizes
                lr = base_lr / (1.0 + math.sqrt(n_samples / 100.0))
                print(f"Auto-scaled learning rate: {lr:.6f} (base_lr={base_lr}, n_samples={n_samples})")

        # Store original kernel and parameters
        init_kernel = self.kernel.model_copy()
        init_sigmasq = self._gp_params.sig2.detach().clone()
        
        logs = []                                       # ⇐ stored afterwards

        # Check if optimizer is a string
        use_enhanced_opt = optimizer is not None
        
        for EPSILON in epsilon_values:
            for J in trace_samples_values:
                # Reset kernel and sigmasq for this grid setting
                self.kernel = init_kernel.model_copy()
                self._gp_params = GPParams(self.kernel, init_sig2=init_sigmasq.item()).to(device)
                
                # Re-register the GPParams module
                if hasattr(self, "gp_params"):
                    delattr(self, "gp_params")
                self.add_module("gp_params", self._gp_params)
                
                start = time.time()
                
                # Track hyperparameters for logging
                tracked_hypers = {name: [] for name in self.kernel.hypers}
                tracked_hypers['sigmasq'] = []  # Add noise parameter separately
                
                # ---- Optimize using the provided optimizer or vanilla SGD -----
                if use_enhanced_opt:
                    # ------ Enhanced optimization with GPParams -------
                    
                    # Create optimizer based on the optimizer parameter
                    if isinstance(optimizer, str):
                        if optimizer.lower() == 'adam':
                            optimizer_instance = Adam(self.parameters(), lr=lr)
                        else:
                            raise ValueError(f"Unsupported optimizer string: {optimizer}. Currently supporting: 'adam'")
                    else:
                        # Use the provided optimizer directly
                        optimizer_instance = optimizer
                    
                    # Register the optimizer for automatic parameter syncing
                    self.register_optimizer(optimizer_instance)
                    
                    # Prepare names for tracking
                    names = self._gp_params.hypers_names + ["sigmasq"]
                    history = {n: [] for n in names}
                    
                    # Optimization loop
                    for it in range(max_iters):
                        # Record current values of all hyperparameters for tracking
                        for name, value in self.kernel.iter_hypers():
                            tracked_hypers[name].append(float(value))
                        tracked_hypers['sigmasq'].append(float(self._gp_params.sig2.item()))
                        
                        # a) sync current hypers back to kernel
                        self.sync_parameters()
                        
                        # b) compute gradients
                        raw_grad = self.compute_gradients(
                            trace_samples=5 if it < max_iters*0.8 else J,
                            do_profiling=profile_gradient and (
                                (profile_first_iter and it == 0) or 
                                (profile_last_iter and it == max_iters - 1) or
                                (not profile_first_iter and not profile_last_iter)
                            ),
                            nufft_eps=EPSILON*0.1,
                            apply_gradients=True,
                            **gkwargs
                        )
                        
                        # c) optimizer update
                        optimizer_instance.zero_grad()
                        optimizer_instance.step()
                        
                        # d) record history
                        pos_list = self._gp_params.pos.detach().cpu().tolist()
                        for name, val in zip(names, pos_list):
                            history[name].append(val)
                        
                        # f) optional logging
                        if it % 10 == 0:
                            # Get current values for printing
                            lengthscale = self.kernel.get_hyper('lengthscale')
                            variance = self.kernel.get_hyper('variance')
                            print(f"[ε={EPSILON} | J={J}] iter {it:>3}  "
                                f"ℓ={lengthscale:.4g}  "
                                f"σ_f²={variance:.4g}  σ_n²={self._gp_params.sig2.item():.4g}")
                            print(f"grad: {raw_grad}")


                ## TODO can probably remove this        
                else:
                    # ------ Original SGD implementation -------
                    lr_curr = lr
                    
                    for it in range(max_iters):
                        # Record current values of all hyperparameters
                        for name, value in self.kernel.iter_hypers():
                            tracked_hypers[name].append(float(value))
                        tracked_hypers['sigmasq'].append(float(self._gp_params.sig2.item()))
                        
                        # Determine profiling
                        do_profile_this_iter = profile_gradient and (
                            (profile_first_iter and it == 0) or 
                            (profile_last_iter and it == max_iters - 1) or
                            (not profile_first_iter and not profile_last_iter)
                        )
                        
                        # Get gradients in the original way (not using raw_grad)
                        grad = efgpnd_gradient_batched(
                            self.x, self.y, 
                            sigmasq=self._gp_params.sig2,
                            kernel=self.kernel,
                            eps=EPSILON,
                            trace_samples=5 if it < max_iters*0.8 else J,
                            x0=self.x0, x1=self.x1,
                            do_profiling=do_profile_this_iter,
                            **gkwargs,
                            nufft_eps=EPSILON*0.1
                        )
                        
                        # Update kernel hyperparameters directly
                        for i, (name, value) in enumerate(self.kernel.iter_hypers()):
                            # Apply log-space update to avoid negative values
                            new_value = math.exp(math.log(value) - lr_curr * value * grad[i].item())
                            
                            # Apply minimum value constraints based on hyperparameter
                            if name == 'lengthscale':
                                new_value = max(new_value, min_lengthscale)
                            else:
                                new_value = max(new_value, 1e-5)
                                
                            # Update the hyperparameter directly
                            self.kernel.set_hyper(name, new_value)
                            
                            # Also update the corresponding raw parameter in _gp_params
                            with torch.no_grad():
                                self._gp_params.raw[i] = torch.log(torch.tensor(new_value, dtype=rdtype, device=device))
                        
                        # Update noise parameter directly
                        new_noise = math.exp(math.log(self._gp_params.sig2.item()) - lr_curr * self._gp_params.sig2.item() * grad[-1].item())
                        new_noise = max(new_noise, 1e-5)
                        
                        # Update the parameter in _gp_params
                        with torch.no_grad():
                            self._gp_params.raw[-1] = torch.log(torch.tensor(new_noise, dtype=rdtype, device=device))
                        
                        # No need to call sync_parameters() since we've updated both the kernel and _gp_params directly
                        
                        # adaptive LR guard against runaway ℓ
                        if it < 3 and self.kernel.get_hyper('lengthscale') < 0.05:
                            lr_curr *= 0.5
                        
                        if it % 10 == 0:
                            # Get current values for printing
                            lengthscale = self.kernel.get_hyper('lengthscale')
                            variance = self.kernel.get_hyper('variance')
                            print(f"[ε={EPSILON} | J={J}] iter {it:>3}  "
                                f"ℓ={lengthscale:.4g}  "
                                f"σ_f²={variance:.4g}  σ_n²={self._gp_params.sig2.item():.4g}")
                            print(f"grad: {grad}")

                # ---- Store log information -------------------------------
                elapsed = time.time() - start

                logs.append({
                    "epsilon"           : EPSILON,
                    "trace_samples"     : J,
                    "tracked_hyperparameters": tracked_hypers,
                    "tracked_lengthscale": tracked_hypers.get('lengthscale', []),
                    "tracked_variance"   : tracked_hypers.get('variance', []),
                    "tracked_noise"      : tracked_hypers.get('sigmasq', []),
                    "iters"              : max_iters,
                    "time_sec"           : elapsed,
                    "final_kernel"       : self.kernel.model_copy(),
                    "final_noise"        : self._gp_params.sig2.detach().clone(),
                    "objective"          : float("nan"),  # No closed-form objective available
                })

                print(f"└─ finished ε={EPSILON}, J={J} in {elapsed:.1f}s")

        # ----------------------------------------------------------------
        # store results in the model
        # ----------------------------------------------------------------
        self.training_log = logs
        return self   # so you can chain: model.optimize_hyperparameters(...).fit()


    # convenience alias
    def fit_predict(self, x_new: torch.Tensor, **pred_kw):
        """Shorthand for `.fit().predict(...)`."""
        return self.fit().predict(x_new, **pred_kw)


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
# def compute_convolution_vector_vectorized_dD(m: int, x: torch.Tensor, h: float) -> torch.Tensor:
#     """
#     Multi‑D type‑1 NUFFT convolution vector:
#       v[k1,...,kd] = sum_n exp(2πi <k, x_n>)
#     """
#     device      = x.device
#     dtype_real  = x.dtype
#     dtype_cmplx = torch.complex64 if dtype_real == torch.float32 else torch.complex128
#     if x.ndim == 1:
#         x = x[:, None]
#     N, d        = x.shape
#     eps         = 1e-15
# 
#     # (d, N) array of phases
#     phi = (2 * math.pi * h * x).T.contiguous().to(dtype_real)
# 
#     # all weights = 1 + 0i
#     c = torch.ones(N, dtype=dtype_cmplx, device=device)
# 
#     # output grid size in each of the d dims
#     OUT = tuple([4*m + 1] * d)
# 
#     v = pff.finufft_type1(
#         phi,    # (d, N)
#         c,      # (N,)
#         OUT,
#         eps=eps,
#         isign=-1,
#         modeord=False
#     )
#     return v
# # ──────────────────────────────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────────────────────────────
def _cmplx(real_dtype: torch.dtype) -> torch.dtype:
    """Matching complex dtype for a given real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


class ToeplitzND:
    """
    ToeplitzND class for multidimensional Toeplitz matrix-vector products using FFTs.
    """
    
    def __init__(self, v: torch.Tensor, *, force_pow2: bool = False, precompute_fft: bool = True):
        """
        Initialize the Toeplitz operator with the first column/row vector.
        
        Args:
            v: First column/row of the Toeplitz matrix, zero-padded to full convolution size
            force_pow2: If True, use power of 2 for FFT sizes (default: False)
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

def _cmplx(real_dtype: torch.dtype) -> torch.dtype:
    """Matching complex dtype for a given real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128




## These are just for testing and sanity checking, not used in EFGNPD 
def compute_gradients_truncated(x, y, sigmasq, kernel, EPSILON):
    """
    Gradients with approximated kernel, exact trace-- forming the matrices directly to sanity check.
    """
    # sigmasq = torch.tensor(0.1, dtype=torch.float64)  # noise variance
    # kernel = SquaredExponential(dimension=1, lengthscale=0.1, variance=1.0)

    # Flatten data to 1D.
    # if x is 1d unsqueeze
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    # x = x.to(dtype=torch.float64).flatten()   # shape: (N,)
    # y = y.to(dtype=torch.float64).flatten()     # shape: (N,)
    # x_new = torch.linspace(0, 5, 1000, dtype=torch.float64)
    d = x.shape[1]
    cdtype = _cmplx(x.dtype)
    x0 = x.min(dim=0).values  
    x1 = x.max(dim=0).values  

    domain_lengths = x1 - x0
    L = domain_lengths.max()
    N = x.shape[0]
    # print(EPSILON)
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=EPSILON, L=L, use_integral=True, l2scaled=False)
    # print(xis_1d.shape)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij') # makes tensor product Jm 
    xis = torch.stack(grids, dim=-1).view(-1, d) 
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d) # (mtot**d,1)

    D = torch.diag(ws).to(dtype=cdtype)         # D: (M, M)

    # Form design features F (N x M): F[n,m] = exp(2pi i * xis[m] * x[n])
    # F = torch.exp(1j * 2 * math.pi * torch.outer(x, xis)).to(dtype=torch.complex128)
    F = torch.exp(1j * 2 * torch.pi * (x @ xis.T))

    # Compute approximate kernel: K = F * D^2 * F^*.
    D2 = D @ D  # This is just diag(ws^2)
    K = F @ D2 @ F.conj().transpose(-2, -1)  # shape: (N, N)
    C = K + sigmasq * torch.eye(N, dtype=cdtype)  # add noise term

    # Directly invert C and compute alpha.
    C_inv = torch.linalg.inv(C)
    alpha = C_inv @ y.to(dtype=cdtype)  # shape: (N,)

    # Compute derivative of the kernel with respect to the kernel hyperparameters.
    # Let spectral_grad = kernel.spectral_grad(xis), shape: (M, n_params)
    spectral_grad = kernel.spectral_grad(xis)  # shape: (M, n_params)
    # Then dK/dtheta for each kernel hyperparameter i is approximated as:
    # dK/dtheta_i = F * diag( h * spectral_grad(:, i) ) * F^*
    dK_dtheta_list = []
    n_params = spectral_grad.shape[1]
    for i in range(n_params):
        dK_i = F @ torch.diag((h**d * spectral_grad[:, i]).to(dtype=cdtype)) @ F.conj().transpose(-2, -1)
        dK_dtheta_list.append(dK_i)
    # The derivative with respect to the noise parameter is simply the identity.
    dK_dtheta_list.append(torch.eye(N, dtype=cdtype))
    n_total = n_params + 1

    # Compute gradient for each hyperparameter using:
    # grad = 0.5 * [trace(C_inv * dK/dtheta) - alpha^H * (dK/dtheta) * alpha]
    grad = torch.zeros(n_total, dtype=cdtype)
    for i in range(n_total):
        if i < n_params:
            term1 = torch.trace(C_inv @ dK_dtheta_list[i])
            term2 = (alpha.conj().unsqueeze(0) @ (dK_dtheta_list[i] @ alpha.unsqueeze(1))).squeeze()
        else:  # noise derivative: dC/d(sigmasq) = I
            term1 = torch.trace(C_inv)
            term2 = (alpha.conj().unsqueeze(0) @ alpha.unsqueeze(1)).squeeze()
        grad[i] = 0.5 * (term1 - term2)
        # print('term1:' ,term1.real)
        # print('term2:', term2.real) 

    # Print the gradients (real parts)
    # print("(Truncated) Direct inversion gradient:")
    # print(f"  dNLL/d(lengthscale) = {grad[0].real.item():.6f}")
    # if n_params > 1:
    #     print(f"  dNLL/d(variance)    = {grad[1].real.item():.6f}")
    # print(f"  dNLL/d(noise)       = {grad[-1].real.item():.6f}")
    true_grad = grad.clone()
    # print("term 1: ", term1.real, "term 2: ", term2.real)
    return true_grad.real

# 3. Define the squared-exponential kernel.
def squared_exponential_kernel(x1, x2, lengthscale, variance):
    # Ensure inputs are 2D
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(1)
    # Compute pairwise squared Euclidean distances.
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)   # shape: (n1, n2, d)
    dist_sq = (diff ** 2).sum(dim=2)             # shape: (n1, n2)
    K = variance * torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
    return K

# -------------------------
# 4. Define the negative log marginal likelihood (NLL)
def negative_log_marginal_likelihood(x, y, lengthscale, variance, noise):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n = x.shape[0]
    # Compute kernel matrix K(X,X) and add noise on the diagonal.
    K = squared_exponential_kernel(x, x, lengthscale, variance) + noise * torch.eye(n, dtype=x.dtype)
    # Compute Cholesky factorization of K.
    L = torch.linalg.cholesky(K)
    # Solve for alpha = K^{-1} y using the Cholesky factors.
    alpha = torch.cholesky_solve(y, L)
    # Compute the log determinant of K via its Cholesky factor.
    logdetK = 2 * torch.sum(torch.log(torch.diag(L)))
    # NLL = 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5*n*log(2π)
    nll = 0.5 * torch.matmul(y.T, alpha) + 0.5 * logdetK + 0.5 * n * math.log(2 * math.pi)
    return nll.squeeze()  # return a scalar
def compute_gradients_vanilla(x, y, sigmasq, kernel):
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)


    # -------------------------
    # 2. Define hyperparameters as torch tensors with gradients.
    lengthscale = torch.tensor(kernel.lengthscale, dtype=x.dtype, requires_grad=True)
    variance    = torch.tensor(kernel.variance, dtype=x.dtype, requires_grad=True)
    noise       = sigmasq.clone().detach().requires_grad_(True)

    # -------------------------


    # -------------------------
    # 5. Compute the NLL and its gradients.
    nll = negative_log_marginal_likelihood(x, y, lengthscale, variance, noise)
    # print("Negative log marginal likelihood:", nll.item())

    nll.backward()

    # print("\n (VANILLA) Gradients of the negative log marginal likelihood:")
    # print("  dNLL/d(lengthscale) =", lengthscale.grad.item())
    # print("  dNLL/d(variance)    =", variance.grad.item())
    # print("  dNLL/d(noise)       =", noise.grad.item())
    grad = torch.tensor([lengthscale.grad.item(), variance.grad.item(), noise.grad.item()])

    return grad.to(dtype=x.dtype)

