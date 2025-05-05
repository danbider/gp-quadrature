# efgp_wrapper.py
from typing import Dict, Optional, Tuple
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

from math import prod
from typing import Dict, Optional, Callable
from torch.fft import fftn, ifftn


# for stochastic variance estimation
def diag_sums_nd(A_apply, J, xis_flat, max_cg_iter, cg_tol,ws):
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
            do_profiling: bool = True,
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
        rdtype  = torch.float64
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

        # ---------- NUFFT Setup --------------------------------------
        with record_function("nufft_setup"):
            OUT        = (mtot,) * d
            nufft_eps  = 1e-15
            xcen       = torch.zeros(d, device=device, dtype=rdtype)
            phi_train  = (TWO_PI * h * (x      - xcen)).T.contiguous()
            phi_test   = (TWO_PI * h * (x_new  - xcen)).T.contiguous()

            def finufft1(phi_in, vals):
                phi_in = phi_in.contiguous()
                vals = vals.contiguous()
                fk = pff.finufft_type1(phi_in, vals.to(dtype=cdtype), OUT, eps=nufft_eps, isign=-1, modeord=False)
                return fk.view(-1)

            def finufft2(phi_in, fk_flat):
                phi_in = phi_in.contiguous()
                fk_flat = fk_flat
                fk_nd = fk_flat.view(OUT).to(dtype=cdtype).contiguous()
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
                    # ---- exact CG ----------------------------------
                    f_x   = torch.exp(TWO_PI * 1j * torch.matmul(x_new, xis.T))
                    rhs_v = ws.unsqueeze(0) * f_x.conj()
                    gamma = BatchConjugateGradients(A_var, rhs_v, x0=torch.zeros_like(rhs_v, dtype=cdtype),
                                                    tol=cg_tol, early_stopping=True).solve()
                    s2    = torch.real(torch.einsum('bm,m,bm->b', f_x, ws, gamma)).clamp_min_(0.0)
                    ytrg["var"] = s2


                    # microbatch = 2048
                    # if device is None:
                    #     device = x_new.device

                    # xis   = xis.to(device)
                    # ws    = ws.to(device)            # (m,)
                    # out   = []                       # chunks to cat at the end
                    # z0    = torch.zeros_like(ws, dtype=cdtype, device=device)  # template for CG

                    # for xb in torch.split(x_new, microbatch, dim=0):           # (b,d)
                    #     xb   = xb.to(device)

                    #     # (b, m) – only this sub‑batch lives in memory
                    #     fx   = torch.exp(TWO_PI * 1j * (xb @ xis.T)).to(cdtype)

                    #     # rhs = diag(ws) ⋅ conj(fx)   – uses broadcasting, no extra copy
                    #     rhs  = ws * fx.conj()

                    #     γ    = BatchConjugateGradients(
                    #                 A_var, rhs, x0=z0.expand_as(rhs),
                    #                 tol=cg_tol, early_stopping=True
                    #         ).solve()              # (b, m)

                    #     s2b  = torch.real((fx * (ws * γ)).sum(dim=-1)).clamp_min(0.0)

                    #     out.append(s2b.cpu())        # keep only the scalar result
                    #     del fx, rhs, γ               # free GPU quickly
                    #     torch.cuda.empty_cache()

                    # ytrg["var"] = torch.cat(out, dim=0)
                elif variance_method == "stochastic":
                    # ---- Hutchinson / FFT --------------------------
                    J = hutchinson_probes
                    est_grid = diag_sums_nd(A_var, J, xis, max_cg_iter, cg_tol,ws)
                    s_est    = nufft_var_est_nd(est_grid, h, torch.zeros(d, device=device, dtype=rdtype), x_new, nufft_eps)
                    ytrg["var"] = s_est.clamp_min_(0.0)

                else:
                    raise ValueError("variance_method must be 'regular' or 'stochastic'")

        # ---------- Optional Log Marginal -----------------------------
        if opts.get("get_log_marginal_likelihood", False):
            ytrg["log_marginal_likelihood"] = torch.tensor(float("nan"), device=device, dtype=rdtype)

    # ----------------- Return ----------------------------------------
    return beta, xis, ytrg, ws, toeplitz



def efgpnd_gradient_batched(
        x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
        *, nufft_eps=1e-15, cg_tol = 1e-4, early_stopping = True, device=None, dtype=torch.float64):
    """
    Gradient of the 1‑D GP log‑marginal likelihood estimated with
    Hutchinson trace + CG, completely torch native.
    """
    # 0)  Book‑keeping ------------------------------------------------------
    device  = device or x.device
    # device = x.device
    rdtype = torch.float64
    cdtype = torch.complex128
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
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    # print(xis_1d.shape)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij') # makes tensor product Jm 
    xis = torch.stack(grids, dim=-1).view(-1, d) 
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=torch.complex128) * h**d) # (mtot**d,1)
    Dprime  = (h**d * kernel.spectral_grad(xis)).to(cmplx)  # (M, 3)

    # 1)  NUFFT adjoint / forward helpers (modeord=False) -------------------
    OUT = (mtot,)*d
    
    nufft_eps = 1e-15
    phi = (2 * math.pi * h * (x - 0.0)).to(dtype).T.contiguous()   # real (N,)
    def finufft1(vals):
        """
        Adjoint NUFFT: nonuniform→uniform.
        vals: (N,) complex
        returns: tensor of shape OUT, then flattened
        """
        arr = pff.finufft_type1(
            phi, vals.to(cdtype).contiguous(), OUT,
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
        # fk_nd = fk_flat
        return pff.finufft_type2(
            phi, fk_nd.to(cdtype), # phi is (d,N), fk_nd is (T, mtot**d)
            eps=nufft_eps, isign=+1, modeord=False
        )


    fadj = lambda v: finufft1(v)    # NU → U
    fwd  = lambda fk: finufft2(fk)  # U  → NU

    # 2)  Toeplitz operator T (cached FFT) ----------------------------------
    m_conv       = (mtot - 1) // 2
    v_kernel     = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz   = ToeplitzND(v_kernel)               # cached once

    # 3)  Linear map A· = D F*F (D·) + σ² I -------------------------------
    def A_apply(beta):
        return ws * toeplitz(ws * beta) + sigmasq * beta

    # 4)  Solve A β = W F* y ---------------------------------------------
    rhs   = ws * fadj(y).reshape(-1)
    beta  = ConjugateGradients(A_apply, rhs,
                               torch.zeros_like(rhs),
                               tol=cg_tol, early_stopping=early_stopping).solve()
    alpha = (y - fwd(ws * beta)) / sigmasq

    # 5)  Term‑2  (α*D'α, α*α) --------------------------------------------
    fadj_alpha = fadj(alpha)
    term2 = torch.stack((
        torch.vdot(fadj_alpha.reshape(-1), Dprime[:, 0] * fadj_alpha.reshape(-1)),
        torch.vdot(fadj_alpha.reshape(-1), Dprime[:, 1] * fadj_alpha.reshape(-1)),
        torch.vdot(alpha,       alpha)
    ))

    # 6)  Monte‑Carlo trace (Term‑1) ---------------------------------------
    T  = trace_samples
    Z  = (2 * torch.randint(0, 2, (T, x.shape[0]), device=device,
                            dtype=dtype) - 1).to(cmplx)
    fadjZ = fadj(Z)

    B_blocks, R_blocks = [], []
    for i in range(3):
        # Dprime[:, i] * fadjZ.reshape(trace_samples,-1) should be shape (trace_samples, M)
        rhs_i = fwd(Dprime[:, i] * fadjZ.reshape(int(trace_samples),-1)) if i < 2 else Z
        R_blocks.append(rhs_i)
        B_blocks.append(ws * fadj(rhs_i).reshape(int(trace_samples),-1))

    B_all = torch.cat(B_blocks, 0)       # (3T, M)
    R_all = torch.cat(R_blocks, 0)       # (3T, N)

    def A_apply_batch(B):
        return ws * toeplitz(ws * B) + sigmasq * B

    Beta_all = BatchConjugateGradients(
        A_apply_batch, B_all, torch.zeros_like(B_all),
        tol=cg_tol, early_stopping=early_stopping).solve()

    Alpha_all = (R_all - fwd(ws * Beta_all)) / sigmasq
    A_chunks  = Alpha_all.chunk(3, 0)

    term1 = torch.stack([(Z * a).sum(1).mean() for a in A_chunks])

    # 7)  Gradient ----------------------------------------------------------
    grad = 0.5 * (term1 - term2)
    return grad.real  



class EFGPND:
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
        sigmasq: float,
        eps: float,
        opts: Optional[Dict] = None,
    ):
        self.x        = x
        self.y        = y
        self.kernel   = kernel
        self.sigmasq  = sigmasq
        self.eps      = eps
        self.opts     = {} if opts is None else opts

        # --- placeholders populated by `fit` -------------------------------------------------
        self._beta       = None        # Fourier weights
        self._xis        = None        # frequency grid
        self._ws         = None        # quadrature weights √S(ξ) h^{d/2}
        self._toeplitz   = None        # convolution operator object
        self._fitted     = False
        self.train_mean  = None        # f̂(x_i) on training set
        self._cache_full = None        # (beta, xis, ytrg, ws, toeplitz) for reuse


    def fit(self) -> "EFGPND":
        """
        Solves the linear system once and stores everything needed for
        fast predictions at new x*.  Returns `self` for chaining.
        """
        # We only need the mean here; keep variance off for speed
        opts = dict(self.opts)               # shallow copy
        opts.setdefault("estimate_variance", False)

        beta, xis, ytrg, ws, toeplitz = efgp_nd(
            x        = self.x,
            y        = self.y,
            sigmasq  = self.sigmasq,
            kernel   = self.kernel,
            eps      = self.eps,
            x_new    = self.x,               # ← predict on training set
            opts     = opts,
            do_profiling = False,
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
            sigmasq  = self.sigmasq,
            kernel   = self.kernel,
            eps      = self.eps,
            x_new    = x_new,
            opts     = opts,
            do_profiling = False,
        )
        mean = ytrg["mean"]
        var  = ytrg["var"] if return_variance else None
        return mean, var

    # ------------------------------------------------------------------
    # placeholder for hyper‑parameter learning
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # put this inside class EFGPND  (replacing the previous stub)
    # ------------------------------------------------------------------
    def optimize_hyperparameters(
        self,
        *,
        # --- search grid ------------------------------------------------
        epsilon_values         = (1e-2,),      # quadrature accuracy grid
        trace_samples_values   = (10,),        # Hutchinson probes grid
        # --- SGD options -----------------------------------------------
        lr: float              = 0.05,         # initial learning‑rate
        max_iters: int         = 50,
        min_lengthscale: float = 5e-3,

        **gkwargs,                             # forwarded to efgpnd_gradient_batched
    ):
        """
        Simple SGD for now 
        """
        device = self.x.device
        rdtype = torch.float64

        init_kernel = self.kernel                       # keep original ref
        best_state  = None                              # for early stopping
        best_obj    = float("inf")
        x0 = self.x.min(dim=0).values  
        x1 = self.x.max(dim=0).values  
        logs = []                                       # ⇐ stored afterwards

        for EPSILON in epsilon_values:
            for J in trace_samples_values:

                # fresh copy of kernel & noise each hypergrid setting
                kernel  = init_kernel.model_copy()
                sigmasq = torch.tensor(self.sigmasq, dtype=rdtype, device=device)

                start   = time.time()
                lr_curr = lr

                tracked_len, tracked_var, tracked_noise = [], [], []

                for it in range(max_iters):
                    # ---- bookkeeping --------------------------------------------------
                    tracked_len.append(float(kernel.lengthscale))
                    tracked_var.append(float(kernel.variance))
                    tracked_noise.append(float(sigmasq))

                    # ---- gradient ------------------------------------------------------
                    if it<40:
                        grad = efgpnd_gradient_batched(
                            self.x, self.y, sigmasq, kernel,
                            EPSILON,
                            trace_samples = 3,
                            x0 = x0, x1 = x1,
                            # **gkwargs
                        )
                    else: 
                        grad = efgpnd_gradient_batched(
                            self.x, self.y, sigmasq, kernel,
                            EPSILON,
                            trace_samples = J,
                            x0 = x0, x1 = x1,
                            # **gkwargs
                        )
                    # grad = [∂L/∂ℓ, ∂L/∂σ_f², ∂L/∂σ_n²]
                    # ---- parameter update (log‑space trick) ---------------------------
                    # length‑scale
                    new_ℓ = math.exp( math.log(kernel.lengthscale)
                                    - lr_curr * kernel.lengthscale * grad[0].item() )
                    kernel.lengthscale = max(new_ℓ, min_lengthscale)

                    # variance
                    new_var = math.exp( math.log(kernel.variance)
                                        - lr_curr * kernel.variance * grad[1].item() )
                    kernel.variance = max(new_var, 1e-5)

                    # noise
                    new_noise = math.exp( math.log(sigmasq.item())
                                        - lr_curr * sigmasq.item() * grad[-1].item() )
                    sigmasq = torch.tensor(max(new_noise, 1e-5), dtype=rdtype, device=device)

                    # adaptive LR guard against runaway ℓ
                    if it < 3 and kernel.lengthscale < 0.05:
                        lr_curr *= 0.5

                    if it % 10 == 0:
                        print(f"[ε={EPSILON} | J={J}] iter {it:>3}  "
                            f"ℓ={kernel.lengthscale:.4g}  "
                            f"σ_f²={kernel.variance:.4g}  σ_n²={sigmasq:.4g}")
                        print(f"grad: {grad}")

                # ---- one full objective eval (optional) -------------------------------
                obj = float("nan")       # <- plug in closed‑form log‑marg if available
                elapsed = time.time() - start

                logs.append({
                    "epsilon"           : EPSILON,
                    "trace_samples"     : J,
                    "tracked_lengthscale": tracked_len,
                    "tracked_variance"   : tracked_var,
                    "tracked_noise"      : tracked_noise,
                    "iters"              : max_iters,
                    "time_sec"           : elapsed,
                    "final_kernel"       : kernel,
                    "final_noise"        : sigmasq,
                    "objective"          : obj,
                })


                best_state = (kernel, sigmasq)

                print(f"└─ finished ε={EPSILON}, J={J} in {elapsed:.1f}s")

        # ----------------------------------------------------------------
        # stash results & update model
        # ----------------------------------------------------------------
        self.training_log = logs                           # everything
        if best_state is not None:
            self.kernel, self.sigmasq = best_state         # take best

        return self   # so you can chain: model.optimize_hyperparameters(...).fit()


    # convenience alias
    def fit_predict(self, x_new: torch.Tensor, **pred_kw):
        """Shorthand for `.fit().predict(...)`."""
        return self.fit().predict(x_new, **pred_kw)


class ToeplitzND:
    """
    Fast d‑dimensional block‑Toeplitz convolution via FFT.
    Caches the FFT of a fixed kernel and then performs
    conv with any flat-or-block input.

    Args:
      v: complex tensor of shape (L1,...,Ld) with Li = 2*ni - 1
      force_pow2: zero‑pad each dim to next power‑of‑2

    Call with x either:
      • flat: 1D tensor of length prod(ni), or
      • block: tensor whose last d dims are exactly (n1,...,nd)
    """
    def __init__(self, v: torch.Tensor, *, force_pow2: bool = True):
        # ensure complex
        if not v.is_complex():
            v = v.to(torch.complex128)
        self.device = v.device

        # dims and block‑sizes
        self.Ls   = list(v.shape)                   # [L1,...,Ld]
        self.ns   = [(L+1)//2 for L in self.Ls]     # [n1,...,nd]
        self.size = prod(self.ns)                   # total block elements

        # fft grid
        if force_pow2:
            self.fft_shape = [1 << (L-1).bit_length() for L in self.Ls]
        else:
            self.fft_shape = self.Ls.copy()

        # pad & cache kernel FFT
        pad = []
        for L, F in zip(reversed(self.Ls), reversed(self.fft_shape)):
            pad += [0, F - L]
        v_pad     = nnf.pad(v.to(self.device), pad)
        self.fft_v = torch.fft.fftn(v_pad, s=self.fft_shape,
                                    dim=list(range(-len(self.Ls), 0)))

        # slice indices for central block
        self.starts = [n-1 for n in self.ns]
        self.ends   = [st+n for st, n in zip(self.starts, self.ns)]


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts x with shape (..., self.size)  – flat batch
                    or  (..., *self.ns)       – block batch
        and returns a tensor of the same outer-batch shape.
        """
        x = x.to(self.device)
        orig_flat = False                     # remember how input looked

        # ---------- detect layout ----------------------------------------
        if x.shape[-1] == self.size:          # (..., N)  – flat batch
            orig_flat = True
            batch_shape = x.shape[:-1]
            x = x.view(*batch_shape, *self.ns)
        elif list(x.shape[-len(self.ns):]) == self.ns:
            batch_shape = x.shape[:-len(self.ns)]
        else:
            raise AssertionError(
                f"Expected trailing dim {self.size} or block {tuple(self.ns)}, got {tuple(x.shape)}"
            )

        # ---------- promote to complex -----------------------------------
        if not x.is_complex():
            x = x.to(torch.complex128)

        # ---------- zero-pad to FFT grid ---------------------------------
        pad = []
        for n, F in zip(reversed(self.ns), reversed(self.fft_shape)):
            pad += [0, F - n]                 # torch.nn.functional.pad wants (dim_k_before, dim_k_after) pairs
        x_pad = nnf.pad(x, pad)

        # ---------- FFT convolution --------------------------------------
        x_fft = torch.fft.fftn(x_pad, s=self.fft_shape,
                            dim=list(range(-len(self.ns), 0)))
        y     = torch.fft.ifftn(self.fft_v * x_fft,
                                s=self.fft_shape,
                                dim=list(range(-len(self.ns), 0)))

        # ---------- crop central block -----------------------------------
        slices = [slice(None)] * y.ndim       # keep batch dims
        for st, en in zip(self.starts, self.ends):
            slices[-len(self.starts) + len(slices) - y.ndim]  # just to calm linters
        slices = [slice(None)] * (y.ndim - len(self.ns))
        for st, en in zip(self.starts, self.ends):
            slices.append(slice(st, en))
        y = y[tuple(slices)]

        # ---------- restore original layout ------------------------------
        # ---------- restore original layout ------------------------------
        if orig_flat:
            # Either option below is fine; reshape is the simpler one.
            y = y.reshape(*batch_shape, self.size)          # works with non-contiguous tensors
            # y = y.contiguous().view(*batch_shape, self.size)  # alternative

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
    eps         = 1e-15

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
# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _cmplx(real_dtype: torch.dtype) -> torch.dtype:
    """Matching complex dtype for a given real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128
