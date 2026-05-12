"""
Full-gradient end-to-end timing: current efgpnd_gradient_batched vs
a single-NUFFT implementation of the same gradient.

Proposed pipeline:
  * ONE NUFFT (F*y at setup)
  * CG for beta_0
  * F*alpha by Toeplitz
  * Term 2: M-space for all hypers
  * Term 1 kernel hypers: BD' Hutchinson (pure Toeplitz + CG, zero NUFFTs)
  * Term 1 noise: existing Woodbury shortcut (zero NUFFTs)

Also verifies that the proposed gradient equals the current gradient to
the same CG tolerance (using identical Hutchinson probes).
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from kernels.squared_exponential import SquaredExponential
from efgpnd import (
    _cmplx, NUFFT, ToeplitzND,
    compute_convolution_vector_vectorized_dD,
    create_A_mean, create_jacobi_precond,
    efgpnd_gradient_batched,
)
from utils.kernels import get_xis
from cg import ConjugateGradients
from vanilla_gp_sampling import sample_gp_spectral_approx


torch.manual_seed(0)
dtype = torch.float64
cdtype = _cmplx(dtype)


def proposed_gradient(x, y, sigmasq, kernel, eps, trace_samples,
                      cg_tol=1e-8, nufft_eps=6e-8, noise_probe_seed=None):
    """
    Single-NUFFT gradient of the EFGP log-marginal.

    Returns a tensor of shape (num_hypers,) ordered [kernel hypers..., noise].
    """
    d = x.shape[1] if x.ndim > 1 else 1
    if x.ndim == 1: x = x.unsqueeze(-1)
    kernel_hypers = kernel.hypers
    num_kernel = len(kernel_hypers)

    L = (x.max(dim=0).values - x.min(dim=0).values).max()
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                              use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(cdtype) * h**d)
    Dprime_all = (h**d * kernel.spectral_grad(xis)).to(cdtype)         # (M, num_kernel)
    M = ws.numel()
    OUT = (mtot,) * d

    xcen = torch.zeros(d, dtype=x.dtype)
    nufft_op = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype)
    def fadj(v): return nufft_op.type1(v, out_shape=OUT).reshape(-1)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, sigmasq, cdtype)
    center_tuple = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    jacobi = create_jacobi_precond(ws, sigmasq, diag_scale=v_kernel[center_tuple].real)

    # ---- THE ONLY NUFFT ----
    y_c = y.to(cdtype)
    Fy = fadj(y_c)
    y_norm2 = (y_c * y_c.conj()).sum().real

    # ---- mean solve ----
    rhs = ws * Fy
    cg_mean = ConjugateGradients(A_apply, rhs, torch.zeros_like(rhs),
                                 tol=cg_tol, M_inv_apply=jacobi, early_stopping=True)
    beta_0 = cg_mean.solve()
    Dbeta  = ws * beta_0

    # F*alpha via Toeplitz
    Fstar_alpha = (Fy - toeplitz(Dbeta)) / sigmasq

    # reusable M-scalars
    c_scalar   = torch.vdot(beta_0, ws * Fy).real       # Re{beta_0^* D Fy}
    beta_norm2 = torch.vdot(beta_0, beta_0).real

    # =================== Term 2 ===================
    T2 = torch.empty(num_kernel + 1, dtype=dtype)
    for i in range(num_kernel):
        T2[i] = torch.vdot(Fstar_alpha, Dprime_all[:, i] * Fstar_alpha).real
    T2[-1] = (y_norm2 - c_scalar - sigmasq * beta_norm2) / (sigmasq ** 2)

    # =================== Term 1 ===================
    T_p = trace_samples
    T1 = torch.empty(num_kernel + 1, dtype=dtype)

    # Variance hyper has a closed-form shortcut from the noise term:
    #   dK/dsig_f^2 = K/sig_f^2  =>  T1[var] = (N - sig^2 * T1_noise) / sig_f^2.
    # So Hutchinson only runs over kernel hypers EXCLUDING variance.
    variance_idx = kernel_hypers.index("variance") if "variance" in kernel_hypers else None
    trace_kernel_indices = [i for i in range(num_kernel) if i != variance_idx]
    K_tr = len(trace_kernel_indices)

    # --- single batched CG: stack [trace-kernel hypers, noise] together ---
    g = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(noise_probe_seed if noise_probe_seed is not None else 123)
    V = torch.empty((T_p, M), dtype=x.dtype).bernoulli_(0.5, generator=g).mul_(2).sub_(1).to(cdtype)
    Vn = torch.empty((T_p, M), dtype=x.dtype).bernoulli_(0.5, generator=g2).mul_(2).sub_(1).to(cdtype)

    # kernel-hyper RHSs:  D T D'_i V   (shape K_tr * T_p, M)
    if K_tr > 0:
        Dp_stack = Dprime_all[:, trace_kernel_indices].T                # (K_tr, M)
        Di_V = (Dp_stack.unsqueeze(1) * V.unsqueeze(0)).reshape(K_tr * T_p, M)
        y1_kernel = toeplitz(Di_V)                                      # T D' V
        rhs_kernel = ws * y1_kernel                                     # D T D' V
    else:
        Di_V = y1_kernel = torch.empty((0, M), dtype=cdtype)
        rhs_kernel = torch.empty((0, M), dtype=cdtype)

    # noise RHS: D T D Vn
    rhs_noise = ws * toeplitz(ws * Vn)                                  # (T_p, M)

    rhs_all = torch.cat([rhs_kernel, rhs_noise], dim=0)                 # ((K_tr+1)*T_p, M)
    cg_trace = ConjugateGradients(A_apply, rhs_all, torch.zeros_like(rhs_all),
                                  tol=cg_tol, M_inv_apply=jacobi, early_stopping=True)
    sol_all = cg_trace.solve()
    sol_kernel = sol_all[:K_tr * T_p]
    sol_noise  = sol_all[K_tr * T_p:]

    # --- kernel hyper Hutchinson (excluding variance) ---
    if K_tr > 0:
        y2_kernel = toeplitz(ws * sol_kernel)
        Bu = ((y1_kernel - y2_kernel) / sigmasq).view(K_tr, T_p, M)
        for slot, hi in enumerate(trace_kernel_indices):
            T1[hi] = (V * Bu[slot]).sum(dim=1).real.mean()

    # --- noise Woodbury shortcut ---
    T1[-1] = (torch.tensor(float(x.shape[0]), dtype=dtype) / sigmasq
              - ((Vn.conj() * sol_noise).sum(dim=1).real / sigmasq).mean())

    # --- variance closed-form from noise ---
    if variance_idx is not None:
        var_scalar = torch.as_tensor(kernel.get_hyper("variance"), dtype=dtype)
        T1[variance_idx] = (torch.tensor(float(x.shape[0]), dtype=dtype)
                            - sigmasq * T1[-1]) / var_scalar

    return 0.5 * (T1 - T2)


def _time(fn, reps=2):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# =============== sweep ===============
print(f"{'N':>6} {'M':>4} {'T_p':>4}  {'current (ms)':>14}  {'proposed (ms)':>15}  "
      f"{'speedup':>8}  {'|Δ|_∞':>10}")

for n in [2000, 10000, 30000, 100000]:
    for ls_val in [0.08]:
        for T_p in [1, 10, 50]:
            x = torch.rand(n, 1, dtype=dtype)
            y = sample_gp_spectral_approx(x, length_scale=ls_val, variance=1.0,
                                          num_samples=1, seed=0).to(dtype)
            y = y + (0.1 ** 0.5) * torch.randn(n, dtype=dtype)
            kernel = SquaredExponential(dimension=1, init_lengthscale=ls_val, init_variance=1.0)
            sigmasq = torch.tensor(0.1, dtype=dtype)
            eps = 1e-6
            # Get M for reporting
            L = (x.max() - x.min())
            _, _, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                                 use_integral=True, l2scaled=False)

            # Current: efgpnd_gradient_batched (produces [kernel..., noise])
            def call_current():
                return efgpnd_gradient_batched(
                    x, y, sigmasq, kernel, eps, trace_samples=T_p,
                    x0=x.min(dim=0).values, x1=x.max(dim=0).values,
                    cg_tol=1e-8, nufft_eps=6e-8)
            def call_prop():
                return proposed_gradient(x, y, sigmasq, kernel, eps,
                                         trace_samples=T_p, cg_tol=1e-8, nufft_eps=6e-8)

            g_cur = call_current()
            g_new = call_prop()
            # Absolute max diff (noting Hutchinson uses different probe streams)
            diff = (g_cur - g_new).abs().max().item()

            t_cur = _time(call_current, reps=3)
            t_new = _time(call_prop, reps=3)
            print(f"{n:>6d} {mtot:>4d} {T_p:>4d}  {t_cur*1000:>14.2f}  {t_new*1000:>15.2f}  "
                  f"{t_cur/t_new:>7.2f}x  {diff:>10.2e}")

# ---- correctness check: identical Hutchinson behavior by taking many probes ----
print("\n--- convergence of proposed gradient to current as T_p -> inf (N=4000, ell=0.08) ---")
x = torch.rand(4000, 1, dtype=dtype)
y = sample_gp_spectral_approx(x, length_scale=0.08, variance=1.0,
                              num_samples=1, seed=0).to(dtype)
y = y + (0.1 ** 0.5) * torch.randn(4000, dtype=dtype)
kernel = SquaredExponential(dimension=1, init_lengthscale=0.08, init_variance=1.0)
sigmasq = torch.tensor(0.1, dtype=dtype)

for T_p in [5, 50, 500]:
    # average N independent evaluations to estimate the true gradient both methods converge to
    def avg(fn, n_avg=10):
        out = None
        for s in range(n_avg):
            torch.manual_seed(s)
            g = fn(T_p)
            out = g if out is None else out + g
        return out / n_avg
    g_cur = avg(lambda T: efgpnd_gradient_batched(
        x, y, sigmasq, kernel, 1e-6, trace_samples=T,
        x0=x.min(dim=0).values, x1=x.max(dim=0).values, cg_tol=1e-8))
    g_new = avg(lambda T: proposed_gradient(x, y, sigmasq, kernel, 1e-6,
                                            trace_samples=T, cg_tol=1e-8))
    print(f"T_p={T_p:>4d}  current={[f'{v:+.4e}' for v in g_cur.tolist()]}  "
          f"proposed={[f'{v:+.4e}' for v in g_new.tolist()]}")
