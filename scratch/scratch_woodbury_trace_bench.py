"""
Benchmark: Woodbury-in-feature-space Hutchinson trace for the lengthscale
           gradient vs. the current N-space trace path in efgpnd.

We estimate  T_1^l = tr( K_tilde^{-1} dK/dl ),  K_tilde = Phi Phi* + s2 I,
                                                 dK/dl  = F D' F*.

Current path (efgpnd.py lines ~193-260): N-dim probes Z, 3 NUFFTs/estimate.
Woodbury path: M-dim probes v, two Toeplitz mat-vecs + one CG solve, 0 NUFFTs.
"""

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from kernels.squared_exponential import SquaredExponential
from efgpnd import (
    _cmplx, NUFFT, ToeplitzND,
    compute_convolution_vector_vectorized_dD,
    create_A_mean, create_jacobi_precond,
)
from utils.kernels import get_xis
from cg import ConjugateGradients
from vanilla_gp_sampling import sample_gp_fast


torch.manual_seed(0)
dtype = torch.float64
cdtype = _cmplx(dtype)

# ---------- problem ----------
d = 1
n = 4000
ls_val, var_val, noise_val = 0.08, 1.0, 0.1
eps = 1e-6

x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=ls_val, variance=var_val,
                   noise_variance=noise_val, num_samples=1).squeeze()

kernel = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
sigmasq = torch.tensor(noise_val, dtype=dtype)

# ---------- spectral grid ----------
L = (x.max(dim=0).values - x.min(dim=0).values).max()
xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                          use_integral=True, l2scaled=False)
grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
xis = torch.stack(grids, dim=-1).view(-1, d)
ws     = torch.sqrt(kernel.spectral_density(xis).to(cdtype) * h**d)   # D diag
Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)                # (M, 2)
ls_idx = kernel.hypers.index("lengthscale")
Dp = Dprime[:, ls_idx]                                                # D' diag
M = ws.numel()
print(f"n={n}, d={d}, mtot={mtot}, M={M}")

# ---------- operators ----------
OUT  = (mtot,) * d
xcen = torch.zeros(d, dtype=dtype)
nufft_op = NUFFT(x, xcen, h, 1e-12, cdtype=cdtype)

def fadj(v):  # N -> M, supports (T, N) batches
    out = nufft_op.type1(v, out_shape=OUT)
    return out.reshape(v.shape[0], -1) if v.ndim == 2 else out.reshape(-1)

def fwd(fk):  # M -> N
    if fk.ndim == 1:
        return nufft_op.type2(fk.reshape(OUT), out_shape=OUT)
    T = fk.shape[0]
    return nufft_op.type2(fk.reshape((T,) + OUT), out_shape=OUT)

m_conv   = (mtot - 1) // 2
v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)
toeplitz = ToeplitzND(v_kernel, force_pow2=True)

A_apply = create_A_mean(ws, toeplitz, sigmasq, cdtype)
center_tuple = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
diag_scale   = v_kernel[center_tuple].real
jacobi       = create_jacobi_precond(ws, sigmasq, diag_scale=diag_scale)

# ---------- ground truth via dense F ----------
print("Building dense F (column by column)...")
F_dense = torch.zeros((n, M), dtype=cdtype)
I_M = torch.eye(M, dtype=cdtype)
for k in range(M):
    F_dense[:, k] = fwd(I_M[k])
D2 = ws * ws.conj()
K_tilde = (F_dense * D2) @ F_dense.conj().T + sigmasq * torch.eye(n, dtype=cdtype)
dK_dl   = (F_dense * Dp) @ F_dense.conj().T
trace_true = torch.linalg.solve(K_tilde, dK_dl).diagonal().sum().real.item()
print(f"Ground truth trace = {trace_true:.8f}")

# Also sanity check the deterministic piece: tr(F*F D') = (F*F)_{ii} * sum(D') ?
FstarF_diag = (F_dense.conj().T @ F_dense).diagonal()
print(f"diag(F*F) range: [{FstarF_diag.real.min():.6f}, {FstarF_diag.real.max():.6f}]  "
      f"v_kernel[center]={v_kernel[center_tuple].real:.6f}")
det_direct = (FstarF_diag * Dp).sum().real.item()
det_woodbury = (v_kernel[center_tuple] * Dp.sum()).real.item()
print(f"tr(F*F D')  direct={det_direct:.6f}  vs  v_kernel[0]*sum(D')={det_woodbury:.6f}")

# ---------- current (N-space Hutchinson) ----------
def current_trace(T_probes, seed, cg_tol=1e-8):
    g = torch.Generator().manual_seed(seed)
    Z = torch.empty((T_probes, n), dtype=dtype)
    Z.bernoulli_(0.5, generator=g).mul_(2).sub_(1)
    Z = Z.to(cdtype)
    fadjZ = fadj(Z)                            # (T, M)
    Di_FZ = Dp * fadjZ                         # (T, M)
    w     = fwd(Di_FZ).reshape(T_probes, -1)   # (T, N)  F D' F* Z
    gamma = ws * toeplitz(Di_FZ)               # (T, M)  D T D' F* Z
    cg = ConjugateGradients(A_apply, gamma, torch.zeros_like(gamma),
                            tol=cg_tol, M_inv_apply=jacobi, early_stopping=True)
    beta = cg.solve()
    FDbeta = fwd(ws * beta).reshape(T_probes, -1)
    alpha  = (w - FDbeta) / sigmasq
    est = (Z * alpha).sum(dim=1).real.mean().item()
    return est, cg.iters_completed

# ---------- Woodbury on B·D' (the "right" M-space operator) ----------
# B = F* K_tilde^{-1} F = s2^{-1} (T - TD (s2 I + DTD)^{-1} DT)
# tr(B D') = tr(K_tilde^{-1} F D' F*) = target (one-shot Hutchinson, no split).
def bd_trace(T_probes, seed, cg_tol=1e-8):
    g = torch.Generator().manual_seed(seed)
    V = torch.empty((T_probes, M), dtype=dtype)
    V.bernoulli_(0.5, generator=g).mul_(2).sub_(1)
    V = V.to(cdtype)
    u   = Dp * V                       # D' v
    y1  = toeplitz(u)                  # T D' v
    rhs = ws * y1                      # D T D' v
    cg  = ConjugateGradients(A_apply, rhs, torch.zeros_like(rhs),
                             tol=cg_tol, M_inv_apply=jacobi, early_stopping=True)
    c   = cg.solve()                   # (s2 I + DTD)^{-1} DTD' v
    y2  = toeplitz(ws * c)             # T D · that
    Bu  = (y1 - y2) / sigmasq          # B D' v
    est = (V * Bu).sum(dim=1).real.mean().item()
    return est, cg.iters_completed

# ---------- Woodbury (M-space Hutchinson, split) ----------
def woodbury_trace(T_probes, seed, cg_tol=1e-8):
    # deterministic first term: s2^{-1} tr(F*F D')
    first = (v_kernel[center_tuple] * Dp.sum()).real.item() / sigmasq.item()
    # Hutchinson second term: s2^{-1} * E[v* (s2 I + DTD)^{-1} D T D' T D v]
    g = torch.Generator().manual_seed(seed)
    V = torch.empty((T_probes, M), dtype=dtype)
    V.bernoulli_(0.5, generator=g).mul_(2).sub_(1)
    V = V.to(cdtype)
    # D T D' T D v, with two Toeplitz applies
    rhs = ws * toeplitz(Dp * toeplitz(ws * V))
    cg = ConjugateGradients(A_apply, rhs, torch.zeros_like(rhs),
                            tol=cg_tol, M_inv_apply=jacobi, early_stopping=True)
    beta = cg.solve()
    second = (V * beta).sum(dim=1).real.mean().item() / sigmasq.item()
    return first - second, cg.iters_completed

# ---------- diagnostic: exact M-space trace (no Hutchinson) ----------
# Second piece = tr((s2 I + DTD)^{-1} DTD'TD). At small M we just build it.
T_dense = torch.zeros((M, M), dtype=cdtype)
for k in range(M):
    e = torch.zeros(M, dtype=cdtype); e[k] = 1.0
    T_dense[:, k] = toeplitz(e)
D_diag     = torch.diag(ws)
Dp_diag    = torch.diag(Dp)
DTD        = D_diag @ T_dense @ D_diag
DTDpTD     = D_diag @ T_dense @ Dp_diag @ T_dense @ D_diag
A_M        = torch.eye(M, dtype=cdtype) * sigmasq + DTD
second_exact = torch.linalg.solve(A_M, DTDpTD).diagonal().sum().real.item()
first_exact  = (v_kernel[center_tuple] * Dp.sum()).real.item()
trace_exact_Mspace = (first_exact - second_exact) / sigmasq.item()
print(f"Exact via M-space Woodbury: {trace_exact_Mspace:.6f}  (dense truth {trace_true:.6f})")

# Rank / variance diagnostics.
# N-space Hutchinson variance on A_N = K_tilde^{-1} F D' F*, rank <= M.
# M-space Hutchinson variance on A_M = (s2 I + DTD)^{-1} DTD'TD (full M x M).
# Variance of Rademacher Hutchinson(A) = 2*(||A||_F^2 - sum A_ii^2).
print("\n--- theoretical per-probe Hutchinson variance ---")
A_N = torch.linalg.solve(K_tilde, dK_dl)
A_M_op = torch.linalg.solve(A_M, DTDpTD) / (-sigmasq.item())  # signed so that sum = (trace - first)
# actually A_M_op doesn't directly estimate anything; the effective operator whose trace we estimate is
# -1/s2 * (s2 I + DTD)^{-1} DTD'TD. With Rademacher probes, variance adds as usual.
# Our estimate = first/s2 - (1/s2) Hutchinson( (s2 I + DTD)^{-1} DTD'TD )
# So relevant operator for variance is (1/s2) * (s2 I + DTD)^{-1} DTD'TD.
A_M_var = torch.linalg.solve(A_M, DTDpTD) / sigmasq.item()
# B D' form
B_dense = F_dense.conj().T @ torch.linalg.solve(K_tilde, F_dense)   # M x M
BDp = B_dense * Dp.unsqueeze(0)                                      # B · diag(D')
def hutch_var(A):
    fro2 = (A.abs() ** 2).sum().real.item()
    diag2 = (A.diagonal().abs() ** 2).sum().real.item()
    return 2 * (fro2 - diag2)
vN = hutch_var(A_N)
vM = hutch_var(A_M_var)
vBD = hutch_var(BDp)
print(f"N-space (A_N):        ||.||_F^2={torch.linalg.norm(A_N).item()**2:.3f}  std/probe={vN**0.5:.3f}")
print(f"M-space split (A_M):  ||.||_F^2={torch.linalg.norm(A_M_var).item()**2:.3f}  std/probe={vM**0.5:.3f}")
print(f"M-space B·D' (A'_M):  ||.||_F^2={torch.linalg.norm(BDp).item()**2:.3f}  std/probe={vBD**0.5:.3f}")
print(f"variance ratio (split/N): {vM/vN:.1f}x   (BD'/N): {vBD/vN:.2f}x")
print(f"tr(B·D') = {BDp.diagonal().sum().real.item():.6f}  (should match truth {trace_true:.6f})")
# Rank of A_N
sN = torch.linalg.svdvals(A_N)
print(f"numerical rank of A_N (>1e-10 of top): {int((sN > sN.max()*1e-10).sum())}")

# ---------- accuracy ----------
print("\n--- accuracy at T=50, CG tol=1e-8 ---")
for seed in [1, 2, 3]:
    e1, it1 = current_trace(50, seed=seed)
    e2, it2 = woodbury_trace(50, seed=seed)
    e3, it3 = bd_trace(50, seed=seed)
    print(f"seed={seed}  true={trace_true:.5f}  "
          f"current={e1:.5f} (err {abs(e1-trace_true):.2e})  "
          f"split-M={e2:.5f} (err {abs(e2-trace_true):.2e})  "
          f"BD'-M={e3:.5f} (err {abs(e3-trace_true):.2e})")

# ---------- estimator variance (many independent replications) ----------
print("\n--- estimator variance (independent replications) ---")
print(f"{'T':>5}  {'method':>10}  {'mean':>10}  {'std':>10}  {'1-probe std':>12}")
n_reps = 60
for T in [5, 20, 100]:
    cur = torch.tensor([current_trace (T, seed=1000 + k)[0] for k in range(n_reps)])
    spl = torch.tensor([woodbury_trace(T, seed=2000 + k)[0] for k in range(n_reps)])
    bdp = torch.tensor([bd_trace      (T, seed=3000 + k)[0] for k in range(n_reps)])
    for name, est in [("current", cur), ("split-M", spl), ("BD'-M", bdp)]:
        m, s = est.mean().item(), est.std().item()
        print(f"{T:>5}  {name:>10}  {m:>+10.4f}  {s:>10.4f}  {s*(T**0.5):>12.4f}")

# ---------- timing ----------
def _time(fn, reps=5):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps

# Exact M-space trace (O(M^3), deterministic)
def exact_mspace_trace():
    T_dense = torch.zeros((M, M), dtype=cdtype)
    e = torch.zeros(M, dtype=cdtype)
    for k in range(M):
        e.zero_(); e[k] = 1.0
        T_dense[:, k] = toeplitz(e)
    DTD_     = (ws.unsqueeze(-1) * T_dense) * ws.unsqueeze(0)
    DTDpTD_  = (ws.unsqueeze(-1) * T_dense) * Dp.unsqueeze(0)
    DTDpTD_  = DTDpTD_ @ T_dense * ws.unsqueeze(0)
    A_M_     = DTD_ + sigmasq * torch.eye(M, dtype=cdtype)
    second   = torch.linalg.solve(A_M_, DTDpTD_).diagonal().sum().real
    first    = v_kernel[center_tuple] * Dp.sum()
    return ((first - second) / sigmasq).real.item()

print("\n--- timing ---")
print(f"{'T':>5} {'current (ms)':>14} {'woodbury (ms)':>15} {'exact-M (ms)':>14} {'speedup cur/new':>17}")
t_exact = _time(exact_mspace_trace)
for T in [10, 50, 200]:
    t_cur = _time(lambda T=T: current_trace(T, seed=0))
    t_new = _time(lambda T=T: woodbury_trace(T, seed=0))
    print(f"{T:>5} {t_cur*1000:>14.2f} {t_new*1000:>15.2f} {t_exact*1000:>14.2f} {t_cur/t_new:>17.2f}x")
print(f"\nexact-M-space result: {exact_mspace_trace():.6f}  vs dense truth {trace_true:.6f}")
