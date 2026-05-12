"""
Sweep: lengthscale vs. trace estimator cost / accuracy in the few-probes regime.

For each lengthscale l, we set up EFGP at eps=1e-6 (which determines M),
compute the dense ground-truth T_1^l = tr(K_tilde^{-1} F D' F*), and measure:

  - current N-space Hutchinson (3 NUFFTs + 1 CG per probe)
  - BD' M-space Hutchinson    (2 Toeplitz + 1 CG per probe, no NUFFT)
  - exact M-space dense       (O(M^3), deterministic)

at T in {1, 2, 4, 10}. Reports wall time, abs error, relative error, and
error std across independent replications.
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

d = 1
n = 4000
var_val, noise_val = 1.0, 0.1
eps = 1e-6

x = torch.rand(n, d, dtype=dtype)
# Use a fixed y (its value does not affect the trace term we are studying)
y = sample_gp_fast(x, length_scale=0.1, variance=var_val,
                   noise_variance=noise_val, num_samples=1).squeeze()

lengthscales = [0.02, 0.04, 0.08, 0.15, 0.3, 0.6]
T_probes_list = [1, 2, 4, 10]
N_REPS = 40         # replications for variance estimate
CG_TOL = 1e-8


def build_machinery(ls):
    """Return dict of all EFGP pieces for a given lengthscale."""
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls, init_variance=var_val)
    sigmasq = torch.tensor(noise_val, dtype=dtype)
    L = (x.max(dim=0).values - x.min(dim=0).values).max()
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                              use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(cdtype) * h**d)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)
    Dp = Dprime[:, kernel.hypers.index("lengthscale")]
    M = ws.numel()

    OUT = (mtot,) * d
    xcen = torch.zeros(d, dtype=dtype)
    nufft_op = NUFFT(x, xcen, h, 1e-12, cdtype=cdtype)

    def fadj(v):
        out = nufft_op.type1(v, out_shape=OUT)
        return out.reshape(v.shape[0], -1) if v.ndim == 2 else out.reshape(-1)

    def fwd(fk):
        if fk.ndim == 1:
            return nufft_op.type2(fk.reshape(OUT), out_shape=OUT)
        T_ = fk.shape[0]
        return nufft_op.type2(fk.reshape((T_,) + OUT), out_shape=OUT)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, sigmasq, cdtype)
    center_tuple = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_scale = v_kernel[center_tuple].real
    jacobi = create_jacobi_precond(ws, sigmasq, diag_scale=diag_scale)

    return dict(ws=ws, Dp=Dp, M=M, mtot=mtot, sigmasq=sigmasq,
                fadj=fadj, fwd=fwd, toeplitz=toeplitz, A_apply=A_apply,
                jacobi=jacobi, v_kernel=v_kernel, center_tuple=center_tuple,
                OUT=OUT)


def dense_truth(m):
    """Dense ground-truth trace via explicit F."""
    M = m["M"]
    F_dense = torch.zeros((n, M), dtype=cdtype)
    I_M = torch.eye(M, dtype=cdtype)
    for k in range(M):
        F_dense[:, k] = m["fwd"](I_M[k])
    D2 = m["ws"] * m["ws"].conj()
    K_tilde = (F_dense * D2) @ F_dense.conj().T + m["sigmasq"] * torch.eye(n, dtype=cdtype)
    dK_dl = (F_dense * m["Dp"]) @ F_dense.conj().T
    return torch.linalg.solve(K_tilde, dK_dl).diagonal().sum().real.item()


def current_trace(m, T_probes, seed):
    g = torch.Generator().manual_seed(seed)
    Z = torch.empty((T_probes, n), dtype=dtype)
    Z.bernoulli_(0.5, generator=g).mul_(2).sub_(1)
    Z = Z.to(cdtype)
    fadjZ = m["fadj"](Z)
    Di_FZ = m["Dp"] * fadjZ
    w = m["fwd"](Di_FZ).reshape(T_probes, -1)
    gamma = m["ws"] * m["toeplitz"](Di_FZ)
    cg = ConjugateGradients(m["A_apply"], gamma, torch.zeros_like(gamma),
                            tol=CG_TOL, M_inv_apply=m["jacobi"], early_stopping=True)
    beta = cg.solve()
    FDbeta = m["fwd"](m["ws"] * beta).reshape(T_probes, -1)
    alpha = (w - FDbeta) / m["sigmasq"]
    return (Z * alpha).sum(dim=1).real.mean().item()


def bd_trace(m, T_probes, seed):
    g = torch.Generator().manual_seed(seed)
    V = torch.empty((T_probes, m["M"]), dtype=dtype)
    V.bernoulli_(0.5, generator=g).mul_(2).sub_(1)
    V = V.to(cdtype)
    u = m["Dp"] * V
    y1 = m["toeplitz"](u)
    rhs = m["ws"] * y1
    cg = ConjugateGradients(m["A_apply"], rhs, torch.zeros_like(rhs),
                            tol=CG_TOL, M_inv_apply=m["jacobi"], early_stopping=True)
    c = cg.solve()
    y2 = m["toeplitz"](m["ws"] * c)
    Bu = (y1 - y2) / m["sigmasq"]
    return (V * Bu).sum(dim=1).real.mean().item()


def exact_dense(m):
    M = m["M"]
    T_dense = torch.zeros((M, M), dtype=cdtype)
    e = torch.zeros(M, dtype=cdtype)
    for k in range(M):
        e.zero_(); e[k] = 1.0
        T_dense[:, k] = m["toeplitz"](e)
    ws = m["ws"]
    DTD = (ws.unsqueeze(-1) * T_dense) * ws.unsqueeze(0)
    left = (ws.unsqueeze(-1) * T_dense) * m["Dp"].unsqueeze(0)
    DTDpTD = left @ T_dense * ws.unsqueeze(0)
    A_ = DTD + m["sigmasq"] * torch.eye(M, dtype=cdtype)
    second = torch.linalg.solve(A_, DTDpTD).diagonal().sum().real
    first = m["v_kernel"][m["center_tuple"]] * m["Dp"].sum()
    return ((first - second) / m["sigmasq"]).real.item()


def _time(fn, reps=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# -------- sweep --------
header = f"{'ell':>6}  {'M':>4}  {'method':>12}  {'T':>3}  {'time (ms)':>10}  {'|err|':>10}  {'rel err':>9}  {'err std':>10}"
print(header)
print("-" * len(header))

for ls in lengthscales:
    m = build_machinery(ls)
    truth = dense_truth(m)
    # exact dense
    t_ex = _time(lambda: exact_dense(m))
    e_ex = abs(exact_dense(m) - truth)
    print(f"{ls:>6.3f}  {m['M']:>4d}  {'exact-M':>12}  {'-':>3}  {t_ex*1000:>10.2f}  {e_ex:>10.2e}  "
          f"{e_ex/abs(truth):>9.2e}  {0.0:>10.2e}")
    for T_ in T_probes_list:
        for name, fn in [("current", current_trace), ("BD'-M", bd_trace)]:
            # timing: average over N_REPS probes' worth of calls with distinct seeds
            seeds = list(range(7000, 7000 + N_REPS))
            # warm up
            fn(m, T_, seeds[0])
            t0 = time.perf_counter()
            ests = []
            for sd in seeds:
                ests.append(fn(m, T_, sd))
            dt = (time.perf_counter() - t0) / N_REPS
            ests = torch.tensor(ests)
            errs = (ests - truth).abs()
            err_std = (ests - truth).std().item()
            print(f"{ls:>6.3f}  {m['M']:>4d}  {name:>12}  {T_:>3d}  {dt*1000:>10.2f}  "
                  f"{errs.mean().item():>10.2e}  {errs.mean().item()/abs(truth):>9.2e}  "
                  f"{err_std:>10.2e}")
    print()
