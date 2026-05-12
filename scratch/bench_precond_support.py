"""
Benchmark the mean-CG preconditioner (kronecker / jacobi / nystrom) on data sets
with different support geometries:

  * 2D uniform-in-square    (product measure, Kronecker should win)
  * 2D uniform-in-disk      (non-product)
  * 3D uniform-in-cube      (product measure)
  * 3D uniform-in-ball      (non-product)
  * 2D OISST                (non-product, land masked out)

Records CG iters and wall time for each preconditioner.

Usage:
    ~/myenv/bin/python scratch/bench_precond_support.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efgpnd import (
    NUFFT,
    ToeplitzND,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
    create_jacobi_precond,
    create_kronecker_precond,
    create_nystrom_precond,
)
from cg import ConjugateGradients
from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis


DTYPE = torch.float64
CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
#  Data generators
# ---------------------------------------------------------------------------

def _rng(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def uniform_square(n: int, d: int = 2, L: float = 2.0, seed: int = 0):
    g = _rng(seed)
    x = (torch.rand(n, d, generator=g, dtype=DTYPE) - 0.5) * 2.0 * L
    return x


def uniform_disk(n: int, d: int = 2, R: float = 2.0, seed: int = 0):
    g = _rng(seed)
    out = torch.empty(n, d, dtype=DTYPE)
    got = 0
    while got < n:
        batch = 2 * (n - got) + 32
        pts = (torch.rand(batch, d, generator=g, dtype=DTYPE) - 0.5) * 2.0 * R
        inside = (pts.pow(2).sum(dim=1) <= R * R)
        take = pts[inside][: n - got]
        out[got : got + take.shape[0]] = take
        got += take.shape[0]
    return out


def uniform_ball(n: int, R: float = 2.0, seed: int = 0):
    return uniform_disk(n, d=3, R=R, seed=seed)


def gp_like_y(x: torch.Tensor, lengthscale: float, seed: int = 0):
    """
    Build a synthetic y that has roughly the lengthscale we're fitting with.
    Draw a small set of anchor points and form y = sum of bumps.
    Cheap (not a true GP sample), but keeps the CG system well-conditioned
    and the target in the RKHS-ish regime.
    """
    g = _rng(seed + 1_000)
    n, d = x.shape
    K = 40
    anchors = x[torch.randint(0, n, (K,), generator=g)]
    weights = torch.randn(K, generator=g, dtype=DTYPE)
    dists2 = ((x[:, None, :] - anchors[None, :, :]) ** 2).sum(dim=2)
    y = (weights * torch.exp(-0.5 * dists2 / lengthscale ** 2)).sum(dim=1)
    y = y + 0.05 * torch.randn(n, generator=g, dtype=DTYPE)
    return y


def load_oisst_2d(n_sub: int, seed: int = 0):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "real" / "oisst"))
    from load_oisst import load_oisst_torch
    x, y = load_oisst_torch(variable="sst", n_sub=n_sub, seed=seed)
    return x.to(DTYPE), y.to(DTYPE)


# ---------------------------------------------------------------------------
#  One benchmark run: fix (x, y, kernel, eps, sigmasq) -> time each preconditioner
# ---------------------------------------------------------------------------

def _build_common(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: SquaredExponential,
    eps: float,
    sigmasq: float,
    nufft_eps: Optional[float] = None,
):
    if nufft_eps is None:
        nufft_eps = eps * 0.1
    device = x.device
    d = x.shape[1]
    # Use the data's own bounding box (what efgpnd does internally).
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    L = (x1 - x0).max().item()

    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                              use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(device=device, dtype=DTYPE)
    h_t = torch.tensor(h, device=device, dtype=DTYPE) if not torch.is_tensor(h) else h.to(DTYPE)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=CDTYPE) * h_t ** d)

    xcen = torch.zeros(d, device=device, dtype=DTYPE)
    nufft_op = NUFFT(x, xcen, h_t, nufft_eps, cdtype=CDTYPE, device=device)
    OUT = (mtot,) * d
    Fy = nufft_op.type1(y.to(CDTYPE), out_shape=OUT).reshape(-1)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h_t).to(dtype=CDTYPE)
    toeplitz = ToeplitzND(v_kernel, force_pow2=False)

    sigmasq_t = torch.tensor(sigmasq, device=device, dtype=DTYPE)
    A_apply = create_A_mean(ws, toeplitz, sigmasq_t, CDTYPE)
    rhs = ws * Fy

    center = tuple(((torch.tensor(v_kernel.shape, device=device) - 1) // 2).tolist())
    diag_scale = v_kernel[center].real

    # --- Non-separability diagnostic: compare v_kernel to its rank-1 separable
    # approximation v_sep(k) = ∏_j v_marg_j(k_j) / v(0)^(d-1).  Report the
    # relative Frobenius norm of the residual.
    ctr = (mtot - 1) // 2 * 2 + 1  # odd grid length of v_kernel (= 2m-1 for m=mtot)
    L_v = v_kernel.shape[0]
    cidx = (L_v - 1) // 2
    v_ctr = v_kernel[(cidx,) * d]
    v_marg = []
    for k in range(d):
        slc = [cidx] * d
        slc[k] = slice(None)
        v_marg.append(v_kernel[tuple(slc)])  # (L_v,)
    # Build v_sep = ⊗_k v_marg[k] / v_ctr^(d-1)
    v_sep = v_marg[0].clone()
    for k in range(1, d):
        v_sep = v_sep.unsqueeze(-1) * v_marg[k].view(*([1]*k), L_v)
    v_sep = v_sep / (v_ctr ** (d - 1))
    sep_err = torch.linalg.norm(v_kernel - v_sep) / torch.linalg.norm(v_kernel)

    return {
        "d": d,
        "mtot": int(mtot),
        "M": int(ws.numel()),
        "ws": ws,
        "v_kernel": v_kernel,
        "toeplitz": toeplitz,
        "A_apply": A_apply,
        "rhs": rhs,
        "sigmasq": sigmasq_t,
        "diag_scale": diag_scale,
        "h": h_t,
        "sep_err": float(sep_err.item()),
    }


def _precond(kind: str, ctx: Dict, *, nystrom_rank: int = 30):
    if kind == "none":
        return None, 0.0
    t0 = time.perf_counter()
    if kind == "jacobi":
        M_inv = create_jacobi_precond(ctx["ws"], ctx["sigmasq"],
                                      diag_scale=ctx["diag_scale"])
    elif kind == "kronecker":
        M_inv = create_kronecker_precond(ctx["ws"], ctx["v_kernel"],
                                         ctx["sigmasq"], d=ctx["d"],
                                         mtot_1d=ctx["mtot"],
                                         cdtype=CDTYPE, rdtype=DTYPE)
    elif kind == "nystrom":
        M_inv = create_nystrom_precond(ctx["A_apply"], M=ctx["M"],
                                       sigmasq_scalar=ctx["sigmasq"],
                                       rank=nystrom_rank, oversample=10,
                                       seed=0, cdtype=CDTYPE, rdtype=DTYPE)
    elif kind == "kron+jac":
        # SPD additive combo: (M_k^{-1} + M_j^{-1}) / 2. Sum of SPD is SPD.
        M_k = create_kronecker_precond(ctx["ws"], ctx["v_kernel"],
                                       ctx["sigmasq"], d=ctx["d"],
                                       mtot_1d=ctx["mtot"],
                                       cdtype=CDTYPE, rdtype=DTYPE)
        M_j = create_jacobi_precond(ctx["ws"], ctx["sigmasq"],
                                    diag_scale=ctx["diag_scale"])
        def M_inv(v):
            return 0.5 * (M_k(v) + M_j(v))
    elif kind == "kron_scaled":
        # Scalar Hutchinson damping: α such that <z, A z> ≈ α <z, M_k z>,
        # i.e. tr(A) ≈ α tr(M_k).  We only have M_k^{-1} apply, so we estimate
        # tr(A M_k^{-1}) / M ≈ α directly (on average, A M_k^{-1} ≈ α I).
        M_k_raw = create_kronecker_precond(ctx["ws"], ctx["v_kernel"],
                                           ctx["sigmasq"], d=ctx["d"],
                                           mtot_1d=ctx["mtot"],
                                           cdtype=CDTYPE, rdtype=DTYPE)
        A_apply = ctx["A_apply"]
        n_probe = 6
        rg = torch.Generator(device="cpu").manual_seed(0)
        alpha_sum, nrm = 0.0, 0.0
        for _ in range(n_probe):
            z = torch.randn(ctx["M"], generator=rg, dtype=DTYPE).to(CDTYPE)
            z = z / torch.linalg.norm(z)
            # <z, A M_k^{-1} z> ≈ α <z, z> = α
            alpha_sum += torch.vdot(z, A_apply(M_k_raw(z))).real.item()
            nrm += 1.0
        alpha = alpha_sum / nrm
        def M_inv(v):
            return M_k_raw(v) / alpha
    elif kind == "kron+nys":
        # SPD additive combo: (M_k^{-1} + M_n^{-1}) / 2.
        M_k = create_kronecker_precond(ctx["ws"], ctx["v_kernel"],
                                       ctx["sigmasq"], d=ctx["d"],
                                       mtot_1d=ctx["mtot"],
                                       cdtype=CDTYPE, rdtype=DTYPE)
        M_n = create_nystrom_precond(ctx["A_apply"], M=ctx["M"],
                                     sigmasq_scalar=ctx["sigmasq"],
                                     rank=nystrom_rank, oversample=10,
                                     seed=0, cdtype=CDTYPE, rdtype=DTYPE)
        def M_inv(v):
            return 0.5 * (M_k(v) + M_n(v))
    elif kind == "kron_sym2":
        # Symmetric two-level multiplicative (SPD): pre- and post-Kron
        # sandwich around a Jacobi middle. Equivalent to one fine-grid
        # correction with Kron as the coarse solver.
        M_k = create_kronecker_precond(ctx["ws"], ctx["v_kernel"],
                                       ctx["sigmasq"], d=ctx["d"],
                                       mtot_1d=ctx["mtot"],
                                       cdtype=CDTYPE, rdtype=DTYPE)
        M_j = create_jacobi_precond(ctx["ws"], ctx["sigmasq"],
                                    diag_scale=ctx["diag_scale"])
        A_apply = ctx["A_apply"]
        def M_inv(v):
            # v  ──Jac──>  z1        (smoother)
            # z1 ── r = v - A z1
            # z2 = z1 + M_k(r)       (coarse correct)
            # r2 = v - A z2
            # z3 = z2 + Jac(r2)      (post-smooth — symmetric)
            z1 = M_j(v)
            r1 = v - A_apply(z1)
            z2 = z1 + M_k(r1)
            r2 = v - A_apply(z2)
            return z2 + M_j(r2)
    else:
        raise ValueError(kind)
    t1 = time.perf_counter()
    return M_inv, t1 - t0


def run_case(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    cg_tol: float,
    preconds: Tuple[str, ...] = ("none", "jacobi", "kronecker",
                                 "kron+jac", "kron_scaled"),
    max_iter: int = 2000,
) -> Dict:
    d = x.shape[1]
    kernel = SquaredExponential(dimension=d)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    ctx = _build_common(x, y, kernel, eps=eps, sigmasq=sigmasq)
    result = {
        "name": name, "d": d, "N": int(x.shape[0]),
        "lengthscale": lengthscale, "variance": variance,
        "sigmasq": sigmasq, "eps": eps, "cg_tol": cg_tol,
        "mtot": ctx["mtot"], "M": ctx["M"],
        "sep_err": ctx["sep_err"],
        "trials": {},
    }
    for kind in preconds:
        M_inv, setup_s = _precond(kind, ctx)
        x0 = torch.zeros_like(ctx["rhs"])
        t0 = time.perf_counter()
        cg = ConjugateGradients(ctx["A_apply"], ctx["rhs"], x0,
                                tol=cg_tol, max_iter=max_iter,
                                early_stopping=True, M_inv_apply=M_inv)
        sol = cg.solve()
        elapsed = time.perf_counter() - t0
        # final residual (just to make sure CG actually converged)
        r = ctx["rhs"] - ctx["A_apply"](sol)
        res_rel = (torch.linalg.norm(r) / torch.linalg.norm(ctx["rhs"])).item()
        result["trials"][kind] = {
            "iters": int(cg.iters_completed),
            "setup_s": setup_s,
            "cg_s": elapsed,
            "total_s": setup_s + elapsed,
            "res_rel": res_rel,
        }
    return result


# ---------------------------------------------------------------------------
#  Run the suite
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n2d", type=int, default=20_000)
    ap.add_argument("--n3d", type=int, default=20_000)
    ap.add_argument("--n_oisst", type=int, default=20_000)
    ap.add_argument("--ls2d", type=float, default=0.25)
    ap.add_argument("--ls3d", type=float, default=0.4)
    ap.add_argument("--ls_oisst", type=float, default=5.0)  # degrees
    ap.add_argument("--sigmasq", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).with_suffix(".json")))
    args = ap.parse_args()

    torch.set_default_dtype(DTYPE)

    results: List[Dict] = []

    # ---- 2D: square (product) vs disk (non-product) ----
    print("== 2D uniform SQUARE (product measure, L=2) ==")
    x_sq = uniform_square(args.n2d, d=2, L=2.0, seed=0)
    y_sq = gp_like_y(x_sq, lengthscale=args.ls2d, seed=0)
    r = run_case("2d_square", x_sq, y_sq,
                 lengthscale=args.ls2d, variance=1.0,
                 sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
    results.append(r); print(json.dumps(r, indent=2))

    print("== 2D uniform DISK (non-product, R=2) ==")
    x_dk = uniform_disk(args.n2d, d=2, R=2.0, seed=0)
    y_dk = gp_like_y(x_dk, lengthscale=args.ls2d, seed=0)
    r = run_case("2d_disk", x_dk, y_dk,
                 lengthscale=args.ls2d, variance=1.0,
                 sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
    results.append(r); print(json.dumps(r, indent=2))

    # ---- 3D: cube vs ball ----
    print("== 3D uniform CUBE (product measure, L=2) ==")
    x_cb = uniform_square(args.n3d, d=3, L=2.0, seed=0)
    y_cb = gp_like_y(x_cb, lengthscale=args.ls3d, seed=0)
    r = run_case("3d_cube", x_cb, y_cb,
                 lengthscale=args.ls3d, variance=1.0,
                 sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
    results.append(r); print(json.dumps(r, indent=2))

    print("== 3D uniform BALL (non-product, R=2) ==")
    x_bl = uniform_ball(args.n3d, R=2.0, seed=0)
    y_bl = gp_like_y(x_bl, lengthscale=args.ls3d, seed=0)
    r = run_case("3d_ball", x_bl, y_bl,
                 lengthscale=args.ls3d, variance=1.0,
                 sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
    results.append(r); print(json.dumps(r, indent=2))

    # ---- OISST (2D lon/lat with land mask, heavily non-product) ----
    try:
        print("== 2D OISST SST (land-masked, non-product) ==")
        x_oi, y_oi = load_oisst_2d(args.n_oisst, seed=0)
        # center y to avoid a huge DC spike
        y_oi = y_oi - y_oi.mean()
        r = run_case("oisst", x_oi, y_oi,
                     lengthscale=args.ls_oisst, variance=float(y_oi.var().item()),
                     sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
        results.append(r); print(json.dumps(r, indent=2))
    except Exception as exc:
        print(f"[skip OISST] {exc}")

    outp = Path(args.out)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote -> {outp}")


if __name__ == "__main__":
    main()
