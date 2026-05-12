"""
Jacobi vs Kronecker on the same 5 cases as bench_precond_support.py, but with
a Matern kernel instead of SE. Matern's spectrum decays as a power law
(not Gaussian), so the effective rank of A is higher and both preconditioners
face a harder problem.

Tested with nu ∈ {1.5, 2.5} (Matern32 and Matern52 — the common choices).

Usage:
    ~/myenv/bin/python scratch/bench_precond_matern.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
)
from cg import ConjugateGradients
from kernels.matern import Matern
from utils.kernels import get_xis

from scratch.bench_precond_support import (
    DTYPE, CDTYPE,
    uniform_square, uniform_disk, uniform_ball, gp_like_y, load_oisst_2d,
)
from scratch.kron_svd_precond import create_kronecker_svd_precond


def _build_common(x, y, kernel, eps, sigmasq, nufft_eps=None):
    if nufft_eps is None:
        nufft_eps = eps * 0.1
    device = x.device
    d = x.shape[1]
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

    return {
        "d": d, "mtot": int(mtot), "M": int(ws.numel()),
        "ws": ws, "v_kernel": v_kernel, "toeplitz": toeplitz,
        "A_apply": A_apply, "rhs": rhs, "sigmasq": sigmasq_t,
        "diag_scale": diag_scale, "h": h_t,
    }


def run_case(name, x, y, *, nu, lengthscale, variance, sigmasq,
             eps, cg_tol, max_iter=2000):
    d = x.shape[1]
    kernel = Matern(dimension=d, nu=nu)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    ctx = _build_common(x, y, kernel, eps=eps, sigmasq=sigmasq)
    out = {"name": name, "d": d, "N": int(x.shape[0]),
           "nu": nu, "lengthscale": lengthscale, "variance": variance,
           "sigmasq": sigmasq, "eps": eps, "cg_tol": cg_tol,
           "mtot": ctx["mtot"], "M": ctx["M"], "trials": {}}

    for kind in ("jacobi", "kronecker", "kron_svd"):
        t0 = time.perf_counter()
        rank1_err = None
        if kind == "jacobi":
            M_inv = create_jacobi_precond(ctx["ws"], ctx["sigmasq"],
                                          diag_scale=ctx["diag_scale"])
        elif kind == "kronecker":
            M_inv = create_kronecker_precond(
                ctx["ws"], ctx["v_kernel"], ctx["sigmasq"],
                d=ctx["d"], mtot_1d=ctx["mtot"],
                cdtype=CDTYPE, rdtype=DTYPE,
            )
        else:  # kron_svd
            M_inv = create_kronecker_svd_precond(
                ctx["ws"], ctx["v_kernel"], ctx["sigmasq"],
                d=ctx["d"], mtot_1d=ctx["mtot"],
                cdtype=CDTYPE, rdtype=DTYPE, hooi_iter=5,
            )
            rank1_err = getattr(M_inv, "rank1_err", None)
        setup_s = time.perf_counter() - t0
        x0 = torch.zeros_like(ctx["rhs"])
        t1 = time.perf_counter()
        cg = ConjugateGradients(ctx["A_apply"], ctx["rhs"], x0,
                                tol=cg_tol, max_iter=max_iter,
                                early_stopping=True, M_inv_apply=M_inv)
        sol = cg.solve()
        cg_s = time.perf_counter() - t1
        r = ctx["rhs"] - ctx["A_apply"](sol)
        res_rel = (torch.linalg.norm(r) / torch.linalg.norm(ctx["rhs"])).item()
        out["trials"][kind] = {
            "iters": int(cg.iters_completed),
            "setup_s": setup_s, "cg_s": cg_s,
            "total_s": setup_s + cg_s, "res_rel": res_rel,
            "rank1_err": rank1_err,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n2d", type=int, default=50_000)
    ap.add_argument("--n3d", type=int, default=30_000)
    ap.add_argument("--n_oisst", type=int, default=50_000)
    ap.add_argument("--ls2d", type=float, default=0.25)
    ap.add_argument("--ls3d", type=float, default=0.4)
    ap.add_argument("--ls_oisst", type=float, default=5.0)
    ap.add_argument("--sigmasq", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--nus", type=str, default="1.5,2.5",
                    help="Comma-separated list of nu values")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).with_suffix(".json")))
    args = ap.parse_args()

    torch.set_default_dtype(DTYPE)
    nus = [float(s) for s in args.nus.split(",")]

    # Pre-build data tensors (reuse across nus)
    x_sq = uniform_square(args.n2d, d=2, L=2.0, seed=0)
    x_dk = uniform_disk(args.n2d, d=2, R=2.0, seed=0)
    x_cb = uniform_square(args.n3d, d=3, L=2.0, seed=0)
    x_bl = uniform_ball(args.n3d, R=2.0, seed=0)
    try:
        x_oi, y_oi_raw = load_oisst_2d(args.n_oisst, seed=0)
        y_oi = y_oi_raw - y_oi_raw.mean()
        have_oisst = True
    except Exception as exc:
        print(f"[skip OISST] {exc}")
        have_oisst = False

    results: List[Dict] = []
    for nu in nus:
        print(f"\n====== Matern nu = {nu} ======")

        y_sq = gp_like_y(x_sq, lengthscale=args.ls2d, seed=0)
        r = run_case(f"2d_square_nu{nu}", x_sq, y_sq, nu=nu,
                     lengthscale=args.ls2d, variance=1.0,
                     sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
        results.append(r); print(json.dumps(r, indent=2))

        y_dk = gp_like_y(x_dk, lengthscale=args.ls2d, seed=0)
        r = run_case(f"2d_disk_nu{nu}", x_dk, y_dk, nu=nu,
                     lengthscale=args.ls2d, variance=1.0,
                     sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
        results.append(r); print(json.dumps(r, indent=2))

        y_cb = gp_like_y(x_cb, lengthscale=args.ls3d, seed=0)
        r = run_case(f"3d_cube_nu{nu}", x_cb, y_cb, nu=nu,
                     lengthscale=args.ls3d, variance=1.0,
                     sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
        results.append(r); print(json.dumps(r, indent=2))

        y_bl = gp_like_y(x_bl, lengthscale=args.ls3d, seed=0)
        r = run_case(f"3d_ball_nu{nu}", x_bl, y_bl, nu=nu,
                     lengthscale=args.ls3d, variance=1.0,
                     sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
        results.append(r); print(json.dumps(r, indent=2))

        if have_oisst:
            r = run_case(f"oisst_nu{nu}", x_oi, y_oi, nu=nu,
                         lengthscale=args.ls_oisst,
                         variance=float(y_oi.var().item()),
                         sigmasq=args.sigmasq, eps=args.eps, cg_tol=args.cg_tol)
            results.append(r); print(json.dumps(r, indent=2))

    outp = Path(args.out)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote -> {outp}")


if __name__ == "__main__":
    main()
