"""
Answer to: "is Kron's win just about avoiding CG?"

Compares (over 30 Adam steps on the notebook data):
  jacobi     - diagonal precond, many CG iters
  kronecker  - Kronecker-structured precond, few CG iters
  dense_eig  - build A as dense m^d x m^d, Cholesky; CG converges in 1 iter
               (this is literally "skip CG, do dense direct solve")

Run: ~/myenv/bin/python -u scratch/scratch_dense_vs_kron.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.optim import Adam

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff

torch.set_default_dtype(torch.float64)
DT = torch.float64


def make_data(n, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = torch.rand(n, d, dtype=DT)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                      num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def run(x, y, d, precond, *, max_iters=30, eps=1e-3, cg_tol=1e-4, lr=0.3,
        init_ls=0.3, init_var=0.5, init_sigmasq=0.3, J=1, noise_floor=1e-5):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_sigmasq, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond})
    opt = Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    mean_cg, trace_cg = [], []
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        opt.step()
        s = model.last_gradient_stats
        mean_cg.append(s.get('mean_cg_iters'))
        trace_cg.append(s.get('trace_cg_iters'))
    total = time.perf_counter() - t0
    ls = model.kernel.get_hyper('lengthscale')
    var = model.kernel.get_hyper('variance')
    sig = model._gp_params.sig2.item()
    return dict(total=total, mean_cg=mean_cg, trace_cg=trace_cg,
                mtot=model.last_gradient_stats.get('mtot'),
                M=model.last_gradient_stats.get('feature_count'),
                ls=ls, var=var, sig=sig)


cases = [
    ("n=100k, d=2",  100_000, 2),
    ("n=250k, d=2",  250_000, 2),
]


if __name__ == "__main__":
    print("Dense direct vs Kron vs Jacobi (30 Adam steps, eps=1e-3, cg_tol=1e-4)", flush=True)
    print("init ℓ=0.3, σ_f²=0.5, σ_n²=0.3\n", flush=True)
    for name, n, d in cases:
        print(f"=== {name} ===", flush=True)
        x, y = make_data(n, d)
        results = {}
        for precond in ["kronecker", "jacobi", "dense_eig"]:
            try:
                r = run(x, y, d, precond)
                results[precond] = r
                cgm = r['mean_cg']; cgt = r['trace_cg']
                print(f"  {precond:<12s} total={r['total']:7.2f}s  "
                      f"mtot={r['mtot']} M={r['M']}  "
                      f"cg-mean(med)={sorted(cgm)[len(cgm)//2]}  "
                      f"cg-trace(med)={sorted(cgt)[len(cgt)//2]}  "
                      f"final ℓ={r['ls']:.4f}",
                      flush=True)
            except Exception as e:
                print(f"  {precond:<12s} FAILED: {type(e).__name__}: {e}", flush=True)

        if "kronecker" in results and "dense_eig" in results:
            k = results["kronecker"]['total']
            de = results["dense_eig"]['total']
            j = results["jacobi"]['total']
            print(f"  --> dense_eig / kron  = {de/k:.2f}x  (>1 means kron is faster)")
            print(f"      jacobi    / kron  = {j/k:.2f}x")
        print(flush=True)
