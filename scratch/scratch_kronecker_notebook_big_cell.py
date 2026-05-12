"""
Literal reproduction of the 'big data' EFGP cells from
scratch/hyperparameter_comparison.ipynb (cells 3, 5, 18-20).

Times each preconditioner option with the SAME code path the notebook uses.

Run: ~/myenv/bin/python -u scratch/scratch_kronecker_notebook_big_cell.py
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
dtype = torch.float64

# --- cell 3: ground-truth config ---
true_ls = 0.05
true_var = 1
true_noise = 0.01
d = 2

# --- cell 5: common init hypers (notebook uses these across methods) ---
init_ls = 0.3
init_var = 0.5
init_noise = 0.3
noise_floor = 1e-5

# --- cell 18: big data ---
n_big = 250_000
n_feat_rff = 2000
torch.manual_seed(1)
x_big = torch.rand(n_big, d, dtype=dtype)
f_big = sample_gp_rff(x_big, length_scale=true_ls, variance=true_var,
                     num_features=n_feat_rff, seed=0)
torch.manual_seed(1)
y_big = f_big + torch.sqrt(torch.tensor(true_noise, dtype=dtype)) * torch.randn(n_big, dtype=dtype)
print(f"Big data: n={n_big}, y std={y_big.std():.3f}", flush=True)

# --- cell 19: training config ---
lr = 0.3
EPSILON = 1e-3
cg_tol = 1e-4
max_iters = 50
J = 1


def run_efgp(precond: str):
    """Exact reproduction of cell 20's inner loop, with the precond
    option passed explicitly so all three paths use the same code."""
    kernel_efgp_big = SquaredExponential(
        dimension=d, init_lengthscale=init_ls, init_variance=init_var
    )
    training_log_eps = {
        'iter': [], 'lengthscale': [], 'variance': [], 'sigmasq': [],
    }
    model_eps = EFGPND(x_big, y_big, kernel=kernel_efgp_big,
                       sigmasq=0.3, eps=EPSILON, estimate_params=False,
                       opts={"mean_cg_preconditioner_type": precond})
    optimizer_eps = Adam(model_eps.parameters(), lr=lr)

    training_log_eps['iter'].append(0)
    training_log_eps['lengthscale'].append(model_eps.kernel.get_hyper('lengthscale'))
    training_log_eps['variance'].append(model_eps.kernel.get_hyper('variance'))
    training_log_eps['sigmasq'].append(model_eps._gp_params.sig2.item())

    t0 = time.time()
    for it in range(max_iters):
        optimizer_eps.zero_grad()
        model_eps.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                    noise_floor=noise_floor)
        optimizer_eps.step()

        ls = model_eps.kernel.get_hyper('lengthscale')
        var = model_eps.kernel.get_hyper('variance')
        sig = model_eps._gp_params.sig2.item()
        training_log_eps['iter'].append(it + 1)
        training_log_eps['lengthscale'].append(ls)
        training_log_eps['variance'].append(var)
        training_log_eps['sigmasq'].append(sig)
        if it % 10 == 0:
            print(f"  [{precond}] iter {it:>3}  l={ls:.4g}  sf2={var:.4g}  sn2={sig:.4g}",
                  flush=True)

    fit_time = time.time() - t0
    print(f"  [{precond}] Final: l={ls:.4g}, sf2={var:.4g}, sn2={sig:.4g}", flush=True)
    print(f"  [{precond}] fit time: {fit_time:.2f}s", flush=True)
    return fit_time


timings = {}
# run kronecker twice to show warmup/JIT variance
for precond in ["kronecker", "kronecker", "jacobi"]:
    print(f"\n--- EFGP EPSILON={EPSILON:.0e}, cg_tol={cg_tol:.0e}, precond={precond} ---",
          flush=True)
    t = run_efgp(precond)
    timings.setdefault(precond, []).append(t)

print("\n" + "=" * 60)
print("Summary (n=250k, d=2, eps=1e-3, cg_tol=1e-4, 50 Adam iters):")
print("=" * 60)
for precond, ts in timings.items():
    runs = ", ".join(f"{t:.2f}s" for t in ts)
    print(f"  {precond:<12s} runs=[{runs}]  best={min(ts):.2f}s")
