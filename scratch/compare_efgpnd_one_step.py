"""Single-step gradient comparison.

Build OLD and CURRENT EFGPND at *exactly* the same hypers, set the same RNG
seed, call compute_gradients once, and print the resulting parameter
gradients. Loops over a few hyper anchors so we see whether the gap is one-off
or systematic.
"""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_data(n=100, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01):
    dt = torch.float64
    torch.manual_seed(42)
    x = torch.rand(n, d, dtype=dt)
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var, num_features=5000, seed=42)
    torch.manual_seed(1)
    y = y + torch.sqrt(torch.tensor(true_noise, dtype=dt)) * torch.randn(n, dtype=dt)
    return x, y


def grad_at(EFGPND_cls, x, y, *, ls, var, sig2, eps, cg_tol, J, noise_floor, seed):
    d = x.shape[1]
    torch.manual_seed(seed)
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls, init_variance=var)
    model = EFGPND_cls(x, y, kernel=kernel, sigmasq=sig2, eps=eps, estimate_params=False)
    # Reseed right before gradient call so probes match across versions.
    torch.manual_seed(seed + 9001)
    model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
    grads = {}
    for n_, p in model.named_parameters():
        if p.grad is not None:
            grads[n_] = p.grad.detach().clone().cpu()
    stats = dict(getattr(model, 'last_gradient_stats', {}) or {})
    return grads, stats


def main():
    cur = load(REPO / 'efgpnd.py', 'efgpnd_cur')
    old = load(Path('/tmp/efgpnd_compare/efgpnd_old.py'), 'efgpnd_old')
    x, y = make_data()
    cfg = dict(eps=1e-4, cg_tol=1e-5, J=1, noise_floor=1e-5)

    anchors = [
        dict(ls=0.30, var=0.5, sig2=0.30),
        dict(ls=0.10, var=0.7, sig2=0.50),
        dict(ls=0.05, var=1.0, sig2=0.05),
        dict(ls=0.07, var=0.5, sig2=0.30),  # similar to where the trajectories diverge
    ]
    seeds = [0, 1, 2]

    for anchor in anchors:
        print("\n" + "=" * 90)
        print(f"hypers: ℓ={anchor['ls']}  σ_f²={anchor['var']}  σ_n²={anchor['sig2']}")
        print("=" * 90)
        for seed in seeds:
            g_old, s_old = grad_at(old.EFGPND, x, y, **anchor, **cfg, seed=seed)
            g_cur, s_cur = grad_at(cur.EFGPND, x, y, **anchor, **cfg, seed=seed)
            keys = sorted(set(g_old) & set(g_cur))
            print(f"  seed={seed}  cg(old/cur mean)={s_old.get('mean_cg_iters')}/{s_cur.get('mean_cg_iters')}  "
                  f"trace={s_old.get('trace_cg_iters')}/{s_cur.get('trace_cg_iters')}  "
                  f"mtot(old/cur)={s_old.get('mtot')}/{s_cur.get('mtot')}  M={s_old.get('feature_count')}/{s_cur.get('feature_count')}")
            for k in keys:
                a, b = g_old[k].flatten(), g_cur[k].flatten()
                if a.numel() == 1:
                    av, bv = float(a.item()), float(b.item())
                    den = max(abs(av), abs(bv), 1e-30)
                    print(f"     {k:>22}:  OLD={av:+.6e}  CUR={bv:+.6e}  Δ={bv-av:+.3e}  rel={(bv-av)/den:+.3e}")
                else:
                    diff = (b - a).abs().max().item()
                    den = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
                    print(f"     {k:>22}:  shape={tuple(g_old[k].shape)}  OLD‖·‖∞={a.abs().max():.3e}  CUR‖·‖∞={b.abs().max():.3e}  max|Δ|={diff:.3e}  rel={diff/den:.3e}")

        # J=16 averaged
        print("  --- J=16 (denoised) ---")
        g_old, s_old = grad_at(old.EFGPND, x, y, **anchor, **{**cfg, 'J': 16}, seed=0)
        g_cur, s_cur = grad_at(cur.EFGPND, x, y, **anchor, **{**cfg, 'J': 16}, seed=0)
        for k in sorted(set(g_old) & set(g_cur)):
            a, b = g_old[k].flatten(), g_cur[k].flatten()
            if a.numel() == 1:
                av, bv = float(a.item()), float(b.item())
                den = max(abs(av), abs(bv), 1e-30)
                print(f"     {k:>22}:  OLD={av:+.6e}  CUR={bv:+.6e}  Δ={bv-av:+.3e}  rel={(bv-av)/den:+.3e}")
            else:
                diff = (b - a).abs().max().item()
                den = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
                print(f"     {k:>22}:  shape={tuple(g_old[k].shape)}  OLD‖·‖∞={a.abs().max():.3e}  CUR‖·‖∞={b.abs().max():.3e}  max|Δ|={diff:.3e}  rel={diff/den:.3e}")


if __name__ == '__main__':
    main()
