"""Same as compare_efgpnd_dist.py but at small lengthscale, low noise."""
import importlib.util, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec); sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_data(n, d, true_ls, true_var, true_noise):
    dt = torch.float64
    torch.manual_seed(42)
    x = torch.rand(n, d, dtype=dt)
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var, num_features=10000, seed=42)
    torch.manual_seed(1)
    y = y + torch.sqrt(torch.tensor(true_noise, dtype=dt)) * torch.randn(n, dtype=dt)
    return x, y


def run_one(EFGPND_cls, x, y, *, init_ls, init_var, init_noise, eps, cg_tol,
            lr, max_iters, J, noise_floor, seed, log=False):
    d = x.shape[1]
    torch.manual_seed(seed)
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
    model = EFGPND_cls(x, y, kernel=kernel, sigmasq=init_noise, eps=eps, estimate_params=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    traj = []
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        opt.step()
        if log:
            traj.append((model.kernel.get_hyper('lengthscale'),
                         model.kernel.get_hyper('variance'),
                         model._gp_params.sig2.item()))
    return (model.kernel.get_hyper('lengthscale'),
            model.kernel.get_hyper('variance'),
            model._gp_params.sig2.item(), traj)


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std()), float(t.std() / max(1, len(vals)) ** 0.5)


def regime(label, *, true_ls, true_noise, init_ls, init_noise, n=200, d=2,
           true_var=1.0, init_var=0.5, lr=0.05, max_iters=80, J=1, nseeds=8,
           eps=1e-4, cg_tol=1e-5, noise_floor=1e-7):
    print("\n" + "#" * 80)
    print(f"# {label}")
    print(f"# true: ℓ={true_ls}  σ_f²={true_var}  σ_n²={true_noise}")
    print(f"# init: ℓ={init_ls}  σ_f²={init_var}  σ_n²={init_noise}")
    print(f"# n={n}  d={d}  lr={lr}  iters={max_iters}  J={J}  noise_floor={noise_floor}")
    print("#" * 80)
    x, y = make_data(n=n, d=d, true_ls=true_ls, true_var=true_var, true_noise=true_noise)

    cur = sys.modules['efgpnd_cur']
    old = sys.modules['efgpnd_old']
    cfg = dict(init_ls=init_ls, init_var=init_var, init_noise=init_noise,
               eps=eps, cg_tol=cg_tol, lr=lr, max_iters=max_iters,
               J=J, noise_floor=noise_floor)

    rows = []
    t0 = time.time()
    for s in range(nseeds):
        l_old, v_old, n_old, _ = run_one(old.EFGPND, x, y, **cfg, seed=s)
        l_cur, v_cur, n_cur, _ = run_one(cur.EFGPND, x, y, **cfg, seed=s)
        rows.append((s, l_old, v_old, n_old, l_cur, v_cur, n_cur))
        print(f"  seed={s:2d}  OLD ℓ={l_old:.5f} σ_f²={v_old:.4f} σ_n²={n_old:.4e}  | "
              f"CUR ℓ={l_cur:.5f} σ_f²={v_cur:.4f} σ_n²={n_cur:.4e}")
    print(f"  ...{time.time()-t0:.1f}s total")

    l_o = [r[1] for r in rows]; v_o = [r[2] for r in rows]; n_o = [r[3] for r in rows]
    l_c = [r[4] for r in rows]; v_c = [r[5] for r in rows]; n_c = [r[6] for r in rows]
    print("  ---- distribution summary (z = Δmean / pooled-SE):")
    for name, oldvals, curvals, truth in [
        ('ℓ',    l_o, l_c, true_ls),
        ('σ_f²', v_o, v_c, true_var),
        ('σ_n²', n_o, n_c, true_noise),
    ]:
        mo, so, seo = mean_std(oldvals); mc, sc, sec = mean_std(curvals)
        z = (mc - mo) / max((seo**2 + sec**2)**0.5, 1e-30)
        print(f"     {name:>5}: TRUE={truth:8.5g}   OLD={mo:8.5g}±{seo:7.2g}   CUR={mc:8.5g}±{sec:7.2g}   z={z:+.2f}σ")


def main():
    cur = load(REPO / 'efgpnd.py', 'efgpnd_cur')
    old = load(Path('/tmp/efgpnd_compare/efgpnd_old.py'), 'efgpnd_old')

    # 1) Small ls, low noise (the user's setup of interest)
    regime("Small ls (0.03), low noise (1e-3), warm init",
           true_ls=0.03, true_noise=1e-3, init_ls=0.05, init_noise=5e-3,
           n=400, d=2, lr=0.05, max_iters=120, J=1, nseeds=6)

    # 2) Same regime but warmer init (close to truth)
    regime("Small ls (0.03), low noise (1e-3), tighter init",
           true_ls=0.03, true_noise=1e-3, init_ls=0.04, init_noise=2e-3,
           n=400, d=2, lr=0.03, max_iters=120, J=1, nseeds=6)

    # 3) Same with J=8 (denoised trace) — should converge much closer
    regime("Small ls (0.03), low noise (1e-3), J=8 denoised",
           true_ls=0.03, true_noise=1e-3, init_ls=0.05, init_noise=5e-3,
           n=400, d=2, lr=0.05, max_iters=80, J=8, nseeds=4)


if __name__ == '__main__':
    main()
