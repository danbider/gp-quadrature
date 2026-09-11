"""
SI appendix: SKI training CG-tolerance robustness (for reviewer 2).

Shows that tightening SKI's *training* conjugate-gradient tolerance just slows it down for a modest
RMSE gain -- which justifies the cg_tolerance=1.0 default used in the main Fig 1D comparison.

Setup: SKI @ n=10k, preconditioner OFF (max_preconditioner_size=0), 50 Adam iters, lr 0.3, float32,
100x100 grid; sweep cg_tolerance over {1.0, 1e-1, 1e-2, 1e-3}. First run is a discarded warmup.
Prediction uses GPyTorch's default eval CG tol (0.01) throughout, so the reported RMSE differences
come only from the learned hyperparameters. Run on a QUIET machine for citable numbers.

  ~/myenv/bin/python paper_figures/scaling/ski_cgtol_appendix.py

Writes ski_cgtol_appendix.json next to this file.
"""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch                       # noqa: E402  (worker pins threads on import)
import worker as W                 # noqa: E402
from utils.ski import fit_ski_gp   # noqa: E402

T = 10_000
CG_TOLS = [1.0, 1e-1, 1e-2, 1e-3]


def run(cgtol):
    x, y, f = W.gen_data(T)
    xe, fe = W.gen_test_points()
    res = fit_ski_gp(x, y, kernel='SE', max_iters=W.MAX_ITERS, lr=0.3,
                     init_lengthscale=W.INIT_LS, init_outputscale=W.INIT_VAR, init_noise=W.INIT_NOISE,
                     dtype=torch.float32, num_trace_samples=1, cg_tolerance=cgtol,
                     target_grid_points=10_000, max_preconditioner_size=0, verbose=False)
    mean = W._predict_gpytorch(res['model'], res['likelihood'], xe)
    return {"cg_tol": cgtol, "learn_s": res['fit_time_sec'],
            "rmse": W.rmse(mean.numpy(), fe.numpy()), "final_loss": res['history']['loss'][-1]}


def main():
    _ = run(1.0)  # warmup, discarded
    rows = []
    print(f"{'cg_tol':>8} {'learn_s':>9} {'RMSE':>9} {'final_loss':>12}", flush=True)
    for c in CG_TOLS:
        r = run(c); rows.append(r)
        print(f"{c:>8.0e} {r['learn_s']:>9.2f} {r['rmse']:>9.5f} {r['final_loss']:>12.4f}", flush=True)
    out = HERE / "ski_cgtol_appendix.json"
    out.write_text(json.dumps({"setup": {"method": "SKI", "n": T, "grid": "100x100",
                   "max_preconditioner_size": 0, "max_iters": W.MAX_ITERS, "lr": 0.3,
                   "eval_cg_tolerance": 0.01, "metric": "RMSE = ||fhat-f||/sqrt(N) at test points"},
                   "rows": rows}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
