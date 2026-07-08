"""
Synthetic NB-GP recovery gate: EFGP (Polya-Gamma) vs inducing-point SVGP.

Both methods fit the IDENTICAL generative model -- exp/logit link
``y ~ NB(r, p=sigmoid(f*))``, i.e. ``E[y] = r*exp(f*)`` -- so differences reflect only the
approximation strategy (Fourier features vs inducing points) + optimization, not model
mismatch. The dispersion ``r`` is LEARNED by both methods.

Ground truth: latent f* is an SE GP-prior draw (lengthscale 0.20, variance 1.0) via the
project's spectral sampler ``vanilla_gp_sampling.sample_gp_spectral_approx``.

Measures, per method: wall-clock fit time; normalized latent recovery (de-meaned to
absorb the r<->offset degeneracy) as normalized RMSE + Pearson correlation; recovered r
vs r_true; recovered lengthscale/outputscale; predictive count MAE.

Also runs a likelihood efficiency-parity micro-benchmark: our exp-link NB vs gpytorch's
built-in softplus NB share the same base class / quadrature, so per-step cost should match.

Run:  ~/myenv/bin/python scratch/synth_nb_gp_compare.py [--n 5000] [--sweep]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gpytorch  # noqa: E402
from vanilla_gp_sampling import sample_gp_spectral_approx  # noqa: E402
from utils.svgp_nb import fit_svgp_nb, ExpLinkNegativeBinomialLikelihood  # noqa: E402
from polyagamma_classification.pg_classifier import (  # noqa: E402
    PolyagammaGPNegativeBinomialRegressor,
)

LENGTHSCALE_TRUE = 0.20
VARIANCE_TRUE = 1.0
R_TRUE = 2.0


# --------------------------------------------------------------------------- data
def make_data(n: int, seed: int = 0):
    """x ~ U([-1,1]^2), f* ~ SE-GP(ls=0.20, var=1), y ~ NB(r_true, logits=f*)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * 2.0 - 1.0
    f_true = sample_gp_spectral_approx(
        x, length_scale=LENGTHSCALE_TRUE, variance=VARIANCE_TRUE, seed=seed + 1
    ).reshape(-1)
    y = torch.distributions.NegativeBinomial(
        total_count=R_TRUE, logits=f_true
    ).sample()
    return x, y, f_true


# ------------------------------------------------------------------------ metrics
def recovery_metrics(f_hat: np.ndarray, f_true: np.ndarray) -> dict:
    """De-mean both (offset degeneracy), then normalized RMSE + Pearson corr."""
    a = f_hat - f_hat.mean()
    b = f_true - f_true.mean()
    nrmse = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    corr = float(np.corrcoef(a, b)[0, 1])
    return {"nrmse": nrmse, "corr": corr}


# --------------------------------------------------------------------------- fits
def fit_efgp(x, y, f_true):
    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=1.0,
        learn_total_count=True,            # <-- r is LEARNED
        total_count_lr=0.05,
        total_count_update_frequency=1,
        total_count_quadrature_nodes=16,
        lengthscale_init=LENGTHSCALE_TRUE,
        variance_init=VARIANCE_TRUE,
        max_iter=50,
        e_step_iters=1,
        final_e_step_iters=2,
        rho0=0.7,
        gamma=1e-3,
        lr=0.05,
        n_e_probes=1,
        n_m_probes=1,
        cg_tol=1e-6,
        nufft_eps=1e-7,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        prediction_batch_size=256,
        predictive_variance_method="chebyshev",
        predictive_variance_chebyshev_nodes=7,
        use_exact_weighted_toeplitz_operator=True,
        random_state=0,
        device="cpu",
        store_history=True,
        verbose=0,
    )
    t0 = time.time()
    reg.fit(x_np, y_np)
    dt = time.time() - t0
    f_hat = np.asarray(reg.decision_function(x_np)).reshape(-1)
    resp = np.asarray(reg.predict_response_mean(x_np)).reshape(-1)
    return {
        "name": "EFGP-PG",
        "time": dt,
        "r": float(reg.total_count_),
        "lengthscale": float(np.ravel(reg.lengthscale_)[0]) if hasattr(reg, "lengthscale_") else float("nan"),
        "outputscale": float(reg.variance_) if hasattr(reg, "variance_") else float("nan"),
        "recovery": recovery_metrics(f_hat, f_true.numpy()),
        "count_mae": float(np.mean(np.abs(resp - y.numpy()))),
    }


def fit_svgp(x, y, f_true, num_inducing=200, max_iters=100, batch_size=None):
    t0 = time.time()
    out = fit_svgp_nb(
        x, y, kernel="SE", num_inducing=num_inducing, max_iters=max_iters,
        batch_size=batch_size, init_total_count=1.0,
        init_lengthscale=LENGTHSCALE_TRUE, init_outputscale=VARIANCE_TRUE,
        device="cpu", verbose=False,
    )
    dt = time.time() - t0                     # fit_svgp_nb also reports fit_time_sec
    f_hat = out["predict_latent_mean"](x).detach().cpu().numpy().reshape(-1)
    resp = out["predict_response_mean"](x).detach().cpu().numpy().reshape(-1)
    model = out["model"]
    return {
        "name": f"SVGP-NB (M={out['num_inducing']}{', bs=%d' % batch_size if batch_size else ''})",
        "time": dt,
        "r": out["total_count"],             # <-- r is LEARNED
        "lengthscale": float(model.base_kernel.lengthscale.detach().reshape(-1).mean()),
        "outputscale": float(model.covar_module.outputscale.detach()),
        "recovery": recovery_metrics(f_hat, f_true.numpy()),
        "count_mae": float(np.mean(np.abs(resp - y.numpy()))),
    }


# ---------------------------------------------------------- efficiency parity bench
def likelihood_parity_benchmark(n=4000, repeats=100, seed=0):
    """Time expected_log_prob (fwd, and fwd+bwd) for our exp-link NB vs built-in NB."""
    from linear_operator.operators import DiagLinearOperator

    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(n, generator=g) * 0.5
    var = torch.rand(n, generator=g) * 0.5 + 0.1
    obs = torch.distributions.NegativeBinomial(
        total_count=2.0, logits=mean
    ).sample().double()

    def make_dist():
        return gpytorch.distributions.MultivariateNormal(mean, DiagLinearOperator(var))

    ours = ExpLinkNegativeBinomialLikelihood(init_total_count=2.0)
    builtin = gpytorch.likelihoods.NegativeBinomialLikelihood()

    def time_fwd(lik):
        # warmup
        for _ in range(5):
            lik.expected_log_prob(obs, make_dist()).sum()
        t0 = time.time()
        for _ in range(repeats):
            lik.expected_log_prob(obs, make_dist()).sum()
        return (time.time() - t0) / repeats

    def time_fwd_bwd(lik):
        params = list(lik.parameters())
        for _ in range(5):
            lik.zero_grad(set_to_none=True)
            lik.expected_log_prob(obs, make_dist()).sum().backward()
        t0 = time.time()
        for _ in range(repeats):
            lik.zero_grad(set_to_none=True)
            lik.expected_log_prob(obs, make_dist()).sum().backward()
        return (time.time() - t0) / repeats

    f_ours, f_bi = time_fwd(ours), time_fwd(builtin)
    b_ours, b_bi = time_fwd_bwd(ours), time_fwd_bwd(builtin)
    return {
        "fwd_ours_ms": f_ours * 1e3, "fwd_builtin_ms": f_bi * 1e3,
        "fwd_ratio": f_ours / f_bi,
        "fwdbwd_ours_ms": b_ours * 1e3, "fwdbwd_builtin_ms": b_bi * 1e3,
        "fwdbwd_ratio": b_ours / b_bi,
    }


# ---------------------------------------------------------------------- reporting
def print_result(res):
    rec = res["recovery"]
    print(
        f"  {res['name']:<24s}  time={res['time']:7.2f}s  "
        f"r={res['r']:6.3f}  ls={res['lengthscale']:6.3f}  os={res['outputscale']:6.3f}  "
        f"corr={rec['corr']:.4f}  nrmse={rec['nrmse']:.4f}  countMAE={res['count_mae']:.4f}"
    )


def run_one(n, num_inducing=200, max_iters=100, batch_size=None):
    print(f"\n=== N={n} (r_true={R_TRUE}, ls_true={LENGTHSCALE_TRUE}, var_true={VARIANCE_TRUE}) ===")
    x, y, f_true = make_data(n)
    print(f"  counts: mean={float(y.mean()):.3f} var={float(y.var()):.3f} "
          f"max={int(y.max())} zero-frac={float((y == 0).float().mean()):.3f}")
    efgp = fit_efgp(x, y, f_true)
    svgp = fit_svgp(x, y, f_true, num_inducing=num_inducing, max_iters=max_iters, batch_size=batch_size)
    print_result(efgp)
    print_result(svgp)
    return efgp, svgp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    if not args.no_bench:
        print("=== likelihood efficiency-parity micro-benchmark (ours vs built-in NB) ===")
        b = likelihood_parity_benchmark()
        print(f"  expected_log_prob fwd:      ours={b['fwd_ours_ms']:.3f}ms  "
              f"builtin={b['fwd_builtin_ms']:.3f}ms  ratio={b['fwd_ratio']:.3f}")
        print(f"  expected_log_prob fwd+bwd:  ours={b['fwdbwd_ours_ms']:.3f}ms  "
              f"builtin={b['fwdbwd_builtin_ms']:.3f}ms  ratio={b['fwdbwd_ratio']:.3f}")
        parity_ok = b["fwd_ratio"] < 1.15 and b["fwdbwd_ratio"] < 1.15
        print(f"  PARITY {'OK' if parity_ok else 'FAIL'} "
              f"(ours within 15% of built-in; expect ~equal or faster)")
    else:
        parity_ok = True

    results = []
    if args.sweep:
        for n in (2000, 8000, 30000):
            bs = 2048 if n >= 30000 else None
            mi = 60 if n >= 30000 else 100
            results.append(run_one(n, batch_size=bs, max_iters=mi))
    else:
        results.append(run_one(args.n))

    # Gate: both learned r near truth, both recover latent well.
    print("\n=== GATE ===")
    ok = parity_ok
    for efgp, svgp in results:
        for res in (efgp, svgp):
            corr_ok = res["recovery"]["corr"] > 0.8
            r_ok = 0.3 * R_TRUE < res["r"] < 3.0 * R_TRUE
            print(f"  {res['name']:<24s}  corr>0.8:{corr_ok}  r-near-truth:{r_ok}")
            ok = ok and corr_ok and r_ok
    print(f"\n{'GATE PASSED' if ok else 'GATE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
