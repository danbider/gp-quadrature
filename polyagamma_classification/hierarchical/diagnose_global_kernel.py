"""
Diagnostic: why does the global kernel recover worse than local?

We instrument the fit loop to record gradient decomposition (term1/term2),
effective SNR, and spectral-weight overlap at each iteration.  Then produce
figures that explain the learning dynamics.

Run:  ~/myenv/bin/python diagnose_global_kernel.py
"""
from __future__ import annotations

import math, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pg_hierarchical import (
    HierarchicalPGNegBinRegressor,
    _make_kernel,
    _build_hierarchical_spectral_state,
    _compute_hierarchical_mstep_gradient,
    _run_hierarchical_estep,
    _pg_omega_expectation,
    _sample_rademacher,
    _negative_binomial_total_count_gradient,
    HierarchicalSpectralState,
)

# ── data generation (same as notebook) ──────────────────────────────────

def rff_gp_sample(x, *, seed, lengthscale, variance, n_features=2000):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    omega = rng.normal(0, 1.0 / lengthscale, size=(n_features, 1))
    phase = rng.uniform(0, 2 * np.pi, size=n_features)
    weights = rng.normal(size=n_features)
    features = math.sqrt(2.0 * variance / n_features) * np.cos(x @ omega.T + phase)
    return features @ weights


TRUE = dict(ls_g=0.2, var_g=1.5, ls_h=0.05, var_h=0.5, r=5.0)
INIT = dict(ls_g=0.35, var_g=0.8, ls_h=0.1, var_h=0.3, r=3.0)


def generate_data(n=5000, L=2, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n)
    locations = np.zeros(n, dtype=int)
    per = n // L
    for ell in range(L):
        s = ell * per
        e = (ell + 1) * per if ell < L - 1 else n
        locations[s:e] = ell
    g = rff_gp_sample(x, seed=seed + 100, lengthscale=TRUE["ls_g"], variance=TRUE["var_g"])
    h = np.zeros(n)
    for ell in range(L):
        mask = locations == ell
        h[mask] = rff_gp_sample(x[mask], seed=seed + 200 + ell,
                                 lengthscale=TRUE["ls_h"], variance=TRUE["var_h"])
    f = g + h
    prob = 1.0 / (1.0 + np.exp(-f))
    y = rng.negative_binomial(TRUE["r"], 1 - prob).astype(np.float64)
    return x, y, locations, g, h, f


# ── instrumented fit ────────────────────────────────────────────────────

def instrumented_fit(x, y, locations, max_iter=80, seed=123):
    """Run the fit loop manually so we can record gradient diagnostics."""
    from cg import ConjugateGradients

    dev = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128
    n = len(x)

    X_t = torch.as_tensor(x[:, None], dtype=rdtype, device=dev)
    y_t = torch.as_tensor(y, dtype=rdtype, device=dev)
    loc_t = torch.as_tensor(locations, dtype=torch.long, device=dev)

    kg = _make_kernel(dimension=1, lengthscale=INIT["ls_g"], variance=INIT["var_g"])
    kh = _make_kernel(dimension=1, lengthscale=INIT["ls_h"], variance=INIT["var_h"])

    r = INIT["r"]
    log_r = torch.nn.Parameter(torch.tensor(math.log(r), dtype=rdtype, device=dev))
    r_opt = torch.optim.Adam([log_r], lr=0.02, maximize=True)
    opt_g = torch.optim.Adam(kg._gp_params_ref.parameters(), lr=0.05, maximize=True)
    opt_h = torch.optim.Adam(kh._gp_params_ref.parameters(), lr=0.05, maximize=True)

    kappa_fn = lambda: 0.5 * (y_t - r)
    pg_b_fn = lambda: y_t + r
    delta = (0.25 * pg_b_fn()).clone()

    records = []
    t0 = time.perf_counter()

    for outer in range(max_iter):
        r = float(torch.exp(log_r).item())
        kappa = kappa_fn()
        pg_b = pg_b_fn()

        spec = _build_hierarchical_spectral_state(
            X_t, loc_t, kg, kh,
            spectral_eps=1e-4, trunc_eps=1e-4, nufft_eps=1e-7,
            rdtype=rdtype, cdtype=cdtype, device=dev,
        )

        e_iters = 3
        delta, mean, sigma_diag, einfo = _run_hierarchical_estep(
            y_t, kappa, pg_b, delta, spec,
            max_iters=e_iters, rho0=0.7, gamma=1e-3, tol=1e-4,
            n_probes=10, cg_tol=1e-5,
            seed=seed + 1000 * outer, verbose=0,
        )

        mout = _compute_hierarchical_mstep_gradient(
            kappa, delta, spec,
            n_probes=10, cg_tol=1e-5,
            seed=seed + 1000 * outer + 500,
        )

        grad_g = mout["grad_g"].real
        grad_h = mout["grad_h"].real
        term1_g = mout["term1_g"].real
        term2_g = mout["term2_g"].real
        term1_h = mout["term1_h"].real
        term2_h = mout["term2_h"].real

        # ── spectral overlap diagnostic ──
        # How much of the global kernel's power sits in frequency bands
        # that the local kernel also occupies significantly?
        sg = spec.ws2_g.real
        sh = spec.ws2_h.real
        sg_n = sg / sg.sum()
        sh_n = sh / sh.sum()
        # Bhattacharyya overlap
        spectral_overlap = float(torch.sum(torch.sqrt(sg_n * sh_n)).item())

        # Fraction of global spectral mass in low-freq half vs high-freq half
        m = sg.numel()
        # Sort by frequency magnitude
        freq_mag = spec.xis.norm(dim=-1)
        order = torch.argsort(freq_mag)
        sg_sorted = sg[order]
        cumfrac = torch.cumsum(sg_sorted, 0) / sg_sorted.sum()
        sh_sorted = sh[order]
        cumfrac_h = torch.cumsum(sh_sorted, 0) / sh_sorted.sum()

        # ── gradient SNR ──
        # With stochastic trace (10 probes), the gradient is noisy.
        # Record the raw gradient and the magnitude of term1 vs term2.
        rec = {
            "iter": outer,
            "elapsed": time.perf_counter() - t0,
            "ls_g": float(kg.lengthscale),
            "var_g": float(kg.variance),
            "ls_h": float(kh.lengthscale),
            "var_h": float(kh.variance),
            "r": r,
            # Raw gradient components (index 0 = lengthscale, 1 = variance)
            "grad_g_ls": float(grad_g[0]),
            "grad_g_var": float(grad_g[1]),
            "grad_h_ls": float(grad_h[0]),
            "grad_h_var": float(grad_h[1]),
            # Term1 (data fit) and term2 (complexity)
            "t1_g_ls": float(term1_g[0]),
            "t1_g_var": float(term1_g[1]),
            "t2_g_ls": float(term2_g[0]),
            "t2_g_var": float(term2_g[1]),
            "t1_h_ls": float(term1_h[0]),
            "t1_h_var": float(term1_h[1]),
            "t2_h_ls": float(term2_h[0]),
            "t2_h_var": float(term2_h[1]),
            # Spectral overlap
            "spectral_overlap": spectral_overlap,
            # Spectral cumulative distributions (store as numpy for plotting)
            "_freq_mag_sorted": freq_mag[order].numpy(),
            "_sg_cumfrac": cumfrac.numpy(),
            "_sh_cumfrac": cumfrac_h.numpy(),
            # Delta stats
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            # mean / sigma_diag stats
            "mean_norm": float(mean.norm()),
            "sigma_diag_mean": float(sigma_diag.mean()),
        }
        records.append(rec)

        # ── apply Adam steps (matching the real fit loop) ──
        raw_g = kg._gp_params_ref.raw
        raw_g.grad = torch.stack([
            grad_g[0].to(dtype=raw_g.dtype) * kg.lengthscale,
            grad_g[1].to(dtype=raw_g.dtype) * kg.variance,
            torch.tensor(0.0, dtype=raw_g.dtype),
        ])
        opt_g.step(); opt_g.zero_grad(set_to_none=True)

        raw_h = kh._gp_params_ref.raw
        raw_h.grad = torch.stack([
            grad_h[0].to(dtype=raw_h.dtype) * kh.lengthscale,
            grad_h[1].to(dtype=raw_h.dtype) * kh.variance,
            torch.tensor(0.0, dtype=raw_h.dtype),
        ])
        opt_h.step(); opt_h.zero_grad(set_to_none=True)

        if (outer + 1) % 3 == 0:
            grad_r = _negative_binomial_total_count_gradient(
                y_t, mean, sigma_diag, total_count=r, quadrature_nodes=12,
            )
            log_r.grad = (grad_r * r).to(dtype=log_r.dtype).detach()
            r_opt.step(); r_opt.zero_grad(set_to_none=True)
            r = float(torch.exp(log_r).item())

        if outer % 10 == 0:
            print(f"[{outer:3d}] ls_g={rec['ls_g']:.4f} var_g={rec['var_g']:.4f} "
                  f"ls_h={rec['ls_h']:.4f} var_h={rec['var_h']:.4f} r={rec['r']:.3f}")

    return records


# ── gradient SNR experiment ─────────────────────────────────────────────

def gradient_snr_experiment(x, y, locations, n_seeds=20):
    """
    At the TRUE hyperparameters, compute the M-step gradient with many
    different probe seeds to measure variance of the stochastic gradient.
    This tells us the signal-to-noise ratio of each gradient component.
    """
    dev = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X_t = torch.as_tensor(x[:, None], dtype=rdtype, device=dev)
    y_t = torch.as_tensor(y, dtype=rdtype, device=dev)
    loc_t = torch.as_tensor(locations, dtype=torch.long, device=dev)

    # Use true hyperparameters
    kg = _make_kernel(dimension=1, lengthscale=TRUE["ls_g"], variance=TRUE["var_g"])
    kh = _make_kernel(dimension=1, lengthscale=TRUE["ls_h"], variance=TRUE["var_h"])

    spec = _build_hierarchical_spectral_state(
        X_t, loc_t, kg, kh,
        spectral_eps=1e-4, trunc_eps=1e-4, nufft_eps=1e-7,
        rdtype=rdtype, cdtype=cdtype, device=dev,
    )

    # Run E-step to get a reasonable delta
    r = TRUE["r"]
    kappa = 0.5 * (y_t - r)
    pg_b = y_t + r
    delta = (0.25 * pg_b).clone()
    for _ in range(10):
        delta, mean, sigma_diag, _ = _run_hierarchical_estep(
            y_t, kappa, pg_b, delta, spec,
            max_iters=5, rho0=0.7, gamma=1e-3, tol=1e-4,
            n_probes=30, cg_tol=1e-5, seed=42, verbose=0,
        )

    # Now compute gradient many times with different probe seeds
    grad_samples = {"g_ls": [], "g_var": [], "h_ls": [], "h_var": [],
                    "t1_g_ls": [], "t1_g_var": [], "t2_g_ls": [], "t2_g_var": [],
                    "t1_h_ls": [], "t1_h_var": [], "t2_h_ls": [], "t2_h_var": []}

    for s in range(n_seeds):
        mout = _compute_hierarchical_mstep_gradient(
            kappa, delta, spec,
            n_probes=10, cg_tol=1e-5, seed=1000 + s * 77,
        )
        g_g = mout["grad_g"].real
        g_h = mout["grad_h"].real
        grad_samples["g_ls"].append(float(g_g[0]))
        grad_samples["g_var"].append(float(g_g[1]))
        grad_samples["h_ls"].append(float(g_h[0]))
        grad_samples["h_var"].append(float(g_h[1]))
        grad_samples["t1_g_ls"].append(float(mout["term1_g"].real[0]))
        grad_samples["t1_g_var"].append(float(mout["term1_g"].real[1]))
        grad_samples["t2_g_ls"].append(float(mout["term2_g"].real[0]))
        grad_samples["t2_g_var"].append(float(mout["term2_g"].real[1]))
        grad_samples["t1_h_ls"].append(float(mout["term1_h"].real[0]))
        grad_samples["t1_h_var"].append(float(mout["term1_h"].real[1]))
        grad_samples["t2_h_ls"].append(float(mout["term2_h"].real[0]))
        grad_samples["t2_h_var"].append(float(mout["term2_h"].real[1]))

    # Convert to arrays
    for k in grad_samples:
        grad_samples[k] = np.array(grad_samples[k])

    return grad_samples


# ── also: gradient at init vs true ──────────────────────────────────────

def gradient_at_params(x, y, locations, ls_g, var_g, ls_h, var_h, r_val,
                       n_warmup=10, n_probes=30, seed=42):
    """Compute well-averaged gradient at arbitrary hyperparameters."""
    dev = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128
    X_t = torch.as_tensor(x[:, None], dtype=rdtype, device=dev)
    y_t = torch.as_tensor(y, dtype=rdtype, device=dev)
    loc_t = torch.as_tensor(locations, dtype=torch.long, device=dev)

    kg = _make_kernel(dimension=1, lengthscale=ls_g, variance=var_g)
    kh = _make_kernel(dimension=1, lengthscale=ls_h, variance=var_h)

    spec = _build_hierarchical_spectral_state(
        X_t, loc_t, kg, kh,
        spectral_eps=1e-4, trunc_eps=1e-4, nufft_eps=1e-7,
        rdtype=rdtype, cdtype=cdtype, device=dev,
    )

    kappa = 0.5 * (y_t - r_val)
    pg_b = y_t + r_val
    delta = (0.25 * pg_b).clone()
    for _ in range(n_warmup):
        delta, mean, sigma_diag, _ = _run_hierarchical_estep(
            y_t, kappa, pg_b, delta, spec,
            max_iters=5, rho0=0.7, gamma=1e-3, tol=1e-4,
            n_probes=30, cg_tol=1e-5, seed=seed, verbose=0,
        )

    mout = _compute_hierarchical_mstep_gradient(
        kappa, delta, spec,
        n_probes=n_probes, cg_tol=1e-5, seed=seed + 500,
    )
    return mout


# ── landscape scan along var_g ──────────────────────────────────────────

def scan_var_g(x, y, locations, var_g_values, seed=42, n_warmup=10):
    """Compute well-averaged gradient w.r.t. var_g along a 1D slice."""
    results = []
    for vg in var_g_values:
        print(f"  var_g={vg:.3f} ...", end=" ", flush=True)
        mout = gradient_at_params(
            x, y, locations,
            ls_g=TRUE["ls_g"], var_g=vg,
            ls_h=TRUE["ls_h"], var_h=TRUE["var_h"],
            r_val=TRUE["r"], n_warmup=n_warmup, n_probes=30, seed=seed,
        )
        g = mout["grad_g"].real
        results.append({
            "var_g": vg,
            "grad_var_g": float(g[1]),
            "t1_var_g": float(mout["term1_g"].real[1]),
            "t2_var_g": float(mout["term2_g"].real[1]),
        })
        print(f"grad={results[-1]['grad_var_g']:.4f}")
    return results


# ── plotting ────────────────────────────────────────────────────────────

def make_figures(records, snr_data, scan_data, x, g_true, h_true, locations):
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.35)

    iters = [r["iter"] for r in records]

    # ── Fig 1: hyperparameter paths (context) ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(iters, [r["ls_g"] for r in records], label="$\\ell_g$")
    ax.plot(iters, [r["ls_h"] for r in records], label="$\\ell_h$")
    ax.axhline(TRUE["ls_g"], color="C0", ls="--", alpha=0.5)
    ax.axhline(TRUE["ls_h"], color="C1", ls="--", alpha=0.5)
    ax.set_ylabel("lengthscale")
    ax.legend(fontsize=8)
    ax.set_title("(a) Lengthscale paths")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(iters, [r["var_g"] for r in records], label="$\\sigma^2_g$")
    ax.plot(iters, [r["var_h"] for r in records], label="$\\sigma^2_h$")
    ax.axhline(TRUE["var_g"], color="C0", ls="--", alpha=0.5)
    ax.axhline(TRUE["var_h"], color="C1", ls="--", alpha=0.5)
    ax.set_ylabel("variance")
    ax.legend(fontsize=8)
    ax.set_title("(b) Variance paths")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(iters, [r["r"] for r in records])
    ax.axhline(TRUE["r"], color="green", ls="--", alpha=0.5)
    ax.set_ylabel("$r$")
    ax.set_title("(c) Shape parameter path")

    # ── Fig 2: raw gradient components ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(iters, [r["grad_g_ls"] for r in records], label="$\\partial L/\\partial \\ell_g$", alpha=0.8)
    ax.plot(iters, [r["grad_h_ls"] for r in records], label="$\\partial L/\\partial \\ell_h$", alpha=0.8)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_ylabel("gradient (raw)")
    ax.legend(fontsize=8)
    ax.set_title("(d) Lengthscale gradients")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(iters, [r["grad_g_var"] for r in records], label="$\\partial L/\\partial \\sigma^2_g$", alpha=0.8)
    ax.plot(iters, [r["grad_h_var"] for r in records], label="$\\partial L/\\partial \\sigma^2_h$", alpha=0.8)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_ylabel("gradient (raw)")
    ax.legend(fontsize=8)
    ax.set_title("(e) Variance gradients")

    # ── Fig 3: term1 vs term2 decomposition for variance ──
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(iters, [r["t1_g_var"] for r in records], label="term1 (data)", alpha=0.8)
    ax.plot(iters, [r["t2_g_var"] for r in records], label="term2 (complexity)", alpha=0.8)
    ax.plot(iters, [r["grad_g_var"] for r in records], "k-", lw=2, alpha=0.6, label="grad (t1-t2)/2")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.legend(fontsize=7)
    ax.set_ylabel("component magnitude")
    ax.set_title("(f) $\\sigma^2_g$ gradient decomposition")

    # ── Fig 4: gradient SNR at true hyperparams ──
    ax = fig.add_subplot(gs[2, 0:2])
    labels = ["$\\partial L/\\partial \\ell_g$", "$\\partial L/\\partial \\sigma^2_g$",
              "$\\partial L/\\partial \\ell_h$", "$\\partial L/\\partial \\sigma^2_h$"]
    keys = ["g_ls", "g_var", "h_ls", "h_var"]
    colors_snr = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    positions = np.arange(len(keys))
    means = [snr_data[k].mean() for k in keys]
    stds = [snr_data[k].std() for k in keys]
    snrs = [abs(m) / s if s > 0 else float("inf") for m, s in zip(means, stds)]

    bars = ax.bar(positions, means, yerr=stds, capsize=5, color=colors_snr, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_ylabel("gradient value (20 seeds)")
    ax.set_title("(g) Gradient mean $\\pm$ std at TRUE hyperparams (SNR = |mean|/std)")
    for i, snr in enumerate(snrs):
        ax.text(i, means[i] + stds[i] + 0.005, f"SNR={snr:.1f}",
                ha="center", fontsize=9, fontweight="bold")

    # ── Fig 5: term1 vs term2 SNR breakdown ──
    ax = fig.add_subplot(gs[2, 2])
    t1_keys = ["t1_g_var", "t2_g_var", "t1_h_var", "t2_h_var"]
    t1_labels = ["t1 $\\sigma^2_g$", "t2 $\\sigma^2_g$", "t1 $\\sigma^2_h$", "t2 $\\sigma^2_h$"]
    t1_means = [snr_data[k].mean() for k in t1_keys]
    t1_stds = [snr_data[k].std() for k in t1_keys]
    t1_snrs = [abs(m) / s if s > 0 else float("inf") for m, s in zip(t1_means, t1_stds)]

    alphas = [0.7, 0.4, 0.7, 0.4]
    bar_colors = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"]
    for bi in range(len(t1_keys)):
        ax.bar(bi, t1_means[bi], yerr=t1_stds[bi], capsize=4,
               color=bar_colors[bi], alpha=alphas[bi])
    ax.set_xticks(range(len(t1_keys)))
    ax.set_xticklabels(t1_labels, fontsize=8)
    ax.set_title("(h) Term1/Term2 SNR (variance)")
    for i, snr in enumerate(t1_snrs):
        ax.text(i, t1_means[i] + t1_stds[i] + 0.002, f"{snr:.1f}",
                ha="center", fontsize=8)

    # ── Fig 6: spectral overlap evolution ──
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(iters, [r["spectral_overlap"] for r in records], "ko-", ms=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Bhattacharyya overlap")
    ax.set_title("(i) Spectral overlap $S_g$ vs $S_h$")
    ax.set_ylim(0, 1)

    # ── Fig 7: spectral CDFs at iter 0 and final ──
    ax = fig.add_subplot(gs[3, 1])
    for idx, label, ls in [(0, "iter 0", "--"), (-1, f"iter {records[-1]['iter']}", "-")]:
        freqs = records[idx]["_freq_mag_sorted"]
        ax.plot(freqs, records[idx]["_sg_cumfrac"], f"C0{ls}", label=f"$S_g$ {label}", lw=1.5)
        ax.plot(freqs, records[idx]["_sh_cumfrac"], f"C1{ls}", label=f"$S_h$ {label}", lw=1.5)
    ax.set_xlabel("$|\\xi|$")
    ax.set_ylabel("cumulative spectral mass")
    ax.legend(fontsize=7)
    ax.set_title("(j) Spectral CDFs: global vs local")

    # ── Fig 8: landscape scan of grad(var_g) ──
    ax = fig.add_subplot(gs[3, 2])
    vgs = [s["var_g"] for s in scan_data]
    grads = [s["grad_var_g"] for s in scan_data]
    t1s = [s["t1_var_g"] for s in scan_data]
    t2s = [s["t2_var_g"] for s in scan_data]
    ax.plot(vgs, grads, "ko-", ms=5, lw=2, label="grad $\\sigma^2_g$")
    ax.plot(vgs, [0.5 * t for t in t1s], "b--", lw=1, alpha=0.6, label="0.5 $\\times$ term1")
    ax.plot(vgs, [-0.5 * t for t in t2s], "r--", lw=1, alpha=0.6, label="$-$0.5 $\\times$ term2")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.axvline(TRUE["var_g"], color="green", ls="--", lw=1.5, label=f"true={TRUE['var_g']}")
    ax.set_xlabel("$\\sigma^2_g$")
    ax.set_ylabel("gradient")
    ax.legend(fontsize=7)
    ax.set_title("(k) $\\partial L / \\partial \\sigma^2_g$ landscape\n(other params at true)")

    # ── Fig 9: cancellation ratio ──
    ax = fig.add_subplot(gs[4, 0])
    # |grad| / max(|term1|, |term2|)  -- how much cancellation?
    cancel_g_var = [abs(r["grad_g_var"]) / max(abs(r["t1_g_var"]), abs(r["t2_g_var"]), 1e-12)
                    for r in records]
    cancel_h_var = [abs(r["grad_h_var"]) / max(abs(r["t1_h_var"]), abs(r["t2_h_var"]), 1e-12)
                    for r in records]
    ax.plot(iters, cancel_g_var, label="$\\sigma^2_g$", lw=1.5)
    ax.plot(iters, cancel_h_var, label="$\\sigma^2_h$", lw=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("|grad| / max(|t1|, |t2|)")
    ax.set_title("(l) Cancellation ratio (lower = more cancellation)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.5)

    # ── Fig 10: term1/term2 decomposition for lengthscale ──
    ax = fig.add_subplot(gs[4, 1])
    ax.plot(iters, [r["t1_g_ls"] for r in records], label="term1 (data)", alpha=0.8)
    ax.plot(iters, [r["t2_g_ls"] for r in records], label="term2 (complexity)", alpha=0.8)
    ax.plot(iters, [r["grad_g_ls"] for r in records], "k-", lw=2, alpha=0.6, label="grad (t1-t2)/2")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.legend(fontsize=7)
    ax.set_ylabel("component magnitude")
    ax.set_xlabel("iteration")
    ax.set_title("(m) $\\ell_g$ gradient decomposition")

    # ── Fig 11: effective gradient magnitude (log scale) ──
    ax = fig.add_subplot(gs[4, 2])
    ax.semilogy(iters, [abs(r["grad_g_ls"]) for r in records], label="$|\\nabla \\ell_g|$", alpha=0.7)
    ax.semilogy(iters, [abs(r["grad_g_var"]) for r in records], label="$|\\nabla \\sigma^2_g|$", alpha=0.7)
    ax.semilogy(iters, [abs(r["grad_h_ls"]) for r in records], label="$|\\nabla \\ell_h|$", ls="--", alpha=0.7)
    ax.semilogy(iters, [abs(r["grad_h_var"]) for r in records], label="$|\\nabla \\sigma^2_h|$", ls="--", alpha=0.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel("|gradient|")
    ax.legend(fontsize=7)
    ax.set_title("(n) Gradient magnitudes (log scale)")

    plt.savefig("global_kernel_diagnostic.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("global_kernel_diagnostic.png", bbox_inches="tight", dpi=150)
    print("\nSaved: global_kernel_diagnostic.pdf / .png")
    plt.close()


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1: Generate data")
    print("=" * 60)
    x, y, locations, g_true, h_true, f_true = generate_data(n=5000, L=2, seed=42)

    print("\n" + "=" * 60)
    print("STEP 2: Instrumented fit (80 iters)")
    print("=" * 60)
    records = instrumented_fit(x, y, locations, max_iter=80, seed=123)

    print("\n" + "=" * 60)
    print("STEP 3: Gradient SNR at true hyperparameters (20 seeds)")
    print("=" * 60)
    snr_data = gradient_snr_experiment(x, y, locations, n_seeds=20)

    for k in ["g_ls", "g_var", "h_ls", "h_var"]:
        m, s = snr_data[k].mean(), snr_data[k].std()
        print(f"  {k:8s}: mean={m:+.5f}  std={s:.5f}  SNR={abs(m)/s:.2f}")

    print("\n" + "=" * 60)
    print("STEP 4: Landscape scan along var_g")
    print("=" * 60)
    var_g_values = np.linspace(0.5, 3.5, 13)
    scan_data = scan_var_g(x, y, locations, var_g_values, seed=42)

    print("\n" + "=" * 60)
    print("STEP 5: Making figures")
    print("=" * 60)
    make_figures(records, snr_data, scan_data, x, g_true, h_true, locations)

    # ── Summary printout ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    # Gradient cancellation at true params
    print("\nAt TRUE hyperparameters:")
    for k in ["g_var", "h_var"]:
        m, s = snr_data[k].mean(), snr_data[k].std()
        t1m = snr_data[f"t1_{k}"].mean()
        t2m = snr_data[f"t2_{k}"].mean()
        cancel = abs(m) / max(abs(t1m), abs(t2m))
        print(f"  {k}: term1={t1m:+.5f}  term2={t2m:+.5f}  "
              f"grad={m:+.5f}  cancellation={1-cancel:.1%}")

    # Landscape flatness
    print("\nLandscape scan: zero crossing of grad(var_g) at:")
    for i in range(len(scan_data) - 1):
        if scan_data[i]["grad_var_g"] * scan_data[i+1]["grad_var_g"] < 0:
            # Linear interpolation
            g1, g2 = scan_data[i]["grad_var_g"], scan_data[i+1]["grad_var_g"]
            v1, v2 = scan_data[i]["var_g"], scan_data[i+1]["var_g"]
            v_cross = v1 + (v2 - v1) * (-g1) / (g2 - g1)
            print(f"  var_g ~ {v_cross:.3f}  (true = {TRUE['var_g']})")


if __name__ == "__main__":
    main()
