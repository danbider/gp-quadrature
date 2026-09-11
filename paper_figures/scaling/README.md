# Synthetic GP-regression scaling study (Fig 1D)

Cost-vs-accuracy comparison of **EFGP** against three approximate-GP baselines —
**SGPR (m=49)**, **SGPR (m=1024)**, and **SKI** — on synthetic 2-D squared-exponential
(SE) GP regression, scaling the number of observations `n` from 10⁴ to 10⁶. This is
panel **D** of Figure 1 in the paper.

Two things are measured per fit:
1. **Wall-clock** for hyperparameter learning (50 Adam iterations, Type-II MLE) **plus
   posterior-mean prediction at the held-out test points** — both are inside the timer.
2. **Latent-recovery accuracy**: standard RMSE of the predicted posterior mean against
   the *true* latent field `f`, `RMSE = ||fhat - f|| / sqrt(N)` (no demeaning, no
   normalization), evaluated at **held-out test points** (not training points)
   (`rmse` in `worker.py`).

## Files

| file | role |
|---|---|
| `run_benchmark.py` | driver: runs each `(method, n)` fit in an isolated subprocess, enforces memory/time caps, records telemetry, writes `scaling_data.json` after every fit, prints a running table |
| `worker.py` | one `(method, n)` fit; owns data generation, the 50-iter loop, prediction, and timing |
| `scaling_data.json` | canonical results consumed by `../fig1_overview.py` (panel D) |
| `scaling_data_trainEval_archive.json` | backup of the previous *train-eval* results (auto-created on first run) |

Run everything with the project interpreter:

```bash
~/myenv/bin/python paper_figures/scaling/run_benchmark.py       # full sweep (~5–6 h)
~/myenv/bin/python paper_figures/fig1_overview.py               # regenerate Fig 1
```

## Synthetic data (fixed, method-neutral)

- Inputs: `x ~ Uniform([0,1]²)`, seed 0; **nested** across `n` (the first 10⁴ of the
  10⁶ draw equal the 10⁴ draw), so `f` is one consistent function at every `n`.
- Latent field: a fixed random-Fourier-feature (RFF) SE draw, `RFF_F=4000`,
  `RFF_SEED=12345`, `ℓ=0.05`, `var=1.0`; defined everywhere via `worker.rff_field(x)`.
- Targets: `y = f + ε`, `ε ~ N(0, 0.01)`, seed 1.
- **Test set**: 50,000 points `Uniform([0,1]²)`, `TEST_SEED=999`, disjoint from training,
  **the same points for every (method, n)**; true `f` there comes from the same RFF
  realization. Recovery RMSE is measured on these points.

> **Why RFF does not favor EFGP.** RFF draws *random* frequencies (Monte-Carlo from the
> spectral density); EFGP uses a *deterministic equispaced quadrature grid* of
> frequencies — algorithmically distinct. Crucially, none of the four estimators ever
> sees the frequencies: each receives only `(x, y)` and fits an SE kernel, so a latent
> `f` that happens to be a Fourier sum does not privilege EFGP's estimator over
> SGPR/SKI. `worker.rff_field` is factored out so a non-Fourier generator (exact
> dense-Cholesky draw at small `n`, or a spatial kernel-basis interpolant) can be dropped
> in for the deferred generator ablation.

Initialisation (all methods, deliberately wrong): `ℓ=0.3`, `var=0.5`, `noise=0.3`.

## Method settings

All methods: 2-D SE kernel, 50 optimizer iterations, Type-II MLE. Same input data.

### EFGP (`efgpnd.EFGPND`)
- `eps = 1e-3` (Fourier quadrature tolerance → grid size), `nufft_eps = eps*0.1 = 1e-4`
- `cg_tol = 1e-4`, `noise_floor = 1e-5`
- `refresh_grid_every = 5` (rebuild the Fourier grid every 5 iters; matches the paper's `B=5`)
- Preconditioner: **kronecker** (mean-CG and trace-CG), `trace_samples = 1`
- Optimizer: Adam, `lr = 0.3`; dtype float64

### SKI (`utils.ski.fit_ski_gp`, GPyTorch `GridInterpolationKernel`)
- **Grid: fixed 100×100** for all `n` (`target_grid_points=10000` →
  `round(√10000)=100`/dim). Cubic (order-4) interpolation, `use_toeplitz=True`.
- Training `cg_tolerance = 1.0` (loose — SKI's inner CG is the cost bottleneck);
  `max_cg_iterations = 1000`, `num_trace_samples = 1`.
- `max_preconditioner_size = 0` (**preconditioner disabled**). The pivoted-Cholesky CG
  preconditioner was measured ~7× slower at N=10⁴ with identical RMSE (and a worse
  marginal likelihood): with the Toeplitz structure + noise floor and loose CG tol, CG
  converges in ~1 step, so the preconditioner is pure per-iteration overhead. Disabling it
  is the faster, charitable-to-SKI configuration.
- Prediction: GPyTorch default `eval_cg_tolerance = 0.01` (posterior-mean solve).
- Optimizer: Adam, `lr = 0.3`; dtype **float32**; `GaussianLikelihood`, `ConstantMean`.

  **Grid-size justification.** SKI grid resolution is set by the kernel *lengthscale*,
  not by `n`. Spacing 0.01 on [0,1]² gives ~5 nodes per true lengthscale (ℓ=0.05; ~3 per
  SKI-learned ℓ≈0.026–0.031) — enough for cubic interpolation. The fixed 100×100 grid
  equals GPyTorch 1.15.2's `gpytorch.utils.grid.choose_grid_size(train_x, ratio=1.0)`
  default **at n=10⁴**; at larger `n` that √n heuristic would demand up to a 1000×1000
  (10⁶-node) grid — intractable on a laptop and far finer than a smooth kernel needs.
  GPyTorch's docs call `choose_grid_size` a sensitive starting point ("for 2–4D you
  can't use as fine a grid… adjust on a validation set"), so a fixed, lengthscale-matched
  grid is the principled and charitable-to-SKI choice. A saturation sweep
  (`G ∈ {50,100,200}`) is the deferred SI robustness check.

### SGPR (`utils.sgpr.fit_sgpr`, Titsias VFE via `InducingPointKernel`)
- Two arms: `m = 49` and `m = 1024` inducing points, locations learned, init = random
  training subset (`inducing_seed=0`).
- Optimizer: Adam, `lr = 0.3`; dtype float64; `ZeroMean`, `ScaleKernel(RBFKernel)`.
  (lr matched to EFGP/SKI so all methods share Adam lr 0.3 / 50 iters.)

## Sweep, caps, and expected drop-outs

- Sizes: `[10k, 100k, 250k, 500k, 1M]` (250k is SKI's largest feasible `n` and its
  headline "over an hour" point).
- Order: `efgp → sgpr49 → sgpr1024 → ski` (fast/high-value first, slow SKI last).
- Memory cap: **14 GB** RSS (of 17 GB physical) → `oom`.
- Per-method time caps: `efgp/sgpr49 = 1800s`, `sgpr1024 = 18000s`, `ski = 7200s`
  → `timeout`. Once a method drops out at some `n`, larger `n` are marked without running.

Expected (laptop): EFGP ok at all sizes (sub-second); SGPR-49 ok at all sizes; SGPR-1024
ok at `{10k,100k,500k}` (500k ≈ 3.8 h) and OOM at 1M; SKI ok at `{10k,100k,250k}` (250k
≈ 71 min) and OOM at ≥500k.

## Fairness & contention controls

Because EFGP fits are <1 s while the baselines are minutes-to-hours and memory-heavy,
the baselines are the ones exposed to contention/swapping/throttling. Controls:

- **Serial isolation**: one fit at a time, isolated subprocess — never concurrent.
- **Identical compute budget**: the worker pins `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB` threads
  + `torch.set_num_threads` = physical core count for *every* method (finufft honours
  `OMP_NUM_THREADS`). Recorded as `threads` per record.
- **Pre-flight quiescence gate**: before each fit, wait (bounded) until 1-min load and
  available RAM are sane; leftover `worker.py` processes are flagged.
- **Per-fit telemetry** in each record: `peak_rss_gb`, `max_load1`, `min_avail_gb`,
  `swap_delta_mb`, and flags `swap_suspected` / `contention_suspected` → re-run any fit
  that trips a flag rather than trust it.
- **Cooldown** (`COOLDOWN_S=15`) before each fit to avoid thermal carry-over.
- **Repeat cheap fits**: fits whose first run is `< CHEAP_SEC (60 s)` run `K_REPS=3`×;
  `time` is the median (`sec_all` / `sec_spread` / `n_reps` recorded). Expensive fits run
  once (can't repeat multi-hour fits) with full telemetry.

Tunable via env vars: `BENCH_THREADS`, `BENCH_MAX_ITERS`, `BENCH_NEVAL`, `BENCH_CHEAP_SEC`,
`BENCH_K_REPS`, `BENCH_COOLDOWN_S`, `BENCH_LOAD_GATE`, `BENCH_MIN_FREE_GB`,
`BENCH_GATE_MAX_WAIT_S`.

## Output schema (`scaling_data.json`)

```jsonc
{
  "config": { /* full settings above, self-documenting */ },
  "results": [
    {
      "method": "efgp", "T": 10000, "status": "ok",
      "eval": "test", "n_test": 50000, "threads": 8,
      "time": 0.51,            // median t_total (learning + prediction) — plotted on x
      "t_learn": 0.47, "t_predict": 0.04, "t_total": 0.51,
      "sec_all": [0.51, 0.52, 0.50], "sec_median": 0.51, "sec_spread": 0.02, "n_reps": 3,
      "rmse": 0.0027,          // latent-recovery RMSE at TEST points — plotted on y
      "ls": 0.05, "var": 1.0, "noise": 0.01,
      "peak_rss_gb": 0.8, "max_load1": 7.9, "min_avail_gb": 9.1,
      "swap_delta_mb": 0.0, "swap_suspected": false, "contention_suspected": false
    }
    // failures carry status "oom" | "timeout" | "error" (+ wall_at_kill_s / error)
    // skipped-larger-n carry a "note"
  ]
}
```

## Deferred (scaffold only — not run in the initial sweep)
- SKI grid **saturation sweep** (`G ∈ {50,100,200}` at one `n`) — SI robustness for the
  grid choice.
- Non-Fourier **ground-truth ablation** (swap `rff_field` for an exact/kernel-basis
  generator) — shows the RFF generator does not change the ranking.
