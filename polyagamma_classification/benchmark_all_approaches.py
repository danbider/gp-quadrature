"""
Benchmark all hierarchical solver approaches on the mosquito dataset.
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

# Load data
df = pd.read_pickle(ROOT / "data" / "mosquito" / "italy_ovitrap.pkl")
df = df.dropna(subset=['value'])
X = (df['week'].values / 52.0).reshape(-1, 1).astype(np.float64)
y = df['value'].values.astype(np.float64)
locations = df['site_idx'].values.astype(np.int64)
n_sites = len(np.unique(locations))
print(f"Data: n={len(y):,}, L={n_sites} sites")

kwargs = dict(
    lengthscale_g_init=0.15, variance_g_init=2.0,
    lengthscale_h_init=0.06, variance_h_init=1.0,
    total_count=3.0, learn_total_count=True, total_count_lr=0.02,
    max_iter=15, e_step_iters=3, lr=0.05, cg_tol=1e-5,
    seed=42, verbose=1,
)

# --- Approach B: Block CD + Schur M-step ---
print("\n" + "="*60)
print("Approach B: Block CD + Schur M-step")
print("="*60)
from hierarchical.pg_hierarchical_blockcd import HierarchicalPGNegBinRegressorBlockCD
t0 = time.perf_counter()
model_b = HierarchicalPGNegBinRegressorBlockCD(cd_sweeps=5, **kwargs)
model_b.fit(X, y, locations)
t_b = time.perf_counter() - t0
print(f"Time: {t_b:.1f}s ({t_b/15:.1f}s/iter)")

# --- Approach C: Schur CG ---
print("\n" + "="*60)
print("Approach C: Schur CG (feature-space M-step)")
print("="*60)
from hierarchical.pg_hierarchical_schur import HierarchicalPGNegBinRegressorSchurCG
t0 = time.perf_counter()
model_c = HierarchicalPGNegBinRegressorSchurCG(**kwargs)
model_c.fit(X, y, locations)
t_c = time.perf_counter() - t0
print(f"Time: {t_c:.1f}s ({t_c/15:.1f}s/iter)")

# --- Approach D: Direct Schur ---
print("\n" + "="*60)
print("Approach D: Direct Schur (dense factorization)")
print("="*60)
from hierarchical.pg_hierarchical_schur import HierarchicalPGNegBinRegressorDirect
t0 = time.perf_counter()
model_d = HierarchicalPGNegBinRegressorDirect(**kwargs)
model_d.fit(X, y, locations)
t_d = time.perf_counter() - t0
print(f"Time: {t_d:.1f}s ({t_d/15:.1f}s/iter)")

# --- Summary ---
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Approach':<35s} {'Total':>8s} {'Per iter':>10s} {'vs Joint':>10s}")
print(f"{'A: Joint solver (baseline)':<35s} {'1331.1':>8s} {'88.7':>10s} {'1.0x':>10s}")
print(f"{'B: Block CD + Schur M-step':<35s} {t_b:>8.1f} {t_b/15:>10.1f} {1331.1/t_b:>10.1f}x")
print(f"{'C: Schur CG (feat-space M)':<35s} {t_c:>8.1f} {t_c/15:>10.1f} {1331.1/t_c:>10.1f}x")
print(f"{'D: Direct Schur (dense)':<35s} {t_d:>8.1f} {t_d/15:>10.1f} {1331.1/t_d:>10.1f}x")

print(f"\n{'':35s} {'ls_g':>8s} {'var_g':>8s} {'ls_h':>8s} {'var_h':>8s} {'r':>8s} {'MAE':>8s}")
for name, m in [("B: Block CD", model_b), ("C: Schur CG", model_c), ("D: Direct", model_d)]:
    print(f"{name:<35s} {m.lengthscale_g_:>8.4f} {m.variance_g_:>8.2f} {m.lengthscale_h_:>8.4f} {m.variance_h_:>8.2f} {m.total_count_:>8.3f} {m.history_[-1]['mae']:>8.1f}")
