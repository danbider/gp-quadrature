# gp-quadrature
Fast Gaussian Process regression with equispaced quadrature rules.

See `efgpnd_basic_ex.ipynb` for typical usage.
See `Hyper_learning_sanitychecks.ipynb` and `efgpnd_sanity_checks.ipynb` for
approximation checks on posterior mean, variance, and hyperparameter gradients.

## Installation
Clone the repository and install the package in development mode:

```
git clone https://github.com/danbider/gp-quadrature.git
cd gp-quadrature
pip install -e .
```

## Branch layout

- `main` — core library, polished demo notebooks, sanity tests. Safe to share.
- `dev` — `main` plus all in-progress scratch scripts, diagnostics, and
  experimental figures. Default working branch. New exploratory code lands
  here; stabilized pieces migrate to `main` via cherry-pick or targeted merge.
- `backup/pre-reorg-2026-04-12` — tag snapshotting the pre-reorganization state.

Recommended local layout: keep the primary checkout on `dev` and add a
worktree for `main` so both branches are browseable side-by-side without
`git checkout` churn:

```
git worktree add ../gp-quadrature-main main
```

