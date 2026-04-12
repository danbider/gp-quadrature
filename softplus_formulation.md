# Matching GPyTorch's parameterization in EFGP

## Problem

EFGP uses `exp` to map raw parameters to positive space:
```
pos = exp(raw)       →   d(pos)/d(raw) = pos
```

GPyTorch uses `softplus`:
```
pos = log(1 + exp(raw))   →   d(pos)/d(raw) = sigmoid(raw)
```

GPyTorch also normalizes the MLL by `1/n` (in `ExactMarginalLogLikelihood.forward`: `res.div_(num_data)`).

These two differences cause the raw-space gradient to differ by a factor of `n * pos / sigmoid(softplus_inv(pos))`, which is ~635× for ls/noise at typical values and ~1157× for variance=2.0. The ratio grows with parameter value for exp but saturates for softplus, causing EFGP to overshoot variance specifically when using Adam.

## Changes to match GPyTorch

### 1. GPParams (kernels/kernel_params.py)

Replace the exp transform with softplus:

```python
# Current:
@property
def pos(self):
    return torch.exp(self.raw)

# New:
import torch.nn.functional as F

@property
def pos(self):
    return F.softplus(self.raw)
```

Update the inverse (used in `__init__` to set raw from initial positive values):

```python
# Current:
self.raw = nn.Parameter(torch.log(torch.tensor(init_values, dtype=dtype)))

# New:
def _softplus_inv(x):
    return torch.log(torch.exp(x) - 1)

self.raw = nn.Parameter(_softplus_inv(torch.tensor(init_values, dtype=dtype)))
```

### 2. compute_gradients (efgpnd.py, ~line 693)

The chain rule currently multiplies by `pos` (the exp Jacobian):

```python
# Current:
raw_grad = torch.stack([
    grads[i].detach() * pos_vec[i]    # d(NLL)/d(raw) = d(NLL)/d(pos) * pos
    for i in range(len(grads))
])
```

With softplus, multiply by `sigmoid(raw)` instead:

```python
# New:
raw_grad = torch.stack([
    grads[i].detach() * torch.sigmoid(self._gp_params.raw[i].detach())
    for i in range(len(grads))
])
```

### 3. Normalize by n

Divide the gradient by `n` before setting `.grad`:

```python
raw_grad = raw_grad / self.x.shape[0]
```

This makes EFGP's gradient scale match GPyTorch's, so the same learning rate produces the same step sizes.

### 4. Anywhere `pos` is read from `raw`

Search for any direct `torch.exp(self._gp_params.raw)` or `exp(raw)` outside of `GPParams.pos` and replace with `F.softplus(...)`. Key locations:
- `efgpnd.py` lines that read `sig2` from `_gp_params`
- Any kernel code that reads raw parameters directly

## Verification

After these changes, running SGD with the same lr as GPyTorch ExactGP should produce near-identical trajectories (confirmed by the softplus test in `scratch_softplus_test.py`).
