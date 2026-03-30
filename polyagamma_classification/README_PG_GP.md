# Polya-Gamma Gaussian Process (PG_GP)

A clean, self-contained implementation of Polya-Gamma augmented Gaussian Processes for binary classification using efficient NUFFT-based computations.

## Overview

This implementation provides a scalable approach to GP classification by:
- Using Polya-Gamma augmentation to handle the non-Gaussian likelihood
- Employing NUFFT (Non-Uniform Fast Fourier Transform) for efficient matrix-vector products
- Using conjugate gradients for solving linear systems
- Implementing Hutchinson's trace estimator for variance computations

## Usage

### Basic Example

```python
import torch
from kernels import SquaredExponential
from PG_GP import PolyagammaGP

# Set up data
n, d = 1000, 2
x = torch.rand(n, d) * 2 - 1  # Random points in [-1, 1]^d
y = torch.randint(0, 2, (n,)).float()  # Binary labels

# Initialize kernel
kernel = SquaredExponential(dimension=d, init_lengthscale=0.5, init_variance=2.0)

# Create model
model = PolyagammaGP(
    x=x, 
    y=y, 
    kernel=kernel,
    eps=1e-4,        # Spectral truncation tolerance
    nufft_eps=1e-8,  # NUFFT tolerance  
    cg_tol=1e-6      # Conjugate gradient tolerance
)

# Fit model (E-step)
m, Sigma_diags, acc, Sz, probes = model.fit(max_iters=20, verbose=True)

# Compute gradients (M-step)
term1, term2, term3 = model.m_step(m, J=20, verbose=True)
gradients = 0.5 * (term1 + term2 - term3)
```

### Key Methods

#### `fit(max_iters=20, rho0=1.0, gamma=1e-3, tol=1e-6, verbose=True, J=20)`
Performs the E-step using variational inference.

**Parameters:**
- `max_iters`: Maximum number of iterations
- `rho0`: Initial step size for natural gradient updates
- `gamma`: Step size decay parameter
- `tol`: Convergence tolerance
- `verbose`: Print progress information
- `J`: Number of probe vectors for Hutchinson estimator

**Returns:**
- `m`: Posterior mean
- `Sigma_diags`: Posterior variance diagonal
- `acc`: Predictive accuracy
- `Sz`: Hutchinson samples
- `probes`: Probe vectors used

#### `m_step(m, J=20, verbose=True)`
Computes gradients w.r.t. kernel hyperparameters.

**Parameters:**
- `m`: Posterior mean from E-step
- `J`: Number of probe vectors for trace estimation
- `verbose`: Print intermediate results

**Returns:**
- `term1`: First gradient term (quadratic form)
- `term2`: Second gradient term (trace of covariance)
- `term3`: Third gradient term (trace of kernel derivative)

### Configuration Parameters

- **`eps`**: Controls spectral truncation accuracy. Smaller values = more accurate but slower.
- **`nufft_eps`**: NUFFT tolerance. Smaller values = more accurate NUFFT.
- **`cg_tol`**: Conjugate gradient tolerance for all linear solves. Smaller values = more accurate but slower convergence.

### Hyperparameter Management

```python
# Get current hyperparameters
params = model.get_hyperparameters()
print(params)  # {'variance': 2.0, 'lengthscale': 0.5}

# Set new hyperparameters
model.set_hyperparameters(variance=1.5, lengthscale=0.3)
```

## Implementation Details

### Architecture
- **Spectral Representation**: Uses kernel spectral density for efficient computations
- **NUFFT Integration**: Leverages NUFFT for fast Fourier-based matrix-vector products
- **Variational Inference**: Implements natural gradient updates for Polya-Gamma parameters
- **Conjugate Gradients**: Unified CG tolerance for all linear system solves

### Key Components
1. **qVariationalParams**: Manages Polya-Gamma variational parameters (Δ)
2. **Spectral Setup**: Computes spectral points, weights, and gradients
3. **NUFFT Operators**: Efficient forward/adjoint transforms
4. **CG Solvers**: Unified tolerance for posterior mean and trace estimation
5. **Hutchinson Estimator**: Stochastic trace estimation for posterior variance

### Memory and Computational Efficiency
- Avoids explicit kernel matrix construction
- Uses batched operations where possible
- Employs efficient spectral representations
- Unified CG tolerance reduces redundant computations

## Dependencies

- PyTorch
- NUFFT implementation (`efgpnd`)
- Conjugate gradient solver (`cg`)
- Kernel utilities (`utils.kernels`)
- Kernel implementations (`kernels`)

## Testing

Run the test script to verify functionality:

```bash
python test_pg_gp.py
```

This will:
1. Generate synthetic 2D classification data
2. Fit the PG-GP model
3. Compute hyperparameter gradients
4. Display results and accuracy metrics
5. Create visualization plots (if matplotlib available)

## Notes

- The implementation is designed for binary classification problems
- All computations use float64 for numerical stability
- The unified `cg_tol` parameter controls accuracy vs. speed tradeoff
- CUDA support is automatic if available
- The class is fully self-contained and follows the same logic as the original notebook 