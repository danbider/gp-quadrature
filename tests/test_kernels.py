"""
Test the kernels.
"""

import pytest
import torch
import math


def test_matern():
    """
    Test the Matern kernel.
    (1) test that the kernel at zero distance is the variance
    (2) test that the kernel at a non-zero distance matches the manual calculation
    """
    from kernels.matern import Matern
    matern = Matern(dimension=1, name='matern12', lengthscale=1.0, variance=1.0)
    assert matern.nu == 0.5
    distance = torch.tensor([0.0])
    assert matern.kernel(distance=distance) == 1.0

def test_matern_multidim():
    """
    Test the Matern kernel with multi-dimensional inputs.
    """
    from kernels.matern import Matern
    
    # Create a Matern kernel with dimension=2
    matern = Matern(dimension=2, name='matern32', lengthscale=1.5, variance=2.0)
    
    # Test kernel_matrix with 2D inputs
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.5, 1.0], [2.0, 3.0]])
    
    K = matern.kernel_matrix(x, y)
    
    # Check shape
    assert K.shape == (2, 2)
    
    # Check diagonal values (should be less than variance since points aren't identical)
    assert K[0, 0] < matern.variance
    assert K[1, 1] < matern.variance
    
    # Test with 1D input that should be treated as multiple 1D points
    x1d = torch.tensor([1.0, 2.0])
    y1d = torch.tensor([0.5, 1.0, 1.5])
    
    K1d = matern.kernel_matrix(x1d, y1d)
    
    # Check shape - should be (2, 3) for 2 x-points and 3 y-points
    assert K1d.shape == (2, 3)

def test_matern_spectral_grad():
    """
    Test the Matern kernel's spectral gradient.
    """
    from kernels.matern import Matern
    
    # Create Matern kernels with different smoothness
    matern12 = Matern(dimension=1, name='matern12', lengthscale=1.2, variance=1.5)
    matern52 = Matern(dimension=2, name='matern52', lengthscale=0.8, variance=2.0)
    
    # Test frequencies
    freqs1d = torch.tensor([0.0, 0.5, 1.0])
    freqs2d = torch.tensor([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])
    
    # Get spectral densities and gradients
    s1 = matern12.spectral_density(freqs1d)
    g1 = matern12.spectral_grad(freqs1d)
    
    s2 = matern52.spectral_density(freqs2d)
    g2 = matern52.spectral_grad(freqs2d)
    
    # Check shapes
    assert s1.shape == (3,)
    assert g1.shape == (3, 2)
    assert s2.shape == (3,)
    assert g2.shape == (3, 2)
    
    # Check gradient with respect to variance
    # Should equal spectral density / variance
    assert torch.allclose(g1[:, 1], s1 / matern12.variance)
    assert torch.allclose(g2[:, 1], s2 / matern52.variance)
    
    # Simple finite difference test for lengthscale gradient
    delta = 1e-5
    matern12_plus = Matern(dimension=1, name='matern12', lengthscale=matern12.lengthscale + delta, variance=matern12.variance)
    s1_plus = matern12_plus.spectral_density(freqs1d)
    fd_grad = (s1_plus - s1) / delta
    
    # Check approximate agreement with analytical gradient (relative tolerance is higher due to finite differences)
    assert torch.allclose(g1[:, 0], fd_grad, rtol=1e-2)

def test_matern_log_marginal():
    """
    Test the Matern kernel's log marginal likelihood.
    """
    from kernels.matern import Matern
    
    # Create a Matern kernel
    matern = Matern(dimension=1, name='matern32', lengthscale=1.0, variance=1.0)
    
    # Create some toy data
    x = torch.linspace(0, 5, 10).reshape(-1, 1)
    y = torch.sin(x.squeeze())
    
    # Compute log marginal likelihood
    sigmasq = 0.1
    log_marg = matern.log_marginal(x, y, sigmasq)
    
    # Check that it's a scalar
    assert isinstance(log_marg, torch.Tensor) and log_marg.numel() == 1
    
    # Check that changing hyperparameters changes the log marginal likelihood
    matern2 = Matern(dimension=1, name='matern32', lengthscale=0.5, variance=1.0)
    log_marg2 = matern2.log_marginal(x, y, sigmasq)
    
    assert log_marg != log_marg2

def test_hyper_methods():
    """
    Test the hyperparameter getter, setter, and iterator methods.
    """
    from kernels.matern import Matern
    
    matern = Matern(dimension=1, name='matern52', lengthscale=1.2, variance=1.5)
    
    # Test get_hyper
    assert matern.get_hyper('lengthscale') == 1.2
    assert matern.get_hyper('variance') == 1.5
    
    # Test set_hyper
    matern.set_hyper('lengthscale', 0.8)
    assert matern.lengthscale == 0.8
    
    # Test iter_hypers
    hypers = dict(matern.iter_hypers())
    assert 'lengthscale' in hypers
    assert 'variance' in hypers
    assert hypers['lengthscale'] == 0.8
    assert hypers['variance'] == 1.5

# test that an invalid name raises a validation error
with pytest.raises(ValueError):
    from kernels.matern import Matern
    Matern(dimension=1, name='matern92', lengthscale=1.0, variance=1.0)

def test_squared_exponential():
    """
    Test the SquaredExponential kernel.
    (1) test that the kernel at zero distance is the variance
    (2) test that the kernel at a non-zero distance matches the manual calculation
    """
    from kernels.squared_exponential import SquaredExponential
    _VAR = 1.12
    _LENGTHSCALE = 0.45 
    _DISTANCE = torch.tensor([0.7071])
    squared_exponential = SquaredExponential(dimension=1, lengthscale=_LENGTHSCALE, variance=_VAR)
    assert squared_exponential.kernel(distance=torch.tensor([0.0])) == _VAR
    manual_kernel = _VAR * torch.exp(-0.5 * (_DISTANCE/_LENGTHSCALE)**2)
    assert torch.allclose(squared_exponential.kernel(distance=_DISTANCE), manual_kernel)


