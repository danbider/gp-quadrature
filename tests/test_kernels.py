"""
Test the kernels.
"""

import pytest
import torch


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


