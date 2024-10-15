import torch
from utils.kernels import GetTruncationBound
from kernels.squared_exponential import SquaredExponential

def test_get_truncation_bound():
    
    test_function = SquaredExponential(dimension=1, lengthscale=1.0, variance=1.0).kernel
    TEST_INPUT = 7.0
    # evaluate the function at some value and call it epsilon 
    epsilon = test_function(torch.tensor(TEST_INPUT))

    bisector = GetTruncationBound(eps=epsilon, kern=test_function)
    L = bisector.find_truncation_bound() # L is the input value where the function is epsilon

    assert abs(L - TEST_INPUT) < 1e-6
    assert abs(test_function(torch.tensor(L)) - epsilon) < 1e-6
