import torch
from typing import Dict, Optional, Callable
from utils.kernels import get_xis
from cg import ConjugateGradients
import time

def compute_convolution_vector_vectorized(m: int, x: torch.Tensor, h: float) -> torch.Tensor:
    """
    Vectorized implementation of compute_convolution_vector
    Args:
        m (int): Half number of nodes (mtot = 2m + 1)
        x (torch.Tensor): Input points
        h (float): Grid spacing
    Returns:
        torch.Tensor: Convolution vector of shape (4m+1,)
    """
    j_indices = torch.arange(-2*m, 2*m + 1).to(dtype=torch.complex128) # j = -2m, ..., 2m
    exponents = 2 * torch.pi * 1j * h * torch.outer(j_indices, x).to(dtype=torch.complex128)
    v = torch.sum(torch.exp(exponents), dim=1).conj()
    # TODO: the final conjugation was necessary to make everything match, but I am not sure it matches the equation
    return v, j_indices

class FFTConv1d:
    def __init__(self, v, b):
        self.v = v
        self.b = b
        self.len_v = len(v)
        self.len_b = len(b)
        self.conv_length = self.len_v + self.len_b - 1  # Full convolution length
        self.valid_length = self.len_v - self.len_b + 1  # = 4m+1 - (2m+1) + 1 = 2m + 1
        self.start = self.len_b - 1  # = 2m + 1 - 1 = 2m
        self.end = self.start + self.valid_length  # = 2m + 2m + 1 = 4m + 1

    def pad_right(self):
        # Find next power of two for efficient FFT computation
        fft_length = 1 << (self.conv_length - 1).bit_length()
        # Pad signals to fft_length (padding with zeros at the right)
        v_padded = torch.nn.functional.pad(self.v, (0, fft_length - self.len_v))
        b_padded = torch.nn.functional.pad(self.b, (0, fft_length - self.len_b))
        return v_padded, b_padded

    def fft_multiply_ifft(self, v_padded, b_padded):
        # Compute FFTs of padded signals
        V = torch.fft.fft(v_padded)
        B = torch.fft.fft(b_padded)
        # Multiply in the frequency domain
        conv_fft = V * B
        # Compute inverse FFT to get convolution result
        return torch.fft.ifft(conv_fft)

    def extract_valid(self, conv_result_full):
        # Extract the valid part of the convolution result
        return conv_result_full[self.start:self.end]  # = conv_result_full[2m:4m+1]

    def __call__(self):
        v_padded, b_padded = self.pad_right()
        conv_result_full = self.fft_multiply_ifft(v_padded, b_padded)
        return self.extract_valid(conv_result_full)

def efgp1d(x: torch.Tensor, y: torch.Tensor, sigmasq: float, kernel: Dict[str, Callable], eps: float, x_new: torch.Tensor, opts: Optional[Dict] = None):
    """
    Fast equispaced Fourier GP regression in 1D.
    Difference from efgp1d_dense is that this version does not use the dense solve, but rather a convolution + CG.
    """

    # Send x, y, x_new to double precision, real dtype
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x_new = x_new.to(dtype=torch.float64)

    # Get problem geometry
    x0, x1 = torch.min(torch.cat([x, x_new])), torch.max(torch.cat([x, x_new]))
    L = x1 - x0
    N = x.shape[0]

    # Get Fourier frequencies and weights
    xis, h, mtot = get_xis(kernel, eps, L)  # assume this exists and returns tensors
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=torch.complex128) * h)
    D = torch.diag(ws)

    ##### compute the right hand side #####
    # Compute the right hand side (expensive, materializing F which is M X N)
    # paper does it via NUFFT
    start_time = time.time()
    F = torch.exp(1j * 2 * torch.pi * torch.outer(x, xis)).to(dtype=torch.complex128)
    right_hand_side = D @ (F.adjoint() @ y.to(dtype=torch.complex128))
    rhs_time = time.time() - start_time

    ##### construct the convolution vector v #####
    # In the original work, this was done using NUFFT, but here we do it manually
    # we can also do it via taking the first row and column of F^* F (see discrete_convolution_tests.ipynb)
    start_time = time.time()
    v, j_indices = compute_convolution_vector_vectorized(m=int((mtot - 1) / 2), x=x, h=h)
    conv_vector_time = time.time() - start_time

    ##### solve linear system (DF^*FD + sigma^2)beta = rhs with the conjugate gradients method #####
    
    # define the application of the operator A to a vector (more efficient than materializing A and multiplying)
    # where we multiply elementwise by ws instead of using the diagonal matrix D 
    Afun = lambda beta: D @ FFTConv1d(v, D @ beta)() + sigmasq * beta

    # Call conjugate gradients
    cg_object = ConjugateGradients(A_apply_function=Afun, b=right_hand_side, x0=torch.zeros_like(right_hand_side))
    
    start_time = time.time()
    beta = cg_object.solve() # beta is the solution to the linear system
    solve_time = time.time() - start_time

    # Evaluate the posterior mean at the new points
    # The paper also uses nuFFT here, but we do it manually for now
    Phi_target = torch.exp(1j * 2 * torch.pi * torch.outer(x_new, xis)) @ D
    yhat = Phi_target @ beta

    ytrg = {'mean': torch.real(yhat)}
    
    timing_results = {
        'rhs_time': rhs_time,
        'construct_system_time': conv_vector_time,
        'solve_time': solve_time
    }

    # Optionally compute posterior variance
    if opts is not None and opts.get('get_var', False):
        # TODO: implement this later
        ytrg['var'] = None

    # returning just part of the args
    return beta, xis, ytrg, timing_results



