import torch
from typing import Dict, Optional, Callable
from utils.kernels import get_xis
from cg import ConjugateGradients
import time
from torch.profiler import profile, record_function, ProfilerActivity

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
    j_indices = torch.arange(-2*m, 2*m + 1).to(dtype=torch.complex128, device=x.device) # j = -2m, ..., 2m
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

def efgp1d(x: torch.Tensor, y: torch.Tensor, sigmasq: float, kernel: Dict[str, Callable], eps: float, x_new: torch.Tensor, do_profiling: bool = True, opts: Optional[Dict] = None):
    """
    Fast equispaced Fourier GP regression in 1D.
    Difference from efgp1d_dense is that this version does not use the dense solve, but rather a convolution + CG.
    """
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if do_profiling else []
    with profile(activities=activities, record_shapes=True) as prof:    

        # Send x, y, x_new to double precision, real dtype
        device = x.device
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        x_new = x_new.to(dtype=torch.float64)

        # Get problem geometry
        x0, x1 = torch.min(torch.cat([x, x_new])), torch.max(torch.cat([x, x_new]))
        L = x1 - x0
        N = x.shape[0]  

        # Get Fourier frequencies and weights
        xis, h, mtot = get_xis(kernel, eps, L.item())
        print(f"Number of quadrature nodes: {mtot}")
        xis = xis.to(device=device)
        ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=torch.complex128, device=device) * h)
        D = torch.diag(ws)

        ##### compute the right hand side #####
        # Compute the right hand side (expensive, materializing F which is M X N)
        # paper does it via NUFFT
        with record_function("right_hand_side"):
            F = torch.exp(1j * 2 * torch.pi * torch.outer(x, xis)).to(dtype=torch.complex128, device=device)
            right_hand_side = D @ (F.adjoint() @ y.to(dtype=torch.complex128))
        

        ##### construct the convolution vector v #####
        # In the original work, this was done using NUFFT, but here we do it manually
        # we can also do it via taking the first row and column of F^* F (see discrete_convolution_tests.ipynb)
        with record_function("convolution_vector"):
            v, j_indices = compute_convolution_vector_vectorized(m=int((mtot - 1) / 2), x=x, h=h)


        ##### solve linear system (DF^*FD + sigma^2)beta = rhs with the conjugate gradients method #####
        
        # define the application of the operator A to a vector (more efficient than materializing A and multiplying)
        # where we multiply elementwise by ws instead of using the diagonal matrix D 
        Afun = lambda beta: D @ FFTConv1d(v, D @ beta)() + sigmasq * beta

        # Call conjugate gradients
        cg_object = ConjugateGradients(A_apply_function=Afun, b=right_hand_side, x0=torch.zeros_like(right_hand_side))
        
        with record_function("solve"):
            beta = cg_object.solve() # beta is the solution to the linear system


        # Evaluate the posterior mean at the new points
        # The paper also uses nuFFT here, but we do it manually for now
        with record_function("posterior_predictive_mean"):
            Phi_target = torch.exp(1j * 2 * torch.pi * torch.outer(x_new, xis)) @ D
            yhat = Phi_target @ beta

        ytrg = {'mean': torch.real(yhat)}

        # Optionally compute posterior variance
        if opts is not None and opts.get('get_var', False):
            # TODO: implement this later
            ytrg['var'] = None
        
        # # Optionally compute log marginal likelihood at training locations
        if opts is not None and opts.get('get_log_marginal_likelihood', False):
            with record_function("log_marginal_likelihood"):
                logdet = N*torch.log(sigmasq) + torch.logdet((D @ F.adjoint() @ F @ D)/sigmasq + torch.eye(mtot, dtype=torch.float64, device=device)).to(dtype=torch.float64)
                alpha = (1/sigmasq) * (y - torch.real(F @ D @ beta))
                log_marg_lik = -0.5 * y.T @ alpha - 0.5 * logdet - 0.5 * N * torch.log(2 * torch.tensor(torch.pi, dtype=torch.float64, device=device))
                ytrg['log_marginal_likelihood'] = log_marg_lik # TODO: this shouldn't be in ytrg, fix later
        
    # Print profiler results
    print("\nProfiler Results Summary:")
    print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=50
        ))
    prof.export_chrome_trace("pytorch_profiler_trace.json")

    # returning just part of the args
    return beta, xis, ytrg


if __name__ == "__main__":
    KERNEL_NAME = "matern32"
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device!")
    # generate data 
    EPSILON = 1e-6
    N = 1000000
    freq = 0.5    
    x = torch.linspace(0, 5, N, device=device)
    y = torch.sin(2 * torch.pi * freq * x) + torch.randn(N, device=device) * 0.1
    from utils.kernels import get_xis
    if KERNEL_NAME == "squared_exponential":
        from kernels.squared_exponential import SquaredExponential
        kernel = SquaredExponential(dimension=1, lengthscale=0.1, variance=1.0)
    elif "matern" in KERNEL_NAME:
        from kernels.matern import Matern
        kernel = Matern(dimension=1, lengthscale=1.0, name=KERNEL_NAME)
    
    efgp1d(x, y, 1.0, kernel, EPSILON, x, opts=None)
