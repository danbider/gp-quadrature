import torch
from typing import Dict, Optional, Callable
from utils.kernels import get_xis
from cg import ConjugateGradients, BatchConjugateGradients
import time
from torch.profiler import profile, record_function, ProfilerActivity
import finufft
import pytorch_finufft.functional as pff
import numpy as np
import math

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
    #TODO multiple dimensions. 
    # v = torch.tensor(finufft.nufft1d1(
    #     x=(2 * torch.pi * h * x).numpy(), 
    #     c=np.ones_like(x,dtype=np.complex128),                
    #     n_modes=4 * m + 1 ,                      
    #     isign=-1,                       
    #     eps=1e-15,
    # ), dtype=torch.complex128)

    device      = x.device
    dtype_real  = x.dtype
    dtype_cmplx = torch.complex64 if dtype_real == torch.float32 else torch.complex128
    eps = 1e-15
    # non‑uniform points mapped to phase  φ = 2π h x
    phi = (2 * math.pi * h * x).to(dtype_real)                # (N,)

    mtot_big = 4 * m + 1                                      # length 4m+1
    OUT      = (mtot_big,)                                    # iterable shape!

    # constant weights c_j = 1 + 0i
    c = torch.ones_like(x, dtype=dtype_cmplx, device=device)

    # type‑1 NUFFT (adjoint): NU → U   in CMCL order (modeord=False)
    v = pff.finufft_type1(
            phi.unsqueeze(0), c, OUT,
            eps=eps, isign=-1, modeord=False)
    return v

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
    


class BatchFFTConv1d:
    def __init__(self, v, b):
        """
        v: Convolution kernel of shape (..., L_v). Can be batched.
        b: Signal to be convolved of shape (..., L_b). Can be batched.
        """
        self.v = v
        self.b = b
        # Use the last dimension for lengths (supports batched tensors)
        self.len_v = v.shape[-1]
        self.len_b = b.shape[-1]
        self.conv_length = self.len_v + self.len_b - 1  # Full convolution length.
        self.valid_length = self.len_v - self.len_b + 1
        self.start = self.len_b - 1
        self.end = self.start + self.valid_length

    def pad_to_length(self, x, target_length):
        pad_size = target_length - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad_size))

    def pad_signals(self):
        # Next power of two length for efficient FFT computation.
        fft_length = 1 << (self.conv_length - 1).bit_length()
        v_padded = self.pad_to_length(self.v, fft_length)
        b_padded = self.pad_to_length(self.b, fft_length)
        return v_padded, b_padded

    def fft_multiply_ifft(self, v_padded, b_padded):
        # Compute FFT along the last dimension.
        V = torch.fft.fft(v_padded, n=v_padded.shape[-1], dim=-1)
        B = torch.fft.fft(b_padded, n=b_padded.shape[-1], dim=-1)
        conv_fft = V * B
        conv_ifft = torch.fft.ifft(conv_fft, n=v_padded.shape[-1], dim=-1)
        return conv_ifft

    def extract_valid(self, conv_result_full):
        # Extract the valid convolution result along the last dimension.
        return conv_result_full[..., self.start:self.end]

    def __call__(self):
        v_padded, b_padded = self.pad_signals()
        conv_result_full = self.fft_multiply_ifft(v_padded, b_padded)
        return self.extract_valid(conv_result_full)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _cmplx(real_dtype: torch.dtype) -> torch.dtype:
    """Matching complex dtype for a given real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


class Toeplitz1D:
    """
    Fast linear convolution with a fixed CMCL‑ordered kernel.
    Caches the FFT once, works with any batch shape of the input.
    """
    def __init__(self, v: torch.Tensor, *, force_pow2: bool = True):
        if not v.is_complex():
            v = v.to(torch.complex128)

        self.L   = v.shape[-1]                       # 2n − 1
        self.n   = (self.L + 1) // 2                 # matrix size  n
        L_fft    = 1 << (self.L - 1).bit_length() if force_pow2 else self.L
        pad_tail = L_fft - self.L

        v_pad    = torch.nn.functional.pad(v, (0, pad_tail))
        self.fft_v = torch.fft.fft(v_pad, dim=-1)    # cached
        self.start = self.n - 1                      # ← use n, not L
        self.end   = self.start + self.n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.complex128)
        assert x.shape[-1] == self.n, \
            f"Input length {x.shape[-1]} ≠ kernel matrix size {self.n}"

        pad_tail = self.fft_v.shape[-1] - x.shape[-1]
        x_pad    = torch.nn.functional.pad(x, (0, pad_tail))

        y_fft = self.fft_v * torch.fft.fft(x_pad, dim=-1)
        y     = torch.fft.ifft(y_fft, dim=-1)[..., self.start:self.end]
        return y
    


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
        # print(f"Number of quadrature nodes: {mtot}")
        xis = xis.to(device=device)
        ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=torch.complex128, device=device) * h)


        # TODO redo with NUFFT

        
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
            v = compute_convolution_vector_vectorized(m=int((mtot - 1) / 2), x=x, h=h)


        ##### solve linear system (DF^*FD + sigma^2)beta = rhs with the conjugate gradients method #####
        
        # define the application of the operator A to a vector (more efficient than materializing A and multiplying)
        # where we multiply elementwise by ws instead of using the diagonal matrix D 
        Afun = lambda beta: D @ FFTConv1d(v, D @ beta)() + sigmasq * beta

        # Call conjugate gradients
        cg_object = ConjugateGradients(A_apply_function=Afun, b=right_hand_side, x0=torch.zeros_like(right_hand_side), early_stopping=opts.get('early_stopping', False))
        
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
    # print("\nProfiler Results Summary:")
    # print(prof.key_averages().table(
    #         sort_by="cuda_time_total", 
    #         row_limit=50
    #     ))
    # prof.export_chrome_trace("pytorch_profiler_trace.json")

    # returning just part of the args
    return beta, xis, ytrg



from typing import Dict, Optional, Callable



def efgp1d_NUFFT(x: torch.Tensor, y: torch.Tensor, sigmasq: float,
                 kernel: dict, eps: float, x_new: torch.Tensor,
                 do_profiling: bool = True, opts: Optional[dict] = None):

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if do_profiling else []
    with profile(activities=activities, record_shapes=True) as prof:

        # ---------- casts ---------------------------------------------------
        device = x.device
        dtype = torch.float64 # Use float64 as in the original example
        rdtype = torch.float64
        cdtype = _cmplx(rdtype)

        x = x.to(dtype=rdtype)
        y = y.to(dtype=rdtype)
        x_new = x_new.to(dtype=rdtype)

        # ---------- geometry -----------------------------------------------
        x0, x1 = torch.min(torch.cat([x, x_new])), torch.max(torch.cat([x, x_new]))
        L = x1 - x0
        N = x.shape[0]

        # ---------- quadrature ---------------------------------------------
        xis, h, mtot = get_xis(kernel, eps, L.item())
        xis = xis.to(device)
        ws  = torch.sqrt(kernel.spectral_density(xis).to(cdtype) * h)  # (M,)
        M   = mtot

        # ---------- NUFFT wrappers (CMCL order) ----------------------------
        OUT = (M,)
        nufft_eps = 1e-15

        def finufft1(phi, vals):
            return pff.finufft_type1(phi.unsqueeze(0), vals.to(cdtype), OUT,
                                     eps=nufft_eps, isign=-1, modeord=False)

        def finufft2(phi, fk):
            return pff.finufft_type2(phi.unsqueeze(0), fk.to(cdtype),
                                     eps=nufft_eps, isign=+1, modeord=False)

        phi = (2 * math.pi * h * (x - 0.0)).to(rdtype)
        fadj = lambda v: finufft1(phi, v)          # NU → U

        # ---------- RHS -----------------------------------------------------
        with record_function("right_hand_side"):
            right_hand_side = ws * fadj(y)

        # ---------- Toeplitz kernel & operator -----------------------------
        with record_function("convolution_vector"):
            v_kernel = compute_convolution_vector_vectorized(
                m=int((mtot - 1) / 2), x=x, h=h).to(cdtype)

        toeplitz = Toeplitz1D(v_kernel)

        def A_apply(beta):
            return ws * toeplitz(ws * beta) + sigmasq * beta

        # ---------- CG solve ------------------------------------------------
        cg_object = ConjugateGradients(A_apply, b=right_hand_side,
                                       x0=torch.zeros_like(right_hand_side),
                                       early_stopping=False)
        with record_function("solve"):
            beta = cg_object.solve()

        # ---------- posterior predictive mean ------------------------------
        with record_function("posterior_predictive_mean"):
            phi_new = (2 * math.pi * h * (x_new - 0.0)).to(rdtype)
            yhat = finufft2(phi_new, ws * beta).real

        ytrg = {'mean': yhat}

            ### Variance calculation


        # (F* F/sigmasq + I/ws**2)
        def C_apply(beta):
            return ws* toeplitz(ws*beta)/sigmasq +  beta
        
        
        
        m  = (mtot - 1) // 2
        k  = torch.cat((
                torch.arange(-m, 0,  device=device, dtype=dtype),  # -m,…,-1
                torch.arange( 0, m+1, device=device, dtype=dtype)   #  0,…, m
        ))              
        # form batches of f_x
        # def f_x_batch(x_batch: torch.Tensor, h: float) -> torch.Tensor:
        #     """
        #     x_batch : (B,)  real
        #     Returns  : (B, M) complex   with CMCL ordering
        #     """
        phase = 1j * 2 * math.pi * h * x_new[:, None] * k[None, :]   # (B,M)
        f_x = torch.exp(phase) 
        # Compute f_x for the batch of test points
        # f_x = f_x_batch(x_new, h)  # (B,M) complex
        
        # Initialize BatchConjugateGradients to solve C^{-1} f_x 
        batch_cg = BatchConjugateGradients(
            C_apply, 
            ws*f_x.conj(), 
            torch.zeros_like(f_x.conj()),
            tol=1e-6, 
            early_stopping=False
        )
        ### End Variance calculation
        
        # Solve the system for all points in the batch simultaneously
        C_inv_f_x = ws *batch_cg.solve()  # (B,M) complex
        
        s2 = torch.real((f_x * C_inv_f_x).sum(dim=1))    # (B,)
        ytrg['var'] = s2

        if opts is not None and opts.get('get_log_marginal_likelihood', False):
            ytrg['log_marginal_likelihood'] = None  # TODO

    # ---------- profiler summary -------------------------------------------
    # print("\nProfiler Results Summary:")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    return beta, xis, ytrg

def efgp1d_gradient_batched(
        x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
        *, nufft_eps=1e-15, device=None, dtype=torch.float64):
    """
    Gradient of the 1‑D GP log‑marginal likelihood estimated with
    Hutchinson trace + CG, completely torch native.
    """
    # 0)  Book‑keeping ------------------------------------------------------
    device  = device or x.device
    x       = x.to(device, dtype).flatten()
    y       = y.to(device, dtype).flatten()
    cmplx   = _cmplx(dtype)

    L       = x1 - x0
    xis, h, mtot = get_xis(kernel, eps, L)           # |xis| = M
    M       = xis.numel()
    ws      = torch.sqrt(kernel.spectral_density(xis) * h).to(cmplx)
    Dprime  = (h * kernel.spectral_grad(xis)).to(cmplx)  # (M, 3)

    # 1)  NUFFT adjoint / forward helpers (modeord=False) -------------------
    OUT = (mtot,)
    
    nufft_eps = 1e-15
    finufft1 = lambda pts, vals: pff.finufft_type1(
        pts.unsqueeze(0), vals.to(cmplx), OUT,
        eps=nufft_eps, isign=-1, modeord=False)

    finufft2 = lambda pts, fk: pff.finufft_type2(
        pts.unsqueeze(0), fk.to(cmplx),
        eps=nufft_eps, isign=+1, modeord=False)
    phi = (2 * math.pi * h * (x - 0.0)).to(dtype)    # real (N,)
    fadj = lambda v: finufft1(phi, v)    # NU → U
    fwd  = lambda fk: finufft2(phi, fk)  # U  → NU

    # 2)  Toeplitz operator T (cached FFT) ----------------------------------
    m          = (M - 1) // 2
    v_kernel   = compute_convolution_vector_vectorized(m, x, h).to(cmplx)
    toeplitz   = Toeplitz1D(v_kernel)               # cached once

    # 3)  Linear map A· = D F*F (D·) + σ² I -------------------------------
    def A_apply(beta):
        return ws * toeplitz(ws * beta) + sigmasq * beta

    # 4)  Solve A β = W F* y ---------------------------------------------
    rhs   = ws * fadj(y)
    beta  = ConjugateGradients(A_apply, rhs,
                               torch.zeros_like(rhs),
                               early_stopping=False).solve()
    alpha = (y - fwd(ws * beta)) / sigmasq

    # 5)  Term‑2  (α*D'α, α*α) --------------------------------------------
    fadj_alpha = fadj(alpha)
    term2 = torch.stack((
        torch.vdot(fadj_alpha, Dprime[:, 0] * fadj_alpha),
        torch.vdot(fadj_alpha, Dprime[:, 1] * fadj_alpha),
        torch.vdot(alpha,       alpha)
    ))

    # 6)  Monte‑Carlo trace (Term‑1) ---------------------------------------
    T  = trace_samples
    Z  = (2 * torch.randint(0, 2, (T, x.numel()), device=device,
                            dtype=dtype) - 1).to(cmplx)
    fadjZ = fadj(Z)

    B_blocks, R_blocks = [], []
    for i in range(3):
        rhs_i = fwd(Dprime[:, i] * fadjZ) if i < 2 else Z
        R_blocks.append(rhs_i)
        B_blocks.append(ws * fadj(rhs_i))

    B_all = torch.cat(B_blocks, 0)       # (3T, M)
    R_all = torch.cat(R_blocks, 0)       # (3T, N)

    def A_apply_batch(B):
        return ws * toeplitz(ws * B) + sigmasq * B

    Beta_all = BatchConjugateGradients(
        A_apply_batch, B_all, torch.zeros_like(B_all),
        tol=1e-6, early_stopping=False).solve()

    Alpha_all = (R_all - fwd(ws * Beta_all)) / sigmasq
    A_chunks  = Alpha_all.chunk(3, 0)

    term1 = torch.stack([(Z * a).sum(1).mean() for a in A_chunks])

    # 7)  Gradient ----------------------------------------------------------
    grad = 0.5 * (term1 - term2)
    return grad.real   

# torch.allclose(ytrg, ytrg_old)


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
    opts = {'early_stopping': False}
    from utils.kernels import get_xis
    if KERNEL_NAME == "squared_exponential":
        from kernels.squared_exponential import SquaredExponential
        kernel = SquaredExponential(dimension=1, lengthscale=0.1, variance=1.0)
    elif "matern" in KERNEL_NAME:
        from kernels.matern import Matern
        kernel = Matern(dimension=1, lengthscale=1.0, name=KERNEL_NAME)
    
    efgp1d(x, y, 1.0, kernel, EPSILON, x, opts=opts)
