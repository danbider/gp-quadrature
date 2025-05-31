import torch
import math
from utils.kernels import get_xis
from efgpnd import _cmplx
## These are just for testing and sanity checking, not used in EFGNPD 
def compute_gradients_truncated(x, y, sigmasq, kernel, EPSILON):
    """
    Gradients with approximated kernel, exact trace-- forming the matrices directly to sanity check.
    """
    # sigmasq = torch.tensor(0.1, dtype=torch.float64)  # noise variance
    # kernel = SquaredExponential(dimension=1, lengthscale=0.1, variance=1.0)

    # Flatten data to 1D.
    # if x is 1d unsqueeze
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    # x = x.to(dtype=torch.float64).flatten()   # shape: (N,)
    # y = y.to(dtype=torch.float64).flatten()     # shape: (N,)
    # x_new = torch.linspace(0, 5, 1000, dtype=torch.float64)
    d = x.shape[1]
    cdtype = _cmplx(x.dtype)
    x0 = x.min(dim=0).values  
    x1 = x.max(dim=0).values  

    domain_lengths = x1 - x0
    L = domain_lengths.max()
    N = x.shape[0]
    # print(EPSILON)
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=EPSILON, L=L, use_integral=True, l2scaled=False)
    # print(xis_1d.shape)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij') # makes tensor product Jm 
    xis = torch.stack(grids, dim=-1).view(-1, d) 
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d) # (mtot**d,1)

    D = torch.diag(ws).to(dtype=cdtype)         # D: (M, M)

    # Form design features F (N x M): F[n,m] = exp(2pi i * xis[m] * x[n])
    # F = torch.exp(1j * 2 * math.pi * torch.outer(x, xis)).to(dtype=torch.complex128)
    F = torch.exp(1j * 2 * torch.pi * (x @ xis.T))

    # Compute approximate kernel: K = F * D^2 * F^*.
    D2 = D @ D  # This is just diag(ws^2)
    K = F @ D2 @ F.conj().transpose(-2, -1)  # shape: (N, N)
    C = K + sigmasq * torch.eye(N, dtype=cdtype)  # add noise term

    # Directly invert C and compute alpha.
    C_inv = torch.linalg.inv(C)
    alpha = C_inv @ y.to(dtype=cdtype)  # shape: (N,)

    # Compute derivative of the kernel with respect to the kernel hyperparameters.
    # Let spectral_grad = kernel.spectral_grad(xis), shape: (M, n_params)
    spectral_grad = kernel.spectral_grad(xis)  # shape: (M, n_params)
    # Then dK/dtheta for each kernel hyperparameter i is approximated as:
    # dK/dtheta_i = F * diag( h * spectral_grad(:, i) ) * F^*
    dK_dtheta_list = []
    n_params = spectral_grad.shape[1]
    for i in range(n_params):
        dK_i = F @ torch.diag((h**d * spectral_grad[:, i]).to(dtype=cdtype)) @ F.conj().transpose(-2, -1)
        dK_dtheta_list.append(dK_i)
    # The derivative with respect to the noise parameter is simply the identity.
    dK_dtheta_list.append(torch.eye(N, dtype=cdtype))
    n_total = n_params + 1

    # Compute gradient for each hyperparameter using:
    # grad = 0.5 * [trace(C_inv * dK/dtheta) - alpha^H * (dK/dtheta) * alpha]
    grad = torch.zeros(n_total, dtype=cdtype)
    for i in range(n_total):
        if i < n_params:
            term1 = torch.trace(C_inv @ dK_dtheta_list[i])
            term2 = (alpha.conj().unsqueeze(0) @ (dK_dtheta_list[i] @ alpha.unsqueeze(1))).squeeze()
        else:  # noise derivative: dC/d(sigmasq) = I
            term1 = torch.trace(C_inv)
            term2 = (alpha.conj().unsqueeze(0) @ alpha.unsqueeze(1)).squeeze()
        grad[i] = 0.5 * (term1 - term2)
        # print('term1:' ,term1.real)
        # print('term2:', term2.real) 

    # Print the gradients (real parts)
    # print("(Truncated) Direct inversion gradient:")
    # print(f"  dNLL/d(lengthscale) = {grad[0].real.item():.6f}")
    # if n_params > 1:
    #     print(f"  dNLL/d(variance)    = {grad[1].real.item():.6f}")
    # print(f"  dNLL/d(noise)       = {grad[-1].real.item():.6f}")
    true_grad = grad.clone()
    # print("term 1: ", term1.real, "term 2: ", term2.real)
    return true_grad.real

# 3. Define the squared-exponential kernel.
def squared_exponential_kernel(x1, x2, lengthscale, variance):
    # Ensure inputs are 2D
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(1)
    # Compute pairwise squared Euclidean distances.
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)   # shape: (n1, n2, d)
    dist_sq = (diff ** 2).sum(dim=2)             # shape: (n1, n2)
    K = variance * torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
    return K

def matern_kernel(x1, x2, lengthscale, variance, nu=1.5):
    # Ensure inputs are 2D
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(1)
    
    # Compute pairwise distances
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)   # shape: (n1, n2, d)
    dist = torch.sqrt((diff ** 2).sum(dim=2))   # shape: (n1, n2)
    scaled_dist = math.sqrt(2 * nu) * dist / lengthscale
    
    if nu == 1.5:
        # Matern 3/2
        poly_term = 1.0 + scaled_dist
        K = variance * poly_term * torch.exp(-scaled_dist)
    elif nu == 2.5:
        # Matern 5/2
        poly_term = 1.0 + scaled_dist + (scaled_dist ** 2) / 3.0
        K = variance * poly_term * torch.exp(-scaled_dist)
    else:
        raise ValueError("Only nu=1.5 (Matern 3/2) and nu=2.5 (Matern 5/2) are implemented")
    
    return K

# -------------------------
# 4. Define the negative log marginal likelihood (NLL)
def negative_log_marginal_likelihood(x, y, lengthscale, variance, noise, kernel_type):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n = x.shape[0]
    # Compute kernel matrix K(X,X) and add noise on the diagonal.
    if kernel_type == "squared_exponential":
        K = squared_exponential_kernel(x, x, lengthscale, variance) + noise * torch.eye(n, dtype=x.dtype)
    elif kernel_type == "matern":
        K = matern_kernel(x, x, lengthscale, variance) + noise * torch.eye(n, dtype=x.dtype)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    # Compute Cholesky factorization of K.
    L = torch.linalg.cholesky(K)
    # Solve for alpha = K^{-1} y using the Cholesky factors.
    alpha = torch.cholesky_solve(y, L)
    # Compute the log determinant of K via its Cholesky factor.
    logdetK = 2 * torch.sum(torch.log(torch.diag(L)))
    # NLL = 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5*n*log(2π)
    nll = 0.5 * torch.matmul(y.T, alpha) + 0.5 * logdetK + 0.5 * n * math.log(2 * math.pi)
    return nll.squeeze()  # return a scalar
def compute_gradients_vanilla(x, y, sigmasq, kernel,kernel_type):
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)


    # -------------------------
    # 2. Define hyperparameters as torch tensors with gradients.
    lengthscale = torch.tensor(kernel.lengthscale, dtype=x.dtype, requires_grad=True)
    variance    = torch.tensor(kernel.variance, dtype=x.dtype, requires_grad=True)
    noise       = sigmasq.clone().detach().requires_grad_(True)

    # -------------------------


    # -------------------------
    # 5. Compute the NLL and its gradients.
    nll = negative_log_marginal_likelihood(x, y, lengthscale, variance, noise,kernel_type)
    # print("Negative log marginal likelihood:", nll.item())

    nll.backward()

    # print("\n (VANILLA) Gradients of the negative log marginal likelihood:")
    # print("  dNLL/d(lengthscale) =", lengthscale.grad.item())
    # print("  dNLL/d(variance)    =", variance.grad.item())
    # print("  dNLL/d(noise)       =", noise.grad.item())
    grad = torch.tensor([lengthscale.grad.item(), variance.grad.item(), noise.grad.item()])

    return grad.to(dtype=x.dtype)