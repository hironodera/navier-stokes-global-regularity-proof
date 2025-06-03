# utils.py
# Auxiliary utilities for Navier-Stokes proof implementation

import numpy as np

def compute_grad_norm(u_hat, solver, order):
    """
    Compute ||∇^order u||_L^2^2 norm using spectral derivatives.

    Parameters:
    - u_hat: Fourier-transformed velocity field (shape: (3, N, N, N))
    - solver: instance of SpectralSolver (provides wave numbers)
    - order: order of spatial derivative

    Returns:
    - L2 norm squared of ∇^order u
    """
    deriv_hat = solver.compute_derivative(u_hat, order)
    deriv_phys = np.fft.ifftn(deriv_hat, axes=(1,2,3)).real
    norm_squared = np.sum(deriv_phys**2)
    return norm_squared

def initialize_energy_levels(u_hat, solver, max_order):
    """
    Compute initial high-order energy levels E_k(0) for k = 0 to max_order.

    Parameters:
    - u_hat: Fourier-transformed velocity field
    - solver: SpectralSolver instance
    - max_order: maximum derivative order k

    Returns:
    - Dictionary {k: E_k(0)}
    """
    Ek_init = {}
    for k in range(max_order + 1):
        norm_squared = compute_grad_norm(u_hat, solver, k)
        Ek_init[k] = 0.5 * norm_squared
    return Ek_init

def compute_grad_norm_dict(u_hat, solver, max_order):
    """
    Compute dictionary of ||∇^{k+1} u||_L^2^2 for k+1 up to max_order+1.

    Parameters:
    - u_hat: Fourier-transformed velocity field
    - solver: SpectralSolver instance
    - max_order: maximum derivative order k

    Returns:
    - Dictionary {k+1: norm}
    """
    grad_norms = {}
    for k in range(max_order + 1):
        norm_squared = compute_grad_norm(u_hat, solver, k+1)
        grad_norms[k+1] = norm_squared
    return grad_norms
