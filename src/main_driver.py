# main_driver.py
# Full integration driver for Navier-Stokes proof implementation

import numpy as np
from src.fft_solver import SpectralSolver
from src.concentration_control import ConcentrationControl
from src.energy_recursion import EnergyRecursion
from src.spatial_decay import SpatialDecay
from src.utils import initialize_energy_levels, compute_grad_norm_dict

# ==== Simulation Parameters ====

N = 64  # grid resolution (adjustable)
L = 2 * np.pi  # periodic box size
viscosity = 0.01
dt = 0.001
T_final = 1.0
max_order = 3  # maximum derivative order k

# Riccati parameter (requires careful calibration based on initial data)
K = 1.0
C_beta = 0.5
beta = 4

# ==== Initialize Spectral Solver ====
solver = SpectralSolver(N=N, L=L, viscosity=viscosity)

# ==== Generate synthetic initial data (example divergence-free u0) ====
def synthetic_initial_data(N, L):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u_x = np.sin(X) * np.cos(Y) * np.cos(Z)
    u_y = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u_z = np.zeros_like(u_x)
    return np.array([u_x, u_y, u_z])

u0_phys = synthetic_initial_data(N, L)
u0_hat = np.fft.fftn(u0_phys, axes=(1,2,3))

# Project initial data to divergence-free subspace
u0_hat_proj = solver.project_div_free(u0_hat)

# ==== Initialize Concentration Functional ====
grad_norm_samples = np.abs(np.gradient(u0_phys))  # rough gradient estimate
grad_norm_flat = grad_norm_samples.flatten()
C0_estimate = ConcentrationControl.estimate_initial_concentration(grad_norm_flat)
concentration_module = ConcentrationControl(C0=C0_estimate, K=K, dt=dt)

# ==== Initialize Energy Levels ====
Ek_init = initialize_energy_levels(u0_hat_proj, solver, max_order)
Ck_sharp_list = {k: EnergyRecursion.compute_Ck_sharp(k, viscosity) for k in range(max_order+1)}
energy_module = EnergyRecursion(Ek_init, Ck_sharp_list, viscosity, dt)

# ==== Initialize Spatial Decay Functional ====
M_beta_0 = 1.0  # initial spatial decay value
spatial_decay_module = SpatialDecay(M_beta_0, C_beta, beta, dt)

# ==== Main Simulation Loop ====

time = 0.0
while time < T_final:

    # Compute high-order gradient norms
    grad_norms = compute_grad_norm_dict(u0_hat_proj, solver, max_order)

    # Update concentration functional
    C_current = concentration_module.update()

    # Update high-order energy levels
    energy_module.update(C_current, grad_norms)

    # Update spatial decay
    spatial_decay_module.update(time)

    # Stability monitoring (example condition)
    if C_current > 1e6 or np.any([v > 1e10 for v in energy_module.Ek.values()]):
        print("Stability condition triggered. Terminating simulation.")
        break

    # Advance time
    time += dt

# ==== Output Results ====

print("Simulation complete.")
print(f"Final Concentration Functional: {concentration_module.get_current_value()}")
print(f"Final Energy Levels: {energy_module.get_current_Ek()}")
print(f"Final Spatial Decay M_beta: {spatial_decay_module.get_current_M_beta()}")
