# verification_driver.py
# Full verification run with stabilized parameters

import numpy as np
from fft_solver import SpectralSolver
from concentration_control import ConcentrationControl
from energy_recursion import EnergyRecursion
from spatial_decay import SpatialDecay
from utils import initialize_energy_levels, compute_grad_norm_dict

# ==== Verification Parameters ====

N = 32           # Moderate grid for verification (safe for initial testing)
L = 2 * np.pi
viscosity = 0.05
dt = 0.0005
T_final = 0.2
max_order = 2    # Keep low for initial stability test

# Riccati parameter
K = 0.5
C_beta = 0.3
beta = 4

# ==== Initialize Spectral Solver ====

solver = SpectralSolver(N=N, L=L, viscosity=viscosity)

# ==== Synthetic Initial Data (Smooth & Safe Divergence-Free Field) ====

def smooth_initial_data(N, L):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u_x = np.sin(X) * np.cos(Y) * np.cos(Z)
    u_y = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u_z = np.zeros_like(u_x)
    return np.array([u_x, u_y, u_z])

u0_phys = smooth_initial_data(N, L)
u0_hat = np.fft.fftn(u0_phys, axes=(1,2,3))
u0_hat_proj = solver.project_div_free(u0_hat)

# ==== Initialize Modules ====

grad_norm_samples = np.abs(np.gradient(u0_phys)).flatten()
C0_estimate = ConcentrationControl.estimate_initial_concentration(grad_norm_samples)
concentration_module = ConcentrationControl(C0=C0_estimate, K=K, dt=dt)

Ek_init = initialize_energy_levels(u0_hat_proj, solver, max_order)
Ck_sharp_list = {k: EnergyRecursion.compute_Ck_sharp(k, viscosity) for k in range(max_order+1)}
energy_module = EnergyRecursion(Ek_init, Ck_sharp_list, viscosity, dt)

M_beta_0 = 1.0
spatial_decay_module = SpatialDecay(M_beta_0, C_beta, beta, dt)

# ==== Main Execution Loop ====

time = 0.0
step = 0
log_interval = 20

while time < T_final:
    grad_norms = compute_grad_norm_dict(u0_hat_proj, solver, max_order)
    C_current = concentration_module.update()
    energy_module.update(C_current, grad_norms)
    spatial_decay_module.update(time)

    # Simple stability threshold
    if C_current > 1e5 or np.any([v > 1e7 for v in energy_module.Ek.values()]):
        print("Stability condition triggered. Terminating at step", step)
        break

    if step % log_interval == 0:
        print(f"Step {step}, t={time:.4f}")
        print(f"  C(u): {C_current:.6f}")
        print(f"  Energy: {energy_module.get_current_Ek()}")
        print(f"  M_beta: {spatial_decay_module.get_current_M_beta():.6f}")

    time += dt
    step += 1

print("Verification run complete.")
print(f"Final C(u): {concentration_module.get_current_value()}")
print(f"Final Energies: {energy_module.get_current_Ek()}")
print(f"Final M_beta: {spatial_decay_module.get_current_M_beta()}")
