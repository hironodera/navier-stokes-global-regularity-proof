# fft_solver.py
# Fourier-based spectral solver for incompressible Navier-Stokes equations

import numpy as np

class SpectralSolver:
    def __init__(self, N, L, viscosity):
        self.N = N                # number of grid points
        self.L = L                # domain size
        self.viscosity = viscosity
        self.dx = L / N
        self.k = self._wave_numbers()
        self.ksqr = np.sum(self.k**2, axis=0)
        self.ksqr[0,0,0] = 1  # avoid division by zero at k=0

    def _wave_numbers(self):
        k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        return np.array([kx, ky, kz])

    def project_div_free(self, u_hat):
        k_dot_u = np.sum(self.k * u_hat, axis=0)
        proj = u_hat - (self.k * k_dot_u) / self.ksqr
        return proj

    def compute_derivative(self, u_hat, order):
        deriv = (1j * self.k) ** order * u_hat
        return deriv

    def curl(self, u_hat):
        ux, uy, uz = u_hat
        curl_x = 1j * (self.k[1]*uz - self.k[2]*uy)
        curl_y = 1j * (self.k[2]*ux - self.k[0]*uz)
        curl_z = 1j * (self.k[0]*uy - self.k[1]*ux)
        return np.array([curl_x, curl_y, curl_z])
