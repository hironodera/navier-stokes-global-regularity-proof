# energy_recursion.py
# High-order energy recursion module

import numpy as np

class EnergyRecursion:
    def __init__(self, Ek_init, Ck_sharp_list, viscosity, dt):
        """
        Initialize high-order energy levels.

        Parameters:
        - Ek_init: dict {k: E_k(0)} for all orders k
        - Ck_sharp_list: dict {k: Ck_sharp value} for all orders k
        - viscosity: ν (positive constant)
        - dt: time step size
        """
        self.Ek = Ek_init.copy()
        self.Ck_sharp = Ck_sharp_list
        self.nu = viscosity
        self.dt = dt

    def update(self, C_current, grad_norms):
        """
        Update each high-order energy level based on the recurrence relation:

        E_k(t+Δt) = E_k(t) + Δt * [ 
            - (ν/2) * ||∇^{k+1} u||_L^2^2 
            + (Ck_sharp/ν) * C(u(t)) * E_k(t)
        ]

        Parameters:
        - C_current: current concentration functional value
        - grad_norms: dict {k+1: ||∇^{k+1} u||_L^2^2 } for all k+1
        """
        for k in self.Ek.keys():
            damping = - (self.nu / 2) * grad_norms.get(k+1, 0)
            coupling = (self.Ck_sharp[k] / self.nu) * C_current * self.Ek[k]
            increment = self.dt * (damping + coupling)
            self.Ek[k] += increment

    def get_current_Ek(self):
        """
        Return current dictionary of energy levels.
        """
        return self.Ek

    @staticmethod
    def compute_Ck_sharp(k, viscosity):
        """
        Compute theoretical Ck_sharp constant for order k.

        Formula:
        Ck_sharp = (k / 4) * (2 ** k) / ν + 2
        """
        return (k / 4) * (2 ** k) / viscosity + 2
