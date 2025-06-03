# spatial_decay.py
# Weighted spatial decay control module (M_beta functional)

import numpy as np

class SpatialDecay:
    def __init__(self, M_beta_0, C_beta, beta, dt):
        """
        Initialize spatial decay control.

        Parameters:
        - M_beta_0: initial weighted spatial decay functional
        - C_beta: coupling constant for decay growth
        - beta: decay exponent (typically beta=4)
        - dt: time step size
        """
        self.M_beta = M_beta_0
        self.C_beta = C_beta
        self.beta = beta
        self.dt = dt
        self.epsilon = 1e-8  # safeguard for t=0 division

    def update(self, t):
        """
        Update M_beta functional based on weighted decay relation:
        M_beta(t+Δt) = M_beta(t) + Δt * [ (C_beta / t^{3/4}) * M_beta(t) + const ]

        Parameters:
        - t: current time (nonzero safe)
        """
        safe_t = max(t, self.epsilon)
        growth = (self.C_beta / safe_t**(3/4)) * self.M_beta
        increment = self.dt * (growth + self._constant_term())
        self.M_beta += increment

    def _constant_term(self):
        """
        Placeholder for possible model-dependent constant source terms.
        """
        return 0.0  # Currently no constant source

    def get_current_M_beta(self):
        """
        Return current M_beta value.
        """
        return self.M_beta
