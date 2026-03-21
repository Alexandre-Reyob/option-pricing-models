import numpy as np
from .heston import Heston

class Bates(Heston):
    
    def __init__(self, S0, v0, kappa, theta, sigma_v, rho, lambda_jump, mu_j, sigma_j):
        super().__init__(S0, v0, kappa, theta, sigma_v, rho)
        self.lambda_jump = lambda_jump
        self.mu_j = mu_j
        self.sigma_j = sigma_j
    
    def generate_paths(self, T, n_steps, n_paths, r, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        jump_adj = self.lambda_jump * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)
        
        for i in range(1, n_steps + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            v_prev = np.maximum(v[:, i-1], 0)
            v[:, i] = v[:, i-1] + self.kappa * (self.theta - v_prev) * dt + self.sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * w2
            v[:, i] = np.maximum(v[:, i], 0)
            
            n_jumps = np.random.poisson(self.lambda_jump * dt, n_paths)
            jump = np.zeros(n_paths)
            for j in range(n_paths):
                if n_jumps[j] > 0:
                    jumps = np.random.normal(self.mu_j, self.sigma_j, n_jumps[j])
                    jump[j] = np.sum(jumps)
            
            S[:, i] = S[:, i-1] * np.exp(
                (r - 0.5 * v_prev - jump_adj) * dt
                + np.sqrt(v_prev) * np.sqrt(dt) * w1
                + jump
            )
        
        return S, v
