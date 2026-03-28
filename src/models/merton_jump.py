import numpy as np


class MertonJumpDiffusion:

    def __init__(self, S0, r, sigma, lambda_jump, mu_j, sigma_j):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def generate_paths(self, T, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0

        # drift correction pour que E[e^jump] = 1 sous Q
        jump_adj = self.lambda_jump * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)

        for i in range(1, n_steps + 1):
            n_jumps = rng.poisson(self.lambda_jump * dt, n_paths)

            # vectorize jump aggregation -- la boucle Python était trop lente
            max_j = int(n_jumps.max()) if n_jumps.max() > 0 else 0
            if max_j > 0:
                raw = rng.normal(self.mu_j, self.sigma_j, (n_paths, max_j))
                mask = np.arange(max_j) < n_jumps[:, None]
                jump = (raw * mask).sum(axis=1)
            else:
                jump = np.zeros(n_paths)

            z = rng.standard_normal(n_paths)
            S[:, i] = S[:, i-1] * np.exp(
                (self.r - 0.5 * self.sigma**2 - jump_adj) * dt
                + self.sigma * np.sqrt(dt) * z
                + jump
            )

        return S

    def price(self, K, T, option_type, n_paths=100000, n_steps=252, seed=None):
        S_paths = self.generate_paths(T, n_steps, n_paths, seed)
        S_T = S_paths[:, -1]

        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        price = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(n_paths)

        return price, std_error
