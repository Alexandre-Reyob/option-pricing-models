import numpy as np
from scipy.optimize import minimize
from ..engine.monte_carlo import generate_gbm_paths

class Heston:
    
    def __init__(self, S0, v0, kappa, theta, sigma_v, rho):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
    
    def generate_paths(self, T, n_steps, n_paths, r, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(1, n_steps + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            v[:, i] = v[:, i-1] + self.kappa * (self.theta - np.maximum(v[:, i-1], 0)) * dt + self.sigma_v * np.sqrt(np.maximum(v[:, i-1], 0)) * np.sqrt(dt) * w2
            v[:, i] = np.maximum(v[:, i], 0)
            
            S[:, i] = S[:, i-1] * np.exp((r - 0.5 * v[:, i-1]) * dt + np.sqrt(np.maximum(v[:, i-1], 0)) * np.sqrt(dt) * w1)
        
        return S, v
    
    def price(self, K, T, r, option_type, n_paths=100000, n_steps=252, seed=None):
        S_paths, _ = self.generate_paths(T, n_steps, n_paths, r, seed)
        S_T = S_paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, std_error
    
    @staticmethod
    def calibrate(market_prices, strikes, maturities, S0, r, initial_guess=None):
        if initial_guess is None:
            initial_guess = [0.04, 2.0, 0.04, 0.3, -0.7]
        
        def objective(params):
            v0, kappa, theta, sigma_v, rho = params
            model = Heston(S0, v0, kappa, theta, sigma_v, rho)
            
            mse = 0
            for i, T in enumerate(maturities):
                for j, K in enumerate(strikes):
                    price, _ = model.price(K, T, r, 'call')
                    mse += (price - market_prices[i][j])**2
            
            return mse
        
        bounds = [
            (0.001, 0.5),     # v0
            (0.1, 10.0),      # kappa
            (0.001, 1.0),     # theta
            (0.01, 2.0),      # sigma_v
            (-0.99, 0.99)     # rho
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        return result.x
