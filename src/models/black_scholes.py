import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        # direct formula -- evite d'appeler _d1() et de tout recalculer
        return (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def price(self, option_type):
        d1, d2 = self._d1(), self._d2()

        if option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self, option_type):
        d1 = self._d1()
        if option_type == 'call':
            return norm.cdf(d1)
        elif option_type == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self):
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100

    def theta(self, option_type):
        d1, d2 = self._d1(), self._d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))

        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return (term1 - term2) / 365
        elif option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return (term1 + term2) / 365
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def rho(self, option_type):
        d2 = self._d2()
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        elif option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        else:
            raise ValueError("option_type must be 'call' or 'put'")
