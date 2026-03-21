import numpy as np

def finite_difference_delta(price_func, S, epsilon=0.01, *args, **kwargs):
    price_up = price_func(S * (1 + epsilon), *args, **kwargs)
    price_down = price_func(S * (1 - epsilon), *args, **kwargs)
    return (price_up - price_down) / (2 * S * epsilon)

def finite_difference_gamma(price_func, S, epsilon=0.01, *args, **kwargs):
    price_up = price_func(S * (1 + epsilon), *args, **kwargs)
    price_down = price_func(S * (1 - epsilon), *args, **kwargs)
    price_mid = price_func(S, *args, **kwargs)
    return (price_up - 2 * price_mid + price_down) / (S * epsilon)**2

def finite_difference_vega(price_func, sigma, epsilon=0.01, *args, **kwargs):
    price_up = price_func(sigma * (1 + epsilon), *args, **kwargs)
    price_down = price_func(sigma * (1 - epsilon), *args, **kwargs)
    return (price_up - price_down) / (2 * sigma * epsilon) / 100
