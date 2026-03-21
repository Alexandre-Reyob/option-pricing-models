import numpy as np

def generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    for i in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return S

def price_european_mc(S_paths, K, r, T, option_type):
    n_paths = S_paths.shape[0]
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

def price_american_mc(S_paths, K, r, dt, option_type):
    n_paths, n_steps = S_paths.shape
    n_steps -= 1
    T = n_steps * dt
    
    if option_type == 'call':
        payoffs = np.maximum(S_paths - K, 0)
    else:
        payoffs = np.maximum(K - S_paths, 0)
    
    cashflow = payoffs[:, -1].copy()
    
    for t in range(n_steps - 1, 0, -1):
        itm = payoffs[:, t] > 0
        
        if np.sum(itm) > 0:
            X = S_paths[itm, t]
            Y = cashflow[itm] * np.exp(-r * dt)
            
            poly = np.polyfit(X, Y, 2)
            continuation = np.polyval(poly, X)
            
            exercise = payoffs[itm, t]
            cashflow[itm] = np.where(exercise > continuation, exercise, cashflow[itm])
    
    price = np.exp(-r * dt) * np.mean(cashflow)
    
    # Pour un call sans dividende, on force le prix européen
    # On calcule le prix européen par MC pour être cohérent
    if option_type == 'call':
        S_T = S_paths[:, -1]
        payoff_terminal = np.maximum(S_T - K, 0)
        european_price = np.exp(-r * T) * np.mean(payoff_terminal)
        return european_price
    
    return price
