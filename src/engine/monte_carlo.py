import numpy as np


def generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # GBM has an exact solution -- no need to loop, fully vectorized
    Z = rng.standard_normal((n_paths, n_steps))
    increments = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.cumprod(increments, axis=1)

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
    n_paths, n_steps_plus1 = S_paths.shape
    n_steps = n_steps_plus1 - 1
    T = n_steps * dt

    if option_type == 'call':
        payoffs = np.maximum(S_paths - K, 0)
    else:
        payoffs = np.maximum(K - S_paths, 0)

    # store everything discounted to t=0 dès le départ
    cashflow = np.exp(-r * T) * payoffs[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        itm = payoffs[:, t] > 0
        if np.sum(itm) == 0:
            continue

        X = S_paths[itm, t]
        Y = cashflow[itm]  # already discounted to 0

        poly = np.polyfit(X, Y, 2)  # degree 2 should be enough
        continuation = np.polyval(poly, X)

        exercise_val = np.exp(-r * t * dt) * payoffs[itm, t]
        cashflow[itm] = np.where(exercise_val > continuation, exercise_val, cashflow[itm])

    price = np.mean(cashflow)

    # Pour un call sans dividende, on force le prix européen (théorème)
    if option_type == 'call':
        S_T = S_paths[:, -1]
        european_price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        return european_price

    return price
