
import numpy as np
import matplotlib.pyplot as plt

def plot_price_surface(prices, strikes, maturities):
    X, Y = np.meshgrid(strikes, maturities)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, prices, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Option Price')
    ax.set_title('Option Price Surface')
    
    plt.colorbar(surf)
    plt.show()

def plot_convergence(prices, std_errors, true_price=None):
    n_paths = np.arange(1000, len(prices) * 1000 + 1, 1000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_paths, prices, label='MC Price')
    
    upper = np.array(prices) + 1.96 * np.array(std_errors)
    lower = np.array(prices) - 1.96 * np.array(std_errors)
    
    plt.fill_between(n_paths, lower, upper, alpha=0.3, label='95% CI')
    
    if true_price is not None:
        plt.axhline(y=true_price, color='r', linestyle='--', label='True Price')
    
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volatility_smile(implied_vols, strikes, spot):
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, implied_vols, 'o-')
    plt.axvline(x=spot, color='gray', linestyle='--', label='Spot')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile')
    plt.legend()
    plt.grid(True)
    plt.show()
