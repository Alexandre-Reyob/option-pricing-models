import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})


def plot_price_surface(prices, strikes, maturities):
    X, Y = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, prices, cmap='plasma', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Strike', labelpad=10)
    ax.set_ylabel('Maturity (y)', labelpad=10)
    ax.set_zlabel('Option Price', labelpad=10)
    ax.set_title('Option Price Surface', pad=15)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_convergence(prices, std_errors, true_price=None):
    n_paths = np.arange(1000, len(prices) * 1000 + 1, 1000)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(n_paths, prices, color='#2c7bb6', lw=1.8, label='MC price')

    upper = np.array(prices) + 1.96 * np.array(std_errors)
    lower = np.array(prices) - 1.96 * np.array(std_errors)
    ax.fill_between(n_paths, lower, upper, alpha=0.25, color='#2c7bb6', label='95% CI')

    if true_price is not None:
        ax.axhline(y=true_price, color='#d7191c', linestyle='--', lw=1.5, label=f'BS price = {true_price:.4f}')

    ax.set_xlabel('Number of paths')
    ax.set_ylabel('Option price')
    ax.set_title('Monte Carlo convergence')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_volatility_smile(implied_vols, strikes, spot):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(strikes, [iv * 100 for iv in implied_vols], 'o-', color='#1a9641', lw=2, ms=6)
    ax.axvline(x=spot, color='gray', linestyle='--', lw=1.2, label=f'Spot = {spot}')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied volatility (%)')
    ax.set_title('Volatility smile')
    ax.legend()
    plt.tight_layout()
    plt.show()
