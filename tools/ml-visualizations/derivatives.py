# ABOUTME: Visualizes derivatives as tangent lines and slopes of curves.
# ABOUTME: Run with `python derivatives.py` to see interactive matplotlib plots.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# The function to visualize (and its analytical derivative for comparison)
def f(x):
    return x**2

def f_derivative(x):
    return 2 * x

FUNCTION_LABEL = "f(x) = x²"

# Point at which to draw the tangent line
TANGENT_POINT = 1.5

# Step size for numerical derivative
H = 0.001

# Range for plots
X_MIN, X_MAX = -3, 3


def numerical_derivative(func, x, h=H):
    """Compute the derivative numerically using the central difference method."""
    return (func(x + h) - func(x - h)) / (2 * h)


def plot_tangent_line():
    """Show a function with its tangent line at a point."""
    x = np.linspace(X_MIN, X_MAX, 200)
    y = f(x)

    # Tangent line at the chosen point
    slope = numerical_derivative(f, TANGENT_POINT)
    y0 = f(TANGENT_POINT)
    tangent_y = slope * (x - TANGENT_POINT) + y0

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, y, "b-", linewidth=2, label=FUNCTION_LABEL)
    ax.plot(x, tangent_y, "r--", linewidth=2,
            label=f"Tangent at x={TANGENT_POINT} (slope={slope:.2f})")
    ax.plot(TANGENT_POINT, y0, "ro", markersize=8)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-1, X_MAX**2 + 1)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("The Derivative = Slope of the Tangent Line", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    # Annotate
    ax.annotate(
        f"At x = {TANGENT_POINT}:\nf(x) = {y0:.2f}\nf'(x) = {slope:.2f}",
        xy=(TANGENT_POINT, y0),
        xytext=(TANGENT_POINT + 0.8, y0 + 2),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )

    return fig


def plot_function_and_derivative():
    """Show a function and its derivative side by side."""
    x = np.linspace(X_MIN, X_MAX, 200)
    y = f(x)
    dy = f_derivative(x)

    # Also compute numerical derivative at sample points
    sample_x = np.linspace(X_MIN + 0.1, X_MAX - 0.1, 20)
    numerical_dy = np.array([numerical_derivative(f, xi) for xi in sample_x])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Function
    axes[0].plot(x, y, "b-", linewidth=2, label=FUNCTION_LABEL)
    axes[0].set_title(f"Function: {FUNCTION_LABEL}", fontsize=13)
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].axvline(x=0, color="k", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    # Derivative
    axes[1].plot(x, dy, "r-", linewidth=2, label="f'(x) = 2x (analytical)")
    axes[1].plot(sample_x, numerical_dy, "ko", markersize=4, alpha=0.7,
                 label="Numerical derivative")
    axes[1].set_title("Derivative: f'(x) = 2x", fontsize=13)
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")

    fig.suptitle("A Function and Its Derivative", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_derivative_as_rate():
    """Show the derivative as rate of change: zoom in on a small interval."""
    x = np.linspace(X_MIN, X_MAX, 200)
    y = f(x)

    # Show the "rise over run" at the tangent point
    dx = 0.8
    x0 = TANGENT_POINT
    x1 = x0 + dx
    y0 = f(x0)
    y1 = f(x1)
    slope = (y1 - y0) / dx

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, y, "b-", linewidth=2, label=FUNCTION_LABEL)

    # Rise-over-run triangle
    ax.plot([x0, x1], [y0, y0], "g-", linewidth=2)   # Run (horizontal)
    ax.plot([x1, x1], [y0, y1], "r-", linewidth=2)   # Rise (vertical)
    ax.plot([x0, x1], [y0, y1], "k--", linewidth=1.5) # Secant line

    ax.plot(x0, y0, "ko", markersize=8)
    ax.plot(x1, y1, "ko", markersize=8)

    # Labels
    ax.annotate(f"run = {dx}", xy=((x0 + x1) / 2, y0 - 0.3), fontsize=11,
                color="green", ha="center", fontweight="bold")
    ax.annotate(f"rise = {y1 - y0:.2f}", xy=(x1 + 0.1, (y0 + y1) / 2), fontsize=11,
                color="red", fontweight="bold")
    ax.annotate(f"slope ≈ {slope:.2f}", xy=((x0 + x1) / 2, (y0 + y1) / 2 + 0.5),
                fontsize=11, fontweight="bold")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-1, X_MAX**2 + 1)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Derivative as Rate of Change: Rise / Run", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Derivatives: Tangent Lines and Rates of Change")
    print("=" * 50)
    print(f"Function: {FUNCTION_LABEL}")
    print(f"Tangent point: x = {TANGENT_POINT}")
    print(f"f({TANGENT_POINT}) = {f(TANGENT_POINT):.4f}")
    print(f"Numerical derivative: f'({TANGENT_POINT}) = {numerical_derivative(f, TANGENT_POINT):.4f}")
    print(f"Analytical derivative: f'({TANGENT_POINT}) = {f_derivative(TANGENT_POINT):.4f}")
    print()
    print("Close each plot window to see the next one.")

    plot_derivative_as_rate()
    plt.show()

    plot_tangent_line()
    plt.show()

    plot_function_and_derivative()
    plt.show()
