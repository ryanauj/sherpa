# ABOUTME: Visualizes loss surfaces as 3D plots and contour maps for small networks.
# ABOUTME: Run with `python loss_surfaces.py` to explore how loss changes with weights.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Simple dataset: 4 points, trying to learn y = 2x + 1
X_DATA = np.array([0, 1, 2, 3], dtype=float)
Y_DATA = np.array([1, 3, 5, 7], dtype=float)

# Range for weight (slope) and bias (intercept) exploration
W_RANGE = (-1, 5)
B_RANGE = (-3, 5)
GRID_RESOLUTION = 50


def mse_loss(y_pred, y_true):
    """Mean Squared Error: average of squared differences."""
    return np.mean((y_pred - y_true) ** 2)


def compute_loss_surface(x_data, y_data, w_range, b_range, resolution):
    """Compute loss for a grid of (weight, bias) values."""
    weights = np.linspace(w_range[0], w_range[1], resolution)
    biases = np.linspace(b_range[0], b_range[1], resolution)
    W, B = np.meshgrid(weights, biases)
    Z = np.zeros_like(W)

    for i in range(resolution):
        for j in range(resolution):
            y_pred = W[i, j] * x_data + B[i, j]
            Z[i, j] = mse_loss(y_pred, y_data)

    return W, B, Z


def plot_3d_loss_surface():
    """Show the loss surface as a 3D plot."""
    W, B, Z = compute_loss_surface(X_DATA, Y_DATA, W_RANGE, B_RANGE, GRID_RESOLUTION)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(W, B, Z, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Weight (slope)")
    ax.set_ylabel("Bias (intercept)")
    ax.set_zlabel("Loss (MSE)")
    ax.set_title("Loss Surface for y = w·x + b\n(trying to learn y = 2x + 1)", fontsize=13)

    # Mark the minimum
    ax.scatter([2], [1], [0], c="red", s=100, zorder=5, label="Minimum (w=2, b=1)")
    ax.legend()

    fig.colorbar(surf, shrink=0.5, label="Loss")
    return fig


def plot_contour_loss():
    """Show the loss surface as a contour plot (top-down view)."""
    W, B, Z = compute_loss_surface(X_DATA, Y_DATA, W_RANGE, B_RANGE, GRID_RESOLUTION)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    contour = ax.contourf(W, B, Z, levels=30, cmap="viridis")
    ax.contour(W, B, Z, levels=15, colors="white", alpha=0.3, linewidths=0.5)
    ax.plot(2, 1, "r*", markersize=15, label="Minimum (w=2, b=1)")

    ax.set_xlabel("Weight (slope)", fontsize=12)
    ax.set_ylabel("Bias (intercept)", fontsize=12)
    ax.set_title("Loss Contour Map\nEach ring = same loss value", fontsize=13)
    ax.legend(fontsize=11)
    fig.colorbar(contour, label="Loss (MSE)")

    return fig


def plot_gradient_descent_on_loss():
    """Show gradient descent path on the loss contour."""
    W, B, Z = compute_loss_surface(X_DATA, Y_DATA, W_RANGE, B_RANGE, GRID_RESOLUTION)

    # Run gradient descent
    lr = 0.01
    w, b = 0.0, -2.0  # Starting point
    path_w, path_b, path_loss = [w], [b], []
    path_loss.append(mse_loss(w * X_DATA + b, Y_DATA))

    for step in range(100):
        # Gradients of MSE for linear model y = w*x + b
        y_pred = w * X_DATA + b
        errors = y_pred - Y_DATA
        dw = 2 * np.mean(errors * X_DATA)
        db = 2 * np.mean(errors)

        w -= lr * dw
        b -= lr * db

        path_w.append(w)
        path_b.append(b)
        path_loss.append(mse_loss(w * X_DATA + b, Y_DATA))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Contour with path
    axes[0].contourf(W, B, Z, levels=30, cmap="viridis")
    axes[0].contour(W, B, Z, levels=15, colors="white", alpha=0.3, linewidths=0.5)
    axes[0].plot(path_w, path_b, "r.-", markersize=4, linewidth=1.5, label="GD path")
    axes[0].plot(path_w[0], path_b[0], "rs", markersize=10, label="Start")
    axes[0].plot(path_w[-1], path_b[-1], "r*", markersize=15, label=f"End (w={w:.2f}, b={b:.2f})")
    axes[0].plot(2, 1, "g*", markersize=15, label="True minimum")
    axes[0].set_xlabel("Weight (slope)")
    axes[0].set_ylabel("Bias (intercept)")
    axes[0].set_title("Gradient Descent on Loss Surface", fontsize=13)
    axes[0].legend(fontsize=9)

    # Loss over steps
    axes[1].plot(path_loss, "b-", linewidth=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].set_title("Training Loss Over Time", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate(f"Final loss: {path_loss[-1]:.4f}",
                     xy=(len(path_loss) - 1, path_loss[-1]),
                     xytext=(-80, 30), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->"), fontsize=10)

    fig.suptitle("Training = Walking Downhill on the Loss Surface", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loss Surfaces for Linear Regression")
    print("=" * 45)
    print(f"Data: x = {X_DATA}")
    print(f"       y = {Y_DATA}")
    print(f"Target: y = 2x + 1")
    print()
    print("The loss surface shows how MSE changes as we vary weight and bias.")
    print("Gradient descent finds the minimum by walking downhill.")
    print()
    print("Close each plot window to see the next one.")

    plot_3d_loss_surface()
    plt.show()

    plot_contour_loss()
    plt.show()

    plot_gradient_descent_on_loss()
    plt.show()
