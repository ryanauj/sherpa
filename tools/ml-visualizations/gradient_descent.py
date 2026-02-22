# ABOUTME: Animated gradient descent on a 2D contour plot showing the optimization path.
# ABOUTME: Run with `python gradient_descent.py` to watch gradient descent find a minimum.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Learning rate — try making this bigger (0.5) or smaller (0.001)
LEARNING_RATE = 0.1

# Starting point for gradient descent
START_X, START_Y = -2.0, 2.0

# Number of gradient descent steps
N_STEPS = 30

# Loss function: a simple bowl (quadratic)
# Try changing the coefficients to make it more elliptical
A, B = 1.0, 3.0  # f(x, y) = A*x² + B*y²


def loss_function(x, y):
    """The function we're trying to minimize."""
    return A * x**2 + B * y**2


def gradient(x, y):
    """The gradient (vector of partial derivatives) of the loss function."""
    df_dx = 2 * A * x
    df_dy = 2 * B * y
    return np.array([df_dx, df_dy])


def run_gradient_descent(start, lr, n_steps):
    """Run gradient descent and record the path."""
    path = [np.array(start)]
    current = np.array(start, dtype=float)
    for _ in range(n_steps):
        grad = gradient(current[0], current[1])
        current = current - lr * grad
        path.append(current.copy())
    return np.array(path)


def plot_contour_with_path():
    """Show gradient descent path on a contour plot of the loss function."""
    # Create contour data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    # Run gradient descent
    path = run_gradient_descent([START_X, START_Y], LEARNING_RATE, N_STEPS)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Contour plot
    contour = ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.3)
    ax.clabel(contour, inline=True, fontsize=8)

    # Gradient descent path
    ax.plot(path[:, 0], path[:, 1], "r.-", markersize=8, linewidth=1.5,
            label="Gradient descent path")
    ax.plot(path[0, 0], path[0, 1], "rs", markersize=12, label="Start")
    ax.plot(path[-1, 0], path[-1, 1], "r*", markersize=15, label="End")

    # Mark the true minimum
    ax.plot(0, 0, "g*", markersize=15, label="True minimum")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Gradient Descent on f(x,y) = {A}x² + {B}y²\n"
        f"Learning rate = {LEARNING_RATE}, Steps = {N_STEPS}",
        fontsize=13,
    )
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig


def plot_loss_over_steps():
    """Show how the loss decreases over gradient descent steps."""
    path = run_gradient_descent([START_X, START_Y], LEARNING_RATE, N_STEPS)
    losses = [loss_function(p[0], p[1]) for p in path]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(len(losses)), losses, "b.-", markersize=8, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Over Gradient Descent Steps", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    ax.annotate(
        f"Start: loss = {losses[0]:.2f}",
        xy=(0, losses[0]),
        xytext=(5, losses[0]),
        fontsize=10,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        f"End: loss = {losses[-1]:.4f}",
        xy=(len(losses) - 1, losses[-1]),
        xytext=(len(losses) - 10, losses[-1] + 2),
        fontsize=10,
        arrowprops=dict(arrowstyle="->"),
    )

    return fig


def plot_learning_rate_comparison():
    """Show how different learning rates affect convergence."""
    learning_rates = [0.01, 0.1, 0.3]
    colors = ["blue", "green", "red"]

    # Create contour data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, lr, color in zip(axes, learning_rates, colors):
        path = run_gradient_descent([START_X, START_Y], lr, N_STEPS)
        losses = [loss_function(p[0], p[1]) for p in path]

        ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.5)
        ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.2)
        ax.plot(path[:, 0], path[:, 1], ".-", color=color, markersize=6, linewidth=1.5)
        ax.plot(path[0, 0], path[0, 1], "s", color=color, markersize=10)
        ax.plot(path[-1, 0], path[-1, 1], "*", color=color, markersize=12)
        ax.plot(0, 0, "g*", markersize=12)

        ax.set_title(f"lr = {lr}\nFinal loss = {losses[-1]:.4f}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Learning Rate on Gradient Descent", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Gradient Descent Visualization")
    print("=" * 50)
    print(f"Loss function: f(x,y) = {A}x² + {B}y²")
    print(f"Starting point: ({START_X}, {START_Y})")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Steps: {N_STEPS}")
    print()

    path = run_gradient_descent([START_X, START_Y], LEARNING_RATE, N_STEPS)
    start_loss = loss_function(START_X, START_Y)
    end_loss = loss_function(path[-1, 0], path[-1, 1])
    print(f"Starting loss: {start_loss:.4f}")
    print(f"Final loss:    {end_loss:.6f}")
    print(f"Final position: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})")
    print()
    print("Close each plot window to see the next one.")

    plot_contour_with_path()
    plt.show()

    plot_loss_over_steps()
    plt.show()

    plot_learning_rate_comparison()
    plt.show()
