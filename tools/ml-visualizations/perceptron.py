# ABOUTME: Visualizes a single perceptron's decision boundary in 2D.
# ABOUTME: Run with `python perceptron.py` to see how weights and bias define a separating line.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Perceptron weights and bias
# The decision boundary is: w1*x + w2*y + bias = 0
WEIGHTS = np.array([1.0, 1.5])
BIAS = -2.0

# Generate some sample 2D data points
np.random.seed(42)
N_POINTS = 50
CLASS_0 = np.random.randn(N_POINTS, 2) * 0.5 + np.array([0, 0])
CLASS_1 = np.random.randn(N_POINTS, 2) * 0.5 + np.array([3, 2])


def perceptron_output(x, weights, bias):
    """Compute perceptron output: 1 if w·x + bias > 0, else 0."""
    return 1 if np.dot(weights, x) + bias > 0 else 0


def plot_decision_boundary():
    """Show the perceptron's decision boundary separating two classes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot data points
    ax.scatter(CLASS_0[:, 0], CLASS_0[:, 1], c="blue", marker="o", label="Class 0", s=40, alpha=0.7)
    ax.scatter(CLASS_1[:, 0], CLASS_1[:, 1], c="red", marker="x", label="Class 1", s=40, alpha=0.7)

    # Decision boundary: w1*x + w2*y + bias = 0  →  y = -(w1*x + bias) / w2
    x_range = np.linspace(-2, 5, 100)
    if WEIGHTS[1] != 0:
        boundary_y = -(WEIGHTS[0] * x_range + BIAS) / WEIGHTS[1]
        ax.plot(x_range, boundary_y, "g-", linewidth=2, label="Decision boundary")

        # Shade the regions
        ax.fill_between(x_range, boundary_y, 5, alpha=0.05, color="red")
        ax.fill_between(x_range, boundary_y, -3, alpha=0.05, color="blue")

    # Draw the weight vector (perpendicular to the boundary)
    # Scale it for visibility
    w_scale = 1.0
    midpoint = np.array([1.0, -(WEIGHTS[0] * 1.0 + BIAS) / WEIGHTS[1]]) if WEIGHTS[1] != 0 else np.array([0, 0])
    ax.annotate(
        "",
        xy=midpoint + w_scale * WEIGHTS / np.linalg.norm(WEIGHTS),
        xytext=midpoint,
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax.annotate(
        f"w = {WEIGHTS}",
        xy=midpoint + w_scale * WEIGHTS / np.linalg.norm(WEIGHTS),
        fontsize=10, color="green", fontweight="bold",
        xytext=(10, 5), textcoords="offset points",
    )

    ax.set_xlim(-2, 5)
    ax.set_ylim(-3, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title(
        f"Perceptron Decision Boundary\n"
        f"Weights: {WEIGHTS}, Bias: {BIAS}\n"
        f"Boundary: {WEIGHTS[0]}x + {WEIGHTS[1]}y + ({BIAS}) = 0",
        fontsize=12,
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    return fig


def plot_xor_problem():
    """Show why a single perceptron can't solve XOR."""
    xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_y = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AND gate — linearly separable
    and_y = np.array([0, 0, 0, 1])
    for i, (point, label) in enumerate(zip(xor_x, and_y)):
        color = "red" if label == 1 else "blue"
        marker = "x" if label == 1 else "o"
        axes[0].scatter(*point, c=color, marker=marker, s=200, zorder=5)

    # Draw a separating line for AND
    x_line = np.linspace(-0.5, 1.5, 100)
    y_line = -x_line + 1.5
    axes[0].plot(x_line, y_line, "g--", linewidth=2, label="Decision boundary")
    axes[0].set_title("AND Gate — Linearly Separable\nA single perceptron CAN solve this", fontsize=11)
    axes[0].set_xlim(-0.5, 1.5)
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")

    # XOR gate — NOT linearly separable
    for i, (point, label) in enumerate(zip(xor_x, xor_y)):
        color = "red" if label == 1 else "blue"
        marker = "x" if label == 1 else "o"
        axes[1].scatter(*point, c=color, marker=marker, s=200, zorder=5)

    axes[1].set_title("XOR Gate — NOT Linearly Separable\nNo single line can separate these!", fontsize=11)
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")

    # Try a few lines to show none work
    for slope, intercept, style in [(-1, 0.5, "r--"), (-1, 1.5, "b--"), (1, -0.5, "g--")]:
        y_line = slope * x_line + intercept
        axes[1].plot(x_line, y_line, style, linewidth=1, alpha=0.5)

    fig.suptitle("The XOR Problem: Why We Need Multiple Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Perceptron Decision Boundary")
    print("=" * 40)
    print(f"Weights: {WEIGHTS}")
    print(f"Bias: {BIAS}")
    print(f"Decision rule: {WEIGHTS[0]}*x₁ + {WEIGHTS[1]}*x₂ + ({BIAS}) > 0 → Class 1")
    print()

    # Classify some points
    test_points = [[0, 0], [3, 2], [1, 1], [2, 1]]
    for p in test_points:
        result = perceptron_output(p, WEIGHTS, BIAS)
        raw = np.dot(WEIGHTS, p) + BIAS
        print(f"  Point {p}: w·x + b = {raw:.2f} → Class {result}")

    print()
    print("Close each plot window to see the next one.")

    plot_decision_boundary()
    plt.show()

    plot_xor_problem()
    plt.show()
