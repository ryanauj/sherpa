# ABOUTME: Visualizes common activation functions (sigmoid, ReLU, tanh) and their derivatives.
# ABOUTME: Run with `python activation_functions.py` to see side-by-side comparisons.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Input range for plots
X_MIN, X_MAX = -5, 5


def sigmoid(x):
    """Squashes any input to the range (0, 1)."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """Passes positive values through, zeros out negatives."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU: 1 for positive x, 0 for negative x."""
    return np.where(x > 0, 1.0, 0.0)


def tanh_fn(x):
    """Squashes any input to the range (-1, 1)."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh: 1 - tanh(x)²."""
    return 1 - np.tanh(x)**2


def plot_all_activations():
    """Show all three activation functions side by side."""
    x = np.linspace(X_MIN, X_MAX, 200)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sigmoid
    axes[0].plot(x, sigmoid(x), "b-", linewidth=2.5)
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].axhline(y=1, color="gray", linewidth=0.5, linestyle="--")
    axes[0].axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    axes[0].axvline(x=0, color="k", linewidth=0.5)
    axes[0].set_title("Sigmoid\nσ(x) = 1 / (1 + e⁻ˣ)", fontsize=12)
    axes[0].set_ylabel("Output")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate("Output range: (0, 1)\nSmoothly squashes\nany input", xy=(0.05, 0.95),
                     xycoords="axes fraction", fontsize=9, va="top",
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # ReLU
    axes[1].plot(x, relu(x), "r-", linewidth=2.5)
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)
    axes[1].set_title("ReLU\nf(x) = max(0, x)", fontsize=12)
    axes[1].set_ylim(-1, X_MAX + 0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate("Output range: [0, ∞)\nZeros out negatives\nPasses positives through", xy=(0.05, 0.95),
                     xycoords="axes fraction", fontsize=9, va="top",
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # Tanh
    axes[2].plot(x, tanh_fn(x), "g-", linewidth=2.5)
    axes[2].axhline(y=0, color="k", linewidth=0.5)
    axes[2].axhline(y=1, color="gray", linewidth=0.5, linestyle="--")
    axes[2].axhline(y=-1, color="gray", linewidth=0.5, linestyle="--")
    axes[2].axvline(x=0, color="k", linewidth=0.5)
    axes[2].set_title("Tanh\ntanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)", fontsize=12)
    axes[2].set_ylim(-1.4, 1.4)
    axes[2].grid(True, alpha=0.3)
    axes[2].annotate("Output range: (-1, 1)\nCentered at zero\nLike sigmoid but symmetric", xy=(0.05, 0.95),
                     xycoords="axes fraction", fontsize=9, va="top",
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))

    for ax in axes:
        ax.set_xlabel("Input")

    fig.suptitle("Activation Functions: How Neurons Squish Their Output", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_activations_with_derivatives():
    """Show each activation function alongside its derivative."""
    x = np.linspace(X_MIN, X_MAX, 200)

    activations = [
        ("Sigmoid", sigmoid, sigmoid_derivative, "blue"),
        ("ReLU", relu, relu_derivative, "red"),
        ("Tanh", tanh_fn, tanh_derivative, "green"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, (name, fn, dfn, color) in enumerate(activations):
        # Function
        axes[0, i].plot(x, fn(x), color=color, linewidth=2.5)
        axes[0, i].axhline(y=0, color="k", linewidth=0.5)
        axes[0, i].axvline(x=0, color="k", linewidth=0.5)
        axes[0, i].set_title(f"{name}", fontsize=13)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylabel("f(x)")

        # Derivative
        axes[1, i].plot(x, dfn(x), color=color, linewidth=2.5, linestyle="--")
        axes[1, i].axhline(y=0, color="k", linewidth=0.5)
        axes[1, i].axvline(x=0, color="k", linewidth=0.5)
        axes[1, i].set_title(f"{name} Derivative", fontsize=13)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylabel("f'(x)")
        axes[1, i].set_xlabel("x")

    fig.suptitle("Activation Functions and Their Derivatives\n(Derivatives matter for backpropagation!)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_why_nonlinearity():
    """Demonstrate why linear activations aren't useful — stacking linears = still linear."""
    x = np.linspace(-3, 3, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear composition: f(g(x)) where f(x) = 2x, g(x) = 3x → 6x (still linear!)
    g = 3 * x
    f_of_g = 2 * g  # 6x
    axes[0].plot(x, g, "b-", linewidth=2, label="g(x) = 3x (layer 1)")
    axes[0].plot(x, f_of_g, "r-", linewidth=2, label="f(g(x)) = 2(3x) = 6x (2 layers)")
    axes[0].plot(x, 6 * x, "g--", linewidth=2, label="h(x) = 6x (1 layer)")
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].axvline(x=0, color="k", linewidth=0.5)
    axes[0].set_title("Without Activation Functions\nStacking linear layers = one linear layer", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # With ReLU: f(relu(g(x))) — can represent nonlinear functions
    g = 3 * x
    relu_g = relu(g)
    f_of_relu_g = 2 * relu_g - 3
    axes[1].plot(x, f_of_relu_g, "purple", linewidth=2.5,
                 label="Layer 2 → ReLU → Layer 1\n(nonlinear!)")
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)
    axes[1].set_title("With ReLU Activation\nThe network can learn curves and bends", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Why Neural Networks Need Nonlinear Activation Functions", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Activation Functions")
    print("=" * 40)
    print()
    print("Testing each function at x = -2, 0, 2:")
    for x_val in [-2, 0, 2]:
        print(f"\n  x = {x_val}:")
        print(f"    sigmoid({x_val}) = {sigmoid(x_val):.4f}")
        print(f"    relu({x_val})    = {relu(x_val):.4f}")
        print(f"    tanh({x_val})    = {tanh_fn(x_val):.4f}")
    print()
    print("Close each plot window to see the next one.")

    plot_all_activations()
    plt.show()

    plot_activations_with_derivatives()
    plt.show()

    plot_why_nonlinearity()
    plt.show()
