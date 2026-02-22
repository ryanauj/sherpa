# ABOUTME: Step-by-step backpropagation trace through a small neural network.
# ABOUTME: Run with `python backprop_trace.py` to see gradients flow backwards layer by layer.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# A tiny network: 2 inputs → 2 hidden → 1 output
# Fixed weights so the trace is reproducible and easy to follow
INPUT = np.array([0.5, 0.8])
TARGET = 1.0

# Layer 1 weights and biases (2 inputs → 2 hidden)
W1 = np.array([[0.3, 0.7],
                [0.5, 0.2]])
B1 = np.array([0.1, -0.1])

# Layer 2 weights and biases (2 hidden → 1 output)
W2 = np.array([[0.6],
                [0.4]])
B2 = np.array([0.05])

LEARNING_RATE = 0.5


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid: σ(x) * (1 - σ(x))."""
    s = sigmoid(x)
    return s * (1 - s)


def forward_pass(x, w1, b1, w2, b2):
    """Forward pass through the network, recording all intermediate values."""
    # Layer 1
    z1 = x @ w1 + b1            # Pre-activation
    a1 = sigmoid(z1)             # Post-activation

    # Layer 2
    z2 = a1 @ w2 + b2           # Pre-activation
    a2 = sigmoid(z2)             # Post-activation (output)

    return {
        "x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2,
    }


def backward_pass(cache, target, w1, b1, w2, b2):
    """Backward pass: compute gradients for all weights and biases."""
    x = cache["x"]
    z1, a1 = cache["z1"], cache["a1"]
    z2, a2 = cache["z2"], cache["a2"]

    # Loss: MSE = (a2 - target)²
    loss = (a2[0] - target) ** 2

    # --- Output layer gradients ---
    # dL/da2 = 2(a2 - target)
    dL_da2 = 2 * (a2[0] - target)

    # da2/dz2 = sigmoid'(z2)
    da2_dz2 = sigmoid_derivative(z2[0])

    # dL/dz2 = dL/da2 * da2/dz2  (chain rule!)
    dL_dz2 = dL_da2 * da2_dz2

    # dz2/dw2 = a1 (the input to this layer)
    # dL/dw2 = dL/dz2 * dz2/dw2
    dL_dw2 = a1.reshape(-1, 1) * dL_dz2

    # dL/db2 = dL/dz2
    dL_db2 = np.array([dL_dz2])

    # --- Hidden layer gradients ---
    # dz2/da1 = w2
    # dL/da1 = dL/dz2 * w2^T
    dL_da1 = dL_dz2 * w2.ravel()

    # da1/dz1 = sigmoid'(z1)
    da1_dz1 = sigmoid_derivative(z1)

    # dL/dz1 = dL/da1 * da1/dz1
    dL_dz1 = dL_da1 * da1_dz1

    # dL/dw1 = x^T * dL/dz1
    dL_dw1 = x.reshape(-1, 1) @ dL_dz1.reshape(1, -1)

    # dL/db1 = dL/dz1
    dL_db1 = dL_dz1

    return {
        "loss": loss,
        "dL_da2": dL_da2, "da2_dz2": da2_dz2, "dL_dz2": dL_dz2,
        "dL_dw2": dL_dw2, "dL_db2": dL_db2,
        "dL_da1": dL_da1, "da1_dz1": da1_dz1, "dL_dz1": dL_dz1,
        "dL_dw1": dL_dw1, "dL_db1": dL_db1,
    }


def print_forward_trace(cache):
    """Print the forward pass step by step."""
    print("FORWARD PASS")
    print("=" * 60)
    print(f"Input: x = {cache['x']}")
    print()
    print("Layer 1:")
    print(f"  z1 = x @ W1 + b1 = {cache['x']} @ W1 + {B1}")
    print(f"     = {cache['z1']}")
    print(f"  a1 = sigmoid(z1) = {cache['a1']}")
    print()
    print("Layer 2:")
    print(f"  z2 = a1 @ W2 + b2 = {cache['a1']} @ W2 + {B2}")
    print(f"     = {cache['z2']}")
    print(f"  a2 = sigmoid(z2) = {cache['a2']}")
    print(f"  (This is the network's prediction: {cache['a2'][0]:.6f})")
    print(f"  (Target: {TARGET})")


def print_backward_trace(grads):
    """Print the backward pass step by step."""
    print()
    print("BACKWARD PASS (Backpropagation)")
    print("=" * 60)
    print(f"Loss = (prediction - target)² = {grads['loss']:.6f}")
    print()
    print("--- Output Layer (working backwards) ---")
    print(f"  dL/da2 = 2(a2 - target) = {grads['dL_da2']:.6f}")
    print(f"  da2/dz2 = sigmoid'(z2)  = {grads['da2_dz2']:.6f}")
    print(f"  dL/dz2 = dL/da2 × da2/dz2 = {grads['dL_dz2']:.6f}  ← Chain rule!")
    print(f"  dL/dW2 = a1ᵀ × dL/dz2   = {grads['dL_dw2'].ravel()}")
    print(f"  dL/db2 = dL/dz2          = {grads['dL_db2']}")
    print()
    print("--- Hidden Layer (keep going backwards) ---")
    print(f"  dL/da1 = dL/dz2 × W2ᵀ   = {grads['dL_da1']}")
    print(f"  da1/dz1 = sigmoid'(z1)   = {grads['da1_dz1']}")
    print(f"  dL/dz1 = dL/da1 × da1/dz1 = {grads['dL_dz1']}  ← Chain rule again!")
    print(f"  dL/dW1 = xᵀ × dL/dz1:")
    print(f"           {grads['dL_dw1']}")
    print(f"  dL/db1 = dL/dz1          = {grads['dL_db1']}")


def print_weight_update(grads, lr):
    """Print the weight update step."""
    print()
    print("WEIGHT UPDATE")
    print("=" * 60)
    print(f"Learning rate: {lr}")
    print()
    print("W2_new = W2 - lr × dL/dW2:")
    new_w2 = W2 - lr * grads["dL_dw2"]
    print(f"  {W2.ravel()} - {lr} × {grads['dL_dw2'].ravel()}")
    print(f"  = {new_w2.ravel()}")
    print()
    print("W1_new = W1 - lr × dL/dW1:")
    new_w1 = W1 - lr * grads["dL_dw1"]
    print(f"  Old W1:\n  {W1}")
    print(f"  New W1:\n  {new_w1}")


def plot_network_diagram(cache, grads):
    """Draw the network as a diagram with values and gradients annotated."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node positions
    input_pos = [(0, 3.5), (0, 1.5)]
    hidden_pos = [(2.5, 4), (2.5, 1)]
    output_pos = [(5, 2.5)]

    node_radius = 0.35

    # Draw connections with weight labels
    for i, ip in enumerate(input_pos):
        for j, hp in enumerate(hidden_pos):
            color = "blue" if grads["dL_dw1"][i, j] < 0 else "red"
            lw = min(3, max(0.5, abs(grads["dL_dw1"][i, j]) * 10))
            ax.annotate("", xy=hp, xytext=ip,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1))
            mid = ((ip[0] + hp[0]) / 2, (ip[1] + hp[1]) / 2)
            ax.text(mid[0], mid[1] + 0.2, f"w={W1[i, j]:.1f}",
                    fontsize=8, ha="center", color="gray")
            ax.text(mid[0], mid[1] - 0.2, f"∂={grads['dL_dw1'][i, j]:.4f}",
                    fontsize=7, ha="center", color=color)

    for j, hp in enumerate(hidden_pos):
        for k, op in enumerate(output_pos):
            color = "blue" if grads["dL_dw2"][j, k] < 0 else "red"
            ax.annotate("", xy=op, xytext=hp,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1))
            mid = ((hp[0] + op[0]) / 2, (hp[1] + op[1]) / 2)
            ax.text(mid[0], mid[1] + 0.3, f"w={W2[j, k]:.1f}",
                    fontsize=8, ha="center", color="gray")
            ax.text(mid[0], mid[1] - 0.1, f"∂={grads['dL_dw2'][j, k]:.4f}",
                    fontsize=7, ha="center", color=color)

    # Draw nodes
    for i, pos in enumerate(input_pos):
        circle = plt.Circle(pos, node_radius, color="lightblue", ec="black", lw=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"x{i+1}\n{cache['x'][i]:.2f}",
                ha="center", va="center", fontsize=9, fontweight="bold")

    for j, pos in enumerate(hidden_pos):
        circle = plt.Circle(pos, node_radius, color="lightyellow", ec="black", lw=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"h{j+1}\n{cache['a1'][j]:.3f}",
                ha="center", va="center", fontsize=9, fontweight="bold")

    for k, pos in enumerate(output_pos):
        circle = plt.Circle(pos, node_radius, color="lightcoral", ec="black", lw=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"out\n{cache['a2'][k]:.4f}",
                ha="center", va="center", fontsize=9, fontweight="bold")

    # Labels
    ax.text(0, 4.5, "Input", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.5, 4.8, "Hidden", ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 3.5, "Output", ha="center", fontsize=12, fontweight="bold")

    # Legend
    ax.text(0, -0.5, f"Target: {TARGET}  |  Prediction: {cache['a2'][0]:.4f}  |  "
            f"Loss: {grads['loss']:.6f}", fontsize=11, fontweight="bold")
    ax.text(0, -0.9, "Gray = weights  |  Blue ∂ = decrease weight  |  Red ∂ = increase weight",
            fontsize=9, color="gray")

    ax.set_title("Backpropagation: Values (forward) and Gradients (backward)", fontsize=14)
    return fig


if __name__ == "__main__":
    cache = forward_pass(INPUT, W1, B1, W2, B2)
    print_forward_trace(cache)

    grads = backward_pass(cache, TARGET, W1, B1, W2, B2)
    print_backward_trace(grads)

    print_weight_update(grads, LEARNING_RATE)

    print()
    print("=" * 60)
    print("Close the plot window to exit.")

    plot_network_diagram(cache, grads)
    plt.tight_layout()
    plt.show()
