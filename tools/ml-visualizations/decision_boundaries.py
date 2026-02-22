# ABOUTME: Visualizes how layers of a neural network warp 2D space to create decision boundaries.
# ABOUTME: Run with `python decision_boundaries.py` to see multi-layer space transformation.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Network architecture: [input_dim, hidden1, hidden2, ..., output_dim]
LAYER_SIZES = [2, 4, 2]

# Random seed for reproducible weights
SEED = 42

# Grid parameters
GRID_RANGE = 2.0
GRID_POINTS = 20


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def make_grid(grid_range, n_points):
    """Create a grid of 2D points."""
    x = np.linspace(-grid_range, grid_range, n_points)
    y = np.linspace(-grid_range, grid_range, n_points)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points, xx, yy


def init_weights(layer_sizes, seed=SEED):
    """Initialize random weights and biases for each layer."""
    np.random.seed(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.8
        b = np.random.randn(layer_sizes[i + 1]) * 0.3
        weights.append(w)
        biases.append(b)
    return weights, biases


def forward_pass(points, weights, biases, activation=relu):
    """Run a forward pass, collecting intermediate representations."""
    layers = [points]
    current = points
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = current @ w + b
        if i < len(weights) - 1:  # Apply activation to all but the last layer
            current = activation(z)
        else:
            current = z
        layers.append(current)
    return layers


def color_by_quadrant(points):
    """Assign colors based on which quadrant each point is in."""
    colors = np.zeros(len(points))
    colors[(points[:, 0] >= 0) & (points[:, 1] >= 0)] = 0  # Q1
    colors[(points[:, 0] < 0) & (points[:, 1] >= 0)] = 1   # Q2
    colors[(points[:, 0] < 0) & (points[:, 1] < 0)] = 2    # Q3
    colors[(points[:, 0] >= 0) & (points[:, 1] < 0)] = 3   # Q4
    return colors


def plot_space_transformation():
    """Show how each layer transforms the input space."""
    points, _, _ = make_grid(GRID_RANGE, GRID_POINTS)
    weights, biases = init_weights(LAYER_SIZES)
    layers = forward_pass(points, weights, biases)
    colors = color_by_quadrant(points)

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    cmap = plt.cm.Set1

    for i, (ax, layer_points) in enumerate(zip(axes, layers)):
        # Plot the first two dimensions (or only available dimensions)
        if layer_points.shape[1] >= 2:
            ax.scatter(layer_points[:, 0], layer_points[:, 1],
                       c=colors, cmap=cmap, s=15, alpha=0.7)

            # Draw grid lines to show warping
            for row_idx in range(GRID_POINTS):
                start = row_idx * GRID_POINTS
                end = start + GRID_POINTS
                ax.plot(layer_points[start:end, 0], layer_points[start:end, 1],
                        "k-", alpha=0.15, linewidth=0.5)

            for col_idx in range(GRID_POINTS):
                indices = list(range(col_idx, GRID_POINTS * GRID_POINTS, GRID_POINTS))
                ax.plot(layer_points[indices, 0], layer_points[indices, 1],
                        "k-", alpha=0.15, linewidth=0.5)
        else:
            ax.scatter(layer_points[:, 0], np.zeros(len(layer_points)),
                       c=colors, cmap=cmap, s=15, alpha=0.7)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.set_title(f"Input Space\n({LAYER_SIZES[0]} dimensions)", fontsize=12)
        elif i == n_layers - 1:
            ax.set_title(f"Output (Layer {i})\n({LAYER_SIZES[-1]} dimensions)", fontsize=12)
        else:
            ax.set_title(f"After Layer {i}\n({LAYER_SIZES[i]} dimensions)\n+ ReLU activation", fontsize=12)

    fig.suptitle(
        f"How a Neural Network Warps Space\n"
        f"Architecture: {' → '.join(str(s) for s in LAYER_SIZES)}  |  "
        f"Colors = original quadrants",
        fontsize=14, y=1.05,
    )
    fig.tight_layout()
    return fig


def plot_xor_solution():
    """Show how a 2-layer network can solve XOR by warping space."""
    # Hand-crafted weights that solve XOR
    # Layer 1: map to a space where XOR is linearly separable
    w1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])
    # Layer 2: linear separator in the transformed space
    w2 = np.array([[1], [-2]])
    b2 = np.array([0])

    points, _, _ = make_grid(1.5, 25)

    # Forward pass through the hand-crafted network
    z1 = points @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    output = sigmoid(z2)

    # XOR data points
    xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_y = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Input space with XOR points
    colors = output.ravel()
    axes[0].contourf(
        points[:, 0].reshape(25, 25),
        points[:, 1].reshape(25, 25),
        colors.reshape(25, 25),
        levels=20, cmap="RdBu_r", alpha=0.5,
    )
    for i, (point, label) in enumerate(zip(xor_x, xor_y)):
        color = "red" if label == 1 else "blue"
        marker = "x" if label == 1 else "o"
        axes[0].scatter(*point, c=color, marker=marker, s=200, zorder=5, edgecolors="black")
    axes[0].set_title("Input Space\nXOR is NOT linearly separable", fontsize=12)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # After first layer + ReLU
    xor_transformed = relu(xor_x @ w1 + b1)
    a1_grid = relu(points @ w1 + b1)
    axes[1].scatter(a1_grid[:, 0], a1_grid[:, 1], c=colors, cmap="RdBu_r", s=10, alpha=0.5)
    for i, (point, label) in enumerate(zip(xor_transformed, xor_y)):
        color = "red" if label == 1 else "blue"
        marker = "x" if label == 1 else "o"
        axes[1].scatter(*point, c=color, marker=marker, s=200, zorder=5, edgecolors="black")
    axes[1].set_title("After Layer 1 + ReLU\nSpace is warped — now separable!", fontsize=12)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # Network output (decision boundary in input space)
    grid_range = np.linspace(-0.5, 1.5, 100)
    xx, yy = np.meshgrid(grid_range, grid_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    z1_g = grid_points @ w1 + b1
    a1_g = relu(z1_g)
    z2_g = a1_g @ w2 + b2
    out_g = sigmoid(z2_g)

    axes[2].contourf(xx, yy, out_g.reshape(100, 100), levels=20, cmap="RdBu_r", alpha=0.5)
    axes[2].contour(xx, yy, out_g.reshape(100, 100), levels=[0.5], colors="green", linewidths=2)
    for i, (point, label) in enumerate(zip(xor_x, xor_y)):
        color = "red" if label == 1 else "blue"
        marker = "x" if label == 1 else "o"
        axes[2].scatter(*point, c=color, marker=marker, s=200, zorder=5, edgecolors="black")
    axes[2].set_title("Network Output\nNonlinear decision boundary!", fontsize=12)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Solving XOR: How a Neural Network Warps Space to Create a Decision Boundary",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Decision Boundaries: How Neural Networks Warp Space")
    print("=" * 55)
    print(f"Network architecture: {' → '.join(str(s) for s in LAYER_SIZES)}")
    print()
    print("Close each plot window to see the next one.")

    plot_space_transformation()
    plt.show()

    plot_xor_solution()
    plt.show()
