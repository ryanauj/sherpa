# ABOUTME: Visualizes linear transformations by applying a matrix to a grid of points.
# ABOUTME: Run with `python linear_transforms.py` to watch a grid warp under different matrices.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# The matrix to apply. Try uncommenting different ones!
MATRIX = np.array([[2, 1],
                    [0, 1]])  # Horizontal shear + stretch

# Other matrices to try:
# MATRIX = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
#                     [np.sin(np.pi/6),  np.cos(np.pi/6)]])  # 30° rotation
# MATRIX = np.array([[1, 0], [0, -1]])     # Reflection over x-axis
# MATRIX = np.array([[0, 1], [1, 0]])      # Reflection over y=x
# MATRIX = np.array([[2, 0], [0, 0.5]])    # Stretch x, squish y
# MATRIX = np.array([[0, -1], [1, 0]])     # 90° rotation

# Grid parameters
GRID_RANGE = 2       # Grid spans [-GRID_RANGE, GRID_RANGE]
GRID_POINTS = 11     # Points per grid line


def make_grid(grid_range, n_points):
    """Create a grid of points as a 2xN matrix."""
    t = np.linspace(-grid_range, grid_range, n_points)
    points = []
    # Horizontal lines
    for y_val in t:
        for x_val in t:
            points.append([x_val, y_val])
    return np.array(points).T  # 2 x N


def make_grid_lines(grid_range, n_points):
    """Create grid lines for nicer visualization (returns pairs of points per line)."""
    t = np.linspace(-grid_range, grid_range, n_points)
    dense_t = np.linspace(-grid_range, grid_range, 100)
    lines = []
    # Horizontal lines
    for y_val in t:
        line = np.array([[x, y_val] for x in dense_t]).T
        lines.append(line)
    # Vertical lines
    for x_val in t:
        line = np.array([[x_val, y] for y in dense_t]).T
        lines.append(line)
    return lines


def plot_grid_lines(ax, lines, color="blue", alpha=0.5, linewidth=1):
    """Plot grid lines on an axes."""
    for line in lines:
        ax.plot(line[0], line[1], color=color, alpha=alpha, linewidth=linewidth)


def plot_transformation():
    """Show a grid before and after applying the matrix."""
    lines = make_grid_lines(GRID_RANGE, GRID_POINTS)
    view_range = GRID_RANGE * 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before
    axes[0].set_xlim(-view_range, view_range)
    axes[0].set_ylim(-view_range, view_range)
    axes[0].set_aspect("equal")
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].axvline(x=0, color="k", linewidth=0.5)
    axes[0].set_title("Original Grid")
    plot_grid_lines(axes[0], lines, color="blue")

    # Highlight basis vectors
    axes[0].annotate("", xy=(1, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    axes[0].annotate("", xy=(0, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="green", lw=2.5))
    axes[0].annotate("e₁", xy=(1, 0), fontsize=12, color="red",
                     fontweight="bold", xytext=(5, -15), textcoords="offset points")
    axes[0].annotate("e₂", xy=(0, 1), fontsize=12, color="green",
                     fontweight="bold", xytext=(-20, 5), textcoords="offset points")

    # After
    transformed_lines = [MATRIX @ line for line in lines]
    te1 = MATRIX @ np.array([1, 0])
    te2 = MATRIX @ np.array([0, 1])

    axes[1].set_xlim(-view_range, view_range)
    axes[1].set_ylim(-view_range, view_range)
    axes[1].set_aspect("equal")
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)
    axes[1].set_title(f"After Transformation\nMatrix: [{MATRIX[0]}] [{MATRIX[1]}]")
    plot_grid_lines(axes[1], transformed_lines, color="purple")

    # Highlight transformed basis vectors
    axes[1].annotate("", xy=tuple(te1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    axes[1].annotate("", xy=tuple(te2), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="green", lw=2.5))
    axes[1].annotate(f"M·e₁ = {te1}", xy=tuple(te1), fontsize=10, color="red",
                     fontweight="bold", xytext=(5, -15), textcoords="offset points")
    axes[1].annotate(f"M·e₂ = {te2}", xy=tuple(te2), fontsize=10, color="green",
                     fontweight="bold", xytext=(-20, 5), textcoords="offset points")

    fig.suptitle("Linear Transformation: How a Matrix Warps Space", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_transformation_gallery():
    """Show several standard transformations side by side."""
    matrices = {
        "Identity": np.array([[1, 0], [0, 1]]),
        "Scale 2x": np.array([[2, 0], [0, 2]]),
        "Rotate 45°": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                 [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
        "Shear": np.array([[1, 1], [0, 1]]),
        "Reflect (x-axis)": np.array([[1, 0], [0, -1]]),
        "Squeeze": np.array([[2, 0], [0, 0.5]]),
    }

    lines = make_grid_lines(GRID_RANGE, GRID_POINTS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (name, M) in zip(axes.flat, matrices.items()):
        transformed = [M @ line for line in lines]
        view_range = GRID_RANGE * 3
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-view_range, view_range)
        ax.set_aspect("equal")
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        # Faded original grid
        plot_grid_lines(ax, lines, color="blue", alpha=0.2)
        # Transformed grid
        plot_grid_lines(ax, transformed, color="purple", alpha=0.6)

        ax.set_title(f"{name}\n[{M[0]}]\n[{M[1]}]", fontsize=10)

    fig.suptitle("Gallery of Linear Transformations", fontsize=14)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Linear Transformations: How Matrices Warp Space")
    print("=" * 50)
    print(f"Matrix:\n{MATRIX}")
    print(f"\nBasis vector e₁ = [1, 0] maps to {MATRIX @ np.array([1, 0])}")
    print(f"Basis vector e₂ = [0, 1] maps to {MATRIX @ np.array([0, 1])}")
    print()
    print("Close each plot window to see the next one.")

    plot_transformation()
    plt.show()

    plot_transformation_gallery()
    plt.show()
