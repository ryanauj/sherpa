# ABOUTME: Visualizes matrix-vector multiplication as a geometric transformation.
# ABOUTME: Run with `python matrices.py` to see how matrices move vectors around.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Matrix to apply (try different ones!)
# Rotation by 45 degrees
MATRIX = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                    [np.sin(np.pi / 4),  np.cos(np.pi / 4)]])

# Vectors to transform
VECTORS = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([2, 0.5]),
]

# Some interesting matrices to try (uncomment one to use it):
# MATRIX = np.array([[2, 0], [0, 2]])       # Uniform scaling (2x)
# MATRIX = np.array([[1, 0], [0, -1]])      # Reflection over x-axis
# MATRIX = np.array([[1, 1], [0, 1]])       # Horizontal shear
# MATRIX = np.array([[0, -1], [1, 0]])      # 90-degree rotation
# MATRIX = np.array([[0.5, 0], [0, 2]])     # Non-uniform scaling


def setup_axes(ax, title, grid_range=4):
    """Configure axes with grid, origin, and labels."""
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_title(title)


def draw_vector(ax, vector, origin=(0, 0), color="blue", label=None, alpha=1.0):
    """Draw a vector as an arrow from origin."""
    ax.annotate(
        "",
        xy=(origin[0] + vector[0], origin[1] + vector[1]),
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=alpha),
    )
    if label:
        ax.annotate(
            label,
            xy=(origin[0] + vector[0], origin[1] + vector[1]),
            fontsize=9,
            color=color,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )


def plot_matrix_transform():
    """Show before and after of matrix-vector multiplication."""
    colors = ["blue", "red", "green", "purple"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Before
    setup_axes(axes[0], "Original Vectors")
    for i, v in enumerate(VECTORS):
        draw_vector(axes[0], v, color=colors[i % len(colors)], label=str(v))

    # After
    setup_axes(axes[1], f"After Multiplying by Matrix\n{MATRIX[0]}\n{MATRIX[1]}")
    for i, v in enumerate(VECTORS):
        transformed = MATRIX @ v
        draw_vector(
            axes[1], transformed,
            color=colors[i % len(colors)],
            label=f"{v} → {np.round(transformed, 2)}",
        )

    fig.suptitle("Matrix-Vector Multiplication as Transformation", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_basis_vectors():
    """Show how a matrix transforms the standard basis vectors."""
    e1 = np.array([1, 0])  # Standard basis vector 1
    e2 = np.array([0, 1])  # Standard basis vector 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Before
    setup_axes(axes[0], "Standard Basis Vectors")
    draw_vector(axes[0], e1, color="blue", label="e₁ = [1, 0]")
    draw_vector(axes[0], e2, color="red", label="e₂ = [0, 1]")

    # After — the columns of the matrix ARE where the basis vectors land
    te1 = MATRIX @ e1
    te2 = MATRIX @ e2
    setup_axes(axes[1], "Transformed Basis Vectors\n(These ARE the columns of the matrix!)")
    draw_vector(axes[1], te1, color="blue", label=f"M·e₁ = {np.round(te1, 2)}")
    draw_vector(axes[1], te2, color="red", label=f"M·e₂ = {np.round(te2, 2)}")

    # Show faded original basis for reference
    draw_vector(axes[1], e1, color="blue", alpha=0.2)
    draw_vector(axes[1], e2, color="red", alpha=0.2)

    fig.suptitle("The columns of a matrix show where the basis vectors go", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Matrix-Vector Multiplication as Transformation")
    print("=" * 50)
    print(f"Matrix:\n{MATRIX}")
    print()
    for v in VECTORS:
        result = MATRIX @ v
        print(f"  {v} → {np.round(result, 2)}")
    print()
    print("Close each plot window to see the next one.")

    plot_basis_vectors()
    plt.show()

    plot_matrix_transform()
    plt.show()
