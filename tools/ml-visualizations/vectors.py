# ABOUTME: Visualizes 2D vector operations — addition, subtraction, and scaling.
# ABOUTME: Run with `python vectors.py` to see interactive matplotlib plots.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Two vectors to visualize
VECTOR_A = np.array([3, 1])
VECTOR_B = np.array([1, 2])

# Scalar for scaling demonstration
SCALAR = 2.0


def setup_axes(ax, title, grid_range=8):
    """Configure axes with grid, origin, and labels."""
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_title(title)


def draw_vector(ax, vector, origin=(0, 0), color="blue", label=None):
    """Draw a vector as an arrow from origin."""
    ax.annotate(
        "",
        xy=(origin[0] + vector[0], origin[1] + vector[1]),
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )
    # Label at the tip
    if label:
        ax.annotate(
            label,
            xy=(origin[0] + vector[0], origin[1] + vector[1]),
            fontsize=10,
            color=color,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )


def plot_vectors_as_arrows():
    """Show two vectors as arrows from the origin."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    setup_axes(ax, f"Vectors as Arrows\na = {VECTOR_A}, b = {VECTOR_B}")

    draw_vector(ax, VECTOR_A, color="blue", label=f"a = {VECTOR_A}")
    draw_vector(ax, VECTOR_B, color="red", label=f"b = {VECTOR_B}")

    # Also show them as points
    ax.plot(*VECTOR_A, "bo", markersize=6)
    ax.plot(*VECTOR_B, "ro", markersize=6)

    return fig


def plot_addition():
    """Show vector addition: a + b, with the parallelogram construction."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    result = VECTOR_A + VECTOR_B
    setup_axes(ax, f"Vector Addition\na + b = {VECTOR_A} + {VECTOR_B} = {result}")

    # Draw a
    draw_vector(ax, VECTOR_A, color="blue", label="a")
    # Draw b starting from tip of a (tail-to-tip method)
    draw_vector(ax, VECTOR_B, origin=tuple(VECTOR_A), color="red", label="b")
    # Draw result
    draw_vector(ax, result, color="green", label=f"a + b = {result}")

    # Draw the parallelogram (b from origin, a from tip of b)
    draw_vector(ax, VECTOR_B, color="red")
    draw_vector(ax, VECTOR_A, origin=tuple(VECTOR_B), color="blue")

    # Dashed lines for parallelogram
    ax.plot(
        [VECTOR_A[0], result[0]],
        [VECTOR_A[1], result[1]],
        "--",
        color="gray",
        alpha=0.5,
    )
    ax.plot(
        [VECTOR_B[0], result[0]],
        [VECTOR_B[1], result[1]],
        "--",
        color="gray",
        alpha=0.5,
    )

    return fig


def plot_scaling():
    """Show scalar multiplication: stretching and shrinking vectors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Positive scalar
    scaled = SCALAR * VECTOR_A
    setup_axes(axes[0], f"Scaling by {SCALAR}\n{SCALAR} * {VECTOR_A} = {scaled}")
    draw_vector(axes[0], VECTOR_A, color="blue", label=f"a = {VECTOR_A}")
    draw_vector(axes[0], scaled, color="green", label=f"{SCALAR}a = {scaled}")

    # Negative scalar
    neg_scaled = -1 * VECTOR_A
    setup_axes(axes[1], f"Scaling by -1\n-1 * {VECTOR_A} = {neg_scaled}")
    draw_vector(axes[1], VECTOR_A, color="blue", label=f"a = {VECTOR_A}")
    draw_vector(axes[1], neg_scaled, color="orange", label=f"-a = {neg_scaled}")

    fig.tight_layout()
    return fig


def plot_subtraction():
    """Show vector subtraction: a - b as a + (-b)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    result = VECTOR_A - VECTOR_B
    setup_axes(ax, f"Vector Subtraction\na - b = {VECTOR_A} - {VECTOR_B} = {result}")

    # Draw both vectors from origin
    draw_vector(ax, VECTOR_A, color="blue", label="a")
    draw_vector(ax, VECTOR_B, color="red", label="b")
    # Draw the difference vector from b to a
    draw_vector(ax, result, origin=tuple(VECTOR_B), color="green", label=f"a - b = {result}")

    return fig


if __name__ == "__main__":
    print("Vectors as Arrows and Data")
    print("=" * 40)
    print(f"Vector a: {VECTOR_A}")
    print(f"Vector b: {VECTOR_B}")
    print(f"a + b = {VECTOR_A + VECTOR_B}")
    print(f"a - b = {VECTOR_A - VECTOR_B}")
    print(f"{SCALAR} * a = {SCALAR * VECTOR_A}")
    print()
    print("Close each plot window to see the next one.")

    plot_vectors_as_arrows()
    plt.show()

    plot_addition()
    plt.show()

    plot_subtraction()
    plt.show()

    plot_scaling()
    plt.show()
