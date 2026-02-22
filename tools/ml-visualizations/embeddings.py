# ABOUTME: Visualizes word embeddings as vectors in 2D space, showing similarity relationships.
# ABOUTME: Run with `python embeddings.py` to see how words cluster by meaning.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# Simulated 2D word embeddings (normally these would be hundreds of dimensions)
# These are positioned to illustrate semantic relationships
WORDS = {
    # Animals
    "cat":     np.array([2.0, 3.0]),
    "dog":     np.array([2.5, 3.2]),
    "fish":    np.array([1.5, 2.5]),
    # Royalty
    "king":    np.array([-2.0, 3.0]),
    "queen":   np.array([-1.5, 3.5]),
    "prince":  np.array([-2.2, 2.5]),
    # Actions
    "run":     np.array([3.0, -1.0]),
    "walk":    np.array([2.5, -1.2]),
    "swim":    np.array([1.8, -0.5]),
    # Objects
    "car":     np.array([-2.0, -2.0]),
    "truck":   np.array([-2.5, -1.8]),
    "bicycle": np.array([-1.5, -2.5]),
}

# Word pairs for similarity comparison
SIMILARITY_PAIRS = [
    ("cat", "dog"),
    ("cat", "car"),
    ("king", "queen"),
    ("run", "walk"),
    ("king", "truck"),
]

# Analogy: king - man + woman ≈ queen (simulated)
ANALOGY_WORDS = {
    "man":   np.array([-2.5, 2.2]),
    "woman": np.array([-2.0, 2.7]),
}


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def plot_word_embeddings():
    """Plot all words as points in 2D space, colored by category."""
    categories = {
        "Animals":  ["cat", "dog", "fish"],
        "Royalty":  ["king", "queen", "prince"],
        "Actions":  ["run", "walk", "swim"],
        "Vehicles": ["car", "truck", "bicycle"],
    }
    colors = {"Animals": "blue", "Royalty": "purple", "Actions": "green", "Vehicles": "red"}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for category, words in categories.items():
        for word in words:
            vec = WORDS[word]
            ax.scatter(*vec, c=colors[category], s=100, zorder=5)
            ax.annotate(word, xy=vec, fontsize=11, fontweight="bold",
                        color=colors[category],
                        xytext=(8, 5), textcoords="offset points")

    # Draw circles around clusters
    for category, words in categories.items():
        vecs = np.array([WORDS[w] for w in words])
        center = vecs.mean(axis=0)
        radius = np.max(np.linalg.norm(vecs - center, axis=1)) + 0.3
        circle = plt.Circle(center, radius, fill=False, color=colors[category],
                            linestyle="--", alpha=0.4, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1] - radius - 0.3, category,
                ha="center", fontsize=10, color=colors[category], style="italic")

    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    ax.set_title("Word Embeddings in 2D Space\nSimilar words cluster together", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    return fig


def plot_similarity_comparison():
    """Show cosine similarity between different word pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of similarities
    pairs = SIMILARITY_PAIRS
    similarities = [cosine_similarity(WORDS[a], WORDS[b]) for a, b in pairs]
    labels = [f"{a} / {b}" for a, b in pairs]
    colors = ["green" if s > 0.5 else "orange" if s > 0 else "red" for s in similarities]

    axes[0].barh(range(len(pairs)), similarities, color=colors)
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_title("Cosine Similarity Between Word Pairs", fontsize=13)
    axes[0].axvline(x=0, color="k", linewidth=0.5)
    axes[0].set_xlim(-1, 1)
    axes[0].grid(True, alpha=0.3, axis="x")

    for i, s in enumerate(similarities):
        axes[0].text(s + 0.03, i, f"{s:.3f}", va="center", fontsize=10)

    # Vector arrows for two pairs
    pair1, pair2 = pairs[0], pairs[1]
    for pair, color_a, color_b, label in [
        (pair1, "blue", "blue", "Similar pair"),
        (pair2, "blue", "red", "Dissimilar pair"),
    ]:
        a_word, b_word = pair
        a_vec, b_vec = WORDS[a_word], WORDS[b_word]

    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect("equal")

    # Draw arrows for a few words
    for word in ["cat", "dog", "car", "king", "queen"]:
        vec = WORDS[word]
        axes[1].annotate("", xy=vec, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", lw=1.5,
                                         color="blue" if word in ["cat", "dog"] else
                                         "red" if word == "car" else "purple"))
        axes[1].annotate(word, xy=vec, fontsize=10, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points")

    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_title("Similar words → similar directions\n(small angle = high cosine similarity)", fontsize=12)

    fig.tight_layout()
    return fig


def plot_analogy():
    """Visualize the king - man + woman ≈ queen analogy."""
    all_words = {**WORDS, **ANALOGY_WORDS}

    king = all_words["king"]
    man = all_words["man"]
    woman = all_words["woman"]
    queen = all_words["queen"]

    # The analogy vector
    result = king - man + woman

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot the relevant words
    for word in ["king", "queen", "man", "woman"]:
        vec = all_words[word]
        ax.scatter(*vec, s=150, zorder=5)
        ax.annotate(word, xy=vec, fontsize=13, fontweight="bold",
                    xytext=(10, 10), textcoords="offset points")

    # Draw the analogy arrows
    # king → man (the "male" direction)
    ax.annotate("", xy=man, xytext=king,
                arrowprops=dict(arrowstyle="->", color="blue", lw=2, linestyle="--"))
    ax.text((king[0] + man[0]) / 2 - 0.5, (king[1] + man[1]) / 2,
            "−man", fontsize=10, color="blue")

    # man → woman (gender shift)
    ax.annotate("", xy=woman, xytext=man,
                arrowprops=dict(arrowstyle="->", color="green", lw=2, linestyle="--"))
    ax.text((man[0] + woman[0]) / 2 + 0.2, (man[1] + woman[1]) / 2 - 0.3,
            "+woman", fontsize=10, color="green")

    # Result point
    ax.scatter(*result, s=150, c="red", marker="*", zorder=5)
    ax.annotate(f"king−man+woman\n≈ queen!", xy=result, fontsize=11,
                color="red", fontweight="bold",
                xytext=(15, -20), textcoords="offset points")

    ax.axhline(y=0, color="k", linewidth=0.3)
    ax.axvline(x=0, color="k", linewidth=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    ax.set_title("Word Analogy: king − man + woman ≈ queen\n"
                 "Embeddings capture semantic relationships as directions",
                 fontsize=13)

    return fig


if __name__ == "__main__":
    print("Word Embeddings Visualization")
    print("=" * 45)
    print()
    print("Cosine similarities between word pairs:")
    for a, b in SIMILARITY_PAIRS:
        sim = cosine_similarity(WORDS[a], WORDS[b])
        print(f"  {a:8s} / {b:8s} = {sim:.4f}")

    print()
    print("Close each plot window to see the next one.")

    plot_word_embeddings()
    plt.show()

    plot_similarity_comparison()
    plt.show()

    plot_analogy()
    plt.show()
