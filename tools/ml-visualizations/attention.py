# ABOUTME: Visualizes the attention mechanism as a heatmap between words in a sentence.
# ABOUTME: Run with `python attention.py` to see how attention weights connect words.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Change these to experiment!
# ============================================================

# The sentence to visualize attention for
SENTENCE = ["The", "cat", "sat", "on", "the", "mat"]

# Embedding dimension (normally 512+ in real transformers, we use small for visibility)
D_MODEL = 8

# Random seed for reproducible embeddings
SEED = 42


def make_embeddings(words, d_model, seed=SEED):
    """Create random embeddings for each word. Real models learn these."""
    np.random.seed(seed)
    # In a real model, similar words would have similar embeddings
    # We simulate this by making some words more related
    embeddings = np.random.randn(len(words), d_model) * 0.5
    return embeddings


def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Q: queries  (seq_len, d_k)
    K: keys     (seq_len, d_k)
    V: values   (seq_len, d_v)

    Returns:
        output: weighted values  (seq_len, d_v)
        weights: attention weights (seq_len, seq_len)
    """
    d_k = K.shape[-1]

    # Step 1: Compute similarity scores (dot product of queries and keys)
    scores = Q @ K.T  # (seq_len, seq_len)

    # Step 2: Scale by sqrt(d_k) to prevent large values
    scores = scores / np.sqrt(d_k)

    # Step 3: Softmax to convert scores to probabilities
    # Subtract max for numerical stability
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Weighted sum of values
    output = weights @ V  # (seq_len, d_v)

    return output, weights


def plot_attention_heatmap(weights, words, title="Attention Weights"):
    """Plot attention weights as a heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(weights, cmap="Blues", aspect="auto")

    # Label axes with words
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=12, rotation=45, ha="right")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=12)

    ax.set_xlabel("Key (attending TO)", fontsize=12)
    ax.set_ylabel("Query (attending FROM)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add text annotations showing the actual values
    for i in range(len(words)):
        for j in range(len(words)):
            color = "white" if weights[i, j] > 0.5 else "black"
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, label="Attention Weight")
    fig.tight_layout()
    return fig


def plot_attention_step_by_step():
    """Show each step of the attention computation."""
    embeddings = make_embeddings(SENTENCE, D_MODEL)

    # In a real transformer, Q, K, V come from learned linear projections
    # Here we use the embeddings directly for simplicity
    Q = embeddings
    K = embeddings
    V = embeddings

    d_k = K.shape[-1]
    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax
    scores_shifted = scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Step 1: Raw scores
    im0 = axes[0].imshow(scores, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(SENTENCE)))
    axes[0].set_xticklabels(SENTENCE, rotation=45, ha="right")
    axes[0].set_yticks(range(len(SENTENCE)))
    axes[0].set_yticklabels(SENTENCE)
    axes[0].set_title("Step 1: Q × Kᵀ\n(raw similarity scores)", fontsize=12)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Step 2: Scaled scores
    im1 = axes[1].imshow(scaled_scores, cmap="RdBu_r", aspect="auto")
    axes[1].set_xticks(range(len(SENTENCE)))
    axes[1].set_xticklabels(SENTENCE, rotation=45, ha="right")
    axes[1].set_yticks(range(len(SENTENCE)))
    axes[1].set_yticklabels(SENTENCE)
    axes[1].set_title(f"Step 2: Scale by √d_k = √{d_k}\n(prevents extreme softmax)", fontsize=12)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Step 3: Attention weights (after softmax)
    im2 = axes[2].imshow(weights, cmap="Blues", aspect="auto")
    axes[2].set_xticks(range(len(SENTENCE)))
    axes[2].set_xticklabels(SENTENCE, rotation=45, ha="right")
    axes[2].set_yticks(range(len(SENTENCE)))
    axes[2].set_yticklabels(SENTENCE)
    axes[2].set_title("Step 3: Softmax\n(rows sum to 1.0)", fontsize=12)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        for i in range(len(SENTENCE)):
            for j in range(len(SENTENCE)):
                data = [scores, scaled_scores, weights][list(axes).index(ax)]
                val = data[i, j]
                color = "white" if abs(val) > 0.5 * np.max(np.abs(data)) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.suptitle("Scaled Dot-Product Attention: Step by Step", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_what_attention_does():
    """Illustrate what attention accomplishes: contextual mixing of information."""
    embeddings = make_embeddings(SENTENCE, D_MODEL)
    output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Before attention: each word is its own independent embedding
    im0 = axes[0].imshow(embeddings, cmap="viridis", aspect="auto")
    axes[0].set_yticks(range(len(SENTENCE)))
    axes[0].set_yticklabels(SENTENCE, fontsize=11)
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_title("Before Attention\nEach word = independent vector", fontsize=12)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # After attention: each word is a weighted mix of all words
    im1 = axes[1].imshow(output, cmap="viridis", aspect="auto")
    axes[1].set_yticks(range(len(SENTENCE)))
    axes[1].set_yticklabels(SENTENCE, fontsize=11)
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_title("After Attention\nEach word = context-aware mix", fontsize=12)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Attention Mixes Information Between Words\n"
                 "Each output vector is a weighted combination of all input vectors",
                 fontsize=14, y=1.05)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Attention Mechanism Visualization")
    print("=" * 45)
    print(f"Sentence: {' '.join(SENTENCE)}")
    print(f"Embedding dimension: {D_MODEL}")
    print()

    embeddings = make_embeddings(SENTENCE, D_MODEL)
    output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)

    print("Attention weights (each row sums to 1.0):")
    print(f"{'':8s}", end="")
    for w in SENTENCE:
        print(f"{w:8s}", end="")
    print()
    for i, word in enumerate(SENTENCE):
        print(f"{word:8s}", end="")
        for j in range(len(SENTENCE)):
            print(f"{weights[i, j]:8.3f}", end="")
        print()

    print()
    print("Close each plot window to see the next one.")

    plot_attention_heatmap(weights, SENTENCE,
                          title=f"Self-Attention: \"{' '.join(SENTENCE)}\"")
    plt.show()

    plot_attention_step_by_step()
    plt.show()

    plot_what_attention_does()
    plt.show()
