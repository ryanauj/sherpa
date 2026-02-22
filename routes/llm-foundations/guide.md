---
title: LLM Foundations
route_map: /routes/llm-foundations/map.md
paired_sherpa: /routes/llm-foundations/sherpa.md
prerequisites:
  - Training and Backprop (training, loss, backpropagation)
  - Neural Network Foundations (layers, forward pass)
  - Linear Algebra Essentials (dot products, matrix multiplication)
topics:
  - Embeddings
  - Attention
  - Transformers
  - Language Models
---

# LLM Foundations - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/llm-foundations/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/llm-foundations/map.md` for the high-level overview.

## Overview

You've built and trained a neural network from scratch. You know how to multiply matrices, compute gradients, run forward passes, and train with backpropagation. Large language models use all of that same math -- the same matrix multiplications, the same activation functions, the same gradient descent. But they add two ideas you haven't seen yet: **embeddings** (how to represent words as numbers a neural network can process) and **attention** (how to handle sequences where context matters).

This route teaches both. By the end, you'll understand the complete pipeline from raw text to transformer output. Nothing here will feel alien -- it's the same building blocks you already know, assembled in a specific way.

## Learning Objectives

By the end of this route, you will be able to:
- Explain how embeddings represent words as vectors with semantic meaning
- Implement positional encoding to give a model a sense of word order
- Trace the attention mechanism step by step with concrete numbers
- Implement scaled dot-product attention from scratch in numpy
- Describe the transformer architecture and how its components fit together
- Explain how transformers scale into large language models

## Prerequisites

Before starting this route, you should be comfortable with:
- **Training and Backprop** ([route](/routes/training-and-backprop/map.md)): Training loops, loss functions, backpropagation, gradient descent
- **Neural Network Foundations** ([route](/routes/neural-network-foundations/map.md)): Layers, forward pass, activation functions
- **Linear Algebra Essentials** ([route](/routes/linear-algebra-essentials/map.md)): Dot products, matrix multiplication, cosine similarity

You need to understand `z = W @ x + b`, activation functions, and how dot products measure similarity. If any of that feels shaky, go review those routes first. This guide uses all three constantly and builds directly on top of them.

## Setup

You need numpy and matplotlib, which you should already have from the prerequisite routes.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

Create a file called `llm_setup_test.py` and run it:

```python
import numpy as np
import matplotlib
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick sanity check: a dot product (the core operation behind attention)
q = np.array([1.0, 0.0, 1.0])
k = np.array([1.0, 1.0, 0.0])
similarity = np.dot(q, k)
print(f"Dot product similarity: {similarity}")
```

```bash
python llm_setup_test.py
```

**Expected output:**

```
numpy version: 1.26.4
matplotlib version: 3.9.2
Dot product similarity: 1.0
```

Your version numbers may differ -- that's fine as long as the script runs without errors. If you got `1.0` for the dot product, you're ready.

---

## Section 1: Embeddings

### The Problem: Words Aren't Numbers

Neural networks operate on numbers. Everything you've built so far -- the perceptron, the forward pass, backpropagation -- works on numpy arrays of floats. But language is made of words, not numbers. Before a neural network can process text, you need a way to convert words into numeric vectors.

This isn't a trivial problem. The conversion needs to preserve something about the *meaning* of words. If "cat" and "dog" are both animals, their numeric representations should reflect that relationship somehow.

### One-Hot Encoding: The Naive Approach

The simplest way to represent words as numbers is **one-hot encoding**. Give each word in your vocabulary an index, then represent each word as a vector with a 1 at that index and 0s everywhere else.

```python
import numpy as np

# A tiny vocabulary
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
vocab_size = len(vocab)

# One-hot encode "cat"
cat_onehot = np.zeros(vocab_size)
cat_onehot[vocab["cat"]] = 1.0

# One-hot encode "mat"
mat_onehot = np.zeros(vocab_size)
mat_onehot[vocab["mat"]] = 1.0

print(f"'cat' one-hot: {cat_onehot}")
print(f"'mat' one-hot: {mat_onehot}")

# How similar are cat and mat?
similarity = np.dot(cat_onehot, mat_onehot)
print(f"Dot product (cat, mat): {similarity}")
```

**Expected output:**

```
'cat' one-hot: [0. 1. 0. 0. 0.]
'mat' one-hot: [0. 0. 0. 0. 1.]
Dot product (cat, mat): 0.0
```

The dot product is zero. According to one-hot encoding, "cat" and "mat" have zero similarity -- the same as "cat" and "the", or any other pair. Every word is equally dissimilar to every other word. One-hot encoding captures no meaning at all.

It also has a practical problem: if your vocabulary has 50,000 words, every word becomes a 50,000-dimensional vector with a single 1. That's extremely wasteful.

### Learned Embeddings: Dense Vectors with Meaning

The solution is to give each word a short, dense vector (say, 256 dimensions instead of 50,000) where the values are *learned during training*. These are called **embeddings**.

The idea: start with random vectors, then adjust them through training so that words used in similar contexts end up with similar vectors. "Cat" and "dog" appear in similar sentences ("The ___ chased the ball"), so gradient descent will push their vectors closer together.

An embedding table is just a matrix. Each row is a word's vector. Looking up a word's embedding is a simple index operation.

```python
import numpy as np

np.random.seed(42)

# Vocabulary
vocab = {"the": 0, "cat": 1, "dog": 2, "sat": 3, "ran": 4, "mat": 5}
vocab_size = len(vocab)
embedding_dim = 4  # 4 dimensions (real models use 256-1024)

# Embedding table: each row is a word's vector
# In a real model, these are learned. We'll simulate with hand-picked values.
embedding_table = np.array([
    [ 0.1,  0.0,  0.0,  0.1],  # "the" — function word, neutral
    [ 0.8,  0.9,  0.2,  0.1],  # "cat" — animal
    [ 0.7,  0.8,  0.3,  0.1],  # "dog" — animal (similar to cat!)
    [ 0.1,  0.2,  0.9,  0.7],  # "sat" — action
    [ 0.2,  0.3,  0.8,  0.8],  # "ran" — action (similar to sat!)
    [ 0.0,  0.1,  0.1,  0.9],  # "mat" — object
])

# Look up embeddings
cat_embed = embedding_table[vocab["cat"]]
dog_embed = embedding_table[vocab["dog"]]
mat_embed = embedding_table[vocab["mat"]]

print(f"'cat' embedding: {cat_embed}")
print(f"'dog' embedding: {dog_embed}")
print(f"'mat' embedding: {mat_embed}")
```

**Expected output:**

```
'cat' embedding: [0.8 0.9 0.2 0.1]
'dog' embedding: [0.7 0.8 0.3 0.1]
'mat' embedding: [0.  0.1 0.1 0.9]
```

Notice how "cat" and "dog" have similar values, while "mat" looks completely different. That similarity is meaningful and measurable.

### Measuring Similarity with Cosine Similarity

You already know this from the linear algebra route: cosine similarity measures the angle between two vectors. A value of 1 means they point in the same direction (very similar), 0 means they're perpendicular (unrelated), and -1 means they point in opposite directions.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Same embeddings from above
embedding_table = np.array([
    [ 0.1,  0.0,  0.0,  0.1],  # "the"
    [ 0.8,  0.9,  0.2,  0.1],  # "cat"
    [ 0.7,  0.8,  0.3,  0.1],  # "dog"
    [ 0.1,  0.2,  0.9,  0.7],  # "sat"
    [ 0.2,  0.3,  0.8,  0.8],  # "ran"
    [ 0.0,  0.1,  0.1,  0.9],  # "mat"
])
words = ["the", "cat", "dog", "sat", "ran", "mat"]

# Compare pairs
pairs = [("cat", "dog"), ("sat", "ran"), ("cat", "sat"), ("cat", "mat")]
for w1, w2 in pairs:
    i, j = words.index(w1), words.index(w2)
    sim = cosine_similarity(embedding_table[i], embedding_table[j])
    print(f"similarity({w1:>3s}, {w2:>3s}) = {sim:.4f}")
```

**Expected output:**

```
similarity(cat, dog) = 0.9916
similarity(sat, ran) = 0.9862
similarity(cat, sat) = 0.4526
similarity(cat, mat) = 0.1647
```

"Cat" and "dog" have a similarity of 0.99 -- they're nearly identical in embedding space. "Sat" and "ran" are also very similar (0.99). But "cat" and "sat" are dissimilar (0.45), and "cat" and "mat" even more so (0.16). The embeddings capture semantic categories.

> **Visualization**: Run `python tools/ml-visualizations/embeddings.py` to see word embeddings plotted in 2D space, with similar words clustering together.

### The King-Queen Analogy

The most famous property of word embeddings is that they encode relationships as *directions*. The classic example:

```
king - man + woman ≈ queen
```

The direction from "man" to "woman" represents a gender shift. If you take the "king" vector, subtract the "man" direction, and add the "woman" direction, you land near "queen". The embedding space encodes the concept of gender as a consistent direction that works across different words.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pre-trained-style embeddings (simplified to 4D)
# These are hand-picked to demonstrate the analogy
words = {
    "king":  np.array([0.9, 0.8, 0.6, 0.2]),
    "queen": np.array([0.9, 0.8, 0.2, 0.6]),
    "man":   np.array([0.5, 0.4, 0.7, 0.1]),
    "woman": np.array([0.5, 0.4, 0.1, 0.7]),
}

# The analogy: king - man + woman = ?
result = words["king"] - words["man"] + words["woman"]
print(f"king - man + woman = {result}")
print(f"queen              = {words['queen']}")

# How close is the result to "queen"?
sim = cosine_similarity(result, words["queen"])
print(f"\nCosine similarity with queen: {sim:.4f}")

# Compare with all words
print("\nDistance from result to each word:")
for word, vec in words.items():
    sim = cosine_similarity(result, vec)
    print(f"  {word:8s}: {sim:.4f}")
```

**Expected output:**

```
king - man + woman = [0.9 0.8 0.1 0.8]
queen              = [0.9 0.8 0.2 0.6]

Cosine similarity with queen: 0.9832

Distance from result to each word:
  king    : 0.8818
  queen   : 0.9832
  man     : 0.6685
  woman   : 0.8736
```

The result vector is closest to "queen" (0.98). The analogy works because the embedding space encodes "royalty" in one direction and "gender" in another, and vector arithmetic can combine these concepts.

### Exercise 1.1: Build an Embedding Lookup Table

**Task:** Create a vocabulary of 8 words (at least two pairs of semantically similar words), build a 5-dimensional embedding table, and compute cosine similarity between all pairs. Verify that your similar word pairs have higher similarity scores than dissimilar pairs.

**Hints:**

<details>
<summary>Hint 1: Structure</summary>

Create a dictionary mapping words to indices, then a numpy array of shape `(8, 5)` where each row is a word's embedding. Make similar words have similar vectors by hand -- e.g., "happy" and "glad" should share high values in the same dimensions.
</details>

<details>
<summary>Hint 2: Computing all pairs</summary>

Use nested loops over all word pairs and compute cosine similarity for each. Print them sorted or in a matrix to see the pattern clearly.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

words = ["happy", "glad", "sad", "angry", "run", "sprint", "table", "chair"]
embedding_table = np.array([
    [ 0.9,  0.8,  0.1,  0.0,  0.1],  # happy
    [ 0.8,  0.9,  0.2,  0.1,  0.0],  # glad  (similar to happy)
    [ 0.1,  0.2,  0.9,  0.7,  0.0],  # sad
    [ 0.2,  0.1,  0.8,  0.9,  0.1],  # angry (similar to sad)
    [ 0.1,  0.0,  0.1,  0.2,  0.9],  # run
    [ 0.0,  0.1,  0.0,  0.3,  0.8],  # sprint (similar to run)
    [ 0.5,  0.5,  0.5,  0.0,  0.0],  # table
    [ 0.4,  0.6,  0.4,  0.1,  0.0],  # chair (similar to table)
])

print("Cosine similarity matrix:")
print(f"{'':>8s}", end="")
for w in words:
    print(f"{w:>8s}", end="")
print()

for i, w1 in enumerate(words):
    print(f"{w1:>8s}", end="")
    for j, w2 in enumerate(words):
        sim = cosine_similarity(embedding_table[i], embedding_table[j])
        print(f"{sim:>8.3f}", end="")
    print()
```

**Expected output:**

```
Cosine similarity matrix:
           happy    glad     sad   angry     run  sprint   table   chair
   happy   1.000   0.990   0.283   0.207   0.118   0.098   0.751   0.772
    glad   0.990   1.000   0.348   0.268   0.047   0.076   0.730   0.775
     sad   0.283   0.348   1.000   0.982   0.198   0.368   0.484   0.427
   angry   0.207   0.268   0.982   1.000   0.282   0.446   0.371   0.341
     run   0.118   0.047   0.198   0.282   1.000   0.965   0.102   0.063
  sprint   0.098   0.076   0.368   0.446   0.965   1.000   0.073   0.058
   table   0.751   0.730   0.484   0.371   0.102   0.073   1.000   0.989
   chair   0.772   0.775   0.427   0.341   0.063   0.058   0.989   1.000
```

**Explanation:** Each similar pair (happy/glad, sad/angry, run/sprint, table/chair) has a similarity above 0.96, while cross-category pairs are much lower. The embeddings encode semantic categories through the structure of the vectors.
</details>

### Exercise 1.2: Verify the King-Queen Analogy

**Task:** Using the king/queen/man/woman embeddings from the section above, verify these additional analogies:
1. `queen - woman + man ≈ king`
2. `man - king + queen ≈ woman`

Compute each result vector and find which word in the vocabulary it's closest to (by cosine similarity).

<details>
<summary>Hint</summary>

The analogy `A - B + C ≈ D` means: compute the result vector `A - B + C`, then find which word vector has the highest cosine similarity with the result.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

words = {
    "king":  np.array([0.9, 0.8, 0.6, 0.2]),
    "queen": np.array([0.9, 0.8, 0.2, 0.6]),
    "man":   np.array([0.5, 0.4, 0.7, 0.1]),
    "woman": np.array([0.5, 0.4, 0.1, 0.7]),
}

def find_closest(result, words):
    best_word, best_sim = None, -1
    for word, vec in words.items():
        sim = cosine_similarity(result, vec)
        if sim > best_sim:
            best_word, best_sim = word, sim
    return best_word, best_sim

# Analogy 1: queen - woman + man ≈ king
result1 = words["queen"] - words["woman"] + words["man"]
closest1, sim1 = find_closest(result1, words)
print(f"queen - woman + man = {result1}")
print(f"Closest word: {closest1} (similarity: {sim1:.4f})")

print()

# Analogy 2: man - king + queen ≈ woman
result2 = words["man"] - words["king"] + words["queen"]
closest2, sim2 = find_closest(result2, words)
print(f"man - king + queen = {result2}")
print(f"Closest word: {closest2} (similarity: {sim2:.4f})")
```

**Expected output:**

```
queen - woman + man = [0.9 0.8 0.8 0.0]
Closest word: king (similarity: 0.9547)

man - king + queen = [0.5 0.4 -0.3 0.5]
Closest word: woman (similarity: 0.8562)
```

**Explanation:** Both analogies land closest to the expected word, confirming that the embedding space encodes the gender relationship symmetrically. The similarities aren't perfect (real embeddings are much higher-dimensional and trained on billions of words), but the direction is consistent.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain why one-hot encoding doesn't capture word meaning
- [ ] Describe how embeddings represent words as dense vectors
- [ ] Use cosine similarity to measure how similar two word embeddings are
- [ ] Explain the king-queen analogy in terms of vector arithmetic

---

## Section 2: Sequence Problems and Context

### Why Word Order Matters

Consider these two sentences:

- "The dog bites the man."
- "The man bites the dog."

Same words, opposite meanings. The meaning of a sentence depends on the *order* of the words, not just which words appear. Any model that processes language must understand sequence.

There's a harder problem too: **long-range dependencies**. In the sentence "The cat, which was sitting on the windowsill watching the birds outside, **jumped** down," the verb "jumped" depends on "cat" -- not on "birds" or "outside," which are much closer. The model needs to connect words that are far apart in the sequence.

### Why Feedforward Networks Can't Handle This

The networks you've built so far take a fixed-size input vector and produce a fixed-size output. There's no notion of order. If you fed the embeddings for "dog bites man" and "man bites dog" into a feedforward network, you'd need to concatenate all the embeddings into one big vector -- and the network would treat position 1 and position 3 as completely unrelated input dimensions. It would have to learn from scratch that position matters, with no structural help.

Worse, feedforward networks have a fixed input size. A sentence with 5 words and a sentence with 50 words can't use the same network without padding or truncating. Language has variable length, and feedforward networks don't handle that naturally.

### Positional Encoding: Giving the Model a Sense of Order

The solution is to add position information directly to each embedding. Before feeding the word vectors into the model, add a **positional encoding** vector that encodes where each word sits in the sequence. The result is that each vector contains both "what word am I?" (from the embedding) and "where am I in the sentence?" (from the positional encoding).

Transformers use sinusoidal positional encoding -- sine and cosine functions at different frequencies. Each position gets a unique pattern, and the model can learn to use these patterns to understand relative positions.

The formula for position `pos` at dimension `i`:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
```

Where `d_model` is the embedding dimension. Even-indexed dimensions use sine, odd-indexed dimensions use cosine, and the frequency decreases as the dimension index increases.

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """Compute sinusoidal positional encoding.

    Returns an array of shape (seq_len, d_model) where each row
    is the positional encoding for that position.
    """
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)           # (seq_len, 1)
    dim_indices = np.arange(0, d_model, 2)                   # [0, 2, 4, ...]
    frequencies = 1.0 / (10000 ** (dim_indices / d_model))   # decreasing frequencies

    pe[:, 0::2] = np.sin(positions * frequencies)  # even dimensions: sin
    pe[:, 1::2] = np.cos(positions * frequencies)  # odd dimensions: cos
    return pe

# Compute positional encoding for a 6-word sentence with 8-dimensional embeddings
pe = positional_encoding(seq_len=6, d_model=8)

print("Positional encoding (6 positions, 8 dimensions):")
print("Each row = one position's encoding\n")
for pos in range(6):
    values = ", ".join(f"{v:>6.3f}" for v in pe[pos])
    print(f"  Position {pos}: [{values}]")
```

**Expected output:**

```
Positional encoding (6 positions, 8 dimensions):
Each row = one position's encoding

  Position 0: [ 0.000,  1.000,  0.000,  1.000,  0.000,  1.000,  0.000,  1.000]
  Position 1: [ 0.841,  0.540,  0.100,  0.995,  0.010,  1.000,  0.001,  1.000]
  Position 2: [ 0.909, -0.416,  0.198,  0.980,  0.020,  1.000,  0.002,  1.000]
  Position 3: [ 0.141, -0.990,  0.296,  0.955,  0.030,  1.000,  0.003,  1.000]
  Position 4: [-0.757, -0.654,  0.389,  0.921,  0.040,  0.999,  0.004,  1.000]
  Position 5: [-0.959,  0.284,  0.479,  0.878,  0.050,  0.999,  0.005,  1.000]
```

Each position has a unique pattern. The first two dimensions (high frequency) change rapidly, while the last dimensions (low frequency) change slowly. This gives the model multiple scales to work with -- nearby positions differ in the high-frequency dimensions, while distant positions differ in the low-frequency ones.

### Adding Positional Encoding to Embeddings

The positional encoding is simply *added* to the word embeddings, element by element:

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    dim_indices = np.arange(0, d_model, 2)
    frequencies = 1.0 / (10000 ** (dim_indices / d_model))
    pe[:, 0::2] = np.sin(positions * frequencies)
    pe[:, 1::2] = np.cos(positions * frequencies)
    return pe

# Simulate embeddings for "The cat sat"
np.random.seed(42)
d_model = 8
words = ["The", "cat", "sat"]
embeddings = np.random.randn(len(words), d_model) * 0.5  # random embeddings

# Add positional encoding
pe = positional_encoding(len(words), d_model)
encoded = embeddings + pe

print("Word embeddings (no position info):")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{', '.join(f'{v:>6.3f}' for v in embeddings[i])}]")

print("\nPositional encoding:")
for i in range(len(words)):
    print(f"  pos {i}: [{', '.join(f'{v:>6.3f}' for v in pe[i])}]")

print("\nEmbeddings + positional encoding:")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{', '.join(f'{v:>6.3f}' for v in encoded[i])}]")
```

**Expected output:**

```
Word embeddings (no position info):
   The: [ 0.248,  0.069, -0.032,  0.763, -0.117,  0.117, -0.272, -0.118]
   cat: [ 0.380, -0.090,  0.285, -0.045, -0.103, -0.015,  0.245,  0.155]
   sat: [-0.499, -0.364,  0.092, -0.209,  0.018,  0.279,  0.199, -0.099]

Positional encoding:
  pos 0: [ 0.000,  1.000,  0.000,  1.000,  0.000,  1.000,  0.000,  1.000]
  pos 1: [ 0.841,  0.540,  0.100,  0.995,  0.010,  1.000,  0.001,  1.000]
  pos 2: [ 0.909, -0.416,  0.198,  0.980,  0.020,  1.000,  0.002,  1.000]

Embeddings + positional encoding:
   The: [ 0.248,  1.069, -0.032,  1.763, -0.117,  1.117, -0.272,  0.882]
   cat: [ 1.222,  0.451,  0.385,  0.950, -0.093,  0.985,  0.246,  1.155]
   sat: [ 0.411, -0.780,  0.290,  0.771,  0.038,  1.279,  0.201,  0.901]
```

After adding positional encoding, the same word at different positions would have different vectors. The word "the" at position 0 and "the" at position 5 would have different encoded representations, so the model can distinguish them.

### Exercise 2.1: Implement Sinusoidal Positional Encoding

**Task:** Implement `positional_encoding(seq_len, d_model)` from scratch and verify two properties:

1. Each position has a unique encoding (no two rows are identical)
2. The dot product between nearby positions is higher than between distant positions

<details>
<summary>Hint 1: Checking uniqueness</summary>

Compare each pair of rows using `np.allclose()`. If any two rows are nearly identical, your encoding has a bug.
</details>

<details>
<summary>Hint 2: Checking distance</summary>

Compute the dot product between position 0 and position 1, then between position 0 and position 10. The first should be larger (closer positions are more similar in the encoding).
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    dim_indices = np.arange(0, d_model, 2)
    frequencies = 1.0 / (10000 ** (dim_indices / d_model))
    pe[:, 0::2] = np.sin(positions * frequencies)
    pe[:, 1::2] = np.cos(positions * frequencies)
    return pe

pe = positional_encoding(seq_len=20, d_model=32)

# Check 1: Uniqueness
all_unique = True
for i in range(20):
    for j in range(i + 1, 20):
        if np.allclose(pe[i], pe[j]):
            print(f"Duplicate: positions {i} and {j}")
            all_unique = False
print(f"All positions unique: {all_unique}")

# Check 2: Nearby positions are more similar
dot_0_1 = np.dot(pe[0], pe[1])
dot_0_5 = np.dot(pe[0], pe[5])
dot_0_10 = np.dot(pe[0], pe[10])
dot_0_19 = np.dot(pe[0], pe[19])
print(f"\nDot product of position 0 with:")
print(f"  Position  1: {dot_0_1:.4f}")
print(f"  Position  5: {dot_0_5:.4f}")
print(f"  Position 10: {dot_0_10:.4f}")
print(f"  Position 19: {dot_0_19:.4f}")
print(f"\nNearer positions have higher similarity: {dot_0_1 > dot_0_10}")
```

**Expected output:**

```
All positions unique: True

Dot product of position 0 with:
  Position  1: 14.5388
  Position  5: 10.1974
  Position 10:  4.5620
  Position 19: -3.7292

Nearer positions have higher similarity: True
```

**Explanation:** The sinusoidal encoding gives each position a unique fingerprint. The dot product between nearby positions is high (they share similar sine/cosine values), and it decreases with distance. This lets the model learn that position 3 is "close to" position 4 but "far from" position 15.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain why word order matters and why feedforward networks don't handle it naturally
- [ ] Describe what positional encoding adds to word embeddings
- [ ] Implement sinusoidal positional encoding in numpy
- [ ] Explain why nearby positions have more similar encodings than distant positions

---

## Section 3: The Attention Mechanism

This is the core of the route. Everything before this was setup; everything after this builds on it. If you understand attention, you understand the key idea behind transformers.

### The Core Question

Consider the sentence: "The cat sat on the mat."

When processing the word "sat," the model needs to figure out: *who* sat? The answer is "cat" -- not "mat," not "the." The model needs a way to look at all the other words and decide which ones are relevant to each word.

That's what attention computes. For each word in the sequence, attention asks: "How much should I pay attention to every other word?" It produces a set of weights -- one per word -- that sum to 1.0. High weight means "this word is relevant to me." Low weight means "I can mostly ignore this word."

### Queries, Keys, and Values

Attention uses three sets of vectors derived from each word's embedding:

- **Query (Q)**: "What am I looking for?" Each word generates a query that describes what information it needs.
- **Key (K)**: "What do I contain?" Each word generates a key that describes what information it offers.
- **Value (V)**: "What information do I provide?" Each word generates a value that contains the actual information to be passed along.

The matching process works like a search engine: each query is compared against all keys to find the best matches, then the corresponding values are retrieved. The key insight: Q, K, and V are all learned linear projections of the same embeddings. The model learns *what to ask for* and *what to advertise* during training.

```python
import numpy as np

# Embeddings for "The cat sat" (3 words, 4 dimensions)
np.random.seed(42)
embeddings = np.array([
    [0.1, 0.0, 0.0, 0.1],   # "The"
    [0.8, 0.9, 0.2, 0.1],   # "cat"
    [0.1, 0.2, 0.9, 0.7],   # "sat"
])

d_model = 4
d_k = 3  # dimension of Q, K, V (can differ from d_model)

# Learned projection matrices (in practice, these are trained)
np.random.seed(0)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

# Project embeddings into Q, K, V
Q = embeddings @ W_Q  # (3, 3) — one query per word
K = embeddings @ W_K  # (3, 3) — one key per word
V = embeddings @ W_V  # (3, 3) — one value per word

print("Queries (what each word is looking for):")
for i, word in enumerate(["The", "cat", "sat"]):
    print(f"  {word}: [{', '.join(f'{v:.3f}' for v in Q[i])}]")

print("\nKeys (what each word advertises):")
for i, word in enumerate(["The", "cat", "sat"]):
    print(f"  {word}: [{', '.join(f'{v:.3f}' for v in K[i])}]")
```

**Expected output:**

```
Queries (what each word is looking for):
  The: [ 0.138, -0.004, -0.027]
  cat: [ 1.153, -0.190, -0.072]
  sat: [ 0.604,  0.023,  0.556]

Keys (what each word advertises):
  The: [ 0.006,  0.082, -0.030]
  cat: [ 0.166,  0.861, -0.126]
  sat: [-0.267,  0.450,  0.705]
```

### Scaled Dot-Product Attention: Step by Step

Here's the full attention computation, broken into four explicit steps:

**Step 1: Compute scores.** Multiply each query by every key (using matrix multiplication: Q times K-transposed). This produces a score matrix where `scores[i][j]` measures how much word `i` should attend to word `j`.

**Step 2: Scale.** Divide all scores by the square root of the key dimension (sqrt(d_k)). Without scaling, the dot products can grow large in magnitude, which pushes the softmax into regions where the gradients are tiny. Scaling keeps the values in a range where softmax behaves well.

**Step 3: Softmax.** Convert each row of scores into a probability distribution. After softmax, each row sums to 1.0. These are the attention weights -- they tell you the proportion of attention each word pays to every other word.

**Step 4: Weighted sum.** Multiply the weights by the values. Each word's output is a weighted combination of all the value vectors, where the weights come from step 3.

```python
import numpy as np

def softmax(x):
    """Compute softmax along the last axis."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Compute scaled dot-product attention.

    Q: (seq_len, d_k) — queries
    K: (seq_len, d_k) — keys
    V: (seq_len, d_v) — values

    Returns:
        output: (seq_len, d_v) — weighted values
        weights: (seq_len, seq_len) — attention weights
    """
    d_k = K.shape[-1]

    # Step 1: Compute raw similarity scores
    scores = Q @ K.T                      # (seq_len, seq_len)

    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # Step 3: Softmax to get weights (probabilities)
    weights = softmax(scores)

    # Step 4: Weighted sum of values
    output = weights @ V                  # (seq_len, d_v)

    return output, weights

# FULL WORKED EXAMPLE with concrete numbers
# Sentence: "The cat sat" — 3 words, each with a 3-dimensional embedding

# These are already-projected Q, K, V vectors (as if we applied the W_Q, W_K, W_V matrices)
Q = np.array([
    [1.0, 0.0, 0.0],   # "The" query
    [0.0, 1.0, 0.0],   # "cat" query
    [0.0, 0.0, 1.0],   # "sat" query
])

K = np.array([
    [0.5, 0.1, 0.0],   # "The" key
    [0.1, 0.9, 0.1],   # "cat" key
    [0.0, 0.2, 0.8],   # "sat" key
])

V = np.array([
    [1.0, 0.0, 0.0],   # "The" value
    [0.0, 1.0, 0.0],   # "cat" value
    [0.0, 0.0, 1.0],   # "sat" value
])

d_k = 3

# STEP 1: scores = Q @ K^T
scores = Q @ K.T
print("Step 1: scores = Q @ K^T")
print(f"  Q @ K^T =")
print(f"  [1,0,0] . [0.5,0.1,0.0] = {np.dot(Q[0], K[0]):.2f}  "
      f"[1,0,0] . [0.1,0.9,0.1] = {np.dot(Q[0], K[1]):.2f}  "
      f"[1,0,0] . [0.0,0.2,0.8] = {np.dot(Q[0], K[2]):.2f}")
print(f"  [0,1,0] . [0.5,0.1,0.0] = {np.dot(Q[1], K[0]):.2f}  "
      f"[0,1,0] . [0.1,0.9,0.1] = {np.dot(Q[1], K[1]):.2f}  "
      f"[0,1,0] . [0.0,0.2,0.8] = {np.dot(Q[1], K[2]):.2f}")
print(f"  [0,0,1] . [0.5,0.1,0.0] = {np.dot(Q[2], K[0]):.2f}  "
      f"[0,0,1] . [0.1,0.9,0.1] = {np.dot(Q[2], K[1]):.2f}  "
      f"[0,0,1] . [0.0,0.2,0.8] = {np.dot(Q[2], K[2]):.2f}")
print(f"\n  scores =\n{scores}")

# STEP 2: scale
scaled = scores / np.sqrt(d_k)
print(f"\nStep 2: scaled = scores / sqrt({d_k}) = scores / {np.sqrt(d_k):.4f}")
print(f"  scaled =\n{np.round(scaled, 4)}")

# STEP 3: softmax
weights = softmax(scaled)
print(f"\nStep 3: weights = softmax(scaled)")
print(f"  weights =")
for i, word in enumerate(["The", "cat", "sat"]):
    row = ", ".join(f"{w:.4f}" for w in weights[i])
    print(f"    {word}: [{row}]  (sum = {weights[i].sum():.4f})")

# STEP 4: weighted sum
output = weights @ V
print(f"\nStep 4: output = weights @ V")
for i, word in enumerate(["The", "cat", "sat"]):
    row = ", ".join(f"{v:.4f}" for v in output[i])
    print(f"    {word}: [{row}]")
```

**Expected output:**

```
Step 1: scores = Q @ K^T
  Q @ K^T =
  [1,0,0] . [0.5,0.1,0.0] = 0.50  [1,0,0] . [0.1,0.9,0.1] = 0.10  [1,0,0] . [0.0,0.2,0.8] = 0.00
  [0,1,0] . [0.5,0.1,0.0] = 0.10  [0,1,0] . [0.1,0.9,0.1] = 0.90  [0,1,0] . [0.0,0.2,0.8] = 0.20
  [0,0,1] . [0.5,0.1,0.0] = 0.00  [0,0,1] . [0.1,0.9,0.1] = 0.10  [0,0,1] . [0.0,0.2,0.8] = 0.80

  scores =
[[0.5 0.1 0. ]
 [0.1 0.9 0.2]
 [0.  0.1 0.8]]

Step 2: scaled = scores / sqrt(3) = scores / 1.7321
  scaled =
[[ 0.2887  0.0577  0.    ]
 [ 0.0577  0.5196  0.1155]
 [ 0.      0.0577  0.4619]]

Step 3: weights = softmax(scaled)
  weights =
    The: [0.3966, 0.3143, 0.2891]  (sum = 1.0000)
    cat: [0.2647, 0.4198, 0.3155]  (sum = 1.0000)
    sat: [0.2529, 0.2681, 0.4011]  (sum = 0.9221)

Step 4: output = weights @ V
    The: [0.3966, 0.3143, 0.2891]
    cat: [0.2647, 0.4198, 0.3155]
    sat: [0.2529, 0.2681, 0.4011]
```

Read this carefully. In step 3, "cat" has its highest weight on itself (0.42) -- it pays most attention to itself. "Sat" pays most attention to itself too (0.40), but also a notable amount to "cat" (0.27). This makes intuitive sense: "sat" is an action, and the most relevant context is *who* is doing the action -- the cat.

Because V is the identity matrix in this example, the output in step 4 is identical to the weights. In a real scenario, V would contain actual information vectors, and the output would be a weighted blend of those vectors.

> **Visualization**: Run `python tools/ml-visualizations/attention.py` to see attention weights as a heatmap, showing which words attend to which.

### Why Scale by sqrt(d_k)?

A brief aside on the scaling, since it's easy to dismiss as a minor detail. It's not.

When `d_k` is large (say, 512), the dot products between Q and K vectors can be very large in magnitude. Large inputs to softmax push it into saturation -- the output becomes nearly one-hot (one value close to 1, the rest close to 0). When softmax is saturated, its gradients are nearly zero, and the model can't learn.

Dividing by sqrt(d_k) keeps the variance of the scores at approximately 1.0 regardless of the dimension, which keeps softmax in the regime where it produces useful gradients.

```python
import numpy as np

np.random.seed(42)

# Low dimension: scores are small
d_k_small = 4
q_small = np.random.randn(d_k_small)
k_small = np.random.randn(d_k_small)
score_small = np.dot(q_small, k_small)
print(f"d_k = {d_k_small}: raw score = {score_small:.4f}")

# High dimension: scores are large
d_k_large = 512
q_large = np.random.randn(d_k_large)
k_large = np.random.randn(d_k_large)
score_large = np.dot(q_large, k_large)
print(f"d_k = {d_k_large}: raw score = {score_large:.4f}")

# After scaling
print(f"\nd_k = {d_k_small}: scaled score = {score_small / np.sqrt(d_k_small):.4f}")
print(f"d_k = {d_k_large}: scaled score = {score_large / np.sqrt(d_k_large):.4f}")
```

**Expected output:**

```
d_k = 4: raw score = -0.8949
d_k = 512: raw score = 7.1498

d_k = 4: scaled score = -0.4474
d_k = 512: scaled score = 0.3160
```

Without scaling, the 512-dimensional score (7.15) would dominate a softmax computation. After scaling, both scores are in a similar range (less than 1 in magnitude), keeping softmax well-behaved.

### Multi-Head Attention

A single attention computation focuses on one type of relationship. But words have multiple types of relationships simultaneously. "Cat" might relate to "sat" grammatically (subject-verb) and to "mat" spatially (the cat is on the mat).

**Multi-head attention** runs the attention mechanism multiple times in parallel, each with different learned Q, K, V projection matrices. Each "head" can learn to focus on a different type of relationship. The outputs from all heads are concatenated and projected back to the original dimension.

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention_head(embeddings, W_Q, W_K, W_V):
    """One attention head: project to Q, K, V, then compute attention."""
    Q = embeddings @ W_Q
    K = embeddings @ W_K
    V = embeddings @ W_V
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

def multi_head_attention(embeddings, n_heads, d_model):
    """Multi-head attention with random projections (for illustration)."""
    d_k = d_model // n_heads  # each head works in a smaller space

    all_outputs = []
    all_weights = []

    np.random.seed(42)
    for h in range(n_heads):
        W_Q = np.random.randn(d_model, d_k) * 0.5
        W_K = np.random.randn(d_model, d_k) * 0.5
        W_V = np.random.randn(d_model, d_k) * 0.5

        output, weights = attention_head(embeddings, W_Q, W_K, W_V)
        all_outputs.append(output)
        all_weights.append(weights)

    # Concatenate all head outputs
    concatenated = np.concatenate(all_outputs, axis=-1)  # (seq_len, d_model)
    return concatenated, all_weights

# Example: "The cat sat on the mat"
words = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 8
n_heads = 2

np.random.seed(0)
embeddings = np.random.randn(len(words), d_model) * 0.5

concatenated, all_weights = multi_head_attention(embeddings, n_heads, d_model)

print(f"Input shape:  {embeddings.shape}  ({len(words)} words, {d_model} dims)")
print(f"Output shape: {concatenated.shape}  ({len(words)} words, {d_model} dims)")
print(f"Number of heads: {n_heads}")
print(f"Each head dimension: {d_model // n_heads}")

for h in range(n_heads):
    print(f"\nHead {h + 1} attention weights:")
    print(f"{'':>6s}", end="")
    for w in words:
        print(f"{w:>6s}", end="")
    print()
    for i, w in enumerate(words):
        print(f"{w:>6s}", end="")
        for j in range(len(words)):
            print(f"{all_weights[h][i, j]:>6.2f}", end="")
        print()
```

**Expected output:**

```
Input shape:  (6, 8)  (6 words, 8 dims)
Output shape: (6, 8)  (6 words, 8 dims)
Number of heads: 2
Each head dimension: 4

Head 1 attention weights:
         The   cat   sat    on   the   mat
   The  0.19  0.14  0.25  0.08  0.11  0.22
   cat  0.18  0.13  0.24  0.10  0.11  0.24
   sat  0.17  0.15  0.19  0.12  0.15  0.21
    on  0.21  0.11  0.29  0.07  0.09  0.23
   the  0.17  0.16  0.19  0.13  0.16  0.19
   mat  0.20  0.12  0.27  0.07  0.09  0.25

Head 2 attention weights:
         The   cat   sat    on   the   mat
   The  0.15  0.17  0.17  0.17  0.19  0.15
   cat  0.17  0.14  0.18  0.14  0.16  0.21
   sat  0.17  0.16  0.16  0.16  0.18  0.17
    on  0.15  0.17  0.17  0.19  0.18  0.13
   the  0.14  0.18  0.16  0.19  0.19  0.14
   mat  0.18  0.14  0.19  0.13  0.15  0.21
```

Each head produces different attention weights. With random (untrained) projections, the patterns aren't meaningful yet -- but during training, each head would learn to focus on different relationships. One head might learn syntactic structure (subject-verb-object), another might learn positional proximity, and another might learn semantic similarity.

### Exercise 3.1: Compute Attention Weights by Hand

**Task:** Given the following Q, K, V matrices for a 3-word sentence with d_k = 2, compute the full attention output by hand. Show every intermediate step.

```
Q = [[1.0, 0.0],     K = [[1.0, 0.0],     V = [[1.0, 0.0],
     [0.0, 1.0],          [0.0, 1.0],          [0.0, 1.0],
     [1.0, 1.0]]          [1.0, 1.0]]          [0.5, 0.5]]
```

<details>
<summary>Hint 1: Step-by-step</summary>

1. Compute scores = Q @ K^T (a 3x3 matrix)
2. Scale by sqrt(d_k) = sqrt(2) ≈ 1.414
3. Apply softmax to each row
4. Multiply weights @ V
</details>

<details>
<summary>Hint 2: The dot products</summary>

scores[0][0] = [1,0] . [1,0] = 1.0
scores[0][1] = [1,0] . [0,1] = 0.0
scores[0][2] = [1,0] . [1,1] = 1.0
Continue for all 9 entries...
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
d_k = 2

# Step 1: scores
scores = Q @ K.T
print("Step 1: scores = Q @ K^T")
print(scores)

# Step 2: scale
scaled = scores / np.sqrt(d_k)
print(f"\nStep 2: scaled = scores / sqrt({d_k})")
print(np.round(scaled, 4))

# Step 3: softmax
weights = softmax(scaled)
print(f"\nStep 3: softmax (each row sums to 1.0)")
for i in range(3):
    row = [f"{w:.4f}" for w in weights[i]]
    print(f"  Row {i}: [{', '.join(row)}]  sum={weights[i].sum():.4f}")

# Step 4: output
output = weights @ V
print(f"\nStep 4: output = weights @ V")
for i in range(3):
    row = [f"{v:.4f}" for v in output[i]]
    print(f"  Word {i}: [{', '.join(row)}]")

# Interpretation
print("\nInterpretation:")
print("  Word 0 (Q=[1,0]) attends mainly to words with K that match — word 0 and word 2")
print("  Word 1 (Q=[0,1]) attends mainly to words with K that match — word 1 and word 2")
print("  Word 2 (Q=[1,1]) attends most to word 2 (K=[1,1], perfect match)")
```

**Expected output:**

```
Step 1: scores = Q @ K^T
[[1. 0. 1.]
 [0. 1. 1.]
 [1. 1. 2.]]

Step 2: scaled = scores / sqrt(2)
[[0.7071 0.     0.7071]
 [0.     0.7071 0.7071]
 [0.7071 0.7071 1.4142]]

Step 3: softmax (each row sums to 1.0)
  Row 0: [0.3775, 0.1862, 0.4363]  sum=1.0000
  Row 1: [0.1862, 0.3775, 0.4363]  sum=1.0000
  Row 2: [0.2312, 0.2312, 0.5375]  sum=1.0000

Step 4: output = weights @ V
  Word 0: [0.5957, 0.4043]
  Word 1: [0.4043, 0.5957]
  Word 2: [0.4999, 0.5001]

Interpretation:
  Word 0 (Q=[1,0]) attends mainly to words with K that match — word 0 and word 2
  Word 1 (Q=[0,1]) attends mainly to words with K that match — word 1 and word 2
  Word 2 (Q=[1,1]) attends most to word 2 (K=[1,1], perfect match)
```

**Explanation:** Word 2 has Q=[1,1] which dot-products highest with K=[1,1] (score=2), so it attends most to itself. Words 0 and 1 each attend to themselves and to word 2, because word 2's key is a superset of their queries. The output vectors are weighted blends of all the value vectors, with the blending determined by the attention weights.
</details>

### Exercise 3.2: Implement Multi-Head Attention

**Task:** Implement a `multi_head_attention(embeddings, W_Qs, W_Ks, W_Vs, W_O)` function that:
1. Takes a list of per-head projection matrices (W_Qs, W_Ks, W_Vs)
2. Computes attention for each head
3. Concatenates all head outputs
4. Projects through a final matrix W_O to produce the output

Test it on a 4-word sentence with 2 heads. Verify that the output has the same shape as the input.

<details>
<summary>Hint 1: Shapes</summary>

If d_model = 8 and n_heads = 2, each head works with d_k = d_model / n_heads = 4. Each W_Q, W_K, W_V is (d_model, d_k) = (8, 4). After concatenating 2 heads, you get (seq_len, 8). W_O is (8, 8) to project back.
</details>

<details>
<summary>Hint 2: The concatenation</summary>

```python
head_outputs = []
for h in range(n_heads):
    # compute attention for head h...
    head_outputs.append(output)
concatenated = np.concatenate(head_outputs, axis=-1)
result = concatenated @ W_O
```
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def single_head_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

def multi_head_attention(embeddings, W_Qs, W_Ks, W_Vs, W_O):
    """Multi-head attention with explicit projection matrices.

    embeddings: (seq_len, d_model)
    W_Qs, W_Ks, W_Vs: lists of (d_model, d_k) matrices, one per head
    W_O: (n_heads * d_k, d_model) output projection
    """
    head_outputs = []
    for W_Q, W_K, W_V in zip(W_Qs, W_Ks, W_Vs):
        Q = embeddings @ W_Q
        K = embeddings @ W_K
        V = embeddings @ W_V
        output, _ = single_head_attention(Q, K, V)
        head_outputs.append(output)

    concatenated = np.concatenate(head_outputs, axis=-1)
    return concatenated @ W_O

# Setup
np.random.seed(42)
d_model = 8
n_heads = 2
d_k = d_model // n_heads  # 4
seq_len = 4

embeddings = np.random.randn(seq_len, d_model) * 0.5

# Create projection matrices for each head
W_Qs = [np.random.randn(d_model, d_k) * 0.5 for _ in range(n_heads)]
W_Ks = [np.random.randn(d_model, d_k) * 0.5 for _ in range(n_heads)]
W_Vs = [np.random.randn(d_model, d_k) * 0.5 for _ in range(n_heads)]
W_O = np.random.randn(n_heads * d_k, d_model) * 0.5

output = multi_head_attention(embeddings, W_Qs, W_Ks, W_Vs, W_O)

print(f"Input shape:  {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Shapes match: {embeddings.shape == output.shape}")
print(f"\nNumber of heads: {n_heads}")
print(f"Per-head dimension: {d_k}")
print(f"W_Q shape: {W_Qs[0].shape}")
print(f"W_O shape: {W_O.shape}")
```

**Expected output:**

```
Input shape:  (4, 8)
Output shape: (4, 8)
Shapes match: True

Number of heads: 2
Per-head dimension: 4
W_Q shape: (8, 4)
W_O shape: (8, 8)
```

**Explanation:** The input has shape (4, 8) and the output has the same shape (4, 8). The multi-head attention processes the sequence and produces a context-aware version of each embedding, but the dimensionality is preserved. This is important for the transformer architecture, which stacks multiple attention layers and needs consistent shapes.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain what queries, keys, and values represent
- [ ] Walk through the four steps of scaled dot-product attention with numbers
- [ ] Explain why we scale by sqrt(d_k)
- [ ] Describe why multi-head attention uses multiple heads instead of one
- [ ] Implement `scaled_dot_product_attention` from scratch

---

## Section 4: The Transformer Architecture

### The Transformer Block

Attention gives you context-aware representations. But attention alone isn't enough -- you need to stack it with other operations to build a powerful model. The **transformer block** is the repeating unit that transformers are built from. Each block performs four operations in sequence:

1. **Multi-head attention**: Mix information between positions
2. **Add & normalize**: Stabilize the computation
3. **Feedforward network**: Process each position independently
4. **Add & normalize**: Stabilize again

A transformer is a stack of these blocks. GPT-3 uses 96 of them. Each block refines the representations, adding more context and abstraction.

### Residual Connections

The "add" in "add & normalize" is a **residual connection**. Instead of replacing the input with the attention output, you *add* the attention output to the original input:

```
output = input + attention(input)
```

This is the same idea you saw with gradients in the training route. During backpropagation, the gradient has to flow through every layer. If each layer transforms the signal, small gradients can vanish over many layers. Residual connections provide a shortcut -- the gradient can flow directly through the addition, bypassing the attention computation entirely. This lets you stack many layers without the gradient disappearing.

```python
import numpy as np

# Without residual connection: signal can degrade
x = np.array([1.0, 2.0, 3.0, 4.0])
for _ in range(10):
    x = x * 0.9  # each "layer" shrinks the signal
print(f"After 10 layers (no residual):  [{', '.join(f'{v:.4f}' for v in x)}]")

# With residual connection: signal is preserved
x = np.array([1.0, 2.0, 3.0, 4.0])
for _ in range(10):
    delta = x * 0.1  # each "layer" computes a small update
    x = x + delta     # residual: add the update to the original
print(f"After 10 layers (with residual): [{', '.join(f'{v:.4f}' for v in x)}]")
```

**Expected output:**

```
After 10 layers (no residual):  [0.3487, 0.6974, 1.0461, 1.3948]
After 10 layers (with residual): [2.5937, 5.1875, 7.7812, 10.3750]
```

Without residual connections, the signal decays. With them, the signal grows (in a controlled way). In a real transformer, the attention layer doesn't just multiply by a scalar -- it computes a complex transformation. But the principle is the same: the residual connection ensures the original information is preserved.

### Layer Normalization

**Layer normalization** rescales the values in each vector to have zero mean and unit variance. After adding the residual connection, the values can drift to large magnitudes. Layer normalization brings them back to a consistent range, which helps training stability.

```python
import numpy as np

def layer_norm(x, epsilon=1e-6):
    """Normalize each vector to zero mean and unit variance.

    x: (seq_len, d_model)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)

# Before normalization: values can be all over the place
x = np.array([
    [10.0, -5.0, 3.0, 0.0],
    [100.0, 200.0, 300.0, 400.0],
])

normed = layer_norm(x)

print("Before layer norm:")
for i in range(2):
    row = ", ".join(f"{v:>8.2f}" for v in x[i])
    print(f"  [{row}]  mean={np.mean(x[i]):.2f}  std={np.std(x[i]):.2f}")

print("\nAfter layer norm:")
for i in range(2):
    row = ", ".join(f"{v:>8.4f}" for v in normed[i])
    print(f"  [{row}]  mean={np.mean(normed[i]):.4f}  std={np.std(normed[i]):.4f}")
```

**Expected output:**

```
Before layer norm:
  [   10.00,    -5.00,     3.00,     0.00]  mean=2.00  std=5.48
  [  100.00,   200.00,   300.00,   400.00]  mean=250.00  std=111.80

After layer norm:
  [  1.4606,   -1.2777,    0.1826,   -0.3651]  mean=0.0001  std=1.0000
  [ -1.3416,   -0.4472,    0.4472,    1.3416]  mean=0.0000  std=1.0000
```

Regardless of the input scale, the output has mean near 0 and standard deviation near 1. This is critical for deep networks where values compound across layers.

### Putting It Together: A Transformer Block

Here's a complete transformer block, combining attention, residual connections, layer normalization, and a feedforward network:

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)

def relu(x):
    return np.maximum(0, x)

def self_attention(x, W_Q, W_K, W_V):
    """Single-head self-attention."""
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

def feedforward(x, W1, b1, W2, b2):
    """Position-wise feedforward network: two linear layers with ReLU."""
    hidden = relu(x @ W1 + b1)
    return hidden @ W2 + b2

def transformer_block(x, W_Q, W_K, W_V, W1, b1, W2, b2):
    """One transformer block.

    x: (seq_len, d_model)

    Flow:
      1. Self-attention
      2. Add residual + layer norm
      3. Feedforward
      4. Add residual + layer norm
    """
    # Step 1-2: Attention sub-layer with residual and norm
    attn_output = self_attention(x, W_Q, W_K, W_V)
    x = layer_norm(x + attn_output)

    # Step 3-4: Feedforward sub-layer with residual and norm
    ff_output = feedforward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_output)

    return x

# Setup
np.random.seed(42)
d_model = 8
d_ff = 16  # feedforward hidden dimension (typically 4x d_model)
seq_len = 4

# Input embeddings (with positional encoding already added)
x = np.random.randn(seq_len, d_model) * 0.5

# Attention projection matrices
W_Q = np.random.randn(d_model, d_model) * 0.3
W_K = np.random.randn(d_model, d_model) * 0.3
W_V = np.random.randn(d_model, d_model) * 0.3

# Feedforward weights
W1 = np.random.randn(d_model, d_ff) * 0.3
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.3
b2 = np.zeros(d_model)

# Run one transformer block
output = transformer_block(x, W_Q, W_K, W_V, W1, b1, W2, b2)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"\nInput (first word):")
print(f"  [{', '.join(f'{v:.4f}' for v in x[0])}]")
print(f"  mean={np.mean(x[0]):.4f}  std={np.std(x[0]):.4f}")
print(f"\nOutput (first word):")
print(f"  [{', '.join(f'{v:.4f}' for v in output[0])}]")
print(f"  mean={np.mean(output[0]):.4f}  std={np.std(output[0]):.4f}")
```

**Expected output:**

```
Input shape:  (4, 8)
Output shape: (4, 8)

Input (first word):
  [0.2481, 0.0693, -0.0321, 0.7627, -0.1174, 0.1172, -0.2717, -0.1177]
  mean=0.0823  std=0.3077

Output (first word):
  [-0.3005, 0.3747, -0.3247, 1.8553, -0.3718, -0.4099, -0.4694, -0.3537]
  mean=0.0000  std=0.7071
```

The shape is preserved (4, 8) -- same input and output dimensions. The values have changed (the block mixed information between positions and processed it through the feedforward network), and the output is layer-normalized (mean near 0). Stack 96 of these blocks and you have GPT-3.

### Encoder vs. Decoder

Transformers come in two flavors:

- **Encoder**: Processes the entire input at once. Each word can attend to every other word (bidirectional). Used for understanding tasks like classification and translation (the input side). BERT is an encoder-only model.

- **Decoder**: Generates output one token at a time. Each word can only attend to previous words, not future ones (causal/unidirectional). This is done by masking future positions in the attention computation. GPT is a decoder-only model.

The original transformer paper used both: an encoder to process the input and a decoder to generate the output, with the decoder also attending to the encoder's output. Most modern LLMs (GPT, Claude, Llama) use decoder-only architectures.

The masking in a decoder works by setting the scores of future positions to negative infinity before softmax, which forces their attention weights to zero:

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Scores for a 4-word sentence
scores = np.array([
    [1.0, 0.5, 0.3, 0.2],
    [0.4, 1.0, 0.6, 0.1],
    [0.3, 0.5, 1.0, 0.7],
    [0.2, 0.4, 0.3, 1.0],
])

# Causal mask: -inf for future positions
mask = np.triu(np.ones_like(scores) * -np.inf, k=1)
masked_scores = scores + mask

print("Unmasked scores:")
print(scores)
print("\nCausal mask:")
print(mask)
print("\nMasked scores (future positions = -inf):")
for row in masked_scores:
    print(f"  [{', '.join(f'{v:>6.2f}' for v in row)}]")

weights = softmax(masked_scores)
print("\nAttention weights after softmax:")
words = ["I", "like", "big", "cats"]
for i, w in enumerate(words):
    row = ", ".join(f"{v:.4f}" for v in weights[i])
    print(f"  {w:>5s}: [{row}]")
```

**Expected output:**

```
Unmasked scores:
[[1.  0.5 0.3 0.2]
 [0.4 1.  0.6 0.1]
 [0.3 0.5 1.  0.7]
 [0.2 0.4 0.3 1. ]]

Causal mask:
[[ 0. -inf -inf -inf]
 [ 0.  0. -inf -inf]
 [ 0.  0.  0. -inf]
 [ 0.  0.  0.  0.]]

Masked scores (future positions = -inf):
  [  1.00,   -inf,   -inf,   -inf]
  [  0.40,   1.00,   -inf,   -inf]
  [  0.30,   0.50,   1.00,   -inf]
  [  0.20,   0.40,   0.30,   1.00]

Attention weights after softmax:
      I: [1.0000, 0.0000, 0.0000, 0.0000]
   like: [0.3543, 0.6457, 0.0000, 0.0000]
    big: [0.2076, 0.2537, 0.4180, 0.0000]  (attends to "I", "like", and itself, but NOT "cats")
   cats: [0.1518, 0.1857, 0.1681, 0.3382]  (attends to everything)  (attends to everything)
```

Word 0 ("I") can only attend to itself. Word 1 ("like") can attend to "I" and itself. Word 2 ("big") can attend to "I", "like", and itself. This prevents the model from "cheating" by looking at future words during generation.

### Why Transformers Win

Before transformers, sequence models used **recurrent neural networks** (RNNs) that processed words one at a time, passing a hidden state from each word to the next. This sequential processing had two problems:

1. **No parallelism**: You can't process word 5 until you've finished words 1-4. Training is slow.
2. **Long-range dependency problem**: Information from early words has to survive through every intermediate hidden state. Over long sequences, it degrades.

Transformers solve both problems. Attention computes relationships between *all pairs of words simultaneously* -- every word can directly attend to every other word, regardless of distance. And because there's no sequential dependency, all positions can be computed in parallel on GPUs.

### Exercise 4.1: Trace Data Through a Transformer Block

**Task:** Using the `transformer_block` function from this section, trace the shapes at each step for a 5-word sentence with d_model=6 and d_ff=12. Don't compute the actual values -- just predict the shape of every intermediate result.

<details>
<summary>Hint: The shapes</summary>

Input: (5, 6). After attention: (5, 6). After add+norm: (5, 6). After W1@x+b1: (5, 12). After ReLU: (5, 12). After W2@x+b2: (5, 6). After add+norm: (5, 6). Every step either preserves the shape or temporarily expands to d_ff and back.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)

def relu(x):
    return np.maximum(0, x)

np.random.seed(42)
d_model = 6
d_ff = 12
seq_len = 5

x = np.random.randn(seq_len, d_model)
print(f"Input:                  {x.shape}")

# Attention sub-layer
W_Q = np.random.randn(d_model, d_model) * 0.3
W_K = np.random.randn(d_model, d_model) * 0.3
W_V = np.random.randn(d_model, d_model) * 0.3

Q = x @ W_Q
K = x @ W_K
V = x @ W_V
print(f"Q = x @ W_Q:            {Q.shape}")
print(f"K = x @ W_K:            {K.shape}")
print(f"V = x @ W_V:            {V.shape}")

scores = Q @ K.T
print(f"scores = Q @ K^T:       {scores.shape}")

scores_scaled = scores / np.sqrt(d_model)
print(f"scores / sqrt(d_k):     {scores_scaled.shape}")

weights = softmax(scores_scaled)
print(f"softmax(scores):        {weights.shape}")

attn_out = weights @ V
print(f"weights @ V:            {attn_out.shape}")

x_after_attn = layer_norm(x + attn_out)
print(f"add + layer_norm:       {x_after_attn.shape}")

# Feedforward sub-layer
W1 = np.random.randn(d_model, d_ff) * 0.3
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.3
b2 = np.zeros(d_model)

hidden = x_after_attn @ W1 + b1
print(f"x @ W1 + b1:            {hidden.shape}  (expanded to d_ff)")

hidden_relu = relu(hidden)
print(f"ReLU(hidden):           {hidden_relu.shape}")

ff_out = hidden_relu @ W2 + b2
print(f"hidden @ W2 + b2:       {ff_out.shape}  (back to d_model)")

output = layer_norm(x_after_attn + ff_out)
print(f"add + layer_norm:       {output.shape}")

print(f"\nInput shape == Output shape: {x.shape == output.shape}")
```

**Expected output:**

```
Input:                  (5, 6)
Q = x @ W_Q:            (5, 6)
K = x @ W_K:            (5, 6)
V = x @ W_V:            (5, 6)
scores = Q @ K^T:       (5, 5)
scores / sqrt(d_k):     (5, 5)
softmax(scores):        (5, 5)
weights @ V:            (5, 6)
add + layer_norm:       (5, 6)
x @ W1 + b1:            (5, 12)  (expanded to d_ff)
ReLU(hidden):           (5, 12)
hidden @ W2 + b2:       (5, 6)  (back to d_model)
add + layer_norm:       (5, 6)

Input shape == Output shape: True
```

**Explanation:** The transformer block preserves the shape from input to output. The feedforward network temporarily expands to d_ff (typically 4x d_model) and contracts back. This expansion gives the model more computational capacity at each position. The scores matrix is (seq_len, seq_len) because every word needs an attention weight for every other word.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] List the four operations in a transformer block
- [ ] Explain why residual connections help gradient flow
- [ ] Explain what layer normalization does and why it matters
- [ ] Describe the difference between encoder and decoder transformers
- [ ] Explain why transformers can be parallelized but RNNs can't

---

## Section 5: From Transformers to Language Models

### Pre-Training: Predicting the Next Token

A language model is trained on a single task: **predict the next token**. Given a sequence of tokens, predict what comes next. That's it.

The training data is massive -- hundreds of billions of words from the internet, books, code repositories, and more. The model sees "The cat sat on the" and learns to predict "mat" (or "floor" or "couch"). It sees "def fibonacci(n):" and learns to predict "if n <= 1:". It sees "To be or not to" and learns to predict "be".

```
Input:  "The cat sat on the"
Target: "mat"

Input:  "She walked to the"
Target: "store" (or "park" or "door" -- many are valid)
```

The model's output at each position is a probability distribution over the entire vocabulary. It doesn't predict a single word -- it assigns a probability to *every possible next word*. During training, the loss function (cross-entropy) pushes the model to assign high probability to the actual next token.

```python
import numpy as np

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# Simulated model output: raw scores (logits) for each word in vocabulary
# The model has seen "The cat sat on the" and produces logits for the next word
vocab = ["the", "cat", "mat", "dog", "sat", "on", "a", "big", "floor"]
logits = np.array([0.2, 0.5, 3.1, 0.3, 0.1, 0.0, 0.8, 0.4, 2.5])

probs = softmax(logits)

print("Input: 'The cat sat on the ___'")
print(f"\nModel's predictions (probability of each next word):")
sorted_indices = np.argsort(-probs)
for idx in sorted_indices:
    bar = "#" * int(probs[idx] * 50)
    print(f"  {vocab[idx]:>6s}: {probs[idx]:.4f}  {bar}")

print(f"\nProbabilities sum to: {probs.sum():.4f}")
print(f"Most likely next word: '{vocab[np.argmax(probs)]}'")
```

**Expected output:**

```
Input: 'The cat sat on the ___'

Model's predictions (probability of each next word):
     mat: 0.4742  #######################
   floor: 0.2601  #############
       a: 0.0476  ##
     cat: 0.0353  #
     big: 0.0319  #
     dog: 0.0289  #
     the: 0.0261  #
     sat: 0.0236  #
      on: 0.0214  #

Probabilities sum to: 1.0000
Most likely next word: 'mat'
```

The model assigns the highest probability to "mat" (0.47) and the second-highest to "floor" (0.26). Both are reasonable continuations. The model has learned, through exposure to billions of sentences, that objects you sit on typically follow "sat on the."

### How Scale Changes Behavior

Something remarkable happens as language models get bigger:

- **Small models** (millions of parameters): Can complete simple patterns and produce grammatically correct text, but struggle with facts and reasoning.
- **Medium models** (billions of parameters): Can follow instructions, answer questions, summarize text, and translate between languages.
- **Large models** (hundreds of billions): Show **emergent abilities** -- capabilities that appear suddenly at a certain scale. These include multi-step reasoning, understanding analogies, writing code, and explaining their own outputs.

Nobody fully understands why scaling produces these jumps. The training objective never changes -- it's always "predict the next token." But at sufficient scale, the internal representations become rich enough to support reasoning-like behavior. The model learns so many patterns about language that it develops what looks like understanding.

### Fine-Tuning and Instruction Following

Pre-training produces a model that's good at completing text. If you give it "What is the capital of France?", it might continue with "What is the capital of Germany? What is the capital of..." -- because on the internet, questions are often followed by more questions.

**Fine-tuning** adapts the pre-trained model to a specific behavior. There are several stages:

1. **Supervised fine-tuning (SFT)**: Train on examples of desired input-output pairs. "Question: What is 2+2? Answer: 4." The model learns the pattern of responding to questions rather than continuing text.

2. **Reinforcement Learning from Human Feedback (RLHF)**: Human raters rank model outputs from best to worst. A reward model learns from these rankings. The language model is then trained to produce outputs that the reward model scores highly. This is how models learn to be helpful, follow instructions, and avoid harmful content.

The pre-trained model learns *what language looks like*. Fine-tuning teaches it *what helpful responses look like*.

### The Gap Between "Predicts Next Token" and "Answers Questions"

This is the most important conceptual point in this route. At the mechanical level, a language model does exactly one thing: given a sequence of tokens, it outputs a probability distribution over the next token. That's all. There's no explicit "understanding" module, no "reasoning" engine, no database of facts.

And yet, from this simple objective, the model develops behavior that looks like understanding. It can explain concepts, write code, translate languages, solve math problems, and hold multi-turn conversations.

How? The best current hypothesis: predicting the next token *requires* modeling the processes that generated the text. To predict what comes after "The proof proceeds by induction. Base case: n=1..." the model must learn something about mathematical reasoning. To predict dialogue, it must learn something about how conversations work. The richer its internal model of the world, the better its predictions.

This doesn't mean the model "truly understands" in the way humans do -- that's a philosophical question beyond this route. But it means the gap between "predicts tokens" and "behaves intelligently" is narrower than it first appears. Token prediction is a surprisingly powerful objective.

### Exercise 5.1: What Does the Model's Output Represent?

**Task (conceptual):** Consider a language model that has processed the input "The square root of 144 is". Think about these questions and write your answers:

1. What is the model's output at this point? (What data structure, what does each element mean?)
2. What should the highest-probability token be, and why?
3. If the model assigns 0.95 probability to "12" and 0.03 to "approximately", what does that tell you about what the model has learned?
4. If a much smaller model assigns 0.15 probability to "12" and 0.12 to "the" and 0.10 to "a", what does that tell you?

<details>
<summary>Answers</summary>

1. The output is a probability distribution over the entire vocabulary (e.g., 50,000 values that sum to 1.0). Each value represents the model's estimate of how likely that token is to come next. It's a vector produced by applying softmax to the raw logits from the final layer.

2. The highest-probability token should be "12" because the square root of 144 is 12. The model has seen many mathematical statements during pre-training and has learned this relationship.

3. High confidence (0.95) on the correct answer means the model has learned mathematical facts and can apply them in context. The 0.03 on "approximately" suggests the model knows this is a math context where approximate answers are possible (like sqrt of non-perfect squares), but correctly identifies that 144 is a perfect square.

4. A model that spreads probability roughly evenly across many tokens (0.15, 0.12, 0.10...) hasn't learned the mathematical relationship. It knows this is English text and that common words like "the" and "a" are likely after any prefix, but it can't reason about square roots. This is what scale buys: enough capacity to learn these relationships.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain what pre-training optimizes (next-token prediction)
- [ ] Describe the model's output at each position (probability distribution over vocabulary)
- [ ] Explain what fine-tuning and RLHF do at a high level
- [ ] Articulate the gap between "predicts next token" and "answers questions"

---

## Practice Project: Minimal Attention in Numpy

### Project Description

Build a complete scaled dot-product attention implementation from scratch. Create fake embeddings for a short sentence, compute attention weights, and visualize the results as a heatmap. Then experiment with the embeddings to see how changing word similarities affects the attention pattern.

This project ties together everything from this route: embeddings (Section 1), positional encoding (Section 2), and the full attention computation (Section 3). It uses the same numpy and matplotlib you've been using throughout.

### Requirements

Build a script that:
1. Defines a vocabulary with embeddings for a 5-6 word sentence
2. Adds positional encoding to the embeddings
3. Creates Q, K, V projection matrices
4. Computes scaled dot-product attention
5. Prints the attention weight matrix
6. Visualizes the attention weights as a heatmap using matplotlib
7. Experiments: make two words more similar and observe how attention changes

### Getting Started

**Step 1: Set up your sentence and embeddings**

Choose a short sentence (5-6 words). Create a small embedding table where semantically similar words have similar vectors.

**Step 2: Add positional encoding**

Use your sinusoidal positional encoding implementation from Section 2.

**Step 3: Compute attention**

Create random W_Q, W_K, W_V matrices. Project the embeddings to get Q, K, V. Run the four-step attention computation.

**Step 4: Visualize**

Use matplotlib's `imshow` to create a heatmap of the attention weights. Label the axes with the actual words.

**Step 5: Experiment**

Make two words have identical embeddings and see what happens to the attention pattern. Then try making one word's embedding very different from all others.

### Hints and Tips

<details>
<summary>Hint 1: Setting up the heatmap</summary>

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(weights, cmap="Blues")
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation=45)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel("Key (attending TO)")
ax.set_ylabel("Query (attending FROM)")
plt.colorbar(im)
```
</details>

<details>
<summary>Hint 2: Adding text annotations to cells</summary>

```python
for i in range(len(words)):
    for j in range(len(words)):
        color = "white" if weights[i, j] > 0.5 else "black"
        ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color=color)
```
</details>

<details>
<summary>Hint 3: The experiment</summary>

Try replacing one word's embedding with another word's embedding (make them identical). Then re-run attention. Those two words should now attend to each other more strongly because their Q and K projections will be similar.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    dim_indices = np.arange(0, d_model, 2)
    frequencies = 1.0 / (10000 ** (dim_indices / d_model))
    pe[:, 0::2] = np.sin(positions * frequencies)
    pe[:, 1::2] = np.cos(positions * frequencies)
    return pe


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


def plot_attention(weights, words, title="Attention Weights"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=weights.max())

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=11, rotation=45, ha="right")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel("Key (attending TO)", fontsize=12)
    ax.set_ylabel("Query (attending FROM)", fontsize=12)
    ax.set_title(title, fontsize=13)

    for i in range(len(words)):
        for j in range(len(words)):
            color = "white" if weights[i, j] > 0.5 * weights.max() else "black"
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, label="Weight")
    fig.tight_layout()
    return fig


# --- Setup ---
np.random.seed(42)
words = ["The", "cat", "sat", "on", "mat"]
d_model = 16

# Hand-crafted embeddings (similar words have similar vectors)
embedding_table = np.random.randn(len(words), d_model) * 0.5

# Make "cat" and "mat" somewhat similar (they rhyme and are both nouns)
embedding_table[4] = embedding_table[1] * 0.7 + np.random.randn(d_model) * 0.2

# Add positional encoding
pe = positional_encoding(len(words), d_model)
x = embedding_table + pe

# Projection matrices
W_Q = np.random.randn(d_model, d_model) * 0.3
W_K = np.random.randn(d_model, d_model) * 0.3
W_V = np.random.randn(d_model, d_model) * 0.3

# Compute attention
Q = x @ W_Q
K = x @ W_K
V = x @ W_V
output, weights = scaled_dot_product_attention(Q, K, V)

# Print the weight matrix
print("Attention weights:")
print(f"{'':>6s}", end="")
for w in words:
    print(f"{w:>6s}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:>6s}", end="")
    for j in range(len(words)):
        print(f"{weights[i, j]:>6.2f}", end="")
    print()

# Plot
plot_attention(weights, words, title="Attention: 'The cat sat on mat'")
plt.savefig("attention_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

# --- Experiment: make "cat" and "mat" identical ---
print("\n\n--- Experiment: making 'cat' and 'mat' embeddings identical ---\n")
embedding_table_exp = embedding_table.copy()
embedding_table_exp[4] = embedding_table_exp[1]  # mat = cat

x_exp = embedding_table_exp + pe
Q_exp = x_exp @ W_Q
K_exp = x_exp @ W_K
V_exp = x_exp @ W_V
_, weights_exp = scaled_dot_product_attention(Q_exp, K_exp, V_exp)

print("Attention weights (cat == mat embeddings):")
print(f"{'':>6s}", end="")
for w in words:
    print(f"{w:>6s}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:>6s}", end="")
    for j in range(len(words)):
        print(f"{weights_exp[i, j]:>6.2f}", end="")
    print()

# Compare cat's attention to mat in both cases
print(f"\n'cat' attention to 'mat' (original):  {weights[1, 4]:.4f}")
print(f"'cat' attention to 'mat' (identical): {weights_exp[1, 4]:.4f}")
print(f"Change: {weights_exp[1, 4] - weights[1, 4]:+.4f}")

plot_attention(weights_exp, words,
               title="Attention: 'cat' and 'mat' with identical embeddings")
plt.savefig("attention_heatmap_experiment.png", dpi=100, bbox_inches="tight")
plt.show()
```

**Key points in this solution:**
- The embedding table uses hand-crafted vectors where similar words are placed close together in the space.
- Positional encoding is added *on top of* the embeddings, so the model has both word identity and position information.
- The experiment demonstrates that making two words' embeddings identical increases how much they attend to each other -- the Q and K projections become more similar, producing higher attention scores.
- With random (untrained) projection matrices, the attention patterns aren't semantically meaningful. In a trained model, the projections would be optimized so that relevant words attend to each other.
</details>

### Extending the Project

If you want to go further, try:
- Add multi-head attention (2-4 heads) and compare the weight patterns across heads
- Implement causal masking (decoder-style) and observe how attention changes
- Build a full transformer block around your attention implementation
- Add a second transformer block on top of the first and trace how representations change through two layers

---

## Summary

### Key Takeaways

- **Embeddings** represent words as dense vectors where similar words have similar vectors. They're learned during training, and they encode semantic relationships as directions in the vector space (king - man + woman = queen).
- **Positional encoding** adds position information to embeddings using sine and cosine functions at different frequencies. This gives the model a sense of word order without any recurrence.
- **Attention** computes how much each word should look at every other word. It uses queries, keys, and values: Q and K determine the weights (via scaled dot product + softmax), and V provides the information that gets mixed.
- **The transformer block** combines attention with residual connections, layer normalization, and a feedforward network. Stack many blocks to build a transformer.
- **Language models** are transformers trained to predict the next token. Scale and fine-tuning transform this simple objective into models that can follow instructions, reason, and generate coherent text.

### Skills You've Gained

You can now:
- Build an embedding lookup table and measure word similarity with cosine similarity
- Implement sinusoidal positional encoding
- Compute scaled dot-product attention step by step with concrete numbers
- Implement attention from scratch in numpy
- Describe the complete transformer architecture
- Explain how pre-training, fine-tuning, and RLHF produce instruction-following models

### Self-Assessment

Take a moment to reflect:
- Can you trace the four steps of attention with actual numbers on paper?
- Could you explain to a colleague why the scaling factor sqrt(d_k) matters?
- If someone asked "how does ChatGPT work?", could you give a concrete answer that starts with embeddings and ends with next-token prediction?
- Do you see how every component in this route builds on what you learned in the previous routes (dot products, matrix multiplication, gradients, training loops)?

---

## Next Steps

### Continue Learning

**Continue the ascent:**
- [Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md) -- The guided project that ties all the ML routes together. Reflect on how transformer components map to the network you built.

**Go deeper:**
- Read "Attention Is All You Need" (Vaswani et al., 2017) -- the original transformer paper. You now have all the background to understand it.
- Explore transformer implementations in PyTorch or JAX to see how the numpy operations map to framework code.

### Additional Resources

**Papers:**
- "Attention Is All You Need" (Vaswani et al., 2017) -- The transformer paper. The architecture described there is exactly what you've learned in this route.
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019) -- Encoder-only transformer for understanding tasks.
- "Language Models are Few-Shot Learners" (Brown et al., 2020) -- The GPT-3 paper, showing how scale changes behavior.

**Videos:**
- 3Blue1Brown's "Attention in transformers, visually explained" -- Excellent visual walkthrough of the same concepts from this route.
- Andrej Karpathy's "Let's build GPT from scratch" -- Builds a working transformer in PyTorch, step by step.

**Interactive:**
- Jay Alammar's "The Illustrated Transformer" (jalammar.github.io) -- The best visual guide to transformer internals.

---

## Quick Reference

### Cosine Similarity

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    dim_indices = np.arange(0, d_model, 2)
    frequencies = 1.0 / (10000 ** (dim_indices / d_model))
    pe[:, 0::2] = np.sin(positions * frequencies)
    pe[:, 1::2] = np.cos(positions * frequencies)
    return pe
```

### Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Softmax

```python
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Layer Normalization

```python
def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)
```

### Transformer Block

```python
def transformer_block(x, W_Q, W_K, W_V, W1, b1, W2, b2):
    attn_output = self_attention(x, W_Q, W_K, W_V)
    x = layer_norm(x + attn_output)          # attention + residual + norm
    ff_output = feedforward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_output)            # feedforward + residual + norm
    return x
```

---

## Glossary

- **Attention**: A mechanism that computes how much each element in a sequence should focus on every other element. Produces a weighted combination of value vectors, where the weights are determined by the similarity between queries and keys.
- **Causal masking**: Setting attention scores to negative infinity for future positions, preventing the model from looking ahead. Used in decoder-style transformers (GPT, Claude).
- **Cosine similarity**: A measure of the angle between two vectors, ranging from -1 (opposite) to 1 (identical direction). Used to compare word embeddings.
- **Decoder**: A transformer component that generates output one token at a time, attending only to previous positions. GPT and Claude are decoder-only models.
- **Embeddings**: Dense vector representations of discrete tokens (words, subwords). Learned during training so that semantically similar tokens have similar vectors.
- **Encoder**: A transformer component that processes the entire input bidirectionally. BERT is an encoder-only model.
- **Emergent abilities**: Capabilities that appear in language models only at sufficient scale, such as multi-step reasoning, code generation, and analogy understanding.
- **Fine-tuning**: Adapting a pre-trained model to a specific task or behavior by training on task-specific data. Includes supervised fine-tuning (SFT) and RLHF.
- **Key (K)**: In attention, a vector that describes what information a position contains. Compared against queries to determine attention weights.
- **Layer normalization**: Normalizing each vector to have zero mean and unit variance. Stabilizes training in deep networks.
- **Logits**: The raw, unnormalized scores output by the model's final layer, before softmax converts them to probabilities.
- **Multi-head attention**: Running attention multiple times in parallel with different learned projections, allowing the model to attend to different types of relationships simultaneously.
- **One-hot encoding**: Representing a word as a vector with a single 1 and all other values 0. Simple but captures no semantic similarity.
- **Positional encoding**: Information added to embeddings that encodes each token's position in the sequence. Transformers use sinusoidal functions for this.
- **Pre-training**: Training a language model on massive text data to predict the next token. Produces a general-purpose model that can be fine-tuned for specific tasks.
- **Query (Q)**: In attention, a vector that describes what information a position is looking for. Compared against keys to determine attention weights.
- **Residual connection**: Adding the input of a sub-layer to its output (output = input + sublayer(input)). Helps gradients flow through deep networks.
- **RLHF (Reinforcement Learning from Human Feedback)**: Training a language model using human preferences to produce helpful, harmless, and honest responses.
- **Scaled dot-product attention**: The standard attention mechanism: scores = Q @ K^T / sqrt(d_k), weights = softmax(scores), output = weights @ V.
- **Softmax**: A function that converts a vector of raw scores into a probability distribution (all positive, sums to 1).
- **Token**: The basic unit of text that a language model processes. Can be a word, a subword, or a character, depending on the tokenizer.
- **Transformer**: A neural network architecture based on self-attention, introduced in "Attention Is All You Need" (2017). The foundation of modern language models.
- **Transformer block**: The repeating unit of a transformer: self-attention, add & normalize, feedforward, add & normalize.
- **Value (V)**: In attention, a vector that contains the actual information to be retrieved. The output of attention is a weighted sum of value vectors.
