---
title: LLM Foundations
route_map: /routes/llm-foundations/map.md
paired_guide: /routes/llm-foundations/guide.md
topics:
  - Embeddings
  - Attention
  - Transformers
  - Language Models
---

# LLM Foundations - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach how large language models work — from word embeddings through the attention mechanism to the transformer architecture. The learner arrives having built and trained a neural network from scratch and leaves understanding the architecture behind GPT-style models.

**Route Map**: See `/routes/llm-foundations/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/llm-foundations/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain how embeddings represent words as vectors with meaning
- Describe the attention mechanism step by step with concrete numbers
- Explain the transformer architecture and how its components fit together
- Understand how transformers scale into large language models
- Implement scaled dot-product attention from scratch in numpy

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/llm-foundations/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has completed:
- **Training and Backprop** — specifically loss functions, backpropagation, and training loops
- **Neural Network Foundations** — specifically layers, forward pass, and the role of activation functions
- **Linear Algebra Essentials** — specifically dot products, matrix multiplication, and the idea that matrices are transformations

**Quick prerequisite check:**
1. "What does a dot product measure?" (Should get: how aligned two vectors are — how much they point in the same direction)
2. "Can you describe the forward pass through a 2-layer network?" (Should get: input goes through W1 @ x + b1, activation, then W2 @ h + b2, activation — each layer transforms the input)
3. "What does backpropagation compute?" (Should get: the gradient of the loss with respect to every weight, using the chain rule backward through the network)

**If prerequisites are missing**: Suggest they work through the prerequisite routes first. This route builds directly on all three. The dot product is the computational core of attention — if they don't understand dot products, attention won't make sense. If they don't understand forward passes and training, the transformer architecture will feel arbitrary.

### Audience Context
The target learner is a backend developer who has completed the four prerequisite routes. They have built a neural network from scratch that classifies handwritten digits. They understand matrices as transformations, derivatives as rates of change, and training as forward-loss-backward-update.

Use this to your advantage:
- Forward pass → data flowing through transformer blocks (same concept, different architecture)
- Training loop → pre-training on next-token prediction (same loop, different data and loss)
- Dot product → attention scores (the dot product they already know IS the core operation in attention)
- Weight matrices → query/key/value projections (familiar linear transformations with specific roles)
- Batch processing → processing all tokens in parallel (the parallelism advantage of transformers)

The key transition in this route: the learner goes from "I know how neural networks work" to "I know how the specific neural network architecture behind LLMs works." The building blocks are the same — what's different is how they're arranged.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). Configuration options include:

**Teaching Style:**
- `tone`: objective, encouraging, humorous (default: objective and respectful)
- `explanation_depth`: concise, balanced, detailed
- `pacing`: learner-led, balanced, structured

**Assessment Format:**
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)
- `quiz_frequency`: after_each_section, after_major_topics, end_of_route
- `feedback_style`: immediate, summary, detailed

**Example `.sherpa-config.yml`:**
```yaml
teaching:
  tone: encouraging
  explanation_depth: balanced
  pacing: learner-led

assessment:
  quiz_type: mixed
  quiz_frequency: after_major_topics
  feedback_style: immediate
```

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types to verify understanding:

**Multiple Choice Questions:**
- Present 3-4 answer options
- Include plausible distractors based on common misconceptions
- Example: "In scaled dot-product attention, why do we divide by sqrt(d_k)? A) To normalize the output to [0, 1] B) To prevent the dot products from growing too large for softmax C) To reduce computation time D) To make the gradients flow better"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess whether they understand the mechanism, not just the vocabulary
- Example: "In your own words, why do we need positional encoding?"

**Tracing Questions:**
- Give concrete numbers and ask the learner to trace attention by hand
- These are the most important assessments in this route
- Example: "Given Q = [1, 0], K = [[1, 0], [0, 1]], V = [[5, 0], [0, 5]], what is the attention output?"

**Prediction Questions:**
- Show a scenario and ask what will happen
- Builds intuition about how attention and transformers behave
- Example: "If two words have very similar embedding vectors, what will their attention score be?"

---

## Teaching Flow

### Introduction

**What to Cover:**
- The learner has built a network that classifies images. Language models do something similar but with text — and the architecture is different.
- The core question: how do you build a neural network that processes language?
- Three problems to solve: (1) turn words into numbers, (2) understand relationships between words, (3) scale it up
- This route covers the architecture — how the pieces fit together. The training process uses the same forward-loss-backward-update loop they already know.

**Opening Questions to Assess Level:**
1. "You've trained a network on MNIST. What's different about processing text vs. images?"
2. "Have you heard terms like 'embeddings,' 'attention,' or 'transformer' before? What's your mental model?"
3. "What are you most curious about regarding how LLMs work?"

**Adapt based on responses:**
- If they've read about transformers: Move through embeddings quickly, spend more time on the mechanics of attention and the hands-on implementation
- If they're curious about ChatGPT: Frame everything as "these are the building blocks that make ChatGPT possible — by the end, you'll understand the architecture"
- If they're intimidated: Ground it immediately: "Everything in this route uses concepts you already know — dot products, matrix multiplication, softmax. The architecture is just a specific arrangement of familiar pieces."
- If they have NLP background: Skip the "why words need numbers" motivation, go straight to learned embeddings and attention mechanics

**Good opening framing:**
"You built a neural network that recognizes handwritten digits. That network takes a grid of pixels (numbers) and classifies them. Language models do something similar — but with text. The challenge is: text isn't numbers. Words don't come in a nice grid. And word order matters in a way that pixel order doesn't. This route covers the key ideas that solve these problems: embeddings turn words into numbers, attention captures relationships between words, and the transformer architecture puts it all together. Every concept builds on things you already know — dot products, matrix multiplication, softmax, and training loops."

---

### Setup Verification

**Check numpy:**
```bash
python -c "import numpy as np; print(np.__version__)"
```

**Check matplotlib (needed for attention heatmaps):**
```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

**If not installed:**
```bash
pip install numpy matplotlib
```

**Check visualization scripts:**
Verify the scripts exist at `/tools/ml-visualizations/`:
- `embeddings.py` — Embedding space visualization
- `attention.py` — Attention weight heatmap visualization

**Notation Reminder:**
"We'll build on notation from previous routes. `@` is matrix multiplication, `W` is a weight matrix. New today: `Q` for queries, `K` for keys, `V` for values, `d_k` for the dimension of key vectors, `softmax()` for the function that converts scores to probabilities. I'll explain each one as it comes up."

---

### Section 1: Embeddings

**Core Concept to Teach:**
Neural networks need numbers. Text is words. Embeddings bridge the gap — they represent each word as a dense vector where the geometry encodes meaning. Similar words end up close together in the vector space. Embeddings are learned, not hand-designed.

**How to Explain:**
1. Start with the problem: "Your MNIST network takes 784 numbers (pixel values). How do you feed the word 'cat' to a network?"
2. Show the naive approach (one-hot encoding) and why it fails
3. Introduce learned embeddings as the solution
4. Show that the geometry of the embedding space encodes meaning

**The Problem — Words Aren't Numbers:**

"Your MNIST network takes a 784-dimensional vector (28x28 pixel values). Each pixel is a number between 0 and 1. Easy. But what about the word 'cat'? It's not a number. You can't multiply a matrix by the string 'cat'. You need a way to represent words as vectors."

**Attempt 1 — One-Hot Encoding:**

```python
import numpy as np

# Vocabulary: 5 words
vocab = ['cat', 'dog', 'fish', 'king', 'queen']

# One-hot: each word is a vector with a 1 in its position
cat   = np.array([1, 0, 0, 0, 0])
dog   = np.array([0, 1, 0, 0, 0])
fish  = np.array([0, 0, 1, 0, 0])
king  = np.array([0, 0, 0, 1, 0])
queen = np.array([0, 0, 0, 0, 1])

# Problem 1: No similarity information
print(f"cat · dog  = {np.dot(cat, dog)}")    # 0 — but cats and dogs are similar!
print(f"cat · fish = {np.dot(cat, fish)}")   # 0 — same distance as cat to king
print(f"king · queen = {np.dot(king, queen)}")  # 0 — no relationship captured

# Problem 2: Dimension grows with vocabulary
# GPT-style models have 50,000+ tokens
# One-hot vectors would be 50,000-dimensional with a single 1
print(f"\nVector dimension: {len(cat)}")
print(f"Non-zero elements: {np.count_nonzero(cat)}")
print(f"Sparsity: {1 - np.count_nonzero(cat)/len(cat):.0%}")
```

"Two problems. First: every pair of words has the same dot product — zero. 'Cat' is as different from 'dog' as it is from 'quantum physics.' That's wrong — cats and dogs are both pets, both animals. One-hot encoding throws away all similarity information. Second: with a vocabulary of 50,000 words, every vector is 50,000-dimensional with a single non-zero element. That's incredibly wasteful."

**The Solution — Learned Embeddings:**

"Instead of a sparse, high-dimensional, meaningless vector, represent each word as a short, dense vector where the numbers encode meaning. 'Cat' and 'dog' should have similar vectors because they're both animals."

```python
# Learned embeddings: each word is a dense vector (say, 4 dimensions)
# These are made up to illustrate — real ones are learned during training
embeddings = {
    'cat':   np.array([0.8, 0.2, -0.1, 0.5]),
    'dog':   np.array([0.7, 0.3, -0.2, 0.4]),
    'fish':  np.array([0.5, 0.1,  0.6, 0.3]),
    'king':  np.array([-0.3, 0.9, 0.1, 0.7]),
    'queen': np.array([-0.2, 0.8, 0.2, 0.8]),
}

# Now similarity makes sense!
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cat · dog  = {cosine_similarity(embeddings['cat'], embeddings['dog']):.4f}")   # High — similar
print(f"cat · fish = {cosine_similarity(embeddings['cat'], embeddings['fish']):.4f}")  # Medium — both animals
print(f"cat · king = {cosine_similarity(embeddings['cat'], embeddings['king']):.4f}")  # Low — unrelated
print(f"king · queen = {cosine_similarity(embeddings['king'], embeddings['queen']):.4f}")  # High — related
```

"In a good embedding space, the dot product (or cosine similarity) between two word vectors reflects how similar those words are. 'Cat' and 'dog' are close. 'King' and 'queen' are close. 'Cat' and 'king' are far apart."

**The Famous Analogy — king - man + woman ≈ queen:**

```python
# Vector arithmetic captures semantic relationships
man   = np.array([-0.4, 0.7, -0.1, 0.3])
woman = np.array([-0.3, 0.6,  0.3, 0.6])

# king - man + woman should be close to queen
result = embeddings['king'] - man + woman
print(f"king - man + woman = {result}")
print(f"queen              = {embeddings['queen']}")
print(f"Similarity: {cosine_similarity(result, embeddings['queen']):.4f}")
```

"This is the famous Word2Vec result. The direction from 'man' to 'woman' captures the concept of gender. Adding that direction to 'king' lands near 'queen.' The embedding space has learned that semantic relationships correspond to geometric directions."

**How Embeddings Are Learned:**

"An embedding is just a weight matrix. Each row is one word's vector."

```python
# An embedding layer is a matrix: vocab_size × embedding_dim
vocab_size = 5
embedding_dim = 4

# The embedding matrix — this IS the layer's weights
E = np.random.randn(vocab_size, embedding_dim) * 0.1
print(f"Embedding matrix shape: {E.shape}")  # (5, 4)

# To get the embedding for word index 2 ("fish"):
word_index = 2
embedding = E[word_index]  # Just a row lookup
print(f"Embedding for word {vocab[word_index]}: {embedding}")
```

"An embedding lookup is just indexing into a matrix. The matrix starts with random values and gets trained like any other weight matrix — through backpropagation. During training, similar words get pushed to similar vectors because they appear in similar contexts."

Reference `/tools/ml-visualizations/embeddings.py` for a 2D projection of word embeddings.

**Common Misconceptions:**
- Misconception: "Embeddings are just lookup tables" → Clarify: "They start as random numbers and are trained to encode semantic meaning. The geometry of the space IS the knowledge."
- Misconception: "You design embeddings by hand" → Clarify: "Embeddings are learned during training. The network discovers the right representation by seeing millions of examples of how words are used."
- Misconception: "Each dimension of an embedding has a clear meaning (like 'animalness')" → Clarify: "Individual dimensions don't usually have interpretable meanings. The meaning is in the relationships between vectors — distances and directions."
- Misconception: "One-hot encoding is useless" → Clarify: "One-hot encoding is the starting point. You can think of the embedding matrix as a learned transformation FROM one-hot space TO a dense, meaningful space. E @ one_hot_vector = embedding."

**Verification Questions:**
1. "Why can't we just use one-hot encoded vectors as inputs to a neural network?" (We can, but they encode no similarity — all words are equally different — and the vectors are huge for large vocabularies)
2. "What does it mean for two word vectors to have a high cosine similarity?" (The words have similar meanings or appear in similar contexts)
3. Multiple choice: "An embedding matrix for a vocabulary of 10,000 words with 256-dimensional embeddings has shape: A) (256, 10000) B) (10000, 256) C) (256, 256) D) (10000, 10000)"
4. "How are embeddings learned?" (Through backpropagation during training — the embedding matrix is a weight matrix that gets updated like any other)

**Good answer indicators:**
- They understand that one-hot encoding loses similarity information
- They can explain cosine similarity as "how aligned two vectors are"
- They answer B (10000 × 256 — one row per word)
- They connect embedding training to the backpropagation they already know

**If they struggle:**
- Connect to MNIST: "In MNIST, each pixel is a feature. For text, each embedding dimension is a feature. The network learns which features matter."
- Make it concrete: "Imagine plotting words on a 2D graph. You'd put 'cat' and 'dog' near each other, 'king' and 'queen' near each other. That's what embeddings do, but in higher dimensions."
- Stay with the matrix indexing view: "An embedding lookup is just `E[word_index]` — grab row 7 from a matrix. That's it mechanically."

**Exercise 1.1:**
"Implement a function that computes cosine similarity between two word vectors. Then create a small vocabulary of 8 words (4 animals, 4 professions) with made-up 4D embedding vectors. Compute the cosine similarity between every pair and verify that animals are more similar to animals and professions are more similar to professions."

**How to Guide Them:**
1. "The cosine similarity formula is dot(a, b) / (norm(a) * norm(b))"
2. "For the embeddings, make animals have similar vectors and professions have similar vectors"
3. If stuck: "Start with the cosine_similarity function from the example above. Then create your own embedding dictionary."

**Solution:**
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = {
    'cat':      np.array([0.9, 0.2, -0.1, 0.4]),
    'dog':      np.array([0.8, 0.3, -0.2, 0.5]),
    'parrot':   np.array([0.7, 0.1,  0.0, 0.3]),
    'hamster':  np.array([0.8, 0.2, -0.1, 0.3]),
    'doctor':   np.array([-0.3, 0.8, 0.5, -0.1]),
    'lawyer':   np.array([-0.2, 0.7, 0.6, -0.2]),
    'teacher':  np.array([-0.1, 0.9, 0.4, -0.1]),
    'engineer': np.array([-0.3, 0.6, 0.7, -0.3]),
}

words = list(embeddings.keys())
print("Cosine similarities:")
for i, w1 in enumerate(words):
    for w2 in words[i+1:]:
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"  {w1:>8s} - {w2:<8s}: {sim:.3f}")
```

**After exercise, ask:**
- "Can you see how the dot product from linear algebra is doing the heavy lifting here?"
- "What would happen if the embedding dimension were 1 instead of 4?" (Too few dimensions to capture the relationships — all words would be projected onto a single number line)

---

### Section 2: Sequence Problems and Context

**Core Concept to Teach:**
Language has order. "Dog bites man" and "man bites dog" use the same words but mean different things. A neural network that treats input as an unordered bag of vectors will miss this structure. Positional encoding solves this by adding position information to each embedding.

**How to Explain:**
1. Show that word order matters with a concrete example
2. Explain why a plain feedforward network ignores order
3. Introduce the long-range dependency problem
4. Present positional encoding as the solution

**Word Order Matters:**

```python
# Same words, different meaning
sentence1 = ["dog", "bites", "man"]   # A dog bit a man
sentence2 = ["man", "bites", "dog"]   # A man bit a dog

# If we just sum the embeddings, they're identical
emb = {
    'dog':   np.array([0.8, 0.2, 0.1]),
    'bites': np.array([0.1, 0.7, -0.3]),
    'man':   np.array([0.3, 0.5, 0.6]),
}

bag1 = sum(emb[w] for w in sentence1)
bag2 = sum(emb[w] for w in sentence2)
print(f"Sentence 1 bag-of-words: {bag1}")
print(f"Sentence 2 bag-of-words: {bag2}")
print(f"Identical? {np.allclose(bag1, bag2)}")  # True — the model can't tell them apart!
```

"If we just add up the word embeddings, 'dog bites man' and 'man bites dog' produce the same vector. The model has no way to distinguish them. This is the bag-of-words problem — it throws away word order."

**Long-Range Dependencies:**

"Order matters, but it's not just about adjacent words. Consider:

'The cat, which was sitting on the mat that the dog had been lying on earlier, **purred**.'

The verb 'purred' needs to connect back to 'cat' — not 'mat' or 'dog.' That's 13 words away. The network needs to maintain that connection across the entire sentence.

In your MNIST network, every pixel is independent — pixel (5,5) doesn't depend on pixel (20,20). In language, the meaning of a word can depend on a word far away in the sentence."

**Why Plain Feedforward Networks Struggle:**

"Your Network class from the previous routes takes a fixed-size input vector and produces a fixed-size output. Two problems with text:
1. Sentences have different lengths — you can't fix the input size
2. Even if you zero-pad to a max length, a fully connected layer treats position 1 and position 50 with different weights — it can't generalize 'the verb goes with the subject' regardless of where they appear in the sentence."

**Positional Encoding — Giving the Model a Sense of Order:**

"Since we're going to process all positions with the same weights (spoiler: that's what attention does), we need another way to encode position. The solution: add a position-dependent vector to each word's embedding."

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """Compute sinusoidal positional encoding."""
    positions = np.arange(seq_len)[:, np.newaxis]       # (seq_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]            # (1, d_model)

    # Even dimensions use sin, odd dimensions use cos
    angles = positions / (10000 ** (2 * (dims // 2) / d_model))
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Even indices
    encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Odd indices
    return encoding

# Visualize positional encoding for 20 positions, 16 dimensions
pe = positional_encoding(20, 16)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position in Sequence')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar(label='Value')
plt.show()
```

"Each row is a position's encoding vector. Each column oscillates at a different frequency — like a clock with hands spinning at different speeds. Position 0 has a specific pattern, position 1 has a slightly different one, and so on. No two positions have the same encoding."

**Why Sinusoids?**

"Two reasons:
1. Each position gets a unique vector, so the model can tell positions apart
2. Relative positions can be computed from the encoding — the encoding for 'position 5 relative to position 3' is the same as 'position 8 relative to position 6.' Sinusoids have this property because sin(a+b) and cos(a+b) can be expressed as linear combinations of sin(a), cos(a), sin(b), cos(b)."

```python
# Combining embedding + positional encoding
d_model = 8
sentence = ["The", "cat", "sat"]

# Fake embeddings for illustration
word_embeddings = np.random.randn(len(sentence), d_model) * 0.1
pe = positional_encoding(len(sentence), d_model)

# Final input = embedding + positional encoding
model_input = word_embeddings + pe

print("Word embeddings (no position info):")
print(word_embeddings)
print("\nPositional encoding:")
print(pe)
print("\nFinal input to model (embedding + position):")
print(model_input)
```

"The model's input for each word is the sum of its word embedding (what the word means) and its positional encoding (where the word is). This gives the model both pieces of information."

**Common Misconceptions:**
- Misconception: "Position is encoded by the order of inputs to the network" → Clarify: "A transformer processes all positions in parallel with shared weights. Without explicit positional encoding, it has no way to know which word came first."
- Misconception: "Positional encoding is learned" → Clarify: "The original transformer uses fixed sinusoidal encoding. Some architectures (like BERT) do learn the positional encoding. Both work."
- Misconception: "Adding position to the embedding corrupts the word meaning" → Clarify: "The model learns to use both signals. The embedding dimensions carry word meaning, the positional encoding adds position. They share the same vector space, and the model learns to separate the two signals during training."

**Verification Questions:**
1. "Why does 'dog bites man' need to be treated differently from 'man bites dog' by the model?" (They have different meanings despite using the same words — word order carries meaning)
2. "What problem does positional encoding solve?" (It tells the model where each word is in the sequence, since the model processes all positions with the same weights)
3. Multiple choice: "In a transformer, positional information is added to the input by: A) Using different weights for each position B) Encoding position as a separate input stream C) Adding a position-dependent vector to each word embedding D) Sorting the words alphabetically"
4. "What's a long-range dependency? Give an example." (When a word's meaning depends on another word far away in the sentence — like 'cat' and 'purred' with many words between them)

**Good answer indicators:**
- They understand why bag-of-words fails
- They can explain positional encoding as "adding position information to the embedding"
- They answer C
- They can give a concrete example of a long-range dependency

**If they struggle:**
- Stay concrete: "Imagine I give you three embedding vectors but don't tell you the order. Can you tell if it's 'dog bites man' or 'man bites dog'?" (No — the order information is lost.)
- Simplify positional encoding: "Just think of it as stamping each word with its position number. The sinusoidal version is a clever way to do that stamping."
- Connect to what they know: "In MNIST, position IS encoded — pixel (0,0) always goes to the same weight. But text has variable length, so we can't hardcode position into the weight structure."

**Exercise 2.1:**
"Implement sinusoidal positional encoding. Generate the encoding for a sequence of length 10 with embedding dimension 16. Visualize it as a heatmap."

**How to Guide Them:**
1. "For each position and dimension, compute the angle: position / (10000 ^ (2i / d_model))"
2. "Use sin for even dimensions, cos for odd dimensions"
3. "Use `plt.imshow()` to visualize the result"

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / (10000 ** (2 * (dims // 2) / d_model))
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding

pe = positional_encoding(10, 16)
plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding')
plt.colorbar()
plt.show()
```

**After exercise, ask:**
- "Can you see how each position has a unique pattern?"
- "What happens at different frequencies? Look at the first few columns vs. the last few."
- "We've solved two problems: turning words into vectors (embeddings) and encoding position. The big remaining question is: how does the model know which words are relevant to each other? That's attention."

---

### Section 3: The Attention Mechanism

**Core Concept to Teach:**
Attention computes a weighted average of values, where the weights come from the similarity between queries and keys. It lets each word "look at" every other word in the sentence and decide how much to attend to each one. The dot product — which the learner already knows from Route 1 — is the core operation.

**THIS IS THE KEY SECTION OF THE ENTIRE ROUTE.** Spend the most time here. The learner must be able to trace attention step by step with concrete numbers before moving on.

**How to Explain:**
1. Start with the intuition: "When you read 'The cat sat on the mat because it was tired,' how do you know 'it' refers to 'the cat'?"
2. Introduce queries, keys, and values with an analogy
3. Show scaled dot-product attention step by step with small numbers
4. Extend to multi-head attention

**The Intuition — Dynamic, Context-Dependent Connections:**

"In your MNIST network, each neuron connects to every input with fixed weights. The connections are the same regardless of the input. Attention is different: the connections change based on the input. Each word dynamically decides which other words are relevant to it.

When you read 'The cat sat on the mat because it was tired,' you know 'it' refers to 'cat,' not 'mat.' Your brain attended to 'cat' when processing 'it.' Attention does something analogous (though the mechanism is different from how brains work)."

**Queries, Keys, Values — The Database Analogy:**

"Think of attention like a database query:
- **Query (Q)**: What am I looking for? Each word generates a query — 'what information do I need?'
- **Key (K)**: What do I contain? Each word generates a key — 'this is what I offer.'
- **Value (V)**: Here's my actual content. Each word generates a value — 'here's the information to use if you attend to me.'

The dot product between a query and a key measures how well they match. High match = high attention weight = more of that value gets included in the output."

```python
import numpy as np

# Simple example: 3 words, each represented by a 4D embedding
# "The" = position 0, "cat" = position 1, "sat" = position 2

# Each word's embedding (after positional encoding)
x = np.array([
    [1.0, 0.0, 1.0, 0.0],  # "The"
    [0.0, 1.0, 0.0, 1.0],  # "cat"
    [1.0, 1.0, 0.0, 0.0],  # "sat"
])

# Weight matrices project embeddings into Q, K, V spaces
# In practice these are learned. We'll use simple ones.
W_q = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # (4, 2) -> d_k = 2
W_k = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # (4, 2)
W_v = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # (4, 2) -> d_v = 2

# Compute Q, K, V for all words at once
Q = x @ W_q   # (3, 2) — one query per word
K = x @ W_k   # (3, 2) — one key per word
V = x @ W_v   # (3, 2) — one value per word

print("Queries (Q):")
print(Q)
print("\nKeys (K):")
print(K)
print("\nValues (V):")
print(V)
```

"Each word's embedding gets multiplied by three different weight matrices to produce its query, key, and value. These are just linear projections — the same W @ x operation from your Network class. The matrices W_q, W_k, W_v are learned during training."

**Scaled Dot-Product Attention — Step by Step:**

"Now we compute attention. The formula is:
`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`

Let's break it down piece by piece."

```python
d_k = Q.shape[1]  # Dimension of keys (2 in our example)

# Step 1: Compute attention scores — Q @ K^T
# Each query dot-producted with each key
scores = Q @ K.T
print("Step 1 — Raw attention scores (Q @ K^T):")
print(scores)
print(f"Shape: {scores.shape}")  # (3, 3) — score for every query-key pair
```

"The score matrix is (num_words × num_words). Entry [i, j] is the dot product of word i's query with word j's key. High score means word i should attend strongly to word j. Remember from linear algebra: the dot product measures alignment. That's exactly what's happening here — Q and K that point in the same direction get a high score."

```python
# Step 2: Scale by sqrt(d_k)
scaled_scores = scores / np.sqrt(d_k)
print(f"\nStep 2 — Scaled scores (÷ sqrt({d_k}) = ÷ {np.sqrt(d_k):.2f}):")
print(scaled_scores)
```

"Why divide by sqrt(d_k)? Because dot products grow with vector dimension. If d_k is large, the dot products can be huge, which pushes softmax into regions where it's nearly 0 or 1 (saturated). Dividing by sqrt(d_k) keeps the values in a reasonable range."

```python
# Step 3: Apply softmax to get attention weights
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

weights = softmax(scaled_scores)
print("\nStep 3 — Attention weights (softmax of scaled scores):")
print(weights)
print(f"Row sums: {weights.sum(axis=1)}")  # Each row sums to 1
```

"Softmax converts scores into probabilities. Each row sums to 1. The highest-scoring key gets the most weight. You already know softmax from the training route — it's the same function used in classification outputs."

```python
# Step 4: Weighted sum of values
output = weights @ V
print("\nStep 4 — Attention output (weights @ V):")
print(output)
```

"The output for each word is a weighted average of all the values, where the weights come from attention. Words that got high attention scores contribute more to the output. The output has the same shape as V — one vector per word."

**Putting It All Together:**

```python
def scaled_dot_product_attention(Q, K, V):
    """Compute scaled dot-product attention."""
    d_k = K.shape[-1]
    scores = Q @ K.T                           # (seq_len, seq_len)
    scaled_scores = scores / np.sqrt(d_k)      # Scale
    weights = softmax(scaled_scores)            # Normalize to probabilities
    output = weights @ V                       # Weighted sum of values
    return output, weights

output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention output:")
print(output)
print("\nAttention weights:")
print(weights)
```

"Four operations: dot product, scale, softmax, weighted sum. Each one uses concepts you already know. The dot product measures similarity. Scaling prevents large values. Softmax converts to probabilities. The weighted sum produces the output."

**Visualizing Attention Weights:**

Reference `/tools/ml-visualizations/attention.py` for an interactive version, or:

```python
import matplotlib.pyplot as plt

words = ["The", "cat", "sat"]

plt.figure(figsize=(6, 5))
plt.imshow(weights, cmap='Blues')
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.xlabel('Attending TO (keys)')
plt.ylabel('Attending FROM (queries)')
plt.title('Attention Weight Heatmap')
plt.colorbar(label='Attention weight')

# Add text annotations
for i in range(len(words)):
    for j in range(len(words)):
        plt.text(j, i, f'{weights[i, j]:.2f}',
                ha='center', va='center', fontsize=12)
plt.show()
```

"This heatmap shows which words attend to which other words. Bright squares mean high attention. Each row sums to 1. In a trained model, you'd see patterns like pronouns attending to their referents ('it' attending to 'cat')."

**Multi-Head Attention — Multiple Perspectives:**

"One attention computation gives one way of looking at the relationships. Multi-head attention runs several attention computations in parallel, each with different learned Q/K/V projections, then concatenates the results."

```python
def multi_head_attention(x, W_q_list, W_k_list, W_v_list, W_o):
    """Multi-head attention with h heads."""
    head_outputs = []

    for W_q, W_k, W_v in zip(W_q_list, W_k_list, W_v_list):
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        head_output, _ = scaled_dot_product_attention(Q, K, V)
        head_outputs.append(head_output)

    # Concatenate all heads
    concatenated = np.concatenate(head_outputs, axis=-1)

    # Project back to original dimension
    output = concatenated @ W_o
    return output
```

"Each head can learn to attend to different things. Head 1 might attend to syntactic relationships (subject-verb). Head 2 might attend to semantic relationships (words in the same topic). Head 3 might attend to adjacent words. The final projection W_o combines all these perspectives."

**Why Multiple Heads Help:**

"A single attention head computes one set of weights per query-key pair. If word A needs to attend to word B for grammar AND word C for meaning, a single head must compromise. With multiple heads, one head can attend to B and another to C."

**Common Misconceptions:**
- Misconception: "Attention means the model pays attention like humans do" → Clarify: "It's a weighted sum based on dot-product similarity. The word 'attention' is a metaphor. The mechanism is a mathematical operation — multiply, scale, softmax, weighted sum."
- Misconception: "Q, K, V are three different inputs" → Clarify: "In self-attention, Q, K, and V all come from the same input — each word's embedding. They're just projected through different weight matrices to play different roles."
- Misconception: "The attention weights are fixed" → Clarify: "The weight matrices (W_q, W_k, W_v) are fixed after training. But the attention weights change for every input — they're computed from the specific words in the sentence."
- Misconception: "More heads always means better attention" → Clarify: "More heads give more perspectives, but each head gets fewer dimensions (the embedding dimension is split across heads). There's a trade-off."

**Verification Questions:**
1. "What does the dot product between a query and a key represent?" (How relevant that key is to that query — how much the word should attend to the other word)
2. "Why do we scale by sqrt(d_k)?" (To prevent large dot products from pushing softmax into saturation, where gradients become tiny)
3. "In a self-attention layer with 5 input tokens, what shape is the attention weight matrix?" ((5, 5) — one weight for every pair of tokens)
4. Multiple choice: "After applying softmax to attention scores, each row of the weight matrix: A) Sums to the number of tokens B) Sums to 1 C) Contains only 0s and 1s D) Is symmetric"
5. "What advantage does multi-head attention have over single-head?" (Each head can attend to different types of relationships simultaneously)

**Good answer indicators:**
- They can describe attention as "dot product for similarity, softmax for weights, weighted sum for output"
- They understand the scaling prevents softmax saturation
- They know the weight matrix shape is (seq_len, seq_len)
- They answer B (rows sum to 1 — they're probability distributions)
- They understand multi-head as multiple simultaneous perspectives

**If they struggle:**
- Return to dot products: "You already know that dot(a, b) measures how aligned a and b are. Attention uses this to measure how relevant each word is to every other word."
- Use a tiny example: 2 words, 2-dimensional embeddings, trace every number
- Separate the steps: "Don't try to understand the whole formula at once. Step 1 is just a matrix multiply. Step 2 is just dividing. Step 3 is softmax (you know this). Step 4 is another matrix multiply."
- Analogy: "Imagine you're at a party and someone says a keyword. You turn to look at (attend to) the person who's most likely to respond to that keyword. The query is the keyword, the keys are what each person is about, and the values are what they actually say."

**Exercise 3.1:**
"Implement `scaled_dot_product_attention` from scratch. Test it with Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], V = [[5, 0], [0, 5]]. Trace through every step and verify the output by hand."

**How to Guide Them:**
1. "Start with the dot product: Q @ K.T"
2. "Scale by sqrt(d_k). What is d_k here?"
3. "Apply softmax to each row"
4. "Multiply by V"
5. "Check: when Q[0] = K[0] exactly and Q[0] is orthogonal to K[1], what should the attention weight be?"

**Solution:**
```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T
    scaled = scores / np.sqrt(d_k)
    weights = softmax(scaled)
    output = weights @ V
    return output, weights

Q = np.array([[1, 0], [0, 1]], dtype=float)
K = np.array([[1, 0], [0, 1]], dtype=float)
V = np.array([[5, 0], [0, 5]], dtype=float)

output, weights = scaled_dot_product_attention(Q, K, V)

print("Scores (Q @ K^T):")
print(Q @ K.T)
print(f"\nScaled scores (÷ sqrt({K.shape[-1]})):")
print((Q @ K.T) / np.sqrt(K.shape[-1]))
print(f"\nAttention weights:")
print(weights)
print(f"\nOutput:")
print(output)
```

"Q[0] = [1, 0] matches K[0] = [1, 0] perfectly (dot product = 1) and is orthogonal to K[1] = [0, 1] (dot product = 0). After scaling and softmax, Q[0] attends mostly to K[0], so the output is close to V[0] = [5, 0]. The attention mechanism routes the right value to the right query."

**Exercise 3.2:**
"Compute attention for the sentence 'I love cats' using random 4-dimensional embeddings and random 4×2 weight matrices for Q, K, V. Visualize the attention weights as a heatmap."

**How to Guide Them:**
1. "Create 3 embedding vectors (one per word), each 4-dimensional"
2. "Create W_q, W_k, W_v — each 4×2"
3. "Compute Q = x @ W_q, K = x @ W_k, V = x @ W_v"
4. "Call your attention function"
5. "Plot the weights as a heatmap with word labels"

**After exercises, ask:**
- "Can you trace the full attention computation — from embeddings to Q/K/V to weights to output?"
- "How is the dot product in attention related to the dot product you learned in linear algebra?" (Same operation — it measures alignment/similarity between vectors)
- "What makes attention different from a regular neural network layer?" (The weights are dynamic — computed from the input — rather than fixed parameters)

---

### Section 4: The Transformer Architecture

**Core Concept to Teach:**
A transformer is attention plus standard neural network components arranged in a specific pattern. Each transformer block does: attention → add & normalize → feedforward → add & normalize. Residual connections help gradients flow. The whole thing can be trained in parallel (unlike recurrent networks), which is why transformers dominate.

**How to Explain:**
1. Show how attention fits into a larger block
2. Explain residual connections (they already know about gradient flow from the training route)
3. Walk through one transformer block
4. Briefly explain encoder vs. decoder
5. Explain why transformers train faster than recurrent networks

**The Transformer Block:**

"Attention is one piece. A transformer block wraps it with standard components:"

```python
import numpy as np

def layer_norm(x, eps=1e-6):
    """Normalize each vector to zero mean and unit variance."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def feedforward(x, W1, b1, W2, b2):
    """Two-layer feedforward network with ReLU."""
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    return hidden @ W2 + b2

def transformer_block(x, W_q, W_k, W_v, W_o, ff_W1, ff_b1, ff_W2, ff_b2):
    """One transformer block: attention → add & norm → feedforward → add & norm."""
    d_model = x.shape[-1]

    # Sub-layer 1: Self-attention
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    attn_out, _ = scaled_dot_product_attention(Q, K, V)
    attn_out = attn_out @ W_o           # Project back to d_model dimensions

    # Residual connection + layer normalization
    x = layer_norm(x + attn_out)        # Add original input, then normalize

    # Sub-layer 2: Feedforward
    ff_out = feedforward(x, ff_W1, ff_b1, ff_W2, ff_b2)

    # Residual connection + layer normalization
    x = layer_norm(x + ff_out)          # Add, then normalize

    return x
```

"Walk through each component:

1. **Self-attention**: Computes attention over the input — each word attends to every other word. This is the step that captures relationships between words.

2. **Residual connection (add)**: Add the attention output back to the original input. `x = x + attention(x)`. This is a highway for gradients — during backpropagation, the gradient can flow directly through the addition, bypassing the attention computation.

3. **Layer normalization**: Normalize each vector to zero mean and unit variance. This stabilizes training — without it, the values can drift to very large or very small numbers across layers.

4. **Feedforward network**: A standard 2-layer network (the same W @ x + b → ReLU → W @ x + b pattern from your neural network route). This processes each position independently. While attention mixes information between positions, the feedforward layer processes each position's representation.

5. **Another residual + normalization**: Same pattern again."

**Residual Connections — Gradient Highways:**

"You learned about vanishing gradients in the training route. In a deep network, gradients shrink as they flow backward through many layers. Residual connections solve this:"

```python
# Without residual connection:
# output = f(x)
# Gradient must flow through f during backprop

# With residual connection:
# output = x + f(x)
# Gradient flows through BOTH paths: directly through x (gradient = 1)
# and through f(x) (gradient = f'(x))
# Even if f'(x) is tiny, the direct path carries the gradient

# In code:
x = np.array([1.0, 2.0, 3.0])

# Some transformation (pretend this is attention or feedforward)
def f(x):
    return x * 0.1  # Tiny output — would cause vanishing gradients

# Without residual
without = f(x)
print(f"Without residual: {without}")  # [0.1, 0.2, 0.3] — signal shrinks

# With residual
with_residual = x + f(x)
print(f"With residual:    {with_residual}")  # [1.1, 2.2, 3.3] — original signal preserved
```

"The residual connection is `x + f(x)` instead of just `f(x)`. During backpropagation, the gradient of `x + f(x)` with respect to x is `1 + f'(x)`. Even if f'(x) vanishes, the gradient is at least 1. This is what makes training deep transformers (with 96+ layers) possible."

**Layer Normalization:**

```python
# Layer norm normalizes each token's vector independently
x = np.array([[10.0, 0.5, -3.0],
              [1.0, 2.0, 3.0]])

normed = layer_norm(x)
print("Before layer norm:")
print(x)
print(f"\nAfter layer norm:")
print(normed)
print(f"\nMean of each row: {np.mean(normed, axis=-1)}")  # ~0
print(f"Std of each row:  {np.std(normed, axis=-1)}")     # ~1
```

"Each row (each token's vector) gets normalized to zero mean and unit variance. This keeps the values in a stable range as data flows through many layers."

**Encoder vs. Decoder:**

"The original transformer paper has two halves:
- **Encoder**: Processes the input sequence. Each token can attend to every other token (bidirectional attention). Used in BERT.
- **Decoder**: Generates the output sequence one token at a time. Each token can only attend to previous tokens (masked/causal attention). Used in GPT.

GPT-style language models use only the decoder half. BERT-style models use only the encoder half. The original application (machine translation) used both."

```python
# Causal (decoder) attention: mask future positions
def causal_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Mask: set future positions to -infinity so softmax gives them 0 weight
    seq_len = scores.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # Upper triangle
    scores = scores - mask * 1e9  # -inf for masked positions

    weights = softmax(scores)
    return weights @ V, weights

# Example: 4 tokens
Q_ex = np.random.randn(4, 3)
K_ex = np.random.randn(4, 3)
V_ex = np.random.randn(4, 3)

_, causal_weights = causal_attention(Q_ex, K_ex, V_ex)
print("Causal attention weights:")
print(np.round(causal_weights, 3))
print("\nNotice: each row only has non-zero weights for current and previous positions")
```

"In causal attention, position 0 can only see itself. Position 1 can see positions 0 and 1. Position 3 can see everything up to and including itself. This prevents the model from 'cheating' during next-token prediction by looking at future tokens."

**Why Transformers Train Faster Than Recurrent Networks:**

"Before transformers, the dominant architecture for sequences was recurrent neural networks (RNNs/LSTMs). They process one token at a time, sequentially:

```
token_0 → RNN → hidden_state_1
token_1 + hidden_state_1 → RNN → hidden_state_2
token_2 + hidden_state_2 → RNN → hidden_state_3
...
```

This is sequential — you can't compute hidden_state_2 until hidden_state_1 is done. For a sequence of 1000 tokens, that's 1000 sequential steps.

Transformers compute attention over all tokens in parallel — it's one big matrix multiplication:

```
all_tokens → Q, K, V → Q @ K^T → softmax → @ V → done
```

This is a matrix operation that GPUs are designed for. All tokens are processed simultaneously. This parallelism is the key practical advantage of transformers — and it's why scaling to massive datasets became feasible."

**Common Misconceptions:**
- Misconception: "The transformer block is complicated" → Clarify: "It's four familiar pieces arranged in a specific pattern: attention (dot products + softmax), residual connections (addition), layer normalization, and a feedforward network (the same W @ x + b → ReLU → W @ x + b you already know). None of these components is new."
- Misconception: "Residual connections are a trick" → Clarify: "They solve a fundamental problem — vanishing gradients in deep networks. Without them, you can't train a 96-layer transformer."
- Misconception: "Encoder and decoder are always used together" → Clarify: "GPT uses only the decoder. BERT uses only the encoder. The original transformer used both because it was designed for translation."
- Misconception: "Transformers replaced RNNs because they're more powerful" → Clarify: "Transformers train much faster because of parallelism. Whether they're fundamentally more powerful is debatable — the key advantage is training efficiency at scale."

**Verification Questions:**
1. "What are the four components of a transformer block?" (Self-attention, residual connection + layer norm, feedforward network, residual connection + layer norm)
2. "What problem do residual connections solve?" (Vanishing gradients in deep networks — they provide a direct path for gradients to flow through)
3. "Why can transformers process all tokens in parallel while RNNs can't?" (Transformers use attention, which is a matrix multiply over all tokens at once. RNNs process one token at a time because each step depends on the previous hidden state.)
4. Multiple choice: "In GPT-style language models, each token can attend to: A) All tokens in the sequence B) Only the token immediately before it C) Only itself and previous tokens D) Only tokens within a fixed window"

**Good answer indicators:**
- They can list the components of a transformer block
- They connect residual connections to the vanishing gradient problem from the training route
- They understand parallelism as transformers' key practical advantage
- They answer C (causal attention — current and previous tokens only)

**If they struggle:**
- Walk through the transformer_block function line by line: "First, compute attention. Then add the input back (residual). Then normalize. Then feedforward. Then add and normalize again."
- For residual connections: "Remember from the training route when we discussed vanishing gradients? `x + f(x)` guarantees the gradient is at least 1."
- For parallelism: "Think about batch processing — your training loop can process a whole batch at once because each sample is independent. Attention treats all tokens like a batch of queries against all keys."

**Exercise 4.1:**
"Implement layer_norm from scratch. Test it with a few vectors and verify the output has zero mean and unit variance."

**How to Guide Them:**
1. "Compute the mean and standard deviation of the input"
2. "Subtract the mean, divide by the standard deviation"
3. "Add a small epsilon to the denominator to avoid division by zero"

**Solution:**
```python
def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

# Test
x = np.array([[10.0, 0.5, -3.0, 2.0],
              [1.0, 1.0, 1.0, 1.0]])
normed = layer_norm(x)
print(f"Input:\n{x}")
print(f"\nNormalized:\n{normed}")
print(f"\nMeans: {np.mean(normed, axis=-1)}")  # ~0
print(f"Stds:  {np.std(normed, axis=-1)}")     # ~1
```

**After exercise, ask:**
- "Can you trace data through a full transformer block? Input → attention → add & norm → feedforward → add & norm → output?"
- "Which parts of the transformer block are familiar from your neural network route?" (Feedforward network, activation functions, the matrix multiplications in attention)
- "Which part is genuinely new?" (Attention — the dynamic, input-dependent weighted sum)

---

### Section 5: From Transformers to Language Models

**Core Concept to Teach:**
A language model is a transformer trained to predict the next token. Scale (more parameters, more data, more compute) transforms a simple next-token predictor into something that appears to understand language. Fine-tuning and RLHF bridge the gap from "predicts likely text" to "answers questions helpfully."

**How to Explain:**
1. Show how next-token prediction works — it's just classification
2. Explain scale and emergent abilities
3. Discuss fine-tuning and RLHF
4. Address the gap between "predicts tokens" and "understands language"

**Pre-Training — The Core Task:**

"The pre-training task is dead simple: given a sequence of tokens, predict the next one. That's it."

```python
# Next-token prediction
# Input:  "The cat sat on the"
# Target: "mat"

# The model sees: ["The", "cat", "sat", "on", "the"]
# For each position, it predicts the next token:
#   "The"                    → predict "cat"
#   "The cat"                → predict "sat"
#   "The cat sat"            → predict "on"
#   "The cat sat on"         → predict "the"
#   "The cat sat on the"     → predict "mat"

# This is just classification! The "classes" are all tokens in the vocabulary.
# vocabulary_size = 50,000+ tokens
# The output layer is a linear projection to vocab_size, followed by softmax.
```

"This is the same classification problem as MNIST — but instead of 10 digit classes, there are 50,000+ token classes. The loss function is cross-entropy (which they know from the training route). The training loop is forward-loss-backward-update (which they've implemented)."

```python
# Simplified pre-training step (pseudocode in numpy terms)

# 1. Tokenize: "The cat sat" → [415, 3797, 3332]
# 2. Embed: look up each token in the embedding matrix
# 3. Add positional encoding
# 4. Pass through N transformer blocks
# 5. Project output to vocabulary size
# 6. Softmax → probability distribution over next token
# 7. Cross-entropy loss against actual next token
# 8. Backprop and update weights

# This is the SAME training loop you built for MNIST, with:
# - A different architecture (transformer instead of feedforward)
# - A different dataset (text instead of images)
# - A different number of output classes (50,000+ instead of 10)
```

"The training data is the entire internet (or large portions of it). Trillions of tokens. The model sees sentence after sentence and learns to predict what comes next. In the process, it learns grammar, facts, reasoning patterns, code syntax — anything that helps it predict the next token."

**Tokens, Not Words:**

"LLMs don't operate on words — they operate on tokens. A tokenizer splits text into subword pieces:"

```python
# Tokenization example (simplified)
# "unhappiness" → ["un", "happiness"]  or  ["un", "happ", "iness"]
# "ChatGPT" → ["Chat", "G", "PT"]
# "print('hello')" → ["print", "('", "hello", "')"]

# Why subwords?
# - Handles rare/unknown words by breaking them into known pieces
# - Keeps vocabulary size manageable (~50,000 tokens)
# - "antidisestablishmentarianism" doesn't need its own token
```

**Scale Changes Behavior:**

"Here's the surprising part. A small transformer trained on next-token prediction learns basic grammar. A medium one learns more complex patterns. A very large one — trained on trillions of tokens — starts doing things nobody explicitly trained it to do:"

```
Model size:       What it can do:
~100M params      Basic grammar, simple completion
~1B params        Coherent paragraphs, some factual recall
~10B params       Complex reasoning, code generation begins
~100B+ params     Multi-step reasoning, instruction following,
                  few-shot learning, translation without
                  being trained on parallel text
```

"These are called 'emergent abilities' — capabilities that appear at scale without being explicitly trained. The model was only ever trained to predict the next token. But predicting the next token well REQUIRES understanding grammar, facts, logic, and reasoning."

**Fine-Tuning and Instruction Following:**

"A pre-trained model is a next-token predictor. It will happily continue any text:
- Input: 'The weather today is' → Output: 'sunny and warm with a high of 72'
- Input: 'Once upon a time' → Output: 'there was a princess who lived in a castle'

But it doesn't naturally answer questions or follow instructions. That requires fine-tuning:"

```
Phase 1: Pre-training
  - Data: trillions of tokens from the internet
  - Task: predict next token
  - Result: a model that can continue any text

Phase 2: Supervised Fine-Tuning (SFT)
  - Data: thousands of (instruction, response) pairs written by humans
  - Task: predict next token (same task, different data)
  - Result: a model that formats responses to questions

Phase 3: RLHF (Reinforcement Learning from Human Feedback)
  - Data: human preferences ("response A is better than B")
  - Task: optimize for human-preferred outputs
  - Result: a model that gives helpful, harmless, honest responses
```

"The architecture doesn't change between phases. It's always the same transformer. What changes is the training data and the optimization objective."

**The Gap — Prediction vs. Understanding:**

"A fundamental question: does an LLM 'understand' language, or does it 'just' predict the next token?

Consider: to predict the next token well in 'The capital of France is ___', the model needs to encode the fact that Paris is the capital of France. To predict well in '2 + 3 = ___', it needs to encode arithmetic. To predict well in complex reasoning chains, it needs to encode logical inference.

Whether this constitutes 'understanding' is a philosophical question. What's not debatable: the mechanism is next-token prediction using the transformer architecture you've now learned. Everything else — knowledge, reasoning, instruction following — emerges from that mechanism at scale."

**Common Misconceptions:**
- Misconception: "LLMs understand language" → Clarify: "They predict the next token based on patterns learned from training data. Whether this constitutes 'understanding' is an open question. The mechanism is purely statistical pattern matching via the attention-based architecture."
- Misconception: "Bigger is always better" → Clarify: "More parameters without correspondingly more data leads to overfitting. And beyond a certain scale, the improvements become marginal. Training costs also grow enormously."
- Misconception: "LLMs are trained to answer questions" → Clarify: "They're pre-trained to predict the next token. Question-answering behavior comes from fine-tuning (SFT + RLHF) — a separate phase that adjusts the model's behavior without changing its architecture."
- Misconception: "Emergent abilities are mysterious" → Clarify: "The model was trained to predict text. Predicting text well requires encoding knowledge about the world. The 'emergence' is that encoding enough knowledge lets the model do things that look intelligent."
- Misconception: "RLHF teaches the model new knowledge" → Clarify: "RLHF adjusts the model's behavior — which responses it prefers to give. The knowledge comes from pre-training. RLHF is about alignment, not learning."

**Verification Questions:**
1. "What is the pre-training task for GPT-style models?" (Predict the next token given all previous tokens)
2. "How is next-token prediction related to the classification you did in MNIST?" (Same structure: the model outputs a probability distribution over classes. In MNIST, 10 classes (digits). In language, 50,000+ classes (tokens). Same cross-entropy loss.)
3. "What does fine-tuning change — the architecture or the behavior?" (The behavior — the architecture stays the same. The model learns to respond to instructions instead of just continuing text.)
4. Multiple choice: "Emergent abilities in LLMs arise from: A) Explicit programming of reasoning skills B) A special reasoning module in the architecture C) Scale — training a large model on vast amounts of text D) RLHF training"

**Good answer indicators:**
- They can describe pre-training as "next-token prediction with cross-entropy loss"
- They connect the output layer to classification (same as MNIST, more classes)
- They understand that fine-tuning changes behavior, not architecture
- They answer C (scale produces emergent abilities)

**If they struggle:**
- Ground it in what they know: "You trained a network to classify digits. Imagine the same thing, but with 50,000 classes (one per token) and trillions of training samples. That's pre-training."
- Make the training loop concrete: "Forward pass through the transformer. Compute cross-entropy loss against the actual next token. Backpropagate. Update weights. That's it — the same loop you built."
- For emergent abilities: "If you've ever seen autocomplete suggest surprisingly relevant text — that's a small language model. Now imagine one 1000x bigger."

---

## Practice Project

**Project Introduction:**
"Time to put the key concept from this route into code. You're going to implement scaled dot-product attention from scratch in numpy, compute attention weights for a small sentence, and visualize the attention weight matrix as a heatmap."

**Requirements:**
Present one at a time:
1. "Implement `scaled_dot_product_attention(Q, K, V)` — the function that computes attention"
2. "Create embedding vectors for a 5-word sentence"
3. "Create W_q, W_k, W_v weight matrices and compute Q, K, V"
4. "Compute attention and print the weights"
5. "Visualize the attention weights as a heatmap with word labels"
6. "Bonus: implement causal (masked) attention and show how the heatmap changes"

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work, offer to answer questions
2. **If they want guidance**: Build it step by step together, starting with the attention function
3. **If they're unsure**: Start with requirement 1 and check in

**Checkpoints During Project:**
- After the attention function: "Test with identity matrices for Q, K, V. When Q = K, what should the attention weights look like?" (Uniform — each token attends equally to all tokens)
- After computing Q, K, V: "Print the shapes. Q, K, V should each be (5, d_k)."
- After the heatmap: "Which words attend most strongly to which other words? Does it make sense?"
- After causal attention: "Compare the two heatmaps. What's different?" (The causal one has zeros above the diagonal)

**Code Review Approach:**
When reviewing their work:
1. Check the attention function: "Does it scale by sqrt(d_k)? Does softmax normalize along the right axis?"
2. Check the shapes: "Q @ K.T should be (seq_len, seq_len). The weights @ V should be (seq_len, d_v)."
3. Ask about understanding: "Can you explain what the attention weights mean for the first word?"
4. Connect to the architecture: "This attention function is the core of the transformer. Everything else — residual connections, layer norm, feedforward — wraps around this."

**If They Get Stuck:**
- On the attention function: "It's four operations: Q @ K.T (dot products), divide by sqrt(d_k), softmax, multiply by V."
- On softmax: "Remember to subtract the max for numerical stability: exp(x - max(x)) / sum(exp(x - max(x)))."
- On the heatmap: "Use plt.imshow() with the weights matrix. Add labels with plt.xticks() and plt.yticks()."
- On causal masking: "Create an upper-triangular matrix of ones. Subtract a large number (1e9) from the scores at those positions before softmax."

**Full Solution (for reference):**
```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=False):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask:
        seq_len = scores.shape[0]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores - causal_mask * 1e9

    weights = softmax(scores)
    output = weights @ V
    return output, weights

# Sentence: "The cat sat on mat"
words = ["The", "cat", "sat", "on", "mat"]
seq_len = len(words)
d_model = 8
d_k = 4

# Random embeddings (in practice, these are learned)
np.random.seed(42)
embeddings = np.random.randn(seq_len, d_model)

# Add positional encoding
positions = np.arange(seq_len)[:, np.newaxis]
dims = np.arange(d_model)[np.newaxis, :]
angles = positions / (10000 ** (2 * (dims // 2) / d_model))
pe = np.zeros((seq_len, d_model))
pe[:, 0::2] = np.sin(angles[:, 0::2])
pe[:, 1::2] = np.cos(angles[:, 1::2])
x = embeddings + pe

# Weight matrices
W_q = np.random.randn(d_model, d_k) * 0.1
W_k = np.random.randn(d_model, d_k) * 0.1
W_v = np.random.randn(d_model, d_k) * 0.1

# Compute Q, K, V
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# Compute attention
output, weights = scaled_dot_product_attention(Q, K, V)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regular attention
axes[0].imshow(weights, cmap='Blues')
axes[0].set_xticks(range(seq_len))
axes[0].set_yticks(range(seq_len))
axes[0].set_xticklabels(words)
axes[0].set_yticklabels(words)
axes[0].set_xlabel('Attending TO (keys)')
axes[0].set_ylabel('Attending FROM (queries)')
axes[0].set_title('Full Attention')
for i in range(seq_len):
    for j in range(seq_len):
        axes[0].text(j, i, f'{weights[i,j]:.2f}',
                    ha='center', va='center', fontsize=9)

# Causal attention
_, causal_weights = scaled_dot_product_attention(Q, K, V, mask=True)
axes[1].imshow(causal_weights, cmap='Blues')
axes[1].set_xticks(range(seq_len))
axes[1].set_yticks(range(seq_len))
axes[1].set_xticklabels(words)
axes[1].set_yticklabels(words)
axes[1].set_xlabel('Attending TO (keys)')
axes[1].set_ylabel('Attending FROM (queries)')
axes[1].set_title('Causal Attention (GPT-style)')
for i in range(seq_len):
    for j in range(seq_len):
        axes[1].text(j, i, f'{causal_weights[i,j]:.2f}',
                    ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.show()
```

**Expected Results:**
- Full attention: each row sums to 1, all entries can be non-zero
- Causal attention: upper triangle is zero, each row still sums to 1
- The heatmaps visually show which words attend to which other words

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- Embeddings represent words as dense vectors where geometry encodes meaning — similar words are close in vector space
- Positional encoding adds position information because transformers process all positions with shared weights
- Attention computes a weighted average of values based on query-key similarity — the dot product (from Route 1!) is the core operation
- A transformer block is attention + residual connection + layer norm + feedforward + residual connection + layer norm
- LLMs are transformers trained to predict the next token. Scale produces emergent abilities. Fine-tuning and RLHF shape behavior.

**Ask them to explain one concept:**
"Can you trace the attention mechanism step by step — from embeddings through Q/K/V to the output?"
(This is the core skill of this route. If they can do it, they understand the building block that makes LLMs work.)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel about how LLMs work?"

**Respond based on answer:**
- 1-4: "Attention is the conceptually dense part. Go back to the small numerical example — trace Q @ K.T, scale, softmax, multiply by V. Once attention clicks, the rest is familiar components arranged in a specific pattern."
- 5-7: "You've got the architecture. The best way to deepen understanding is to read the original 'Attention Is All You Need' paper — you now have the background to understand it. Try implementing a full transformer block."
- 8-10: "You understand the architecture behind the most impactful technology of the decade. You can read transformer papers, understand model architectures, and reason about how LLMs work. Consider exploring PyTorch's transformer implementation to see how this scales to production."

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice more: Implement a full transformer block in numpy — attention + residual + layer norm + feedforward"
- "To go deeper: Read 'Attention Is All You Need' (the original 2017 paper). You now have the vocabulary and understanding to follow it."
- "For the big picture: [Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md) — reflect on how transformer components map to the building blocks you've learned"
- "To see it in practice: Look at the nanoGPT project by Andrej Karpathy — a minimal GPT implementation in ~300 lines of PyTorch"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there a part of the transformer architecture that still feels shaky?"
"Anything about how LLMs work that you want to discuss further?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Can't trace attention step by step
- Confused about Q/K/V — what each one represents
- Losing track of shapes (seq_len, d_model, d_k)
- Overwhelmed by the number of components in a transformer block

**Strategies:**
- Reduce to the absolute minimum: 2 words, 2-dimensional embeddings, trace every number
- Stay with the database analogy: "Query = what am I looking for? Key = what do I contain? Value = what's my content."
- Focus on dot products: "You already know dot products measure similarity. That's ALL the attention score is — a dot product."
- Break the transformer block into individual pieces: "Forget the full block. Can you do attention? Good. Can you do residual connection? Good. Can you do layer norm? Good. Now chain them."
- If shapes confuse them: "Q is (num_words, d_k). K is (num_words, d_k). Q @ K.T is (num_words, num_words). The d_k's match and disappear. You've seen this before with matrix shapes."
- Let them use the code to verify every step — don't require hand computation until the pattern is clear

### If Learner is Excelling

**Signs:**
- Implements attention quickly and correctly
- Asks about position-specific details (RoPE, ALiBi, relative position encoding)
- Wants to know about specific architectures (GPT, BERT, Llama)
- Asks about efficiency (FlashAttention, KV caching)

**Strategies:**
- Move faster through the conceptual material, focus on implementation
- Discuss modern positional encoding methods: RoPE (rotary position encoding) used in Llama
- Explain KV caching: during generation, you don't need to recompute K and V for previous tokens
- Discuss the quadratic cost of attention (seq_len^2) and why long context is expensive
- Challenge: "Implement multi-head attention from scratch" or "Implement a full transformer block"
- Preview: "The next frontier is mixture-of-experts, retrieval-augmented generation, and multimodal models — all built on the transformer foundation you now understand"
- Read nanoGPT together — they can now understand every line

### If Learner Seems Disengaged

**Signs:**
- Short responses, not asking questions
- Rushing through exercises without engaging
- Seems uninterested in the architecture details

**Strategies:**
- Jump to the "why it matters" perspective: "Every AI tool you use — ChatGPT, Copilot, image generators — is built on this architecture"
- Show attention patterns from a real model — visualize what a trained model actually attends to
- Connect to their practical interests: "Understanding how transformers work helps you debug prompts, understand model limitations, and build better AI-powered applications"
- Make it hands-on: run the attention computation on their own text, visualize the weights
- If they're interested in the practical side more than theory: focus on Section 5 (how models are trained and fine-tuned)

### Different Learning Styles

**Visual learners:**
- Use the attention heatmap visualization extensively
- Show the positional encoding heatmap
- Draw the transformer block as a flow diagram: boxes for each component, arrows for data flow
- Reference `/tools/ml-visualizations/attention.py` for interactive visualizations

**Hands-on learners:**
- Code first, explain after
- "Implement attention, run it, then let's understand why it works"
- Exercise-driven: less lecture, more building and experimenting
- Modify parameters and observe: "What happens if you double d_k?"

**Conceptual learners:**
- Spend more time on why: why attention instead of RNNs, why multi-head, why scale matters
- Discuss the information-theoretic view: attention computes a dynamic, context-dependent representation
- Go deeper on emergent abilities: what does it mean for capabilities to emerge from scale?
- They may want to discuss the philosophical question of whether LLMs "understand"

**Example-driven learners:**
- Trace through specific numbers for every concept before generalizing
- Use the "The cat sat on the mat" example as a running thread
- Every formula gets a concrete computation before any abstract notation
- Build up from 2-word examples to full sentences

---

## Troubleshooting Common Issues

### Shape Mismatches in Attention

**"shapes not aligned" during Q @ K.T:**
```python
# Q should be (seq_len, d_k), K should be (seq_len, d_k)
# Q @ K.T should be (seq_len, seq_len)
print(f"Q shape: {Q.shape}, K shape: {K.shape}")

# Common error: W_q has wrong shape
# W_q should be (d_model, d_k), so x @ W_q gives (seq_len, d_k)
```

"If Q and K have different d_k dimensions, the transpose won't align. Check that W_q and W_k project to the same dimension."

### Softmax Producing NaN or All Zeros

```python
# Problem: large values in the scores cause exp() to overflow
scores = np.array([[100, 200, 300]])
print(np.exp(scores))  # [overflow, overflow, overflow]

# Fix: subtract the max before exponentiating
scores_stable = scores - np.max(scores, axis=-1, keepdims=True)
print(np.exp(scores_stable))  # [exp(-200), exp(-100), exp(0)] — no overflow
```

"Always subtract the max before computing exp in softmax. This is numerically identical (it cancels out in the division) but prevents overflow."

### Causal Mask Not Working

"If future positions still have non-zero attention weights:
1. Check the mask is upper-triangular (np.triu with k=1)
2. Check you're subtracting a large number (1e9), not adding
3. Check the mask is applied BEFORE softmax, not after"

### Attention Weights All Uniform

"If every word attends equally to every other word:
1. Check that Q and K are actually different (not all zeros)
2. Check that d_k isn't too large — large d_k with small score values means softmax produces near-uniform weights
3. If using random weights, try a different seed — some initializations produce degenerate behavior"

### Concept-Specific Confusion

**If confused about Q/K/V:**
- "Forget the names. There are three projections of the input. The first two get dot-producted to compute similarity (that's Q and K). The third gets weighted by those similarities (that's V)."
- Code it step by step: "Q = x @ W_q. That's just a matrix multiply. Same for K and V. Three matrix multiplies."

**If confused about why attention uses dot products:**
- "You already know dot products measure alignment. Two vectors pointing the same way = high dot product = high similarity. That's EXACTLY what we want: words that are 'relevant' to each other should have high similarity."

**If confused about multi-head:**
- "Run attention once with one set of W_q, W_k, W_v. Now run it again with different W_q, W_k, W_v. Each run finds different relationships. Concatenate the results."

**If confused about the whole pipeline:**
- Trace the full flow for one sentence: "Text → tokenize → embed → add position → attention → add & norm → feedforward → add & norm → output logits → softmax → predicted next token"

---

## Teaching Notes

**Key Emphasis Points:**
- The attention mechanism is the core innovation of this route. Everything else — embeddings, positional encoding, residual connections, layer norm, feedforward — is either motivation for attention or infrastructure around it. Invest the most time in Section 3.
- The dot product is the bridge from Route 1 to this route. The learner already knows that dot products measure similarity. Attention uses this to compute which words are relevant to which other words. Keep making this connection.
- The transformer block is built from familiar pieces. Don't let it seem alien — point out every component they've seen before: matrix multiplication, ReLU, softmax, residual connections.
- Connect pre-training to the training loop they built. The LLM training loop is the same forward-loss-backward-update pattern. The architecture changed, not the learning process.

**Pacing Guidance:**
- Section 1 (embeddings) should be relatively quick — it's setup for the main event. Spend enough time for them to understand why words need vector representations, but don't linger.
- Section 2 (sequences and position) is transitional. Make sure they understand WHY position encoding is needed before showing HOW.
- Section 3 (attention) is where the learner will spend the most time. Budget at least 40% of the session here. The step-by-step numerical trace is critical — don't skip it.
- Section 4 (transformer architecture) can be faster if they understand attention. The components are familiar. Focus on how they fit together.
- Section 5 (LLMs) is the payoff — connecting the architecture to the systems they've heard about. It can be more conversational and shorter on code.
- Allow plenty of time for the practice project — implementing attention cements the core concept.

**Success Indicators:**
You'll know they've got it when they:
- Can trace attention step by step with concrete numbers: scores → scale → softmax → weighted sum
- Explain attention as "dot product for similarity, softmax for weights, weighted sum of values"
- Connect the dot product in attention to the dot product from linear algebra
- Describe a transformer block as "familiar components (feedforward, residual, normalization) wrapped around attention"
- Explain LLMs as "transformers trained on next-token prediction, then fine-tuned"
- Ask questions about specific models, training details, or efficiency — this means they've internalized the fundamentals

**Most Common Confusion Points:**
1. **Q/K/V roles**: What each one represents and why there are three separate projections. Remedy: the database analogy and the step-by-step trace with concrete numbers.
2. **Why scale by sqrt(d_k)**: It feels arbitrary. Remedy: "Without scaling, large d_k makes dot products large, which pushes softmax to extremes (one weight near 1, rest near 0). Scaling keeps the values moderate."
3. **Residual connections**: Why adding the input back helps. Remedy: connect to vanishing gradients from the training route. "The gradient is at least 1 through the addition."
4. **Emergence**: How next-token prediction produces reasoning. Remedy: "Predicting well REQUIRES encoding knowledge. The reasoning is a side effect of prediction at scale."

**Teaching Philosophy:**
- The learner has built a neural network from scratch and trained it on MNIST. This route is about showing them how the SAME fundamental operations (matrix multiplication, activation, training) compose into the architecture behind LLMs.
- Attention is the one genuinely new concept. Everything else is recombination of things they already know. Frame it that way.
- Code is the primary medium. Every concept gets a numpy implementation. The code IS the understanding.
- Visualization is essential for attention. The heatmap of attention weights should make the abstract concept concrete.
- This route succeeds when the learner reads "transformer architecture" and thinks "attention, residual connections, layer norm, feedforward — I know what each of those does" instead of "mysterious black box AI."
