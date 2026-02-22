---
title: "Neural Net from Scratch"
routes:
  - linear-algebra-essentials
  - calculus-for-ml
  - neural-network-foundations
  - training-and-backprop
  - llm-foundations
---

# Neural Net from Scratch

## Overview

This ascent guides you through building a neural network from scratch in Python — using only numpy, no ML frameworks. You'll start with basic linear algebra utilities and end with a network that classifies handwritten digits. Each checkpoint corresponds to a route: learn the math and concepts, then write the code.

## The Summit

A working neural network that:
- Performs matrix operations with your own linear algebra module
- Finds function minimums using gradient descent
- Computes a forward pass through multiple layers with activations
- Trains using backpropagation
- Classifies MNIST digits 0-4 at >80% accuracy
- Uses nothing beyond numpy — every operation is something you understand

## Prerequisites

No ML experience required — this ascent builds it from the ground up. You should have:
- Python experience (comfortable writing classes, using numpy)
- Willingness to work through math visually and by hand
- A working Python 3 environment with numpy and matplotlib installed

## Route Checkpoints

### Checkpoint 1: Linear Algebra Toolkit → Route: linear-algebra-essentials

**What you'll learn**: Vectors, matrices, dot products, and linear transformations — all visualized geometrically.

**Apply to the project**:
- Create `linalg.py` with functions: `dot(a, b)`, `matmul(A, B)`, `transpose(A)`
- Write tests that verify your implementations against numpy's built-in functions
- These are the building blocks everything else depends on

**Milestone**: A tested linear algebra module where you understand every operation geometrically.

---

### Checkpoint 2: Optimization Engine → Route: calculus-for-ml

**What you'll learn**: Derivatives, gradients, the chain rule, and gradient descent.

**Apply to the project**:
- Create `optimize.py` with functions: `numerical_gradient(f, x)`, `gradient_descent(f, x0, lr, steps)`
- Test `numerical_gradient` against known analytical derivatives
- Test `gradient_descent` by finding the minimum of a simple quadratic function

**Milestone**: A working optimizer that finds the minimum of any differentiable function.

---

### Checkpoint 3: Network Structure → Route: neural-network-foundations

**What you'll learn**: Perceptrons, activation functions, layers, and the forward pass.

**Apply to the project**:
- Create `network.py` with a `Network` class
- Implement `__init__` to set up layers with random weights and biases
- Implement `forward(X)` that passes data through all layers
- Implement activation functions (sigmoid and ReLU) as helper functions
- Test that the forward pass produces output of the correct shape

**Milestone**: A Network object that computes (nonsense) output — the structure is right, it just hasn't learned anything yet.

---

### Checkpoint 4: Training → Route: training-and-backprop

**What you'll learn**: Loss functions, backpropagation, and training loops.

**Apply to the project**:
- Add `cross_entropy_loss(predictions, targets)` to your network
- Implement `backward()` that computes gradients for every weight and bias using the chain rule
- Implement `train(X, y, lr, epochs, batch_size)` that runs the full training loop
- Download MNIST data using `download_data.py`
- Train your network on digits 0-4
- Plot the training loss over time

**Milestone**: Your network classifies handwritten digits 0-4 at >80% accuracy. You built every piece yourself.

---

### Checkpoint 5: Reflection → Route: llm-foundations

**What you'll learn**: Embeddings, attention, transformers — how the building blocks compose into LLMs.

**Apply to the project**:
- No code for this checkpoint — this is a conceptual exercise
- Map transformer components to what you built: where does matrix multiplication appear? Where does the chain rule show up? How is attention different from a dense layer?
- Write a short reflection (in comments or a markdown file) connecting your network to the transformer architecture
- Consider: what would you need to add to your network to handle sequences?

**Milestone**: Conceptual understanding of how your from-scratch network relates to the models behind ChatGPT.

## Summit Review

Congratulations — you've built a neural network from nothing but numpy and understanding. Your project:
- [x] Performs linear algebra with functions you wrote and tested
- [x] Finds function minimums using gradient descent
- [x] Computes forward passes through multi-layer networks
- [x] Trains using backpropagation with the chain rule
- [x] Classifies handwritten digits at >80% accuracy
- [x] Uses no ML frameworks — you understand every line

## Extending the Ascent

Ideas for going further:
- Add more activation functions (tanh, leaky ReLU) and compare training results
- Train on all 10 MNIST digits instead of just 0-4
- Implement momentum or Adam optimizer in your gradient descent
- Add dropout regularization
- Try a convolutional layer (much harder, but powerful for image tasks)
- Implement mini-batch gradient descent with shuffling
