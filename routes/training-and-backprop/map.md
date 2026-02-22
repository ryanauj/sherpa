---
title: Training and Backpropagation
topics:
  - Loss Functions
  - Backpropagation
  - Training Loops
  - Optimization
related_routes:
  - neural-network-foundations
  - calculus-for-ml
  - linear-algebra-essentials
  - llm-foundations
---

# Training and Backpropagation - Route Map

## Overview

This route teaches how neural networks learn — loss functions measure how wrong the network is, backpropagation computes which direction to adjust each weight, and training loops repeat the process until the network gets good. You'll trace backprop through a small network by hand and train a network on real data.

## What You'll Learn

By following this route, you will:
- Define and compute loss functions (MSE, cross-entropy)
- Trace backpropagation step by step through a small network
- Implement backward() from scratch using the chain rule
- Build a complete training loop with batching and learning rate
- Train a neural network to classify handwritten digits

## Prerequisites

Before starting this route:
- **Required**: [Neural Network Foundations](/routes/neural-network-foundations/map.md) (forward pass, layers, activations)
- **Required**: [Calculus for ML](/routes/calculus-for-ml/map.md) (chain rule, gradients, gradient descent)
- **Required**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (matrix operations)

## Route Structure

### 1. Loss Functions
- What a loss function measures: the gap between prediction and truth
- Mean squared error for regression
- Cross-entropy loss for classification
- Visualizing the loss surface

### 2. Backpropagation
- The key insight: the chain rule applied layer by layer, backwards
- Tracing gradients through a 2-layer network by hand
- Computing weight gradients: how much each weight contributed to the error

### 3. The Training Loop
- Forward pass → compute loss → backward pass → update weights
- Learning rate and its effect on convergence
- Batching: why we don't use the whole dataset at once
- Epochs and when to stop training

### 4. Training in Practice
- Monitoring loss over time (the training curve)
- Overfitting and underfitting: what they look like
- Simple regularization techniques
- Debugging a training run that isn't converging

### 5. Practice Project — Train on MNIST Digits
- Use the Network class from the previous route
- Add backward() and train() methods
- Train on a subset of MNIST (digits 0-4)
- Achieve >80% accuracy

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Visualization scripts in `/tools/ml-visualizations/` (loss_surfaces.py, backprop_trace.py)
- MNIST data download script in `/ascents/neural-net-from-scratch/download_data.py`
- numpy for all computation (no ML frameworks)

## Next Steps

After completing this route:
- **[LLM Foundations](/routes/llm-foundations/map.md)** — How these building blocks compose into language models
- **[Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md)** — The ascent project tying everything together
