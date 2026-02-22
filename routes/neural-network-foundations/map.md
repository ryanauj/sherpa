---
title: Neural Network Foundations
topics:
  - Perceptrons
  - Activation Functions
  - Layers
  - Forward Pass
related_routes:
  - linear-algebra-essentials
  - calculus-for-ml
  - training-and-backprop
---

# Neural Network Foundations - Route Map

## Overview

This route teaches what neural networks actually are — from a single perceptron to multi-layer networks — through visualization and code. You'll see how layers of simple math operations warp space to create decision boundaries, and build a forward pass from scratch.

## What You'll Learn

By following this route, you will:
- Understand a perceptron as a weighted sum plus a decision threshold
- Explain what activation functions do and why networks need them
- Trace data through a multi-layer network by hand and in code
- Visualize how each layer transforms the input space
- Build a Network class with a working forward pass

## Prerequisites

Before starting this route:
- **Required**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (matrix-vector multiplication)
- **Required**: [Calculus for ML](/routes/calculus-for-ml/map.md) (derivatives, gradients)
- **Helpful**: Comfort reading Python classes

## Route Structure

### 1. The Perceptron
- A single neuron: weights, bias, activation
- The perceptron as a linear classifier (a line that separates two classes)
- Why a single perceptron can't solve XOR

### 2. Activation Functions
- Why linear operations alone aren't enough
- Sigmoid, ReLU, and tanh — what each looks like and when to use it
- Activation functions as "squishing" or "gating" operations

### 3. Layers and the Forward Pass
- Stacking perceptrons into a layer (matrix multiplication + activation)
- The forward pass: data flows from input to output, layer by layer
- Tracing a concrete input through a 2-layer network by hand

### 4. Multi-Layer Networks
- How depth creates power: each layer warps space further
- Visualizing 2D decision boundaries as layers transform the space
- The representation learning perspective: layers learn useful features

### 5. Practice Project — Network Anatomy Visualization
- Build a Network class that computes a forward pass
- Visualize what each layer does to a grid of 2D input points
- Experiment with different architectures (more layers, more neurons)

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Visualization scripts in `/tools/ml-visualizations/` (perceptron.py, activation_functions.py, decision_boundaries.py)
- numpy for matrix operations
- matplotlib for visualizing decision boundaries and space transformations

## Next Steps

After completing this route:
- **[Training and Backprop](/routes/training-and-backprop/map.md)** — How networks learn: loss functions, backpropagation, and training loops
- **[LLM Foundations](/routes/llm-foundations/map.md)** — How these building blocks compose into language models
