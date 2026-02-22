---
title: Linear Algebra Essentials
topics:
  - Vectors
  - Matrices
  - Dot Products
  - Linear Transformations
related_routes:
  - linear-algebra-deep-dive
  - calculus-for-ml
---

# Linear Algebra Essentials - Route Map

## Overview

This route teaches the linear algebra you need for machine learning — vectors, matrices, dot products, and linear transformations — through geometric visualization and Python code. Every concept is shown visually before any equation appears.

## What You'll Learn

By following this route, you will:
- Represent data as vectors and understand what operations on them mean geometrically
- Compute dot products and interpret them as similarity measures
- Understand matrices as transformations that warp space
- Perform matrix multiplication and understand why it works the way it does
- Build a 2D transformation sandbox that visualizes matrix operations

## Prerequisites

Before starting this route:
- **Required**: Python basics (variables, loops, functions, lists)
- **Required**: Comfort running Python scripts from the terminal
- **Helpful**: High school algebra (variables, equations)

## Route Structure

### 1. Vectors as Arrows and Data
- What a vector is: a list of numbers, an arrow in space, a point
- Vector addition, subtraction, and scaling — visually
- Why ML uses vectors to represent everything

### 2. Dot Products and Similarity
- Computing a dot product by hand and in code
- Geometric meaning: projection and angle between vectors
- Dot products as a measure of similarity

### 3. Matrices as Transformations
- A matrix as a function that takes a vector in and sends a vector out
- Visualizing what different matrices do to a grid of points
- Identity, scaling, rotation, and shear matrices

### 4. Matrix Operations
- Matrix-vector multiplication as applying a transformation
- Matrix-matrix multiplication as composing transformations
- Transpose and its geometric meaning

### 5. Practice Project — 2D Transformation Sandbox
- Build an interactive script that applies user-chosen matrices to a grid
- See how combining transformations (matrix multiplication) works visually

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Visualization scripts in `/tools/ml-visualizations/` (vectors.py, matrices.py, linear_transforms.py)
- numpy for array operations
- matplotlib for plotting

## Next Steps

After completing this route:
- **[Calculus for ML](/routes/calculus-for-ml/map.md)** — Derivatives, gradients, and gradient descent
- **[Linear Algebra Deep Dive](/routes/linear-algebra-deep-dive/map.md)** — Eigenvalues, SVD, and vector spaces (optional side quest)
