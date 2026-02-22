---
title: Calculus for ML
topics:
  - Derivatives
  - Gradients
  - Chain Rule
  - Gradient Descent
related_routes:
  - linear-algebra-essentials
  - neural-network-foundations
---

# Calculus for ML - Route Map

## Overview

This route teaches the calculus concepts that power machine learning — derivatives, gradients, the chain rule, and gradient descent — through visual intuition and Python code. You'll see loss surfaces, watch gradient descent find minimums, and understand why these ideas matter before touching an equation.

## What You'll Learn

By following this route, you will:
- Understand derivatives as rates of change and slopes of curves
- Compute partial derivatives and gradients for functions of multiple variables
- Apply the chain rule to compose functions (the foundation of backpropagation)
- Implement gradient descent from scratch and watch it optimize a function
- Build a linear regression model using only gradient descent and numpy

## Prerequisites

Before starting this route:
- **Required**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (vectors, dot products, matrices)
- **Required**: Python basics and comfort with numpy arrays
- **Helpful**: Any prior exposure to the idea of a "slope"

## Route Structure

### 1. Derivatives as Rates of Change
- What a derivative measures: how fast something changes
- Computing derivatives numerically (the finite difference trick)
- Visualizing the tangent line to a curve

### 2. Partial Derivatives and Gradients
- Functions of multiple inputs (like ML loss functions)
- Partial derivatives: holding everything else constant
- The gradient vector: which direction increases the function fastest

### 3. The Chain Rule
- Composing functions: feeding one function's output into another
- The chain rule: how derivatives flow through composed functions
- Why this matters for neural networks (preview of backpropagation)

### 4. Gradient Descent
- The core idea: walk downhill on the loss surface
- Learning rate: how big your steps are
- Watching gradient descent converge on a 2D contour plot
- Local minima, saddle points, and why they matter

### 5. Practice Project — Linear Regression from Scratch
- Define a loss function (mean squared error)
- Compute gradients by hand and verify numerically
- Train a linear model with gradient descent

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Visualization scripts in `/tools/ml-visualizations/` (derivatives.py, gradient_descent.py)
- numpy for computation
- matplotlib for plotting loss surfaces and gradient paths

## Next Steps

After completing this route:
- **[Neural Network Foundations](/routes/neural-network-foundations/map.md)** — Perceptrons, layers, and the forward pass
- **[Training and Backprop](/routes/training-and-backprop/map.md)** — Loss functions and backpropagation (uses chain rule heavily)
