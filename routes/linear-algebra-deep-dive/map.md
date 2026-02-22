---
title: Linear Algebra Deep Dive
topics:
  - Vector Spaces
  - Eigenvalues
  - Eigenvectors
  - SVD
related_routes:
  - linear-algebra-essentials
  - calculus-for-ml
---

# Linear Algebra Deep Dive - Route Map

## Overview

This side-quest route goes deeper into linear algebra — vector spaces, eigenvalues, eigenvectors, and singular value decomposition (SVD). These concepts appear throughout ML (PCA, recommendation systems, stable diffusion) but aren't required for the main learning path. Take this route when you're curious about why certain ML techniques work the way they do.

## What You'll Learn

By following this route, you will:
- Understand vector spaces, basis, and dimension
- Compute eigenvalues and eigenvectors and explain what they mean geometrically
- Decompose a matrix using eigendecomposition
- Apply SVD to compress an image
- Connect these concepts to ML applications (PCA, latent spaces)

## Prerequisites

Before starting this route:
- **Required**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (vectors, matrices, transformations)
- **Helpful**: [Calculus for ML](/routes/calculus-for-ml/map.md) (helpful for understanding optimization context)

## Route Structure

### 1. Vector Spaces and Basis
- What a vector space is (and isn't)
- Basis vectors: the minimum set that spans the space
- Dimension and why it matters
- Changing basis: same vector, different coordinates

### 2. Eigenvalues and Eigenvectors
- The key question: which vectors don't change direction when transformed?
- Computing eigenvalues and eigenvectors for 2x2 matrices
- Geometric interpretation: stretching along special directions

### 3. Eigendecomposition
- Decomposing a matrix into its eigenvalues and eigenvectors
- What decomposition reveals about a transformation
- When eigendecomposition works (and when it doesn't)

### 4. Singular Value Decomposition (SVD)
- SVD as a generalization that always works
- The three matrices: U, Σ, V^T — what each one means
- Low-rank approximation: keeping only the important parts

### 5. Practice Project — Image Compression with SVD
- Load a grayscale image as a matrix
- Compute the SVD
- Reconstruct the image using fewer singular values
- Visualize the quality vs. compression tradeoff

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- numpy (np.linalg.eig, np.linalg.svd) for computations
- matplotlib for visualizing transformations and image compression
- Visualization scripts from `/tools/ml-visualizations/` (reuses linear_transforms.py from linear-algebra-essentials)

## Next Steps

After completing this route:
- Explore PCA (Principal Component Analysis) — direct application of eigendecomposition
- Read about latent spaces in generative models — SVD intuition applies directly
- Look into spectral methods in graph neural networks — eigenvalues of graph matrices
