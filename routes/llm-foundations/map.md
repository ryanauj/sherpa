---
title: LLM Foundations
topics:
  - Embeddings
  - Attention
  - Transformers
  - Language Models
related_routes:
  - training-and-backprop
  - neural-network-foundations
  - linear-algebra-essentials
---

# LLM Foundations - Route Map

## Overview

This route teaches how large language models work — from word embeddings to attention to the transformer architecture. You'll understand the building blocks that make LLMs possible, see how attention computes relationships between words, and connect everything back to the neural network fundamentals you already know.

## What You'll Learn

By following this route, you will:
- Explain how embeddings represent words as vectors with meaning
- Describe the problem that attention solves (long-range dependencies in sequences)
- Trace the attention mechanism step by step with concrete numbers
- Explain the transformer architecture and how its components fit together
- Understand how transformers scale into large language models

## Prerequisites

Before starting this route:
- **Required**: [Training and Backprop](/routes/training-and-backprop/map.md) (training, loss, backpropagation)
- **Required**: [Neural Network Foundations](/routes/neural-network-foundations/map.md) (layers, forward pass)
- **Required**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (dot products, matrix multiplication)
- **Helpful**: Curiosity about how ChatGPT-style models work

## Route Structure

### 1. Embeddings
- Representing words as vectors (why one-hot encoding falls short)
- Semantic meaning in vector space: king - man + woman = queen
- How embeddings are learned during training

### 2. Sequence Problems and Context
- Why word order matters: "dog bites man" vs "man bites dog"
- The challenge of long-range dependencies
- Why plain neural networks struggle with sequences
- Positional encoding: giving the model a sense of order

### 3. The Attention Mechanism
- Queries, keys, and values — what each one does
- Computing attention weights with dot products
- Scaled dot-product attention step by step
- Multi-head attention: looking at the input from multiple perspectives

### 4. The Transformer Architecture
- Encoder and decoder blocks
- Layer normalization and residual connections
- How components connect: embedding → attention → feedforward → output
- Why transformers train faster than recurrent networks

### 5. From Transformers to Language Models
- Pre-training: predicting the next token on massive text
- How scale changes behavior (emergent abilities)
- Fine-tuning and instruction following
- The gap between "understands next token" and "answers questions"

### 6. Practice Project — Minimal Attention in Numpy
- Implement scaled dot-product attention from scratch
- Compute attention weights for a small example sentence
- Visualize the attention weight matrix as a heatmap

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Visualization scripts in `/tools/ml-visualizations/` (embeddings.py, attention.py)
- numpy for matrix computations
- matplotlib for attention heatmaps

## Next Steps

After completing this route:
- **[Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md)** — The ascent project: reflect on how transformer components map to what you built
- Explore transformer implementations in PyTorch or JAX
- Read "Attention Is All You Need" (the original transformer paper) — you'll now have the background to understand it
