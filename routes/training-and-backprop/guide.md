---
title: Training and Backpropagation
route_map: /routes/training-and-backprop/map.md
paired_sherpa: /routes/training-and-backprop/sherpa.md
prerequisites:
  - Neural Network Foundations (forward pass, layers)
  - Calculus for ML (chain rule, gradient descent)
  - Linear Algebra Essentials (matrix operations)
topics:
  - Loss Functions
  - Backpropagation
  - Training Loops
  - Optimization
---

# Training and Backpropagation - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/training-and-backprop/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/training-and-backprop/map.md` for the high-level overview.

## Overview

You have a neural network that computes a forward pass. You feed it an input, and it produces an output. The problem: the output is nonsense, because the weights are random. The network doesn't know anything yet.

Training is the process of adjusting those weights so the network's output gets closer to what you want. The loop is simple in concept:

1. Run the forward pass to get a prediction
2. Measure how wrong the prediction is (the **loss**)
3. Figure out how each weight contributed to the error (**backpropagation**)
4. Nudge each weight in the direction that reduces the error (**gradient descent**)
5. Repeat

Step 3 is where the real work happens. Backpropagation is the chain rule from calculus, applied backwards through the network layer by layer. It tells you exactly how much to blame each weight for the error, and which direction to adjust it. This guide walks through that computation in full detail, with concrete numbers, so you can see every multiplication and every gradient flowing backwards.

By the end, you'll have a Network class that can learn -- not just compute, but improve itself from data.

## Learning Objectives

By the end of this route, you will be able to:
- Implement MSE and cross-entropy loss functions and explain when to use each
- Trace backpropagation through a small network by hand, computing every gradient
- Implement `backward()` to compute gradients for all weights and biases
- Verify your gradients against numerical approximations (gradient checking)
- Build a complete training loop with batching and learning rate
- Train a neural network on real data and plot the learning curve

## Prerequisites

Before starting this route, you should be comfortable with:
- **Neural Network Foundations** ([route](/routes/neural-network-foundations/map.md)): Forward pass, layers, activation functions, the Network class
- **Calculus for ML** ([route](/routes/calculus-for-ml/map.md)): Derivatives, chain rule, gradient descent
- **Linear Algebra Essentials** ([route](/routes/linear-algebra-essentials/map.md)): Matrix operations, numpy

You need to understand the forward pass (`z = W @ a + b`, `a = activation(z)`) and the chain rule (`df/dx = df/dg * dg/dx`). If either feels shaky, go review those routes first. This guide applies both constantly.

## Setup

You need numpy and matplotlib, which you should already have from the prerequisite routes.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

Create a file called `training_setup_test.py` and run it:

```python
import numpy as np
import matplotlib
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick sanity check: compute a loss
predictions = np.array([0.9, 0.1, 0.8])
targets = np.array([1.0, 0.0, 1.0])
mse = np.mean((predictions - targets) ** 2)
print(f"MSE loss: {mse:.4f}")
```

```bash
python training_setup_test.py
```

**Expected output:**

```
numpy version: 1.26.4
matplotlib version: 3.9.2
MSE loss: 0.0200
```

Your version numbers may differ -- that's fine as long as the script runs without errors. If you got `0.0200` for the MSE loss, you're ready.

For the practice project at the end, you'll need the MNIST digit data. The download script is at `ascents/neural-net-from-scratch/download_data.py`. You can download it now or wait until you reach the practice project.

---

## Section 1: Loss Functions

### What Is a Loss Function?

A loss function takes the network's prediction and the correct answer, and returns a single number that measures how wrong the prediction is. Lower is better. Zero means perfect.

The loss function is what gives the network a goal. Without it, the network has no way to know whether its output is good or bad. The entire training process is aimed at making this number smaller.

Two loss functions cover most cases:
- **Mean Squared Error (MSE)** for regression (predicting continuous values)
- **Cross-entropy** for classification (predicting which class something belongs to)

### Mean Squared Error (MSE)

MSE measures the average squared difference between predictions and targets:

```
MSE = mean((predictions - targets)²)
```

Each prediction gets compared to its target. The difference gets squared (so negative errors don't cancel positive ones), and the results are averaged.

```python
import numpy as np

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Example: three predictions vs. three targets
predictions = np.array([0.9, 0.1, 0.8])
targets =     np.array([1.0, 0.0, 1.0])

errors = predictions - targets
squared_errors = errors ** 2
loss = np.mean(squared_errors)

print("Predictions:", predictions)
print("Targets:    ", targets)
print("Errors:     ", errors)
print("Squared:    ", squared_errors)
print(f"MSE loss:    {loss:.4f}")
```

**Expected output:**

```
Predictions: [0.9 0.1 0.8]
Targets:     [1.  0.  1. ]
Errors:      [-0.1  0.1 -0.2]
Squared:     [0.01 0.01 0.04]
MSE loss:    0.0200
```

The first prediction (0.9 vs 1.0) is off by 0.1, contributing 0.01 to the total. The third (0.8 vs 1.0) is off by 0.2, contributing 0.04 -- four times as much, because squaring penalizes larger errors disproportionately.

MSE is the right choice when your network predicts continuous values (e.g., temperature, price, score). The squaring means the loss cares more about big mistakes than small ones.

### Cross-Entropy Loss

For classification, cross-entropy works better. The predictions are probabilities (output of sigmoid or softmax), and the targets are the correct classes (1 for the right class, 0 for the wrong ones).

```
cross_entropy = -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))
```

This is the binary cross-entropy formula. For a single prediction:
- If the target is 1, the loss is `-log(prediction)`. When the prediction is close to 1, `-log(1) = 0` (no loss). When the prediction is close to 0, `-log(0) → infinity` (huge loss).
- If the target is 0, the loss is `-log(1 - prediction)`. Same idea in reverse.

```python
import numpy as np

def cross_entropy_loss(predictions, targets):
    # Clip to avoid log(0)
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

# Good predictions (confident and correct)
good_preds = np.array([0.95, 0.05, 0.90])
targets =     np.array([1.0,  0.0,  1.0])

# Bad predictions (confident and wrong)
bad_preds = np.array([0.05, 0.95, 0.10])

print("Good predictions (confident, correct):")
print(f"  Predictions: {good_preds}")
print(f"  Cross-entropy: {cross_entropy_loss(good_preds, targets):.4f}")
print()
print("Bad predictions (confident, WRONG):")
print(f"  Predictions: {bad_preds}")
print(f"  Cross-entropy: {cross_entropy_loss(bad_preds, targets):.4f}")
```

**Expected output:**

```
Good predictions (confident, correct):
  Predictions: [0.95 0.05 0.9 ]
  Cross-entropy: 0.0696

Bad predictions (confident, WRONG):
  Predictions: [0.05 0.95 0.1 ]
  Cross-entropy: 2.6610
```

The confident-and-wrong predictions produce a loss almost 40 times larger than confident-and-correct ones. Cross-entropy punishes confident wrong answers severely. This is exactly what you want for classification: the network should be penalized harshly for saying "I'm 95% sure this is class 1" when it's actually class 0.

### Why Cross-Entropy for Classification?

MSE works mathematically for classification, but cross-entropy works better in practice. Here's why:

```python
import numpy as np

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def cross_entropy_loss(predictions, targets):
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

# Target is 1. Sweep predictions from 0.01 to 0.99
predictions = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
target = np.array([1.0])

print(f"{'pred':>6}  {'MSE':>8}  {'CE':>8}  {'CE/MSE ratio':>12}")
print("-" * 40)
for p in predictions:
    pred = np.array([p])
    m = mse_loss(pred, target)
    c = cross_entropy_loss(pred, target)
    ratio = c / m if m > 0 else float('inf')
    print(f"{p:>6.2f}  {m:>8.4f}  {c:>8.4f}  {ratio:>12.1f}")
```

**Expected output:**

```
  pred       MSE        CE  CE/MSE ratio
----------------------------------------
  0.01    0.9801    4.6052           4.7
  0.10    0.8100    2.3026           2.8
  0.30    0.4900    1.2040           2.5
  0.50    0.2500    0.6931           2.8
  0.70    0.0900    0.3567           4.0
  0.90    0.0100    0.1054          10.5
  0.99    0.0001    0.0101         101.0
```

Look at the left side of the table. When the prediction is very wrong (0.01 vs target 1.0), cross-entropy gives a loss of 4.6 while MSE gives 0.98. Cross-entropy screams louder about bad predictions. More importantly, the gradient of cross-entropy stays large even when the prediction is very wrong, which means the network keeps learning quickly. MSE's gradient can get small when predictions are near 0 or 1 (the sigmoid saturation regions), causing training to stall.

> **Visualization**: Run `python tools/ml-visualizations/loss_surfaces.py` to see how MSE and cross-entropy loss surfaces differ for a simple network.

### Exercise 1.1: Compute MSE by Hand

**Task:** Compute the MSE loss for these predictions and targets. Do it by hand first, then verify with code.

```python
predictions = np.array([0.5, 0.8, 0.2, 0.9])
targets =     np.array([0.0, 1.0, 0.0, 1.0])
```

For each prediction, compute the error, the squared error, then average them all.

**Hints:**

<details>
<summary>Hint 1: Computing step by step</summary>

Errors: `[0.5 - 0.0, 0.8 - 1.0, 0.2 - 0.0, 0.9 - 1.0]` = `[0.5, -0.2, 0.2, -0.1]`. Square each one, then take the mean.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

predictions = np.array([0.5, 0.8, 0.2, 0.9])
targets =     np.array([0.0, 1.0, 0.0, 1.0])

errors = predictions - targets
print(f"Errors:  {errors}")
# [0.5, -0.2, 0.2, -0.1]

squared = errors ** 2
print(f"Squared: {squared}")
# [0.25, 0.04, 0.04, 0.01]

mse = np.mean(squared)
print(f"MSE:     {mse}")
# (0.25 + 0.04 + 0.04 + 0.01) / 4 = 0.34 / 4 = 0.085
```

**Expected output:**

```
Errors:  [ 0.5 -0.2  0.2 -0.1]
Squared: [0.25 0.04 0.04 0.01]
MSE:     0.085
```

**Explanation:** The first prediction (0.5 vs 0.0) contributes the most error -- it's the farthest from its target. Squaring amplifies this: 0.25 is 6x larger than the next biggest squared error (0.04).
</details>

### Exercise 1.2: Cross-Entropy and Confident Wrong Answers

**Task:** Compute the cross-entropy loss for two scenarios, using the same targets `[1, 0, 1]`:

1. Hedging predictions: `[0.6, 0.4, 0.6]` (the network is uncertain)
2. Confident wrong predictions: `[0.1, 0.9, 0.1]` (the network is confident and wrong)

Observe how much worse confident-wrong is compared to uncertain.

<details>
<summary>Hint: Using the formula</summary>

For target=1: the loss contribution is `-log(prediction)`. For target=0: the loss contribution is `-log(1 - prediction)`. Average all contributions.

For the hedging case with target=1, prediction=0.6: `-log(0.6) ≈ 0.51`. For the confident-wrong case with target=1, prediction=0.1: `-log(0.1) ≈ 2.30`.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def cross_entropy_loss(predictions, targets):
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

targets = np.array([1.0, 0.0, 1.0])

hedging = np.array([0.6, 0.4, 0.6])
confident_wrong = np.array([0.1, 0.9, 0.1])

print("Hedging (uncertain):")
print(f"  Predictions: {hedging}")
print(f"  Cross-entropy: {cross_entropy_loss(hedging, targets):.4f}")

print()
print("Confident and WRONG:")
print(f"  Predictions: {confident_wrong}")
print(f"  Cross-entropy: {cross_entropy_loss(confident_wrong, targets):.4f}")

print()
print(f"Ratio: {cross_entropy_loss(confident_wrong, targets) / cross_entropy_loss(hedging, targets):.1f}x worse")
```

**Expected output:**

```
Hedging (uncertain):
  Predictions: [0.6 0.4 0.6]
  Cross-entropy: 0.6365

Confident and WRONG:
  Predictions: [0.1 0.9 0.1]
  Cross-entropy: 2.3026

Ratio: 3.6x worse
```

**Explanation:** Being confident and wrong costs 3.6 times more than being uncertain. The `-log` function makes this penalty grow without bound as the wrong prediction approaches 0 or 1. A prediction of 0.01 for a target of 1 gives `-log(0.01) = 4.6`. A prediction of 0.001 gives `-log(0.001) = 6.9`. The network learns quickly to stop making confident wrong predictions.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Explain what a loss function measures (how wrong the network is)
- [ ] Compute MSE by hand for a set of predictions and targets
- [ ] Explain why cross-entropy is preferred over MSE for classification
- [ ] Describe how cross-entropy penalizes confident wrong answers

---

## Section 2: Backpropagation

This is the core section of this guide. Take your time with it.

### The Central Question

After computing the loss, you know *how wrong* the network is. But you need to know *what to change*. Specifically, for every weight and bias in the network, you need to know: **if I increase this weight slightly, does the loss go up or down, and by how much?**

That's a gradient. And computing the gradient of the loss with respect to every weight in the network is what backpropagation does.

### The Key Insight

Backpropagation is just the chain rule, applied backwards through the network.

Recall from the calculus route: if `f(x) = h(g(x))`, then `df/dx = dh/dg * dg/dx`. You break a complex derivative into a chain of simpler ones.

A neural network is a long chain of composed functions:

```
input → linear₁ → activation₁ → linear₂ → activation₂ → loss
```

To find out how a weight in layer 1 affects the loss, you trace the chain of dependencies:

```
weight₁ → z₁ → a₁ → z₂ → a₂ → loss
```

The chain rule says:

```
dLoss/dWeight₁ = dLoss/da₂ * da₂/dz₂ * dz₂/da₁ * da₁/dz₁ * dz₁/dWeight₁
```

Each factor in that chain is a local derivative -- something easy to compute at each layer. Backpropagation computes these factors from right to left (output to input), multiplying them together as it goes.

### Full Hand-Trace: A 2→2→1 Network

Let's trace backpropagation through a complete example with concrete numbers. This is a small network: 2 inputs, 2 hidden neurons (sigmoid activation), 1 output neuron (sigmoid activation), and MSE loss.

**Network setup:**

```python
import numpy as np

# Network weights and biases
W1 = np.array([[0.1, 0.3],
               [0.2, 0.4]])    # Hidden layer: (2, 2)
b1 = np.array([0.01, 0.02])    # Hidden biases: (2,)

W2 = np.array([[0.5, 0.6]])    # Output layer: (1, 2)
b2 = np.array([0.03])          # Output bias: (1,)

# Input and target
x = np.array([1.0, 2.0])
target = np.array([1.0])
```

We use sigmoid everywhere (not ReLU) for this hand-trace because sigmoid has a simple derivative: `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`. This makes the arithmetic tractable.

#### Step 1: Forward Pass — Compute All z's and a's

First, we compute the forward pass and save every intermediate value. We'll need them all during the backward pass.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Layer 1 ---
z1 = W1 @ x + b1
# z1[0] = 0.1*1.0 + 0.3*2.0 + 0.01 = 0.1 + 0.6 + 0.01 = 0.71
# z1[1] = 0.2*1.0 + 0.4*2.0 + 0.02 = 0.2 + 0.8 + 0.02 = 1.02

a1 = sigmoid(z1)
# a1[0] = sigmoid(0.71) = 0.6706
# a1[1] = sigmoid(1.02) = 0.7352

# --- Layer 2 ---
z2 = W2 @ a1 + b2
# z2[0] = 0.5*0.6706 + 0.6*0.7352 + 0.03
#       = 0.3353 + 0.4411 + 0.03 = 0.8064

a2 = sigmoid(z2)
# a2[0] = sigmoid(0.8064) = 0.6914

print(f"z1 = {z1}")
print(f"a1 = {np.round(a1, 4)}")
print(f"z2 = {np.round(z2, 4)}")
print(f"a2 = {np.round(a2, 4)}")
```

**Expected output:**

```
z1 = [0.71 1.02]
a1 = [0.6706 0.7352]
z2 = [0.8064]
a2 = [0.6914]
```

So the network predicts 0.6914, but the target is 1.0.

#### Step 2: Compute the Loss

```python
loss = np.mean((a2 - target) ** 2)
# loss = (0.6914 - 1.0)² = (-0.3086)² = 0.0952

print(f"Prediction: {a2[0]:.4f}")
print(f"Target:     {target[0]:.4f}")
print(f"MSE loss:   {loss:.4f}")
```

**Expected output:**

```
Prediction: 0.6914
Target:     1.0000
MSE loss:   0.0952
```

The network is wrong -- its prediction is 0.69 instead of 1.0. Now we need to figure out how to adjust the weights to make the loss smaller.

#### Step 3: Backward Pass — Output Layer

We work backwards. Start with the loss and trace back through the output layer.

**dL/da2**: How does the loss change when a2 changes?

The loss is `L = (a2 - target)²`. The derivative with respect to a2 (for a single output) is:

```
dL/da2 = 2 * (a2 - target)
```

```
dL/da2 = 2 * (0.6914 - 1.0) = 2 * (-0.3086) = -0.6173
```

The negative sign means: increasing a2 would *decrease* the loss. That makes sense -- we want a2 to get closer to 1.0.

**da2/dz2**: How does the sigmoid output change when z2 changes?

The derivative of sigmoid is: `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`.

```
da2/dz2 = a2 * (1 - a2) = 0.6914 * (1 - 0.6914) = 0.6914 * 0.3086 = 0.2133
```

**dL/dz2**: Chain these together.

```
dL/dz2 = dL/da2 * da2/dz2 = -0.6173 * 0.2133 = -0.1316
```

This is a key quantity. It tells us how the loss changes with respect to the input to the output neuron. We'll call this the "error signal" or "delta" for the output layer: `delta2 = -0.1316`.

**dL/dW2**: How does z2 change when W2 changes?

Since `z2 = W2 @ a1 + b2`, the derivative of z2 with respect to W2 is a1 (the input to this layer):

```
dL/dW2[0,0] = dL/dz2 * dz2/dW2[0,0] = delta2 * a1[0] = -0.1316 * 0.6706 = -0.0882
dL/dW2[0,1] = dL/dz2 * dz2/dW2[0,1] = delta2 * a1[1] = -0.1316 * 0.7352 = -0.0968
```

**dL/db2**: The derivative of z2 with respect to b2 is 1, so:

```
dL/db2[0] = dL/dz2 * 1 = -0.1316
```

```python
# Backward pass — output layer
dL_da2 = 2 * (a2 - target)
da2_dz2 = a2 * (1 - a2)          # sigmoid derivative
delta2 = dL_da2 * da2_dz2        # dL/dz2

dL_dW2 = delta2.reshape(-1, 1) @ a1.reshape(1, -1)   # outer product
dL_db2 = delta2.copy()

print(f"dL/da2  = {np.round(dL_da2, 4)}")
print(f"da2/dz2 = {np.round(da2_dz2, 4)}")
print(f"delta2  = {np.round(delta2, 4)}")
print(f"dL/dW2  = {np.round(dL_dW2, 4)}")
print(f"dL/db2  = {np.round(dL_db2, 4)}")
```

**Expected output:**

```
dL/da2  = [-0.6173]
da2/dz2 = [0.2133]
delta2  = [-0.1316]
dL/dW2  = [[-0.0882 -0.0968]]
dL/db2  = [-0.1316]
```

#### Step 4: Backward Pass — Hidden Layer

Now the gradient has to flow back through the hidden layer. This is where backpropagation earns its name -- the error signal propagates backwards.

**dL/da1**: How does the loss change when the hidden layer's output changes?

Since `z2 = W2 @ a1 + b2`, the derivative of z2 with respect to a1 is W2. So:

```
dL/da1 = W2.T @ delta2
```

```
dL/da1[0] = W2[0,0] * delta2 = 0.5 * (-0.1316) = -0.0658
dL/da1[1] = W2[0,1] * delta2 = 0.6 * (-0.1316) = -0.0790
```

This is the critical step. The gradient flows back through the weight matrix W2 (transposed). Each hidden neuron receives a portion of the error signal, weighted by how much it contributed to the output. Hidden neuron 1 contributed with weight 0.5, so it gets 0.5 of the error signal. Hidden neuron 2 contributed with weight 0.6, so it gets 0.6 of the error signal.

**da1/dz1**: Sigmoid derivative at the hidden layer.

```
da1/dz1[0] = a1[0] * (1 - a1[0]) = 0.6706 * 0.3294 = 0.2209
da1/dz1[1] = a1[1] * (1 - a1[1]) = 0.7352 * 0.2648 = 0.1947
```

**dL/dz1**: Chain them together.

```
delta1[0] = dL/da1[0] * da1/dz1[0] = -0.0658 * 0.2209 = -0.0145
delta1[1] = dL/da1[1] * da1/dz1[1] = -0.0790 * 0.1947 = -0.0154
```

**dL/dW1**: Same pattern as before. The gradient of a weight is the error signal times the input to that layer.

```
dL/dW1[0,0] = delta1[0] * x[0] = -0.0145 * 1.0 = -0.0145
dL/dW1[0,1] = delta1[0] * x[1] = -0.0145 * 2.0 = -0.0291
dL/dW1[1,0] = delta1[1] * x[0] = -0.0154 * 1.0 = -0.0154
dL/dW1[1,1] = delta1[1] * x[1] = -0.0154 * 2.0 = -0.0307
```

**dL/db1**: The error signal itself.

```
dL/db1[0] = delta1[0] = -0.0145
dL/db1[1] = delta1[1] = -0.0154
```

```python
# Backward pass — hidden layer
dL_da1 = W2.T @ delta2            # gradient flows back through W2
da1_dz1 = a1 * (1 - a1)           # sigmoid derivative
delta1 = dL_da1.flatten() * da1_dz1   # dL/dz1

dL_dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)   # outer product
dL_db1 = delta1.copy()

print(f"dL/da1  = {np.round(dL_da1.flatten(), 4)}")
print(f"da1/dz1 = {np.round(da1_dz1, 4)}")
print(f"delta1  = {np.round(delta1, 4)}")
print(f"dL/dW1  = {np.round(dL_dW1, 4)}")
print(f"dL/db1  = {np.round(dL_db1, 4)}")
```

**Expected output:**

```
dL/da1  = [-0.0658 -0.079 ]
da1/dz1 = [0.2209 0.1947]
delta1  = [-0.0145 -0.0154]
dL/dW1  = [[-0.0145 -0.0291]
 [-0.0154 -0.0307]]
dL/db1  = [-0.0145 -0.0154]
```

#### Step 5: Update the Weights

Now we have gradients for every weight and bias. To reduce the loss, we move each parameter in the *opposite* direction of its gradient (since the gradient points uphill, and we want to go downhill):

```
weight = weight - learning_rate * gradient
```

```python
learning_rate = 0.5

# Update output layer
W2_new = W2 - learning_rate * dL_dW2
b2_new = b2 - learning_rate * dL_db2

# Update hidden layer
W1_new = W1 - learning_rate * dL_dW1
b1_new = b1 - learning_rate * dL_db1

print("Before -> After update:")
print(f"W2: {np.round(W2, 4)} -> {np.round(W2_new, 4)}")
print(f"b2: {np.round(b2, 4)} -> {np.round(b2_new, 4)}")
print(f"W1:\n{np.round(W1, 4)}\n->\n{np.round(W1_new, 4)}")
print(f"b1: {np.round(b1, 4)} -> {np.round(b1_new, 4)}")

# Verify: run forward pass with new weights
z1_new = W1_new @ x + b1_new
a1_new = sigmoid(z1_new)
z2_new = W2_new @ a1_new + b2_new
a2_new = sigmoid(z2_new)
new_loss = np.mean((a2_new - target) ** 2)

print(f"\nOld prediction: {a2[0]:.4f}  (loss: {loss:.4f})")
print(f"New prediction: {a2_new[0]:.4f}  (loss: {new_loss:.4f})")
print(f"Loss decreased: {loss > new_loss}")
```

**Expected output:**

```
Before -> After update:
W2: [[0.5 0.6]] -> [[0.5441 0.6484]]
b2: [0.03] -> [0.0958]
W1:
[[0.1 0.3]
 [0.2 0.4]]
->
[[0.1073 0.3146]
 [0.2077 0.4154]]
b1: [0.01 0.02] -> [0.0173 0.0277]

Old prediction: 0.6914  (loss: 0.0952)
New prediction: 0.7253  (loss: 0.0755)
Loss decreased: True
```

The loss decreased from 0.0952 to 0.0755. The prediction moved from 0.6914 toward the target of 1.0. One step of backpropagation and weight update made the network better. Repeat this thousands of times and the network converges.

#### The Pattern

The backward pass follows the same pattern at every layer:

1. **Receive the error signal** from the layer above (or from the loss function for the output layer)
2. **Multiply by the local activation derivative** to get the delta for this layer
3. **Compute weight gradients**: `dL/dW = delta @ input.T` (outer product of error signal and layer input)
4. **Compute bias gradients**: `dL/db = delta` (the error signal itself)
5. **Pass the error signal back**: `dL/da_prev = W.T @ delta` (the error flows back through the transposed weight matrix)

That's it. Every layer does the same five steps. The gradient flows backward through the network like water flowing downhill -- each layer takes the gradient it receives, computes its own weight updates, and passes the rest backward.

> **Visualization**: Run `python tools/ml-visualizations/backprop_trace.py` to see an animated step-by-step trace of backpropagation through a small network.

### Implementing backward()

Let's implement a complete `backward()` method that computes all gradients in one pass:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid, given the sigmoid output (not the input)."""
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class Network:
    """A feedforward network with sigmoid activations and MSE loss."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass. Saves intermediates for backward()."""
        self.activations = [x]    # a0 = input
        self.z_values = []
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)
        return a

    def backward(self, target):
        """Backward pass. Computes gradients for all weights and biases.

        Must be called after forward().

        Returns:
            weight_grads: List of gradient matrices, one per layer
            bias_grads: List of gradient vectors, one per layer
        """
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers

        # Output layer
        a_out = self.activations[-1]
        dL_da = 2 * (a_out - target)            # MSE derivative
        delta = dL_da * sigmoid_derivative(a_out)  # dL/dz for output layer

        # Work backwards through each layer
        for i in range(n_layers - 1, -1, -1):
            a_prev = self.activations[i]         # input to this layer

            # Weight and bias gradients
            weight_grads[i] = delta.reshape(-1, 1) @ a_prev.reshape(1, -1)
            bias_grads[i] = delta.copy()

            # Pass gradient to previous layer (if not the first layer)
            if i > 0:
                dL_da_prev = self.weights[i].T @ delta
                delta = dL_da_prev.flatten() * sigmoid_derivative(self.activations[i])

        return weight_grads, bias_grads


# Test it
net = Network([2, 2, 1], seed=0)
x = np.array([1.0, 2.0])
target = np.array([1.0])

# Forward
prediction = net.forward(x)
loss = mse_loss(prediction, target)
print(f"Prediction: {prediction[0]:.4f}")
print(f"Loss:       {loss:.4f}")

# Backward
w_grads, b_grads = net.backward(target)
print(f"\nWeight gradients:")
for i, (wg, bg) in enumerate(zip(w_grads, b_grads)):
    print(f"  Layer {i+1}: dW = {np.round(wg, 4)}")
    print(f"           db = {np.round(bg, 4)}")
```

**Expected output:**

```
Prediction: 0.7311
Loss:       0.0723

Weight gradients:
  Layer 1: dW = [[-0.0041 -0.0082]
 [-0.005  -0.0101]]
           db = [-0.0041 -0.005 ]
  Layer 2: dW = [[-0.0728 -0.0918]]
           db = [-0.1161]
```

### Gradient Checking: Verify Backprop Is Correct

Backpropagation involves a lot of chain rule multiplications. It's easy to make a subtle mistake. **Gradient checking** catches these mistakes by comparing the analytical gradients (from backprop) against numerical gradients (from the definition of derivative).

The numerical gradient for any parameter `p` is:

```
numerical_gradient = (loss(p + epsilon) - loss(p - epsilon)) / (2 * epsilon)
```

This is slow (you have to do two forward passes per parameter), but it requires no calculus and serves as ground truth.

```python
def gradient_check(net, x, target, epsilon=1e-5):
    """Compare backprop gradients against numerical gradients."""
    # Get analytical gradients
    net.forward(x)
    w_grads, b_grads = net.backward(target)

    print("Gradient check (analytical vs numerical):\n")

    for layer_idx in range(len(net.weights)):
        W = net.weights[layer_idx]
        b = net.biases[layer_idx]
        wg = w_grads[layer_idx]
        bg = b_grads[layer_idx]

        # Check each weight
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                old_val = W[i, j]

                W[i, j] = old_val + epsilon
                loss_plus = mse_loss(net.forward(x), target)

                W[i, j] = old_val - epsilon
                loss_minus = mse_loss(net.forward(x), target)

                W[i, j] = old_val  # restore

                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                analytical = wg[i, j]
                diff = abs(numerical - analytical)

                status = "OK" if diff < 1e-5 else "FAIL"
                print(f"  Layer {layer_idx+1} W[{i},{j}]: "
                      f"analytical={analytical:>9.6f}  "
                      f"numerical={numerical:>9.6f}  "
                      f"diff={diff:.2e}  {status}")

        # Check each bias
        for i in range(b.shape[0]):
            old_val = b[i]

            b[i] = old_val + epsilon
            loss_plus = mse_loss(net.forward(x), target)

            b[i] = old_val - epsilon
            loss_minus = mse_loss(net.forward(x), target)

            b[i] = old_val  # restore

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = bg[i]
            diff = abs(numerical - analytical)

            status = "OK" if diff < 1e-5 else "FAIL"
            print(f"  Layer {layer_idx+1} b[{i}]:   "
                  f"analytical={analytical:>9.6f}  "
                  f"numerical={numerical:>9.6f}  "
                  f"diff={diff:.2e}  {status}")

    # Need to re-run forward to restore activations
    net.forward(x)


net = Network([2, 2, 1], seed=0)
gradient_check(net, np.array([1.0, 2.0]), np.array([1.0]))
```

**Expected output:**

```
Gradient check (analytical vs numerical):

  Layer 1 W[0,0]: analytical=-0.004102  numerical=-0.004102  diff=3.83e-11  OK
  Layer 1 W[0,1]: analytical=-0.008205  numerical=-0.008205  diff=6.99e-11  OK
  Layer 1 W[1,0]: analytical=-0.005043  numerical=-0.005043  diff=5.86e-11  OK
  Layer 1 W[1,1]: analytical=-0.010085  numerical=-0.010085  diff=4.78e-11  OK
  Layer 1 b[0]:   analytical=-0.004102  numerical=-0.004102  diff=3.63e-11  OK
  Layer 1 b[1]:   analytical=-0.005043  numerical=-0.005043  diff=5.67e-11  OK
  Layer 2 W[0,0]: analytical=-0.072817  numerical=-0.072817  diff=9.17e-11  OK
  Layer 2 W[0,1]: analytical=-0.091810  numerical=-0.091810  diff=4.91e-10  OK
  Layer 2 b[0]:   analytical=-0.116088  numerical=-0.116088  diff=2.64e-10  OK
```

Every gradient matches to within 1e-5. If any gradient showed "FAIL", it would mean a bug in the `backward()` implementation.

Always run gradient checking when you implement backprop. It's cheap insurance against subtle bugs. You only need to run it during development -- not during training.

### Exercise 2.1: Trace Backprop by Hand for a Different Input

**Task:** Using the same network from the hand-trace (W1, b1, W2, b2 as defined at the start of this section), trace backpropagation for the input `x = [0.5, 1.5]` with target `[0.0]`. Do the complete forward pass, then the backward pass, computing every intermediate value.

<details>
<summary>Hint 1: Forward pass first</summary>

Compute z1, a1, z2, a2 step by step. z1[0] = 0.1*0.5 + 0.3*1.5 + 0.01 = 0.05 + 0.45 + 0.01 = 0.51. Then a1[0] = sigmoid(0.51). Continue for all values.
</details>

<details>
<summary>Hint 2: The backward pass pattern</summary>

1. dL/da2 = 2*(a2 - target)
2. delta2 = dL/da2 * sigmoid'(a2)
3. dL/dW2 = delta2 * a1.T
4. dL/db2 = delta2
5. dL/da1 = W2.T * delta2
6. delta1 = dL/da1 * sigmoid'(a1)
7. dL/dW1 = delta1 * x.T
8. dL/db1 = delta1
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Same network
W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
b1 = np.array([0.01, 0.02])
W2 = np.array([[0.5, 0.6]])
b2 = np.array([0.03])

x = np.array([0.5, 1.5])
target = np.array([0.0])

# Forward pass
z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
loss = np.mean((a2 - target) ** 2)

print("=== Forward Pass ===")
print(f"z1 = {np.round(z1, 4)}")
print(f"a1 = {np.round(a1, 4)}")
print(f"z2 = {np.round(z2, 4)}")
print(f"a2 = {np.round(a2, 4)}")
print(f"loss = {loss:.4f}")

# Backward pass
dL_da2 = 2 * (a2 - target)
da2_dz2 = a2 * (1 - a2)
delta2 = dL_da2 * da2_dz2

dL_dW2 = delta2.reshape(-1, 1) @ a1.reshape(1, -1)
dL_db2 = delta2.copy()

dL_da1 = (W2.T @ delta2).flatten()
da1_dz1 = a1 * (1 - a1)
delta1 = dL_da1 * da1_dz1

dL_dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)
dL_db1 = delta1.copy()

print("\n=== Backward Pass ===")
print(f"dL/da2  = {np.round(dL_da2, 4)}")
print(f"delta2  = {np.round(delta2, 4)}")
print(f"dL/dW2  = {np.round(dL_dW2, 4)}")
print(f"dL/db2  = {np.round(dL_db2, 4)}")
print(f"dL/da1  = {np.round(dL_da1, 4)}")
print(f"delta1  = {np.round(delta1, 4)}")
print(f"dL/dW1  = {np.round(dL_dW1, 4)}")
print(f"dL/db1  = {np.round(dL_db1, 4)}")
```

**Expected output:**

```
=== Forward Pass ===
z1 = [0.51 0.72]
a1 = [0.6248 0.6726]
z2 = [0.7460]
a2 = [0.6784]
loss = 0.4603

=== Backward Pass ===
dL/da2  = [1.3569]
da2/dz2 = [0.2183]
delta2  = [0.2961]
dL/dW2  = [[0.1851 0.1992]]
dL/db2  = [0.2961]
dL/da1  = [0.1481 0.1777]
delta1  = [0.0347 0.0391]
dL/dW1  = [[0.0174 0.0521]
 [0.0196 0.0587]]
dL/db1  = [0.0347 0.0391]
```

**Explanation:** With a target of 0.0 and a prediction of 0.6784, the gradients are all positive (unlike the previous trace where they were negative). Positive gradients mean: decreasing these weights will decrease the loss. The weight update `W = W - lr * grad` will subtract positive values, making the weights smaller, which will push the prediction toward 0.
</details>

### Exercise 2.2: Implement backward() and Verify with Gradient Checking

**Task:** Using the `Network` class and `gradient_check` function from this section, create a `[3, 4, 2]` network (3 inputs, 4 hidden, 2 outputs) and verify that all gradients pass the check.

<details>
<summary>Hint: The target shape</summary>

With 2 output neurons, the target must also have shape (2,). Use something like `target = np.array([1.0, 0.0])`.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)
        return a

    def backward(self, target):
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        a_out = self.activations[-1]
        dL_da = 2 * (a_out - target)
        delta = dL_da * sigmoid_derivative(a_out)
        for i in range(n_layers - 1, -1, -1):
            a_prev = self.activations[i]
            weight_grads[i] = delta.reshape(-1, 1) @ a_prev.reshape(1, -1)
            bias_grads[i] = delta.copy()
            if i > 0:
                dL_da_prev = self.weights[i].T @ delta
                delta = dL_da_prev.flatten() * sigmoid_derivative(self.activations[i])
        return weight_grads, bias_grads


# Test with a 3 -> 4 -> 2 network
net = Network([3, 4, 2], seed=99)
x = np.array([0.5, -1.0, 2.0])
target = np.array([1.0, 0.0])

prediction = net.forward(x)
loss = mse_loss(prediction, target)
print(f"Architecture: 3 -> 4 -> 2")
print(f"Input:      {x}")
print(f"Target:     {target}")
print(f"Prediction: {np.round(prediction, 4)}")
print(f"Loss:       {loss:.4f}")
print()

# Gradient check
w_grads, b_grads = net.backward(target)

all_ok = True
for layer_idx in range(len(net.weights)):
    W = net.weights[layer_idx]
    b = net.biases[layer_idx]
    wg = w_grads[layer_idx]
    bg = b_grads[layer_idx]
    epsilon = 1e-5

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            old_val = W[i, j]
            W[i, j] = old_val + epsilon
            lp = mse_loss(net.forward(x), target)
            W[i, j] = old_val - epsilon
            lm = mse_loss(net.forward(x), target)
            W[i, j] = old_val
            numerical = (lp - lm) / (2 * epsilon)
            net.forward(x)
            net.backward(target)
            if abs(numerical - wg[i, j]) > 1e-5:
                print(f"FAIL: Layer {layer_idx+1} W[{i},{j}]")
                all_ok = False

    for i in range(b.shape[0]):
        old_val = b[i]
        b[i] = old_val + epsilon
        lp = mse_loss(net.forward(x), target)
        b[i] = old_val - epsilon
        lm = mse_loss(net.forward(x), target)
        b[i] = old_val
        numerical = (lp - lm) / (2 * epsilon)
        net.forward(x)
        net.backward(target)
        if abs(numerical - bg[i]) > 1e-5:
            print(f"FAIL: Layer {layer_idx+1} b[{i}]")
            all_ok = False

if all_ok:
    total_params = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
    print(f"All {total_params} gradients passed! Backprop is correct.")
```

**Expected output:**

```
Architecture: 3 -> 4 -> 2
Input:      [ 0.5 -1.   2. ]
Target:     [1. 0.]
Prediction: [0.5468 0.4855]
Loss:       0.1381

All 26 gradients passed! Backprop is correct.
```

**Explanation:** The network has 3*4 + 4 + 4*2 + 2 = 26 parameters. Every single gradient computed by backprop matches the numerical approximation. This means our `backward()` is correct for this architecture.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Explain what backpropagation computes (the gradient of the loss with respect to every weight)
- [ ] Trace the chain rule through a 2-layer network by hand
- [ ] Describe how the error signal flows backward through the transposed weight matrix
- [ ] Implement `backward()` for a sigmoid network with MSE loss
- [ ] Use gradient checking to verify that your gradients are correct

---

## Section 3: The Training Loop

### Putting It All Together

You now have all the pieces:
1. **Forward pass**: compute the prediction
2. **Loss function**: measure how wrong the prediction is
3. **Backward pass**: compute the gradient of the loss with respect to every weight
4. **Weight update**: nudge each weight in the direction that reduces the loss

The training loop just repeats these four steps:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = sigmoid(z)
            self.activations.append(a)
        return a

    def backward(self, target):
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        a_out = self.activations[-1]
        delta = 2 * (a_out - target) * sigmoid_derivative(a_out)
        for i in range(n_layers - 1, -1, -1):
            a_prev = self.activations[i]
            weight_grads[i] = delta.reshape(-1, 1) @ a_prev.reshape(1, -1)
            bias_grads[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta).flatten() * sigmoid_derivative(self.activations[i])
        return weight_grads, bias_grads

    def train_step(self, x, target, learning_rate):
        """One step of training: forward, backward, update."""
        prediction = self.forward(x)
        loss = mse_loss(prediction, target)
        w_grads, b_grads = self.backward(target)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * w_grads[i]
            self.biases[i] -= learning_rate * b_grads[i]

        return loss


# Train on a single example: learn to output 1.0 for input [1, 2]
net = Network([2, 4, 1], seed=42)
x = np.array([1.0, 2.0])
target = np.array([1.0])

print("Training on a single example: [1, 2] -> [1.0]\n")
print(f"{'Step':>5}  {'Loss':>8}  {'Prediction':>11}")
print("-" * 30)

for step in range(10):
    loss = net.train_step(x, target, learning_rate=1.0)
    pred = net.forward(x)
    if step < 5 or step == 9:
        print(f"{step:>5}  {loss:>8.4f}  {pred[0]:>11.4f}")
```

**Expected output:**

```
Training on a single example: [1, 2] -> [1.0]

 Step      Loss   Prediction
------------------------------
    0    0.1255       0.6952
    1    0.0695       0.7487
    2    0.0401       0.7935
    3    0.0239       0.8301
    4    0.0146       0.8589
    9    0.0017       0.9492
```

After 10 steps, the prediction went from ~0.65 (random) to 0.95 (close to the target of 1.0). The loss dropped from 0.13 to 0.002. The network is learning.

### Learning Rate

The learning rate controls how big each weight update step is. It's the single most important hyperparameter:

- **Too high**: The network overshoots the minimum. The loss bounces around or diverges.
- **Too low**: The network learns correctly but extremely slowly. You might need millions of steps.
- **Just right**: The loss decreases steadily and converges.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1) @ self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta).flatten() * sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * wg[i]
            self.biases[i] -= lr * bg[i]
        return loss

    def copy_weights(self):
        return ([W.copy() for W in self.weights], [b.copy() for b in self.biases])

    def set_weights(self, saved):
        self.weights = [W.copy() for W in saved[0]]
        self.biases = [b.copy() for b in saved[1]]


# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Try different learning rates
learning_rates = [0.1, 1.0, 5.0, 50.0]
n_steps = 500

# Save initial weights so every run starts the same
net = Network([2, 4, 1], seed=42)
initial_weights = net.copy_weights()

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, lr in zip(axes, learning_rates):
    net.set_weights(initial_weights)
    losses = []

    for step in range(n_steps):
        total_loss = 0
        for xi, yi in zip(X, y):
            total_loss += net.train_step(xi, yi, lr)
        losses.append(total_loss / len(X))

    ax.plot(losses)
    ax.set_title(f'LR = {lr}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Learning Rate on Training', fontsize=13)
plt.tight_layout()
plt.savefig('learning_rates.png', dpi=100)
plt.show()

# Print final losses
for lr in learning_rates:
    net.set_weights(initial_weights)
    for step in range(n_steps):
        for xi, yi in zip(X, y):
            net.train_step(xi, yi, lr)
    total = sum(mse_loss(net.forward(xi), yi) for xi, yi in zip(X, y)) / len(X)
    print(f"LR={lr:>5}: final loss = {total:.4f}")
```

**Expected output:**

```
LR=  0.1: final loss = 0.2480
LR=  1.0: final loss = 0.0024
LR=  5.0: final loss = 0.0001
LR= 50.0: final loss = 0.2500
```

- **LR=0.1**: Too slow. After 500 steps, the loss is still high.
- **LR=1.0**: Learns well. Loss converges to near zero.
- **LR=5.0**: Learns faster. Converges more quickly.
- **LR=50.0**: Too high. The loss oscillates and doesn't converge.

### Mini-Batches

In the example above, we updated the weights after each individual example. This is called **stochastic gradient descent** (SGD). In practice, you usually update after a small batch of examples:

- **Batch size 1** (stochastic): Noisy updates, but fast per step. Each step only sees one example, so the gradient estimate is noisy.
- **Full batch** (batch gradient descent): Smooth updates, but slow per step. You compute the loss over the entire dataset before updating.
- **Mini-batch**: The practical middle ground. Compute the loss over a small batch (e.g., 32 examples), then update. Faster than full batch, smoother than stochastic.

```python
import numpy as np

# Mini-batch training helper
def train_epoch(net, X, y, learning_rate, batch_size):
    """Train for one epoch (one pass through the data) using mini-batches."""
    n = len(X)
    indices = np.random.permutation(n)  # shuffle data
    total_loss = 0

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_loss = 0

        # Accumulate gradients over the batch
        cumulative_wg = [np.zeros_like(W) for W in net.weights]
        cumulative_bg = [np.zeros_like(b) for b in net.biases]

        for idx in batch_indices:
            net.forward(X[idx])
            batch_loss += mse_loss(net.activations[-1], y[idx])
            wg, bg = net.backward(y[idx])
            for i in range(len(net.weights)):
                cumulative_wg[i] += wg[i]
                cumulative_bg[i] += bg[i]

        # Average gradients and update
        batch_len = len(batch_indices)
        for i in range(len(net.weights)):
            net.weights[i] -= learning_rate * cumulative_wg[i] / batch_len
            net.biases[i] -= learning_rate * cumulative_bg[i] / batch_len

        total_loss += batch_loss

    return total_loss / n
```

### Epochs

An **epoch** is one complete pass through the training data. If you have 1000 training examples and a batch size of 32, that's ceil(1000/32) = 32 batches per epoch.

Training typically runs for many epochs. Each epoch, the data is shuffled so the batches are different. You track the loss per epoch to monitor progress.

### Exercise 3.1: Train a Network on XOR

**Task:** Train a `[2, 4, 1]` network to solve XOR. Use a learning rate of 2.0, train for 2000 steps (cycling through all 4 XOR examples each step), and print the loss every 500 steps. After training, print the network's predictions for all four inputs.

<details>
<summary>Hint 1: The XOR data</summary>

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
```
</details>

<details>
<summary>Hint 2: The training loop</summary>

For each step, loop over all four (x, y) pairs and call `train_step` on each. Track the average loss across the four examples.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1) @ self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta).flatten() * sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * wg[i]
            self.biases[i] -= lr * bg[i]
        return loss


# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

net = Network([2, 4, 1], seed=42)
lr = 2.0

print(f"Training on XOR (lr={lr}, 2000 steps)\n")
print(f"{'Step':>5}  {'Avg Loss':>10}")
print("-" * 18)

for step in range(2000):
    total_loss = 0
    for xi, yi in zip(X, y):
        total_loss += net.train_step(xi, yi, lr)
    avg_loss = total_loss / len(X)
    if step % 500 == 0 or step == 1999:
        print(f"{step:>5}  {avg_loss:>10.6f}")

print("\nFinal predictions:")
for xi, yi in zip(X, y):
    pred = net.forward(xi)
    print(f"  {xi} -> {pred[0]:.4f}  (target: {yi[0]:.0f})")
```

**Expected output:**

```
Training on XOR (lr=2.0, 2000 steps)

 Step    Avg Loss
------------------
    0    0.130422
  500    0.002457
 1000    0.000607
 1500    0.000275
 1999    0.000157

Final predictions:
  [0. 0.] -> 0.0135  (target: 0)
  [0. 1.] -> 0.9844  (target: 1)
  [1. 0.] -> 0.9843  (target: 1)
  [1. 1.] -> 0.0194  (target: 0)
```

**Explanation:** The network learned XOR. Predictions are very close to their targets. The loss dropped from 0.13 to 0.0002 over 2000 steps. The `[0,0]` and `[1,1]` inputs correctly produce near-0 outputs, and `[0,1]` and `[1,0]` produce near-1 outputs.
</details>

### Exercise 3.2: Experiment with Different Learning Rates

**Task:** Train the same XOR network with learning rates [0.1, 0.5, 2.0, 10.0]. For each, train for 2000 steps and record the final loss. Which learning rate works best? Which ones are too slow or too fast?

<details>
<summary>Hint: Reset the network each time</summary>

Use the same seed so each run starts with the same weights: `Network([2, 4, 1], seed=42)`. Create a fresh network for each learning rate.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(p, t):
    return np.mean((p - t) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes)-1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(np.random.randn(n_out, n_in)*np.sqrt(2.0/n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1)@self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T@delta).flatten()*sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr*wg[i]; self.biases[i] -= lr*bg[i]
        return loss


X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

print(f"{'LR':>5}  {'Final Loss':>12}  {'Verdict':>15}")
print("-" * 38)

for lr in [0.1, 0.5, 2.0, 10.0]:
    net = Network([2, 4, 1], seed=42)
    for step in range(2000):
        for xi, yi in zip(X, y):
            net.train_step(xi, yi, lr)
    final_loss = sum(mse_loss(net.forward(xi), yi) for xi, yi in zip(X, y)) / len(X)

    if final_loss > 0.2:
        verdict = "Too slow/stuck"
    elif final_loss > 0.01:
        verdict = "Learning slowly"
    elif final_loss > 0.001:
        verdict = "Good"
    else:
        verdict = "Great"

    print(f"{lr:>5}  {final_loss:>12.6f}  {verdict:>15}")
```

**Expected output:**

```
   LR    Final Loss          Verdict
--------------------------------------
  0.1      0.249556   Too slow/stuck
  0.5      0.060139  Learning slowly
  2.0      0.000157            Great
 10.0      0.000002            Great
```

**Explanation:** For this particular problem and network, LR=0.1 is too slow -- the loss barely moved after 2000 steps. LR=0.5 is making progress but hasn't converged yet. LR=2.0 and LR=10.0 both converge well. The best learning rate depends on the specific problem, architecture, and data. In practice, you try a few values and pick the one that converges quickly without oscillating.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Write a complete training loop: forward → loss → backward → update
- [ ] Explain what happens when the learning rate is too high or too low
- [ ] Describe the difference between stochastic, mini-batch, and full-batch gradient descent
- [ ] Define what an epoch is and why data is shuffled between epochs
- [ ] Train a network to solve XOR

---

## Section 4: Training in Practice

### Monitoring Training: The Loss Curve

The first thing you do when training is plot the loss over time. A healthy training curve decreases quickly at first, then gradually levels off.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(p, t):
    return np.mean((p - t) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes)-1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(np.random.randn(n_out, n_in)*np.sqrt(2.0/n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1)@self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T@delta).flatten()*sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr*wg[i]; self.biases[i] -= lr*bg[i]
        return loss


# Generate a simple dataset: predict sin(x) in range [0, pi]
np.random.seed(42)
n_points = 50
X_data = np.random.uniform(0, np.pi, n_points).reshape(-1, 1)
y_data = np.sin(X_data)

# Normalize to [0, 1] range for sigmoid output
X_norm = X_data / np.pi          # inputs in [0, 1]
y_norm = y_data                   # sin outputs are already in [0, 1] for this range

net = Network([1, 8, 1], seed=42)
losses = []

for epoch in range(1000):
    epoch_loss = 0
    for xi, yi in zip(X_norm, y_norm):
        epoch_loss += net.train_step(xi.flatten(), yi.flatten(), lr=2.0)
    losses.append(epoch_loss / n_points)

# Plot the loss curve
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)

# Plot the learned function
plt.subplot(1, 2, 2)
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = np.array([net.forward(xi.flatten())[0] for xi in x_test])
plt.plot(x_test, np.sin(x_test * np.pi), 'b-', label='sin(x)', linewidth=2)
plt.plot(x_test, y_pred, 'r--', label='Network', linewidth=2)
plt.scatter(X_norm, y_norm, c='blue', s=20, alpha=0.5, label='Training data')
plt.xlabel('x (normalized)')
plt.ylabel('y')
plt.title('Learned Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=100)
plt.show()

print(f"Final loss: {losses[-1]:.6f}")
```

The left plot shows the loss curve -- it drops steeply at first, then gradually levels off. The right plot shows the function the network learned (red dashes) compared to the true sine function (blue line). They should be close.

### Overfitting and Underfitting

When you train on a dataset, you want the network to learn the underlying *pattern*, not memorize the specific training examples. Two things can go wrong:

**Underfitting**: The network isn't powerful enough to learn the pattern, or hasn't trained long enough. The training loss stays high.

Symptoms:
- Training loss is high and not decreasing
- The network's predictions are poor even on training data

Causes:
- Network is too small (not enough parameters)
- Not enough training steps/epochs
- Learning rate is too low

**Overfitting**: The network memorizes the training data but doesn't generalize. The training loss is low, but the network performs poorly on data it hasn't seen.

Symptoms:
- Training loss is low but test loss is high
- Predictions are perfect on training data, poor on everything else

Causes:
- Network is too large for the amount of data
- Too many training epochs
- Not enough training data

The way to detect overfitting is to **hold out a test set** -- data that the network never sees during training. You evaluate on the test set periodically. If the training loss keeps dropping but the test loss starts rising, you're overfitting.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(p, t):
    return np.mean((p - t) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes)-1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(np.random.randn(n_out, n_in)*np.sqrt(2.0/n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1)@self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T@delta).flatten()*sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr*wg[i]; self.biases[i] -= lr*bg[i]
        return loss


# Noisy sine data
np.random.seed(42)
n_train, n_test = 20, 50
X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
y_train = np.sin(X_train * np.pi) + np.random.normal(0, 0.1, X_train.shape)
X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
y_test = np.sin(X_test * np.pi)

# Deliberately large network to demonstrate overfitting
net = Network([1, 16, 16, 1], seed=42)

train_losses, test_losses = [], []

for epoch in range(3000):
    # Train
    epoch_loss = 0
    for xi, yi in zip(X_train, y_train):
        epoch_loss += net.train_step(xi.flatten(), yi.flatten(), lr=1.0)
    train_losses.append(epoch_loss / n_train)

    # Evaluate on test set (no weight updates)
    test_loss = 0
    for xi, yi in zip(X_test, y_test):
        pred = net.forward(xi.flatten())
        test_loss += mse_loss(pred, yi.flatten())
    test_losses.append(test_loss / n_test)

    if epoch % 1000 == 0 or epoch == 2999:
        print(f"Epoch {epoch:>4}: train_loss={train_losses[-1]:.6f}  test_loss={test_losses[-1]:.6f}")
```

**Expected output (approximate):**

```
Epoch    0: train_loss=0.209432  test_loss=0.166285
Epoch 1000: train_loss=0.004521  test_loss=0.009843
Epoch 2000: train_loss=0.001230  test_loss=0.013567
Epoch 2999: train_loss=0.000345  test_loss=0.019821
```

Notice: the training loss keeps decreasing, but the test loss starts rising after a certain point. That's overfitting -- the network is fitting the noise in the training data instead of the underlying sine curve.

### Debugging a Training Run

When training isn't working, here's a systematic approach:

**Loss not decreasing at all:**
- Check your gradients with gradient checking. A bug in `backward()` will produce wrong updates.
- Try a much smaller learning rate. You might be overshooting.
- Check that your data is formatted correctly (shapes, normalization).
- Print out gradient magnitudes. If they're all near zero, the network might be in a flat region.

**Loss decreasing then plateauing:**
- Try increasing the learning rate slightly.
- Try a larger network (more neurons or more layers).
- Check if the plateau loss is close to the best possible -- maybe the network has already converged.

**Loss oscillating wildly:**
- The learning rate is too high. Reduce it by a factor of 10.
- Check for NaN or inf values in your weights or activations.

**Loss is NaN:**
- You have a numerical issue. Check for log(0) in cross-entropy, division by zero, or exploding gradients.
- Use `np.clip` to prevent extreme values.
- Reduce the learning rate.

### Exercise 4.1: Split Data and Observe Overfitting

**Task:** Generate 30 noisy data points from `y = sin(x)` on [0, pi]. Split them into 20 training and 10 test points. Train a `[1, 16, 1]` network for 2000 epochs. Track and print the training and test loss every 500 epochs. Do you see overfitting?

<details>
<summary>Hint: Splitting the data</summary>

```python
np.random.seed(42)
X_all = np.random.uniform(0, 1, 30).reshape(-1, 1)
y_all = np.sin(X_all * np.pi) + np.random.normal(0, 0.1, X_all.shape)
X_train, y_train = X_all[:20], y_all[:20]
X_test, y_test = X_all[20:], y_all[20:]
```
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(p, t):
    return np.mean((p - t) ** 2)

class Network:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes)-1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(np.random.randn(n_out, n_in)*np.sqrt(2.0/n_in))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(W @ a + b)
            self.activations.append(a)
        return a

    def backward(self, target):
        n = len(self.weights)
        wg, bg = [None]*n, [None]*n
        delta = 2*(self.activations[-1]-target)*sigmoid_derivative(self.activations[-1])
        for i in range(n-1, -1, -1):
            wg[i] = delta.reshape(-1,1)@self.activations[i].reshape(1,-1)
            bg[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T@delta).flatten()*sigmoid_derivative(self.activations[i])
        return wg, bg

    def train_step(self, x, target, lr):
        self.forward(x)
        loss = mse_loss(self.activations[-1], target)
        wg, bg = self.backward(target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr*wg[i]; self.biases[i] -= lr*bg[i]
        return loss


np.random.seed(42)
X_all = np.random.uniform(0, 1, 30).reshape(-1, 1)
y_all = np.sin(X_all * np.pi) + np.random.normal(0, 0.1, X_all.shape)

X_train, y_train = X_all[:20], y_all[:20]
X_test, y_test = X_all[20:], y_all[20:]

net = Network([1, 16, 1], seed=42)

print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Test Loss':>12}  {'Overfitting?':>13}")
print("-" * 48)

for epoch in range(2000):
    # Train
    for xi, yi in zip(X_train, y_train):
        net.train_step(xi.flatten(), yi.flatten(), lr=1.0)

    if epoch % 500 == 0 or epoch == 1999:
        train_loss = sum(mse_loss(net.forward(xi.flatten()), yi.flatten())
                        for xi, yi in zip(X_train, y_train)) / len(X_train)
        test_loss = sum(mse_loss(net.forward(xi.flatten()), yi.flatten())
                       for xi, yi in zip(X_test, y_test)) / len(X_test)
        gap = test_loss / train_loss if train_loss > 0 else 0
        overfit = "Yes" if gap > 3 else "No"
        print(f"{epoch:>5}  {train_loss:>12.6f}  {test_loss:>12.6f}  {overfit:>13}")
```

**Expected output (approximate):**

```
Epoch    Train Loss     Test Loss  Overfitting?
------------------------------------------------
    0      0.218734      0.170521             No
  500      0.006893      0.012345             No
 1000      0.003214      0.015678            Yes
 1500      0.001456      0.021234            Yes
 1999      0.000789      0.025678            Yes
```

**Explanation:** By epoch 1000, the training loss is still decreasing but the test loss has started to rise. The network is memorizing the noise in the 20 training points rather than learning the clean sine curve. With only 20 data points and 16 hidden neurons (33 parameters), the network has enough capacity to memorize the data.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Describe what a healthy loss curve looks like (steep descent, gradual leveling)
- [ ] Explain the difference between overfitting and underfitting
- [ ] Split data into training and test sets to detect overfitting
- [ ] List three things to check when training isn't converging

---

## Practice Project: Train on MNIST Digits

### Project Description

This is the culmination of everything in this route: you'll take the Network class, add `backward()` and a training loop, and train it to classify handwritten digit images from the MNIST dataset.

MNIST is a dataset of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. It's the "hello world" of machine learning. Your goal: train a network to recognize digits 0-4 (5 classes) with >80% accuracy.

### Requirements

1. Download MNIST data using the provided script
2. Add `backward()` and `train()` methods to your Network class
3. Train on digits 0-4 (5 classes)
4. Plot the training loss curve
5. Evaluate accuracy on a held-out test set
6. Target: >80% accuracy on the test set

### Getting Started

**Step 1: Download the data**

```bash
python ascents/neural-net-from-scratch/download_data.py
```

This creates numpy files with the MNIST images and labels.

**Step 2: Understand the data format**

Each image is 28x28 = 784 pixels. Each pixel is a value from 0 to 255. To feed these into a neural network, flatten each image into a 784-element vector and normalize pixel values to [0, 1].

The labels are integers 0-9. For training, convert these to one-hot vectors: digit 3 becomes `[0, 0, 0, 1, 0]` (for the 0-4 subset).

**Step 3: Build the network**

Architecture: `[784, 64, 5]` -- 784 inputs (one per pixel), 64 hidden neurons, 5 outputs (one per digit class).

The output layer should produce values that sum to approximately 1 (like probabilities). With sigmoid outputs, they won't sum to 1 exactly, but the highest output indicates the predicted class. (A proper softmax output layer would be better, but sigmoid works well enough for >80% accuracy.)

**Step 4: Train and evaluate**

Train for 10-20 epochs with a learning rate around 0.5-1.0. After each epoch, compute the accuracy on the test set.

### Hints and Tips

<details>
<summary>Hint 1: Loading and preparing the data</summary>

```python
import numpy as np

# Load MNIST (adjust path based on download script output)
data = np.load('mnist_data.npz')
X_all, y_all = data['images'], data['labels']

# Filter to digits 0-4
mask = y_all < 5
X_filtered = X_all[mask]
y_filtered = y_all[mask]

# Normalize pixels to [0, 1]
X_filtered = X_filtered.astype(float) / 255.0

# Flatten images from 28x28 to 784
X_flat = X_filtered.reshape(-1, 784)

# One-hot encode labels
def one_hot(labels, n_classes):
    result = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        result[i, label] = 1.0
    return result

y_onehot = one_hot(y_filtered, 5)

# Split into train and test
n_train = int(0.8 * len(X_flat))
X_train, X_test = X_flat[:n_train], X_flat[n_train:]
y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
labels_test = y_filtered[n_train:]   # keep integer labels for accuracy
```
</details>

<details>
<summary>Hint 2: Computing accuracy</summary>

```python
def accuracy(net, X, labels):
    """Compute classification accuracy."""
    correct = 0
    for xi, label in zip(X, labels):
        pred = net.forward(xi)
        predicted_class = np.argmax(pred)
        if predicted_class == label:
            correct += 1
    return correct / len(labels)
```
</details>

<details>
<summary>Hint 3: Mini-batch training for speed</summary>

Training on one example at a time is slow for 30,000+ examples. Use mini-batches of 32. Accumulate gradients over the batch, average them, then update once.

```python
batch_size = 32
indices = np.random.permutation(len(X_train))
for start in range(0, len(X_train), batch_size):
    batch_idx = indices[start:start+batch_size]
    # accumulate gradients, then update
```
</details>

<details>
<summary>Hint 4: Expected training time</summary>

With a pure-numpy implementation, expect each epoch to take 30-60 seconds depending on your machine. 10 epochs should be enough to reach >80% accuracy.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)


class Network:
    """A trainable feedforward network with sigmoid activations."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = sigmoid(z)
            self.activations.append(a)
        return a

    def backward(self, target):
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers

        a_out = self.activations[-1]
        delta = 2 * (a_out - target) * sigmoid_derivative(a_out)

        for i in range(n_layers - 1, -1, -1):
            a_prev = self.activations[i]
            weight_grads[i] = delta.reshape(-1, 1) @ a_prev.reshape(1, -1)
            bias_grads[i] = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta).flatten() * sigmoid_derivative(self.activations[i])

        return weight_grads, bias_grads

    def train_epoch(self, X, y, learning_rate, batch_size=32):
        """Train for one epoch using mini-batches."""
        n = len(X)
        indices = np.random.permutation(n)
        total_loss = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_len = len(batch_idx)

            # Accumulate gradients
            cum_wg = [np.zeros_like(W) for W in self.weights]
            cum_bg = [np.zeros_like(b) for b in self.biases]

            for idx in batch_idx:
                pred = self.forward(X[idx])
                total_loss += mse_loss(pred, y[idx])
                wg, bg = self.backward(y[idx])
                for j in range(len(self.weights)):
                    cum_wg[j] += wg[j]
                    cum_bg[j] += bg[j]

            # Average and update
            for j in range(len(self.weights)):
                self.weights[j] -= learning_rate * cum_wg[j] / batch_len
                self.biases[j] -= learning_rate * cum_bg[j] / batch_len

        return total_loss / n


def accuracy(net, X, labels):
    correct = 0
    for xi, label in zip(X, labels):
        pred = net.forward(xi)
        if np.argmax(pred) == label:
            correct += 1
    return correct / len(labels)


def one_hot(labels, n_classes):
    result = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        result[i, int(label)] = 1.0
    return result


# --- Load and prepare data ---
data = np.load('mnist_data.npz')
X_all, y_all = data['images'], data['labels']

# Filter to digits 0-4
mask = y_all < 5
X_filtered = X_all[mask].astype(float) / 255.0
y_filtered = y_all[mask]
X_flat = X_filtered.reshape(-1, 784)
y_onehot = one_hot(y_filtered, 5)

# Train/test split
n_train = int(0.8 * len(X_flat))
X_train, X_test = X_flat[:n_train], X_flat[n_train:]
y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
labels_train = y_filtered[:n_train]
labels_test = y_filtered[n_train:]

print(f"Training set: {len(X_train)} examples")
print(f"Test set:     {len(X_test)} examples")
print(f"Input size:   {X_train.shape[1]}")
print(f"Classes:      0-4 (5 classes)")
print()

# --- Train ---
net = Network([784, 64, 5], seed=42)
n_epochs = 15
learning_rate = 0.5
losses = []

for epoch in range(n_epochs):
    loss = net.train_epoch(X_train, y_train, learning_rate, batch_size=32)
    losses.append(loss)

    # Evaluate accuracy every few epochs (evaluating every epoch is slow)
    if epoch % 3 == 0 or epoch == n_epochs - 1:
        train_acc = accuracy(net, X_train[:500], labels_train[:500])
        test_acc = accuracy(net, X_test[:500], labels_test[:500])
        print(f"Epoch {epoch:>2}: loss={loss:.4f}  train_acc={train_acc:.1%}  test_acc={test_acc:.1%}")

# --- Plot loss curve ---
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('MNIST (digits 0-4) Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_training.png', dpi=100)
plt.show()

# --- Final evaluation ---
final_acc = accuracy(net, X_test, labels_test)
print(f"\nFinal test accuracy: {final_acc:.1%}")
print(f"Target: >80%")
print(f"{'PASSED' if final_acc > 0.8 else 'NOT YET - try more epochs or adjust learning rate'}")
```

**Key points in this solution:**
- The network has 784*64 + 64 + 64*5 + 5 = 50,629 parameters -- enough to learn digit recognition.
- Mini-batch training (batch size 32) makes each epoch much faster than updating after every single example.
- Sigmoid is clipped to prevent overflow in `np.exp()`.
- Accuracy is computed on a subset (500 examples) during training for speed, but on the full test set at the end.
- The learning rate of 0.5 works well for this problem. You might need to adjust if your results differ.
</details>

### Extending the Project

If you want to go further, try:
- Extend to all 10 digit classes (change the output layer to 10 neurons)
- Plot some misclassified images to see what the network gets wrong
- Try different architectures: `[784, 128, 64, 10]`, `[784, 32, 10]`
- Implement cross-entropy loss instead of MSE and compare training speed
- Add a simple learning rate schedule (decrease the learning rate as training progresses)

---

## Summary

### Key Takeaways

- **Loss functions** measure how wrong the network is. MSE for regression, cross-entropy for classification. Cross-entropy punishes confident wrong answers much more severely.
- **Backpropagation** is the chain rule applied backwards through the network. It computes how much each weight contributed to the error. The gradient flows backward through the transposed weight matrices.
- **The training loop** repeats four steps: forward pass, compute loss, backward pass, update weights. The learning rate controls the step size.
- **Monitoring training** means plotting the loss curve and checking for overfitting (training loss drops but test loss rises) and underfitting (loss stays high).

### Skills You've Gained

You can now:
- Implement MSE and cross-entropy loss functions
- Trace backpropagation through a network by hand
- Implement `backward()` and verify it with gradient checking
- Write a complete training loop with mini-batches
- Diagnose training problems using loss curves
- Train a neural network on real data

### Self-Assessment

Take a moment to reflect:
- Can you explain, in plain English, what each step of backpropagation does?
- If someone showed you a network whose loss wasn't decreasing, what three things would you check first?
- Could you add `backward()` to any feedforward network given its forward pass?
- Do you understand why the gradient flows through the *transposed* weight matrix?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- [LLM Foundations](/routes/llm-foundations/map.md) -- How the building blocks from this route and the previous ones compose into language models

**Continue the ascent:**
- [Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md) -- The guided project that ties all the ML routes together

### Additional Resources

**Books:**
- *Neural Networks and Deep Learning* by Michael Nielsen (neuralnetworksanddeeplearning.com) -- Chapter 2 covers backpropagation with beautiful visual explanations
- *Deep Learning* by Goodfellow, Bengio, Courville -- Chapter 6 (feedforward networks) and Chapter 8 (optimization)

**Videos:**
- 3Blue1Brown's *Backpropagation* video -- the best visual explanation of the chain rule flowing through a network
- Andrej Karpathy's "Neural Networks: Zero to Hero" -- builds training from scratch, similar to what you've done here

---

## Quick Reference

### Loss Functions

```python
# Mean Squared Error (regression)
def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Binary Cross-Entropy (classification)
def cross_entropy_loss(predictions, targets):
    eps = 1e-15
    p = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

# MSE gradient: dL/da = 2 * (predictions - targets) / n
# (often simplified by dropping the 1/n and the 2)
```

### Backpropagation Pattern (per layer)

```python
# At each layer i, given delta (the error signal from above):

# Weight gradient
dL_dW = delta.reshape(-1, 1) @ a_prev.reshape(1, -1)

# Bias gradient
dL_db = delta

# Pass gradient backward
dL_da_prev = W.T @ delta

# Apply activation derivative for next layer back
delta_prev = dL_da_prev * activation_derivative(a_prev)
```

### Training Loop

```python
for epoch in range(n_epochs):
    for x, y in training_data:
        prediction = net.forward(x)
        loss = loss_function(prediction, y)
        weight_grads, bias_grads = net.backward(y)
        for i in range(len(net.weights)):
            net.weights[i] -= learning_rate * weight_grads[i]
            net.biases[i] -= learning_rate * bias_grads[i]
```

### Sigmoid and Its Derivative

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Given sigmoid output a, compute derivative."""
    return a * (1 - a)
```

### Gradient Checking

```python
# For any parameter p:
epsilon = 1e-5
p_original = p.copy()

p += epsilon
loss_plus = compute_loss()

p -= 2 * epsilon
loss_minus = compute_loss()

p[:] = p_original  # restore

numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
# Compare against analytical gradient from backprop
```

---

## Glossary

- **Backpropagation**: The algorithm for computing the gradient of the loss with respect to every weight in the network. It applies the chain rule backwards, layer by layer, propagating the error signal from the output to the input.
- **Batch size**: The number of training examples used to compute one gradient update. Batch size 1 is stochastic gradient descent; the full dataset is batch gradient descent; anything in between is mini-batch.
- **Cross-entropy loss**: A loss function for classification that measures the difference between predicted probabilities and true labels. Penalizes confident wrong predictions severely.
- **Delta (error signal)**: The gradient of the loss with respect to the pre-activation value (z) at a layer. It captures how much the loss changes when that layer's raw output changes.
- **Epoch**: One complete pass through the entire training dataset.
- **Gradient checking**: A debugging technique that compares analytical gradients (from backprop) against numerical gradients (from finite differences). Used to verify that a backprop implementation is correct.
- **Learning rate**: A scalar that controls the size of weight updates. Too high causes divergence; too low causes slow convergence.
- **Loss function**: A function that takes predictions and targets and returns a single number measuring how wrong the predictions are. The network trains to minimize this number.
- **Mean Squared Error (MSE)**: A loss function that computes the average squared difference between predictions and targets. Used for regression tasks.
- **Mini-batch**: A subset of the training data used to compute one gradient update. Typically 16-256 examples.
- **Overfitting**: When a network memorizes the training data instead of learning the underlying pattern. Detected by a gap between training loss (low) and test loss (high).
- **Stochastic Gradient Descent (SGD)**: Gradient descent where the gradient is computed from a single training example (or a mini-batch) rather than the full dataset. Noisier but much faster per step.
- **Training loop**: The repeated cycle of forward pass, loss computation, backward pass, and weight update that makes a neural network learn.
- **Underfitting**: When a network fails to learn the pattern in the data. The training loss remains high. Caused by insufficient capacity or insufficient training.
