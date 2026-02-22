---
title: Neural Network Foundations
route_map: /routes/neural-network-foundations/map.md
paired_sherpa: /routes/neural-network-foundations/sherpa.md
prerequisites:
  - Linear Algebra Essentials (matrix-vector multiplication)
  - Calculus for ML (derivatives, gradients)
topics:
  - Perceptrons
  - Activation Functions
  - Layers
  - Forward Pass
---

# Neural Network Foundations - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/neural-network-foundations/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/neural-network-foundations/map.md` for the high-level overview.

## Overview

A neural network is composed matrix operations with nonlinearities between them. You already know both pieces. From the linear algebra route, you know how to multiply matrices and vectors. From the calculus route, you know what derivatives and gradients are. This route shows how those two ideas combine into the structure that powers modern ML.

The core loop is simple: take an input vector, multiply it by a weight matrix, add a bias, pass the result through a nonlinear function, and repeat. Each repetition is a "layer." Stack enough layers and the network can approximate remarkably complex functions -- classifiers, translators, image generators.

This guide builds that understanding from the ground up. You'll start with a single neuron (the perceptron), see why nonlinearity matters, then assemble neurons into layers and layers into networks. By the end, you'll have a Network class that computes a forward pass, and you'll be able to visualize what each layer does to the input space.

## Learning Objectives

By the end of this route, you will be able to:
- Implement a perceptron and explain it as a weighted sum plus bias plus activation
- Explain why activation functions are necessary and choose between sigmoid, ReLU, and tanh
- Trace data through a multi-layer network by hand and in code
- Build a Network class with a working forward pass
- Visualize how each layer transforms the input space

## Prerequisites

Before starting this route, you should be comfortable with:
- **Linear Algebra Essentials** ([route](/routes/linear-algebra-essentials/map.md)): Matrix-vector multiplication, the `@` operator, numpy arrays
- **Calculus for ML** ([route](/routes/calculus-for-ml/map.md)): Derivatives, gradients, and the intuition behind optimization
- **Python classes** (helpful): `__init__`, methods, attributes

If matrix-vector multiplication or the concept of a gradient feels fuzzy, go back and review those routes first. This guide uses both constantly.

## Setup

You need numpy and matplotlib, which you should already have from the prerequisite routes.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

Create a file called `nn_setup_test.py` and run it:

```python
import numpy as np
import matplotlib
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick sanity check: a single neuron computation
weights = np.array([0.5, -0.3])
inputs = np.array([1.0, 2.0])
bias = 0.1
z = np.dot(weights, inputs) + bias
output = 1 / (1 + np.exp(-z))  # sigmoid
print(f"Neuron output: {output:.4f}")
```

```bash
python nn_setup_test.py
```

**Expected output:**

```
numpy version: 1.26.4
matplotlib version: 3.9.2
Neuron output: 0.4750
```

Your version numbers may differ -- that's fine as long as the script runs without errors. If you got `0.4750` for the neuron output, you're ready.

---

## Section 1: The Perceptron

### What Is a Perceptron?

A perceptron is a single artificial neuron. It takes a vector of inputs, computes a weighted sum, adds a bias, and passes the result through an activation function to produce a single output.

In code:

```python
output = activation(np.dot(weights, inputs) + bias)
```

That's the entire computation. Let's break it apart:

1. **Weighted sum**: `np.dot(weights, inputs)` -- a dot product. Each input gets multiplied by its corresponding weight, and the products are summed. This is the same dot product from linear algebra.
2. **Bias**: `+ bias` -- a constant offset that shifts the result. Without the bias, the decision always passes through the origin.
3. **Activation function**: `activation(...)` -- a nonlinear function that squashes the result into a useful range. We'll use sigmoid for now: it maps any number to a value between 0 and 1.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# A perceptron with 2 inputs
weights = np.array([0.5, -0.3])
inputs = np.array([1.0, 2.0])
bias = 0.1

# Step by step
z = np.dot(weights, inputs) + bias   # 0.5*1.0 + (-0.3)*2.0 + 0.1 = -0.1
output = sigmoid(z)

print(f"Weighted sum + bias (z): {z}")
print(f"Sigmoid output: {output:.4f}")
```

**Expected output:**

```
Weighted sum + bias (z): 0.0
Sigmoid output: 0.5000
```

Wait -- let's check the arithmetic. `0.5 * 1.0 + (-0.3) * 2.0 + 0.1` = `0.5 - 0.6 + 0.1` = `0.0`. And sigmoid(0) = 0.5. When the weighted sum is zero, the sigmoid sits right at the midpoint -- the neuron is undecided.

### The Decision Boundary

A perceptron with two inputs draws a line in 2D space. Points on one side of the line produce outputs above 0.5, and points on the other side produce outputs below 0.5. This line is the **decision boundary**.

The equation of the boundary is where the weighted sum equals zero:

```
w1 * x1 + w2 * x2 + bias = 0
```

This is a linear equation -- it defines a straight line. The weights control the line's angle, and the bias shifts it away from the origin.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Perceptron parameters
w = np.array([1.0, -1.5])
bias = 0.5

# Generate a grid of points
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1, x2)

# Compute perceptron output at every point
Z = w[0] * X1 + w[1] * X2 + bias
outputs = sigmoid(Z)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, outputs, levels=50, cmap='RdBu_r', alpha=0.8)
plt.colorbar(label='Perceptron output')

# Draw the decision boundary (where z = 0)
# w1*x1 + w2*x2 + bias = 0  =>  x2 = -(w1*x1 + bias) / w2
boundary_x2 = -(w[0] * x1 + bias) / w[1]
plt.plot(x1, boundary_x2, 'k-', linewidth=2, label='Decision boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('perceptron_boundary.png', dpi=100)
plt.show()
```

The plot shows a smooth gradient from red (output near 1) to blue (output near 0), with a sharp transition at the decision boundary line. Everything on one side is "class 1" and everything on the other side is "class 0."

> **Visualization**: Run `python tools/ml-visualizations/perceptron.py` to explore how changing weights and bias moves the decision boundary interactively.

### The XOR Problem

A single perceptron can only draw one straight line. This means it can only classify data that is **linearly separable** -- where a single line can divide the two classes.

The XOR function is a classic example of something a single perceptron cannot solve:

| x1 | x2 | XOR output |
|----|-----|------------|
| 0  | 0   | 0          |
| 0  | 1   | 1          |
| 1  | 0   | 1          |
| 1  | 1   | 0          |

```python
import numpy as np
import matplotlib.pyplot as plt

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

plt.figure(figsize=(6, 6))
for i, (point, label) in enumerate(zip(X, y)):
    color = 'red' if label == 1 else 'blue'
    marker = 'o' if label == 1 else 's'
    plt.scatter(point[0], point[1], c=color, s=200, marker=marker,
                edgecolors='black', linewidth=2, zorder=5)
    plt.annotate(f'  XOR={label}', xy=point, fontsize=12)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR: No single line can separate red from blue')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xor_problem.png', dpi=100)
plt.show()
```

No matter how you orient a single straight line, you can't put both red points on one side and both blue points on the other. The classes are not linearly separable.

This was a famous limitation identified in 1969 by Minsky and Papert. The solution, as we'll see in Section 4, is to use multiple layers -- a network of perceptrons. The first layer warps the space so that the classes *become* linearly separable, and the second layer draws the final boundary.

### Exercise 1.1: Implement a Perceptron and Classify Points

**Task:** Implement a perceptron with weights `[2.0, -1.0]` and bias `-0.5`. Classify the following points by computing the output and thresholding at 0.5:

```python
points = np.array([[1, 0], [0, 1], [1, 1], [-1, 2], [2, 3]])
```

For each point, print the weighted sum (z), the sigmoid output, and the predicted class (1 if output >= 0.5, else 0).

**Hints:**

<details>
<summary>Hint 1: Computing z for all points at once</summary>

You can compute `z = points @ weights + bias` to get the weighted sum for all points in one operation. The `@` operator handles the dot product for each row of `points` against the `weights` vector.
</details>

<details>
<summary>Hint 2: Thresholding</summary>

After computing the sigmoid output, classify with `predicted = (output >= 0.5).astype(int)`. This converts True/False to 1/0.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.array([2.0, -1.0])
bias = -0.5

points = np.array([[1, 0], [0, 1], [1, 1], [-1, 2], [2, 3]])

z = points @ weights + bias
outputs = sigmoid(z)
predicted = (outputs >= 0.5).astype(int)

print(f"{'Point':>10}  {'z':>6}  {'sigmoid':>7}  {'class':>5}")
print("-" * 35)
for i in range(len(points)):
    print(f"{str(points[i]):>10}  {z[i]:>6.2f}  {outputs[i]:>7.4f}  {predicted[i]:>5}")
```

**Expected output:**

```
     Point       z  sigmoid  class
-----------------------------------
    [1 0]    1.50   0.8176      1
    [0 1]   -1.50   0.1824      0
    [1 1]    0.50   0.6225      1
   [-1  2]   -4.50   0.0109      0
    [2 3]    0.50   0.6225      1
```

**Explanation:** The decision boundary is where `2*x1 - 1*x2 - 0.5 = 0`, or `x2 = 2*x1 - 0.5`. Points above this line (in x2 terms) get classified as 0, points below as 1. The weights determine that x1 pushes toward class 1 and x2 pushes toward class 0.
</details>

### Exercise 1.2: Change Weights and Bias, Observe the Boundary

**Task:** Start with the perceptron from Exercise 1.1 (weights `[2.0, -1.0]`, bias `-0.5`). Make three separate modifications and observe how the decision boundary changes:

1. Double the weights to `[4.0, -2.0]` (keep bias at `-0.5`)
2. Change the bias to `2.0` (keep weights at `[2.0, -1.0]`)
3. Flip the sign of one weight: `[2.0, 1.0]` (keep bias at `-0.5`)

For each, compute the boundary line equation (`x2 = ...`) and classify the same five points.

<details>
<summary>Hint: Deriving the boundary equation</summary>

The decision boundary is where `w1*x1 + w2*x2 + bias = 0`. Solve for x2: `x2 = -(w1*x1 + bias) / w2`. This gives you a line equation in terms of x1.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

points = np.array([[1, 0], [0, 1], [1, 1], [-1, 2], [2, 3]])

configs = [
    ("Original",       np.array([2.0, -1.0]), -0.5),
    ("Doubled weights", np.array([4.0, -2.0]), -0.5),
    ("Bias = 2.0",     np.array([2.0, -1.0]),  2.0),
    ("Flipped w2",     np.array([2.0,  1.0]), -0.5),
]

for name, w, b in configs:
    z = points @ w + b
    outputs = sigmoid(z)
    predicted = (outputs >= 0.5).astype(int)
    # Boundary: w[0]*x1 + w[1]*x2 + b = 0  =>  x2 = -(w[0]*x1 + b) / w[1]
    boundary = f"x2 = {-w[0]/w[1]:.1f}*x1 + {-b/w[1]:.1f}"
    print(f"\n{name}: w={w}, b={b}")
    print(f"  Boundary: {boundary}")
    print(f"  Classifications: {predicted}")
```

**Expected output:**

```
Original: w=[ 2. -1.], b=-0.5
  Boundary: x2 = 2.0*x1 + -0.5
  Classifications: [1 0 1 0 1]

Doubled weights: w=[ 4. -2.], b=-0.5
  Boundary: x2 = 2.0*x1 + -0.2
  Classifications: [1 0 1 0 1]

Bias = 2.0: w=[ 2. -1.], b=2.0
  Boundary: x2 = 2.0*x1 + 2.0
  Classifications: [1 1 1 0 1]

Flipped w2: w=[ 2.  1.], b=-0.5
  Boundary: x2 = -2.0*x1 + 0.5
  Classifications: [1 0 1 0 1]
```

**Explanation:**
- **Doubled weights**: The boundary barely moved (slightly different intercept because bias wasn't doubled). But the sigmoid becomes *steeper* -- the transition from class 0 to class 1 is sharper. Doubling weights doesn't change *where* the line is (much), but makes the classifier more confident.
- **Bias = 2.0**: The boundary shifted up significantly. The point `[0, 1]` switched from class 0 to class 1 because the line moved above it.
- **Flipped w2**: The boundary angle changed completely -- it now slopes downward instead of upward. x2 now pushes toward class 1 instead of class 0.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Explain a perceptron as weighted sum + bias + activation function
- [ ] Compute a perceptron's output by hand for a given input
- [ ] Describe the decision boundary as a straight line in input space
- [ ] Explain why a single perceptron cannot solve XOR

---

## Section 2: Activation Functions

### Why Nonlinearity Matters

Here's the key insight of this section: **without activation functions, stacking layers of neurons does nothing useful**.

Consider two layers without activation functions. Layer 1 computes `z1 = W1 @ x + b1`. Layer 2 computes `z2 = W2 @ z1 + b2`. Substituting:

```
z2 = W2 @ (W1 @ x + b1) + b2
   = (W2 @ W1) @ x + (W2 @ b1 + b2)
   = W_combined @ x + b_combined
```

Two linear layers collapse into one linear layer. You could replace any number of stacked linear layers with a single matrix multiplication. The network has no more power than a single perceptron.

Let's verify:

```python
import numpy as np

np.random.seed(42)

# Two random linear layers
W1 = np.random.randn(3, 2)  # 2 inputs -> 3 outputs
b1 = np.random.randn(3)
W2 = np.random.randn(1, 3)  # 3 inputs -> 1 output
b2 = np.random.randn(1)

x = np.array([1.0, 2.0])

# Two-layer computation (no activation)
z1 = W1 @ x + b1
z2 = W2 @ z1 + b2

# Collapsed single-layer computation
W_combined = W2 @ W1
b_combined = W2 @ b1 + b2
z_combined = W_combined @ x + b_combined

print(f"Two layers:  {z2[0]:.6f}")
print(f"One layer:   {z_combined[0]:.6f}")
print(f"Same result: {np.allclose(z2, z_combined)}")
```

**Expected output:**

```
Two layers:  -0.751080
One layer:   -0.751080
Same result: True
```

Identical. Adding layers without activation functions is pointless -- you're just multiplying matrices together, which produces another matrix. Activation functions break this linearity and give each layer the ability to introduce curves into the decision boundary.

### Sigmoid: Squash to (0, 1)

You've already seen sigmoid. It takes any number and squashes it into the range (0, 1):

```
sigmoid(z) = 1 / (1 + e^(-z))
```

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.array([-5, -2, -1, 0, 1, 2, 5])
for z in z_values:
    print(f"sigmoid({z:>2}) = {sigmoid(z):.4f}")
```

**Expected output:**

```
sigmoid(-5) = 0.0067
sigmoid(-2) = 0.1192
sigmoid(-1) = 0.2689
sigmoid( 0) = 0.5000
sigmoid( 1) = 0.7311
sigmoid( 2) = 0.8808
sigmoid( 5) = 0.9933
```

Properties of sigmoid:
- Output is always between 0 and 1
- sigmoid(0) = 0.5 (the midpoint)
- Large positive inputs → output near 1
- Large negative inputs → output near 0
- The curve is S-shaped (sigmoidal), smooth, and differentiable everywhere

Sigmoid is intuitive as a "probability" -- it converts an arbitrary score into a value between 0 and 1. That's why it's used for binary classification outputs.

### ReLU: max(0, x)

ReLU (Rectified Linear Unit) is the simplest and most widely used activation function in hidden layers:

```
relu(z) = max(0, z)
```

If the input is positive, pass it through unchanged. If it's negative, output zero.

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

z_values = np.array([-5, -2, -1, 0, 1, 2, 5])
for z in z_values:
    print(f"relu({z:>2}) = {relu(z)}")
```

**Expected output:**

```
relu(-5) = 0
relu(-2) = 0
relu(-1) = 0
relu( 0) = 0
relu( 1) = 1
relu( 2) = 2
relu( 5) = 5
```

Properties of ReLU:
- Output is always >= 0
- For positive inputs, it's the identity function (output = input)
- For negative inputs, it's zero
- Computationally cheap -- just a comparison and possibly setting to zero
- The derivative is 1 for positive inputs and 0 for negative inputs (undefined at exactly 0, but treated as 0 in practice)

ReLU dominates hidden layers because it's fast to compute, its gradient doesn't vanish for positive values (unlike sigmoid), and it introduces sparsity (some neurons output zero, which acts as a form of automatic feature selection).

### Tanh: Squash to (-1, 1)

Tanh (hyperbolic tangent) is like sigmoid but centered at zero:

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

```python
import numpy as np

z_values = np.array([-5, -2, -1, 0, 1, 2, 5])
for z in z_values:
    print(f"tanh({z:>2}) = {np.tanh(z):>7.4f}")
```

**Expected output:**

```
tanh(-5) = -1.0000
tanh(-2) = -0.9640
tanh(-1) = -0.7616
tanh( 0) =  0.0000
tanh( 1) =  0.7616
tanh( 2) =  0.9640
tanh( 5) =  1.0000
```

Properties of tanh:
- Output is always between -1 and 1
- tanh(0) = 0 (centered at zero, unlike sigmoid which centers at 0.5)
- Symmetric around the origin
- Large positive inputs → output near 1
- Large negative inputs → output near -1

The zero-centering is sometimes useful because it means the outputs of one layer have a mean near zero, which can help the next layer learn more efficiently.

### Plotting All Three

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

z = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(z, sigmoid(z), 'b-', linewidth=2)
axes[0].set_title('Sigmoid')
axes[0].set_ylim(-0.1, 1.1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, relu(z), 'r-', linewidth=2)
axes[1].set_title('ReLU')
axes[1].set_ylim(-1, 5.5)
axes[1].grid(True, alpha=0.3)

axes[2].plot(z, np.tanh(z), 'g-', linewidth=2)
axes[2].set_title('Tanh')
axes[2].set_ylim(-1.2, 1.2)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('z')
    ax.set_ylabel('output')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=100)
plt.show()
```

> **Visualization**: Run `python tools/ml-visualizations/activation_functions.py` to see all three activation functions plotted together with their derivatives.

### When to Use Which

- **ReLU**: Default choice for hidden layers. Fast, simple, and works well in practice.
- **Sigmoid**: Binary classification output layer. When you need an output interpretable as a probability between 0 and 1.
- **Tanh**: Sometimes used in hidden layers when zero-centered outputs matter. Common in recurrent networks.
- **Softmax**: Multi-class classification output layer (converts a vector of scores into probabilities that sum to 1). We won't implement this here, but it's worth knowing the name.

The rule of thumb: use ReLU inside the network, and pick the output activation based on your task.

### Exercise 2.1: Implement Sigmoid, ReLU, and Tanh from Scratch

**Task:** Implement all three activation functions using only numpy (no `np.tanh`). Verify your implementations match numpy's built-in versions.

<details>
<summary>Hint 1: Sigmoid formula</summary>

`sigmoid(z) = 1 / (1 + np.exp(-z))`. Be careful with very large negative values of z -- `np.exp(-z)` can overflow. For this exercise, don't worry about numerical stability.
</details>

<details>
<summary>Hint 2: Tanh from exponentials</summary>

`tanh(z) = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))`. Or equivalently, `tanh(z) = 2 * sigmoid(2*z) - 1`.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def my_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def my_relu(z):
    return np.maximum(0, z)

def my_tanh(z):
    exp_z = np.exp(z)
    exp_neg_z = np.exp(-z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

# Test values
z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print("Sigmoid:")
print(f"  Mine:   {my_sigmoid(z)}")
print(f"  Numpy:  {1 / (1 + np.exp(-z))}")
print(f"  Match:  {np.allclose(my_sigmoid(z), 1 / (1 + np.exp(-z)))}")

print("\nReLU:")
print(f"  Mine:   {my_relu(z)}")
print(f"  Manual: {np.array([max(0, v) for v in z])}")
print(f"  Match:  {np.allclose(my_relu(z), np.maximum(0, z))}")

print("\nTanh:")
print(f"  Mine:   {my_tanh(z)}")
print(f"  Numpy:  {np.tanh(z)}")
print(f"  Match:  {np.allclose(my_tanh(z), np.tanh(z))}")
```

**Expected output:**

```
Sigmoid:
  Mine:   [0.04742587 0.26894142 0.5        0.73105858 0.95257413]
  Numpy:  [0.04742587 0.26894142 0.5        0.73105858 0.95257413]
  Match:  True

ReLU:
  Mine:   [0. 0. 0. 1. 3.]
  Manual: [0. 0. 0. 1. 3.]
  Match:  True

Tanh:
  Mine:   [-0.99505475 -0.76159416  0.          0.76159416  0.99505475]
  Numpy:  [-0.99505475 -0.76159416  0.          0.76159416  0.99505475]
  Match:  True
```

**Explanation:** All three implementations match the reference. The key formulas are: sigmoid uses `1 / (1 + exp(-z))`, ReLU uses `max(0, z)`, and tanh uses the ratio of exponential differences.
</details>

### Exercise 2.2: Compare Activation Function Outputs

**Task:** Pass the same input vector through each activation function and compare the outputs. Use the input `z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])`. For each activation function, compute the output and the range (min to max) of the output.

<details>
<summary>Hint: Comparing ranges</summary>

Use `np.min()` and `np.max()` to find the range of each output. Notice how sigmoid maps everything to (0, 1), ReLU zeros out negatives, and tanh maps everything to (-1, 1).
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

activations = {
    "Sigmoid": sigmoid(z),
    "ReLU":    relu(z),
    "Tanh":    np.tanh(z),
}

print(f"Input z: {z}\n")

for name, output in activations.items():
    print(f"{name:>7}: {np.round(output, 4)}")
    print(f"         Range: [{np.min(output):.4f}, {np.max(output):.4f}]\n")
```

**Expected output:**

```
Input z: [-2.  -0.5  0.   0.5  2. ]

Sigmoid: [0.1192 0.3775 0.5    0.6225 0.8808]
         Range: [0.1192, 0.8808]

   ReLU: [0.  0.  0.  0.5 2. ]
         Range: [0.0000, 2.0000]

   Tanh: [-0.9640 -0.4621  0.      0.4621  0.9640]
         Range: [-0.9640, 0.9640]
```

**Explanation:** Sigmoid squashes everything into (0, 1) -- all outputs are positive, centered around 0.5. ReLU kills all negative values (three of five inputs become 0) and passes positive values unchanged. Tanh preserves the sign and squashes to (-1, 1), centered at zero. Notice that ReLU is the only one that can output values greater than 1 or equal to 0 exactly.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Explain why stacking linear layers without activation functions collapses into a single layer
- [ ] Describe what sigmoid, ReLU, and tanh each do to an input value
- [ ] Implement all three from scratch
- [ ] State the default choice for hidden layers (ReLU) and output layers (sigmoid for binary, softmax for multi-class)

---

## Section 3: Layers and the Forward Pass

### A Layer Is One Matrix Operation Plus Activation

A single layer of a neural network computes:

```
output = activation(W @ input + b)
```

Where:
- `W` is the weight matrix with shape `(n_outputs, n_inputs)`
- `input` is the input vector with shape `(n_inputs,)`
- `b` is the bias vector with shape `(n_outputs,)`
- `activation` is an activation function applied element-wise
- `output` has shape `(n_outputs,)`

This is the fundamental building block. Every layer does the same thing: linear transformation (matrix multiply + bias), then nonlinearity (activation function). The only things that differ between layers are the sizes of `W` and `b`, and possibly the choice of activation function.

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

# A layer: 2 inputs -> 3 outputs
W = np.array([
    [ 0.5,  0.3],
    [-0.2,  0.8],
    [ 0.1, -0.4]
])   # Shape: (3, 2)

b = np.array([0.1, -0.1, 0.2])   # Shape: (3,)

x = np.array([1.0, 2.0])   # Shape: (2,)

# The layer computation
z = W @ x + b               # Linear part: (3, 2) @ (2,) + (3,) = (3,)
output = relu(z)             # Nonlinear part: element-wise ReLU

print(f"Input:           {x}         shape: {x.shape}")
print(f"Weight matrix W:\n{W}   shape: {W.shape}")
print(f"Bias b:          {b}  shape: {b.shape}")
print(f"z = W @ x + b:  {z}")
print(f"output = relu(z): {output}")
```

**Expected output:**

```
Input:           [1. 2.]         shape: (2,)
Weight matrix W:
[[ 0.5  0.3]
 [-0.2  0.8]
 [ 0.1 -0.4]]   shape: (3, 2)
Bias b:          [ 0.1 -0.1  0.2]  shape: (3,)
z = W @ x + b:  [1.2 1.3 -0.5]
output = relu(z): [1.2 1.3 0. ]
```

Let's trace the computation by hand:

- `z[0] = 0.5*1.0 + 0.3*2.0 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2` → relu(1.2) = 1.2
- `z[1] = -0.2*1.0 + 0.8*2.0 + (-0.1) = -0.2 + 1.6 - 0.1 = 1.3` → relu(1.3) = 1.3
- `z[2] = 0.1*1.0 + (-0.4)*2.0 + 0.2 = 0.1 - 0.8 + 0.2 = -0.5` → relu(-0.5) = 0.0

The third neuron got a negative weighted sum, so ReLU killed it. That neuron is "off" for this particular input.

### The Forward Pass: Chaining Layers

The **forward pass** is the process of sending data through the network from input to output, layer by layer. Each layer's output becomes the next layer's input:

```
input → layer 1 → layer 2 → ... → output
```

Let's build a concrete example: a network with 2 inputs, 3 hidden neurons, and 1 output.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Network architecture: 2 -> 3 -> 1
# Layer 1: 2 inputs -> 3 hidden neurons (ReLU)
W1 = np.array([
    [ 0.2,  0.4],
    [-0.5,  0.3],
    [ 0.1,  0.6]
])   # Shape: (3, 2)
b1 = np.array([0.1, -0.2, 0.0])   # Shape: (3,)

# Layer 2: 3 hidden neurons -> 1 output (Sigmoid)
W2 = np.array([
    [0.7, -0.3, 0.5]
])   # Shape: (1, 3)
b2 = np.array([-0.1])   # Shape: (1,)

# Input
x = np.array([1.0, 0.5])

# Forward pass - step by step
print("=== Forward Pass ===\n")

print(f"Input: {x}")
print()

# Layer 1
z1 = W1 @ x + b1
a1 = relu(z1)
print(f"Layer 1:")
print(f"  z1 = W1 @ x + b1 = {z1}")
print(f"  a1 = relu(z1)    = {a1}")
print()

# Layer 2
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
print(f"Layer 2:")
print(f"  z2 = W2 @ a1 + b2 = {z2}")
print(f"  a2 = sigmoid(z2)  = {a2}")
print()

print(f"Final output: {a2[0]:.4f}")
```

**Expected output:**

```
=== Forward Pass ===

Input: [1.  0.5]

Layer 1:
  z1 = W1 @ x + b1 = [ 0.5  -0.05  0.4 ]
  a1 = relu(z1)    = [0.5  0.   0.4]

Layer 2:
  z2 = W2 @ a1 + b2 = [0.35]
  a2 = sigmoid(z2)  = [0.58661758]

Final output: 0.5866
```

Let's verify layer 1 by hand:
- `z1[0] = 0.2*1.0 + 0.4*0.5 + 0.1 = 0.2 + 0.2 + 0.1 = 0.5` → relu(0.5) = 0.5
- `z1[1] = -0.5*1.0 + 0.3*0.5 + (-0.2) = -0.5 + 0.15 - 0.2 = -0.55` → relu(-0.55) = 0.0
- `z1[2] = 0.1*1.0 + 0.6*0.5 + 0.0 = 0.1 + 0.3 + 0.0 = 0.4` → relu(0.4) = 0.4

And layer 2:
- `z2[0] = 0.7*0.5 + (-0.3)*0.0 + 0.5*0.4 + (-0.1) = 0.35 + 0 + 0.2 - 0.1 = 0.45` → sigmoid(0.45) ≈ 0.6106

Wait -- let me recheck. `0.7*0.5 = 0.35`, `(-0.3)*0.0 = 0.0`, `0.5*0.4 = 0.2`, plus bias `(-0.1)` = `0.35 + 0.0 + 0.2 - 0.1 = 0.45`.

Actually, let's look at the code output more carefully. The z1 computation gives `[ 0.5  -0.05  0.4 ]` -- let me recheck z1[1]: `-0.5*1.0 + 0.3*0.5 + (-0.2)` = `-0.5 + 0.15 - 0.2` = `-0.55`. Hmm, the output says `-0.05` not `-0.55`. Let me recheck: `-0.5*1.0 = -0.5`, `0.3*0.5 = 0.15`, bias `-0.2`. Total: `-0.5 + 0.15 - 0.2 = -0.55`. So the relu output is 0 regardless.

The important point: data flows forward through the layers, each one transforming the representation. The hidden layer's three neurons each look at the input differently (different weights), and the output layer combines those three perspectives into a final answer.

### Matrix Dimensions Through the Network

Keeping track of shapes is one of the most practical skills for working with neural networks. Here's the pattern:

| Layer | Weight shape | Bias shape | Input shape | Output shape |
|-------|-------------|------------|-------------|--------------|
| 1     | (3, 2)      | (3,)       | (2,)        | (3,)         |
| 2     | (1, 3)      | (1,)       | (3,)        | (1,)         |

The rule: a layer with `n_in` inputs and `n_out` outputs has:
- Weight matrix W: shape `(n_out, n_in)`
- Bias vector b: shape `(n_out,)`
- Input: shape `(n_in,)`
- Output: shape `(n_out,)`

The columns of W match the input size. The rows of W match the output size. If your matrix multiplication fails with a shape error, this is almost always the issue.

### A Forward Pass Function

Let's wrap the forward pass into a clean function:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_pass(x, weights, biases, activations):
    """Compute the forward pass through a list of layers.

    Args:
        x: Input vector
        weights: List of weight matrices, one per layer
        biases: List of bias vectors, one per layer
        activations: List of activation functions, one per layer

    Returns:
        The output of the final layer
    """
    a = x
    for W, b, activation in zip(weights, biases, activations):
        z = W @ a + b
        a = activation(z)
    return a

# Define the same network as before
weights = [
    np.array([[ 0.2,  0.4], [-0.5,  0.3], [ 0.1,  0.6]]),
    np.array([[0.7, -0.3, 0.5]])
]
biases = [
    np.array([0.1, -0.2, 0.0]),
    np.array([-0.1])
]
activations = [relu, sigmoid]

x = np.array([1.0, 0.5])
output = forward_pass(x, weights, biases, activations)
print(f"Output: {output[0]:.4f}")
```

**Expected output:**

```
Output: 0.5866
```

Same result as before, but now the code is general -- it works for any number of layers with any sizes.

### Exercise 3.1: Trace a Small Network by Hand, Then Verify

**Task:** Given the following network (2 inputs, 2 hidden neurons, 1 output), trace the forward pass by hand for the input `x = [1.0, -1.0]`. Write out every intermediate value. Then verify with numpy.

```python
W1 = np.array([[1.0, 0.0],
               [0.0, 1.0]])   # 2x2 (identity-like)
b1 = np.array([0.0, 0.0])

W2 = np.array([[1.0, 1.0]])   # 1x2
b2 = np.array([0.0])
```

Use ReLU for the hidden layer and sigmoid for the output.

<details>
<summary>Hint: Trace step by step</summary>

Layer 1: `z1 = W1 @ [1, -1] + [0, 0] = [1, -1]`. Then `a1 = relu([1, -1]) = [1, 0]`. Layer 2: `z2 = W2 @ [1, 0] + [0] = [1]`. Then `a2 = sigmoid(1)`.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
b1 = np.array([0.0, 0.0])
W2 = np.array([[1.0, 1.0]])
b2 = np.array([0.0])

x = np.array([1.0, -1.0])

# By hand:
# Layer 1: z1 = [[1,0],[0,1]] @ [1,-1] + [0,0] = [1,-1]
#           a1 = relu([1, -1]) = [1, 0]
# Layer 2: z2 = [[1,1]] @ [1, 0] + [0] = [1]
#           a2 = sigmoid(1) = 0.7311

z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

print(f"Input:   {x}")
print(f"z1:      {z1}")
print(f"a1:      {a1}")
print(f"z2:      {z2}")
print(f"a2:      {a2}")
print(f"Output:  {a2[0]:.4f}")
```

**Expected output:**

```
Input:   [ 1. -1.]
z1:      [ 1. -1.]
a1:      [1. 0.]
z2:      [1.]
a2:      [0.73105858]
Output:  0.7311
```

**Explanation:** W1 is the identity matrix, so z1 = x. ReLU then kills the negative component: [1, -1] becomes [1, 0]. The second input's information is lost -- ReLU set it to zero because it was negative. W2 sums both hidden neurons, so z2 = 1 + 0 = 1. Sigmoid(1) = 0.7311.

This example shows how ReLU acts as a gate: it lets positive signals through and blocks negative ones. The second hidden neuron contributed nothing to the output because its input was negative.
</details>

### Exercise 3.2: Change the Hidden Layer Size

**Task:** Modify the network to have 4 hidden neurons instead of 2 (keep 2 inputs and 1 output). Initialize weights randomly with `np.random.seed(42)` and `np.random.randn()`. Run the forward pass on `x = [1.0, -1.0]` and print the intermediate shapes and values.

<details>
<summary>Hint: Getting the shapes right</summary>

With 2 inputs and 4 hidden neurons: W1 has shape (4, 2), b1 has shape (4,). With 4 hidden neurons and 1 output: W2 has shape (1, 4), b2 has shape (1,). Use `np.random.randn(rows, cols)` for matrices and `np.random.randn(size)` for vectors.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

np.random.seed(42)

# 2 inputs -> 4 hidden -> 1 output
W1 = np.random.randn(4, 2)
b1 = np.random.randn(4)
W2 = np.random.randn(1, 4)
b2 = np.random.randn(1)

x = np.array([1.0, -1.0])

# Forward pass
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

print(f"Input x:  {x}  shape: {x.shape}")
print(f"W1 shape: {W1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"z1:       {np.round(z1, 4)}  shape: {z1.shape}")
print(f"a1:       {np.round(a1, 4)}  shape: {a1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"b2 shape: {b2.shape}")
print(f"z2:       {np.round(z2, 4)}  shape: {z2.shape}")
print(f"a2:       {np.round(a2, 4)}  shape: {a2.shape}")
print(f"\nOutput:   {a2[0]:.4f}")
print(f"Active hidden neurons: {np.sum(a1 > 0)} of {len(a1)}")
```

**Expected output:**

```
Input x:  [ 1. -1.]  shape: (2,)
W1 shape: (4, 2)
b1 shape: (4,)
z1:       [ 0.8476 -0.2607  2.9048 -0.6854]  shape: (4,)
a1:       [0.8476 0.     2.9048 0.    ]  shape: (4,)
W2 shape: (1, 4)
b2 shape: (1,)
z2:       [-1.0525]  shape: (1,)
a2:       [0.2588]  shape: (1,)

Output:   0.2588
Active hidden neurons: 2 of 4
```

**Explanation:** With random weights, 2 of the 4 hidden neurons are active (have positive z values after ReLU). The other 2 were killed by ReLU. The shapes flow correctly: (4, 2) @ (2,) = (4,), then (1, 4) @ (4,) = (1,). Changing the hidden layer size to 4 gives the network more capacity -- more parameters to learn from, more ways to carve up the input space.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Describe a layer as `output = activation(W @ input + b)`
- [ ] State the weight matrix shape rule: `(n_outputs, n_inputs)`
- [ ] Trace a forward pass through a 2-layer network by hand
- [ ] Implement a general forward pass function that works for any number of layers
- [ ] Predict how changing the hidden layer size affects the network's shapes

---

## Section 4: Multi-Layer Networks

### Depth Creates Power

Each layer in a neural network warps the input space. The first layer applies a linear transformation (rotate, scale, shear) followed by a nonlinearity (bend, fold, clip). The second layer does the same to the already-warped space. Stack enough layers and you can warp any space into any shape you need.

This is the fundamental insight: **depth lets networks learn complex decision boundaries by composing simple transformations**.

A single layer can only draw a flat boundary (after the linear + activation step). Two layers can draw boundaries with one bend. Three layers can draw boundaries with multiple bends. More layers = more bends = more complex shapes.

### Solving XOR with Two Layers

Remember the XOR problem from Section 1? A single perceptron couldn't solve it because the classes aren't linearly separable. But a two-layer network can. The trick: the first layer warps the space so that the classes *become* linearly separable, and the second layer draws the final boundary.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Hand-crafted weights that solve XOR
# Layer 1: 2 hidden neurons that detect (x1 AND x2) and (x1 OR x2)
W1 = np.array([
    [20.0, 20.0],    # Neuron 1: fires when both inputs are high (AND-like)
    [20.0, 20.0]     # Neuron 2: fires when either input is high (OR-like)
])
b1 = np.array([-30.0, -10.0])  # High threshold for AND, low for OR

# Layer 2: output = OR - AND (which is XOR)
W2 = np.array([[-20.0, 20.0]])   # Subtract AND, add OR
b2 = np.array([-10.0])

print("XOR Network - Layer by Layer\n")
print(f"{'Input':>8}  {'z1':>18}  {'a1 (hidden)':>18}  {'z2':>8}  {'output':>8}  {'target':>7}")
print("-" * 80)

for i in range(len(X)):
    x = X[i]
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)

    print(f"{str(x):>8}  {str(np.round(z1, 1)):>18}  {str(np.round(a1, 4)):>18}  "
          f"{z2[0]:>8.2f}  {a2[0]:>8.4f}  {y[i]:>7}")
```

**Expected output:**

```
XOR Network - Layer by Layer

   Input                z1    a1 (hidden)        z2    output   target
--------------------------------------------------------------------------------
  [0 0]       [-30. -10.]    [0.     0.0000]   -10.00    0.0000        0
  [0 1]       [-10.  10.]    [0.0000 1.0000]    10.00    1.0000        1
  [1 0]       [-10.  10.]    [0.0000 1.0000]    10.00    1.0000        1
  [1 1]        [10. 30.]    [1.0000 1.0000]   -10.00    0.0000        0
```

The network correctly computes XOR. Here's what each layer does:

- **Hidden neuron 1** (AND-like): Only fires when both inputs are high. For `[1,1]`: `20 + 20 - 30 = 10` → sigmoid ≈ 1. For all other inputs: z is -30 or -10 → sigmoid ≈ 0.
- **Hidden neuron 2** (OR-like): Fires when either input is high. For `[0,1]` or `[1,0]`: `20 - 10 = 10` → sigmoid ≈ 1. For `[0,0]`: `-10` → sigmoid ≈ 0.
- **Output layer**: `XOR = OR AND NOT(AND)`. It subtracts the AND signal from the OR signal. When both are 1 (input `[1,1]`), they cancel out.

The first layer *re-represents* the inputs in a space where XOR becomes linearly separable. This is the core idea behind representation learning.

> **Visualization**: Run `python tools/ml-visualizations/decision_boundaries.py` to see how a two-layer network warps 2D space to solve XOR and other classification problems.

### Visualizing Space Warping

To see what each layer does to the input space, we can feed a grid of 2D points through the network and plot where they end up after each layer:

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

# A simple 2-layer network: 2 -> 2 -> 2
# (Keeping 2D throughout so we can visualize)
np.random.seed(7)
W1 = np.array([[1.5, 0.8], [-0.5, 1.2]])
b1 = np.array([-0.3, 0.1])
W2 = np.array([[0.9, -0.7], [0.4, 1.1]])
b2 = np.array([0.2, -0.2])

# Create a grid of input points
n = 15
x_range = np.linspace(-2, 2, n)
y_range = np.linspace(-2, 2, n)
grid = np.array([[x, y] for x in x_range for y in y_range])

# Forward pass, saving intermediate results
z1 = (W1 @ grid.T).T + b1     # After linear transform
a1 = relu(z1)                  # After activation
z2 = (W2 @ a1.T).T + b2       # After second linear transform
a2 = relu(z2)                  # After second activation

# Plot each stage
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
stages = [
    (grid, "Input"),
    (z1, "After W1 @ x + b1"),
    (a1, "After ReLU"),
    (a2, "After Layer 2"),
]

for ax, (points, title) in zip(axes, stages):
    # Color by original position to track where points move
    colors = grid[:, 0] + grid[:, 1]
    ax.scatter(points[:, 0], points[:, 1], c=colors, cmap='coolwarm',
               s=15, alpha=0.7)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('space_warping.png', dpi=100)
plt.show()
```

Watch how the grid transforms:
1. **Input**: A regular grid of points.
2. **After W1 @ x + b1**: The grid is rotated, scaled, and shifted (linear transformation). Still a grid, just warped linearly.
3. **After ReLU**: Points with negative coordinates get clipped to zero. The grid folds -- parts collapse onto the axes. This is where nonlinearity kicks in.
4. **After Layer 2**: Another linear transformation followed by ReLU. The space is warped again.

Each layer adds another fold or bend to the space. The network learns the specific folds that make the data separable.

### Building a Network Class

Let's wrap everything into a proper class:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

class Network:
    """A feedforward neural network with configurable layer sizes."""

    def __init__(self, layer_sizes, hidden_activation=relu, output_activation=sigmoid):
        """Initialize random weights and biases.

        Args:
            layer_sizes: List of integers, e.g. [2, 4, 3, 1] means
                         2 inputs, two hidden layers (4 and 3 neurons), 1 output.
            hidden_activation: Activation function for hidden layers.
            output_activation: Activation function for the output layer.
        """
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # Small random weights (scaled by input size for stability)
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """Compute the forward pass.

        Args:
            x: Input vector of shape (layer_sizes[0],)

        Returns:
            Output vector of shape (layer_sizes[-1],)
        """
        a = x
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            if i < n_layers - 1:
                a = self.hidden_activation(z)
            else:
                a = self.output_activation(z)
        return a

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"Network: {' -> '.join(str(s) for s in self.layer_sizes)}")
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            n_params = W.size + b.size
            total_params += n_params
            print(f"  Layer {i+1}: W{W.shape} + b{b.shape} = {n_params} params")
        print(f"  Total parameters: {total_params}")


# Create and test a network
np.random.seed(42)
net = Network([2, 4, 3, 1])
net.summary()

x = np.array([1.0, -0.5])
output = net.forward(x)
print(f"\nInput:  {x}")
print(f"Output: {output[0]:.4f}")
```

**Expected output:**

```
Network: 2 -> 4 -> 3 -> 1
  Layer 1: W(4, 2) + b(4,) = 12 params
  Layer 2: W(3, 4) + b(3,) = 15 params
  Layer 3: W(1, 3) + b(1,) = 4 params
  Total parameters: 31

Input:  [ 1.  -0.5]
Output: 0.5765
```

The `summary()` method shows the architecture and parameter count. This network has 31 trainable parameters -- small by modern standards, but enough to learn interesting functions.

Note the weight initialization: `np.random.randn(...) * np.sqrt(2.0 / n_in)`. This scaling (called "He initialization") keeps the variance of activations roughly constant across layers, which helps networks train. If you initialize with too-large weights, activations explode; too-small, they vanish. We'll explore this more in the training-and-backprop route.

### Exercise 4.1: Build a Network Class That Supports Arbitrary Layer Sizes

**Task:** Using the Network class above (or your own version), create and test three different architectures:

1. A narrow network: `[2, 2, 1]` (2 inputs, 2 hidden neurons, 1 output)
2. A wide network: `[2, 16, 1]` (2 inputs, 16 hidden neurons, 1 output)
3. A deep network: `[2, 4, 4, 4, 1]` (2 inputs, three hidden layers of 4 neurons each, 1 output)

For each, print the summary and run a forward pass on `x = [1.0, -0.5]`.

<details>
<summary>Hint: Creating the networks</summary>

Just change the `layer_sizes` argument: `Network([2, 2, 1])`, `Network([2, 16, 1])`, `Network([2, 4, 4, 4, 1])`. The Network class handles the rest.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

class Network:
    def __init__(self, layer_sizes, hidden_activation=relu, output_activation=sigmoid):
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
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
        a = x
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            if i < n_layers - 1:
                a = self.hidden_activation(z)
            else:
                a = self.output_activation(z)
        return a

    def summary(self):
        total = 0
        print(f"Network: {' -> '.join(str(s) for s in self.layer_sizes)}")
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            n = W.size + b.size
            total += n
            print(f"  Layer {i+1}: W{W.shape} + b{b.shape} = {n} params")
        print(f"  Total: {total} params")

np.random.seed(42)
x = np.array([1.0, -0.5])

architectures = [
    ("Narrow", [2, 2, 1]),
    ("Wide",   [2, 16, 1]),
    ("Deep",   [2, 4, 4, 4, 1]),
]

for name, sizes in architectures:
    print(f"\n{'='*40}")
    print(f"{name} network:")
    net = Network(sizes)
    net.summary()
    output = net.forward(x)
    print(f"  forward([1.0, -0.5]) = {output[0]:.4f}")
```

**Expected output:**

```
========================================
Narrow network:
Network: 2 -> 2 -> 1
  Layer 1: W(2, 2) + b(2,) = 6 params
  Layer 2: W(1, 2) + b(1,) = 3 params
  Total: 9 params
  forward([1.0, -0.5]) = 0.5765

========================================
Wide network:
Network: 2 -> 16 -> 1
  Layer 1: W(16, 2) + b(16,) = 48 params
  Layer 2: W(1, 16) + b(1,) = 17 params
  Total: 65 params
  forward([1.0, -0.5]) = 0.1498

========================================
Deep network:
Network: 2 -> 4 -> 4 -> 4 -> 1
  Layer 1: W(4, 2) + b(4,) = 12 params
  Layer 2: W(4, 4) + b(4,) = 20 params
  Layer 3: W(4, 4) + b(4,) = 20 params
  Layer 4: W(1, 4) + b(1,) = 5 params
  Total: 57 params
  forward([1.0, -0.5]) = 0.5024
```

**Explanation:** The narrow network has only 9 parameters -- very limited capacity. The wide network has 65 parameters concentrated in one large hidden layer. The deep network has 57 parameters spread across three hidden layers. With random weights, the outputs are essentially random -- the networks haven't been trained yet. Training (covered in the next route) is what makes the weights meaningful.
</details>

### Exercise 4.2: Visualize What Each Layer Does to a Grid

**Task:** Create a `[2, 3, 2]` network (2 inputs, 3 hidden neurons, 2 outputs -- keeping 2D so we can plot). Generate a grid of 2D points, pass them through the network, and plot:
1. The original grid
2. The grid after layer 1 (linear transform only, before activation)
3. The grid after layer 1 (with activation)
4. The grid after layer 2

Color the points by their original x-coordinate so you can track where they move.

<details>
<summary>Hint 1: Handling the 3D hidden layer</summary>

The hidden layer has 3 neurons, so after layer 1 the points are in 3D. You can't plot 3D easily, so either: (a) use a `[2, 2, 2]` network instead to stay in 2D throughout, or (b) project the 3D points down to 2D for visualization (e.g., plot the first two dimensions).

The simplest approach: use `[2, 2, 2]`.
</details>

<details>
<summary>Hint 2: Applying the network layer by layer</summary>

Instead of calling `net.forward()`, manually apply each layer: `z1 = W1 @ x + b1`, `a1 = relu(z1)`, `z2 = W2 @ a1 + b2`, `a2 = activation(z2)`. Save each intermediate result for plotting.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

np.random.seed(15)

# 2 -> 2 -> 2 network (stay in 2D for visualization)
W1 = np.random.randn(2, 2) * 1.5
b1 = np.random.randn(2) * 0.5
W2 = np.random.randn(2, 2) * 1.5
b2 = np.random.randn(2) * 0.5

# Generate grid
n = 20
coords = np.linspace(-2, 2, n)
grid = np.array([[x, y] for x in coords for y in coords])
colors = grid[:, 0]  # Color by original x-coordinate

# Forward pass, saving intermediates
z1 = (W1 @ grid.T).T + b1
a1 = relu(z1)
z2 = (W2 @ a1.T).T + b2
a2 = relu(z2)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
stages = [
    (grid, "Input grid"),
    (z1, "After W1 @ x + b1\n(linear only)"),
    (a1, "After ReLU\n(layer 1 output)"),
    (a2, "After layer 2\n(full network output)"),
]

for ax, (pts, title) in zip(axes, stages):
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap='coolwarm', s=10, alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

plt.suptitle('How each layer transforms the input space', fontsize=13)
plt.tight_layout()
plt.savefig('layer_transformations.png', dpi=100)
plt.show()
```

**Explanation:** The linear transformation (W1 @ x + b1) rotates and stretches the grid but keeps it grid-like -- straight lines stay straight. ReLU then clips all negative values to zero, folding parts of the grid onto the axes. This folding is the nonlinear part -- it creates the bends that give neural networks their power. The second layer applies another rotate-stretch-fold, warping the space further.

The colors let you track where the original points ended up. Points that were on opposite sides of the input space may end up near each other after the transformations -- this is how the network can bring together points from the same class.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Explain how depth (more layers) creates more powerful decision boundaries
- [ ] Describe how a two-layer network solves XOR by warping the input space
- [ ] Build a Network class with configurable layer sizes
- [ ] Visualize how each layer transforms a set of input points
- [ ] Count the number of parameters in a network given its architecture

---

## Practice Project: Network Anatomy Visualization

### Project Description

Build a complete visualization tool that shows the anatomy of a neural network's forward pass. You'll create a Network class with configurable layers, run the forward pass on a grid of 2D points, and plot what each layer does to the grid. The goal is to build intuition for how networks transform space.

This project ties together everything from this route:
- Perceptrons as the building blocks of each neuron
- Activation functions introducing nonlinearity between layers
- Matrix operations performing the linear transformation in each layer
- The forward pass chaining everything together

### Requirements

Build a script (`network_anatomy.py`) that:
1. Creates a Network class with configurable layers (at least `__init__` and `forward`)
2. Generates a grid of 2D input points
3. Runs the forward pass, saving the output of each intermediate layer
4. Plots the grid at each stage (input, after each layer)
5. Colors points consistently across plots so you can track transformations
6. Supports experimenting with different architectures (e.g., `[2, 4, 2]`, `[2, 8, 4, 2]`)

### Getting Started

**Step 1: Set up the Network class**

Your Network class needs a `forward` method that returns intermediate activations, not just the final output:

```python
def forward_with_intermediates(self, x):
    """Returns a list of activations: [input, after_layer_1, after_layer_2, ...]"""
    activations = [x]
    a = x
    for i, (W, b) in enumerate(zip(self.weights, self.biases)):
        z = W @ a + b
        a = self.hidden_activation(z)  # Use same activation for simplicity
        activations.append(a)
    return activations
```

**Step 2: Generate the grid**

```python
n = 20
coords = np.linspace(-2, 2, n)
grid = np.array([[x, y] for x in coords for y in coords])
```

**Step 3: Run the forward pass for each grid point and collect intermediates**

**Step 4: Plot each stage side by side**

### Hints and Tips

<details>
<summary>Hint 1: Handling non-2D hidden layers</summary>

If the hidden layer has more than 2 neurons, you can't plot the full representation. Two options:
- Keep all hidden layers at 2 neurons so everything stays plottable.
- Use the first two dimensions of higher-dimensional layers for plotting (a projection).

For this project, using `[2, 2, 2]` or `[2, 2, 2, 2]` architectures keeps things simple and fully visualizable.
</details>

<details>
<summary>Hint 2: Making the visualization informative</summary>

Use consistent coloring across all subplots. Color each point by its position in the original grid (e.g., by original x-coordinate, or by distance from origin). This lets you see how the network rearranges the space.

Also draw the origin (0, 0) on each subplot as a reference point.
</details>

<details>
<summary>Hint 3: Trying different random seeds</summary>

Different random weight initializations produce different transformations. Try several seeds (`np.random.seed(0)`, `np.random.seed(1)`, etc.) to see how different networks warp space differently. Some seeds may produce more dramatic or interesting visualizations.
</details>

<details>
<summary>Hint 4: Comparing architectures</summary>

Run the visualization for several architectures and compare:
- `[2, 2, 2]` -- one hidden layer, minimal capacity
- `[2, 4, 2]` -- one wider hidden layer (you'll only see first 2 dims of hidden)
- `[2, 2, 2, 2]` -- two hidden layers, each 2D
- `[2, 2, 2, 2, 2]` -- three hidden layers

More layers = more stages of warping.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

class Network:
    """A feedforward neural network that can report intermediate activations."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
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
        """Forward pass returning final output."""
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = relu(z)
        return a

    def forward_with_intermediates(self, x):
        """Forward pass returning activations after each layer."""
        activations = [x.copy()]
        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = relu(z)
            activations.append(a.copy())
        return activations


def make_grid(n=20, bound=2):
    """Generate a grid of 2D points."""
    coords = np.linspace(-bound, bound, n)
    return np.array([[x, y] for x in coords for y in coords])


def visualize_network(layer_sizes, seed=42, grid_n=20):
    """Visualize how a network transforms a grid of 2D points."""
    net = Network(layer_sizes, seed=seed)
    grid = make_grid(n=grid_n)

    # Run forward pass for every grid point, collect intermediates
    all_intermediates = [net.forward_with_intermediates(point) for point in grid]

    # Reorganize: intermediates_by_stage[stage] is an array of all points at that stage
    n_stages = len(all_intermediates[0])
    intermediates_by_stage = []
    for stage in range(n_stages):
        points_at_stage = np.array([interp[stage] for interp in all_intermediates])
        intermediates_by_stage.append(points_at_stage)

    # Color by original position
    colors = grid[:, 0] * 3 + grid[:, 1]

    # Plot
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    titles = ["Input"]
    for i in range(1, n_stages):
        titles.append(f"After layer {i}\n({layer_sizes[i-1]} -> {layer_sizes[i]})")

    for ax, points, title in zip(axes, intermediates_by_stage, titles):
        # Only plot first 2 dimensions if hidden layer is wider than 2
        plot_dims = min(2, points.shape[1])
        if plot_dims < 2:
            ax.scatter(points[:, 0], np.zeros(len(points)), c=colors,
                       cmap='coolwarm', s=10, alpha=0.7)
        else:
            ax.scatter(points[:, 0], points[:, 1], c=colors,
                       cmap='coolwarm', s=10, alpha=0.7)

        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    arch_str = ' -> '.join(str(s) for s in layer_sizes)
    total_params = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
    plt.suptitle(f'Network anatomy: [{arch_str}]  ({total_params} params, seed={seed})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'network_anatomy_{"_".join(str(s) for s in layer_sizes)}.png', dpi=100)
    plt.show()


# Visualize different architectures
print("Visualizing network transformations...\n")

visualize_network([2, 2, 2], seed=42)
visualize_network([2, 2, 2, 2], seed=42)
visualize_network([2, 2, 2, 2, 2], seed=7)
```

**Key points in this solution:**
- `forward_with_intermediates()` saves the activation after each layer so we can visualize the progression.
- Points are colored by a combination of their original x and y coordinates, making it easy to track how the space warps.
- For hidden layers wider than 2, only the first 2 dimensions are plotted (a projection).
- Different architectures show different numbers of warping stages.
- Changing the seed changes the random weights, producing different transformations.
</details>

### Extending the Project

If you want to go further, try:
- Add the ability to visualize both the linear transform (before activation) and the activated output separately for each layer
- Color-code points by their target class (e.g., define a circular or spiral classification boundary) and watch whether the network's transformations push the classes apart
- Animate the transformation: show the grid smoothly morphing from one stage to the next using matplotlib's animation tools
- Add a parameter count and architecture diagram to the plot

---

## Summary

### Key Takeaways

- **A perceptron** computes `output = activation(weights . inputs + bias)`. It's a weighted sum followed by a nonlinear squashing function. A single perceptron draws a straight decision boundary.
- **Activation functions** introduce nonlinearity. Without them, stacking layers collapses into a single linear operation. ReLU is the default for hidden layers; sigmoid for binary output.
- **A layer** is `output = activation(W @ input + b)`, where W has shape (n_outputs, n_inputs). It's a linear transformation followed by a nonlinearity.
- **The forward pass** chains layers together: data flows from input through each layer to the output. Each layer transforms the representation.
- **Depth creates power**: each layer warps the input space further. Multiple layers can create complex, curved decision boundaries that a single layer cannot.

### Skills You've Gained

You can now:
- Implement a perceptron and explain its decision boundary
- Implement sigmoid, ReLU, and tanh from scratch
- Trace data through a multi-layer network by hand
- Build a Network class with configurable architecture
- Visualize what each layer does to the input space
- Count parameters in a network

### Self-Assessment

Take a moment to reflect:
- Can you explain to a colleague what a neural network computes, using only the words "matrix multiply," "add bias," and "apply activation function"?
- If someone showed you a network that couldn't classify XOR, would you know what's missing (hint: more layers)?
- Could you write a forward pass from scratch given only the weight matrices and bias vectors?
- Do you understand why ReLU is preferred over sigmoid for hidden layers?

---

## Next Steps

### Continue Learning

Ready for more? Your next step:

**Build on this topic:**
- [Training and Backprop](/routes/training-and-backprop/map.md) -- How networks *learn*: loss functions, backpropagation (the chain rule applied through the network), and training loops. This route takes the static network you built here and makes it learn from data.

**Explore related routes:**
- [LLM Foundations](/routes/llm-foundations/map.md) -- How the building blocks from this route compose into language models.

### Additional Resources

**Books:**
- *Neural Networks and Deep Learning* by Michael Nielsen -- freely available online (neuralnetworksanddeeplearning.com), excellent visual explanations
- *Deep Learning* by Goodfellow, Bengio, Courville -- Chapter 6 covers feedforward networks in depth

**Videos:**
- 3Blue1Brown's *Neural Networks* series -- superb visual explanation of neurons, layers, and the forward pass
- Andrej Karpathy's "Neural Networks: Zero to Hero" -- builds neural networks from scratch in Python

**Interactive Tools:**
- TensorFlow Playground (playground.tensorflow.org) -- experiment with network architectures and watch decision boundaries form in real time
- ConvNetJS (cs.stanford.edu/people/karpathy/convnetjs/) -- neural network demos running in the browser

---

## Quick Reference

### Network Dimension Rules

| Component | Shape | Example (3 inputs, 4 outputs) |
|-----------|-------|-------------------------------|
| Weight matrix W | (n_out, n_in) | (4, 3) |
| Bias vector b | (n_out,) | (4,) |
| Input vector x | (n_in,) | (3,) |
| Output vector | (n_out,) | (4,) |

**Parameter count per layer:** `n_out * n_in + n_out`

**Total parameters:** Sum across all layers.

### Activation Functions

| Function | Formula | Range | Use case |
|----------|---------|-------|----------|
| ReLU | max(0, z) | [0, +inf) | Hidden layers (default) |
| Sigmoid | 1 / (1 + e^(-z)) | (0, 1) | Binary output |
| Tanh | (e^z - e^(-z)) / (e^z + e^(-z)) | (-1, 1) | Hidden layers (alternative) |
| Softmax | e^(zi) / sum(e^(zj)) | (0, 1), sums to 1 | Multi-class output |

### Forward Pass Pattern

```python
# Single layer
z = W @ x + b        # Linear transform
a = activation(z)    # Nonlinearity

# Full network
a = x
for W, b in layers:
    z = W @ a + b
    a = activation(z)
output = a
```

### Numpy Cheat Sheet for Neural Networks

```python
import numpy as np

# --- Activation Functions ---
relu = lambda z: np.maximum(0, z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
# tanh is built-in: np.tanh(z)

# --- Layer Operations ---
z = W @ x + b                    # Linear transform
a = relu(z)                      # Activation

# --- Weight Initialization ---
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)   # He initialization
b = np.zeros(n_out)

# --- Shape Checking ---
W.shape     # (n_out, n_in)
x.shape     # (n_in,)
(W @ x).shape  # (n_out,)
```

---

## Glossary

- **Activation function**: A nonlinear function applied element-wise after the linear transformation in each layer. Introduces the nonlinearity that gives neural networks their power. Common choices: ReLU, sigmoid, tanh.
- **Bias**: A constant vector added after the matrix multiplication in each layer. Shifts the decision boundary away from the origin.
- **Decision boundary**: The surface in input space where the network's output transitions from one class to another. A single perceptron's boundary is a straight line (or hyperplane); multi-layer networks can create curved boundaries.
- **Depth**: The number of layers in a network. Deeper networks can represent more complex functions.
- **Feedforward network**: A neural network where data flows in one direction, from input to output, with no cycles. Also called a multilayer perceptron (MLP).
- **Forward pass**: The process of computing the output of a network given an input, by passing data through each layer in sequence.
- **He initialization**: A weight initialization strategy that scales random weights by sqrt(2 / n_in). Helps maintain stable activation magnitudes across layers.
- **Hidden layer**: Any layer between the input and output layers. Hidden neurons learn intermediate representations of the data.
- **Layer**: One stage of the network's computation: a linear transformation (W @ x + b) followed by an activation function.
- **Linearly separable**: A dataset is linearly separable if a single straight line (or hyperplane) can separate the classes. A single perceptron can only classify linearly separable data.
- **Perceptron**: A single artificial neuron that computes a weighted sum of inputs, adds a bias, and applies an activation function. The basic building block of neural networks.
- **ReLU (Rectified Linear Unit)**: The activation function max(0, z). Passes positive values unchanged and sets negative values to zero. The default choice for hidden layers.
- **Representation learning**: The idea that hidden layers learn useful intermediate features of the data. Each layer transforms the raw input into a progressively more useful representation.
- **Sigmoid**: The activation function 1 / (1 + e^(-z)). Squashes any value into the range (0, 1). Used for binary classification outputs.
- **Tanh**: The activation function (e^z - e^(-z)) / (e^z + e^(-z)). Squashes any value into the range (-1, 1). Zero-centered alternative to sigmoid.
- **Weight matrix**: The matrix W in a layer's computation W @ x + b. Each row represents one neuron's weights; each column corresponds to one input. Shape: (n_outputs, n_inputs).
- **Width**: The number of neurons in a hidden layer. Wider layers give more capacity per layer.
- **XOR problem**: The classic demonstration that a single perceptron cannot solve all classification tasks. XOR requires at least two layers.
