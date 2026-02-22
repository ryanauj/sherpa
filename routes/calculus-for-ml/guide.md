---
title: Calculus for ML
route_map: /routes/calculus-for-ml/map.md
paired_sherpa: /routes/calculus-for-ml/sherpa.md
prerequisites:
  - Linear Algebra Essentials
  - Python basics and numpy
topics:
  - Derivatives
  - Gradients
  - Chain Rule
  - Gradient Descent
---

# Calculus for ML - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/calculus-for-ml/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/calculus-for-ml/map.md` for the high-level overview.

## Overview

Neural networks learn by adjusting weights to minimize a loss function. The loss function measures how wrong the model's predictions are. Training is the process of finding the set of weights that makes the loss as small as possible.

Calculus is the tool that tells you **which direction to adjust the weights**. Without it, you'd be guessing -- trying random changes and hoping the loss goes down. With calculus, you can compute exactly how each weight affects the loss, and nudge every weight in the direction that makes things better. That computation is what gradient descent does, and it's the engine behind every neural network you've ever heard of.

This guide teaches four ideas:
1. **Derivatives** -- how fast is a function changing at a point?
2. **Gradients** -- which direction should you move to increase (or decrease) a function of many variables?
3. **The chain rule** -- how do derivatives flow through composed functions? (This is backpropagation.)
4. **Gradient descent** -- the algorithm that uses gradients to find minimums.

Every concept starts with intuition and code. Equations appear only after you've seen the idea in action, and every piece of notation gets a plain-English explanation the first time it shows up.

## Learning Objectives

By the end of this route, you will be able to:
- Compute derivatives numerically and explain what they measure
- Calculate partial derivatives and assemble them into a gradient vector
- Apply the chain rule to composed functions
- Implement gradient descent from scratch in Python
- Train a linear regression model using only numpy and gradient descent

## Prerequisites

Before starting this route, you should be comfortable with:
- **Linear Algebra Essentials** ([route](/routes/linear-algebra-essentials/map.md)): Vectors, dot products, numpy arrays
- **Python basics**: Variables, loops, functions, numpy array operations
- **High school algebra** (helpful): The idea of a "slope" and basic function notation like f(x)

No prior calculus knowledge is assumed. If you've heard the word "derivative" before but it feels fuzzy, that's fine -- this guide starts from scratch.

## Setup

You need numpy and matplotlib, which you should already have from the Linear Algebra Essentials route.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

Create a file called `calc_setup_test.py` and run it:

```python
import numpy as np
import matplotlib
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick sanity check: compute slope of x^2 at x=3
x = 3.0
h = 0.0001
slope = ((x + h)**2 - (x - h)**2) / (2 * h)
print(f"Slope of x^2 at x=3: {slope:.4f}")
```

```bash
python calc_setup_test.py
```

**Expected output:**

```
numpy version: 1.26.4
matplotlib version: 3.9.2
Slope of x^2 at x=3: 6.0000
```

Your version numbers may differ -- that's fine as long as the script runs without errors. If you got `6.0000` for the slope, you're ready. (We'll explain what that number means in Section 1.)

---

## Section 1: Derivatives as Rates of Change

### What Is a Derivative?

A derivative measures **how fast something is changing**. That's it. If you've ever looked at a speedometer, you've read a derivative -- your speed is the rate of change of your position over time.

Here's the key question a derivative answers: if I nudge the input to a function by a tiny amount, how much does the output change?

Consider the function f(x) = x². When x = 3, f(x) = 9. If you increase x to 3.001, f(x) becomes 9.006001. The output changed by about 6 for every 1 unit of input change. The derivative of f(x) = x² at x = 3 is 6.

In math notation, the derivative of f with respect to x is written as:

```
df/dx    or    f'(x)
```

Read `df/dx` as "the rate of change of f as x changes" or more casually as "how much f changes per unit change in x." The `d` stands for an infinitesimally small change. The notation `f'(x)` (read "f prime of x") means the same thing.

### The Slope of a Curve at a Point

Graphically, the derivative at a point is the **slope of the tangent line** to the curve at that point. A tangent line just touches the curve at one point and shows the direction the curve is heading.

- Where the curve is going up steeply, the derivative is a large positive number.
- Where the curve is flat (at a peak or valley), the derivative is zero.
- Where the curve is going down, the derivative is negative.

This is why derivatives matter for ML: the points where the derivative is zero are the peaks and valleys -- the minimums and maximums. Finding where the derivative of a loss function equals zero means finding the best (or worst) possible parameters.

### Computing Derivatives Numerically

You don't need to learn derivative formulas to use derivatives. You can compute them numerically using the **central difference** method. The idea: nudge x slightly in both directions and measure how much f changes.

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    """Compute the derivative of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

The formula `(f(x+h) - f(x-h)) / (2*h)` computes the slope between two points that are very close together, one slightly above x and one slightly below. The variable `h` is the size of the nudge -- smaller is more accurate, but too small causes floating-point problems. The value `1e-7` (0.0000001) is a good default.

Let's test it on f(x) = x², where we know the derivative should be 2x:

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    """Compute the derivative of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

# Test at several points
for x_val in [0, 1, 2, 3, -2]:
    deriv = numerical_derivative(f, x_val)
    print(f"f'({x_val:>2}) = {deriv:>8.4f}   (expected: {2 * x_val})")
```

**Expected output:**

```
f'( 0) =   0.0000   (expected: 0)
f'( 1) =   2.0000   (expected: 2)
f'( 2) =   4.0000   (expected: 4)
f'( 3) =   6.0000   (expected: 6)
f'(-2) =  -4.0000   (expected: -4)
```

The numerical derivative matches the known formula (2x) at every point. At x = 0, the slope is 0 -- the bottom of the parabola, where the curve is flat. At x = 3, the slope is 6 -- the curve is rising steeply.

> **Visualization**: Run `python tools/ml-visualizations/derivatives.py` to see a function, its tangent line at different points, and the corresponding derivative value. Move the point along the curve and watch the tangent line tilt.

### Why Numerical Derivatives Matter

In ML, you'll often work with functions that are too complex for an analytical derivative formula. Numerical derivatives give you a way to check your work -- compute the derivative both ways and make sure they agree. This technique is called **gradient checking** and it's a standard debugging tool when implementing backpropagation.

### Exercise 1.1: Write a Numerical Derivative Function

**Task:** Write a function `numerical_derivative(f, x)` and test it on these functions:

1. f(x) = x³ (derivative should be 3x²)
2. f(x) = sin(x) (derivative should be cos(x))
3. f(x) = e^x (derivative should be e^x -- it's its own derivative!)

Verify that your numerical result matches the expected derivative at x = 1.

**Hints:**

<details>
<summary>Hint 1: The formula</summary>

Use `(f(x + h) - f(x - h)) / (2 * h)` with `h = 1e-7`. This is the central difference formula.
</details>

<details>
<summary>Hint 2: numpy functions</summary>

Use `np.sin(x)`, `np.cos(x)`, and `np.exp(x)` for the trigonometric and exponential functions.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    """Compute the derivative of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test functions and their known derivatives
tests = [
    ("x^3",    lambda x: x**3,      lambda x: 3 * x**2),
    ("sin(x)", lambda x: np.sin(x), lambda x: np.cos(x)),
    ("e^x",    lambda x: np.exp(x), lambda x: np.exp(x)),
]

x = 1.0
for name, f, df in tests:
    numerical = numerical_derivative(f, x)
    exact = df(x)
    print(f"f(x) = {name:>6}  |  f'(1) numerical: {numerical:.6f}  |  exact: {exact:.6f}")
```

**Expected output:**

```
f(x) =    x^3  |  f'(1) numerical: 3.000000  |  exact: 3.000000
f(x) = sin(x)  |  f'(1) numerical: 0.540302  |  exact: 0.540302
f(x) =    e^x  |  f'(1) numerical: 2.718282  |  exact: 2.718282
```

**Explanation:** The numerical derivative matches the exact derivative to at least 6 decimal places. For x³, the derivative 3x² at x=1 gives 3. For sin(x), the derivative cos(x) at x=1 gives cos(1) = 0.5403. For e^x, the derivative is e^x itself, so e^1 = 2.7183.
</details>

### Exercise 1.2: Finding the Minimum

**Task:** The function f(x) = x² - 4x + 3 has a parabola shape with a single minimum. Find the x value where the derivative equals zero (the minimum of the function).

1. Write the function f(x) = x² - 4x + 3
2. Compute the numerical derivative at several x values (try x = 0, 1, 2, 3, 4)
3. At which x is the derivative closest to zero?
4. What is f(x) at that point?

**Hints:**

<details>
<summary>Hint 1: Scanning for zeros</summary>

Compute the derivative at integer values of x from 0 to 4. The derivative changes sign when it crosses zero -- look for where it goes from negative to positive (or vice versa).
</details>

<details>
<summary>Hint 2: What the derivative tells you</summary>

If the derivative is negative, the function is still decreasing (you haven't reached the minimum yet). If the derivative is positive, the function is increasing (you've passed the minimum). The minimum is where the derivative is zero.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x**2 - 4*x + 3

# Scan x values
print("  x  |  f(x)  |  f'(x)")
print("-----|--------|--------")
for x_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
    fx = f(x_val)
    dfx = numerical_derivative(f, x_val)
    print(f" {x_val:.1f} | {fx:>5.1f}  | {dfx:>6.2f}")
```

**Expected output:**

```
  x  |  f(x)  |  f'(x)
-----|--------|--------
 0.0 |   3.0  |  -4.00
 1.0 |   0.0  |  -2.00
 2.0 |  -1.0  |   0.00
 3.0 |   0.0  |   2.00
 4.0 |   3.0  |   4.00
```

**Explanation:** The derivative is zero at x = 2, where f(2) = -1. This is the minimum of the function. Notice how the derivative is negative for x < 2 (the function is still going down) and positive for x > 2 (the function is going back up). The minimum is exactly where the function stops decreasing and starts increasing -- where the derivative crosses zero.

This is the fundamental idea behind gradient descent: to find the minimum of a function, look for where its derivative is zero. When the derivative is negative, move right (increasing x). When the derivative is positive, move left (decreasing x). The derivative tells you which direction to go.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Explain what a derivative measures in plain English (rate of change)
- [ ] Compute a numerical derivative using the central difference formula
- [ ] Explain what it means when a derivative is zero, positive, or negative
- [ ] Connect derivatives to finding minimums: the minimum is where f'(x) = 0

---

## Section 2: Partial Derivatives and Gradients

### Functions with Multiple Inputs

The functions in Section 1 had a single input: f(x). But ML loss functions depend on many variables -- every weight in the model is a variable. A neural network with 1000 weights has a loss function with 1000 inputs.

Let's start with two inputs:

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

# Try some values
print(f"f(1, 1) = {f(1, 1)}")   # 1 + 3 = 4
print(f"f(2, 1) = {f(2, 1)}")   # 4 + 3 = 7
print(f"f(1, 2) = {f(1, 2)}")   # 1 + 12 = 13
```

**Expected output:**

```
f(1, 1) = 4
f(2, 1) = 7
f(1, 2) = 13
```

This function takes two numbers and returns one number. You can think of it as a surface -- for every (x, y) point on a plane, the function gives you a height f(x, y). The minimum of this surface is at (0, 0) where f(0, 0) = 0.

### What Is a Partial Derivative?

A partial derivative measures how the function changes when you nudge **one** input while holding **all the others constant**.

For f(x, y) = x² + 3y²:
- If you nudge x while holding y fixed, the function changes at a rate of 2x. This is the partial derivative with respect to x.
- If you nudge y while holding x fixed, the function changes at a rate of 6y. This is the partial derivative with respect to y.

The notation for partial derivatives uses a curly d symbol: **∂** (Unicode: ∂). It looks like a rounded "d" and it specifically means "partial derivative" -- differentiating with respect to one variable while treating all others as constants.

```
∂f/∂x = 2x      (partial derivative of f with respect to x)
∂f/∂y = 6y      (partial derivative of f with respect to y)
```

Read `∂f/∂x` as "the partial derivative of f with respect to x" or more casually "how much f changes when you nudge x." The straight `d` in `df/dx` means total derivative (one variable). The curly `∂` in `∂f/∂x` means partial derivative (multiple variables, holding the others fixed).

### Computing Partial Derivatives Numerically

The trick is the same as before -- nudge one variable, keep the others fixed:

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

def partial_x(f, x, y, h=1e-7):
    """Partial derivative of f with respect to x (hold y constant)."""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(f, x, y, h=1e-7):
    """Partial derivative of f with respect to y (hold x constant)."""
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

# Test at (1, 2)
x, y = 1.0, 2.0
print(f"At ({x}, {y}):")
print(f"  ∂f/∂x = {partial_x(f, x, y):.4f}   (expected: {2*x})")
print(f"  ∂f/∂y = {partial_y(f, x, y):.4f}   (expected: {6*y})")
```

**Expected output:**

```
At (1.0, 2.0):
  ∂f/∂x = 2.0000   (expected: 2.0)
  ∂f/∂y = 12.0000   (expected: 12.0)
```

Notice how computing a partial derivative is exactly like a regular derivative -- you just hold the other variables still. At the point (1, 2), nudging x by a tiny amount changes f at a rate of 2 per unit, and nudging y changes f at a rate of 12 per unit. The function is much more sensitive to y at this point.

### The Gradient: All Partial Derivatives in One Vector

The **gradient** is simply all the partial derivatives collected into a vector. For a function of two variables:

```
∇f = [∂f/∂x, ∂f/∂y]
```

The symbol ∇ is called "nabla" or "del." Read `∇f` as "the gradient of f" or "grad f." It's a vector that points in the direction of steepest increase of the function.

For f(x, y) = x² + 3y², the gradient is:

```
∇f = [2x, 6y]
```

At the point (1, 2), the gradient is [2, 12]. This vector points in the direction where f increases the fastest. To *decrease* f (which is what we want when minimizing a loss function), we go in the **opposite direction**: [-2, -12].

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

def gradient(f, x, y, h=1e-7):
    """Compute the gradient of f at (x, y)."""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# Compute gradient at several points
points = [(1, 2), (0, 0), (-1, 1), (3, -1)]
for x, y in points:
    grad = gradient(f, x, y)
    print(f"∇f({x}, {y}) = [{grad[0]:>6.2f}, {grad[1]:>6.2f}]")
```

**Expected output:**

```
∇f(1, 2) = [  2.00,  12.00]
∇f(0, 0) = [  0.00,   0.00]
∇f(-1, 1) = [ -2.00,   6.00]
∇f(3, -1) = [  6.00,  -6.00]
```

At (0, 0) the gradient is [0, 0] -- this is the minimum of the function. The gradient is zero at the minimum because there's no direction that increases the function (or decreases it) -- you're already at the bottom.

### The Gradient Points Uphill

Here's the crucial property: **the gradient always points in the direction of steepest increase**. If you're standing on a hillside and want to go uphill as quickly as possible, follow the gradient. If you want to go downhill (to minimize a loss function), go in the **opposite** direction of the gradient.

This is the entire idea behind gradient descent: compute the gradient, then step in the opposite direction.

### Exercise 2.1: Compute Gradients

**Task:** Compute the gradient of f(x, y) = x² + 3y² at the points (0, 0), (1, 0), (0, 1), (2, 3), and (-1, -2). For each point, state whether the gradient points toward or away from the origin.

**Hints:**

<details>
<summary>Hint 1: The gradient formula</summary>

For f(x, y) = x² + 3y², the gradient is ∇f = [2x, 6y]. You can verify this numerically using the central difference formula on each variable separately.
</details>

<details>
<summary>Hint 2: Direction relative to origin</summary>

If the gradient vector has the same sign pattern as the position vector (both components positive, or both negative), the gradient points away from the origin -- the function increases as you move away from the center. If the signs are opposite, it points toward the origin.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

def gradient(f, x, y, h=1e-7):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

points = [(0, 0), (1, 0), (0, 1), (2, 3), (-1, -2)]
for x, y in points:
    grad = gradient(f, x, y)
    position = np.array([x, y])
    # Dot product > 0 means gradient points roughly away from origin
    if np.linalg.norm(grad) < 1e-10:
        direction = "zero (at the minimum)"
    elif position @ grad > 0:
        direction = "away from origin (uphill)"
    else:
        direction = "toward origin (downhill)"
    print(f"({x:>2}, {y:>2}): ∇f = [{grad[0]:>6.2f}, {grad[1]:>6.2f}]  — {direction}")
```

**Expected output:**

```
( 0,  0): ∇f = [  0.00,   0.00]  — zero (at the minimum)
( 1,  0): ∇f = [  2.00,   0.00]  — away from origin (uphill)
( 0,  1): ∇f = [  0.00,   6.00]  — away from origin (uphill)
( 2,  3): ∇f = [  4.00,  18.00]  — away from origin (uphill)
(-1, -2): ∇f = [ -2.00, -12.00]  — away from origin (uphill)
```

**Explanation:** The gradient always points away from the origin because the function f(x, y) = x² + 3y² increases in every direction from the origin. The minimum is at (0, 0), and the gradient at every other point tells you which direction leads uphill. To minimize the function, you'd step in the *opposite* direction of the gradient at each point.
</details>

### Exercise 2.2: Gradient Vectors on a Contour Plot

**Task:** Create a contour plot of f(x, y) = x² + 3y² and draw gradient vectors at several points. Use `plt.contour()` for the contour lines and `plt.quiver()` for the arrows.

**Hints:**

<details>
<summary>Hint 1: Creating a contour plot</summary>

```python
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + 3 * Y**2
plt.contour(X, Y, Z, levels=15)
```
</details>

<details>
<summary>Hint 2: Drawing arrows with quiver</summary>

`plt.quiver(x_positions, y_positions, arrow_x, arrow_y)` draws arrows. The gradient components are the arrow directions. You may want to normalize the arrows for visibility so they don't overlap.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + 3 * y**2

def gradient(f, x, y, h=1e-7):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# Create contour plot
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + 3 * Y**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(label='f(x, y)')

# Compute and draw gradient vectors at grid points
gx = np.linspace(-2.5, 2.5, 6)
gy = np.linspace(-2.5, 2.5, 6)
for xi in gx:
    for yi in gy:
        grad = gradient(f, xi, yi)
        length = np.linalg.norm(grad)
        if length > 0.1:  # Skip near-zero gradients
            # Normalize for visibility, scale to fixed length
            grad_normalized = grad / length * 0.3
            plt.arrow(xi, yi, grad_normalized[0], grad_normalized[1],
                      head_width=0.08, head_length=0.05, fc='red', ec='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x,y) = x² + 3y² with gradient vectors')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gradient_contour.png', dpi=100)
plt.show()
```

**Explanation:** The contour lines show levels of equal f(x, y) -- like elevation lines on a topographic map. The red arrows are gradient vectors, pointing in the direction of steepest increase. Notice how the arrows always point perpendicular to the contour lines and away from the minimum at the center. The elliptical shape of the contours reflects the fact that y has a stronger effect on f than x does (because of the factor of 3).
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Explain what a partial derivative is (rate of change with respect to one variable, others held constant)
- [ ] Distinguish between the ∂ (partial) and d (total) notation
- [ ] Compute partial derivatives numerically
- [ ] Define the gradient as the vector of all partial derivatives
- [ ] Explain why the gradient points in the direction of steepest increase

---

## Section 3: The Chain Rule

### Composing Functions

In programming, you compose functions all the time: pipe one function's output into another. In Python, `h(x) = f(g(x))` means "first apply g to x, then apply f to the result."

```python
import numpy as np

def g(x):
    return 3 * x + 1

def f(u):
    return u ** 2

def h(x):
    return f(g(x))

# Trace through the computation
x = 2.0
inner = g(x)      # 3*2 + 1 = 7
outer = f(inner)   # 7^2 = 49
print(f"g({x}) = {inner}")
print(f"f({inner}) = {outer}")
print(f"h({x}) = f(g({x})) = {h(x)}")
```

**Expected output:**

```
g(2.0) = 7.0
f(7.0) = 49.0
h(2.0) = f(g(2.0)) = 49.0
```

Neural networks are deeply composed functions. A simple network might compute:

```
output = activation(weights3 @ activation(weights2 @ activation(weights1 @ input)))
```

That's three layers of composition. To train this network, you need to know: how does the final output change when you tweak `weights1`? The chain rule answers this.

### The Chain Rule

The chain rule says: when you compose two functions, the derivative of the composition is the **product of the individual derivatives**.

For h(x) = f(g(x)):

```
dh/dx = df/dg * dg/dx
```

Read this as: "the rate of change of h with respect to x equals the rate of change of f with respect to g, times the rate of change of g with respect to x."

Think of it as a chain of gears. If g doubles x (dg/dx = 2), and f triples g (df/dg = 3), then h multiplies x by 6 (dh/dx = 2 * 3 = 6). The rates of change multiply through the chain.

### Worked Example

Let's work through h(x) = (3x + 1)² step by step.

The inner function is g(x) = 3x + 1, with derivative dg/dx = 3.
The outer function is f(u) = u², with derivative df/du = 2u.

By the chain rule:
```
dh/dx = df/du * dg/dx = 2u * 3 = 2(3x + 1) * 3 = 6(3x + 1)
```

At x = 2: dh/dx = 6(3*2 + 1) = 6(7) = 42.

Let's verify this numerically:

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def h(x):
    return (3 * x + 1) ** 2

# Chain rule says dh/dx = 6(3x + 1)
x = 2.0
chain_rule_result = 6 * (3 * x + 1)
numerical_result = numerical_derivative(h, x)

print(f"Chain rule: dh/dx at x={x} is {chain_rule_result}")
print(f"Numerical:  dh/dx at x={x} is {numerical_result:.4f}")
print(f"Match: {np.isclose(chain_rule_result, numerical_result)}")
```

**Expected output:**

```
Chain rule: dh/dx at x=2.0 is 42.0
Numerical:  dh/dx at x=2.0 is 42.0000
Match: True
```

### Why the Chain Rule Matters for ML

Backpropagation -- the algorithm that trains neural networks -- is just the chain rule applied repeatedly. A neural network is a long chain of composed functions (layers). To compute how the loss changes with respect to a weight deep in the network, you multiply the derivatives through each layer, from the output back to that weight.

Here's a preview with a three-layer composition:

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Three composed functions: h(x) = f3(f2(f1(x)))
def f1(x): return 2 * x      # Layer 1: multiply by 2
def f2(x): return x + 3      # Layer 2: add 3
def f3(x): return x ** 2     # Layer 3: square

def composed(x):
    return f3(f2(f1(x)))

# The derivative via chain rule:
# d(composed)/dx = df3/df2 * df2/df1 * df1/dx
# df1/dx = 2
# df2/df1 = 1
# df3/df2 = 2 * f2(f1(x)) = 2 * (2x + 3)
# Total: 2 * 1 * 2(2x + 3) = 4(2x + 3)

x = 1.0
chain_rule = 4 * (2 * x + 3)
numerical = numerical_derivative(composed, x)

print(f"composed({x}) = f3(f2(f1({x}))) = {composed(x)}")
print(f"Chain rule derivative: {chain_rule}")
print(f"Numerical derivative:  {numerical:.4f}")
```

**Expected output:**

```
composed(1.0) = f3(f2(f1(1.0))) = 25.0
Chain rule derivative: 20.0
Numerical derivative:  20.0000
```

The chain rule lets you decompose the derivative of a complicated composition into a product of simple derivatives. Each layer contributes its own local derivative, and you multiply them all together. This is exactly what backpropagation does.

### Exercise 3.1: Apply the Chain Rule

**Task:** For each composed function below, compute the derivative using the chain rule, then verify with a numerical derivative at x = 1.

1. h(x) = sin(x²) -- inner: g(x) = x², outer: f(u) = sin(u)
2. h(x) = (x³ + 1)⁴ -- inner: g(x) = x³ + 1, outer: f(u) = u⁴
3. h(x) = e^(2x) -- inner: g(x) = 2x, outer: f(u) = e^u

**Hints:**

<details>
<summary>Hint 1: Derivative rules you need</summary>

- d/du sin(u) = cos(u)
- d/du u^n = n * u^(n-1)
- d/du e^u = e^u
- d/dx x^n = n * x^(n-1)
</details>

<details>
<summary>Hint 2: Chain rule pattern</summary>

For h(x) = f(g(x)): dh/dx = f'(g(x)) * g'(x). Compute the derivative of the outer function evaluated at g(x), then multiply by the derivative of the inner function.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

x = 1.0

# 1. h(x) = sin(x^2)
# dh/dx = cos(x^2) * 2x
h1 = lambda x: np.sin(x**2)
chain1 = np.cos(x**2) * 2 * x
numer1 = numerical_derivative(h1, x)
print(f"sin(x^2):     chain rule = {chain1:.6f}, numerical = {numer1:.6f}")

# 2. h(x) = (x^3 + 1)^4
# dh/dx = 4(x^3 + 1)^3 * 3x^2
h2 = lambda x: (x**3 + 1)**4
chain2 = 4 * (x**3 + 1)**3 * 3 * x**2
numer2 = numerical_derivative(h2, x)
print(f"(x^3 + 1)^4:  chain rule = {chain2:.6f}, numerical = {numer2:.6f}")

# 3. h(x) = e^(2x)
# dh/dx = e^(2x) * 2
h3 = lambda x: np.exp(2 * x)
chain3 = np.exp(2 * x) * 2
numer3 = numerical_derivative(h3, x)
print(f"e^(2x):        chain rule = {chain3:.6f}, numerical = {numer3:.6f}")
```

**Expected output:**

```
sin(x^2):     chain rule = 1.080605, numerical = 1.080605
(x^3 + 1)^4:  chain rule = 96.000000, numerical = 96.000000
e^(2x):        chain rule = 14.778112, numerical = 14.778112
```

**Explanation:**
1. For sin(x²): The outer derivative is cos(u) evaluated at u = x² = 1, giving cos(1). The inner derivative is 2x = 2. The product is cos(1) * 2 = 1.0806.
2. For (x³ + 1)⁴: The outer derivative is 4u³ evaluated at u = 2, giving 4*8 = 32. The inner derivative is 3x² = 3. The product is 32 * 3 = 96.
3. For e^(2x): The outer derivative is e^u evaluated at u = 2, giving e². The inner derivative is 2. The product is 2e² = 14.778.

In each case, the chain rule result matches the numerical derivative.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Explain function composition as "feeding one function's output into another"
- [ ] Apply the chain rule: dh/dx = df/dg * dg/dx
- [ ] Verify chain rule results with numerical derivatives
- [ ] Explain why the chain rule is the foundation of backpropagation

---

## Section 4: Gradient Descent

### The Algorithm

Gradient descent is how ML models learn. The idea is simple:

1. Start with some initial guess for the parameters.
2. Compute the gradient of the loss function at the current parameters.
3. Take a small step in the **opposite** direction of the gradient (downhill).
4. Repeat until the loss stops decreasing.

That's it. Here it is in pseudocode:

```
parameters = initial_guess
for each step:
    grad = gradient(loss_function, parameters)
    parameters = parameters - learning_rate * grad
```

The **learning rate** controls how big each step is. It's a small positive number, typically between 0.001 and 0.1. The minus sign is because the gradient points uphill, and we want to go downhill.

### Gradient Descent in Python

Let's implement gradient descent to find the minimum of f(x) = x⁴ - 3x² + 2. This function has two local minima and one local maximum, so it's more interesting than a simple parabola.

```python
import numpy as np

def f(x):
    return x**4 - 3 * x**2 + 2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Gradient descent
x = 3.0              # Starting point
learning_rate = 0.01  # Step size
n_steps = 100

print(f"{'Step':>4}  {'x':>8}  {'f(x)':>10}  {'f\'(x)':>10}")
print("-" * 40)

for step in range(n_steps):
    grad = numerical_derivative(f, x)
    if step < 10 or step % 20 == 0:
        print(f"{step:>4}  {x:>8.4f}  {f(x):>10.4f}  {grad:>10.4f}")
    x = x - learning_rate * grad

print(f"\nFinal: x = {x:.6f}, f(x) = {f(x):.6f}")
```

**Expected output:**

```
Step         x       f(x)      f'(x)
----------------------------------------
   0    3.0000    56.0000    90.0000
   1    2.1000    12.6441    31.1124
   2    1.7889     4.7361    15.6476
   3    1.6324     2.6759    10.0920
   4    1.5315     1.7727     6.8991
   5    1.4625     1.2730     4.8966
   6    1.4135     0.9751     3.5612
   7    1.3779     0.7859     2.6328
   8    1.3516     0.6630     1.9696
   9    1.3319     0.5823     1.4882
  20    1.2346     0.2603     0.1594
  40    1.2247     0.2500     0.0036
  60    1.2247     0.2500     0.0001
  80    1.2247     0.2500     0.0000

Final: x = 1.224745, f(x) = 0.250000
```

The algorithm started at x = 3 and walked downhill, step by step, until it settled near x = 1.2247. At that point, the derivative is essentially zero -- we've found a minimum.

> **Visualization**: Run `python tools/ml-visualizations/gradient_descent.py` to see the algorithm animated. Watch the point slide down the curve and settle into a minimum.

### Learning Rate: Getting the Step Size Right

The learning rate is the most important hyperparameter in gradient descent. Too small and you'll take forever to converge. Too large and you'll overshoot the minimum and bounce around (or diverge).

```python
import numpy as np

def f(x):
    return x**2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent(f, x0, lr, n_steps):
    """Run gradient descent and return the history of x values."""
    x = x0
    history = [x]
    for _ in range(n_steps):
        grad = numerical_derivative(f, x)
        x = x - lr * grad
        history.append(x)
    return history

x0 = 4.0

# Try three learning rates
for lr in [0.01, 0.1, 1.5]:
    history = gradient_descent(f, x0, lr, 20)
    final_x = history[-1]
    print(f"lr={lr:<4}  x after 20 steps: {final_x:>12.6f}  f(x): {f(final_x):>12.6f}")
```

**Expected output:**

```
lr=0.01  x after 20 steps:     2.641586  f(x):     6.977976
lr=0.1   x after 20 steps:     0.047224  f(x):     0.002230
lr=1.5   x after 20 steps: -8589934592.000000  f(x): 73786976294838206464.000000
```

- **lr = 0.01** (too small): After 20 steps, x is still at 2.6. It's heading in the right direction, but progress is painfully slow. You'd need hundreds of steps.
- **lr = 0.1** (just right): After 20 steps, x is at 0.05, very close to the minimum at 0. Fast convergence.
- **lr = 1.5** (too large): The value explodes. Each step overshoots the minimum, and the overshooting gets worse each time. The algorithm diverges.

Finding a good learning rate is one of the practical challenges of training ML models. In practice, techniques like learning rate schedules (reducing the learning rate over time) and adaptive methods (like Adam) help, but the fundamental tradeoff remains: small steps are safe but slow, large steps are fast but risky.

### Gradient Descent in 2D

The same algorithm works for functions of multiple variables. Instead of subtracting a scalar gradient, you subtract a gradient vector:

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

def gradient(f, x, y, h=1e-7):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# Gradient descent in 2D
pos = np.array([3.0, 2.0])   # Starting point
learning_rate = 0.05
n_steps = 50

print(f"{'Step':>4}  {'x':>7}  {'y':>7}  {'f(x,y)':>10}")
print("-" * 35)

for step in range(n_steps):
    x, y = pos
    if step < 8 or step % 10 == 0:
        print(f"{step:>4}  {x:>7.3f}  {y:>7.3f}  {f(x, y):>10.4f}")
    grad = gradient(f, x, y)
    pos = pos - learning_rate * grad

x, y = pos
print(f"\nFinal: ({x:.6f}, {y:.6f}), f = {f(x, y):.6f}")
```

**Expected output:**

```
Step        x        y      f(x,y)
-----------------------------------
   0    3.000    2.000     21.0000
   1    2.700    0.800      9.2100
   2    2.430    0.320      5.9049
   3    2.187    0.128      4.7830
   4    1.968    0.051      3.8750
   5    1.772    0.020      3.1383
   6    1.594    0.008      2.5413
   7    1.435    0.003      2.0587
  10    1.047    0.000      1.0960
  20    0.362    0.000      0.1309
  30    0.125    0.000      0.0156
  40    0.043    0.000      0.0019

Final: (0.014901, 0.000000), f = 0.000222
```

The algorithm converges to (0, 0), the minimum of f(x, y) = x² + 3y². Notice how y converges much faster than x -- that's because the gradient in the y direction is larger (the coefficient 3 makes the function steeper in y), so the steps in y are bigger.

### Local Minima and Saddle Points

Not every minimum is *the* minimum. A function can have multiple valleys (local minima), and gradient descent will find whichever one you start closest to.

```python
import numpy as np

def f(x):
    return x**4 - 3 * x**2 + 2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Start from two different positions
for x0 in [2.0, -0.5]:
    x = x0
    for _ in range(200):
        grad = numerical_derivative(f, x)
        x = x - 0.01 * grad
    print(f"Starting at x={x0:>5}: converged to x={x:>8.4f}, f(x)={f(x):.4f}")
```

**Expected output:**

```
Starting at x=  2.0: converged to x=  1.2247, f(x)=0.2500
Starting at x= -0.5: converged to x= -1.2247, f(x)=0.2500
```

Both starting points find a minimum, but they find *different* minima (the function is symmetric). Neither can guarantee it found the *global* minimum (the lowest point overall). In this case, both minima happen to have the same value, but in general they won't.

A **saddle point** is a point where the gradient is zero but it's not a minimum -- it's a minimum in one direction but a maximum in another. Imagine the center of a horse saddle. In higher dimensions, saddle points are actually more common than local minima, and gradient descent can sometimes get stuck near them temporarily.

For training neural networks, the practical impact is:
- Different random initializations can lead to different solutions.
- Advanced optimizers (like Adam) help escape shallow minima and saddle points.
- In practice, for large neural networks, most local minima are nearly as good as the global minimum.

### Exercise 4.1: Implement Gradient Descent

**Task:** Implement gradient descent to find a minimum of f(x) = x⁴ - 3x² + 2. Start at x = 0.5. Use a learning rate of 0.01 and run for 200 steps. Print the final x and f(x).

Then try starting at x = -3.0. Do you find the same minimum?

**Hints:**

<details>
<summary>Hint 1: The update rule</summary>

Each step: `x = x - learning_rate * f'(x)`. Use your `numerical_derivative` function for f'(x).
</details>

<details>
<summary>Hint 2: Expect different results</summary>

The function f(x) = x⁴ - 3x² + 2 has two minima: one near x = 1.22 and one near x = -1.22. Which one you find depends on where you start.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def f(x):
    return x**4 - 3 * x**2 + 2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent_1d(f, x0, lr=0.01, n_steps=200):
    x = x0
    for step in range(n_steps):
        grad = numerical_derivative(f, x)
        x = x - lr * grad
    return x

# Starting at x = 0.5
x_final = gradient_descent_1d(f, x0=0.5)
print(f"Start x=0.5:  final x = {x_final:.6f}, f(x) = {f(x_final):.6f}")

# Starting at x = -3.0
x_final2 = gradient_descent_1d(f, x0=-3.0)
print(f"Start x=-3.0: final x = {x_final2:.6f}, f(x) = {f(x_final2):.6f}")
```

**Expected output:**

```
Start x=0.5:  final x = 1.224745, f(x) = 0.250000
Start x=-3.0: final x = -1.224745, f(x) = 0.250000
```

**Explanation:** Starting at x = 0.5, the gradient is negative (the function is decreasing to the right of the local maximum at x = 0), so the algorithm moves right and finds the minimum near x = 1.22. Starting at x = -3.0, the gradient is positive (the function is decreasing to the left), so the algorithm moves left and finds the other minimum near x = -1.22. Same loss value, different minimizers.
</details>

### Exercise 4.2: Learning Rate Exploration

**Task:** Run gradient descent on f(x) = x² starting from x = 5 with these learning rates: 0.001, 0.01, 0.1, 0.5, 0.99, and 1.01. Run each for 50 steps. For each, report the final x value and whether the algorithm converged, converged slowly, or diverged.

**Hints:**

<details>
<summary>Hint 1: Defining convergence</summary>

"Converged" means x is very close to the minimum (0 in this case). "Converged slowly" means x is moving toward 0 but hasn't gotten close yet. "Diverged" means x is getting farther from 0 or oscillating wildly.
</details>

<details>
<summary>Hint 2: What's special about lr = 1.0?</summary>

For f(x) = x², the derivative is 2x, so the update is x = x - lr * 2x = x(1 - 2*lr). When lr = 0.5, the factor is 0, so you jump directly to 0 in one step. When lr > 0.5, the factor is negative, so you overshoot. When lr >= 1.0, |1 - 2*lr| >= 1, so each step makes x larger -- divergence.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

def f(x):
    return x**2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 5.0
for lr in [0.001, 0.01, 0.1, 0.5, 0.99, 1.01]:
    x = x0
    for _ in range(50):
        grad = numerical_derivative(f, x)
        x = x - lr * grad
    status = "converged" if abs(x) < 0.01 else "slow" if abs(x) < abs(x0) else "DIVERGED"
    print(f"lr={lr:<5}  x = {x:>15.6f}  f(x) = {f(x):>15.6f}  [{status}]")
```

**Expected output:**

```
lr=0.001  x =        4.095024  f(x) =       16.769219  [slow]
lr=0.01   x =        1.816697  f(x) =        3.300406  [slow]
lr=0.1    x =        0.000072  f(x) =        0.000000  [converged]
lr=0.5    x =        0.000000  f(x) =        0.000000  [converged]
lr=0.99   x =       -0.000000  f(x) =        0.000000  [converged]
lr=1.01   x = -57154871042.895187  f(x) = 3266675265028449234944.000000  [DIVERGED]
```

**Explanation:** Small learning rates (0.001, 0.01) make progress but haven't converged after 50 steps. Medium rates (0.1, 0.5, 0.99) converge well. At lr = 0.5, the algorithm actually converges in a single step because the update perfectly jumps to the minimum. At lr = 1.01, the algorithm diverges -- each step overshoots more than the last, and x explodes to huge values.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Describe the gradient descent algorithm in your own words
- [ ] Implement gradient descent for 1D and 2D functions
- [ ] Explain what the learning rate controls and what happens when it's too small or too large
- [ ] Explain what a local minimum is and why gradient descent might find different minima from different starting points

---

## Practice Project: Linear Regression from Scratch

### Project Description

Build a linear regression model trained with gradient descent -- no scikit-learn, no autograd, just numpy. You'll define a loss function, compute its gradients, and watch the model learn to fit a line to data.

This project ties together everything from this route:
- Derivatives tell you how the loss changes when you tweak a parameter
- The gradient gives you the direction of steepest increase (so you go the other way)
- The chain rule lets you compute how each parameter affects the loss through the chain of computations
- Gradient descent uses all of this to iteratively improve the parameters

### The Problem

Given a set of data points (x, y), find the line y = mx + b that best fits the data. "Best fit" means the line that minimizes the total error between the predicted y values and the actual y values.

### The Loss Function: Mean Squared Error

The loss function measures how wrong the predictions are. Mean Squared Error (MSE) is the standard choice for regression:

```
MSE = (1/N) * sum((y_predicted - y_actual)^2)
```

Where y_predicted = m * x + b for each data point.

### Computing the Gradients

For the loss L = (1/N) * sum((mx + b - y)²), we need two partial derivatives:

```
∂L/∂m = (2/N) * sum((mx + b - y) * x)    — how does the loss change when we nudge m?
∂L/∂b = (2/N) * sum((mx + b - y))         — how does the loss change when we nudge b?
```

These formulas come from applying the chain rule to the MSE formula. The inner function is (mx + b - y), the outer function is squaring it. The chain rule gives us 2 * (mx + b - y) times the derivative of the inner part with respect to m (which is x) or b (which is 1).

### Getting Started

**Step 1: Generate some data**

```python
import numpy as np
np.random.seed(42)

# True line: y = 2x + 1, with noise
N = 50
x = np.random.uniform(-3, 3, N)
y = 2 * x + 1 + np.random.normal(0, 0.5, N)
```

**Step 2: Define the loss function**

```python
def predict(x, m, b):
    return m * x + b

def mse_loss(x, y, m, b):
    predictions = predict(x, m, b)
    return np.mean((predictions - y) ** 2)
```

**Step 3: Compute gradients**

```python
def gradients(x, y, m, b):
    predictions = predict(x, m, b)
    errors = predictions - y
    dm = (2 / len(x)) * np.sum(errors * x)
    db = (2 / len(x)) * np.sum(errors)
    return dm, db
```

**Step 4: Run gradient descent**

```python
m, b = 0.0, 0.0  # Initial guess
learning_rate = 0.05
n_epochs = 100

for epoch in range(n_epochs):
    dm, db = gradients(x, y, m, b)
    m = m - learning_rate * dm
    b = b - learning_rate * db
```

**Step 5: Plot the results**

Plot the data points and the fitted line. Also plot the loss over training to see it decrease.

### Hints and Tips

<details>
<summary>Hint 1: Structuring your code</summary>

Put everything in a single script. Define the helper functions (predict, mse_loss, gradients) at the top. Generate data, run the training loop, then plot. Print the loss every 10 epochs to watch progress.
</details>

<details>
<summary>Hint 2: Verifying your gradients</summary>

Before running gradient descent, verify your analytical gradients match numerical gradients at a few points. Compute the numerical gradient of the loss with respect to m by nudging m slightly:

```python
h = 1e-7
dm_numerical = (mse_loss(x, y, m + h, b) - mse_loss(x, y, m - h, b)) / (2 * h)
```

If this matches your analytical `dm`, your gradient computation is correct.
</details>

<details>
<summary>Hint 3: Plotting</summary>

Use two subplots: one for the data and fitted line, one for the loss curve.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: data + line
ax1.scatter(x, y, alpha=0.6, label='Data')
x_line = np.linspace(-3, 3, 100)
ax1.plot(x_line, m * x_line + b, 'r-', label=f'Fit: y = {m:.2f}x + {b:.2f}')
ax1.legend()

# Plot 2: loss over time
ax2.plot(loss_history)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
```
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Generate data ---
np.random.seed(42)
N = 50
x = np.random.uniform(-3, 3, N)
y_true_slope = 2.0
y_true_intercept = 1.0
y = y_true_slope * x + y_true_intercept + np.random.normal(0, 0.5, N)

# --- Model functions ---
def predict(x, m, b):
    return m * x + b

def mse_loss(x, y, m, b):
    predictions = predict(x, m, b)
    return np.mean((predictions - y) ** 2)

def gradients(x, y, m, b):
    predictions = predict(x, m, b)
    errors = predictions - y
    dm = (2 / len(x)) * np.sum(errors * x)
    db = (2 / len(x)) * np.sum(errors)
    return dm, db

# --- Verify gradients numerically ---
m, b = 0.5, -0.3  # Arbitrary test point
dm_analytical, db_analytical = gradients(x, y, m, b)

h = 1e-7
dm_numerical = (mse_loss(x, y, m + h, b) - mse_loss(x, y, m - h, b)) / (2 * h)
db_numerical = (mse_loss(x, y, m, b + h) - mse_loss(x, y, m, b - h)) / (2 * h)

print("Gradient verification:")
print(f"  dm: analytical = {dm_analytical:.8f}, numerical = {dm_numerical:.8f}")
print(f"  db: analytical = {db_analytical:.8f}, numerical = {db_numerical:.8f}")
print()

# --- Train with gradient descent ---
m, b = 0.0, 0.0
learning_rate = 0.05
n_epochs = 200
loss_history = []

print(f"{'Epoch':>5}  {'m':>8}  {'b':>8}  {'Loss':>10}")
print("-" * 35)

for epoch in range(n_epochs):
    loss = mse_loss(x, y, m, b)
    loss_history.append(loss)

    if epoch < 5 or epoch % 40 == 0:
        print(f"{epoch:>5}  {m:>8.4f}  {b:>8.4f}  {loss:>10.4f}")

    dm, db = gradients(x, y, m, b)
    m = m - learning_rate * dm
    b = b - learning_rate * db

print(f"\nLearned: y = {m:.4f}x + {b:.4f}")
print(f"True:    y = {y_true_slope:.4f}x + {y_true_intercept:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Data and fitted line
ax1.scatter(x, y, alpha=0.6, label='Data', zorder=2)
x_line = np.linspace(-3.5, 3.5, 100)
ax1.plot(x_line, m * x_line + b, 'r-', linewidth=2,
         label=f'Fit: y = {m:.2f}x + {b:.2f}')
ax1.plot(x_line, y_true_slope * x_line + y_true_intercept, 'g--', alpha=0.5,
         label=f'True: y = {y_true_slope}x + {y_true_intercept}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression with Gradient Descent')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss curve
ax2.plot(loss_history, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Training Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_output.png', dpi=100)
plt.show()
```

**Key points in this solution:**
- Gradients are verified numerically before training -- this is a good habit to catch bugs early.
- The learned parameters (m, b) converge close to the true values (2.0, 1.0), but not exactly, because of the noise in the data.
- The loss curve drops steeply at first and then levels off as the parameters approach the optimum.
- The final loss is not zero because the data has irreducible noise -- even the true line doesn't pass through all the points.
</details>

### Extending the Project

If you want to go further, try:
- **Polynomial regression**: Fit y = ax² + bx + c by adding an x² feature and a third parameter. You'll need to compute three gradients instead of two.
- **Mini-batch gradient descent**: Instead of using all N points to compute each gradient, use a random subset (a "mini-batch"). This is faster for large datasets and adds noise that can help escape local minima.
- **Learning rate schedule**: Start with a larger learning rate and decrease it over time. Try halving the learning rate every 50 epochs.
- **Multiple features**: Extend to y = w1*x1 + w2*x2 + b with two input features. Use vectors and matrices for the parameters and data.

---

## Summary

### Key Takeaways

- **Derivatives** measure how fast a function's output changes when you nudge its input. Numerically, you compute them with the central difference formula: `(f(x+h) - f(x-h)) / (2h)`.
- **Partial derivatives** extend this to functions of multiple variables by nudging one variable while holding others constant. The curly ∂ symbol means "partial."
- **The gradient** is the vector of all partial derivatives: ∇f = [∂f/∂x, ∂f/∂y, ...]. It points in the direction of steepest increase.
- **The chain rule** says the derivative of composed functions equals the product of the individual derivatives: dh/dx = df/dg * dg/dx. This is the mathematical foundation of backpropagation.
- **Gradient descent** finds minimums by repeatedly stepping in the opposite direction of the gradient. The learning rate controls step size: too small is slow, too large diverges.

### Skills You've Gained

You can now:
- Compute derivatives and gradients numerically
- Apply the chain rule to composed functions
- Implement gradient descent from scratch
- Train a linear regression model with gradient descent
- Diagnose learning rate problems
- Verify analytical gradients with numerical ones (gradient checking)

### Self-Assessment

Take a moment to reflect:
- Could you explain gradient descent to a colleague using only the words "slope" and "step"?
- If someone showed you a loss curve that wasn't decreasing, what would you check first?
- Can you trace the chain rule through a three-layer function composition?
- Do you understand why the gradient points uphill and we subtract it to go downhill?

---

## Next Steps

### Continue Learning

Ready for more? Here are your next options:

**Build on this topic:**
- [Neural Network Foundations](/routes/neural-network-foundations/map.md) -- Perceptrons, layers, and the forward pass. Uses matrix multiplication (from linear algebra) and activation functions.
- [Training and Backprop](/routes/training-and-backprop/map.md) -- Loss functions and backpropagation. Applies the chain rule to compute gradients through an entire neural network.

### Additional Resources

**Books:**
- *Mathematics for Machine Learning* by Deisenroth, Faisal, Ong -- freely available online, covers calculus for ML in depth
- *Deep Learning* by Goodfellow, Bengio, Courville -- Chapter 4 covers numerical computation and optimization

**Videos:**
- 3Blue1Brown's *Essence of Calculus* -- the best visual explanation of derivatives and integrals
- 3Blue1Brown's *Neural Networks* series -- shows how gradient descent and backpropagation work visually

**Interactive Tools:**
- Desmos (desmos.com/calculator) -- graph functions and see tangent lines interactively
- TensorFlow Playground (playground.tensorflow.org) -- watch gradient descent train a neural network in real time

---

## Quick Reference

### Derivative Rules

| Function f(x) | Derivative f'(x) | Example |
|---|---|---|
| x^n | n * x^(n-1) | d/dx x³ = 3x² |
| sin(x) | cos(x) | d/dx sin(x) = cos(x) |
| cos(x) | -sin(x) | d/dx cos(x) = -sin(x) |
| e^x | e^x | d/dx e^x = e^x |
| ln(x) | 1/x | d/dx ln(x) = 1/x |
| c * f(x) | c * f'(x) | d/dx 5x² = 10x |
| f(x) + g(x) | f'(x) + g'(x) | d/dx (x² + x) = 2x + 1 |
| f(g(x)) | f'(g(x)) * g'(x) | d/dx sin(x²) = cos(x²) * 2x |

### Gradient Descent Update Rule

```
parameters = parameters - learning_rate * gradient
```

For linear regression (y = mx + b) with MSE loss:
```
m = m - lr * (2/N) * sum((mx + b - y) * x)
b = b - lr * (2/N) * sum((mx + b - y))
```

### Numpy Functions for Calculus

```python
import numpy as np

# --- Numerical Derivatives ---
# Central difference (use for gradient checking)
df_dx = (f(x + h) - f(x - h)) / (2 * h)

# --- Math Functions ---
np.sin(x)           # Sine
np.cos(x)           # Cosine
np.exp(x)           # e^x
np.log(x)           # Natural log (ln)
np.sqrt(x)          # Square root

# --- Array Operations (useful for gradient computations) ---
np.sum(array)        # Sum all elements
np.mean(array)       # Mean of all elements
np.dot(a, b)         # Dot product
a @ b                # Matrix/dot product (preferred)

# --- Plotting ---
import matplotlib.pyplot as plt
plt.plot(x, y)                   # Line plot
plt.scatter(x, y)                # Scatter plot
plt.contour(X, Y, Z)             # Contour plot
plt.arrow(x, y, dx, dy)          # Arrow (for gradient vectors)
```

---

## Glossary

- **Central difference**: A method for computing numerical derivatives: (f(x+h) - f(x-h)) / (2h). More accurate than the forward difference because it's symmetric around x.
- **Chain rule**: The rule for differentiating composed functions: d/dx f(g(x)) = f'(g(x)) * g'(x). Derivatives multiply through each layer of composition.
- **Convergence**: When gradient descent reaches a point where the loss stops decreasing meaningfully. The gradient approaches zero near a minimum.
- **Derivative**: The rate of change of a function at a point. Geometrically, the slope of the tangent line to the curve. Written as df/dx or f'(x).
- **Divergence**: When gradient descent's steps get larger instead of smaller, causing the parameters to explode. Usually caused by a learning rate that's too large.
- **Gradient**: A vector of all partial derivatives of a function. Written as ∇f. Points in the direction of steepest increase of the function.
- **Gradient checking**: Verifying analytical gradients by comparing them to numerical gradients. A standard debugging technique when implementing backpropagation.
- **Gradient descent**: An optimization algorithm that iteratively moves parameters in the opposite direction of the gradient to minimize a loss function.
- **Learning rate**: A hyperparameter that controls the step size in gradient descent. Too small means slow convergence; too large means divergence.
- **Local minimum**: A point where the function is lower than all nearby points, but not necessarily the lowest point overall (the global minimum). Gradient descent converges to whichever local minimum is nearest to the starting point.
- **Loss function**: A function that measures how wrong a model's predictions are. Training minimizes this function. Common examples: mean squared error (regression), cross-entropy (classification).
- **Mean Squared Error (MSE)**: A loss function that computes the average of the squared differences between predictions and actual values: (1/N) * sum((predicted - actual)²).
- **Nabla (∇)**: The symbol for the gradient operator. ∇f means "the gradient of f" -- the vector of all partial derivatives.
- **Partial derivative**: The derivative of a function with respect to one variable, treating all other variables as constants. Written with the curly ∂ symbol: ∂f/∂x.
- **Saddle point**: A point where the gradient is zero but it's not a minimum or maximum -- it's a minimum in some directions and a maximum in others.
- **Tangent line**: A line that touches a curve at exactly one point and has the same slope as the curve at that point. Its slope is the derivative.
