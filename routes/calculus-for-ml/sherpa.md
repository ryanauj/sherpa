---
title: Calculus for ML
route_map: /routes/calculus-for-ml/map.md
paired_guide: /routes/calculus-for-ml/guide.md
topics:
  - Derivatives
  - Gradients
  - Chain Rule
  - Gradient Descent
---

# Calculus for ML - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach the calculus fundamentals that power machine learning — derivatives, gradients, the chain rule, and gradient descent. Every concept is introduced visually and through programming analogies before any formal notation appears.

**Route Map**: See `/routes/calculus-for-ml/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/calculus-for-ml/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Understand derivatives as rates of change and slopes of curves
- Compute partial derivatives and gradients for functions of multiple variables
- Apply the chain rule to compose functions (the foundation of backpropagation)
- Implement gradient descent from scratch and watch it optimize a function
- Build a linear regression model using only gradient descent and numpy

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/calculus-for-ml/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Completed the Linear Algebra Essentials route (vectors, dot products, matrices)
- Python basics and comfort with numpy arrays
- matplotlib installed for visualizations

**If prerequisites are missing**: If they haven't done Linear Algebra Essentials, suggest they work through that route first — gradient descent relies on understanding vectors and dot products. If numpy or matplotlib aren't installed, help them install (`pip install numpy matplotlib`).

### Audience Context
The target learner is a backend developer moving toward machine learning. They completed Linear Algebra Essentials, so they're comfortable with vectors, dot products, and matrix operations in numpy. Their math background beyond that is limited.

Use their programming intuition:
- Function return values → derivatives measure how outputs change relative to inputs
- Loops that adjust a variable → gradient descent iterates toward a minimum
- Function composition / pipelines → chain rule passes rates of change through each stage
- Error metrics in monitoring dashboards → loss functions measure how wrong a model is

Always show the picture BEFORE the equation. When you introduce math notation for the first time, explain it explicitly — the learner may not remember calculus notation from school, if they ever learned it.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). Configuration options include:

**Teaching Style:**
- `tone`: objective, encouraging, humorous (default: objective and respectful)
- `explanation_depth`: concise, balanced, detailed
- `pacing`: learner-led, balanced, structured

**Assessment Format:**
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)
- `quiz_frequency`: after_each_section, after_major_topics, end_of_route
- `feedback_style`: immediate, summary, detailed

**Example `.sherpa-config.yml`:**
```yaml
teaching:
  tone: encouraging
  explanation_depth: balanced
  pacing: learner-led

assessment:
  quiz_type: mixed
  quiz_frequency: after_major_topics
  feedback_style: immediate
```

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types to verify understanding:

**Multiple Choice Questions:**
- Present 3-4 answer options
- Include one correct answer and plausible distractors based on common misconceptions
- Example: "If the gradient at a point is [2, -3], which direction should gradient descent step? A) [2, -3] B) [-2, 3] C) [3, 2] D) [0, 0]"

**Explanation Questions:**
- Ask the learner to explain concepts in their own words
- Assess whether they've built intuition, not just memorized steps
- Example: "In your own words, why does gradient descent move in the opposite direction of the gradient?"

**Prediction Questions:**
- Show a function or a gradient descent setup and ask what will happen before running code
- Builds intuition and catches memorization-without-understanding
- Example: "Before running this, predict: will gradient descent converge, diverge, or oscillate with this learning rate?"

**Code Questions:**
- Ask the learner to write a small function that applies a concept
- Example: "Write a function that computes the numerical derivative of any function at any point"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after new concepts
- Use explanation questions for conceptual understanding (especially chain rule and gradient direction)
- Use prediction questions before running visualization scripts
- Use code questions for derivatives, gradients, and gradient descent implementation

---

## Teaching Flow

### Introduction

**What to Cover:**
- Calculus sounds intimidating, but the core ideas are simple: how fast is something changing, and which direction should I go to make it change less?
- The whole reason ML needs calculus: gradient descent. Neural networks learn by adjusting their parameters to reduce error, and gradient descent uses derivatives to figure out which way to adjust.
- By the end of this route, they'll implement gradient descent from scratch and train a linear regression model with it
- Everything will be hands-on with Python and numpy — no proofs, no epsilon-delta definitions

**Opening Questions to Assess Level:**
1. "Do you remember anything about derivatives or calculus from school? Even vaguely?"
2. "Have you ever heard of gradient descent or seen it mentioned in ML tutorials?"
3. "What got you interested in this — building neural networks, understanding how ML training works, curiosity?"

**Adapt based on responses:**
- If they remember some calculus: Skip the "what is a slope" basics, lean into the gradient and chain rule faster
- If they've seen gradient descent conceptually: Spend less time on motivation, more on implementation details
- If complete beginner to calculus: Go slower on derivatives, use lots of visual examples, don't rush to notation
- If coming from ML motivation: Frame everything around "this is how your model learns" — loss functions, parameter updates, convergence

**Good opening framing:**
"Here's the punchline: when you train a neural network, you're doing one thing — adjusting numbers to make errors smaller. Calculus tells you which direction to adjust them. That's it. Derivatives tell you 'if I nudge this number, how does the error change?' Gradient descent says 'okay, nudge it in the direction that makes the error go down.' Everything in this route is building toward that one idea."

---

### Setup Verification

**Check numpy:**
```bash
python -c "import numpy as np; print(np.__version__)"
```

**If not installed:**
```bash
pip install numpy
```

**Check matplotlib:**
```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

**If not installed:**
```bash
pip install matplotlib
```

**Check visualization scripts:**
Verify the scripts exist at `/tools/ml-visualizations/`. These will be used throughout the session:
- `derivatives.py` — Tangent line and numerical derivative visualizations
- `gradient_descent.py` — Gradient descent on 2D and 3D surfaces

**Quick Notation Primer:**
Before diving in, set a ground rule: "I'm going to explain every piece of math notation the first time it comes up. Calculus has a lot of notation — dx, ∂, ∇ — and none of it is obvious. If I ever use a symbol you don't recognize, stop me immediately. There are no dumb questions about notation."

---

### Section 1: Derivatives as Rates of Change

**Core Concept to Teach:**
A derivative measures how fast a function's output changes when you nudge its input. It's the slope of a curve at a specific point. If you've ever looked at a speedometer — that's a derivative. Speed is the rate of change of position.

**How to Explain:**
1. Start with a picture: Plot a curve (like `f(x) = x²`) and draw a tangent line at one point. "The derivative at that point is the slope of this line."
2. Programming analogy: "Imagine a function `f(x)`. The derivative tells you: if I increase x by a tiny amount, how much does `f(x)` change? It's the sensitivity of the output to the input."
3. The speedometer analogy: "Your car's position is a function of time. The derivative of position with respect to time is velocity — how fast your position is changing. The derivative of velocity is acceleration — how fast your speed is changing. Derivatives measure rates of change."
4. Connect to ML: "In ML, your loss function takes model parameters as input. The derivative tells you how sensitive the loss is to each parameter. That's exactly the information you need to improve the model."

**Visualize First:**
Reference `/tools/ml-visualizations/derivatives.py` or have them run:

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

x = np.linspace(-3, 3, 100)
plt.figure(figsize=(8, 6))
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x²')

# Tangent line at x = 1
x0 = 1
slope = 2 * x0  # derivative of x² is 2x
tangent_y = slope * (x - x0) + f(x0)
plt.plot(x, tangent_y, 'r--', linewidth=1.5, label=f'Tangent at x={x0} (slope={slope})')
plt.plot(x0, f(x0), 'ro', markersize=8)

plt.xlim(-3, 3)
plt.ylim(-1, 9)
plt.grid(True)
plt.legend()
plt.title("Derivative = Slope of the Tangent Line")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

"See that red dashed line? That's the tangent line at x = 1. Its slope is 2. That means: at x = 1, if you increase x by a tiny amount, f(x) increases by about twice that amount."

**Introducing Notation:**
"The derivative of f with respect to x is written several ways:

- `f'(x)` — pronounced 'f prime of x'
- `df/dx` — pronounced 'dee f dee x' — looks like a fraction but it's a single symbol meaning 'rate of change of f with respect to x'
- `d/dx f(x)` — the 'd/dx' part is an operator that says 'take the derivative of whatever follows'

They all mean the same thing: how fast does f change when x changes?"

**The Numerical Derivative:**
"You don't need to know calculus rules to compute a derivative. You can approximate it numerically — just measure the slope between two very close points:"

```python
def numerical_derivative(f, x, h=1e-7):
    """Compute the derivative of f at x using the central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

# Derivative of x² at x = 3
print(numerical_derivative(f, 3))  # ~6.0 (exact answer: 2*3 = 6)

# Derivative of x² at x = -1
print(numerical_derivative(f, -1))  # ~-2.0 (exact answer: 2*(-1) = -2)
```

"This is called the **central difference formula**: `(f(x+h) - f(x-h)) / (2h)`. You're measuring the slope between two points that are very close together (separated by 2h). As h gets smaller, this gets closer to the true derivative."

**Why Central Difference?**
"You might think `(f(x+h) - f(x)) / h` would work — and it does, roughly. But the central difference `(f(x+h) - f(x-h)) / (2h)` is more accurate because it measures the slope centered on x rather than starting from x. It's a small detail but it matters for numerical stability."

**Walk Through — Some Derivative Rules:**
"While the numerical method always works, knowing a few basic rules speeds things up and helps you read ML papers:

- Constant: derivative of 5 is 0 (a flat line has no slope)
- Power rule: derivative of xⁿ is n·xⁿ⁻¹ (e.g., derivative of x² is 2x, derivative of x³ is 3x²)
- Sum rule: derivative of f + g is f' + g' (derivatives distribute over addition)
- Scalar multiple: derivative of c·f is c·f' (constants pull out)

You don't need to memorize these. The numerical method is your safety net — you can always check."

**Verify Understanding of Rules:**
```python
import numpy as np

# Verify power rule: d/dx[x³] = 3x²
def f(x):
    return x ** 3

def analytical_derivative(x):
    return 3 * x ** 2

x = 2.0
print(f"Numerical:   {numerical_derivative(f, x):.6f}")
print(f"Analytical:  {analytical_derivative(x):.6f}")
# Both should print ~12.0
```

**Common Misconceptions:**
- Misconception: "Derivatives are abstract math with no practical use" → Clarify: "Derivatives are literally how ML models train. Every time a neural network updates its weights, it's using derivatives to figure out which direction to adjust."
- Misconception: "You need to master all the derivative rules before using them" → Clarify: "The numerical method works on any function. Analytical rules are a shortcut, not a prerequisite. Start with numerical, learn rules as patterns emerge."
- Misconception: "The derivative IS the tangent line" → Clarify: "The derivative is a number (the slope). The tangent line is a line with that slope passing through the point. The derivative tells you the slope; it doesn't give you the line itself."
- Misconception: "A derivative of 0 means the function isn't doing anything" → Clarify: "A derivative of 0 means the function is momentarily flat — it's at a peak, a valley, or a saddle point. The function still has a value there, it's just not changing at that instant."

**Verification Questions:**
1. "In your own words, what does a derivative measure?"
2. "If f(x) = x² and the derivative at x = 3 is 6, what does that 6 mean concretely?"
3. Multiple choice: "The derivative of a function at a peak is: A) Very large B) Zero C) Negative D) Undefined"
4. "Write the numerical derivative function from memory. What are the three ingredients?"

**Good answer indicators:**
- They describe it as "rate of change" or "how sensitive the output is to the input"
- They say something like "if you increase x by a tiny amount from 3, f(x) increases by about 6 times that amount"
- They can answer B (zero at a peak — the tangent line is flat)
- They recall: the function f, the point x, and a small step h

**If they struggle:**
- Go back to the speedometer analogy: "You're in a car. The derivative of your position is your speed. If the speedometer reads 60, that means your position is changing at 60 mph."
- Use a table of values: compute f(x) for x = 2.0, 2.001, 2.01, 2.1 and show how the ratio of changes approaches the derivative
- "Forget the word 'derivative.' Think of it as the 'nudge ratio' — if I nudge the input, how much does the output nudge?"
- Run the visualization script and move the tangent line to different points: "Watch how the slope changes as you move along the curve"

**Exercise 1.1:**
"Write a function `numerical_derivative(f, x, h=1e-7)` that computes the derivative of any function f at any point x. Then use it to compute derivatives of:
- `f(x) = x²` at x = 0, 1, 2, 3
- `f(x) = x³ - 2x` at x = 1
- `f(x) = np.sin(x)` at x = 0"

**How to Guide Them:**
1. "Start with the central difference formula: (f(x+h) - f(x-h)) / (2h)"
2. If stuck on sin: "np.sin is just a function. Plug it in the same way."
3. After they compute: "Do you see a pattern in the x² derivatives at 0, 1, 2, 3?"

**Solution:**
```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# f(x) = x²
f1 = lambda x: x ** 2
for x in [0, 1, 2, 3]:
    print(f"f(x)=x² at x={x}: derivative = {numerical_derivative(f1, x):.4f}")
# Output: 0, 2, 4, 6 — the derivative of x² is 2x

# f(x) = x³ - 2x
f2 = lambda x: x ** 3 - 2 * x
print(f"f(x)=x³-2x at x=1: derivative = {numerical_derivative(f2, 1):.4f}")
# Output: ~1.0 (3*1² - 2 = 1)

# f(x) = sin(x)
f3 = np.sin
print(f"f(x)=sin(x) at x=0: derivative = {numerical_derivative(f3, 0):.4f}")
# Output: ~1.0 (derivative of sin is cos, cos(0) = 1)
```

**Exercise 1.2:**
"Plot f(x) = x³ - 3x and its tangent line at x = -1, x = 0, and x = 1. Where is the function going up? Going down? Flat?"

**How to Guide Them:**
1. "Compute the derivative at each point first."
2. "A positive derivative means the function is going up (increasing). Negative means going down. Zero means flat."
3. "Plot the curve and add tangent lines like the example earlier."

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 3 - 3 * x

x = np.linspace(-2.5, 2.5, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x³ - 3x')

for x0, color in [(-1, 'red'), (0, 'green'), (1, 'orange')]:
    slope = numerical_derivative(f, x0)
    tangent = slope * (x - x0) + f(x0)
    plt.plot(x, tangent, '--', color=color, linewidth=1.5,
             label=f'Tangent at x={x0} (slope={slope:.1f})')
    plt.plot(x0, f(x0), 'o', color=color, markersize=8)

plt.ylim(-5, 5)
plt.grid(True)
plt.legend()
plt.title("Tangent Lines at Different Points")
plt.show()
```

"At x = -1 the derivative is 0 (local maximum). At x = 0 the derivative is -3 (going down steeply). At x = 1 the derivative is 0 again (local minimum). See how the flat tangent lines correspond to peaks and valleys?"

**After exercises, ask:**
- "Does the connection between slope and rate of change make sense?"
- "Can you see how knowing the derivative tells you which direction the function is increasing?"
- Adjust pacing based on response

---

### Section 2: Partial Derivatives and Gradients

**Core Concept to Teach:**
Most ML functions take multiple inputs — a loss function depends on many model parameters. A partial derivative measures the rate of change with respect to one input while holding the others constant. The gradient collects all partial derivatives into a vector that points in the direction of steepest ascent.

**How to Explain:**
1. Start with the picture: "Imagine a hilly landscape. You're standing at a point on a hill. The gradient tells you which direction is the steepest uphill climb from where you're standing."
2. Programming analogy: "If you have a function `loss(w1, w2)`, the partial derivative with respect to w1 tells you 'if I nudge w1 while keeping w2 fixed, how does the loss change?' It's like running A/B tests — change one variable, measure the effect."
3. Connect to Section 1: "A partial derivative IS a regular derivative — you just pretend all the other variables are constants. You already know how to compute derivatives. This extends the same idea to multiple dimensions."

**Walk Through — A Function of Two Variables:**

```python
import numpy as np

def f(x, y):
    return x ** 2 + 3 * y ** 2

# At the point (1, 2):
# f(1, 2) = 1 + 12 = 13
```

"This function takes two inputs and returns one output. You can think of it as a surface — for every (x, y) point on the ground, f gives a height."

**Visualize the Surface:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + 3 * Y ** 2

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = x² + 3y²')
plt.show()
```

"It's a bowl. The minimum is at the origin (0, 0). Gradient descent will try to find the bottom of this bowl."

**Introducing Notation — Partial Derivatives:**
"The partial derivative of f with respect to x is written:

`∂f/∂x`

That curly 'd' (∂) means 'partial' — it's calculus's way of saying 'I'm only looking at how f changes when x changes, while y stays fixed.'

For `f(x, y) = x² + 3y²`:
- `∂f/∂x = 2x` — treat y as a constant, differentiate with respect to x
- `∂f/∂y = 6y` — treat x as a constant, differentiate with respect to y"

**Computing Partial Derivatives Numerically:**
```python
def partial_derivative(f, point, variable_index, h=1e-7):
    """Compute the partial derivative of f at a point with respect to one variable."""
    point_plus = np.array(point, dtype=float)
    point_minus = np.array(point, dtype=float)
    point_plus[variable_index] += h
    point_minus[variable_index] -= h
    return (f(*point_plus) - f(*point_minus)) / (2 * h)

def f(x, y):
    return x ** 2 + 3 * y ** 2

# Partial derivatives at (1, 2)
df_dx = partial_derivative(f, [1, 2], 0)  # ~2.0 (analytical: 2*1 = 2)
df_dy = partial_derivative(f, [1, 2], 1)  # ~12.0 (analytical: 6*2 = 12)
print(f"∂f/∂x at (1,2) = {df_dx:.4f}")
print(f"∂f/∂y at (1,2) = {df_dy:.4f}")
```

"Same trick as before — nudge one variable, hold the others still, measure the change."

**The Gradient Vector:**
"The gradient is all the partial derivatives packed into a vector:

`∇f = [∂f/∂x, ∂f/∂y]`

That upside-down triangle (∇) is called 'nabla' or 'del.' It's just a symbol meaning 'gradient of.'

For `f(x, y) = x² + 3y²` at the point (1, 2):
- `∇f = [2x, 6y] = [2, 12]`"

```python
def gradient(f, point, h=1e-7):
    """Compute the gradient of f at a point."""
    grad = np.zeros(len(point))
    for i in range(len(point)):
        grad[i] = partial_derivative(f, point, i, h)
    return grad

grad = gradient(f, [1, 2])
print(f"Gradient at (1, 2): {grad}")  # [2.0, 12.0]
```

**The Key Insight — Gradient Direction:**
"The gradient vector points in the direction of steepest ASCENT — the direction where the function increases the fastest.

This is critical: **the gradient points uphill, not downhill.** If you want to minimize a function (which is what ML training does), you go in the OPPOSITE direction of the gradient."

**Visualize Gradient Vectors:**
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x ** 2 + 3 * y ** 2

# Contour plot with gradient arrows
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + 3 * Y ** 2

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(label='f(x, y)')

# Plot gradient arrows at a few points
points = [(-2, -1), (1, 2), (2, -2), (-1, 1)]
for px, py in points:
    gx, gy = 2 * px, 6 * py  # analytical gradient
    # Normalize for visualization
    length = np.sqrt(gx**2 + gy**2)
    plt.arrow(px, py, gx/length * 0.4, gy/length * 0.4,
              head_width=0.1, head_length=0.05, fc='red', ec='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Vectors on Contour Plot (point uphill)')
plt.gca().set_aspect('equal')
plt.show()
```

"See how every red arrow points away from the center (the minimum) toward higher values? That's the gradient — steepest ascent. Gradient descent goes the opposite way."

**Common Misconceptions:**
- Misconception: "The gradient points toward the minimum" → Clarify: "It points toward the MAXIMUM (steepest ascent). You go in the OPPOSITE direction of the gradient to descend. This is the single most important thing to get right."
- Misconception: "Partial derivatives are a different kind of math from regular derivatives" → Clarify: "A partial derivative is just a regular derivative where you treat the other variables as constants. If f(x, y) = x² + 3y², then ∂f/∂x is the derivative of x² + constant, which is 2x."
- Misconception: "The gradient is a scalar (a single number)" → Clarify: "The gradient is a vector — it has one component per input variable. For a function of 2 variables, the gradient is a 2D vector. For 100 variables, it's a 100-dimensional vector."
- Misconception: "You need to know the analytical formula to compute gradients" → Clarify: "The numerical method works on any function. Compute each partial derivative numerically by nudging one variable at a time."

**Verification Questions:**
1. "What does ∂f/∂x mean in plain English?"
2. "If `f(x, y) = x² + 3y²`, what is ∂f/∂y? How did you compute it?"
3. Multiple choice: "The gradient of a function at a point gives: A) The value of the function B) The direction of steepest descent C) The direction of steepest ascent D) The minimum of the function"
4. "If the gradient at a point is [2, -3], which direction should gradient descent step?" (Answer: [-2, 3])

**Good answer indicators:**
- They describe ∂f/∂x as "how f changes when x changes, with y held fixed"
- They correctly compute ∂f/∂y = 6y by treating x as a constant
- They answer C (steepest ascent) and immediately add "so you go the opposite way"
- They negate the gradient for the descent direction

**If they struggle:**
- Analogy: "Imagine you're standing on a hillside. ∂f/∂x is 'how steep is the hill if I walk purely east?' ∂f/∂y is 'how steep if I walk purely north?' The gradient combines both into 'which direction is steepest overall.'"
- Simplify: start with a function of one variable, compute the derivative, then add a second variable
- "Hold y = 2 constant. Now f(x, 2) = x² + 12, which is just a function of x. Its derivative is 2x. That's ∂f/∂x."
- Use the contour plot with gradient arrows — the visual makes it click

**Exercise 2.1:**
"Compute the gradient of `f(x, y) = 3x² + xy + y²` at the point (1, -1). Do it both analytically and numerically, and verify they match."

**How to Guide Them:**
1. "Start analytically: to find ∂f/∂x, treat y as a constant and differentiate. What's the derivative of 3x²? Of xy (where y is constant)? Of y² (with respect to x)?"
2. If stuck on xy: "If y is a constant, then xy is just y·x. The derivative of y·x with respect to x is y."
3. "Now verify numerically using your gradient function."

**Solution:**
```python
import numpy as np

def f(x, y):
    return 3 * x**2 + x * y + y**2

# Analytical:
# ∂f/∂x = 6x + y → at (1, -1): 6(1) + (-1) = 5
# ∂f/∂y = x + 2y → at (1, -1): 1 + 2(-1) = -1
# Gradient = [5, -1]

# Numerical:
grad = gradient(f, [1, -1])
print(f"Numerical gradient: {grad}")   # [5.0, -1.0]
print(f"Analytical gradient: [5, -1]")
```

**Exercise 2.2:**
"The function `f(x, y, z) = x² + 2y² + 3z²` takes three inputs. Compute the gradient at (1, 1, 1) numerically. What does the gradient tell you about which variable has the most influence on the function at this point?"

**How to Guide Them:**
1. "Use the gradient function — it works for any number of variables."
2. "Which component of the gradient is largest? That variable has the steepest rate of change."
3. "What does this mean for gradient descent? Which parameter would change the most?"

**Solution:**
```python
def f3(x, y, z):
    return x**2 + 2*y**2 + 3*z**2

grad = gradient(f3, [1, 1, 1])
print(f"Gradient: {grad}")  # [2.0, 4.0, 6.0]
# z has the largest gradient component — the function is most
# sensitive to z at this point
```

**After exercises, ask:**
- "Does the gradient make intuitive sense as 'direction of steepest ascent'?"
- "Can you see how this applies to ML — if you have a loss function with 1000 parameters, the gradient tells you which parameters to change and by how much?"

---

### Section 3: The Chain Rule

**Core Concept to Teach:**
Functions are often composed: the output of one function becomes the input to another. The chain rule tells you how to compute the derivative of a composed function. This is not an abstract math concept — it IS how backpropagation works in neural networks.

**How to Explain:**
1. Programming analogy: "Think of a data pipeline: `raw_data → clean(raw_data) → transform(cleaned) → score(transformed)`. If you want to know how a change in the raw data affects the final score, you need to trace through each stage. The chain rule tells you: multiply the rates of change at each stage."
2. Visual: Draw the pipeline as boxes connected by arrows. Each box has a local rate of change (derivative). The total rate of change is the product of all the local rates.
3. Connect to ML: "In a neural network, data flows through layers. Each layer is a function. Backpropagation uses the chain rule to compute how the final loss depends on each weight by tracing backward through the layers and multiplying derivatives at each step."

**The Chain Rule — One Variable:**
"If `y = f(g(x))` — that is, x goes into g, and the result goes into f — then:

`dy/dx = f'(g(x)) · g'(x)`

In English: the derivative of the whole composition equals the derivative of the outer function (evaluated at the inner function's output) times the derivative of the inner function."

**Concrete Example:**
```python
import numpy as np

# y = (3x + 1)²
# This is f(g(x)) where:
#   g(x) = 3x + 1    (inner function)
#   f(u) = u²         (outer function, where u = g(x))

# Chain rule:
#   g'(x) = 3
#   f'(u) = 2u
#   dy/dx = f'(g(x)) · g'(x) = 2(3x + 1) · 3 = 6(3x + 1)

def y(x):
    return (3 * x + 1) ** 2

# Verify at x = 2:
# Analytical: 6(3*2 + 1) = 6 * 7 = 42
print(f"Numerical:  {numerical_derivative(y, 2):.4f}")
print(f"Analytical: {6 * (3 * 2 + 1)}")  # 42
```

**The Pipeline View:**
"Think of it as rates of change multiplying through a pipeline:

```
x → [g: multiply by 3, add 1] → u → [f: square it] → y

If x changes by 1:
  u changes by 3        (g'(x) = 3)
  y changes by 2u       (f'(u) = 2u)
  Total: y changes by 3 · 2u = 6u = 6(3x + 1)
```

Each stage multiplies its own rate of change. The chain rule is just: multiply them all together."

**Multi-Stage Chain Rule:**
"What if you have three functions composed? Same idea — just keep multiplying:

`d/dx f(g(h(x))) = f'(g(h(x))) · g'(h(x)) · h'(x)`"

```python
import numpy as np

# y = sin(x²)³  → three compositions
# h(x) = x²
# g(u) = sin(u)
# f(v) = v³
# dy/dx = f'(g(h(x))) · g'(h(x)) · h'(x)
#       = 3·sin(x²)² · cos(x²) · 2x

def y(x):
    return np.sin(x ** 2) ** 3

# Verify at x = 1:
x = 1.0
analytical = 3 * np.sin(1)**2 * np.cos(1) * 2
print(f"Numerical:  {numerical_derivative(y, x):.6f}")
print(f"Analytical: {analytical:.6f}")
```

**Connecting to Backpropagation:**
"In a neural network, each layer applies a function:

```
input → [layer 1] → [activation] → [layer 2] → [activation] → ... → loss
```

Backpropagation asks: how does the loss change if I tweak a weight in layer 1? The chain rule answers this by working backward — multiplying the local derivatives at each stage from the loss back to that weight.

You'll implement this in the Neural Network Foundations route. For now, just understand that the chain rule is the mechanism that makes training possible."

**Walk Through — Manual Backprop for a Tiny Graph:**
```python
# Forward pass: a simple computation graph
# x = 2
# a = x * 3      (multiply by 3)
# b = a + 1      (add 1)
# y = b ** 2     (square)

x = 2.0
a = x * 3       # 6
b = a + 1       # 7
y = b ** 2      # 49

# Backward pass: chain rule, working backward from y
dy_db = 2 * b       # d(b²)/db = 2b = 14
db_da = 1           # d(a+1)/da = 1
da_dx = 3           # d(3x)/dx = 3

# Total derivative: dy/dx = dy/db · db/da · da/dx
dy_dx = dy_db * db_da * da_dx  # 14 * 1 * 3 = 42

print(f"dy/dx = {dy_dx}")  # 42

# Verify:
def y_func(x):
    return (3 * x + 1) ** 2

print(f"Numerical: {numerical_derivative(y_func, 2):.4f}")  # ~42
```

"This is backpropagation in its simplest form. Each step computes a local derivative, and you multiply them backward through the chain. That's the chain rule in action."

**Common Misconceptions:**
- Misconception: "The chain rule only applies to special cases" → Clarify: "Almost every function you'll encounter in ML is a composition. Layers in a neural network, activation functions applied to linear transformations, loss functions applied to predictions — the chain rule is everywhere."
- Misconception: "You add the derivatives in a chain" → Clarify: "You MULTIPLY them. Each stage multiplies its local rate of change. If stage 1 doubles the rate and stage 2 triples it, the total is 6x, not 5x."
- Misconception: "Backpropagation is a different algorithm from the chain rule" → Clarify: "Backpropagation IS the chain rule applied to a computation graph. It's not a separate algorithm — it's just an efficient way to apply the chain rule to a network with many parameters."
- Misconception: "You need to know calculus proofs to use the chain rule" → Clarify: "Numerical computation works. Understanding is what matters, not formal proofs. If you can trace a computation graph and multiply local derivatives, you understand the chain rule."

**Verification Questions:**
1. "In your own words, what does the chain rule do?"
2. "If `y = f(g(x))`, and g doubles its input (g'(x) = 2) and f triples its input (f'(u) = 3), what is dy/dx?" (Answer: 6)
3. Multiple choice: "In the chain rule for `f(g(h(x)))`, the derivatives are: A) Added together B) Multiplied together C) Divided D) The maximum is taken"
4. "Why is the chain rule essential for training neural networks?"

**Good answer indicators:**
- They describe it as "how derivatives pass through composed functions"
- They multiply (2 × 3 = 6), not add
- They answer B (multiplied)
- They connect it to backpropagation: "it lets you trace how the loss depends on any weight by multiplying derivatives backward through the layers"

**If they struggle:**
- Domino analogy: "Imagine dominoes falling. The first one pushes the second, which pushes the third. Each domino amplifies or dampens the force. The chain rule tells you the total amplification from first to last."
- Break it down into explicit steps: compute each intermediate value and its derivative separately, then multiply
- Use numbers, not symbols: "g doubles, f triples, so the total is 6x — not complicated"
- Run the numerical derivative on the composed function and compare to the product of local derivatives

**Exercise 3.1:**
"Compute the derivative of `y = sin(x²)` at x = 1 using the chain rule, then verify numerically."

**How to Guide Them:**
1. "What's the outer function? What's the inner function?"
2. "Inner: g(x) = x². Outer: f(u) = sin(u)."
3. "g'(x) = 2x. f'(u) = cos(u). Chain rule: f'(g(x)) · g'(x) = cos(x²) · 2x."
4. "Evaluate at x = 1."

**Solution:**
```python
import numpy as np

# Analytical: dy/dx = cos(x²) · 2x
# At x = 1: cos(1) · 2 ≈ 1.0806

def y(x):
    return np.sin(x ** 2)

analytical = np.cos(1) * 2
numerical = numerical_derivative(y, 1)
print(f"Analytical: {analytical:.6f}")
print(f"Numerical:  {numerical:.6f}")
```

**Exercise 3.2:**
"Trace the following computation graph forward and backward:
```
x = 3
a = 2 * x
b = a ** 2
c = b - 10
```
What is dc/dx? Compute it by multiplying local derivatives backward, then verify numerically."

**How to Guide Them:**
1. "First, compute the forward pass: what are a, b, c?"
2. "Now go backward: dc/db = ?, db/da = ?, da/dx = ?"
3. "Multiply them: dc/dx = dc/db · db/da · da/dx"

**Solution:**
```python
# Forward: x=3, a=6, b=36, c=26
# Backward:
# dc/db = 1 (derivative of b - 10 with respect to b)
# db/da = 2a = 12 (derivative of a² with respect to a)
# da/dx = 2 (derivative of 2x with respect to x)
# dc/dx = 1 * 12 * 2 = 24

def c_func(x):
    a = 2 * x
    b = a ** 2
    return b - 10

print(f"Chain rule:  24")
print(f"Numerical:   {numerical_derivative(c_func, 3):.4f}")  # ~24
```

**After exercises, ask:**
- "Does the chain rule feel mechanical (multiply local derivatives) or still abstract?"
- "Can you see how this extends to a neural network with many layers?"

---

### Section 4: Gradient Descent

**Core Concept to Teach:**
Gradient descent is the optimization algorithm that powers ML training. The idea is simple: you have a function you want to minimize (the loss), you compute the gradient (which direction is uphill), and you take a step in the opposite direction (downhill). Repeat until you reach a minimum.

**How to Explain:**
1. Start with the picture: "Imagine you're blindfolded on a hilly landscape. You want to reach the lowest valley. You can feel the slope of the ground under your feet. What do you do? Walk downhill. That's gradient descent."
2. The algorithm is three lines: "Compute the gradient. Step in the opposite direction. Repeat."
3. The learning rate: "How big a step do you take? That's the learning rate. Too big and you overshoot the valley. Too small and you take forever."

**The Algorithm:**
```
repeat:
    gradient = compute_gradient(loss_function, current_parameters)
    current_parameters = current_parameters - learning_rate * gradient
```

"That's it. The entire core of ML training fits in two lines of math."

**Walk Through — 1D Gradient Descent:**

```python
import numpy as np

def f(x):
    return (x - 3) ** 2 + 1  # Minimum at x = 3, f(3) = 1

def df(x):
    return 2 * (x - 3)  # Derivative

# Gradient descent
x = 0.0            # Starting point
lr = 0.1            # Learning rate
history = [x]

for step in range(30):
    grad = df(x)
    x = x - lr * grad  # Step opposite to gradient
    history.append(x)

print(f"Final x: {x:.4f}")  # Should be close to 3.0
print(f"Final f(x): {f(x):.4f}")  # Should be close to 1.0
```

**Visualize the Descent:**
Reference `/tools/ml-visualizations/gradient_descent.py` or:
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 3) ** 2 + 1

x_range = np.linspace(-1, 7, 100)
plt.figure(figsize=(10, 6))
plt.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x)')

# Plot gradient descent steps
x = 0.0
lr = 0.1
for step in range(15):
    x_new = x - lr * 2 * (x - 3)
    plt.plot(x, f(x), 'ro', markersize=6)
    plt.annotate('', xy=(x_new, f(x_new)), xytext=(x, f(x)),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    x = x_new

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Finding the Minimum')
plt.legend()
plt.grid(True)
plt.show()
```

"Watch the red dots march toward the bottom of the curve. Each step, the gradient says 'uphill is this way,' and gradient descent goes the opposite direction."

**The Learning Rate — Why It Matters:**

"The learning rate (often written as α or η) controls the step size. This is one of the most important hyperparameters in ML."

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 3) ** 2 + 1

def run_gd(x0, lr, steps=30):
    x = x0
    history = [x]
    for _ in range(steps):
        x = x - lr * 2 * (x - 3)
        history.append(x)
    return history

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x_range = np.linspace(-2, 8, 100)

for ax, lr, title in zip(axes,
    [0.01, 0.1, 0.9],
    ['Too Small (lr=0.01)', 'Just Right (lr=0.1)', 'Too Large (lr=0.9)']):

    history = run_gd(0.0, lr)
    ax.plot(x_range, f(x_range), 'b-', linewidth=2)
    ax.plot(history, [f(x) for x in history], 'ro-', markersize=4)
    ax.set_title(title)
    ax.set_ylim(-1, 20)
    ax.grid(True)

plt.tight_layout()
plt.show()
```

"Three scenarios:
- **Too small** (lr = 0.01): Gradient descent crawls. It will eventually get there, but it's painfully slow.
- **Just right** (lr = 0.1): Converges smoothly to the minimum in a reasonable number of steps.
- **Too large** (lr = 0.9): Overshoots the minimum, bounces back and forth, and may diverge entirely."

**2D Gradient Descent:**
"Now let's do it in 2D — a function with two parameters, just like a real loss function:"

```python
import numpy as np

def loss(params):
    x, y = params
    return x ** 2 + 3 * y ** 2

def grad_loss(params):
    x, y = params
    return np.array([2 * x, 6 * y])

# Gradient descent in 2D
params = np.array([4.0, 3.0])  # Starting point
lr = 0.05
history = [params.copy()]

for step in range(50):
    g = grad_loss(params)
    params = params - lr * g
    history.append(params.copy())

print(f"Final params: {params}")  # Should be close to [0, 0]
print(f"Final loss: {loss(params):.6f}")  # Should be close to 0
```

**Visualize 2D Gradient Descent:**
```python
import numpy as np
import matplotlib.pyplot as plt

def loss(params):
    x, y = params
    return x ** 2 + 3 * y ** 2

history = np.array(history)

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X ** 2 + 3 * Y ** 2

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Loss')
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4, linewidth=1)
plt.plot(history[0, 0], history[0, 1], 'rs', markersize=10, label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on f(x,y) = x² + 3y²')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()
```

"The path zigzags toward the center. Notice it descends faster in the y direction (because the gradient is steeper there — the coefficient of y² is larger)."

**Local Minima and Saddle Points:**
"For simple bowl-shaped functions, gradient descent finds the global minimum easily. But real loss surfaces can have:

- **Local minima**: Valleys that aren't the deepest one. Gradient descent can get stuck.
- **Saddle points**: Points that are a minimum in one direction and a maximum in another. The gradient is zero, but it's not a minimum.

In practice, for neural networks with many parameters, saddle points are more common than local minima. Advanced optimizers (Adam, RMSProp) handle these better than basic gradient descent."

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# A function with a saddle point at the origin
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = X ** 2 - Y ** 2  # Saddle: goes up in x, down in y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax.set_title('Saddle Point: f(x,y) = x² - y²')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.show()
```

"At the origin, the gradient is [0, 0] — it looks like a minimum to gradient descent. But it's only a minimum in the x direction. In the y direction, it's actually a maximum. This is a saddle point."

**Common Misconceptions:**
- Misconception: "Gradient descent always finds the global minimum" → Clarify: "It finds A minimum, which might be local. For convex functions (bowl-shaped), local = global. For complex loss surfaces, it depends on where you start and your learning rate."
- Misconception: "The learning rate doesn't matter much" → Clarify: "It's one of the most important hyperparameters. Too high = divergence (loss goes to infinity). Too low = training takes forever. Getting it right is crucial."
- Misconception: "You just run gradient descent until the loss is zero" → Clarify: "For most problems, the loss won't reach zero — you stop when it's 'good enough' or stops improving. Also, zero training loss often means overfitting."
- Misconception: "More gradient descent steps are always better" → Clarify: "After convergence, more steps don't help. And with noisy data (stochastic gradient descent), too many steps can actually hurt by overfitting."

**Verification Questions:**
1. "What are the two pieces of information gradient descent needs at each step?" (The gradient and the learning rate)
2. "If the gradient at a point is [2, -3], and the learning rate is 0.1, what is the parameter update?" (Answer: step = -0.1 × [2, -3] = [-0.2, 0.3])
3. Multiple choice: "A very large learning rate will cause gradient descent to: A) Converge faster B) Overshoot and possibly diverge C) Find the global minimum D) Need fewer iterations"
4. "What happens at a saddle point?"

**Good answer indicators:**
- They know gradient descent needs the gradient direction and learning rate
- They correctly compute the update by negating and scaling the gradient
- They answer B (overshoot/diverge)
- They can describe a saddle point as flat in the gradient (zero) but not actually a minimum

**If they struggle:**
- Return to the blindfolded hillwalker analogy: "Feel the slope. Step downhill. That's all it is."
- Start with 1D only. Don't move to 2D until they can trace the 1D algorithm step by step.
- Walk through 3 steps of gradient descent by hand on paper with concrete numbers
- "The gradient tells you the direction. The learning rate tells you how far. Multiply and subtract."
- Show the three learning rate plots side by side — the visual makes the learning rate intuition click

**Exercise 4.1:**
"Implement gradient descent to minimize `f(x) = x⁴ - 3x² + 2` starting from x = 2. Use a learning rate of 0.01 and run for 200 steps. Plot the path of x values over time. Does it converge? To which minimum?"

**How to Guide Them:**
1. "First, plot the function to see what it looks like — how many minima does it have?"
2. "Write the gradient descent loop: compute the derivative numerically, update x."
3. "Try different starting points — does it always find the same minimum?"

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 4 - 3 * x ** 2 + 2

# Plot the function
x_range = np.linspace(-2.5, 2.5, 200)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x_range, f(x_range), 'b-', linewidth=2)
plt.title('f(x) = x⁴ - 3x² + 2')
plt.grid(True)

# Gradient descent
x = 2.0
lr = 0.01
history = [x]

for _ in range(200):
    grad = numerical_derivative(f, x)
    x = x - lr * grad
    history.append(x)

plt.subplot(1, 2, 2)
plt.plot(history, 'r-', linewidth=1.5)
plt.xlabel('Step')
plt.ylabel('x')
plt.title(f'x converges to {x:.4f}')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final x: {x:.4f}, f(x): {f(x):.4f}")
```

"The function has two local minima. Starting from x = 2, gradient descent finds the one near x ≈ 1.22. Starting from x = -2 would find the other one near x ≈ -1.22. This illustrates that gradient descent finds LOCAL minima — the one it finds depends on where you start."

**Exercise 4.2:**
"Implement 2D gradient descent on `loss(x, y) = (x - 1)² + 10(y + 2)²` with learning rate 0.05, starting from (5, 5). Plot the contour path. Why does the path converge faster in one direction than the other?"

**How to Guide Them:**
1. "This is the same template as the earlier 2D example."
2. "Compute the gradient analytically: ∂f/∂x = ?, ∂f/∂y = ?"
3. "After running, look at the contour plot. The y coefficient is 10 (vs 1 for x), so the surface is much steeper in y."

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

def loss(x, y):
    return (x - 1)**2 + 10 * (y + 2)**2

def grad_loss(x, y):
    return np.array([2 * (x - 1), 20 * (y + 2)])

params = np.array([5.0, 5.0])
lr = 0.05
history = [params.copy()]

for _ in range(100):
    g = grad_loss(*params)
    params = params - lr * g
    history.append(params.copy())

history = np.array(history)

x_range = np.linspace(-1, 6, 100)
y_range = np.linspace(-5, 6, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (X - 1)**2 + 10 * (Y + 2)**2

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=3)
plt.plot(history[0, 0], history[0, 1], 'rs', markersize=10, label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
plt.title('Gradient Descent: y converges faster (steeper gradient)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final: ({params[0]:.3f}, {params[1]:.3f})")
# Should be close to (1, -2)
```

"The loss is 10x steeper in y than in x, so the gradient in y is much larger and y converges faster. This is a real issue in ML — parameters with very different gradient magnitudes cause zigzagging. Adaptive learning rates (like Adam) address this by giving each parameter its own effective learning rate."

**After exercises, ask:**
- "Do you feel confident implementing gradient descent from scratch?"
- "Can you see how this extends to ML — instead of x and y, you have millions of weights?"

---

## Practice Project

**Project Introduction:**
"Let's put everything together. You're going to build linear regression from scratch — no scikit-learn, no autograd, just numpy. You'll define a loss function, compute its gradient, and use gradient descent to find the best-fit line for a dataset."

**Requirements:**
Present one at a time:
1. "Generate synthetic data: a line with noise — `y = 2x + 1 + noise`"
2. "Define the mean squared error (MSE) loss function for a linear model `y_pred = w*x + b`"
3. "Compute the gradient of MSE with respect to w and b (analytically)"
4. "Implement gradient descent to learn w and b from data"
5. "Plot the loss curve over training steps and the final fit line against the data"

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work, offer to answer questions
2. **If they want guidance**: Build it step by step together
3. **If they're unsure**: Start with step 1 (data generation) and check in

**Step-by-Step Guidance:**

**Step 1 — Generate Data:**
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 50
x_data = np.random.uniform(-5, 5, n)
y_data = 2 * x_data + 1 + np.random.normal(0, 1, n)  # True: w=2, b=1

plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data')
plt.grid(True)
plt.show()
```

**Step 2 — Define MSE Loss:**
"Mean Squared Error: average the squared differences between predictions and actual values.

```
MSE = (1/n) · Σ(y_pred - y_actual)²
    = (1/n) · Σ(w·x + b - y)²
```"

```python
def mse_loss(w, b, x, y):
    predictions = w * x + b
    return np.mean((predictions - y) ** 2)

# Initial loss with random w, b
print(f"Initial loss: {mse_loss(0, 0, x_data, y_data):.4f}")
```

**Step 3 — Compute the Gradient:**
"Derive the partial derivatives of MSE with respect to w and b:

```
∂MSE/∂w = (2/n) · Σ(w·x + b - y) · x
∂MSE/∂b = (2/n) · Σ(w·x + b - y)
```

In code:"

```python
def mse_gradient(w, b, x, y):
    n = len(x)
    predictions = w * x + b
    errors = predictions - y
    dw = (2 / n) * np.sum(errors * x)
    db = (2 / n) * np.sum(errors)
    return dw, db
```

"Encourage them to verify against numerical gradients:"
```python
# Numerical check
h = 1e-7
w, b = 0.5, 0.5
dw_num = (mse_loss(w + h, b, x_data, y_data) - mse_loss(w - h, b, x_data, y_data)) / (2 * h)
db_num = (mse_loss(w, b + h, x_data, y_data) - mse_loss(w, b - h, x_data, y_data)) / (2 * h)
dw_analytical, db_analytical = mse_gradient(w, b, x_data, y_data)
print(f"dw — numerical: {dw_num:.6f}, analytical: {dw_analytical:.6f}")
print(f"db — numerical: {db_num:.6f}, analytical: {db_analytical:.6f}")
```

**Step 4 — Gradient Descent Training:**
```python
# Training
w, b = 0.0, 0.0
lr = 0.01
epochs = 200
loss_history = []

for epoch in range(epochs):
    loss = mse_loss(w, b, x_data, y_data)
    loss_history.append(loss)
    dw, db = mse_gradient(w, b, x_data, y_data)
    w = w - lr * dw
    b = b - lr * db

print(f"Learned w: {w:.4f} (true: 2)")
print(f"Learned b: {b:.4f} (true: 1)")
print(f"Final loss: {loss_history[-1]:.4f}")
```

**Step 5 — Visualize Results:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(loss_history, 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True)

# Fit line
axes[1].scatter(x_data, y_data, alpha=0.7, label='Data')
x_line = np.linspace(-5, 5, 100)
axes[1].plot(x_line, w * x_line + b, 'r-', linewidth=2,
             label=f'Learned: y = {w:.2f}x + {b:.2f}')
axes[1].plot(x_line, 2 * x_line + 1, 'g--', linewidth=1.5,
             label='True: y = 2x + 1')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Linear Regression Fit')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

**Checkpoints During Project:**
- After data generation: "Can you see the linear trend in the scatter plot?"
- After MSE definition: "Plug in w=0, b=0. Is the loss high? Does that make sense?"
- After gradient computation: "Verify your analytical gradient matches the numerical one. They should agree to at least 5 decimal places."
- After training: "How close are your learned w and b to the true values? Why aren't they exact?" (noise in the data)
- After visualization: "Does the loss curve decrease smoothly? If it's jagged or increasing, check your gradient computation or learning rate."

**Code Review Approach:**
When reviewing their work:
1. Start with what works: "The loss is decreasing — the gradient is correct"
2. Check understanding: "Why did you choose this learning rate? What happens if you double it?"
3. Push deeper: "What would happen with more noise in the data? Less data?"
4. Connect to ML: "You just trained a model using gradient descent. A neural network does exactly this, but with more parameters and the chain rule for gradients."

**If They Get Stuck:**
- "Which step are you on? Data, loss function, gradient, training loop, or visualization?"
- If gradient is wrong: "Check your gradient against the numerical derivative. They must match."
- If loss increases: "Your learning rate is too high, or there's a sign error in the update. Are you subtracting the gradient?"
- If loss doesn't decrease: "Learning rate might be too small. Try 10x larger."

**Extension Ideas if They Finish Early:**
- "Try different learning rates and plot the loss curves on the same graph"
- "Add a stopping criterion: stop when the loss changes by less than 1e-6 between epochs"
- "Extend to polynomial regression: fit y = ax² + bx + c using gradient descent on three parameters"
- "Add momentum: keep a running average of past gradients and use that for the update"
- "Compare your results to `np.polyfit(x_data, y_data, 1)` — the closed-form solution"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- A derivative measures the rate of change — how sensitive a function's output is to its input
- Partial derivatives measure the rate of change with respect to one variable while holding others constant
- The gradient is a vector of all partial derivatives — it points in the direction of steepest ascent
- The chain rule tells you how derivatives flow through composed functions (multiply local derivatives)
- Gradient descent walks downhill on the loss surface by stepping opposite to the gradient
- You built linear regression from scratch using these ideas — the same ideas that train neural networks

**Ask them to explain one concept:**
"Can you explain in your own words why gradient descent moves in the OPPOSITE direction of the gradient?"
(This reinforces the key insight — the gradient points uphill, descent goes downhill)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with calculus for ML?"

**Respond based on answer:**
- 1-4: "That's okay. Calculus builds slowly. The numerical methods are your safety net — you can always compute a derivative even if the analytical version is hard. Spend time running the visualization scripts and changing parameters. Seeing gradient descent converge is worth more than any formula."
- 5-7: "Good progress. You have the foundation. When you see gradient descent in a real ML library, you'll know what's happening under the hood. Practice computing gradients for different loss functions."
- 8-10: "Solid. You're ready for neural networks. The next route applies everything you learned here to build and train a network from scratch."

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice more: Implement gradient descent for a quadratic function in higher dimensions (3+ variables)"
- "For a challenge: Implement mini-batch stochastic gradient descent (SGD) on the linear regression problem"
- "When you're ready: Neural Network Foundations uses derivatives, gradients, and the chain rule to build a working neural network"
- "For deeper understanding: Read about automatic differentiation (autograd) — how libraries like PyTorch compute derivatives without numerical approximation"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there a specific concept you want to revisit?"
"Was there anything that felt shaky or that you want more practice with?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused by derivative notation (df/dx, ∂f/∂x, ∇)
- Can't connect the numerical method to the concept
- Gets lost in multi-variable functions
- Doesn't see why any of this relates to ML

**Strategies:**
- Stay in 1D until derivatives click. Don't introduce partial derivatives until single-variable is solid.
- Always use the numerical method first — it's concrete and debuggable. Analytical rules are a bonus.
- "Forget the symbols. The derivative just answers: if I nudge the input, how much does the output change?"
- Use the speedometer analogy repeatedly — it maps perfectly to the concept
- Run visualization scripts for everything — see the tangent line, see the gradient arrows, watch gradient descent converge
- Connect every concept back to ML: "This is literally how your model learns"
- If notation is the blocker: write code alongside every equation. "∂f/∂x just means `partial_derivative(f, point, 0)`"
- Check if math anxiety is the issue vs. actual conceptual confusion — adjust tone accordingly

### If Learner is Excelling

**Signs:**
- Completes exercises quickly and correctly
- Asks about higher-order derivatives, Hessians, or second-order methods
- Wants to know about automatic differentiation
- Starts writing more complex loss functions on their own

**Strategies:**
- Move at faster pace, skip basic numerical verification if they're computing analytically
- Introduce the concept of the Hessian (matrix of second derivatives) and what it tells you about curvature
- Discuss momentum and adaptive learning rates (Adam, RMSProp) conceptually
- Preview automatic differentiation: "Libraries like PyTorch track computations and apply the chain rule automatically"
- Challenge: "Implement gradient descent with momentum. How does it change the convergence path?"
- Discuss learning rate schedules: cosine annealing, warm restarts, step decay
- Introduce the concept of convexity and why it matters (convex = gradient descent finds the global minimum)
- Ask: "Can you implement Newton's method? It uses second derivatives to choose better step sizes."

### If Learner Seems Disengaged

**Signs:**
- Short, minimal responses
- Not asking questions
- Rushing through exercises without engaging
- Seems to find the content too abstract

**Strategies:**
- Connect to their goals immediately: "What do you want to build? Let's see how gradient descent fits in."
- Show something impressive: train the linear regression model live and watch the line snap into place
- Skip ahead to gradient descent if derivatives feel too abstract — motivation flows backward
- "Every AI model you've ever used was trained with gradient descent. Let me show you what that looks like."
- Make it competitive: "Can you find a learning rate that converges in fewer than 50 steps?"
- Reduce theory, increase coding. Let them experiment rather than listen.

### Different Learning Styles

**Visual learners:**
- Run every visualization script. The contour plots with gradient arrows are essential.
- Always plot before and after. Show the loss curve decreasing.
- Use contour plots for 2D functions — they're more readable than 3D surface plots.

**Hands-on learners:**
- Get them coding immediately. "Here's a function, compute its derivative numerically, let's go."
- Exercise-driven: less lecture, more experimentation
- "Try changing the learning rate. What happens? Try a different function. What changes?"

**Conceptual learners:**
- Spend extra time on why the chain rule multiplies rather than adds
- Discuss the geometric meaning of the gradient in detail
- Connect derivatives to linear approximation: "The derivative tells you the best linear approximation of a function at a point"
- They may want to understand why the central difference is more accurate than the forward difference

**Example-driven learners:**
- Show code first, explain after
- Use concrete numbers everywhere before generalizing
- Build up from 1D to 2D to the linear regression project

---

## Troubleshooting Common Issues

### numpy Not Installed
```bash
pip install numpy
# or
pip3 install numpy
# or in a virtual environment
python -m pip install numpy
```
Verify: `python -c "import numpy; print(numpy.__version__)"`

### matplotlib Not Showing Plots
- On macOS, they may need a backend: `pip install pyobjc` or use `matplotlib.use('TkAgg')`
- In a remote session (SSH): plots won't display. Use `plt.savefig('output.png')` instead of `plt.show()`
- In VS Code: plots may appear in a separate window or inline depending on settings
- If nothing appears: add `plt.savefig('plot.png')` before `plt.show()` as a fallback

### Numerical Derivative Returns NaN or Inf
- h is too small: floating-point precision issues. Use h = 1e-7 (not 1e-15)
- h is too large: the approximation is inaccurate. Stay between 1e-5 and 1e-8.
- The function itself has a discontinuity or is undefined at that point

### Gradient Descent Diverges (Loss Goes to Infinity)
- Learning rate is too high. Cut it in half and try again.
- Sign error: make sure you're SUBTRACTING the gradient, not adding it
- Gradient computation is wrong: verify against numerical gradient

### Gradient Descent Converges Too Slowly
- Learning rate is too small. Try 2x or 10x larger.
- The function has very different scales in different directions — adaptive learning rates help
- Starting point is far from the minimum

### Analytical and Numerical Gradients Don't Match
- Check the analytical formula for errors — common mistakes: missing factor of 2, sign error, wrong variable
- Make sure h isn't too large or too small
- Check that you're evaluating both at the same point

### Concept-Specific Confusion

**If confused about derivatives:**
- Go back to the speedometer: "Speed is the derivative of position. Acceleration is the derivative of speed."
- Use a table of values and compute slopes between points
- "The derivative is the nudge ratio. Nudge x, measure how much f(x) nudges."

**If confused about partial derivatives:**
- "Hold y constant. Now it's just a regular derivative."
- Compute f(1, 2), f(1.001, 2), and show the difference. That's the partial derivative with respect to x at (1, 2).
- Use the A/B test analogy: change one variable, measure the effect.

**If confused about the chain rule:**
- Use the domino analogy: each stage amplifies or dampens the signal
- Work through the computation graph step by step with concrete numbers
- "Just multiply the local derivatives. That's the whole rule."

**If confused about gradient descent:**
- Return to the blindfolded hillwalker
- Walk through 5 steps by hand with concrete numbers
- Plot everything — the function, the steps, the loss curve

---

## Teaching Notes

**Key Emphasis Points:**
- "The gradient points uphill, descent goes downhill" — the single most common error is getting this backward. Reinforce it multiple times.
- The numerical derivative is the learner's safety net. If they can compute it, they can always check their analytical work.
- The chain rule = multiply local derivatives. Keep returning to this simple framing rather than the formal notation.
- Gradient descent is the punchline of the entire route. Sections 1-3 build toward it. Keep previewing where you're headed.
- Code first, math second. Every equation should have a corresponding Python expression right next to it.

**Pacing Guidance:**
- Don't rush Section 1 (derivatives). If they don't feel the "rate of change" intuition, nothing else will click.
- Section 2 (gradients) extends Section 1 to multiple variables. If derivatives clicked, this should be straightforward — spend more time on the gradient direction insight.
- Section 3 (chain rule) is conceptually the hardest section. Take your time. Use the pipeline analogy and concrete computation graphs. Don't move on until they can multiply through a 3-stage chain.
- Section 4 (gradient descent) is the payoff. Give it plenty of time for experimentation — different learning rates, different functions, 1D and 2D.
- The practice project integrates everything. Allow plenty of time. This is where the concepts solidify.

**Success Indicators:**
You'll know they've got it when they:
- Can compute a numerical derivative of any function without looking at notes
- Know that the gradient points uphill and descent goes opposite
- Can trace through a chain rule computation by multiplying local derivatives
- Can implement gradient descent from scratch and debug it (learning rate, sign errors)
- Ask questions like "what learning rate should I use?" or "what if the loss has multiple minima?"
- Connect the concepts to ML: "so backpropagation is just the chain rule" or "the weights are the parameters gradient descent optimizes"

**Most Common Confusion Points:**
1. **Gradient direction**: Uphill, not downhill. Say it three times.
2. **Chain rule**: Multiplication, not addition. Walk through concrete examples.
3. **Learning rate**: The Goldilocks problem. Show all three regimes visually.
4. **Notation**: ∂, ∇, df/dx — translate every symbol to code.
5. **Why bother with analytical derivatives**: The numerical method always works, but analytical is faster and exact. Analytical lets you understand the structure of the gradient.

**Teaching Philosophy:**
- The learner already thinks computationally — leverage that. Code is their native language for understanding math.
- Visualization is not supplementary, it's primary. The contour plot with gradient arrows IS the understanding.
- Every equation is a translation of an intuitive idea. If the intuition isn't clear, the equation is useless.
- Let numpy do the arithmetic. The learner's job is to build intuition, not practice computation.
- This route succeeds when the learner sees "gradient descent" in an ML tutorial and thinks "I know exactly what that means and I could implement it from scratch."
