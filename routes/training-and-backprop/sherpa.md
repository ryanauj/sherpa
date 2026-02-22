---
title: Training and Backpropagation
route_map: /routes/training-and-backprop/map.md
paired_guide: /routes/training-and-backprop/guide.md
topics:
  - Loss Functions
  - Backpropagation
  - Training Loops
  - Optimization
---

# Training and Backpropagation - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach how neural networks learn — from measuring error with loss functions, through computing gradients with backpropagation, to building a complete training loop. The learner arrives with a working forward pass and leaves with a network that learns from data.

**Route Map**: See `/routes/training-and-backprop/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/training-and-backprop/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Define and compute loss functions (MSE, cross-entropy)
- Trace backpropagation step by step through a small network
- Implement `backward()` from scratch using the chain rule
- Build a complete training loop with batching and learning rate
- Train a neural network to classify handwritten digits (MNIST)

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/training-and-backprop/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has completed:
- **Neural Network Foundations** — specifically the forward pass, Layer/Network class, and the role of activation functions
- **Calculus for ML** — specifically the chain rule, partial derivatives, and the concept of gradient descent
- **Linear Algebra Essentials** — matrix operations, transpose, matrix-vector products

**Quick prerequisite check:**
1. "Can you describe what happens during a forward pass through a 2-layer network?" (Should get: input goes through W1 @ x + b1, activation, then W2 @ h + b2, activation — each layer transforms the input)
2. "What's the chain rule?" (Should get: the derivative of f(g(x)) is f'(g(x)) * g'(x) — you multiply the derivatives along the chain)
3. "What does gradient descent do with the gradient?" (Should get: it moves parameters in the opposite direction of the gradient to decrease the function value)

**If prerequisites are missing**: Suggest they work through the prerequisite routes first. The chain rule is not optional here — backpropagation IS the chain rule. If they can't compute d/dx of f(g(x)), they need [Calculus for ML](/routes/calculus-for-ml/map.md) first. If they don't have a working Network class, they need [Neural Network Foundations](/routes/neural-network-foundations/map.md).

### Audience Context
The target learner is a backend developer who has completed the three prerequisite math/ML routes. They have a Network class with Layer objects that can compute a forward pass. They understand matrices as transformations and derivatives as rates of change.

Use this to your advantage:
- Request/response cycle → forward/backward pass (data flows forward, gradients flow backward)
- Error logging/monitoring → loss functions and training curves
- Retry loops with exponential backoff → gradient descent with learning rate (both iteratively improve toward a goal, with step size control)
- Batch processing / chunked API calls → mini-batch training (process data in chunks for efficiency)
- Unit tests → loss functions (both tell you how wrong something is)

The key transition in this route: the learner goes from a network that computes arbitrary (wrong) outputs to one that actually learns from data. This is the moment neural networks stop being a curiosity and become useful.

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
- Include plausible distractors based on common misconceptions
- Example: "If the loss is decreasing on training data but increasing on test data, what's happening? A) The learning rate is too high B) Overfitting C) Underfitting D) The loss function is wrong"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess whether they understand the mechanics, not just the vocabulary
- Example: "In your own words, why do we compute gradients backward through the network instead of forward?"

**Tracing Questions:**
- Give concrete numbers and ask the learner to trace backprop by hand
- These are the most important assessments in this route
- Example: "Given output 0.7, target 1.0, and sigmoid activation, what's the gradient of the loss with respect to the pre-activation value z?"

**Prediction Questions:**
- Show a training scenario and ask what will happen
- Builds intuition about training dynamics
- Example: "I set the learning rate to 100. What do you think will happen to the loss?"

---

## Teaching Flow

### Introduction

**What to Cover:**
- The learner has a network that computes outputs — but those outputs are nonsense because the weights are random
- The question: how does a network learn the right weights?
- Three-step answer: (1) measure how wrong it is (loss), (2) figure out which direction to adjust each weight (backprop), (3) repeat (training loop)
- By the end of this route, their Network class will learn from data

**Opening Questions to Assess Level:**
1. "Your Network class can compute a forward pass. What happens if you run it with random weights on real data?"
2. "Do you have any intuition for how gradient descent might apply to neural networks?"
3. "Have you heard the term 'backpropagation' before? What's your mental model of it?"

**Adapt based on responses:**
- If they understand gradient descent well: Move through loss functions quickly, spend more time on the chain rule mechanics of backprop
- If they're math-cautious: Lead with code. Show the numpy first, explain the calculus second
- If they've heard of backprop and think it's complex/special: Immediately ground it: "It's literally just the chain rule from calculus, applied systematically"
- If they're excited about training on real data: Use MNIST as motivation throughout — "every concept we learn today gets us closer to recognizing handwritten digits"

**Good opening framing:**
"You have a network that can compute outputs, but the outputs are garbage because the weights are random. Today you're going to make it learn. The whole process boils down to three questions: How wrong is it? (loss functions) Which direction should each weight move? (backpropagation) How do we repeat this efficiently? (training loops) By the end, you'll train a network to read handwritten digits."

---

### Setup Verification

**Check numpy:**
```bash
python -c "import numpy as np; print(np.__version__)"
```

**Check matplotlib (needed for training curves):**
```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

**If not installed:**
```bash
pip install numpy matplotlib
```

**Check that they have a Network class:**
The learner should have a working `Network` class from the neural-network-foundations route. If not, they'll need at minimum:
```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Layer:
    def __init__(self, weights, bias, activation=relu):
        self.weights = np.array(weights, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.activation = activation

    def forward(self, x):
        z = self.weights @ x + self.bias
        return self.activation(z)

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

**Check visualization scripts:**
Verify the scripts exist at `/tools/ml-visualizations/`:
- `loss_surfaces.py` — Loss surface visualization
- `backprop_trace.py` — Step-by-step backprop tracer

**Notation Reminder:**
"We'll build on notation from previous routes. Quick refresher: `@` is matrix multiplication, `W` is a weight matrix, `b` is a bias vector, `z` is the pre-activation value (linear output), `a` is the post-activation value. New today: `L` for loss, `y_hat` (ŷ) for predictions, `y` for targets, `∂L/∂W` for the gradient of the loss with respect to a weight matrix."

---

### Section 1: Loss Functions

**Core Concept to Teach:**
A loss function measures how wrong the network's predictions are. It takes the network's output and the true answer and produces a single number — higher means worse. The goal of training is to minimize this number by adjusting the weights.

**How to Explain:**
1. Start with the analogy: "A loss function is like a unit test that returns a score instead of pass/fail. Zero means perfect; larger numbers mean more wrong."
2. Start with the simplest case: one prediction, one target
3. Build up to MSE (for regression) and cross-entropy (for classification)
4. Show what the loss surface looks like

**Mean Squared Error (MSE):**

"MSE is the most intuitive loss function. It measures the average squared distance between predictions and targets."

```python
import numpy as np

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# One prediction vs target
prediction = np.array([0.7])
target = np.array([1.0])
loss = mse_loss(prediction, target)
print(f"Prediction: {prediction[0]}, Target: {target[0]}, MSE Loss: {loss}")
# Loss = (0.7 - 1.0)^2 = 0.09

# Multiple predictions
predictions = np.array([0.7, 0.3, 0.9])
targets = np.array([1.0, 0.0, 1.0])
loss = mse_loss(predictions, targets)
print(f"Predictions: {predictions}, Targets: {targets}, MSE Loss: {loss:.4f}")
# Loss = ((0.7-1)^2 + (0.3-0)^2 + (0.9-1)^2) / 3 = (0.09 + 0.09 + 0.01) / 3 = 0.0633
```

"Walk through this: each prediction has an error (prediction - target). We square it so that errors in either direction are positive and large errors are penalized more. We average across all samples. Perfect predictions give a loss of 0."

**Why square the errors?**
"Two reasons: (1) it makes all errors positive (an error of -0.3 is just as bad as +0.3), and (2) it penalizes big errors more than small ones — an error of 2.0 costs 4x as much as an error of 1.0. That's usually what we want."

**Cross-Entropy Loss:**

"For classification, we use cross-entropy instead of MSE. Cross-entropy measures how surprised the model is by the correct answer."

```python
def cross_entropy_loss(predictions, targets):
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

# Binary classification: model says 0.9 for a true positive
pred = np.array([0.9])
target = np.array([1.0])
loss = cross_entropy_loss(pred, target)
print(f"Confident and correct — Loss: {loss:.4f}")  # Low loss

# Model says 0.1 for a true positive — very wrong
pred = np.array([0.1])
target = np.array([1.0])
loss = cross_entropy_loss(pred, target)
print(f"Confident and WRONG — Loss: {loss:.4f}")    # High loss

# Model says 0.5 — uncertain
pred = np.array([0.5])
target = np.array([1.0])
loss = cross_entropy_loss(pred, target)
print(f"Uncertain — Loss: {loss:.4f}")               # Medium loss
```

"Cross-entropy uses logarithms. When the model is confident and correct (predicts 0.9 for a true positive), `-log(0.9) ≈ 0.105` — low loss. When the model is confident and WRONG (predicts 0.1 for a true positive), `-log(0.1) ≈ 2.303` — high loss. The logarithm makes being confidently wrong extremely expensive."

**When to Use Which:**
- MSE → regression (predicting continuous values like temperature, price)
- Cross-entropy → classification (predicting categories like digit 0-9, spam/not-spam)
- "We'll use cross-entropy for MNIST because it's a classification problem."

**Visualizing the Loss Surface:**

Reference `/tools/ml-visualizations/loss_surfaces.py` for an interactive version, or:

```python
import matplotlib.pyplot as plt

# For a simple case: one weight, one bias, one data point
# Show how loss changes as we vary the weight
w_values = np.linspace(-3, 3, 100)
losses = []

x_data = np.array([1.0])
y_target = np.array([0.5])

for w in w_values:
    prediction = sigmoid(w * x_data)
    loss = mse_loss(prediction, y_target)
    losses.append(loss)

plt.figure(figsize=(8, 5))
plt.plot(w_values, losses, linewidth=2)
plt.xlabel('Weight value')
plt.ylabel('Loss')
plt.title('Loss as a function of one weight')
plt.grid(True)
plt.show()
```

"This is the loss surface — loss plotted against weight values. Training is gradient descent: start somewhere, compute the slope, step downhill. Repeat until you reach a minimum."

**Common Misconceptions:**
- Misconception: "Lower loss always means a better model" → Clarify: "Lower loss on training data might mean overfitting. What matters is loss on data the model hasn't seen."
- Misconception: "MSE and cross-entropy are interchangeable" → Clarify: "MSE works poorly for classification because its gradients become very small when predictions are far from targets (sigmoid output near 0 or 1). Cross-entropy's gradients stay useful."
- Misconception: "The loss function is something the model learns" → Clarify: "You choose the loss function. It's a design decision, like choosing which metric to optimize."

**Verification Questions:**
1. "What does a loss of 0 mean?" (The predictions exactly match the targets — perfect predictions)
2. "Why do we square the errors in MSE instead of just taking the absolute value?" (Squaring penalizes large errors more and is differentiable everywhere — abs value has a kink at 0)
3. Multiple choice: "For a binary classifier, which loss function is more appropriate? A) MSE B) Cross-entropy C) Both work equally well D) Neither"
4. "If the model predicts 0.99 for a target of 1.0, is the cross-entropy loss high or low?" (Low — the model is confident and correct)

**Good answer indicators:**
- They can compute MSE by hand for simple examples
- They understand that cross-entropy penalizes confident wrong predictions harshly
- They answer B (cross-entropy for classification)
- They grasp that loss is a design choice, not something learned

**If they struggle:**
- Stay with MSE first — it's the most intuitive. "How far off was each prediction? Square those distances. Average them. Done."
- For cross-entropy, focus on the intuition: "How surprised is the model by the correct answer? Very surprised = high loss."
- Connect to things they know: "A loss function is like an automated code review score. Lower is better."

**Exercise 1.1:**
"Implement `mse_loss` and `cross_entropy_loss` from scratch. Test them with these predictions and targets:
- predictions = [0.8, 0.3, 0.6], targets = [1.0, 0.0, 1.0]
- Compute both losses and explain which one is higher and why."

**How to Guide Them:**
1. "For MSE: subtract, square, average. Three operations."
2. "For cross-entropy: use the formula with log. Don't forget to clip predictions to avoid log(0)."
3. If stuck on cross-entropy: "The formula is `-mean(y * log(ŷ) + (1-y) * log(1-ŷ))`"

**Solution:**
```python
import numpy as np

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def cross_entropy_loss(predictions, targets):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

predictions = np.array([0.8, 0.3, 0.6])
targets = np.array([1.0, 0.0, 1.0])

print(f"MSE Loss:           {mse_loss(predictions, targets):.4f}")
print(f"Cross-Entropy Loss: {cross_entropy_loss(predictions, targets):.4f}")
```

**After exercise, ask:**
- "Can you see how the loss gives us a single number to optimize?"
- "If we could magically adjust the weights to make this number go to zero, what would that mean?"
- "How might we figure out which DIRECTION to adjust each weight?" (This leads into backpropagation)

---

### Section 2: Backpropagation

**Core Concept to Teach:**
Backpropagation computes the gradient of the loss with respect to every weight in the network. It does this by applying the chain rule layer by layer, starting from the loss and working backward through the network. Each layer's gradient depends on the gradient flowing back from the layer above it.

**THIS IS THE KEY SECTION OF THE ENTIRE ROUTE.** Spend the most time here. The learner must be able to trace backprop by hand through a small network before moving on.

**How to Explain:**
1. Start with the simplest possible case: one neuron, one weight
2. Apply the chain rule to compute how the loss changes when that weight changes
3. Expand to a 2-layer network and trace every gradient by hand
4. Show the pattern: each layer receives a gradient from above, computes local gradients, passes gradients down

**Step 1 — One Neuron, One Weight:**

"Let's start with the absolute simplest case. One input, one weight, one output, MSE loss."

```python
import numpy as np

# Forward pass
x = 1.5          # input
w = 0.8          # weight
b = 0.1          # bias
z = w * x + b    # pre-activation: 0.8 * 1.5 + 0.1 = 1.3
a = sigmoid(z)   # activation: sigmoid(1.3) ≈ 0.786

# Loss
target = 1.0
loss = (a - target) ** 2  # MSE for one sample: (0.786 - 1.0)^2 ≈ 0.046

print(f"x = {x}")
print(f"z = w*x + b = {z}")
print(f"a = sigmoid(z) = {a:.4f}")
print(f"loss = (a - target)^2 = {loss:.4f}")
```

"Now the question: if we nudge w slightly, how does the loss change? That's ∂L/∂w — the gradient of the loss with respect to the weight. If we know this, we know which direction to adjust w."

**Applying the Chain Rule:**

"The chain rule says: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w. We compute each piece."

```python
# Chain rule: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w

# ∂L/∂a = 2 * (a - target)
dL_da = 2 * (a - target)
print(f"∂L/∂a = 2 * (a - target) = 2 * ({a:.4f} - {target}) = {dL_da:.4f}")

# ∂a/∂z = sigmoid(z) * (1 - sigmoid(z))    [derivative of sigmoid]
da_dz = a * (1 - a)
print(f"∂a/∂z = sigmoid(z) * (1 - sigmoid(z)) = {a:.4f} * {1-a:.4f} = {da_dz:.4f}")

# ∂z/∂w = x    [because z = w*x + b, so derivative w.r.t. w is x]
dz_dw = x
print(f"∂z/∂w = x = {dz_dw}")

# Chain them together
dL_dw = dL_da * da_dz * dz_dw
print(f"\n∂L/∂w = {dL_da:.4f} * {da_dz:.4f} * {dz_dw} = {dL_dw:.4f}")
```

"That's it. The chain rule decomposes the gradient into a product of local derivatives. Each piece is trivial to compute. The magic is in chaining them."

**Verify with Numerical Gradient:**

"We can check our work by nudging the weight slightly and seeing how the loss changes."

```python
# Numerical gradient: (L(w+epsilon) - L(w-epsilon)) / (2*epsilon)
epsilon = 1e-5

def compute_loss(w_val):
    z_val = w_val * x + b
    a_val = sigmoid(z_val)
    return (a_val - target) ** 2

numerical_grad = (compute_loss(w + epsilon) - compute_loss(w - epsilon)) / (2 * epsilon)
print(f"Analytical gradient: {dL_dw:.6f}")
print(f"Numerical gradient:  {numerical_grad:.6f}")
print(f"Match: {np.isclose(dL_dw, numerical_grad)}")
```

"Numerical gradients are slow (you need two forward passes per weight), but they're a sanity check. If your analytical gradient doesn't match the numerical one, you have a bug. Always verify."

**Step 2 — A Full 2-Layer Network (THE HAND-TRACE):**

"Now let's trace backprop through a real (tiny) network. 2 inputs, 2 hidden neurons, 1 output."

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Network: 2 -> 2 -> 1
# Layer 1: 2 inputs, 2 hidden neurons
W1 = np.array([[0.1, 0.3],
               [0.2, 0.4]])
b1 = np.array([0.01, 0.02])

# Layer 2: 2 hidden neurons, 1 output
W2 = np.array([[0.5, 0.6]])
b2 = np.array([0.03])

# Input and target
x = np.array([1.0, 0.5])
target = np.array([1.0])
```

**Forward Pass (trace every value):**

```python
# === FORWARD PASS ===

# Layer 1
z1 = W1 @ x + b1
# z1[0] = 0.1*1.0 + 0.3*0.5 + 0.01 = 0.1 + 0.15 + 0.01 = 0.26
# z1[1] = 0.2*1.0 + 0.4*0.5 + 0.02 = 0.2 + 0.20 + 0.02 = 0.42
a1 = sigmoid(z1)
# a1[0] = sigmoid(0.26) ≈ 0.5646
# a1[1] = sigmoid(0.42) ≈ 0.6034

# Layer 2
z2 = W2 @ a1 + b2
# z2[0] = 0.5*0.5646 + 0.6*0.6034 + 0.03 = 0.2823 + 0.3620 + 0.03 = 0.6743
a2 = sigmoid(z2)
# a2[0] = sigmoid(0.6743) ≈ 0.6624

# Loss (MSE)
loss = np.mean((a2 - target) ** 2)
# loss = (0.6624 - 1.0)^2 = 0.1140

print(f"z1 = {z1}")
print(f"a1 = {a1}")
print(f"z2 = {z2}")
print(f"a2 = {a2}")
print(f"loss = {loss:.4f}")
```

**Backward Pass (trace every gradient):**

"Now we go backwards. Start from the loss, work back through each layer."

```python
# === BACKWARD PASS ===

# Step 1: ∂L/∂a2 (gradient of loss w.r.t. output)
# L = (a2 - target)^2, so ∂L/∂a2 = 2*(a2 - target)
dL_da2 = 2 * (a2 - target)
print(f"∂L/∂a2 = {dL_da2}")

# Step 2: ∂L/∂z2 (gradient through sigmoid)
# ∂a2/∂z2 = a2 * (1 - a2)   [sigmoid derivative]
# ∂L/∂z2 = ∂L/∂a2 * ∂a2/∂z2
da2_dz2 = a2 * (1 - a2)
dL_dz2 = dL_da2 * da2_dz2
print(f"∂L/∂z2 = ∂L/∂a2 * sigmoid'(z2) = {dL_dz2}")

# Step 3: Gradients for Layer 2 weights and bias
# z2 = W2 @ a1 + b2
# ∂L/∂W2 = ∂L/∂z2 * a1^T   (outer product)
# ∂L/∂b2 = ∂L/∂z2
dL_dW2 = dL_dz2.reshape(-1, 1) @ a1.reshape(1, -1)
dL_db2 = dL_dz2
print(f"∂L/∂W2 = {dL_dW2}")
print(f"∂L/∂b2 = {dL_db2}")

# Step 4: ∂L/∂a1 (gradient flowing back to layer 1)
# z2 = W2 @ a1 + b2, so ∂z2/∂a1 = W2^T
# ∂L/∂a1 = W2^T @ ∂L/∂z2
dL_da1 = W2.T @ dL_dz2
print(f"∂L/∂a1 = {dL_da1}")

# Step 5: ∂L/∂z1 (gradient through layer 1's sigmoid)
da1_dz1 = a1 * (1 - a1)
dL_dz1 = dL_da1 * da1_dz1
print(f"∂L/∂z1 = {dL_dz1}")

# Step 6: Gradients for Layer 1 weights and bias
dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
dL_db1 = dL_dz1
print(f"∂L/∂W1 = {dL_dW1}")
print(f"∂L/∂b1 = {dL_db1}")
```

**Walk Through the Pattern:**

"Let's step back and see the pattern. For each layer, going backward:
1. Receive ∂L/∂a from the layer above (or from the loss if this is the output layer)
2. Compute ∂L/∂z = ∂L/∂a * activation_derivative(z) — multiply by the local activation gradient
3. Compute ∂L/∂W = ∂L/∂z * input^T — this is the gradient for the weights
4. Compute ∂L/∂b = ∂L/∂z — this is the gradient for the biases
5. Compute ∂L/∂input = W^T @ ∂L/∂z — this is the gradient to pass down to the previous layer

That's the entire algorithm. Steps 1-2 compute how much the loss changes through this layer's activation. Steps 3-4 tell you how to adjust this layer's parameters. Step 5 passes the gradient backward to the next layer."

**The Key Insight:**
"Each layer's gradient depends on the gradient flowing back from above. That's why it's called BACK-propagation — the error signal propagates backward through the network, layer by layer, carrying the chain rule with it."

**Verify Everything with Numerical Gradients:**

```python
def compute_loss_for_param(param_name, param_idx, delta):
    """Compute loss with one parameter nudged by delta."""
    W1_copy = W1.copy()
    b1_copy = b1.copy()
    W2_copy = W2.copy()
    b2_copy = b2.copy()

    if param_name == 'W1':
        W1_copy.flat[param_idx] += delta
    elif param_name == 'b1':
        b1_copy[param_idx] += delta
    elif param_name == 'W2':
        W2_copy.flat[param_idx] += delta
    elif param_name == 'b2':
        b2_copy[param_idx] += delta

    z1_ = W1_copy @ x + b1_copy
    a1_ = sigmoid(z1_)
    z2_ = W2_copy @ a1_ + b2_copy
    a2_ = sigmoid(z2_)
    return np.mean((a2_ - target) ** 2)

eps = 1e-5

# Check ∂L/∂W2
print("Verifying ∂L/∂W2:")
for i in range(W2.size):
    numerical = (compute_loss_for_param('W2', i, eps) - compute_loss_for_param('W2', i, -eps)) / (2*eps)
    analytical = dL_dW2.flat[i]
    print(f"  W2[{i}]: analytical={analytical:.6f}, numerical={numerical:.6f}, match={np.isclose(analytical, numerical)}")

# Check ∂L/∂W1
print("Verifying ∂L/∂W1:")
for i in range(W1.size):
    numerical = (compute_loss_for_param('W1', i, eps) - compute_loss_for_param('W1', i, -eps)) / (2*eps)
    analytical = dL_dW1.flat[i]
    print(f"  W1[{i}]: analytical={analytical:.6f}, numerical={numerical:.6f}, match={np.isclose(analytical, numerical)}")
```

"All analytical gradients should match numerical gradients to within a tiny tolerance. If they don't, you have a bug in the backward pass."

**Activation Function Derivatives:**

"We used sigmoid above. Here are the derivatives for the activation functions you know:"

```python
# Sigmoid derivative: sigmoid(z) * (1 - sigmoid(z))
def sigmoid_derivative(a):
    """Derivative of sigmoid, given the activated value a = sigmoid(z)."""
    return a * (1 - a)

# ReLU derivative: 1 if z > 0, 0 if z <= 0
def relu_derivative(z):
    """Derivative of ReLU, given the pre-activation value z."""
    return (z > 0).astype(float)
```

"ReLU's derivative is elegant: if the neuron was active (z > 0), the gradient passes straight through (multiplied by 1). If the neuron was dead (z <= 0), the gradient is blocked (multiplied by 0). It's a gradient gate."

Reference `/tools/ml-visualizations/backprop_trace.py` for a step-by-step animated trace.

**Common Misconceptions:**
- Misconception: "Backprop is some special algorithm separate from calculus" → Clarify: "It IS the chain rule. Nothing more. Applied systematically, layer by layer, backward through the network."
- Misconception: "Gradients tell you the right weight values" → Clarify: "Gradients tell you the DIRECTION to adjust — which way is downhill. Not how far to go (that's the learning rate) and not the destination."
- Misconception: "You need to store all intermediate values" → Clarify: "You do. The forward pass computes and stores z and a for each layer. The backward pass needs them. This is why neural networks are memory-hungry."
- Misconception: "Deeper networks have better gradients" → Clarify: "Deeper networks can have vanishing or exploding gradients — the gradient signal can shrink or grow exponentially as it flows backward through many layers."

**Verification Questions:**
1. "In the chain rule ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w, what does each term represent?" (How loss changes with output, how output changes through activation, how pre-activation changes with weight)
2. "Why is the gradient for the bias ∂L/∂b simply equal to ∂L/∂z?" (Because z = W @ x + b, so ∂z/∂b = 1)
3. "What role does W^T play in backpropagation?" (It distributes the gradient backward — it tells each input how much it contributed to each output's error)
4. Multiple choice: "What happens to the gradient at a ReLU neuron that had a negative input during the forward pass? A) The gradient doubles B) The gradient passes through unchanged C) The gradient becomes zero D) The gradient becomes negative"

**Good answer indicators:**
- They can decompose the chain rule into local derivatives
- They understand the bias gradient is just ∂L/∂z because ∂z/∂b = 1
- They see W^T as redistributing error backward
- They answer C (gradient becomes zero — dead ReLU blocks gradients)

**If they struggle:**
- Go back to the single-neuron case. "Forget the network. One input, one weight. z = w*x. L = (sigmoid(z) - target)^2. How does L change when w changes?"
- Draw the computation graph: x → [*w] → z → [sigmoid] → a → [loss] → L. "Each arrow has a derivative. Multiply them along the path."
- If matrix notation is confusing, trace element by element: "W2[0,0] * a1[0] + W2[0,1] * a1[1] + b2[0] = z2[0]. So ∂z2[0]/∂W2[0,0] = a1[0]."
- Numerical gradients are the ultimate crutch: "Don't trust the math? Nudge the weight, recompute the loss, and check."

**Exercise 2.1:**
"Trace backprop by hand through this 2→2→1 network with sigmoid activations:
- W1 = [[0.5, -0.3], [0.2, 0.8]], b1 = [0, 0]
- W2 = [[0.4, 0.6]], b2 = [0]
- Input: [1, 0], Target: [1]
Show every intermediate value (z1, a1, z2, a2, loss, and every gradient). Verify at least one weight gradient numerically."

**How to Guide Them:**
1. "Start with the forward pass: compute z1, a1, z2, a2, loss."
2. "Then backward: start with ∂L/∂a2 and work back."
3. "For each layer: multiply ∂L/∂a by the activation derivative to get ∂L/∂z. Then compute ∂L/∂W and ∂L/∂b."
4. "Pick any weight and verify with a numerical gradient."
5. If stuck: "What's ∂L/∂a2? It's 2*(a2 - target). Start there."

**Exercise 2.2:**
"Implement a `backward_layer` function that takes the gradient from above (∂L/∂a), the layer's stored z and input, and the weight matrix, and returns ∂L/∂W, ∂L/∂b, and ∂L/∂input."

**How to Guide Them:**
1. "What are the three things you need to compute?" (∂L/∂W, ∂L/∂b, ∂L/∂input)
2. "∂L/∂z = ∂L/∂a * activation_derivative. ∂L/∂W = outer product of ∂L/∂z and input. ∂L/∂b = ∂L/∂z. ∂L/∂input = W^T @ ∂L/∂z."

**Solution:**
```python
def backward_layer(dL_da, z, layer_input, W, activation_deriv):
    """
    Compute gradients for one layer.

    Args:
        dL_da: gradient of loss w.r.t. this layer's output (activated)
        z: pre-activation values from forward pass
        layer_input: input to this layer from forward pass
        W: weight matrix of this layer
        activation_deriv: derivative of activation, applied to z or a

    Returns:
        dL_dW: gradient for weights
        dL_db: gradient for biases
        dL_dinput: gradient to pass to previous layer
    """
    dL_dz = dL_da * activation_deriv
    dL_dW = dL_dz.reshape(-1, 1) @ layer_input.reshape(1, -1)
    dL_db = dL_dz
    dL_dinput = W.T @ dL_dz
    return dL_dW, dL_db, dL_dinput
```

**After exercises, ask:**
- "Can you see the pattern? Every layer does the same three things backward."
- "What would happen if the activation derivatives were all very small — close to zero?" (Hint: vanishing gradients)
- "Do you feel confident enough to trace backprop through a 3-layer network?" (If yes, they're ready to move on)

---

### Section 3: The Training Loop

**Core Concept to Teach:**
Training is the loop that makes the network learn. Each iteration: compute the forward pass, compute the loss, compute the gradients (backward pass), update the weights. Repeat thousands of times. Mini-batching and learning rate control how the learning happens.

**How to Explain:**
1. Show the full loop in pseudocode first
2. Implement it for real with the tiny network from Section 2
3. Discuss learning rate: too high = overshoot, too low = slow
4. Introduce mini-batches: why we don't use one sample or all samples

**The Training Loop in 20 Lines:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(W1, b1, W2, b2, X, Y, learning_rate=0.1, epochs=1000):
    """Train a 2-layer network with sigmoid activations and MSE loss."""
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for x, y in zip(X, Y):
            # Forward
            z1 = W1 @ x + b1
            a1 = sigmoid(z1)
            z2 = W2 @ a1 + b2
            a2 = sigmoid(z2)

            # Loss
            loss = np.mean((a2 - y) ** 2)
            total_loss += loss

            # Backward
            dL_da2 = 2 * (a2 - y)
            dL_dz2 = dL_da2 * a2 * (1 - a2)
            dL_dW2 = dL_dz2.reshape(-1, 1) @ a1.reshape(1, -1)
            dL_db2 = dL_dz2
            dL_da1 = W2.T @ dL_dz2
            dL_dz1 = dL_da1 * a1 * (1 - a1)
            dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
            dL_db1 = dL_dz1

            # Update (gradient descent)
            W1 -= learning_rate * dL_dW1
            W2 -= learning_rate * dL_dW2
            b1 -= learning_rate * dL_db1
            b2 -= learning_rate * dL_db2

        losses.append(total_loss / len(X))

    return W1, W2, b1, b2, losses
```

"That's a complete training loop. Forward, loss, backward, update. Four steps, repeated thousands of times. Every neural network in the world trains this way — PyTorch, TensorFlow, everything. They just add optimization tricks on top."

**Train it on XOR:**

```python
# XOR dataset
X = [np.array([0, 0]), np.array([0, 1]),
     np.array([1, 0]), np.array([1, 1])]
Y = [np.array([0]), np.array([1]),
     np.array([1]), np.array([0])]

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros(2)
W2 = np.random.randn(1, 2) * 0.5
b2 = np.zeros(1)

# Train
W1, W2, b1, b2, losses = train(W1, b1, W2, b2, X, Y,
                                 learning_rate=1.0, epochs=5000)

# Test
print("After training:")
for x, y in zip(X, Y):
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    print(f"  Input: {x}, Target: {y[0]}, Prediction: {a2[0]:.4f}")

# Plot training curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve — XOR')
plt.grid(True)
plt.show()
```

"Look at the training curve. The loss starts high (random weights produce garbage) and decreases over epochs as the weights adjust. This is the network learning."

**Learning Rate — The Step Size:**

```python
# Show the effect of different learning rates
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, lr in enumerate([0.01, 1.0, 100.0]):
    np.random.seed(42)
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.zeros(2)
    W2 = np.random.randn(1, 2) * 0.5
    b2 = np.zeros(1)

    _, _, _, _, losses = train(W1, b1, W2, b2, X, Y,
                                learning_rate=lr, epochs=5000)

    axes[idx].plot(losses)
    axes[idx].set_title(f'LR = {lr}')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].grid(True)

plt.tight_layout()
plt.show()
```

"Three scenarios:
- **Learning rate too small (0.01)**: the loss decreases, but painfully slowly. You'd need 100x more epochs.
- **Learning rate just right (1.0)**: the loss drops steadily to near zero. The network learns efficiently.
- **Learning rate too large (100.0)**: the loss oscillates wildly or explodes. The gradient steps overshoot the minimum."

**Mini-Batching:**

"In the XOR example, we update weights after every single sample. That works for 4 data points, but MNIST has 60,000. Three strategies:

1. **Stochastic (batch size = 1)**: update after each sample. Noisy but fast per step.
2. **Full batch (batch size = all)**: average gradients over all samples, then update. Stable but slow per step and memory-heavy.
3. **Mini-batch (batch size = 32, 64, 128)**: compromise. Average gradients over a small batch, then update.

Mini-batching is the standard. It gives you the noise that helps escape local minima, plus the stability of averaging, plus manageable memory usage."

```python
def train_batched(network, X, Y, learning_rate=0.01, epochs=100, batch_size=32):
    """Training loop with mini-batching."""
    losses = []
    n = len(X)

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        epoch_loss = 0
        num_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # Accumulate gradients over the batch
            batch_grad_W = [np.zeros_like(layer.weights) for layer in network.layers]
            batch_grad_b = [np.zeros_like(layer.bias) for layer in network.layers]
            batch_loss = 0

            for x, y in zip(X_batch, Y_batch):
                # Forward and backward for each sample
                # (accumulate gradients)
                pass  # The learner will implement this

            # Average gradients and update
            actual_batch_size = end - start
            for i, layer in enumerate(network.layers):
                layer.weights -= learning_rate * batch_grad_W[i] / actual_batch_size
                layer.bias -= learning_rate * batch_grad_b[i] / actual_batch_size

            epoch_loss += batch_loss / actual_batch_size
            num_batches += 1

        losses.append(epoch_loss / num_batches)

    return losses
```

**Terminology Recap:**
- **Epoch**: one pass through the entire training dataset
- **Batch**: a subset of training samples processed together
- **Iteration/Step**: one weight update (one batch processed)
- "If you have 1000 samples and batch size 100, one epoch = 10 iterations."

**Common Misconceptions:**
- Misconception: "More epochs is always better" → Clarify: "After a point, more epochs means the network memorizes the training data (overfitting) instead of learning general patterns."
- Misconception: "The loss should decrease every single step" → Clarify: "With mini-batches, the loss is noisy — it bounces around. The overall trend should be downward, but individual steps may go up."
- Misconception: "Gradient descent finds the global minimum" → Clarify: "For non-convex loss surfaces (which neural networks have), gradient descent finds a local minimum. In practice, local minima in high-dimensional spaces tend to be good enough."
- Misconception: "The learning rate should stay constant" → Clarify: "In practice, learning rate schedules (starting high, decreasing over time) often work better. But a fixed rate is fine to start with."

**Verification Questions:**
1. "What are the four steps of one training iteration?" (Forward pass, compute loss, backward pass, update weights)
2. "If the learning rate is too large, what happens?" (The loss oscillates or explodes — gradient steps overshoot the minimum)
3. "Why do we shuffle the data before each epoch?" (To prevent the network from learning the order of samples rather than the content)
4. Multiple choice: "With 10,000 training samples and a batch size of 100, how many weight updates happen per epoch? A) 1 B) 100 C) 1000 D) 10000"

**Good answer indicators:**
- They can list the four steps in order
- They understand learning rate as step size — too big overshoots, too small is slow
- They understand shuffling prevents order-dependent learning
- They answer B (100 batches of 100 = 100 updates per epoch)

**If they struggle:**
- Reduce to the simplest possible training loop: one weight, one data point, update by hand
- "w_new = w_old - learning_rate * gradient. That's it. If the gradient is positive, w decreases. If negative, w increases. You're walking downhill."
- Show the XOR training curve and point at it: "Look — it's working. The loss goes down. That means the network is getting less wrong over time."

**Exercise 3.1:**
"Train the 2→2→1 network on XOR. Plot the training curve. Then try three different learning rates (0.1, 1.0, 10.0) and compare the curves."

**How to Guide Them:**
1. "Use the `train` function above (or write your own)."
2. "For each learning rate, start with the same initial weights (use the same seed)."
3. "Plot all three training curves on the same plot."

**Exercise 3.2:**
"Add `backward()` and `train()` methods to your Network class from the previous route. The backward method should compute and store gradients for each layer. The train method should run the full loop."

**How to Guide Them:**
1. "Start with backward(). Each layer needs to store z and its input during the forward pass."
2. "Modify `Layer.forward()` to save `self.z` and `self.input`."
3. "Add `Layer.backward(dL_da)` that computes gradients and returns dL_dinput."
4. "The Network.backward() calls each layer's backward in reverse order."

**Solution sketch:**
```python
class Layer:
    def __init__(self, weights, bias, activation='sigmoid'):
        self.weights = np.array(weights, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.activation = activation
        # Stored during forward pass for backprop
        self.z = None
        self.input = None
        # Gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x.copy()
        self.z = self.weights @ x + self.bias
        if self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'relu':
            self.output = relu(self.z)
        return self.output

    def backward(self, dL_da):
        if self.activation == 'sigmoid':
            da_dz = self.output * (1 - self.output)
        elif self.activation == 'relu':
            da_dz = (self.z > 0).astype(float)

        dL_dz = dL_da * da_dz
        self.dW = dL_dz.reshape(-1, 1) @ self.input.reshape(1, -1)
        self.db = dL_dz
        dL_dinput = self.weights.T @ dL_dz
        return dL_dinput
```

**After exercises, ask:**
- "You just trained a neural network from scratch. How does it feel?"
- "Can you trace the whole flow: forward → loss → backward → update?"
- "What determines how fast the network learns?" (Learning rate and the gradient magnitudes)

---

### Section 4: Training in Practice

**Core Concept to Teach:**
Real training isn't just "run the loop and wait." You need to monitor the loss curve, diagnose problems (overfitting, underfitting, divergence), and apply fixes. This section builds practical intuition for debugging training runs.

**How to Explain:**
1. Show what a healthy training curve looks like
2. Show what overfitting looks like (training loss drops, test loss rises)
3. Show what underfitting looks like (loss stays high)
4. Discuss common training failures and how to diagnose them

**Monitoring the Training Curve:**

"The training curve is your primary debugging tool. Plot loss vs. epoch. Here's what to look for:"

```python
import matplotlib.pyplot as plt

epochs = np.arange(100)

# Healthy training
healthy = 2.0 * np.exp(-0.05 * epochs) + 0.1 + np.random.randn(100) * 0.03

# Overfitting
train_loss = 2.0 * np.exp(-0.08 * epochs) + 0.02
test_loss = 2.0 * np.exp(-0.06 * epochs[:30]).tolist() + (0.3 + 0.005 * epochs[30:]).tolist()
test_loss = np.array(test_loss) + np.random.randn(100) * 0.02

# Underfitting
underfit = 1.8 - 0.3 * (1 - np.exp(-0.01 * epochs)) + np.random.randn(100) * 0.05

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, healthy, label='Training loss')
axes[0].set_title('Healthy Training')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

axes[1].plot(epochs, train_loss, label='Training loss')
axes[1].plot(epochs, test_loss, label='Test loss')
axes[1].set_title('Overfitting')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(epochs, underfit, label='Training loss')
axes[2].set_title('Underfitting')
axes[2].grid(True)

for ax in axes:
    ax.set_xlabel('Epoch')

plt.tight_layout()
plt.show()
```

**Diagnosing Problems:**

"Your training curve tells you what's wrong:

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss decreases steadily | Healthy | Keep going |
| Training loss low, test loss high | Overfitting | More data, regularization, simpler model |
| Loss stays high | Underfitting | Bigger model, more epochs, check data |
| Loss oscillates wildly | Learning rate too high | Decrease learning rate |
| Loss is NaN | Exploding gradients or bad data | Check data, decrease learning rate, gradient clipping |
| Loss decreases then plateaus | Possible local minimum | Try learning rate schedule, different initialization |"

**Overfitting — Seeing It Happen:**

"Overfitting is when the network memorizes training data instead of learning general patterns. It's like a student who memorizes exam answers but can't solve variations."

```python
# Create a simple dataset where overfitting is easy to observe
np.random.seed(42)
# 10 training points (easy to memorize)
X_train = np.random.randn(10, 2)
Y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float).reshape(-1, 1)

# 100 test points (harder to do well on without generalization)
X_test = np.random.randn(100, 2)
Y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(float).reshape(-1, 1)

# Train a network that's too big for this simple task
# Track both training and test loss
# ...
# (The learner implements this — the key observation is train loss → 0 while test loss rises)
```

**Simple Regularization:**

"When overfitting is a problem, regularization penalizes complexity. The simplest form is L2 regularization (weight decay): add a term to the loss that penalizes large weights."

```python
def loss_with_regularization(predictions, targets, weights_list, lambda_reg=0.01):
    """MSE loss + L2 regularization."""
    mse = np.mean((predictions - targets) ** 2)
    l2_penalty = sum(np.sum(W ** 2) for W in weights_list)
    return mse + lambda_reg * l2_penalty
```

"L2 regularization pushes weights toward zero, which prevents the network from relying too heavily on any single feature. It's a blunt instrument but effective."

**Debugging Checklist:**

"When your training run isn't working:
1. **Check the data**: Are inputs and targets correctly aligned? Are values in reasonable ranges? Any NaN or inf?
2. **Check the forward pass**: Does the output shape match the target shape? For one sample, does the loss make sense?
3. **Check gradients**: Do numerical gradients match analytical gradients? This catches backward pass bugs.
4. **Check the learning rate**: Start with a small value (0.001) and increase. If loss immediately explodes, it's too high.
5. **Check initialization**: Very large initial weights can cause sigmoid to saturate (all outputs near 0 or 1), killing gradients."

**Common Misconceptions:**
- Misconception: "If the training loss is low, the model is good" → Clarify: "Only if the test loss is also low. Low training loss with high test loss is overfitting — the model memorized instead of learned."
- Misconception: "Regularization always helps" → Clarify: "Too much regularization causes underfitting — the model becomes too constrained to learn the actual pattern."
- Misconception: "NaN loss means the code is broken" → Clarify: "Often it's a numerical issue — log(0), division by zero, or weights that exploded to infinity. Clipping and smaller learning rates usually fix it."

**Verification Questions:**
1. "You're training a network. Training loss is 0.01, test loss is 2.5. What's happening?" (Overfitting — the network memorized training data but doesn't generalize)
2. "Your loss is NaN after 50 epochs. What are three things you'd check?" (Data for NaN/inf, learning rate too high, log of zero in loss function)
3. "What does L2 regularization do to the weights during training?" (It adds a penalty for large weights, pushing them toward zero)

**If they struggle:**
- Focus on the training curve as a diagnostic tool: "This single plot tells you almost everything about how training is going."
- Use the overfitting analogy: "The network is cramming for the test. It memorized the answers to the practice problems, but it can't handle new questions."

**Exercise 4.1:**
"Train a 2→8→1 network on the 10-point dataset above. Track both training and test loss. Train for 2000 epochs and plot both curves. Identify the epoch where overfitting starts."

**How to Guide Them:**
1. "You'll need to compute the loss on the test set after each epoch without updating weights."
2. "The overfitting point is where test loss starts increasing while training loss keeps decreasing."
3. "Try adding L2 regularization and see how it changes the curves."

**Exercise 4.2:**
"Intentionally break training in three ways and observe the effects:
1. Set the learning rate to 1000. What happens?
2. Set all initial weights to 0. What happens?
3. Feed the network targets that don't match the inputs (random labels). What happens?"

**How to Guide Them:**
1. "For each experiment, plot the training curve."
2. "For learning rate 1000: the loss should explode."
3. "For zero weights: all neurons compute the same thing (symmetry problem)."
4. "For random labels: the network tries to memorize noise — training loss goes down but generalization is impossible."

**After exercises, ask:**
- "What's the first thing you'd look at if your training run isn't working?" (The training curve)
- "How would you decide between 'model is too small' and 'model is too big'?" (Underfitting = too small, overfitting = too big)

---

## Practice Project

**Project Introduction:**
"Time to put it all together. You're going to add `backward()` and `train()` to your Network class and train it on real data — a subset of MNIST handwritten digits."

**Requirements:**
Present these one at a time:
1. "Extend your Layer class to save intermediate values during forward and compute gradients in backward"
2. "Add a `backward()` method to Network that computes gradients for all layers"
3. "Add a `train()` method with mini-batching, configurable learning rate, and loss tracking"
4. "Load MNIST digits 0-4 (a 5-class classification problem)"
5. "Train the network to achieve >80% accuracy on the test set"
6. "Plot the training curve (loss vs. epoch) and report final accuracy"

**MNIST Data Setup:**

```python
# Download MNIST subset
# Reference: /ascents/neural-net-from-scratch/download_data.py
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, Y = mnist.data, mnist.target.astype(int)

# Filter to digits 0-4
mask = Y < 5
X = X[mask]
Y = Y[mask]

# Normalize pixel values to [0, 1]
X = X / 255.0

# One-hot encode targets
Y_onehot = np.zeros((len(Y), 5))
Y_onehot[np.arange(len(Y)), Y] = 1

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y_onehot[:split], Y_onehot[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input dimension: {X_train.shape[1]}")  # 784 (28x28 pixels)
print(f"Output classes: {Y_train.shape[1]}")    # 5 (digits 0-4)
```

"If sklearn is not available, use the download script at `/ascents/neural-net-from-scratch/download_data.py`."

**Architecture Recommendation:**
"A good starting architecture for MNIST: 784 → 128 (ReLU) → 64 (ReLU) → 5 (softmax or sigmoid). This gives the network enough capacity to learn the digit patterns without being so large that training is slow."

**Softmax for Multi-Class Output:**

"For multi-class classification, we replace sigmoid with softmax in the output layer. Softmax converts a vector of raw scores into probabilities that sum to 1."

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

# Example
z = np.array([2.0, 1.0, 0.5, -1.0, 0.1])
probs = softmax(z)
print(f"Scores: {z}")
print(f"Probabilities: {probs}")
print(f"Sum: {np.sum(probs)}")  # 1.0
```

"For the backward pass, the derivative of cross-entropy loss combined with softmax simplifies to: ∂L/∂z = predictions - targets. That's it — much simpler than computing the softmax derivative separately."

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work, offer to answer questions. Check in when they have training running.
2. **If they want guidance**: Build it step by step. Start with extending Layer, then Network, then data loading, then training.
3. **If they're unsure**: Start with requirement 1 (extending Layer) and check in.

**Checkpoints During Project:**
- After extending Layer: "Run one forward pass and one backward pass on a single MNIST sample. Do the gradient shapes look right?"
- After Network.backward(): "Verify gradients numerically for at least two weights."
- After train(): "Train on XOR first — you already know it should work. Then switch to MNIST."
- During MNIST training: "After 1 epoch, the loss should have decreased from its initial value. If not, check the learning rate."
- After training: "Plot the training curve. What accuracy did you get? If below 80%, try more epochs or adjust the learning rate."

**Code Review Approach:**
When reviewing their work:
1. Check gradient computation: "Are you storing z and input during forward? Are the shapes right in backward?"
2. Check the training loop: "Are you shuffling data each epoch? Averaging gradients over the batch?"
3. Check initialization: "Large initial weights can kill training. Try `W = np.random.randn(...) * 0.01`"
4. Ask about design: "How does your Network.backward() handle the chain of layers?"

**If They Get Stuck:**
- On extending Layer: "You need to save two things in forward: `self.z = W @ x + b` (before activation) and `self.input = x`. The backward method uses both."
- On Network backward: "Call each layer's backward in reverse order. The first call passes ∂L/∂output. Each layer returns the gradient for the layer below."
- On MNIST accuracy: "If stuck below 80%: try learning rate 0.1, batch size 32, 20 epochs, and Xavier initialization (`W * sqrt(2/fan_in)`)."
- On softmax gradient: "The combined cross-entropy + softmax gradient is just `predictions - targets`. This is a well-known simplification."

**Expected Results:**
- Training loss should decrease from ~1.6 (random guessing over 5 classes) to ~0.1-0.3
- Test accuracy should reach 80-95% depending on architecture and hyperparameters
- Training 20 epochs on ~25,000 samples should take a few minutes with pure numpy

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- Loss functions measure how wrong predictions are: MSE for regression, cross-entropy for classification
- Backpropagation is the chain rule applied backward through the network, computing ∂L/∂W for every weight
- Each layer in backward: receive gradient from above, multiply by activation derivative, compute weight/bias gradients, pass gradient below
- The training loop: forward → loss → backward → update, repeated with mini-batches over many epochs
- Practical training requires monitoring loss curves and diagnosing overfitting, underfitting, and divergence

**Ask them to explain one concept:**
"Can you trace backprop through a 2-layer network and explain what each gradient represents?"
(This is the core skill of this route. If they can do it, they truly understand how neural networks learn.)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel about training neural networks from scratch?"

**Respond based on answer:**
- 1-4: "Backprop is genuinely the hardest concept in this learning path. The hand-tracing exercise is the key — try it again with different numbers. Once you can trace it automatically, everything else clicks."
- 5-7: "You've got the mechanics down. The next step is intuition — which comes from training different architectures on different data and watching what happens. Experiment with the MNIST project."
- 8-10: "You can train a neural network from scratch — that puts you ahead of most people who use ML frameworks. You're ready for the next route, where these building blocks compose into language models."

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice more: Try different MNIST architectures — deeper, wider, different activations. Can you beat your initial accuracy?"
- "To go deeper: Implement momentum, learning rate decay, or dropout. These are the optimization tricks that make training work well in practice."
- "When you're ready: [LLM Foundations](/routes/llm-foundations/map.md) shows how these building blocks compose into language models — the architecture behind GPT"
- "For the complete picture: [Neural Net from Scratch](/ascents/neural-net-from-scratch/ascent.md) is the ascent project that ties all routes together"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there a part of backprop that still feels shaky?"
"Anything about the training loop that doesn't make sense?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Can't trace the chain rule through even one layer
- Confused about which gradient goes where
- Losing track of shapes during backward pass
- Getting frustrated with the math

**Strategies:**
- Reduce to one neuron, one weight. "z = w*x, L = (z - target)^2, ∂L/∂w = 2*(z - target) * x. That's the entire concept."
- Use numerical gradients as a crutch: "Don't trust the math? Nudge the weight, compute the loss twice, subtract and divide. That's the gradient."
- Draw the computation graph and label each arrow with its derivative
- Let them use code to verify every step before doing anything by hand
- If shapes confuse them: "∂L/∂W has the same shape as W. Always. If your gradient has a different shape, something is wrong."
- Stay concrete: use specific numbers, not variables, until the pattern is clear
- Break the backward pass into a checklist they can follow mechanically

### If Learner is Excelling

**Signs:**
- Completes hand-tracing quickly and correctly
- Asks about optimization algorithms (Adam, RMSProp)
- Experiments beyond what's asked
- Connects concepts to frameworks they've seen

**Strategies:**
- Move faster, skip basic exercises, go straight to MNIST
- Discuss optimization beyond vanilla gradient descent: momentum, Adam, learning rate schedules
- Introduce batch normalization concept: normalizing activations between layers
- Discuss computational graphs and automatic differentiation (how PyTorch/JAX work under the hood)
- Challenge: "Implement backprop for a convolutional layer" or "Add dropout to your network"
- Preview the next route: "In LLM foundations, the attention mechanism is just a specific layer architecture. You now know how to train any layer."

### If Learner Seems Disengaged

**Signs:**
- Short responses, not asking questions
- Rushing through exercises
- Not connecting with the material

**Strategies:**
- Jump to the MNIST project: "Let's skip ahead and train on real data. We'll come back to the theory as needed."
- Show the result first: "Here's a network recognizing handwritten digits. Now let's understand how it learned to do that."
- Make it tangible: draw a digit, photograph it, feed it to the network
- Connect to their goals: "What made you want to learn ML? Let's frame everything in terms of that."

### Different Learning Styles

**Visual learners:**
- Use the backprop_trace.py visualization extensively
- Draw computation graphs for every example
- Plot gradients flowing through the network
- Show loss surfaces and gradient descent paths

**Hands-on learners:**
- Code first, explain after
- "Implement the backward pass, run it, then let's understand why it works"
- Exercise-driven: less lecture, more building and debugging

**Conceptual learners:**
- Spend more time on why: why the chain rule, why backward, why mini-batches
- Discuss the information-theoretic interpretation of cross-entropy
- Explain the geometry of gradient descent in high-dimensional weight space

**Example-driven learners:**
- Trace through specific numbers first, generalize after
- Use the XOR example as the running thread — they already know it from the previous route
- Every concept gets a concrete computation before any formula

---

## Troubleshooting Common Issues

### Shape Mismatches in Backward Pass

**"shapes not aligned" during gradient computation:**
```python
# Check: ∂L/∂W should have the same shape as W
print(f"W shape: {W.shape}, dL_dW shape: {dL_dW.shape}")

# The outer product: dL_dz is (output,), input is (input,)
# dL_dW = dL_dz.reshape(-1, 1) @ input.reshape(1, -1)
# Result shape: (output, input) — same as W
```

"The most common backward pass bug is transposing shapes. Remember: ∂L/∂W = ∂L/∂z (column) @ input (row). The outer product naturally produces the right shape."

### Numerical Gradient Mismatch

"If analytical and numerical gradients don't match:
1. Check you're using the right activation derivative (sigmoid vs ReLU)
2. Check the chain rule order — are you multiplying in the right sequence?
3. Check that you're using the saved values from the forward pass, not recomputing them
4. Use a small epsilon (1e-5) for numerical gradients"

### Loss Not Decreasing

"If the loss stays flat:
1. **Learning rate too small**: try 10x larger
2. **All-zero weights**: breaks symmetry — use random initialization
3. **Sigmoid saturation**: initial weights too large cause sigmoid outputs near 0 or 1 where gradients are tiny. Use smaller initial weights.
4. **Dead ReLU**: if too many neurons have negative inputs, gradients are zero. Try smaller weights or add small positive biases."

### Loss is NaN

"Almost always one of:
1. **log(0)** in cross-entropy: clip predictions to [epsilon, 1-epsilon]
2. **Exploding gradients**: decrease learning rate or add gradient clipping
3. **Bad data**: check for NaN or inf in the input data
4. **Numerical overflow in exp()**: in softmax, subtract the max value before exponentiating"

### Concept-Specific Confusion

**If confused about which gradient flows where:**
- "Gradients flow backward, opposite to data. Data flows input → output. Gradients flow loss → output → hidden → input."
- Draw arrows on the computation graph: forward arrows for data, backward arrows for gradients

**If confused about why W^T appears in backprop:**
- "During forward, W mixes inputs into outputs. During backward, W^T distributes the output gradients back to the inputs. It's the transpose because you're going in the opposite direction."

**If confused about mini-batching:**
- "Instead of updating weights after every single sample (noisy), or after all samples (slow), update after every 32 samples (balanced). That's all mini-batching is."

---

## Teaching Notes

**Key Emphasis Points:**
- Backpropagation is the chain rule. If the learner walks away with one thing, it should be this. It's not a special algorithm, not a black box — it's the chain rule from calculus, applied systematically backward through the network.
- The hand-trace in Section 2 is the single most important exercise in this route. Don't skip it. Don't rush it. The learner must be able to trace gradients through a 2-layer network before moving on.
- The training loop is mechanically simple: forward, loss, backward, update. The difficulty is in getting the backward pass right and tuning hyperparameters.
- Numerical gradient checking is the learner's safety net. Teach them to always verify new backward implementations against numerical gradients.

**Pacing Guidance:**
- Don't rush Section 1 (loss functions). Loss is the starting point for all gradients — if they don't understand what they're minimizing, backprop won't make sense.
- Section 2 (backpropagation) is where the learner will spend the most time. Budget at least 40% of the session here. The one-neuron example should be fast; the full hand-trace will take time.
- Section 3 (training loop) should feel like a payoff — "we can finally make it learn!" If the learner is engaged, let them experiment with different learning rates and architectures.
- Section 4 (practical training) can be shorter for learners who are already comfortable debugging code. Focus on the overfitting/underfitting distinction.
- Allow plenty of time for the MNIST practice project — it's where everything comes together.

**Success Indicators:**
You'll know they've got it when they:
- Can trace backprop through a 2-layer network by hand, computing every gradient
- Explain backpropagation as "the chain rule applied backward through the network"
- Verify their gradients numerically without being prompted
- Build and run a training loop that makes the loss decrease
- Train on MNIST and get reasonable accuracy
- Ask questions about optimization tricks or architecture choices — this means they've internalized the fundamentals and are ready for the next level

**Most Common Confusion Points:**
1. **Chain rule mechanics**: Which derivatives multiply together and in what order. Remedy: trace the computation graph, label each edge with its derivative, and multiply along the path.
2. **Shapes in the backward pass**: The outer product `dL_dz @ input^T` produces the weight gradient. Learners often transpose the wrong thing. Remedy: "∂L/∂W has the same shape as W. Always."
3. **What gradients mean**: Gradients tell you which direction to move, not where to go. The learning rate controls the step size. They're a compass, not a GPS.
4. **Overfitting**: Learners may not see why low training loss is bad. The exam analogy helps: "If a student memorizes answers, they ace the practice test but fail the real one."

**Teaching Philosophy:**
- The learner already knows the chain rule and gradient descent from calculus-for-ml. This route is about applying those tools to neural networks. Don't re-derive the chain rule — apply it.
- Code is the primary medium. Every gradient gets computed in numpy before (or instead of) being written as a formula. The code IS the math.
- Numerical gradient verification is not optional. It's the equivalent of a unit test for your backward pass. Make it a habit.
- The MNIST project is the reward. The learner goes from abstract math to a network that reads handwriting. That transition from theory to capability is what makes this route meaningful.
- This route succeeds when the learner sees a training loop and thinks "forward, loss, backward, update — I know exactly what each step does" instead of "mysterious optimization process."
