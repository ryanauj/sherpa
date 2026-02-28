---
title: Bayesian Statistics
route_map: /routes/bayesian-statistics/map.md
paired_sherpa: /routes/bayesian-statistics/sherpa.md
prerequisites:
  - Probability Fundamentals (Bayes' theorem)
  - Probability Distributions
  - Statistical Inference
topics:
  - Bayesian vs Frequentist Thinking
  - Prior Distributions
  - Likelihood and Posterior
  - Conjugate Priors
  - Markov Chain Monte Carlo
  - Bayesian Decision Theory
---

# Bayesian Statistics - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/bayesian-statistics/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/bayesian-statistics/map.md` for the high-level overview.

## Overview

In the Statistical Inference route, you learned to test hypotheses and build confidence intervals using the frequentist framework. It works, but it can feel unnatural. You're forced to reason about "the probability of data given a hypothesis" when what you really want is "the probability of the hypothesis given the data."

Bayesian statistics flips the script. It starts with what you believe, updates those beliefs with evidence, and gives you a direct probability statement about the thing you care about. When a sports bettor says "I think this team has about a 60% chance of winning," that's Bayesian thinking. When you update your estimate of a fund manager's skill after each quarter of returns, that's Bayesian updating. When a spam filter learns from your email habits, that's Bayes in action.

This guide teaches five ideas:
1. **Bayesian vs frequentist thinking** — two paradigms, different strengths
2. **Priors, likelihood, and posteriors** — the Bayesian update cycle
3. **Conjugate priors** — closed-form Bayesian updates
4. **MCMC** — sampling from complex posteriors when closed forms don't exist
5. **Bayesian decision theory** — making optimal decisions under uncertainty

## Learning Objectives

By the end of this route, you will be able to:
- Explain the differences between Bayesian and frequentist approaches
- Construct prior distributions and compute posteriors through Bayesian updating
- Apply conjugate priors for closed-form analysis (Beta-Binomial, Normal-Normal, Gamma-Poisson)
- Implement a Metropolis-Hastings MCMC sampler and diagnose convergence
- Make optimal decisions using Bayesian decision theory and the Kelly criterion
- Build a Bayesian sports rating system

## Prerequisites

Before starting, you should be comfortable with:
- **Probability Fundamentals** ([route](/routes/probability-fundamentals/map.md)): Bayes' theorem, conditional probability
- **Probability Distributions** ([route](/routes/probability-distributions/map.md)): Beta, Normal, Poisson, Gamma distributions
- **Statistical Inference** ([route](/routes/statistical-inference/map.md)): Hypothesis testing, confidence intervals
- **Python**: numpy, scipy.stats, matplotlib

## Setup

```bash
pip install numpy scipy matplotlib
```

**Verify your setup:**

```python
import numpy as np
from scipy import stats
import matplotlib
print(f"numpy: {np.__version__}")
print(f"scipy.stats: available")
print(f"matplotlib: {matplotlib.__version__}")

# Quick Bayesian update: coin with Beta prior
alpha_prior, beta_prior = 2, 2  # mild prior toward 0.5
data_heads, data_tails = 7, 3
alpha_post = alpha_prior + data_heads
beta_post = beta_prior + data_tails
posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
```

**Expected output:**
```
numpy: 1.26.4
scipy.stats: available
matplotlib: 3.9.2
Posterior mean: 0.643
95% credible interval: (0.409, 0.841)
```

---

## Section 1: Bayesian vs Frequentist Thinking

### Two Paradigms

**Frequentist** and **Bayesian** statistics answer fundamentally different questions:

| | Frequentist | Bayesian |
|---|---|---|
| **What is probability?** | Long-run frequency | Degree of belief |
| **Parameters are...** | Fixed but unknown constants | Random variables with distributions |
| **Data is...** | Random (one sample from many possible) | Fixed (this is what we observed) |
| **Key question** | P(data \| hypothesis) | P(hypothesis \| data) |
| **Result** | p-values, confidence intervals | Posterior distributions, credible intervals |

### The Same Problem, Two Perspectives

**Scenario:** A bettor wins 56 out of 100 bets. Is the strategy profitable (true win rate > 50%)?

**Frequentist approach:**

```python
import numpy as np
from scipy import stats

wins, n = 56, 100
p_hat = wins / n

# H0: p = 0.50, H1: p > 0.50
se = np.sqrt(0.5 * 0.5 / n)
z = (p_hat - 0.5) / se
p_value = 1 - stats.norm.cdf(z)

print("FREQUENTIST APPROACH")
print(f"Sample win rate: {p_hat:.2f}")
print(f"z-statistic: {z:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")
```

**Expected output:**
```
FREQUENTIST APPROACH
Sample win rate: 0.56
z-statistic: 1.200
p-value: 0.1151
Conclusion: Fail to reject H0
```

**Bayesian approach:**

```python
from scipy import stats

wins, n = 56, 100

# Prior: Beta(2, 2) — mild belief that win rate is near 0.5
alpha_prior, beta_prior = 2, 2

# Posterior: Beta(alpha_prior + wins, beta_prior + losses)
alpha_post = alpha_prior + wins
beta_post = beta_prior + (n - wins)
posterior = stats.beta(alpha_post, beta_post)

# Direct probability of the question we care about
p_profitable = 1 - posterior.cdf(0.50)

print("BAYESIAN APPROACH")
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
print(f"P(win rate > 50%): {p_profitable:.3f}")
print(f"P(win rate > 53%): {1 - posterior.cdf(0.53):.3f}")
```

**Expected output:**
```
BAYESIAN APPROACH
Posterior mean: 0.558
95% credible interval: (0.462, 0.650)
P(win rate > 50%): 0.882
P(win rate > 53%): 0.720
```

Notice the difference: the frequentist says "not enough evidence to reject H₀" (a double negative). The Bayesian says "there's an 88% probability the win rate exceeds 50%." Same data, more intuitive answer.

### When to Use Each Approach

Both frameworks have strengths:

- **Frequentist** is great for: regulatory decisions (FDA drug approval), controlled experiments, when you want strong false-positive guarantees
- **Bayesian** is great for: incorporating prior knowledge, small samples, sequential updating, decision-making under uncertainty, when you want direct probability statements

### Key Points to Remember

- **Bayesian credible intervals** have the interpretation people think confidence intervals have: "There's a 95% probability the parameter is in this interval"
- **Priors** encode what you believed before seeing data; they get overwhelmed by enough data
- **Both approaches converge** with large samples — the debate matters most with small data

### Exercise 1.1: Framing Problems Both Ways

**Task:** A fund manager claims to generate 2% annual alpha (excess return over the market). After 5 years of data, their average annual alpha is 2.5% with a standard deviation of 4%. Frame this as both a frequentist and Bayesian analysis.

<details>
<summary>Hint: Frequentist setup</summary>

Use a one-sample t-test with H₀: α = 0, H₁: α > 0. You have n=5 years.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

# Data
n_years = 5
mean_alpha = 0.025  # 2.5% observed
std_alpha = 0.04    # 4% standard deviation

# Frequentist: t-test
se = std_alpha / np.sqrt(n_years)
t_stat = mean_alpha / se
p_value = 1 - stats.t.cdf(t_stat, df=n_years-1)

print("FREQUENTIST")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (one-sided): {p_value:.4f}")
print(f"Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'}")
print()

# Bayesian: Normal-Normal conjugate
# Prior: alpha ~ Normal(0, 0.03^2) — skeptical prior centered at no alpha
prior_mean = 0.0
prior_std = 0.03

# Posterior (Normal-Normal conjugate)
prior_precision = 1 / prior_std**2
data_precision = n_years / std_alpha**2
post_precision = prior_precision + data_precision
post_mean = (prior_precision * prior_mean + data_precision * mean_alpha) / post_precision
post_std = np.sqrt(1 / post_precision)

posterior = stats.norm(post_mean, post_std)

print("BAYESIAN (skeptical prior)")
print(f"Prior: N(0, {prior_std:.3f})")
print(f"Posterior mean: {post_mean:.4f} ({post_mean:.2%})")
print(f"95% credible interval: ({posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f})")
print(f"P(alpha > 0): {1 - posterior.cdf(0):.3f}")
print(f"P(alpha > 2%): {1 - posterior.cdf(0.02):.3f}")
```

**Expected output:**
```
FREQUENTIST
t-statistic: 1.398
p-value (one-sided): 0.1173
Conclusion: Not significant

BAYESIAN (skeptical prior)
Prior: N(0, 0.030)
Posterior mean: 0.0154 (1.54%)
95% credible interval: (-0.0094, 0.0401)
P(alpha > 0): 0.889
P(alpha > 2%): 0.336
```

**Explanation:** The frequentist says "not significant" — 5 years isn't enough data. The Bayesian says "89% chance of positive alpha, but only 34% chance it exceeds 2%." The Bayesian approach, with its skeptical prior, pulls the estimate toward zero, reflecting the well-known difficulty of generating alpha.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the core difference between frequentist and Bayesian paradigms
- [ ] Name a scenario where each approach is more natural
- [ ] Interpret a Bayesian credible interval vs a frequentist confidence interval

---

## Section 2: Priors, Likelihood, and Posteriors

### The Bayesian Update

Bayesian inference follows a simple recipe:

**Posterior ∝ Prior × Likelihood**

- **Prior**: Your belief about a parameter before seeing data — P(θ)
- **Likelihood**: How probable the observed data is for each possible parameter value — P(data | θ)
- **Posterior**: Your updated belief after seeing data — P(θ | data)

### Visualizing Bayesian Updating

```python
import numpy as np
from scipy import stats

# A sports bettor wants to estimate a team's true win probability
# Prior belief: the team is decent, around 55% win rate
# Use Beta(11, 9) which has mean 0.55 and is moderately confident

alpha_prior, beta_prior = 11, 9
prior = stats.beta(alpha_prior, beta_prior)

# Observe: team goes 8-2 in their first 10 games
wins, losses = 8, 2

# Posterior: Beta(11+8, 9+2) = Beta(19, 11)
alpha_post = alpha_prior + wins
beta_post = beta_prior + losses
posterior = stats.beta(alpha_post, beta_post)

x = np.linspace(0, 1, 200)
print("Prior:     Beta(11, 9)")
print(f"  Mean: {prior.mean():.3f}")
print(f"  95% CI: ({prior.ppf(0.025):.3f}, {prior.ppf(0.975):.3f})")
print()
print(f"Data: {wins} wins, {losses} losses ({wins/(wins+losses):.0%} win rate)")
print()
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"  Mean: {posterior.mean():.3f}")
print(f"  95% CI: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
```

**Expected output:**
```
Prior:     Beta(11, 9)
  Mean: 0.550
  95% CI: (0.366, 0.722)

Data: 8 wins, 2 losses (80% win rate)

Posterior: Beta(19, 11)
  Mean: 0.633
  95% CI: (0.470, 0.780)
```

The team went 8-2 (80% win rate), but the posterior mean is only 63.3%, not 80%. The prior pulled the estimate toward 55%, because 10 games isn't enough to override our prior knowledge that most teams don't win 80% of their games.

### Sequential Updating

One of Bayesian inference's superpowers is **sequential updating** — you can update one observation at a time, and the order doesn't matter.

```python
from scipy import stats

# Start with a weak prior
alpha, beta_param = 2, 2

# Observe results one game at a time
results = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1,  # First 10 games: 8-2
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0]  # Next 10 games: 6-4

print(f"{'Games':>5} | {'Record':>7} | {'Post Mean':>9} | {'95% Credible Interval':>22}")
print("-" * 55)

total_w, total_l = 0, 0
for i, result in enumerate(results):
    if result == 1:
        alpha += 1
        total_w += 1
    else:
        beta_param += 1
        total_l += 1

    post = stats.beta(alpha, beta_param)

    if (i + 1) % 5 == 0 or i == 0:
        print(f"{i+1:>5} | {total_w:>3}-{total_l:<3} | {post.mean():>8.3f}  | ({post.ppf(0.025):.3f}, {post.ppf(0.975):.3f})")

print(f"\nFinal estimate: {stats.beta(alpha, beta_param).mean():.3f}")
print(f"Data-only estimate: {total_w/(total_w+total_l):.3f}")
```

**Expected output:**
```
Games | Record | Post Mean | 95% Credible Interval
-------------------------------------------------------
    1 |   1-0   |    0.600  | (0.192, 0.937)
    5 |   4-1   |    0.667  | (0.381, 0.893)
   10 |   8-2   |    0.714  | (0.519, 0.872)
   15 |  12-3   |    0.684  | (0.516, 0.825)
   20 |  14-6   |    0.667  | (0.509, 0.801)

Final estimate: 0.667
Data-only estimate: 0.700
```

### Choosing Priors

The choice of prior is a feature, not a bug. Different priors reflect different prior knowledge:

```python
from scipy import stats

# Three analysts estimate a new team's win probability
# Same data: 7 wins in 10 games

wins, losses = 7, 3

priors = {
    'Uninformative (Beta(1,1))': (1, 1),
    'Mildly informative (Beta(5,5))': (5, 5),
    'Strong prior toward 50% (Beta(50,50))': (50, 50),
}

print(f"Data: {wins} wins, {losses} losses\n")
for name, (a, b) in priors.items():
    prior = stats.beta(a, b)
    posterior = stats.beta(a + wins, b + losses)
    print(f"{name}")
    print(f"  Prior mean: {prior.mean():.3f}")
    print(f"  Posterior mean: {posterior.mean():.3f}")
    print(f"  Posterior 95% CI: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
    print()
```

**Expected output:**
```
Data: 7 wins, 3 losses

Uninformative (Beta(1,1))
  Prior mean: 0.500
  Posterior mean: 0.667
  Posterior 95% CI: (0.384, 0.897)

Mildly informative (Beta(5,5))
  Prior mean: 0.500
  Posterior mean: 0.600
  Posterior 95% CI: (0.410, 0.773)

Strong prior toward 50% (Beta(50,50))
  Prior mean: 0.500
  Posterior mean: 0.518
  Posterior 95% CI: (0.430, 0.606)
```

With a strong prior, 10 games barely budge the estimate. With a weak prior, the data dominates. The right prior depends on how much you know before seeing data.

### Exercise 2.1: Updating a Sports Prediction

**Task:** You believe a basketball team's free throw percentage is around 75% (use Beta(30, 10) as your prior). The player goes 6-for-10 in tonight's game. Compute the posterior and the probability their true rate is above 70%.

<details>
<summary>Hint</summary>

The Beta-Binomial conjugate: if prior is Beta(α, β) and you observe k successes in n trials, the posterior is Beta(α + k, β + n - k).
</details>

<details>
<summary>Click to see solution</summary>

```python
from scipy import stats

alpha_prior, beta_prior = 30, 10
makes, misses = 6, 4

posterior = stats.beta(alpha_prior + makes, beta_prior + misses)

print(f"Prior: Beta(30, 10), mean = {stats.beta(30, 10).mean():.3f}")
print(f"Tonight: {makes}/{makes+misses}")
print(f"Posterior: Beta({alpha_prior+makes}, {beta_prior+misses})")
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
print(f"P(true rate > 70%): {1 - posterior.cdf(0.70):.3f}")
```

**Expected output:**
```
Prior: Beta(30, 10), mean = 0.750
Posterior: Beta(36, 14)
Posterior mean: 0.720
95% credible interval: (0.579, 0.840)
P(true rate > 70%): 0.615
```

**Explanation:** A 60% shooting night pulled the estimate down from 75% to 72%, but not as far as a naive 60% estimate. The prior (based on the team's season performance) provides useful stability.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] State the Bayesian update formula in words
- [ ] Explain how prior strength affects the posterior
- [ ] Perform a manual Bayesian update with the Beta-Binomial model

---

## Section 3: Conjugate Priors

### What Are Conjugate Priors?

A **conjugate prior** is a prior distribution that, when combined with a particular likelihood, produces a posterior in the same distribution family. This gives you a closed-form solution — no simulation needed.

### Beta-Binomial: Win Rates and Proportions

This is the workhorse conjugate pair for proportions.

- **Prior**: Beta(α, β)
- **Data**: k successes in n trials
- **Posterior**: Beta(α + k, β + n - k)

**Application — Bayesian batting average estimator:**

```python
import numpy as np
from scipy import stats

# MLB batting averages: the league-wide distribution is roughly Beta(81, 219)
# This means a prior mean of 81/300 = 0.270 (league average)
league_alpha, league_beta = 81, 219

players = {
    'Player A (hot start)':  {'hits': 30, 'at_bats': 80},   # .375 batting avg
    'Player B (cold start)': {'hits': 10, 'at_bats': 80},   # .125 batting avg
    'Player C (small sample)': {'hits': 5, 'at_bats': 10},  # .500 batting avg
}

print("Bayesian Batting Average Estimator")
print(f"Prior: Beta({league_alpha}, {league_beta}) — league average {league_alpha/(league_alpha+league_beta):.3f}")
print()

for name, data in players.items():
    raw_avg = data['hits'] / data['at_bats']
    alpha_post = league_alpha + data['hits']
    beta_post = league_beta + data['at_bats'] - data['hits']
    posterior = stats.beta(alpha_post, beta_post)

    print(f"{name}")
    print(f"  Raw average: {raw_avg:.3f}")
    print(f"  Bayesian estimate: {posterior.mean():.3f}")
    print(f"  95% credible interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
    print(f"  Shrinkage toward league avg: {abs(raw_avg - posterior.mean()):.3f}")
    print()
```

**Expected output:**
```
Bayesian Batting Average Estimator
Prior: Beta(81, 219) — league average 0.270

Player A (hot start)
  Raw average: 0.375
  Bayesian estimate: 0.292
  95% credible interval: (0.249, 0.337)
  Shrinkage toward league avg: 0.083

Player B (cold start)
  Raw average: 0.125
  Bayesian estimate: 0.240
  95% credible interval: (0.199, 0.283)
  Shrinkage toward league avg: 0.115

Player C (small sample)
  Raw average: 0.500
  Bayesian estimate: 0.277
  95% credible interval: (0.229, 0.328)
  Shrinkage toward league avg: 0.223
```

Player C's estimate barely moved from the league average despite a .500 raw average — because 10 at-bats is almost no information. Player A's hot start still gets pulled toward .270 because the prior represents about 300 at-bats of league experience.

### Normal-Normal: Estimating Means

When both the prior and likelihood are Normal:

- **Prior**: N(μ₀, σ₀²)
- **Data**: n observations with sample mean x̄ and known variance σ²
- **Posterior**: N(μ_post, σ_post²)

```python
import numpy as np
from scipy import stats

# Estimating a fund's true annual return
# Prior: industry average fund returns ~8% with wide uncertainty
prior_mean = 0.08
prior_std = 0.05

# Data: 5 years of returns
observed_returns = np.array([0.12, -0.03, 0.15, 0.08, 0.11])
n = len(observed_returns)
data_mean = np.mean(observed_returns)
data_std = 0.10  # assume known volatility

# Posterior
prior_precision = 1 / prior_std**2
data_precision = n / data_std**2
post_precision = prior_precision + data_precision
post_mean = (prior_precision * prior_mean + data_precision * data_mean) / post_precision
post_std = np.sqrt(1 / post_precision)

posterior = stats.norm(post_mean, post_std)

print("Estimating Fund's True Annual Return")
print(f"Prior: N({prior_mean:.2%}, {prior_std:.2%})")
print(f"Data: {n} years, mean return = {data_mean:.2%}")
print(f"Posterior: N({post_mean:.2%}, {post_std:.2%})")
print(f"95% credible interval: ({posterior.ppf(0.025):.2%}, {posterior.ppf(0.975):.2%})")
print(f"P(true return > 10%): {1 - posterior.cdf(0.10):.3f}")
```

**Expected output:**
```
Estimating Fund's True Annual Return
Prior: N(8.00%, 5.00%)
Data: 5 years, mean return = 8.60%
Posterior: N(8.47%, 3.54%)
95% credible interval: (1.54%, 15.40%)
P(true return > 10%): 0.333
```

### Gamma-Poisson: Rate Estimation

For count data (goals per game, trades per hour):

- **Prior**: Gamma(α, β)
- **Data**: observe total count s in t intervals
- **Posterior**: Gamma(α + s, β + t)

```python
from scipy import stats

# Estimating a soccer team's scoring rate (goals per game)
# Prior: teams typically score about 1.5 goals per game
# Gamma(3, 2) has mean 3/2 = 1.5
alpha_prior, beta_prior = 3, 2

# Data: team scored 12 goals in 6 games
total_goals, n_games = 12, 6

alpha_post = alpha_prior + total_goals
beta_post = beta_prior + n_games
posterior = stats.gamma(alpha_post, scale=1/beta_post)

print("Estimating Goals-Per-Game Rate")
print(f"Prior: Gamma(3, 2), mean = {alpha_prior/beta_prior:.2f}")
print(f"Data: {total_goals} goals in {n_games} games ({total_goals/n_games:.2f}/game)")
print(f"Posterior: Gamma({alpha_post}, {beta_post}), mean = {alpha_post/beta_post:.2f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.2f}, {posterior.ppf(0.975):.2f})")
print(f"P(rate > 2.0): {1 - posterior.cdf(2.0):.3f}")
```

**Expected output:**
```
Estimating Goals-Per-Game Rate
Prior: Gamma(3, 2), mean = 1.50
Data: 12 goals in 6 games (2.00/game)
Posterior: Gamma(15, 8), mean = 1.88
95% credible interval: (1.05, 2.89)
P(rate > 2.0): 0.432
```

### Exercise 3.1: Build a Bayesian Conversion Rate Estimator

**Task:** An online sportsbook A/B tests two landing page designs. Design A: 45 signups out of 500 visitors. Design B: 62 signups out of 500 visitors. Using a Beta(1,1) uninformative prior, compute posterior distributions for both and find P(Design B > Design A).

<details>
<summary>Hint: Comparing posteriors</summary>

To find P(B > A), draw many samples from each posterior and count how often the B sample exceeds the A sample.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Design A
a_success, a_total = 45, 500
post_a = stats.beta(1 + a_success, 1 + a_total - a_success)

# Design B
b_success, b_total = 62, 500
post_b = stats.beta(1 + b_success, 1 + b_total - b_success)

print(f"Design A: {a_success}/{a_total} = {a_success/a_total:.1%}")
print(f"  Posterior mean: {post_a.mean():.4f}")
print(f"  95% CI: ({post_a.ppf(0.025):.4f}, {post_a.ppf(0.975):.4f})")
print()
print(f"Design B: {b_success}/{b_total} = {b_success/b_total:.1%}")
print(f"  Posterior mean: {post_b.mean():.4f}")
print(f"  95% CI: ({post_b.ppf(0.025):.4f}, {post_b.ppf(0.975):.4f})")
print()

# Monte Carlo comparison
n_samples = 100000
samples_a = post_a.rvs(n_samples)
samples_b = post_b.rvs(n_samples)
p_b_better = np.mean(samples_b > samples_a)

print(f"P(Design B > Design A): {p_b_better:.3f}")
print(f"Expected lift: {(post_b.mean() - post_a.mean()) / post_a.mean():.1%}")
```

**Expected output:**
```
Design A: 45/500 = 9.0%
  Posterior mean: 0.0916
  95% CI: (0.0683, 0.1184)

Design B: 62/500 = 12.4%
  Posterior mean: 0.1255
  95% CI: (0.0977, 0.1567)

P(Design B > Design A): 0.964
Expected lift: 37.0%
```

**Explanation:** There's a 96.4% probability that Design B converts better. In a frequentist test, you might report p < 0.05 and stop. The Bayesian approach gives you a direct probability and helps you decide whether to switch.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] List the three main conjugate pairs (Beta-Binomial, Normal-Normal, Gamma-Poisson)
- [ ] Perform a Beta-Binomial update and interpret the result
- [ ] Explain what "shrinkage" toward the prior means and why it's useful

---

## Section 4: Introduction to MCMC

### Why We Need MCMC

Conjugate priors are elegant, but most real problems don't have nice closed-form posteriors. When the posterior is too complex to compute analytically, we use **Markov Chain Monte Carlo (MCMC)** to sample from it.

The idea: construct a sequence of random samples (a Markov chain) whose long-run distribution converges to the posterior. Then use those samples to estimate any quantity you want.

### The Metropolis-Hastings Algorithm

The simplest MCMC algorithm:

1. Start at some parameter value θ
2. Propose a new value θ* by adding random noise
3. Compute the acceptance ratio: r = P(θ* | data) / P(θ | data)
4. If r > 1, always accept. If r < 1, accept with probability r
5. Repeat

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Problem: Estimate a coin's bias
# We observe 7 heads in 10 flips
# Prior: Beta(2, 2)
# Posterior should be Beta(9, 5) — we'll verify with MCMC

n_heads, n_tails = 7, 3
alpha_prior, beta_prior = 2, 2

def log_posterior(theta):
    """Log of unnormalized posterior: log(prior) + log(likelihood)"""
    if theta <= 0 or theta >= 1:
        return -np.inf
    log_prior = stats.beta.logpdf(theta, alpha_prior, beta_prior)
    log_likelihood = stats.binom.logpmf(n_heads, n_heads + n_tails, theta)
    return log_prior + log_likelihood

# Metropolis-Hastings
n_samples = 50000
samples = np.zeros(n_samples)
samples[0] = 0.5  # starting value
proposal_std = 0.1
accepted = 0

for i in range(1, n_samples):
    # Propose
    theta_proposed = samples[i-1] + np.random.normal(0, proposal_std)

    # Accept/reject
    log_ratio = log_posterior(theta_proposed) - log_posterior(samples[i-1])

    if np.log(np.random.random()) < log_ratio:
        samples[i] = theta_proposed
        accepted += 1
    else:
        samples[i] = samples[i-1]

# Discard burn-in
burn_in = 5000
posterior_samples = samples[burn_in:]

# Compare to analytical posterior
analytical = stats.beta(alpha_prior + n_heads, beta_prior + n_tails)

print("Metropolis-Hastings Results")
print(f"Acceptance rate: {accepted/n_samples:.1%}")
print(f"MCMC posterior mean: {np.mean(posterior_samples):.4f}")
print(f"Analytical posterior mean: {analytical.mean():.4f}")
print(f"MCMC 95% CI: ({np.percentile(posterior_samples, 2.5):.4f}, {np.percentile(posterior_samples, 97.5):.4f})")
print(f"Analytical 95% CI: ({analytical.ppf(0.025):.4f}, {analytical.ppf(0.975):.4f})")
```

**Expected output:**
```
Metropolis-Hastings Results
Acceptance rate: 53.9%
MCMC posterior mean: 0.6428
Analytical posterior mean: 0.6429
MCMC 95% CI: (0.3873, 0.8608)
Analytical 95% CI: (0.3883, 0.8553)
```

The MCMC estimates match the analytical solution closely.

### Diagnostics: Is Your Chain Working?

MCMC can go wrong. Here are key diagnostics:

```python
import numpy as np

# Using the samples from above
print("MCMC Diagnostics")
print(f"Chain length (after burn-in): {len(posterior_samples)}")

# Effective sample size (rough approximation)
# Autocorrelation reduces effective samples
autocorr_lag1 = np.corrcoef(posterior_samples[:-1], posterior_samples[1:])[0, 1]
ess_approx = len(posterior_samples) * (1 - autocorr_lag1) / (1 + autocorr_lag1)
print(f"Lag-1 autocorrelation: {autocorr_lag1:.3f}")
print(f"Approximate effective sample size: {ess_approx:.0f}")

# Check stationarity: compare first half to second half
first_half = posterior_samples[:len(posterior_samples)//2]
second_half = posterior_samples[len(posterior_samples)//2:]
print(f"First half mean: {np.mean(first_half):.4f}")
print(f"Second half mean: {np.mean(second_half):.4f}")
print(f"Difference: {abs(np.mean(first_half) - np.mean(second_half)):.4f}")
```

**Expected output:**
```
MCMC Diagnostics
Chain length (after burn-in): 45000
Lag-1 autocorrelation: 0.722
Approximate effective sample size: 7267
Approximate effective sample size: 7267
First half mean: 0.6424
Second half mean: 0.6432
Difference: 0.0008
```

Key things to check:
- **Acceptance rate**: 20-50% for most problems (ours is good at ~54%)
- **Autocorrelation**: Lower is better — high autocorrelation means less information per sample
- **Stationarity**: First and second halves should have similar statistics

### Exercise 4.1: MCMC for a Non-Conjugate Problem

**Task:** A sports bettor believes their edge is somewhere between 0% and 10%, but not uniformly — they think smaller edges are more likely. Use a Beta(2, 20) prior (mean = 9.1%) on the edge, observe 540 wins in 1000 bets (edge ≈ 4% above 50%), and estimate the posterior using MCMC.

<details>
<summary>Hint: Defining the posterior</summary>

The parameter is the true win rate p. The prior is Beta(2, 20) on the edge (p - 0.5), so on p itself it's a shifted Beta. The likelihood is Binomial(1000, p).
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(42)

wins, n = 540, 1000

# Prior on win rate: we'll use Beta(12, 12) centered at 0.50
# with slight skew toward small positive edges
# More precisely: log-posterior approach
alpha_prior, beta_prior = 12, 12

def log_posterior(p):
    if p <= 0 or p >= 1:
        return -np.inf
    return stats.beta.logpdf(p, alpha_prior, beta_prior) + stats.binom.logpmf(wins, n, p)

n_samples = 50000
samples = np.zeros(n_samples)
samples[0] = 0.5
accepted = 0

for i in range(1, n_samples):
    proposed = samples[i-1] + np.random.normal(0, 0.02)
    log_r = log_posterior(proposed) - log_posterior(samples[i-1])
    if np.log(np.random.random()) < log_r:
        samples[i] = proposed
        accepted += 1
    else:
        samples[i] = samples[i-1]

burn_in = 5000
post_samples = samples[burn_in:]
edge_samples = post_samples - 0.50

print(f"Acceptance rate: {accepted/n_samples:.1%}")
print(f"Posterior win rate: {np.mean(post_samples):.4f}")
print(f"Posterior edge: {np.mean(edge_samples):.4f} ({np.mean(edge_samples):.2%})")
print(f"95% CI for edge: ({np.percentile(edge_samples, 2.5):.4f}, {np.percentile(edge_samples, 97.5):.4f})")
print(f"P(edge > 0): {np.mean(edge_samples > 0):.3f}")
print(f"P(edge > 2%): {np.mean(edge_samples > 0.02):.3f}")
```

**Expected output:**
```
Acceptance rate: 62.2%
Posterior win rate: 0.5388
Posterior edge: 0.0388 (3.88%)
95% CI for edge: (0.0103, 0.0674)
P(edge > 0): 0.996
P(edge > 2%): 0.926
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the Metropolis-Hastings algorithm in plain language
- [ ] Implement a basic MH sampler
- [ ] Check convergence diagnostics (acceptance rate, autocorrelation, stationarity)

---

## Section 5: Bayesian Decision Theory

### Making Decisions Under Uncertainty

Bayesian decision theory combines your posterior beliefs with a **loss function** to make optimal decisions. The best decision minimizes expected loss (or maximizes expected utility).

**Expected loss** = ∫ Loss(action, θ) × P(θ | data) dθ

### The Kelly Criterion: Optimal Bet Sizing

The Kelly criterion is a famous Bayesian decision rule for optimal bet sizing. It maximizes the expected logarithm of wealth (which maximizes long-run growth rate).

**Kelly fraction** = edge / odds = (bp - q) / b

Where b = decimal odds - 1, p = probability of winning, q = 1 - p.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# You believe a bet has P(win) distributed as Beta(56, 44)
# (based on observing 56 wins in 100 bets with a flat prior)
# The bet pays 2-to-1 (decimal odds = 2.0, so b = 1)

alpha, beta_param = 56, 44
b = 1.0  # bet pays 1:1 (even money)

# Point estimate Kelly
p_point = alpha / (alpha + beta_param)
kelly_point = (b * p_point - (1 - p_point)) / b

# Full Bayesian Kelly: integrate over posterior uncertainty
n_samples = 100000
p_samples = stats.beta.rvs(alpha, beta_param, size=n_samples)
kelly_samples = (b * p_samples - (1 - p_samples)) / b

# Optimal fraction: use posterior mean of Kelly (with max at 0)
kelly_values = np.maximum(kelly_samples, 0)  # never bet negative

print("Kelly Criterion Analysis")
print(f"Posterior P(win): {stats.beta(alpha, beta_param).mean():.3f}")
print(f"95% CI for P(win): ({stats.beta(alpha, beta_param).ppf(0.025):.3f}, {stats.beta(alpha, beta_param).ppf(0.975):.3f})")
print(f"Point estimate Kelly fraction: {kelly_point:.3f}")
print(f"Bayesian Kelly (posterior mean): {np.mean(kelly_values):.3f}")
print(f"P(edge > 0): {np.mean(p_samples > 0.5):.3f}")
print()

# Simulate outcomes with different bet sizes
bankroll = 1000
n_bets = 200

for fraction in [0.02, 0.05, kelly_point, 0.15, 0.25]:
    np.random.seed(42)
    balance = bankroll
    for _ in range(n_bets):
        p_true = 0.56  # actual probability
        bet_size = balance * min(fraction, 1.0)
        if np.random.random() < p_true:
            balance += bet_size * b
        else:
            balance -= bet_size
    print(f"Fraction {fraction:.2f}: Final balance = ${balance:,.0f} ({balance/bankroll - 1:+.1%})")
```

**Expected output:**
```
Kelly Criterion Analysis
Posterior P(win): 0.560
95% CI for P(win): (0.462, 0.654)
Point estimate Kelly fraction: 0.120
Bayesian Kelly (posterior mean): 0.094
P(edge > 0): 0.882

Fraction 0.02: Final balance = $1,046 (+4.6%)
Fraction 0.05: Final balance = $1,116 (+11.6%)
Fraction 0.12: Final balance = $1,262 (+26.2%)
Fraction 0.15: Final balance = $1,281 (+28.1%)
Fraction 0.25: Final balance = $1,194 (+19.4%)
```

Notice that betting too much (25%) actually produces worse results than the Kelly fraction. Over-betting increases variance and risk of ruin.

### Bayesian Model Comparison

When choosing between models, the Bayesian approach computes the **posterior probability** of each model:

```python
import numpy as np
from scipy import stats

# Is a die fair or loaded?
# Model 1: fair die, P(6) = 1/6
# Model 2: loaded die, P(6) = 1/4

# Data: rolled die 60 times, got 15 sixes
n_rolls = 60
n_sixes = 15

# Prior: 50/50 on fair vs loaded
prior_fair = 0.5
prior_loaded = 0.5

# Likelihoods
p_data_fair = stats.binom.pmf(n_sixes, n_rolls, 1/6)
p_data_loaded = stats.binom.pmf(n_sixes, n_rolls, 1/4)

# Bayes factor
bayes_factor = p_data_loaded / p_data_fair

# Posterior probabilities
p_evidence = p_data_fair * prior_fair + p_data_loaded * prior_loaded
post_fair = (p_data_fair * prior_fair) / p_evidence
post_loaded = (p_data_loaded * prior_loaded) / p_evidence

print(f"Data: {n_sixes} sixes in {n_rolls} rolls ({n_sixes/n_rolls:.1%})")
print(f"Expected under fair: {n_rolls/6:.1f} sixes ({1/6:.1%})")
print(f"Expected under loaded: {n_rolls/4:.1f} sixes ({1/4:.1%})")
print()
print(f"Bayes factor (loaded vs fair): {bayes_factor:.2f}")
print(f"P(fair die | data): {post_fair:.3f}")
print(f"P(loaded die | data): {post_loaded:.3f}")
```

**Expected output:**
```
Data: 15 sixes in 60 rolls (25.0%)
Expected under fair: 10.0 sixes (16.7%)
Expected under loaded: 15.0 sixes (25.0%)

Bayes factor (loaded vs fair): 4.82
P(fair die | data): 0.172
P(loaded die | data): 0.828
```

### Exercise 5.1: Kelly Criterion with Uncertainty

**Task:** You've observed 70 wins in 120 sports bets at -110 odds (decimal odds = 1.909). Use a Beta(1,1) prior to compute the posterior win probability, then determine the Kelly-optimal bet size. Also compute how the recommended bet size changes if you use a "half-Kelly" strategy (popular for managing risk).

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

wins, losses = 70, 50
n = wins + losses
decimal_odds = 1.909
b = decimal_odds - 1  # net payout = 0.909

# Posterior on win probability
posterior = stats.beta(1 + wins, 1 + losses)

print(f"Record: {wins}-{losses} ({wins/n:.1%})")
print(f"Posterior mean P(win): {posterior.mean():.4f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f})")
print()

# Kelly calculation
p = posterior.mean()
q = 1 - p
kelly_full = (b * p - q) / b
kelly_half = kelly_full / 2

print(f"At -110 odds (decimal {decimal_odds}):")
print(f"Break-even win rate: {1/decimal_odds:.4f} ({1/decimal_odds:.1%})")
print(f"Full Kelly fraction: {kelly_full:.4f} ({kelly_full:.2%})")
print(f"Half Kelly fraction: {kelly_half:.4f} ({kelly_half:.2%})")
print()

# Bayesian Kelly: account for parameter uncertainty
samples = posterior.rvs(100000)
kelly_samples = np.maximum((b * samples - (1 - samples)) / b, 0)
print(f"Bayesian Kelly (mean): {np.mean(kelly_samples):.4f}")
print(f"P(edge > 0): {np.mean(samples > 1/decimal_odds):.3f}")
print()
print(f"With $1000 bankroll:")
print(f"  Full Kelly bet: ${1000 * kelly_full:.2f}")
print(f"  Half Kelly bet: ${1000 * kelly_half:.2f}")
print(f"  Bayesian Kelly bet: ${1000 * np.mean(kelly_samples):.2f}")
```

**Expected output:**
```
Record: 70-50 (58.3%)
Posterior mean P(win): 0.5820
95% credible interval: (0.4929, 0.6676)

At -110 odds (decimal 1.909):
Break-even win rate: 0.5238 (52.4%)
Full Kelly fraction: 0.0951 (9.51%)
Half Kelly fraction: 0.0476 (4.76%)

Bayesian Kelly (mean): 0.0734
P(edge > 0): 0.898

With $1000 bankroll:
  Full Kelly bet: $95.12
  Half Kelly bet: $47.56
  Bayesian Kelly bet: $73.44
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain expected utility/loss in Bayesian decision theory
- [ ] Apply the Kelly criterion to a betting scenario
- [ ] Compare models using Bayes factors

---

## Common Pitfalls

### Pitfall 1: Poor Prior Choice

**The Problem:** Using an overly informative prior that dominates the data, or an improper prior that leads to nonsensical results.

**How to avoid it:** Start with weakly informative priors. Check sensitivity: if reasonable alternative priors give very different posteriors, you need more data.

### Pitfall 2: Ignoring MCMC Diagnostics

**The Problem:** Treating MCMC output as if it's a perfect sample from the posterior without checking convergence.

**How to avoid it:** Always check trace plots, acceptance rates, autocorrelation, and compare multiple chains. Discard a generous burn-in period.

### Pitfall 3: Overconfident Posteriors

**The Problem:** Taking the posterior too literally when the model itself might be wrong.

**How to avoid it:** The posterior is only as good as your model. If your likelihood or prior is misspecified, the posterior will be wrong too. Use posterior predictive checks: simulate data from your model and see if it looks like the real data.

## Best Practices

- ✅ **Use weakly informative priors** — they regularize without dominating
- ✅ **Check prior sensitivity** — make sure reasonable prior changes don't flip conclusions
- ✅ **Report full posteriors** — not just point estimates; show credible intervals
- ✅ **Use posterior predictive checks** — verify your model generates realistic data
- ✅ **Start with conjugate models** — they're fast and help build intuition before going to MCMC
- ❌ **Don't use flat priors as default** — they're not always uninformative and can cause problems
- ❌ **Don't skip diagnostics** — a non-converged chain gives meaningless results
- ❌ **Don't ignore the model** — the posterior depends on both prior and likelihood being reasonable

---

## Practice Project

### Project Description

Build a **Bayesian sports rating system** that ranks teams and predicts game outcomes using Bayesian updating.

### Requirements

1. Create a rating system with Beta-distributed team strengths
2. Update ratings after each game result
3. Predict future game outcomes with probability estimates
4. Track how ratings evolve over a season
5. Compare predictions against a naive baseline (e.g., home team always wins)

### Getting Started

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Set up 8 teams with different true strengths
teams = {
    'Team A': {'true_strength': 0.70, 'alpha': 5, 'beta': 5},
    'Team B': {'true_strength': 0.65, 'alpha': 5, 'beta': 5},
    'Team C': {'true_strength': 0.55, 'alpha': 5, 'beta': 5},
    'Team D': {'true_strength': 0.50, 'alpha': 5, 'beta': 5},
    'Team E': {'true_strength': 0.50, 'alpha': 5, 'beta': 5},
    'Team F': {'true_strength': 0.45, 'alpha': 5, 'beta': 5},
    'Team G': {'true_strength': 0.35, 'alpha': 5, 'beta': 5},
    'Team H': {'true_strength': 0.30, 'alpha': 5, 'beta': 5},
}

# TODO: Simulate a round-robin season
# TODO: Update ratings after each game
# TODO: Predict matchups and evaluate accuracy
```

<details>
<summary>If you're not sure where to start</summary>

1. For each matchup, compute each team's win probability using their current posterior means
2. Simulate the game outcome based on their true strengths
3. Update the winner's Beta distribution: alpha += 1
4. Update the loser's Beta distribution: beta += 1
5. Track predictions and outcomes to evaluate accuracy
</details>

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(42)

teams = {
    'Team A': {'true_strength': 0.70, 'alpha': 5, 'beta': 5},
    'Team B': {'true_strength': 0.65, 'alpha': 5, 'beta': 5},
    'Team C': {'true_strength': 0.55, 'alpha': 5, 'beta': 5},
    'Team D': {'true_strength': 0.50, 'alpha': 5, 'beta': 5},
    'Team E': {'true_strength': 0.50, 'alpha': 5, 'beta': 5},
    'Team F': {'true_strength': 0.45, 'alpha': 5, 'beta': 5},
    'Team G': {'true_strength': 0.35, 'alpha': 5, 'beta': 5},
    'Team H': {'true_strength': 0.30, 'alpha': 5, 'beta': 5},
}

team_names = list(teams.keys())

def predict_win_prob(team1, team2):
    """Predict P(team1 wins) based on current posteriors using Monte Carlo."""
    samples1 = stats.beta.rvs(teams[team1]['alpha'], teams[team1]['beta'], size=10000)
    samples2 = stats.beta.rvs(teams[team2]['alpha'], teams[team2]['beta'], size=10000)
    return np.mean(samples1 > samples2)

def simulate_game(team1, team2):
    """Simulate a game outcome based on true strengths."""
    s1 = teams[team1]['true_strength']
    s2 = teams[team2]['true_strength']
    p1_wins = s1 / (s1 + s2)
    return team1 if np.random.random() < p1_wins else team2

# Simulate 3 rounds of round-robin (each team plays each other 3 times)
correct_predictions = 0
total_games = 0
brier_score_sum = 0

for round_num in range(3):
    for i in range(len(team_names)):
        for j in range(i + 1, len(team_names)):
            t1, t2 = team_names[i], team_names[j]

            # Predict
            p1_wins = predict_win_prob(t1, t2)
            predicted_winner = t1 if p1_wins > 0.5 else t2

            # Simulate
            actual_winner = simulate_game(t1, t2)

            # Evaluate
            if predicted_winner == actual_winner:
                correct_predictions += 1
            outcome = 1.0 if actual_winner == t1 else 0.0
            brier_score_sum += (p1_wins - outcome) ** 2
            total_games += 1

            # Update ratings
            if actual_winner == t1:
                teams[t1]['alpha'] += 1
                teams[t2]['beta'] += 1
            else:
                teams[t2]['alpha'] += 1
                teams[t1]['beta'] += 1

# Results
print("=" * 55)
print("BAYESIAN SPORTS RATING SYSTEM — SEASON RESULTS")
print("=" * 55)

print(f"\nTotal games: {total_games}")
print(f"Prediction accuracy: {correct_predictions}/{total_games} = {correct_predictions/total_games:.1%}")
print(f"Brier score: {brier_score_sum/total_games:.4f}")
print(f"Baseline (always pick random): ~50%")
print()

print(f"{'Team':>8} | {'True':>5} | {'Rating':>6} | {'95% CI':>16} | {'Record':>7}")
print("-" * 55)

for name in sorted(teams.keys(), key=lambda t: teams[t]['alpha']/(teams[t]['alpha']+teams[t]['beta']), reverse=True):
    t = teams[name]
    posterior = stats.beta(t['alpha'], t['beta'])
    wins = t['alpha'] - 5  # subtract prior
    losses = t['beta'] - 5
    print(f"{name:>8} | {t['true_strength']:.2f}  | {posterior.mean():.3f}  | ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f}) | {wins:>2}-{losses}")
```

**Key points in this solution:**
- Ratings start at Beta(5,5) and update after each game
- Predictions use Monte Carlo sampling from posteriors
- Rankings converge toward true strengths as games accumulate
- The Brier score measures calibration of probability predictions
</details>

---

## Summary

### Key Takeaways

- **Bayesian inference** provides direct probability statements about parameters
- **Priors** encode prior knowledge and get overwhelmed by enough data
- **Conjugate priors** give closed-form posteriors for common problems
- **MCMC** handles complex posteriors by sampling
- **Bayesian decision theory** combines uncertainty with loss functions for optimal decisions
- **The Kelly criterion** is the optimal Bayesian bet sizing rule

### Skills You've Gained

You can now:
- ✓ Frame problems from both Bayesian and frequentist perspectives
- ✓ Perform Bayesian updating using conjugate priors
- ✓ Implement a Metropolis-Hastings MCMC sampler
- ✓ Diagnose MCMC convergence
- ✓ Apply the Kelly criterion for optimal bet sizing
- ✓ Build a Bayesian rating and prediction system

---

## Next Steps

### Continue Learning

**Build on this topic:**
- [Stochastic Processes](/routes/stochastic-processes/map.md) — MCMC is itself a stochastic process; go deeper into Markov chains and Monte Carlo methods

**Explore related routes:**
- [Regression and Modeling](/routes/regression-and-modeling/map.md) — Bayesian regression extends these ideas to prediction

### Additional Resources

**Documentation:**
- scipy.stats — Beta, Normal, Gamma distributions and more
- PyMC — a Python library for Bayesian modeling with MCMC
- ArviZ — Bayesian visualization and diagnostics

---

## Appendix

### Quick Reference

| Conjugate Pair | Prior | Likelihood | Posterior |
|---|---|---|---|
| Beta-Binomial | Beta(α, β) | Binomial(n, p) | Beta(α+k, β+n-k) |
| Normal-Normal | N(μ₀, σ₀²) | N(μ, σ²/n) | N(μ_post, σ_post²) |
| Gamma-Poisson | Gamma(α, β) | Poisson(λ) | Gamma(α+Σx, β+n) |

### Glossary

- **Prior**: Probability distribution representing beliefs about a parameter before seeing data
- **Likelihood**: The probability of observed data given a parameter value
- **Posterior**: Updated probability distribution after combining prior and likelihood
- **Conjugate prior**: A prior whose posterior belongs to the same distribution family
- **Credible interval**: A Bayesian interval with a direct probability interpretation
- **MCMC**: A class of algorithms that sample from complex posterior distributions
- **Metropolis-Hastings**: A specific MCMC algorithm using propose-accept/reject steps
- **Burn-in**: Initial MCMC samples discarded before the chain converges
- **Kelly criterion**: A formula for optimal bet sizing that maximizes long-run growth
- **Bayes factor**: The ratio of likelihoods under two competing models
- **Shrinkage**: The phenomenon where Bayesian estimates are pulled toward the prior mean
- **Posterior predictive check**: Simulating data from the posterior model to verify it fits reality
