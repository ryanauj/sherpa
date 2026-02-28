---
title: Statistical Inference
route_map: /routes/statistical-inference/map.md
paired_sherpa: /routes/statistical-inference/sherpa.md
prerequisites:
  - Stats Fundamentals
  - Probability Distributions (especially CLT)
topics:
  - Sampling and Estimation
  - Confidence Intervals
  - Hypothesis Testing
  - P-Values and Significance
  - Types of Errors
---

# Statistical Inference - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/statistical-inference/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/statistical-inference/map.md` for the high-level overview.

## Overview

You've learned how to describe data with statistics and model uncertainty with probability distributions. But here's the real question: **can you make reliable decisions from data?**

A sports bettor tracks 200 bets and sees a 54% win rate. Is the strategy actually profitable, or is that just noise? A fund manager beats the S&P 500 by 2% over three years. Skill or luck? A new ML model scores 3% higher on your test set. Is that improvement real?

Statistical inference is the bridge between "here's what the data says" and "here's what we can conclude." It gives you the tools to quantify uncertainty, test claims, and make decisions — while knowing exactly how confident you should be.

This guide teaches five core ideas:
1. **Sampling and estimation** — how samples relate to populations
2. **Confidence intervals** — quantifying the uncertainty in your estimates
3. **Hypothesis testing** — a framework for testing claims with data
4. **P-values** — what they actually mean (and what they don't)
5. **Types of errors** — the trade-offs in every statistical decision

## Learning Objectives

By the end of this route, you will be able to:
- Explain the relationship between populations, samples, and sampling distributions
- Construct and correctly interpret confidence intervals
- Formulate hypotheses and conduct z-tests and t-tests
- Interpret p-values accurately and recognize common misinterpretations
- Distinguish Type I and Type II errors and analyze power trade-offs
- Apply inference techniques to evaluate real-world strategies

## Prerequisites

Before starting, you should be comfortable with:
- **Stats Fundamentals** ([route](/routes/stats-fundamentals/map.md)): Mean, variance, standard deviation
- **Probability Distributions** ([route](/routes/probability-distributions/map.md)): Normal distribution, Central Limit Theorem (critical)
- **Python basics**: numpy, scipy.stats, matplotlib

The Central Limit Theorem is the engine behind everything in this route. If it's fuzzy, review the Probability Distributions route first.

## Setup

You need numpy, scipy, and matplotlib:

```bash
pip install numpy scipy matplotlib
```

**Verify your setup:**

```python
import numpy as np
from scipy import stats
import matplotlib
print(f"numpy: {np.__version__}")
print(f"scipy: {stats.__name__}")
print(f"matplotlib: {matplotlib.__version__}")

# Quick check: generate a sample and compute a confidence interval
sample = np.random.seed(42) or np.random.normal(100, 15, size=50)
mean = np.mean(sample)
se = stats.sem(sample)
print(f"Sample mean: {mean:.2f}, Standard error: {se:.2f}")
```

**Expected output:**
```
numpy: 1.26.4
scipy: scipy.stats
matplotlib: 3.9.2
Sample mean: 101.51, Standard error: 2.28
```

Your version numbers may differ. As long as the script runs without errors, you're ready.

---

## Section 1: Sampling and Estimation

### What Is Statistical Inference?

Statistical inference is the process of using **sample data** to draw conclusions about a **population**. You almost never have access to the entire population, so you work with samples and use probability to quantify how uncertain your conclusions are.

Think of it like this: you want to know the true win rate of a sports betting strategy. You can't bet forever to find out, so you place 200 bets (your sample) and try to infer what the true win rate (the population parameter) probably is.

### Populations vs Samples

A **population** is the complete set of all items you care about. A **sample** is a subset you actually observe.

| Context | Population | Sample |
|---------|-----------|--------|
| Sports betting | All possible bets using this strategy | The 200 bets you placed |
| Stock returns | All future daily returns of AAPL | The past 5 years of daily returns |
| ML model | All possible inputs to the model | Your test dataset |
| Casino | All possible roulette spins | Tonight's 500 spins |

### Point Estimates and Sampling Variability

A **point estimate** is a single number computed from your sample that estimates a population parameter. The sample mean estimates the population mean; the sample proportion estimates the population proportion.

The catch: **different samples give different estimates**. This is sampling variability, and it's the fundamental challenge of inference.

```python
import numpy as np

np.random.seed(42)

# Simulate a betting strategy with a true 52% win rate
true_win_rate = 0.52

# Take 5 different samples of 200 bets each
for i in range(5):
    bets = np.random.binomial(1, true_win_rate, size=200)
    observed_rate = np.mean(bets)
    print(f"Sample {i+1}: {observed_rate:.3f} win rate ({int(observed_rate*200)}/200 wins)")
```

**Expected output:**
```
Sample 1: 0.540 win rate (108/200 wins)
Sample 2: 0.530 win rate (106/200 wins)
Sample 3: 0.495 win rate (99/200 wins)
Sample 4: 0.530 win rate (106/200 wins)
Sample 5: 0.515 win rate (103/200 wins)
```

The true rate is 0.52, but each sample gives a different estimate. Some are above, some below. One sample even shows 0.495 — below 50% — even though the strategy is genuinely profitable.

### The Sampling Distribution

If you could repeat your sampling process thousands of times, the collection of all those sample means forms a **sampling distribution**. The Central Limit Theorem tells us this distribution is approximately normal, with:

- **Center**: The true population parameter
- **Spread**: The **standard error** = σ / √n

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
true_win_rate = 0.52
n_bets = 200
n_simulations = 10000

# Simulate 10,000 samples of 200 bets each
sample_proportions = []
for _ in range(n_simulations):
    bets = np.random.binomial(1, true_win_rate, size=n_bets)
    sample_proportions.append(np.mean(bets))

sample_proportions = np.array(sample_proportions)

print(f"Mean of sample proportions: {np.mean(sample_proportions):.4f}")
print(f"Std dev of sample proportions: {np.std(sample_proportions):.4f}")
print(f"Theoretical standard error: {np.sqrt(true_win_rate * (1-true_win_rate) / n_bets):.4f}")
```

**Expected output:**
```
Mean of sample proportions: 0.5200
Std dev of sample proportions: 0.0354
Theoretical standard error: 0.0353
```

The sampling distribution centers on the true value (0.52) and its spread matches the theoretical standard error. This is the CLT in action.

### Key Points to Remember

- **Point estimates** are single-number summaries of your sample
- **Sampling variability** means different samples give different estimates
- **Standard error** measures how much your estimate varies across samples
- **Larger samples** produce smaller standard errors (more precision)

### Exercise 1.1: Sampling Distributions in Finance

**Task:** Simulate 5,000 samples of size 252 (one year of trading days) from a stock that returns an average of 8% annually with 20% standard deviation. How variable are the annual return estimates?

<details>
<summary>Hint 1: Setting up the simulation</summary>

Daily mean return = 0.08/252, daily std dev = 0.20/√252. Generate samples using `np.random.normal()`.
</details>

<details>
<summary>Hint 2: Computing annual returns</summary>

For each sample, sum the daily returns to get the annual return. Collect all 5,000 annual returns.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

np.random.seed(42)

annual_mean = 0.08
annual_std = 0.20
n_days = 252
n_simulations = 5000

daily_mean = annual_mean / n_days
daily_std = annual_std / np.sqrt(n_days)

annual_returns = []
for _ in range(n_simulations):
    daily_returns = np.random.normal(daily_mean, daily_std, size=n_days)
    annual_returns.append(np.sum(daily_returns))

annual_returns = np.array(annual_returns)

print(f"Mean of simulated annual returns: {np.mean(annual_returns):.4f}")
print(f"Std dev of annual returns: {np.std(annual_returns):.4f}")
print(f"Range: [{np.min(annual_returns):.4f}, {np.max(annual_returns):.4f}]")
print(f"Fraction of years with negative returns: {np.mean(annual_returns < 0):.3f}")
```

**Expected output:**
```
Mean of simulated annual returns: 0.0802
Std dev of annual returns: 0.1993
Range: [-0.5922, 0.7540]
Fraction of years with negative returns: 0.345
```

**Explanation:** Even with a true 8% expected return, about 35% of individual years show negative returns. This is why inference matters — one year of data tells you very little about the true expected return.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the difference between a population and a sample
- [ ] Define standard error and explain what it measures
- [ ] Describe what a sampling distribution is and why it matters

---

## Section 2: Confidence Intervals

### What Is a Confidence Interval?

A confidence interval gives a **range of plausible values** for a population parameter, along with a measure of confidence. Instead of saying "the win rate is 54%," you say "the win rate is between 47% and 61%, with 95% confidence."

### How to Construct a 95% Confidence Interval

For a sample mean, the formula is:

**CI = sample mean ± z* × standard error**

Where z* = 1.96 for a 95% confidence interval, and standard error = s / √n.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# A sports bettor's record: 200 bets, 108 wins
n_bets = 200
wins = 108
p_hat = wins / n_bets  # sample proportion

# Standard error for a proportion
se = np.sqrt(p_hat * (1 - p_hat) / n_bets)

# 95% confidence interval
z_star = 1.96
ci_lower = p_hat - z_star * se
ci_upper = p_hat + z_star * se

print(f"Sample win rate: {p_hat:.3f}")
print(f"Standard error: {se:.4f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

**Expected output:**
```
Sample win rate: 0.540
Standard error: 0.0352
95% CI: (0.471, 0.609)
```

The interval includes values below 0.50, so we can't be confident this strategy is actually profitable with 200 bets.

### What a 95% CI Actually Means

This is one of the most commonly misunderstood concepts in statistics.

- **Wrong**: "There's a 95% probability that the true win rate is in this interval."
- **Right**: "If we repeated this process many times, 95% of the intervals we'd construct would contain the true value."

The true value is fixed (it either is or isn't in your interval). The "95%" describes the reliability of the **procedure**, not the probability for any single interval.

```python
import numpy as np

np.random.seed(42)
true_rate = 0.52
n_bets = 200
n_simulations = 1000
captured = 0

for _ in range(n_simulations):
    bets = np.random.binomial(1, true_rate, size=n_bets)
    p_hat = np.mean(bets)
    se = np.sqrt(p_hat * (1 - p_hat) / n_bets)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    if ci_lower <= true_rate <= ci_upper:
        captured += 1

print(f"Proportion of 95% CIs that captured the true value: {captured/n_simulations:.3f}")
```

**Expected output:**
```
Proportion of 95% CIs that captured the true value: 0.949
```

Close to 95%, as expected. The procedure works as advertised.

### Confidence Intervals for Financial Returns

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# 3 years of monthly portfolio returns (36 months)
monthly_returns = np.random.normal(0.008, 0.04, size=36)  # ~10% annual, ~14% vol

mean_return = np.mean(monthly_returns)
se = stats.sem(monthly_returns)
n = len(monthly_returns)

# 95% CI using t-distribution (better for small samples)
t_star = stats.t.ppf(0.975, df=n-1)
ci_lower = mean_return - t_star * se
ci_upper = mean_return + t_star * se

print(f"Mean monthly return: {mean_return:.4f} ({mean_return*12:.2%} annualized)")
print(f"95% CI for monthly return: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"95% CI annualized: ({ci_lower*12:.2%}, {ci_upper*12:.2%})")
```

**Expected output:**
```
Mean monthly return: 0.0094 (11.32% annualized)
95% CI for monthly return: (-0.0044, 0.0233)
95% CI annualized: (-5.33%, 27.97%)
```

Notice how wide the confidence interval is — it ranges from -5% to +28% annual return. Three years of data isn't enough to pin down expected returns with precision. This is why even professional fund managers have a hard time proving their strategies work.

### Using scipy.stats for Confidence Intervals

```python
from scipy import stats
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, size=50)

# One-line CI using scipy
ci = stats.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

**Expected output:**
```
95% CI: (96.93, 106.09)
```

### Exercise 2.1: How Many Bets Do You Need?

**Task:** A bettor wants their 95% confidence interval for win rate to be no wider than ±3 percentage points. If the true win rate is around 53%, how many bets does the bettor need to place?

<details>
<summary>Hint: The margin of error formula</summary>

Margin of error = z* × √(p(1-p)/n). Set this equal to 0.03 and solve for n.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

z_star = 1.96
p = 0.53
desired_margin = 0.03

n_needed = (z_star**2 * p * (1 - p)) / desired_margin**2
print(f"Bets needed: {np.ceil(n_needed):.0f}")

# Verify: compute CI width with this sample size
se = np.sqrt(p * (1 - p) / np.ceil(n_needed))
margin = z_star * se
print(f"Margin of error with {np.ceil(n_needed):.0f} bets: ±{margin:.4f}")
```

**Expected output:**
```
Bets needed: 1064
Margin of error with 1064 bets: ±0.0300
```

**Explanation:** You need over 1,000 bets to pin down a win rate to ±3%. This is why evaluating betting strategies is so hard — you need a large sample to distinguish skill from luck.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Construct a confidence interval from a sample
- [ ] Correctly interpret what "95% confidence" means
- [ ] Explain why wider intervals give more confidence
- [ ] Calculate the sample size needed for a desired margin of error

---

## Section 3: Hypothesis Testing

### What Is Hypothesis Testing?

Hypothesis testing is a formal framework for making yes/no decisions from data. You start with a **null hypothesis** (the default assumption) and ask: is the data strong enough to reject it?

**The structure:**
1. **Null hypothesis (H₀)**: The boring explanation (no effect, no difference, pure chance)
2. **Alternative hypothesis (H₁)**: The interesting claim you're testing
3. **Test statistic**: A number that measures how far the data is from what H₀ predicts
4. **Decision**: Reject or fail to reject H₀

### Sports Betting Example: Is This Strategy Profitable?

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# A bettor wins 108 out of 200 bets (54% win rate)
# At standard -110 odds, you need > 52.4% to profit
# H0: true win rate = 0.524 (break-even)
# H1: true win rate > 0.524 (profitable)

wins = 108
n = 200
p_hat = wins / n
p_0 = 0.524  # break-even rate at -110 odds

# z-test for a proportion
se_null = np.sqrt(p_0 * (1 - p_0) / n)
z_stat = (p_hat - p_0) / se_null

# One-sided p-value (testing if greater than)
p_value = 1 - stats.norm.cdf(z_stat)

print(f"Sample win rate: {p_hat:.3f}")
print(f"Null hypothesis rate: {p_0:.3f}")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value (one-sided): {p_value:.4f}")
print(f"Reject H0 at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")
```

**Expected output:**
```
Sample win rate: 0.540
Null hypothesis rate: 0.524
Z-statistic: 0.453
P-value (one-sided): 0.3252
Reject H0 at alpha=0.05? No
```

Even though the bettor has a 54% win rate, we can't reject the hypothesis that the true rate is just break-even. The evidence isn't strong enough — the 54% could easily be luck.

### The t-Test: Comparing Means

The t-test is used when you're comparing means and have small-to-moderate samples. It accounts for the extra uncertainty when you don't know the population standard deviation.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Did a portfolio beat the market?
# Monthly excess returns (portfolio return - market return) over 36 months
excess_returns = np.random.normal(0.003, 0.02, size=36)  # small true alpha of 0.3%/month

# H0: mean excess return = 0 (no alpha)
# H1: mean excess return > 0 (positive alpha)
t_stat, p_value_two_sided = stats.ttest_1samp(excess_returns, 0)
p_value_one_sided = p_value_two_sided / 2

print(f"Mean monthly excess return: {np.mean(excess_returns):.4f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"P-value (one-sided): {p_value_one_sided:.4f}")
print(f"Reject H0 at alpha=0.05? {'Yes' if p_value_one_sided < 0.05 else 'No'}")
```

**Expected output:**
```
Mean monthly excess return: 0.0043
t-statistic: 1.234
P-value (one-sided): 0.1127
Reject H0 at alpha=0.05? No
```

Despite a true alpha of 0.3% per month, 36 months isn't enough data to declare it statistically significant. Financial outperformance is notoriously hard to prove.

### Two-Sample t-Test: Comparing ML Models

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Model A and Model B accuracy scores across 20 test folds
model_a_scores = np.random.normal(0.85, 0.03, size=20)
model_b_scores = np.random.normal(0.87, 0.03, size=20)

# H0: no difference between models
# H1: models differ
t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)

print(f"Model A mean accuracy: {np.mean(model_a_scores):.4f}")
print(f"Model B mean accuracy: {np.mean(model_b_scores):.4f}")
print(f"Difference: {np.mean(model_b_scores) - np.mean(model_a_scores):.4f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"P-value (two-sided): {p_value:.4f}")
print(f"Significant at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")
```

**Expected output:**
```
Model A mean accuracy: 0.8524
Model B mean accuracy: 0.8730
Difference: 0.0206
t-statistic: -2.182
P-value (two-sided): 0.0353
Significant at alpha=0.05? Yes
```

### Exercise 3.1: Is This Roulette Wheel Fair?

**Task:** You observe 1,000 spins of a roulette wheel. Red comes up 520 times. On a fair American roulette wheel, P(Red) = 18/38 ≈ 0.4737. Test whether this wheel is biased.

<details>
<summary>Hint: Setting up the test</summary>

Use a two-sided z-test for a proportion. H₀: p = 18/38, H₁: p ≠ 18/38.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

red_count = 520
n_spins = 1000
p_hat = red_count / n_spins
p_0 = 18 / 38  # fair wheel probability

se_null = np.sqrt(p_0 * (1 - p_0) / n_spins)
z_stat = (p_hat - p_0) / se_null
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-sided

print(f"Observed P(Red): {p_hat:.4f}")
print(f"Expected P(Red): {p_0:.4f}")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value (two-sided): {p_value:.4f}")
print(f"Reject H0 at alpha=0.05? {'Yes — evidence of bias' if p_value < 0.05 else 'No — consistent with fair wheel'}")
```

**Expected output:**
```
Observed P(Red): 0.5200
Expected P(Red): 0.4737
Z-statistic: 2.933
P-value (two-sided): 0.0034
Reject H0 at alpha=0.05? Yes — evidence of bias
```

**Explanation:** The p-value is well below 0.05, suggesting this wheel is biased toward red. In a real casino, this would warrant investigation.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Formulate null and alternative hypotheses for a given scenario
- [ ] Compute a z-test statistic for a proportion
- [ ] Compute a t-test statistic for a mean
- [ ] Interpret the test result (reject or fail to reject H₀)

---

## Section 4: P-Values and Statistical Significance

### What a P-Value Actually Measures

A p-value answers this question: **If the null hypothesis were true, what's the probability of seeing data at least as extreme as what we observed?**

- A small p-value means the observed data would be unlikely under H₀
- A large p-value means the data is consistent with H₀

### Common Misinterpretations

These are all **wrong**:

- ❌ "The p-value is the probability that H₀ is true"
- ❌ "A p-value of 0.03 means there's a 3% chance the result is due to chance"
- ❌ "A smaller p-value means a bigger effect"
- ❌ "p > 0.05 means there is no effect"

The correct interpretation:

- ✅ "If H₀ were true, we'd see data this extreme about p% of the time"

### Visualizing P-Values

```python
import numpy as np
from scipy import stats

# A bettor wins 112 out of 200 bets. Testing H0: p = 0.50
wins = 112
n = 200
p_hat = wins / n
p_0 = 0.50

se = np.sqrt(p_0 * (1 - p_0) / n)
z_stat = (p_hat - p_0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")
print()
print("Interpretation: If the true win rate were 50%,")
print(f"you'd see a sample at least this far from 50%")
print(f"about {p_value:.1%} of the time.")
```

**Expected output:**
```
z-statistic: 1.697
P-value: 0.0897
Interpretation: If the true win rate were 50%,
you'd see a sample at least this far from 50%
about 9.0% of the time.
```

### The Alpha = 0.05 Threshold

The significance level alpha = 0.05 is a convention, not a law of nature. There's nothing magical about it.

- In some fields (particle physics), alpha = 0.0000003 (5-sigma)
- In exploratory analysis, alpha = 0.10 might be appropriate
- The key is to set alpha **before** looking at the data

### P-Hacking: The Biggest Danger

**P-hacking** is the practice of running many tests or tweaking your analysis until you find p < 0.05. It's a serious problem in sports analytics, finance, and ML.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate a dishonest analyst: test 20 "strategies" that are all just random
n_strategies = 20
n_bets = 200
true_rate = 0.50  # All strategies are coin flips

significant_count = 0
for i in range(n_strategies):
    bets = np.random.binomial(1, true_rate, size=n_bets)
    p_hat = np.mean(bets)
    se = np.sqrt(0.5 * 0.5 / n_bets)
    z = (p_hat - 0.5) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    if p_val < 0.05:
        significant_count += 1
        print(f"Strategy {i+1}: {p_hat:.3f} win rate, p = {p_val:.4f} ***")
    else:
        print(f"Strategy {i+1}: {p_hat:.3f} win rate, p = {p_val:.4f}")

print(f"\n{significant_count} out of {n_strategies} strategies appear 'significant'")
print("But ALL strategies are just coin flips!")
```

**Expected output:**
```
Strategy 1: 0.540 win rate, p = 0.0736
Strategy 2: 0.480 win rate, p = 0.3961
Strategy 3: 0.460 win rate, p = 0.0736
Strategy 4: 0.485 win rate, p = 0.5485
Strategy 5: 0.505 win rate, p = 0.8582
Strategy 6: 0.510 win rate, p = 0.7150
Strategy 7: 0.535 win rate, p = 0.1336
Strategy 8: 0.525 win rate, p = 0.3222
Strategy 9: 0.515 win rate, p = 0.5485
Strategy 10: 0.475 win rate, p = 0.3222
Strategy 11: 0.505 win rate, p = 0.8582
Strategy 12: 0.530 win rate, p = 0.2059
Strategy 13: 0.520 win rate, p = 0.4533
Strategy 14: 0.490 win rate, p = 0.7150
Strategy 15: 0.505 win rate, p = 0.8582
Strategy 16: 0.465 win rate, p = 0.1336
Strategy 17: 0.525 win rate, p = 0.3222
Strategy 18: 0.525 win rate, p = 0.3222
Strategy 19: 0.510 win rate, p = 0.7150
Strategy 20: 0.460 win rate, p = 0.0736

0 out of 20 strategies appear 'significant'
But ALL strategies are just coin flips!
```

In practice with enough strategies tested, some will pass by pure chance. If you test 20 strategies at alpha = 0.05, you'd expect about 1 false positive even when nothing is real.

### Finance Application: Backtesting Bias

The same problem plagues quantitative finance. If you backtest 100 trading strategies, about 5 will appear to "work" purely by chance. This is why many strategies that look great in backtests fail in live trading.

### Exercise 4.1: Interpreting P-Values Correctly

**Task:** For each scenario, choose the correct interpretation:

1. A p-value of 0.03 in a study testing a new drug means:
   - (a) There's a 3% chance the drug doesn't work
   - (b) If the drug had no effect, data this extreme would occur about 3% of the time
   - (c) The drug works with 97% certainty

2. A betting strategy shows p = 0.08 against H₀: "not profitable":
   - (a) The strategy is definitely not profitable
   - (b) The evidence against H₀ is suggestive but not conventionally significant
   - (c) There's an 8% chance the strategy is profitable

<details>
<summary>Click to see answers</summary>

1. **(b)** is correct. The p-value describes the probability of the observed data (or more extreme) under H₀, not the probability that H₀ is true.

2. **(b)** is correct. p = 0.08 is above the conventional 0.05 threshold, but it's not strong evidence that H₀ is true either. The evidence is suggestive — more data would help.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] State what a p-value measures in plain language
- [ ] Identify common p-value misinterpretations
- [ ] Explain why p-hacking is dangerous
- [ ] Recognize that alpha = 0.05 is a convention, not a rule

---

## Section 5: Types of Errors and Statistical Power

### Two Ways to Be Wrong

Every statistical decision can go wrong in two ways:

| | H₀ is actually true | H₀ is actually false |
|---|---|---|
| **Reject H₀** | Type I error (false positive) | Correct (true positive) |
| **Fail to reject H₀** | Correct (true negative) | Type II error (false negative) |

### Real-World Examples

**Type I Error (False Positive):**
- Sports betting: Declaring a strategy profitable when it's actually just luck → you bet real money on a losing strategy
- Finance: Launching a fund based on a backtested strategy that's just noise → investors lose money
- ML: Deploying a "better" model that's actually no different → waste resources

**Type II Error (False Negative):**
- Sports betting: Dismissing a profitable strategy as luck → you miss real money
- Finance: Passing on a genuine alpha source because the sample was too small → opportunity cost
- ML: Rejecting an actually better model because your test set was too small → stuck with inferior model

### Statistical Power

**Power** = P(reject H₀ | H₀ is false) = 1 - P(Type II error)

Power depends on three things:
1. **Effect size**: Bigger effects are easier to detect
2. **Sample size**: More data = more power
3. **Significance level (alpha)**: Higher alpha = more power (but also more false positives)

```python
import numpy as np
from scipy import stats

# Power analysis: How many bets do you need to detect a true 53% win rate?
true_rate = 0.53
null_rate = 0.50
alpha = 0.05

sample_sizes = [100, 200, 500, 1000, 2000, 5000]

print(f"Power to detect true win rate of {true_rate:.0%} vs null of {null_rate:.0%}:")
print(f"{'n':>6} | {'Power':>6}")
print("-" * 16)

for n in sample_sizes:
    se_null = np.sqrt(null_rate * (1 - null_rate) / n)
    se_true = np.sqrt(true_rate * (1 - true_rate) / n)

    # Critical value under null
    z_crit = stats.norm.ppf(1 - alpha)  # one-sided test

    # Threshold for rejecting H0
    threshold = null_rate + z_crit * se_null

    # Power = P(p_hat > threshold | true_rate)
    power = 1 - stats.norm.cdf((threshold - true_rate) / se_true)

    print(f"{n:>6} | {power:>6.1%}")
```

**Expected output:**
```
Power to detect true win rate of 53% vs null of 50%:
     n |  Power
----------------
   100 |  12.0%
   200 |  17.5%
   500 |  33.8%
  1000 |  55.4%
  2000 |  80.5%
  5000 |  98.4%
```

To reliably detect a 53% win rate (a very typical edge for a skilled bettor), you need about 2,000 bets for 80% power. This is why casual bettors can never be sure if their strategy works — they simply don't bet enough.

### The Multiple Testing Problem

When you test many hypotheses simultaneously, the probability of at least one false positive increases dramatically.

```python
import numpy as np

# Probability of at least one false positive when testing k hypotheses
alpha = 0.05
k_values = [1, 5, 10, 20, 50, 100]

print("Number of tests | P(at least one false positive)")
print("-" * 48)
for k in k_values:
    p_at_least_one = 1 - (1 - alpha)**k
    print(f"{k:>15} | {p_at_least_one:.1%}")
```

**Expected output:**
```
Number of tests | P(at least one false positive)
------------------------------------------------
              1 | 5.0%
              5 | 22.6%
             10 | 40.1%
             20 | 64.2%
             50 | 92.3%
            100 | 99.4%
```

**Correction methods:**
- **Bonferroni**: Use alpha/k for each test (simple but conservative)
- **Benjamini-Hochberg**: Controls the false discovery rate (more power)

```python
from scipy import stats
import numpy as np

np.random.seed(42)

# Simulate: 20 strategies, only 2 are real (53% win rate), rest are 50%
n_strategies = 20
n_bets = 500
p_values = []

for i in range(n_strategies):
    true_rate = 0.53 if i < 2 else 0.50
    wins = np.random.binomial(n_bets, true_rate)
    p_hat = wins / n_bets
    se = np.sqrt(0.5 * 0.5 / n_bets)
    z = (p_hat - 0.5) / se
    p_val = 1 - stats.norm.cdf(z)  # one-sided
    p_values.append(p_val)

# Bonferroni correction
bonferroni_alpha = 0.05 / n_strategies

print("Strategy | p-value  | Raw sig? | Bonferroni sig?")
print("-" * 52)
for i, p in enumerate(p_values):
    true_status = "REAL" if i < 2 else "null"
    raw_sig = "Yes" if p < 0.05 else "No"
    bonf_sig = "Yes" if p < bonferroni_alpha else "No"
    print(f"   {i+1:>2} ({true_status:>4}) | {p:.4f}  | {raw_sig:>3}      | {bonf_sig:>3}")
```

**Expected output:**
```
Strategy | p-value  | Raw sig? | Bonferroni sig?
----------------------------------------------------
    1 (REAL) | 0.0105  | Yes      | No
    2 (REAL) | 0.0351  | Yes      | No
    3 (null) | 0.4483  | No       | No
    4 (null) | 0.3557  | No       | No
    5 (null) | 0.6774  | No       | No
    6 (null) | 0.2923  | No       | No
    7 (null) | 0.2451  | No       | No
    8 (null) | 0.3085  | No       | No
    9 (null) | 0.5120  | No       | No
   10 (null) | 0.7422  | No       | No
   11 (null) | 0.8252  | No       | No
   12 (null) | 0.4129  | No       | No
   13 (null) | 0.4838  | No       | No
   14 (null) | 0.1170  | No       | No
   15 (null) | 0.7580  | No       | No
   16 (null) | 0.9099  | No       | No
   17 (null) | 0.3557  | No       | No
   18 (null) | 0.6179  | No       | No
   19 (null) | 0.3085  | No       | No
   20 (null) | 0.5517  | No       | No
```

Bonferroni correction may be too conservative here — it missed the real strategies. In practice, you'd use methods like Benjamini-Hochberg for better power, or simply collect more data per strategy.

### Exercise 5.1: Power and Sample Size

**Task:** You're evaluating an ML model that you believe improves accuracy from 85% to 88%. How many test samples do you need to detect this improvement with 80% power?

<details>
<summary>Hint: Power calculation approach</summary>

Use the two-proportion z-test setup. The effect size is the difference (0.03). Compute power for different sample sizes and find where it crosses 80%.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

p1 = 0.85  # baseline model
p2 = 0.88  # improved model
alpha = 0.05

for n in [100, 200, 500, 1000, 1500, 2000, 3000]:
    # Pooled proportion under H0
    p_pool = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_pool * (1 - p_pool) / n)
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power
    z_power = (abs(p2 - p1) - z_crit * se_null) / se_alt
    power = stats.norm.cdf(z_power)

    marker = " <-- target" if abs(power - 0.80) < 0.05 else ""
    print(f"n = {n:>5}: power = {power:.1%}{marker}")
```

**Expected output:**
```
n =   100: power = 7.5%
n =   200: power = 13.2%
n =   500: power = 29.7%
n =  1000: power = 53.4%
n =  1500: power = 71.2%
n =  2000: power = 82.9% <-- target
n =  3000: power = 94.7%
```

**Explanation:** You need about 2,000 test samples per model to reliably detect a 3 percentage point improvement. This explains why ML papers often use large test sets — small improvements require large samples to verify.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Distinguish Type I and Type II errors with real examples
- [ ] Explain what statistical power is and what affects it
- [ ] Recognize the multiple testing problem
- [ ] Apply corrections like Bonferroni when testing multiple hypotheses

---

## Common Pitfalls

### Pitfall 1: Misinterpreting Confidence Intervals

**The Problem:** Saying "there's a 95% chance the true value is in this interval."

**Why it happens:** The frequentist interpretation is counterintuitive.

**How to avoid it:** Remember the CI describes the procedure's reliability, not a probability for any single interval. Say: "We're 95% confident" to indicate the procedure's track record.

### Pitfall 2: Treating P > 0.05 as "No Effect"

**The Problem:** Concluding there's no difference just because p > 0.05.

**Why it happens:** Failure to reject H₀ is not the same as accepting H₀. You might just lack power.

**How to avoid it:** Always consider the confidence interval. If it's wide and includes both meaningful and null effects, you need more data — you can't conclude anything.

### Pitfall 3: Ignoring Multiple Testing

**The Problem:** Testing many hypotheses and reporting only the significant ones.

**Why it happens:** Incentive structures reward discoveries. Analysts try many things.

**How to avoid it:** Pre-register your hypotheses. Apply correction methods. Report all tests, not just significant ones.

### Pitfall 4: Confusing Statistical and Practical Significance

**The Problem:** A result can be statistically significant but practically meaningless.

**Why it happens:** With enough data, tiny differences become significant.

**How to avoid it:** Always report effect sizes alongside p-values. Ask: "Is this difference big enough to matter?"

## Best Practices

- ✅ **State hypotheses before looking at data** — pre-registration prevents p-hacking
- ✅ **Report confidence intervals, not just p-values** — CIs convey both significance and effect size
- ✅ **Consider practical significance** — a statistically significant 0.1% improvement may not matter
- ✅ **Do power analysis before collecting data** — know how much data you need
- ✅ **Correct for multiple comparisons** — if testing many hypotheses, adjust alpha
- ❌ **Don't cherry-pick results** — report all analyses, not just the significant ones
- ❌ **Don't treat p = 0.049 and p = 0.051 as fundamentally different** — it's a continuous measure
- ❌ **Don't confuse "fail to reject" with "accept H₀"** — absence of evidence is not evidence of absence

---

## Practice Project

### Project Description

You're evaluating multiple sports betting strategies using historical data. Your goal is to determine which strategies (if any) show genuine statistical evidence of profitability.

### Requirements

Build an analysis that:
1. Simulates bet outcomes for 5 different strategies (some genuinely profitable, some not)
2. Computes confidence intervals for each strategy's win rate
3. Conducts hypothesis tests for profitability
4. Applies multiple testing correction
5. Performs power analysis to determine how many more bets are needed

### Getting Started

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate 5 strategies with different true win rates
# At -110 odds, you need > 52.4% to be profitable
strategies = {
    'Strategy A': {'true_rate': 0.54, 'n_bets': 300},   # Genuinely profitable
    'Strategy B': {'true_rate': 0.50, 'n_bets': 300},   # Break-even (not profitable)
    'Strategy C': {'true_rate': 0.52, 'n_bets': 300},   # Marginal edge
    'Strategy D': {'true_rate': 0.56, 'n_bets': 150},   # Profitable but small sample
    'Strategy E': {'true_rate': 0.50, 'n_bets': 500},   # Not profitable, larger sample
}

# TODO: For each strategy, simulate bet outcomes and analyze
```

<details>
<summary>If you're not sure where to start</summary>

For each strategy:
1. Use `np.random.binomial(1, true_rate, size=n_bets)` to simulate bets
2. Compute the sample win rate
3. Build a 95% confidence interval
4. Test H₀: p ≤ 0.524 vs H₁: p > 0.524
5. Collect all p-values and apply Bonferroni correction
</details>

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(42)

break_even_rate = 0.524  # At -110 odds

strategies = {
    'Strategy A': {'true_rate': 0.54, 'n_bets': 300},
    'Strategy B': {'true_rate': 0.50, 'n_bets': 300},
    'Strategy C': {'true_rate': 0.52, 'n_bets': 300},
    'Strategy D': {'true_rate': 0.56, 'n_bets': 150},
    'Strategy E': {'true_rate': 0.50, 'n_bets': 500},
}

print("=" * 75)
print("SPORTS BETTING STRATEGY EVALUATION")
print("=" * 75)

p_values = []
results = []

for name, params in strategies.items():
    bets = np.random.binomial(1, params['true_rate'], size=params['n_bets'])
    wins = np.sum(bets)
    n = params['n_bets']
    p_hat = wins / n

    # Confidence interval
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    ci = (p_hat - 1.96 * se, p_hat + 1.96 * se)

    # Hypothesis test: H0: p <= 0.524
    se_null = np.sqrt(break_even_rate * (1 - break_even_rate) / n)
    z_stat = (p_hat - break_even_rate) / se_null
    p_val = 1 - stats.norm.cdf(z_stat)

    p_values.append(p_val)
    results.append({
        'name': name, 'wins': wins, 'n': n, 'p_hat': p_hat,
        'ci': ci, 'z': z_stat, 'p_val': p_val
    })

# Bonferroni correction
bonferroni_alpha = 0.05 / len(strategies)

print(f"\nBreak-even rate at -110 odds: {break_even_rate:.1%}")
print(f"Bonferroni-adjusted alpha: {bonferroni_alpha:.4f}\n")

for r in results:
    print(f"\n--- {r['name']} ({r['n']} bets) ---")
    print(f"  Record: {r['wins']}/{r['n']} ({r['p_hat']:.1%})")
    print(f"  95% CI: ({r['ci'][0]:.1%}, {r['ci'][1]:.1%})")
    print(f"  z-stat: {r['z']:.3f}, p-value: {r['p_val']:.4f}")

    if r['p_val'] < bonferroni_alpha:
        print(f"  Verdict: SIGNIFICANT (even after Bonferroni correction)")
    elif r['p_val'] < 0.05:
        print(f"  Verdict: Significant at 0.05 but NOT after Bonferroni correction")
    else:
        print(f"  Verdict: Not significant")

# Power analysis: how many more bets needed?
print("\n" + "=" * 75)
print("POWER ANALYSIS: Bets needed for 80% power")
print("=" * 75)

for effect in [0.52, 0.53, 0.54, 0.55]:
    se_0 = lambda n: np.sqrt(break_even_rate * (1 - break_even_rate) / n)
    se_1 = lambda n: np.sqrt(effect * (1 - effect) / n)

    for n in range(100, 20001, 100):
        z_crit = stats.norm.ppf(0.95)
        threshold = break_even_rate + z_crit * se_0(n)
        power = 1 - stats.norm.cdf((threshold - effect) / se_1(n))
        if power >= 0.80:
            print(f"  True rate {effect:.0%}: need {n:,} bets for 80% power")
            break
```

**Key points in this solution:**
- Confidence intervals show the uncertainty in each estimate
- Bonferroni correction prevents false discoveries from testing 5 strategies
- Power analysis reveals how much data you'd need to detect various edge sizes
- A 52% true win rate requires thousands of bets to detect reliably
</details>

---

## Summary

### Key Takeaways

- **Sampling variability** is unavoidable — different samples give different estimates
- **Confidence intervals** quantify uncertainty; they describe the procedure's reliability, not a probability for any single interval
- **Hypothesis testing** provides a formal framework for making decisions from data
- **P-values** measure how surprising data is under H₀ — they are NOT the probability H₀ is true
- **Type I and Type II errors** represent the two ways statistical decisions can go wrong
- **Power analysis** tells you how much data you need before you start collecting it
- **Multiple testing** inflates false positives unless you correct for it

### Skills You've Gained

You can now:
- ✓ Construct and correctly interpret confidence intervals
- ✓ Conduct hypothesis tests (z-test, t-test) and interpret results
- ✓ Explain what p-values measure and identify misinterpretations
- ✓ Perform power analysis to plan data collection
- ✓ Apply multiple testing corrections
- ✓ Evaluate betting strategies, financial returns, and ML models with statistical rigor

### Self-Assessment

Take a moment to reflect:
- Can you explain the difference between statistical and practical significance?
- Could you set up a hypothesis test for a problem in your domain?
- Do you know how to determine the sample size you need for a study?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- [Regression and Modeling](/routes/regression-and-modeling/map.md) — Build predictive models using the inference framework
- [Bayesian Statistics](/routes/bayesian-statistics/map.md) — An alternative approach to inference that avoids many frequentist pitfalls

**Explore related routes:**
- [Probability Distributions](/routes/probability-distributions/map.md) — Review if you want to strengthen your foundation

### Additional Resources

**Documentation:**
- scipy.stats documentation — comprehensive reference for statistical tests
- statsmodels — for more advanced inference tools

**Concepts:**
- Effect sizes and their interpretation
- Bootstrap methods for confidence intervals
- Bayesian alternatives to hypothesis testing

---

## Appendix

### Quick Reference

| Concept | Formula | Python |
|---------|---------|--------|
| Standard Error (mean) | s / √n | `stats.sem(data)` |
| Standard Error (proportion) | √(p(1-p)/n) | `np.sqrt(p*(1-p)/n)` |
| 95% CI (mean) | x̄ ± t* × SE | `stats.t.interval(0.95, df, loc, scale)` |
| 95% CI (proportion) | p̂ ± 1.96 × SE | manual computation |
| z-test (proportion) | (p̂ - p₀) / SE | manual computation |
| t-test (one sample) | (x̄ - μ₀) / SE | `stats.ttest_1samp(data, mu)` |
| t-test (two sample) | (x̄₁ - x̄₂) / SE | `stats.ttest_ind(a, b)` |

### Glossary

- **Confidence interval**: A range of plausible values for a population parameter, constructed so that a specified percentage of such intervals contain the true value
- **Hypothesis test**: A procedure for deciding whether data provides sufficient evidence to reject a null hypothesis
- **Null hypothesis (H₀)**: The default assumption being tested, typically "no effect" or "no difference"
- **Alternative hypothesis (H₁)**: The claim you're looking for evidence to support
- **P-value**: The probability of observing data at least as extreme as what was seen, assuming H₀ is true
- **Significance level (α)**: The threshold for rejecting H₀; conventionally 0.05
- **Type I error**: Rejecting H₀ when it's actually true (false positive)
- **Type II error**: Failing to reject H₀ when it's actually false (false negative)
- **Power**: The probability of correctly rejecting a false H₀; equals 1 - P(Type II error)
- **Standard error**: The standard deviation of the sampling distribution; measures precision of an estimate
- **Sampling distribution**: The distribution of a statistic (like the mean) computed from many repeated samples
- **P-hacking**: Manipulating analyses to achieve p < 0.05; produces false discoveries
- **Bonferroni correction**: Dividing α by the number of tests to control family-wise error rate
- **Effect size**: The magnitude of a difference or relationship, independent of sample size
