---
title: Statistical Inference
route_map: /routes/statistical-inference/map.md
paired_guide: /routes/statistical-inference/guide.md
topics:
  - Sampling and Estimation
  - Confidence Intervals
  - Hypothesis Testing
  - P-Values and Significance
  - Types of Errors
---

# Statistical Inference - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach statistical inference -- the science of drawing conclusions about populations from samples. Every concept is motivated by real-world applications in sports betting, casino gambling, finance, and machine learning before any formula appears.

**Route Map**: See `/routes/statistical-inference/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/statistical-inference/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain the relationship between populations, samples, and sampling distributions
- Construct and correctly interpret confidence intervals
- Formulate null and alternative hypotheses and conduct z-tests and t-tests
- Interpret p-values accurately and recognize common misuses
- Distinguish Type I and Type II errors and analyze power trade-offs
- Apply inference techniques to evaluate betting strategies, financial returns, and ML model comparisons

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/statistical-inference/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Completed the Stats Fundamentals route (mean, variance, standard deviation, distributions)
- Completed the Probability Distributions route (especially the Central Limit Theorem)
- Python basics and comfort with numpy and scipy.stats

**If prerequisites are missing**: If they haven't covered the Central Limit Theorem, suggest they work through Probability Distributions first -- the CLT is the engine that makes confidence intervals and hypothesis testing work. If they lack stats fundamentals, point them to Stats Fundamentals. If scipy isn't installed, help them install it (`pip install scipy numpy matplotlib`).

### Audience Context
The target learner understands descriptive statistics and probability distributions, including the Central Limit Theorem. They want to move from "describing data" to "making decisions from data." They may be motivated by sports analytics, financial modeling, data science, or general curiosity about how scientific claims are evaluated.

Use their domain interests:
- Sports betting: Can you tell if a strategy is profitable or just lucky?
- Casino games: Is the house edge real, or could this wheel be biased?
- Finance: Did this portfolio actually beat the market, or was it random noise?
- ML/Data Science: Is model A really better than model B, or just lucky on this test set?

Always ground abstract concepts in these concrete scenarios BEFORE introducing notation or formulas. The learner should understand WHY a technique exists before learning HOW it works.

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
- Present 3-4 answer options with plausible distractors based on common misconceptions
- Example: "A 95% confidence interval for a team's win rate is [0.48, 0.62]. Which is correct? A) There is a 95% probability the true win rate is in this interval B) If we repeated this study many times, about 95% of the intervals would contain the true win rate C) The team wins between 48% and 62% of games D) We are 95% sure the team is above .500"

**Explanation Questions:**
- Ask the learner to explain concepts in their own words
- Example: "In your own words, explain why we say 'fail to reject H0' instead of 'accept H0'."

**Prediction Questions:**
- Show a scenario and ask what the result will be before computing
- Example: "If we double the sample size, what happens to the width of the confidence interval?"

**Code Questions:**
- Ask the learner to write functions that implement inference procedures
- Example: "Write a function that computes a confidence interval for a proportion."

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after CI and hypothesis testing definitions
- Use explanation questions for p-value interpretation and error types
- Use prediction questions before running simulations
- Use code questions for computing CIs, test statistics, and power analysis

---

## Teaching Flow

### Introduction

**What to Cover:**
- Statistical inference is the bridge between "here's what my data looks like" (descriptive stats) and "here's what I can conclude" (decision-making)
- The core problem: you observe a sample, but you care about the population. How do you reason from one to the other?
- By the end, they'll be able to rigorously evaluate whether a sports betting strategy is genuinely profitable, whether a trading strategy beats the market, or whether one ML model is truly better than another

**Opening Questions to Assess Level:**
1. "Have you encountered hypothesis testing or p-values before? Even informally?"
2. "What's your main motivation -- sports analytics, finance, data science, general statistics?"
3. "Do you remember the Central Limit Theorem from the probability distributions route?"

**Adapt based on responses:**
- If they've seen hypothesis testing: Move faster through setup, focus on correct interpretation and common pitfalls
- If motivated by sports betting: Lead with betting examples, use casino scenarios for intuition
- If motivated by finance: Lead with portfolio returns and market comparisons
- If motivated by ML: Frame everything as "is this model improvement real or noise?"
- If CLT is fuzzy: Spend extra time in Section 1 on sampling distributions -- everything else depends on it

**Good opening framing:**
"You've learned to describe data -- means, variances, distributions. But here's the question that actually matters: when you see a pattern in your data, is it real or is it noise? If a sports bettor wins 55% of bets over 200 games, is that skill or luck? If a stock portfolio returns 12% when the market returns 10%, did the manager actually beat the market? Statistical inference gives you rigorous tools to answer these questions. That's what this route is about."

---

### Setup Verification

**Check scipy:**
```bash
python -c "import scipy.stats; print(f'scipy version: {scipy.__version__}')"
```

**Check numpy and matplotlib:**
```bash
python -c "import numpy as np; import matplotlib; print(f'numpy: {np.__version__}, matplotlib: {matplotlib.__version__}')"
```

**If not installed:**
```bash
pip install scipy numpy matplotlib
```

**Quick CLT Refresher:**
If the learner's CLT understanding seems shaky, run this quick demo before proceeding:

```python
import numpy as np

# The CLT in action: means of random samples form a normal distribution
population = np.random.exponential(scale=5, size=100000)
sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(1000)]
print(f"Population mean: {np.mean(population):.2f}")
print(f"Mean of sample means: {np.mean(sample_means):.2f}")
print(f"Std of sample means: {np.std(sample_means):.2f}")
print(f"Population std / sqrt(50): {np.std(population) / np.sqrt(50):.2f}")
```

If the last two numbers are close, the CLT is working. Make sure the learner sees this before moving on -- it's the foundation for everything that follows.

---

### Section 1: Sampling and Estimation

**Core Concept to Teach:**
You almost never observe an entire population. You observe a sample and use it to estimate population parameters. The sampling distribution tells you how much your estimate would vary across different samples, and that variability is what makes inference possible.

**How to Explain:**
1. Start with a concrete scenario: "Imagine you want to know the true win rate of the Kansas City Chiefs. The 'population' is every game they'd ever play under current conditions -- infinitely many hypothetical games. You only observe one season of 17 games. That season is a sample."
2. Point estimates: "Your best guess for the true win rate is the sample win rate. If they won 11 of 17, your point estimate is 11/17 = 0.647. Simple -- but how confident should you be?"
3. Sampling distributions: "If the Chiefs played 1000 different 17-game seasons (same true ability), they wouldn't go 11-6 every time. Sometimes 12-5, sometimes 9-8. The distribution of those sample win rates across hypothetical seasons is the sampling distribution. It tells you how much your estimate would bounce around."
4. Standard error: "The standard deviation of the sampling distribution is called the standard error. It measures the precision of your estimate. More games (bigger sample) = smaller standard error = more precise estimate."

**Sports Example:**
"Can you judge an NBA player's true three-point shooting percentage from 10 games? From 50? From 100? With 10 games, you might see anywhere from 20% to 55% even if the true rate is 38%. With 100 games, you'd see something much closer to the truth. The sampling distribution gets narrower as n increases."

**Finance Example:**
"A hedge fund reports 15% annual returns over 3 years. Is the true expected return 15%? With only 3 data points and high volatility, the standard error is enormous. The true expected return could easily be 5% or 25%. You need more data -- or you need to accept wide uncertainty."

**ML Example:**
"You test a model on a holdout set of 500 examples and get 82% accuracy. That's a point estimate. If you had a different holdout set of 500 examples from the same distribution, you'd get a slightly different accuracy. The sampling distribution of accuracy scores tells you how much to trust that 82%."

**Common Misconceptions:**
- "My sample IS the population" -> Clarify: Your sample is one realization. If you collected data again, you'd get different numbers. The population is the theoretical entity you're trying to learn about.
- "Bigger sample is always better" -> Clarify: Bigger is better, but with diminishing returns. Standard error shrinks as 1/sqrt(n), so going from 100 to 400 observations cuts the error in half. Going from 10,000 to 40,000 also cuts it in half. The first improvement is more impactful in practice.
- "A biased sample becomes good if it's large enough" -> Clarify: Sample size fixes random error, not systematic bias. Surveying 10 million Twitter users doesn't tell you what all Americans think.

**Verification Questions:**
1. "If a basketball player shoots 40% from three over 20 games, what's the point estimate of their true shooting percentage? How confident should you feel?"
2. "What happens to the standard error if you quadruple the sample size?"
3. "Why can't you just look at more data to fix a biased sample?"

**Good answer indicators:**
- They recognize that 40% is the point estimate but express appropriate uncertainty
- They say the standard error is halved (1/sqrt(4) = 1/2)
- They distinguish between random error (fixed by more data) and systematic bias (not fixed by more data)

**If they struggle:**
- Use a physical analogy: "Imagine pulling colored marbles from a bag. Each handful is a sample. Sometimes you get more red, sometimes more blue. The more marbles you grab, the closer your proportion gets to the true proportion in the bag."
- Run a simulation: generate 1000 samples of size n from a known population, plot the distribution of sample means, show how it narrows as n increases
- Emphasize that this is about repeated sampling -- "if we did this again and again"

**Exercise 1.1: Simulate a Sampling Distribution**
Present this exercise: "Write Python code that simulates a population of roulette wheel outcomes (38 slots, 18 red, 18 black, 2 green). Draw samples of size 50, 200, and 1000. For each sample size, compute the sample proportion of red outcomes 5000 times and plot the sampling distribution. What happens as sample size increases?"

**How to Guide Them:**
1. First ask: "How would you set up the simulation?"
2. If stuck on the population: "Think of each spin as a Bernoulli trial with p = 18/38"
3. If stuck on sampling: "Use np.random.binomial(n, p, size=5000) / n to get 5000 sample proportions"
4. If stuck on visualization: "Use plt.hist() to plot the three distributions side by side"

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

p_red = 18 / 38  # true probability of red on a roulette wheel

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n in zip(axes, [50, 200, 1000]):
    sample_proportions = np.random.binomial(n, p_red, size=5000) / n
    ax.hist(sample_proportions, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(p_red, color='red', linestyle='--', label=f'True p = {p_red:.4f}')
    ax.set_title(f'n = {n}, SE = {np.std(sample_proportions):.4f}')
    ax.set_xlabel('Sample proportion of red')
    ax.legend()
plt.tight_layout()
plt.show()
```

**After exercise, ask:**
- "What happened to the spread of the distributions as n increased?"
- "If you were a casino owner, why would you want lots of spins?"

---

### Section 2: Confidence Intervals

**Core Concept to Teach:**
A confidence interval gives a range of plausible values for an unknown population parameter. A 95% confidence interval means: if you repeated the sampling process many times, about 95% of the intervals you'd construct would contain the true parameter. It does NOT mean there's a 95% probability the parameter is in this particular interval.

**How to Explain:**
1. Start with the problem: "A point estimate is a single number -- your best guess. But it doesn't tell you how uncertain you are. A confidence interval adds that uncertainty: 'We estimate the win rate is 0.55, with a 95% CI of [0.48, 0.62].' That range communicates how much the estimate could plausibly vary."
2. Construction using CLT: "The CLT tells us that sample means are approximately normally distributed. A 95% CI for a mean is: sample_mean +/- 1.96 * standard_error. The 1.96 comes from the normal distribution -- 95% of a normal distribution falls within 1.96 standard deviations of the mean."
3. The correct interpretation: "The confidence level refers to the procedure, not this specific interval. Imagine constructing 100 different 95% CIs from 100 different samples. About 95 of those intervals would contain the true parameter. About 5 would miss it. You don't know if THIS interval is one of the 95 or one of the 5."

**Sports Betting Example:**
"A bettor wins 110 out of 200 bets. Point estimate: 55% win rate. Is this bettor actually skilled, or could a 50% (no-skill) bettor get this result by luck? The 95% CI for the true win rate is:"

```python
import numpy as np
from scipy import stats

wins, n = 110, 200
p_hat = wins / n
se = np.sqrt(p_hat * (1 - p_hat) / n)
ci_lower = p_hat - 1.96 * se
ci_upper = p_hat + 1.96 * se
print(f"Win rate: {p_hat:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

Walk through: "The CI is [0.481, 0.619]. Since 0.50 is inside this interval, we can't rule out that this bettor has no real edge. The sample size isn't large enough to distinguish 55% skill from 50% luck."

**Finance Example:**
"A portfolio returned an average of 1.2% per month over 36 months with a standard deviation of 4%. The 95% CI for the true monthly return is:"

```python
import numpy as np

x_bar = 1.2   # sample mean monthly return (%)
s = 4.0       # sample standard deviation (%)
n = 36
se = s / np.sqrt(n)
ci_lower = x_bar - 1.96 * se
ci_upper = x_bar + 1.96 * se
print(f"Mean monthly return: {x_bar:.1f}%")
print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
```

Walk through: "The CI is [-0.11%, 2.51%]. The interval includes zero, which means we can't confidently say this portfolio has positive expected returns -- even though the sample mean is 1.2%."

**ML Example:**
"You test a classifier on 500 examples and get 82% accuracy. The 95% CI for the true accuracy is approximately [78.6%, 85.4%]. If a competing model gets 79% on the same test set, the CIs overlap substantially -- you probably can't claim one model is better."

**Common Misconceptions:**
- "95% probability the parameter is in this interval" -> Clarify: The parameter is a fixed number -- it's either in the interval or it isn't. The 95% refers to the long-run reliability of the procedure. This is the single most common misinterpretation in all of statistics.
- "A wider CI means I'm more certain" -> Clarify: A wider CI means MORE uncertainty, not less. You're saying the parameter could be in a larger range.
- "Non-overlapping CIs mean significant difference" -> Clarify: CIs can overlap and the difference can still be significant. And non-overlapping CIs almost certainly indicate a significant difference, but this is a rough heuristic, not a formal test.

**Verification Questions:**
1. "A 95% CI for a team's scoring average is [22.1, 26.3] points per game. What does this mean? What does it NOT mean?"
2. "If you increase the sample size from 50 to 200 games, what happens to the CI width?"
3. "Why do we use 1.96 for a 95% CI?"

**Good answer indicators:**
- They give the repeated-sampling interpretation, not the "probability" interpretation
- They say the CI gets narrower (approximately halved, since sqrt(200/50) = 2)
- They connect 1.96 to the normal distribution (95% of values within 1.96 SDs)

**If they struggle:**
- The archery analogy: "A confidence interval is like an archer who hits the target 95% of the time. Each arrow either hits or misses. For any single arrow, you can't say there's a 95% chance it hit -- it already landed. The 95% describes the archer's long-run accuracy."
- Simulation: generate 100 CIs from 100 samples of a known population. Show that roughly 95 capture the true mean and 5 miss. This makes the frequentist interpretation visceral.
- If the finance example resonates: "Value at Risk (VaR) is conceptually related -- it's a range of losses you'd expect with some confidence level."

**Exercise 2.1: Confidence Intervals for Sports Data**
"A soccer team scored 1.8 goals per game over a 38-game season, with a standard deviation of 1.3. Compute the 95% and 99% confidence intervals for the team's true scoring rate. Then answer: could this team's true average be as low as 1.4 goals per game?"

**How to Guide Them:**
1. "What's your point estimate? What's the standard error?"
2. If stuck on the z-value: "For 99%, use 2.576 instead of 1.96. Look up the standard normal distribution."
3. If they get the numbers but not the interpretation: "Is 1.4 inside the 95% CI? What does that tell you?"

**Solution:**
```python
import numpy as np

x_bar = 1.8
s = 1.3
n = 38
se = s / np.sqrt(n)

ci_95 = (x_bar - 1.96 * se, x_bar + 1.96 * se)
ci_99 = (x_bar - 2.576 * se, x_bar + 2.576 * se)

print(f"Standard error: {se:.4f}")
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"99% CI: [{ci_99[0]:.3f}, {ci_99[1]:.3f}]")
print(f"Is 1.4 in the 95% CI? {ci_95[0] <= 1.4 <= ci_95[1]}")
```

**After exercise, ask:**
- "Why is the 99% CI wider than the 95% CI?"
- "If you only had a 10-game sample, how would that change things?"

---

### Section 3: Hypothesis Testing

**Core Concept to Teach:**
Hypothesis testing is a formal framework for deciding whether observed data is consistent with a specific claim (the null hypothesis) or whether the evidence is strong enough to reject that claim. You start by assuming the null hypothesis is true, compute how likely your data would be under that assumption, and decide accordingly.

**How to Explain:**
1. The courtroom analogy: "Hypothesis testing works like a criminal trial. The null hypothesis is 'innocent until proven guilty.' You assume it's true and look for evidence against it. If the evidence is overwhelming, you reject the null. If not, you fail to reject -- which is NOT the same as saying it's true. You haven't proven innocence; you just don't have enough evidence to convict."
2. The framework:
   - H0 (null hypothesis): the boring explanation -- nothing special is happening
   - H1 (alternative hypothesis): the interesting claim -- something real is going on
   - Collect data, compute a test statistic
   - Ask: "If H0 were true, how unusual would this data be?"
   - If very unusual: reject H0. If not: fail to reject H0.
3. Test statistics: "A test statistic converts your data into a single number that measures how far your observation is from what H0 predicts. For means, the z-test and t-test are common choices."

**Sports Betting Example:**
"Is this betting strategy actually profitable, or just lucky?"

```python
import numpy as np
from scipy import stats

# Bettor claims their strategy beats 50/50
wins = 112
total_bets = 200
p_hat = wins / total_bets  # observed win rate
p_0 = 0.50                 # null hypothesis: no edge

# z-test for a proportion
se = np.sqrt(p_0 * (1 - p_0) / total_bets)
z = (p_hat - p_0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed

print(f"Observed win rate: {p_hat:.3f}")
print(f"z-statistic: {z:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H0 at alpha=0.05? {p_value < 0.05}")
```

Walk through: "H0: true win rate = 0.50 (no skill). H1: true win rate != 0.50. We compute a z-score measuring how many standard errors the observed rate is from 0.50. If the p-value is below our threshold (usually 0.05), we reject H0 and conclude there's evidence of a real edge."

**Casino Example:**
"Is this roulette wheel fair? You observe 1000 spins and red comes up 540 times. Under a fair wheel, red should appear about 473 times (18/38 = 47.37%). Is 540 out of 1000 suspicious?"

Walk them through the z-test for this proportion: H0: p = 18/38, H1: p != 18/38. The z-statistic will be large, and the p-value tiny -- this wheel is almost certainly biased.

**Finance Example:**
"Did this portfolio outperform the market? A fund returned an average of 0.8% per month over 48 months, with a standard deviation of 3.2%. The market returned 0.6% per month. Is the difference real?"

```python
import numpy as np
from scipy import stats

# One-sample t-test: does the excess return differ from 0?
excess_returns = np.random.normal(0.2, 3.2, 48)  # simulated excess returns
t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
print(f"Mean excess return: {np.mean(excess_returns):.3f}%")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
```

Walk through: "With high volatility and only 48 months of data, the signal-to-noise ratio is poor. Even genuinely skilled managers often can't produce statistically significant outperformance over short periods. This is why evaluating fund managers is so hard."

**ML Example:**
"Is model A (84% accuracy) significantly better than model B (81% accuracy) on a 1000-example test set? Use a two-proportion z-test or McNemar's test to find out."

**Common Misconceptions:**
- "Rejecting H0 proves H1 is true" -> Clarify: You've found evidence against H0, not proof of H1. There could be other explanations you haven't considered.
- "Failing to reject H0 means H0 is true" -> Clarify: It means you don't have enough evidence to reject H0. Maybe the effect is real but your sample is too small to detect it (Type II error).
- "A smaller p-value means a bigger effect" -> Clarify: p-values mix effect size with sample size. A tiny, practically meaningless effect can produce a tiny p-value if the sample is large enough.

**Verification Questions:**
1. "A gambler claims their roulette system beats the house. Formulate H0 and H1 for testing this claim."
2. "You run a z-test and get z = 1.5. What does this mean in plain English?"
3. "Why do we say 'fail to reject H0' instead of 'accept H0'?"

**Good answer indicators:**
- H0: the gambler has no edge (win rate = expected rate under fair play). H1: the gambler does have an edge.
- z = 1.5 means the observed result is 1.5 standard errors from what H0 predicts -- unusual but not extreme
- They explain that absence of evidence is not evidence of absence

**If they struggle:**
- Return to the courtroom analogy. "The jury doesn't say 'the defendant is innocent.' They say 'not guilty' -- we didn't find enough evidence. Same idea."
- Walk through a very simple example: flip a coin 10 times, get 7 heads. Is it unfair? Compute the probability of 7+ heads under a fair coin. Show that it's not that surprising (p ~ 0.17).
- If z-scores and t-scores are confusing, focus on the intuition: "How many standard errors away from the null hypothesis is our observation?" More standard errors = more surprising = stronger evidence against H0.

**Exercise 3.1: Test a Betting Strategy**
"A sports bettor has won 118 out of 210 bets against the spread. The null hypothesis is that they have no edge (50% win rate). Conduct a two-tailed z-test at the alpha = 0.05 level. What do you conclude?"

**How to Guide Them:**
1. "What are H0 and H1?"
2. "Compute the standard error under the null hypothesis."
3. "Compute the z-statistic and look up (or compute) the p-value."
4. "Compare the p-value to alpha = 0.05."

**Solution:**
```python
from scipy import stats
import numpy as np

wins, n = 118, 210
p_hat = wins / n
p_0 = 0.50
se = np.sqrt(p_0 * (1 - p_0) / n)
z = (p_hat - p_0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Win rate: {p_hat:.4f}")
print(f"z = {z:.4f}")
print(f"p-value = {p_value:.4f}")
print(f"Conclusion: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")
```

**After exercise, ask:**
- "If the bettor had the same win rate but over 1000 bets instead of 210, would the conclusion change? Why?"
- "What would this bettor need to track besides win rate?"

---

### Section 4: P-Values and Statistical Significance

**Core Concept to Teach:**
A p-value is the probability of observing data as extreme as (or more extreme than) what you actually observed, IF the null hypothesis were true. It is NOT the probability that the null hypothesis is true. It is NOT the probability that your results are due to chance. It measures how surprising your data is under the assumption of H0.

**How to Explain:**
1. Start with a precise definition: "The p-value answers one question: 'If nothing interesting were happening (H0 is true), how often would I see results this extreme just by chance?' A small p-value means your data would be very unlikely under H0."
2. The significance threshold: "Conventionally, we use alpha = 0.05 as a cutoff: if p < 0.05, reject H0. But 0.05 is arbitrary. There's nothing magical about it. It was a convention proposed by Fisher, and it stuck. A p-value of 0.049 and 0.051 represent essentially the same evidence, but one 'passes' and the other 'fails.' This binary thinking is a major source of bad science."
3. Effect size: "Statistical significance tells you whether an effect exists, not whether it's big enough to matter. With a large enough sample, you can detect trivially small effects. Always ask: is this effect practically meaningful?"

**Sports Example - Cherry-Picking:**
"Imagine a sports analyst tests 20 different 'trends' -- does this team play better on Tuesdays? After road trips? In odd-numbered months? Even if ALL trends are fake (H0 true for all), you'd expect about 1 out of 20 to produce p < 0.05 by pure chance. The analyst then publishes: 'Statistically significant trend found!' This is the multiple testing problem, and it's rampant in sports analytics."

**Finance Example - Backtesting Bias:**
"A quantitative trader tests 100 different trading strategies on historical data. Five produce p < 0.05 'significant' returns. The trader picks the best one and launches a fund. But those 5 significant results are exactly what you'd expect from testing 100 random strategies. The 'significant' backtested return is likely an artifact of overfitting to historical noise."

**ML Example - Data Dredging:**
"You try 50 different feature engineering approaches, 10 model architectures, and 5 hyperparameter settings. That's 2500 combinations. Some will appear to perform significantly better -- but if you only report the best result, you've p-hacked your way to a false conclusion. This is why holdout test sets and pre-registration matter."

**P-Hacking Demonstration:**
Walk through a concrete example of p-hacking:

```python
import numpy as np
from scipy import stats

# Simulate: test 20 "hypotheses" that are all null (no real effect)
np.random.seed(42)
significant_count = 0
for i in range(20):
    # Two groups with NO real difference
    group_a = np.random.normal(0, 1, 30)
    group_b = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(group_a, group_b)
    if p < 0.05:
        significant_count += 1
        print(f"  Test {i+1}: p = {p:.4f} -- 'SIGNIFICANT!'")
    else:
        print(f"  Test {i+1}: p = {p:.4f}")

print(f"\n{significant_count} out of 20 tests were 'significant' (expected ~1)")
```

Walk through: "Every single test here has NO real effect. Yet some still come out 'significant.' If you only report the significant ones, you've manufactured false discoveries."

**Common Misconceptions:**
- "The p-value is the probability H0 is true" -> Clarify: The p-value assumes H0 is true and asks how surprising the data is. It tells you P(data | H0), not P(H0 | data). To get P(H0 | data), you need Bayes' theorem and a prior.
- "p = 0.03 means there's only a 3% chance these results are due to chance" -> Clarify: It means that IF chance alone were operating, data this extreme would occur about 3% of the time. That's different from saying there's a 3% chance it IS due to chance.
- "Not significant means not real" -> Clarify: It might mean the sample is too small to detect the effect (low power). Absence of evidence is not evidence of absence.
- "Significant means important" -> Clarify: Statistically significant and practically significant are different things. A drug that lowers blood pressure by 0.1 mmHg could be statistically significant with n = 100,000 patients but clinically meaningless.

**Verification Questions:**
1. "A study finds p = 0.03. A journalist writes: 'There is only a 3% chance the drug doesn't work.' What's wrong with this statement?"
2. "If you test 100 true null hypotheses at alpha = 0.05, how many do you expect to reject?"
3. "A betting system shows a statistically significant edge with p = 0.01 over 10,000 bets. The edge is 0.5% (you win 50.5% of bets). Is this practically useful?"

**Good answer indicators:**
- They explain the journalist confused P(data | H0) with P(H0 | data)
- They say about 5 out of 100 (5%)
- They recognize that a 0.5% edge, while real, might not cover transaction costs (the vig/juice in sports betting)

**If they struggle:**
- Use the "rare disease test" analogy: a test with a 5% false positive rate doesn't mean a positive result gives you a 5% chance of not having the disease. It depends on how common the disease is (the base rate). Similarly, a p-value doesn't tell you the probability of H0.
- If the multiple testing problem is confusing: "Imagine 20 people each flip a fair coin 5 times. At least one will probably get 5 heads. That doesn't mean their coin is unfair -- you just gave randomness enough chances to produce something that looks surprising."

**Exercise 4.1: Interpreting P-Values**
Present 4 statements about p-values and ask the learner to identify which are correct and which are misconceptions. Then have them explain why the incorrect ones are wrong.

Statements:
1. "The p-value is the probability that the null hypothesis is true."
2. "If p < 0.05, the result is scientifically important."
3. "The p-value tells us how likely the observed data would be if the null hypothesis were true."
4. "A p-value of 0.001 means the alternative hypothesis is almost certainly true."

**Solution:**
- Statement 1: WRONG. p-value assumes H0 is true; it doesn't give the probability of H0.
- Statement 2: WRONG. Statistical significance doesn't imply practical significance.
- Statement 3: CLOSEST TO CORRECT (it's the probability of data this extreme or more extreme under H0).
- Statement 4: WRONG. Small p-values are strong evidence against H0 but don't directly give the probability of H1 being true.

---

### Section 5: Types of Errors and Power

**Core Concept to Teach:**
Every hypothesis test can make two kinds of mistakes. Type I error: rejecting H0 when it's actually true (false alarm). Type II error: failing to reject H0 when it's actually false (missed detection). The probability of Type I error is alpha (you set this). The probability of Type II error is beta. Statistical power = 1 - beta = probability of correctly detecting a real effect.

**How to Explain:**
1. Set up the 2x2 table:
   - H0 true + Reject H0 = Type I error (false positive)
   - H0 true + Fail to reject H0 = Correct decision (true negative)
   - H0 false + Reject H0 = Correct decision (true positive) -- this is POWER
   - H0 false + Fail to reject H0 = Type II error (false negative)
2. Real stakes: "In medicine, a Type I error means approving a drug that doesn't work. A Type II error means rejecting a drug that does work. Both are bad, but in different ways. The same trade-off appears in every domain."

**Sports Betting Example:**
- Type I error: You conclude a betting strategy is profitable when it's actually just luck. You bet real money on a worthless system and lose.
- Type II error: You dismiss a betting strategy as luck when it actually has a real edge. You miss a profitable opportunity.
- "Which error is worse? For most bettors, Type I is worse -- you're losing real money. So you'd want a low alpha (strict threshold for declaring a strategy 'real')."

**Finance Example:**
- Type I error: A quant fund deploys a trading signal that appeared significant in backtesting but has no real predictive power. They lose capital.
- Type II error: A fund dismisses a genuinely profitable signal because it didn't quite reach significance. They miss returns.
- "In finance, Type I errors are expensive -- you trade on false signals and lose money. This is why quants often use stricter significance thresholds (alpha = 0.01 or even 0.001)."

**ML Example:**
- Type I error: You deploy model B to production because it appeared significantly better than model A, but the improvement was noise. Users get a worse experience.
- Type II error: You stick with model A because the improvement of model B wasn't statistically significant, missing a real improvement.
- "Connection to precision and recall: in a classification setting, false positives (precision) correspond to Type I errors, and false negatives (recall) correspond to Type II errors. Same fundamental trade-off."

**Power Analysis:**
"Before running an experiment, you should ask: 'Given the effect size I care about and my sample size, what's the probability I'll detect the effect if it exists?' This is power analysis."

```python
from scipy import stats
import numpy as np

def compute_power(effect_size, n, alpha=0.05):
    """Power of a one-sample z-test for a proportion."""
    p_0 = 0.50  # null hypothesis
    p_1 = p_0 + effect_size  # alternative
    se_0 = np.sqrt(p_0 * (1 - p_0) / n)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    se_1 = np.sqrt(p_1 * (1 - p_1) / n)
    z_power = (p_1 - p_0 - z_crit * se_0) / se_1
    power = stats.norm.cdf(z_power)
    return power

# How much data do you need to detect a 3% edge in sports betting?
for n in [100, 500, 1000, 2000, 5000]:
    pwr = compute_power(0.03, n)
    print(f"n = {n:>5}: power = {pwr:.3f}")
```

Walk through: "If the true edge is 3% (win rate 53%), you need thousands of bets to have a good chance of detecting it statistically. With 100 bets, the power is terrible -- you'd almost certainly miss the effect even if it's real. This is why professional sports bettors track thousands of bets."

**Multiple Testing Problem:**
"If you test 20 hypotheses at alpha = 0.05, the probability of at least one Type I error is 1 - (1 - 0.05)^20 = 0.64. That's a 64% chance of a false positive somewhere. Solutions include the Bonferroni correction (use alpha/20 for each test) or the Benjamini-Hochberg procedure (controls false discovery rate)."

```python
# Bonferroni correction
alpha = 0.05
num_tests = 20
bonferroni_alpha = alpha / num_tests
print(f"Original alpha: {alpha}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha}")
print(f"P(at least one false positive, uncorrected): {1 - (1 - alpha)**num_tests:.3f}")
print(f"P(at least one false positive, Bonferroni): {1 - (1 - bonferroni_alpha)**num_tests:.3f}")
```

**Common Misconceptions:**
- "I should always minimize Type I error" -> Clarify: Reducing alpha increases beta (more Type II errors). It's a trade-off. The right balance depends on the costs of each error type.
- "Power of 80% is always good enough" -> Clarify: 80% is a convention, not a law of nature. If detecting the effect is critical (a profitable trading strategy), you might want 90% or 95% power.
- "If my test is not significant, the effect size must be zero" -> Clarify: You may lack power. Check whether your sample size was large enough to detect the effect size you care about.

**Verification Questions:**
1. "In sports betting, which is worse: a Type I error or a Type II error? Why?"
2. "You have power = 0.30 to detect a 2% edge. What does this mean in plain English?"
3. "If you test 50 trading strategies and 3 are 'significant' at alpha = 0.05, should you trade them?"

**Good answer indicators:**
- They reason about the costs: Type I means losing money on a bad strategy, Type II means missing a good one
- "There's a 30% chance I'd detect the 2% edge if it existed. That means a 70% chance I'd miss it."
- They recognize the multiple testing problem: 50 * 0.05 = 2.5 expected false positives, so 3 significant results could easily all be false

**If they struggle:**
- The fire alarm analogy: Type I = alarm goes off when there's no fire (annoying). Type II = no alarm when there IS a fire (dangerous). You want both to be low, but you can't drive both to zero simultaneously.
- If power analysis is confusing: "Power is just the probability you'll find what you're looking for, IF it's there. Low power means your study is like looking for a needle in a haystack with a weak magnet."
- Visualize it: draw two overlapping bell curves (null distribution and alternative distribution), show alpha region, show how power is the area under the alternative curve past the critical value

**Exercise 5.1: Power and Error Trade-offs**
"A casino suspects a roulette wheel is biased. They want to test whether the proportion of red outcomes differs from 18/38. If the true proportion is 0.50 (biased toward red), how many spins do they need to detect this with 80% power at alpha = 0.05? Use the power function from above (adapted for this scenario) to find the answer."

**How to Guide Them:**
1. "What's the effect size here? It's 0.50 - 18/38 = 0.0263"
2. "Try different values of n and see when power crosses 0.80"
3. "Or use scipy.stats to compute it directly"

**Solution:**
```python
from scipy import stats
import numpy as np

p_0 = 18 / 38  # null: fair wheel
p_1 = 0.50     # alternative: biased wheel
effect_size = p_1 - p_0
alpha = 0.05

for n in [100, 500, 1000, 2000, 3000, 5000]:
    se_0 = np.sqrt(p_0 * (1 - p_0) / n)
    se_1 = np.sqrt(p_1 * (1 - p_1) / n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_power = (effect_size - z_crit * se_0) / se_1
    power = stats.norm.cdf(z_power)
    print(f"n = {n:>5}: power = {power:.3f}")
```

**After exercise, ask:**
- "Is it practical for a casino to observe this many spins?"
- "What would happen if the bias were larger (say p = 0.55)? Would you need fewer spins?"

---

## Practice Project: A/B Testing a Sports Betting Strategy

**Project Introduction:**
"Now let's put everything together. You're a sports analytics consultant. A client has been tracking their bets against the spread in NBA games. They've used two approaches: Strategy A (their 'gut feel' picks) and Strategy B (a model-based approach). You need to determine whether either strategy is profitable and whether Strategy B is significantly better than Strategy A."

**Dataset Setup:**
```python
import numpy as np

np.random.seed(42)

# Simulated betting results (1 = win, 0 = loss)
# Strategy A: slight edge (true win rate 52%)
strategy_a = np.random.binomial(1, 0.52, size=300)
# Strategy B: better edge (true win rate 55%)
strategy_b = np.random.binomial(1, 0.55, size=300)
```

**Requirements:**
Present these to the learner one at a time or all at once depending on their preference:

1. **Compute point estimates and confidence intervals** for each strategy's win rate
2. **Test whether each strategy is profitable**: H0: win rate = 0.50 vs H1: win rate > 0.50
3. **Test whether Strategy B is better than Strategy A**: two-proportion z-test
4. **Compute the power** of your test to detect a 3% difference between strategies with 300 bets each
5. **Address multiple testing**: the client also tested Strategies C, D, E (none profitable). Does this change your conclusions about A and B?

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work and offer to answer questions at each requirement
2. **If they want guidance**: Walk through each requirement step by step
3. **If they're unsure**: Start with requirement 1 (CIs) and check in before moving on

**Checkpoints During Project:**
- After CIs: "What do the confidence intervals tell you about each strategy?"
- After hypothesis tests: "Which strategy appears profitable? How strong is the evidence?"
- After comparison test: "Can you conclude B is better than A?"
- After power analysis: "Was your test powerful enough to detect the difference?"
- After multiple testing: "How does knowing about strategies C, D, E change your interpretation?"

**Full Solution:**
```python
import numpy as np
from scipy import stats

np.random.seed(42)
strategy_a = np.random.binomial(1, 0.52, size=300)
strategy_b = np.random.binomial(1, 0.55, size=300)

# 1. Point estimates and CIs
p_a = np.mean(strategy_a)
p_b = np.mean(strategy_b)
se_a = np.sqrt(p_a * (1 - p_a) / len(strategy_a))
se_b = np.sqrt(p_b * (1 - p_b) / len(strategy_b))

print("=== Point Estimates and Confidence Intervals ===")
print(f"Strategy A: {p_a:.4f}, 95% CI: [{p_a - 1.96*se_a:.4f}, {p_a + 1.96*se_a:.4f}]")
print(f"Strategy B: {p_b:.4f}, 95% CI: [{p_b - 1.96*se_b:.4f}, {p_b + 1.96*se_b:.4f}]")

# 2. Test each strategy vs 50%
print("\n=== Individual Hypothesis Tests (H0: p = 0.50) ===")
for name, p_hat, n in [("A", p_a, 300), ("B", p_b, 300)]:
    se_0 = np.sqrt(0.50 * 0.50 / n)
    z = (p_hat - 0.50) / se_0
    p_val = 1 - stats.norm.cdf(z)  # one-tailed (H1: p > 0.50)
    print(f"Strategy {name}: z = {z:.4f}, p-value = {p_val:.4f}, "
          f"{'Reject H0' if p_val < 0.05 else 'Fail to reject H0'}")

# 3. Compare A vs B
print("\n=== Comparing Strategies (H0: p_B - p_A = 0) ===")
p_pool = (np.sum(strategy_a) + np.sum(strategy_b)) / (len(strategy_a) + len(strategy_b))
se_diff = np.sqrt(p_pool * (1 - p_pool) * (1/len(strategy_a) + 1/len(strategy_b)))
z_diff = (p_b - p_a) / se_diff
p_val_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
print(f"Difference: {p_b - p_a:.4f}")
print(f"z = {z_diff:.4f}, p-value = {p_val_diff:.4f}")
print(f"{'Reject H0' if p_val_diff < 0.05 else 'Fail to reject H0'}")

# 4. Power analysis
print("\n=== Power Analysis ===")
effect = 0.03
se_power = np.sqrt(2 * 0.50 * 0.50 / 300)  # under null for two-proportion test
z_crit = stats.norm.ppf(0.975)
power = 1 - stats.norm.cdf(z_crit - effect / se_power)
print(f"Power to detect 3% difference with n=300 each: {power:.3f}")

# 5. Multiple testing consideration
print("\n=== Multiple Testing ===")
num_strategies = 5  # A, B, C, D, E
bonferroni_alpha = 0.05 / num_strategies
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.3f}")
print("Re-evaluate: do the p-values from step 2 survive the corrected threshold?")
```

**Code Review Approach:**
When reviewing their work:
1. Start with praise: "Good job setting up the hypotheses correctly."
2. Ask questions: "What does the power analysis tell you about your ability to detect real differences?"
3. Guide improvements: "Consider whether one-tailed or two-tailed is more appropriate here."
4. Relate to concepts: "See how the multiple testing adjustment makes some results non-significant? That's the Bonferroni correction doing its job."

**If They Get Stuck:**
- On CIs: "Start with the formula: p_hat +/- 1.96 * sqrt(p_hat * (1 - p_hat) / n)"
- On hypothesis tests: "Set up H0 (no edge) and compute how far the observed rate is from 0.50 in standard error units"
- On the comparison: "Pool the proportions to get a common standard error, then compute z for the difference"
- On power: "How often would you detect a 3% difference given this sample size and variance?"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned:"
- Sampling distributions are the foundation of inference -- they tell you how your estimates would vary
- Confidence intervals quantify uncertainty about population parameters (and are frequently misinterpreted)
- Hypothesis testing is a framework for deciding whether evidence is strong enough to reject a claim
- P-values measure how surprising your data is under H0 -- nothing more
- Type I and II errors represent fundamental trade-offs, and power analysis helps you design studies that can actually detect what you're looking for

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel about statistical inference?"
- 1-4: Focus on the foundational concepts (CIs and hypothesis tests). Suggest working through more examples with the paired guide. Offer to revisit specific misconceptions.
- 5-7: Normal! Suggest practicing with real data (sports reference databases, Yahoo Finance data). Offer to work through more exercises from the guide.
- 8-10: Great! Suggest Bayesian Statistics as an alternative inference framework, or Regression and Modeling to apply inference to more complex settings.

**Suggest Next Steps:**
Based on their progress and interests:
- Sports betting track: "Try analyzing your own betting records or historical odds data"
- Finance track: "Apply these tools to analyze portfolio returns or compare investment strategies"
- ML track: "Use statistical tests to rigorously compare models in your next project"
- Academic track: "Consider the Bayesian Statistics route for an alternative framework"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there anything you'd like me to clarify or explain differently?"
"Which domain applications were most useful to your understanding?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Slow down and use more analogies (courtroom, fire alarm, archer)
- Focus on one domain consistently rather than switching between sports/finance/ML
- Run more simulations -- seeing 1000 confidence intervals where 950 capture the true mean is more convincing than a verbal explanation
- Do exercises together step by step rather than having them work alone
- Check CLT understanding -- if it's shaky, inference will feel like magic rather than logic
- Use concrete numbers: "If the true win rate is 0.50 and you bet 200 times..."

### If Learner is Excelling
- Move faster through standard CI/hypothesis testing mechanics
- Focus on interpretation subtleties: the correct interpretation of CIs, the difference between statistical and practical significance
- Introduce more advanced topics: Bayesian inference as an alternative, bootstrap confidence intervals, permutation tests
- Discuss real-world challenges: multiple testing in genomics, effect size reporting in psychology, the replication crisis
- Challenge them with harder scenarios: "What if the observations aren't independent?" "What if the distribution isn't normal?"

### If Learner Seems Disengaged
- Check in: "How are you feeling about this?"
- Ask about their specific goals: "What made you interested in statistical inference?"
- Switch domain examples to match their interests
- If they find the math tedious, focus on simulation-based approaches
- If they find it too theoretical, jump to the practice project earlier
- Take a break if needed

### Different Learning Styles
- **Visual learners**: Run simulations and plot results. Show the sampling distribution narrowing, show 100 CIs with some missing the true mean, plot overlapping distributions for Type I/II errors.
- **Hands-on learners**: Jump to exercises quickly, explain concepts as they arise during coding
- **Conceptual learners**: Spend more time on the "why" -- why does the CLT make CIs possible, why does the frequentist framework work this way, what are its philosophical limitations
- **Example-driven learners**: Lead with the sports betting or finance scenario, derive the concepts from the example rather than presenting theory first

---

## Troubleshooting Common Issues

### Technical Setup Problems
- scipy not installed: `pip install scipy`
- numpy version conflicts: `pip install --upgrade numpy scipy`
- matplotlib plots not showing: may need `plt.show()` or `%matplotlib inline` in Jupyter

### Concept-Specific Confusion

**If confused about sampling distributions:**
- Run the simulation from Exercise 1.1 with different sample sizes
- Emphasize: "This is about HYPOTHETICAL repetition. What WOULD happen if you collected data again?"
- Connect to their experience: "Every time a new NBA season starts, we get a new 'sample' of that team's performance"

**If confused about CI interpretation:**
- Run the simulation that generates 100 CIs and shows which ones capture the true mean
- Use the archer analogy: the 95% describes the procedure's long-run accuracy, not any single interval
- Ask: "Is the population mean a random variable or a fixed number?" (Fixed.) "Then can we assign a probability to it being in an interval?" (Not in the frequentist framework.)

**If confused about p-values:**
- Return to the definition: "Assume H0 is true. How often would we see data this extreme?"
- Run a simulation: generate data from H0, compute test statistics, show where the observed statistic falls
- Use the conditional probability framing: "p = P(data this extreme | H0 true), NOT P(H0 true | data this extreme)"

**If confused about Type I vs Type II errors:**
- Draw the 2x2 decision table on paper
- Use domain-specific stakes: "Type I means you bet real money on a fake edge. Type II means you miss a real opportunity."
- Connect to the alpha level: "Alpha IS the Type I error rate. You choose it directly."

---

## Additional Resources to Suggest

**If they want more practice:**
- Work through the exercises in the paired guide (`/routes/statistical-inference/guide.md`)
- Analyze real sports data from sports-reference.com
- Apply inference to their own data (betting records, portfolio returns, model evaluations)

**If they want deeper understanding:**
- The [Bayesian Statistics](/routes/bayesian-statistics/map.md) route for an alternative framework
- "Statistical Rethinking" by Richard McElreath for a more conceptual approach
- "All of Statistics" by Larry Wasserman for a rigorous mathematical treatment

**If they want to see real applications:**
- FiveThirtyEight's methodology articles for sports and election forecasting
- Academic papers on the replication crisis in psychology and medicine
- Kaggle competitions for ML model comparison with statistical rigor

---

## Teaching Notes

**Key Emphasis Points:**
- The CI interpretation is the single most misunderstood concept in statistics. Take extra time here. If the learner can explain it correctly, they understand inference better than most professionals.
- P-values are the second most misunderstood concept. Spend time on what they are NOT before explaining what they ARE.
- The connection between all five sections: sampling distributions enable CIs, which enable hypothesis tests, which produce p-values, which can produce errors. It's one logical chain.
- Always tie back to stakes: "Getting this wrong means losing money / deploying a bad model / publishing a false finding."

**Pacing Guidance:**
- Don't rush Section 1 (sampling distributions). If the learner doesn't deeply understand why sample statistics vary, CIs and hypothesis tests will feel like arbitrary formulas.
- Section 2 (CIs) and Section 3 (hypothesis testing) are the technical core -- allow plenty of time for exercises.
- Section 4 (p-values) is conceptually subtle but computationally simple. Focus on interpretation, not calculation.
- Section 5 (errors and power) ties everything together. Make sure there's enough time for it.
- Allow plenty of time for the practice project -- it integrates all five sections.

**Success Indicators:**
You'll know they've got it when they:
- Can explain the CI interpretation correctly on the first try (without the "95% probability" mistake)
- Can formulate H0 and H1 for a new scenario without prompting
- Spontaneously ask "but is this effect practically meaningful?" when seeing a significant result
- Recognize the multiple testing problem when evaluating several strategies
- Can reason about Type I vs Type II error trade-offs in context
