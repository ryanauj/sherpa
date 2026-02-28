---
title: Probability Distributions
route_map: /routes/probability-distributions/map.md
paired_guide: /routes/probability-distributions/guide.md
topics:
  - Discrete Distributions
  - Continuous Distributions
  - Expected Value and Variance
  - Law of Large Numbers
  - Central Limit Theorem
---

# Probability Distributions - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach probability distributions -- the mathematical models that describe random processes. Every distribution is introduced through real-world applications in gambling, sports betting, finance, and machine learning before any formulas appear.

**Route Map**: See `/routes/probability-distributions/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/probability-distributions/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Identify which probability distribution models a given real-world scenario
- Compute probabilities, expected values, and variances for Bernoulli, Binomial, Poisson, Geometric, Uniform, Normal, and Exponential distributions
- Use scipy.stats in Python to work with any distribution (PMF/PDF, CDF, sampling)
- Explain why the Law of Large Numbers guarantees the house always wins
- Apply the Central Limit Theorem to understand why sample means are normally distributed
- Build Monte Carlo simulations to verify theoretical results

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/probability-distributions/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Completed the Probability Fundamentals route (sample spaces, probability rules, conditional probability, expected value)
- Basic Python skills (variables, loops, functions)
- Familiarity with numpy arrays

**If prerequisites are missing**: If they haven't done Probability Fundamentals, suggest they work through that route first -- distributions build directly on probability rules and expected value. If they need a quick refresher, spend a few minutes reviewing: "What does P(A) = 0.3 mean? What's expected value? Can you compute the expected value of a dice roll?"

### Audience Context
The target learner understands basic probability (from probability-fundamentals) and wants to model real-world randomness. They may be motivated by:
- Gambling/sports betting: understanding the math behind casino games and betting strategies
- Data science/ML: distributions underpin nearly every statistical model and ML algorithm
- Finance: modeling returns, risk, and rare events

Use their motivation to pick examples. If they're interested in sports betting, lead with casino and sports examples. If they're into ML, emphasize how distributions appear in classifiers, regression, and generative models. If finance, lead with portfolio returns and risk modeling.

Always show the real-world example BEFORE the formula. When introducing notation, explain it in plain English first.

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
- Example: "A roulette wheel has 18 red, 18 black, and 2 green slots. You bet on red 100 times. Which distribution models the number of wins? A) Poisson B) Binomial C) Normal D) Exponential"

**Explanation Questions:**
- Ask the learner to explain concepts in their own words
- Example: "In your own words, why does the casino always win in the long run, even though individual gamblers sometimes win big?"

**Prediction Questions:**
- Show a distribution setup and ask what will happen before running code
- Example: "Before running this simulation of 10,000 roulette spins, predict: will the average payout per spin be above, below, or exactly at the theoretical expected value?"

**Code Questions:**
- Ask the learner to write a function using scipy.stats or numpy
- Example: "Write a function that takes a probability p and number of trials n, and returns the probability of getting at least k successes"

---

## Teaching Flow

### Introduction

**What to Cover:**
- In probability fundamentals, they learned to compute the probability of individual events. Now they're learning to describe entire random processes -- all the possible outcomes and all their probabilities, packaged into one mathematical object called a distribution
- Distributions are the language of uncertainty. Every casino game, every financial model, every ML algorithm uses them
- By the end, they'll simulate casino games and prove mathematically why the house always wins

**Opening Questions to Assess Level:**
1. "Can you explain what expected value means? If I asked you to compute the expected value of rolling a die, how would you do it?"
2. "Have you encountered the term 'normal distribution' or 'bell curve' before? In what context?"
3. "What's motivating you to learn distributions -- gambling math, data science, finance, or something else?"

**Adapt based on responses:**
- If they're solid on expected value: Move quickly through E[X] review, focus on distributions themselves
- If they've seen the normal distribution: Use it as a reference point ("you already know the most famous one -- now let's build up to it")
- If they're motivated by gambling: Lead with casino examples in every section
- If they're motivated by ML: Emphasize how distributions appear in classifiers, feature engineering, and generative models
- If they're motivated by finance: Lead with portfolio returns, risk, and Black-Scholes connections

**Good opening framing:**
"A probability tells you the chance of one thing happening. A probability distribution tells you the chance of everything that could happen, all at once. When you roll a die, you don't just want to know 'what's the chance of a 6?' -- you want to know the complete picture: each face has probability 1/6. That complete picture is a distribution. Every casino game has one. Every stock has one. Every ML model produces one. Once you can describe a random process with a distribution, you can compute anything -- expected profit, risk, the chance of going broke, the chance of beating the market."

---

### Setup Verification

**Check scipy, numpy, matplotlib:**
```bash
python -c "import scipy.stats; import numpy as np; import matplotlib; print('All good')"
```

**If not installed:**
```bash
pip install scipy numpy matplotlib
```

**Quick test:**
```python
from scipy import stats
import numpy as np

# Generate 5 random samples from a normal distribution
samples = stats.norm.rvs(loc=0, scale=1, size=5, random_state=42)
print(f"Samples: {samples}")
print(f"Mean: {np.mean(samples):.4f}")
```

---

### Section 1: Discrete Distributions

**Core Concept to Teach:**
A discrete distribution assigns probabilities to countable outcomes -- things you can list out. How many free throws does a player make? How many games does a team win? How many goals are scored? Discrete distributions answer "how likely is each count?"

**How to Explain:**
1. Start with something they know: "You already know that a fair coin has P(heads) = 0.5. That's actually the simplest distribution -- the Bernoulli distribution. It has exactly two outcomes."
2. Build up: "What if you flip 10 coins? Now you're counting successes -- that's the Binomial distribution."
3. Shift to rare events: "What about events that happen rarely and unpredictably, like goals in a soccer game or stock market crashes? That's the Poisson distribution."
4. Add waiting: "How many trials until you get your first win? That's the Geometric distribution."

#### Bernoulli Distribution

**How to Explain:**
"The Bernoulli distribution is the simplest possible random process: something either happens or it doesn't. A single coin flip. A single free throw attempt. A single stock trade that either profits or loses. One trial, two outcomes."

**Example to Present:**
```python
from scipy import stats

# LeBron James free throw percentage: ~73%
ft = stats.bernoulli(p=0.73)

print(f"P(make) = {ft.pmf(1):.2f}")    # 0.73
print(f"P(miss) = {ft.pmf(0):.2f}")    # 0.27
print(f"Expected value: {ft.mean():.2f}")  # 0.73
print(f"Variance: {ft.var():.4f}")       # 0.1971

# Simulate 10 free throws
np.random.seed(42)
shots = ft.rvs(size=10)
print(f"Simulated shots: {shots}")  # Array of 0s and 1s
print(f"Made {shots.sum()} out of {len(shots)}")
```

**Walk Through:**
- `p=0.73` is the probability of success (making the shot)
- `pmf(1)` gives P(X=1) = probability of success. PMF stands for "probability mass function" -- it tells you the probability of each specific outcome
- `rvs(size=10)` generates random samples -- simulates 10 free throws

**Finance connection:** "A Bernoulli trial can model whether a stock goes up or down on a given day. If a stock goes up 55% of days, that's Bernoulli(p=0.55)."

**ML connection:** "Binary classification is fundamentally Bernoulli -- each prediction is either correct or incorrect."

#### Binomial Distribution

**How to Explain:**
"The Binomial distribution counts the number of successes in a fixed number of independent trials. If LeBron shoots 10 free throws, how many does he make? If a baseball player with a .300 batting average gets 4 at-bats, how many hits? If you bet on red 100 times at roulette, how many times do you win?"

**Key parameters:** n (number of trials) and p (probability of success on each trial).

**Example to Present:**
```python
from scipy import stats
import numpy as np

# A baseball player with .300 batting average gets 4 at-bats per game
binom = stats.binom(n=4, p=0.300)

# Probability of exactly k hits
for k in range(5):
    print(f"P({k} hits) = {binom.pmf(k):.4f}")

# P(0 hits) = 0.2401  -- hitless game
# P(1 hit)  = 0.4116  -- most likely outcome
# P(2 hits) = 0.2646
# P(3 hits) = 0.0756
# P(4 hits) = 0.0081  -- going 4-for-4 is rare

print(f"\nExpected hits per game: {binom.mean():.1f}")  # 1.2
print(f"Std dev: {binom.std():.4f}")                     # 0.9165
print(f"P(at least 1 hit): {1 - binom.pmf(0):.4f}")     # 0.7599
```

**Walk Through:**
- "The most likely outcome is exactly 1 hit (41% chance). Going hitless isn't that rare (24%). Going 4-for-4 is very rare (0.8%)."
- "Expected hits per game: n * p = 4 * 0.3 = 1.2. Over a 162-game season, that's about 194 hits."
- "P(at least 1 hit) = 1 - P(0 hits) = 0.76. So about 3 out of 4 games, he gets at least one hit."

**Sports betting application:**
"If a team wins 60% of their games, what's the probability they win at least 50 out of 82 games in an NBA season?"

```python
season = stats.binom(n=82, p=0.60)
p_at_least_50 = 1 - season.cdf(49)  # cdf(49) = P(X <= 49)
print(f"P(at least 50 wins): {p_at_least_50:.4f}")  # ~0.5765
```

**Finance application:**
"In a portfolio of 20 bonds, each with a 5% default probability (independently), how many defaults should you expect?"

```python
defaults = stats.binom(n=20, p=0.05)
print(f"Expected defaults: {defaults.mean():.1f}")      # 1.0
print(f"P(0 defaults): {defaults.pmf(0):.4f}")          # 0.3585
print(f"P(3+ defaults): {1 - defaults.cdf(2):.4f}")     # 0.0755
```

#### Poisson Distribution

**How to Explain:**
"The Poisson distribution models the number of events that occur in a fixed interval when events happen independently at a constant average rate. Goals per soccer match. Accidents per month. Server errors per hour. Stock market crashes per decade. The key parameter is lambda -- the average number of events per interval."

**Example to Present:**
```python
from scipy import stats

# Average goals per match in the Premier League: ~2.7
goals = stats.poisson(mu=2.7)

for k in range(8):
    print(f"P({k} goals) = {goals.pmf(k):.4f}")

# P(0 goals) = 0.0672  -- goalless draw is uncommon
# P(1 goal)  = 0.1815
# P(2 goals) = 0.2450  -- most likely
# P(3 goals) = 0.2205
# P(4 goals) = 0.1488
# P(5 goals) = 0.0804
# P(6 goals) = 0.0362
# P(7 goals) = 0.0140

print(f"\nExpected goals: {goals.mean():.1f}")  # 2.7
print(f"P(4+ goals, for over bet): {1 - goals.cdf(3):.4f}")  # 0.2694
```

**Walk Through:**
- "Lambda = 2.7 means on average, 2.7 goals per match. The most likely single outcome is exactly 2 goals."
- "Sports bettors care about P(4+ goals) for over/under bets -- here it's about 27%."
- "Notice the Poisson naturally handles the fact that you can't have negative goals and there's no theoretical upper limit."

**Finance connection:** "Poisson models rare events well. If stock market crashes (drops > 10%) happen about 0.3 times per year on average, P(2+ crashes in a year) = 1 - poisson.cdf(1, mu=0.3) -- very small but not zero."

**ML connection:** "Count data in ML -- number of clicks, number of words, number of purchases -- often follows a Poisson distribution. Poisson regression is built on this."

#### Geometric Distribution

**How to Explain:**
"The Geometric distribution counts how many trials you need until your first success. How many hands of poker until you get a winning hand? How many sales calls until you close a deal? How many at-bats until a hit?"

**Example to Present:**
```python
from scipy import stats

# Probability of winning a hand of poker: ~15%
# How many hands until first win?
hands = stats.geom(p=0.15)

print(f"Expected hands until first win: {hands.mean():.1f}")  # ~6.7
print(f"P(win on 1st hand): {hands.pmf(1):.4f}")             # 0.1500
print(f"P(win within 5 hands): {hands.cdf(5):.4f}")          # 0.5563
print(f"P(need 10+ hands): {1 - hands.cdf(9):.4f}")          # 0.2316
```

**Walk Through:**
- "On average, you'll wait about 6.7 hands. But there's a 23% chance you'll need 10 or more hands. Gambling streaks feel personal, but they're just geometric randomness."
- "The geometric distribution is memoryless: no matter how many hands you've lost, the probability of winning the next hand is still 15%. Past losses don't make a win 'due' -- this is the gambler's fallacy."

**Connect to gambler's fallacy:** "This is a powerful teaching moment. Ask: 'If you've lost 10 hands in a row, is a win now more likely?' The answer is no. The geometric distribution is memoryless. Each trial is independent."

**Common Misconceptions for All Discrete Distributions:**
- Misconception: "Binomial requires p = 0.5" --> Clarify: "p can be any value between 0 and 1. A team winning 60% of games is Binomial with p=0.6."
- Misconception: "Poisson is only for time intervals" --> Clarify: "Poisson works for any fixed interval -- time, space, area. Typos per page, defects per square meter, accidents per intersection per year."
- Misconception: "After many failures, a success is 'due'" --> Clarify: "The geometric distribution is memoryless. Independence means past results don't affect future probabilities."
- Misconception: "Expected value means the most likely outcome" --> Clarify: "E[X] = 1.2 hits doesn't mean 1.2 is the most likely outcome (you can't get 1.2 hits). The most likely outcome is the mode, which may differ from the mean."

**Verification Questions:**
1. "A basketball player makes 80% of free throws. She shoots 5. What distribution models the number she makes, and what are the parameters?"
2. "A website gets an average of 3 server errors per day. What's the probability of a perfect day (0 errors)? Which distribution did you use?"
3. "Why can't you use a Binomial distribution to model goals per soccer match? (Hint: what's n?)"

**Good answer indicators:**
- They correctly identify Binomial(n=5, p=0.8) for the free throws
- They use Poisson(mu=3) and compute P(0) = e^(-3) for the server errors
- They explain that Binomial requires a fixed number of trials, but goals in a soccer match don't have a clear n (there's no fixed number of "goal attempts")

**If they struggle:**
- Go back to concrete counting: "Can you list all possible outcomes? 0, 1, 2, 3, 4, 5 makes for free throws -- those are countable. That's discrete."
- Draw the decision tree: "Each free throw is a Bernoulli trial. String 5 together and you get Binomial."
- Use the scipy.stats documentation interactively: have them look up `stats.binom` and experiment with parameters

**Exercise 1.1: Choosing the Right Distribution**
Present this exercise: "For each scenario, identify which discrete distribution applies and state its parameters:
1. A coin is flipped 20 times. Count the number of heads.
2. A call center gets an average of 8 calls per hour. Count calls in the next hour.
3. A scratch-off lottery ticket wins 1 time in 50. How many tickets until you win?
4. A single trade either profits or loses. You profit 40% of the time."

**Solution:**
1. Binomial(n=20, p=0.5)
2. Poisson(mu=8)
3. Geometric(p=1/50 = 0.02)
4. Bernoulli(p=0.4)

---

### Section 2: Continuous Distributions

**Core Concept to Teach:**
Continuous distributions model outcomes that can take any value in a range -- not just whole numbers. Stock returns can be 2.37% or -0.0041%. Sprint times can be 9.58 seconds or 10.23 seconds. When outcomes are continuous, we can't assign probabilities to individual points (the probability of a stock returning exactly 2.37000...% is zero). Instead, we work with probability density and compute probabilities over intervals.

**How to Explain:**
1. Key distinction: "For discrete distributions, we ask 'what's the probability of exactly 3?' For continuous distributions, we ask 'what's the probability of being between 2.5 and 3.5?' Individual points have probability zero -- only intervals have probability."
2. PDF vs PMF: "The probability density function (PDF) replaces the PMF. The height of the PDF at a point isn't a probability -- it's a density. You get probabilities by computing the area under the curve between two points."
3. CDF is your friend: "The cumulative distribution function, CDF, gives P(X <= x). It's the most useful function for computing actual probabilities. P(a < X < b) = CDF(b) - CDF(a)."

#### Uniform Distribution

**How to Explain:**
"The uniform distribution is the simplest continuous distribution: every value in the interval [a, b] is equally likely. A perfectly fair spinner. A random number generator. The arrival time of a bus if you know nothing about the schedule."

**Example to Present:**
```python
from scipy import stats
import numpy as np

# A roulette wheel spinner lands at a random angle between 0 and 360 degrees
spinner = stats.uniform(loc=0, scale=360)

print(f"Mean: {spinner.mean():.1f} degrees")       # 180.0
print(f"Variance: {spinner.var():.1f}")              # 10800.0
print(f"P(land in first quadrant, 0-90): {spinner.cdf(90) - spinner.cdf(0):.4f}")  # 0.25
print(f"P(exact angle = 45): practically 0")
```

**Walk Through:**
- "scipy.stats uses loc (starting point) and scale (width). Uniform(loc=0, scale=360) means values range from 0 to 360."
- "P(0 to 90) = 90/360 = 0.25. One-quarter of the circle, one-quarter of the probability."
- "The density is constant at 1/360 everywhere. That's what 'uniform' means -- flat, equal density."

#### Normal (Gaussian) Distribution

**How to Explain:**
"The normal distribution -- the bell curve -- is the single most important distribution in statistics. It's defined by two parameters: mean (mu, the center) and standard deviation (sigma, the spread). It appears everywhere:
- Heights of people in a population
- Test scores across a large group
- Measurement errors in experiments
- Player performance metrics in sports
- Stock returns (approximately)
- Many ML algorithms assume features are normally distributed"

**Example to Present:**
```python
from scipy import stats
import numpy as np

# NBA player heights: mean = 78.5 inches, std = 3.5 inches
heights = stats.norm(loc=78.5, scale=3.5)

print(f"P(height < 72 inches / 6 ft): {heights.cdf(72):.4f}")    # ~0.0317
print(f"P(height > 84 inches / 7 ft): {1 - heights.cdf(84):.4f}") # ~0.0582
print(f"P(between 75 and 82): {heights.cdf(82) - heights.cdf(75):.4f}") # ~0.6827

# The 68-95-99.7 rule
for k in [1, 2, 3]:
    lower = 78.5 - k * 3.5
    upper = 78.5 + k * 3.5
    prob = heights.cdf(upper) - heights.cdf(lower)
    print(f"P(within {k} std devs): {prob:.4f}")
# ~0.6827, ~0.9545, ~0.9973
```

**Walk Through:**
- "Only about 3% of NBA players are under 6 feet tall -- that's in the far left tail."
- "The 68-95-99.7 rule: about 68% of values fall within 1 standard deviation, 95% within 2, and 99.7% within 3. This rule works for any normal distribution."

**Finance application -- stock returns:**
```python
# Daily S&P 500 returns: mean ~0.04%, std ~1.1%
returns = stats.norm(loc=0.0004, scale=0.011)

print(f"P(daily loss > 3%): {returns.cdf(-0.03):.6f}")    # ~0.0029
print(f"P(daily loss > 5%): {returns.cdf(-0.05):.8f}")    # very small
```

**Critical caveat -- fat tails:**
"Tell the learner: 'The normal distribution says a 5% daily drop is almost impossible. But in reality, the S&P 500 has had daily drops of 7%, 10%, even 20%. Real financial returns have fat tails -- extreme events happen more often than the normal distribution predicts. This is a famous problem in finance. The normal distribution is a useful approximation for typical days, but it dangerously underestimates the probability of extreme events. The 2008 financial crisis was partly caused by models that assumed normal distributions when reality had fat tails.'"

**ML connection:** "Many ML algorithms assume features are normally distributed -- Gaussian Naive Bayes, Linear Discriminant Analysis, assumptions behind linear regression. Understanding when data is and isn't normally distributed is critical for choosing the right model."

#### Exponential Distribution

**How to Explain:**
"The exponential distribution models the time between events in a Poisson process. If goals in a soccer match follow a Poisson distribution (average rate), the time between goals follows an exponential distribution. If server errors arrive at a Poisson rate, the time between errors is exponential."

**Example to Present:**
```python
from scipy import stats

# If a soccer match averages 2.7 goals per 90 minutes,
# the average time between goals is 90/2.7 = 33.3 minutes
time_between_goals = stats.expon(scale=33.3)  # scale = 1/lambda = mean

print(f"Mean time between goals: {time_between_goals.mean():.1f} min")
print(f"P(goal within 10 min): {time_between_goals.cdf(10):.4f}")       # ~0.2592
print(f"P(no goal for 45+ min): {1 - time_between_goals.cdf(45):.4f}")  # ~0.2584
print(f"P(goal between 20-40 min): {time_between_goals.cdf(40) - time_between_goals.cdf(20):.4f}")
```

**Walk Through:**
- "About 26% chance of a goal within the first 10 minutes. About 26% chance of going a full half without a goal."
- "The exponential distribution is also memoryless -- if no goal has been scored in 30 minutes, the probability of a goal in the next 10 minutes is exactly the same as the probability of a goal in the first 10 minutes. The past doesn't matter."

**Finance connection:** "Time between trades, time between market-moving events, time between defaults -- all often modeled as exponential."

**Common Misconceptions for Continuous Distributions:**
- Misconception: "The PDF value at a point is the probability of that point" --> Clarify: "PDF values can be greater than 1. They're densities, not probabilities. You need to integrate (compute area) over an interval to get a probability."
- Misconception: "Everything is normally distributed" --> Clarify: "Income is right-skewed. Insurance claims have fat tails. Earthquake magnitudes follow a power law. Always check whether the normal assumption is reasonable for your data."
- Misconception: "The mean of a normal distribution is the most likely value" --> Clarify: "For a normal distribution, the mean, median, and mode are all the same -- so the mean IS the most likely value. But this is a special property of the normal distribution, not true of all distributions."

**Verification Questions:**
1. "Stock returns are approximately normal with mean 0.04% and std 1.1%. What's the probability of a return between -1% and +1%?"
2. "If customers arrive at a store at an average rate of 5 per hour, what distribution models the time between arrivals, and what's the probability of waiting more than 30 minutes for the next customer?"
3. "A PDF value at x=10 is 2.5. Does this mean there's a 250% chance of x being 10?"

**Good answer indicators:**
- They compute P(-0.01 < X < 0.01) using CDF subtraction for the stock returns question
- They identify Exponential(scale=12 minutes) and compute 1 - CDF(30) for the customer arrival question
- They correctly say no -- PDF values are densities, not probabilities

**If they struggle:**
- Draw the analogy: "Think of a PDF like a terrain map. The height of the terrain at each point isn't the 'amount of land' there -- it's how concentrated the land is. To find the actual amount of land in a region, you measure the area."
- Use the CDF as a crutch: "If the PDF is confusing, work exclusively with the CDF. CDF(x) = P(X <= x). That IS a probability, and it always makes sense."

**Exercise 2.1: Fitting Distributions to Scenarios**
Present this exercise: "For each scenario, identify the distribution and compute the requested probability:
1. You generate a random number between 0 and 1. What's P(0.3 < X < 0.7)?
2. Adult male heights have mean 70 inches, std 3 inches. What percentage are taller than 76 inches?
3. A taxi arrives every 8 minutes on average. What's the probability you wait more than 15 minutes?"

**Solution:**
1. Uniform(0, 1): P = 0.7 - 0.3 = 0.4
2. Normal(70, 3): P(X > 76) = 1 - norm.cdf(76, 70, 3) = 0.0228 = 2.28%
3. Exponential(scale=8): P(X > 15) = 1 - expon.cdf(15, scale=8) = 0.1534 = 15.3%

---

### Section 3: Expected Value and Variance of Distributions

**Core Concept to Teach:**
Every distribution has an expected value (long-run average) and a variance (measure of spread). These two numbers summarize a distribution's center and width. In gambling, E[X] tells you the house edge. In finance, E[X] is expected return and Var(X) is risk. In ML, expected prediction error decomposes into bias (related to E[X]) and variance.

**How to Explain:**
1. "You already know expected value from probability fundamentals. Now we're computing it for named distributions. Each distribution has a formula for E[X] and Var(X) in terms of its parameters."
2. "The real power: once you know the distribution, you instantly know the expected value and variance. You don't need to enumerate every outcome."

**Present the Formulas:**

| Distribution | E[X] | Var(X) |
|---|---|---|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Poisson(lambda) | lambda | lambda |
| Geometric(p) | 1/p | (1-p)/p^2 |
| Uniform(a,b) | (a+b)/2 | (b-a)^2/12 |
| Normal(mu, sigma) | mu | sigma^2 |
| Exponential(lambda) | 1/lambda | 1/lambda^2 |

"Notice: for the Poisson, the mean and variance are both lambda. That's a unique property -- if you see count data where the mean and variance are very different, it's probably not Poisson."

**Casino Application -- Computing the House Edge:**

**Example to Present:**
```python
import numpy as np

# American Roulette: 38 slots (1-36, 0, 00)
# Bet $1 on red: 18 red slots win $1, 20 non-red slots lose $1
p_win = 18/38
p_lose = 20/38

ev_per_bet = (1 * p_win) + (-1 * p_lose)
print(f"Expected value per $1 bet: ${ev_per_bet:.4f}")  # -$0.0526

# Over 1000 bets
total_bets = 1000
expected_loss = ev_per_bet * total_bets
print(f"Expected loss over {total_bets} bets: ${expected_loss:.2f}")  # -$52.63

# Variance per bet
var_per_bet = (1 - ev_per_bet)**2 * p_win + (-1 - ev_per_bet)**2 * p_lose
std_per_bet = np.sqrt(var_per_bet)
print(f"Std dev per bet: ${std_per_bet:.4f}")  # ~$0.9986

# Std dev of total over 1000 independent bets
std_total = std_per_bet * np.sqrt(total_bets)
print(f"Std dev over {total_bets} bets: ${std_total:.2f}")  # ~$31.57
```

**Walk Through:**
- "Every $1 bet on roulette costs you 5.26 cents on average. That's the house edge."
- "Over 1000 bets, you expect to lose $52.63. The standard deviation is about $31.57, so you might be up or down by $30-$60 from that expected loss."
- "This is why casinos are profitable. They don't need to win every hand -- they just need a negative expected value for the player and enough volume."

**Finance Application -- Portfolio Expected Return and Risk:**
```python
from scipy import stats
import numpy as np

# Two stocks: Tech stock (high return, high risk) vs Bond fund (low return, low risk)
tech = stats.norm(loc=0.12, scale=0.25)   # 12% expected return, 25% std dev
bond = stats.norm(loc=0.04, scale=0.05)   # 4% expected return, 5% std dev

# 60/40 portfolio
w_tech, w_bond = 0.6, 0.4
portfolio_mean = w_tech * tech.mean() + w_bond * bond.mean()
# Assuming independence (simplification)
portfolio_var = (w_tech**2) * tech.var() + (w_bond**2) * bond.var()
portfolio_std = np.sqrt(portfolio_var)

print(f"Portfolio expected return: {portfolio_mean:.2%}")  # 8.80%
print(f"Portfolio std dev (risk): {portfolio_std:.2%}")    # 15.20%
print(f"P(portfolio loses money): {stats.norm.cdf(0, portfolio_mean, portfolio_std):.4f}")
```

**Walk Through:**
- "By mixing stocks and bonds, you get 8.8% expected return (between 12% and 4%) but with reduced risk compared to all-stock."
- "This is the heart of portfolio theory: combining assets to balance expected return and risk."

**ML Preview -- Bias-Variance Tradeoff:**
"In ML, a model's expected prediction error decomposes into bias squared plus variance plus irreducible noise. A model with high bias consistently predicts the wrong thing (its expected prediction is far from truth). A model with high variance gives wildly different predictions depending on the training data. You'll see this in detail in later routes, but the language is the same -- expected value and variance."

**Verification Questions:**
1. "A casino game pays $10 with probability 0.4 and costs $7 to play. What's the expected value? Should you play?"
2. "Why does the house edge guarantee the casino profits in the long run but not on any single bet?"
3. Multiple choice: "A Poisson distribution has lambda = 5. What is its variance? A) 25 B) 5 C) sqrt(5) D) 2.5"

**If they struggle:**
- Go back to the definition: "E[X] = sum of (outcome * probability) for all outcomes"
- Use concrete money examples: "You pay $1 to play. You win $2 half the time and $0 half the time. E[X] = 2(0.5) + 0(0.5) - 1 = $0. It's a fair game."

**Exercise 3.1: House Edge Calculations**
"Compute the expected value per bet and house edge for:
1. European Roulette bet on a single number: pays 35-to-1, 37 slots (1-36, 0)
2. Craps pass-line bet: wins with probability 244/495, pays 1-to-1
3. A sports bet at -110 odds (bet $110 to win $100) on a 50/50 game"

---

### Section 4: Law of Large Numbers

**Core Concept to Teach:**
The Law of Large Numbers (LLN) says: as you take more and more independent samples from a distribution, the sample mean gets closer and closer to the true expected value. This is why casinos can predict their profits, why insurance companies can set premiums, and why you need a large sample of bets to evaluate a betting strategy.

**How to Explain:**
1. Intuition first: "Flip a coin 10 times -- you might get 70% heads. Flip it 10,000 times -- you'll get very close to 50%. The more you flip, the closer the average gets to the true probability."
2. The casino connection: "A gambler might win big in one session. But over millions of bets across thousands of gamblers, the casino's average profit per bet converges to the house edge. Individual luck washes out."
3. The formal statement: "If X_1, X_2, ..., X_n are independent draws from a distribution with mean mu, then the sample mean converges to mu as n goes to infinity."

**Simulation to Demonstrate:**
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate roulette: bet on red, win +1 or lose -1
p_red = 18/38
n_bets = 10000

# Generate outcomes
outcomes = np.where(np.random.random(n_bets) < p_red, 1, -1)

# Running average
running_avg = np.cumsum(outcomes) / np.arange(1, n_bets + 1)

# Theoretical expected value
ev = (18/38) * 1 + (20/38) * (-1)  # -0.0526

plt.figure(figsize=(10, 6))
plt.plot(running_avg, linewidth=0.8)
plt.axhline(y=ev, color='r', linestyle='--', label=f'E[X] = {ev:.4f}')
plt.xlabel('Number of Bets')
plt.ylabel('Average Profit per Bet ($)')
plt.title('Law of Large Numbers: Roulette Running Average')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Walk Through:**
- "Early on, the running average bounces around wildly. After a few hundred bets, it starts settling near the expected value. After several thousand, it's very close to -$0.0526."
- "This is LLN in action. Short-run luck is real. Long-run averages are predictable."

**THIS is Why Casinos Always Win:**
"Ask the learner: 'A casino doesn't know whether you personally will win or lose tonight. But they know that across all their tables, all night, all week, the average profit per bet will be very close to the house edge. They don't gamble -- they collect a mathematical tax. LLN is their business model.'"

**Sports Betting Application:**
"A sports bettor who wins 53% of bets at -110 odds has a positive expected value. But:
- After 100 bets, there's still a significant chance of being down (the sample is too small for LLN to kick in)
- After 1000 bets, the bettor's winning percentage will be very close to 53%, and profits become more predictable
- This is why professional bettors talk about 'sample size' -- you can't evaluate a strategy on 20 bets"

**Finance Application:**
"Diversification works because of LLN. If you own one stock, its return is highly variable. If you own 500 stocks, the average return across all of them converges to the market's expected return. Individual stock risk washes out. This is the mathematical foundation of index fund investing."

**Common Misconceptions:**
- Misconception: "LLN means results even out in the short run" --> Clarify: "LLN says the average converges, not that the outcomes balance out. If you lose 10 bets in a row, LLN does NOT mean you'll win the next 10. The future bets don't 'remember' the past ones."
- Misconception: "LLN means you can't win at gambling" --> Clarify: "LLN means the casino can't lose in aggregate. An individual gambler can absolutely win in a session. The key is that the gambler's expected value is negative, and LLN ensures the casino's average converges to that negative number times volume."
- Misconception: "LLN says the sample mean equals the expected value" --> Clarify: "LLN says the sample mean converges to the expected value. With a finite sample, there's always some deviation. It gets smaller with more samples, but never hits exactly zero."

**Verification Questions:**
1. "If you flip a fair coin 1,000,000 times, will you get exactly 500,000 heads?"
2. "A gambler has been losing all night. Does LLN suggest they're 'due for a win'?"
3. "Why do insurance companies survive even though they can't predict which specific customers will file claims?"

**Exercise 4.1: LLN Simulation**
"Simulate rolling a fair die 10,000 times. Plot the running average and verify it converges to 3.5 (the true expected value). Then repeat with a loaded die where P(6) = 0.3 and all other faces have equal probability. What does the running average converge to?"

---

### Section 5: Central Limit Theorem

**Core Concept to Teach:**
The Central Limit Theorem (CLT) says: if you take many independent samples from any distribution and compute their mean, the distribution of those sample means is approximately normal -- even if the original distribution is nothing like a bell curve. This is why the normal distribution is everywhere and why so many statistical methods work.

**How to Explain:**
1. The shocking claim: "Take any distribution -- Poisson, Exponential, Uniform, something bizarre and skewed -- draw samples of size n, compute the mean, repeat thousands of times. The histogram of those means will look like a bell curve. Every time."
2. Why it matters: "This is why confidence intervals work. This is why hypothesis tests work. This is why the normal distribution appears so often in practice -- not because individual data points are normal, but because averages and sums of random things are approximately normal."
3. The conditions: "The CLT needs independent samples and a finite variance. For most real-world scenarios, these are satisfied. The bigger the sample size n, the better the normal approximation."

**Simulation to Demonstrate:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Start with an exponential distribution (very skewed, not normal at all)
lam = 2.0
true_mean = 1 / lam  # 0.5

# Draw sample means for different sample sizes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sample_sizes = [1, 5, 30, 100]
n_experiments = 5000

for ax, n in zip(axes.flat, sample_sizes):
    # Each experiment: draw n values, compute the mean
    sample_means = [np.mean(np.random.exponential(scale=1/lam, size=n))
                    for _ in range(n_experiments)]

    ax.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Overlay the theoretical normal (CLT prediction)
    x = np.linspace(min(sample_means), max(sample_means), 100)
    theoretical_std = (1/lam) / np.sqrt(n)
    ax.plot(x, stats.norm.pdf(x, true_mean, theoretical_std), 'r-', linewidth=2)

    ax.set_title(f'Sample size n = {n}')
    ax.set_xlabel('Sample Mean')
    ax.axvline(true_mean, color='green', linestyle='--', alpha=0.7)

plt.suptitle('CLT: Sample Means from Exponential Distribution', fontsize=14)
plt.tight_layout()
plt.show()
```

**Walk Through:**
- "With n=1, the distribution of 'means' is just the exponential distribution itself -- very skewed."
- "With n=5, it's already starting to look symmetric."
- "With n=30, it's essentially a bell curve. The red line is the normal distribution predicted by CLT."
- "With n=100, it's almost perfectly normal. The CLT is working."
- "The green dashed line is the true mean. Notice the sample means are centered on it (LLN) and their spread decreases as n grows."

**The Key Formula:**
"If the original distribution has mean mu and standard deviation sigma, the CLT says the sample mean has:
- Mean = mu (the same as the original)
- Standard deviation = sigma / sqrt(n) (called the 'standard error')

So larger samples give tighter estimates of the mean. This is why polling organizations survey thousands of people -- to make the standard error small."

**Finance Application:**
"A portfolio of many uncorrelated assets: each asset's return has its own distribution (possibly skewed, possibly fat-tailed). But the portfolio return is an average of many returns, and by the CLT, it's approximately normally distributed. This is why portfolio theory uses the normal distribution even though individual stock returns aren't perfectly normal."

**ML Application:**
"Many ML algorithms assume that features or residuals are normally distributed. The CLT provides theoretical justification: if a feature is the sum or average of many small, independent factors, it will be approximately normal. Height is the result of many genetic and environmental factors -- CLT explains why it's normally distributed."

**Why This is 'The Most Important Theorem in Statistics':**
"Ask the learner to think about this: 'Without the CLT, every statistical method would need to know the exact distribution of the data. With the CLT, you can use the same methods (confidence intervals, hypothesis tests, regression) regardless of the underlying distribution, because sample means are always approximately normal. It's a universal approximation theorem for averages.'"

**Common Misconceptions:**
- Misconception: "CLT means all data is normally distributed" --> Clarify: "CLT says sample MEANS are normally distributed, not the data itself. Individual data points can follow any distribution."
- Misconception: "You always need n >= 30" --> Clarify: "30 is a common rule of thumb, not a law. For symmetric distributions, n = 10 may be enough. For heavily skewed distributions, you might need n = 100+. It depends on how 'non-normal' the original distribution is."
- Misconception: "CLT works for everything" --> Clarify: "CLT requires finite variance. Distributions with infinite variance (like the Cauchy distribution) break CLT. In finance, this matters -- some models of extreme events have very heavy tails."

**Verification Questions:**
1. "You roll a fair die and compute the mean. For one die, the distribution is uniform (1-6). If you roll 50 dice and compute the mean, what shape will its distribution have?"
2. "A distribution has mean 100 and standard deviation 20. You take samples of size 400. What's the standard error of the sample mean?"
3. Multiple choice: "The CLT applies to: A) Individual data points B) Sample means C) Population parameters D) Standard deviations"

**Exercise 5.1: CLT Simulation**
"Simulate the CLT using three different starting distributions:
1. Uniform(0, 1) -- symmetric, no skew
2. Exponential(lambda=1) -- right-skewed
3. A custom 'loaded die' distribution -- highly non-uniform

For each: draw 5000 samples of size n=50, compute means, plot the histogram, and overlay the theoretical normal curve predicted by CLT. Verify that all three produce approximately normal distributions of sample means."

---

## Practice Project: Monte Carlo Casino Simulation

**Project Introduction:**
"Now let's put everything together. You'll build a Monte Carlo simulation of casino games that demonstrates every concept from this route: distributions model the games, expected value quantifies the house edge, LLN shows how profits converge, and CLT explains the distribution of outcomes across many sessions."

**Requirements:**
Present these requirements:
1. Simulate three casino games: roulette (bet on red), simplified blackjack (hit or stand based on hand total), and craps (pass-line bet)
2. Run each game for 1,000 bets per session, across 1,000 sessions (gamblers)
3. For each game: compute theoretical expected value and compare to simulated average
4. Plot the convergence of average profit over bets (LLN visualization)
5. Plot the histogram of session outcomes across all 1,000 gamblers (CLT visualization)
6. Calculate the probability of a gambler finishing a session with a profit

**Scaffolding Strategy:**
1. **If they want to try alone**: "Start with roulette -- it's the simplest to simulate. Get that working, then add the other games."
2. **If they want guidance**: Walk through roulette together, then let them implement the others
3. **If they're unsure**: Provide the roulette skeleton and have them fill in the logic

**Roulette Skeleton:**
```python
import numpy as np

def simulate_roulette_session(n_bets, bet_amount=1):
    """Simulate n_bets on red in American roulette. Return total profit."""
    p_red = 18/38
    wins = np.random.random(n_bets) < p_red
    profit = np.sum(np.where(wins, bet_amount, -bet_amount))
    return profit
```

**Checkpoints During Project:**
- After roulette simulation: "Run 1000 sessions. What's the average profit? How does it compare to the theoretical EV?"
- After adding LLN plot: "Does the running average converge? How many bets does it take?"
- After adding CLT histogram: "What shape is the histogram of session outcomes? Can you overlay the predicted normal curve?"
- Final: "What fraction of gamblers walk away with a profit? Is it close to what you'd predict from the normal distribution?"

**Code Review Approach:**
- Check that they're using numpy vectorization (not Python loops for individual bets)
- Verify they compute theoretical EV independently and compare to simulation
- Make sure the CLT histogram has a normal curve overlay with correct parameters
- Ask: "What would change if you simulated a game with a smaller house edge? With a bigger one?"

**If They Get Stuck:**
- On simulation mechanics: "Start simple. Can you simulate a single bet? Good. Now wrap it in a loop for n bets. Now wrap that in a loop for n sessions."
- On CLT visualization: "The session total profit is the sum of 1000 independent bets. By CLT, this sum is approximately normal. Its mean is 1000 * EV_per_bet. Its standard deviation is sqrt(1000) * std_per_bet."
- On probability of profit: "If session outcomes are approximately normal with mean mu and std sigma, P(profit > 0) = P(X > 0) = 1 - norm.cdf(0, mu, sigma)"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned:
1. **Distributions** are complete models of random processes -- they tell you the probability of every possible outcome
2. **Discrete distributions** (Bernoulli, Binomial, Poisson, Geometric) model countable outcomes
3. **Continuous distributions** (Uniform, Normal, Exponential) model outcomes on a continuous scale
4. **Expected value and variance** summarize a distribution's center and spread -- they quantify risk and reward
5. **Law of Large Numbers** ensures averages converge to expected values -- this is why casinos always win
6. **Central Limit Theorem** ensures sample means are approximately normal -- this is why the bell curve is everywhere"

Ask them to explain one concept back to you. A good test: "Can you explain to me why a casino is guaranteed to make money in the long run, using the specific terms we learned today?"

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with probability distributions?"
- 1-4: Suggest reviewing specific sections, focus on hands-on exercises
- 5-7: Normal for an intermediate topic. Suggest building more simulations for practice
- 8-10: Ready for statistical inference and Bayesian statistics

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice more, try simulating other games -- poker, sports betting, options pricing"
- "For the next route, statistical-inference builds directly on CLT to teach confidence intervals and hypothesis testing"
- "If you're interested in Bayesian methods, bayesian-statistics combines distributions with Bayes' theorem"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Slow down, use more concrete examples with money (gambling is great for this)
- Focus on one distribution at a time -- don't compare them until each is solid
- Let them experiment with scipy.stats interactively rather than working through theory
- Draw the connection: "A distribution is just a table of outcomes and probabilities. We're packaging that table into a formula."
- Check prerequisites: if they can't compute basic expected values, revisit probability-fundamentals

### If Learner is Excelling
- Move faster through the basic distributions, spend more time on applications
- Introduce distribution fitting: "Given real data, which distribution fits best?"
- Discuss the limitations of each distribution: when does the normal approximation break down?
- Bring in moment-generating functions or characteristic functions as a preview of advanced topics
- Challenge them with multi-distribution problems: "A factory produces items with normally distributed weights. Each item is inspected with a Bernoulli trial (p = probability of passing). How many items in a batch of 100 pass inspection?"

### If Learner Seems Disengaged
- Check in: "Is this too abstract? Let's make it more concrete."
- Switch to their domain of interest: if finance bores them, try sports. If sports bores them, try ML
- Move to the practice project early -- some learners need to build before they understand theory
- Ask what they want to be able to DO, then connect the material to that goal

### Different Learning Styles
- **Visual learners**: Plot every distribution. Show histograms. Animate convergence.
- **Hands-on learners**: Give them scipy.stats and let them experiment. "What happens if you change lambda from 2 to 10?"
- **Conceptual learners**: Spend time on why CLT works (averaging washes out extreme values), why LLN is inevitable (independence + averaging = convergence)
- **Example-driven learners**: Lead with the casino simulation, then work backward to the theory that explains what they observed

---

## Troubleshooting Common Issues

### Technical Setup Problems
- **scipy not installed**: `pip install scipy`
- **Matplotlib not showing plots**: Try `plt.savefig('plot.png')` as an alternative, or use `%matplotlib inline` in Jupyter
- **Import errors**: Make sure they're using `from scipy import stats`, not `import scipy.stats` (both work, but the former is more common in examples)

### Concept-Specific Confusion

**If confused about PMF vs PDF:**
- "PMF is for discrete distributions: it gives the probability of each exact value. PDF is for continuous distributions: it gives the density, and you need to compute area under the curve for actual probabilities."
- "Quick test: can the outcome only be a whole number (0, 1, 2, 3...)? Then PMF. Can it be any decimal value? Then PDF."

**If confused about when to use which distribution:**
- Walk through the decision tree: "Is it yes/no? Bernoulli. Is it counting successes in fixed trials? Binomial. Is it counting events in an interval? Poisson. Is it time between events? Exponential. Is it a measurement that varies around an average? Probably Normal."

**If confused about CDF:**
- "CDF(x) answers one question: what's the probability that X is less than or equal to x? That's it. P(X <= x). Everything else is subtraction: P(X > x) = 1 - CDF(x). P(a < X < b) = CDF(b) - CDF(a)."

**If confused about LLN vs CLT:**
- "LLN says: the sample average converges to the true average. CLT says: the distribution of the sample average is approximately normal. LLN is about convergence (where does the average end up?). CLT is about the shape (what does the distribution of averages look like?)."

---

## Additional Resources to Suggest

**If they want more practice:**
- Simulate different casino games and compute house edges
- Model real sports data (batting averages, scoring distributions) with appropriate distributions
- Use real stock return data from Yahoo Finance and test normality assumptions

**If they want deeper understanding:**
- Probability and Statistics for Engineering and the Sciences (Jay Devore) -- thorough textbook treatment
- Think Stats by Allen Downey -- free, Python-based, simulation-heavy approach
- 3Blue1Brown YouTube channel -- visual explanations of CLT and distributions

**If they want to see real applications:**
- Quantitative finance: options pricing relies on distributional assumptions (Black-Scholes uses normal)
- Sports analytics: Poisson models for soccer scoring, Binomial models for basketball shooting
- ML: Naive Bayes classifiers, Gaussian Mixture Models, distributional reinforcement learning

---

## Teaching Notes

**Key Emphasis Points:**
- Really emphasize the connection between distributions and real-world applications. Every distribution should have at least one gambling, one finance, and one ML example
- The LLN section is the emotional climax for gambling-motivated learners: this is THE answer to "why does the house always win?"
- CLT is the most conceptually challenging section. Take extra time here. The simulation is essential -- don't try to teach CLT without running the simulation
- Fat tails in finance is a critical caveat. Don't let the learner walk away thinking "everything is normal"

**Pacing Guidance:**
- Don't rush Section 1 (Discrete Distributions) -- getting the intuition right here makes everything else easier
- Section 2 (Continuous) can be faster if they grasped discrete well -- the concepts transfer
- Section 3 (E[X] and Var) is review plus application -- pace depends on how solid their expected value knowledge is
- Allow plenty of time for the practice project -- it ties everything together

**Success Indicators:**
You'll know they've got it when they:
- Can look at a real-world scenario and immediately identify which distribution applies
- Can explain the casino's business model using LLN without prompting
- Can describe CLT in their own words and explain why it matters for statistics
- Build working simulations that match theoretical predictions
