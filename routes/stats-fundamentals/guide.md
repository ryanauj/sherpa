---
title: Statistics Fundamentals
route_map: /routes/stats-fundamentals/map.md
paired_sherpa: /routes/stats-fundamentals/sherpa.md
prerequisites:
  - Basic arithmetic
  - Comfort with a calculator or spreadsheet
topics:
  - Data Types and Collection
  - Measures of Central Tendency
  - Measures of Spread
  - Data Visualization
  - Exploratory Data Analysis
---

# Statistics Fundamentals

## Overview

Statistics is the science of learning from data. Every time you read that a team averages 110 points per game, that a stock returned 12% last year, or that a machine learning model achieved 95% accuracy, you are looking at descriptive statistics. Understanding these fundamentals lets you see past the headline number and ask the right follow-up questions: Is that average misleading? How consistent is that return? What does the full distribution look like?

This route teaches descriptive statistics from the ground up using real-world examples from three domains:

- **Sports betting and casino gambling**: Analyzing player performance, understanding point spreads, and seeing why the house always wins in the long run.
- **Finance**: Evaluating stock returns, measuring risk through volatility, and comparing investment options.
- **Data science and machine learning**: Exploring datasets, understanding feature distributions, and preparing data for modeling.

By the end, you will be able to take a raw dataset, compute summary statistics, build meaningful visualizations, and draw actionable conclusions.

> **Note for AI assistants**: This route has a paired teaching guide at `/routes/stats-fundamentals/sherpa.md` that provides structured guidance for teaching this material interactively.

## Learning Objectives

By the end of this route, you will be able to:
- Classify data into categorical (nominal, ordinal) and numerical (discrete, continuous) types
- Compute and interpret mean, median, and mode and choose the right measure for a given context
- Calculate range, variance, standard deviation, and IQR and explain what they reveal about variability and risk
- Create histograms, box plots, and scatter plots to visualize distributions and relationships
- Perform a complete exploratory data analysis on a real-world dataset

## Prerequisites

Before starting, you should be comfortable with:
- Basic arithmetic: addition, subtraction, multiplication, division, squaring numbers, and square roots
- Using a calculator or spreadsheet to perform calculations

If you have programming experience, great -- the code examples will feel natural. If not, do not worry. The Python code is explained line by line, and you can follow along even if you have never written code before.

## Setup

This route uses Python with three libraries. If you already have Python installed, install the libraries:

```bash
pip install numpy pandas matplotlib
```

**Verify your setup:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"NumPy version:      {np.__version__}")
print(f"pandas version:     {pd.__version__}")
print(f"matplotlib version: {plt.matplotlib.__version__}")
```

You should see version numbers printed without errors. If you do not have Python, you can use [Google Colab](https://colab.research.google.com/) -- a free browser-based environment with all libraries pre-installed.

**Alternative**: If you prefer not to use Python, you can follow along with a spreadsheet (Google Sheets, Excel, or LibreOffice Calc). The concepts are the same; only the tool differs.

---

## Section 1: Types of Data

### What Are Data Types?

Before you can analyze data, you need to know what kind of data you have. This is not about programming types like `int` or `string` -- it is about the statistical nature of the information. The data type determines which summary statistics make sense, which charts are appropriate, and how you should interpret results.

All data falls into two broad families: **categorical** and **numerical**.

### Categorical Data

Categorical data represents groups, labels, or qualities. There are two subtypes:

**Nominal** -- categories with no natural order:
- Sports: team name, player position (guard, forward, center), home vs away
- Finance: stock ticker (AAPL, GOOGL), industry sector (tech, healthcare, energy)
- ML: color, country, device type

**Ordinal** -- categories with a meaningful order, but no consistent numeric distance between them:
- Sports: draft round (1st, 2nd, undrafted), league standing (1st place through last)
- Finance: credit rating (AAA, AA, A, BBB), risk level (low, medium, high)
- ML: survey responses (strongly disagree, disagree, neutral, agree, strongly agree)

The key distinction: with ordinal data, you know that AAA is better than BBB, but you cannot say it is "twice as good." The gaps between categories are not necessarily equal.

### Numerical Data

Numerical data represents quantities you can measure or count. Two subtypes:

**Discrete** -- countable values, typically whole numbers:
- Sports: goals scored, wins, assists, number of fouls
- Finance: number of shares traded, number of transactions
- ML: count of features, number of training epochs

**Continuous** -- measurable values that can take any number in a range, including decimals:
- Sports: batting average (.312), completion percentage (67.4%), player height (6'7")
- Finance: stock price ($142.58), daily return percentage (+1.23%), P/E ratio (22.4)
- ML: model accuracy (0.943), loss value (0.0372), feature importance score

### Why Data Types Matter

The data type dictates what you can do:

| Operation | Nominal | Ordinal | Discrete | Continuous |
|-----------|---------|---------|----------|------------|
| Count frequencies | Yes | Yes | Yes | Yes |
| Rank / order | No | Yes | Yes | Yes |
| Compute mean | No | No | Yes | Yes |
| Compute differences | No | No | Yes | Yes |

Computing the "average jersey number" or the "mean zip code" is technically possible but meaningless. Jersey numbers and zip codes are nominal -- they are labels, not quantities.

### Code Example: Identifying Data Types

```python
import pandas as pd

# Sample sports dataset
data = {
    'team': ['Lakers', 'Celtics', 'Warriors', 'Heat', 'Bucks'],
    'conference': ['West', 'East', 'West', 'East', 'East'],
    'playoff_seed': [7, 2, 6, 8, 1],
    'wins': [43, 57, 44, 44, 58],
    'ppg': [117.2, 112.8, 118.9, 109.5, 116.9],
    'win_pct': [0.524, 0.695, 0.537, 0.537, 0.707]
}
df = pd.DataFrame(data)
print(df)
print("\nPandas dtypes:")
print(df.dtypes)
```

**Expected Output:**
```
       team conference  playoff_seed  wins    ppg  win_pct
0   Lakers       West             7    43  117.2    0.524
1  Celtics       East             2    57  112.8    0.695
2 Warriors       West             6    44  118.9    0.537
3     Heat       East             8    44  109.5    0.537
4    Bucks       East             1    58  116.9    0.707

Pandas dtypes:
team            object
conference      object
playoff_seed     int64
wins             int64
ppg            float64
win_pct        float64
dtype: object
```

**What's happening here:**
- pandas calls strings `object` and numbers `int64` or `float64`. But these are programming types, not statistical types.
- `playoff_seed` shows up as `int64`, but statistically it is ordinal -- the difference between seed 1 and seed 2 is not the same as between seed 7 and seed 8.
- You, the analyst, must know the statistical type. The software cannot figure that out for you.

### Exercise 1.1: Classify the Columns

Here is a row from a sports betting dataset:

```
game_id:       20231105_LAL_BOS
home_team:     BOS
away_team:     LAL
home_score:    114
away_score:    109
overtime:      True
spread:        -5.5
total_points:  223
venue:         TD Garden
attendance:    19156
```

**Task:** Classify each column as nominal, ordinal, discrete, or continuous.

<details>
<summary>Hint 1: Start with the obvious ones</summary>

`home_team`, `away_team`, `venue`, and `game_id` are clearly labels -- they are names and identifiers with no numeric meaning.
</details>

<details>
<summary>Hint 2: Think about the numbers</summary>

Can `home_score` be 114.5? No -- scores are whole numbers (discrete). Can `spread` be -5.5? Yes -- spreads use half-points (continuous). What about `overtime`? It is True/False -- a boolean category.
</details>

<details>
<summary>Click to see solution</summary>

| Column | Type | Reasoning |
|--------|------|-----------|
| game_id | Nominal | Identifier, no order or numeric meaning |
| home_team | Nominal | Team name, no inherent order |
| away_team | Nominal | Team name, no inherent order |
| home_score | Discrete | Whole number count of points |
| away_score | Discrete | Whole number count of points |
| overtime | Nominal | Boolean category (True/False) |
| spread | Continuous | Can take half-point values like -5.5 |
| total_points | Discrete | Sum of two discrete values, still a whole number |
| venue | Nominal | Place name, no inherent order |
| attendance | Discrete | Count of people, whole numbers |

**Key insight:** `overtime` might look like it could be ordinal (True > False?), but there is no meaningful ordering -- it is simply a category with two values.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Distinguish between categorical and numerical data
- [ ] Explain the difference between nominal and ordinal
- [ ] Explain the difference between discrete and continuous
- [ ] Identify why some numbers (jersey numbers, zip codes) are actually categorical

---

## Section 2: Measures of Central Tendency

### What Is Central Tendency?

Central tendency answers the question: "What is a typical value in this dataset?" It gives you a single number that represents the center of the data. The three main measures are **mean**, **median**, and **mode**.

### Mean (Arithmetic Average)

The mean is what most people think of as "the average." Add up all the values and divide by how many there are.

**Formula:** mean = (sum of all values) / (number of values)

```python
import numpy as np

# Points scored by a basketball team in 10 games
points = [98, 102, 105, 99, 145, 101, 100, 97, 103, 100]

mean_pts = np.mean(points)
print(f"Mean points per game: {mean_pts}")
```

**Expected Output:**
```
Mean points per game: 105.0
```

The mean is 105. But look at the data -- most games are in the 97-105 range. That one 145-point game pulls the mean up. The mean is sensitive to outliers.

### Median (Middle Value)

The median is the middle value when you sort the data. Half the values are above it, half below. If there is an even number of values, the median is the average of the two middle values.

```python
median_pts = np.median(points)
print(f"Median points per game: {median_pts}")

# To see why, let's sort the data:
sorted_pts = sorted(points)
print(f"Sorted: {sorted_pts}")
print(f"Middle two values: {sorted_pts[4]} and {sorted_pts[5]}")
print(f"Median: ({sorted_pts[4]} + {sorted_pts[5]}) / 2 = {median_pts}")
```

**Expected Output:**
```
Median points per game: 100.5
Sorted: [97, 98, 99, 100, 100, 101, 102, 103, 105, 145]
Middle two values: 100 and 101
Median: (100 + 101) / 2 = 100.5
```

The median (100.5) is much closer to a "typical" game than the mean (105.0). The 145-point outlier has almost no effect on the median.

### Mode (Most Frequent Value)

The mode is the value that appears most often. It is the only central tendency measure that works with categorical data.

```python
from collections import Counter

mode_pts = Counter(points).most_common(1)[0][0]
print(f"Mode: {mode_pts}")

# For categorical data:
positions = ['Guard', 'Guard', 'Forward', 'Center', 'Guard',
             'Forward', 'Guard', 'Center', 'Forward', 'Guard']
mode_pos = Counter(positions).most_common(1)[0][0]
print(f"Most common position: {mode_pos}")
```

**Expected Output:**
```
Mode: 100
Most common position: Guard
```

### When to Use Each Measure

| Situation | Best Measure | Why |
|-----------|-------------|-----|
| Symmetric data, no outliers | Mean | Uses all data, most precise |
| Skewed data or outliers | Median | Resistant to extreme values |
| Categorical data | Mode | Only option for labels |
| Salary/income data | Median | Always skewed by high earners |
| Home prices | Median | A few mansions distort the mean |
| Sports averages | Depends | Check for blowout games |
| Stock returns | Both | Compare mean vs median to detect skew |

### Real-World Example: Mean vs Median Income

```python
import numpy as np

# Simulated annual salaries at a small company (in thousands)
salaries = [45, 48, 52, 55, 50, 47, 53, 51, 49, 450]

print(f"Mean salary:   ${np.mean(salaries):.0f}k")
print(f"Median salary: ${np.median(salaries):.0f}k")
print(f"Difference:    ${np.mean(salaries) - np.median(salaries):.0f}k")
```

**Expected Output:**
```
Mean salary:   $90k
Median salary: $51k
Difference:    $39k
```

The CEO making $450k pulls the mean up to $90k. But 9 out of 10 employees earn between $45k and $55k. The median ($51k) tells you what a typical employee actually earns. This is why the US Census Bureau reports median household income, not mean.

### Sports Betting Application

If you are evaluating a team for an over/under bet, the mean points per game can mislead you. A team that had one 160-point overtime game will have an inflated average. Check the median to see what a "normal" game looks like.

```python
import numpy as np

# Team that had a few blowout wins
team_scores = [98, 101, 95, 160, 99, 103, 97, 155, 100, 102]

print(f"Mean:   {np.mean(team_scores):.1f}")
print(f"Median: {np.median(team_scores):.1f}")
print(f"\nThe over/under line might be set near {np.mean(team_scores):.0f}.")
print(f"But a typical game is closer to {np.median(team_scores):.0f}.")
```

**Expected Output:**
```
Mean:   111.0
Median: 100.5

The over/under line might be set near 111.
But a typical game is closer to 101.
```

### Exercise 2.1: Compare Central Tendency Measures

**Task:** Compute the mean, median, and mode for each dataset below. Then explain which measure is most informative and why.

```python
# Dataset A: Daily stock returns (%)
returns_a = [1.2, -0.5, 0.8, 1.1, -0.3, 0.7, 25.0, 0.9, -0.2, 0.6]

# Dataset B: Customer ratings (1-5 stars)
ratings_b = [5, 4, 5, 3, 5, 4, 5, 2, 5, 4]

# Dataset C: Number of goals scored per soccer match
goals_c = [1, 2, 1, 0, 3, 1, 1, 2, 0, 1, 4, 1, 2, 1, 0]
```

<details>
<summary>Hint 1: Getting started</summary>

Use `np.mean()` and `np.median()` for each dataset. For mode, use `Counter().most_common(1)`. Remember to think about what type of data each dataset contains.
</details>

<details>
<summary>Hint 2: Which measure is most informative?</summary>

For Dataset A, check whether the mean is being distorted by an outlier. For Dataset B, think about whether mean makes sense for star ratings. For Dataset C, all three measures are useful -- but which gives the best single summary?
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from collections import Counter

# Dataset A
returns_a = [1.2, -0.5, 0.8, 1.1, -0.3, 0.7, 25.0, 0.9, -0.2, 0.6]
print("Dataset A (Stock Returns):")
print(f"  Mean:   {np.mean(returns_a):.2f}%")
print(f"  Median: {np.median(returns_a):.2f}%")
print(f"  Mode:   {Counter(returns_a).most_common(1)[0][0]}%")
# Mean: 2.93%, Median: 0.75%, Mode: varies (no repeats likely)

# Dataset B
ratings_b = [5, 4, 5, 3, 5, 4, 5, 2, 5, 4]
print("\nDataset B (Customer Ratings):")
print(f"  Mean:   {np.mean(ratings_b):.1f}")
print(f"  Median: {np.median(ratings_b):.1f}")
print(f"  Mode:   {Counter(ratings_b).most_common(1)[0][0]}")
# Mean: 4.2, Median: 4.5, Mode: 5

# Dataset C
goals_c = [1, 2, 1, 0, 3, 1, 1, 2, 0, 1, 4, 1, 2, 1, 0]
print("\nDataset C (Goals per Match):")
print(f"  Mean:   {np.mean(goals_c):.2f}")
print(f"  Median: {np.median(goals_c):.1f}")
print(f"  Mode:   {Counter(goals_c).most_common(1)[0][0]}")
# Mean: 1.33, Median: 1.0, Mode: 1
```

**Expected Output:**
```
Dataset A (Stock Returns):
  Mean:   2.93%
  Median: 0.75%
  Mode:   1.2%

Dataset B (Customer Ratings):
  Mean:   4.2
  Median: 4.5
  Mode:   5

Dataset C (Goals per Match):
  Mean:   1.33
  Median: 1.0
  Mode:   1
```

**Interpretation:**
- **Dataset A**: The median (0.75%) is most informative. The 25% outlier inflates the mean to 2.93%, which does not represent a typical day.
- **Dataset B**: The mode (5) is most informative for star ratings -- it tells you the most common response. The mean (4.2) is useful too, but averaging ordinal data is debatable.
- **Dataset C**: The mode (1) is most informative -- one goal is the most common outcome. The mean (1.33) is reasonable here since there are no extreme outliers.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Calculate mean, median, and mode by hand for a small dataset
- [ ] Explain when the mean is misleading (outliers, skewed data)
- [ ] Choose the appropriate measure for a given situation
- [ ] Describe the difference between mean and median salary/income

---

## Section 3: Measures of Spread

### Why Central Tendency Is Not Enough

Consider two basketball teams that both average 100 points per game:

- **Team A**: scores between 95 and 105 every night. Consistent, predictable.
- **Team B**: scores anywhere from 75 to 130. Volatile, unpredictable.

The mean tells you they are "the same." But they are not. If you are betting on the over/under, managing a portfolio, or training an ML model, the spread around the center matters enormously.

### Range

The simplest measure of spread: maximum value minus minimum value.

```python
import numpy as np

team_a = [98, 102, 100, 97, 103, 101, 99, 100, 104, 96]
team_b = [130, 85, 110, 75, 120, 88, 115, 78, 105, 94]

print(f"Team A range: {max(team_a) - min(team_a)}")  # 8
print(f"Team B range: {max(team_b) - min(team_b)}")  # 55
```

**Expected Output:**
```
Team A range: 8
Team B range: 55
```

The range is easy to understand but fragile -- a single extreme value can blow it up. It uses only two data points and ignores everything in between.

### Variance and Standard Deviation

Variance measures the average squared distance from the mean. Standard deviation is the square root of variance, bringing it back to the original units.

**How variance is computed, step by step:**
1. Find the mean
2. Subtract the mean from each value (these are "deviations")
3. Square each deviation (to make them all positive)
4. Average the squared deviations

For a **sample** (which is almost always what you have), divide by (n - 1) instead of n. This is called Bessel's correction.

```python
import numpy as np

data = [4, 8, 6, 5, 3, 7, 9, 5]

# Step by step
mean = np.mean(data)
deviations = [x - mean for x in data]
squared_devs = [d**2 for d in deviations]
variance = sum(squared_devs) / (len(data) - 1)  # sample variance
std_dev = variance ** 0.5

print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Deviations: {[f'{d:+.2f}' for d in deviations]}")
print(f"Squared deviations: {[f'{d:.2f}' for d in squared_devs]}")
print(f"Sample variance: {variance:.2f}")
print(f"Sample std dev:  {std_dev:.2f}")

# Verify with numpy
print(f"\nNumPy variance:  {np.var(data, ddof=1):.2f}")
print(f"NumPy std dev:   {np.std(data, ddof=1):.2f}")
```

**Expected Output:**
```
Data: [4, 8, 6, 5, 3, 7, 9, 5]
Mean: 5.875
Deviations: ['-1.88', '+2.12', '+0.12', '-0.88', '-2.88', '+1.12', '+3.12', '-0.88']
Squared deviations: ['3.52', '4.52', '0.02', '0.77', '8.27', '1.27', '9.77', '0.77']
Sample variance: 4.12
Sample std dev:  2.03

NumPy variance:  4.12
NumPy std dev:   2.03
```

**Important:** In numpy, use `ddof=1` for sample statistics. The default `ddof=0` computes population statistics, which underestimates the true variability when working with a sample.

### Interquartile Range (IQR)

The IQR is the range of the middle 50% of the data: Q3 (75th percentile) minus Q1 (25th percentile). Like the median, it is resistant to outliers.

```python
import numpy as np

data = [4, 8, 6, 5, 3, 7, 9, 5, 50]  # Note the outlier: 50

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(f"Data (sorted): {sorted(data)}")
print(f"Q1 (25th percentile): {q1}")
print(f"Q3 (75th percentile): {q3}")
print(f"IQR: {iqr}")
print(f"Range: {max(data) - min(data)}")
print(f"Std Dev: {np.std(data, ddof=1):.2f}")
```

**Expected Output:**
```
Data (sorted): [3, 4, 5, 5, 6, 7, 8, 9, 50]
Q1 (25th percentile): 4.5
Q3 (75th percentile): 8.5
IQR: 4.0
Range: 47
Std Dev: 14.24
```

The range (47) and std dev (14.24) are both inflated by the outlier. The IQR (4.0) tells you that the middle half of the data is tightly clustered, which is the more useful story.

### Comparing Team A and Team B

```python
import numpy as np

team_a = [98, 102, 100, 97, 103, 101, 99, 100, 104, 96]
team_b = [130, 85, 110, 75, 120, 88, 115, 78, 105, 94]

for name, data in [("Team A", team_a), ("Team B", team_b)]:
    print(f"\n{name}:")
    print(f"  Mean:     {np.mean(data):.1f}")
    print(f"  Median:   {np.median(data):.1f}")
    print(f"  Std Dev:  {np.std(data, ddof=1):.1f}")
    print(f"  IQR:      {np.percentile(data,75) - np.percentile(data,25):.1f}")
    print(f"  Range:    {max(data) - min(data)}")
```

**Expected Output:**
```
Team A:
  Mean:     100.0
  Median:   100.0
  Std Dev:  2.5
  IQR:      3.2
  Range:    8

Team B:
  Mean:     100.0
  Median:   102.5
  Std Dev:  18.9
  IQR:      28.5
  Range:    55
```

Same mean. Completely different teams. The spread tells the real story.

### Casino Application: Why the House Always Wins

In casino gambling, the expected value (mean outcome) is always slightly negative for the player. But variance is what keeps people playing -- in the short term, swings make winning feel possible.

```python
import numpy as np

np.random.seed(42)

# Simulate 1000 roulette bets on red, $10 each
# American roulette: 18 red, 18 black, 2 green (38 total)
win_prob = 18 / 38
loss_prob = 20 / 38

outcomes = np.random.choice([10, -10], size=1000, p=[win_prob, loss_prob])

print(f"Expected value per bet: ${10 * win_prob + (-10) * loss_prob:.2f}")
print(f"Std dev per bet: ${np.std(outcomes, ddof=1):.2f}")
print(f"\nAfter 10 bets:   ${np.sum(outcomes[:10])}")
print(f"After 100 bets:  ${np.sum(outcomes[:100])}")
print(f"After 1000 bets: ${np.sum(outcomes)}")
```

**Expected Output:**
```
Expected value per bet: $-0.53
Std dev per bet: $10.00

After 10 bets:   $0
After 100 bets:  $-40
After 1000 bets: $-60
```

After 10 bets, you might be even or ahead -- the variance masks the edge. After 1,000 bets, the mean dominates and the casino collects. This is the law of large numbers in action. The house does not need a big edge; it just needs many bets.

### Finance Application: Volatility

In finance, standard deviation of returns is called **volatility**. It is the primary measure of risk.

```python
import numpy as np

np.random.seed(42)

# Simulated annual returns over 10 years
stock_a = np.random.normal(0.10, 0.05, 10)  # 10% avg return, 5% volatility
stock_b = np.random.normal(0.10, 0.25, 10)  # 10% avg return, 25% volatility

for name, returns in [("Stock A (low vol)", stock_a), ("Stock B (high vol)", stock_b)]:
    print(f"\n{name}:")
    print(f"  Mean annual return:  {np.mean(returns)*100:.1f}%")
    print(f"  Volatility (std):    {np.std(returns, ddof=1)*100:.1f}%")
    print(f"  Best year:           {np.max(returns)*100:+.1f}%")
    print(f"  Worst year:          {np.min(returns)*100:+.1f}%")
```

**Expected Output:**
```
Stock A (low vol):
  Mean annual return:  10.4%
  Volatility (std):    4.5%
  Best year:           +17.6%
  Worst year:          +3.3%

Stock B (high vol):
  Mean annual return:  12.0%
  Volatility (std):    22.5%
  Best year:           +47.9%
  Worst year:          -16.4%
```

Both stocks have similar average returns. But Stock B's high volatility means you could lose 16% in a bad year or gain 48% in a good one. A conservative investor prefers Stock A; an aggressive investor might accept Stock B's risk for the chance of higher gains.

### Exercise 3.1: Spread and Decision-Making

**Task:** Two mutual funds have the following monthly returns (%) over 8 months. Compute the mean, standard deviation, and IQR for each. Then answer: Which fund would you recommend to a retiree who needs stable income? Which to a young investor with a long time horizon?

```python
fund_stable = [0.8, 1.0, 0.6, 0.9, 0.7, 1.1, 0.8, 0.9]
fund_growth = [3.5, -2.0, 4.2, -1.5, 5.0, -3.0, 6.1, -0.8]
```

<details>
<summary>Hint 1: Computing the statistics</summary>

Use `np.mean()`, `np.std(data, ddof=1)`, and `np.percentile()` for Q1 and Q3. Remember IQR = Q3 - Q1.
</details>

<details>
<summary>Hint 2: Interpreting for different investors</summary>

A retiree needs predictable returns because they are withdrawing money regularly. A young investor can tolerate swings because they have time to recover from bad months.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

fund_stable = [0.8, 1.0, 0.6, 0.9, 0.7, 1.1, 0.8, 0.9]
fund_growth = [3.5, -2.0, 4.2, -1.5, 5.0, -3.0, 6.1, -0.8]

for name, data in [("Stable Fund", fund_stable), ("Growth Fund", fund_growth)]:
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    print(f"\n{name}:")
    print(f"  Mean:    {np.mean(data):.2f}%")
    print(f"  Std Dev: {np.std(data, ddof=1):.2f}%")
    print(f"  IQR:     {q3 - q1:.2f}%")
```

**Expected Output:**
```
Stable Fund:
  Mean:    0.85%
  Std Dev: 0.16%
  IQR:     0.17%

Growth Fund:
  Mean:    1.44%
  Std Dev: 3.55%
  IQR:     5.73%
```

**Interpretation:**
- **Retiree**: Recommend the Stable Fund. Its mean return (0.85%) is lower, but its standard deviation (0.16%) is tiny. Returns are predictable month to month -- no nasty surprises.
- **Young investor**: The Growth Fund might be better. Its mean return (1.44%) is higher, but it comes with large swings (std dev 3.55%). Over a long time horizon, the higher average compounds while the volatility evens out.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain what standard deviation tells you in plain language
- [ ] Compute variance and standard deviation step by step
- [ ] Describe the difference between range and IQR
- [ ] Use `ddof=1` for sample statistics in numpy
- [ ] Connect spread to real decisions (betting, investing, risk)

---

## Section 4: Data Visualization

### Why Visualize?

Numbers are essential, but they can hide important patterns. Consider Anscombe's Quartet -- four datasets with nearly identical means, standard deviations, and correlations, but wildly different shapes when plotted. Visualization reveals what summary statistics alone cannot: skewness, clusters, outliers, and relationships.

### Histograms: The Shape of a Distribution

A histogram divides a numerical variable into bins and counts how many values fall in each bin. It answers: "What does the distribution look like?"

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# NBA player salaries (simulated, in millions)
salaries = np.concatenate([
    np.random.exponential(5, 200),   # Most players: lower salaries
    np.random.normal(35, 5, 20)      # Stars: high salaries
])
salaries = np.clip(salaries, 0.5, 50)

plt.figure(figsize=(8, 4))
plt.hist(salaries, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Annual Salary ($ millions)')
plt.ylabel('Number of Players')
plt.title('Distribution of NBA Player Salaries')
plt.axvline(np.mean(salaries), color='red', linestyle='--', label=f'Mean: ${np.mean(salaries):.1f}M')
plt.axvline(np.median(salaries), color='green', linestyle='--', label=f'Median: ${np.median(salaries):.1f}M')
plt.legend()
plt.tight_layout()
plt.savefig('salary_histogram.png', dpi=100, bbox_inches='tight')
plt.show()
```

**What to look for:**
- **Shape**: Is the distribution symmetric, right-skewed, left-skewed, or bimodal?
- **Center**: Where is the peak?
- **Spread**: How wide is the distribution?
- **Outliers**: Are there isolated bars far from the main cluster?
- **Mean vs median**: In skewed data, the mean and median will be in different positions.

In this example, the salary distribution is right-skewed: most players earn relatively modest salaries, but a few stars earn $30M+. The mean is pulled right; the median better represents a "typical" player.

### Box Plots: Comparing Groups

A box plot shows the median (line), quartiles (box), range (whiskers at 1.5 * IQR), and outliers (dots). It is excellent for comparing distributions across groups.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Points per game by position
guards = np.random.normal(15, 6, 80).clip(0, 40)
forwards = np.random.normal(14, 5, 70).clip(0, 35)
centers = np.random.normal(12, 4, 50).clip(0, 30)

plt.figure(figsize=(8, 5))
bp = plt.boxplot([guards, forwards, centers],
                  labels=['Guards', 'Forwards', 'Centers'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue'))
plt.ylabel('Points Per Game')
plt.title('Scoring Distribution by Position')
plt.tight_layout()
plt.savefig('position_boxplot.png', dpi=100, bbox_inches='tight')
plt.show()

# Print the statistics the box plot shows
for name, data in [('Guards', guards), ('Forwards', forwards), ('Centers', centers)]:
    print(f"{name}: median={np.median(data):.1f}, "
          f"Q1={np.percentile(data,25):.1f}, Q3={np.percentile(data,75):.1f}, "
          f"IQR={np.percentile(data,75)-np.percentile(data,25):.1f}")
```

**Expected Output (statistics):**
```
Guards: median=15.0, Q1=10.8, Q3=19.2, IQR=8.4
Forwards: median=14.3, Q1=10.6, Q3=17.9, IQR=7.3
Centers: median=12.2, Q1=9.4, Q3=14.9, IQR=5.5
```

**Reading a box plot:**
- The line inside the box is the **median** (not the mean).
- The box spans from **Q1** to **Q3** -- the middle 50% of the data.
- The whiskers extend to the most extreme values within 1.5 * IQR of the box.
- Dots beyond the whiskers are **outliers**.

From this plot, you can immediately see that guards have the highest median scoring and the widest range, while centers are more clustered around a lower value.

### Scatter Plots: Relationships Between Variables

A scatter plot shows two numerical variables against each other, one on each axis. Each data point becomes a dot. It answers: "Is there a relationship between these two variables?"

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
n = 30

# Simulated NBA team data
ppg = np.random.normal(110, 5, n)
opp_ppg = np.random.normal(110, 5, n)
point_diff = ppg - opp_ppg
wins = np.clip(41 + point_diff * 2.5 + np.random.normal(0, 3, n), 10, 72).astype(int)

plt.figure(figsize=(8, 5))
plt.scatter(point_diff, wins, alpha=0.7, s=60, color='steelblue', edgecolors='black')
plt.xlabel('Average Point Differential')
plt.ylabel('Wins')
plt.title('Point Differential vs Wins (NBA Season)')
plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(41, color='gray', linestyle=':', alpha=0.5, label='.500 record')
plt.legend()
plt.tight_layout()
plt.savefig('pointdiff_wins_scatter.png', dpi=100, bbox_inches='tight')
plt.show()

# Compute correlation
corr = np.corrcoef(point_diff, wins)[0, 1]
print(f"Correlation between point differential and wins: {corr:.3f}")
```

**Expected Output:**
```
Correlation between point differential and wins: 0.892
```

**What to look for:**
- **Direction**: Do the dots go up-and-right (positive relationship) or up-and-left (negative)?
- **Strength**: Are the dots tightly clustered around a line (strong) or scattered (weak)?
- **Outliers**: Any dots far from the main pattern?
- **Shape**: Is the relationship linear or curved?

Here, point differential and wins have a strong positive relationship (correlation 0.89). Teams that outscore their opponents tend to win more -- not surprising, but the visualization shows exactly how tight that relationship is.

### Choosing the Right Chart

| Question | Chart Type | Data Needed |
|----------|-----------|-------------|
| What does the distribution look like? | Histogram | One numerical variable |
| How do groups compare? | Box plot | One numerical + one categorical variable |
| Is there a relationship between two variables? | Scatter plot | Two numerical variables |
| What are the category frequencies? | Bar chart | One categorical variable |

### Exercise 4.1: Create and Interpret Visualizations

**Task:** Using the dataset below, create three visualizations and write a one-sentence interpretation of each.

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'team': [f'Team {i}' for i in range(1, 31)],
    'conference': np.random.choice(['East', 'West'], 30),
    'wins': np.random.randint(15, 65, 30),
    'ppg': np.random.normal(110, 5, 30).round(1),
    'opp_ppg': np.random.normal(110, 5, 30).round(1),
    'three_pct': np.random.normal(0.36, 0.02, 30).round(3)
})
```

1. Histogram of `wins`
2. Box plot of `ppg` grouped by `conference`
3. Scatter plot of `three_pct` vs `wins`

<details>
<summary>Hint 1: Setting up the plots</summary>

Use `plt.hist(df['wins'], ...)` for the histogram. For the box plot, split the data: `east = df[df['conference']=='East']['ppg']` and do the same for West. For the scatter plot, use `plt.scatter(df['three_pct'], df['wins'])`.
</details>

<details>
<summary>Hint 2: Interpretation tips</summary>

For the histogram, describe the shape (symmetric? skewed?). For the box plot, compare the medians and spreads. For the scatter plot, describe whether there appears to be a relationship (and how strong).
</details>

<details>
<summary>Click to see solution</summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    'team': [f'Team {i}' for i in range(1, 31)],
    'conference': np.random.choice(['East', 'West'], 30),
    'wins': np.random.randint(15, 65, 30),
    'ppg': np.random.normal(110, 5, 30).round(1),
    'opp_ppg': np.random.normal(110, 5, 30).round(1),
    'three_pct': np.random.normal(0.36, 0.02, 30).round(3)
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Histogram of wins
axes[0].hist(df['wins'], bins=10, edgecolor='black', color='steelblue', alpha=0.7)
axes[0].set_xlabel('Wins')
axes[0].set_ylabel('Number of Teams')
axes[0].set_title('Distribution of Wins')

# 2. Box plot of PPG by conference
east_ppg = df[df['conference'] == 'East']['ppg']
west_ppg = df[df['conference'] == 'West']['ppg']
axes[1].boxplot([east_ppg, west_ppg], labels=['East', 'West'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_ylabel('Points Per Game')
axes[1].set_title('PPG by Conference')

# 3. Scatter plot of three_pct vs wins
axes[2].scatter(df['three_pct'], df['wins'], alpha=0.7, color='steelblue', edgecolors='black')
axes[2].set_xlabel('Three-Point Percentage')
axes[2].set_ylabel('Wins')
axes[2].set_title('Three-Point % vs Wins')

plt.tight_layout()
plt.savefig('exercise_4_1.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Interpretations:**
1. **Histogram**: The distribution of wins is roughly uniform across the league, with no strong clustering at any particular win total.
2. **Box plot**: The East and West conferences have similar median PPG, though the specific spread and outliers may differ slightly.
3. **Scatter plot**: There is no obvious linear relationship between three-point shooting percentage and wins in this simulated data, suggesting three-point percentage alone does not predict team success.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Create a histogram, box plot, and scatter plot in matplotlib
- [ ] Read and interpret each chart type
- [ ] Choose the right chart for a given question and data type
- [ ] Identify the difference between the median line in a box plot and the mean

---

## Common Pitfalls

### Pitfall 1: Using the Mean Without Checking for Outliers

**The Problem:** You report the mean and make decisions based on it, but outliers have distorted the value.

**Why it happens:** The mean is the default "average" everyone learns first. It feels natural and complete.

**How to avoid it:** Always compute both the mean and median. If they differ substantially, investigate why. Use a histogram to see the shape of the distribution.

```python
import numpy as np

data_clean = [50, 52, 48, 51, 49, 53, 47, 50, 52, 48]
data_outlier = [50, 52, 48, 51, 49, 53, 47, 50, 52, 500]

print("Clean data:")
print(f"  Mean: {np.mean(data_clean):.1f}, Median: {np.median(data_clean):.1f}")
print("Data with outlier:")
print(f"  Mean: {np.mean(data_outlier):.1f}, Median: {np.median(data_outlier):.1f}")
```

**Expected Output:**
```
Clean data:
  Mean: 50.0, Median: 50.0
Data with outlier:
  Mean: 95.2, Median: 50.5
```

### Pitfall 2: Confusing Population and Sample Statistics

**The Problem:** Using `ddof=0` (population) when you should use `ddof=1` (sample), leading to underestimated variability.

**Why it happens:** numpy defaults to `ddof=0`, and many tutorials do not explain the difference.

**How to avoid it:** Ask yourself: "Is this the entire population, or a sample?" If it is a sample (it almost always is), use `ddof=1`.

```python
import numpy as np

data = [4, 8, 6, 5, 3]
print(f"Population std (ddof=0): {np.std(data, ddof=0):.4f}")
print(f"Sample std (ddof=1):     {np.std(data, ddof=1):.4f}")
```

**Expected Output:**
```
Population std (ddof=0): 1.7205
Sample std (ddof=1):     1.9235
```

### Pitfall 3: Choosing the Wrong Chart

**The Problem:** Using a pie chart for 15 categories, a histogram for categorical data, or a line chart when there is no time dimension.

**How to avoid it:** Use the chart selection table from Section 4. Ask yourself: "What type is my data? What question am I trying to answer?"

### Pitfall 4: Ignoring Spread When Comparing Groups

**The Problem:** Comparing two groups by their means alone and concluding they are "the same" or "different" without considering variability.

**How to avoid it:** Always report a measure of spread alongside central tendency. Two groups with the same mean but different standard deviations are not the same.

---

## Best Practices

- **Always explore before summarizing.** Look at the raw data (or at least the first few rows) before computing statistics. Check for missing values, obvious errors, and data types.
- **Report mean and median together.** If they are close, the data is roughly symmetric. If they differ, there is skew or outliers -- investigate.
- **Use `ddof=1` for sample statistics.** Unless you are certain you have the entire population, use Bessel's correction.
- **Match the chart to the question.** Do not default to bar charts for everything. Think about what you want to reveal.
- **Label your axes and title your charts.** A chart without labels is a chart nobody can interpret.
- **Be skeptical of small samples.** Summary statistics from 5 data points are unreliable. The more data you have, the more trustworthy your statistics become.

---

## Practice Project: Analyzing a Sports Season

### Project Description

You are a sports analyst tasked with understanding which statistics best predict team success in a basketball season. You will build a complete dataset, compute summary statistics, create visualizations, and write up your findings.

This project integrates everything you have learned: data types, central tendency, spread, and visualization.

### Requirements

1. Create a dataset with 30 teams and at least 8 columns (mix of categorical and numerical)
2. Classify each column by statistical data type
3. Compute mean, median, and standard deviation for at least 4 numerical columns
4. Create at least 3 visualizations (one of each type: histogram, box plot, scatter plot)
5. Compute correlations between key metrics and wins
6. Write a summary paragraph explaining which metrics best predict success

### Getting Started

**Step 1: Create your dataset**

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 30

teams = pd.DataFrame({
    'team': [f'Team {i}' for i in range(1, n + 1)],
    'conference': np.random.choice(['East', 'West'], n),
    'wins': np.random.randint(15, 65, n),
    'ppg': np.random.normal(110, 5, n).round(1),
    'opp_ppg': np.random.normal(110, 5, n).round(1),
    'rebounds': np.random.normal(44, 3, n).round(1),
    'assists': np.random.normal(25, 3, n).round(1),
    'turnovers': np.random.normal(14, 2, n).round(1),
    'ft_pct': np.random.normal(0.77, 0.03, n).round(3),
    'three_pct': np.random.normal(0.36, 0.02, n).round(3)
})
teams['losses'] = 82 - teams['wins']
teams['point_diff'] = (teams['ppg'] - teams['opp_ppg']).round(1)

print(teams.head())
```

**Step 2: Classify your columns** (write this out -- which are nominal, ordinal, discrete, continuous?)

**Step 3: Compute summary statistics**

**Step 4: Build your visualizations**

**Step 5: Compute correlations and write your findings**

### Hints and Tips

<details>
<summary>If you are not sure where to start</summary>

Run `teams.describe()` to get a quick overview of all numerical columns. Then pick the column you think is most related to wins and make a scatter plot.
</details>

<details>
<summary>If you are stuck on the box plot</summary>

Try comparing `ppg` or `rebounds` between the East and West conferences. Split the data using boolean indexing: `teams[teams['conference'] == 'East']['ppg']`.
</details>

<details>
<summary>If you are stuck on correlations</summary>

Use `teams['column_name'].corr(teams['wins'])` to compute the correlation between any column and wins. Do this for every numerical column and see which has the strongest correlation (closest to +1 or -1).
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Create dataset
np.random.seed(42)
n = 30

teams = pd.DataFrame({
    'team': [f'Team {i}' for i in range(1, n + 1)],
    'conference': np.random.choice(['East', 'West'], n),
    'wins': np.random.randint(15, 65, n),
    'ppg': np.random.normal(110, 5, n).round(1),
    'opp_ppg': np.random.normal(110, 5, n).round(1),
    'rebounds': np.random.normal(44, 3, n).round(1),
    'assists': np.random.normal(25, 3, n).round(1),
    'turnovers': np.random.normal(14, 2, n).round(1),
    'ft_pct': np.random.normal(0.77, 0.03, n).round(3),
    'three_pct': np.random.normal(0.36, 0.02, n).round(3)
})
teams['losses'] = 82 - teams['wins']
teams['point_diff'] = (teams['ppg'] - teams['opp_ppg']).round(1)

# 2. Data type classification
print("=== DATA TYPE CLASSIFICATION ===")
print("team:       Nominal (label)")
print("conference: Nominal (category)")
print("wins:       Discrete (count)")
print("ppg:        Continuous (measurement)")
print("opp_ppg:    Continuous (measurement)")
print("rebounds:   Continuous (measurement)")
print("assists:    Continuous (measurement)")
print("turnovers:  Continuous (measurement)")
print("ft_pct:     Continuous (proportion)")
print("three_pct:  Continuous (proportion)")
print("losses:     Discrete (count)")
print("point_diff: Continuous (derived measurement)")

# 3. Summary statistics
print("\n=== SUMMARY STATISTICS ===")
cols = ['wins', 'ppg', 'opp_ppg', 'point_diff', 'rebounds', 'assists']
for col in cols:
    print(f"\n{col}:")
    print(f"  Mean:    {teams[col].mean():.2f}")
    print(f"  Median:  {teams[col].median():.2f}")
    print(f"  Std Dev: {teams[col].std():.2f}")
    print(f"  IQR:     {teams[col].quantile(0.75) - teams[col].quantile(0.25):.2f}")

# 4. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram of wins
axes[0].hist(teams['wins'], bins=10, edgecolor='black', color='steelblue', alpha=0.7)
axes[0].axvline(teams['wins'].mean(), color='red', linestyle='--', label=f"Mean: {teams['wins'].mean():.0f}")
axes[0].axvline(teams['wins'].median(), color='green', linestyle='--', label=f"Median: {teams['wins'].median():.0f}")
axes[0].set_xlabel('Wins')
axes[0].set_ylabel('Number of Teams')
axes[0].set_title('Distribution of Wins')
axes[0].legend()

# Box plot of PPG by conference
east = teams[teams['conference'] == 'East']['ppg']
west = teams[teams['conference'] == 'West']['ppg']
axes[1].boxplot([east, west], labels=['East', 'West'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_ylabel('Points Per Game')
axes[1].set_title('PPG by Conference')

# Scatter plot: point differential vs wins
axes[2].scatter(teams['point_diff'], teams['wins'], alpha=0.7, s=60,
                color='steelblue', edgecolors='black')
axes[2].set_xlabel('Point Differential')
axes[2].set_ylabel('Wins')
axes[2].set_title('Point Differential vs Wins')

plt.tight_layout()
plt.savefig('practice_project.png', dpi=100, bbox_inches='tight')
plt.show()

# 5. Correlations
print("\n=== CORRELATIONS WITH WINS ===")
numeric_cols = ['ppg', 'opp_ppg', 'rebounds', 'assists', 'turnovers',
                'ft_pct', 'three_pct', 'point_diff']
for col in numeric_cols:
    r = teams[col].corr(teams['wins'])
    print(f"  {col:12s}: {r:+.3f}")

# 6. Summary
print("\n=== FINDINGS ===")
print("""
With simulated data, point differential shows the strongest correlation
with wins. This makes intuitive sense: teams that outscore opponents by
larger margins tend to accumulate more wins over a season. Individual
statistics like PPG or rebounds show weaker correlations because they
capture only one dimension of team performance. The combination of
offensive and defensive efficiency (captured by point differential)
is a more holistic predictor of success.
""")
```

**Key points in this solution:**
- The data type classification shows understanding of Section 1.
- Summary statistics use both mean and median, revealing whether distributions are symmetric.
- The three visualizations each serve a different purpose: distribution shape, group comparison, and bivariate relationship.
- Correlations quantify what the scatter plot shows visually.
- The interpretation connects statistical findings to real-world reasoning.
</details>

### Extending the Project

If you want to go further, try these extensions:

- **Sports betting extension**: Add a column for each team's over/under line (set at the mean PPG) and compute how often the actual PPG is over vs under. Does the median predict better than the mean?
- **Finance extension**: Replace the sports data with stock returns for 30 companies. Compute the Sharpe-like ratio (mean return / std dev) and rank the stocks by risk-adjusted performance.
- **Data science extension**: Add missing values to some columns and practice handling them (dropping vs imputing with mean or median). Discuss which imputation method is appropriate and why.

---

## Summary

### Key Takeaways

- **Data types come first.** Categorical (nominal, ordinal) and numerical (discrete, continuous) data types determine which statistics and charts are valid. Always classify before computing.
- **Central tendency has three faces.** The mean is powerful but sensitive to outliers. The median is robust but ignores the magnitude of extreme values. The mode works for categorical data. Report multiple measures, not just one.
- **Spread is as important as center.** Two datasets with the same mean can be completely different. Standard deviation and IQR quantify how much the data varies around the center, which directly maps to risk, consistency, and reliability.
- **Visualization reveals patterns that numbers hide.** Histograms show shape, box plots compare groups, scatter plots show relationships. The right chart for the right question is a skill worth developing.
- **Descriptive statistics is a workflow.** Explore, summarize, visualize, interpret. Each step builds on the previous one.

### Skills You've Gained

You can now:
- Classify any dataset's columns by statistical data type
- Compute mean, median, mode, range, variance, standard deviation, and IQR using Python
- Choose the right summary statistic for a given context
- Build histograms, box plots, and scatter plots with matplotlib
- Perform a complete exploratory data analysis and communicate findings
- Apply statistical thinking to sports, finance, and data science problems

### Self-Assessment

Take a moment to reflect:
- Can you explain mean vs median to someone who has never studied statistics?
- If someone quotes an "average," do you instinctively wonder about the spread?
- Could you take a new dataset and produce a summary report with statistics and charts?
- Do you understand why the casino always wins in the long run?

---

## Next Steps

### Continue Learning

Ready for more? Here are your next options:

**Build on this topic:**
- [Probability Fundamentals](/routes/probability-fundamentals/map.md) -- Learn the rules of probability, conditional probability, and Bayes' theorem. This is the natural next step after descriptive statistics.

**Explore related routes:**
- [Probability Distributions](/routes/probability-distributions/map.md) -- Understand normal distributions, binomial distributions, and the central limit theorem. You will connect the histogram shapes you learned here to formal probability models.
- [Regression and Modeling](/routes/regression-and-modeling/map.md) -- Move from describing data to predicting outcomes. The scatter plots and correlations you explored here are the foundation of regression analysis.

### Practice More

- Analyze a real dataset from a domain you care about (sports stats from Basketball Reference, stock data from Yahoo Finance, public datasets on Kaggle)
- Try computing all the statistics and building all the charts from memory, without referring to the guide
- Explore the `seaborn` library for more polished statistical visualizations
- Work through the practice project extensions to connect these skills to betting, finance, or ML contexts

---

## Appendix

### Quick Reference: Formulas

| Statistic | Formula | Python |
|-----------|---------|--------|
| Mean | sum(x) / n | `np.mean(data)` |
| Median | Middle value of sorted data | `np.median(data)` |
| Mode | Most frequent value | `Counter(data).most_common(1)[0][0]` |
| Range | max - min | `max(data) - min(data)` |
| Variance (sample) | sum((x - mean)^2) / (n - 1) | `np.var(data, ddof=1)` |
| Std Dev (sample) | sqrt(variance) | `np.std(data, ddof=1)` |
| Q1 | 25th percentile | `np.percentile(data, 25)` |
| Q3 | 75th percentile | `np.percentile(data, 75)` |
| IQR | Q3 - Q1 | `np.percentile(data, 75) - np.percentile(data, 25)` |
| Correlation | cov(x,y) / (std_x * std_y) | `np.corrcoef(x, y)[0,1]` or `df['a'].corr(df['b'])` |

### Glossary

- **Categorical data**: Data that represents groups or labels, not quantities
- **Continuous data**: Numerical data that can take any value in a range, including decimals
- **Correlation**: A measure of the linear relationship between two variables, ranging from -1 to +1
- **Descriptive statistics**: Methods for summarizing and describing the features of a dataset
- **Discrete data**: Numerical data that can only take countable values (usually whole numbers)
- **Exploratory data analysis (EDA)**: The process of examining data through summary statistics and visualization before formal modeling
- **Interquartile range (IQR)**: The range of the middle 50% of the data (Q3 - Q1), resistant to outliers
- **Mean**: The arithmetic average; sum of values divided by the count
- **Median**: The middle value in sorted data; resistant to outliers
- **Mode**: The most frequently occurring value in a dataset
- **Nominal data**: Categorical data with no natural order (e.g., team names, colors)
- **Ordinal data**: Categorical data with a meaningful order but unequal spacing (e.g., credit ratings)
- **Outlier**: A data point that is far from the rest of the data, potentially distorting summary statistics
- **Population**: The complete set of all items of interest
- **Quartile**: Values that divide data into four equal parts (Q1 = 25th percentile, Q2 = median, Q3 = 75th percentile)
- **Range**: The difference between the maximum and minimum values
- **Sample**: A subset of the population used for analysis
- **Skewness**: Asymmetry in a distribution; right-skewed means the tail extends to the right
- **Standard deviation**: The square root of variance; measures spread in the original units
- **Variance**: The average squared deviation from the mean; measures spread
- **Volatility**: In finance, the standard deviation of returns; a measure of risk
