---
title: Statistics Atlas
domain: statistics
description: Comprehensive reference of statistical methods, tests, distributions, and concepts across the statistics route family.
routes:
  - stats-fundamentals
  - probability-fundamentals
  - probability-distributions
  - statistical-inference
  - bayesian-statistics
  - stochastic-processes
  - regression-and-modeling
last_updated: 2026-04-05
---

# Statistics Atlas

A comprehensive reference for statistics. Use this atlas to look up methods, concepts, and techniques — find what each one does, when to use it, and which route teaches it.

## How to Use This Atlas

- **Looking for a specific method?** Use your editor's search (Ctrl+F / Cmd+F) or ask an AI assistant to find it.
- **Browsing by topic?** Entries are organized into categories below.
- **Want to learn in depth?** Each entry links to the route(s) where the concept is taught with full explanations, examples, and exercises.

---

## Descriptive Statistics

### Mean (Arithmetic Mean)

The sum of all values divided by the count. The most common measure of central tendency, but sensitive to outliers — a single extreme value can shift the mean dramatically.

- **Use when**: You need a single summary of a dataset's center and the data is roughly symmetric without extreme outliers
- **Watch out**: Skewed distributions (e.g., income data) make the mean misleading; prefer the median in those cases
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 2

### Median

The middle value when data is sorted. Exactly half the values fall above and half below.

- **Use when**: Your data is skewed or contains outliers and you need a robust measure of center
- **Watch out**: The median ignores the magnitude of extreme values, which is a strength for robustness but means it discards information
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 2

### Mode

The most frequently occurring value in a dataset. The only measure of central tendency that works with categorical data.

- **Use when**: You're working with categorical data, or you want to know the most common value in a distribution
- **Watch out**: A dataset can have no mode, one mode, or multiple modes (bimodal, multimodal); not always a useful summary for continuous data
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 2

### Range

The difference between the maximum and minimum values. The simplest measure of spread.

- **Use when**: You need a quick sense of how spread out the data is
- **Watch out**: Extremely sensitive to outliers — a single extreme value changes the range dramatically
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Variance

The average of the squared deviations from the mean. Quantifies how spread out values are, but in squared units.

- **Use when**: You need a mathematical foundation for further calculations (standard deviation, hypothesis tests, ANOVA)
- **Watch out**: The units are squared (e.g., dollars²), making it hard to interpret directly — use standard deviation for reporting
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Standard Deviation

The square root of the variance. Measures spread in the same units as the original data.

- **Use when**: You want to describe how much values typically deviate from the mean in interpretable units
- **Watch out**: Like the mean, standard deviation is sensitive to outliers; consider IQR for skewed data
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Interquartile Range (IQR)

The difference between the 75th percentile (Q3) and the 25th percentile (Q1). Measures the spread of the middle 50% of the data.

- **Use when**: Your data is skewed or has outliers and you need a robust measure of spread
- **Watch out**: Ignores the tails of the distribution entirely, which is its strength and limitation
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Percentiles and Quartiles

Percentiles divide data into 100 equal parts; quartiles divide it into 4. The median is the 50th percentile (Q2).

- **Use when**: You want to understand where a specific value falls relative to the rest of the distribution
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Skewness

A measure of how asymmetric a distribution is. Positive skew means a long right tail; negative skew means a long left tail.

- **Use when**: You need to assess whether the mean is a reliable measure of center, or to check distributional assumptions
- **Watch out**: Highly sensitive to outliers; always visualize the distribution rather than relying solely on the skewness number
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Kurtosis

A measure of how heavy-tailed a distribution is compared to the normal distribution. High kurtosis means more extreme outliers.

- **Use when**: You need to assess tail risk — common in finance (fat-tailed return distributions) and quality control
- **Watch out**: Often misinterpreted as "peakedness"; it's really about tail weight and outlier propensity
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Five-Number Summary

The minimum, Q1, median, Q3, and maximum. A compact description of a distribution's shape and spread.

- **Use when**: You want a quick, comprehensive snapshot of a distribution — this is what a box plot visualizes
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

### Z-Score (Standard Score)

The number of standard deviations a value is from the mean: z = (x - μ) / σ. Standardizes values for comparison across different scales.

- **Use when**: You need to compare values from different distributions, detect outliers, or standardize features for ML
- **Watch out**: Assumes the data is roughly normally distributed for meaningful interpretation of "how unusual" a value is
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 3

---

## Data Types and Visualization

### Categorical Data (Nominal and Ordinal)

Data that represents categories or groups. Nominal categories have no inherent order (colors, species); ordinal categories have a meaningful order (rankings, education levels).

- **Use when**: You're classifying observations into groups rather than measuring quantities
- **Watch out**: Don't compute means on ordinal data — the intervals between categories aren't necessarily equal
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 1

### Numerical Data (Discrete and Continuous)

Data that represents measurable quantities. Discrete data takes countable values (goals scored, items sold); continuous data can take any value in a range (temperature, time).

- **Use when**: You're measuring or counting quantities and need to choose appropriate summary statistics and charts
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 1

### Histogram

A chart that bins numerical data into intervals and shows the frequency of each bin as a bar. Reveals the shape of a distribution.

- **Use when**: You want to see the shape, center, spread, and skewness of a numerical variable's distribution
- **Watch out**: Bin width dramatically affects appearance — too few bins hide patterns, too many create noise
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 4

### Box Plot

A chart showing the five-number summary as a box (Q1 to Q3) with whiskers and individual outlier points. Compact comparison of distributions.

- **Use when**: You want to compare distributions across groups or quickly identify outliers, quartiles, and spread
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 4

### Scatter Plot

A chart plotting two numerical variables as points on an x-y plane. Reveals relationships, clusters, and outliers.

- **Use when**: You want to explore the relationship between two numerical variables before computing correlations or fitting models
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 4

### Exploratory Data Analysis (EDA)

A systematic approach to summarizing, visualizing, and understanding a dataset before formal modeling. Combines summary statistics, visualizations, and domain knowledge.

- **Use when**: You're starting work with a new dataset — always do EDA before jumping to modeling or hypothesis testing
- **Routes**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) > Section 5

---

## Probability Foundations

### Sample Space

The set of all possible outcomes of a random experiment. For a coin flip: {Heads, Tails}. For a die roll: {1, 2, 3, 4, 5, 6}.

- **Use when**: You're defining a probability problem — the sample space is the foundation for computing any probability
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 1

### Event

A subset of the sample space — one or more outcomes you're interested in. "Rolling an even number" is the event {2, 4, 6}.

- **Use when**: You want to compute the probability of something specific happening within the sample space
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 1

### Probability

A number between 0 and 1 representing how likely an event is. Can be interpreted as classical (equally likely outcomes), frequentist (long-run relative frequency), or subjective (degree of belief).

- **Use when**: You need to quantify uncertainty about whether an event will occur
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 1

### Addition Rule

P(A or B) = P(A) + P(B) - P(A and B). For mutually exclusive events, simplifies to P(A) + P(B).

- **Use when**: You want the probability that at least one of two events occurs
- **Watch out**: Forgetting to subtract the overlap P(A and B) is the most common error
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 2

### Multiplication Rule

P(A and B) = P(A) × P(B|A). For independent events, simplifies to P(A) × P(B).

- **Use when**: You want the probability that two events both occur
- **Watch out**: Only use the simplified form P(A) × P(B) when the events are truly independent
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 2

### Complement Rule

P(not A) = 1 - P(A). The probability of an event not happening.

- **Use when**: It's easier to calculate the probability of the opposite event — common in "at least one" problems
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 2

### Conditional Probability

P(A|B) = P(A and B) / P(B). The probability of A given that B has occurred.

- **Use when**: You have new information (B has happened) and want to update the probability of A
- **Watch out**: P(A|B) ≠ P(B|A) — confusing these is called the "prosecutor's fallacy"
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 3

### Independence

Two events are independent if knowing one occurred doesn't change the probability of the other: P(A|B) = P(A).

- **Use when**: You need to determine whether to use the simplified multiplication rule or need to account for dependence
- **Watch out**: Independence is an assumption, not a given — always verify it with domain knowledge or data
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 3

### Bayes' Theorem

P(A|B) = P(B|A) × P(A) / P(B). Reverses conditional probabilities — updates prior beliefs with new evidence.

- **Use when**: You know P(B|A) but need P(A|B), or you want to update a prior belief after observing data
- **Watch out**: Requires a prior P(A), which can be subjective; the result is only as good as the prior and the likelihood
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 4, [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 1

### Expected Value

The long-run average of a random variable: E[X] = Σ x·P(x). Weighs each outcome by its probability.

- **Use when**: You need to evaluate a bet, strategy, or decision by its average payoff over many repetitions
- **Watch out**: Expected value can be misleading for one-shot decisions or when variance is high
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 5, [Probability Distributions](/routes/probability-distributions/map.md) > Section 3

### Law of Total Probability

P(B) = Σ P(B|Aᵢ) × P(Aᵢ) over all partitions Aᵢ. Breaks a complex probability into simpler conditional pieces.

- **Use when**: You can't compute P(B) directly but can compute it conditional on different scenarios that cover all possibilities
- **Routes**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) > Section 4

---

## Probability Distributions

### Bernoulli Distribution

A single trial with two outcomes: success (p) or failure (1-p). The building block for more complex distributions.

- **Use when**: You're modeling a single yes/no outcome — a coin flip, a free throw, a stock going up or down
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 1

### Binomial Distribution

The number of successes in n independent Bernoulli trials. Parameters: n (trials) and p (success probability).

- **Use when**: You're counting successes in a fixed number of independent trials — wins in a season, conversions out of visitors, defective items in a batch
- **Watch out**: Requires independence between trials and a fixed probability p; if p changes over time, the binomial doesn't apply
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 1

### Poisson Distribution

Models the count of events in a fixed interval when events occur independently at a constant average rate λ.

- **Use when**: You're counting rare events in a fixed window — goals per match, server errors per hour, insurance claims per year
- **Watch out**: Assumes events are independent and the rate is constant; clustering or time-varying rates violate this
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 1

### Geometric Distribution

The number of trials until the first success. Models waiting time in a sequence of independent Bernoulli trials.

- **Use when**: You want to know how long until something happens for the first time — hands until a poker win, calls until a sale
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 1

### Uniform Distribution (Discrete and Continuous)

All outcomes are equally likely. Discrete: rolling a fair die. Continuous: a random number between 0 and 1.

- **Use when**: You have no reason to favor any outcome over another — random number generation, baseline models
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 2

### Normal Distribution (Gaussian)

The bell curve. Defined by mean μ and standard deviation σ. Arises naturally from the sum of many small independent effects (Central Limit Theorem).

- **Use when**: Your data is approximately symmetric and bell-shaped, or you're working with sample means (which are approximately normal by CLT)
- **Watch out**: Not all data is normal — always check with a histogram or normality test before assuming
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 2

### Exponential Distribution

Models the time between events in a Poisson process. The continuous analog of the geometric distribution.

- **Use when**: You're modeling waiting times between independent events — time between goals, time between trades, time to failure
- **Watch out**: Has the memoryless property — the probability of waiting another t minutes doesn't depend on how long you've already waited
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 2

### Probability Mass Function (PMF)

For discrete distributions, P(X = x) — the probability that a random variable takes a specific value.

- **Use when**: You need the exact probability of a specific outcome for a discrete variable
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 1

### Probability Density Function (PDF)

For continuous distributions, f(x) — the relative likelihood at a specific value. Probabilities come from integrating the PDF over an interval.

- **Use when**: You're working with continuous distributions and need to compute probabilities over ranges
- **Watch out**: The PDF value at a point is not a probability — only areas under the curve are probabilities
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 2

### Cumulative Distribution Function (CDF)

F(x) = P(X ≤ x) — the probability that a random variable is less than or equal to x. Works for both discrete and continuous distributions.

- **Use when**: You want "what's the probability of getting x or less?" — percentile calculations, p-value computation
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 2

### Law of Large Numbers

As sample size increases, the sample mean converges to the expected value. The mathematical guarantee behind "the house always wins."

- **Use when**: You need to justify why large samples give reliable estimates, or explain why casinos and insurance companies are profitable
- **Watch out**: Doesn't say anything about small samples — the "gambler's fallacy" is the mistaken belief that short-run results must balance out
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 4

### Central Limit Theorem

The distribution of sample means approaches a normal distribution as sample size grows, regardless of the population's shape. The foundation of statistical inference.

- **Use when**: You're constructing confidence intervals, performing hypothesis tests, or justifying why the normal distribution appears in so many statistical methods
- **Watch out**: Requires sufficiently large samples (n ≥ 30 as a rough rule); highly skewed populations need larger samples
- **Routes**: [Probability Distributions](/routes/probability-distributions/map.md) > Section 5

---

## Sampling and Estimation

### Population vs. Sample

A population is every member of the group you're studying; a sample is a subset you actually measure. Inference bridges the gap.

- **Use when**: You need to clarify what you're measuring (the sample) versus what you're trying to learn about (the population)
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 1

### Point Estimate

A single value used to estimate a population parameter — e.g., the sample mean x̄ estimates the population mean μ.

- **Use when**: You need a best guess for a population parameter from sample data
- **Watch out**: A point estimate alone says nothing about uncertainty — always pair it with a confidence interval
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 1

### Sampling Distribution

The probability distribution of a statistic (like the sample mean) across all possible samples of a given size.

- **Use when**: You need to understand how much a statistic varies from sample to sample — this is the foundation for confidence intervals and hypothesis tests
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 1

### Standard Error

The standard deviation of a sampling distribution. Measures how much a statistic (like the sample mean) varies across samples.

- **Use when**: You need to quantify the precision of an estimate — smaller standard error means more precise
- **Watch out**: Standard error is not the same as standard deviation; SE shrinks with larger samples, SD describes the data itself
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 1

### Confidence Interval

A range of values that, with a stated confidence level, is expected to contain the true population parameter. A 95% CI means: if we repeated the sampling process many times, 95% of the resulting intervals would contain the true parameter.

- **Use when**: You want to communicate both your estimate and its uncertainty — report CIs alongside point estimates
- **Watch out**: A 95% CI does NOT mean "there's a 95% probability the true value is in this interval" — the true value is fixed, the interval is random
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 2

### Margin of Error

Half the width of a confidence interval. Quantifies the maximum expected difference between the estimate and the true value at a given confidence level.

- **Use when**: You want a simple ± expression of uncertainty — "the approval rating is 52% ± 3%"
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 2

### Sample Size Determination

Calculating how many observations you need to achieve a desired margin of error or statistical power. Depends on variability, confidence level, and effect size.

- **Use when**: You're planning a study or experiment and need to know how much data to collect
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 2, Section 5
---

## Hypothesis Testing

### Null Hypothesis

The default claim that there is no effect or no difference — the status quo. Denoted H₀. Hypothesis testing asks: is the evidence strong enough to reject this?

- **Use when**: You're setting up a hypothesis test — the null is always the claim you're trying to find evidence against
- **Watch out**: Failing to reject H₀ does not mean H₀ is true — it means you didn't find sufficient evidence against it
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### Alternative Hypothesis

The claim that there is an effect or difference — what you're trying to find evidence for. Denoted H₁ or Hₐ.

- **Use when**: You're defining what you hope to show — "this drug works," "this strategy beats random," "these groups differ"
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### Test Statistic

A value computed from sample data that measures how far the observed result is from what the null hypothesis predicts. Examples: z-statistic, t-statistic, chi-square statistic.

- **Use when**: You need to convert raw data into a standardized measure that can be compared to a known distribution
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### P-Value

The probability of observing data as extreme as (or more extreme than) your sample, assuming the null hypothesis is true. Small p-values suggest the null is unlikely.

- **Use when**: You need to quantify the strength of evidence against the null hypothesis
- **Watch out**: A p-value is NOT the probability that the null is true; it does NOT measure effect size; statistical significance ≠ practical significance
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 4

### Significance Level (Alpha)

The threshold below which you reject the null hypothesis, typically 0.05. Represents your tolerance for Type I error.

- **Use when**: You're deciding your rejection criterion before running a test — always set alpha before looking at the data
- **Watch out**: The 0.05 threshold is a convention, not a law of nature; different fields and stakes call for different thresholds
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 4

### Z-Test

A hypothesis test using the standard normal distribution. Requires known population standard deviation or large sample sizes.

- **Use when**: You're testing a mean or proportion with a large sample (n ≥ 30) and known or estimable variance
- **Watch out**: Rarely used in practice because the population standard deviation is almost never known — t-tests are more common
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### T-Test (One-Sample, Two-Sample, Paired)

A hypothesis test using the t-distribution, which accounts for uncertainty in estimating the standard deviation. One-sample tests a mean against a value; two-sample compares two group means; paired compares matched observations.

- **Use when**: You're comparing means with small to moderate samples and unknown population standard deviation
- **Watch out**: Assumes approximately normal data (less critical with larger samples) and equal variances for the two-sample version (use Welch's t-test if variances differ)
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### One-Tailed vs. Two-Tailed Test

A two-tailed test checks for any difference (greater or less); a one-tailed test checks for a difference in a specific direction.

- **Use when**: Use two-tailed by default; use one-tailed only when you have a strong directional hypothesis specified before seeing the data
- **Watch out**: Choosing one-tailed after seeing the data is a form of p-hacking
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### Type I Error (False Positive)

Rejecting the null hypothesis when it's actually true. Probability = α (significance level). Concluding there's an effect when there isn't one.

- **Use when**: You need to understand the risk of a false alarm — e.g., approving an ineffective drug, deploying a model change that doesn't actually help
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 5

### Type II Error (False Negative)

Failing to reject the null hypothesis when it's actually false. Probability = β. Missing a real effect.

- **Use when**: You need to understand the risk of missing something real — e.g., failing to detect a disease, not noticing a profitable strategy
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 5

### Statistical Power

The probability of correctly rejecting a false null hypothesis: Power = 1 - β. Higher power means you're more likely to detect a real effect.

- **Use when**: You're planning a study and need to ensure it can actually detect the effect you care about
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 5

### Power Analysis

Calculating the sample size needed to achieve a desired level of statistical power, given the expected effect size and significance level.

- **Use when**: You're designing an experiment and need to determine how many subjects or observations to collect
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 5

### Effect Size

A standardized measure of the magnitude of a difference or relationship, independent of sample size. Examples: Cohen's d, Pearson's r, odds ratio.

- **Use when**: You want to know whether a statistically significant result is also practically meaningful
- **Watch out**: A tiny effect can be statistically significant with a large enough sample — always report effect sizes alongside p-values
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 4

### Multiple Testing Correction (Bonferroni)

When performing many hypothesis tests simultaneously, the probability of at least one false positive increases. Bonferroni correction divides α by the number of tests.

- **Use when**: You're running multiple tests on the same dataset — e.g., comparing many groups, testing many features
- **Watch out**: Bonferroni is conservative — it reduces power significantly; consider alternatives like Benjamini-Hochberg (FDR) for large numbers of tests
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 5

### Chi-Square Test

Tests whether observed frequencies in categorical data differ significantly from expected frequencies. Used for goodness-of-fit and tests of independence.

- **Use when**: You're analyzing categorical data — testing if a die is fair, if two categorical variables are related, if proportions differ across groups
- **Watch out**: Requires expected frequencies ≥ 5 in each cell; for small samples, use Fisher's exact test instead
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### F-Test

Tests whether two population variances are equal, or (in ANOVA) whether group means differ. Based on the ratio of two variances.

- **Use when**: You're comparing variances or testing whether multiple group means are all equal (as part of ANOVA)
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### ANOVA (Analysis of Variance)

Tests whether the means of three or more groups differ significantly. Extends the t-test to multiple groups by comparing between-group variance to within-group variance.

- **Use when**: You're comparing means across 3+ groups — e.g., does performance differ across teams, treatment groups, or product versions?
- **Watch out**: ANOVA tells you that at least one group differs, but not which one — follow up with post-hoc tests (Tukey, Bonferroni)
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 3

### A/B Testing

A controlled experiment comparing two variants (A and B) to determine which performs better. Applies hypothesis testing to product and business decisions.

- **Use when**: You want to test whether a change (new design, pricing, algorithm) improves a measurable outcome
- **Watch out**: Requires sufficient sample size, random assignment, and patience — peeking at results early inflates false positive rates
- **Routes**: [Statistical Inference](/routes/statistical-inference/map.md) > Section 6

---

## Correlation and Regression

### Correlation (Pearson r)

A measure of the linear relationship between two numerical variables, ranging from -1 (perfect negative) to +1 (perfect positive). Zero means no linear relationship.

- **Use when**: You want to quantify how strongly two variables move together — before fitting a regression model
- **Watch out**: Correlation measures only linear relationships; two variables can be strongly related in a nonlinear way with r ≈ 0
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 1

### Spurious Correlation

A statistical relationship between two variables that has no causal basis — often driven by a lurking third variable or pure coincidence.

- **Use when**: You need to critically evaluate whether a correlation reflects a real relationship or a coincidence
- **Watch out**: Large datasets make it easy to find spurious correlations; always ask "what mechanism could explain this?"
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 1

### Confounding Variable

A third variable that influences both the independent and dependent variables, creating a misleading association between them.

- **Use when**: You're trying to determine whether a relationship is causal or merely associational
- **Watch out**: Observational studies are especially vulnerable to confounding — randomized experiments are the gold standard for eliminating confounders
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 1

### Simpson's Paradox

A trend that appears in several groups of data reverses or disappears when the groups are combined. Caused by a lurking variable that changes the group proportions.

- **Use when**: You see contradictory results between subgroup analysis and aggregate analysis — always investigate why
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 1

### Simple Linear Regression (OLS)

Fits a straight line y = mx + b to data by minimizing the sum of squared residuals. The most fundamental predictive model.

- **Use when**: You want to predict a continuous outcome from a single predictor variable, or quantify the relationship between two variables
- **Watch out**: Assumes a linear relationship, constant variance of residuals, and approximately normal residuals; always check residual plots
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 2

### Slope and Intercept

The slope (m) represents the change in y for each one-unit increase in x. The intercept (b) is the predicted value of y when x = 0.

- **Use when**: You're interpreting a regression model — the slope is the key coefficient that quantifies the relationship
- **Watch out**: The intercept often has no meaningful interpretation if x = 0 is outside the range of your data
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 2

### Residuals

The differences between observed values and predicted values: residual = y - ŷ. A well-fitting model has small, randomly scattered residuals.

- **Use when**: You're diagnosing model fit — patterns in residuals indicate violations of model assumptions
- **Watch out**: Systematic patterns (curves, funnels, clusters) in residual plots indicate the model is missing something important
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 2

### Multiple Linear Regression

Extends simple linear regression to multiple predictor variables: y = b₀ + b₁x₁ + b₂x₂ + ... Each coefficient represents the effect of that predictor while holding others constant.

- **Use when**: You want to predict an outcome from several variables simultaneously, or control for confounders
- **Watch out**: Adding too many predictors risks overfitting; correlated predictors (multicollinearity) make individual coefficients unreliable
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 3

### Multicollinearity

When predictor variables are highly correlated with each other, making it difficult to isolate their individual effects. Coefficients become unstable and hard to interpret.

- **Use when**: Your regression coefficients have unexpected signs or huge standard errors — check the correlation matrix and variance inflation factors (VIF)
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 3

### Feature Selection

Choosing which predictor variables to include in a model. Methods include forward selection, backward elimination, and regularization (LASSO, Ridge).

- **Use when**: You have many candidate predictors and need to determine which ones actually contribute to prediction
- **Watch out**: Stepwise methods can overfit and are sensitive to the order of variable entry; cross-validation-based approaches are more reliable
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 3

### Logistic Regression

A regression model for binary outcomes (yes/no, win/lose, default/no-default). Uses the sigmoid function to map predictions to probabilities between 0 and 1.

- **Use when**: Your outcome variable is binary and you want to model the probability of the positive class
- **Watch out**: Despite the name, logistic regression is a classification method, not a regression method; don't use it for continuous outcomes
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 4

### Odds Ratio

The ratio of the odds of an event in one group to the odds in another. In logistic regression, eᵇ gives the odds ratio for a one-unit increase in the predictor.

- **Use when**: You're interpreting logistic regression coefficients or comparing event rates between groups
- **Watch out**: Odds ratios are not the same as relative risk (risk ratios) — they diverge substantially when the event is common
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 4

### Sigmoid Function

The S-shaped function σ(z) = 1 / (1 + e⁻ᶻ) that maps any real number to a value between 0 and 1. The core transformation in logistic regression.

- **Use when**: You need to convert a linear combination of features into a probability
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 4

---

## Model Evaluation

### R-Squared (Coefficient of Determination)

The proportion of variance in the dependent variable explained by the model. Ranges from 0 (no explanatory power) to 1 (perfect prediction).

- **Use when**: You want a quick summary of how well your regression model fits the data
- **Watch out**: R² always increases when you add more predictors, even useless ones — use adjusted R² for models with multiple predictors
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Adjusted R-Squared

A modified R² that penalizes for the number of predictors. Only increases if a new predictor improves the model more than expected by chance.

- **Use when**: You're comparing regression models with different numbers of predictors
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Mean Squared Error (MSE)

The average of the squared residuals. Penalizes large errors more heavily than small ones.

- **Use when**: You need a loss function for optimization or want to emphasize large errors in your evaluation
- **Watch out**: Units are squared, making interpretation difficult — RMSE is usually more interpretable
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Root Mean Squared Error (RMSE)

The square root of MSE. Gives error magnitude in the same units as the dependent variable.

- **Use when**: You want to communicate "on average, predictions are off by about X units"
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Mean Absolute Error (MAE)

The average of the absolute residuals. Less sensitive to outliers than MSE/RMSE.

- **Use when**: You want a robust error metric that isn't dominated by a few large outliers
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### AIC (Akaike Information Criterion)

A model comparison metric that balances goodness of fit against model complexity. Lower AIC indicates a better model. AIC = 2k - 2ln(L), where k is the number of parameters.

- **Use when**: You're comparing multiple candidate models and want to find the one that balances fit and simplicity
- **Watch out**: AIC values are only meaningful relative to other models on the same data — the absolute number has no interpretation
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### BIC (Bayesian Information Criterion)

Similar to AIC but with a stronger penalty for complexity. Tends to prefer simpler models than AIC, especially with large samples.

- **Use when**: You want model comparison with a stronger preference for parsimony, or when you believe the true model is relatively simple
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Cross-Validation

Evaluating a model by training on a subset of data and testing on the held-out remainder. K-fold CV repeats this k times with different splits.

- **Use when**: You want to estimate how well your model generalizes to unseen data — the gold standard for model evaluation
- **Watch out**: Don't use cross-validation for model training and evaluation simultaneously — this leads to data leakage
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Overfitting

When a model captures noise in the training data rather than the underlying pattern. Performs well on training data but poorly on new data.

- **Use when**: Your model has high training accuracy but low test accuracy — a classic sign of overfitting
- **Watch out**: More complex models (more features, more parameters) are more prone to overfitting; regularization and cross-validation are the main defenses
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Underfitting

When a model is too simple to capture the underlying pattern in the data. Performs poorly on both training and test data.

- **Use when**: Your model has low accuracy even on training data — it needs more features, more flexibility, or a different approach
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

### Bias-Variance Tradeoff

The fundamental tension in modeling: simple models have high bias (systematic error) but low variance (stable predictions); complex models have low bias but high variance (unstable predictions). The best model minimizes total error.

- **Use when**: You're deciding how complex to make your model — this framework guides the decision
- **Routes**: [Regression and Modeling](/routes/regression-and-modeling/map.md) > Section 5

---

## Bayesian Methods

### Prior Distribution

A probability distribution representing your beliefs about a parameter before observing data. Can be uninformative (vague), weakly informative, or strongly informative.

- **Use when**: You're building a Bayesian model and need to encode what you know (or don't know) before seeing data
- **Watch out**: The choice of prior matters most with small samples; with large samples, the data overwhelms the prior
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Likelihood Function

L(θ|data) — how well each possible parameter value θ explains the observed data. Not a probability distribution over θ, but a function of θ for fixed data.

- **Use when**: You need to connect observed data to parameter values — the likelihood is the bridge between data and inference in both Bayesian and frequentist frameworks
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Posterior Distribution

The updated distribution of a parameter after combining the prior with the likelihood via Bayes' theorem: posterior ∝ prior × likelihood.

- **Use when**: You want the full picture of what you believe about a parameter after seeing data — not just a point estimate, but a complete distribution
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Conjugate Priors

Prior distributions that, when combined with a specific likelihood, produce a posterior in the same distribution family. Enables closed-form Bayesian updates without simulation.

- **Use when**: Your data model matches a known conjugate pair (e.g., Beta-Binomial, Normal-Normal) and you want exact analytical updates
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Beta-Binomial Model

A conjugate model for binary data. Beta prior on the success probability p, binomial likelihood. Posterior is also Beta with updated parameters.

- **Use when**: You're estimating a proportion — win rates, conversion rates, free throw percentages — and want to incorporate prior information
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Normal-Normal Model

A conjugate model for estimating a mean with normally distributed data. Normal prior on the mean, normal likelihood. Posterior is also Normal.

- **Use when**: You're estimating a population mean and have prior information about its likely range
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Gamma-Poisson Model

A conjugate model for count data. Gamma prior on the rate parameter λ, Poisson likelihood. Posterior is also Gamma.

- **Use when**: You're estimating an event rate — goals per game, defects per batch, arrivals per hour
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Uninformative Prior

A prior that expresses minimal prior knowledge, such as a uniform distribution or a very wide normal. Lets the data drive the posterior.

- **Use when**: You have no strong prior beliefs and want the analysis to be driven primarily by the data
- **Watch out**: Truly uninformative priors can be improper (don't integrate to 1) or can lead to surprising behavior — weakly informative priors are often preferable
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Weakly Informative Prior

A prior that gently constrains parameters to reasonable ranges without strongly influencing the posterior. A practical default for most Bayesian analyses.

- **Use when**: You want to regularize your model and rule out absurd parameter values without injecting strong opinions
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Bayesian Updating

The process of revising your posterior as new data arrives. Yesterday's posterior becomes today's prior.

- **Use when**: You're processing data sequentially and want to update beliefs incrementally — sports ratings, real-time estimation, adaptive systems
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 2

### Markov Chain Monte Carlo (MCMC)

A family of algorithms that sample from complex posterior distributions by constructing a Markov chain whose stationary distribution is the target posterior.

- **Use when**: Your posterior has no closed-form solution (most real-world models) and you need to approximate it via sampling
- **Watch out**: Requires convergence diagnostics — trace plots, effective sample size, R-hat — to verify the chain has converged
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 4, [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 3

### Metropolis-Hastings Algorithm

A specific MCMC algorithm that proposes new parameter values and accepts or rejects them based on how much they improve the posterior probability.

- **Use when**: You're implementing MCMC from scratch or need to understand the foundational algorithm behind more sophisticated samplers
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 4

### Bayesian Decision Theory

A framework for making optimal decisions under uncertainty by minimizing expected loss (or maximizing expected utility) over the posterior distribution.

- **Use when**: You need to make a decision (bet, invest, treat, deploy) and want to account for full parameter uncertainty
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 5

### Loss Function

A function that quantifies the cost of making a wrong decision. Common choices: squared error (penalizes large errors), absolute error (robust to outliers), 0-1 loss (classification).

- **Use when**: You're formalizing what "wrong" means for your specific decision problem — the optimal decision depends on the loss function
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 5

### Expected Utility

The probability-weighted average of utility across all possible outcomes. Rational decision-making under uncertainty maximizes expected utility.

- **Use when**: You're choosing between options with uncertain outcomes and different payoffs — betting strategies, portfolio allocation, treatment decisions
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 5

### Kelly Criterion

A formula for optimal bet sizing that maximizes long-run wealth growth: bet a fraction of your bankroll proportional to your edge divided by the odds.

- **Use when**: You have a positive expected value and want to size your bets to maximize long-term growth without risking ruin
- **Watch out**: Assumes you know your true edge accurately — overestimating your edge leads to overbetting and potential ruin
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 5

### Credible Interval

The Bayesian analog of a confidence interval: a range that contains the parameter with a stated probability (e.g., 95% credible interval means "there's a 95% probability the parameter is in this range").

- **Use when**: You want to communicate uncertainty about a parameter in Bayesian terms — more intuitive than frequentist confidence intervals
- **Watch out**: Unlike confidence intervals, credible intervals depend on the prior; with uninformative priors, they often coincide numerically
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 5

### Empirical Bayes

An approach that estimates the prior from the data itself, rather than specifying it in advance. A practical compromise between fully Bayesian and frequentist methods.

- **Use when**: You have many similar estimation problems (e.g., batting averages for many players) and want to borrow strength across them
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Shrinkage Estimator

An estimator that pulls extreme values toward the overall mean. Bayesian priors naturally produce shrinkage — extreme observations are "shrunk" toward the prior mean.

- **Use when**: You have noisy estimates for many groups and want to improve accuracy by borrowing information across groups
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 3

### Hierarchical Model

A model with multiple levels: individual parameters are drawn from a group-level distribution, which itself has hyperparameters. Enables partial pooling between groups.

- **Use when**: You have data from multiple related groups (teams, schools, patients) and want each group's estimate to borrow strength from the others
- **Routes**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) > Section 4

---

## Stochastic Processes

### Random Walk

A stochastic process where each step is a random increment from the current position. The simplest model of a process that evolves unpredictably over time.

- **Use when**: You're modeling cumulative effects of random events — stock prices, a gambler's bankroll, particle diffusion
- **Watch out**: The random walk hypothesis suggests stock prices follow a random walk, making them inherently unpredictable — but the hypothesis is debated
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 1

### Gambler's Ruin

The probability that a gambler with finite wealth will eventually go broke when playing a fair or unfavorable game repeatedly. A classic random walk result.

- **Use when**: You're analyzing the long-term viability of a strategy with repeated bets — bankroll management, risk of ruin calculations
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 1

### Markov Chain

A stochastic process where the probability of each future state depends only on the current state, not the history. The "memoryless" property.

- **Use when**: You're modeling a system that transitions between discrete states — credit ratings, game states, weather patterns, PageRank
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 2

### Transition Matrix

A matrix where entry (i, j) gives the probability of moving from state i to state j. Fully describes a Markov chain's dynamics.

- **Use when**: You're specifying or analyzing a Markov chain — multiply the matrix by itself to get multi-step transition probabilities
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 2

### Stationary Distribution

The long-run equilibrium distribution of a Markov chain. After many steps, the chain settles into this distribution regardless of where it started.

- **Use when**: You want to know the long-run behavior of a system — what fraction of time does it spend in each state?
- **Watch out**: Not all Markov chains have a unique stationary distribution — the chain must be irreducible and aperiodic
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 2

### Absorbing State

A state in a Markov chain that, once entered, can never be left. Probability of entering an absorbing state = 1 for absorbing chains.

- **Use when**: You're modeling processes with permanent outcomes — gambler's ruin (going broke), disease progression to terminal states
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 2

### Monte Carlo Simulation

Using random sampling to estimate quantities that are difficult to compute analytically. Generate many random scenarios and aggregate the results.

- **Use when**: The problem is too complex for an analytical solution — pricing options, estimating integrals, simulating complex systems
- **Watch out**: Results are approximate and require enough samples for accuracy; always report confidence intervals on Monte Carlo estimates
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 3

### Importance Sampling

A variance-reduction technique that focuses Monte Carlo samples on the most informative regions of the distribution, then reweights to correct for the biased sampling.

- **Use when**: Standard Monte Carlo is inefficient because the important events are rare — tail risk estimation, rare event simulation
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 3

### Poisson Process

A model for events occurring randomly and independently in continuous time at a constant average rate λ. The number of events in any interval follows a Poisson distribution.

- **Use when**: You're modeling arrival times or event counts — goals in a match, trades arriving at an exchange, insurance claims
- **Watch out**: Assumes events are independent and the rate is constant; real-world events often cluster or have time-varying rates
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 4

### Memorylessness

The property that the probability of future events doesn't depend on how long you've already waited. Unique to exponential (continuous) and geometric (discrete) distributions.

- **Use when**: You're modeling waiting times where the process "resets" at every moment — the remaining wait has the same distribution regardless of elapsed time
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 4

### Geometric Brownian Motion

A continuous-time stochastic process where the logarithm of the variable follows a random walk with drift. The standard model for stock price dynamics.

- **Use when**: You're modeling asset prices, option pricing (Black-Scholes), or any quantity that grows multiplicatively with random shocks
- **Watch out**: Assumes constant volatility and log-normal returns — real markets exhibit fat tails and volatility clustering
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 1

---

## Time Series

### Stationarity

A time series is stationary if its statistical properties (mean, variance, autocorrelation) don't change over time. Most time series methods assume stationarity.

- **Use when**: You're checking whether a time series method is appropriate — non-stationary data needs to be differenced or detrended first
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 5

### Autocorrelation

The correlation of a time series with a lagged version of itself. Measures how much past values predict future values.

- **Use when**: You're exploring temporal patterns in data — seasonality, trends, or momentum effects
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 5

### Autoregressive Model (AR)

A time series model where the current value depends linearly on its own previous values: X_t = c + φ₁X_{t-1} + ... + φₚX_{t-p} + ε.

- **Use when**: Past values of the series are predictive of future values — stock momentum, temperature patterns, team performance streaks
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 5

### Moving Average Model (MA)

A time series model where the current value depends linearly on past forecast errors: X_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}.

- **Use when**: The series shows short-term dependencies driven by random shocks that persist for a few periods
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 5

### ARMA Model

Combines autoregressive and moving average components: AR(p) + MA(q). Captures both the persistence of past values and the persistence of past shocks.

- **Use when**: Neither a pure AR nor a pure MA model adequately captures the time series dynamics
- **Watch out**: Requires stationary data; for non-stationary data, use ARIMA (which adds differencing)
- **Routes**: [Stochastic Processes](/routes/stochastic-processes/map.md) > Section 5
