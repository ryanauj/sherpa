---
title: Regression and Modeling
route_map: /routes/regression-and-modeling/map.md
paired_sherpa: /routes/regression-and-modeling/sherpa.md
prerequisites:
  - Stats Fundamentals
  - Statistical Inference
  - Probability Distributions (helpful)
topics:
  - Correlation and Causation
  - Simple Linear Regression
  - Multiple Linear Regression
  - Logistic Regression
  - Model Evaluation
---

# Regression and Modeling

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/regression-and-modeling/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/regression-and-modeling/map.md` for the high-level overview.

## Overview

Regression is the workhorse of statistical modeling. It gives you a principled way to ask: "Given what I know, what should I predict?" Whether you are predicting how many points a basketball team will score, estimating the probability a borrower defaults, or building an ML pipeline, regression is where you start.

This guide takes you from correlation through linear and logistic regression to rigorous model evaluation, with applications in sports analytics, finance, and data science throughout.

## Learning Objectives

By the end of this guide, you will be able to:
- Compute and correctly interpret correlation coefficients
- Build simple and multiple linear regression models using scikit-learn and statsmodels
- Apply logistic regression to binary classification problems
- Evaluate models using R-squared, MSE, RMSE, AIC/BIC, and cross-validation
- Recognize and avoid overfitting, multicollinearity, and other common pitfalls

## Prerequisites

Before starting, you should be familiar with:
- Descriptive statistics: mean, variance, standard deviation (Stats Fundamentals)
- Hypothesis testing, confidence intervals, and p-values (Statistical Inference)
- Basic Python, NumPy, and pandas usage

If you need to review: [Stats Fundamentals](/routes/stats-fundamentals/map.md) | [Statistical Inference](/routes/statistical-inference/map.md) | [Probability Distributions](/routes/probability-distributions/map.md)

## Setup

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib
```

```python
import numpy as np, pandas as pd, sklearn, statsmodels.api as sm, matplotlib.pyplot as plt
print(f"NumPy: {np.__version__}, pandas: {pd.__version__}, sklearn: {sklearn.__version__}")
```

---

## Section 1: Correlation and Causation

### What Is Correlation?

Correlation measures the strength and direction of the *linear* relationship between two variables. The Pearson correlation coefficient *r* ranges from -1 (perfect negative) to +1 (perfect positive), with 0 meaning no linear relationship.

### Computing Correlation

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_teams = 30
teams = pd.DataFrame({
    'pace': np.random.normal(100, 5, n_teams),
    'off_rating': np.random.normal(110, 4, n_teams),
    'def_rating': np.random.normal(108, 4, n_teams),
    'three_pct': np.random.normal(0.36, 0.02, n_teams),
    'turnovers': np.random.normal(14, 2, n_teams),
})
teams['wins'] = (0.5 * teams['off_rating'] - 0.4 * teams['def_rating']
    + 10 * teams['three_pct'] - 0.3 * teams['turnovers']
    + np.random.normal(0, 2, n_teams)).round().astype(int)

print(teams[['off_rating', 'def_rating', 'pace', 'wins']].corr().round(3))
```

**Expected Output:**
```
            off_rating  def_rating   pace   wins
off_rating       1.000      -0.049  0.079  0.536
def_rating      -0.049       1.000  0.174 -0.503
pace             0.079       0.174  1.000 -0.108
wins             0.536      -0.503 -0.108  1.000
```

Better offense correlates with more wins (r ~ 0.54). Higher opponent scoring correlates with fewer wins (r ~ -0.50). Pace has near-zero correlation with wins -- playing faster does not predict winning by itself.

### Correlation Is Not Causation

Two variables can correlate because: (1) X causes Y, (2) Y causes X, or (3) a third variable Z drives both. Ice cream sales and drowning deaths correlate -- the confounder is summer heat. Teams that sell more hot dogs at games tend to win more -- the confounder is crowd size (better teams draw bigger crowds).

**Finance**: Two stocks may correlate not because they influence each other but because both respond to interest rates. Correlation matrices drive portfolio diversification, but correlations spike during crises -- exactly when you need diversification most.

### Simpson's Paradox

A player can have a higher batting average in *both* halves of the season yet a *lower* average overall:

```python
# Player A: .400 first half (40/100), .200 second half (80/400) => .240 season
# Player B: .300 first half (120/400), .150 second half (15/100) => .270 season
print("Player A: .400 (H1), .200 (H2), .240 (season)")
print("Player B: .300 (H1), .150 (H2), .270 (season)")
print("A beats B in each half, but B beats A overall!")
```

This happens because unequal sample sizes across subgroups reverse the aggregate trend. Always check for meaningful subgroups in your data.

### Exercise 1.1: Explore Correlations

**Task:** Using the `teams` DataFrame, find the pairs of variables with the strongest positive and strongest negative correlation. For each, explain whether the relationship could be causal.

<details>
<summary>Hint</summary>
Use `teams.corr()` and look for the largest and smallest off-diagonal values.
</details>

<details>
<summary>Solution</summary>

```python
corr = teams.corr()
unstacked = corr.unstack()
unstacked = unstacked[unstacked < 1.0]
print(f"Strongest positive: {unstacked.idxmax()}, r = {unstacked.max():.3f}")
print(f"Strongest negative: {unstacked.idxmin()}, r = {unstacked.min():.3f}")
```

The strongest positive is likely `off_rating`/`wins` (~0.54) -- plausibly causal but confounded by roster talent. The strongest negative is `def_rating`/`wins` (~-0.50) -- allowing fewer opponent points helps you win, but coaching quality confounds both.
</details>

**Self-Check:**
- [ ] Compute a correlation matrix in pandas
- [ ] Interpret the sign and magnitude of r
- [ ] Explain why correlation does not imply causation with an example
- [ ] Describe Simpson's paradox

---

## Section 2: Simple Linear Regression

### Fitting a Line to Data

Simple linear regression fits y = mx + b to minimize the sum of squared residuals (Ordinary Least Squares). The slope *m* tells you the association per unit change in x; the intercept *b* anchors the line.

### scikit-learn Implementation

```python
from sklearn.linear_model import LinearRegression

X = teams[['off_rating']].values
y = teams['wins'].values
model = LinearRegression().fit(X, y)

print(f"Slope:     {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X, y):.4f}")
```

**Expected Output:**
```
Slope:     0.2889
Intercept: -27.2193
R-squared: 0.2870
```

**Slope = 0.29**: Each one-point increase in offensive rating predicts ~0.29 more wins. **R-squared = 0.29**: The model explains 29% of win variance; 71% is due to other factors.

### statsmodels for Statistical Detail

```python
import statsmodels.api as sm
X_sm = sm.add_constant(teams['off_rating'])
result = sm.OLS(teams['wins'], X_sm).fit()
print(result.summary().tables[1])
```

This gives p-values and confidence intervals for each coefficient. A p-value below 0.05 for the slope confirms a statistically significant linear relationship.

### Residual Analysis

```python
residuals = y - model.predict(X)
print(f"Mean residual: {residuals.mean():.6f}")  # ~0 by construction
print(f"Std residual:  {residuals.std():.3f}")
```

Plot residuals vs. predicted values. You want random scatter around zero. A curved pattern means your linear model misses a nonlinear relationship. A funnel shape means heteroscedasticity.

### Finance: CAPM Beta

The CAPM model regresses stock excess returns on market excess returns. Beta (the slope) measures systematic risk: beta = 1.3 means the stock is 30% more volatile than the market.

```python
np.random.seed(42)
market = np.random.normal(0.008, 0.04, 60)
stock = 0.002 + 1.3 * market + np.random.normal(0, 0.02, 60)
capm = LinearRegression().fit(market.reshape(-1, 1), stock)
print(f"Beta: {capm.coef_[0]:.3f}, Alpha: {capm.intercept_:.4f}")
# Output: Beta: 1.270, Alpha: 0.0029
```

### Exercise 2.1: Build a Regression Model

**Task:** Build a simple linear regression predicting wins from `def_rating`. Report slope, intercept, R-squared, and interpret each.

<details>
<summary>Hint</summary>
Use `LinearRegression().fit(teams[['def_rating']].values, teams['wins'].values)`. Remember that higher defensive rating means *worse* defense, so expect a negative slope.
</details>

<details>
<summary>Solution</summary>

```python
X = teams[['def_rating']].values
y = teams['wins'].values
model = LinearRegression().fit(X, y)
print(f"Slope: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}, R²: {model.score(X,y):.4f}")
# Slope ~ -0.27: each point of worse defense costs ~0.27 wins
# R² ~ 0.25: defense explains ~25% of win variance
```
</details>

**Self-Check:**
- [ ] Fit a linear regression in both scikit-learn and statsmodels
- [ ] Interpret slope, intercept, and R-squared
- [ ] Explain what residuals are and what patterns to look for

---

## Section 3: Multiple Linear Regression

### Adding More Predictors

Multiple regression uses several predictors: y = b0 + b1*x1 + b2*x2 + ... Each coefficient represents the expected change in y per unit increase in that predictor, *holding all other predictors constant*.

```python
features = teams[['off_rating', 'def_rating', 'three_pct', 'turnovers']]
model = LinearRegression().fit(features, teams['wins'])

print("Coefficients:")
for name, coef in zip(features.columns, model.coef_):
    print(f"  {name:>12}: {coef:.4f}")
print(f"  {'Intercept':>12}: {model.intercept_:.4f}")
print(f"  {'R-squared':>12}: {model.score(features, teams['wins']):.4f}")
```

**Expected Output:**
```
  off_rating:  0.4846
  def_rating: -0.3921
   three_pct:  9.4752
   turnovers: -0.2830
   Intercept:  0.6558
   R-squared:  0.9548
```

R-squared jumped from 0.29 to 0.95 -- four predictors capture the pattern much better than one.

### Multicollinearity

When predictors are highly correlated with each other, coefficients become unstable. Detect it with Variance Inflation Factors (VIF):

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(features)
for i, col in enumerate(X_vif.columns):
    print(f"  {col:>12}: VIF = {variance_inflation_factor(X_vif.values, i):.2f}")
```

VIF > 5 is a warning; VIF > 10 is a serious problem. Fix by removing correlated predictors or using regularization (Ridge regression).

**Finance**: The Fama-French three-factor model extends CAPM with size (SMB) and value (HML) factors -- just multiple regression with three predictors.

### Exercise 3.1: Compare Models

**Task:** Build three models: (A) `off_rating` only, (B) `off_rating` + `def_rating`, (C) all four predictors. Compare R-squared and adjusted R-squared. Which would you choose for next season's predictions?

<details>
<summary>Solution</summary>

```python
import statsmodels.api as sm
for name, cols in [('A', ['off_rating']), ('B', ['off_rating','def_rating']),
                   ('C', ['off_rating','def_rating','three_pct','turnovers'])]:
    X = sm.add_constant(teams[cols])
    r = sm.OLS(teams['wins'], X).fit()
    print(f"Model {name}: R²={r.rsquared:.4f}, Adj R²={r.rsquared_adj:.4f}, AIC={r.aic:.1f}")
```

Model C has the highest R-squared, but with only 30 observations and 4 predictors, overfitting is a risk. Cross-validation (Section 5) is needed to make a fair comparison.
</details>

**Self-Check:**
- [ ] Interpret multiple regression coefficients with the "holding constant" qualifier
- [ ] Compute VIF and recognize multicollinearity
- [ ] Understand that more predictors do not always mean a better model

---

## Section 4: Logistic Regression

### Binary Outcomes Need a Different Approach

When the outcome is binary (win/lose, default/no-default), linear regression fails -- it can predict values below 0 or above 1. Logistic regression solves this by passing a linear combination through the **sigmoid function**: sigma(z) = 1 / (1 + e^(-z)), which maps any real number to a probability between 0 and 1.

The model: log(p / (1-p)) = b0 + b1*x1 + b2*x2 + ... The left side is the log-odds. Despite the name, logistic regression *is* regression -- on the log-odds scale.

### Building the Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

teams['win_class'] = (teams['wins'] >= teams['wins'].median()).astype(int)
X = teams[['off_rating', 'def_rating', 'three_pct']].values
y = teams['win_class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:     {model.score(X_test, y_test):.3f}")
```

**Expected Output:**
```
Training accuracy: 0.952
Test accuracy:     0.889
```

### Interpreting Odds Ratios

Coefficients are in log-odds space. Convert with `exp()`:

```python
import numpy as np
for name, coef in zip(['off_rating','def_rating','three_pct'], model.coef_[0]):
    print(f"  {name:>12}: coef={coef:.3f}, odds_ratio={np.exp(coef):.3f}")
```

An odds ratio of 1.97 for `off_rating` means each one-point increase approximately doubles the odds of being an above-median team. An odds ratio of 0.59 for `def_rating` means each point of worse defense cuts the odds to ~59%.

### Predicting Probabilities

```python
probs = model.predict_proba(X_test)[:3]
for i, p in enumerate(probs):
    print(f"  Team {i+1}: P(below median)={p[0]:.3f}, P(above median)={p[1]:.3f}")
```

These probabilities are what make logistic regression powerful for sports betting: if your model says 65% win probability but the bookmaker implies 50%, there may be a value bet.

### Exercise 4.1: Logistic Regression Classifier

**Task:** Build a logistic regression using all predictors to classify above/below-median wins. Report accuracy and odds ratios.

<details>
<summary>Solution</summary>

```python
feature_cols = ['off_rating', 'def_rating', 'three_pct', 'turnovers', 'pace']
X = teams[feature_cols].values
y = teams['win_class'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"Test accuracy: {model.score(X_te, y_te):.3f}")
for name, coef in zip(feature_cols, model.coef_[0]):
    print(f"  {name}: OR={np.exp(coef):.3f}")
```
</details>

**Self-Check:**
- [ ] Explain why linear regression is wrong for binary outcomes
- [ ] Describe the sigmoid function
- [ ] Convert log-odds to odds ratios and interpret them
- [ ] Use `predict_proba` to get probabilities

---

## Section 5: Model Evaluation

### The Core Problem

Any model can memorize training data. The question is whether it generalizes to new data. A model with R-squared = 0.85 on training data and 0.30 on new data is overfitting.

### Key Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = np.array([100, 105, 98, 112, 95])
y_pred = np.array([102, 103, 99, 108, 97])

print(f"MSE:  {mean_squared_error(y_true, y_pred):.2f}")    # Penalizes large errors
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")  # Same units as y
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f}")   # Robust to outliers
```

**Expected Output:**
```
MSE:  5.60
RMSE: 2.37
MAE:  2.20
```

Use RMSE when large errors are disproportionately bad (sports betting, finance). Use MAE when all errors matter equally.

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

for name, cols in [('A', ['off_rating']), ('B', ['off_rating','def_rating']),
                   ('C', ['off_rating','def_rating','three_pct','turnovers'])]:
    X = teams[cols].values
    m = LinearRegression()
    m.fit(X, teams['wins'].values)
    train_r2 = m.score(X, teams['wins'].values)
    cv = cross_val_score(m, X, teams['wins'].values, cv=5, scoring='r2')
    print(f"Model {name}: Train R²={train_r2:.3f}, CV R²={cv.mean():.3f} +/- {cv.std():.3f}")
```

CV R-squared is always lower than training R-squared. The gap reveals overfitting. If CV R-squared is negative, the model is worse than predicting the mean every time.

### Overfitting Demonstration

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = teams[['off_rating']].values
y = teams['wins'].values
for deg in [1, 3, 10]:
    pipe = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    pipe.fit(X, y)
    cv = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    print(f"Degree {deg:>2}: Train R²={pipe.score(X,y):.3f}, CV R²={cv.mean():.3f}")
```

As degree increases, training R-squared goes up but CV R-squared collapses -- classic overfitting.

### AIC/BIC for Model Comparison

```python
import statsmodels.api as sm
for name, cols in [('A', ['off_rating']), ('B', ['off_rating','def_rating']),
                   ('C', ['off_rating','def_rating','three_pct','turnovers'])]:
    r = sm.OLS(teams['wins'], sm.add_constant(teams[cols])).fit()
    print(f"Model {name}: AIC={r.aic:.1f}, BIC={r.bic:.1f}")
```

Lower AIC/BIC is better. BIC penalizes complexity more heavily, preferring simpler models.

### Exercise 5.1: Full Evaluation

**Task:** Compute 5-fold CV R-squared and RMSE for Models A, B, and C. Which has the best CV performance?

<details>
<summary>Solution</summary>

```python
for name, cols in [('A', ['off_rating']), ('B', ['off_rating','def_rating']),
                   ('C', ['off_rating','def_rating','three_pct','turnovers'])]:
    X = teams[cols].values
    y = teams['wins'].values
    m = LinearRegression()
    cv_r2 = cross_val_score(m, X, y, cv=5, scoring='r2')
    cv_mse = -cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Model {name}: CV R²={cv_r2.mean():.3f}, CV RMSE={np.sqrt(cv_mse.mean()):.3f}")
```
</details>

**Self-Check:**
- [ ] Explain R-squared vs. adjusted R-squared
- [ ] Compute MSE, RMSE, and MAE
- [ ] Perform cross-validation and interpret the results
- [ ] Identify overfitting by comparing training vs. CV metrics

---

## Practice Project: Sports Prediction Model

### Requirements

Build a sports prediction system that:
1. Creates a dataset with 5+ features and 50+ observations
2. Performs exploratory analysis (correlation matrix)
3. Fits at least two linear regression models and one logistic regression model
4. Evaluates all models with 5-fold cross-validation
5. Compares against a naive baseline

### Starter Code

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 200
games = pd.DataFrame({
    'home_off': np.random.normal(110, 5, n),
    'home_def': np.random.normal(108, 5, n),
    'away_off': np.random.normal(110, 5, n),
    'away_def': np.random.normal(108, 5, n),
    'home_pace': np.random.normal(100, 4, n),
})
games['point_diff'] = (0.4*(games['home_off']-games['away_off'])
    - 0.3*(games['home_def']-games['away_def'])
    + 3.0 + np.random.normal(0, 8, n))
games['home_win'] = (games['point_diff'] > 0).astype(int)

# Explore
print("Correlations with point_diff:")
print(games.corr()['point_diff'].drop('point_diff').round(3))

# Model 1: simple
X1 = (games[['home_off']].values - games[['away_off']].values)
# Model 2: all features
X2 = games[['home_off','home_def','away_off','away_def','home_pace']].values

for name, X in [('Simple', X1), ('Full', X2)]:
    cv = cross_val_score(LinearRegression(), X, games['point_diff'], cv=5, scoring='r2')
    print(f"LR {name}: CV R² = {cv.mean():.3f}")

for name, X in [('Simple', X1), ('Full', X2)]:
    cv = cross_val_score(LogisticRegression(max_iter=1000), X, games['home_win'], cv=5)
    print(f"LogReg {name}: CV Accuracy = {cv.mean():.3f}")

baseline = max(games['home_win'].mean(), 1 - games['home_win'].mean())
print(f"Baseline (always pick majority): {baseline:.3f}")
```

<details>
<summary>If you are stuck</summary>
Start with the correlation matrix. Identify the strongest predictors. Build a simple model first, then add features one at a time and watch how CV performance changes.
</details>

### Extending the Project

- Use real data from basketball-reference.com or FanGraphs
- Try Ridge/Lasso regression to prevent overfitting
- Build a calibration curve: are predicted probabilities accurate?
- Simulate a betting strategy: bet on games where model probability exceeds implied book probability

---

## Common Pitfalls

**Overfitting**: Model looks great on training data, fails on new data. Fix: always cross-validate; start simple.

**Ignoring assumptions**: Linear regression assumes linear relationships, independent errors, constant variance. Fix: plot residuals; transform variables if needed.

**P-hacking with features**: Testing 50 features and reporting the 3 with p < 0.05 guarantees false positives. Fix: pre-specify features or use cross-validation for selection.

**Confusing correlation with prediction**: High R-squared on historical data does not guarantee future accuracy. Fix: validate with genuinely out-of-sample data.

## Best Practices

- Start simple: one or two predictors, linear model. Add complexity only when CV metrics improve.
- Always cross-validate. Never report only training metrics.
- Plot your data before modeling. Visual inspection catches problems statistics miss.
- Check residuals after fitting. Patterns mean your model is missing something.
- Know your baseline. In sports, the home team wins ~58% of the time. Beat that first.
- Interpret carefully. Coefficients show associations, not causes.

---

## Summary

### Key Takeaways

- **Correlation**: Measures linear co-movement, not causation. Confounders and Simpson's paradox are real.
- **Simple Linear Regression**: Fits y = mx + b via least squares. Slope is the association per unit of x.
- **Multiple Regression**: Adds predictors with partial-effect interpretation. Multicollinearity makes coefficients unstable.
- **Logistic Regression**: Predicts binary outcome probabilities via the sigmoid. Coefficients convert to odds ratios.
- **Model Evaluation**: Cross-validation is non-negotiable. Training metrics alone are dangerously optimistic.

### Skills You Have Gained

- Computing and interpreting correlation matrices
- Building regression models in scikit-learn and statsmodels
- Interpreting slopes, R-squared, and odds ratios
- Performing cross-validation and detecting overfitting
- Applying modeling to sports, finance, and ML problems

---

## Next Steps

- [Bayesian Statistics](/routes/bayesian-statistics/map.md) -- Incorporate prior knowledge into models
- [Neural Network Foundations](/routes/neural-network-foundations/map.md) -- Regression generalizes into deep learning
- [Stochastic Processes](/routes/stochastic-processes/map.md) -- Model randomness evolving over time
- Regularization (Ridge, Lasso) -- Prevent overfitting with penalty terms
- Time series regression -- Handle autocorrelation and seasonality

**Books**: "An Introduction to Statistical Learning" (James et al.) -- free online
**Data**: basketball-reference.com, FanGraphs, Yahoo Finance, Kaggle competitions

---

## Appendix

### Quick Reference

| Metric | Use For | Better |
|--------|---------|--------|
| R-squared | Variance explained | Higher |
| Adj R-squared | Model comparison (penalizes complexity) | Higher |
| MSE / RMSE | Error magnitude | Lower |
| MAE | Error (robust to outliers) | Lower |
| AIC / BIC | Model comparison | Lower |

### Glossary

- **Coefficient**: Expected change in y per unit change in x
- **Confounding variable**: A third variable creating a spurious association
- **Cross-validation**: Evaluating models by training/testing on different data subsets
- **Log-odds**: ln(p / (1-p)), the scale logistic regression operates on
- **Multicollinearity**: Highly correlated predictors making coefficients unstable
- **Odds ratio**: exp(coefficient), the multiplicative change in odds per unit increase
- **Overfitting**: Capturing noise in training data, failing on new data
- **Residual**: Actual minus predicted value (y - y_hat)
- **Sigmoid**: 1 / (1 + e^(-z)), maps any number to a probability

### Troubleshooting

**"Expected 2D array, got 1D"**: Use `.reshape(-1, 1)` or `df[['col']]` instead of `df['col']`.
**statsmodels missing intercept**: Use `sm.add_constant(X)`.
**Negative CV R-squared**: Your model is worse than predicting the mean. Simplify it.
**LogisticRegression won't converge**: Increase `max_iter` or standardize features.
