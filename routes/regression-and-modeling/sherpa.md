---
title: Regression and Modeling
route_map: /routes/regression-and-modeling/map.md
paired_guide: /routes/regression-and-modeling/guide.md
topics:
  - Correlation and Causation
  - Simple Linear Regression
  - Multiple Linear Regression
  - Logistic Regression
  - Model Evaluation
---

# Regression and Modeling - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach regression analysis and statistical modeling effectively through structured interaction. It covers correlation, simple and multiple linear regression, logistic regression, and model evaluation -- all grounded in applications from sports analytics, finance, and machine learning.

**Route Map**: See `/routes/regression-and-modeling/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/regression-and-modeling/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Distinguish correlation from causation and identify spurious relationships
- Build, interpret, and diagnose simple linear regression models
- Extend to multiple linear regression with feature selection and multicollinearity awareness
- Apply logistic regression for binary classification and interpret odds ratios
- Evaluate models using R-squared, MSE, RMSE, AIC/BIC, and cross-validation
- Recognize overfitting, underfitting, and common modeling traps

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/regression-and-modeling/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they have covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Comfort with means, variance, standard deviation, and basic data visualization (Stats Fundamentals)
- Understanding of hypothesis testing, confidence intervals, and p-values (Statistical Inference)
- Familiarity with Python, NumPy, and pandas (at least basic usage)

**If prerequisites are missing**: Suggest they review [Stats Fundamentals](/routes/stats-fundamentals/map.md) or [Statistical Inference](/routes/statistical-inference/map.md) first. If they are close but rusty, offer a 2-minute refresher on the specific concept before proceeding.

### Learner Preferences Configuration

Check for a `.sherpa-config.yml` in the repository root. Relevant options:
- `tone`: objective, encouraging, humorous (default: objective)
- `explanation_depth`: concise, balanced, detailed
- `primary_domain`: sports, finance, ml, general (default: general -- use all three)
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)

If no configuration file exists, use defaults.

### Assessment Strategies

Use a mix of:
- **Multiple choice**: Quick concept checks (e.g., "r = -0.85 means: A) Causation B) Strong negative linear relationship C) Guaranteed prediction D) No relationship")
- **Explanation**: Deeper understanding (e.g., "Why doesn't high R-squared guarantee good predictions?")
- **Calculation/Code**: Hands-on (e.g., "Given this output, predict y when x = 10")
Use multiple choice for new concepts, explanation for foundational ideas, calculation after mechanics.

---

## Teaching Flow

### Introduction

**What to Cover:**
- Regression is about building mathematical models that capture relationships in data
- It is foundational to sports analytics (predicting scores, win probabilities), finance (pricing risk, forecasting returns), and machine learning (supervised learning)
- By the end, they will be able to build a model, evaluate whether it is any good, and know when not to trust it

**Opening Questions to Assess Level:**
1. "Have you worked with regression before, even informally -- fitting a trendline in Excel, for instance?"
2. "When you hear 'correlation does not imply causation,' can you give me an example of why that is true?"
3. "What kind of predictions are you most interested in -- sports outcomes, financial returns, general data science, or something else?"

**Adapt based on responses:**
- If experienced: Skip basics of correlation, move quickly to regression mechanics, focus on evaluation and pitfalls
- If complete beginner: Spend extra time on correlation intuition and the meaning of "fitting a line"
- If specific domain interest: Weight examples toward their domain (sports, finance, or ML) throughout

---

### Section 1: Correlation and Causation

**Core Concept to Teach:**
Correlation measures the strength and direction of a linear relationship between two variables. It is the starting point for regression, but it comes with traps that catch even experienced analysts.

**How to Explain:**
1. Start with intuition: "Correlation tells you whether two things tend to move together. When one goes up, does the other go up (positive), go down (negative), or do its own thing (near zero)?"
2. Introduce the Pearson correlation coefficient r: ranges from -1 to +1
3. Show that r only captures *linear* relationships -- a perfect parabola has r near 0
4. Drive home: correlation does not tell you *why* two things move together

**Example to Present:**
```python
import numpy as np

# Simulated data: team pace (possessions per game) and points scored
np.random.seed(42)
pace = np.random.normal(100, 5, 30)       # possessions per game
points = 0.9 * pace + np.random.normal(0, 3, 30)  # points ~ pace + noise

r = np.corrcoef(pace, points)[0, 1]
print(f"Correlation between pace and points: r = {r:.3f}")
# Output: Correlation between pace and points: r = 0.832
```

**Walk Through:**
- r = 0.83 means a strong positive linear relationship
- Teams that play faster tend to score more -- that makes basketball sense
- But does playing faster *cause* more points? Or do better offensive teams both play faster and score more?

**Spurious Correlations -- The Big Warning:**
Present a fun example: "Did you know that the number of Nicolas Cage films released in a year correlates with the number of swimming pool drownings? Should we ban Nicolas Cage movies to save lives?"

- Spurious correlations arise from coincidence, confounders, or shared trends
- Sports example: teams that sell more hot dogs at games tend to win more. Confounder? Better teams draw bigger crowds, and bigger crowds buy more hot dogs.
- Finance example: two stocks may be correlated not because they influence each other but because they both respond to interest rates.

**Simpson's Paradox:**
"Here is a genuinely mind-bending result. A baseball player can have a higher batting average than another player in *both* halves of the season, yet a *lower* average for the full season."

Explain the mechanism: unequal sample sizes across subgroups reverse the aggregate trend. This is not a curiosity -- it shows up in medical studies, admissions data, and sports stats.

Ask: "Why does this matter for modeling?" Answer: aggregation can mislead. Always check whether your data has subgroups that behave differently.

**Domain Applications:**
- **Sports**: Do more home runs predict more wins? (Some correlation, but pitching and defense are confounders.)
- **Finance**: Correlation matrices drive portfolio diversification. When correlations spike during a crisis, diversification fails exactly when you need it.
- **ML**: Highly correlated features cause problems in regression (multicollinearity). Feature correlation analysis is a standard preprocessing step.

**Common Misconceptions:**
- "High correlation means one causes the other" -- Clarify: correlation measures co-movement, not causation. You need experiments or careful causal reasoning.
- "No correlation means no relationship" -- Clarify: r measures *linear* association. A perfect U-shape has r near 0. Always plot your data.
- "Correlation of 0.5 means 50% relationship" -- Clarify: r-squared (0.25) is the proportion of variance explained. The scale of r is not intuitive as a percentage.

**Verification Questions:**
1. "What does a Pearson correlation of -0.7 tell you? What does it not tell you?"
2. "Can you think of two variables that are correlated but where one clearly does not cause the other?"
3. "If I told you the correlation between two stocks is 0.9, would you say they are good for diversification? Why or why not?"

**Good answer indicators:**
- They separate description (co-movement) from explanation (causation)
- They can generate their own spurious correlation example
- They mention confounders or third variables unprompted

**If they struggle:**
- Use the ice cream / drowning example: both increase in summer (confounder: temperature)
- Draw out the logic: "Just because A and B move together, can you think of a C that might drive both?"
- Show Anscombe's quartet: four datasets with the same r but wildly different shapes

**Exercise 1.1: Compute and Interpret Correlations**
Present this exercise: "Here is a dataset of NBA team statistics for one season: pace, offensive rating, defensive rating, three-point percentage, and win total. Compute the correlation matrix and identify: (a) the strongest positive correlation, (b) the strongest negative correlation, (c) a pair you think might be spuriously correlated."

**How to Guide Them:**
1. Ask: "How would you compute correlations for multiple variables at once?"
2. If stuck: "pandas DataFrames have a `.corr()` method"
3. Push on interpretation: "Why are those two variables correlated? Is there a confounder?"

**After exercise, ask:**
- "Which correlations surprised you? For the strongest, is the relationship causal?"

---

### Section 2: Simple Linear Regression

**Core Concept to Teach:**
Simple linear regression fits the equation y = mx + b to data, finding the line that minimizes the sum of squared residuals (errors). It is the most fundamental predictive model and the building block for everything that follows.

**How to Explain:**
1. Start with the analogy: "Imagine you have a scatter plot of data. Regression finds the single straight line that best summarizes the trend."
2. Define the components: y is the outcome (dependent variable), x is the predictor (independent variable), m is the slope, b is the intercept
3. Explain least squares: "Best fit means the line that minimizes the total squared distance between each data point and the line. Why squared? Because we want to penalize big errors more than small ones, and squaring prevents positive and negative errors from canceling out."

**Example to Present:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data: turnovers per game (x) and points scored (y)
np.random.seed(42)
turnovers = np.random.uniform(10, 20, 25)
points = 120 - 2.5 * turnovers + np.random.normal(0, 4, 25)

model = LinearRegression()
model.fit(turnovers.reshape(-1, 1), points)

print(f"Slope:     {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
# Output:
# Slope:     -2.330
# Intercept: 118.194
```

**Walk Through:**
- Slope of -2.33: "For each additional turnover per game, the model predicts about 2.3 fewer points scored."
- Intercept of 118.2: "A team with zero turnovers would be predicted to score about 118 points." (Note: extrapolation beyond data range -- no team has zero turnovers.)
- This is a *model*, not a law. The true relationship has noise.

**Residuals -- What the Model Gets Wrong:**
```python
predictions = model.predict(turnovers.reshape(-1, 1))
residuals = points - predictions
print(f"Mean residual: {residuals.mean():.6f}")   # Should be ~0
print(f"Std of residuals: {residuals.std():.3f}")
```

Explain: residuals should have no pattern. If you plot residuals vs. predictions and see a curve or a funnel, the model is missing something.

**Domain Applications:**
- **Sports**: Predicting points from a single stat (turnovers, pace, shooting percentage). This is how you start building a game prediction model.
- **Finance**: The Capital Asset Pricing Model (CAPM) says expected return = risk-free rate + beta * (market return - risk-free rate). Beta is literally the slope of a regression of stock returns on market returns.
- **ML**: Simple linear regression is the simplest supervised learning algorithm. It introduces the core ML workflow: fit on training data, evaluate on test data.

**Common Misconceptions:**
- "The line passes through all points" -- Clarify: the line passes through the *center* of the data cloud. Individual points scatter around it. That scatter is the noise your model cannot explain.
- "A good fit means good predictions" -- Clarify: a model can fit historical data perfectly (especially with few data points) and predict new data terribly. This is overfitting, and we will cover it in Section 5.
- "The slope tells you about causation" -- Clarify: the slope tells you the *association* in the data. Whether turnovers cause fewer points, or whether both are driven by opponent quality, requires causal reasoning beyond the regression itself.

**Verification Questions:**
1. "If the slope of a regression is 3.5, what does that mean in plain language?"
2. "Why do we use squared errors instead of just absolute errors in ordinary least squares?"
3. "What would it mean if the residuals showed a clear curved pattern?"

**Good answer indicators:**
- They interpret slope in the units of x and y
- They understand that residual patterns indicate model misspecification
- They do not claim the regression proves causation

**If they struggle:**
- Go back to a concrete example: "If slope = -2, and a team commits one more turnover, the model predicts 2 fewer points."
- Draw or describe the idea of vertical distances from points to the line
- Show that minimizing squared errors gives more weight to outliers, which is both a strength and a vulnerability

**Exercise 2.1: Build a Simple Linear Regression**
Present this exercise: "Using the NBA team data from Exercise 1.1, pick the predictor that had the strongest correlation with wins. Fit a simple linear regression model. Report the slope, intercept, and R-squared. Then interpret each in plain English."

**How to Guide Them:**
1. Ask: "Which predictor will you choose, and why?"
2. If stuck on code: "scikit-learn's `LinearRegression` or statsmodels' `OLS` both work. Which would you like to try?"
3. Push on interpretation: "What does that slope mean for a real team?"

**Solution:**
```python
import statsmodels.api as sm

X = sm.add_constant(data['off_rating'])
y = data['wins']
model = sm.OLS(y, X).fit()
print(model.summary())
```

**After exercise, ask:**
- "Does the R-squared value surprise you? Is it high enough to be useful?"
- "What other variables might improve the model?" (This leads into Section 3.)

---

### Section 3: Multiple Linear Regression

**Core Concept to Teach:**
Multiple linear regression adds more predictors to the model: y = b0 + b1*x1 + b2*x2 + ... + bn*xn. Each coefficient represents the effect of that predictor *holding all other predictors constant*. This is a powerful tool but introduces new complications.

**How to Explain:**
1. Transition from simple: "Your single-predictor model explained some variance in wins. But wins depend on many things -- offense, defense, shooting, turnovers. Multiple regression lets you include all of them."
2. Key interpretation: "The coefficient on offensive rating now means: the expected change in wins for a one-unit increase in offensive rating, *holding defensive rating, three-point percentage, and everything else constant.*"
3. This "holding constant" interpretation is what makes multiple regression so useful -- and so easy to misinterpret.

**Example to Present:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Using our team data
features = data[['off_rating', 'def_rating', 'three_pct']]
target = data['wins']

model = LinearRegression()
model.fit(features, target)

print("Coefficients:")
for name, coef in zip(features.columns, model.coef_):
    print(f"  {name}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R-squared: {model.score(features, target):.3f}")
```

**Walk Through:**
- Compare R-squared to the simple model. Did adding predictors help?
- Each coefficient has a "holding other variables constant" interpretation
- The intercept is the predicted wins when all predictors are zero (usually meaningless -- no team has zero offensive rating)

**Feature Selection: Which Variables Matter?**
- Not every variable improves the model
- Adding irrelevant predictors increases complexity without increasing true predictive power
- Use statsmodels to get p-values for each coefficient; drop predictors with high p-values
- Adjusted R-squared penalizes for number of predictors (unlike regular R-squared, which always goes up)

**Multicollinearity -- The Hidden Danger:**
"What happens when two predictors are highly correlated with each other? If offensive rating and points per game are both in the model, and they are correlated at r = 0.95, the model cannot tell which one matters. The coefficients become unstable -- small changes in the data produce wildly different coefficients."

How to detect: compute Variance Inflation Factors (VIF). A VIF above 5-10 signals a problem.

How to fix: remove one of the correlated predictors, or combine them (e.g., average them, or use PCA).

**Domain Applications:**
- **Sports**: Predicting game outcomes from multiple team stats. A model with offensive rating, defensive rating, and pace captures more than any single stat alone.
- **Finance**: The Fama-French three-factor model adds size and value factors beyond market beta. Multi-factor models are the standard in quantitative finance.
- **ML**: Feature engineering and selection are central to ML pipelines. Adding too many features leads to overfitting. Removing useless features improves generalization.

**Common Misconceptions:**
- "More predictors always make a better model" -- Clarify: more predictors always increase R-squared on training data, but can decrease performance on new data. Adjusted R-squared, AIC, and cross-validation address this.
- "Each coefficient tells me the total effect of that variable" -- Clarify: each coefficient is the *partial* effect, conditional on other predictors. Change the set of predictors and the coefficients change.
- "High VIF means I should definitely remove that variable" -- Clarify: multicollinearity affects coefficient interpretation but does not necessarily hurt prediction. It depends on your goal.

**Verification Questions:**
1. "You add a new predictor to a regression model and R-squared goes up from 0.65 to 0.66. Is that a meaningful improvement?"
2. "What does it mean to say 'the effect of offensive rating, holding defensive rating constant'?"
3. "Two predictors have a correlation of 0.92. What problems might this cause?"

**If they struggle:**
- Use a concrete analogy: "If you want to know whether coffee improves productivity, you also need to hold sleep constant. Otherwise you might confuse the effect of coffee with the effect of being well-rested."
- Show a before/after: fit a model with and without a correlated predictor, and show how coefficients change dramatically
- Emphasize that multiple regression is asking: "What is the unique contribution of each predictor?"

**Exercise 3.1: Build and Compare Multiple Regression Models**
Present this exercise: "Build three models for predicting wins: (a) offensive rating only, (b) offensive + defensive rating, (c) all available predictors. Compare R-squared, adjusted R-squared, and interpret the coefficients. Which model would you choose and why?"

**How to Guide Them:**
1. Let them build all three models
2. Push on comparison: "Does adding defensive rating help? By how much?"
3. Ask about trade-offs: "The full model has the highest R-squared. Is it necessarily the best?"
4. If they focus only on R-squared, prompt: "What other metrics might we use to compare?" (Foreshadows Section 5.)

**Solution:**
```python
import statsmodels.api as sm

# Model A: offensive rating only
X_a = sm.add_constant(data[['off_rating']])
model_a = sm.OLS(data['wins'], X_a).fit()

# Model B: offensive + defensive rating
X_b = sm.add_constant(data[['off_rating', 'def_rating']])
model_b = sm.OLS(data['wins'], X_b).fit()

# Model C: all predictors
X_c = sm.add_constant(data[['off_rating', 'def_rating', 'three_pct', 'pace']])
model_c = sm.OLS(data['wins'], X_c).fit()

for name, m in [('A', model_a), ('B', model_b), ('C', model_c)]:
    print(f"Model {name}: R²={m.rsquared:.3f}, Adj R²={m.rsquared_adj:.3f}, AIC={m.aic:.1f}")
```

---

### Section 4: Logistic Regression

**Core Concept to Teach:**
Logistic regression is used when the outcome is binary -- win or lose, default or no default, click or no click. Instead of predicting a continuous number, it predicts the *probability* that the outcome is 1 (the positive class). Despite the name, logistic regression *is* a regression technique -- it regresses on the log-odds of the outcome.

**How to Explain:**
1. Motivate: "What if we want to predict whether a team wins, not how many points they score? The outcome is 0 or 1. A straight line does not work because it can predict values below 0 or above 1."
2. Introduce the sigmoid function: sigma(z) = 1 / (1 + e^(-z)). It maps any real number to a value between 0 and 1 -- a probability.
3. The model: log(p / (1-p)) = b0 + b1*x1 + b2*x2 + ... The left side is the log-odds. The right side is a linear combination, just like regular regression.

**Example to Present:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200
point_diff = np.random.normal(0, 10, n)  # positive = home advantage
home_win = (point_diff + np.random.normal(0, 3, n) > 0).astype(int)

model = LogisticRegression()
model.fit(point_diff.reshape(-1, 1), home_win)

print(f"Coefficient: {model.coef_[0][0]:.3f}")
print(f"Intercept:   {model.intercept_[0]:.3f}")
print(f"Accuracy:    {accuracy_score(home_win, model.predict(point_diff.reshape(-1, 1))):.3f}")
```

**Walk Through:**
- The coefficient on point_diff is positive: as point differential increases, the probability of a home win increases
- Unlike linear regression, the coefficient is in log-odds space. To interpret: "For each one-unit increase in point differential, the *log-odds* of winning increase by [coefficient]."
- Converting to odds ratios: exp(coefficient) gives the multiplicative change in odds

**Odds Ratios -- Making Coefficients Interpretable:**
```python
import numpy as np
odds_ratio = np.exp(model.coef_[0][0])
print(f"Odds ratio: {odds_ratio:.3f}")
# If odds_ratio = 1.2, then each unit increase in point_diff
# multiplies the odds of winning by 1.2 (20% increase in odds)
```

Explain the difference between probability and odds:
- Probability: p = 0.75 means 75% chance
- Odds: 0.75 / 0.25 = 3 to 1 (three times more likely to happen than not)
- Log-odds: ln(3) = 1.099

**Domain Applications:**
- **Sports betting**: "Given a team's stats, what is the probability they win? If your model says 60% but the bookmaker's odds imply 50%, there may be a value bet." This is exactly how sports betting models work.
- **Finance**: Credit scoring -- predict the probability a borrower defaults. Logistic regression has been a workhorse in banking for decades.
- **ML**: Logistic regression is the foundational classification algorithm. It is fast, interpretable, and often the first model you try. If it works well enough, you do not need a neural network.

**Common Misconceptions:**
- "Logistic regression is not regression" -- Clarify: It absolutely is regression. It regresses the log-odds on the predictors. The name is correct.
- "The output is a hard class label" -- Clarify: The output is a probability. You choose a threshold (often 0.5) to convert to a class label. The threshold choice matters and should be tuned.
- "Accuracy is the right metric" -- Clarify: If 90% of games are won by the home team, a model that always predicts "home win" has 90% accuracy but is useless. We need better metrics (precision, recall, log-loss). Section 5 covers this.

**Verification Questions:**
1. "Why can't we just use linear regression to predict a binary outcome?"
2. "A logistic regression coefficient is 0.5. What is the odds ratio, and what does it mean?"
3. "Your model predicts a 55% chance of a home win. The betting line implies 50%. What would you do?"

**Good answer indicators:**
- They understand the sigmoid maps to probabilities
- They can convert between log-odds, odds ratios, and probabilities
- They recognize that accuracy alone can be misleading

**If they struggle:**
- Focus on the sigmoid curve: draw/describe it. "For very negative inputs, the probability is near 0. For very positive inputs, near 1. In between, it transitions smoothly."
- Use concrete numbers: "If odds ratio = 2, it means the odds double for each unit increase in x. If a team with 5 turnovers has 2:1 odds of winning, 6 turnovers would give them 4:1 odds."
- Compare with a real example: "Should you bet on a team your model gives a 60% chance to win if the payout is 2:1? Yes -- expected value is positive."

**Exercise 4.1: Build a Logistic Regression Classifier**
Present this exercise: "Using team statistics (offensive rating, defensive rating, three-point percentage), build a logistic regression model to predict whether a team wins. Report the accuracy, print the odds ratios for each feature, and interpret them."

**How to Guide Them:**
1. Ask them to create the binary outcome variable (wins above/below median, or simulate game outcomes)
2. Push on odds ratio interpretation: "A team improves their offensive rating by 1 point. How much do their odds of winning change?"
3. Ask: "Is accuracy the best way to evaluate this model? What could go wrong?"

**Solution:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Create binary outcome: above-median wins
data['win_class'] = (data['wins'] >= data['wins'].median()).astype(int)

X = data[['off_rating', 'def_rating', 'three_pct']]
y = data['win_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print("\nOdds Ratios:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name}: {np.exp(coef):.3f}")
```

---

### Section 5: Model Evaluation

**Core Concept to Teach:**
Building a model is only half the job. Evaluating whether the model is any good -- and whether it will generalize to new data -- is the other half. This section covers the metrics and methods that separate useful models from dangerous ones.

**How to Explain:**
1. Motivate: "Your regression model has an R-squared of 0.85 on the training data. Is that good? What if I told you it has an R-squared of 0.30 on new data? That gap is the difference between a model that memorizes and a model that learns."
2. Evaluation is about answering: "How wrong is this model, and will it be this wrong on data it has not seen yet?"

**R-squared and Adjusted R-squared:**
- R-squared: the proportion of variance in y explained by the model. R-squared = 1 - (SS_res / SS_tot).
- It ranges from 0 (model explains nothing) to 1 (model explains everything).
- Problem: R-squared always increases when you add predictors, even useless ones.
- Adjusted R-squared penalizes for the number of predictors. It can decrease if a new predictor does not help enough to justify the added complexity.

**MSE, RMSE, and MAE:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = np.array([100, 105, 98, 112, 95])
y_pred = np.array([102, 103, 99, 108, 97])

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)

print(f"MSE:  {mse:.2f}")    # Mean Squared Error
print(f"RMSE: {rmse:.2f}")   # Root Mean Squared Error (same units as y)
print(f"MAE:  {mae:.2f}")    # Mean Absolute Error (less sensitive to outliers)
```

Explain the tradeoff: MSE/RMSE penalize large errors more heavily. MAE treats all errors equally. Which you prefer depends on whether large errors are disproportionately bad in your application. In sports betting, a prediction that is off by 20 points is far worse than two predictions off by 10 points each -- use MSE/RMSE.

**AIC and BIC -- Comparing Models:**
- Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) both balance goodness of fit against model complexity
- Lower AIC/BIC is better
- BIC penalizes complexity more heavily than AIC, preferring simpler models
- Use them to compare non-nested models: "Model A has two predictors and AIC = 150. Model B has four predictors and AIC = 148. Model B is slightly better, but barely -- the extra complexity is almost not justified."

**Cross-Validation -- The Gold Standard:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, features, target, cv=5, scoring='r2')
print(f"CV R-squared scores: {scores.round(3)}")
print(f"Mean CV R-squared:   {scores.mean():.3f}")
print(f"Std CV R-squared:    {scores.std():.3f}")
```

Explain: cross-validation splits the data into k folds, trains on k-1, tests on the held-out fold, and rotates. It gives a realistic estimate of how the model will perform on unseen data. If your training R-squared is 0.85 but your cross-validated R-squared is 0.40, you are overfitting.

**Overfitting vs. Underfitting:**
- Overfitting: the model is too complex. It memorizes noise in the training data and fails on new data. Signs: high training score, low test/CV score.
- Underfitting: the model is too simple. It cannot capture the real pattern. Signs: low training score, low test/CV score.
- The goal is the sweet spot: a model complex enough to capture the real signal, but not so complex that it captures noise.

**Domain Applications:**
- **Sports**: "Your model predicts game outcomes with 65% accuracy. The baseline (pick the home team) is 58%. Is that improvement real, or just overfitting? Cross-validate to find out."
- **Finance**: Backtesting a trading strategy on historical data is like evaluating a model on training data. Without proper out-of-sample testing, you have no idea if the strategy actually works. This is why so many backtested strategies fail in live trading.
- **ML**: Train/test split and cross-validation are non-negotiable in ML. Never report training accuracy as your model's performance.

**Common Misconceptions:**
- "Higher R-squared is always better" -- Clarify: R-squared can be inflated by adding useless predictors or by overfitting. Always use cross-validated metrics for model comparison.
- "My model has 95% accuracy, so it's great" -- Clarify: check the baseline. If the classes are imbalanced (e.g., 95% of the data is one class), a model that always predicts the majority class gets 95% accuracy and learns nothing.
- "Cross-validation tells you the true performance" -- Clarify: cross-validation estimates expected performance on similar data. If future data has a different distribution (e.g., a new NBA season with rule changes), even cross-validated estimates may be optimistic.

**Verification Questions:**
1. "Your model has R-squared of 0.90 on training data and 0.45 on cross-validation. What is happening?"
2. "Model A has RMSE = 5.2 and 3 predictors. Model B has RMSE = 5.0 and 10 predictors. Which would you choose?"
3. "Why should you never use training accuracy as your final evaluation metric?"

**If they struggle:**
- Use the exam analogy: "Imagine studying only the answer key. You will ace that specific test, but fail any other test on the same material. That is overfitting."
- Show a concrete example: fit a polynomial of degree 15 to 20 data points. It passes through every point but oscillates wildly between them.
- Emphasize the practical consequence: "In sports betting, overfitting means your model looks profitable on historical data but loses money on future games."

**Exercise 5.1: Evaluate and Compare Models**
Present this exercise: "Take the three models from Exercise 3.1 (one, two, and all predictors). For each model, compute: (a) training R-squared, (b) 5-fold cross-validated R-squared, (c) RMSE. Which model performs best on cross-validation? Is it the same model that has the highest training R-squared?"

**How to Guide Them:**
1. Let them compute all metrics
2. Ask: "What pattern do you notice between training and CV R-squared as you add predictors?"
3. Push: "If the most complex model overfits, what would you do about it?"
4. Discuss: "How would you explain your model choice to someone who says 'just use all the data'?"

---

## Practice Project

**Project Introduction:**
"Now let's put everything together. You are going to build a sports prediction model from scratch. You will explore the data, build multiple models, evaluate them properly, and see if you can beat a simple baseline."

**Requirements:**
Present these requirements:
- Load or create a dataset of team/game statistics (at least 5 features and 50+ observations)
- Perform exploratory analysis: correlation matrix, scatter plots, identify potential issues
- Build at least two regression models to predict a continuous outcome (e.g., point differential)
- Build at least one logistic regression model to predict a binary outcome (e.g., win/loss)
- Evaluate all models using cross-validation (not just training metrics)
- Compare against a naive baseline (e.g., home team always wins, or predict the league average)
- Report results clearly with metrics and at least one visualization

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work, offer to answer questions. Check in after they have built their first model.
2. **If they want guidance**: Walk through each step together. Start with data exploration, then build one model at a time.
3. **If they are unsure**: Suggest starting with the correlation matrix and identifying the two best predictors.

**Checkpoints During Project:**
- After data exploration: "What did you learn from the correlation matrix? Any surprises?"
- After first regression model: "How does the R-squared look? What does it mean for your predictions?"
- After logistic model: "What accuracy did you get? How does it compare to the baseline?"
- After cross-validation: "Did any of your models overfit? How can you tell?"
- At completion: "Which model would you actually use to make predictions? Why?"

**Code Review Approach:**
Start with praise, ask questions rather than correct ("What happens if you change the threshold?"), and relate their work back to concepts covered earlier.

**If They Get Stuck:**
- On data: suggest synthetic data with known relationships
- On modeling: walk through the sklearn API step by step
- On evaluation: provide the `cross_val_score` template
- On interpretation: ask them to state what each number means in plain English

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned:"
- Correlation measures linear association but not causation. Always look for confounders and plot your data.
- Simple linear regression fits a line to data. The slope tells you the association; residuals tell you what the model misses.
- Multiple regression adds predictors but introduces multicollinearity. More predictors are not always better.
- Logistic regression predicts probabilities of binary outcomes. Coefficients are in log-odds space; convert to odds ratios for interpretation.
- Model evaluation requires cross-validation or held-out test data. Training metrics alone are dangerously optimistic.

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with regression and modeling?"
- 1-4: Re-read the guide, offer to re-teach the weakest topic with different examples
- 5-7: Normal for first exposure -- suggest practice with real datasets from Kaggle or sports-reference.com
- 8-10: Suggest regularization (Ridge, Lasso), polynomial regression, or Bayesian approaches

**Suggest Next Steps:**
- [Bayesian Statistics](/routes/bayesian-statistics/map.md) -- alternative framework with prior beliefs
- [Neural Network Foundations](/routes/neural-network-foundations/map.md) -- regression generalizes into deep learning
- Time Series Analysis -- regression with temporal structure
- Practice with real data from basketball-reference.com, FanGraphs, or Yahoo Finance

**Encourage Questions:**
"Do you have any questions? Is there a specific domain where you want to go deeper?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Slow down, use concrete examples with real numbers and scatter plots
- Do exercises together rather than assigning them solo
- Use tiny datasets (5-10 points) to illustrate concepts
- Check prerequisites -- they may need Stats Fundamentals or Statistical Inference

### If Learner is Excelling
- Move faster, introduce regularization (Ridge, Lasso), polynomial features, interaction terms
- Bring up the bias-variance tradeoff mathematically
- Challenge with real-world complications: missing data, outliers, non-linearity

### If Learner Seems Disengaged
- Connect to their domain interest: sports betting examples for sports fans, stock data for finance
- Switch between formulas and code to find what resonates

### Different Learning Styles
- **Visual**: Scatter plots, residual plots, the sigmoid curve
- **Hands-on**: Jump to code, let them experiment
- **Conceptual**: "Why" before "how" -- why least squares? Why log-odds?
- **Example-driven**: Lead with real sports/finance data, extract concepts after

---

## Troubleshooting Common Issues

### Technical Setup Problems
- **scikit-learn not installed**: `pip install scikit-learn`
- **statsmodels not installed**: `pip install statsmodels`
- **Import errors**: Verify Python environment and package versions. `sklearn` requires `import sklearn`, not `import scikit-learn`.
- **Data format issues**: scikit-learn expects 2D arrays for X. Use `.reshape(-1, 1)` for single-feature arrays.

### Concept-Specific Confusion

- **Correlation vs. causation**: Use the third-variable explanation: "A correlates with B, but maybe C causes both." Have them generate their own spurious example.
- **Residuals**: Draw vertical distances from points to the line. Residuals = actual - predicted.
- **Logistic regression output**: Walk through one prediction: compute log-odds, convert to odds, convert to probability. "log-odds = 0 means odds = 1:1 means probability = 0.5."
- **Overfitting**: Memorization analogy: "You memorized last year's exam answers. You ace that test but fail this year's."

---

## Additional Resources to Suggest

- **Practice**: Kaggle competitions, basketball-reference.com, FanGraphs, Yahoo Finance
- **Deeper understanding**: "An Introduction to Statistical Learning" (James et al.), statsmodels docs
- **Real applications**: FiveThirtyEight prediction methodology, "The Signal and the Noise" (Nate Silver)

---

## Teaching Notes

**Key Emphasis Points:**
- Really emphasize "correlation is not causation" in Section 1. It is the most important concept in the entire route and the most commonly violated in practice.
- Make sure they understand residuals before moving to multiple regression. Residual analysis is how you know when a model is wrong.
- In Section 5, drive home that training metrics alone are worthless. Cross-validation is not optional -- it is the minimum standard for responsible modeling.

**Pacing Guidance:**
- Do not rush Sections 1-2. They set the conceptual foundation.
- Sections 3-4 can move faster if they are comfortable with Section 2, since the ideas extend naturally.
- Allow plenty of time for the practice project -- this is where everything comes together.
- If time is limited, Section 5 (Model Evaluation) is the most important section to cover thoroughly.

**Success Indicators:**
You will know they have got it when they:
- Can explain the difference between correlation and causation with their own example
- Interpret regression coefficients in plain language with correct units
- Choose cross-validation over training metrics without being prompted
- Question whether a model is "good enough" rather than accepting any R-squared value
- Connect the concepts to their domain of interest (sports, finance, or ML)
