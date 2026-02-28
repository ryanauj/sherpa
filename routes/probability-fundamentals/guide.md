---
title: Probability Fundamentals
route_map: /routes/probability-fundamentals/map.md
paired_sherpa: /routes/probability-fundamentals/sherpa.md
prerequisites:
  - Basic arithmetic and fractions
  - Stats Fundamentals (helpful)
topics:
  - Sample Spaces and Events
  - Probability Rules
  - Conditional Probability
  - Bayes' Theorem
  - Expected Value
---

# Probability Fundamentals

## Overview

Probability is the mathematics of uncertainty. It is the reason casinos are profitable, the foundation of machine learning classifiers, and the language that financial analysts use to talk about risk. Whether you want to understand why the house always wins, build a spam filter, or evaluate an investment, you need probability.

This guide teaches probability from the ground up using real-world examples from gambling, sports betting, data science, and finance. By the end, you will be able to compute probabilities, update beliefs with new evidence using Bayes' theorem, and calculate expected values that tell you whether a bet or decision is worth taking.

> **Note for AI assistants**: This route has a paired teaching guide at `/routes/probability-fundamentals/sherpa.md` that provides structured guidance for teaching this material interactively.

## Learning Objectives

By the end of this route, you will be able to:
- Define probability and enumerate sample spaces for common random experiments
- Apply addition, multiplication, and complement rules to compute combined probabilities
- Compute conditional probabilities and explain how new information changes outcomes
- Apply Bayes' theorem to update beliefs when new evidence arrives
- Calculate expected value and use it to identify favorable and unfavorable bets

## Prerequisites

Before starting this guide, you should be comfortable with:
- Basic arithmetic: addition, subtraction, multiplication, division
- Fractions, decimals, and percentages (e.g., converting 1/4 to 0.25 to 25%)

If you want broader statistical context first, check out:
- [Stats Fundamentals](/routes/stats-fundamentals/map.md) -- descriptive statistics, data types, and visualization

## Setup

This guide uses Python for code examples and exercises. You need:
- Python 3.8 or later
- numpy (for the practice project)

```bash
pip install numpy
```

**Verify your setup:**
```bash
python3 -c "import numpy; print(f'numpy {numpy.__version__} ready')"
```

You should see:
```
numpy 1.24.0 ready
```

(Your version number may differ -- any recent version works.)

---

## Section 1: What Is Probability?

### The Meaning of Probability

Probability assigns a number between 0 and 1 to an event, measuring how likely it is to occur. A probability of 0 means the event is impossible. A probability of 1 means it is certain. Everything else falls somewhere in between.

Think of probability as a confidence meter. When a weather app says "70% chance of rain," it is telling you that in similar conditions, it rains about 7 out of 10 times. It is not a guarantee -- it is a quantified best estimate.

### Sample Spaces and Events

Every probability problem starts with two things:

1. **Sample space (S)**: The set of all possible outcomes.
2. **Event (E)**: A subset of the sample space -- the outcomes you care about.

The probability of an event is:

**P(E) = number of favorable outcomes / total number of outcomes**

(This formula applies when all outcomes are equally likely.)

### Examples

Let's build sample spaces for common scenarios:

```python
# Coin flip
coin_sample_space = {"Heads", "Tails"}
p_heads = 1 / len(coin_sample_space)
print(f"P(Heads) = {p_heads}")  # 0.5

# Six-sided die
die_sample_space = {1, 2, 3, 4, 5, 6}
even_numbers = {2, 4, 6}
p_even = len(even_numbers) / len(die_sample_space)
print(f"P(even) = {p_even}")  # 0.5

# Standard deck of 52 cards
total_cards = 52
aces = 4
p_ace = aces / total_cards
print(f"P(Ace) = {p_ace:.4f} = {aces}/{total_cards}")  # 0.0769 = 4/52
```

**Expected Output:**
```
P(Heads) = 0.5
P(even) = 0.5
P(Ace) = 0.0769 = 4/52
```

### Casino Application: The Roulette Wheel

An American roulette wheel has 38 slots: numbers 1 through 36 (colored red or black), plus 0 and 00 (colored green). The sample space has exactly 38 equally likely outcomes.

```python
# American roulette sample space
total_slots = 38
red_slots = 18
black_slots = 18
green_slots = 2  # 0 and 00

p_red = red_slots / total_slots
p_single_number = 1 / total_slots

print(f"P(red) = {red_slots}/{total_slots} = {p_red:.4f}")
print(f"P(single number) = 1/{total_slots} = {p_single_number:.4f}")
print(f"P(green) = {green_slots}/{total_slots} = {green_slots/total_slots:.4f}")
```

**Expected Output:**
```
P(red) = 18/38 = 0.4737
P(single number) = 1/38 = 0.0263
P(green) = 2/38 = 0.0526
```

Notice that P(red) is not 1/2 -- the two green slots tilt the odds. This is the foundation of the house edge.

### Three Interpretations of Probability

There are three ways to think about what probability means:

1. **Classical**: When all outcomes are equally likely, count favorable outcomes divided by total. This is what we have been doing with dice and cards.

2. **Frequentist**: Probability is the long-run frequency. If you flip a coin 10,000 times, about 5,000 flips will be heads. The more flips, the closer the frequency gets to 0.5.

3. **Subjective**: Probability as a degree of belief. A sports analyst says "I think there is a 65% chance Team A wins." There is no sample space to count -- this is a personal assessment of uncertainty.

All three interpretations are useful. Casino games use classical probability. Scientific experiments use frequentist probability. Bayesian statistics and betting markets use subjective probability.

### The Gambler's Fallacy

This is one of the most important concepts in probability. The **gambler's fallacy** is the mistaken belief that past outcomes affect future independent events.

If a roulette wheel has landed on red 10 times in a row, what is the probability the next spin is red? It is still 18/38. The wheel has no memory. Each spin is independent.

```python
# The wheel doesn't remember
# After 10 reds in a row, the next spin is still:
p_red_next = 18 / 38
print(f"P(red on next spin) = {p_red_next:.4f}")
print("This is true regardless of what happened on previous spins.")
```

**Expected Output:**
```
P(red on next spin) = 0.4737
This is true regardless of what happened on previous spins.
```

Casinos love the gambler's fallacy. It keeps people betting after long streaks, convinced that the pattern "must" break.

### Exercise 1.1: Building Sample Spaces

**Task:** For each scenario below, list the complete sample space and compute the requested probability.

(a) Rolling a six-sided die: P(number greater than 4)
(b) Drawing one card from a standard 52-card deck: P(face card) -- face cards are Jack, Queen, King
(c) Flipping three coins: P(exactly two heads)

**Hints:**
<details>
<summary>Hint 1: Getting started</summary>
For each problem, start by writing out every possible outcome. For (c), use a systematic approach: list all combinations starting with HHH, HHT, HTH, etc.
</details>

<details>
<summary>Hint 2: Counting three-coin outcomes</summary>
Three coins means 2 x 2 x 2 = 8 total outcomes. List them all, then circle the ones with exactly two heads.
</details>

**Solution:**
<details>
<summary>Click to see solution</summary>

```python
# (a) Die roll: P(greater than 4)
die_space = {1, 2, 3, 4, 5, 6}
greater_than_4 = {5, 6}
p_a = len(greater_than_4) / len(die_space)
print(f"(a) P(>4) = {len(greater_than_4)}/{len(die_space)} = {p_a:.4f}")

# (b) Card draw: P(face card)
total_cards = 52
face_cards = 12  # 3 face cards (J, Q, K) x 4 suits
p_b = face_cards / total_cards
print(f"(b) P(face card) = {face_cards}/{total_cards} = {p_b:.4f}")

# (c) Three coin flips: P(exactly 2 heads)
three_coin_space = [
    "HHH", "HHT", "HTH", "HTT",
    "THH", "THT", "TTH", "TTT"
]
exactly_two_heads = [outcome for outcome in three_coin_space
                     if outcome.count("H") == 2]
p_c = len(exactly_two_heads) / len(three_coin_space)
print(f"(c) Sample space: {three_coin_space}")
print(f"    Exactly 2 heads: {exactly_two_heads}")
print(f"    P(exactly 2 heads) = {len(exactly_two_heads)}/{len(three_coin_space)} = {p_c:.4f}")
```

**Expected Output:**
```
(a) P(>4) = 2/6 = 0.3333
(b) P(face card) = 12/52 = 0.2308
(c) Sample space: ['HHH', 'HHT', 'HTH', 'HTT', 'THH', 'THT', 'TTH', 'TTT']
    Exactly 2 heads: ['HHT', 'HTH', 'THH']
    P(exactly 2 heads) = 3/8 = 0.3750
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Define what a sample space is and why it matters
- [ ] Compute the probability of an event by counting favorable outcomes
- [ ] Explain the gambler's fallacy and why it is wrong
- [ ] Describe the three interpretations of probability

---

## Section 2: Probability Rules

Now that you can compute probabilities for single events, you need rules for combining them. There are three fundamental rules: addition (OR), multiplication (AND), and complement (NOT).

### The Addition Rule (OR)

When you want the probability of event A **or** event B occurring:

**P(A or B) = P(A) + P(B) - P(A and B)**

If A and B are **mutually exclusive** (they cannot both happen), then P(A and B) = 0, and the formula simplifies to:

**P(A or B) = P(A) + P(B)**

```python
# Mutually exclusive: drawing a heart OR a diamond
# A card cannot be both a heart and a diamond
p_heart = 13 / 52
p_diamond = 13 / 52
p_heart_or_diamond = p_heart + p_diamond
print(f"P(heart or diamond) = {p_heart} + {p_diamond} = {p_heart_or_diamond:.4f}")

# NOT mutually exclusive: drawing a heart OR a king
# The king of hearts is both!
p_heart = 13 / 52
p_king = 4 / 52
p_heart_and_king = 1 / 52  # king of hearts
p_heart_or_king = p_heart + p_king - p_heart_and_king
print(f"P(heart or king) = {13}/{52} + {4}/{52} - {1}/{52} = {p_heart_or_king:.4f}")
```

**Expected Output:**
```
P(heart or diamond) = 0.25 + 0.25 = 0.5000
P(heart or king) = 13/52 + 4/52 - 1/52 = 0.3077
```

### The Multiplication Rule (AND)

When you want the probability of event A **and** event B both occurring:

**P(A and B) = P(A) x P(B)** (when A and B are independent)

Two events are **independent** if the outcome of one does not affect the other.

```python
# Independent events: rolling a 6 AND flipping heads
p_six = 1 / 6
p_heads = 1 / 2
p_six_and_heads = p_six * p_heads
print(f"P(roll 6 AND flip heads) = {p_six:.4f} x {p_heads:.4f} = {p_six_and_heads:.4f}")

# Independent: rolling a 6 twice in a row
p_two_sixes = (1/6) * (1/6)
print(f"P(two sixes in a row) = (1/6) x (1/6) = {p_two_sixes:.4f}")

# NOT independent: drawing two aces from a deck WITHOUT replacement
p_first_ace = 4 / 52
p_second_ace_given_first = 3 / 51  # one ace removed, one card removed
p_two_aces = p_first_ace * p_second_ace_given_first
print(f"P(two aces without replacement) = {p_first_ace:.4f} x {p_second_ace_given_first:.4f} = {p_two_aces:.4f}")
```

**Expected Output:**
```
P(roll 6 AND flip heads) = 0.1667 x 0.5000 = 0.0833
P(two sixes in a row) = (1/6) x (1/6) = 0.0278
P(two aces without replacement) = 0.0769 x 0.0588 = 0.0045
```

### The Complement Rule (NOT)

The probability of an event NOT happening is:

**P(not A) = 1 - P(A)**

This is surprisingly useful. Often it is easier to compute the probability of something NOT happening and subtract from 1.

```python
# What is the probability of getting at least one 6 in four die rolls?
# Direct approach: complex (one 6, two 6s, three 6s, four 6s)
# Complement approach: easy!
p_no_six_one_roll = 5 / 6
p_no_six_four_rolls = p_no_six_one_roll ** 4
p_at_least_one_six = 1 - p_no_six_four_rolls

print(f"P(no 6 in one roll) = {p_no_six_one_roll:.4f}")
print(f"P(no 6 in four rolls) = {p_no_six_four_rolls:.4f}")
print(f"P(at least one 6 in four rolls) = {p_at_least_one_six:.4f}")
```

**Expected Output:**
```
P(no 6 in one roll) = 0.8333
P(no 6 in four rolls) = 0.4823
P(at least one 6 in four rolls) = 0.5177
```

### Casino Application: Craps

In craps, you roll two dice and sum them. On the "come-out roll," you win immediately with a 7 or 11, and lose immediately with a 2, 3, or 12. Let's compute these probabilities.

```python
# All possible outcomes when rolling two dice
two_dice_outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
total_outcomes = len(two_dice_outcomes)

# Count outcomes for each sum
sums = [d1 + d2 for d1, d2 in two_dice_outcomes]

# Winning sums: 7 or 11
wins = sum(1 for s in sums if s in [7, 11])
p_win = wins / total_outcomes

# Losing sums: 2, 3, or 12
losses = sum(1 for s in sums if s in [2, 3, 12])
p_lose = losses / total_outcomes

# Everything else establishes a "point"
p_point = 1 - p_win - p_lose

print(f"Total two-dice outcomes: {total_outcomes}")
print(f"P(win on come-out: 7 or 11) = {wins}/{total_outcomes} = {p_win:.4f}")
print(f"P(lose on come-out: 2, 3, or 12) = {losses}/{total_outcomes} = {p_lose:.4f}")
print(f"P(establish a point) = {p_point:.4f}")
```

**Expected Output:**
```
Total two-dice outcomes: 36
P(win on come-out: 7 or 11) = 8/36 = 0.2222
P(lose on come-out: 2, 3, or 12) = 4/36 = 0.1111
P(establish a point) = 0.6667
```

### Finance Application: Portfolio Risk

If two investments each have a 5% chance of losing money, and they move independently, what is the probability that both lose money? What about at least one?

```python
p_loss_each = 0.05

# Both lose (independent)
p_both_lose = p_loss_each * p_loss_each
print(f"P(both investments lose) = {p_both_lose:.4f}")

# At least one loses (complement approach)
p_neither_loses = (1 - p_loss_each) ** 2
p_at_least_one_loses = 1 - p_neither_loses
print(f"P(at least one loses) = {p_at_least_one_loses:.4f}")

# WARNING: if investments are correlated (e.g., both tech stocks),
# you CANNOT simply multiply. Correlation increases the probability
# of both losing simultaneously.
print("\nNote: This assumes independence. Correlated assets")
print("have a HIGHER probability of both losing together.")
```

**Expected Output:**
```
P(both investments lose) = 0.0025
P(at least one loses) = 0.0975
Note: This assumes independence. Correlated assets
have a HIGHER probability of both losing together.
```

### Exercise 2.1: Combining Probabilities

**Task:** Solve the following probability problems:

(a) You draw one card from a standard deck. What is P(spade OR face card)?
(b) You flip a fair coin 5 times. What is P(at least one heads)?
(c) A slot machine has a 1/500 chance of a payout on any spin. You play 500 spins. What is P(at least one payout)?

**Hints:**
<details>
<summary>Hint 1: Part (a)</summary>
Spades and face cards overlap -- some face cards are spades. Use the inclusion-exclusion formula: P(A or B) = P(A) + P(B) - P(A and B). How many cards are both spades and face cards?
</details>

<details>
<summary>Hint 2: Parts (b) and (c)</summary>
For "at least one" problems, use the complement: P(at least one) = 1 - P(none). For (b), P(no heads in 5 flips) = (1/2)^5.
</details>

**Solution:**
<details>
<summary>Click to see solution</summary>

```python
# (a) P(spade or face card)
p_spade = 13 / 52
p_face = 12 / 52
p_spade_and_face = 3 / 52  # J, Q, K of spades
p_spade_or_face = p_spade + p_face - p_spade_and_face
print(f"(a) P(spade or face) = {13}/52 + {12}/52 - {3}/52 = {22}/52 = {p_spade_or_face:.4f}")

# (b) P(at least one heads in 5 flips)
p_no_heads = (1/2) ** 5
p_at_least_one_heads = 1 - p_no_heads
print(f"(b) P(at least one heads) = 1 - (1/2)^5 = 1 - {p_no_heads:.4f} = {p_at_least_one_heads:.4f}")

# (c) P(at least one payout in 500 spins)
p_no_payout_one_spin = 499 / 500
p_no_payout_500_spins = p_no_payout_one_spin ** 500
p_at_least_one_payout = 1 - p_no_payout_500_spins
print(f"(c) P(at least one payout) = 1 - (499/500)^500 = {p_at_least_one_payout:.4f}")
print(f"    Approximately {p_at_least_one_payout*100:.1f}% -- NOT 100%!")
```

**Expected Output:**
```
(a) P(spade or face) = 13/52 + 12/52 - 3/52 = 22/52 = 0.4231
(b) P(at least one heads) = 1 - (1/2)^5 = 1 - 0.0312 = 0.9688
(c) P(at least one payout) = 1 - (499/500)^500 = 0.6323
    Approximately 63.2% -- NOT 100%!
```

**Explanation:** Part (c) is a common surprise. Even with 500 spins at a 1/500 chance, you only have about a 63% chance of hitting at least one payout. Many people incorrectly assume it should be close to 100%.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Apply the addition rule and know when to subtract the overlap
- [ ] Apply the multiplication rule for independent events
- [ ] Use the complement rule for "at least one" problems
- [ ] Explain the difference between mutually exclusive and independent events

---

## Section 3: Conditional Probability

### What Is Conditional Probability?

Conditional probability answers the question: "Given that one thing is true, what is the probability of another?"

The notation is **P(A|B)**, read as "the probability of A given B." The formula is:

**P(A|B) = P(A and B) / P(B)**

Think of it this way: when you condition on B, you shrink the sample space to only the outcomes where B occurred. Then you count how many of those also include A.

### A Card Example

What is the probability that a card is a king, given that it is a face card?

```python
# P(king | face card)
# There are 12 face cards in a deck. 4 of them are kings.
p_king_and_face = 4 / 52   # all kings are face cards
p_face = 12 / 52

p_king_given_face = p_king_and_face / p_face
print(f"P(king | face card) = (4/52) / (12/52) = {p_king_given_face:.4f}")
print(f"This equals 4/12 = 1/3, which makes sense:")
print(f"of the 12 face cards, 4 are kings.")
```

**Expected Output:**
```
P(king | face card) = (4/52) / (12/52) = 0.3333
This equals 4/12 = 1/3, which makes sense:
of the 12 face cards, 4 are kings.
```

### Sports Betting Application

Conditional probability is the backbone of sports betting. Oddsmakers constantly update probabilities based on new information.

```python
# A basketball team's win probability depends on context
overall_win_rate = 0.55
win_rate_at_home = 0.70
win_rate_on_road = 0.40
win_rate_star_healthy = 0.60
win_rate_star_injured = 0.35

print("=== Conditional Win Probabilities ===")
print(f"P(win) = {overall_win_rate}")
print(f"P(win | home) = {win_rate_at_home}")
print(f"P(win | road) = {win_rate_on_road}")
print(f"P(win | star healthy) = {win_rate_star_healthy}")
print(f"P(win | star injured) = {win_rate_star_injured}")
print()
print("Each piece of information changes the probability.")
print("Smart bettors look for cases where the sportsbook")
print("hasn't fully adjusted for new information.")
```

**Expected Output:**
```
=== Conditional Win Probabilities ===
P(win) = 0.55
P(win | home) = 0.7
P(win | road) = 0.4
P(win | star healthy) = 0.6
P(win | star injured) = 0.35

Each piece of information changes the probability.
Smart bettors look for cases where the sportsbook
hasn't fully adjusted for new information.
```

### ML Application: The Foundation of Classification

In machine learning, classification is conditional probability. When a spam filter says "this email is 92% likely to be spam," it is computing:

**P(spam | email features)**

The Naive Bayes classifier computes this directly using Bayes' theorem (which we will cover next). Every time you use autocomplete, a recommendation system, or a medical diagnostic tool, conditional probability is doing the work.

### Independence Revisited

Two events are **independent** if knowing one tells you nothing about the other:

**P(A|B) = P(A)**

If this equality holds, A and B are independent. If it does not, they are dependent -- knowing B changes your estimate of A.

```python
# Testing independence
# Is "rolling a 6" independent of "flipping heads"?
p_six = 1/6
p_six_given_heads = 1/6  # coin result doesn't affect the die
print(f"P(six) = {p_six:.4f}")
print(f"P(six | heads) = {p_six_given_heads:.4f}")
print(f"Independent? {abs(p_six - p_six_given_heads) < 0.0001}")

print()

# Is "winning" independent of "playing at home"?
p_win = 0.55
p_win_given_home = 0.70
print(f"P(win) = {p_win}")
print(f"P(win | home) = {p_win_given_home}")
print(f"Independent? {abs(p_win - p_win_given_home) < 0.0001}")
```

**Expected Output:**
```
P(six) = 0.1667
P(six | heads) = 0.1667
Independent? True

P(win) = 0.55
P(win | home) = 0.7
Independent? False
```

### A Critical Warning: P(A|B) is NOT P(B|A)

This is one of the most common and most dangerous mistakes in probability.

- P(positive test | disease) = how often sick people test positive (sensitivity)
- P(disease | positive test) = how often positive testers are actually sick (predictive value)

These can be wildly different. A test that catches 99% of sick people can still produce mostly false positives if the disease is rare.

```python
# P(wet sidewalk | rain) is very high
p_wet_given_rain = 0.95

# P(rain | wet sidewalk) is much lower
# Sidewalks can be wet from sprinklers, washing, etc.
p_rain_given_wet = 0.40

print(f"P(wet sidewalk | rain) = {p_wet_given_rain}")
print(f"P(rain | wet sidewalk) = {p_rain_given_wet}")
print("These are NOT the same!")
```

**Expected Output:**
```
P(wet sidewalk | rain) = 0.95
P(rain | wet sidewalk) = 0.4
These are NOT the same!
```

### Exercise 3.1: Conditional Probability

**Task:** Use the following data about a baseball player's hitting performance:
- Overall batting average: .280 (P(hit) = 0.280)
- Batting average vs left-handed pitchers: .320
- Batting average vs right-handed pitchers: .260
- Batting average with runners in scoring position: .310
- 35% of at-bats are against left-handed pitchers

(a) What is P(hit | left-handed pitcher)?
(b) Are "getting a hit" and "facing a left-handed pitcher" independent? How do you know?
(c) What is P(left-handed pitcher | hit)? Use the relationship P(A and B) = P(A|B) x P(B).

**Hints:**
<details>
<summary>Hint 1: Part (b)</summary>
Two events are independent if P(A|B) = P(A). Compare P(hit | lefty) to P(hit).
</details>

<details>
<summary>Hint 2: Part (c)</summary>
P(lefty | hit) = P(hit and lefty) / P(hit). You can compute P(hit and lefty) = P(hit | lefty) x P(lefty).
</details>

**Solution:**
<details>
<summary>Click to see solution</summary>

```python
p_hit = 0.280
p_hit_given_lefty = 0.320
p_lefty = 0.35

# (a) Directly stated
print(f"(a) P(hit | lefty) = {p_hit_given_lefty}")

# (b) Independence check
independent = abs(p_hit_given_lefty - p_hit) < 0.001
print(f"\n(b) P(hit) = {p_hit}, P(hit | lefty) = {p_hit_given_lefty}")
print(f"    Independent? {independent}")
print(f"    No -- the player hits better against lefties.")

# (c) P(lefty | hit)
p_hit_and_lefty = p_hit_given_lefty * p_lefty
p_lefty_given_hit = p_hit_and_lefty / p_hit
print(f"\n(c) P(hit and lefty) = {p_hit_given_lefty} x {p_lefty} = {p_hit_and_lefty:.4f}")
print(f"    P(lefty | hit) = {p_hit_and_lefty:.4f} / {p_hit} = {p_lefty_given_hit:.4f}")
print(f"    About {p_lefty_given_hit*100:.1f}% of this player's hits come against lefties,")
print(f"    even though only {p_lefty*100:.0f}% of at-bats are against lefties.")
```

**Expected Output:**
```
(a) P(hit | lefty) = 0.32

(b) P(hit) = 0.28, P(hit | lefty) = 0.32
    Independent? False
    No -- the player hits better against lefties.

(c) P(hit and lefty) = 0.32 x 0.35 = 0.1120
    P(lefty | hit) = 0.1120 / 0.28 = 0.4000
    About 40.0% of this player's hits come against lefties,
    even though only 35% of at-bats are against lefties.
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Compute P(A|B) using the formula
- [ ] Explain what "conditioning" does to the sample space
- [ ] Test whether two events are independent
- [ ] Explain why P(A|B) and P(B|A) are different

---

## Section 4: Bayes' Theorem

### The Problem Bayes Solves

You often know one conditional probability but need the reverse. You know P(evidence | hypothesis) but need P(hypothesis | evidence).

- A medical test: You know P(positive test | disease) = 0.99, but you need P(disease | positive test).
- A spam filter: You know P(contains "free" | spam) = 0.80, but you need P(spam | contains "free").
- A sports bet: You know P(strong start | good team) but you need P(good team | strong start).

### The Formula

**P(A|B) = P(B|A) x P(A) / P(B)**

The components have intuitive names:
- **Prior**: P(A) -- your initial belief before seeing evidence
- **Likelihood**: P(B|A) -- how likely is the evidence if your hypothesis is true
- **Evidence**: P(B) -- how likely is the evidence overall
- **Posterior**: P(A|B) -- your updated belief after seeing evidence

### The Classic Medical Testing Example

A disease affects 1% of the population. A test for the disease has:
- Sensitivity: 99% (P(positive | disease) = 0.99)
- False positive rate: 5% (P(positive | no disease) = 0.05)

You test positive. What is the probability you actually have the disease?

```python
# Medical testing with Bayes' theorem
prior = 0.01               # P(disease) -- 1% prevalence
sensitivity = 0.99          # P(positive | disease)
false_positive_rate = 0.05  # P(positive | no disease)

# P(positive) using the law of total probability
p_positive = sensitivity * prior + false_positive_rate * (1 - prior)

# Bayes' theorem
posterior = (sensitivity * prior) / p_positive

print("=== Medical Test: Bayes' Theorem ===")
print(f"Prior P(disease) = {prior}")
print(f"P(positive | disease) = {sensitivity}")
print(f"P(positive | no disease) = {false_positive_rate}")
print(f"P(positive) = {p_positive:.4f}")
print(f"P(disease | positive) = {posterior:.4f}")
print(f"\nEven with a 99% sensitive test, a positive result")
print(f"only means a {posterior*100:.1f}% chance of having the disease!")
print(f"This is because the disease is rare (1% prevalence).")
print(f"Most positives are false positives.")
```

**Expected Output:**
```
=== Medical Test: Bayes' Theorem ===
Prior P(disease) = 0.01
P(positive | disease) = 0.99
P(positive | no disease) = 0.05
P(positive) = 0.0594
P(disease | positive) = 0.1667

Even with a 99% sensitive test, a positive result
only means a 16.7% chance of having the disease!
This is because the disease is rare (1% prevalence).
Most positives are false positives.
```

### Intuition: The Natural Frequency Approach

If formulas feel abstract, think in terms of concrete numbers.

```python
# Natural frequency approach: imagine 10,000 people
population = 10000
with_disease = int(population * 0.01)        # 100 people
without_disease = population - with_disease   # 9,900 people

# Testing
true_positives = int(with_disease * 0.99)           # 99
false_positives = int(without_disease * 0.05)        # 495
total_positives = true_positives + false_positives   # 594

p_disease_given_positive = true_positives / total_positives

print("=== Natural Frequency Approach ===")
print(f"Out of {population:,} people:")
print(f"  {with_disease} have the disease")
print(f"  {without_disease:,} do not")
print(f"\nAfter testing:")
print(f"  {true_positives} true positives (sick + positive)")
print(f"  {false_positives} false positives (healthy + positive)")
print(f"  {total_positives} total positives")
print(f"\nP(disease | positive) = {true_positives}/{total_positives} = {p_disease_given_positive:.4f}")
```

**Expected Output:**
```
=== Natural Frequency Approach ===
Out of 10,000 people:
  100 have the disease
  9,900 do not

After testing:
  99 true positives (sick + positive)
  495 false positives (healthy + positive)
  594 total positives

P(disease | positive) = 99/594 = 0.1667
```

### Spam Filtering with Bayes

This is exactly how Naive Bayes spam filters work.

```python
# Spam filter example
p_spam = 0.30                   # 30% of emails are spam
p_free_given_spam = 0.80        # 80% of spam contains "free"
p_free_given_not_spam = 0.10    # 10% of legit emails contain "free"

# P("free" appears)
p_free = p_free_given_spam * p_spam + p_free_given_not_spam * (1 - p_spam)

# P(spam | contains "free")
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print("=== Spam Filter: Bayes' Theorem ===")
print(f"Prior P(spam) = {p_spam}")
print(f"P('free' | spam) = {p_free_given_spam}")
print(f"P('free' | not spam) = {p_free_given_not_spam}")
print(f"P('free') = {p_free:.4f}")
print(f"P(spam | 'free') = {p_spam_given_free:.4f}")
print(f"\nAn email containing 'free' has a {p_spam_given_free*100:.1f}% chance of being spam.")
```

**Expected Output:**
```
=== Spam Filter: Bayes' Theorem ===
Prior P(spam) = 0.3
P('free' | spam) = 0.8
P('free' | not spam) = 0.1
P('free') = 0.31
P(spam | 'free') = 0.7742

An email containing 'free' has a 77.4% chance of being spam.
```

### Updating Sports Predictions

Bayes' theorem is how smart bettors update their beliefs as a season unfolds.

```python
# Evaluating a team's true strength
# Prior: before the season, you think there's a 30% chance this team is "elite"
p_elite = 0.30
p_average = 0.70

# Evidence: the team starts 9-1 (wins 9 of first 10 games)
# P(9-1 start | elite team) -- elite teams do this about 25% of the time
p_hot_start_given_elite = 0.25
# P(9-1 start | average team) -- average teams do this about 3% of the time
p_hot_start_given_average = 0.03

# P(9-1 start)
p_hot_start = (p_hot_start_given_elite * p_elite +
               p_hot_start_given_average * p_average)

# Posterior
p_elite_given_hot_start = (p_hot_start_given_elite * p_elite) / p_hot_start

print("=== Updating Team Assessment ===")
print(f"Prior P(elite) = {p_elite}")
print(f"P(9-1 start | elite) = {p_hot_start_given_elite}")
print(f"P(9-1 start | average) = {p_hot_start_given_average}")
print(f"P(9-1 start) = {p_hot_start:.4f}")
print(f"Posterior P(elite | 9-1 start) = {p_elite_given_hot_start:.4f}")
print(f"\nYour belief that the team is elite went from")
print(f"{p_elite*100:.0f}% to {p_elite_given_hot_start*100:.1f}% after seeing their strong start.")
```

**Expected Output:**
```
=== Updating Team Assessment ===
Prior P(elite) = 0.3
P(9-1 start | elite) = 0.25
P(9-1 start | average) = 0.03
P(9-1 start) = 0.096
Posterior P(elite | 9-1 start) = 0.7812

Your belief that the team is elite went from
30% to 78.1% after seeing their strong start.
```

### Base Rate Neglect

The most common mistake with Bayes' theorem is ignoring the prior (base rate). Humans tend to focus on the evidence and forget how rare or common the hypothesis is.

When a disease affects 1 in 10,000 people and you test positive, your brain screams "I have the disease!" But the math says otherwise -- the vast majority of positives will be false positives from the 9,999 healthy people.

This same bias affects sports bettors ("this team won 8 straight, they must be great!") and investors ("this stock has gone up 5 days in a row, it must keep going!"). Always ask: what was the base rate before the evidence?

### Exercise 4.1: Applying Bayes' Theorem

**Task:** A factory has two machines that produce widgets.
- Machine A produces 60% of all widgets, with a 2% defect rate
- Machine B produces 40% of all widgets, with a 5% defect rate

A widget is randomly selected and found to be defective. What is the probability it came from Machine A?

**Hints:**
<details>
<summary>Hint 1: Identify the components</summary>
Prior: P(Machine A) = 0.60. Likelihood: P(defective | Machine A) = 0.02. You need P(Machine A | defective).
</details>

<details>
<summary>Hint 2: Compute P(defective)</summary>
P(defective) = P(defective | A) x P(A) + P(defective | B) x P(B) = 0.02 x 0.60 + 0.05 x 0.40
</details>

**Solution:**
<details>
<summary>Click to see solution</summary>

```python
# Factory machine problem
p_A = 0.60       # P(Machine A)
p_B = 0.40       # P(Machine B)
p_def_A = 0.02   # P(defective | Machine A)
p_def_B = 0.05   # P(defective | Machine B)

# P(defective)
p_def = p_def_A * p_A + p_def_B * p_B

# P(Machine A | defective)
p_A_given_def = (p_def_A * p_A) / p_def

print(f"P(Machine A) = {p_A}")
print(f"P(defective | A) = {p_def_A}")
print(f"P(defective | B) = {p_def_B}")
print(f"P(defective) = {p_def:.4f}")
print(f"P(Machine A | defective) = {p_A_given_def:.4f}")
print(f"\nEven though Machine A produces 60% of widgets,")
print(f"only {p_A_given_def*100:.1f}% of defective widgets come from Machine A.")
print(f"Machine B's higher defect rate means it produces")
print(f"a disproportionate share of defects.")
```

**Expected Output:**
```
P(Machine A) = 0.6
P(defective | A) = 0.02
P(defective | B) = 0.05
P(defective) = 0.0320
P(Machine A | defective) = 0.3750

Even though Machine A produces 60% of widgets,
only 37.5% of defective widgets come from Machine A.
Machine B's higher defect rate means it produces
a disproportionate share of defects.
```

**Explanation:** Despite Machine A producing more widgets overall, Machine B's higher defect rate (5% vs 2%) means that most defective widgets come from Machine B. This is a practical application of base rate reasoning.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] State Bayes' theorem and identify prior, likelihood, evidence, and posterior
- [ ] Apply Bayes' theorem to compute a posterior probability
- [ ] Explain base rate neglect and why it is dangerous
- [ ] Use the natural frequency approach as an alternative to the formula

---

## Section 5: Expected Value

### What Is Expected Value?

Expected value (EV) is the average outcome you would get if you repeated a random process many, many times. It is calculated by multiplying each possible outcome by its probability, then summing:

**EV = sum of (outcome x probability)**

Expected value is the single most important concept for making rational decisions under uncertainty. It tells you whether a bet, investment, or decision is favorable in the long run.

### A Simple Example

```python
# Simple coin flip bet: heads you win $10, tails you lose $6
p_heads = 0.5
win_amount = 10
lose_amount = -6

ev = p_heads * win_amount + (1 - p_heads) * lose_amount
print(f"EV = {p_heads} x ${win_amount} + {1-p_heads} x ${lose_amount}")
print(f"EV = ${p_heads * win_amount:.2f} + ${(1-p_heads) * lose_amount:.2f}")
print(f"EV = ${ev:.2f} per flip")
print(f"\nThis is a +EV bet. Over 1000 flips, you'd expect")
print(f"to profit about ${ev * 1000:.0f}.")
```

**Expected Output:**
```
EV = 0.5 x $10 + 0.5 x $-6
EV = $5.00 + $-3.00
EV = $2.00 per flip

This is a +EV bet. Over 1000 flips, you'd expect
to profit about $2000.
```

### Why Casinos Always Win

Every casino game has negative expected value for the player. This is the house edge -- the mathematical guarantee that, over millions of bets, the casino profits.

```python
def casino_ev(game_name, bet, win_payout, p_win):
    """Calculate expected value for a casino bet."""
    p_lose = 1 - p_win
    ev = p_win * win_payout + p_lose * (-bet)
    house_edge = -ev / bet * 100
    return ev, house_edge

# American Roulette: bet on a single number
ev, edge = casino_ev("Single Number", bet=1, win_payout=35, p_win=1/38)
print(f"=== American Roulette (single number) ===")
print(f"  EV per $1 bet: ${ev:.4f}")
print(f"  House edge: {edge:.2f}%")

# Roulette: bet on red
ev, edge = casino_ev("Red", bet=1, win_payout=1, p_win=18/38)
print(f"\n=== American Roulette (red/black) ===")
print(f"  EV per $1 bet: ${ev:.4f}")
print(f"  House edge: {edge:.2f}%")

# Craps: pass line bet
ev, edge = casino_ev("Pass Line", bet=1, win_payout=1, p_win=244/495)
print(f"\n=== Craps (pass line) ===")
print(f"  EV per $1 bet: ${ev:.4f}")
print(f"  House edge: {edge:.2f}%")

# Blackjack: with basic strategy
ev, edge = casino_ev("Blackjack", bet=1, win_payout=1, p_win=0.4975)
print(f"\n=== Blackjack (basic strategy, approximate) ===")
print(f"  EV per $1 bet: ${ev:.4f}")
print(f"  House edge: {edge:.2f}%")

print("\n--- Every game is negative EV for the player. ---")
print("The house always wins in the long run.")
```

**Expected Output:**
```
=== American Roulette (single number) ===
  EV per $1 bet: $-0.0526
  House edge: 5.26%

=== American Roulette (red/black) ===
  EV per $1 bet: $-0.0526
  House edge: 5.26%

=== Craps (pass line) ===
  EV per $1 bet: $-0.0141
  House edge: 1.41%

=== Blackjack (basic strategy, approximate) ===
  EV per $1 bet: $-0.0050
  House edge: 0.50%

--- Every game is negative EV for the player. ---
The house always wins in the long run.
```

Notice that blackjack has the lowest house edge, which is why card counters target it -- with enough skill, they can flip the EV to slightly positive.

### Sports Betting: Finding +EV Bets

In sports betting, you can sometimes find bets where your estimated probability of winning exceeds the implied probability from the odds. These are positive expected value (+EV) bets.

```python
def american_odds_to_probability(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def sports_bet_ev(stake, american_odds, true_probability):
    """Calculate EV of a sports bet."""
    implied_prob = american_odds_to_probability(american_odds)
    if american_odds > 0:
        profit = stake * (american_odds / 100)
    else:
        profit = stake * (100 / abs(american_odds))

    ev = true_probability * profit + (1 - true_probability) * (-stake)
    return ev, implied_prob

# Example 1: Underdog at +200, you think they have 40% chance
ev, implied = sports_bet_ev(100, +200, 0.40)
print(f"=== Bet 1: Underdog at +200 ===")
print(f"  Implied probability: {implied*100:.1f}%")
print(f"  Your estimate: 40.0%")
print(f"  EV per $100 bet: ${ev:.2f}")
print(f"  {'POSITIVE EV -- good bet!' if ev > 0 else 'Negative EV -- bad bet.'}")

# Example 2: Favorite at -150, you think they have 55% chance
ev, implied = sports_bet_ev(100, -150, 0.55)
print(f"\n=== Bet 2: Favorite at -150 ===")
print(f"  Implied probability: {implied*100:.1f}%")
print(f"  Your estimate: 55.0%")
print(f"  EV per $100 bet: ${ev:.2f}")
print(f"  {'POSITIVE EV -- good bet!' if ev > 0 else 'Negative EV -- bad bet.'}")

# Example 3: Even money (-110), you think 53% chance
ev, implied = sports_bet_ev(110, -110, 0.53)
print(f"\n=== Bet 3: Standard -110 line ===")
print(f"  Implied probability: {implied*100:.1f}%")
print(f"  Your estimate: 53.0%")
print(f"  EV per $110 bet: ${ev:.2f}")
print(f"  {'POSITIVE EV -- good bet!' if ev > 0 else 'Negative EV -- bad bet.'}")
```

**Expected Output:**
```
=== Bet 1: Underdog at +200 ===
  Implied probability: 33.3%
  Your estimate: 40.0%
  EV per $100 bet: $20.00
  POSITIVE EV -- good bet!

=== Bet 2: Favorite at -150 ===
  Implied probability: 60.0%
  Your estimate: 55.0%
  EV per $100 bet: $-8.33
  Negative EV -- bad bet.

=== Bet 3: Standard -110 line ===
  Implied probability: 52.4%
  Your estimate: 53.0%
  EV per $110 bet: $-0.30
  Negative EV -- bad bet.
```

The key insight: a bet is only +EV if your estimated win probability exceeds the implied probability from the odds. The sportsbook's "vig" (juice) means you typically need to be right more than 52.4% of the time on standard -110 bets just to break even.

### Finance: Expected Return on Investment

Expected value applies directly to investment decisions.

```python
# Investment scenario: tech startup
# 30% chance of 3x return, 40% chance of 1.2x return, 30% chance of total loss
outcomes = [3.0, 1.2, 0.0]     # multipliers on investment
probs = [0.30, 0.40, 0.30]
investment = 10000

ev_multiplier = sum(o * p for o, p in zip(outcomes, probs))
ev_dollar = investment * ev_multiplier
expected_profit = ev_dollar - investment

print("=== Tech Startup Investment ===")
print(f"Investment: ${investment:,}")
for o, p in zip(outcomes, probs):
    print(f"  {p*100:.0f}% chance of {o}x return (${investment * o:,.0f})")
print(f"\nExpected value multiplier: {ev_multiplier:.2f}x")
print(f"Expected value: ${ev_dollar:,.0f}")
print(f"Expected profit: ${expected_profit:,.0f}")
print(f"Expected return: {(ev_multiplier - 1) * 100:.0f}%")

print("\n=== Comparison: Safe Bond ===")
bond_return = 1.05  # 5% guaranteed return
print(f"Bond return: {(bond_return-1)*100:.0f}% guaranteed")
print(f"Bond EV: ${investment * bond_return:,.0f}")
print(f"\nThe startup has higher EV, but also higher variance.")
print(f"Risk tolerance determines which is the better choice.")
```

**Expected Output:**
```
=== Tech Startup Investment ===
Investment: $10,000
  30% chance of 3.0x return ($30,000)
  40% chance of 1.2x return ($12,000)
  30% chance of 0.0x return ($0)

Expected value multiplier: 1.38x
Expected value: $13,800
Expected profit: $3,800
Expected return: 38%

=== Comparison: Safe Bond ===
Bond return: 5% guaranteed
Bond EV: $10,500

The startup has higher EV, but also higher variance.
Risk tolerance determines which is the better choice.
```

### EV is Not Everything: Variance Matters

Two bets can have the same expected value but feel completely different. This is because of variance -- how spread out the outcomes are.

```python
# Same EV, very different variance
print("=== Same EV, Different Variance ===\n")

# Bet A: 50% chance of winning $100, 50% chance of losing $90
ev_a = 0.5 * 100 + 0.5 * (-90)
print(f"Bet A: 50% win $100, 50% lose $90")
print(f"  EV = ${ev_a:.2f}")

# Bet B: 1% chance of winning $5,050, 99% chance of losing $45.96
ev_b = 0.01 * 5050 + 0.99 * (-45.96)
print(f"\nBet B: 1% win $5,050, 99% lose $45.96")
print(f"  EV = ${ev_b:.2f}")

print(f"\nBoth have EV = ~${ev_a:.2f}, but Bet B is much riskier.")
print("You could lose $45.96 ninety-nine times before winning once.")
print("Positive EV does NOT mean guaranteed profit in the short run.")
```

**Expected Output:**
```
=== Same EV, Different Variance ===

Bet A: 50% win $100, 50% lose $90
  EV = $5.00

Bet B: 1% win $5,050, 99% lose $45.96
  EV = $5.04

Both have EV = ~$5.00, but Bet B is much riskier.
You could lose $45.96 ninety-nine times before winning once.
Positive EV does NOT mean guaranteed profit in the short run.
```

### Exercise 5.1: Expected Value Calculations

**Task:** Calculate the expected value for each scenario and determine whether it is favorable:

(a) You play a game where you roll a six-sided die. You win $10 if you roll a 6, and lose $2 for any other result. What is the EV?

(b) A lottery ticket costs $5. The prizes are: $10,000 (probability 1/50,000), $100 (probability 1/500), $10 (probability 1/50). What is the EV?

(c) You are offered insurance on a $1,000 electronic device. The insurance costs $120. There is a 5% chance the device breaks and needs full replacement. What is the EV of buying insurance?

**Hints:**
<details>
<summary>Hint 1: Part (a)</summary>
List each outcome and its probability. P(roll 6) = 1/6, P(not 6) = 5/6. Multiply each outcome by its probability and sum.
</details>

<details>
<summary>Hint 2: Part (c)</summary>
Think about it from your perspective: if you buy insurance, you pay $120 for sure. If the device breaks (5% chance), you save $1,000. If it does not break (95%), you spent $120 for nothing. EV = 0.05 x ($1000 - $120) + 0.95 x (-$120)? Or more simply: expected payout - cost.
</details>

**Solution:**
<details>
<summary>Click to see solution</summary>

```python
# (a) Die roll game
ev_a = (1/6) * 10 + (5/6) * (-2)
print(f"(a) Die roll game:")
print(f"    EV = (1/6)(10) + (5/6)(-2) = {ev_a:.4f}")
print(f"    {'Favorable!' if ev_a > 0 else 'Unfavorable.'}")

# (b) Lottery ticket
cost = 5
ev_prizes = (1/50000) * 10000 + (1/500) * 100 + (1/50) * 10
ev_b = ev_prizes - cost
print(f"\n(b) Lottery ticket:")
print(f"    Expected prize = ${ev_prizes:.4f}")
print(f"    Cost = ${cost}")
print(f"    EV = ${ev_b:.4f}")
print(f"    {'Favorable!' if ev_b > 0 else 'Unfavorable.'}")

# (c) Insurance
insurance_cost = 120
device_cost = 1000
p_break = 0.05
# Expected savings from insurance = P(break) x device_cost
expected_benefit = p_break * device_cost
ev_c = expected_benefit - insurance_cost
print(f"\n(c) Device insurance:")
print(f"    Expected benefit = {p_break} x ${device_cost} = ${expected_benefit:.2f}")
print(f"    Insurance cost = ${insurance_cost}")
print(f"    EV of buying insurance = ${ev_c:.2f}")
print(f"    {'Favorable!' if ev_c > 0 else 'Unfavorable.'}")
print(f"    (But EV alone doesn't tell the whole story --")
print(f"    if $1000 would be devastating, insurance has value beyond EV.)")
```

**Expected Output:**
```
(a) Die roll game:
    EV = (1/6)(10) + (5/6)(-2) = 0.0000
    Unfavorable.

(b) Lottery ticket:
    Expected prize = $0.6000
    Cost = $5
    EV = $-4.4000
    Unfavorable.

(c) Device insurance:
    Expected benefit = 0.05 x $1000 = $50.00
    Insurance cost = $120
    EV of buying insurance = $-70.00
    Unfavorable.
    (But EV alone doesn't tell the whole story --
    if $1000 would be devastating, insurance has value beyond EV.)
```

**Explanation:** The die game is exactly fair (EV = 0). The lottery has terrible EV -- you expect to lose $4.40 per ticket. Insurance is also negative EV on average (that is how insurance companies make money), but it can still be rational if you cannot afford the loss.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Compute expected value for any scenario with known outcomes and probabilities
- [ ] Explain why casino games all have negative EV for the player
- [ ] Determine whether a sports bet has positive or negative expected value
- [ ] Explain why variance matters even when EV is positive

---

## Common Pitfalls

### Pitfall 1: The Gambler's Fallacy

**The Problem:** Believing that past independent events influence future ones. "Red has come up 8 times in a row, so black is due!"

**Why it happens:** Our brains are pattern-seeking machines. We expect randomness to "look random" with frequent alternation, but true randomness includes long streaks.

**How to avoid it:** Always ask: "Does the system have memory?" Dice, coins, and roulette wheels do not. Each trial is independent. Card draws without replacement DO depend on previous draws -- that is different.

### Pitfall 2: Base Rate Neglect

**The Problem:** Ignoring how common or rare a condition is when interpreting evidence. "The test is 99% accurate, so a positive result means I'm 99% likely to have the disease."

**Why it happens:** We focus on the evidence (the test result) and ignore the prior (how rare the disease is).

**How to avoid it:** Always start with Bayes' theorem. Ask: "How common is this condition in the population?" before interpreting any test result.

### Pitfall 3: Confusing P(A|B) with P(B|A)

**The Problem:** Assuming that P(guilty | evidence) is the same as P(evidence | guilty). This is called the "prosecutor's fallacy" in legal contexts.

**Why it happens:** The two conditional probabilities sound similar in everyday language.

**How to avoid it:** Always write out the conditional probability notation explicitly. Ask: "What is given, and what am I computing?"

### Best Practices

- Always start by defining the sample space before computing any probability
- When you see "at least one," think complement rule
- When combining events, ask: "Am I computing OR or AND?" Then: "Are these mutually exclusive / independent?"
- For Bayes' theorem, use the natural frequency approach (imagine 10,000 people) to build intuition
- When evaluating bets or decisions, always compute EV before committing
- Remember that EV is about the long run -- short-run results can deviate significantly

---

## Practice Project: Probability and EV Calculator

### Project Description

Build a Python tool that computes probabilities and expected values for common gambling and decision-making scenarios. This project integrates every concept from this route.

This project will help you:
- Apply sample space enumeration and probability rules
- Implement Bayes' theorem as a reusable function
- Compute expected values for real-world scenarios
- Evaluate whether gambling bets are favorable

### Requirements

Build a Python program that includes:
- A function to compute basic probabilities from a sample space
- A function to apply Bayes' theorem
- A function to compute expected value
- Analysis of at least two casino games (compute their house edge)
- Evaluation of a sports bet for positive expected value

### Getting Started

**Step 1: Create your project file**
```bash
touch probability_calculator.py
```

**Step 2: Plan your approach**
Think about:
1. What inputs does each function need?
2. How will you validate inputs (e.g., probabilities must sum to 1)?
3. What real-world scenarios will you analyze?

**Step 3: Start with the expected value function**, since it is the simplest and most widely useful.

### Starter Code

```python
import numpy as np

def expected_value(outcomes, probabilities):
    """
    Compute expected value.

    Args:
        outcomes: list of numerical outcomes
        probabilities: list of probabilities (must sum to 1)

    Returns:
        The expected value
    """
    # Your code here
    pass

def bayes_theorem(prior, likelihood_if_true, likelihood_if_false):
    """
    Apply Bayes' theorem.

    Args:
        prior: P(hypothesis)
        likelihood_if_true: P(evidence | hypothesis is true)
        likelihood_if_false: P(evidence | hypothesis is false)

    Returns:
        Posterior probability P(hypothesis | evidence)
    """
    # Your code here
    pass

def evaluate_bet(stake, payout_if_win, p_win):
    """
    Evaluate whether a bet has positive expected value.

    Args:
        stake: amount risked
        payout_if_win: amount received on a win (net profit, not including stake)
        p_win: probability of winning

    Returns:
        Tuple of (expected_value, is_favorable)
    """
    # Your code here
    pass

# === Casino Game Analysis ===
# Analyze roulette, craps, or blackjack

# === Sports Bet Evaluation ===
# Evaluate a specific sports bet
```

### Hints and Tips

<details>
<summary>If you are not sure how to start</summary>
Begin with the expected_value function. It just needs to multiply each outcome by its probability and sum: return sum(o * p for o, p in zip(outcomes, probabilities)). Add a check that probabilities sum to 1.
</details>

<details>
<summary>If you are stuck on Bayes' theorem</summary>
The function needs to compute P(evidence) first using the law of total probability: evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior). Then return (likelihood_if_true * prior) / evidence.
</details>

<details>
<summary>If you are stuck on the casino analysis</summary>
For American roulette betting on red: stake = 1, payout = 1, p_win = 18/38. Call your evaluate_bet function with these values. The EV should be about -$0.053.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np

def expected_value(outcomes, probabilities):
    """Compute expected value given outcomes and their probabilities."""
    outcomes = np.array(outcomes, dtype=float)
    probabilities = np.array(probabilities, dtype=float)
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError(f"Probabilities must sum to 1, got {sum(probabilities):.6f}")
    return float(np.sum(outcomes * probabilities))

def bayes_theorem(prior, likelihood_if_true, likelihood_if_false):
    """Apply Bayes' theorem to compute posterior probability."""
    evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior)
    posterior = (likelihood_if_true * prior) / evidence
    return posterior

def evaluate_bet(stake, payout_if_win, p_win):
    """Evaluate whether a bet has positive expected value."""
    p_lose = 1 - p_win
    ev = p_win * payout_if_win + p_lose * (-stake)
    return ev, ev > 0

def analyze_roulette():
    """Analyze common roulette bets."""
    print("=" * 50)
    print("AMERICAN ROULETTE ANALYSIS")
    print("=" * 50)

    bets = [
        ("Single number (35:1)", 1, 35, 1/38),
        ("Red/Black (1:1)", 1, 1, 18/38),
        ("Dozen (2:1)", 1, 2, 12/38),
        ("Column (2:1)", 1, 2, 12/38),
    ]

    for name, stake, payout, p_win in bets:
        ev, favorable = evaluate_bet(stake, payout, p_win)
        house_edge = -ev / stake * 100
        print(f"\n  {name}:")
        print(f"    P(win) = {p_win:.4f}")
        print(f"    EV per ${stake} bet: ${ev:.4f}")
        print(f"    House edge: {house_edge:.2f}%")

def analyze_sports_bet():
    """Evaluate specific sports betting scenarios."""
    print("\n" + "=" * 50)
    print("SPORTS BET EVALUATION")
    print("=" * 50)

    scenarios = [
        ("Underdog +250, you estimate 35%", 100, 250, 0.35),
        ("Underdog +250, you estimate 45%", 100, 250, 0.45),
        ("Favorite -200, you estimate 70%", 200, 100, 0.70),
        ("Even money -110, you estimate 55%", 110, 100, 0.55),
    ]

    for name, stake, payout, p_win in scenarios:
        ev, favorable = evaluate_bet(stake, payout, p_win)
        implied_prob = stake / (stake + payout) if payout > 0 else payout / (payout + stake)
        print(f"\n  {name}:")
        print(f"    Stake: ${stake}, Payout: ${payout}")
        print(f"    Your P(win): {p_win*100:.0f}%")
        print(f"    EV: ${ev:.2f}")
        print(f"    Verdict: {'POSITIVE EV' if favorable else 'Negative EV'}")

def demo_bayes():
    """Demonstrate Bayes' theorem with practical examples."""
    print("\n" + "=" * 50)
    print("BAYES' THEOREM DEMONSTRATIONS")
    print("=" * 50)

    # Medical test
    posterior = bayes_theorem(prior=0.01, likelihood_if_true=0.99,
                             likelihood_if_false=0.05)
    print(f"\n  Medical test (1% prevalence, 99% sensitivity, 5% FPR):")
    print(f"    P(disease | positive test) = {posterior:.4f}")

    # Spam filter
    posterior = bayes_theorem(prior=0.30, likelihood_if_true=0.80,
                             likelihood_if_false=0.10)
    print(f"\n  Spam filter (30% spam rate, 'free' in 80% spam / 10% legit):")
    print(f"    P(spam | contains 'free') = {posterior:.4f}")

    # Sports prediction
    posterior = bayes_theorem(prior=0.30, likelihood_if_true=0.25,
                             likelihood_if_false=0.03)
    print(f"\n  Team evaluation (30% prior elite, 9-1 start):")
    print(f"    P(elite | 9-1 start) = {posterior:.4f}")

if __name__ == "__main__":
    # Run all analyses
    analyze_roulette()
    analyze_sports_bet()
    demo_bayes()

    # Custom EV calculation
    print("\n" + "=" * 50)
    print("CUSTOM EV CALCULATION")
    print("=" * 50)
    # Dice game: roll a die, win $amount_shown, costs $4 to play
    outcomes = [1 - 4, 2 - 4, 3 - 4, 4 - 4, 5 - 4, 6 - 4]  # net outcomes
    probs = [1/6] * 6
    ev = expected_value(outcomes, probs)
    print(f"\n  Dice game (win face value, costs $4):")
    print(f"    EV = ${ev:.4f}")
    print(f"    {'Favorable' if ev > 0 else 'Unfavorable'}")
```

**Key points in this solution:**
- Input validation (probabilities must sum to 1)
- Clean separation of concerns (each function does one thing)
- Practical analysis of real gambling scenarios
- Both casino games and sports bets are covered
</details>

### Extending the Project

If you want to go further, try:
- Add a function to simulate N trials and compare actual results to EV predictions
- Implement a Kelly Criterion calculator (optimal bet sizing given edge and bankroll)
- Build an interactive version that accepts user input for custom bet evaluation
- Add European roulette (37 slots) and compare its house edge to American roulette

---

## Summary

Congratulations on completing Probability Fundamentals. Let's review what you learned.

### Key Takeaways

- **Probability** measures uncertainty on a 0-to-1 scale, starting with sample spaces
- **Three rules** (addition, multiplication, complement) let you combine probabilities for complex events
- **Conditional probability** captures how new information changes the odds
- **Bayes' theorem** provides a systematic way to update beliefs with new evidence
- **Expected value** tells you the long-run average, making it the foundation of rational decision-making

### Skills You've Gained

You can now:
- Enumerate sample spaces and compute probabilities for card games, dice, roulette, and more
- Combine probabilities using the addition, multiplication, and complement rules
- Compute conditional probabilities and test for independence
- Apply Bayes' theorem to update predictions given new evidence
- Calculate expected values to evaluate bets, investments, and decisions

### Self-Assessment

Take a moment to reflect:
- What concepts do you feel most confident about?
- What areas might you want to revisit?
- Can you think of a real situation in your life where expected value could guide a decision?

---

## Next Steps

### Continue Learning

Ready for more? Here are your next options:

**Build on this topic:**
- [Probability Distributions](/routes/probability-distributions/map.md) -- Learn about binomial, normal, Poisson, and other distributions that model real-world randomness
- [Bayesian Statistics](/routes/bayesian-statistics/map.md) -- Go deeper into Bayesian inference, prior selection, and Bayesian modeling

**Explore related routes:**
- [Stats Fundamentals](/routes/stats-fundamentals/map.md) -- If you have not already, build your foundation in descriptive and inferential statistics

### Practice More

- Compute the EV for every casino game you encounter
- When you see sports odds, practice converting them to implied probabilities
- Apply Bayes' theorem to real-world news: when you learn new information, ask "how should this update my beliefs?"

---

## Appendix

### Probability Rules Cheat Sheet

| Rule | Formula | When to Use |
|------|---------|------------|
| Basic probability | P(E) = favorable / total | When all outcomes are equally likely |
| Addition (OR) | P(A or B) = P(A) + P(B) - P(A and B) | When you want "either A or B or both" |
| Addition (mutually exclusive) | P(A or B) = P(A) + P(B) | When A and B cannot both happen |
| Multiplication (AND, independent) | P(A and B) = P(A) x P(B) | When A and B do not affect each other |
| Multiplication (AND, dependent) | P(A and B) = P(A) x P(B\|A) | When A affects B's probability |
| Complement (NOT) | P(not A) = 1 - P(A) | When it is easier to compute P(not A) |
| Conditional probability | P(A\|B) = P(A and B) / P(B) | When you know B has occurred |
| Bayes' theorem | P(A\|B) = P(B\|A) x P(A) / P(B) | When you need to reverse a conditional |
| Expected value | EV = sum of (outcome x probability) | To find the long-run average outcome |

### Glossary

- **Bayes' theorem**: A formula for computing P(A|B) from P(B|A), P(A), and P(B). Used to update beliefs with new evidence.
- **Complement**: The opposite of an event. If event A is "rolling a 6," the complement is "not rolling a 6."
- **Conditional probability**: The probability of an event given that another event has occurred. Written P(A|B).
- **Event**: A subset of the sample space -- one or more outcomes that you are interested in.
- **Expected value (EV)**: The long-run average outcome of a random process. Computed as the sum of each outcome times its probability.
- **Gambler's fallacy**: The incorrect belief that past independent events affect future probabilities.
- **House edge**: The casino's mathematical advantage, expressed as a percentage of each bet. Equal to -EV/bet.
- **Independence**: Two events are independent if the occurrence of one does not affect the probability of the other. P(A|B) = P(A).
- **Mutually exclusive**: Two events that cannot both occur. P(A and B) = 0.
- **Positive EV (+EV)**: A bet or decision where the expected value is greater than zero, indicating long-run profitability.
- **Posterior**: The updated probability after applying Bayes' theorem with new evidence.
- **Prior**: The initial probability before observing new evidence.
- **Sample space**: The set of all possible outcomes of a random experiment.
