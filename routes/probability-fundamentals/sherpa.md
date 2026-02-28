---
title: Probability Fundamentals
route_map: /routes/probability-fundamentals/map.md
paired_guide: /routes/probability-fundamentals/guide.md
topics:
  - Sample Spaces and Events
  - Probability Rules
  - Conditional Probability
  - Bayes' Theorem
  - Expected Value
---

# Probability Fundamentals - AI Teaching Guide

**Purpose**: This guide helps AI assistants teach foundational probability concepts effectively. It provides a structured teaching flow with verification questions, real-world examples from gambling, sports betting, finance, and machine learning, and adaptive strategies for different learner levels.

**Paired Route**: This guide corresponds to `/routes/probability-fundamentals/guide.md` which learners may be reading alongside.

---

## Teaching Overview

### Learning Objectives

By the end of this session, the learner should be able to:
- Define probability and explain the 0-to-1 scale using sample spaces and events
- Apply addition, multiplication, and complement rules to compute combined probabilities
- Compute conditional probabilities and explain how new information changes outcomes
- Apply Bayes' theorem to update beliefs given new evidence
- Calculate expected value and use it to evaluate decisions under uncertainty

### Prior Sessions

Before starting, check `.sessions/index.md` and `.sessions/probability-fundamentals/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify

Before starting, verify the learner has:
- Comfort with basic arithmetic (addition, multiplication, division)
- Understanding of fractions, decimals, and percentages (e.g., converting 1/4 to 0.25 to 25%)
- Basic familiarity with Python (helpful but not strictly required)

**If prerequisites are missing**: Suggest they review the stats-fundamentals route first, or offer a quick primer on fractions and percentages before diving in.

### Audience Context

Learners coming to this route typically fall into one or more of these groups:
- **Gambling and sports betting enthusiasts** who want to understand the math behind odds, house edges, and expected value
- **Aspiring data scientists and ML engineers** who need probability as the foundation for statistical inference, Bayesian methods, and classification algorithms
- **Finance professionals or students** who want to understand risk assessment, portfolio theory, and probabilistic modeling

Identify which group(s) the learner belongs to early on, and tailor examples accordingly. Many learners span multiple groups.

### Learner Preferences Configuration

At the start of the session, ask:
1. "What draws you to learning probability? Gambling, data science, finance, or something else?"
2. "Do you prefer seeing formulas first, or examples first?"
3. "Are you comfortable with Python, or would you prefer me to focus on the concepts and math?"

Adjust the balance of gambling, finance, and ML examples based on their answers. If they want Python, include code. If not, focus on the math and intuition.

### Assessment Strategies

Use a mix of:
- **Conceptual questions**: "In your own words, what does a probability of 0.3 mean?"
- **Computation questions**: "What is the probability of drawing two hearts in a row from a standard deck?"
- **Application questions**: "A casino offers a roulette bet that pays 35:1. The probability of winning is 1/38. Is this a good bet? Why?"
- **Multiple choice with explanation**: Present options, then ask the learner to explain their reasoning regardless of which they pick

---

## Teaching Flow

### Introduction

**What to Cover:**
- Probability is the mathematics of uncertainty -- it gives us a precise language to talk about how likely things are
- It is the foundation of casino games, sports betting odds, financial risk models, weather forecasts, medical testing, and machine learning
- By the end, they will be able to look at any uncertain situation and compute the odds

**Opening Questions to Assess Level:**
1. "Have you encountered probability before -- in a math class, a statistics course, or just thinking about odds?"
2. "What's your main interest in learning probability? Betting, data science, finance, general curiosity?"
3. "If I flip a fair coin twice, what do you think the probability of getting two heads is?"

**Adapt based on responses:**
- If experienced: Move faster through the basics, spend more time on Bayes' theorem and expected value applications
- If complete beginner: Spend extra time on sample spaces and the meaning of probability before moving to rules
- If gambling-focused: Lead with casino and betting examples throughout
- If ML-focused: Emphasize how each concept maps to classification, prediction, and model evaluation
- If finance-focused: Use investment, risk, and portfolio examples

---

### Section 1: What Is Probability?

**Core Concept to Teach:**
Probability quantifies uncertainty. It assigns a number between 0 (impossible) and 1 (certain) to events. Every probabilistic situation starts with a sample space -- the set of all possible outcomes.

**How to Explain:**
1. Start with a relatable analogy: "Probability is like a confidence meter. Zero means 'no way,' one means 'guaranteed,' and everything in between tells you how likely something is."
2. Introduce the sample space: "Before you can compute any probability, you need to list out everything that could possibly happen. That list is the sample space."
3. Show simple examples:
   - Coin flip: sample space = {Heads, Tails}, P(Heads) = 1/2
   - Six-sided die: sample space = {1, 2, 3, 4, 5, 6}, P(rolling a 4) = 1/6
   - Standard deck of 52 cards: P(drawing an Ace) = 4/52 = 1/13
4. Introduce the three interpretations:
   - **Classical**: When all outcomes are equally likely, probability = favorable outcomes / total outcomes
   - **Frequentist**: Probability is the long-run frequency (flip a coin 10,000 times, about 5,000 will be heads)
   - **Subjective**: Probability as degree of belief (a sports analyst says "I think there's a 70% chance this team wins")

**Casino Application:**
"Every casino game has a precisely defined sample space. A roulette wheel has 38 slots (in American roulette: 1-36, 0, 00). The casino knows the exact probability of every outcome. That is how they guarantee profit in the long run -- they don't need to win every spin, they just need the math on their side."

**Finance Application:**
"In finance, we model uncertain outcomes -- will a stock go up or down? Will a bond default? The sample space might be continuous (any price from 0 to infinity), but the core idea is the same: list the possibilities, assign probabilities."

**ML Application:**
"In machine learning, a classifier outputs probabilities. When a spam filter says 'this email is 92% likely to be spam,' it is assigning a probability to a classification outcome. Understanding what that number means is essential."

**Common Misconceptions:**
- "Probability predicts what will happen next" -- Clarify: Probability tells you about long-run patterns, not individual outcomes. A 10% chance of rain does not mean it won't rain today.
- "Past outcomes affect future independent events" (the gambler's fallacy) -- Clarify: If a coin landed heads 5 times in a row, the next flip is still 50/50. The coin has no memory. This is one of the most important misconceptions to address early.
- "A probability of 0.5 means it will happen half the time" -- Clarify: Over many trials it converges to half the time. In a small number of trials, anything can happen.

**Verification Questions:**
1. "What is the sample space for rolling two dice and summing the results?"
2. "If a roulette wheel has 38 slots and you bet on a single number, what is the probability of winning?"
3. "A weather forecast says there's a 30% chance of rain. It doesn't rain. Was the forecast wrong?"

**Good answer indicators:**
- For question 1: They list sums from 2 to 12 (or, better yet, note that the underlying sample space is 36 ordered pairs)
- For question 2: They say 1/38
- For question 3: They explain that 30% means it rains about 3 out of 10 such days, so not raining is the more likely outcome

**If they struggle:**
- Use physical objects: "Imagine you have a bag with 3 red balls and 7 blue balls. If you reach in without looking, what is the probability of drawing a red ball?"
- Draw out the sample space explicitly: list every possibility
- Emphasize that probability is about the setup (the sample space), not about predicting individual results

**Exercise 1.1: Enumerating Sample Spaces**
Present this exercise: "List the complete sample space for each scenario, then compute the requested probability:
(a) Rolling a six-sided die -- P(even number)
(b) Drawing one card from a standard 52-card deck -- P(face card)
(c) Flipping three coins -- P(exactly two heads)"

**How to Guide Them:**
1. First ask: "How would you approach listing all outcomes for three coin flips?"
2. If stuck, provide hints progressively:
   - Hint 1: "For the coins, try listing them systematically: HHH, HHT, HTH, ..."
   - Hint 2: "How many total outcomes are there for three coin flips? Think about it as 2 x 2 x 2."
   - Hint 3: "There are 8 outcomes total. Which ones have exactly two heads?"
3. Encourage them to try before showing solution

**Solution:**
(a) Sample space = {1, 2, 3, 4, 5, 6}. Even numbers = {2, 4, 6}. P(even) = 3/6 = 1/2.
(b) Face cards = J, Q, K in each of 4 suits = 12 cards. P(face card) = 12/52 = 3/13.
(c) Sample space = {HHH, HHT, HTH, HTT, THH, THT, TTH, TTT}. Exactly two heads = {HHT, HTH, THH}. P(exactly 2 heads) = 3/8.

**After exercise, ask:**
- "How did that feel? Was the process of listing sample spaces clear?"
- "Can you see how every probability problem starts with 'what could possibly happen?'"

---

### Section 2: Probability Rules

**Core Concept to Teach:**
Three fundamental rules let you combine simple probabilities into more complex ones: the addition rule (OR), the multiplication rule (AND), and the complement rule (NOT).

**How to Explain:**
1. "Now that you can compute probabilities for simple events, let's learn how to combine them. There are three key operations: OR, AND, and NOT."
2. **Addition Rule (OR)**: P(A or B) = P(A) + P(B) - P(A and B). For mutually exclusive events, P(A and B) = 0, so it simplifies to P(A) + P(B).
   - Example: "What is the probability of drawing a heart OR a diamond from a deck? Hearts and diamonds are mutually exclusive (a card can't be both), so P(heart or diamond) = 13/52 + 13/52 = 26/52 = 1/2."
   - Non-mutually-exclusive example: "What is the probability of drawing a heart OR a king? Some kings are hearts, so P(heart or king) = 13/52 + 4/52 - 1/52 = 16/52 = 4/13."
3. **Multiplication Rule (AND)**: P(A and B) = P(A) x P(B) when events are independent.
   - Example: "What is the probability of rolling a 6 on a die AND flipping heads on a coin? These are independent: P = 1/6 x 1/2 = 1/12."
   - Casino example: "What is the probability of rolling a 7 on two dice? That requires thinking about all the ways to get a sum of 7."
4. **Complement Rule (NOT)**: P(not A) = 1 - P(A).
   - Example: "What is the probability of NOT rolling a 6? P(not 6) = 1 - 1/6 = 5/6."
   - Practical use: "Often it is easier to compute the probability of something NOT happening. What is the probability of getting at least one 6 in four rolls? Compute 1 - P(no sixes in four rolls) = 1 - (5/6)^4."

**Casino Application:**
"In poker, you constantly combine probabilities. What is the probability of being dealt a pair? You need the multiplication rule. What is the probability of completing a flush draw? You need the complement rule. The best poker players are doing this math intuitively."

**Finance Application:**
"If two stocks each have a 10% chance of losing more than 20% in a year, and their movements are independent, what is the probability both lose more than 20%? P = 0.10 x 0.10 = 0.01, or 1%. But what if they are NOT independent -- say they are both tech stocks? Then you cannot simply multiply. This is where understanding independence becomes critical."

**Common Misconceptions:**
- "Always multiply probabilities" -- Clarify: You only multiply for AND (intersection), and only when events are independent. For OR (union), you add. Mixing these up is one of the most common errors.
- "Mutually exclusive and independent are the same thing" -- Clarify: They are opposites in a sense. Mutually exclusive events CANNOT happen together (P(A and B) = 0), while independent events do not affect each other (P(A|B) = P(A)). If two events are mutually exclusive and both have nonzero probability, they are NOT independent.

**Verification Questions:**
1. "In a standard deck, what is the probability of drawing a card that is a spade OR a face card?"
2. "You flip a fair coin three times. What is the probability of getting all heads?"
3. "A slot machine has a 1/1000 chance of hitting the jackpot on any given pull. What is the probability of NOT hitting the jackpot in 1000 pulls?"

**Good answer indicators:**
- For question 1: They use the inclusion-exclusion formula: 13/52 + 12/52 - 3/52 = 22/52 = 11/26
- For question 2: (1/2)^3 = 1/8
- For question 3: (999/1000)^1000, which is approximately 0.368 (about 36.8%). Many people incorrectly think it is close to 0%.

**If they struggle:**
- Use Venn diagrams described verbally: "Imagine two overlapping circles. The OR probability covers both circles. The AND probability is just the overlap."
- Go back to counting: "In a deck of 52 cards, how many are spades? How many are face cards? How many are both?"
- Start with very small examples: two-card decks, two-sided dice

**Exercise 2.1: Card Game Probabilities**
Present this exercise: "You draw two cards from a standard 52-card deck without replacement. Calculate:
(a) P(both cards are aces)
(b) P(at least one card is a heart)
(c) P(first card is a king AND second card is a queen)"

**How to Guide Them:**
1. Ask: "What makes drawing without replacement different from rolling dice twice?"
2. Hints:
   - Hint 1: "For part (a), after drawing the first ace, how many aces are left? How many cards total?"
   - Hint 2: "For part (b), try the complement: P(at least one heart) = 1 - P(no hearts in two draws)."
   - Hint 3: "For the complement approach: P(first card not a heart) = 39/52. Given the first was not a heart, P(second not a heart) = 38/51."

**Solution:**
(a) P(both aces) = 4/52 x 3/51 = 12/2652 = 1/221
(b) P(at least one heart) = 1 - P(no hearts) = 1 - (39/52 x 38/51) = 1 - 1482/2652 = 1170/2652 = approx 0.441
(c) P(king then queen) = 4/52 x 4/51 = 16/2652 = 4/663

---

### Section 3: Conditional Probability

**Core Concept to Teach:**
Conditional probability answers the question: "Given that one thing has happened (or is true), what is the probability of another thing?" It is written P(A|B) and read "the probability of A given B."

**How to Explain:**
1. Start with intuition: "Imagine you know it is cloudy outside. Does that change your estimate of the probability of rain? Of course it does. That updated probability is a conditional probability: P(rain | cloudy)."
2. Formula: P(A|B) = P(A and B) / P(B)
3. Walk through a concrete example:
   - "In a standard deck, what is P(face card | spade)?"
   - P(face card and spade) = 3/52 (Jack, Queen, King of spades)
   - P(spade) = 13/52 = 1/4
   - P(face card | spade) = (3/52) / (13/52) = 3/13
   - "This makes sense: given we know it is a spade, there are 13 spades, 3 of which are face cards."
4. Emphasize: "Conditioning on B shrinks the sample space to only the outcomes where B is true."

**Sports Betting Application:**
"Sports bettors use conditional probability constantly. What is the probability a team wins given that their star player is injured? P(win | star injured) is different from P(win). Oddsmakers adjust the line based on this conditional probability."

"Consider this: a basketball team has a 60% overall win rate. But when they are at home, they win 75% of games. P(win | home) = 0.75. This is the kind of conditional probability that matters for betting."

**ML Application:**
"Conditional probability IS machine learning. When a classifier predicts 'this email is spam,' it is computing P(spam | email features). Naive Bayes, one of the simplest and most effective classifiers, is built directly on conditional probability. We will see Bayes' theorem next, which formalizes exactly how this works."

**Finance Application:**
"Credit risk models compute P(default | economic recession). Insurance companies compute P(claim | age, driving history, location). These are all conditional probabilities, and getting them right is worth billions of dollars."

**Common Misconceptions:**
- Confusing P(A|B) with P(B|A): "The probability of testing positive given you have a disease is NOT the same as the probability of having the disease given you tested positive. This confusion costs lives in medical settings."
- Assuming correlation means conditional change: "Just because P(A|B) differs from P(A) does not mean B causes A."
- Forgetting that independence means P(A|B) = P(A): "If knowing B does not change the probability of A, the events are independent."

**Verification Questions:**
1. "A bag has 5 red and 5 blue marbles. You draw one marble and it is red. You do not replace it. What is the probability the second marble is also red?"
2. "70% of students pass a class. Of those who attend every lecture, 90% pass. What is P(pass | perfect attendance)?"
3. "Is P(rain | cloudy) the same as P(cloudy | rain)? Give an example showing why or why not."

**Good answer indicators:**
- For question 1: 4/9 (4 red left out of 9 total)
- For question 2: 0.90 (this is directly stated, but they should recognize it as a conditional probability)
- For question 3: No. It almost always rains when it is cloudy, but it is cloudy many days when it does not rain. P(rain | cloudy) might be 0.3, while P(cloudy | rain) might be 0.95.

**If they struggle:**
- Use the "shrinking sample space" visual: "When we condition on B, pretend B is the entire world. Only look at outcomes where B happened."
- Use frequency tables: "Out of 1000 people, 100 have a disease, 900 do not. Of the 100 with the disease, 95 test positive. Of the 900 without, 45 test positive. Now answer: if someone tests positive, what is the probability they have the disease?"
- Work through the marble bag example physically step by step

**Exercise 3.1: Conditional Probability in Context**
Present this exercise: "A sports analyst has the following data on a basketball team:
- Overall win rate: 55%
- Win rate when at home: 70%
- Win rate when the opposing team's best player is out: 75%
- 50% of games are at home

(a) What is P(win | home)?
(b) What is P(home | win)? (Use Bayes' theorem or a frequency table.)
(c) Are 'winning' and 'playing at home' independent? How do you know?"

**How to Guide Them:**
1. Part (a) is directly stated -- this checks if they can read conditional probability notation
2. For part (b), suggest: "Try imagining 100 games. In 50 home games, they win 70% = 35. In 50 away games, they win 55% overall means 55 total wins. So away wins = 55 - 35 = 20. P(home | win) = 35/55."
3. For part (c): "Are the events independent? What would have to be true for independence?"

**Solution:**
(a) P(win | home) = 0.70
(b) P(home | win) = P(win and home) / P(win) = (0.50 x 0.70) / 0.55 = 0.35 / 0.55 = 7/11 = approx 0.636
(c) Not independent, because P(win | home) = 0.70, which does not equal P(win) = 0.55. If they were independent, knowing they are at home would not change the win probability.

---

### Section 4: Bayes' Theorem

**Core Concept to Teach:**
Bayes' theorem provides a formula for computing conditional probabilities by flipping them around: it lets you go from P(B|A) to P(A|B). It is the mathematical tool for updating beliefs in light of new evidence.

**How to Explain:**
1. Start with the problem: "You often know P(evidence | hypothesis) but need P(hypothesis | evidence). For example, a medical test is 99% accurate (P(positive test | disease) = 0.99), but what you actually want to know is: P(disease | positive test). These are very different numbers."
2. The formula: P(A|B) = P(B|A) x P(A) / P(B)
3. Explain the components:
   - **Prior**: P(A) -- your initial belief before seeing evidence (e.g., how common is the disease?)
   - **Likelihood**: P(B|A) -- how likely is the evidence if the hypothesis is true?
   - **Evidence**: P(B) -- how likely is the evidence overall? (Often computed using the law of total probability)
   - **Posterior**: P(A|B) -- your updated belief after seeing evidence
4. Walk through the medical testing example in detail:
   - Disease prevalence: 1% (P(disease) = 0.01)
   - Test sensitivity: 99% (P(positive | disease) = 0.99)
   - Test false positive rate: 5% (P(positive | no disease) = 0.05)
   - P(positive) = P(positive | disease) x P(disease) + P(positive | no disease) x P(no disease) = 0.99 x 0.01 + 0.05 x 0.99 = 0.0099 + 0.0495 = 0.0594
   - P(disease | positive) = 0.0099 / 0.0594 = approx 0.167 or about 16.7%
   - "Even with a 99% accurate test, a positive result only means a 17% chance of having the disease! This is because the disease is rare (1% prevalence). Most positives are false positives."

**Sports Betting Application:**
"Before the season, you estimate a team has a 40% chance of making the playoffs (your prior). After they win 8 of their first 10 games, you update your estimate. Bayes' theorem tells you exactly how much to update. If strong teams win 8 of 10 about 30% of the time, and weak teams do so only 5% of the time, you can compute your new probability precisely."

**ML Application:**
"Naive Bayes classification is literally Bayes' theorem applied to features. To classify an email as spam, the algorithm computes: P(spam | words in email) using P(words | spam) x P(spam) / P(words). Each word is a piece of evidence that updates the probability."

"Bayesian neural networks use the same idea at a larger scale: update beliefs about model parameters given training data."

**Finance Application:**
"Risk analysts update P(recession | leading indicators). When unemployment ticks up, Bayes' theorem tells you how much to increase your recession probability. The prior is your baseline estimate, and the new economic data is the evidence."

**Common Misconceptions:**
- "Bayes' theorem is only for complex math" -- Clarify: It is a simple formula. The hard part is thinking about priors and likelihoods clearly, not the arithmetic.
- "Base rate neglect": Humans naturally focus on the evidence (the positive test result) and ignore the base rate (how rare the disease is). This leads to massive overestimates of P(disease | positive test). Emphasize this heavily.
- "The prior doesn't matter" -- Clarify: The prior matters enormously, especially when evidence is ambiguous. A rare disease stays unlikely even after one positive test.

**Verification Questions:**
1. "In the medical testing example, what happens if the disease prevalence is 10% instead of 1%? Does P(disease | positive) go up or down?"
2. "A spam filter knows that 30% of emails are spam. Spam emails contain the word 'free' 80% of the time. Non-spam emails contain 'free' 10% of the time. What is P(spam | contains 'free')?"
3. "Why do experienced gamblers talk about 'updating their priors'?"

**Good answer indicators:**
- For question 1: It goes up significantly. Higher base rate means positive results are more likely to be true positives.
- For question 2: P(spam | free) = (0.8 x 0.3) / (0.8 x 0.3 + 0.1 x 0.7) = 0.24 / 0.31 = approx 0.774
- For question 3: They are using Bayesian reasoning -- starting with an initial estimate and revising it as new information comes in during a game or season.

**If they struggle:**
- Use frequency tables: "Out of 10,000 people, 100 have the disease (1%). Of those 100, 99 test positive. Of the 9,900 without, 495 test positive. Total positives = 594. How many of those 594 actually have the disease?"
- Work through simpler examples first: "A bag has 2 red balls and 8 blue balls. You pick one without looking. Your friend peeks and says 'it's not blue.' What's the probability it's red?" (Obviously 1, but it illustrates updating with evidence.)
- Emphasize the natural frequency approach over the formula for intuition

**Exercise 4.1: Bayes' Theorem in Practice**
Present this exercise: "You are a sports bettor evaluating a new basketball tipster (someone who sells predictions). The tipster claims 70% accuracy. Before testing them, you are skeptical -- you believe there is only a 10% chance they are truly skilled (your prior), and a 90% chance they are just guessing (50% accuracy). The tipster makes 10 predictions and gets 8 correct.

(a) What is P(8 correct out of 10 | skilled)? (Use the binomial formula or a calculator.)
(b) What is P(8 correct out of 10 | guessing)?
(c) Using Bayes' theorem, what is the posterior probability that the tipster is truly skilled?"

**How to Guide Them:**
1. This problem integrates several concepts. Walk through it step by step.
2. Hints:
   - Hint 1: "For the binomial probability, use: P(k successes in n trials) = C(n,k) x p^k x (1-p)^(n-k)"
   - Hint 2: "C(10,8) = 45. For skilled: 45 x 0.7^8 x 0.3^2. For guessing: 45 x 0.5^10."
   - Hint 3: "Apply Bayes': P(skilled | 8 correct) = P(8 correct | skilled) x P(skilled) / P(8 correct)"
3. Let them work through the arithmetic. Offer Python as a computation tool.

**Solution:**
(a) P(8 correct | skilled) = C(10,8) x 0.7^8 x 0.3^2 = 45 x 0.05765 x 0.09 = 45 x 0.005188 = 0.2335
(b) P(8 correct | guessing) = C(10,8) x 0.5^10 = 45 x 0.000977 = 0.04395
(c) P(8 correct) = P(8|skilled) x P(skilled) + P(8|guessing) x P(guessing) = 0.2335 x 0.1 + 0.04395 x 0.9 = 0.02335 + 0.03956 = 0.06291
P(skilled | 8 correct) = 0.02335 / 0.06291 = approx 0.371 or about 37%

"Even after getting 8 out of 10 right, there is only a 37% chance the tipster is genuinely skilled. Your skeptical prior (10%) has updated upward, but the result is not conclusive. This is why Bayesian thinking protects you from overreacting to small samples."

---

### Section 5: Expected Value

**Core Concept to Teach:**
Expected value (EV) is the long-run average outcome of a random process. It is calculated by multiplying each outcome by its probability and summing. EV is the single most important concept for making rational decisions under uncertainty.

**How to Explain:**
1. Start with the core idea: "If you repeated a bet or decision a million times, what would you average per trial? That average is the expected value."
2. Formula: EV = sum of (outcome x probability) for all possible outcomes
3. Simple example: "You bet $1 on a coin flip. Heads you win $2, tails you lose your $1. EV = 0.5 x (+$2) + 0.5 x (-$1) = $1.00 - $0.50 = +$0.50. Positive EV -- this is a good bet."

**Casino Application -- Why the House Always Wins:**
Walk through specific casino games:

"**Roulette (American, double-zero)**:
You bet $1 on a single number. 38 slots on the wheel.
- Win: You get $36 (your $1 back + $35 profit). Probability: 1/38.
- Lose: You lose $1. Probability: 37/38.
- EV = (1/38) x $35 + (37/38) x (-$1) = $0.921 - $0.974 = -$0.053

Every dollar you bet on roulette, you expect to lose about 5.3 cents. The house edge is 5.26%. It does not matter which bet you make on the roulette table -- single number, red/black, odd/even -- the house edge is the same (except the five-number bet, which is worse)."

"**Blackjack** (with basic strategy):
The house edge drops to about 0.5%. EV per $1 bet = roughly -$0.005. This is why card counters target blackjack -- they can occasionally flip the EV to positive."

**Sports Betting Application -- Finding +EV Bets:**
"This is where expected value becomes actionable for sports bettors. A bet has positive expected value when your estimated probability of winning exceeds the implied probability from the odds.

Example: A sportsbook offers +150 on Team A (meaning you win $150 on a $100 bet). The implied probability is 100/250 = 40%. If you believe Team A's true probability of winning is 50%, the EV is:
- EV = 0.50 x $150 + 0.50 x (-$100) = $75 - $50 = +$25 per $100 bet

This is a +EV bet. Professional sports bettors build their entire strategy around finding these edges."

**Finance Application:**
"Expected return on an investment works the same way. If a stock has a 60% chance of going up 20% and a 40% chance of going down 10%, the expected return is:
EV = 0.60 x 20% + 0.40 x (-10%) = 12% - 4% = 8%

Portfolio theory, options pricing, and insurance all depend on expected value calculations."

**ML Application:**
"Expected loss is central to training machine learning models. The loss function computes the 'cost' of each prediction, and the expected loss over the training data is what the model minimizes. Cost-sensitive classification uses expected value explicitly: misclassifying a cancer patient as healthy (false negative) has a higher cost than the reverse."

**Common Misconceptions:**
- "Expected value means I'll get that result" -- Clarify: EV is an average over many trials. In any single trial, you might get much more or much less. A lottery ticket has negative EV, but someone does win.
- "Ignoring variance" -- Clarify: Two bets can have the same EV but very different risk. A 50% chance of winning $100 vs. a 1% chance of winning $5,000 both have EV = $50, but they feel very different. Variance matters for decision-making.
- "Positive EV means guaranteed profit" -- Clarify: Positive EV means profit in the long run over many repetitions. In the short run, you can still lose. You need a bankroll that can absorb the variance.

**Verification Questions:**
1. "A lottery ticket costs $2. The jackpot is $1,000,000 and the probability of winning is 1 in 10,000,000. What is the EV of buying a ticket?"
2. "A sports bet pays +200 (win $200 on a $100 bet). You think there is a 40% chance of winning. Is this a +EV bet?"
3. "Why do casinos stay in business even though individual players sometimes win big?"

**Good answer indicators:**
- For question 1: EV = (1/10,000,000) x $999,998 + (9,999,999/10,000,000) x (-$2) = $0.10 - $2.00 = -$1.90. Strongly negative EV.
- For question 2: EV = 0.40 x $200 + 0.60 x (-$100) = $80 - $60 = +$20. Yes, positive EV.
- For question 3: Every game has negative EV for the player. Over millions of bets, the law of large numbers guarantees the casino's average return converges to the expected value, which is always in their favor.

**If they struggle:**
- Use a concrete small example: "If I flip a coin and pay you $3 for heads but you pay me $1 for tails, would you play? Let's compute the EV."
- Emphasize the "repeat a million times" framing: "If you played this game a million times, how much total would you expect to win or lose?"
- Connect to their intuition: "Does this bet feel fair? Let's see if the math agrees."

**Exercise 5.1: EV Calculations**
Present this exercise: "Calculate the expected value for each scenario and determine whether it is a favorable bet/decision:
(a) American roulette: betting $10 on red (18/38 chance of winning, pays 1:1)
(b) A sports bet at -110 odds (bet $110 to win $100) where you believe you have a 55% chance of winning
(c) An investment with a 70% chance of returning 15% and a 30% chance of losing 5%"

**Solution:**
(a) EV = (18/38) x $10 + (20/38) x (-$10) = $4.74 - $5.26 = -$0.53. Negative EV, unfavorable.
(b) EV = 0.55 x $100 + 0.45 x (-$110) = $55 - $49.50 = +$5.50 per bet. Positive EV, favorable.
(c) EV = 0.70 x 15% + 0.30 x (-5%) = 10.5% - 1.5% = +9.0%. Positive EV, favorable.

---

## Practice Project

**Project Introduction:**
"Now let's put everything together by building a probability and expected value calculator in Python. This will be a set of functions that compute probabilities and EV for common gambling scenarios."

**Requirements:**
Present these requirements:
- A function that computes the probability of simple events given a sample space
- A function that applies Bayes' theorem given prior, likelihood, and evidence
- A function that computes expected value given outcomes and their probabilities
- Apply these to compute the house edge for at least two casino games (roulette and a simplified blackjack hand)
- Evaluate a sports bet for positive expected value

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work, offer to answer questions
2. **If they want guidance**: Work through it step by step together, starting with the EV function
3. **If they're unsure**: Suggest starting with the expected_value function, then building up

**Checkpoints During Project:**
- After the EV function: "Let's test it with the roulette example. Does it return -$0.053 per dollar bet?"
- After the Bayes function: "Let's test with the medical example. Does it return about 16.7%?"
- After casino analysis: "Which game would you rather play based on EV? Does that match your intuition?"
- After sports bet evaluation: "If you found a +EV bet, how confident are you in your probability estimate?"

**Code Review Approach:**
When reviewing their work:
1. Start with praise: "Your EV function handles the edge cases well."
2. Ask questions: "What happens if someone passes in probabilities that don't sum to 1?"
3. Guide improvements: "You could add input validation to make this more robust."
4. Relate to concepts: "Notice how the roulette EV is always negative -- that's the house edge in action."

**If They Get Stuck:**
- Ask them to explain what they are trying to compute
- Provide the function signature and let them fill in the body
- Walk through the math on paper first, then translate to code
- Offer to do the first function together so they see the pattern

**Reference Solution:**
```python
import numpy as np
from math import comb

def expected_value(outcomes, probabilities):
    """Compute expected value given outcomes and their probabilities."""
    outcomes = np.array(outcomes)
    probabilities = np.array(probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-9, "Probabilities must sum to 1"
    return np.sum(outcomes * probabilities)

def bayes_theorem(prior, likelihood, evidence):
    """Apply Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)"""
    return (likelihood * prior) / evidence

def bayes_full(prior, likelihood_if_true, likelihood_if_false):
    """Apply Bayes' theorem computing evidence from components."""
    evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior)
    posterior = (likelihood_if_true * prior) / evidence
    return posterior

def roulette_ev(bet_amount, payout_multiplier, winning_slots=1, total_slots=38):
    """Compute EV for a roulette bet."""
    p_win = winning_slots / total_slots
    p_lose = 1 - p_win
    ev = p_win * (bet_amount * payout_multiplier) + p_lose * (-bet_amount)
    return ev

def sports_bet_ev(stake, decimal_odds, true_probability):
    """Evaluate a sports bet. Decimal odds: 2.50 means win $2.50 per $1 bet (including stake)."""
    profit = stake * (decimal_odds - 1)
    ev = true_probability * profit + (1 - true_probability) * (-stake)
    return ev

# --- Demo ---
print("=== Roulette Expected Value ===")
ev = roulette_ev(bet_amount=10, payout_multiplier=1, winning_slots=18, total_slots=38)
print(f"$10 bet on red: EV = ${ev:.2f}")

ev = roulette_ev(bet_amount=1, payout_multiplier=35, winning_slots=1, total_slots=38)
print(f"$1 bet on single number: EV = ${ev:.4f}")

print("\n=== Bayes' Theorem: Medical Test ===")
posterior = bayes_full(prior=0.01, likelihood_if_true=0.99, likelihood_if_false=0.05)
print(f"P(disease | positive test) = {posterior:.3f}")

print("\n=== Sports Bet EV ===")
ev = sports_bet_ev(stake=100, decimal_odds=2.50, true_probability=0.50)
print(f"$100 at 2.50 odds, 50% win chance: EV = ${ev:.2f}")
```

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- Probability is a 0-to-1 scale for uncertainty, and it starts with sample spaces
- Three rules (addition, multiplication, complement) let you combine probabilities
- Conditional probability captures how new information changes the odds
- Bayes' theorem lets you flip conditional probabilities and update beliefs
- Expected value is the long-run average and the foundation of rational decision-making

Ask them to explain one concept back to you, ideally with a real-world example.

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with probability fundamentals?"
- 1-4: Suggest reviewing the guide.md material, focusing on the areas that felt shakiest. Offer to revisit specific sections.
- 5-7: Normal for a first pass! Suggest working through the exercises in guide.md independently, then returning for the practice project.
- 8-10: Great foundation! Suggest moving to probability-distributions or bayesian-statistics.

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice more, try the exercises in the guide at your own pace"
- "When you're ready, the probability-distributions route covers binomial, normal, and Poisson distributions"
- "If you want to go deeper into Bayesian reasoning, check out the bayesian-statistics route"
- For gambling enthusiasts: "Try computing the EV for other casino games or your favorite sports bets"
- For ML learners: "Start thinking about how Naive Bayes uses everything we covered today"
- For finance learners: "Try modeling a simple portfolio decision using expected value"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there anything you'd like me to explain differently or work through one more time?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Slow down and use physical analogies: coins, dice, bags of marbles
- Avoid formulas initially -- work through everything by counting outcomes
- Use natural frequency tables (out of 1000 people...) instead of abstract probabilities
- Do exercises together rather than having them work alone
- Check prerequisites: they may need more comfort with fractions and percentages

### If Learner is Excelling
- Move at a faster pace, skip basic counting examples
- Present more challenging variations: without-replacement problems, multi-step scenarios
- Introduce connections to advanced topics: how Bayes' theorem connects to Bayesian inference, how EV connects to utility theory
- Ask deeper "why" questions: "Why is the house edge the same for every roulette bet?"
- Challenge them with paradoxes: the Monty Hall problem, the birthday problem

### If Learner Seems Disengaged
- Check in: "How are you feeling about this?"
- Ask about their goals: "What made you interested in learning probability?"
- Switch to examples from their area of interest: if they came for gambling but you have been using finance examples, pivot
- Try a more interactive approach: "Let's simulate this -- you tell me what happens when we flip the coin"
- Take a break if needed

### Different Learning Styles
- **Visual learners**: Describe Venn diagrams, probability trees, and frequency tables. Use the "imagine 1000 people" framing.
- **Hands-on learners**: Jump to exercises quickly, explain concepts through working problems
- **Conceptual learners**: Spend more time on "why" -- why does the addition rule work? What does independence really mean?
- **Example-driven learners**: Lead with the casino/betting example, extract the general principle after

---

## Troubleshooting Common Issues

### Technical Setup Problems
- **Python not installed**: Guide them to python.org or suggest using an online REPL (replit.com, Google Colab)
- **numpy not available**: Use `pip install numpy`. For exercises that don't need numpy, pure Python works fine.
- **Floating point issues**: Explain that 0.1 + 0.2 != 0.3 in floating point -- use `round()` or `abs(a - b) < 1e-9` for comparisons

### Concept-Specific Confusion

**If confused about sample spaces:**
- Go back to the very basics: "A sample space is just a list of everything that could happen. For a coin, there are only two things that can happen."
- Use enumeration: actually write out every possibility
- Emphasize that sample spaces can be large (deck of cards = 52 outcomes) but the principle is the same

**If confused about the difference between OR and AND:**
- OR = at least one happens (addition rule, bigger probability)
- AND = both happen (multiplication rule, smaller probability)
- "If you OR two things together, probability goes up. If you AND two things together, probability goes down."

**If confused about conditional probability:**
- Always go back to the "shrinking sample space" analogy
- Use concrete two-way tables with numbers, not abstract formulas
- Work through P(A|B) vs P(B|A) with a specific example until the difference clicks

**If confused about Bayes' theorem:**
- Use the natural frequency approach exclusively (out of 10,000 people...)
- Build a tree diagram verbally: "Start with 10,000 people. Split into disease/no disease. Then split each into test positive/negative."
- Do not rush -- Bayes' theorem is the hardest concept in this route

**If confused about expected value:**
- Frame it as: "What would you average if you did this a million times?"
- Start with clearly unfair games (flip a coin: heads you win $10, tails you lose $20) before fair games
- Compute EV step by step, writing out each outcome x probability term

---

## Teaching Notes

**Key Emphasis Points:**
- Really emphasize the gambler's fallacy in Section 1 -- it is the most common and most costly misconception
- Make sure they understand the difference between mutually exclusive and independent in Section 2 before moving to conditional probability
- Base rate neglect in Bayes' theorem (Section 4) deserves extra time -- walk through multiple examples
- Expected value (Section 5) is the most directly applicable concept -- make sure they can compute it confidently

**Pacing Guidance:**
- Don't rush Section 1 -- the sample space concept and gambler's fallacy are foundational
- Section 2 can be quicker if they grasped Section 1, since it builds directly on counting
- Section 3 (conditional probability) is a conceptual leap -- allow extra time
- Section 4 (Bayes' theorem) is the most challenging section -- take as long as needed
- Section 5 (expected value) is often the most engaging -- let their enthusiasm drive the pace
- Allow plenty of time for the practice project

**Success Indicators:**
You'll know they've got it when they:
- Can explain why the gambler's fallacy is wrong using probability concepts
- Correctly distinguish when to add vs multiply probabilities
- Can compute P(A|B) and explain why it differs from P(B|A)
- Successfully apply Bayes' theorem to a novel problem
- Compute expected value and correctly identify favorable vs unfavorable bets
- Ask questions like "What's the EV of..." or "How would I update my prior for..."
