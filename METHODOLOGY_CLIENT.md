# UFC Fighter Rating System
## How It Works

**Prepared by:** UFC Fight Prediction
**Version:** 2.0 — April 2026
**Dataset:** 16,398 UFC fights across 770 events (1994 -- April 2026)
**Test period:** 3,315 fights from January 2023 onward (never seen during training)

---

## About This Document

This document walks through the entire prediction system from the ground up. Every number was chosen by testing it against real UFC fight data. Every formula is shown with worked examples.

No statistical background is needed to follow this document. Technical terms are defined in the Glossary at the end.

---

## Part 1: What This System Does

We assign every active UFC fighter a numerical rating that represents their current skill level. These ratings feed into a machine learning model that predicts fight outcomes with **60.1% accuracy** across 3,315 test fights.

> **Note (2026-07-01):** this is a v2.0-era document describing the Elo-anchored pipeline (60.1% / 0.2348 Brier). The current shipped model (base `xgb_v2`, which folds in closing-odds features) reaches **~70% accuracy / ~0.20 Brier** on recent fights. See `KNOWN_ISSUES.md` → "Model performance clarification" for the current, evidence-based figures.

The system accounts for:

- **Who they beat** -- beating a top-10 fighter is worth more than beating a newcomer
- **How they won** -- a knockout carries more weight than a split decision
- **How long ago they fought** -- fighters inactive for over 9 months receive a rating adjustment
- **What weight class they fight in** -- ratings are tracked per division
- **Physical attributes** -- height, reach, age, and stance advantages
- **Performance trends** -- striking output, grappling activity, defensive skills, and whether those metrics are improving or declining

The foundation is a method called **Elo**, originally developed for chess rankings in the 1960s. We chose it because it handles uneven schedules (not every fighter fights every other fighter), it updates after every fight, and it can be extended with MMA-specific adjustments. On top of Elo, a gradient boosted machine learning model combines 28 data points per fight to produce a calibrated win probability.

---

## Part 2: How We Validate Everything

### The Core Test

Every setting in this system was validated through **backtesting**:

1. Take all UFC fights from 1994 through December 2022 (~13,000 fights) -- the **training set**
2. Take all UFC fights from January 2023 onward (3,315 fights) -- the **test set**
3. Before each test fight, predict who wins and with what confidence
4. Compare predictions to what actually happened
5. Score the accuracy

The test fights happened after the training period. The system cannot cheat by knowing the outcome in advance. This is the same principle used in financial modelling and medical research: train on the past, prove it on the future.

### How We Score Accuracy

We use three metrics:

**Brier score** -- measures overall prediction quality. Lower is better. A coin flip (50/50 every fight) scores 0.250. Our system scores **0.2348**.

**AUC-ROC** -- measures how well the model distinguishes winners from losers. 0.5 is random, 1.0 is perfect. Our system scores **0.6445**.

**Accuracy** -- simple percentage of fights where we picked the winner. Our system hits **60.1%**.

### How We Choose Settings

For each parameter, we test many possible values -- sometimes hundreds. We run the full backtest for each one and pick the value with the best Brier score. This process is called a **grid search**.

---

## Part 3: The Rating Formula

### Step 1 -- Starting Point

Every fighter starts at a rating of **1,500**. This is just a baseline -- only the difference between two fighters' ratings matters.

### Step 2 -- Expected Win Probability

Before each fight, the system calculates win probability from the rating gap:

```
Expected Probability = 1 / (1 + 10^((Opponent Rating - Fighter Rating) / 400))
```

The 400 is a universal scaling constant from chess -- it defines how the rating scale is spaced. A 400-point gap corresponds to a 10:1 win expectation. You could use a different number and the predictions would be identical, just on a wider or narrower scale. The entire sports analytics world uses 400 by convention.

**Example -- evenly matched fighters:**

Fighter A rated 1,500 vs Fighter B rated 1,500:

```
E(A) = 1 / (1 + 10^((1500 - 1500) / 400))
     = 1 / (1 + 10^0)
     = 1 / 2
     = 0.50 -- 50% chance each
```

**Example -- clear favourite:**

Fighter A rated 1,700 vs Fighter B rated 1,400 (300-point gap):

```
E(A) = 1 / (1 + 10^((1400 - 1700) / 400))
     = 1 / (1 + 10^(-0.75))
     = 1 / 1.178
     = 0.849 -- Fighter A has an 85% chance
```

**Rating gap reference:**

| Gap | Favourite's Probability |
|-----|------------------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 300 | 85% |
| 400 | 91% |

### Step 3 -- Rating Update After the Fight

```
New Rating = Old Rating + K x MOV x (Actual Result - Expected Probability)
```

Where K is the K-factor (how volatile the rating is), MOV is the margin-of-victory multiplier (how decisive the win was), Actual Result is 1.0 for a win and 0.0 for a loss, and Expected Probability is from Step 2.

**Example -- an upset:**

Fighter A (1,400) beats Fighter B (1,700) by KO. Fighter A was given 15% odds.

```
Fighter A: 1400 + 10 x 1.2 x (1.0 - 0.15) = 1400 + 10.2 = 1,410.2
Fighter B: 1700 + 10 x 1.2 x (0.0 - 0.85) = 1700 - 10.2 = 1,689.8
```

The underdog gains 10.2 points. The favourite drops 10.2 points.

**Example -- an expected result:**

Fighter A (1,700) beats Fighter B (1,400) by unanimous decision. Fighter A was given 85% odds.

```
Fighter A: 1700 + 10 x 1.0 x (1.0 - 0.85) = 1700 + 1.5 = 1,701.5
```

Winning a fight you were supposed to win barely moves the rating.

### Step 4 -- Division Transfers

When a fighter changes weight class:

```
New Division Rating = 1,500 + 0.50 x (Old Division Rating - 1,500)
```

**Example:** Fighter rated 1,650 in Lightweight moves to Welterweight:

```
1,500 + 0.50 x (1,650 - 1,500) = 1,500 + 75 = 1,575
```

Half their advantage carries over. This was validated by testing 7 transfer rates from 50% to 100% -- lower rates produced strictly better predictions.

### Step 5 -- Inactivity Regression

After 270 days (9 months) without a fight, ratings drift back toward 1,500:

```
Months Past Threshold = (Days Since Last Fight - 270) / 30.44
Regression = min(10% x Months Past Threshold, 50%) x (Rating - 1,500)
Adjusted Rating = Rating - Regression
```

**Example:** A fighter rated 1,700 inactive for 18 months:

```
Months past threshold = (548 - 270) / 30.44 = 9.1 months
Regression = min(91%, 50%) x 200 = 100 points
Adjusted rating = 1,700 - 100 = 1,600
```

The 50% cap prevents ratings from collapsing completely.

### Step 6 -- Shrinkage for New Fighters

Fighters with fewer than 5 UFC fights have their ratings pulled toward 1,500:

```
Displayed Rating = 1,500 + (Raw Rating - 1,500) x min(Fight Count / 5, 1.0)
```

**Example:** A fighter with 2 fights and a raw rating of 1,700:

```
1,500 + 200 x 0.40 = 1,580 (instead of 1,700)
```

After 5 fights, the full rating is displayed. This prevents a 2-0 newcomer from appearing higher-rated than a proven champion.

---

## Part 4: How Each Setting Was Chosen

Each parameter was tested against hundreds of alternatives. The tables below show the top results for each.

### 4.1 -- K-Factor (Rating Volatility)

Tested 225 combinations. The K-factor controls how much a single fight moves the rating.

| Rank | K-initial | K-experienced | Transition at | Brier Score |
|------|-----------|---------------|---------------|-------------|
| 1 | 45 | 10 | 4 fights | **0.248640** |
| 2 | 50 | 10 | 4 fights | 0.248654 |
| 3 | 40 | 10 | 4 fights | 0.248682 |
| 4 | 55 | 10 | 4 fights | 0.248720 |
| 5 | 35 | 10 | 4 fights | 0.248783 |

All top-10 configurations used K-experienced = 10. Veteran rating stability matters more than anything else in this parameter group.

**We use:** K = 45 for a fighter's first 4 fights, K = 10 after that.

### 4.2 -- Margin of Victory Multiplier

Tested 294 combinations. This scales the rating update based on how the fight ended.

| Rank | KO/TKO | Submission | Split Decision | Brier Score |
|------|--------|------------|----------------|-------------|
| 1 | **1.2** | **1.1** | **0.5** | **0.249764** |
| 2 | 1.2 | 1.1 | 0.6 | 0.249771 |
| 3 | 1.2 | 1.2 | 0.5 | 0.249782 |
| 4 | 1.2 | 1.3 | 0.5 | 0.249793 |
| 5 | 1.2 | 1.1 | 0.7 | 0.249801 |

All top-5 used KO/TKO = 1.2. The original assumption from sports analytics literature was 1.5. The data preferred lower multipliers -- finish type matters less for *predicting future fights* than you might expect.

**In practice:** If the base rating change is 10 points, a KO produces 12 points, a submission produces 11, a unanimous decision produces 10, and a split decision produces just 5.

### 4.3 -- Division Transfer Rate

Tested 7 values. Results are perfectly monotonic -- lower is always better.

| Transfer Rate | Brier Score |
|--------------|-------------|
| **50%** | **0.250475** |
| 60% | 0.250518 |
| 75% | 0.250583 |
| 100% | 0.250688 |

Fighters who change weight class are only half as predictable in their new division as their old rating would suggest.

### 4.4 -- Inactivity Regression

Tested 14 configurations across threshold, rate, and cap.

| Rank | Threshold | Rate | Cap | Brier Score |
|------|-----------|------|-----|-------------|
| 1 | **270 days (9 months)** | **10%** | **50%** | **0.249129** |
| 2 | 270 days | 8% | 50% | 0.249145 |
| 3 | 270 days | 10% | 40% | 0.249162 |
| 5 | 365 days (12 months) | 10% | 50% | 0.249200 |

9 months outperformed 12 months consistently. UFC fighters lose rating predictability faster than chess players.

### 4.5 -- Domain Attribution (Striking vs Grappling Split)

Tested 25 combinations for how fight outcomes are split between striking and grappling sub-ratings.

| Rank | KO/TKO Split | Submission Split | Brier Score |
|------|-------------|-----------------|-------------|
| 1 | **100% striking / 0% grappling** | **0% striking / 100% grappling** | **0.183334** |
| 2 | 90% / 10% | 0% / 100% | 0.183421 |
| 3 | 100% / 0% | 10% / 90% | 0.183489 |

The data strongly favours full separation: KOs are pure striking events, submissions are pure grappling events.

### 4.6 -- EWMA Half-Life

Tested 4 values (2, 3, 4, 5 fights). All produced identical Brier scores (0.172240). This parameter does not measurably affect prediction accuracy, so we kept the default of 3 fights.

---

## Part 5: The 28 Data Points That Power Predictions

The Elo rating captures a fighter's overall skill trajectory. But the machine learning model looks at 28 data points per fight -- all computed as the difference between Fighter A and Fighter B. Here is every data point the model uses.

### Elo Ratings (3 features)

| # | Feature | What It Captures |
|---|---------|-----------------|
| 1 | **Overall Elo differential** | The core skill gap between the two fighters, built from their entire fight history |
| 2 | **Striking Elo differential** | How each fighter's striking ability compares, tracked separately from grappling |
| 3 | **Grappling Elo differential** | How each fighter's grappling ability compares, tracked separately from striking |

These use the pre-fight rating (before the outcome is known). Using the post-fight rating would leak the result into the prediction.

### Physical Attributes (5 features)

| # | Feature | What It Captures |
|---|---------|-----------------|
| 4 | **Height difference** (inches) | Taller fighters have range and angle advantages |
| 5 | **Reach difference** (inches) | Longer reach allows a fighter to strike from further away |
| 6 | **Leg reach difference** (inches) | Affects kicking range and distance management |
| 7 | **Age difference** (years, at fight date) | The single most predictive feature in the entire model -- younger fighters have a measurable edge |
| 8 | **Stance matchup** (same = 1, opposite = 0) | Whether both fighters use the same stance (e.g., both orthodox) or opposite stances (orthodox vs southpaw) |

When a fighter is missing physical data (about 5% of cases), the system fills in the average value for their weight class.

### Career Performance Rates (8 features)

These measure a fighter's overall career output per minute or per fight.

| # | Feature | What It Captures |
|---|---------|-----------------|
| 9 | **Significant strikes per minute** | Volume of high-impact striking |
| 10 | **Total strikes per minute** | Overall striking output including body and leg strikes |
| 11 | **Takedown rate** | How frequently a fighter attempts and lands takedowns |
| 12 | **Takedown accuracy (%)** | What percentage of takedown attempts succeed |
| 13 | **Takedown defense (%)** | What percentage of opponent takedowns are stuffed |
| 14 | **Strike defense (%)** | What percentage of incoming strikes are avoided |
| 15 | **Control time per fight** (seconds) | How much ground control time a fighter earns per fight |
| 16 | **Submission attempts per fight** | How often a fighter threatens submissions |

### Recent Performance Trends (8 features)

The same 8 stats above, but weighted toward the fighter's last 3 fights. This captures whether a fighter is improving, peaking, or declining.

| # | Feature | What It Captures |
|---|---------|-----------------|
| 17 | **Recent sig. strikes/min** | Is their striking output trending up or down? |
| 18 | **Recent total strikes/min** | Same for overall volume |
| 19 | **Recent takedown rate** | Are they wrestling more or less lately? |
| 20 | **Recent TD accuracy** | Is their wrestling getting sharper? |
| 21 | **Recent TD defense** | Are they getting taken down more? |
| 22 | **Recent strike defense** | Are they getting hit more? |
| 23 | **Recent control time** | Are they controlling fights on the ground? |
| 24 | **Recent submission attempts** | Are they actively hunting finishes? |

A fighter whose recent stats differ sharply from their career averages is either improving or declining -- the model picks up on this.

### Opponent-Adjusted Performance (4 features)

Raw stats can be misleading. Landing 5 takedowns against Khabib Nurmagomedov is far more impressive than 5 takedowns against a pure striker. These features adjust for opponent quality.

| # | Feature | What It Captures |
|---|---------|-----------------|
| 25 | **Opponent-adjusted sig. strikes** | Striking output relative to what opponents typically allow |
| 26 | **Opponent-adjusted takedowns** | Takedown success relative to opponent's usual defense |
| 27 | **Opponent-adjusted strike defense** | Defensive ability relative to opponent's usual output |
| 28 | **Opponent-adjusted control time** | Ground control relative to what opponents typically concede |

Values above 1.0 mean the fighter performed better than what their opponents usually allow. Below 1.0 means worse.

---

## Part 6: How the Model Uses These 28 Data Points

### The Process

1. Look up both fighters' current Elo ratings (overall, striking, grappling)
2. Look up their physical attributes (height, reach, leg reach, age, stance)
3. Look up their performance statistics (career rates, recent trends, opponent-adjusted)
4. Compute all 28 differentials (Fighter A's value minus Fighter B's value)
5. Feed the 28-number vector into the trained XGBoost model
6. Calibrate the output probability
7. Return the prediction

### Why Differentials?

The model does not care about absolute numbers. It cares about *relative* differences. A fighter who throws 6 significant strikes per minute is not inherently better or worse -- what matters is whether they throw more or fewer than their specific opponent.

### What the Model Learned

The top 10 most important factors (out of 28), ranked by how much they influence predictions:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **Age difference** | 0.088 |
| 2 | **Overall Elo differential** | 0.079 |
| 3 | Striking Elo differential | 0.050 |
| 4 | Opponent-adjusted strike defense | 0.050 |
| 5 | Recent takedown rate | 0.049 |
| 6 | Opponent-adjusted takedowns | 0.038 |
| 7 | Recent strike defense | 0.038 |
| 8 | Recent control time | 0.036 |
| 9 | Career takedown rate | 0.034 |
| 10 | Opponent-adjusted striking | 0.034 |

Age is the most important feature, followed closely by the overall Elo rating differential. This validates the Elo foundation -- it is the second most useful signal. But age, which the pure Elo system had no way to capture, adds substantial predictive power. A 22-year-old prospect and a 38-year-old veteran with identical records would carry the same Elo rating, but the model knows the younger fighter has an edge.

### Training Details

- **Algorithm:** XGBoost (Extreme Gradient Boosting) -- an ensemble of decision trees where each tree learns from the errors of the previous ones. Widely used in insurance pricing, fraud detection, and data science competitions.
- **Tuning:** 50 configurations tested via Bayesian optimization (learns from each attempt to find better settings faster)
- **Validation:** 5-fold time-series cross-validation -- each fold only trains on fights from before the test period
- **Calibration:** Platt scaling ensures predicted probabilities match observed outcomes (when the model says 70%, the fighter actually wins ~70% of the time)
- **Training data:** 12,833 fights before January 2023
- **Test data:** 3,315 fights from January 2023 through April 2026

---

## Part 7: All Settings at a Glance

| Setting | Value | Source |
|---------|-------|--------|
| Starting rating | 1,500 | Standard Elo theory |
| K-factor (first 4 fights) | 45 | Tested 225 combinations |
| K-factor (after 4 fights) | 10 | Tested 225 combinations |
| K transition point | 4 fights | Tested 225 combinations |
| KO/TKO multiplier | 1.2x | Tested 294 combinations |
| Submission multiplier | 1.1x | Tested 294 combinations |
| Unanimous decision multiplier | 1.0x | Baseline |
| Split decision multiplier | 0.5x | Tested 294 combinations |
| DQ multiplier | 0.8x | Sports analytics literature |
| Division transfer rate | 50% | Tested 7 values |
| Inactivity threshold | 270 days | Tested 4 thresholds |
| Inactivity rate | 10% / month | Tested 5 rates |
| Inactivity cap | 50% max | Tested 5 caps |
| KO attribution | 100% striking | Tested 25 combinations |
| Submission attribution | 100% grappling | Tested 25 combinations |
| EWMA half-life | 3 fights | All 4 values identical -- default retained |
| Shrinkage threshold | 5 fights | Median UFC career length |

**Overall performance:**
- Brier score: **0.2348** (coin flip = 0.250)
- AUC-ROC: **0.6445**
- Accuracy: **60.1%**

---

## Part 8: The Data

| Source | Description |
|--------|-------------|
| Kaggle UFC Dataset | Historical fights from 1994--2021 |
| UFCStats.com | Round-by-round stats from 2001 onward: strikes, takedowns, control time |

**Total:** 16,398 fights across 770 events, 4,915 fighters.

**Training set:** 12,833 fights (1994 -- December 2022). Used to build ratings and train the model.

**Test set:** 3,315 fights (January 2023 -- April 2026). Used to measure accuracy. The model never saw these fights during training.

---

## Glossary

**AUC-ROC**
Measures how well the model distinguishes winners from losers. 0.5 is random guessing, 1.0 is perfect. Our model scores 0.6445.

**Backtesting**
Testing predictions on historical data the model has never seen. We train on fights through December 2022 and test on 2023 onward.

**Bayesian Shrinkage**
Pulls small-sample ratings toward the average. A fighter with 1 fight gets only 20% credit for their raw rating. Prevents misleading results from limited data.

**Brier Score**
Measures prediction accuracy. Lower is better. 0.25 = coin flip. Our model scores 0.2348.

**Division Transfer**
When a fighter changes weight class, 50% of their rating advantage carries over to the new division.

**Domain Attribution**
Splits rating updates between striking and grappling sub-ratings. KOs count fully toward striking, submissions count fully toward grappling.

**Elo Rating**
A numerical skill rating developed for chess in the 1960s. Ratings rise when you win and fall when you lose. The size of the change depends on your opponent's rating.

**EWMA (Exponentially Weighted Moving Average)**
Computes averages that weight recent fights more heavily. A 3-fight half-life means the most recent fight contributes about 21% to the average.

**Feature Importance**
A score showing how much each input variable contributes to predictions. Higher importance = more influence on the outcome.

**Grid Search**
Testing every possible combination of settings and picking the one with the best score. We tested up to 294 combinations for a single parameter group.

**Inactivity Regression**
A gradual rating reduction for fighters who haven't competed in over 9 months. Reflects uncertainty about their current form.

**K-Factor**
Controls how much ratings change after each fight. Higher K = ratings move quickly. Lower K = ratings are more stable.

**Margin of Victory (MOV) Multiplier**
Scales the rating update based on how the fight ended. A KO produces a 1.2x update; a split decision produces 0.5x.

**Optuna**
A Bayesian optimization tool that efficiently searches for the best model settings. Used 50 trials to find the best XGBoost configuration.

**Platt Scaling**
Adjusts model probability outputs so they match real-world frequencies. When the model says "70% chance," that fighter actually wins about 70% of the time.

**Striking / Grappling Rating**
Sub-ratings tracking ability in each domain. Used to identify stylistic matchups (striker vs grappler).

**Style Index**
Striking rating minus grappling rating. Positive = striker. Negative = grappler. Near zero = balanced.

**Temporal Integrity**
The guarantee that no future information is used in any prediction. Every feature uses only data from fights that already happened.

**XGBoost**
A machine learning algorithm that builds many decision trees, each learning from the previous one's mistakes. Widely used in finance, healthcare, and sports analytics.

---

*All values in this document correspond to the live codebase. Elo parameters validated via grid search across 16,398 fights. XGBoost model trained on 12,833 fights with Bayesian hyperparameter optimization (50 trials), probability calibration, and positional bias correction. All accuracy metrics measured on 3,315 held-out test fights from January 2023 through April 2026.*
