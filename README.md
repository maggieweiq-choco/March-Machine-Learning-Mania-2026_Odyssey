# NCAA March Madness 2026 — Win Probability Prediction

> 🏆 **Ranked 37th out of 3,462 teams (Top 1.1%) — Kaggle March Machine Learning Mania 2026**

> **Kaggle Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
> **Task:** Given any two teams, predict the probability that the lower-ID team wins. Covers both Men's and Women's NCAA tournaments.  
> **Evaluation Metric:** Brier Score (lower is better) — penalizes confident wrong predictions more severely than uncertain ones.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-calibration-green) ![Kaggle](https://img.shields.io/badge/Kaggle-Top%201.1%25-gold)

**Tech Stack:** `Python` · `XGBoost` · `scikit-learn` · `statsmodels` · `ELO Rating System` · `Probability Calibration` · `Leave-One-Season-Out CV` · `GLM Quality Rating`

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Data Sources](#2-data-sources)
3. [Methodology Overview](#3-methodology-overview)
4. [Feature Engineering (53 Features)](#4-feature-engineering-53-features)
5. [Modeling Architecture](#5-modeling-architecture)
6. [Calibration & Post-Processing](#6-calibration--post-processing)
7. [Results & Evaluation](#7-results--evaluation)
8. [Repository Structure](#8-repository-structure)
9. [Reproduction Guide](#9-reproduction-guide)
10. [Dependencies](#10-dependencies)

---

## 1. Problem Framing

The NCAA tournament is a single-elimination bracket where each game's outcome is inherently noisy — upsets are common, especially in early rounds. The prediction task is not merely binary classification (win/lose), but **probability calibration**: assigning a well-formed probability $p \in (0,1)$ such that across all games, teams assigned probability $p$ of winning actually win approximately $p$ fraction of the time.

This distinguishes the problem from standard classification tasks. A model that always predicts 0.95 for the favorite may be wrong more often than a model that predicts 0.65 with appropriate uncertainty. The Brier score $BS = \frac{1}{N}\sum_{i=1}^{N}(f_i - o_i)^2$ directly measures this calibration quality.

**Key design principles in this solution:**
- Predict **point margin** (a continuous, informative signal) rather than directly predicting win probability, then calibrate margin to probability
- Use **leave-one-season-out cross-validation** to simulate real-world temporal prediction with zero look-ahead bias
- Build features that capture **multiple independent dimensions of team quality** beyond raw win/loss records

---

## 2. Data Sources

All data is sourced from the Kaggle competition dataset.

| File | Role in Pipeline |
|---|---|
| `MRegularSeasonDetailedResults` / `WRegularSeasonDetailedResults` | Primary source for all per-team box score features (2003–2025) |
| `MNCAATourneyDetailedResults` / `WNCAATourneyDetailedResults` | Tournament game box scores — used for training targets |
| `MNCAATourneyCompactResults` / `WNCAATourneyCompactResults` | Historical tournament win/loss — training labels |
| `MNCAATourneySeeds` / `WNCAATourneySeeds` | Seed numbers — one of the strongest single predictors |
| `MSampleSubmissionStage2` / `WSampleSubmissionStage2` | All 2026 matchup IDs requiring predictions |

Men's and Women's data are **pooled into a unified dataset** with a binary `men_women` indicator feature. This doubles training data and encourages the model to learn gender-agnostic basketball dynamics, while retaining the ability to account for systematic differences between the two tournaments.

Data is filtered to seasons >= 2003, the first year with detailed box score availability.

---

## 3. Methodology Overview

The full pipeline consists of five stages:

```
Raw Game Logs
     |
     v
[Stage 1] Data Normalization
  - Overtime adjustment (stats scaled to 40-min equivalent)
  - Team-role symmetry: each game duplicated with T1<->T2 swapped
  - Home/Away encoded as +1 / -1 / NaN (with sign correction on swap)
     |
     v
[Stage 2] Feature Engineering
  - Per-team season-level aggregates (box score stats, efficiency metrics)
  - ELO ratings computed from full regular season game-by-game sequence
  - GLM quality ratings from logistic regression on point differentials
  - Recency-weighted win rates, away performance, head-to-head priors
  - Differential features (T1 - T2) for all per-team metrics
     |
     v
[Stage 3] Model Training — XGBoost (Point Margin Regression)
  - Target: signed point differential (T1_Score - T2_Score)
  - Leave-one-season-out CV across 22 seasons (2003-2025, excl. 2020)
  - 22 separate XGBoost models trained, one held-out season each
     |
     v
[Stage 4] Probability Calibration
  - Per-season logistic regression: pred_margin -> P(T1 wins)
  - Grid search over C, solver, penalty, class_weight (Brier-optimized)
  - Manual tail adjustment: fine-tune extreme probabilities
     |
     v
[Stage 5] Inference
  - Apply all 22 calibrated models to 2026 matchup grid
  - Average probabilities across all model folds
  - Output: predictions.csv
```

---

## 4. Feature Engineering (53 Features)

Features are computed from regular season data for each team, then joined to each tournament matchup as a T1/T2 pair.

### 4.1 Overtime Normalization

All box score statistics are scaled by an overtime adjustment factor before aggregation:

```
adj_stat = raw_stat / ((40 + 5 * NumOT) / 40)
```

This ensures that games that go to overtime do not inflate per-game averages.

### 4.2 Team Symmetry (Data Augmentation)

Each game is duplicated with T1 and T2 roles swapped. This prevents the model from learning spurious patterns based on which team ID is numerically lower. The Home/Away encoding is sign-flipped correctly on the swapped copy, so venue context remains accurate.

### 4.3 Per-Team Features

The following features are computed independently for both T1 and T2:

| Feature | Description | Motivation |
|---|---|---|
| `seed` | Tournament seed (1–16) | Strongest single predictor; encodes the selection committee's holistic team assessment |
| `elo` | ELO rating at end of regular season | Sequentially updated from game-by-game results (k=80); captures relative strength without schedule bias |
| `quality` | GLM quality rating | Logistic regression on point differentials; separates teams with similar records by margin of victory |
| `eFG` | Effective field goal % = (FGM + 0.5 x FGM3) / FGA | Shooting efficiency adjusted for 3-point value |
| `FGR3` | 3-point attempt rate = FGA3 / FGA | Stylistic tendency; perimeter-heavy teams behave differently under bracket pressure |
| `FTR` | Free throw rate = FTA / FGA | Aggressiveness and ability to draw fouls — sustainable under tournament pressure |
| `AstR` | Assist rate = Ast / FGM | Proxy for ball movement and team cohesion |
| `Poss` | Possessions per game | Pace of play; essential for comparing efficiency metrics across teams |
| `TOR` | Turnover rate = TO / Poss | Ball security; turnover-prone teams are punished more in single-elimination |
| `ORR` | Offensive rebound rate | Second-chance scoring — often decisive in close games |
| `DRR` | Defensive rebound rate | Limits opponent second chances |
| `Score_avg` | Average points scored (OT-adjusted) | Raw offensive output |
| `PointDiff_avg` | Average point differential | Overall dominance relative to opponents faced |
| `OffRtg` | Points scored per 100 possessions | Pace-adjusted offensive efficiency |
| `DefRtg` | Points allowed per 100 possessions | Pace-adjusted defensive efficiency |
| `WinRatio14d` | Win rate in final 14 days of regular season | Captures momentum — teams peaking at tournament time |
| `away_wins` | Win rate in away games | Performance under hostile conditions |
| `awins` | Strength-of-schedule adjusted win metric | Corrects for record inflation against weak opponents |

### 4.4 ELO Rating System

ELO ratings are computed from scratch each season using the full sequence of regular season games:

- **Initial ELO:** 1000 for all teams at the start of each season
- **K-factor:** 80 (tuned empirically)
- **ELO width:** 400 (standard Elo scaling)
- **Expected win probability:** `E_A = 1 / (1 + 10^((ELO_B - ELO_A) / 400))`
- **Update rule:** winner gains `K * (1 - E_A)`, loser loses same amount

ELO resets each season to prevent historical program prestige from overshadowing current-year performance.

### 4.5 GLM Quality Rating

A logistic regression (GLM) model is fitted on regular season game data using point differential as the predictor and win as the binary outcome. The resulting predicted log-odds serves as a continuous team quality score. This is more expressive than win/loss ratio because it separates teams that win by 20 points versus teams that barely survive.

**Validation:** Quality AUC (0.829) outperforms seed AUC (0.811) across the full historical dataset, and beats seed in **20 out of 22 seasons evaluated**.

### 4.6 Laplace Head-to-Head Prior (`laplace_matchup`)

For each unique T1-T2 pair that has appeared in the historical regular season record, a head-to-head win rate is computed with Laplace smoothing to handle sparse data. This captures any historically documented competitive dynamics between specific programs. The majority of matchups default to 0.5 (uninformative), providing a soft informative prior only where genuine history exists.

### 4.7 Differential Features (15 additional features)

For each of the 15 per-team continuous features, a T1-T2 difference is computed as an additional explicit feature. These allow the model to directly learn how quality gaps between opponents translate into win probability, rather than relying on the model to implicitly compute differences internally.

**Total feature count: 53**
- 1 `men_women` indicator
- 18 T1 features + 18 T2 features = 36
- 15 differential features + 1 `laplace_matchup` = 16

---

## 5. Modeling Architecture

### 5.1 Why Predict Point Margin, Not Win Probability?

Directly predicting a binary win/loss label discards a large amount of information. A 1-point overtime win and a 30-point blowout look identical to a classifier, but carry very different signals about team quality. Predicting point margin as a continuous regression target extracts this richer signal and results in better-calibrated downstream probabilities.

### 5.2 XGBoost Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `reg:squarederror` | Point differential is a continuous regression target |
| `booster` | `gbtree` | Gradient boosted decision trees |
| `tree_method` | `hist` | Histogram-based splits — efficient for this feature dimensionality |
| `grow_policy` | `lossguide` | Leaf-wise growth for better fit on asymmetric data distributions |
| `num_parallel_tree` | 10 | Random forest-style bagging within each boosting round — implicit ensemble |
| `eta` | 0.01 | Low learning rate for smooth, stable convergence |
| `max_depth` | 3 | Shallow trees prevent overfitting on the small tournament dataset (~500 games/season) |
| `min_child_weight` | 20 | Requires meaningful sample count per leaf — strong regularization |
| `subsample` | 0.35 | Aggressive row subsampling further prevents overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `gamma` | 10 | High minimum loss reduction threshold — only meaningful splits allowed |
| `num_boost_round` | 500 | Fixed rounds across all folds for consistent model structure |

### 5.3 Cross-Validation: Leave-One-Season-Out

One XGBoost model is trained per tournament season (22 models total). For held-out season `s`:
- **Train set:** All tournament data from all seasons except `s`
- **Validation set:** Tournament games from season `s` only

This strictly respects temporal ordering, ensuring zero information leakage from future seasons. It also mirrors the actual competition task: use past years to predict a future year you have never seen.

**Out-of-fold MAE across all 22 seasons: 9.21 points**

---

## 6. Calibration & Post-Processing

### 6.1 Logistic Calibration: Margin to Probability

Raw XGBoost margin predictions are passed through a per-season logistic regression to convert them into valid win probabilities:

```
P(T1 wins | predicted_margin) = sigmoid(b0 + b1 * predicted_margin)
```

The calibration model is itself trained leave-one-season-out (same fold structure as XGBoost) to avoid overfitting the calibration curve.

**Hyperparameter grid search:**
- C: 13 values log-spaced from 1e-4 to 1e2
- Solvers: liblinear, lbfgs, saga
- Penalties: L1, L2
- Class weights: None, balanced

**Best configuration found:** `C=0.001, solver=saga, penalty=L2, class_weight=None`  
**Best OOF Brier Score (logistic only):** 0.16549

### 6.2 Manual Tail Adjustment

After logistic calibration, a threshold-based correction is applied to the tails of the probability distribution to fine-tune extreme predictions:

```
If p > upper_threshold: p = p + upper_delta
If p < lower_threshold: p = p - lower_delta
```

**Optimized parameters:**

| Parameter | Value |
|---|---|
| `upper_threshold` | 0.75 |
| `upper_delta` | +0.01 |
| `lower_threshold` | 0.25 |
| `lower_delta` | 0.08 (pulls toward 0.5) |

**Final OOF Brier Score:** 0.16543 (improvement over logistic-only baseline of 0.16549)

### 6.3 Final Prediction Ensemble

For each 2026 matchup, predictions are generated from all 22 season-specific calibrated models and averaged. This provides a natural ensemble effect across different training configurations, reducing variance in the final output.

---

## 7. Results & Evaluation

### 7.1 Model Performance Summary

| Model / Signal | Metric | Score |
|---|---|---|
| Seed difference baseline | ROC-AUC | 0.811 |
| GLM quality difference | ROC-AUC | 0.829 |
| XGBoost OOF (point margin regression) | Mean MAE | 9.21 pts |
| Logistic calibration OOF | Brier Score | 0.16549 |
| **Final model (logistic + tail adjustment)** | **Brier Score** | **0.16543** |
| 🏆 **Kaggle Leaderboard** | **Final Rank** | **37 / 3,462 (Top 1.1%)** |

### 7.2 Key Finding: Quality Rating Outperforms Seeds in 20/22 Seasons

The GLM quality rating — derived purely from regular season point differentials — consistently outperforms the tournament seed as a predictor of game outcomes. This suggests that the committee's seeding process, while strong, leaves measurable predictive signal on the table that statistical models can capture.

### 7.3 Per-Season XGBoost Out-of-Fold MAE

| Season | MAE (pts) | Season | MAE (pts) |
|---|---|---|---|
| 2003 | 8.86 | 2015 | 7.93 |
| 2004 | 7.98 | 2016 | 9.97 |
| 2005 | 7.98 | 2017 | 9.75 |
| 2006 | 8.34 | 2018 | 10.59 |
| 2007 | 7.76 | 2019 | 9.18 |
| 2008 | 9.58 | 2021 | 10.56 |
| 2009 | 9.12 | 2022 | 10.22 |
| 2010 | 8.46 | 2023 | 9.66 |
| 2011 | 9.34 | 2024 | 9.30 |
| 2012 | 8.12 | 2025 | 10.03 |
| 2013 | 9.81 | **Mean** | **9.21** |
| 2014 | 10.07 | | |

Higher MAE in later seasons (2018, 2021, 2022) likely reflects increased parity in modern college basketball rather than model degradation.

---

## 8. Repository Structure

```
March-Machine-Learning-Mania-2026_Odyssey/
|
+-- data/                                  <- Raw Kaggle CSV files (not tracked in git)
|
+-- archive/                               <- Earlier experimental notebooks
|   +-- NCAA version1.ipynb                   Seed-only baseline
|   +-- NCAA version2.ipynb                   Adding GLM quality rating
|   +-- NCAA version3.ipynb                   ELO rating experiments
|   +-- NCAA version5.ipynb                   SVR / NuSVR ensemble experiments
|   +-- Kaggle_ncaa-2026.ipynb                Early Stage 1 submission draft
|   +-- Kaggle_ncaa-2026_2.ipynb              Revised submission draft
|
+-- NCAA version7.ipynb                    <- MAIN: Current best model (this README)
+-- K-ncaa-25Using2026Data_Ensemble.ipynb  <- XGB + SVM ensemble experiment
+-- final-solution-ncaa-2025.ipynb         <- 2025 competition final submission
|
+-- predictions.csv                        <- Stage 2 submission file
+-- predictions.xls                        <- Submission (Excel format)
+-- README.md
+-- .gitignore
```

---

## 9. Reproduction Guide

**Environment setup**

```bash
# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost catboost statsmodels scipy matplotlib seaborn tqdm
```

> **Requirements:** Python >= 3.10 · RAM >= 8GB recommended (22-fold training loop)

**Step 1 — Download data**

Download all competition CSV files from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and place them in the `data/` directory.

**Step 2 — Configure path**

Open `NCAA version7.ipynb`. In the data loading cell, update the file paths to point to your `data/` directory:

```python
M_regular_results = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
```

**Step 3 — Run the notebook**

```
Kernel > Restart & Run All
```

**Step 4 — Collect output**

`predictions.csv` is written to the working directory on completion.

**Expected runtime:** 15–30 minutes, dominated by the 22-fold XGBoost training loop and the logistic regression grid search (~78 solver/penalty/C combinations × 22 folds).

---

## 10. Dependencies

```bash
pip install pandas numpy scikit-learn xgboost catboost statsmodels scipy matplotlib seaborn tqdm
```

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and merging |
| `numpy` | Numerical computation |
| `scikit-learn` | Logistic regression, metrics, preprocessing |
| `xgboost` | Primary gradient boosting model |
| `catboost` | Secondary gradient boosting model |
| `statsmodels` | GLM quality rating computation |
| `scipy` | Statistical utilities |
| `matplotlib` / `seaborn` | Visualization and EDA |
| `tqdm` | Progress bars |

**Python >= 3.10 required** — walrus operator (`:=`) used in evaluation cells.
