# March-Machine-Learning-Mania-2026_Odyssey
## Data

| File | Description |
|---|---|
| `MRegularSeasonDetailedResults.csv` | Men's regular season box scores |
| `MNCAATourneyDetailedResults.csv` | Men's tournament results |
| `MNCAATourneySeeds.csv` | Men's tournament seeds |
| `WRegularSeasonDetailedResults.csv` | Women's regular season box scores |
| `WNCAATourneyDetailedResults.csv` | Women's tournament results |
| `WNCAATourneySeeds.csv` | Women's tournament seeds |

Men's and women's data are combined into a single pipeline. Data from **2003 onward** is used for men's, **2010 onward** for women's.

---

## Data Preparation

Each game is entered **twice** — once as `(T1=Winner, T2=Loser)` and once flipped. This symmetry ensures the model sees both perspectives, and `win=1` simply means T1 won.

**Overtime normalization:** All box score stats are divided by `(40 + 5×OT) / 40` to rescale overtime games back to a standard 40-minute baseline.

---

## Features

### Easy — Seeding

| Feature | Description |
|---|---|
| `men_women` | `1` = men's bracket, `0` = women's bracket |
| `T1_seed` | T1's tournament seed (1 = best, 16 = worst) |
| `T2_seed` | T2's tournament seed |
| `Seed_diff` | `T2_seed − T1_seed` — positive means T1 is seeded higher |

---

### Medium — Regular Season Averages

For each team, two perspectives are computed from regular season data:

**Team's own per-game stats:**

| Feature | Description |
|---|---|
| `T1_avg_Score` | Points scored per game |
| `T1_avg_FGA` | Field goal attempts per game |
| `T1_avg_OR` | Offensive rebounds per game |
| `T1_avg_DR` | Defensive rebounds per game |
| `T1_avg_Blk` | Blocks per game |
| `T1_avg_PF` | Personal fouls per game |
| `T1_avg_PointDiff` | Average scoring margin |

**What opponents did *against* this team** (reflects defensive strength):

| Feature | Description |
|---|---|
| `T1_avg_opponent_FGA` | Opponent shot attempts allowed per game |
| `T1_avg_opponent_Blk` | Blocks opponents recorded vs this team |
| `T1_avg_opponent_PF` | Fouls opponents committed against this team |

> All 13 features above exist for both T1 and T2 (`T2_avg_*`).

---

### Hard — Elo Ratings

Elo is a running skill rating updated after every regular season game. All teams start at **1000**. Wins vs strong opponents gain more; losses to weak opponents lose more.

expected_win  = 1 / (1 + 10^((opponent_elo − team_elo) / 400))
elo_change    = 100 × (1 − expected_win)



| Feature | Description |
|---|---|
| `T1_elo` | T1's Elo at end of regular season |
| `T2_elo` | T2's Elo at end of regular season |
| `elo_diff` | `T1_elo − T2_elo` — positive means T1 is stronger |

---

### Hardest — GLM Quality Score

A Generalized Linear Model is fit on regular season data with team dummy variables to estimate each team's **strength-of-schedule-adjusted** quality:

PointDiff ~ -1 + T1_TeamID + T2_TeamID



Each team gets a coefficient representing how much better or worse they perform than average, **after controlling for opponent strength**. Only tournament teams (and teams that beat a tournament team) are included to keep the regression tractable.

| Feature | Description |
|---|---|
| `T1_quality` | T1's GLM-estimated adjusted strength |
| `T2_quality` | T2's GLM-estimated adjusted strength |

---

## Model — XGBoost

**Target:** `PointDiff` (continuous regression, not binary). Predicting the margin gives richer signal than win/loss.

**Hyperparameters:**

| Parameter | Value | Effect |
|---|---|---|
| `eta` | `0.0093` | Low learning rate for stability |
| `num_boost_round` | `704` | Many rounds to compensate for low eta |
| `max_depth` | `4` | Shallow trees to prevent overfitting |
| `subsample` | `0.6` | 60% of rows sampled per tree |
| `num_parallel_tree` | `2` | Random forest–style bagging per round |
| `colsample_bynode` | `0.8` | 80% of features sampled per split |

**Training strategy:** Leave-one-season-out cross-validation — one model trained per held-out season. Final predictions are an **ensemble average** across all season models.

---

## Calibration — Spline

XGBoost outputs a predicted point margin. To convert to a win probability:

1. Clip predictions to `[−25, +25]`
2. Fit a degree-5 `UnivariateSpline` mapping `predicted_margin → historical win rate` from out-of-fold data
3. Clip final probabilities to `[0.01, 0.99]`

This is more accurate than a sigmoid because the margin-to-probability relationship is empirically fit rather than assumed.

---

## Submission

For each matchup in `SampleSubmissionStage2.csv`:

1. All season models predict a point margin
2. Predictions are **averaged** across all season models
3. A **+10% confidence boost** is applied when `pred < 0.85`
4. Six specific matchups are **manually overridden** with hardcoded probabilities

Output file: `predictions.csv` with columns `ID`, `Pred`.

---

## Key Variables

| Variable | Description |
|---|---|
| `tourney_data` | Main modeling dataframe — one row per team-pair per direction |
| `regular_data` | Source of season averages and Elo updates |
| `models[season]` | Dict of XGBoost models, one per held-out season |
| `spline_model` | Converts predicted margin to win probability |
| `oof_preds` | Out-of-fold predictions used to fit the spline |
| `X` | Submission dataframe with all features merged in |
| `t = 25` | Clipping threshold for the spline (±25 point margin) |
