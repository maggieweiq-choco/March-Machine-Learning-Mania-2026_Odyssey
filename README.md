# NCAA March Madness 2026: Ensemble Learning for Tournament Outcome Prediction

## Abstract

This project develops a machine learning pipeline to predict the win probability of NCAA Men's and Women's Basketball Tournament matchups for the 2026 season. We construct a rich feature set from historical game data, train an ensemble of gradient boosting and support vector regression models under a leave-one-season-out cross-validation scheme, and apply spline-based probability calibration to produce well-formed probabilistic outputs.

---

## 1. Problem Formulation

Given two teams $T_1$ and $T_2$, the task is to estimate $P(\text{T}_1 \text{ wins})$ for every possible first-round through championship matchup in the 2026 NCAA tournament. Predictions are evaluated using the Brier score on the held-out tournament games submitted to the Kaggle March Machine Learning Mania 2026 competition.

---

## 2. Data

All data is sourced from the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition dataset, covering both Men's (M) and Women's (W) tournaments.

### 2.1 Game Results

| File | Description |
|---|---|
| `MRegularSeasonDetailedResults.csv` | Men's regular season game results with box score stats (2003–2025) |
| `WRegularSeasonDetailedResults.csv` | Women's regular season game results with box score stats |
| `MNCAATourneyCompactResults.csv` | Men's tournament game results (2003–2025) |
| `WNCAATourneyCompactResults.csv` | Women's tournament game results |

Each row encodes the winning team (`WTeamID`, `WScore`) and losing team (`LTeamID`, `LScore`), along with game location (`WLoc`: H/A/N) and day number within the season.

### 2.2 Team Metadata

| File | Description |
|---|---|
| `MTeams.csv` / `WTeams.csv` | Team IDs and names |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | Tournament seeds per team per season (e.g. W01, X16) |
| `MTeamCoaches.csv` | Head coach assignments per team per season, with first/last day coached |

### 2.3 Third-Party Rankings

| File | Description |
|---|---|
| `MMasseyOrdinals.csv` | Daily ratings from 30+ external ranking systems (e.g. RPI, KenPom, BPI) aggregated at end of regular season |

Rankings are averaged across all available systems to produce a single `MasseyRank` per team per season.

### 2.4 Submission Template

| File | Description |
|---|---|
| `MSampleSubmissionStage2.csv` / `WSampleSubmissionStage2.csv` | All valid 2026 matchup IDs to predict |

Each `ID` is formatted as `Season_LowerTeamID_HigherTeamID`.

---

## 3. Feature Engineering

Features are computed at the team level from regular season data and then joined to each tournament matchup as a T1/T2 pair. To enforce permutation invariance, each training game is duplicated with T1 and T2 swapped, so the model must learn team-agnostic relationships.

| Feature Group | Features | Description |
|---|---|---|
| Massey Rankings | `T1_MasseyRank`, `T2_MasseyRank` | Mean end-of-season ranking across all Massey rating systems |
| GLM Quality | `glm_quality_T1`, `glm_quality_T2` | Per-team offensive/defensive quality estimated by logistic regression on regular season point differentials |
| Recent Performance | `T1_WinRatio14d`, `T2_WinRatio14d` | Win rate in the final 14 days of the regular season |
| Adjusted Performance | `T1_awins`, `T2_awins` | Strength-of-schedule-adjusted win rate |
| Coaching | `T1_coach_exp`, `T1_coach_winrate` | Cumulative coaching experience (seasons) and career tournament win rate |
| Seeding | `T1_seed`, `T2_seed` | Tournament seed; seed differential as an additional feature |
| Home/Away | `T1_Home`, `T2_Home` | Game location indicator: home (+1), away (−1), neutral (NaN) |

---

## 4. Methodology

### 4.1 Cross-Validation

A **leave-one-season-out** (LOSO) scheme is used for validation. For each held-out tournament year $t$, models are trained on all seasons $\{s : s \neq t\}$ and evaluated on the tournament games of season $t$. This mirrors the temporal structure of the prediction task and avoids look-ahead bias.

### 4.2 Models

Three regression models are trained per fold to predict the point differential (a continuous proxy for win probability):

**XGBoost** with a custom Cauchy loss function:

$$\mathcal{L}_{\text{Cauchy}}(r) = \log\left(1 + \left(\frac{r}{\delta}\right)^2\right)$$

where $r$ is the residual and $\delta$ is a scale parameter. This loss is more robust to outliers than squared error, which is desirable given the high variance of tournament outcomes.

**SVR / NuSVR** (scikit-learn) with RBF kernel. Prior to fitting, features are median-imputed and standardized:

$$\tilde{x} = \frac{x - \mu_{\text{median}}}{\sigma}$$

### 4.3 Ensemble

Final predictions are the simple average of the three model outputs:

$$\hat{y} = \frac{1}{3}\left(\hat{y}_{\text{XGB}} + \hat{y}_{\text{SVR}} + \hat{y}_{\text{NuSVR}}\right)$$

### 4.4 Probability Calibration

Raw ensemble scores are calibrated to $[0, 1]$ probabilities using a monotonic `UnivariateSpline` fit on out-of-fold predictions against binary win/loss outcomes. This corrects for distributional shift between the regression target and the probability scale required by the evaluation metric.

---

## 5. Repository Structure

```
Kaggele_ncaa-2026.ipynb   # Full pipeline: feature engineering, training, prediction, submission
README.md
```

---

## 6. Dependencies

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
xgboost >= 1.7
statsmodels >= 0.14
scipy >= 1.10
```

Install:

```bash
pip install pandas numpy scikit-learn xgboost statsmodels scipy
```

---

## 7. Reproducing Results

1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) and set the data directory path in the notebook.
2. Run **Kernel > Restart & Run All**.
3. `submission.csv` is written to the working directory upon completion.

---

## 8. Submission Format

```
ID,Pred
2026_1101_1102,0.623
2026_1101_1103,0.441
...
```

`ID` encodes `Season_LowerTeamID_HigherTeamID`. `Pred` is the probability that the lower-ID team wins.
