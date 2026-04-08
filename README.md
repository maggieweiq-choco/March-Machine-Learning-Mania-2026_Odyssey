# NCAA March Madness 2026 — Win Probability Prediction

Kaggle competition entry for [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

Given two teams, predict the probability that the lower-ID team wins. Covers both Men's and Women's tournaments.

---

## Datasets

All data comes from the Kaggle competition. Key files used:

| File | What it provides |
|---|---|
| `MRegularSeasonDetailedResults` / `WRegularSeasonDetailedResults` | Box score stats for every regular season game (2003–2025) — used to compute team strength features |
| `MNCAATourneyDetailedResults` / `WNCAATourneyDetailedResults` | Detailed tournament game results — used for training |
| `MNCAATourneyCompactResults` / `WNCAATourneyCompactResults` | Historical tournament outcomes — training labels |
| `MNCAATourneySeeds` / `WNCAATourneySeeds` | Tournament seeds — strong baseline signal for upset probability |
| `MSampleSubmissionStage2` / `WSampleSubmissionStage2` | All 2026 matchup IDs to predict |

---

## Pipeline

### 1. Data Preparation

Men's and Women's data are combined into a unified dataset. All box score stats are adjusted for overtime (normalized to 40-minute equivalents). Each game is duplicated with T1 and T2 roles swapped so the model learns team-agnostic patterns rather than memorizing team IDs. Home/Away context is encoded as `+1 / -1 / NaN` from the `WLoc` field, with the sign flipped correctly for the swapped copy.

### 2. Feature Engineering

53 features are constructed from regular season data and joined to each tournament matchup as a T1/T2 pair.

**Per-team features (computed for both T1 and T2):**

| Feature | Why |
|---|---|
| **Tournament seed** | Strong baseline; one of the highest single-feature predictors |
| **ELO rating** (k=80) | Rolling Elo computed from all regular season results each year; captures head-to-head strength better than raw win/loss |
| **GLM quality rating** | Logistic regression on regular season point differentials; separates teams with similar records by margin of victory |
| **eFG%** (effective field goal %) | Shooting efficiency adjusted for 3-pointers |
| **FGR3** (3-point attempt rate) | Stylistic tendency toward perimeter offense |
| **FTR** (free throw rate) | Measures ability to get to the line |
| **AstR** (assist rate) | Proxy for ball movement and team cohesion |
| **Poss** (possessions per game) | Pace of play |
| **TOR** (turnover rate) | Ball security |
| **ORR / DRR** (offensive / defensive rebound rate) | Rebounding dominance |
| **Score_avg / PointDiff_avg** | Raw scoring output and margin |
| **OffRtg / DefRtg** (offensive / defensive rating) | Points per 100 possessions — pace-adjusted efficiency |
| **WinRatio14d** | Win rate over the final 14 days of the regular season — captures teams peaking at the right time |
| **away_wins** | Win rate in away games — measures performance under pressure |
| **awins** | Strength-of-schedule adjusted win metric |

**Differential features (T1 minus T2):** All per-team features above are also included as T1–T2 differences, plus `Seed_diff`, `elo_diff`, `diff_quality`, and `laplace_matchup`.

**laplace_matchup:** A head-to-head prior built from historical regular season matchups between the same two teams, smoothed with a Laplace prior to handle sparse data.

### 3. Model Training

Trained with **leave-one-season-out cross-validation** — each tournament year is held out in turn, models train on all other years (2003–2025, excluding 2020). This prevents look-ahead bias and mirrors the real prediction task.

Two model types are used:

- **XGBoost** — gradient boosted trees trained on point differential as the target, with predictions calibrated to probabilities via a per-season logistic regression model
- **CatBoost** — gradient boosted trees with built-in categorical handling

Final probability output uses manual threshold-based adjustment (`adjust_probability`) to correct the output distribution at the tails.

### 4. Evaluation

Per-season AUC is computed for both the seed baseline and the GLM quality rating:

- Seed AUC (baseline): **0.811**
- Quality AUC: **0.829**

The quality rating outperforms pure seed in 20 out of 22 seasons evaluated.

---

## File Structure

```
📁 data/                         ← Raw Kaggle competition CSV files
📁 archive/                      ← Earlier experimental notebooks (v1–v5)
📄 NCAA version7.ipynb           ← Latest model (XGBoost + CatBoost, ELO, 53 features)
📄 K-ncaa-25Using2026Data_Ensemble.ipynb  ← XGB + SVM ensemble experiment
📄 final-solution-ncaa-2025.ipynb         ← 2025 competition final submission
📄 predictions.csv / predictions.xls     ← 2026 submission output
📄 README.md
```

---

## Usage

1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
2. Place all CSV files in the `data/` folder (or update the path at the top of the notebook)
3. Open `NCAA version7.ipynb`
4. Run **Kernel > Restart & Run All**
5. `predictions.csv` is written to the working directory

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost catboost statsmodels scipy matplotlib seaborn tqdm
```
