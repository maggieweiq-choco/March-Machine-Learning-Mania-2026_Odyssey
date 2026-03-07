# NCAA March Madness 2026 — Win Probability Prediction

Kaggle competition entry for [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

Given two teams, predict the probability that the lower-ID team wins. Covers both Men's and Women's tournaments.

---

## Datasets

All data comes from the Kaggle competition. Key files used:

| File | What it provides |
|---|---|
| `MRegularSeasonDetailedResults` / `WRegularSeasonDetailedResults` | Box score stats for every regular season game (2003–2025) — used to compute team strength features |
| `MNCAATourneyCompactResults` / `WNCAATourneyCompactResults` | Historical tournament outcomes — training labels |
| `MNCAATourneySeeds` / `WNCAATourneySeeds` | Tournament seeds — strong baseline signal for upset probability |
| `MTeamCoaches` | Head coach history — coaching experience is a proxy for program quality |
| `MMasseyOrdinals` | Daily ratings from 30+ external systems (RPI, KenPom, BPI, etc.) — captures ranking signals the raw box scores miss |
| `MSampleSubmissionStage2` / `WSampleSubmissionStage2` | All 2026 matchup IDs to predict |

---

## Pipeline

### 1. Feature Engineering

Features are built from regular season data and joined to each tournament matchup as a T1/T2 pair. Each game is also duplicated with T1 and T2 swapped so the model learns team-agnostic patterns rather than memorizing team IDs.

| Feature | Why |
|---|---|
| **Massey Ordinals** — aggregated rank across 30+ systems | Single best proxy for overall team quality; captures information that raw win/loss records miss |
| **GLM quality rating** — logistic regression on regular season point differentials | Separates teams with similar records by margin of victory |
| **Recent win rate** — last 14 days of regular season | Teams peaking at the right time tend to outperform seed expectations |
| **Adjusted win rate** — strength-of-schedule weighted | Corrects for teams padding records against weak opponents |
| **Coach experience** — career tournament win rate + seasons coached | Experienced coaches consistently outperform in single-elimination formats |
| **Tournament seed** + seed differential | Strong baseline; seed gap is one of the highest single-feature predictors |
| **Home/Away** — derived from `WLoc` (H/A/N) | Regular season home/away context; neutral for tournament games |

### 2. Model Training

Trained with **leave-one-season-out cross-validation** — each tournament year is held out in turn, models train on all other years. This prevents look-ahead bias and mirrors the real prediction task.

Three models are ensembled:

- **XGBoost** with a custom Cauchy loss — more robust to blowout games and upsets than squared error
- **SVR** (RBF kernel) — captures non-linear relationships between features without overfitting on a small tournament dataset
- **NuSVR** (RBF kernel) — provides a second SVM estimate with different regularization behavior

SVR/NuSVR inputs are median-imputed (handles missing Massey data for some teams) and standardized before fitting.

### 3. Ensemble + Calibration

Final prediction = simple average of XGBoost, SVR, and NuSVR outputs.

Raw scores are then calibrated to well-formed probabilities using a **spline fit** on out-of-fold predictions. This corrects the output distribution to match the Brier score metric.

---

## Usage

1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
2. Set the data directory path at the top of `Kaggele_ncaa-2026.ipynb`
3. Run **Kernel > Restart & Run All**
4. `submission.csv` is written to the working directory

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost statsmodels scipy
```
