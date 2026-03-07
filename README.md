# NCAA March Madness 2026 — Win Probability Prediction

Kaggle competition entry for [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

Predicts the win probability for every possible NCAA tournament matchup using an ensemble of XGBoost and SVR models trained on 20+ years of historical data.

---

## How It Works

### Features
- **Massey Ordinals** — aggregated rankings from 30+ rating systems (RPI, KenPom, BPI, etc.)
- **GLM quality ratings** — per-team offensive/defensive strength from regular season results
- **Recent win rate** — win rate in the last 14 days of the regular season
- **Adjusted win rate** — strength-of-schedule adjusted performance
- **Coach experience** — historical coach win rate and tenure
- **Tournament seed** — seed number and seed differential
- **Home/Away** — derived from game location (home/away/neutral)

### Models
- **XGBoost** with a custom Cauchy loss (robust to outliers)
- **SVR** + **NuSVR** with RBF kernel (median imputation + standard scaling)
- Final prediction = simple average of all three

### Validation
Leave-one-season-out cross-validation across 2003–2025 tournament seasons.

### Calibration
Out-of-fold predictions are calibrated to probabilities using a spline fit.

---

## Usage

1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
2. Set the data path at the top of the notebook
3. Run **Kernel > Restart & Run All**
4. `submission.csv` is generated automatically

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost statsmodels scipy
```

## Output

```
ID,Pred
2026_1101_1102,0.623
2026_1101_1103,0.441
```
