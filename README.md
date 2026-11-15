# Household Electricity Bill Prediction

Predicting a household's monthly electricity bill (₹) from household characteristics
(rooms, occupants, house area, appliances, income, location), and comparing a linear
model with a random forest against a mean baseline.

CS3006 – Data Science for Engineers, Part A project.

## Dataset

- **Source:** Kaggle, *Household Monthly Electricity Bill* by gireeshs —
  https://www.kaggle.com/datasets/gireeshs/household-monthly-electricity-bill
- **File:** `Household energy bill data.csv` (1,000 rows × 10 columns)
- **Note:** the dataset author describes it as a mock-up dataset prepared for practice.
  See the *Limitations* section of the report.

| Column | Meaning |
|---|---|
| `num_rooms` | Number of rooms in the house |
| `num_people` | Number of people in the household |
| `housearea` | House area (sq ft) |
| `is_ac`, `is_tv` | Household has an AC / a TV (1 = yes) |
| `is_flat` | Home is a flat/apartment (1 = yes) |
| `ave_monthly_income` | Average monthly income (₹) |
| `num_children` | Number of children |
| `is_urban` | Home is in an urban area (1 = yes) |
| `amount_paid` | **Target** — monthly electricity bill (₹) |

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. put "Household energy bill data.csv" in the same folder as project.py, then run
python project.py
```

The script runs end to end with no manual steps. All randomness uses a fixed seed
(`random_state=42`), so every run gives identical results.

## What the script does

1. **Cleaning** – removes physically impossible rows (rooms ≤ 0, people ≤ 0, negative
   income, bill ≥ income): 1,000 → 905 rows.
2. **Outliers** – IQR rule on house area, income and bill; 9 rows flagged, all kept
   after inspection (physically plausible).
3. **EDA** – bill histogram, binary-feature box plots, joint group means, numeric
   scatter plots, correlation heatmap, imbalance and small-group checks.
4. **Encoding check + feature engineering** – no categorical columns need encoding;
   adds `area_per_person` and `people_per_room`.
5. **Modelling** – 80/20 train/test split; mean baseline, Linear Regression
   (scaled inside a Pipeline) and Random Forest compared with 5-fold cross-validation
   on the training set only.
6. **Evaluation** – single final evaluation of all models on the held-out test set;
   residual analysis and error by subgroup.
7. **Bill estimator** – saves the trained pipeline to `bill_model.joblib` and provides
   `estimate_bill(...)` for new households (set `ASK_USER = True` for interactive input).

## Outputs

| File | Content |
|---|---|
| `bill_hist.png` | Distribution of the monthly bill |
| `binary_vs_bill.png` | Bill by AC / TV / flat / urban |
| `numeric_vs_bill.png` | Numeric features vs bill |
| `corr_heatmap.png` | Correlation matrix |
| `error_analysis.png` | Predicted vs actual, residual distribution |
| `bill_model.joblib` | Trained Linear Regression pipeline (scaler + model) |

## Key results (test set, 181 households)

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Baseline (mean) | 151.7 | 187.2 | 0.000 |
| **Linear Regression** | **53.6** | **62.9** | **0.887** |
| Random Forest | 60.5 | 75.9 | 0.835 |

Linear Regression was selected using cross-validation (CV MAE ₹53.8 ± 1.3 vs
₹59.3 ± 4.4 for Random Forest) before the test set was used.
