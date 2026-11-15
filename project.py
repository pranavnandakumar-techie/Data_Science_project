import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.stdout.reconfigure(encoding="utf-8")
df=pd.read_csv("Household energy bill data.csv")
print("Length before: ",len(df))
df=df[df["num_rooms"]>0]
df=df[df["num_people"]>0]
df=df[(df["is_ac"]).isin([0,1])]
df=df[(df["is_tv"]).isin([0,1])]
df=df[(df["is_flat"]).isin([0,1])]
df=df[(df["ave_monthly_income"]>=0) & (df["amount_paid"]<df["ave_monthly_income"])]
df=df[df["num_children"]>=0]
df=df[(df["is_urban"]).isin([0,1])]

print("Length after: ",len(df))
pd.set_option('display.max_columns', None)  # show every column, no "..."
pd.set_option('display.width', 200)         # let each printed line be up to 200 characters wide
q1_house=df["housearea"].quantile(0.25)
q3_house=df["housearea"].quantile(0.75)
iqr_house=q3_house-q1_house
lout_house=q1_house-1.5*iqr_house
mout_house=q3_house+1.5*iqr_house

q1_income=df["ave_monthly_income"].quantile(0.25)
q3_income=df["ave_monthly_income"].quantile(0.75)
iqr_income=q3_income-q1_income
lout_income=q1_income-1.5*iqr_income
mout_income=q3_income+1.5*iqr_income

q1_paid=df["amount_paid"].quantile(0.25)
q3_paid=df["amount_paid"].quantile(0.75)
iqr_paid=q3_paid-q1_paid
lout_paid=q1_paid-1.5*iqr_paid
mout_paid=q3_paid+1.5*iqr_paid
print(lout_house, mout_house, lout_income, mout_income, lout_paid, mout_paid)

print(df[(df["housearea"] < lout_house) | (df["housearea"] > mout_house) |
         (df["ave_monthly_income"] < lout_income) | (df["ave_monthly_income"] > mout_income) |
         (df["amount_paid"] < lout_paid) | (df["amount_paid"] > mout_paid)])
mean_amt=statistics.mean(df["amount_paid"])
median_amt=statistics.median(df["amount_paid"])
plt.hist(df["amount_paid"],bins=30)
plt.axvline(mean_amt,color='red',linestyle='dashed',linewidth=2,label=f'Mean: {mean_amt:.2f}')
plt.axvline(median_amt,color='blue',linestyle='dashed',linewidth=2,label=f'Median: {median_amt:.2f}')
print(df.groupby(["is_ac","is_tv","is_flat","is_urban"])["amount_paid"].agg(meanof="mean"))
features = ["num_rooms", "num_people", "housearea", "ave_monthly_income", "num_children"]
xlabels  = ["Number of rooms", "Number of people", "House area (sq ft)",
            "Avg monthly income (₹)", "Number of children"]

fig, ax = plt.subplots(1, 5, figsize=(22, 4))

for i, col in enumerate(features):
    ax[i].scatter(df[col], df["amount_paid"], color='blue', s=10, alpha=0.3)
    ax[i].set_xlabel(xlabels[i])
    ax[i].set_ylabel("Monthly bill (₹)")
    ax[i].set_title(f"{col} vs bill  (r = {df[col].corr(df['amount_paid']):.2f})")

fig.suptitle("Numeric features vs monthly electricity bill")
fig.tight_layout()
fig.savefig("numeric_vs_bill.png", dpi=150)


corr=df.corr()
print(corr)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, label="Pearson r")
ax.set_title("Correlation matrix")
fig.tight_layout()
fig.savefig("corr_heatmap.png", dpi=150)

# 2. Imbalance: % of households with each binary feature = 1
binary = ["is_ac", "is_tv", "is_flat", "is_urban"]
print((df[binary].mean() * 100).round(1))

# 3. Small groups: how many households in each value
print(df["num_children"].value_counts().sort_index())
print(df["num_rooms"].value_counts().sort_index())
print(df["num_people"].value_counts().sort_index())
# ---------- 1. Check data types ----------
print(df.dtypes)

# ---------- 2. Check unique values in every column ----------
for col in df.columns:
    n_unique = df[col].nunique()
    sample = sorted(df[col].unique())[:8]
    print(f"{col:20s}  unique={n_unique:4d}  values={sample}")

# ---------- 3. Classify each feature ----------
binary_cols     = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
count_cols      = ["num_rooms", "num_people", "num_children"]
continuous_cols = ["housearea", "ave_monthly_income"]
text_cols       = df.select_dtypes(include=["object", "string"]).columns.tolist()

print("Binary     :", binary_cols)
print("Counts     :", count_cols)
print("Continuous :", continuous_cols)
print("Text (needs one-hot):", text_cols)

# ---------- Feature engineering ----------
df["area_per_person"] = df["housearea"] / df["num_people"]   # space per person (sq ft)
df["people_per_room"] = df["num_people"] / df["num_rooms"]   # crowding

compare = ["area_per_person", "people_per_room", "housearea", "num_people", "num_rooms"]
print(df[compare].corrwith(df["amount_paid"]).round(3))

X = df.drop(columns="amount_paid")
y = df["amount_paid"]

# ---------- 2. Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------- 3. Baseline: always predict the training mean ----------
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

# ---------- 4. Score it ----------
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)
print(f"Baseline  MAE = ₹{mae:.1f}   RMSE = ₹{rmse:.1f}   R² = {r2:.3f}")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------- 1. Define the CV scheme ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 2. Build the model as a pipeline: scale -> regress ----------
lin_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

# ---------- 3. Cross-validate on TRAIN only ----------
mae_scores = -cross_val_score(lin_model, X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error")
r2_scores  =  cross_val_score(lin_model, X_train, y_train, cv=cv, scoring="r2")

print("Linear  CV MAE per fold:", mae_scores.round(1))
print(f"Linear  CV MAE = ₹{mae_scores.mean():.1f} ± {mae_scores.std():.1f}")
print(f"Linear  CV R²  = {r2_scores.mean():.3f}")

# ---------- 4. Same CV for the baseline, for a fair comparison ----------
base_mae = -cross_val_score(DummyRegressor(strategy="mean"), X_train, y_train,
                            cv=cv, scoring="neg_mean_absolute_error")
print(f"Baseline CV MAE = ₹{base_mae.mean():.1f} ± {base_mae.std():.1f}")

# ---------- 5. Fit on all training data and look at the coefficients ----------
lin_model.fit(X_train, y_train)
coefs = pd.Series(lin_model.named_steps["lr"].coef_, index=X_train.columns)
print(coefs.sort_values(ascending=False).round(1))
from sklearn.ensemble import RandomForestRegressor

# ---------- 1. Define the model ----------
rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

# ---------- 2. Cross-validate on the SAME folds ----------
rf_mae = -cross_val_score(rf_model, X_train, y_train, cv=cv,
                          scoring="neg_mean_absolute_error")
rf_r2  =  cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="r2")

print("RF  CV MAE per fold:", rf_mae.round(1))
print(f"RF  CV MAE = ₹{rf_mae.mean():.1f} ± {rf_mae.std():.1f}")
print(f"RF  CV R²  = {rf_r2.mean():.3f}")

# ---------- 3. Overfitting check: train error vs CV error ----------
rf_model.fit(X_train, y_train)
train_mae = mean_absolute_error(y_train, rf_model.predict(X_train))
print(f"RF  TRAIN MAE = ₹{train_mae:.1f}")

# ---------- 4. Feature importance ----------
imp = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print(imp.sort_values(ascending=False).round(3))

# ================= FINAL EVALUATION ON TEST SET =================
def evaluate(name, model):
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2   = r2_score(y_test, pred)
    print(f"{name:10s} TEST  MAE = ₹{mae:.1f}   RMSE = ₹{rmse:.1f}   R² = {r2:.3f}")
    return pred

# both models were already fit on the full X_train above
evaluate("Baseline", baseline)
lin_pred = evaluate("Linear", lin_model)
evaluate("RF", rf_model)

# ================= ERROR ANALYSIS (chosen model: linear) =================
residuals = y_test - lin_pred
print(f"Residual mean = ₹{residuals.mean():.1f}   std = ₹{residuals.std():.1f}")
print("Median |error| = ₹%.1f   90th percentile |error| = ₹%.1f"
      % tuple(np.percentile(residuals.abs(), [50, 90])))

# --- Slice the error by group ---
err = X_test.assign(abs_err=residuals.abs())
print(err.groupby("is_urban")["abs_err"].mean().round(1))
print(err.groupby("num_children")["abs_err"].agg(["mean", "count"]).round(1))

# --- Plots ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(y_test, lin_pred, s=15, alpha=0.6)
lims = [y_test.min(), y_test.max()]
ax[0].plot(lims, lims, "r--", label="Perfect prediction")
ax[0].set_xlabel("Actual bill (₹)")
ax[0].set_ylabel("Predicted bill (₹)")
ax[0].set_title("Linear regression: predicted vs actual (test set)")
ax[0].legend()

ax[1].hist(residuals, bins=25)
ax[1].axvline(0, color="red", linestyle="--")
ax[1].set_xlabel("Residual = actual − predicted (₹)")
ax[1].set_ylabel("Number of households")
ax[1].set_title("Distribution of errors")

fig.tight_layout()
fig.savefig("error_analysis.png", dpi=150)
plt.show()

# =====================================================================
# CODE ADDITIONS for project.py
# Three blocks. Each one says exactly where it goes in your script.
# =====================================================================


# ---------------------------------------------------------------------
# BLOCK 1 - Finish the histogram
# WHERE: replace your lines 48-50 (plt.hist + the two plt.axvline lines)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["amount_paid"], bins=30)
ax.axvline(mean_amt, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_amt:.2f}')
ax.axvline(median_amt, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_amt:.2f}')
ax.set_xlabel("Monthly bill (₹)")
ax.set_ylabel("Number of households")
ax.set_title("Distribution of monthly electricity bill")
ax.legend()
fig.tight_layout()
fig.savefig("bill_hist.png", dpi=150)
# Why fig, ax instead of plt.hist: plt.* draws on "whatever figure is current".
# Later plots create new figures, so giving the histogram its own fig keeps it
# from being mixed into another plot, and fig.savefig saves exactly this one.


# ---------------------------------------------------------------------
# BLOCK 2 (optional) - Box plots for the 4 binary features
# WHERE: right after your groupby print (your line 51)
# ---------------------------------------------------------------------
binary_feats = ["is_ac", "is_tv", "is_flat", "is_urban"]
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(binary_feats):
    groups = [df.loc[df[col] == 0, "amount_paid"],      # bills where feature = 0
              df.loc[df[col] == 1, "amount_paid"]]      # bills where feature = 1
    ax[i].boxplot(groups)
    ax[i].set_xticks([1, 2])
    ax[i].set_xticklabels(["0", "1"])
    ax[i].set_xlabel(col)
    ax[i].set_ylabel("Monthly bill (₹)")
    ax[i].set_title(f"Bill by {col}")
fig.tight_layout()
fig.savefig("binary_vs_bill.png", dpi=150)
# df.loc[condition, "amount_paid"] = the bill column, only for rows where the
# condition is True. boxplot draws one box per list item: box 1 = group 0, box 2 = group 1.


# ---------------------------------------------------------------------
# BLOCK 3 - Save the model + estimate a bill from user inputs (objective 6)
# WHERE: at the very end, just BEFORE your final plt.show()
# ---------------------------------------------------------------------
import joblib

# Save scaler + linear regression together (the whole Pipeline)
joblib.dump(lin_model, "bill_model.joblib")
print("Saved trained model to bill_model.joblib")


def estimate_bill(num_rooms, num_people, housearea, is_ac, is_tv, is_flat,
                  ave_monthly_income, num_children, is_urban):
    # 1. one household -> one-row DataFrame with the training column names
    row = pd.DataFrame([{
        "num_rooms": num_rooms, "num_people": num_people, "housearea": housearea,
        "is_ac": is_ac, "is_tv": is_tv, "is_flat": is_flat,
        "ave_monthly_income": ave_monthly_income,
        "num_children": num_children, "is_urban": is_urban,
    }])
    # 2. SAME engineered features as before training (avoids training-serving skew)
    row["area_per_person"] = row["housearea"] / row["num_people"]
    row["people_per_room"] = row["num_people"] / row["num_rooms"]
    # 3. SAME column order as the training data
    row = row[X_train.columns]
    # 4. predict() returns an array -> take the first (only) value
    return lin_model.predict(row)[0]


# Test households
print(f"Urban, AC, 2 children : ₹{estimate_bill(2, 4, 750, 1, 1, 1, 30000, 2, 1):.0f}")   # ~₹923
print(f"Rural, no AC, 0 child : ₹{estimate_bill(1, 3, 500, 0, 1, 0, 15000, 0, 0):.0f}")   # ~₹243

# Live demo for the viva: set to True and type in any household
ASK_USER = True
if ASK_USER:
    print("\nEnter household details:")
    rooms    = float(input("Number of rooms: "))
    people   = float(input("Number of people: "))
    area     = float(input("House area (sq ft): "))
    ac       = float(input("Has AC? (1/0): "))
    tv       = float(input("Has TV? (1/0): "))
    flat     = float(input("Is a flat? (1/0): "))
    income   = float(input("Average monthly income (₹): "))
    children = float(input("Number of children: "))
    urban    = float(input("Urban? (1/0): "))
    print(f"Estimated monthly bill: ₹{estimate_bill(rooms, people, area, ac, tv, flat, income, children, urban):.0f}")
