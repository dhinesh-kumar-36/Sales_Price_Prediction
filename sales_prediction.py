
# =============================================================================
# SALES PREDICTION USING LINEAR REGRESSION
# =============================================================================
# Dataset  : advertising.csv (TV, Radio, Newspaper, Sales)
# Algorithm: Linear Regression
# =============================================================================

# ─── 1. IMPORT LIBRARIES ─────────────────────────────────────────────────────

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 60)
print("       SALES PREDICTION USING LINEAR REGRESSION")
print("=" * 60)

# ─── 2. GENERATE / LOAD DATASET ──────────────────────────────────────────────
CSV_PATH = "advertising.csv"

if not os.path.exists(CSV_PATH):
    print("\n⚠️  advertising.csv not found — generating synthetic dataset...")
    np.random.seed(42)
    n = 200
    TV        = np.random.uniform(0.7, 296.4, n).round(1)
    Radio     = np.random.uniform(0.0,  49.6, n).round(1)
    Newspaper = np.random.uniform(0.3, 114.0, n).round(1)
    # Sales formula with realistic noise
    Sales = (0.046 * TV + 0.188 * Radio + 0.004 * Newspaper + 2.9
             + np.random.normal(0, 1.0, n)).round(2)
    pd.DataFrame({"TV": TV, "Radio": Radio,
                  "Newspaper": Newspaper, "Sales": Sales}).to_csv(CSV_PATH, index=False)
    print(f"   ✅ '{CSV_PATH}' created with {n} rows.")

# Load dataset
df = pd.read_csv(CSV_PATH)

# ─── 3. DATA LOADING & BASIC CLEANING ────────────────────────────────────────

print("\n📋 DATASET INFORMATION")
print("-" * 40)
print(f"Shape  : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)")
print(f"Columns: {list(df.columns)}")

# Check for missing values
missing = df.isnull().sum().sum()
print(f"Missing values: {missing}")
if missing > 0:
    df.dropna(inplace=True)        
    print(f"   Rows after cleaning: {len(df)}")

print("\n📊 FIRST 5 ROWS:")
print("-" * 40)
print(df.head().to_string(index=False))

print("\n📈 BASIC STATISTICS:")
print("-" * 40)
print(df.describe().round(2))

# ─── 4. CORRELATION HEATMAP ──────────────────────────────────────────────────

print("\n🎨 Generating Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(7, 5))
corr = df.corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    square=True,
    ax=ax
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved as 'correlation_heatmap.png'")

# ─── 5. SCATTER PLOTS — TV / RADIO / NEWSPAPER vs SALES ──────────────────────

print("\n🎨 Generating Scatter Plots...")

features   = ["TV", "Radio", "Newspaper"]
colors     = ["#E74C3C", "#2ECC71", "#3498DB"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Advertising Spend vs Sales", fontsize=15, fontweight="bold", y=1.02)

for ax, feature, color in zip(axes, features, colors):
    ax.scatter(df[feature], df["Sales"], color=color, alpha=0.6, edgecolors="white", s=55)
    # Trend line
    m, b = np.polyfit(df[feature], df["Sales"], 1)
    x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"{feature} Budget ($)", fontsize=11)
    ax.set_ylabel("Sales (units)",         fontsize=11)
    ax.set_title(f"{feature} vs Sales",    fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved as 'scatter_plots.png'")

# ─── 6. PREPARE FEATURES & TARGET ────────────────────────────────────────────

X = df[["TV", "Radio", "Newspaper"]]   # Feature matrix
y = df["Sales"]                        # Target variable

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔀 TRAIN / TEST SPLIT")
print("-" * 40)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ─── 7. TRAIN LINEAR REGRESSION MODEL ────────────────────────────────────────

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n🤖 MODEL TRAINED: Linear Regression")

# ─── 8. MODEL COEFFICIENTS & INTERCEPT ───────────────────────────────────────

print("\n📐 MODEL COEFFICIENTS")
print("-" * 40)
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:12s}: {coef:.4f}")
print(f"  {'Intercept':12s}: {model.intercept_:.4f}")

# Interpretation: Sales = 0.046*TV + 0.188*Radio + 0.004*Newspaper + 2.9

# ─── 9. EVALUATION METRICS ───────────────────────────────────────────────────

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 MODEL EVALUATION METRICS")
print("=" * 60)
print(f"  R² Score (Accuracy) : {r2:.4f}   ({r2*100:.2f}%)")
print(f"  MAE  (Mean Abs Err) : {mae:.4f}")
print(f"  MSE  (Mean Sq  Err) : {mse:.4f}")
print(f"  RMSE (Root MSE)     : {rmse:.4f}")

# ─── 10. ACTUAL vs PREDICTED PLOT ────────────────────────────────────────────

print("\n🎨 Generating Actual vs Predicted Plot...")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred, color="#8E44AD", alpha=0.7, edgecolors="white", s=60)
# Perfect prediction line
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Sales",    fontsize=12)
ax.set_ylabel("Predicted Sales", fontsize=12)
ax.set_title("Actual vs Predicted Sales", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved as 'actual_vs_predicted.png'")

# ─── 11. SAMPLE PREDICTION ───────────────────────────────────────────────────

print("\n🛒 SAMPLE PREDICTION")
print("=" * 60)

# Input: [TV budget, Radio budget, Newspaper budget]
sample_input = np.array([[200.0, 30.0, 50.0]])
sample_df    = pd.DataFrame(sample_input, columns=features)

predicted_sales = model.predict(sample_df)[0]

print(f"  TV Budget        : $200.0")
print(f"  Radio Budget     : $30.0")
print(f"  Newspaper Budget : $50.0")
print(f"  ─────────────────────────")
print(f"  Predicted Sales  : {predicted_sales:.2f} units")

# ─── 12. SAVE MODEL ──────────────────────────────────────────────────────────

os.makedirs("saved_model", exist_ok=True)
model_path = "saved_model/sales_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 MODEL SAVED → {model_path}")

# Verify reload
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

verify = loaded_model.predict(sample_df)[0]
print(f"🔄 Reload Verification — Prediction: {verify:.2f} units")

print("\n" + "=" * 60)
print("   PROJECT COMPLETE — ALL STEPS FINISHED SUCCESSFULLY!")
print("=" * 60)
