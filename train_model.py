import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DESHARNAIS DATASET
# ==========================================

DATA_PATH = "3.data/desharnais.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Number of projects:", len(df))
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "TeamExp",
    "ManagerExp",
    "Length",
    "Transactions",
    "Entities",
    "Adjustment",
    "Language"
]

target = "Effort"


# ==========================================
# 3. CLEAN DATA
# ==========================================

df = df[features + [target]].copy()

# Convert all selected columns to numeric
for column in features + [target]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing values
df = df.dropna()

print("\nClean dataset size:", len(df))


# ==========================================
# 4. PREPARE X AND Y
# ==========================================

X = df[features]
y = df[target]


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 7. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. CALCULATE PERFORMANCE METRICS
# ==========================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

# Relative error
relative_errors = (
    np.abs(y_test - y_pred)
    / np.maximum(np.abs(y_test), 1e-10)
)

mmre = np.mean(relative_errors)

mdmre = np.median(relative_errors)

pred25 = np.mean(
    relative_errors <= 0.25
) * 100


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f} hours")
print(f"RMSE     : {rmse:.2f} hours")
print(f"MMRE     : {mmre:.4f}")
print(f"MdMRE    : {mdmre:.4f}")
print(f"PRED(25) : {pred25:.2f}%")

print("======================================")


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ==========================================
# 11. TRAIN FINAL MODEL ON ALL DATA
# ==========================================

final_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

final_model.fit(X, y)


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    final_model,
    "model.pkl"
)

print("\nFinal model trained using all available projects.")

print("Model saved successfully as model.pkl")