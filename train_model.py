import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Correct path
DATA_PATH = "3.data/projects.csv"

# Load data
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())

# Features (X) and Target (y)
X = df.drop("effort_hours", axis=1)
y = df["effort_hours"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")