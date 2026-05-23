import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
import joblib

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Feature Engineering
df['rooms_per_house'] = df['AveRooms'] / df['HouseAge']
df['bedroom_ratio'] = df['AveBedrms'] / df['AveRooms']

# Features & Target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X, y)

# Save model + feature names
joblib.dump((model, X.columns.tolist()), "model.pkl")

print("✅ Model saved successfully!")