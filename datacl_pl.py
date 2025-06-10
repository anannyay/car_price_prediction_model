import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load raw dataset
df = pd.read_csv("train-data.csv")
df.drop(columns=["Unnamed: 0", "New_Price"], errors="ignore", inplace=True)

# Clean columns
df["Mileage"] = df["Mileage"].str.extract(r"([\d.]+)").astype(float)
df["Engine"] = df["Engine"].str.extract(r"([\d.]+)").astype(float)
df["Power"] = df["Power"].str.extract(r"([\d.]+)").astype(float)
df["Seats"] = pd.to_numeric(df["Seats"], errors="coerce")

# Add derived features
df["Car_Age"] = datetime.now().year - df["Year"]
df["Brand"] = df["Name"].str.split().str[0]
df.drop(columns=["Year", "Name"], inplace=True)

# Remove outliers
df = df[df["Price"] < 100]
df = df[df["Kilometers_Driven"] < 500000]

# Save cleaned data for training
df.to_csv("cleaned_data.csv", index=False)

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Define columns
num_features = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "Car_Age"]
cat_features = ["Location", "Fuel_Type", "Transmission", "Owner_Type", "Brand"]

# Define pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Fit and transform the preprocessor
X_processed = preprocessor.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_regression, k=20)
selector.fit(X_processed, y)

# Save preprocessor and selector
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)

print("✅ Preprocessing pipeline and feature selector saved.")
