import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")
X = df.drop("Price", axis=1)
y = df["Price"]

# Load pipeline parts
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# Transform and select features
X_processed = preprocessor.transform(X)
X_selected = selector.transform(X_processed)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_selected, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
