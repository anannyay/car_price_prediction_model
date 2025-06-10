import pandas as pd
import pickle

# Load all pipeline parts
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def make_prediction(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_processed = preprocessor.transform(input_df)
    input_selected = selector.transform(input_processed)
    prediction = model.predict(input_selected)
    return prediction[0]
