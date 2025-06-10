from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_pl import make_prediction
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Car Price Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarFeatures(BaseModel):
    Location: str
    Kilometers_Driven: int
    Fuel_Type: str
    Transmission: str
    Owner_Type: str
    Mileage: float
    Engine: float
    Power: float
    Seats: float
    Brand: str
    Car_Age: int

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        input_dict = features.dict()
        price = make_prediction(input_dict)
        return {"predicted_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
