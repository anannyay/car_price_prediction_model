from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_pl import make_prediction # Assuming predict_pl.py exists and make_prediction works
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

# NEW: Add a root endpoint to provide a welcoming message
@app.get("/")
async def read_root():
    """
    Root endpoint for the Car Price Predictor API.
    Provides a simple welcome message and instructions to use the /predict endpoint.
    """
    return {
        "message": "Welcome to the Car Price Predictor API!",
        "instructions": "Send a POST request to /predict with CarFeatures JSON to get a price prediction."
    }

@app.post("/predict")
def predict_price(features: CarFeatures):
    """
    Predicts the price of a car based on the provided features.
    Expects a JSON payload conforming to the CarFeatures BaseModel.
    """
    try:
        input_dict = features.dict()
        # Call your prediction function
        price = make_prediction(input_dict)
        return {"predicted_price": round(price, 2)}
    except Exception as e:
        # Catch any errors during prediction and return a 400 status
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
