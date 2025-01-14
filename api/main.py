from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load model and scaler
try:
    model = joblib.load("../models/rf_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")


# Define the request data schema using Pydantic
class PredictionRequest(BaseModel):
    data: List[dict]  # List of dictionaries representing the data rows


# Define the prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data into a DataFrame
        df = pd.DataFrame(request.data)

        # Check if DataFrame is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty.")

        # Scale the data
        scaled_data = scaler.transform(df)

        # Make predictions
        predictions = model.predict(scaled_data)

        # Return predictions as a JSON response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application using Uvicorn for production
# You can execute this script using: uvicorn api.app:app --reload
