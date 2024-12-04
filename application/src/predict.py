from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
from pydantic import BaseModel

router = APIRouter()

# Input schema for prediction
class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    InternetService: str
    Contract: str
    PaymentMethod: str
    Tenure_Bin: str

@router.post("/predict/")
def predict(input_data: ChurnInput, request: Request):
    """
    Predict churn probability for the given input data.
    """
    # Access the model from the application state
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    try:
        # Make prediction
        predictions = model.predict_proba(input_df)[:, 1]  # Probability of churn

        # Convert numpy.float32 to Python float for JSON serialization
        churn_probability = float(predictions[0])

        return {"churn_probability": churn_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
