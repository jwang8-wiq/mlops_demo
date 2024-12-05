from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
from pydantic import BaseModel, Field

router = APIRouter()

# Input schema for prediction
class ChurnInput(BaseModel):
    tenure: float = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=850.0)
    gender: str = Field(..., example="Male")
    Partner: str = Field(..., example="No")
    Dependents: str = Field(..., example="No")
    PhoneService: str = Field(..., example="Yes")
    InternetService: str = Field(..., example="Fiber optic")
    Contract: str = Field(..., example="Month-to-month")
    PaymentMethod: str = Field(..., example="Credit card (automatic)")
    Tenure_Bin: str = Field(..., example="1-2 yrs")

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
