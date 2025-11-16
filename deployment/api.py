"""
PHASE 7: MODEL DEPLOYMENT
==========================

This module handles model deployment as a REST API using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn
from typing import List, Dict

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in telecom",
    version="1.0.0"
)

# Load model and preprocessor at startup
models_dir = Path("models")
MODEL = None
PREPROCESSOR = None
FEATURE_NAMES = None

@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global MODEL, PREPROCESSOR, FEATURE_NAMES
    
    MODEL = joblib.load(models_dir / 'churn_model.joblib')
    PREPROCESSOR = joblib.load(models_dir / 'preprocessor.joblib')
    
    # Load feature names
    import json
    with open(Path("data/processed/feature_names.json"), 'r') as f:
        FEATURE_NAMES = json.load(f)
    
    print(" Model loaded successfully!")

# Input schema
class CustomerData(BaseModel):
    """Schema for customer data input"""
    gender: str = Field(..., example="Female")
    SeniorCitizen: str = Field(..., example="No")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=24)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=75.50)
    TotalCharges: float = Field(..., ge=0, example=1800.0)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.50,
                "TotalCharges": 1800.0
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    churn_prediction: str
    churn_probability: float
    risk_level: str
    confidence: float

def engineer_features(df):
    """Apply feature engineering (same as training)"""
    # Tenure groups
    df['TenureGroup'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 48, 100],
                               labels=['0-12', '12-24', '24-48', '48+'])
    
    # Average monthly spend
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Service count
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']
    
    df['ServiceCount'] = 0
    for col in service_cols:
        if col in df.columns:
            df['ServiceCount'] += (df[col] == 'Yes').astype(int)
    
    # Is high value customer
    df['IsHighValue'] = (df['MonthlyCharges'] > 64.0).astype(int)  # Median from training
    
    # Has fiber optic
    df['HasFiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    
    # Charges ratio
    df['ChargesRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    
    return df

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict customer churn
    
    Returns:
        - churn_prediction: "Yes" or "No"
        - churn_probability: Probability of churn (0-1)
        - risk_level: "Low", "Medium", or "High"
        - confidence: Model confidence (0-1)
    """
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Engineer features
        df = engineer_features(df)
        
        # Apply preprocessing
        df_processed = PREPROCESSOR.transform(df)
        
        # Get prediction
        prediction = MODEL.predict(df_processed)[0]
        probability = MODEL.predict_proba(df_processed)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate confidence
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            churn_prediction="Yes" if prediction == 1 else "No",
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: List[CustomerData]):
    """
    Predict churn for multiple customers
    
    Returns list of predictions
    """
    try:
        predictions = []
        
        for customer in customers:
            # Convert to DataFrame
            customer_dict = customer.dict()
            df = pd.DataFrame([customer_dict])
            
            # Engineer features
            df = engineer_features(df)
            
            # Apply preprocessing
            df_processed = PREPROCESSOR.transform(df)
            
            # Get prediction
            prediction = MODEL.predict(df_processed)[0]
            probability = MODEL.predict_proba(df_processed)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            confidence = max(probability, 1 - probability)
            
            predictions.append({
                "churn_prediction": "Yes" if prediction == 1 else "No",
                "churn_probability": round(probability, 4),
                "risk_level": risk_level,
                "confidence": round(confidence, 4)
            })
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 7: MODEL DEPLOYMENT")
    print("=" * 60)
    print("\n Starting FastAPI server...")
    print(" API will be available at: http://localhost:8000")
    print(" API documentation at: http://localhost:8000/docs")
    print("\n Deployment Ready!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
