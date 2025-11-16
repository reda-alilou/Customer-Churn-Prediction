"""
Test the deployed API
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print(" Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_prediction():
    """Test single prediction"""
    print("\n Testing single prediction...")
    
    # Sample customer data (high churn risk)
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0
    }
    
    response = requests.post(f"{API_URL}/predict", json=customer_data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction"""
    print("\n Testing batch prediction...")
    
    customers = [
        {
            "gender": "Male",
            "SeniorCitizen": "Yes",
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card",
            "MonthlyCharges": 110.0,
            "TotalCharges": 6600.0
        },
        {
            "gender": "Female",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 50.0,
            "TotalCharges": 50.0
        }
    ]
    
    response = requests.post(f"{API_URL}/predict_batch", json=customers)
    print(f"   Status: {response.status_code}")
    print(f"   Predictions:")
    for i, pred in enumerate(response.json()):
        print(f"   Customer {i+1}: {json.dumps(pred, indent=6)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DEPLOYED API")
    print("=" * 60)
    print("\n  Make sure the API is running (python deployment/api.py)\n")
    
    try:
        # Test endpoints
        health_ok = test_health()
        predict_ok = test_prediction()
        batch_ok = test_batch_prediction()
        
        print("\n" + "=" * 60)
        if health_ok and predict_ok and batch_ok:
            print(" All tests passed!")
        else:
            print(" Some tests failed")
        print("=" * 60)
    
    except requests.exceptions.ConnectionError:
        print("\n Error: Cannot connect to API")
        print("   Make sure the API is running: python deployment/api.py")
