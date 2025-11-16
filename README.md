# Customer Churn Prediction - Complete ML Lifecycle

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![API](https://img.shields.io/badge/API-FastAPI-009688.svg)

A complete machine learning system for predicting customer churn in telecommunications, implementing all 8 phases of the ML lifecycle from problem definition through deployment and monitoring.

## Overview

This project predicts which customers are likely to cancel their service subscription, enabling proactive retention strategies. The model achieves 89.67% recall in identifying churning customers, with a business ROI of 1,580%.

## Project Structure

```
.
|-- src/                           # Source code for phases 1-6
|   |-- problem_definition.py     # Phase 1: Business objectives and metrics
|   |-- data_collection.py        # Phase 2: Data generation and quality assessment
|   |-- data_preparation.py       # Phase 3: Cleaning and preprocessing
|   |-- feature_engineering.py    # Phase 4: Feature transformation
|   |-- train_model.py            # Phase 5: Model training with hyperparameter tuning
|   +-- evaluate_model.py         # Phase 6: Performance evaluation
|
|-- deployment/                    # Phase 7: Model deployment
|   |-- api.py                     # FastAPI REST endpoint
|   +-- test_api.py                # API testing script
|
|-- monitoring/                    # Phase 8: Monitoring and maintenance
|   +-- monitor.py                 # Performance tracking and drift detection
|
|-- data/                          # Data storage
|   |-- telecom_churn.csv          # Raw dataset (7,000 records)
|   +-- processed/                 # Prepared and engineered datasets
|
|-- models/                        # Trained models and reports
|   |-- churn_model.joblib         # Final XGBoost model
|   |-- preprocessor.joblib        # Feature preprocessing pipeline
|   |-- evaluation_report.json     # Performance metrics
|   |-- confusion_matrix.png       # Confusion matrix visualization
|   |-- roc_curve.png              # ROC curve
|   +-- feature_importance.png     # Feature importance chart
|
|-- run_pipeline.py                # Execute complete ML pipeline
|-- requirements.txt               # Python dependencies
|-- Dockerfile                     # Container configuration
+-- README.md                      # This file
```

## API Deployment

### Start the API Server

```bash
python deployment/api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `POST /predict` - Single customer churn prediction
- `POST /predict_batch` - Batch predictions
- `GET /docs` - Interactive API documentation (Swagger UI)

### Test the API

```bash
# In a new terminal
python deployment/test_api.py
```

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
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
  }'
```

### Docker Deployment

```bash
# Build image
docker build -t churn-prediction-api .

# Run container
docker run -p 8000:8000 churn-prediction-api
```

### 1. Problem Definition
- Business objective: Reduce customer churn
- Success metrics: Recall >=75%, Precision >=65%
- Expected F1-Score: >=0.70

### 2. Data Collection
- Dataset: Telecom Customer Churn
- Features: Demographics, services, account info, usage patterns
- 7,000 customer records with 21 features

### 3. Data Preparation
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Train/validation/test split: 70/10/20

### 4. Feature Engineering
- Create tenure groups
- Engineer service usage features
- Calculate customer lifetime value indicators
- Final feature count: 39 features

### 5. Model Training
- Algorithms: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with cross-validation
- Best model: XGBoost (CV Recall: 87.59%)

### 6. Model Evaluation
- Test Recall: 89.67%
- Test Precision: 75.64%
- Test F1-Score: 82.06%
- ROC-AUC: 91.11%

### 7. Model Deployment
- FastAPI REST endpoint
- Model serialization with joblib
- Interactive API documentation

### 8. Model Monitoring
- Performance tracking
- Data drift detection (KS test, Chi-square)
- Monitoring dashboard

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/reda-alilou/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

Execute all 8 ML lifecycle phases:

```bash
python run_pipeline.py
```

This will:
1. Define problem and success metrics
2. Generate and collect 7,000 customer records
3. Clean and prepare data (70/10/20 train/val/test split)
4. Engineer 39 features from 21 original features
5. Train 3 models with hyperparameter tuning
6. Evaluate and select best model (XGBoost)
7. Prepare deployment artifacts
8. Run monitoring and drift detection

### Run Individual Phases

```bash
# Phase 1: Problem Definition
python src/problem_definition.py

# Phase 2: Data Collection
python src/data_collection.py

# Phase 3: Data Preparation
python src/data_preparation.py

# Phase 4: Feature Engineering
python src/feature_engineering.py

# Phase 5: Model Training
python src/train_model.py

# Phase 6: Model Evaluation
python src/evaluate_model.py

# Phase 7: Start API Server
python deployment/api.py

# Phase 8: Run Monitoring
python monitoring/monitor.py
```

## Model Performance

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Recall | 89.67% | >= 75% | Pass |
| Precision | 75.64% | >= 65% | Pass |
| F1-Score | 82.06% | >= 70% | Pass |
| ROC-AUC | 91.11% | >= 80% | Pass |
| Accuracy | 81.57% | >= 75% | Pass |

## Business Impact

- **Campaign Cost**: $50 per customer
- **Customer Lifetime Value**: $840/year ($70/month)
- **Model Recall**: 89.67% (saves ~90 out of 100 at-risk customers)

**Example**: For 100 at-risk customers
- Retention campaign cost: $4,450
- Revenue saved: $74,760 annually
- Net benefit: $70,310
- ROI: 1,580%
