"""
PHASE 4: FEATURE ENGINEERING
=============================

This module handles feature selection, transformation, and creation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

class FeatureEngineering:
    """Handles feature engineering and transformation"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def load_prepared_data(self):
        """Load prepared data splits"""
        train_df = pd.read_csv(self.processed_dir / 'train.csv')
        val_df = pd.read_csv(self.processed_dir / 'val.csv')
        test_df = pd.read_csv(self.processed_dir / 'test.csv')
        
        # Load feature info
        with open(self.processed_dir / 'feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        # Split features and target
        X_train = train_df.drop('Churn', axis=1)
        y_train = train_df['Churn']
        
        X_val = val_df.drop('Churn', axis=1)
        y_val = val_df['Churn']
        
        X_test = test_df.drop('Churn', axis=1)
        y_test = test_df['Churn']
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_info
    
    def create_new_features(self, X):
        """Create new features from existing ones"""
        X_new = X.copy()
        
        print("   Creating new features...")
        
        # 1. Tenure groups
        X_new['TenureGroup'] = pd.cut(X_new['tenure'], 
                                      bins=[0, 12, 24, 48, 100],
                                      labels=['0-12', '12-24', '24-48', '48+'])
        
        # 2. Average monthly spend
        X_new['AvgMonthlySpend'] = X_new['TotalCharges'] / (X_new['tenure'] + 1)
        
        # 3. Service bundle score (number of services subscribed)
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        X_new['ServiceCount'] = 0
        for col in service_cols:
            if col in X_new.columns:
                X_new['ServiceCount'] += (X_new[col] == 'Yes').astype(int)
        
        # 4. Is high value customer
        X_new['IsHighValue'] = (X_new['MonthlyCharges'] > X_new['MonthlyCharges'].median()).astype(int)
        
        # 5. Has premium internet
        if 'InternetService' in X_new.columns:
            X_new['HasFiberOptic'] = (X_new['InternetService'] == 'Fiber optic').astype(int)
        
        # 6. Monthly to total charges ratio
        X_new['ChargesRatio'] = X_new['MonthlyCharges'] / (X_new['TotalCharges'] + 1)
        
        print(f"     Created 6 new features")
        
        return X_new
    
    def select_features(self, X, y=None):
        """Select relevant features for modeling"""
        print("   Selecting relevant features...")
        
        # For this project, we'll use all features
        # In production, you might use feature importance or correlation analysis
        
        selected_features = X.columns.tolist()
        print(f"     Selected {len(selected_features)} features")
        
        return selected_features
    
    def build_preprocessor(self, categorical_features, numerical_features):
        """Build preprocessing pipeline"""
        print("   Building preprocessing pipeline...")
        
        # Define transformers
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def transform_features(self, X_train, X_val, X_test, categorical_features, numerical_features):
        """Transform features using preprocessing pipeline"""
        print("    Transforming features...")
        
        # Build preprocessor
        preprocessor = self.build_preprocessor(categorical_features, numerical_features)
        
        # Fit on training data and transform all sets
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Save preprocessor
        joblib.dump(preprocessor, self.models_dir / 'preprocessor.joblib')
        print(f"     Preprocessor saved to {self.models_dir / 'preprocessor.joblib'}")
        
        # Get feature names after transformation
        feature_names = self._get_feature_names(preprocessor, numerical_features, categorical_features)
        
        # Convert to DataFrames
        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        print(f"     Final feature count: {len(feature_names)}")
        
        return X_train_df, X_val_df, X_test_df, preprocessor, feature_names
    
    def _get_feature_names(self, preprocessor, numerical_features, categorical_features):
        """Get feature names after transformation"""
        feature_names = []
        
        # Numerical features
        feature_names.extend(numerical_features)
        
        # Categorical features (one-hot encoded)
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
        
        return feature_names
    
    def save_engineered_data(self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
        """Save engineered features"""
        print("   Saving engineered features...")
        
        # Save as CSV
        train_df = X_train.copy()
        train_df['Churn'] = y_train.values
        train_df.to_csv(self.processed_dir / 'train_engineered.csv', index=False)
        
        val_df = X_val.copy()
        val_df['Churn'] = y_val.values
        val_df.to_csv(self.processed_dir / 'val_engineered.csv', index=False)
        
        test_df = X_test.copy()
        test_df['Churn'] = y_test.values
        test_df.to_csv(self.processed_dir / 'test_engineered.csv', index=False)
        
        # Save feature names
        with open(self.processed_dir / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        print(f"     Saved to {self.processed_dir}")
    
    def engineer_pipeline(self):
        """Main feature engineering pipeline"""
        print("=" * 60)
        print("PHASE 4: FEATURE ENGINEERING")
        print("=" * 60)
        
        # Load prepared data
        print("\n Loading prepared data...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_info = self.load_prepared_data()
        
        # Create new features
        print("\n Engineering features...")
        X_train_eng = self.create_new_features(X_train)
        X_val_eng = self.create_new_features(X_val)
        X_test_eng = self.create_new_features(X_test)
        
        # Update feature lists
        categorical_features = X_train_eng.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n Feature summary:")
        print(f"   Total features: {len(X_train_eng.columns)}")
        print(f"   Categorical: {len(categorical_features)}")
        print(f"   Numerical: {len(numerical_features)}")
        
        # Transform features
        print("\n Transforming features...")
        X_train_final, X_val_final, X_test_final, preprocessor, feature_names = self.transform_features(
            X_train_eng, X_val_eng, X_test_eng, categorical_features, numerical_features
        )
        
        # Save engineered data
        print("\n Saving engineered data...")
        self.save_engineered_data(
            X_train_final, X_val_final, X_test_final,
            y_train, y_val, y_test, feature_names
        )
        
        print("\n Feature Engineering Complete!")
        
        return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, feature_names

if __name__ == "__main__":
    fe = FeatureEngineering()
    results = fe.engineer_pipeline()
