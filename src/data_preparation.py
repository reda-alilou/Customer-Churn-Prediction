"""
PHASE 3: DATA PREPARATION
==========================

This module handles data cleaning, formatting, and structuring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataPreparation:
    """Handles data cleaning and preparation"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_raw_data(self, filename="telecom_churn.csv"):
        """Load raw data"""
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df):
        """Clean the dataset"""
        df_clean = df.copy()
        
        # 1. Handle missing values
        print("   Handling missing values...")
        if df_clean['TotalCharges'].isnull().any():
            # Fill missing TotalCharges with median or calculate from tenure
            df_clean['TotalCharges'] = df_clean.apply(
                lambda row: row['MonthlyCharges'] * row['tenure'] 
                if pd.isnull(row['TotalCharges']) 
                else row['TotalCharges'], 
                axis=1
            )
        
        # 2. Remove duplicates
        print("    Removing duplicates...")
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"     Removed {removed} duplicate rows")
        
        # 3. Handle inconsistent values
        print("   Standardizing values...")
        # Ensure consistent capitalization
        for col in df_clean.select_dtypes(include=['object']).columns:
            if col != 'customerID':
                df_clean[col] = df_clean[col].str.strip()
        
        # 4. Convert data types
        print("   Converting data types...")
        # TotalCharges should be numeric
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Convert SeniorCitizen to Yes/No for consistency
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        return df_clean
    
    def format_data(self, df):
        """Format and structure data for modeling"""
        df_formatted = df.copy()
        
        print("   Formatting data structure...")
        
        # Separate features and target
        X = df_formatted.drop(['customerID', 'Churn'], axis=1)
        y = df_formatted['Churn'].map({'No': 0, 'Yes': 1})
        
        # Identify feature types
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"     Categorical features: {len(categorical_features)}")
        print(f"     Numerical features: {len(numerical_features)}")
        
        return X, y, categorical_features, numerical_features
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("    Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"     Train set: {len(X_train)} samples")
        print(f"     Validation set: {len(X_val)} samples")
        print(f"     Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_prepared_data(self, X_train, X_val, X_test, y_train, y_val, y_test,
                          categorical_features, numerical_features):
        """Save prepared datasets"""
        print("   Saving prepared data...")
        
        # Save splits
        train_df = X_train.copy()
        train_df['Churn'] = y_train.values
        train_df.to_csv(self.processed_dir / 'train.csv', index=False)
        
        val_df = X_val.copy()
        val_df['Churn'] = y_val.values
        val_df.to_csv(self.processed_dir / 'val.csv', index=False)
        
        test_df = X_test.copy()
        test_df['Churn'] = y_test.values
        test_df.to_csv(self.processed_dir / 'test.csv', index=False)
        
        # Save feature names
        feature_info = {
            'categorical': categorical_features,
            'numerical': numerical_features
        }
        
        import json
        with open(self.processed_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"     Saved to {self.processed_dir}")
    
    def prepare_pipeline(self, filename="telecom_churn.csv"):
        """Main preparation pipeline"""
        print("=" * 60)
        print("PHASE 3: DATA PREPARATION")
        print("=" * 60)
        
        # Load data
        print("\n Loading raw data...")
        df = self.load_raw_data(filename)
        print(f"   Loaded {len(df)} rows")
        
        # Clean data
        print("\n Cleaning data...")
        df_clean = self.clean_data(df)
        print(f"    Data cleaned: {len(df_clean)} rows remaining")
        
        # Format data
        print("\n Formatting data...")
        X, y, categorical_features, numerical_features = self.format_data(df_clean)
        
        # Check target distribution
        churn_dist = y.value_counts()
        print(f"\n Target distribution:")
        print(f"   No Churn (0): {churn_dist[0]} ({churn_dist[0]/len(y)*100:.1f}%)")
        print(f"   Churn (1): {churn_dist[1]} ({churn_dist[1]/len(y)*100:.1f}%)")
        
        # Split data
        print("\n Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Save prepared data
        print("\n Saving prepared data...")
        self.save_prepared_data(
            X_train, X_val, X_test, y_train, y_val, y_test,
            categorical_features, numerical_features
        )
        
        print("\n Data Preparation Complete!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, categorical_features, numerical_features

if __name__ == "__main__":
    prep = DataPreparation()
    results = prep.prepare_pipeline()
