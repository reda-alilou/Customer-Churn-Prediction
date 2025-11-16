"""
PHASE 5: MODEL TRAINING
========================

This module handles model selection, training, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import joblib
import time

class ModelTrainer:
    """Handles model training and optimization"""
    
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir = Path(data_dir) / "processed"
        
    def load_engineered_data(self):
        """Load engineered features"""
        train_df = pd.read_csv(self.data_dir / 'train_engineered.csv')
        val_df = pd.read_csv(self.data_dir / 'val_engineered.csv')
        
        X_train = train_df.drop('Churn', axis=1)
        y_train = train_df['Churn']
        
        X_val = val_df.drop('Churn', axis=1)
        y_val = val_df['Churn']
        
        return X_train, X_val, y_train, y_val
    
    def define_models(self):
        """Define candidate models"""
        print("   Defining candidate models...")
        
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        print(f"     Defined {len(models)} models")
        return models
    
    def train_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """Train a single model with hyperparameter tuning"""
        print(f"\n   Training {model_name}...")
        
        start_time = time.time()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['params'],
            cv=5,
            scoring='recall',  # Optimize for recall (catch churners)
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Training time
        training_time = time.time() - start_time
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='recall')
        
        # Validation score
        val_score = best_model.score(X_val, y_val)
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'cv_recall_mean': cv_scores.mean(),
            'cv_recall_std': cv_scores.std(),
            'val_accuracy': val_score,
            'training_time': training_time,
            'best_model': best_model
        }
        
        print(f"      Best params: {grid_search.best_params_}")
        print(f"      CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"      Training time: {training_time:.2f}s")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all candidate models"""
        print("\n Training all models...")
        
        models = self.define_models()
        results = {}
        
        for model_name, model_config in models.items():
            result = self.train_model(model_name, model_config, X_train, y_train, X_val, y_val)
            results[model_name] = result
        
        return results
    
    def select_best_model(self, results):
        """Select the best model based on validation performance"""
        print("\n Selecting best model...")
        
        # Rank by CV recall score
        best_model_name = max(results, key=lambda x: results[x]['cv_recall_mean'])
        best_result = results[best_model_name]
        
        print(f"     Best model: {best_model_name}")
        print(f"     CV Recall: {best_result['cv_recall_mean']:.4f}")
        
        return best_model_name, best_result['best_model']
    
    def save_model(self, model, model_name, results):
        """Save trained model and metadata"""
        print("\n Saving model...")
        
        # Save model
        model_path = self.models_dir / 'churn_model.joblib'
        joblib.dump(model, model_path)
        print(f"     Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'best_params': results[model_name]['best_params'],
            'cv_recall_mean': results[model_name]['cv_recall_mean'],
            'cv_recall_std': results[model_name]['cv_recall_std'],
            'training_time': results[model_name]['training_time']
        }
        
        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"     Metadata saved to {self.models_dir / 'model_metadata.json'}")
    
    def train_pipeline(self):
        """Main training pipeline"""
        print("=" * 60)
        print("PHASE 5: MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        print("\n Loading engineered data...")
        X_train, X_val, y_train, y_val = self.load_engineered_data()
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train models
        print("\n Training models with hyperparameter tuning...")
        results = self.train_all_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        best_model_name, best_model = self.select_best_model(results)
        
        # Save model
        self.save_model(best_model, best_model_name, results)
        
        print("\n Model Training Complete!")
        
        return best_model, best_model_name, results

if __name__ == "__main__":
    trainer = ModelTrainer()
    model, model_name, results = trainer.train_pipeline()
