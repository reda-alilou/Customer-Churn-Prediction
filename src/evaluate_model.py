"""
PHASE 6: MODEL EVALUATION
==========================

This module handles model validation, testing, and performance analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ModelEvaluator:
    """Handles model evaluation and performance analysis"""
    
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir) / "processed"
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        # Load model
        model = joblib.load(self.models_dir / 'churn_model.joblib')
        
        # Load test data
        test_df = pd.read_csv(self.data_dir / 'test_engineered.csv')
        X_test = test_df.drop('Churn', axis=1)
        y_test = test_df['Churn']
        
        return model, X_test, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test set"""
        print("   Evaluating model on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics, cm, report, y_pred, y_pred_proba
    
    def check_success_criteria(self, metrics):
        """Check if model meets success criteria"""
        print("\n   Checking success criteria...")
        
        from problem_definition import ProblemDefinition
        
        criteria = {
            'Recall': (metrics['recall'], ProblemDefinition.MIN_RECALL),
            'Precision': (metrics['precision'], ProblemDefinition.MIN_PRECISION),
            'F1-Score': (metrics['f1_score'], ProblemDefinition.MIN_F1_SCORE),
            'ROC-AUC': (metrics['roc_auc'], ProblemDefinition.MIN_ROC_AUC),
            'Accuracy': (metrics['accuracy'], ProblemDefinition.MIN_ACCURACY)
        }
        
        all_passed = True
        for metric_name, (actual, threshold) in criteria.items():
            passed = actual >= threshold
            status = "" if passed else ""
            print(f"     {status} {metric_name}: {actual:.4f} (threshold: {threshold:.2f})")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def analyze_errors(self, y_test, y_pred, X_test):
        """Analyze model errors"""
        print("\n   Analyzing model errors...")
        
        # False positives and false negatives
        fp_mask = (y_test == 0) & (y_pred == 1)
        fn_mask = (y_test == 1) & (y_pred == 0)
        
        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        
        print(f"     False Positives: {fp_count}")
        print(f"     False Negatives: {fn_count}")
        
        error_analysis = {
            'false_positives': int(fp_count),
            'false_negatives': int(fn_count),
            'total_errors': int(fp_count + fn_count)
        }
        
        return error_analysis
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance"""
        print(f"\n   Plotting top {top_n} feature importances...")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("     Model doesn't support feature importance")
            return None
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        plot_path = self.models_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved to {plot_path}")
        
        return feature_importance_df
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        print("\n   Plotting confusion matrix...")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        plot_path = self.models_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved to {plot_path}")
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        print("\n   Plotting ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.models_dir / 'roc_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved to {plot_path}")
    
    def generate_evaluation_report(self, metrics, cm, report, error_analysis, all_passed):
        """Generate comprehensive evaluation report"""
        print("\n   Generating evaluation report...")
        
        report_data = {
            'test_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'error_analysis': error_analysis,
            'success_criteria_met': all_passed
        }
        
        # Save JSON report
        with open(self.models_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"     Saved to {self.models_dir / 'evaluation_report.json'}")
    
    def evaluate_pipeline(self):
        """Main evaluation pipeline"""
        print("=" * 60)
        print("PHASE 6: MODEL EVALUATION")
        print("=" * 60)
        
        # Load model and data
        print("\n Loading model and test data...")
        model, X_test, y_test = self.load_model_and_data()
        print(f"   Test samples: {len(X_test)}")
        
        # Evaluate model
        print("\n Evaluating model performance...")
        metrics, cm, report, y_pred, y_pred_proba = self.evaluate_model(model, X_test, y_test)
        
        # Print metrics
        print(f"\n Test Set Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Check success criteria
        all_passed = self.check_success_criteria(metrics)
        
        # Analyze errors
        error_analysis = self.analyze_errors(y_test, y_pred, X_test)
        
        # Visualizations
        print("\n Creating visualizations...")
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # Load feature names
        with open(self.data_dir / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        self.plot_feature_importance(model, feature_names)
        
        # Generate report
        self.generate_evaluation_report(metrics, cm, report, error_analysis, all_passed)
        
        if all_passed:
            print("\n Model Evaluation Complete - All Success Criteria Met!")
        else:
            print("\n  Model Evaluation Complete - Some Criteria Not Met")
        
        return metrics, all_passed

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    metrics, passed = evaluator.evaluate_pipeline()
