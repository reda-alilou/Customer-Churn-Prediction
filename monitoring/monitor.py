"""
PHASE 8: MODEL MONITORING & MAINTENANCE
========================================

This module handles model performance monitoring, data drift detection, and A/B testing.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMonitor:
    """Handles model monitoring and drift detection"""
    
    def __init__(self, models_dir="models", monitoring_dir="monitoring"):
        self.models_dir = Path(models_dir)
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)
        self.logs_dir = self.monitoring_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load production model"""
        return joblib.load(self.models_dir / 'churn_model.joblib')
    
    def log_prediction(self, customer_data, prediction, probability, timestamp=None):
        """Log prediction for monitoring"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'customer_data': customer_data,
            'prediction': prediction,
            'probability': probability
        }
        
        # Append to log file
        log_file = self.logs_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def monitor_performance(self, y_true, y_pred, period='daily'):
        """Monitor model performance over time"""
        print(f"   Monitoring {period} performance...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'sample_size': len(y_true)
        }
        
        # Check for performance degradation
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / 'src'))
        from problem_definition import ProblemDefinition
        
        degraded = []
        if metrics['recall'] < ProblemDefinition.MIN_RECALL:
            degraded.append('recall')
        if metrics['precision'] < ProblemDefinition.MIN_PRECISION:
            degraded.append('precision')
        if metrics['f1_score'] < ProblemDefinition.MIN_F1_SCORE:
            degraded.append('f1_score')
        
        metrics['performance_degraded'] = len(degraded) > 0
        metrics['degraded_metrics'] = degraded
        
        # Save metrics
        metrics_file = self.monitoring_dir / f"performance_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"     Accuracy: {metrics['accuracy']:.4f}")
        print(f"     Recall: {metrics['recall']:.4f}")
        print(f"     Precision: {metrics['precision']:.4f}")
        
        if metrics['performance_degraded']:
            print(f"       Performance degradation detected in: {', '.join(degraded)}")
        else:
            print(f"      Performance within acceptable range")
        
        return metrics
    
    def detect_data_drift(self, reference_data, production_data, threshold=0.05):
        """Detect data drift using statistical tests"""
        print("\n   Detecting data drift...")
        
        from scipy import stats
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'threshold': threshold,
            'features_drifted': [],
            'drift_scores': {}
        }
        
        # Numerical features - use KS test
        numerical_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in production_data.columns:
                statistic, p_value = stats.ks_2samp(
                    reference_data[col].dropna(),
                    production_data[col].dropna()
                )
                
                drift_report['drift_scores'][col] = {
                    'test': 'ks_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drifted': bool(p_value < threshold)
                }
                
                if p_value < threshold:
                    drift_report['features_drifted'].append(col)
                    print(f"       Drift detected in '{col}' (p={p_value:.4f})")
        
        # Categorical features - use chi-square test
        categorical_cols = reference_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in production_data.columns:
                # Create contingency table
                ref_counts = reference_data[col].value_counts()
                prod_counts = production_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(prod_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                prod_aligned = [prod_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                contingency_table = np.array([ref_aligned, prod_aligned])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                drift_report['drift_scores'][col] = {
                    'test': 'chi_square',
                    'statistic': float(chi2),
                    'p_value': float(p_value),
                    'drifted': bool(p_value < threshold)
                }
                
                if p_value < threshold:
                    drift_report['features_drifted'].append(col)
                    print(f"       Drift detected in '{col}' (p={p_value:.4f})")
        
        drift_report['drift_detected'] = len(drift_report['features_drifted']) > 0
        
        # Save report
        drift_file = self.monitoring_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(drift_file, 'w') as f:
            json.dump(drift_report, f, indent=2)
        
        if drift_report['drift_detected']:
            print(f"\n       Data drift detected in {len(drift_report['features_drifted'])} features")
            print(f"     Consider retraining the model")
        else:
            print(f"\n      No significant drift detected")
        
        return drift_report
    
    def ab_test(self, model_a, model_b, X_test, y_test, model_a_name="Current", model_b_name="New"):
        """Perform A/B testing between two models"""
        print(f"\n    A/B Testing: {model_a_name} vs {model_b_name}...")
        
        # Predictions
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)
        
        # Metrics for Model A
        metrics_a = {
            'accuracy': accuracy_score(y_test, y_pred_a),
            'precision': precision_score(y_test, y_pred_a),
            'recall': recall_score(y_test, y_pred_a),
            'f1_score': f1_score(y_test, y_pred_a)
        }
        
        # Metrics for Model B
        metrics_b = {
            'accuracy': accuracy_score(y_test, y_pred_b),
            'precision': precision_score(y_test, y_pred_b),
            'recall': recall_score(y_test, y_pred_b),
            'f1_score': f1_score(y_test, y_pred_b)
        }
        
        # Compare
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'model_a': model_a_name,
            'model_b': model_b_name,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'winner': {},
            'test_size': len(X_test)
        }
        
        print(f"\n     {model_a_name} Performance:")
        for metric, value in metrics_a.items():
            print(f"       {metric}: {value:.4f}")
        
        print(f"\n     {model_b_name} Performance:")
        for metric, value in metrics_b.items():
            print(f"       {metric}: {value:.4f}")
        
        print(f"\n     Comparison:")
        for metric in metrics_a.keys():
            diff = metrics_b[metric] - metrics_a[metric]
            improvement = (diff / metrics_a[metric]) * 100
            
            if metrics_b[metric] > metrics_a[metric]:
                winner = model_b_name
                symbol = ""
            elif metrics_b[metric] < metrics_a[metric]:
                winner = model_a_name
                symbol = ""
            else:
                winner = "Tie"
                symbol = ""
            
            comparison['winner'][metric] = winner
            print(f"       {symbol} {metric}: {improvement:+.2f}% ({winner})")
        
        # Save comparison
        ab_file = self.monitoring_dir / f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(ab_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def plot_performance_over_time(self, metrics_history):
        """Plot model performance metrics over time"""
        print("\n   Plotting performance over time...")
        
        if len(metrics_history) < 2:
            print("     Not enough data points for time series plot")
            return
        
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Over Time', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            ax.plot(df['timestamp'], df[metric], marker='o', linewidth=2)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Time')
            ax.set_ylabel('Score')
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = self.monitoring_dir / 'performance_over_time.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved to {plot_path}")
    
    def generate_monitoring_dashboard(self):
        """Generate monitoring dashboard summary"""
        print("\n   Generating monitoring dashboard...")
        
        # Load latest metrics
        metrics_files = list(self.monitoring_dir.glob("performance_*.json"))
        
        if not metrics_files:
            print("     No performance metrics found")
            return
        
        latest_metrics = sorted(metrics_files, key=lambda x: x.stat().st_mtime)[-1]
        
        with open(latest_metrics, 'r') as f:
            metrics = json.load(f)
        
        # Load drift reports
        drift_files = list(self.monitoring_dir.glob("drift_report_*.json"))
        latest_drift = None
        
        if drift_files:
            latest_drift_file = sorted(drift_files, key=lambda x: x.stat().st_mtime)[-1]
            with open(latest_drift_file, 'r') as f:
                latest_drift = json.load(f)
        
        # Create dashboard
        dashboard = {
            'generated_at': datetime.now().isoformat(),
            'model_health': 'healthy' if not metrics.get('performance_degraded', False) else 'degraded',
            'latest_performance': metrics,
            'drift_status': latest_drift,
            'recommendations': []
        }
        
        # Generate recommendations
        if metrics.get('performance_degraded', False):
            dashboard['recommendations'].append("  Model performance has degraded - consider retraining")
        
        if latest_drift and latest_drift.get('drift_detected', False):
            dashboard['recommendations'].append(f"  Data drift detected in {len(latest_drift['features_drifted'])} features")
        
        if not dashboard['recommendations']:
            dashboard['recommendations'].append(" Model is performing well - no action needed")
        
        # Save dashboard
        with open(self.monitoring_dir / 'dashboard.json', 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"     Dashboard saved to {self.monitoring_dir / 'dashboard.json'}")
        print(f"     Model health: {dashboard['model_health']}")
        
        return dashboard
    
    def monitoring_pipeline(self):
        """Main monitoring pipeline"""
        print("=" * 60)
        print("PHASE 8: MODEL MONITORING & MAINTENANCE")
        print("=" * 60)
        
        # Load test data as proxy for production data (use engineered features)
        print("\n Loading data for monitoring...")
        test_df = pd.read_csv(Path("data/processed/test_engineered.csv"))
        X_test = test_df.drop('Churn', axis=1)
        y_test = test_df['Churn']
        
        # Load model
        print("\n Loading model...")
        model = self.load_model()
        
        # Generate predictions
        print("\n Generating predictions...")
        y_pred = model.predict(X_test)
        
        # Monitor performance
        print("\n Monitoring performance...")
        metrics = self.monitor_performance(y_test, y_pred, period='test')
        
        # Detect drift (using train as reference, test as production)
        print("\n Checking for data drift...")
        train_df = pd.read_csv(Path("data/processed/train_engineered.csv"))
        X_train = train_df.drop('Churn', axis=1)
        
        drift_report = self.detect_data_drift(X_train, X_test)
        
        # Generate dashboard
        print("\n Generating monitoring dashboard...")
        dashboard = self.generate_monitoring_dashboard()
        
        print("\n Model Monitoring Complete!")
        print("\n Recommendations:")
        for rec in dashboard['recommendations']:
            print(f"   {rec}")
        
        return metrics, drift_report, dashboard

if __name__ == "__main__":
    monitor = ModelMonitor()
    results = monitor.monitoring_pipeline()
