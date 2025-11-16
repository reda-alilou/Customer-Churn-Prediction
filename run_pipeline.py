"""
Main pipeline script to run all ML lifecycle phases
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_all_phases():
    """Run complete ML lifecycle"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPLETE ML LIFECYCLE PROJECT")
    print(" " * 10 + "Customer Churn Prediction - November 2025")
    print("=" * 70)
    
    # Phase 1: Problem Definition
    print("\n" + ">" * 35)
    from problem_definition import ProblemDefinition
    summary = ProblemDefinition.get_problem_summary()
    print(f"\n[OK] PHASE 1 COMPLETE - Problem Defined")
    
    # Phase 2: Data Collection
    print("\n" + ">" * 35)
    from data_collection import DataCollector
    collector = DataCollector()
    df = collector.collect_and_save_data()
    print(f"\n[OK] PHASE 2 COMPLETE - Data Collected: {len(df)} samples")
    
    # Phase 3: Data Preparation
    print("\n" + ">" * 35)
    from data_preparation import DataPreparation
    prep = DataPreparation()
    X_train, X_val, X_test, y_train, y_val, y_test, cat_features, num_features = prep.prepare_pipeline()
    print(f"\n[OK] PHASE 3 COMPLETE - Data Prepared")
    
    # Phase 4: Feature Engineering
    print("\n" + ">" * 35)
    from feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    X_train_eng, X_val_eng, X_test_eng, y_train, y_val, y_test, feature_names = fe.engineer_pipeline()
    print(f"\n[OK] PHASE 4 COMPLETE - Features Engineered: {len(feature_names)} features")
    
    # Phase 5: Model Training
    print("\n" + ">" * 35)
    from train_model import ModelTrainer
    trainer = ModelTrainer()
    model, model_name, training_results = trainer.train_pipeline()
    print(f"\n[OK] PHASE 5 COMPLETE - Model Trained: {model_name}")
    
    # Phase 6: Model Evaluation
    print("\n" + ">" * 35)
    from evaluate_model import ModelEvaluator
    evaluator = ModelEvaluator()
    metrics, success = evaluator.evaluate_pipeline()
    print(f"\n[OK] PHASE 6 COMPLETE - Model Evaluated")
    
    # Phase 7: Model Deployment (Info only - API runs separately)
    print("\n" + ">" * 35)
    print("=" * 60)
    print("PHASE 7: MODEL DEPLOYMENT")
    print("=" * 60)
    print("\n[OK] Deployment artifacts ready!")
    print("\nTo start the API server, run:")
    print("   python deployment/api.py")
    print("\nAPI documentation will be at: http://localhost:8000/docs")
    print("\nTo test the API, run:")
    print("   python deployment/test_api.py")
    
    # Phase 8: Model Monitoring
    print("\n" + ">" * 35)
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'monitoring'))
    from monitor import ModelMonitor
    monitor = ModelMonitor()
    monitor_metrics, drift_report, dashboard = monitor.monitoring_pipeline()
    print(f"\n[OK] PHASE 8 COMPLETE - Monitoring Setup")
    
    # Final Summary
    print("\n" + "=" * 70)
    print(" " * 20 + "PROJECT COMPLETE!")
    print("=" * 70)
    
    print("\nFINAL RESULTS:")
    print(f"   Model: {model_name}")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test Recall: {metrics['recall']:.4f}")
    print(f"   Test Precision: {metrics['precision']:.4f}")
    print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    if success:
        print("\n[SUCCESS] All success criteria met!")
    else:
        print("\n[WARNING] Some success criteria not met - consider model improvements")
    
    print("\nPROJECT ARTIFACTS:")
    print("   +- data/telecom_churn.csv - Raw data")
    print("   +- data/processed/ - Prepared datasets")
    print("   +- models/churn_model.joblib - Trained model")
    print("   +- models/preprocessor.joblib - Feature preprocessor")
    print("   +- models/evaluation_report.json - Performance metrics")
    print("   +- deployment/api.py - REST API")
    print("   +- monitoring/dashboard.json - Monitoring dashboard")
    
    print("\nNEXT STEPS:")
    print("   1. Review evaluation visualizations in models/")
    print("   2. Start API: python deployment/api.py")
    print("   3. Test API: python deployment/test_api.py")
    print("   4. Check monitoring: type monitoring/dashboard.json")
    
    print("\n" + "=" * 70)
    
    return {
        'model': model,
        'metrics': metrics,
        'success': success,
        'dashboard': dashboard
    }

if __name__ == "__main__":
    results = run_all_phases()
