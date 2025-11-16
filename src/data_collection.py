"""
PHASE 2: DATA COLLECTION
=========================

This module handles data collection, quality assessment, and initial exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataCollector:
    """Handles data collection and initial quality assessment"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def assess_data_quality(self, df):
        """Assess data quality and generate report"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        return quality_report
    
    def visualize_data(self, df, output_dir="data"):
        """Create initial data visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Collection: Initial Exploration', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = df['Churn'].value_counts()
        axes[0, 0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Churn Distribution')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(churn_counts.values):
            axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # 2. Tenure Distribution
        axes[0, 1].hist(df['tenure'], bins=30, color='#3498db', edgecolor='black')
        axes[0, 1].set_title('Customer Tenure Distribution')
        axes[0, 1].set_xlabel('Tenure (months)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Monthly Charges Distribution
        axes[1, 0].hist(df['MonthlyCharges'], bins=30, color='#9b59b6', edgecolor='black')
        axes[1, 0].set_title('Monthly Charges Distribution')
        axes[1, 0].set_xlabel('Monthly Charges ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Contract Type
        contract_counts = df['Contract'].value_counts()
        axes[1, 1].barh(contract_counts.index, contract_counts.values, color='#f39c12')
        axes[1, 1].set_title('Contract Type Distribution')
        axes[1, 1].set_xlabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_path / 'data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Visualization saved to {output_path / 'data_exploration.png'}")
    
    def collect_and_save_data(self, filename="telecom_churn.csv"):
        """Main method to collect and save data"""
        print("=" * 60)
        print("PHASE 2: DATA COLLECTION")
        print("=" * 60)
        
        # Load existing data
        raw_data_path = self.data_dir / filename
        print(f"\n Loading data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path)
        print(f" Data loaded successfully")
        print(f"   Shape: {df.shape}")
        
        # Assess quality
        print("\n Assessing data quality...")
        quality_report = self.assess_data_quality(df)
        print(f"   Total rows: {quality_report['total_rows']}")
        print(f"   Total columns: {quality_report['total_columns']}")
        print(f"   Duplicate rows: {quality_report['duplicate_rows']}")
        
        missing_total = sum(quality_report['missing_values'].values())
        if missing_total > 0:
            print(f"   Missing values: {missing_total}")
            for col, count in quality_report['missing_values'].items():
                if count > 0:
                    print(f"      - {col}: {count}")
        
        # Create visualizations
        print("\n Creating data visualizations...")
        self.visualize_data(df, output_dir=self.data_dir)
        
        # Display sample
        print("\n Sample data (first 5 rows):")
        print(df.head())
        
        print("\n Data Collection Complete!")
        return df

if __name__ == "__main__":
    collector = DataCollector()
    df = collector.collect_and_save_data()
