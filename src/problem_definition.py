"""
PHASE 1: PROBLEM DEFINITION
============================

Business Objective:
- Predict customer churn to enable proactive retention strategies
- Reduce customer attrition rate by 20%
- Optimize retention campaign costs by targeting high-risk customers

Success Metrics:
1. PRIMARY METRIC: Recall  75%
   - We want to catch at least 75% of churning customers
   - Missing a churner is costly (lost revenue)

2. SECONDARY METRIC: Precision  65%
   - Minimize false alarms to avoid wasting retention resources
   - Too many false positives = inefficient campaigns

3. OVERALL METRIC: F1-Score  0.70
   - Balanced measure of precision and recall

Expected Performance:
- ROC-AUC  0.80 (good discrimination between classes)
- Accuracy  75% (overall correctness)

Business Impact:
- Cost of acquiring new customer: $500
- Cost of retention campaign: $50
- Monthly revenue per customer: $70
- ROI = If we save 100 customers = $7,000/month revenue vs $5,000 campaign cost
"""

class ProblemDefinition:
    """
    Defines the business problem and success criteria
    """
    
    PROBLEM_TYPE = "binary_classification"
    TARGET_VARIABLE = "Churn"
    
    # Success Metrics
    MIN_RECALL = 0.75
    MIN_PRECISION = 0.65
    MIN_F1_SCORE = 0.70
    MIN_ROC_AUC = 0.80
    MIN_ACCURACY = 0.75
    
    # Business Metrics
    CUSTOMER_ACQUISITION_COST = 500
    RETENTION_CAMPAIGN_COST = 50
    MONTHLY_REVENUE_PER_CUSTOMER = 70
    TARGET_CHURN_REDUCTION = 0.20  # 20% reduction
    
    @staticmethod
    def get_problem_summary():
        """Returns a summary of the problem definition"""
        return {
            "problem_type": ProblemDefinition.PROBLEM_TYPE,
            "target_variable": ProblemDefinition.TARGET_VARIABLE,
            "success_metrics": {
                "recall": f">= {ProblemDefinition.MIN_RECALL}",
                "precision": f">= {ProblemDefinition.MIN_PRECISION}",
                "f1_score": f">= {ProblemDefinition.MIN_F1_SCORE}",
                "roc_auc": f">= {ProblemDefinition.MIN_ROC_AUC}",
                "accuracy": f">= {ProblemDefinition.MIN_ACCURACY}"
            },
            "business_impact": {
                "target_reduction": f"{ProblemDefinition.TARGET_CHURN_REDUCTION * 100}%",
                "acquisition_cost": f"${ProblemDefinition.CUSTOMER_ACQUISITION_COST}",
                "retention_cost": f"${ProblemDefinition.RETENTION_CAMPAIGN_COST}",
                "monthly_revenue": f"${ProblemDefinition.MONTHLY_REVENUE_PER_CUSTOMER}"
            }
        }
    
    @staticmethod
    def calculate_roi(customers_saved):
        """Calculate ROI of retention campaign"""
        campaign_cost = customers_saved * ProblemDefinition.RETENTION_CAMPAIGN_COST
        revenue_saved = customers_saved * ProblemDefinition.MONTHLY_REVENUE_PER_CUSTOMER * 12
        roi = ((revenue_saved - campaign_cost) / campaign_cost) * 100
        return {
            "customers_saved": customers_saved,
            "campaign_cost": campaign_cost,
            "annual_revenue_saved": revenue_saved,
            "roi_percentage": roi
        }

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("PHASE 1: PROBLEM DEFINITION")
    print("=" * 60)
    
    summary = ProblemDefinition.get_problem_summary()
    print("\n Problem Summary:")
    print(json.dumps(summary, indent=2))
    
    print("\n ROI Analysis (if we save 100 customers):")
    roi = ProblemDefinition.calculate_roi(100)
    print(json.dumps(roi, indent=2))
    
    print("\n Problem Definition Complete!")
