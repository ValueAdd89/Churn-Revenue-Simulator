# applications/api/main.py
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date
import joblib
import os
from sqlalchemy import create_engine, text
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Revenue Simulator API",
    description="API for churn prediction and revenue analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://churn_user:churn_password@localhost:5432/churn_db")
engine = create_engine(DATABASE_URL)

# Pydantic models
class CustomerData(BaseModel):
    customer_id: str
    company_size: str = Field(..., description="Company size category")
    industry: str = Field(..., description="Industry sector")
    customer_tenure_months: int = Field(..., ge=0, description="Customer tenure in months")
    total_revenue: float = Field(..., ge=0, description="Total customer revenue")
    avg_daily_usage: float = Field(..., ge=0, description="Average daily usage sessions")
    features_adopted: int = Field(..., ge=0, description="Number of features adopted")
    last_activity_days_ago: int = Field(..., ge=0, description="Days since last activity")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    satisfaction_score: float = Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")

class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    key_factors: List[str] = Field(..., description="Top factors contributing to churn risk")
    health_score: float = Field(..., ge=0, le=100, description="Customer health score (0-100)")
    recommendations: List[str] = Field(..., description="Recommended actions")

class CustomerSegment(BaseModel):
    segment_name: str
    customer_count: int
    avg_revenue: float
    avg_churn_risk: float
    characteristics: List[str]

class RevenueMetrics(BaseModel):
    total_mrr: float
    total_arr: float
    churn_rate: float
    avg_customer_ltv: float
    revenue_at_risk: float

class CohortAnalysis(BaseModel):
    cohort_month: str
    cohort_size: int
    current_customers: int
    retention_rate: float
    total_revenue: float
    avg_ltv: float

# Mock model for demonstration (replace with actual trained model)
class MockChurnModel:
    def __init__(self):
        self.feature_names = [
            'customer_tenure_months', 'avg_daily_usage', 'features_adopted',
            'last_activity_days_ago', 'support_tickets', 'satisfaction_score'
        ]
        # Mock feature importance
        self.feature_importance_ = np.array([0.15, 0.25, 0.12, 0.20, 0.18, 0.10])
    
    def predict_proba(self, X):
        """Mock prediction - replace with actual model"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Simple heuristic for demonstration
        probabilities = []
        for row in X:
            tenure, usage, features, last_activity, tickets, satisfaction = row
            
            # Calculate risk factors
            usage_risk = max(0, (10 - usage) / 10)
            activity_risk = min(1, last_activity / 30)
            support_risk = min(1, tickets / 5)
            satisfaction_risk = max(0, (3 - satisfaction) / 2)
            tenure_risk = max(0, (6 - tenure) / 6)
            
            # Weighted combination
            churn_prob = (
                usage_risk * 0.25 +
                activity_risk * 0.20 +
                support_risk * 0.18 +
                satisfaction_risk * 0.10 +
                tenure_risk * 0.15 +
                np.random.uniform(0, 0.12)  # Random factor
            )
            
            churn_prob = max(0, min(1, churn_prob))
            probabilities.append([1 - churn_prob, churn_prob])
        
        return np.array(probabilities)

# Load model (mock for demonstration)
try:
    # In production, load actual trained model
    # model = joblib.load("models/churn_model.pkl")
    model = MockChurnModel()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = MockChurnModel()

# Helper functions
def get_key_factors(customer_data: CustomerData, churn_prob: float) -> List[str]:
    """Identify key factors contributing to churn risk"""
    factors = []
    
    if customer_data.last_activity_days_ago > 14:
        factors.append("High inactivity (14+ days since last login)")
    
    if customer_data.avg_daily_usage < 2:
        factors.append("Low product usage (<2 sessions/day)")
    
    if customer_data.support_tickets > 2:
        factors.append("High support volume (3+ tickets)")
    
    if customer_data.satisfaction_score <= 2:
        factors.append("Low satisfaction score (≤2)")
    
    if customer_data.features_adopted < 2:
        factors.append("Low feature adoption (<2 features)")
    
    if customer_data.customer_tenure_months < 3:
        factors.append("New customer (high risk period)")
    
    # If no specific factors, provide general ones
    if not factors:
        if churn_prob > 0.5:
            factors.append("Multiple moderate risk indicators")
        else:
            factors.append("Overall engagement patterns")
    
    return factors[:3]  # Return top 3 factors

def get_recommendations(customer_data: CustomerData, risk_level: str) -> List[str]:
    """Generate recommendations based on risk level and customer characteristics"""
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "Schedule immediate customer success call",
            "Offer personalized product training session",
            "Consider account manager assignment"
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "Send targeted feature adoption campaign",
            "Schedule proactive check-in call",
            "Provide usage optimization tips"
        ])
    else:
        recommendations.extend([
            "Include in customer advocacy program",
            "Offer referral incentives",
            "Consider upsell opportunities"
        ])
    
    # Add specific recommendations based on factors
    if customer_data.last_activity_days_ago > 14:
        recommendations.append("Send re-engagement email campaign")
    
    if customer_data.features_adopted < 2:
        recommendations.append("Provide guided feature tour")
    
    if customer_data.satisfaction_score <= 2:
        recommendations.append("Conduct satisfaction survey and follow-up")
    
    return list(set(recommendations))[:4]  # Return unique recommendations, max 4

def execute_query(query: str) -> pd.DataFrame:
    """Execute SQL query and return DataFrame"""
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return pd.DataFrame()

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Churn Revenue Simulator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "customers": "/customers",
            "metrics": "/metrics",
            "segments": "/segments",
            "cohorts": "/cohorts"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "database": db_status
    }

@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a customer"""
    try:
        # Prepare features for model
        feature_data = pd.DataFrame([{
            'customer_tenure_months': customer.customer_tenure_months,
            'avg_daily_usage': customer.avg_daily_usage,
            'features_adopted': customer.features_adopted,
            'last_activity_days_ago': customer.last_activity_days_ago,
            'support_tickets': customer.support_tickets,
            'satisfaction_score': customer.satisfaction_score
        }])
        
        # Make prediction
        churn_prob = model.predict_proba(feature_data)[0, 1]
        
        # Determine risk level
        if churn_prob >= 0.7:
            risk_level = "high"
        elif churn_prob >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Calculate health score
        health_score = max(0, min(100, 100 - (churn_prob * 100)))
        
        # Get key factors and recommendations
        key_factors = get_key_factors(customer, churn_prob)
        recommendations = get_recommendations(customer, risk_level)
        
        return ChurnPrediction(
            customer_id=customer.customer_id,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            key_factors=key_factors,
            health_score=round(health_score, 1),
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/customers", tags=["Customers"])
async def get_customers(
    limit: int = Query(100, ge=1, le=1000, description="Number of customers to return"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    company_size: Optional[str] = Query(None, description="Filter by company size")
):
    """Get customer list with optional filters"""
    try:
        query = """
        SELECT 
            customer_id,
            company_name,
            industry,
            company_size,
            customer_tenure_months,
            current_mrr,
            churn_risk_score,
            churn_risk_level,
            customer_health_score,
            last_activity_date
        FROM marts.mart_churn_analysis
        WHERE 1=1
        """
        
        if risk_level:
            query += f" AND churn_risk_level = '{risk_level}'"
        
        if company_size:
            query += f" AND company_size = '{company_size}'"
        
        query += f" ORDER BY churn_risk_score DESC LIMIT {limit}"
        
        df = execute_query(query)
        
        if df.empty:
            return {"customers": [], "count": 0}
        
        return {
            "customers": df.to_dict('records'),
            "count": len(df)
        }
    
    except Exception as e:
        logger.error(f"Failed to get customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=RevenueMetrics, tags=["Analytics"])
async def get_revenue_metrics():
    """Get key revenue and churn metrics"""
    try:
        query = """
        SELECT 
            SUM(current_mrr) as total_mrr,
            SUM(current_mrr) * 12 as total_arr,
            AVG(CASE WHEN is_churned THEN 1.0 ELSE 0.0 END) as churn_rate,
            AVG(total_revenue) as avg_customer_ltv,
            SUM(CASE WHEN churn_risk_level = 'high' THEN current_mrr ELSE 0 END) * 12 as revenue_at_risk
        FROM marts.mart_churn_analysis
        """
        
        df = execute_query(query)
        
        if df.empty:
            # Return default values if no data
            return RevenueMetrics(
                total_mrr=0,
                total_arr=0,
                churn_rate=0,
                avg_customer_ltv=0,
                revenue_at_risk=0
            )
        
        row = df.iloc[0]
        return RevenueMetrics(
            total_mrr=float(row['total_mrr'] or 0),
            total_arr=float(row['total_arr'] or 0),
            churn_rate=float(row['churn_rate'] or 0),
            avg_customer_ltv=float(row['avg_customer_ltv'] or 0),
            revenue_at_risk=float(row['revenue_at_risk'] or 0)
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments", response_model=List[CustomerSegment], tags=["Analytics"])
async def get_customer_segments():
    """Get customer segmentation analysis"""
    try:
        query = """
        SELECT 
            CASE 
                WHEN churn_risk_level = 'high' AND current_mrr > 100 THEN 'High Value At Risk'
                WHEN churn_risk_level = 'high' THEN 'At Risk'
                WHEN churn_risk_level = 'low' AND current_mrr > 200 THEN 'Champions'
                WHEN customer_tenure_months <= 3 THEN 'New Customers'
                WHEN churn_risk_level = 'low' THEN 'Healthy'
                ELSE 'Requires Attention'
            END as segment_name,
            COUNT(*) as customer_count,
            AVG(current_mrr) as avg_revenue,
            AVG(churn_risk_score) as avg_churn_risk
        FROM marts.mart_churn_analysis
        GROUP BY segment_name
        ORDER BY customer_count DESC
        """
        
        df = execute_query(query)
        
        if df.empty:
            return []
        
        segments = []
        for _, row in df.iterrows():
            # Generate characteristics based on segment
            characteristics = []
            segment = row['segment_name']
            
            if 'At Risk' in segment:
                characteristics = ["High churn probability", "Requires immediate attention", "Revenue impact significant"]
            elif 'Champions' in segment:
                characteristics = ["High value customers", "Low churn risk", "Expansion opportunities"]
            elif 'New' in segment:
                characteristics = ["Recently acquired", "Onboarding critical", "High potential value"]
            elif 'Healthy' in segment:
                characteristics = ["Stable usage patterns", "Satisfied customers", "Retention focused"]
            else:
                characteristics = ["Mixed risk factors", "Monitoring required", "Intervention opportunities"]
            
            segments.append(CustomerSegment(
                segment_name=segment,
                customer_count=int(row['customer_count']),
                avg_revenue=float(row['avg_revenue'] or 0),
                avg_churn_risk=float(row['avg_churn_risk'] or 0),
                characteristics=characteristics
            ))
        
        return segments
    
    except Exception as e:
        logger.error(f"Failed to get segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cohorts", response_model=List[CohortAnalysis], tags=["Analytics"])
async def get_cohort_analysis(months: int = Query(12, ge=1, le=24, description="Number of months to analyze")):
    """Get cohort retention analysis"""
    try:
        query = f"""
        SELECT 
            TO_CHAR(cohort_month, 'YYYY-MM') as cohort_month,
            cohort_size,
            cohort_size - churned_customers as current_customers,
            (cohort_size - churned_customers)::float / cohort_size as retention_rate,
            total_cohort_revenue,
            average_customer_ltv
        FROM marts.mart_revenue_analysis
        WHERE cohort_month >= CURRENT_DATE - INTERVAL '{months} months'
        ORDER BY cohort_month DESC
        """
        
        df = execute_query(query)
        
        if df.empty:
            return []
        
        cohorts = []
        for _, row in df.iterrows():
            cohorts.append(CohortAnalysis(
                cohort_month=row['cohort_month'],
                cohort_size=int(row['cohort_size']),
                current_customers=int(row['current_customers']),
                retention_rate=float(row['retention_rate'] or 0),
                total_revenue=float(row['total_cohort_revenue'] or 0),
                avg_ltv=float(row['average_customer_ltv'] or 0)
            ))
        
        return cohorts
    
    except Exception as e:
        logger.error(f"Failed to get cohort analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers/{customer_id}", tags=["Customers"])
async def get_customer_details(customer_id: str):
    """Get detailed information for a specific customer"""
    try:
        query = f"""
        SELECT *
        FROM marts.mart_churn_analysis
        WHERE customer_id = '{customer_id}'
        """
        
        df = execute_query(query)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        customer = df.iloc[0].to_dict()
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in customer.items():
            if pd.isna(value):
                customer[key] = None
            elif hasattr(value, 'item'):
                customer[key] = value.item()
        
        return {"customer": customer}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get customer details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/customers/batch-predict", tags=["Prediction"])
async def batch_predict_churn(customers: List[CustomerData]):
    """Predict churn for multiple customers"""
    try:
        if len(customers) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 customers per batch request")
        
        predictions = []
        for customer in customers:
            # Reuse single prediction logic
            prediction = await predict_churn(customer)
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "high_risk_count": sum(1 for p in predictions if p.risk_level == "high"),
            "medium_risk_count": sum(1 for p in predictions if p.risk_level == "medium"),
            "low_risk_count": sum(1 for p in predictions if p.risk_level == "low")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)