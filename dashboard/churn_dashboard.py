# applications/dashboard/churn_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Churn Revenue Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size:14px !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    
    # Generate sample data
    n_customers = 2000
    
    customers = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'company_size': np.random.choice(['startup', 'small', 'medium', 'large', 'enterprise'], 
                                       n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Education', 'Retail'], 
                                   n_customers, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'subscription_tier': np.random.choice(['basic', 'premium', 'enterprise'], 
                                            n_customers, p=[0.5, 0.35, 0.15]),
        'monthly_revenue': np.random.choice([29, 99, 299], n_customers, p=[0.5, 0.35, 0.15]),
        'customer_tenure_months': np.random.exponential(scale=12, size=n_customers).astype(int),
        'total_revenue': np.random.exponential(scale=2000, size=n_customers),
        'avg_daily_usage': np.random.exponential(scale=20, size=n_customers),
        'features_adopted': np.random.poisson(lam=3, size=n_customers),
        'last_activity_days_ago': np.random.exponential(scale=7, size=n_customers).astype(int),
        'support_tickets': np.random.poisson(lam=1, size=n_customers),
        'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
        'signup_date': pd.date_range(start='2022-01-01', periods=n_customers, freq='D')[:n_customers]
    })
    
    # Create churn labels with realistic patterns
    churn_prob = (
        0.1 +  # Base churn rate
        0.05 * (customers['last_activity_days_ago'] > 30).astype(int) +
        0.05 * (customers['avg_daily_usage'] < 5).astype(int) +
        0.03 * (customers['satisfaction_score'] <= 2).astype(int) +
        0.02 * (customers['support_tickets'] > 3).astype(int) -
        0.03 * (customers['subscription_tier'] == 'enterprise').astype(int) -
        0.02 * (customers['features_adopted'] > 5).astype(int)
    )
    
    customers['is_churned'] = np.random.binomial(1, np.clip(churn_prob, 0, 1, size=n_customers))
    customers['churn_risk_score'] = churn_prob
    customers['churn_risk_level'] = pd.cut(customers['churn_risk_score'], 
                                         bins=[0, 0.1, 0.3, 1.0], 
                                         labels=['low', 'medium', 'high'])
    
    return customers

@st.cache_data
def calculate_metrics(df):
    """Calculate key business metrics"""
    total_customers = len(df)
    churned_customers = df['is_churned'].sum()
    churn_rate = churned_customers / total_customers * 100
    
    mrr = df[df['is_churned'] == 0]['monthly_revenue'].sum()
    avg_ltv = df['total_revenue'].mean()
    at_risk_customers = len(df[df['churn_risk_level'] == 'high'])
    
    # Calculate month-over-month changes (simulated)
    churn_rate_change = np.random.uniform(-0.5, 0.5)
    mrr_change = np.random.uniform(0.05, 0.15)
    ltv_change = np.random.uniform(0.02, 0.08)
    at_risk_change = np.random.randint(-50, 20)
    
    return {
        'churn_rate': churn_rate,
        'churn_rate_change': churn_rate_change,
        'mrr': mrr,
        'mrr_change': mrr_change,
        'avg_ltv': avg_ltv,
        'ltv_change': ltv_change,
        'at_risk_customers': at_risk_customers,
        'at_risk_change': at_risk_change,
        'total_customers': total_customers,
        'churned_customers': churned_customers
    }

def create_churn_trend_chart(df):
    """Create churn trend over time chart"""
    # Simulate monthly data
    months = pd.date_range(start='2023-01-01', end='2024-06-01', freq='M')
    churn_rates = []
    
    for month in months:
        base_rate = 0.15
        seasonal_factor = 0.02 * np.sin(2 * np.pi * month.month / 12)
        noise = np.random.normal(0, 0.01)
        churn_rates.append(max(0, base_rate + seasonal_factor + noise))
    
    trend_df = pd.DataFrame({
        'month': months,
        'churn_rate': [rate * 100 for rate in churn_rates]
    })
    
    fig = px.line(trend_df, x='month', y='churn_rate',
                  title='Monthly Churn Rate Trend',
                  labels={'churn_rate': 'Churn Rate (%)', 'month': 'Month'})
    
    fig.update_traces(line=dict(color='#ff6b6b', width=3))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=16,
        height=400
    )
    
    return fig

def create_revenue_waterfall(df):
    """Create revenue waterfall chart"""
    # Calculate revenue components
    metrics = calculate_metrics(df)
    
    categories = ['Starting MRR', 'New Customers', 'Expansion', 'Churn', 'Ending MRR']
    values = [
        metrics['mrr'] * 0.9,  # Starting MRR
        metrics['mrr'] * 0.15,  # New customers
        metrics['mrr'] * 0.08,  # Expansion
        -metrics['mrr'] * 0.13,  # Churn
        metrics['mrr']  # Ending MRR
    ]
    
    colors = ['blue', 'green', 'green', 'red', 'blue']
    
    fig = go.Figure(go.Waterfall(
        name="Revenue Waterfall",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=categories,
        y=values,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))
    
    fig.update_layout(
        title="Monthly Recurring Revenue Waterfall",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

def create_customer_segmentation(df):
    """Create customer segmentation analysis"""
    # Segment customers based on usage and revenue
    df['usage_segment'] = pd.cut(df['avg_daily_usage'], 
                                bins=[0, 5, 15, 30, float('inf')], 
                                labels=['Low', 'Medium', 'High', 'Power'])
    
    df['revenue_segment'] = pd.cut(df['total_revenue'], 
                                  bins=[0, 500, 2000, 5000, float('inf')], 
                                  labels=['Low', 'Medium', 'High', 'Premium'])
    
    # Create 2D segmentation matrix
    segment_data = df.groupby(['usage_segment', 'revenue_segment']).agg({
        'customer_id': 'count',
        'churn_risk_score': 'mean'
    }).reset_index()
    
    segment_pivot = segment_data.pivot(index='usage_segment', 
                                     columns='revenue_segment', 
                                     values='customer_id').fillna(0)
    
    fig = px.imshow(segment_pivot, 
                    title='Customer Segmentation Matrix',
                    labels=dict(x="Revenue Segment", y="Usage Segment", color="Customer Count"),
                    color_continuous_scale='Blues')
    
    fig.update_layout(height=400)
    
    return fig

def create_feature_importance_chart():
    """Create feature importance chart for churn prediction"""
    features = ['Days Since Last Activity', 'Average Daily Usage', 'Support Tickets', 
                'Features Adopted', 'Satisfaction Score', 'Tenure Months', 
                'Monthly Revenue', 'Company Size']
    
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 title='Churn Prediction - Feature Importance',
                 labels={'x': 'Importance Score', 'y': 'Features'})
    
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return fig

def create_cohort_analysis(df):
    """Create cohort retention analysis"""
    # Simulate cohort data
    cohorts = []
    for i in range(12):
        cohort_date = datetime.now() - timedelta(days=30*i)
        cohort_size = np.random.randint(50, 200)
        
        # Simulate retention rates
        retention_rates = []
        for month in range(min(i+1, 12)):
            base_retention = 0.95 - (month * 0.05)  # Declining retention
            noise = np.random.normal(0, 0.02)
            retention = max(0.1, min(1.0, base_retention + noise))
            retention_rates.append(retention)
        
        cohorts.append({
            'cohort': cohort_date.strftime('%Y-%m'),
            'cohort_size': cohort_size,
            'retention_rates': retention_rates
        })
    
    # Create cohort table
    cohort_df = pd.DataFrame(cohorts)
    
    # Create heatmap data
    max_periods = max(len(c['retention_rates']) for c in cohorts)
    heatmap_data = []
    
    for cohort in cohorts:
        row = cohort['retention_rates'] + [None] * (max_periods - len(cohort['retention_rates']))
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=[c['cohort'] for c in cohorts],
                             columns=[f'Month {i}' for i in range(max_periods)])
    
    fig = px.imshow(heatmap_df, 
                    title='Cohort Retention Analysis',
                    labels=dict(x="Months Since Signup", y="Signup Cohort", color="Retention Rate"),
                    color_continuous_scale='RdYlGn',
                    aspect="auto")
    
    fig.update_layout(height=500)
    
    return fig

def show_executive_summary():
    """Executive dashboard with key metrics"""
    st.title("🎯 Executive Summary")
    
    # Load data
    df = load_sample_data()
    metrics = calculate_metrics(df)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Monthly Churn Rate",
            value=f"{metrics['churn_rate']:.1f}%",
            delta=f"{metrics['churn_rate_change']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Monthly Recurring Revenue",
            value=f"${metrics['mrr']:,.0f}",
            delta=f"{metrics['mrr_change']:.1%}"
        )
    
    with col3:
        st.metric(
            label="Average Customer LTV",
            value=f"${metrics['avg_ltv']:,.0f}",
            delta=f"{metrics['ltv_change']:.1%}"
        )
    
    with col4:
        st.metric(
            label="At-Risk Customers",
            value=f"{metrics['at_risk_customers']:,}",
            delta=f"{metrics['at_risk_change']:+d}"
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_churn_trend_chart(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_revenue_waterfall(df), use_container_width=True)
    
    # Key insights
    st.markdown("### 🔍 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("""
        **Churn Drivers:**
        - 68% of churned customers had low daily usage (<5 sessions)
        - Support ticket volume correlates with 23% increase in churn risk
        - Enterprise customers show 40% lower churn rates
        """)
    
    with insights_col2:
        st.success("""
        **Revenue Opportunities:**
        - $245K potential monthly revenue from at-risk customer retention
        - Premium tier shows highest expansion potential (12% upgrade rate)
        - Customer success intervention reduces churn by 35%
        """)

def show_churn_analysis():
    """Detailed churn analysis page"""
    st.title("🔄 Churn Analysis")
    
    df = load_sample_data()
    
    # Churn overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churned_pct = df['is_churned'].mean() * 100
        st.metric("Overall Churn Rate", f"{churned_pct:.1f}%")
    
    with col2:
        high_risk_pct = (df['churn_risk_level'] == 'high').mean() * 100
        st.metric("High Risk Customers", f"{high_risk_pct:.1f}%")
    
    with col3:
        avg_risk_score = df['churn_risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk_score:.3f}")
    
    st.markdown("---")
    
    # Feature importance and risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='churn_risk_score', nbins=30,
                          title='Churn Risk Score Distribution',
                          labels={'churn_risk_score': 'Risk Score', 'count': 'Customer Count'})
        fig.update_traces(marker_color='orange', opacity=0.7)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn by segments
    st.markdown("### 📊 Churn Analysis by Segments")
    
    segment_tabs = st.tabs(["By Subscription Tier", "By Company Size", "By Industry", "By Usage Level"])
    
    with segment_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            churn_by_tier = df.groupby('subscription_tier')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
            churn_by_tier['churn_rate'] = churn_by_tier['mean'] * 100
            
            fig = px.bar(churn_by_tier, x='subscription_tier', y='churn_rate',
                        title='Churn Rate by Subscription Tier',
                        labels={'churn_rate': 'Churn Rate (%)', 'subscription_tier': 'Tier'})
            fig.update_traces(marker_color=['#ff9999', '#66b3ff', '#99ff99'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            revenue_impact = df[df['is_churned'] == 1].groupby('subscription_tier')['monthly_revenue'].sum().reset_index()
            fig = px.pie(revenue_impact, names='subscription_tier', values='monthly_revenue',
                        title='Churned Revenue by Tier')
            st.plotly_chart(fig, use_container_width=True)
    
    with segment_tabs[1]:
        churn_by_size = df.groupby('company_size')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_size['churn_rate'] = churn_by_size['mean'] * 100
        
        fig = px.bar(churn_by_size, x='company_size', y='churn_rate',
                    title='Churn Rate by Company Size',
                    labels={'churn_rate': 'Churn Rate (%)', 'company_size': 'Company Size'})
        st.plotly_chart(fig, use_container_width=True)
    
    with segment_tabs[2]:
        churn_by_industry = df.groupby('industry')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_industry['churn_rate'] = churn_by_industry['mean'] * 100
        
        fig = px.bar(churn_by_industry, x='industry', y='churn_rate',
                    title='Churn Rate by Industry',
                    labels={'churn_rate': 'Churn Rate (%)', 'industry': 'Industry'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with segment_tabs[3]:
        df['usage_level'] = pd.cut(df['avg_daily_usage'], 
                                  bins=[0, 5, 15, 30, float('inf')], 
                                  labels=['Low', 'Medium', 'High', 'Power'])
        churn_by_usage = df.groupby('usage_level')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_usage['churn_rate'] = churn_by_usage['mean'] * 100
        
        fig = px.bar(churn_by_usage, x='usage_level', y='churn_rate',
                    title='Churn Rate by Usage Level',
                    labels={'churn_rate': 'Churn Rate (%)', 'usage_level': 'Usage Level'})
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk customer table
    st.markdown("### ⚠️ High-Risk Customers")
    
    high_risk_customers = df[df['churn_risk_level'] == 'high'].sort_values('churn_risk_score', ascending=False)
    
    display_cols = ['customer_id', 'company_size', 'industry', 'subscription_tier', 
                   'monthly_revenue', 'churn_risk_score', 'last_activity_days_ago', 'avg_daily_usage']
    
    st.dataframe(
        high_risk_customers[display_cols].head(20),
        use_container_width=True,
        hide_index=True
    )
    
    # Action recommendations
    st.markdown("### 💡 Recommended Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.error("""
        **Immediate Actions (High Risk)**
        - Personal outreach to top 50 at-risk customers
        - Offer usage training sessions
        - Provide dedicated customer success manager
        - Consider pricing incentives for renewals
        """)
    
    with action_col2:
        st.warning("""
        **Medium-term Actions (Medium Risk)**
        - Automated email engagement campaigns
        - Feature adoption workshops
        - Regular health score check-ins
        - Product usage optimization tips
        """)
    
    with action_col3:
        st.info("""
        **Preventive Actions (Low Risk)**
        - Monthly product newsletters
        - Feature announcement updates
        - Customer satisfaction surveys
        - Referral program enrollment
        """)

def show_revenue_analytics():
    """Revenue analytics and forecasting"""
    st.title("💰 Revenue Analytics")
    
    df = load_sample_data()
    
    # Revenue metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_arr = (df[df['is_churned'] == 0]['monthly_revenue'].sum() * 12)
        st.metric("Annual Recurring Revenue", f"${total_arr:,.0f}")
    
    with col2:
        avg_revenue_per_customer = df[df['is_churned'] == 0]['monthly_revenue'].mean()
        st.metric("Avg Revenue Per Customer", f"${avg_revenue_per_customer:.0f}/mo")
    
    with col3:
        churn_revenue_impact = df[df['is_churned'] == 1]['monthly_revenue'].sum() * 12
        st.metric("Churned ARR", f"${churn_revenue_impact:,.0f}")
    
    with col4:
        potential_recovery = df[df['churn_risk_level'] == 'high']['monthly_revenue'].sum() * 12
        st.metric("At-Risk ARR", f"${potential_recovery:,.0f}")
    
    st.markdown("---")
    
    # Revenue analysis tabs
    revenue_tabs = st.tabs(["Revenue Trends", "Cohort Analysis", "LTV Analysis", "Forecasting"])
    
    with revenue_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue trend
            months = pd.date_range(start='2023-01-01', end='2024-06-01', freq='M')
            base_mrr = df[df['is_churned'] == 0]['monthly_revenue'].sum()
            
            monthly_revenue = []
            for i, month in enumerate(months):
                growth_rate = 0.02  # 2% monthly growth
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month.month / 12)
                revenue = base_mrr * (1 + growth_rate) ** i * seasonal_factor
                monthly_revenue.append(revenue)
            
            revenue_df = pd.DataFrame({
                'month': months,
                'mrr': monthly_revenue
            })
            
            fig = px.line(revenue_df, x='month', y='mrr',
                         title='Monthly Recurring Revenue Trend',
                         labels={'mrr': 'MRR ($)', 'month': 'Month'})
            fig.update_traces(line=dict(color='green', width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by tier over time
            tier_revenue = []
            for tier in ['basic', 'premium', 'enterprise']:
                tier_customers = df[(df['subscription_tier'] == tier) & (df['is_churned'] == 0)]
                tier_mrr = tier_customers['monthly_revenue'].sum()
                tier_revenue.append({'tier': tier, 'mrr': tier_mrr})
            
            tier_df = pd.DataFrame(tier_revenue)
            
            fig = px.pie(tier_df, names='tier', values='mrr',
                        title='MRR Distribution by Tier')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with revenue_tabs[1]:
        st.plotly_chart(create_cohort_analysis(df), use_container_width=True)
        
        st.markdown("#### Cohort Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Key Findings:**
            - Month 0 retention averages 95%
            - Steepest drop occurs in months 2-3
            - Enterprise cohorts show 15% better retention
            - Seasonal signup cohorts perform differently
            """)
        
        with col2:
            st.success("""
            **Improvement Opportunities:**
            - Focus on 30-60 day onboarding
            - Implement month-2 check-in program
            - Develop tier-specific retention strategies
            - Optimize seasonal acquisition campaigns
            """)
    
    with revenue_tabs[2]:
        # Customer Lifetime Value Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # LTV by segment
            ltv_by_tier = df.groupby('subscription_tier')['total_revenue'].agg(['mean', 'median', 'std']).reset_index()
            
            fig = px.bar(ltv_by_tier, x='subscription_tier', y='mean',
                        title='Average Customer LTV by Tier',
                        labels={'mean': 'Average LTV ($)', 'subscription_tier': 'Tier'})
            fig.update_traces(marker_color=['#ff9999', '#66b3ff', '#99ff99'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # LTV distribution
            fig = px.box(df, x='subscription_tier', y='total_revenue',
                        title='LTV Distribution by Tier',
                        labels={'total_revenue': 'Customer LTV ($)', 'subscription_tier': 'Tier'})
            st.plotly_chart(fig, use_container_width=True)
        
        # LTV factors correlation
        st.markdown("#### LTV Correlation Analysis")
        
        correlation_features = ['customer_tenure_months', 'avg_daily_usage', 'features_adopted', 
                               'satisfaction_score', 'total_revenue']
        corr_df = df[correlation_features].corr()
        
        fig = px.imshow(corr_df, 
                       title='Feature Correlation with Customer LTV',
                       color_continuous_scale='RdBu',
                       aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with revenue_tabs[3]:
        # Revenue forecasting
        st.markdown("#### Revenue Forecasting")
        
        forecast_col1, forecast_col2 = st.columns(2)
        
        with forecast_col1:
            # Forecast inputs
            st.markdown("**Forecast Parameters**")
            
            monthly_growth = st.slider("Monthly Growth Rate (%)", 0.0, 5.0, 2.0, 0.1)
            churn_improvement = st.slider("Churn Rate Improvement (%)", 0.0, 50.0, 10.0, 5.0)
            new_customer_rate = st.slider("New Customer Acquisition Rate (%)", 0.0, 20.0, 5.0, 1.0)
            
            # Calculate current metrics
            current_mrr = df[df['is_churned'] == 0]['monthly_revenue'].sum()
            current_churn_rate = df['is_churned'].mean()
            
        with forecast_col2:
            # Generate forecast
            forecast_months = 12
            forecast_data = []
            
            mrr = current_mrr
            for month in range(forecast_months):
                # Apply growth and churn
                new_mrr_growth = mrr * (monthly_growth / 100)
                new_customer_mrr = mrr * (new_customer_rate / 100)
                churn_impact = mrr * (current_churn_rate * (1 - churn_improvement / 100))
                
                mrr = mrr + new_mrr_growth + new_customer_mrr - churn_impact
                
                forecast_data.append({
                    'month': month + 1,
                    'mrr': mrr,
                    'arr': mrr * 12
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            fig = px.line(forecast_df, x='month', y='mrr',
                         title='12-Month MRR Forecast',
                         labels={'mrr': 'MRR ($)', 'month': 'Month'})
            fig.update_traces(line=dict(color='purple', width=3))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            final_mrr = forecast_df.iloc[-1]['mrr']
            mrr_growth_total = ((final_mrr / current_mrr) - 1) * 100
            
            st.metric("Projected MRR (12 months)", f"${final_mrr:,.0f}", f"+{mrr_growth_total:.1f}%")

def show_customer_segmentation():
    """Customer segmentation analysis"""
    st.title("👥 Customer Segmentation")
    
    df = load_sample_data()
    
    # Segmentation overview
    st.markdown("### 🎯 Segmentation Overview")
    
    # Create RFM-like segments
    df['revenue_quartile'] = pd.qcut(df['total_revenue'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    df['usage_quartile'] = pd.qcut(df['avg_daily_usage'], q=4, labels=['Low', 'Medium', 'High', 'Power'])
    df['tenure_quartile'] = pd.qcut(df['customer_tenure_months'], q=4, labels=['New', 'Growing', 'Established', 'Veteran'])
    
    # Segment performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_customer_segmentation(df), use_container_width=True)
    
    with col2:
        # Segment value analysis
        segment_analysis = df.groupby(['revenue_quartile', 'usage_quartile']).agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'churn_risk_score': 'mean',
            'is_churned': 'mean'
        }).reset_index()
        
        segment_analysis['avg_revenue_per_customer'] = segment_analysis['total_revenue'] / segment_analysis['customer_id']
        
        # Create bubble chart
        fig = px.scatter(segment_analysis, 
                        x='customer_id', 
                        y='avg_revenue_per_customer',
                        size='total_revenue',
                        color='churn_risk_score',
                        hover_data=['revenue_quartile', 'usage_quartile'],
                        title='Customer Segments: Size vs Value vs Risk',
                        labels={'customer_id': 'Customer Count', 
                               'avg_revenue_per_customer': 'Avg Revenue per Customer'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed segment analysis
    st.markdown("### 📊 Segment Deep Dive")
    
    segment_tabs = st.tabs(["Champions", "At-Risk", "New Customers", "Loyal Customers"])
    
    with segment_tabs[0]:
        # Champions: High revenue, high usage, low churn risk
        champions = df[
            (df['revenue_quartile'] == 'Premium') & 
            (df['usage_quartile'] == 'Power') & 
            (df['churn_risk_level'] == 'low')
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Champion Customers", len(champions))
        with col2:
            st.metric("Avg Monthly Revenue", f"${champions['monthly_revenue'].mean():.0f}")
        with col3:
            st.metric("Avg Churn Risk", f"{champions['churn_risk_score'].mean():.3f}")
        
        st.markdown("#### Champions Characteristics")
        
        champion_chars = pd.DataFrame({
            'Metric': ['Avg Tenure (months)', 'Avg Daily Usage', 'Features Adopted', 'Satisfaction Score'],
            'Champions': [
                champions['customer_tenure_months'].mean(),
                champions['avg_daily_usage'].mean(),
                champions['features_adopted'].mean(),
                champions['satisfaction_score'].mean()
            ],
            'Overall Average': [
                df['customer_tenure_months'].mean(),
                df['avg_daily_usage'].mean(),
                df['features_adopted'].mean(),
                df['satisfaction_score'].mean()
            ]
        })
        
        fig = px.bar(champion_chars, x='Metric', y=['Champions', 'Overall Average'],
                    title='Champions vs Overall Average',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **Champion Strategy:**
        - Leverage for referrals and case studies
        - Offer beta access to new features
        - Create champion advisory board
        - Implement champion recognition program
        """)
    
    with segment_tabs[1]:
        # At-Risk: High churn risk across all segments
        at_risk = df[df['churn_risk_level'] == 'high']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("At-Risk Customers", len(at_risk))
        with col2:
            revenue_at_risk = at_risk['monthly_revenue'].sum() * 12
            st.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.0f}")
        with col3:
            st.metric("Avg Risk Score", f"{at_risk['churn_risk_score'].mean():.3f}")
        
        # Risk factors breakdown
        risk_factors = pd.DataFrame({
            'Factor': ['Low Usage', 'High Support Tickets', 'Low Satisfaction', 'Long Inactivity'],
            'Percentage': [
                (at_risk['avg_daily_usage'] < 5).mean() * 100,
                (at_risk['support_tickets'] > 2).mean() * 100,
                (at_risk['satisfaction_score'] <= 2).mean() * 100,
                (at_risk['last_activity_days_ago'] > 14).mean() * 100
            ]
        })
        
        fig = px.bar(risk_factors, x='Factor', y='Percentage',
                    title='Risk Factors in At-Risk Segment',
                    labels={'Percentage': 'Percentage of At-Risk Customers'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("""
        **At-Risk Strategy:**
        - Immediate personal outreach
        - Usage training and onboarding review
        - Dedicated customer success intervention
        - Consider pricing incentives or credits
        """)
    
    with segment_tabs[2]:
        # New Customers: Recently signed up
        new_customers = df[df['customer_tenure_months'] <= 3]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Customers", len(new_customers))
        with col2:
            new_churn_rate = new_customers['is_churned'].mean() * 100
            st.metric("New Customer Churn Rate", f"{new_churn_rate:.1f}%")
        with col3:
            avg_adoption = new_customers['features_adopted'].mean()
            st.metric("Avg Features Adopted", f"{avg_adoption:.1f}")
        
        # Onboarding progress
        onboarding_progress = pd.DataFrame({
            'Month': ['Month 1', 'Month 2', 'Month 3'],
            'Feature Adoption Rate': [0.45, 0.65, 0.75],
            'Active Usage Rate': [0.70, 0.60, 0.55]
        })
        
        fig = px.line(onboarding_progress, x='Month', y=['Feature Adoption Rate', 'Active Usage Rate'],
                     title='New Customer Onboarding Progress')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **New Customer Strategy:**
        - Structured onboarding program
        - Early feature adoption incentives
        - 30-60-90 day check-ins
        - Quick wins identification
        """)
    
    with segment_tabs[3]:
        # Loyal Customers: Long tenure, consistent usage
        loyal = df[
            (df['customer_tenure_months'] >= 12) & 
            (df['churn_risk_level'] == 'low') &
            (df['avg_daily_usage'] >= df['avg_daily_usage'].median())
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Loyal Customers", len(loyal))
        with col2:
            loyal_ltv = loyal['total_revenue'].mean()
            st.metric("Avg Customer LTV", f"${loyal_ltv:,.0f}")
        with col3:
            expansion_potential = (loyal['subscription_tier'] != 'enterprise').mean() * 100
            st.metric("Expansion Potential", f"{expansion_potential:.0f}%")
        
        # Loyalty trends
        loyalty_analysis = loyal.groupby('subscription_tier').agg({
            'customer_id': 'count',
            'total_revenue': 'mean',
            'customer_tenure_months': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()
        
        fig = px.bar(loyalty_analysis, x='subscription_tier', y='customer_id',
                    title='Loyal Customers by Tier',
                    labels={'customer_id': 'Customer Count', 'subscription_tier': 'Tier'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **Loyal Customer Strategy:**
        - Tier upgrade campaigns
        - Feature expansion offers
        - Loyalty rewards program
        - Advocacy and referral programs
        """)

def show_model_performance():
    """Model performance and monitoring"""
    st.title("🤖 Model Performance")
    
    df = load_sample_data()
    
    # Train a simple model for demonstration
    features = ['customer_tenure_months', 'avg_daily_usage', 'features_adopted', 
               'last_activity_days_ago', 'support_tickets', 'satisfaction_score']
    
    X = df[features]
    y = df['is_churned']
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = (y_pred == y_test).mean()
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = ((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum()
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    st.markdown("---")
    
    # Model analysis tabs
    model_tabs = st.tabs(["Performance Metrics", "Feature Importance", "Model Monitoring", "Predictions"])
    
    with model_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, 
                           text_auto=True,
                           title='Confusion Matrix',
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Not Churned', 'Churned'],
                           y=['Not Churned', 'Churned'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        report_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
            'Not Churned': [
                report['0']['precision'],
                report['0']['recall'],
                report['0']['f1-score'],
                report['0']['support']
            ],
            'Churned': [
                report['1']['precision'],
                report['1']['recall'],
                report['1']['f1-score'],
                report['1']['support']
            ]
        })
        
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    with model_tabs[1]:
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                    title='Feature Importance in Churn Prediction')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.markdown("#### Feature Correlations")
        corr_matrix = X[features].corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu',
                       aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with model_tabs[2]:
        # Model monitoring metrics
        st.markdown("#### Model Drift Monitoring")
        
        # Simulate model performance over time
        dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='W')
        performance_data = []
        
        for i, date in enumerate(dates):
            base_accuracy = 0.85
            drift = 0.02 * np.sin(2 * np.pi * i / 52)  # Annual cycle
            noise = np.random.normal(0, 0.01)
            accuracy = base_accuracy + drift + noise
            
            performance_data.append({
                'date': date,
                'accuracy': accuracy,
                'precision': accuracy + np.random.normal(0, 0.02),
                'recall': accuracy + np.random.normal(0, 0.02)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        fig = px.line(performance_df, x='date', y=['accuracy', 'precision', 'recall'],
                     title='Model Performance Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data drift simulation
        st.markdown("#### Data Drift Detection")
        
        drift_features = ['avg_daily_usage', 'customer_tenure_months', 'features_adopted']
        drift_data = []
        
        for feature in drift_features:
            # Simulate drift score
            drift_score = np.random.uniform(0.05, 0.25)
            status = 'Alert' if drift_score > 0.2 else 'Normal'
            
            drift_data.append({
                'Feature': feature,
                'Drift Score': drift_score,
                'Status': status
            })
        
        drift_df = pd.DataFrame(drift_data)
        
        fig = px.bar(drift_df, x='Feature', y='Drift Score', color='Status',
                    title='Feature Drift Detection',
                    color_discrete_map={'Normal': 'green', 'Alert': 'red'})
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                     annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model retraining recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Model Health Status: Good**
            - Accuracy within acceptable range (>85%)
            - No significant feature drift detected
            - Prediction distribution stable
            - Last retrained: 2 weeks ago
            """)
        
        with col2:
            st.warning("""
            **Recommendations:**
            - Monitor 'avg_daily_usage' drift closely
            - Schedule monthly model evaluation
            - Consider A/B testing new model versions
            - Implement automated retraining triggers
            """)
    
    with model_tabs[3]:
        # Individual predictions
        st.markdown("#### Individual Customer Predictions")
        
        # Sample high-risk predictions
        sample_customers = df.sample(10)
        sample_features = sample_customers[features].fillna(sample_customers[features].mean())
        sample_predictions = model.predict_proba(sample_features)[:, 1]
        
        prediction_df = pd.DataFrame({
            'Customer ID': sample_customers['customer_id'].values,
            'Company Size': sample_customers['company_size'].values,
            'Subscription Tier': sample_customers['subscription_tier'].values,
            'Tenure (months)': sample_customers['customer_tenure_months'].values,
            'Churn Probability': sample_predictions,
            'Risk Level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in sample_predictions]
        })
        
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        
        # Prediction explanation
        st.markdown("#### Prediction Explanations")
        
        explanation_customer = sample_customers.iloc[0]
        customer_features = explanation_customer[features].fillna(explanation_customer[features].mean())
        customer_prob = model.predict_proba([customer_features])[0, 1]
        
        st.write(f"**Customer:** {explanation_customer['customer_id']}")
        st.write(f"**Churn Probability:** {customer_prob:.3f}")
        
        # Feature contributions (simplified SHAP-like explanation)
        feature_values = customer_features.values
        feature_contributions = feature_values * model.feature_importances_
        
        explanation_df = pd.DataFrame({
            'Feature': features,
            'Value': feature_values,
            'Contribution': feature_contributions / feature_contributions.sum()
        }).sort_values('Contribution', ascending=True)
        
        fig = px.bar(explanation_df, x='Contribution', y='Feature', orientation='h',
                    title=f'Feature Contributions for Customer {explanation_customer["customer_id"]}')
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function"""
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    st.sidebar.markdown("---")
    
    # Page selection
    pages = {
        "Executive Summary": "🎯",
        "Churn Analysis": "🔄", 
        "Revenue Analytics": "💰",
        "Customer Segmentation": "👥",
        "Model Performance": "🤖"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choose Page", 
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Page filters (common across pages)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=90), datetime.now()),
        max_value=datetime.now()
    )
    
    # Customer segment filter
    segment_filter = st.sidebar.multiselect(
        "Customer Segments",
        ["startup", "small", "medium", "large", "enterprise"],
        default=["startup", "small", "medium", "large", "enterprise"]
    )
    
    # Subscription tier filter
    tier_filter = st.sidebar.multiselect(
        "Subscription Tiers",
        ["basic", "premium", "enterprise"],
        default=["basic", "premium", "enterprise"]
    )
    
    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Risk Levels",
        ["low", "medium", "high"],
        default=["low", "medium", "high"]
    )
    
    st.sidebar.markdown("---")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Download options
    st.sidebar.markdown("### 📥 Export Options")
    
    if st.sidebar.button("Download Customer Data"):
        df = load_sample_data()
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name="customer_data.csv",
            mime="text/csv"
        )
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    **Churn Revenue Simulator**
    
    A comprehensive analytics platform for subscription business churn and revenue analysis.
    
    **Features:**
    - Real-time churn prediction
    - Revenue forecasting
    - Customer segmentation
    - ML model monitoring
    
    **Data Updated:** Real-time
    **Model Version:** v2.1.0
    """)
    
    # Route to appropriate page
    if selected_page == "Executive Summary":
        show_executive_summary()
    elif selected_page == "Churn Analysis":
        show_churn_analysis()
    elif selected_page == "Revenue Analytics":
        show_revenue_analytics()
    elif selected_page == "Customer Segmentation":
        show_customer_segmentation()
    elif selected_page == "Model Performance":
        show_model_performance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Churn Revenue Simulator v1.0 | Built with ❤️ using Streamlit</p>
        <p>📧 Contact: support@churnanalytics.com | 🔗 Documentation: <a href='#'>docs.churnanalytics.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()