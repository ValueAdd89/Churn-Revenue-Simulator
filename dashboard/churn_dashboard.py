# applications/dashboard/churn_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv_data():
    """Load and process the actual CSV data"""
    try:
        # Load all CSV files
        customers = pd.read_csv('data/customers.csv')
        subscriptions = pd.read_csv('data/subscriptions.csv')
        support_interactions = pd.read_csv('data/support_interactions.csv')
        usage_events = pd.read_csv('data/usage_events.csv')
        
        # Convert date columns
        customers['signup_date'] = pd.to_datetime(customers['signup_date'])
        subscriptions['subscription_start_date'] = pd.to_datetime(subscriptions['subscription_start_date'])
        subscriptions['subscription_end_date'] = pd.to_datetime(subscriptions['subscription_end_date'])
        support_interactions['interaction_date'] = pd.to_datetime(support_interactions['interaction_date'])
        usage_events['event_date'] = pd.to_datetime(usage_events['event_date'])
        
        return customers, subscriptions, support_interactions, usage_events
    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def create_analytics_dataset():
    """Create a unified analytics dataset from the CSV files"""
    customers, subscriptions, support_interactions, usage_events = load_csv_data()
    
    if customers is None:
        # Fallback to sample data if CSV files not found
        return load_sample_data()
    
    try:
        # Start with customers as base
        analytics_df = customers.copy()
        
        # Add subscription information (get latest active subscription per customer)
        active_subs = subscriptions[subscriptions['is_active'] == True].copy()
        latest_subs = active_subs.groupby('customer_id').first().reset_index()
        
        analytics_df = analytics_df.merge(
            latest_subs[['customer_id', 'subscription_tier', 'monthly_revenue', 'subscription_start_date', 'payment_method']], 
            on='customer_id', 
            how='left'
        )
        
        # Calculate customer tenure in months
        analytics_df['customer_tenure_months'] = (
            (datetime.now() - analytics_df['subscription_start_date']).dt.days / 30.44
        ).fillna(0).round(1)
        
        # Add support ticket counts and satisfaction
        support_stats = support_interactions.groupby('customer_id').agg({
            'interaction_id': 'count',
            'satisfaction_score': 'mean',
            'resolution_time_hours': 'mean'
        }).reset_index()
        support_stats.columns = ['customer_id', 'support_tickets', 'avg_satisfaction_score', 'avg_resolution_time']
        
        analytics_df = analytics_df.merge(support_stats, on='customer_id', how='left')
        analytics_df['support_tickets'] = analytics_df['support_tickets'].fillna(0)
        analytics_df['avg_satisfaction_score'] = analytics_df['avg_satisfaction_score'].fillna(3.0)
        
        # Add usage statistics
        usage_stats = usage_events.groupby('customer_id').agg({
            'session_duration_minutes': 'mean',
            'pages_viewed': 'sum',
            'api_calls': 'sum',
            'feature_used': 'nunique',
            'event_date': 'max'
        }).reset_index()
        usage_stats.columns = ['customer_id', 'avg_session_duration', 'total_pages_viewed', 
                              'total_api_calls', 'features_adopted', 'last_activity_date']
        
        analytics_df = analytics_df.merge(usage_stats, on='customer_id', how='left')
        
        # Calculate days since last activity
        analytics_df['last_activity_days_ago'] = (
            datetime.now() - analytics_df['last_activity_date']
        ).dt.days.fillna(30)
        
        # Calculate average daily usage (sessions per day based on tenure)
        analytics_df['avg_daily_usage'] = np.where(
            analytics_df['customer_tenure_months'] > 0,
            analytics_df['total_pages_viewed'] / (analytics_df['customer_tenure_months'] * 30.44),
            0
        )
        
        # Calculate total revenue (estimate based on tenure and monthly revenue)
        analytics_df['total_revenue'] = (
            analytics_df['monthly_revenue'] * analytics_df['customer_tenure_months']
        ).fillna(0)
        
        # Create churn risk scoring
        analytics_df = calculate_churn_risk(analytics_df)
        
        # Clean up missing values
        analytics_df = analytics_df.fillna({
            'monthly_revenue': 29,
            'subscription_tier': 'basic',
            'features_adopted': 1,
            'avg_daily_usage': 1,
            'total_revenue': 100
        })
        
        return analytics_df
        
    except Exception as e:
        st.error(f"Error creating analytics dataset: {e}")
        return load_sample_data()

def calculate_churn_risk(df):
    """Calculate churn risk based on available metrics"""
    try:
        # Normalize factors for risk calculation
        df['risk_inactivity'] = np.where(df['last_activity_days_ago'] > 30, 0.3, 0)
        df['risk_low_usage'] = np.where(df['avg_daily_usage'] < 1, 0.25, 0)
        df['risk_support'] = np.where(df['support_tickets'] > 5, 0.2, 0)
        df['risk_satisfaction'] = np.where(df['avg_satisfaction_score'] < 3, 0.15, 0)
        df['risk_new_customer'] = np.where(df['customer_tenure_months'] < 3, 0.1, 0)
        
        # Enterprise customers get risk reduction
        df['risk_reduction'] = np.where(df['subscription_tier'] == 'enterprise', -0.15, 0)
        
        # Calculate overall churn risk score
        df['churn_risk_score'] = (
            0.1 +  # Base risk
            df['risk_inactivity'] + 
            df['risk_low_usage'] + 
            df['risk_support'] + 
            df['risk_satisfaction'] + 
            df['risk_new_customer'] +
            df['risk_reduction']
        ).clip(0, 1)
        
        # Create risk levels
        df['churn_risk_level'] = pd.cut(
            df['churn_risk_score'], 
            bins=[0, 0.2, 0.5, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        # Simulate actual churn based on risk scores
        df['is_churned'] = np.random.binomial(1, df['churn_risk_score'], size=len(df))
        
        return df
    except Exception as e:
        st.error(f"Error calculating churn risk: {e}")
        return df

@st.cache_data  
def load_sample_data():
    """Fallback sample data if CSV files are not available"""
    np.random.seed(42)
    n_customers = 500
    
    return pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'company_size': np.random.choice(['startup', 'small', 'medium', 'large', 'enterprise'], n_customers),
        'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Education', 'Retail'], n_customers),
        'subscription_tier': np.random.choice(['basic', 'premium', 'enterprise'], n_customers),
        'monthly_revenue': np.random.choice([29, 99, 299], n_customers),
        'customer_tenure_months': np.random.exponential(scale=12, size=n_customers),
        'total_revenue': np.random.exponential(scale=2000, size=n_customers),
        'avg_daily_usage': np.random.exponential(scale=5, size=n_customers),
        'features_adopted': np.random.poisson(lam=3, size=n_customers),
        'last_activity_days_ago': np.random.exponential(scale=7, size=n_customers),
        'support_tickets': np.random.poisson(lam=2, size=n_customers),
        'avg_satisfaction_score': np.random.choice([1, 2, 3, 4, 5], n_customers),
        'churn_risk_score': np.random.uniform(0, 1, n_customers),
        'churn_risk_level': np.random.choice(['low', 'medium', 'high'], n_customers),
        'is_churned': np.random.binomial(1, 0.15, n_customers)
    })

def calculate_metrics(df):
    """Calculate key business metrics"""
    try:
        total_customers = len(df)
        churned_customers = df['is_churned'].sum() if 'is_churned' in df.columns else 0
        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
        
        # Calculate MRR from active customers
        active_customers = df[df['is_churned'] == 0] if 'is_churned' in df.columns else df
        mrr = active_customers['monthly_revenue'].sum() if 'monthly_revenue' in active_customers.columns else 0
        
        # Average customer metrics
        avg_ltv = df['total_revenue'].mean() if 'total_revenue' in df.columns else 0
        avg_tenure = df['customer_tenure_months'].mean() if 'customer_tenure_months' in df.columns else 0
        
        # At-risk customers
        at_risk = len(df[df['churn_risk_level'] == 'high']) if 'churn_risk_level' in df.columns else 0
        
        return {
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'churn_rate': churn_rate,
            'mrr': mrr,
            'avg_ltv': avg_ltv,
            'avg_tenure': avg_tenure,
            'at_risk_customers': at_risk
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}

def create_churn_by_segment_chart(df, segment_col):
    """Create churn rate chart by segment"""
    try:
        if segment_col not in df.columns or 'is_churned' not in df.columns:
            return None
            
        segment_stats = df.groupby(segment_col).agg({
            'customer_id': 'count',
            'is_churned': ['sum', 'mean']
        }).reset_index()
        
        segment_stats.columns = [segment_col, 'total_customers', 'churned_customers', 'churn_rate']
        segment_stats['churn_rate'] = segment_stats['churn_rate'] * 100
        
        fig = px.bar(
            segment_stats, 
            x=segment_col, 
            y='churn_rate',
            title=f'Churn Rate by {segment_col.replace("_", " ").title()}',
            labels={'churn_rate': 'Churn Rate (%)', segment_col: segment_col.replace("_", " ").title()}
        )
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating segment chart: {e}")
        return None

def create_revenue_trend_chart(df):
    """Create revenue trend chart"""
    try:
        if 'subscription_start_date' not in df.columns:
            return None
            
        # Group by month and calculate MRR
        df['signup_month'] = df['subscription_start_date'].dt.to_period('M')
        monthly_revenue = df.groupby('signup_month')['monthly_revenue'].sum().reset_index()
        monthly_revenue['signup_month'] = monthly_revenue['signup_month'].astype(str)
        
        fig = px.line(
            monthly_revenue, 
            x='signup_month', 
            y='monthly_revenue',
            title='Monthly Recurring Revenue Trend',
            labels={'monthly_revenue': 'MRR ($)', 'signup_month': 'Month'}
        )
        fig.update_layout(height=400)
        fig.update_xaxes(tickangle=45)
        return fig
    except Exception as e:
        st.error(f"Error creating revenue trend: {e}")
        return None

def main():
    """Main dashboard function"""
    st.title("🎯 Churn Revenue Analytics Dashboard")
    
    # Load and display data info
    with st.spinner("Loading data..."):
        df = create_analytics_dataset()
    
    if df is None or len(df) == 0:
        st.error("No data available")
        return
    
    st.success(f"✅ Loaded {len(df)} customer records")
    
    # Debug section
    with st.expander("🔍 Data Overview"):
        st.write(f"**Dataset shape:** {df.shape}")
        st.write(f"**Available columns:** {list(df.columns)}")
        st.dataframe(df.head(), use_container_width=True)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    if not metrics:
        st.error("Could not calculate metrics")
        return
    
    # Display key metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{metrics.get('total_customers', 0):,}")
    
    with col2:
        st.metric("Churn Rate", f"{metrics.get('churn_rate', 0):.1f}%")
    
    with col3:
        st.metric("Monthly Recurring Revenue", f"${metrics.get('mrr', 0):,.0f}")
    
    with col4:
        st.metric("At-Risk Customers", f"{metrics.get('at_risk_customers', 0):,}")
    
    st.markdown("---")
    
    # Analytics section
    st.markdown("### 📈 Analytics")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Churn Analysis", "Revenue Analysis", "Customer Segments"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'subscription_tier' in df.columns:
                chart = create_churn_by_segment_chart(df, 'subscription_tier')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        
        with col2:
            if 'company_size' in df.columns:
                chart = create_churn_by_segment_chart(df, 'company_size')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
    
    with tab2:
        revenue_chart = create_revenue_trend_chart(df)
        if revenue_chart:
            st.plotly_chart(revenue_chart, use_container_width=True)
        else:
            st.info("Revenue trend chart not available with current data structure")
    
    with tab3:
        if 'churn_risk_level' in df.columns:
            risk_dist = df['churn_risk_level'].value_counts()
            fig = px.pie(
                values=risk_dist.values, 
                names=risk_dist.index,
                title='Customer Risk Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # High-risk customers table
    if 'churn_risk_level' in df.columns:
        st.markdown("### ⚠️ High-Risk Customers")
        high_risk = df[df['churn_risk_level'] == 'high'].head(10)
        
        if len(high_risk) > 0:
            display_cols = [col for col in ['customer_id', 'company_name', 'subscription_tier', 
                           'monthly_revenue', 'churn_risk_score', 'last_activity_days_ago'] 
                           if col in high_risk.columns]
            st.dataframe(high_risk[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No high-risk customers found")

if __name__ == "__main__":
    main()
