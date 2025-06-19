
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Churn Revenue Dashboard", layout="wide")
st.title("🔄 Churn Revenue Simulator - Executive Dashboard")

# Load example data
df = pd.read_csv("data/raw/customers.csv")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{df['customer_id'].nunique()}")
col2.metric("Avg Tenure (Months)", f"{df['customer_tenure_months'].mean():.1f}")
col3.metric("Churn Rate", f"{(df['churn_risk_score'] > 0.7).mean()*100:.1f}%")

# Churn Risk Histogram
st.subheader("Churn Risk Distribution")
fig1 = px.histogram(df, x="churn_risk_score", nbins=20, title="Churn Risk Score Histogram")
st.plotly_chart(fig1, use_container_width=True)

# Revenue Breakdown by Industry
if 'industry' in df.columns:
    st.subheader("Revenue by Industry")
    rev_df = df.groupby('industry')['total_revenue'].sum().reset_index()
    fig2 = px.bar(rev_df, x='industry', y='total_revenue', title="Total Revenue by Industry")
    st.plotly_chart(fig2, use_container_width=True)

# Time Series Placeholder
st.subheader("Time Series (Simulated)")
dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
monthly_churn = pd.Series((df['churn_risk_score'] > 0.7).sample(12, replace=True).values, index=dates)
fig3 = px.line(monthly_churn, title="Monthly Churn Indicator (Sample)")
st.plotly_chart(fig3, use_container_width=True)
