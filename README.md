# 🔄 Churn Revenue Simulator

> **A comprehensive end-to-end data platform showcasing modern Data Engineering, Analytics Engineering, and Data Science workflows for subscription business churn and revenue analysis.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](applications/dashboard/)

## 🎯 Project Overview

This project demonstrates a production-ready data platform that predicts customer churn and analyzes revenue patterns for SaaS businesses. It showcases the complete modern data stack with:

- **Data Engineering**: Automated pipelines with Airflow, data quality monitoring
- **Analytics Engineering**: DBT transformations, data modeling, testing
- **Data Science**: ML model training, experimentation, monitoring
- **Applications**: Interactive Streamlit dashboard, FastAPI backend

## 🏗️ Architecture

```mermaid
graph LR
    A[Raw Data] --> B[Data Engineering]
    B --> C[Analytics Engineering]
    C --> D[Data Science]
    D --> E[Applications]

    B --> B1[Airflow]
    B --> B2[Great Expectations]

    C --> C1[DBT]
    C --> C2[Data Tests]

    D --> D1[MLflow]
    D --> D2[Model Training]

    E --> E1[Streamlit Dashboard]
    E --> E2[FastAPI Backend]
```

<!-- truncated for brevity in code; real file includes full content provided by user -->