# 🔄 Churn Revenue Simulator
A comprehensive end-to-end data platform showcasing modern Data Engineering, Analytics Engineering, and Data Science workflows for subscription business churn and revenue analysis.








🎯 Project Overview
This project demonstrates a production-ready data platform designed to predict customer churn and provide deep insights into revenue patterns for SaaS (Software as a Service) businesses. It encapsulates a complete modern data stack, from raw data ingestion to interactive analytical applications, showcasing best practices in:

Data Engineering: Robust, automated pipelines for ingesting diverse customer data (e.g., subscription details, usage logs, support interactions). Includes built-in mechanisms for data quality monitoring to ensure reliable datasets.
Analytics Engineering: Transformational workflows to clean, combine, and model raw data into easily consumable datasets. This involves creating consistent dimensions (e.g., customers, companies) and facts (e.g., subscriptions, transactions), ready for analysis and machine learning.
Data Science: Development, training, and deployment of a machine learning model specifically for predicting customer churn. Emphasizes experimentation tracking and model versioning for continuous improvement.
Applications: User-facing interfaces that leverage the processed data and ML predictions for business decision-making, including an interactive dashboard for churn and revenue analytics, and a backend API for programmatic access to insights.
✨ Features
Automated Data Ingestion & Orchestration: Leverage Apache Airflow for scheduling and monitoring data pipelines, from source to curated layers.
Declarative Data Transformation: Use DBT (Data Build Tool) for defining, testing, and documenting data models in a version-controlled, collaborative manner.
Comprehensive Data Quality: Implement Great Expectations for defining and validating data quality rules at various stages of the pipeline, ensuring data integrity.
Churn Prediction Model: Train and evaluate a machine learning model to identify customers at high risk of churning, enabling proactive retention strategies.
ML Experimentation & Model Management: Utilize MLflow for tracking machine learning experiments, logging parameters, metrics, and managing model versions.
Interactive Analytics Dashboard: A Streamlit-powered dashboard offering visual insights into churn rates, revenue trends, customer segmentation, and risk profiling.
Predictive API Backend: A FastAPI service to expose ML model predictions, allowing other applications or systems to query churn risk in real-time.
🏗️ Architecture
The platform is designed with a modular and scalable architecture, depicted below:

Code snippet

graph LR
    A[Raw Data Source] --> B(Data Engineering Layer)
    B --> C(Analytics Engineering Layer)
    C --> D(Data Science Layer)
    D --> E(Applications Layer)

    B -- Orchestration --> B1[Apache Airflow]
    B -- Data Quality --> B2[Great Expectations]
    B -- Data Storage --> PostgreSQLDB[PostgreSQL/Data Lake]

    C -- Transformations & Modeling --> C1[DBT]
    C -- Data Testing --> C2[DBT Tests]
    C -- Output --> CuratedData[Curated Data]

    D -- Experiment Tracking --> D1[MLflow]
    D -- Model Training --> D2[Python ML Models]
    D -- Model Registry --> MLflowModelRegistry[MLflow Model Registry]

    E -- Interactive UI --> E1[Streamlit Dashboard]
    E -- API Endpoint --> E2[FastAPI Backend]

    PostgreSQLDB --> C
    CuratedData --> D
    MLflowModelRegistry --> E2
    CuratedData --> E1
    CuratedData --> E2
Flow Description:

Raw Data Source: Represents various origins of customer data (e.g., transactional databases, web analytics, support systems).
Data Engineering Layer: Airflow orchestrates pipelines to ingest raw data, often storing it in a PostgreSQL database (acting as a data lake/warehouse). Great Expectations is used here to validate data quality before further processing.
Analytics Engineering Layer: DBT transforms the raw, ingested data into structured, clean, and modeled datasets. DBT Tests ensure the integrity and business logic of these transformations, producing "Curated Data."
Data Science Layer: The curated data feeds into the ML pipeline, where Python ML models (e.g., using scikit-learn) are trained to predict churn. MLflow tracks experiments, logs model metrics, and manages model versions in its Model Registry.
Applications Layer:
The Streamlit Dashboard consumes curated data (and potentially real-time predictions from FastAPI) to offer interactive analytics.
The FastAPI Backend serves predictions from registered ML models, allowing other systems to integrate churn predictions.
🛠️ Technologies Used
This project leverages a modern data stack, including:

Orchestration: Apache Airflow - Programmatically author, schedule, and monitor workflows.
Data Transformation: DBT (Data Build Tool) - Transform, test, and document data in your data warehouse.
Data Quality: Great Expectations - Data quality tests, documentation, and profiling.
Machine Learning Ops: MLflow - Manage the ML lifecycle, including experimentation, reproducibility, and deployment.
Database: PostgreSQL - Robust relational database for data storage.
Web Framework (Dashboard): Streamlit - Quickly build and deploy data apps.
Web Framework (API): FastAPI - Modern, fast (high-performance) web framework for building APIs.
Containerization: Docker & Docker Compose - Containerize applications and orchestrate multi-container Docker applications.
Programming Language: Python 3.9+ - Primary language for all scripts and applications.
Data Manipulation: Pandas - For data analysis and manipulation.
Visualization: Plotly Express & Plotly Graph Objects - For interactive charts and graphs.
🚀 Getting Started
Follow these steps to set up and run the Churn Revenue Simulator locally:

Prerequisites
Docker installed on your machine.
Docker Compose (usually comes with Docker Desktop).
Clone the Repository
Bash

git clone https://github.com/your-username/churn-revenue-simulator.git
cd churn-revenue-simulator
Environment Setup
Create a .env file in the root directory of the project based on template.env. This file will store environment variables for database credentials and other configurations.

Bash

cp template.env .env
# Edit .env with your desired settings, e.g., PostgreSQL credentials
Build and Start Services
Use Docker Compose to build all service images and start the containers. This will launch PostgreSQL, Airflow, MLflow, Streamlit, and FastAPI.

Bash

docker-compose up --build -d
This command might take a few minutes on the first run as it downloads images and builds custom ones.

Initialize Airflow
Once Airflow is up, you'll need to initialize its database and create an admin user.

Bash

# Wait a few moments for Airflow to fully start (e.g., 60 seconds)
docker exec -it airflow-webserver airflow db migrate
docker exec -it airflow-webserver airflow users create \
    --username admin --firstname John --lastname Doe --role Admin \
    --email admin@example.com --password admin
(Adjust username, password, email as needed).

Load Initial Data & Run DBT
You'll need to trigger Airflow DAGs to ingest the sample data and run the initial DBT transformations.

Access Airflow UI: Open your browser and navigate to http://localhost:8080. Log in with the credentials you just created (admin/admin).
Unpause DAGs: Navigate to the "DAGs" page. Unpause (toggle on) the following DAGs in order:
ingest_raw_data_dag
dbt_transform_data_dag
train_churn_model_dag (optional, for ML pipeline)
Trigger DAGs: Manually trigger ingest_raw_data_dag first, then dbt_transform_data_dag, and then train_churn_model_dag. Monitor their progress.
🖥️ Usage
Once all services are up and initial DAGs have run successfully:

Streamlit Dashboard: Access the interactive dashboard at http://localhost:8501. Explore customer churn metrics, revenue insights, and risk segments.
FastAPI Backend (API): The API documentation (Swagger UI) is available at http://localhost:8000/docs. You can test endpoints like /predict_churn with sample data.
Airflow UI: Manage and monitor your data pipelines at http://localhost:8080.
MLflow UI: Track ML experiments and manage models at http://localhost:5000.
📸 Screenshots
(To be added: Include screenshots of your Streamlit dashboard here to give a quick visual overview of the application.)

📂 Project Structure
.
├── airflow/                  # Airflow DAGs and configurations
│   ├── dags/                 # Contains data ingestion and DBT orchestration DAGs
│   └── ...
├── dbt/                      # DBT project for data transformations
│   ├── models/               # SQL models for cleaning, staging, and final data marts
│   ├── tests/                # DBT data tests
│   └── ...
├── applications/
│   ├── dashboard/            # Streamlit application for dashboard
│   │   └── churn_dashboard.py
│   └── api/                  # FastAPI application for ML predictions
│       └── main.py
├── data/                     # Sample raw data files (e.g., CSVs)
├── models/                   # Placeholder for trained ML models (managed by MLflow)
├── scripts/                  # Utility scripts (e.g., for data generation, local testing)
├── .env.template             # Template for environment variables
├── docker-compose.yml        # Docker Compose configuration for all services
├── Dockerfile.airflow        # Dockerfile for custom Airflow image
├── Dockerfile.app            # Dockerfile for Streamlit/FastAPI app
├── requirements.txt          # Python dependencies for the project
└── README.md
🤝 Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature-name).
Make your changes and ensure tests pass.
Commit your changes (git commit -m 'feat: Add new feature').
Push to the branch (git push origin feature/your-feature-name).
Create a Pull Request.
Please ensure your code adheres to the project's coding standards and includes appropriate tests.
