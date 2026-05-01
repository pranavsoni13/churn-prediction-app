# Customer Churn Prediction App — Complete Project Documentation

---

# 1. Project Overview

## Purpose of the Project

The Customer Churn Prediction App is an end-to-end machine learning project designed to predict whether a customer is likely to churn (leave the service) or stay with the company.

The project focuses on solving a real-world business problem commonly found in industries such as:

* Telecom
* SaaS
* Subscription platforms
* Streaming services
* Banking & Insurance

## Problem It Solves

Customer acquisition is significantly more expensive than customer retention.

Businesses often lose customers without early warning, which results in:

* Revenue loss
* Increased marketing costs
* Reduced customer lifetime value
* Lower profitability

This system helps businesses:

* Identify at-risk customers early
* Understand major churn factors
* Take proactive retention actions

## Target Users

### Primary Users

* Business Analysts
* Data Scientists
* ML Engineers
* Product Managers
* Customer Retention Teams

### Secondary Users

* Students learning ML
* Academic project submissions
* Recruiters evaluating ML portfolio projects

## High-Level Summary

This project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Model optimization
* Threshold tuning
* Feature importance analysis
* Web app deployment using Streamlit

Final output:

A live deployed web app where users can input customer details and predict churn probability.

---

# 2. Tech Stack

## Languages

* Python 3.x

## Frameworks

* Streamlit (Web Application UI)

## Libraries

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

## Tools & Infrastructure

* Git
* GitHub
* Streamlit Cloud
* Jupyter Notebook
* VS Code

---

# 3. Project Structure

## Folder Structure

churn_project/

├── app/
│   └── app.py

├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py

├── models/
│   ├── churn_model.pkl
│   └── columns.pkl

├── notebooks/
│   └── churn_analysis.ipynb

├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv

├── requirements.txt
├── README.md
└── .gitignore

## Directory Explanation

### /app

Contains the deployed Streamlit frontend.

Key File:
app.py

Responsible for:

* UI rendering
* User input handling
* Prediction display

### /src

Core business logic.

Contains:

* preprocessing logic
* model training scripts
* prediction pipeline

### /models

Serialized trained model files.

Contains:

* saved ML model
* training feature columns

### /notebooks

Jupyter notebooks used during experimentation and EDA.

### /data

Raw dataset storage.

## Entry Points

### Development

src/train_model.py

### Production

app/app.py

---

# 4. Architecture

## Architecture Type

### Monolithic ML Application

This project follows a monolithic architecture where:

* preprocessing
* training
* prediction
* UI

are part of the same application.

Suitable for:

* academic projects
* small-scale ML deployment
* prototype systems

## Design Patterns Used

### Functional Modular Design

Each responsibility is isolated into dedicated modules:

* preprocessing module
* training module
* prediction module
* UI module

This improves:

* readability
* maintainability
* reusability

## Data Flow

Dataset
↓
Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Model Persistence (.pkl)
↓
Streamlit UI
↓
User Input
↓
Prediction Pipeline
↓
Final Prediction + Probability

---

# 5. Features & Modules

## Core Features

* Customer churn prediction
* Churn probability estimation
* Feature importance analysis
* Model comparison
* Threshold tuning
* Streamlit-based UI
* Live deployment

## Modules

### Module 1: Data Preprocessing

File:
src/data_preprocessing.py

Responsibilities:

* Handle missing values
* Convert data types
* Encode categorical columns
* One-hot encoding
* Feature alignment

### Module 2: Model Training

File:
src/train_model.py

Responsibilities:

* Load processed data
* Train Random Forest model
* Train Logistic Regression baseline
* Save trained model

### Module 3: Prediction Pipeline

File:
src/predict.py

Responsibilities:

* Load saved model
* Apply same preprocessing to user input
* Return prediction + probability

### Module 4: Streamlit UI

File:
app/app.py

Responsibilities:

* Input form rendering
* User interaction
* Display predictions

---

# 6. Code Flow

## Step-by-Step Execution Flow

### Step 1 — Dataset Load

pd.read_csv()

Loads Telco Customer Churn dataset.

### Step 2 — Cleaning

* TotalCharges converted to numeric
* Null values removed
* customerID removed

### Step 3 — Encoding

* Yes/No → Binary
* Gender → Numeric
* Remaining → One-hot encoding

### Step 4 — Train-Test Split

train_test_split()

80/20 split.

### Step 5 — Model Training

Primary model:

RandomForestClassifier()

### Step 6 — Threshold Tuning

Default threshold:
0.5

Updated threshold:
0.3

Improved churn recall significantly.

### Step 7 — Model Saving

joblib.dump()

Stores:

* trained model
* feature columns

### Step 8 — Streamlit Prediction

User inputs data → preprocessing → prediction → result display.

---

# 7. API & Data Layer

## API Structure

### Internal Function-Based API

No REST API implemented.

Communication occurs through:

* Python functions
* local model inference

## Main Prediction Interface

predict(data_dict)

### Input

Dictionary of customer features.

### Output

(prediction, probability)

## Data Model

### Target Variable

Churn

### Features

Examples:

* gender
* tenure
* MonthlyCharges
* TotalCharges
* Contract
* InternetService
* PaymentMethod

## Database Layer

### No Database Used

This project uses:

CSV → Pandas DataFrame

No SQL/NoSQL integration.

---

# 8. Configuration

## Environment Variables

Currently:

Not used

Possible future use:

MODEL_PATH=
APP_ENV=
DEBUG=

## Config Files

### requirements.txt

Defines project dependencies.

## Build Setup

### Local

pip install -r requirements.txt
streamlit run app/app.py

### Production

Deployment via Streamlit Cloud.

---

# 9. Testing

## Current State

No formal automated testing implemented.

This is a known improvement area.

## Recommended Future Testing

### Unit Tests

For:

* preprocessing
* prediction
* feature transformation

### Integration Tests

For:

* model loading
* Streamlit prediction flow

---

# 10. Dependencies

## Important Dependencies

### streamlit

Used for UI deployment.

### pandas

Used for:

* data loading
* preprocessing
* transformation

### scikit-learn

Used for:

* model training
* evaluation
* threshold tuning

### joblib

Used for:

* model persistence

---

# 11. Code Quality

## Strengths

* Modular design
* Clean separation of concerns
* End-to-end ML workflow
* Real deployment included
* Business-driven evaluation approach

## Issues / Anti-Patterns

* No automated tests
* No config management
* No database abstraction
* Limited input validation
* No logging framework

## Maintainability

Moderate

Good for:

* small teams
* academic projects

Needs improvement for:

* production systems

---

# 12. Security

## Current State

Minimal security requirements due to:

* public demo project
* no authentication
* no sensitive data

## Potential Vulnerabilities

* No input validation
* No abuse protection
* No access control

---

# 13. Performance

## Current Performance

Good for small-scale usage.

Prediction latency is minimal.

## Bottlenecks

Potential issues:

* model loading on startup
* no caching strategy
* no batch inference

## Optimization Opportunities

* caching model load
* Redis layer
* API-based serving
* Docker deployment

---

# 14. Development Workflow

## Version Control

Git + GitHub

## Branching Strategy

Currently:

main branch only

Recommended:

* main
* develop
* feature/*

## CI/CD

Currently:

Not implemented

Recommended:

* GitHub Actions
* automated tests
* auto deployment validation

---

# 15. Recommendations

## Immediate Improvements

* Add unit testing
* Add input validation
* Improve Streamlit UX
* Add model versioning

## Refactoring Suggestions

* Move constants to config files
* Add service layer abstraction
* Introduce prediction service class

## Scalability Ideas

* FastAPI backend
* PostgreSQL integration
* Docker containerization
* Cloud deployment (AWS/GCP/Azure)

---

# 16. Assumptions & Unknowns

## Assumptions Made

* Single-user usage pattern
* CSV-based input source
* No authentication needed
* Local model inference only

## Missing Information

* Formal testing strategy
* Monitoring/logging setup
* CI/CD configuration
* Production-grade deployment architecture

🙌 Author
Pranav Soni AI & Data Science Student

⭐ If you like this project, consider giving it a star!
