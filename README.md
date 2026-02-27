# ⚽ Player Performance & Transfer Risk Analytics Platform

An end-to-end football analytics system designed to quantify player
availability, financial exposure, and transfer instability using
predictive modeling and explainable AI.

This project simulates a professional scouting intelligence pipeline
used by football clubs.

------------------------------------------------------------------------

## 📌 Executive Summary

Football clubs manage high-value player assets under medical
uncertainty, contract risk, and market volatility.

This platform provides:

-   📉 Short-term injury probability prediction\
-   🔁 Transfer likelihood estimation\
-   🧠 Aggregated decision scoring (AL / İZLE / ALMA)\
-   📊 Interactive scout dashboard with explainability

The system converts structured player data into actionable technical
decisions.

------------------------------------------------------------------------

## 🏗 System Architecture

**Data Layer** - Synthetic structured player dataset

**Feature Engineering** - ACWR (Acute/Chronic Workload Ratio) - Load
delta - Congestion index - Contract decay logic - Agent risk scoring

**Model Layer** - Logistic Regression pipelines - Threshold optimization
(F1-based) - Model versioning - Explainability (SHAP + coefficients)

**Decision Engine** - Weighted injury & transfer aggregation -
Rule-based action classification

**Visualization Layer** - Streamlit interactive dashboard - Risk
distribution analytics - Radar visualization - KPI executive metrics

------------------------------------------------------------------------

## 🧠 Predictive Models

### 1️⃣ Injury Risk Model

Objective: Predict probability of injury within the next 14 days.

Engineered features: - Acute/Chronic Workload Ratio (ACWR) - Load
delta - Congestion score - Re-injury indicator - Position bias

Output: Probability ∈ \[0,1\]

Explainability: - SHAP global feature importance - Logistic regression
coefficient analysis

------------------------------------------------------------------------

### 2️⃣ Transfer Risk Model

Objective: Estimate likelihood of transfer risk.

Features: - Market value - Contract months remaining - Salary - Agent
risk - External interest level

Evaluation: - ROC-AUC - PR-AUC - F1-optimized decision threshold

------------------------------------------------------------------------

## 🎯 Decision Engine

Final Score = weighted combination of:

-   Injury Risk\
-   Transfer Risk

Action mapping:

  Final Score   Action
  ------------- ---------
  Low           🟢 AL
  Medium        🟡 İZLE
  High          🔴 ALMA

This converts raw probabilities into executive-ready decisions for
technical staff.

------------------------------------------------------------------------

## 📊 Streamlit Scout Dashboard

Interactive features:

-   🔴 High Injury Risk KPI\
-   🟡 Watchlist KPI\
-   🟢 Buy Candidates KPI\
-   Position-based risk distribution\
-   Top injury & transfer risk charts\
-   Player radar visualization\
-   SHAP global explainability\
-   Model coefficient visualization\
-   Advanced filtering (age, contract, action, position)

Run locally:

``` bash
streamlit run dashboard/app.py
```

------------------------------------------------------------------------

## 📁 Project Structure

    player-performance-transfer-risk/
    │
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │
    ├── models/
    │   ├── injury_model_v2.pkl
    │   ├── transfer_model_v2.pkl
    │
    ├── src/
    │   ├── injury_model.py
    │   ├── transfer_model.py
    │   ├── decision_board.py
    │   └── data/generate_synthetic.py
    │
    ├── dashboard/
    │   └── app.py
    │
    └── notebooks/

------------------------------------------------------------------------

## 🛠 Technology Stack

-   Python\
-   Pandas\
-   Scikit-learn\
-   SHAP\
-   Streamlit\
-   Matplotlib\
-   Joblib

------------------------------------------------------------------------

## 🔬 Design Principles

-   Interpretable models over black-box complexity\
-   Decision-oriented outputs instead of raw probabilities\
-   Domain-driven feature engineering\
-   Modular and scalable architecture

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Time-based validation split\
-   Player-level local SHAP explanation\
-   Scenario simulation engine\
-   Executive PDF export\
-   Real match & GPS dataset integration

------------------------------------------------------------------------

## ⚠ Disclaimer

The dataset is synthetic and generated for modeling demonstration
purposes.\
The architecture is designed to be directly adaptable to real club
datasets.
