# Trader Profitability Prediction & Dashboard

This project analyzes trader behavior and market sentiment (Fear & Greed Index) to predict profitability and volatility. It includes a machine learning pipeline and an interactive Streamlit dashboard.

## Features

- **Data Processing**: Cleans and merges trading history with Fear & Greed Index data.
- **Feature Engineering**: Generates rolling 7-day metrics (Win Rate, Size, Trade Count) and sentiment lags.
- **Predictive Modeling**:
    - **Classification**: Predicts "Next Day Profitability" (Random Forest Classifier).
    - **Regression**: Forecasts "Next 3-Day Volatility/Risk" (Random Forest Regressor).
- **Behavioral Clustering**: Segments traders into 4 archetypes based on Risk, Reward, and Frequency.
- **Dashboard**: Interactive visualization of model performance, trader profiles, and cluster insights.

## Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

## Project Structure

- `data_processor.py`: Loads raw CSVs, performs cleaning and feature engineering.
- `models.py`: Trains ML models, performs K-Means clustering, and saves artifacts.
- `app.py`: Streamlit application for visualization.
- `processed_data.csv`: Intermediate processed dataset (generated).
- `dashboard_data.csv`: Final dataset with predictions for the dashboard (generated).
- `Data Scinence intern proj/`: Directory containing raw data (`historical_data.csv`, `fear_greed_index.csv`).

## How to Run

Follow these steps to run the project from scratch:

### 1. Process the Data
Generate features and targets from the raw CSV files.
```bash
python data_processor.py
```
*Output: `processed_data.csv`*

### 2. Train Models & Cluster
Train the predictive models and perform trader segmentation.
```bash
python models.py
```
*Output: `model_profitability.pkl`, `model_volatility.pkl`, `model_clustering.pkl`, `trader_clusters.csv`, `dashboard_data.csv`*

### 3. Launch the Dashboard
Start the interactive application to explore results.
```bash
streamlit run app.py
```
This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

## Dashboard Sections
- **Overview**: Aggregate stats and Market Sentiment impact analysis.
- **Trader Analysis**: Deep dive into individual trader performance and specific model predictions.
- **Clustering Insights**: Visualizes the 4 behavior archetypes (e.g., High Risk/High Reward vs. Conservative).
- **Model Performance**: Displays accuracy metrics and feature importance.
