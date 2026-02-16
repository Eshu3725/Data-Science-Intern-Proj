import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib

def load_processed_data(path='processed_data.csv'):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def train_predictive_models(df):
    print("Training predictive models...")
    
    # Features
    feature_cols = [
        'rolling_7d_Net PnL_mean', 'rolling_7d_Net PnL_std',
        'rolling_7d_Size USD_mean', 'rolling_7d_Size USD_std',
        'rolling_7d_Trade Count_mean', 'rolling_7d_Trade Count_std',
        'sentiment_value'
    ]
    
    X = df[feature_cols]
    y_class = df['target_profitable']
    y_reg = df['target_volatility']
    
    # Split (Shuffle=False to simulate time-series somewhat, though strictly we should split by date)
    # Using random split for simplicity as per "Simple predictive model" request, 
    # but let's try to be slightly robust
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 1. Classification Model (Next Day Profitability)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_class_train)
    y_pred_class = clf.predict(X_test)
    
    print("\n--- Profitability Classification (Next Day > 0) ---")
    print(f"Accuracy: {accuracy_score(y_class_test, y_pred_class):.4f}")
    print(classification_report(y_class_test, y_pred_class))
    
    # 2. Regression Model (Volatility)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_pred_reg = reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    print("\n--- Volatility Regression ---")
    print(f"RMSE: {rmse:.4f}")
    
    return clf, reg, feature_cols

def perform_clustering(df):
    print("\nPerforming behavioral clustering...")
    
    # Aggregate data per trader for clustering (Archetypes)
    trader_profile = df.groupby('Account').agg({
        'Net PnL': 'sum',
        'Size USD': 'mean', # Avg Size
        'Trade Count': 'sum', # Total Trades
        'target_profitable': 'mean' # Win Rate (daily level)
    }).rename(columns={'target_profitable': 'Win Rate'})
    
    # Features for clustering
    cluster_cols = ['Net PnL', 'Size USD', 'Trade Count', 'Win Rate']
    X = trader_profile[cluster_cols]
    
    # Clean and Normalize
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    trader_profile['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Name clusters based on centroids
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_cols)
    print("\nCluster Centroids:")
    print(centroids)
    
    # Simple mapping logic (can be refined manually looking at centroids)
    # For now, just return specific cluster labels 0-3
    
    return trader_profile, kmeans, scaler

if __name__ == "__main__":
    df = load_processed_data()
    
    clf, reg, feature_cols = train_predictive_models(df)
    
    trader_clusters, kmeans, scaler = perform_clustering(df)
    
    # Save results
    # Save the models
    joblib.dump(clf, 'model_profitability.pkl')
    joblib.dump(reg, 'model_volatility.pkl')
    joblib.dump(kmeans, 'model_clustering.pkl')
    
    # Save clustered data for Dashboard
    trader_clusters.to_csv('trader_clusters.csv')
    print("\nSaved models and trader_clusters.csv")
    
    # Add predictions to main DF for dashboard visualization of "What-if" or historical view
    df['pred_profitability'] = clf.predict(df[feature_cols])
    df['pred_volatility'] = reg.predict(df[feature_cols])
    df.to_csv('dashboard_data.csv', index=False)
    print("Saved dashboard_data.csv")
