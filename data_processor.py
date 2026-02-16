import pandas as pd
import numpy as np

def load_data(fg_path, trade_path):
    """Loads and standardizes date formats for Fear & Greed and Trade data."""
    print("Loading data...")
    try:
        fg_df = pd.read_csv(fg_path)
        trade_df = pd.read_csv(trade_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None

    # Fear & Greed Dates
    if 'date' in fg_df.columns:
        fg_df['date'] = pd.to_datetime(fg_df['date'])
    else:
        fg_df['date'] = pd.to_datetime(fg_df['timestamp'], unit='s')
    
    fg_df['date'] = fg_df['date'].dt.normalize()
    fg_df = fg_df.drop_duplicates(subset=['date']).sort_values('date')

    # Trade Dates
    # Try parsing 'Timestamp IST' first, else fallback to 'Timestamp'
    try:
         trade_df['datetime'] = pd.to_datetime(trade_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    except Exception:
        trade_df['datetime'] = pd.to_datetime(trade_df['Timestamp'], unit='ms')
    
    trade_df['date'] = trade_df['datetime'].dt.normalize()
    
    # Numerical Conversions
    for col in ['Closed PnL', 'Size USD', 'Fee']:
        trade_df[col] = pd.to_numeric(trade_df[col], errors='coerce').fillna(0)
        
    trade_df['Net PnL'] = trade_df['Closed PnL'] - trade_df['Fee']
    return fg_df, trade_df

def feature_engineering(fg_df, trade_df):
    """Generates rolling features and targets."""
    print("Engineering features...")
    
    # 1. Merge Sentiment
    # We want yesterday's sentiment to predict today/tomorrow, but let's just merge on date first
    df = pd.merge(trade_df, fg_df[['date', 'value', 'classification']], on='date', how='left')
    
    # 2. Trader-Level Aggregates (Daily)
    # We need to aggregate trades to the daily level per trader to create a time series
    daily_stats = df.groupby(['Account', 'date']).agg({
        'Net PnL': 'sum',
        'Size USD': 'sum', # Total Volume
        'Fee': 'sum',
        'Trade ID': 'count', # Num Trades
    }).rename(columns={'Trade ID': 'Trade Count'}).reset_index()
    
    # Merge Sentiment back to daily stats
    daily_stats = pd.merge(daily_stats, fg_df[['date', 'value']], on='date', how='left')
    daily_stats = daily_stats.rename(columns={'value': 'sentiment_value'})
    
    # 3. Rolling Features (Past 7 days)
    daily_stats = daily_stats.sort_values(['Account', 'date'])
    
    metrics = ['Net PnL', 'Size USD', 'Trade Count']
    
    for m in metrics:
        daily_stats[f'rolling_7d_{m}_mean'] = daily_stats.groupby('Account')[m].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        daily_stats[f'rolling_7d_{m}_std'] = daily_stats.groupby('Account')[m].transform(lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))

    # 4. Target Generation
    # Predict Next Day's Profitability (Binary)
    daily_stats['next_day_pnl'] = daily_stats.groupby('Account')['Net PnL'].shift(-1)
    daily_stats['target_profitable'] = (daily_stats['next_day_pnl'] > 0).astype(int)
    
    # Predict Next Day's Volatility (using rolling std of next few days? or just next day squared error? 
    # Let's use next day's absolute PnL as a proxy for volatility/magnitude if actual volatility is hard on daily data)
    # OR: Predict volatility over the NEXT 3 days
    daily_stats['target_volatility'] = daily_stats.groupby('Account')['Net PnL'].transform(lambda x: x.shift(-1).rolling(window=3).std())
    
    # Drop rows without targets (last rows per trader)
    model_df = daily_stats.dropna(subset=['target_profitable', 'target_volatility']).copy()
    
    return model_df

if __name__ == "__main__":
    fg_path = 'Data Scinence intern proj/fear_greed_index.csv'
    trade_path = 'Data Scinence intern proj/historical_data.csv'
    
    fg_df, trade_df = load_data(fg_path, trade_path)
    if fg_df is not None:
        model_df = feature_engineering(fg_df, trade_df)
        print(f"Processed DataFrame Shape: {model_df.shape}")
        print(model_df.head())
        model_df.to_csv('processed_data.csv', index=False)
        print("Saved processed_data.csv")
