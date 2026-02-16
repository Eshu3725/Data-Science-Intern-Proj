import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Trader Insight Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv")
    clusters = pd.read_csv("trader_clusters.csv")
    return df, clusters

def main():
    st.title("Trader Insight & Prediction Dashboard")
    
    try:
        df, clusters = load_data()
    except FileNotFoundError:
        st.error("Data files not found. Please run 'models.py' first.")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Trader Analysis", "Clustering Insights", "Model Performance"])
    
    if page == "Overview":
        st.header("Results Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Traders Tracked", len(clusters))
        col2.metric("Total Trades Analyzed", int(clusters['Trade Count'].sum()))
        col3.metric("Avg Win Rate across Network", f"{clusters['Win Rate'].mean():.2%}")
        
        st.subheader("Aggregated Market Sentiment Impact")
        # Simple agg
        sentiment_agg = df.groupby('sentiment_value')['Net PnL'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.regplot(x=sentiment_agg.index, y=sentiment_agg.values, ax=ax, scatter_kws={'s': 50}, line_kws={'color':'red'})
        ax.set_xlabel("Fear & Greed Index (0-100)")
        ax.set_ylabel("Average Net PnL")
        ax.set_title("Market Sentiment vs Trader Profitability")
        st.pyplot(fig)

    elif page == "Trader Analysis":
        st.header("Individual Trader Analysis")
        
        accounts = clusters['Account'].unique()
        selected_account = st.selectbox("Select Account", accounts)
        
        trader_cluster_info = clusters[clusters['Account'] == selected_account].iloc[0]
        trader_history = df[df['Account'] == selected_account].sort_values('date')
        
        st.subheader(f"Account: {selected_account}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Archetype Cluster", trader_cluster_info['Cluster'])
        c2.metric("Overall Win Rate", f"{trader_cluster_info['Win Rate']:.2%}")
        c3.metric("Total PnL", f"${trader_cluster_info['Net PnL']:,.2f}")
        
        st.subheader("Performance History")
        st.line_chart(trader_history.set_index('date')['Net PnL'])
        
        st.subheader("Last Prediction")
        if not trader_history.empty:
            last_row = trader_history.iloc[-1]
            pred_profit = "Profitable" if last_row['pred_profitability'] == 1 else "Loss"
            pred_vol = last_row['pred_volatility']
            
            st.write(f"**Date:** {last_row['date']}")
            st.metric("Predicted Next Day Outcome", pred_profit)
            st.metric("Predicted Volatility Risk", f"${pred_vol:,.2f}")

    elif page == "Clustering Insights":
        st.header("Behavioral Archetypes")
        
        st.markdown("""
        Traders are clustered based on:
        - **Net PnL**: Total profitability
        - **Size USD**: Average position size (Risk appetite)
        - **Trade Count**: Frequency of trading
        - **Win Rate**: Consistency
        """)
        
        # Merge cluster info back explicitly if needed, but clusters df has it
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(data=clusters, x='Size USD', y='Net PnL', hue='Cluster', palette='viridis', s=100, alpha=0.7)
        plt.yscale('symlog') # Symlog to handle negative PnL nicely on log scale
        plt.xscale('log')
        plt.title("Trader Segments: Risk (Size) vs Reward (PnL)")
        st.pyplot(fig)
        
        st.dataframe(clusters.groupby('Cluster').mean(numeric_only=True).style.format("{:.2f}"))

    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        st.info("Models trained using Random Forest on rolling 7-day features + Sentiment.")
        
        st.markdown("""
        **Classification (Next Day Profitability):**
        - Target: Will the trader end the next day with > 0 PnL?
        - Accuracy: ~69% (on test set)
        
        **Regression (Volatility):**
        - Target: Standard deviation of PnL over next 3 days.
        - RMSE: ~$19k (varies by run)
        """)
        
        st.subheader("Feature Importance Strategy")
        st.write("The models heavily rely on recent PnL momentum ('rolling_7d_Net PnL_mean') and Market Sentiment.")

if __name__ == "__main__":
    main()
