import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import logging


def connect():
    """Establish database connection"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='k007',
            database='stock_analysis'
        )
        
        print("Database Connected")
        return conn
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        print("Database Connection Failed")
    
def close(conn):
    """Close database connection"""
    if conn:
        conn.close()



class StockAnalyzer:
    def __init__(self):
        #self.conn = connection_string
        pass
        
    def load_data(self):
        query = """
        SELECT s.ticker, s.name, s.sector, 
               dp.date, dp.close, dp.volume,
               pm.daily_return, pm.cumulative_return
        FROM stocks s
        JOIN daily_prices dp ON s.id = dp.stock_id
        JOIN performance_metrics pm ON s.id = pm.stock_id AND dp.date = pm.date
        """
        conn = connect()
        cursor = conn.cursor()
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        stock_ids = cursor.fetchall()
        df = pd.DataFrame(stock_ids, columns=columns)
        
        
        conn.commit()
        cursor.close()
        close(conn)
        
        return df
    
    def calculate_volatility(self, df, window=30):
        """Calculate rolling volatility for each stock"""
        return df.groupby('ticker')['daily_return'].rolling(window=window).std().reset_index()
    
    def get_top_volatile_stocks(self, df, n=10):
        """Get top n volatile stocks"""
        volatility = df.groupby('ticker')['daily_return'].std().sort_values(ascending=False)
        return volatility.head(n)

    
    def calculate_correlation_matrix(self, df):
        """Calculate correlation matrix between stocks"""
        pivot_table = df.pivot(index='date', columns='ticker', values='daily_return')
        return pivot_table.corr()
    
    def calculate_sector_correlation(self, df):
        """Calculate sector-wise correlation matrix"""
        sector_pivot = df.groupby(['date', 'sector'])['daily_return'].mean().unstack()
        return sector_pivot.corr()
    
    def get_monthly_performers(self, df):
        """Get top 5 gainers and losers for each month"""
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_returns = df.groupby(['month', 'ticker'])['daily_return'].sum()
        #print(monthly_returns.dtypes)
        performers = {}
        for month in df['month'].unique():
            month_data = monthly_returns.xs(month)
            performers[month] = {
                'gainers': month_data.nlargest(5),
                'losers': month_data.nsmallest(5)
            }
        return performers
    
    def get_top_performers(self, df, n=10):
        latest_returns = df.groupby('ticker')['cumulative_return'].last()
        return latest_returns.nlargest(n)
    
    def get_worst_performers(self, df, n=10):
        latest_returns = df.groupby('ticker')['cumulative_return'].last()
        return latest_returns.nsmallest(n)
    
    def get_stock_ranking(self, df, n=10):
        """Identify top 10 best and worst performing stocks over the past year"""
        one_year_ago = datetime.now() - timedelta(days=365)
        df['date'] = pd.to_datetime(df['date'])
        yearly_data = df[df['date'] >= one_year_ago]
        latest_returns = yearly_data.groupby('ticker')['cumulative_return'].last()
        
        best_stocks = latest_returns.nlargest(n)
        worst_stocks = latest_returns.nsmallest(n)
        
        return best_stocks, worst_stocks
    
    def get_market_overview(self, df):
        """Calculate market summary and percentage of green vs. red stocks"""
        avg_performance = df['daily_return'].mean()
        latest_returns = df.groupby('ticker')['cumulative_return'].last()
        
        green_stocks = (latest_returns > 0).sum()
        red_stocks = (latest_returns <= 0).sum()
        total_stocks = len(latest_returns)
        
        green_percentage = (green_stocks / total_stocks) * 100
        red_percentage = (red_stocks / total_stocks) * 100
        
        return avg_performance, green_percentage, red_percentage

def main():
    st.set_page_config(page_title="Advanced Stock Analysis", layout="wide")
    st.title("Advanced Stock Market Analysis Dashboard")
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    df = analyzer.load_data()

    df['ticker'] = df['ticker'].astype('str')  # Convert to string
    df['name'] = df['name'].astype('str')  # Convert to string
    df['sector'] = df['sector'].astype('str')  # Convert to string
    
    df['close'] = df['close'].astype('float')  # Convert to float
    df['volume'] = df['volume'].astype('float')  # Convert to float
    df['daily_return'] = df['daily_return'].astype('float') *100 # Convert to float
    df['cumulative_return'] = df['cumulative_return'].astype('float') *100 # Convert to float
    
    #df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
    
    
    # Sidebar filters
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min(), df['date'].max())
    )
    
    # Filter data based on date range
    mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
    filtered_df = df[mask]
    
    
    
    
    
    # Market Overview
    st.header("Market Overview")
    avg_performance, green_percentage, red_percentage = analyzer.get_market_overview(df)
    
    st.metric(label="Average Market Performance (%)", value=f"{avg_performance:.2f}%")
    
    # Pie Chart for Green vs. Red Stocks
    market_data = pd.DataFrame({
        'Category': ['Green Stocks', 'Red Stocks'],
        'Percentage': [green_percentage, red_percentage]
    })
    
    fig_pie = px.pie(
        market_data, values='Percentage', names='Category',
        title="Market Sentiment: Green vs. Red Stocks",
        color='Category',
        color_discrete_map={'Green Stocks': 'green', 'Red Stocks': 'red'}
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
       
       
       
    
    
    
    st.header("Stock Performance Ranking")
    past_year_date = df['date'].max() - timedelta(days=365)
    past_year_data = df[df['date'] >= past_year_date]
    
    top_10_best = analyzer.get_top_performers(past_year_data, 10)
    top_10_worst = analyzer.get_worst_performers(past_year_data, 10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Best-Performing Stocks")
        fig_best = px.bar(
            top_10_best.reset_index(),
            x='ticker',
            y='cumulative_return',
            title="Top 10 Best-Performing Stocks",
            labels={'cumulative_return': 'Cumulative Return (%)'},
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig_best, use_container_width=True)
        st.dataframe(top_10_best)
    
    with col2:
        st.subheader("Top 10 Worst-Performing Stocks")
        fig_worst = px.bar(
            top_10_worst.reset_index(),
            x='ticker',
            y='cumulative_return',
            title="Top 10 Worst-Performing Stocks",
            labels={'cumulative_return': 'Cumulative Return (%)'},
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_worst, use_container_width=True)
        st.dataframe(top_10_worst)
        
    
    # 1. Volatility Analysis
    st.header("1. Volatility Analysis")
    top_volatile = analyzer.get_top_volatile_stocks(filtered_df)
    fig_volatile = px.bar(
        top_volatile.reset_index(),
        x='ticker',
        y='daily_return',
        title="Top 10 Most Volatile Stocks",
        labels={'daily_return': 'Volatility (Std Dev)'}
    )
    st.plotly_chart(fig_volatile, use_container_width=True)
    
    # 2. Cumulative Returns
    st.header("2. Top Performers - Cumulative Returns")
    top_performers = analyzer.get_top_performers(filtered_df)
    perf_df = filtered_df[filtered_df['ticker'].isin(top_performers.index)]
    fig_returns = px.line(
        perf_df,
        x='date',
        y='cumulative_return',
        color='ticker',
        title="Top 5 Performing Stocks - Cumulative Returns"
    )
    st.plotly_chart(fig_returns, use_container_width=True)
    
    # 3. Sector Performance
    st.header("3. Sector-wise Performance")
    sector_returns = filtered_df.groupby('sector')['daily_return'].mean().sort_values()
    fig_sector = px.bar(
        sector_returns.reset_index(),
        x='sector',
        y='daily_return',
        title="Average Returns by Sector",
        labels={'daily_return': 'Average Return'}
    )
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # 4. Correlation Matrix
    st.header("4. Stock Price Correlation")
    corr_matrix = analyzer.calculate_correlation_matrix(filtered_df)
    fig_corr = px.imshow(
        corr_matrix,
        title="Stock Return Correlation Matrix",
        labels=dict(color="Correlation"),
        width=800, height=800
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.header("Sector-wise Correlation Matrix")
    sector_corr = analyzer.calculate_sector_correlation(filtered_df)
    fig_sector_corr = px.imshow(
        sector_corr,
        title="Sector-wise Return Correlation Matrix",
        labels=dict(color="Correlation"),
        width=800, height=800
    )
    st.plotly_chart(fig_sector_corr, use_container_width=True)
    
    
    
    
    # 5. Monthly Top Gainers and Losers
    st.header("5. Monthly Top Gainers and Losers")
    monthly_performers = analyzer.get_monthly_performers(filtered_df)
    
    # Create tabs for each month
    months = list(monthly_performers.keys())
    selected_month = st.selectbox("Select Month", months)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Gainers")
        fig_gainers = px.bar(
            monthly_performers[selected_month]['gainers'],
            title=f"Top Gainers - {selected_month}",
            labels={'value': 'Return %'},
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig_gainers, use_container_width=True)
    
    with col2:
        st.subheader("Top 5 Losers")
        fig_losers = px.bar(
            monthly_performers[selected_month]['losers'],
            title=f"Top Losers - {selected_month}",
            labels={'value': 'Return %'},
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_losers, use_container_width=True)
    
    # Additional Metrics Table
    st.header("Summary Statistics")
    summary_stats = filtered_df.groupby('ticker').agg({
        'close': 'mean',
        'daily_return': ['mean', 'std'],
        'cumulative_return': 'last',
        'volume': 'mean'

        
    }).round(4)
    summary_stats.columns = ['Avg Price','Avg Daily Return', 'Volatility', 'Total Return', 'Avg Volume']
    st.dataframe(summary_stats)

if __name__ == "__main__":
    main()
    
