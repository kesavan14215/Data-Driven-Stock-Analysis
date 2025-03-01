# stock_data_loader.py
import yaml
import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
import os
import yaml
import pandas as pd
from pathlib import Path

def read_yaml_files(folder):
    dataframes = []
    
    #for folder in folders:
            
    folder_path = Path(folder)
    #for yaml_file in folder_path.glob('**/*.yaml'):
    #    print(yaml_file)
    
    # Check if folder exists
    if not folder_path.exists():
        print(f"Warning: Folder {folder} does not exist")
        #continue
        
    # Get all yaml files in the folder
    yaml_files = list(folder_path.glob('**/*.yaml')) + list(folder_path.glob('**/*.yml'))
    #print(yaml_files)
    
    for file_path in yaml_files:
        try:
            # Read YAML file
            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            # Convert to DataFrame
            if isinstance(yaml_data, list):
                df = pd.DataFrame(yaml_data)
            elif isinstance(yaml_data, dict):
                df = pd.DataFrame([yaml_data])
            else:
                print(f"Warning: Unsupported YAML structure in {file_path}")
                continue
            
            # Add source file information
            #df['source_file'] = file_path.name
            #df['source_folder'] = folder_path.name
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    return pd.concat(dataframes, ignore_index=True)

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


def create_tables():
    """Create necessary database tables"""
    create_tables_queries = [
        """
        CREATE TABLE IF NOT EXISTS stocks (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        name VARCHAR(100),
        sector VARCHAR(50),
        UNIQUE KEY unique_ticker (ticker)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        stock_id BIGINT NOT NULL,
        date DATE NOT NULL,
        open DECIMAL(10, 2) NOT NULL,
        high DECIMAL(10, 2) NOT NULL,
        low DECIMAL(10, 2) NOT NULL,
        close DECIMAL(10, 2) NOT NULL,
        volume BIGINT NOT NULL,
        FOREIGN KEY (stock_id) REFERENCES stocks(id),
        UNIQUE KEY unique_stock_date (stock_id, date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS performance_metrics (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        stock_id BIGINT NOT NULL,
        date DATE NOT NULL,
        daily_return DECIMAL(10, 4),
        volatility DECIMAL(10, 4),
        cumulative_return DECIMAL(10, 4),
        FOREIGN KEY (stock_id) REFERENCES stocks(id),
        UNIQUE KEY unique_stock_date_metrics (stock_id, date)
        );
        """
    ]
    
    conn = connect()
    cursor = conn.cursor()
    for query in create_tables_queries:
        print(query)
        cursor.execute(query)
    cursor.close()
    close(conn)
        
def drop_tables():
    """Create necessary database tables"""
    drop_tables_queries = [
        "DROP TABLE IF EXISTS daily_prices",
        "DROP TABLE IF EXISTS performance_metrics",
        "DROP TABLE IF EXISTS stocks"
    ]
    
    conn = connect()
    cursor = conn.cursor()
    for query in drop_tables_queries:
        cursor.execute(query)
    cursor.close()
    close(conn)
    
def add_date_features(df):
    """Add various date-derived columns useful for financial analysis."""
    # Basic date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Quarter and semester
    df['quarter'] = df['date'].dt.quarter
    df['semester'] = (df['quarter'] + 1) // 2
    
    # Trading day features
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_quarter_start'] = df['date'].dt.is_quarter_start
    df['is_quarter_end'] = df['date'].dt.is_quarter_end
    df['is_year_start'] = df['date'].dt.is_year_start
    df['is_year_end'] = df['date'].dt.is_year_end
    
    # Time series features
    df['year_month'] = df['date'].dt.to_period('M')
    df['year_week'] = df['date'].dt.strftime('%Y-W%U')
    
    # Trading day of week/month/quarter/year
    df['trading_day_of_week'] = df.groupby(['Ticker', 'year', 'week_of_year']).cumcount() + 1
    df['trading_day_of_month'] = df.groupby(['Ticker', 'year', 'month']).cumcount() + 1
    df['trading_day_of_quarter'] = df.groupby(['Ticker', 'year', 'quarter']).cumcount() + 1
    df['trading_day_of_year'] = df.groupby(['Ticker', 'year']).cumcount() + 1
    
    # Previous trading day's date
    df['prev_trading_date'] = df.groupby('Ticker')['date'].shift(1)
    df['days_since_last_trade'] = (df['date'] - df['prev_trading_date']).dt.days
    df.to_csv("calendar_table.csv", index=False)
    return df
    

def compute_rsi(series, window=14):
    delta = series.diff()
    print(delta)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi




def calculate_metrics( df):
    """Calculate key performance metrics"""
    #metrics = {}
    
    # Calculate daily returns
    df['daily_return'] = df.groupby('Ticker')['close'].pct_change()
    
    # Calculate volatility (standard deviation of daily returns)
    df['volatility'] = df.groupby('Ticker')['daily_return'].transform('std')

    
    # Calculate cumulative returns
    df['cumulative_returns'] = df.groupby('Ticker')['daily_return'].cumsum()
    
    df['prev_close'] = df.groupby('Ticker')['close'].shift(1)
    
    # Calculate Indicators by Grouping by Ticker
    df['ma20'] = df.groupby('Ticker')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['ma50'] = df.groupby('Ticker')['close'].transform(lambda x: x.rolling(window=50).mean())
    #df['RSI']  = df.groupby('Ticker')['close'].transform(lambda x: compute_rsi(x))
    #df['RSI'] = df.groupby('Ticker', group_keys=False)['close'].apply(compute_rsi)
   
    return df

def load_to_database(df):

    conn = connect()
    cursor = conn.cursor()
        
    # Prepare daily prices data
    data_to_insert = [
        (
            row['Ticker'],
            row['COMPANY'],
            row['sector']
        )
        for _, row in df[['Ticker','COMPANY','sector']].drop_duplicates().iterrows()
    ]
    #print(data_to_insert)
    
    query = """
            INSERT IGNORE INTO stocks (ticker,name,sector) 
            VALUES (%s,%s,%s)
            """
    
    cursor.executemany(query, data_to_insert)
    
    ###############

    cursor.execute("SELECT * FROM stocks")
    columns = [desc[0] for desc in cursor.description]
    print(columns)
    stock_ids = cursor.fetchall()
    n_df = pd.DataFrame(stock_ids, columns=columns)
    n_df.to_csv("stocks.csv" ,index=False)
    #n_df = n_df[['id','ticker']] 
    
    n_df = pd.merge(df,n_df,on='Ticker')
    
    query = """
                INSERT INTO daily_prices 
                (stock_id, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume)
            """
    data_to_insert = [
    (
            row['id'],
            row['date'].strftime('%Y-%m-%d'),  # ✅ Apply strftime to row, not Series
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume'])
    )
    for _, row in n_df.iterrows()
    ]
    
    n_df[['id','date','open','high','low','close','volume']].to_csv("daily_prices.csv", index=False)
    
    # Insert daily prices in batches
    batch_size = 1000
    for i in range(0, len(data_to_insert), batch_size):
        #print(i)
        batch = data_to_insert[i:i + batch_size]
        cursor.executemany(query, batch)

    query = """
                INSERT INTO performance_metrics 
                (stock_id, date, daily_return, volatility, cumulative_return)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                daily_return = VALUES(daily_return),
                volatility = VALUES(volatility),
                cumulative_return = VALUES(cumulative_return)
            """
    data_to_insert = [
    (
            row['id'],
            row['date'].strftime('%Y-%m-%d'),  # ✅ Apply strftime to row, not Series
            float(row['daily_return']),
            float(row['volatility']),
            float(row['cumulative_returns'])
    )
    
    for _, row in n_df.iterrows()
    ]
    
    
    n_df[['id','date','daily_return','volatility','cumulative_returns']].to_csv("performance_metrics.csv", index=False)
    
    # Insert daily prices in batches
    batch_size = 1000
    for i in range(0, len(data_to_insert), batch_size):
        #print(i)
        batch = data_to_insert[i:i + batch_size]
        cursor.executemany(query, batch)
 
    conn.commit()
    cursor.close()
    close(conn)


    
# Example usage
if __name__ == "__main__":
    
    # Specify folders containing YAML files
    folders = 'data'
    
    # Read all YAML files and convert to DataFrames
    dfs = read_yaml_files(folders)
        
        
    
        
        
        
    df = dfs
    
    # Example with merging on specific column
    # merged_df = merge_dataframes(dfs, merge_on='id', how='outer')
    
    
    
    
    # 1. First convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Sort by both Ticker and date
    df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
    
    
    add_date_features(df[['Ticker', 'date']])
    df = calculate_metrics(df)
    
    
    print("Number of DataFrames:", len(df))
    print("\nMerged DataFrame shape:", df.shape)
    print("\nMerged DataFrame columns:", df.columns.tolist())
    
    # 4. Verify the sorting worked
    print("\nFirst few rows after sorting:")
    print(df[['Ticker', 'date', 'close']].head(10))
    
    
    print(df[['Ticker', 'date', 'close','daily_return','volatility','cumulative_returns']].head(10))
    
    
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
    df = calculate_metrics(df)
    sector_df= pd.read_csv('Sector_data.csv')
    
    sector_df['Ticker'] = sector_df['Symbol'].str.split(':').str[1].str.strip()
    
    df = pd.merge(df,sector_df,on='Ticker')
    
    
    drop_tables()
    
    create_tables()
    
    load_to_database(df)
    
    
    

    
    

