# Data-Driven-Stock-Analysis

Here’s a detailed README.md file and documentation for your project:

---

## Stock Data Pipeline

### Project Overview
This project is a stock data processing pipeline that loads stock data from YAML files, calculates various financial performance metrics, and stores the processed data into a MySQL database. It covers essential steps such as:
- Reading stock data from YAML files.
- Cleaning and processing data (e.g., calculating daily returns, volatility).
- Adding date-based features for analysis.
- Inserting the processed data into a relational database.
  
### Features
- Read and Parse YAML Files: Load stock data from YAML files into pandas DataFrames.
- Add Financial Metrics: Calculates performance metrics such as daily return, volatility, and cumulative returns.
- Database Integration: Inserts stock data and performance metrics into a MySQL database.
- Dynamic Date Features: Adds date-based features such as day of the week, quarter, and trading days.
- Batch Data Insertion: Efficient batch insertion of stock and price data into MySQL.

### Tech Stack
- Python: The primary programming language used.
- pandas: For data manipulation and calculations.
- MySQL: For data storage.
- YAML: For input data in YAML format.
- NumPy: For numerical computations.
- Logging: For error logging and debugging.

---

### Installation Guide

To run this project, ensure that you have the following dependencies installed:

#### Step 1: Install Python and Dependencies
If you don't have Python installed, download and install the latest version from [Python's official website](https://www.python.org/).

Install the necessary Python libraries:

bash
pip install pandas numpy mysql-connector-python PyYAML


#### Step 2: Install MySQL
Install MySQL and create a database named stock_analysis by following these steps:
1. Install MySQL on your system (follow the guide for your platform).
2. Create a database for storing stock data:
    sql
    CREATE DATABASE stock_analysis;
    

---

### Configuration

Ensure that the following configurations are set up correctly for your environment:

1. Database Connection:  
   - The connection to MySQL is established through the function connect() in the stock_data_loader.py file. The parameters are:
     - host: The host of your MySQL server (e.g., localhost).
     - user: Your MySQL username (e.g., root).
     - password: Your MySQL password.
     - database: Name of the MySQL database (stock_analysis).

2. Folder Structure:
   Ensure your YAML data files are stored in a folder named data/, which will be read and processed.

---

### How to Use

1. Prepare Your YAML Files:  
   Place your stock data in YAML files within the data/ folder. Each file should contain stock-related data (e.g., ticker, date, close, volume).

2. Run the Script:  
   To start processing the data and load it into the database, run the script:

   bash
   python stock_data_loader.py
   

   The script will:
   - Read YAML files from the data/ folder.
   - Process the data to calculate financial metrics.
   - Insert the data into the MySQL database (stock_analysis).

---

### Code Overview

The main script is stock_data_loader.py which contains the following key functions:

#### 1. read_yaml_files(folder)
- Purpose: Reads YAML files from the specified folder and converts them into pandas DataFrames.
- Parameters:
  - folder: Path to the folder containing YAML files.
- Returns: A concatenated DataFrame containing all the data.

#### 2. connect()
- Purpose: Establishes a connection to the MySQL database.
- Returns: MySQL connection object.

#### 3. close(conn)
- Purpose: Closes the database connection.

#### 4. create_tables()
- Purpose: Creates necessary tables (stocks, daily_prices, performance_metrics) in the MySQL database if they do not exist.
- SQL Queries: Contains SQL queries for table creation.

#### 5. drop_tables()
- Purpose: Drops the existing tables in the MySQL database if needed (for reinitialization).

#### 6. add_date_features(df)
- Purpose: Adds date-based features such as year, month, quarter, and other useful trading day features.
- Parameters:
  - df: DataFrame containing stock data with a date column.
- Returns: Modified DataFrame with additional date features.

#### 7. calculate_metrics(df)
- Purpose: Calculates performance metrics such as daily return, volatility, and cumulative returns.
- Parameters:
  - df: DataFrame containing stock data.
- Returns: Modified DataFrame with financial metrics.

#### 8. load_to_database(df)
- Purpose: Loads the processed stock data into the MySQL database, inserting data into the stocks, daily_prices, and performance_metrics tables.
- Parameters:
  - df: DataFrame containing the final processed stock data.

---

### Sample Data

Sample YAML files can have the following structure:

yaml
- ticker: AAPL
  date: "2021-01-01"
  open: 135.0
  high: 137.0
  low: 134.0
  close: 136.0
  volume: 100000
  sector: Technology
  company: Apple Inc.
  
- ticker: AAPL
  date: "2021-01-02"
  open: 136.5
  high: 138.0
  low: 135.0
  close: 137.0
  volume: 120000
  sector: Technology
  company: Apple Inc.


---

### Database Schema

1. stocks Table:  
   Stores unique stock information (ticker, name, sector).
   - id: Primary Key, auto-incremented.
   - ticker: Stock ticker symbol (unique).
   - name: Stock company name.
   - sector: The sector in which the company operates.

2. daily_prices Table:  
   Stores daily stock price data.
   - id: Primary Key, auto-incremented.
   - stock_id: Foreign Key referencing stocks.id.
   - date: Date of the stock price.
   - open: Opening price.
   - high: Highest price.
   - low: Lowest price.
   - close: Closing price.
   - volume: Trading volume.

3. performance_metrics Table:  
   Stores calculated performance metrics.
   - id: Primary Key, auto-incremented.
   - stock_id: Foreign Key referencing stocks.id.
   - date: Date of the performance metrics.
   - daily_return: Daily return.
   - volatility: Volatility of daily returns.
   - cumulative_return: Cumulative return.
