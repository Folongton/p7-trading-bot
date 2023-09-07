import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.stats import geom
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from src.data.get_data import CSVsLoader as CSVs

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('Solarize_Light2')

@dataclass
class G:
    ''' Global variables and methods for project
    '''
    name: str 
    growth_tickers = ['AMZN','BABA','GOOGL','MSFT','TSLA']
    value_tikers = ['ABBV','F','INTC','JNJ','VZ']
    all_nasdaq_tickers = pd.read_csv(os.path.join(str(Path(__file__).resolve().parents[2]), r'data\00_raw\NASDAQ All Tickers.csv'))['ticker'].tolist()

    @staticmethod
    def get_project_root(as_path=False):
        """Returns project root folder."""
        if as_path:
            return Path(__file__).resolve().parents[2]
        else:
            return str(Path(__file__).resolve().parents[2])

    @staticmethod
    def setup_logging(console_level=logging.INFO, log_file='DEFAULT', log_file_level=logging.INFO, logger_name=__name__):
        ''' Setup Logging in log file and console
            Creates a logger object with appropriate formatting and handlers.
        INPUT: console_level: logging level for console, 
                    log_file: name of log file, 
            log_file_level: logging level for log file
        OUTPUT: logger object
        '''
        log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_fmt)
        console_handler.setLevel(console_level)

        file_handler = logging.FileHandler(filename=f'./logs/{log_file}.log')
        file_handler.setFormatter(log_fmt)
        file_handler.setLevel(log_file_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger 
    
class Analysis:
    '''Global Methods related to stock data analysis and manipulation
    '''

    @staticmethod
    def assign_return_bucket(stock):
        '''Assigns a bucket to a percent change in stock price
        INPUT: stock: dataframe of stock data
        OUTPUT: dataframe of stock data with 'return bucket' column and 'UP or DOWN' column
        '''
        df_name = stock.name
        stock['Daily Return'] = stock['5. adjusted close'].pct_change() # Calculate the daily returns for each stock into a new column called 'Daily Return'
        conditions = [stock['Daily Return'] < -0.05, 
                    stock['Daily Return'].between(-0.05, -0.04, inclusive='left'), 
                    stock['Daily Return'].between(-0.04, -0.03, inclusive='left'), 
                    stock['Daily Return'].between(-0.03, -0.02, inclusive='left'), 
                    stock['Daily Return'].between(-0.02, -0.01, inclusive='left'), 
                    stock['Daily Return'].between(-0.01, 0, inclusive='left'), 
                    stock['Daily Return'].between(0, 0.01, inclusive='left'), 
                    stock['Daily Return'].between(0.01, 0.02, inclusive='left'), 
                    stock['Daily Return'].between(0.02, 0.03, inclusive='left'), 
                    stock['Daily Return'].between(0.03, 0.04, inclusive='left'), 
                    stock['Daily Return'].between(0.04, 0.05, inclusive='left'), 
                    stock['Daily Return'] >= 0.05]
        values = [-5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5]
        stock['return bucket'] = np.select(conditions, values)
        stock = stock[~stock['Daily Return'].isnull()]

        # create column called 'UP or DOWN' to indicate whether the stock price increased or decreased and populate it based on the 'return bucket' column
        conditions = [stock['return bucket'].isin([ -1, -2, -3, -4, -5]),
                    stock['return bucket'].isin([0, 1, 2, 3, 4, 5])]
        values = [-1, 1]
        stock = stock.copy(deep=True)
        stock.name = df_name
        stock['UP or DOWN'] = np.select(conditions, values)

        return stock
    
    @staticmethod
    def calc_proba_price_change_based(stock_df, percentChange, days_back, print_proba=False):
        '''Calculates probabilities of a stock to go up or down next day, based on percent change in price of present day.
        Calculated based on previous occurences of percent change in price of stock.
        INPUT: stock_df: dataframe of stock data,
                percentChange: percent change in price of stock,
                days: number of days to look back for calculating
                print_proba: whether to print probabilities or only return dataframe with all probabilities
        OUTPUT: dataframe with probabilities of stock to go up or down next day
        '''
        df = Analysis.assign_return_bucket(stock_df)
        df = df.copy(deep=True)[-days_back:]
        # shift return bucket column up by one row to match the previous day's return bucket
        df['previous_day_return_bucket'] = df['return bucket'].rolling(2).apply(lambda x: x.iloc[0])
        # calculate probability of up or down given previous day's return bucket
        proba_df = df[['previous_day_return_bucket','UP or DOWN']].value_counts(normalize=True).to_frame()
        proba_df = proba_df.rename(columns={'proportion':'Probability'}).pivot_table(columns='previous_day_return_bucket', index='UP or DOWN', values='Probability')

        percentChange = float(percentChange)
        final_series = proba_df[percentChange].rename(index={-1.0:'Negative', 1.0:'Positive'})
        for r in final_series.index:
            if print_proba:
                final_series = final_series / final_series.sum() * 100
                print(f'Probability of {r} move next day is {round(final_series[r],2)}%')
        return proba_df
    
    @staticmethod
    def correlation_of_stock_price_reaction_to_price_change(all_stocks, days_to_look_back=365):
        ''' Compares stocks based on probability of price change next day, based on percent change in price of present day.
        Basically answering question : ***How correlated are the stocks in terms of price change next day, based on percent change in price of present day.***
        INPUT: all_stocks: list of stocks to compare,
                days_to_look_back: number of days to look back for calculating
        OUTPUT: dataframe with colinearity of stocks based on probability of price change next day, based on percent change in price of present day.
        '''
        colinearity_df = pd.DataFrame()
        for stock in all_stocks:
            df = CSVs.load_daily(stock)
            df = Analysis.calc_proba_price_change_based(df, percentChange=0, days_back=days_to_look_back, print_proba=False) # here we getting full probability dataframe to use for correlation, so here we can disregard percentChange argument as well as print_proba.
            df = df.iloc[1].to_frame().T
            df.index = [stock] * len(df)
            colinearity_df = pd.concat([colinearity_df, df], axis=0)
        sns.heatmap(colinearity_df.T.corr(), annot=True, cmap='coolwarm')
        return colinearity_df
    
    @staticmethod
    def proba_up_down_geometric(stock_df, cont_days, days_back, direction):
        '''Calculates probability of a stock to go on specified direction for a given number of days in a row.
        Basically Geometric Distribution of a stock going up or down for a given number of days in a row based on previous days going up or down.
        INPUT: stock_df: dataframe of stock data with calculated 'UP or DOWN' column,
                        To calculate 'UP or DOWN' column use Analysis.assign_return_bucket() method,
                cont_days: number of days in a row to check for,
                period_days: number of days to look back for calculating,
                direction: direction to check for ('up' or 'down')
        OUTPUT: probability of a stock to go on specified direction for a given number of days in a row
        '''
        df_name = stock_df.name
        if direction == 'up':
            direction_int = 1
        if direction == 'down':
            direction_int = -1
        df = stock_df.copy(deep=True)[-days_back:]
        df.name = df_name
        proba = geom.pmf(cont_days, df['UP or DOWN'].value_counts(normalize=True).loc[direction_int])
        print(f'Probability of {stock_df.name} going "{direction.upper()}" for {cont_days} days in a row is {round(proba*100,3)}%. Based on {days_back} days of data.')

    @staticmethod
    def aggregate_close_prices(directory, calendar_days_back=90 ,intraday=False, day_for_intraday='2023-03-24'):
        '''
        This function takes all the csv files in the directory and aggregates them into one dataframe.
        directory: The directory where the data is stored.
        intraday: If True it will take only points from '10:00' to '13:30'.
        '''
        df_agg = pd.DataFrame()
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, file), index_col=0, parse_dates=True)
                if df.empty:
                    continue

                if intraday:
                    df = df.loc[day_for_intraday]
                    df = df.between_time('10:00', '13:30')
                    df = df[['5. adjusted close']].iloc[::-1] 
                    if df.empty:
                        continue
                    if len(df) < 24: #24 periods of 5 min = 2 hours
                        continue
                    if df['5. adjusted close'].nunique() == 1:
                        continue

                else:
                    start_date = datetime.now() - timedelta(days=calendar_days_back) 
                    start_date = start_date.strftime('%Y-%m-%d')
                    df = df.loc[df.index > start_date]
                    df = df[['5. adjusted close']].iloc[::-1]
                    if df.empty:
                        continue
                    if df['5. adjusted close'].nunique() == 1:
                        continue
                
                df.columns = [file.split('-')[0]]
                df_agg = pd.concat([df_agg, df], axis=1)
                    
        return df_agg
    
    @staticmethod
    def find_best_linear_stock(df):
        '''
        IN: dataframe of ticker - columns and close price - rows 
        OUT: dataframe with R2, Slope for each ticker.
        '''
        df_stats = pd.DataFrame(columns=['ticker', 'R2', 'slope'])
        for col in df.columns:
            series = df[col].copy().ffill()
            series = series.copy().bfill()

            X = series.reset_index().index.values.reshape(-1,1)
            y = series.values.reshape(-1,1)
            model = LinearRegression().fit(X,y)
        
            df_stats = pd.concat([df_stats, pd.DataFrame({'ticker': col, 'R2': model.score(X,y), 'slope': model.coef_[0]}, index=[0])], ignore_index=True)
        return df_stats
    
    @staticmethod
    def plot_correlation(series, intervals='Days'):
        '''
        This function plots the correlation between the series - Price and the index of series - Time.
        '''
        series = series.ffill()
        series = series.bfill()

        X = series.reset_index().index.values.reshape(-1,1) # 60 days of data in exact same interval - 1 day.
        y = series.values.reshape(-1,1) # prices 

        model = LinearRegression()
        model.fit(X, y)

        plt.style.use('Solarize_Light2')
        plt.figure(figsize=(10,6))
        
        plt.scatter(X, y, s=10, color='#294dba')
        plt.plot(model.predict(X), color='#d9344f')
        
        plt.xlabel(intervals)
        plt.ylabel('Price')
        plt.title(f'Linear Correlation of {series.name} with linear regression line')
        plt.show()
