import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class G:
    ''' Global variables and methods for project
    '''
    name: str 

    @staticmethod
    def get_project_root() -> Path:
        """Returns project root folder."""
        print(Path(__file__).resolve().parents[2])
        return Path(__file__).resolve().parents[2]
    
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
    
class StockRelated:
    '''Global Methods related to stock data
    '''

    @staticmethod
    def assign_return_bucket(stock):
        '''Assigns a bucket to a percent change in stock price
        '''
        stock['Daily Return'] = stock['Close'].pct_change() # Calculate the daily returns for each stock into a new column called 'Daily Return'
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
        stock['UP or DOWN'] = np.select(conditions, values)

        return stock

    
    @staticmethod
    def calc_proba_price_change_based(stock_df, percentChange, days, print_proba=False):
        '''Calculates probabilities of a stock to go up or down next day, based on percent change in price of present day.
        '''
        
        df = stock_df.copy(deep=True)[-days:]
        # shift return bucket column up by one row to match the previous day's return bucket
        df['previous_day_return_bucket'] = df['return bucket'].rolling(2).apply(lambda x: x[0])
        # calculate probability of up or down given previous day's return bucket
        proba_df = df[['previous_day_return_bucket','UP or DOWN']].value_counts(normalize=True).to_frame().rename(columns={0:'Probability'}).pivot_table(columns='previous_day_return_bucket', index='UP or DOWN', values='Probability')
        percentChange = float(percentChange)
        final_series = proba_df[percentChange].rename(index={-1.0:'Negative', 1.0:'Positive'})
        for r in final_series.index:
            if print_proba:
                final_series = final_series / final_series.sum() * 100
                print(f'Probability of {r} is {round(final_series[r],2)}%')
        return proba_df