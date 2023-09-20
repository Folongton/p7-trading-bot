from src.common.logs import setup_logging
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('Solarize_Light2')

PROJECT_PATH = Path(__file__).resolve().parents[2]

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

@dataclass
class G:
    ''' Global variables and methods for project
    '''
    name: str 
    growth_tickers = ['AMZN','BABA','GOOGL','MSFT','TSLA']
    value_tikers = ['ABBV','F','INTC','JNJ','VZ']
    all_nasdaq_tickers = pd.read_csv(os.path.join(str(PROJECT_PATH), r'data\00_raw\NASDAQ All Tickers.csv'))['ticker'].tolist()
    raw_daily_full_dir = os.path.join(str(PROJECT_PATH), r'data\00_raw\daily_full')

    @staticmethod
    def get_project_root(as_path=False):
        """Returns project root folder."""
        if as_path:
            return Path(__file__).resolve().parents[2]
        else:
            return str(Path(__file__).resolve().parents[2])

 
def split_train_valid_test(df, train_size, valid_size, test_size):
    ''' Splits dataframe into train, valid, test sets
    INPUT: df: dataframe, 
            train_size: float, 
            valid_size: float, 
            test_size: float
    OUTPUT: df_train: dataframe, 
            df_valid: dataframe, 
            df_test: dataframe
    '''
    assert train_size + valid_size + test_size == 1, 'train_size + valid_size + test_size must equal 1'
    train_end_index = int(len(df) * train_size)
    valid_end_index = int(len(df) * (train_size + valid_size))
    df_train = df.iloc[:train_end_index]
    df_valid = df.iloc[train_end_index:valid_end_index]
    df_test = df.iloc[valid_end_index:]

    logger.info(f"Train DF    : {df_train.shape}")
    logger.info(f"Val DF      : {df_valid.shape}")
    logger.info(f"Test DF     : {df_test.shape}")
    logger.info(f"Total values: {df_train.shape[0]} + {df_valid.shape[0]} + {df_test.shape[0]} = {df_train.shape[0] + df_valid.shape[0] + df_test.shape[0]}")
    return df_train, df_valid, df_test

def get_naive_forecast(df):
    ''' Naive forecast is the previous day's close price '''
    naive_series = df['Adj Close'].shift(1)
    return naive_series

def mean_absolute_scaled_error(y_true, y_pred, naive_forecast):
    ''' MASE = MAE\MAE_naive '''
    mae = mean_absolute_percentage_error(y_true, y_pred)
    mae_naive = mean_absolute_percentage_error(y_true, naive_forecast)
    return mae/mae_naive

def calc_errors(y_true, y_pred, naive_forecast):
    ''' Calculates errors for model

    INPUT:  y_true: series, 
            y_pred: series, 
            naive_forecast: series
    OUTPUT: rmse: float,
            mae: float,
            mape: float,
            mase: float
    '''
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred, naive_forecast)

    logger.info(f'Test RMSE: $ {round(rmse, 3)}')
    logger.info(f'Test MAE : $ {round(mae, 3)}')
    logger.info(f'Test MAPE:   {round(mape, 3)}')
    logger.info(f'Test MASE:   {round(mase, 3)}')
    return round(rmse, 3), round(mae, 3), round(mape, 3), round(mase, 3)

def save_errors_to_table(model, errors):
    ''' Saves errors to csv file
    INPUT:  model: str,
            errors: dict
    OUTPUT: None - updates csv file table with errors for all models.
    '''
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(os.path.join(str(PROJECT_PATH), r'logs/models_table.csv'))

    for new_col in errors.keys():
        if new_col not in df.columns:
            df[new_col] = None
    
    dict_for_df = {'model': model, 'timestamp': timestamp}
    dict_for_df.update(errors)
    df = pd.concat([df, pd.DataFrame(dict_for_df, index=[0])], ignore_index=True)

    df.to_csv(os.path.join(str(PROJECT_PATH), r'logs/models_table.csv'), index=False)

    logger.info(f'Errors saved to for {model} model to "logs/models_table.csv" file.')

def save_model(model, model_name):
    ''' Saves model
    '''
    joblib.dump(model, os.path.join(str(PROJECT_PATH), f'models_trained/{model_name}.pkl'))
    logger.info(f'Model saved: "{PROJECT_PATH}/models_trained/{model_name}.pkl"')