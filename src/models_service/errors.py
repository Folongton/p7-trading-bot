import numpy as np
import pandas as pd
import logging, os
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from src.common.logs import setup_logging
from src.common.globals import G
PROJECT_PATH = G.get_project_root()

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)



class ErrorsCalculation():
    ''' 
    Class for calculating errors for models
    And saving errors to csv file
    '''

    @staticmethod
    def get_naive_forecast(df):
        ''' Naive forecast is the previous day's close price '''
        naive_series = df['Adj Close'].shift(1)
        return naive_series
    
    @staticmethod
    def mean_absolute_scaled_error(y_true, y_pred, naive_forecast):
        ''' MASE = MAE\MAE_naive '''
        mae = mean_absolute_percentage_error(y_true, y_pred)
        mae_naive = mean_absolute_percentage_error(y_true, naive_forecast)
        return mae/mae_naive

    @staticmethod
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
        mase = ErrorsCalculation.mean_absolute_scaled_error(y_true, y_pred, naive_forecast)

        logger.info(f'Test RMSE: $ {round(rmse, 3)}')
        logger.info(f'Test MAE : $ {round(mae, 3)}')
        logger.info(f'Test MAPE:   {round(mape, 3)}')
        logger.info(f'Test MASE:   {round(mase, 3)}')
        return round(rmse, 3), round(mae, 3), round(mape, 3), round(mase, 3)

    @staticmethod
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