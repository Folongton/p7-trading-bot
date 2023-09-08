from src.common.globals import G, split_train_valid_test, calc_errors, save_errors_to_table, get_naive_forecast
from src.data.get_data import CSVsLoader
from src.common.analysis_and_plots import Visualize as V
from src.features.build_features import FeatureEngineering as FE


import os
import joblib
from src.common.logs import setup_logging
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

PROJECT_PATH = G.get_project_root()
DATA_DIR_PROCESSED = os.path.join(PROJECT_PATH, r'data\03_processed\daily_full')

config = {
    'AV': {
        'key': '',
        'ticker': 'MSFT',
        'outputsize': 'full',
        'key_adjusted_close': 'Adj Close',
        'key_volume': 'Volume',
    },
    'data': {
        'train_size': 0.85,
        'valid_size': 0.145,
        'test_size': 0.005,
    }, 
    'model': {
        'name': 'ARIMA', 
        'window': 20,
    },
}

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

def main():
    # Load data
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)
    # Log Scale data for ARIMA better performance
    df = FE.log_scale(df, column=config['AV']['key_adjusted_close'])
    # Plot
    V.plot_1_by_2(df, model_name=config['model']['name'],
                  col_before=config['AV']['key_adjusted_close'], 
                  col_after=config['AV']['key_adjusted_close'] + ' - log',
                  title2='Log Scale',)
    # Create features
    df = FE.create_lags(df, column=config['AV']['key_adjusted_close'], lag=config['model']['window'])





if __name__ == '__main__':
    main()