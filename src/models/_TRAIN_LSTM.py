import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.common.analysis_and_plots import Visualize as V
from src.features.build_features import FeatureEngineering as FE
from src.common.globals import G
from src.common.globals import split_train_valid_test, get_naive_forecast, calc_errors, save_errors_to_table
from src.data.get_data import CSVsLoader
from src.common.logs import setup_logging
import logging
import os

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

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
        'test_size_int': 30,
    }, 
    'model': {
        'name': 'LSTM', 
        'window': 20,
        'batch_size' : 32,
        'shuffle_buffer_size' : 1000,
    },
}

def model_summary(model):
    logger.info(str(model.summary()))

def main():
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)

    df_train = df.iloc[:-config['data']['test_size_int']]
    df_test = df.iloc[-config['data']['test_size_int']:]

    train_dataset = FE.windowed_dataset(df_train['Adj Close'], 
                                        window_size=config['model']['window'], 
                                        batch_size=config['model']['batch_size'], 
                                        shuffle_buffer=config['model']['shuffle_buffer_size'])
    
    # Build the Model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                        strides=1,
                        activation="relu",
                        padding='causal',
                        input_shape=[config['model']['window'], 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    # Print the model summary
    model_summary(model)


if __name__ == '__main__':
    main()