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
        'test_size': 0.05,
    }, 
    'model': {
        'name': 'LSTM', 
        'window': 20,
        'batch_size' : 32,
        'shuffle_buffer_size' : 1000,
        'epochs': 500,
    },
}


def main():
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)

    test_size_int = int(len(df) * config['data']['test_size'])
    print('test size:',test_size_int)
    df_train = df.iloc[:-test_size_int]
    df_test = df.iloc[-test_size_int:]

    # Normalize the training data
    max_value = df_train['Adj Close'].max()
    df_train['Adj Close'] = df_train['Adj Close'] / max_value

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
                                        tf.keras.layers.Lambda(lambda x: x * max_value) # Unnormalize
                                        ])

    model.summary(print_fn=logger.info)

    # Set the training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Default is 0.001 = 1e-3
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=optimizer, 
                  metrics=["mae"])    

    # Train the model
    history = model.fit(train_dataset, epochs=config['model']['epochs'])

    # Get mae and loss from history log
    mae=history.history['mae']
    loss=history.history['loss']

    # Plot MAE and Loss
    V.plot_series(x=config['model']['epochs'],
                    y=(mae, loss),
                    title='MAE and Loss',
                    xlabel='MAE',
                    ylabel='Loss',
                    legend=['MAE', 'Loss']
                )

    # Save the model
    model.save(os.path.join(PROJECT_PATH, r'models_trained\lstm_model.h5'))
    

if __name__ == '__main__':
    main()