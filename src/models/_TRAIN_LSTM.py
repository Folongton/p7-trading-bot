import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from src.common.analysis_and_plots import Visualize as V
from src.features.build_features import FeatureEngineering as FE
from src.common.globals import G
from src.common.globals import split_train_valid_test, get_naive_forecast, calc_errors, save_errors_to_table
from src.data.get_data import CSVsLoader
from src.common.logs import setup_logging, log_model_info

import logging
import os
from datetime import datetime

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
        'batch_size' : 30,
        'shuffle_buffer_size' : 1000,
        'epochs' : 300,
        'optimizer': tf.keras.optimizers.Adam(),
        'loss': tf.keras.losses.Huber(),
    },
}


def main():
    # -----------------------------Data----------------------------------------
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)

    test_size_int = int(len(df) * config['data']['test_size'])
    df_train = df.iloc[:-test_size_int]
    df_test = df.iloc[-test_size_int:]

    # DeNormalize the training data in the last layer of the model
    max_value = df_train['Adj Close'].max()

    train_dataset = FE.windowed_dataset(df_train['Adj Close'], 
                                    window_size=config['model']['window'], 
                                    batch_size=config['model']['batch_size'], 
                                    shuffle_buffer=config['model']['shuffle_buffer_size'])
    

    # -----------------------------Model---------------------------------------
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
            tf.keras.layers.Lambda(lambda x: x * max_value) # Unnormalize because of tanh activation in LSTM, whicn outputs values [-1, 1]
            ],
        name=config['model']['name'])

    model._name = f"{model._name}_{str(model.count_params())}_{datetime.now().strftime('%Y-%m-%d--%H-%M')}"
    log_model_info(config, model, logger)

    # Set the training parameters
    model.compile(loss=config['model']['loss'], 
                optimizer=config['model']['optimizer'], 
                metrics=["mae"],
                )    

    # Train the model
    history = model.fit(train_dataset, epochs=config['model']['epochs'])

    # Plot MAE and Loss
    mae=history.history['mae']
    loss=history.history['loss']
    zoom = int(len(mae) * 1)
    V.plot_series(x=range(config['model']['epochs'])[-zoom:],
                    y=(mae[-zoom:],loss[-zoom:]),
                    model_name=config['model']['name'],
                    title='MAE_and_Loss',
                    xlabel='MAE',
                    ylabel='Loss',
                    legend=['MAE', 'Loss']
                )

    # Save the model
    FE.model_save(model, logger)

    # -----------------------------Predictions---------------------------------
    forecast_series = df['Adj Close'].iloc[-test_size_int - config['model']['window']:-1]

    forecast = FE.model_forecast(model=model, 
                                series=forecast_series, 
                                window_size=config['model']['window'], 
                                batch_size=config['model']['batch_size'])

    # Drop single dimensional axis
    results = forecast.squeeze()

    V.plot_series(  x=df_test.index, 
                    y=(df_test['Adj Close'], results),
                    model_name=config['model']['name'])
    
    # -----------------------Calculate Errors----------------------------------
    naive_forecast = get_naive_forecast(df).iloc[-len(df_test['Adj Close']):]
    rmse, mae, mape, mase = calc_errors(df_test['Adj Close'], results, naive_forecast)
    save_errors_to_table(config['model']['name'], {'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase})
        

if __name__ == '__main__':
    main()