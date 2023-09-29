import os, sys, logging
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ---------- Add project root to path ----------
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), verbose=True) # Example:  AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)
# ----------------------------------------------

from src.common.plots import Visualize as V
from src.data.get_data import CSVsLoader
from src.common.logs import setup_logging, log_model_info
from src.features.build_features import FeatureEngineering as FE

from src.models_service.models_service import TensorflowDataPreparation as TFDataPrep
from src.models_service.models_service import TensorflowModelService as TFModelService
from src.models_service.errors import ErrorsCalculation as ErrorCalc


logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

DATA_DIR_PROCESSED = os.path.join(PROJECT_ROOT, r'data\03_processed\daily_full')

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
        'shuffle_buffer_size' : 5600, # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
        'epochs' : 3,
        'optimizer': tf.keras.optimizers.Adam(),
        'loss': tf.keras.losses.Huber(),
    },
    'plots': {
        'loss_zoom': 0.9,
        'show': True,
    },
}

def main():
    # -----------------------------Data----------------------------------------
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)
    df = FE.create_features(df, logger)
    df_train, df_test = TFDataPrep.split_train_test(df, config['data']['test_size'], logger)

    df_test_X = df_test.drop(columns=['Adj Close'])
    df_test_y = df_test['Adj Close']

    df_train_X = df_train.drop(columns=['Adj Close'])
    df_train_y = df_train['Adj Close']


    train_dataset_X, scalers_X = TFDataPrep.windowed_dataset_X(df_train_X, 
                                                                window_size=config['model']['window'], 
                                                                logger=logger,
                                                                verbose=True)
    train_dataset_y = TFDataPrep.windowed_dataset_y(df_train_y, 
                                        window_size=config['model']['window'], 
                                        logger=logger,
                                        verbose=True)
    train_dataset = TFDataPrep.combine_datasets(train_dataset_X, train_dataset_y, config, logger, verbose=False)

    # -----------------------------Model Architecture--------------------------
    model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None,2)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
            ],
        name=config['model']['name'])

    model._name = f"{model._name}_{str(model.count_params())}_{datetime.now().strftime('%Y-%m-%d--%H-%M')}"
    log_model_info(config, model, logger)


    # -----------------------------Model Training-------------------------------
    model.compile(loss=config['model']['loss'], 
                optimizer=config['model']['optimizer'], 
                metrics=["mae"],
                )    

    history = model.fit(train_dataset, epochs=config['model']['epochs'])

    # Plot MAE and Loss
    mae=history.history['mae']
    loss=history.history['loss']
    zoom = int(len(mae) * config['plots']['loss_zoom'])

    V.plot_series(x=range(config['model']['epochs'])[-zoom:],
                    y=(mae[-zoom:],loss[-zoom:]),
                    model_name=config['model']['name'],
                    title='MAE_and_Loss',
                    xlabel='MAE',
                    ylabel='Loss',
                    legend=['MAE', 'Loss'],
                    show=config['plots']['show'],
                )

    # Save the model
    TFModelService.save_model(model=model, logger=logger)
    # save the scalers
    print('scalers_X:', scalers_X)
    
    TFModelService.save_scalers(scalers=scalers_X, model_name=model._name ,logger=logger)



    #------------------------Load the model if necessary--------------------------
    model = TFModelService.load_model(model_name='LSTM_42113_2023-09-20--15-58', logger=logger)


    # -----------------------------Predictions-----------------------------------
    results = TFModelService.model_forecast(model=model, 
                                            df=df_test_X,
                                            window_size=config['model']['window'],
                                            scalers=scalers_X,
                                            verbose=True)
    
    df_test_plot_y = TFModelService.prep_test_df_shape(df_test_y, config)

    V.plot_series(  x=df_test_plot_y.index,  # as dates
                    y=(df_test_plot_y, results),
                    model_name=config['model']['name'],
                    title='Predictions',
                    xlabel='Date',
                    ylabel='Price',
                    show=config['plots']['show'],)


    # -----------------------Calculate Errors----------------------------------
    naive_forecast = ErrorCalc.get_naive_forecast(df).loc[df_test_plot_y.index] # Getting same days as results
    rmse, mae, mape, mase = ErrorCalc.calc_errors(df_test_plot_y, results, naive_forecast)
    ErrorCalc.save_errors_to_table(config['model']['name'], {'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase})


if __name__ == '__main__':
    main()