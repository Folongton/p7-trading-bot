from abc import ABC, abstractmethod

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os, joblib
from datetime import datetime
import re
import pandas as pd

from src.common.globals import G
PROJECT_PATH = G.get_project_root()
DATA_DIR_PROCESSED = (f'{PROJECT_PATH}/data/03_processed/daily_full')
from src.data.get_data import CSVsLoader
from src.common.logs import log_model_info
from src.features.build_features import FeatureEngineering as FE
from src.common.plots import Visualize as V
from src.models_service.errors import ErrorsCalculation as ErrorCalc


# -----------------------------Model Pre Train --------------------------
class DataPreparationService(ABC):
    ''' 
    Classes for data preparation services to fit the model.
    Also a base class for general methods
    '''
    def __init__(self, config):
        self.config = config


    @staticmethod
    def split_train_test(df, test_size, logger):
        '''
        Splits a dataframe into train and test sets

        Args:
            df (pandas dataframe) - dataframe to split
            test_size (float) - size of the test set
            logger (logger) - logger to use

        Returns:
            df_train (pandas dataframe) - train set
            df_test (pandas dataframe) - test set
        '''

        test_size_int = int(len(df) * test_size)
        df_train = df.iloc[:-test_size_int].copy(deep=True)
        df_test = df.iloc[-test_size_int:].copy(deep=True)

        logger.info(f'df_train.shape: {df_train.shape}, df_test.shape: {df_test.shape}')
        logger.info(f'Last record in df_train:\n{df_train.iloc[-1]}')
        logger.info(f'Last record in df_test:\n{df_test.iloc[-1]}')
        logger.info(f'First record in df_test:\n{df_test.iloc[0]}')

        return df_train, df_test

    @staticmethod
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

    @staticmethod
    def label_column_to_end(df, last_column):
        ''' 
        Changes positions of columns in df to put the target column at the end

        Args:
            df (pandas dataframe) - dataframe to change
            last_column (string) - name of the column to put at the end
            
        Returns:
            df (pandas dataframe) - dataframe with the target column at the end
        '''
        cols = df.columns.tolist()
        cols.remove(last_column)
        cols.append(last_column)

        return df[cols]
    
    def data_prep(self, logger, verbose=False):
        pass
    
class TensorflowDataPreparation(DataPreparationService):
    ''' Methods for data preparation services to fit the model in tensorflow'''
    def __init__(self, config):
        super().__init__(config)


    @staticmethod # for 1 feature (close price)
    def windowed_dataset_1_feature(series, window_size, batch_size, shuffle_buffer):
        """Generates dataset windows

        Args:
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the feature
        batch_size (int) - the batch size
        shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
        dataset (TF Dataset) - TF Dataset containing time windows
        """

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

        # Create tuples with features and labels
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        for element in dataset:
            print(type(element))
            print(element)
            break

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset
    
    @staticmethod # for N features
    def windowed_dataset_X(df, window_size, logger, verbose=True):
        '''
        Creates a windowed dataset X from a dataframe

        Args:
            df (pandas dataframe) - dataframe to create the dataset from
            window_size (int) - size of the window
            logger (logger) - logger to use
            verbose (bool) - whether to print debug info or not

        Returns:
            dataset (tf dataset) - dataset with the windowed data
            scalers (dict) - dictionary with the scalers used for each column
        '''

        X_df = df.copy(deep=True)

        scalers = {}
        for col in X_df.columns:
            scaler = MinMaxScaler()
            X_df[col] = scaler.fit_transform(X_df[col].values.reshape(-1,1))
            scalers[col] = scaler
        
        if verbose:
            logger.info('-------------------windowed_dataset_X() - scalers----------------------')
            logger.info (f'scalers: {scalers}')
            logger.info('-'*100)


        # Creating X 
        X = X_df.values

        if verbose:
            logger.info('--------------------windowed_dataset_X() - X.shape----------------------')
            logger.info (f'X.shape: {X.shape}')
            logger.info('-'*100)

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(X)
        if verbose:
            logger.info('---------------windowed_dataset_X() - from_tensor_slices()-----------------')
            for element in dataset:
                last_element = element
            logger.info(f'Last element in dataset: {last_element}')
            logger.info('-'*100)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        if verbose:
            logger.info('-------------------windowed_dataset_X() - window()-------------------------')
            for window in dataset:
                last_window = window
            logger.info(f'Last window elements: {list(last_window.as_numpy_iterator())}')
            logger.info('-'*100)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        if verbose:
            logger.info('--------------------windowed_dataset_X() - flat_map()----------------------')
            for window in dataset:
                last_window = window
            logger.info(f'Last window after flatting: {last_window}')
            logger.info('-'*100)

        if verbose:
            logger.info(f'Lenght of X out of windowed_dataset_X() - {len(list(dataset.as_numpy_iterator()))}')
            logger.info('-'*100)

        return dataset, scalers

    @staticmethod # for N features
    def dataset_y(df, window_size, logger, verbose=True):
        '''
        Creates a windowed dataset y from a dataframe

        Args:
            df (pandas dataframe) - dataframe to create the dataset from
            window_size (int) - size of the window
            logger (logger) - logger to use
            verbose (bool) - whether to print debug info or not

        Returns:
            dataset (tf dataset) - dataset with the windowed data
        '''

        y = df.copy(deep=True)

        if verbose:
            logger.info('---------------------------dataset_y() - y.shape----------------------------')
            logger.info (f' y.shape: {y.shape}')
            logger.info('-'*100)
        
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(y)
        if verbose:
            logger.info('--------------------------dataset_y() - from_tensor_slices()----------------')
            for i,element in enumerate(dataset):
                if i > len(y) - 4:
                    logger.info(f'Element {i}: {element}')
                last_element = element
            logger.info(f'Last element in dataset: {last_element}')
            logger.info('-'*100)


        # Window dataset as we did for X and batch it for 1 - as we need to predict only 1 value
        dataset = dataset.window(window_size, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(window_size))

        # Map each window to its last element so we have only 1 value per window as our target
        dataset = dataset.map(lambda window: window[-1])

        if verbose:
            logger.info('----------------dataset_y() - dataset.take(len(y) - remainder)---------------')
            for element in dataset:
                last_element = element
            logger.info(f'Last element in dataset: {last_element}')
            logger.info(f'Lenght of dataset y out of dataset_y() - {len(list(dataset.as_numpy_iterator()))}')
            logger.info('-'*100)


        return dataset
   
    @staticmethod
    def combine_datasets(train_dataset_X, train_dataset_y, config, logger, verbose=True):
        ''' 
        Combines the X and y datasets into one dataset
        Then shuffles the dataset
        Then batches the dataset and prefetches 1 batch
        
        Args:
            train_dataset_X (tf dataset) - dataset with the features
            train_dataset_y (tf dataset) - dataset with the target
            config (dict) - dictionary with the configuration
            logger (logger) - logger
            verbose (bool) - if True, prints the shape of the dataset
        Returns:
            zipped_dataset (tf dataset) - dataset with the features and the target
        '''
        zipped_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_y))
        
        number_of_windows = len(list(zipped_dataset.as_numpy_iterator()))
        shuffle_buffer_size = int(config['model']['shuffle_buffer_size'] * number_of_windows)
        zipped_dataset = zipped_dataset.shuffle(shuffle_buffer_size)

        
        if verbose:
            logger.info('--------------------------------zipped_dataset-----------------------------------')
            logger.info(f'Len of zipped_dataset (number of windows): {number_of_windows}')
            logger.info(f"shuffle_buffer_size: {shuffle_buffer_size}")
        
        zipped_dataset = zipped_dataset.batch(config['model']['batch_size']).prefetch(1)

        if verbose:
            for x, y in zipped_dataset:
                logger.info(f'x.shape: {x.numpy().shape}, y.shape: {y.numpy().shape}')
                break
        
            input_shape = zipped_dataset.element_spec[0].shape
            logger.info(f'Full Dataset shape: {input_shape}')
            logger.info(f'Input for the model: {input_shape[1:]}')

        return zipped_dataset
    
   
    def choose_y(self, df_train, df_test, logger):
        '''
        To support 2 models - price and trend, we need to choose the y column to use for training and testing. This will mean we have to drop the other y column from the dataframe.
        If the model is price, we will drop the trend column and vice versa.
            IN:
                df_train - pandas dataframe
                df_test - pandas dataframe
            OUT:
                df_train - pandas dataframe updated
                df_test - pandas dataframe updated
                y_column - string - name of the y column
            '''
        if self.config['model']['type'] == 'trend':
            y_column = [y for y in df_train.columns if 'slope' in y][0]

            df_train = df_train.drop(columns=['Tomorrow'])   
            df_test = df_test.drop(columns=['Tomorrow'])

        elif self.config['model']['type'] == 'price':
            y_column = 'Tomorrow'
            trend_col = [y for y in df_train.columns if 'slope' in y][0]

            df_train = df_train.drop(columns=[trend_col])
            df_test = df_test.drop(columns=[trend_col])

        else:
            raise ValueError(f"Model type '{self.config['model']['type']}' is not supported. Supported types are 'price' and 'trend'")
        
        return df_train, df_test, y_column

    def data_prep(self, logger, verbose=False):
        '''Data preparation for model training
        IN:
            self - class object with config and model
        OUT:
            train_dataset - tf dataset
            scalers_X - dict
            df_test_X - pandas dataframe
            df_test_y - pandas dataframe
        '''
        df = CSVsLoader(ticker=self.config['AV']['ticker'], logger=logger, directory=DATA_DIR_PROCESSED)
        df = FE.create_features(df, logger)
        df = FE.create_slope_column(df, logger, slope_window=self.config['model']['trend_prediction_window'])


        df_train, df_test = TensorflowDataPreparation.split_train_test(df, self.config['data']['test_size'], logger)

        df_train, df_test, y_column = self.choose_y(df_train, df_test, logger)

        df_train_X = df_train.drop(columns=[y_column])
        df_train_y = df_train[y_column]

        df_test_X = df_test.drop(columns=[y_column])
        df_test_y = df_test[y_column]


        train_dataset_X, scalers_X = TensorflowDataPreparation.windowed_dataset_X(df_train_X, 
                                                                    window_size=self.config['model']['window'], 
                                                                    logger=logger,
                                                                    verbose=verbose)
        train_dataset_y = TensorflowDataPreparation.dataset_y(df_train_y, 
                                            window_size=self.config['model']['window'], 
                                            logger=logger,
                                            verbose=verbose)
        train_dataset = TensorflowDataPreparation.combine_datasets(train_dataset_X, train_dataset_y, self.config, logger, verbose=True)


        return train_dataset, scalers_X, df_test_X, df_test_y, df
    
class SklearnDataPreparation(DataPreparationService):
    ''' Methods for data preparation services to fit the model in sklearn'''
    
    def __init__(self, config):
        super().__init__(config)


    def data_prep(self, logger, verbose=False):
        '''Data preparation for model in sklearn
        IN:
            self - class object with config and model
        OUT:
            df_train_X - pandas dataframe
            df_train_y - pandas dataframe
            df_test_X - pandas dataframe
            df_test_y - pandas dataframe
            df - pandas dataframe - full dataframe
        '''
        df = CSVsLoader(ticker=self.config['AV']['ticker'], logger=logger, directory=DATA_DIR_PROCESSED)
        df = FE.create_features(df, logger)
        df_train, df_test = TensorflowDataPreparation.split_train_test(df, self.config['data']['test_size'], logger)

        df_train_X = df_train.drop(columns=['Tomorrow'])
        df_train_y = df_train['Tomorrow']

        df_test_X = df_test.drop(columns=['Tomorrow'])
        df_test_y = df_test['Tomorrow']

        return df_train_X, df_train_y, df_test_X, df_test_y, df


# -----------------------------Model Post Train --------------------------
class ModelService(ABC):
    ''' Abstract class for model services like training, prediction, saving, loading etc.'''

    @abstractmethod
    def save_model(self, model, model_name, logger):
        pass

    @abstractmethod
    def load_model(self, model_name, logger):
        pass
    
class TensorflowModelService(ModelService):
    ''' 
    Methods for model services like training, prediction, saving, loading etc. in tensorflow
    After model was Trained
    '''
 

    @staticmethod
    def save_model(model, logger=None):

        model_name = model._name
        model.save(os.path.join(PROJECT_PATH, rf'models_trained/{model_name}.keras'))
        logger.info(f"Model saved as {model._name}.keras")

    @staticmethod
    def load_model(model_name, logger):
        # try to load model recursively in all subfolders of models_trained
        for dirpath, dirnames, filenames in os.walk(os.path.join(PROJECT_PATH, 'models_trained')):
            for name in filenames:
                if name == f'{model_name}.keras':
                    model_path = os.path.join(dirpath, name)
                    model = tf.keras.models.load_model(model_path)
                    logger.info(f"Model loaded from: {model_path}")
                    return model
                
        raise FileNotFoundError(f"Model {model_name}.keras not found in {os.path.join(PROJECT_PATH, 'models_trained')}")
    
    @staticmethod
    def save_scalers(scalers, model_name, logger):
        ''' Saves scalers
        '''
        joblib.dump(scalers, os.path.join(str(PROJECT_PATH), f'models_trained/{model_name}_scalers.pkl'))
        logger.info(f'Scalers saved: "{PROJECT_PATH}/models_trained/{model_name}_scalers.pkl"')

    @staticmethod
    def load_scalers(model_name, logger):
        ''' Loads scalers
        '''
        # try to load scalers recursively in all subfolders of models_trained
        for dirpath, dirnames, filenames in os.walk(os.path.join(PROJECT_PATH, 'models_trained')):
            for name in filenames:
                if name == f'{model_name}_scalers.pkl':
                    scalers_path = os.path.join(dirpath, name)
                    scalers = joblib.load(scalers_path)

                    # # Create a dictionary with the updated keys
                    # key_mapping = {}
                    # for key in scalers.keys():
                    #     key_mapping[key] = key[:-4]

                    # # Create a new dictionary with the updated keys and values
                    # new_dict = {key_mapping[key]: value for key, value in scalers.items()}

                    logger.info(f'Scalers loaded: {scalers_path}')
                    return scalers

        raise FileNotFoundError(f"Scalers {model_name}_scalers.pkl not found in {os.path.join(PROJECT_PATH, 'models_trained')}")
    

    @staticmethod
    def name_model(model, config):
        '''
        Add model name with parameters to the model object for subsequent saving and logging

        IN:
            model: model object
            config: config dict
        OUT:
            model: model object with name attribute updated
        '''
        ticker=config['AV']['ticker']
        name=config['model']['name']
        window = str(config['model']['window'])
        shuffle_buffer_size = str(config['model']['shuffle_buffer_size']).replace('.','')
        batch_size = str(config['model']['batch_size'])
        epochs = str(config['model']['epochs'])
        model_type = config['model']['type']
        tpw = str(config['model']['trend_prediction_window'])
        n_params = str(model.count_params())

        if model_type == 'price':
            model._name = f"{ticker}_{name}_W{window}_SBS{shuffle_buffer_size}_B{batch_size}_E{epochs}_P{n_params}_{datetime.now().strftime('%Y_%m_%d__%H_%M')}"
        elif model_type == 'trend':
            model._name = f"{ticker}_{name}_trend{tpw}_W{window}_SBS{shuffle_buffer_size}_B{batch_size}_E{epochs}_P{n_params}_{datetime.now().strftime('%Y_%m_%d__%H_%M')}"
        else:
            raise ValueError(f"Model type '{model_type}' is not supported. Supported types are 'price' and 'trend'")
        
        return model
    
    @staticmethod
    def model_forecast_1_feature(model, series, window_size, batch_size):
        """Uses an input model to generate predictions on data windows
        This method is used for transforming data to match windowed_dataset() method

        Args:
        model (TF Keras Model) - model that accepts data windows
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the window
        batch_size (int) - the batch size

        Returns:
        forecast (numpy array) - array containing predictions
        """

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)
        print('--------------------------from_tensor_slices--------------------------')
        for element in dataset:
            print(element)
            break
        print('-'*100)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        print ('-------------------------------window-----------------------------------')
        for window in dataset:
            for element in window:
                print(element)
            break
        print('-'*100)

        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda w: w.batch(window_size))
        print('--------------------------------flat_map--------------------------------')
        for x in dataset:
            print(x)
            break
        print('-'*100)

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)
        print('--------------------------------batch-----------------------------------')
        for x in dataset:
            print(x)
            break
        print('-'*100)
    
        # Get predictions on the entire dataset
        forecast = model.predict(dataset)
        print('--------------------------------forecast-----------------------------------')
        for x in forecast:
            print(x)
            break
        print('-'*100)

        return forecast
    
    @staticmethod
    def model_forecast(model, df, window_size, scalers:dict, verbose=True, logger=None):
        '''
        Generates a forecast from the model

        Args:
            model (tf model) - model to use
            window_size (int) - size of the window
            df (pandas dataframe) - df_test_X - dataframe to create the dataset from
            scalers (dict) - dictionary with the scalers used for each column
            verbose (bool) - whether to logger.info debug info or not

        Returns:
            forecast (numpy array) - array with the forecast
        '''

        X_df = df.copy(deep=True)
        if verbose:
            logger.info('---------------------------------model_forecast() - X_df shape-------------------------------------')
            logger.info (f'X_df.shape: {X_df.shape}')
            logger.info(f'Last 2 records:\n{X_df.tail(2)}')
            logger.info('-'*100)

        # Scale the data
        for col in X_df.columns:
            scaler = scalers[col]
            X_df[col] = scaler.transform(X_df[col].values.reshape(-1,1))
        
        if verbose:
            logger.info('---------------------------------model_forecast() - scalers-------------------------------------')
            logger.info (f'scalers: {scalers}')


        # Creating X as numpy array
        X = X_df.values

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(X)
        if verbose:
            logger.info('--------------------------model_forecast() - from_tensor_slices()--------------------------')
            for element in dataset:
                last_element = element
            logger.info(f'Last element in dataset: {last_element}')
            logger.info('-'*100)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        if verbose:
            logger.info('-------------------------------model_forecast() - window()-----------------------------------')
            for window in dataset:
                last_window = window
            logger.info(f'Last window elements: {list(last_window.as_numpy_iterator())}')
            logger.info('-'*100)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        if verbose:
            logger.info('--------------------------------model_forecast() - flat_map()--------------------------------')
            for window in dataset:
                last_window = window
            logger.info(f'Last window after flatting: {last_window}')
            logger.info('-'*100)

        # batch the data
        dataset = dataset.batch(1)
        if verbose:
            logger.info('--------------------------------model_forecast() - batch()-----------------------------------')
            for batch in dataset:
                last_batch = batch
            logger.info(f'Last batch: {last_batch}')
            logger.info('-'*100)
            
        # Get predictions on the entire dataset
        forecast = model.predict(dataset)
        if verbose:
            logger.info('------------------------model_forecast() - forecast for 2 last ---------------------------')
            for i,x in enumerate(forecast[-2:]):
                logger.info(f'(Prediction {i-2} - {x})')
            logger.info(f'Predicted shape: {forecast.shape}')
            logger.info('-'*100)


        forecast = forecast.squeeze()

        return forecast

    @staticmethod
    def add_tomorrow(df_test_y):
        ''' 
        DEPRECATED - NOT USED. KEPT TO DELETE LATER.
        Adds tomorrow to the dataframe and returns index - which is dates    

        Args:
            df_test_y (pandas dataframe) - dataframe to change
        Returns:
            DFIndex (pandas index) - Index of dates for plotting
        '''

        next_date = df_test_y.index[-1] + pd.DateOffset(days=1)
        new_row = pd.Series(index=[next_date], data=[None]) # taking previous today's price as tomorrow's price for plotting
        df_test_y = pd.concat([df_test_y, new_row])

        logger.info(f"df_test_y.shape after adding next day (tomorrow) to existing data: {df_test_y.shape}")
        logger.info(f"df_test_y.tail(3):\n{df_test_y.tail(3)}")
        return df_test_y.index
    
    @staticmethod
    def get_window_size_from_model_name(model_name):
        ''' 
        Gets the window size from the model name

        Args:
            model_name (string) - name of the model usually obtained from model._name
        Returns:
            window_size (int) - window size
        '''
        window_size = int(re.search(r'W(\d+)_', model_name).group(1))
        return window_size
    
    @staticmethod
    # model name example: 'VGT_LSTM_trend20_W20_SBS1_B128_E1500_P42625_2023_11_19__01_30'
    def get_trend_window_size_from_model_name(model_name):
        ''' 
        Gets the trend window size from the model name

        Args:
            model_name (string) - name of the model usually obtained from model._name
        Returns:
            trend_window_size (int) - trend window size
        '''
        trend_window_size = int(re.search(r'trend(\d+)_', model_name).group(1))
        print(f'trend_window_size: {trend_window_size}')
        return trend_window_size
    
class SklearnModelService(ModelService):
    ''' 
    Methods for model services like training, prediction, saving, loading etc. in sklearn
    After model was Trained
    '''

    @staticmethod
    def save_model(model, model_name, logger):
        ''' Saves model
        '''
        joblib.dump(model, os.path.join(str(PROJECT_PATH), f'models_trained/{model_name}.pkl'))
        logger.info(f'Model saved: "{PROJECT_PATH}/models_trained/{model_name}.pkl"')

    @staticmethod
    def load_model(model_name, logger):
        ''' Loads model
        '''
        model = joblib.load(os.path.join(str(PROJECT_PATH), f'models_trained/{model_name}.pkl'))
        logger.info(f'Model loaded: "{PROJECT_PATH}/models_trained/{model_name}.pkl"')
        return model
    

# --------------------Model Train Hyper-parameters Tuning------------------
class ModelTuningService(ABC):
    ''' Abstract class for model tuning services like grid search, random search etc.'''
    def __init__(self, model, config):
        self.model = model
        self.config = config

    
    @abstractmethod
    def grid_search(self, train_X, train_y, valid_X, valid_y, logger):
        ''' 
        IN: 
            train_X, train_y, valid_X, valid_y - dataframes
        OUT:
            best_params - dict
        '''
        pass


class TensorflowModelTuningService(ModelTuningService):
    '''
    Methods for model tuning services like grid search, random search etc. in tensorflow
    '''
    def __init__(self, model, config):
        super().__init__(model, config)

    
    def grid_search(self, logger, verbose=False):
        ''' 
        IN: 
            self - class object with config and model
        OUT:
            best_params - dict
        '''
        best_params = {}

        _config = self.config.copy()

        windows = self.config['model']['window']
        shuffle_buffer_sizes = self.config['model']['shuffle_buffer_size']
        batch_sizes = self.config['model']['batch_size']
        epochs = self.config['model']['epochs']

        for window in windows:
            _config['model']['window'] = window

            for sbs in shuffle_buffer_sizes:
                _config['model']['shuffle_buffer_size'] = sbs

                for batch in batch_sizes:
                    _config['model']['batch_size'] = batch

                    for e in epochs:
                        _config['model']['epochs'] = e

                        # ----------------------------- Data Preparation -----------------------------
                        train_dataset, scalers_X, df_test_X, df_test_y, initial_df = TensorflowDataPreparation(config=self.config).data_prep(logger, verbose)


                        # -----------------------------Model Training-------------------------------
                        tf.keras.backend.clear_session()

                        self.model = TensorflowModelService.name_model(self.model, _config)
                        window_size = TensorflowModelService.get_window_size_from_model_name(self.model._name)
                        log_model_info(_config, self.model, logger)

                        # self.model.load_weights(f'{PROJECT_PATH}/models_trained/keep/default_model_weights.h5')

                        self.model.compile(loss=_config['model']['loss'], 
                                            optimizer=_config['model']['optimizer'], 
                                            metrics=_config['model']['metrics'],
                                            )      
                        
                        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, verbose=1, restore_best_weights=True)
                        history = self.model.fit(train_dataset, epochs=_config['model']['epochs'], callbacks=[callback])

                        # Plot MAE and Loss
                        mae=history.history['mae']
                        loss=history.history['loss']
                        zoom = int(len(mae) * _config['plots']['loss_zoom'])

                        V.plot_series(x=(range(_config['model']['epochs'])[-zoom:], range(_config['model']['epochs'])[-zoom:]),
                                        y=(mae[-zoom:], loss[-zoom:]),
                                        model_name=self.model._name,
                                        title='Loss',
                                        xlabel='Epochs',
                                        ylabel=f'MAE and Loss',
                                        legend=['MAE', f'Loss - {_config["model"]["loss"]}'],
                                        show=_config['plots']['show'],
                                        signal=False
                                    )
                        
                        # -----------------------------Model Save-----------------------------
                        TensorflowModelService.save_model(model=self.model, logger=logger)    
                        TensorflowModelService.save_scalers(scalers=scalers_X, model_name=self.model._name ,logger=logger)

                        
                        # -----------------------------Predictions-----------------------------------
                        results = TensorflowModelService.model_forecast(model=self.model, 
                                                                df=df_test_X,
                                                                window_size=window_size,
                                                                scalers=scalers_X,
                                                                verbose=verbose,
                                                                logger=logger)

                        logger.info(f'results.shape: {results.shape}')
                        logger.info(f'results[-3:]: {results[-3:]}')
                        logger.info(f"df_test_y.shape: {df_test_y.shape}")
                        logger.info(f"df_test_y.tail(3):\n{df_test_y.tail(3)}")

                        V.plot_series( x=(df_test_y.index, df_test_y.index[window_size-1:]),  # as dates
                                        y=(df_test_y, results),                               # as values
                                        model_name=self.model._name,
                                        title='Pred',
                                        xlabel='Date',
                                        ylabel='Price',
                                        legend=['Actual', 'Predicted'],
                                        show=_config['plots']['show'],
                                        signal=_config['plots']['buy_sell_signal'])
                        
                        # -----------------------Calculate Errors----------------------------------
                        df_test_y_no_window = df_test_y.iloc[window_size-1:]      # actual values without window size in the beginning

                        naive_forecast = ErrorCalc.get_naive_forecast(initial_df).loc[df_test_y_no_window.index] 
                        rmse, mae, mape, mase = ErrorCalc.calc_errors(df_test_y_no_window, results, naive_forecast, _config['model']['type'])
                        ErrorCalc.save_errors_to_table(self.model._name, {'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase})


                        # -----------------------Log Best Params----------------------------------
                        if best_params == {}:
                            best_params = {'window': window, 'shuffle_buffer_size': sbs, 'batch_size': batch, 'epochs': e, 'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase}
                        else:
                            if _config['model']['type'] == 'price':
                                if mase < best_params['mase']:
                                    best_params = {'window': window, 'shuffle_buffer_size': sbs, 'batch_size': batch, 'epochs': e, 'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase}
                                    
                            elif _config['model']['type'] == 'trend':
                                if rmse < best_params['rmse']:
                                    best_params = {'window': window, 'shuffle_buffer_size': sbs, 'batch_size': batch, 'epochs': e, 'rmse': rmse, 'mae': mae, 'mape': mape, 'mase': mase}

        logger.info(f'Best params: {best_params}')



