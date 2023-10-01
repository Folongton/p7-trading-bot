from abc import ABC, abstractmethod

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os, joblib
import logging

from src.common.globals import G
PROJECT_PATH = G.get_project_root()
from src.common.logs import setup_logging

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO,
                        log_file_level=logging.INFO)

# -----------------------------Model Pre Train --------------------------
class DataPreparationService(ABC):
    ''' 
    Classes for data preparation services to fit the model.
    Also a base class for general methods
    '''


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
    
class TensorflowDataPreparation(DataPreparationService):
    ''' Methods for data preparation services to fit the model in tensorflow'''
    

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
            logger.info('---------------------------------scalers-------------------------------------')
            logger.info (f'scalers: {scalers}')
            logger.info('-'*100)


        # Creating X
        X = X_df.values

        if verbose:
            logger.info('---------------------------------X shape-----------------------------')
            logger.info (f'X.shape: {X.shape}')
            logger.info('-'*100)

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(X)
        if verbose:
            logger.info('--------------------------from_tensor_slices--------------------------')
            for element in dataset:
                logger.info(element)
                break
            logger.info('-'*100)

        # Window the data but only take those with the specified size
        # And add + 1 to the window size to account for the label, which we will separate later
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        if verbose:
            logger.info('-------------------------------window-----------------------------------')
            for window in dataset:
                logger.info(type(window))
                logger.info(list(window.as_numpy_iterator()))
                break
            logger.info('-'*100)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        if verbose:
            logger.info('--------------------------------flat_map--------------------------------')
            for window in dataset:
                logger.info(window)
                break
            logger.info('-'*100)

        if verbose:
            logger.info(f'Lenght of X = {len(list(dataset.as_numpy_iterator()))}')

        return dataset, scalers

    @staticmethod # for N features
    def windowed_dataset_y(df, window_size, logger, verbose=True):
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
            logger.info('---------------------------------y shape-------------------------------------')
            logger.info (f' y.shape: {y.shape}')
            logger.info('-'*100)
        
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(y)
        if verbose:
            logger.info('--------------------------from_tensor_slices--------------------------')
            for element in dataset:
                logger.info(element)
                break
            logger.info('-'*100)

        # calculate number of points we need to cut to make series evenly divisible by window_size
        remainder = window_size - 1

        # Remove the reminder elements from the end of dataset
        dataset = dataset.take(len(y) - remainder)
        if verbose:
            logger.info('--------------------------------take len(y)-reminder--------------------------------')
            for window in dataset:
                logger.info(window)
                break
            logger.info('-'*100)

        if verbose:
            logger.info(f'Lenght of y = {len(list(dataset.as_numpy_iterator()))}')
        
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
        zipped_dataset = zipped_dataset.shuffle(config['model']['shuffle_buffer_size'])
        zipped_dataset = zipped_dataset.batch(config['model']['batch_size']).prefetch(1)

        if verbose:
            for x, y in zipped_dataset:
                logger.info(f'x.shape: {x.numpy().shape}, y.shape: {y.numpy().shape}')
                break
        
            input_shape = zipped_dataset.element_spec[0].shape
            logger.info(f'Full Dataset shape: {input_shape}')
            logger.info(f'Input for the model: {input_shape[1:]}')

        return zipped_dataset
    
class SklearnDataPreparation(DataPreparationService):
    ''' Methods for data preparation services to fit the model in sklearn'''
    pass


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
    def save_model(model, model_name=None, logger=None):
        if model_name is None:
            model_name = model._name
        model.save(os.path.join(PROJECT_PATH, rf'models_trained/{model_name}.keras'))
        logger.info(f"Model saved as {model._name}.keras")

    @staticmethod
    def load_model(model_name, logger):

        model_path = os.path.join(PROJECT_PATH, rf'models_trained/keep/{model_name}.keras')
        print (model_path)
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    
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
        scalers = joblib.load(os.path.join(str(PROJECT_PATH), f'models_trained/keep/{model_name}_scalers.pkl'))

        # Create a dictionary with the updated keys
        key_mapping = {}
        for key in scalers.keys():
            key_mapping[key] = key[:-4]

        # Create a new dictionary with the updated keys and values
        new_dict = {key_mapping[key]: value for key, value in scalers.items()}

        logger.info(f'Scalers loaded: "{PROJECT_PATH}/models_trained/{model_name}_scalers.pkl"')
        return new_dict
    
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
    def model_forecast(model, df, window_size, scalers:dict, verbose=True):
        '''
        Generates a forecast from the model

        Args:
            model (tf model) - model to use
            window_size (int) - size of the window
            df (pandas dataframe) - dataframe to create the dataset from
            scalers (dict) - dictionary with the scalers used for each column
            verbose (bool) - whether to logger.info debug info or not

        Returns:
            forecast (numpy array) - array with the forecast
        '''

        # change the position of the target column to the end
        # df = DataPreparationService.label_column_to_end(df, 'Adj Close')
        # if verbose:
        #     logger.info('---------------------------------df shape-------------------------------------')
        #     logger.info (f'df.shape: {df.shape}')
        #     logger.info(df.iloc[:2])
        #     logger.info('-'*100)

        X_df = df.copy(deep=True)
        if verbose:
            logger.info('---------------------------------X_df shape-------------------------------------')
            logger.info (f'X_df.shape: {X_df.shape}')
            logger.info(X_df.iloc[:2])
            logger.info('-'*100)

        # Scale the data
        for col in X_df.columns:
            scaler = scalers[col]
            X_df[col] = scaler.transform(X_df[col].values.reshape(-1,1))
        
        # Creating X
        X = X_df.values
        if verbose:
            logger.info('---------------------------------X shape-------------------------------------')
            logger.info (f'X.shape: {X.shape}')
            logger.info('-'*100)


        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(X)
        if verbose:
            logger.info('--------------------------from_tensor_slices--------------------------')
            for element in dataset:
                logger.info(element)
                break
            logger.info('-'*100)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        if verbose:
            logger.info('-------------------------------window-----------------------------------')
            for window in dataset:
                logger.info(type(window))
                logger.info(list(window.as_numpy_iterator()))
                break
            logger.info('-'*100)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        if verbose:
            logger.info('--------------------------------flat_map--------------------------------')
            for window in dataset:
                logger.info(window)
                break
            logger.info('-'*100)

        # batch the data
        dataset = dataset.batch(1)
        if verbose:
            logger.info('--------------------------------batch-----------------------------------')
            for batch in dataset:
                logger.info(batch)
                break
            logger.info('-'*100)
            
        # Get predictions on the entire dataset
        forecast = model.predict(dataset)
        if verbose:
            logger.info('------------------------forecast for 2 first ---------------------------')
            for i,x in enumerate(forecast):
                if i > 1:
                    break
                logger.info(f'(Prediction {i} - {x})')
            logger.info(f'Predicted shape: {forecast.shape}')
            logger.info('-'*100)


        forecast = forecast.squeeze()

        return forecast

    @staticmethod
    def prep_test_df_shape(test_df, config):
        ''' 
        Prepares the test dataframe to plot the results.
        Where -config['model']['window']+1 is to account for the window size. 
        We can't predict for 19 days if the window size is 20, because we don't have the 20th day yet.
        Say we have 100 days, we can only predict for 81 days, because we don't have the 82nd day yet to form the window of 20 days.

        Args:
            test_df (pandas dataframe) - dataframe to change
            results (numpy array) - array with the forecast
            config (dict) - dictionary with the configuration
        Returns:
            df_test_minus_window (pandas dataframe) - dataframe with the last n days removed
        '''
        df_test_minus_window = test_df.iloc[:-config['model']['window']+1].copy(deep=True)
        return df_test_minus_window
    
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