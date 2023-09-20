import numpy as np
import pandas as pd
import tensorflow as tf
import os

from src.common.globals import G

PROJECT_PATH = G.get_project_root()

class FeatureEngineering(pd.DataFrame):
    def __init__(self):
        super().__init__()

    def create_lags(self, column, lag):
        ''' Create lagged columns'''
        for lag in range(1, lag + 1):
            self[f"{column} - {lag}"] = self[column].shift(lag)
            self.dropna(how='any', inplace=True)
        return self
    
    def log_scale(self, column):
        ''' Log scale column'''
        self[f"{column} - log"] = np.log(self[column])
        return self
    
    @staticmethod
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
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
    
    @staticmethod
    def model_forecast(model, series, window_size, batch_size):
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
    def model_save(model, logger):
        model.save(os.path.join(PROJECT_PATH, rf'models_trained\{model._name}.keras'))
        logger.info(f"Model saved as {model._name}.h5")


    # def create_rolling_min(self, column, window):
    #     self.data[f"{column} - rolling min"] = self.data[column].rolling(window).min()
    #     return self.data
    
    # def create_rolling_max(self, column, window):
    #     self.data[f"{column} - rolling max"] = self.data[column].rolling(window).max()
    #     return self.data


    # def create_rolling_std(self, column, window):
    #     self.data[f"{column} - rolling std"] = self.data[column].rolling(window).std()
    #     return self.data
    
    # def create_rolling_mean(self, column, window):
    #     self.data[f"{column} - rolling mean"] = self.data[column].rolling(window).mean()
    #     return self.data


    # def feature_engineering(self):
        # self.data["date"] = pd.to_datetime(self.data["date"])
        # self.data["month"] = self.data["date"].dt.month
        # self.data["day"] = self.data["date"].dt.day
        # self.data["year"] = self.data["date"].dt.year
        # self.data["week"] = self.data["date"].dt.week
        # self.data["dayofweek"] = self.data["date"].dt.dayofweek
        # self.data["dayofyear"] = self.data["date"].dt.dayofyear
        # self.data["quarter"] = self.data["date"].dt.quarter
        # self.data["is_month_start"] = self.data["date"].dt.is_month_start
        # self.data["is_month_end"] = self.data["date"].dt.is_month_end
        # self.data["is_quarter_start"] = self.data["date"].dt.is_quarter_start
        # self.data["is_quarter_end"] = self.data["date"].dt.is_quarter_end
        # self.data["is_year_start"] = self.data["date"].dt.is_year_start
        # self.data["is_year_end"] = self.data["date"].dt.is_year_end
        # self.data["is_leap_year"] = self.data["date"].dt.is_leap_year
        # self.data["days_in_month"] = self.data["date"].dt.days_in_month
        # self.data["is_weekend"] = np.where(
        #     self.data["dayofweek"].isin([5, 6]), 1, 0
        # )
        # self.data["is_weekday"] = np.where(
        #     self.data["dayofweek"].isin([0, 1, 2, 3, 4]), 1, 0
        # )
        # self.data["is_workingday"] = np.where(
        #     self.data["dayofweek"].isin([0, 1, 2, 3, 4]), 1, 0
        # )
        # self.data["is_holiday"] = np.where(
        #     self.data["dayofweek"].isin([5, 6]), 1, 0
        # )
        # self.data["is_prev_holiday"] = np.where(
        #     self.data["dayofweek"].isin([5, 6]), 1, 0
        # )
        # self.data["is_next_holiday"] = np.where(
        #     self.data["dayofweek"].isin([5, 6]), 1, 0
        # )
        
