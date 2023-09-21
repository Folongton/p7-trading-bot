import numpy as np
import pandas as pd


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
    
    def drop_non_features(self):
        ''' 
        Drops all columns except for the Adj Close and the features
        
        Args:
            df (pandas dataframe) - dataframe to change
        Returns:
            df (pandas dataframe) - dataframe with only the Adj Close and the features
            '''
        for col in self.columns:
            if col != 'Adj Close' and col.endswith(' - 1') == False:
                self = self.drop(columns=[col])

        return self
    
    def create_features(self, logger):
        '''
        Creates features from the original dataframe by shifting the columns by 1 day
        And then drops the columns that are not features or the 'Adj Close'

        Args:
            df (pandas dataframe) - dataframe to change
        Returns:
            df (pandas dataframe) - dataframe with only the Adj Close and the features
        '''
        for col in self.columns:
            self[f'{col} - 1'] = self[col].shift(1)
        self = self.dropna()
        self = FeatureEngineering.drop_non_features(self)
        logger.info(f'df.shape: {self.shape}')
        logger.info(f'df.columns: {self.columns}')

        return self

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
        
