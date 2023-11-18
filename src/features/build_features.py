import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
        Creates column 'Tomorrow' which is the Adj Close shifted by -1,
        With other columns as features.

        Args:
            df (pandas dataframe) - dataframe to change
        Returns:
            df (pandas dataframe) - dataframe with 'Tomorrow' column and the features
        '''

        self['Tomorrow'] = self['Adj Close'].shift(-1)
        self = self.dropna()
        
        logger.info('--------------------create_features() - shift(-1)--------------------')
        # logger.info(self) # TEMPORARY
        logger.info(f'df.shape: {self.shape}')
        logger.info(f'df.columns: {self.columns}')

        return self

    def rename_shifted_columns(self):
        ''' 
        Renames the shifted columns to remove the ' - 1' at the end of the column name

        Args:
            df (pandas dataframe) - dataframe to change
        Returns:
            df (pandas dataframe) - dataframe with the shifted columns renamed
        '''
        for col in self.columns:
            if col.endswith(' - 1'):
                self = self.rename(columns={col: col[:-4]})
        return self
    
    @staticmethod
    def slope(ser):
        '''
        Calculates the normilized slope of a pandas Series - Trend. 
        Trend is calcualted by fitting a linear regression line on
        the series and returning its slope.

            INPUT: ser - pandas Series
            OUTPUT: slope - float
        '''
        x = np.array(range(len(ser)))
        x = (x - x.min())/(x.max() - x.min())
        x = x.reshape(-1,1)

        y = ser.values
        y = (y - y.min())/(y.max() - y.min())
        y = y.reshape(-1,1)

        lr = LinearRegression()
        lr.fit(x, y)

        return lr.coef_[0][0]

    @staticmethod
    def calc_window_slopes(ser, window_size):
        '''
        Calculates the slope of the series for a given window size.
        The slope is calcualted for each point of the series by using the slope function above. 
        Slope calculated feature window_size days. 
        Meaning that the first value of the returned list is the slope of the first window_size days of the series. 
        EXAMPLE:
        [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] - series
        [x1, x2, x3, x4, x5] - first window_size days if window_size = 5.
        If y - is a slope array, then [y1] - slope of the first window [x1, x2, x3, x4, x5].

            INPUT: ser - pandas Series
                window_size - int
            OUTPUT: slopes - list
        '''
        slopes = []
        for i in range(0, len(ser)+1 - window_size):
            slopes.append(FeatureEngineering.slope(ser[i:i+window_size]))

        slopes.extend([np.nan] * (window_size - 1))

        return slopes

    def create_slope_column(self, logger, slope_window): 
        '''
        Creates slope column for each slope window.
        slope_window are the number of days to calculate the slope on. 
        EXAMPLE:
        slope_windows = 20
        The function will create a column.
        The values in the column will be the slope of the series for the feature n days, starting from the current day including it.
        Meaning that the value in the slope_20 column for row 100 will be the slope of the series for days 100-120.
  

            INPUT: df - pandas
                    slope_window - int
            OUTPUT: df - pandas with slope column
        '''

        self[f'slope_{slope_window}'] = FeatureEngineering.calc_window_slopes(self['Adj Close'], slope_window)
        self = self.dropna()
        
        logger.info('--------------------FE.create_slope_column()--------------------')
        # logger.info(self) # TEMPORARY
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
        
