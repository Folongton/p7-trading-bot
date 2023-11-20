import pandas as pd
import numpy as np

import os

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.common.globals import G

import logging
from src.common.logs import setup_logging

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('Solarize_Light2')

PROJECT_DIR = G.get_project_root()
DATA_DIR = f'{PROJECT_DIR}\data\\00_raw\daily_full'


logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)


class Visualize:
    @staticmethod
    def calc_prct_gain(series):
        return series.apply(lambda x: (x - series.iloc[0])/series.iloc[0])
    
    @staticmethod
    def plot_prct_gain(stocks, from_date, to_date=datetime.now()-timedelta(days=1) , title='Percent Gain'):
        ''' Plots percent gain for list of stocks'''
        plt.figure(figsize=(15, 5))
        for stock in stocks:
            df = pd.read_csv(os.path.join(DATA_DIR, rf'{stock}-daily-full.csv'), index_col=0, parse_dates=True)
            df.index.rename('Date', inplace=True)
            df = df.copy()[::-1]
            df = df.loc[from_date:to_date].copy(deep=True)

            Visualize.calc_prct_gain(df['5. adjusted close']).plot()

        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        plt.gca().yaxis.tick_right()
        plt.legend(stocks)
        plt.title(title)
        plt.legend(stocks)
        plt.show()  

    @staticmethod
    def construct_fig_path(model_name, title):
        '''Constructs path for saving figure'''
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-5]
        file_name = f'{title}-{model_name}.png'
        fig_dir = os.path.join(PROJECT_DIR, r'figures')
        fig_path = os.path.join(fig_dir, file_name)
        return fig_path, file_name
    
    @staticmethod
    def plot_1_by_2(df, model_name, col_before='Adj Close', col_after='Adj Close - log', title1='Original Series', title2='Modified Series', show=True):
        ''' Plots and saaves plot of 2 subplots'''
        fig, axes = plt.subplots(1, 2, figsize=(16,5))
        axes[0].plot(df[col_before]); axes[0].set_title(title1)
        axes[1].plot(df[col_after]); axes[1].set_title(title2)

        fig_path, file_name = Visualize.construct_fig_path(model_name, title='1 by 2')
        plt.savefig(fig_path)
        logger.info(f'Plot "{file_name}" saved to "{fig_path}"')

        if show == True:
            plt.show()
    
    @staticmethod
    def plot_correlation(series, intervals='Days'):
        '''
        This function plots the correlation between the series - Price and the index of series - Time.
        '''
        series = series.ffill()
        series = series.bfill()

        X = series.reset_index().index.values.reshape(-1,1) # 60 days of data in exact same interval - 1 day.
        y = series.values.reshape(-1,1) # prices 

        model = LinearRegression()
        model.fit(X, y)

        plt.style.use('Solarize_Light2')
        plt.figure(figsize=(10,6))
        
        plt.scatter(X, y, s=10, color='#294dba')
        plt.plot(model.predict(X), color='#d9344f')
        
        plt.xlabel(intervals)
        plt.ylabel('Price')
        plt.title(f'Linear Correlation of {series.name} with linear regression line')
        plt.show()

    @staticmethod # Deprecated - use more general plot_series() instead
    def plot_pred_vs_actual(y_true, y_pred, model_name, title='Prediction vs Actual', show=True):
        ''' Plots and saves plot for actual vs predicted values
        ----- DEPRECATED --- USE plot_series() INSTEAD----
        INPUT: y_true: actual values,
                y_pred: predicted values,
                model_name: name of the model,
                title: title of the plot,
                show: whether to show the plot
        OUTPUT: plot saved to figures folder and shown plot if show=True
        '''
        try:
            plt.plot(y_true.values)
            plt.plot(y_pred.values, color='red')
        except AttributeError:
            plt.plot(y_true)
            plt.plot(y_pred, color='red')
        plt.legend(['Actual', 'Predicted'])
        
        fig_path, file_name = Visualize.construct_fig_path(model_name, title)
        plt.savefig(fig_path)
        logger.info(f'Plot "{file_name}" saved to "{fig_path}"')
        
        if show == True:
            plt.show()
        
    @staticmethod
    def plot_series(x, y, model_name, signal=True, format="-", show=True, title=None, xlabel=None, ylabel=None, legend=None):
        """
        Visualizes time series data

        Args:
            x (array of int) - contains values for the x-axis
            y (array of int or tuple of arrays) - contains the values for the y-axis
            format (string) - line style when plotting the graph
            start (int) - first time step to plot
            end (int) - last time step to plot
            title (string) - title of the plot
            xlabel (string) - label for the x-axis
            ylabel (string) - label for the y-axis
            legend (list of strings) - legend for the plot

        Returns:
            fig_path (string) - path to the saved figure
        """
        plt.clf()
        # Check if there are more than two series to plot
        if type(y) is tuple and type(x) is tuple:
            for i, zipped_t in enumerate(zip(x, y)):
                x_curr, y_curr = zipped_t
                if i == 0:
                    # Plot the first y values (blue - Actual values)
                    plt.plot(x_curr, y_curr, format)
                elif i == 1:
                    # Plot the second y values (orange - Predicted values)
                    # cut x from start to be the same as y_curr (since y_curr is predicted values and is shorter than x by window_size)
                    plt.plot(x_curr, y_curr, format, color='orange')

                    if signal == True:
                        # Calc derivative
                        derivatives = list(np.gradient(y_curr, 1))
                        # if derivatives are negative 3 days in a row, plot a dot on the graph and label it "Sell"
                        # if derivatives are positive 3 days in a row, plot a dot on the graph and label it "Buy"
                        bought = None 
                        for i, derivative in enumerate(derivatives):
                            if i < 2:
                                continue
                            elif (derivative > 0) and (derivatives[i-1] > 0) and (derivatives[i-2] > 0) and (bought in [None, False]):
                                bought = True
                                plt.plot(x_curr[i], y_curr[i], 'go')
                                plt.text(x_curr[i], y_curr[i], 'Buy', color='green', fontsize=14, ha='center')

                            elif (derivative < 0) and (derivatives[i-1] < 0) and (derivatives[i-2] < 0) and (bought in [True]):
                                bought = False
                                plt.plot(x_curr[i], y_curr[i], 'ro')
                                plt.text(x_curr[i], y_curr[i], 'Sell', color='red', fontsize=14, ha='center')

                else:
                    raise ValueError('More than 2 series to plot')                     
            

        else:
            plt.plot(x, y, format)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    
        try:
            plt.legend(legend)
        except TypeError:
            print('No legend provided. Plotting without legend.')

        plt.title(title)
        plt.grid(True)

        fig_path, file_name = Visualize.construct_fig_path(model_name, title)
        plt.savefig(fig_path)
        logger.info(f'Plot "{file_name}" saved to "{fig_path}"')
        
        if show == True:
            plt.show()

        return fig_path
    
    @staticmethod
    def slopes(results_time_price_adjusted: tuple, time_index: pd.DatetimeIndex, pred_trend_window: int):
        """
        Calculates x, y values for slope lines. Slopes coming as predictions from a model.
        And we want to plot them on top of the original price time series.
        *** THESE ARE SLOPE LINES, NOT PRICE LINES ***
        ***We are plotting the trend and strength of the trend, not the price.***

        This function computes the slope values for a series of points defined by the
        time and price adjustments. It is intended to be used in financial or time series
        analysis for trend analysis.

        Parameters:
        - results_time_price_adjusted (tuple): A tuple containing three elements (slopes, time positions, prices).
        - time_index (pd.DatetimeIndex): A Pandas DatetimeIndex representing time points.
        - pred_trend_window (int): The prediction window size for trend calculation.

        Returns:
        - list: A list of tuples, where each tuple contains arrays of x and y values representing the slopes.
        """
        slopes_values = []
        for i, (slope, tpa, price_) in enumerate(zip(*results_time_price_adjusted)):
            if i == len(results_time_price_adjusted[0]) - 1:
                break

            x = np.linspace(tpa, results_time_price_adjusted[1][i+1] - 1, pred_trend_window)
            b = price_ - slope * tpa
            y = slope * x + b

            for inx, time_index_value in enumerate(time_index):
                if inx == tpa:
                    x = time_index[inx: inx+pred_trend_window]

            slopes_values.append((x, y))

        return slopes_values

    @staticmethod
    def plot_slopes_price(slopes_values, time_index, price, window_size):
        """
        Plot slope lines against a price time series.

        This function plots the calculated slope lines on top of the original price time series,
        providing a visual representation of the trends.

        Parameters:
        - slopes_values: Calculated slope values from the 'slopes' function.
        - time_index (pd.DatetimeIndex): A Pandas DatetimeIndex representing time points.
        - price (array-like): An array-like object of price values.
        - window_size (int): The size of the window used in slope calculation.
        """
        for x, y in slopes_values:
            plt.plot(x, y, color='orange')

        plt.plot(time_index, price[window_size-1:])

    @staticmethod
    def plot_future_slope(results_adjusted, time_positions_adjusted, price_adjusted, time_index, pred_trend_window):
        """
        Plot the future slope based on the last point of the given time series.

        This function extends the time series into the future and plots the projected slope
        starting from the last point of the original series.
        *** THIS PLOTS A SLOPE FOR THE NEXT N DAYS, NOT THE NEXT N POINTS IN THE SERIES ***
        ***Slope represents the trend and strength of the trend, not the price.***

        Parameters:
        - results_adjusted: Adjusted results from the slope calculation.
        - time_positions_adjusted: Adjusted time positions corresponding to the results.
        - price_adjusted: Adjusted price values.
        - time_index (pd.DatetimeIndex): The original time index of the series.
        - pred_trend_window (int): The prediction window size.
        """
        last_point = time_index[-1]
        extended_time_index = pd.date_range(start=last_point, periods=pred_trend_window, freq='D')

        x = np.linspace(time_positions_adjusted[-1], time_positions_adjusted[-1]+(pred_trend_window-1), pred_trend_window)
        b = price_adjusted.iloc[-1] - results_adjusted[-1]*time_positions_adjusted[-1]
        y = results_adjusted[-1]*x + b

        x = extended_time_index[:pred_trend_window]
        plt.plot(x, y, color='orange')

    @staticmethod
    def plot_trends(results, price, df_test_y, window_size, pred_trend_window):
        """
        Plot the trends of a given time series data with future projection.

        This function integrates several steps: adjusting results and price data,
        calculating slopes, and plotting both historical trends and future projection.

        Parameters:
        - results: Computed results from a trend analysis model.
        - price: Original price data.
        - df_test_y (pd.DataFrame): A DataFrame containing the test dataset.
        - window_size (int): The window size used in trend analysis.
        - pred_trend_window (int): The prediction window size for future projection.
        """
        # Adjust the results, price, and time positions
        results_adjusted = results[::-pred_trend_window][::-1]
        price_adjusted = price[window_size-1:][::-pred_trend_window][::-1]
        time_positions_adjusted = [i for i in range(len(df_test_y.index[window_size-1:]))][::-pred_trend_window][::-1]
        time_index = df_test_y.index[window_size-1:]

        slopes_values = Visualize.slopes((results_adjusted, time_positions_adjusted, price_adjusted), time_index, pred_trend_window)

        Visualize.plot_slopes_price(slopes_values, time_index, price, window_size)
        Visualize.plot_future_slope(results_adjusted, time_positions_adjusted, price_adjusted, time_index, pred_trend_window)
        
        plt.show()