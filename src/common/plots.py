import pandas as pd

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
    def plot_series(x, y, model_name, format="-", start=0, show=True, end=None, title=None, xlabel=None, ylabel=None, legend=None):
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
        """
        plt.clf()
        # Check if there are more than two series to plot
        if type(y) is tuple:

            # Loop over the y elements
            for i, y_curr in enumerate(y):
                if i == 0:
                    # Plot the x and current y values
                    plt.plot(x[start:end], y_curr[start:end], format)
                if i == 1:
                    # Plot the current y values with a different style
                    plt.plot(x[start:end], y_curr[start:end], format, color='orange')

        else:
            # Plot the x and y values
            plt.plot(x[start:end], y[start:end], format)

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