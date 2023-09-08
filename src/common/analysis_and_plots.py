import numpy as np
import pandas as pd

import os
from pathlib import Path

import seaborn as sns
from scipy.stats import geom
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

from ta.volatility import  BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator
from ta.momentum import RSIIndicator, KAMAIndicator
from ta.volume import OnBalanceVolumeIndicator

from src.data.get_data import CSVsLoader as CSVs
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

class Analysis:
    '''Global Methods related to stock data analysis and manipulation
    '''

    @staticmethod
    def assign_return_bucket(stock):
        '''Assigns a bucket to a percent change in stock price
        INPUT: stock: dataframe of stock data
        OUTPUT: dataframe of stock data with 'return bucket' column and 'UP or DOWN' column
        '''
        df_name = stock.name
        stock['Daily Return'] = stock['5. adjusted close'].pct_change() # Calculate the daily returns for each stock into a new column called 'Daily Return'
        conditions = [stock['Daily Return'] < -0.05, 
                    stock['Daily Return'].between(-0.05, -0.04, inclusive='left'), 
                    stock['Daily Return'].between(-0.04, -0.03, inclusive='left'), 
                    stock['Daily Return'].between(-0.03, -0.02, inclusive='left'), 
                    stock['Daily Return'].between(-0.02, -0.01, inclusive='left'), 
                    stock['Daily Return'].between(-0.01, 0, inclusive='left'), 
                    stock['Daily Return'].between(0, 0.01, inclusive='left'), 
                    stock['Daily Return'].between(0.01, 0.02, inclusive='left'), 
                    stock['Daily Return'].between(0.02, 0.03, inclusive='left'), 
                    stock['Daily Return'].between(0.03, 0.04, inclusive='left'), 
                    stock['Daily Return'].between(0.04, 0.05, inclusive='left'), 
                    stock['Daily Return'] >= 0.05]
        values = [-5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5]
        stock['return bucket'] = np.select(conditions, values)
        stock = stock[~stock['Daily Return'].isnull()]

        # create column called 'UP or DOWN' to indicate whether the stock price increased or decreased and populate it based on the 'return bucket' column
        conditions = [stock['return bucket'].isin([ -1, -2, -3, -4, -5]),
                    stock['return bucket'].isin([0, 1, 2, 3, 4, 5])]
        values = [-1, 1]
        stock = stock.copy(deep=True)
        stock.name = df_name
        stock['UP or DOWN'] = np.select(conditions, values)

        return stock
    
    @staticmethod
    def calc_proba_price_change_based(stock_df, percentChange, days_back, print_proba=False):
        '''Calculates probabilities of a stock to go up or down next day, based on percent change in price of present day.
        Calculated based on previous occurences of percent change in price of stock.
        INPUT: stock_df: dataframe of stock data,
                percentChange: percent change in price of stock,
                days: number of days to look back for calculating
                print_proba: whether to print probabilities or only return dataframe with all probabilities
        OUTPUT: dataframe with probabilities of stock to go up or down next day
        '''
        df = Analysis.assign_return_bucket(stock_df)
        df = df.copy(deep=True)[-days_back:]
        # shift return bucket column up by one row to match the previous day's return bucket
        df['previous_day_return_bucket'] = df['return bucket'].rolling(2).apply(lambda x: x.iloc[0])
        # calculate probability of up or down given previous day's return bucket
        proba_df = df[['previous_day_return_bucket','UP or DOWN']].value_counts(normalize=True).to_frame()
        proba_df = proba_df.rename(columns={'proportion':'Probability'}).pivot_table(columns='previous_day_return_bucket', index='UP or DOWN', values='Probability')

        percentChange = float(percentChange)
        final_series = proba_df[percentChange].rename(index={-1.0:'Negative', 1.0:'Positive'})
        for r in final_series.index:
            if print_proba:
                final_series = final_series / final_series.sum() * 100
                print(f'Probability of {r} move next day is {round(final_series[r],2)}%')
        return proba_df
    
    @staticmethod
    def correlation_of_stock_price_reaction_to_price_change(all_stocks, days_to_look_back=365):
        ''' Compares stocks based on probability of price change next day, based on percent change in price of present day.
        Basically answering question : ***How correlated are the stocks in terms of price change next day, based on percent change in price of present day.***
        INPUT: all_stocks: list of stocks to compare,
                days_to_look_back: number of days to look back for calculating
        OUTPUT: dataframe with colinearity of stocks based on probability of price change next day, based on percent change in price of present day.
        '''
        colinearity_df = pd.DataFrame()
        for stock in all_stocks:
            df = CSVs.load_daily(stock)
            df = Analysis.calc_proba_price_change_based(df, percentChange=0, days_back=days_to_look_back, print_proba=False) # here we getting full probability dataframe to use for correlation, so here we can disregard percentChange argument as well as print_proba.
            df = df.iloc[1].to_frame().T
            df.index = [stock] * len(df)
            colinearity_df = pd.concat([colinearity_df, df], axis=0)
        sns.heatmap(colinearity_df.T.corr(), annot=True, cmap='coolwarm')
        return colinearity_df
    
    @staticmethod
    def proba_up_down_geometric(stock_df, cont_days, days_back, direction):
        '''Calculates probability of a stock to go on specified direction for a given number of days in a row.
        Basically Geometric Distribution of a stock going up or down for a given number of days in a row based on previous days going up or down.
        INPUT: stock_df: dataframe of stock data with calculated 'UP or DOWN' column,
                        To calculate 'UP or DOWN' column use Analysis.assign_return_bucket() method,
                cont_days: number of days in a row to check for,
                period_days: number of days to look back for calculating,
                direction: direction to check for ('up' or 'down')
        OUTPUT: probability of a stock to go on specified direction for a given number of days in a row
        '''
        df_name = stock_df.name
        if direction == 'up':
            direction_int = 1
        if direction == 'down':
            direction_int = -1
        df = stock_df.copy(deep=True)[-days_back:]
        df.name = df_name
        proba = geom.pmf(cont_days, df['UP or DOWN'].value_counts(normalize=True).loc[direction_int])
        print(f'Probability of {stock_df.name} going "{direction.upper()}" for {cont_days} days in a row is {round(proba*100,3)}%. Based on {days_back} days of data.')

    @staticmethod
    def aggregate_close_prices(directory, calendar_days_back=90 ,intraday=False, day_for_intraday='2023-03-24'):
        '''
        This function takes all the csv files in the directory and aggregates them into one dataframe.
        directory: The directory where the data is stored.
        intraday: If True it will take only points from '10:00' to '13:30'.
        '''
        df_agg = pd.DataFrame()
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, file), index_col=0, parse_dates=True)
                if df.empty:
                    continue

                if intraday:
                    df = df.loc[day_for_intraday]
                    df = df.between_time('10:00', '13:30')
                    df = df[['5. adjusted close']].iloc[::-1] 
                    if df.empty:
                        continue
                    if len(df) < 24: #24 periods of 5 min = 2 hours
                        continue
                    if df['5. adjusted close'].nunique() == 1:
                        continue

                else:
                    start_date = datetime.now() - timedelta(days=calendar_days_back) 
                    start_date = start_date.strftime('%Y-%m-%d')
                    df = df.loc[df.index > start_date]
                    df = df[['5. adjusted close']].iloc[::-1]
                    if df.empty:
                        continue
                    if df['5. adjusted close'].nunique() == 1:
                        continue
                
                df.columns = [file.split('-')[0]]
                df_agg = pd.concat([df_agg, df], axis=1)
                    
        return df_agg
    
    @staticmethod
    def find_best_linear_stock(df):
        '''
        IN: dataframe of ticker - columns and close price - rows 
        OUT: dataframe with R2, Slope for each ticker.
        '''
        df_stats = pd.DataFrame(columns=['ticker', 'R2', 'slope'])
        for col in df.columns:
            series = df[col].copy().ffill()
            series = series.copy().bfill()

            X = series.reset_index().index.values.reshape(-1,1)
            y = series.values.reshape(-1,1)
            model = LinearRegression().fit(X,y)
        
            df_stats = pd.concat([df_stats, pd.DataFrame({'ticker': col, 'R2': model.score(X,y), 'slope': model.coef_[0]}, index=[0])], ignore_index=True)
        return df_stats
    
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

class Indicators:
    @staticmethod
    def calc_plot_BollingersBands(df, stock_name, close="5. adjusted close", window=20, window_dev=2, plot=True, plot_days_back=100):
        '''
        Calculates and plots Bollinger Bands for a given stock.
        Function Calculates Bollinger Bands for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Bollinger Bands, 
            window_dev - number of standard deviations to calculate Bollinger Bands, 
            plot_days_back - number of days back to plot Bollinger Bands
        OUT: df with Bollinger Bands and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_bb = BollingerBands(close=df[close], 
                                    window=window, 
                                    window_dev=window_dev)
        # Add Bollinger Bands features
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()
        if plot:
            fig, ax = plt.subplots(figsize=(15, 7))

            ax.plot(df[close].iloc[-plot_days_back:], color='#4C72B0', linewidth=2)
            ax.plot(df['bb_bbh'].iloc[-plot_days_back:], color='#C44E52', linewidth=1, linestyle='--')
            ax.plot(df['bb_bbl'].iloc[-plot_days_back:], color='#C44E52', linewidth=1, linestyle='--')
            ax.set_title(f'Bollinger Bands for {stock_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(['Adjusted Close Price', 'Bollinger High Band', 'Bollinger Low Band'])

            plt.show()

        return df[['bb_bbm', 'bb_bbh', 'bb_bbl']]
    
    @staticmethod
    def calc_plot_SMA(df, stock_name, close="5. adjusted close", window=50, plot=True, plot_days_back=100):
        '''
        Calculates and plots Simple Moving Average for a given stock.
        Function Calculates Simple Moving Average for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Simple Moving Average, 
            plot_days_back - number of days back to plot Simple Moving Average
        OUT: df with Simple Moving Average and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_sma = SMAIndicator(close=df[close], window=window)
        df['sma'] = indicator_sma.sma_indicator()
        if plot:
            fig, ax = plt.subplots(figsize=(15, 7))

            ax.plot(df[close].iloc[-plot_days_back:], color='#4C72B0', linewidth=2)
            ax.plot(df['sma'].iloc[-plot_days_back:], color='#C44E52', linewidth=1)
            ax.set_title(f'Simple Moving Average for {stock_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(['Adjusted Close Price', 'SMA'])

            plt.show()
        return df[['sma']]
    
    @staticmethod
    def calc_plot_EMA(df, stock_name, close="5. adjusted close", window=50, plot=True, plot_days_back=100):
        '''
        Calculates and plots Exponential Moving Average for a given stock.
        Function Calculates Exponential Moving Average for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Exponential Moving Average, 
            plot_days_back - number of days back to plot Exponential Moving Average
        OUT: df with Exponential Moving Average and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_ema = EMAIndicator(close=df[close], window=window)
        df['ema'] = indicator_ema.ema_indicator()
        if plot:
            fig, ax = plt.subplots(figsize=(15, 7))

            ax.plot(df[close].iloc[-plot_days_back:], color='#4C72B0', linewidth=2)
            ax.plot(df['ema'].iloc[-plot_days_back:], color='#C44E52', linewidth=1)
            ax.set_title(f'Exponential Moving Average for {stock_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(['Adjusted Close Price', 'EMA'])

            plt.show()
        return df[['ema']]
    
    @staticmethod
    def calc_plot_WMA(df, stock_name,  close="5. adjusted close", window=50, plot=True, plot_days_back=100):
        '''
        Calculates and plots Weighted Moving Average for a given stock.
        Function Calculates Weighted Moving Average for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Weighted Moving Average, 
            plot_days_back - number of days back to plot Weighted Moving Average
        OUT: df with Weighted Moving Average and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_wma = WMAIndicator(close=df[close], window=window)
        df['wma'] = indicator_wma.wma()
        if plot:
            fig = plt.figure(figsize=(15, 7))
            ax = fig.add_subplot(111)

            ax.plot(df[close].iloc[-plot_days_back:], color='#4C72B0', linewidth=2)
            ax.plot(df['wma'].iloc[-plot_days_back:], color='#C44E52', linewidth=1)
            ax.set_title(f'Weighted Moving Average for {stock_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(['Adjusted Close Price', 'WMA'])

            plt.show()

        return df[['wma']]

    @staticmethod
    def calc_plot_AMA(df, stock_name, close='5. adjusted close',  window=50,  pow1=2, pow2=30, plot=True, plot_days_back=100):
        '''
        Calculates and plots Arnaud Legoux Moving Average for a given stock.
        Function Calculates Arnaud Legoux Moving Average for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Arnaud Legoux Moving Average, 
            plot_days_back - number of days back to plot Arnaud Legoux Moving Average
        OUT: df with Arnaud Legoux Moving Average and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_ama = KAMAIndicator(close=df[close], window=window, pow1=pow1, pow2=pow2)
        df['ama'] = indicator_ama.kama()
        if plot: 
            fig = plt.figure(figsize=(15,7))
            ax = fig.add_subplot(111)

            ax.plot(df[[close]].iloc[-plot_days_back:], color='#4C72B0', linewidth=2)
            ax.plot(df[['ama']].iloc[-plot_days_back:], color='#C44E52', linewidth=1)
            
            ax.set_title(f'Kaufman Adaptive Moving Average for {stock_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(['Adjusted Close Price', 'AMA'])
            
            plt.show()
        return df[['ama']]
    
    @staticmethod
    def calc_plot_RSI(df, stock_name, close='5. adjusted close', window=14, plot=True, plot_days_back=100):
        '''
        Calculates and plots Relative Strength Index for a given stock.
        Function Calculates Relative Strength Index for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            window - number of days to calculate Relative Strength Index, 
            plot_days_back - number of days back to plot Relative Strength Index
        OUT: df with Relative Strength Index and plot
        '''
        plt.style.use('Solarize_Light2')
        df = df.sort_index(ascending=True)
        indicator_rsi = RSIIndicator(close=df[close], window=window)
        df['rsi'] = indicator_rsi.rsi()
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))

            # plot the price
            x = df.iloc[-plot_days_back:].index
            y1 = df[close].iloc[-plot_days_back:]
            ax1.fill_between(x, y1, color='#4C72B0', alpha=0.5) 
            # show only price between min and max price
            ax1.set_ylim([y1.min()-plot_days_back/100, y1.max()+plot_days_back/100])
            ax1.set_title(f'Price {stock_name}')
            ax1.set_ylabel('Price')

            # plot the RSI
            y2 = df['rsi'].iloc[-plot_days_back:]
            ax2.plot(x, y2, color='#55A868')
            ax2.set_title(f'Relative Strength Index {stock_name}')
            ax2.set_ylabel('RSI')

            ax2.axhline(70, linestyle='--', color='#E83030', linewidth=1)
            ax2.axhline(30, linestyle='--', color='#E83030', linewidth=1)
            ax2.set_yticks([30, 70])

            # adjust size of subplots
            fig.subplots_adjust(hspace=0.15)
            ax1.set_position([0.1, 0.5, 0.8, 0.5])
            ax2.set_position([0.1, 0.15, 0.8, 0.27])

            # Label the x-axis
            ax2.set_xlabel('Date')

            plt.show()
        return df[['rsi']]
    
    @staticmethod
    def calc_plot_OBV(df, stock_name, close = "5. adjusted close", volume_col='6. volume', plot=True, plot_days_back=100):
        '''
        Calculates and plots On Balance Volume for a given stock.
        Function Calculates On Balance Volume for all days in the dataframe, but we can specify how many days back we want to plot.
        IN: df - dataframe with stock data, 
            plot_days_back - number of days back to plot On Balance Volume
        OUT: df with On Balance Volume and plot
        '''
        df = df.sort_index(ascending=True)
        indicator_obv = OnBalanceVolumeIndicator(close=df[close], volume=df[volume_col])
        df['obv'] = indicator_obv.on_balance_volume()
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))

            # plot the price
            x = df.iloc[-plot_days_back:].index
            y1 = df[close].iloc[-plot_days_back:]
            ax1.fill_between(x, y1, color='#4C72B0', alpha=0.5)

            # show only price between min and max price
            ax1.set_ylim([y1.min()-plot_days_back/100, y1.max()+plot_days_back/100])
            ax1.set_title(f'Price {stock_name}')
            ax1.set_ylabel('Price')

            # plot the volume
            ax2.bar(df.iloc[-plot_days_back:].index, df[volume_col].iloc[-plot_days_back:], width=0.8, color='#55A868')
            ax2.set_title(f'On Balance Volume {stock_name}')
            ax2.set_ylabel('OBV Volume (in millions)')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}M'.format(y * 1e-6))) 

            # adjust size of subplots
            fig.subplots_adjust(hspace=0.15)
            ax1.set_position([0.1, 0.5, 0.8, 0.5])
            ax2.set_position([0.1, 0.15, 0.8, 0.27])

            # Label the x-axis
            ax2.set_xlabel('Date')
            plt.show()
            
        return df[['obv']]

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
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f'{model_name}-{title}-{now}.png'
        fig_dir = os.path.join(PROJECT_DIR, r'figures')
        fig_path = os.path.join(fig_dir, file_name)
        return fig_path, file_name
    
    @staticmethod
    def plot_pred_vs_actual(y_true, y_pred, model_name, title='Prediction vs Actual', show=True):
        ''' Plots and saves plot for actual vs predicted values
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
        
