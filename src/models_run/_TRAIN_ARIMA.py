from src.common.globals import G, split_train_valid_test, calc_errors, save_errors_to_table, get_naive_forecast
from src.data.get_data import CSVsLoader
from src.common.plots import Visualize as V
from src.features.build_features import FeatureEngineering as FE


import os
import joblib
from src.common.logs import setup_logging
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from statsmodels.tsa.arima.model import ARIMA


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
        'train_size': 0.85,
        'valid_size': 0.15,
        'test_size': 0.05, # 300 days 
    }, 
    'model': {
        'name': 'ARIMA', 
        'window': 20,
    },
}

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

def plot_split(train_data, test_data, show=True):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(train_data, color='green')
    plt.plot(test_data, color='blue')

    train_data = mpatches.Patch(color='green', label='Train Data')
    test_data = mpatches.Patch(color='blue', label='Test Data')
    plt.legend(handles=[train_data, test_data], loc='best')  

    fig_path, file_name = V.construct_fig_path(model_name=config['model']['name'], title='Split Data')
    plt.savefig(fig_path)
    logger.info(f'Plot "{file_name}" saved to "{fig_path}"')

    if show == True:
        plt.show()

def plot_ARIMA_forecast( test_data, fc_series, lower_series, upper_series, train_data=None, prediction_days=None, show=True):
    # Plot last PREDICTION_DAYS*10 days of training set and forecast results
    plt.figure(figsize=(15,5), dpi=100)
    if train_data is not None:
        plt.plot(train_data.iloc[-prediction_days*10:], label='training data')
    plt.plot(test_data, color = 'green', label='Actual Stock Price')
    plt.plot(fc_series, color = 'red',label='Predicted Stock Price')

    # fill between lower and upper confidence band 
    plt.fill_between(lower_series.index, lower_series, upper_series, color='r', alpha=.20, label='Confidence Interval', interpolate=True)
    plt.title(f"{config['AV']['ticker']} Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f"{config['AV']['ticker']} Stock Price")
    plt.legend(loc='upper left', fontsize=8)

    fig_path, file_name = V.construct_fig_path(model_name=config['model']['name'], title='ARIMA Forecast')
    plt.savefig(fig_path)
    logger.info(f'Plot "{file_name}" saved to "{fig_path}"')

    if show == True:
        plt.show()  


def main():
    # Load data
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)
    # Log Scale data for ARIMA better performance
    df = FE.log_scale(df, column=config['AV']['key_adjusted_close'])
    # Plot
    V.plot_1_by_2(df, model_name=config['model']['name'],
                  col_before=config['AV']['key_adjusted_close'], 
                  col_after=config['AV']['key_adjusted_close'] + ' - log',
                  title2='Log Scale',
                  show=False)

    # Split data ( using only 450 last points as our ACF plot shows the impact of the first 450 points only)
    prediction_days = int(df.shape[0]*config['data']['test_size'])
    train_data = df.iloc[-1450-prediction_days:-prediction_days]['Adj Close - log']
    test_data = df.iloc[-prediction_days:]['Adj Close - log']

    plot_split(train_data, test_data, show=False)
    
    # Build Model
    model = ARIMA(train_data, order=(0,1,0), trend='t')  # (p,d,q) - p: lag order, d: degree of differencing, q: order of moving average model - chosen in Analysis file "..\notebooks\exploratory\ARIMA_analysis.ipynb"
    fittedARIMA = model.fit()  
    logger.info(fittedARIMA.summary())

    # Forecast
    fc = fittedARIMA.forecast(prediction_days)

    # Results
    fc_results = fittedARIMA.get_forecast(prediction_days)
    fc_params_summary = fc_results.summary_frame()

    # Plot
    # Make as pandas series for plotting
    fc.index = test_data.index
    fc_series = fc
    lower_series = pd.Series(fc_params_summary['mean_ci_lower'].values, index=test_data.index)
    upper_series = pd.Series(fc_params_summary['mean_ci_upper'].values, index=test_data.index)

    plot_ARIMA_forecast(test_data, fc_series, lower_series, upper_series, train_data, prediction_days, show=False)

    # transforming to original scale
    fc_series_exp = np.exp(fc_series)
    lower_series_exp = np.exp(lower_series)
    upper_series_exp = np.exp(upper_series)
    test_data_exp = np.exp(test_data)

    # Plot Zoomed
    plot_ARIMA_forecast(test_data_exp, fc_series_exp, lower_series_exp, upper_series_exp, train_data=None, prediction_days=None, show=False)

    # Evaluate model
    y_pred = fc_series_exp
    y_true = test_data_exp

    naive_forecast = get_naive_forecast(df).iloc[-len(y_true):]
    rmse, mae, mape, mase = calc_errors(y_true, y_pred, naive_forecast)
    save_errors_to_table(config['model']['name'], {'rmse': rmse,  'mae': mae, 'mape': mape, 'mase': mase})         


if __name__ == '__main__':
    main()