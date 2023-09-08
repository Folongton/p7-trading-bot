from src.common.globals import G, split_train_valid_test, calc_errors, save_errors_to_table, get_naive_forecast
from src.data.get_data import CSVsLoader
from src.common.analysis_and_plots import Visualize as V


import os
from src.common.logs import setup_logging
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

PROJECT_PATH = G.get_project_root()
DATA_DIR_PROCESSED = os.path.join(PROJECT_PATH, r'data\03_processed\daily_full')

config = {
    'AV': {
        'key': '',
        'ticker': 'MSFT',
        'outputsize': 'full',
        'key_adjusted_close': '5. adjusted close',
        'key_volume': '6. volume',
    },
    'data': {
        'train_size': 0.85,
        'valid_size': 0.145,
        'test_size': 0.005,
    }, 
    'model_params': {
        'input_size': '', 
    },
}

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

def main():
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)

    df['Close - 1'] = df['Adj Close'].shift(1)
    df['Volume - 1'] = df['Volume'].shift(1)
    df = df.dropna(how='any')

    X = df[['Close - 1', 'Volume - 1']]
    y = df['Adj Close'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=7, shuffle=False)

    model = LinearRegression().fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    print(f'R^2: {r2}')

    y_pred = model.predict(X_test)
    y_true = y_test

    naive_forecast = get_naive_forecast(df).iloc[-len(y_true):]
    rmse, mae, mape, mase = calc_errors(y_true, y_pred, naive_forecast)
    save_errors_to_table('LinearRegression', {'rmse': rmse,  'mae': mae, 'mape': mape, 'mase': mase,'r2': r2})

    V.plot_pred_vs_actual(y_true, y_pred, os.path.basename(__file__), show=False)

if __name__ == '__main__':
    main()