import os
from src.common.globals import G, split_train_valid_test, calc_errors, save_errors_to_table
from src.data.get_data import CSVsLoader
from src.common.plots import Visualize as V


PROJECT_PATH = G.get_project_root()
DATA_DIR_PROCESSED = os.path.join(PROJECT_PATH, r'data\03_processed\daily_full')

config = {
    'AV': {
        'key': '',
        'ticker': 'MSFT',
    },
    'data': {
        'train_size': 0.80,
        'valid_size': 0.15,
        'test_size': 0.05,
    }
}


def main():
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)
    df['Close - 1'] = df['Adj Close'].shift(1)
    df_train, df_valid, df_test = split_train_valid_test(df, train_size=config['data']['train_size'],
                                                            valid_size=config['data']['valid_size'],
                                                            test_size=config['data']['test_size'])
    y_true = df['Adj Close'].iloc[-df_test.shape[0]:]
    y_pred = df['Close - 1'].iloc[-df_test.shape[0]:]
    naive_forecast = y_pred # naive forecast is the previous day's close price

    rmse, mae, mape, mase = calc_errors(y_true, y_pred, naive_forecast)
    save_errors_to_table('Base', {'rmse': rmse,  'mae': mae, 'mape': mape, 'mase': mase})

    V.plot_pred_vs_actual(y_true, y_pred, os.path.basename(__file__), show=False)

    
if __name__ == '__main__':
    main()