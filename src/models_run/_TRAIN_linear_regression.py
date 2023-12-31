from src.common.globals import G
from src.models_service.errors import ErrorsCalculation as ErrorCalc
from src.models_service.models_service import SklearnModelService as SkModelSrvc
from src.data.get_data import CSVsLoader
from src.common.plots import Visualize as V
from src.features.build_features import FeatureEngineering as FE


import os
import joblib
from src.common.logs import setup_logging
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

PROJECT_PATH = G.get_project_root()
DATA_DIR_PROCESSED = G.processed_daily_full_dir

config = {
    'AV': {
        'key': '',
        'ticker': 'TEST',
        'outputsize': 'full',
        'key_adjusted_close': 'Adj Close',
        'key_volume': 'Volume',
    },
    'data': {
        'train_size': 0.80,
        'valid_size': 0.15,
        'test_size': 0.05,
    }, 
    'model': {
        'name': 'LenearRegression', 
        'window': 20,
    },
}

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

def main():
    # Load data
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)
    # Create features
    df = FE.create_lags(df, column=config['AV']['key_adjusted_close'], lag=config['model']['window'])
    # Split data
    X = df.drop(columns=[config['AV']['key_adjusted_close'], 
                         config['AV']['key_volume']])
    y = df['Adj Close'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=7, shuffle=False)

    # Train model
    model = LinearRegression().fit(X_train, y_train)
    model_name = config['model']['name']
    r2 = round(model.score(X_test, y_test), 3)
    print(f'R^2: {r2}')

    # Evaluate model
    y_pred = model.predict(X_test)
    y_true = y_test

    naive_forecast = ErrorCalc.get_naive_forecast(df).iloc[-len(y_true):]
    rmse, mae, mape, mase = ErrorCalc.calc_errors(y_true, y_pred, naive_forecast)
    ErrorCalc.save_errors_to_table(model_name, {'rmse': rmse,  'mae': mae, 'mape': mape, 'mase': mase,'r2': r2})

    V.plot_pred_vs_actual(y_true, y_pred, os.path.basename(__file__), show=True)

    # save model
    SkModelSrvc.save_model(model, model_name, logger)

if __name__ == '__main__':
    main()