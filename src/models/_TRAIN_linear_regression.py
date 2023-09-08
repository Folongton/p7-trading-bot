# Sync it with 'models\models_table.csv' file.
# 'models\models_table.csv' file will contain columns : model, accuracy, precision ...(metrics), parameter1, parameter2, parameter3 ...(parameters), date, comments. 
# This format will give me ability to see everrything in one place.
# -------------------------------------------------------------------------------------------------------------
import pandas as pd
from src.common.globals import G
from src.data.get_data import CSVsLoader
import os

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
        'window_size': 20,
        'train_split_size': 0.80,
        'val_split_size': 0.20,
    }, 
    'model_params': {
        'input_size': '', 
    },
}

def main():
    df = CSVsLoader(ticker=config['AV']['ticker'], directory=DATA_DIR_PROCESSED)



if __name__ == '__main__':
    main()