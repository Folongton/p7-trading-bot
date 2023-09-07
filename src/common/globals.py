import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import os

import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('Solarize_Light2')

PROJECT_PATH = Path(__file__).resolve().parents[2]

@dataclass
class G:
    ''' Global variables and methods for project
    '''
    name: str 
    growth_tickers = ['AMZN','BABA','GOOGL','MSFT','TSLA']
    value_tikers = ['ABBV','F','INTC','JNJ','VZ']
    all_nasdaq_tickers = pd.read_csv(os.path.join(str(PROJECT_PATH), r'data\00_raw\NASDAQ All Tickers.csv'))['ticker'].tolist()
    daily_full_dir = os.path.join(str(PROJECT_PATH), r'data\00_raw\daily_full')

    @staticmethod
    def get_project_root(as_path=False):
        """Returns project root folder."""
        if as_path:
            return Path(__file__).resolve().parents[2]
        else:
            return str(Path(__file__).resolve().parents[2])

    @staticmethod
    def setup_logging(console_level=logging.INFO, log_file='DEFAULT', log_file_level=logging.INFO, logger_name=__name__):
        ''' Setup Logging in log file and console
            Creates a logger object with appropriate formatting and handlers.
        INPUT: console_level: logging level for console, 
                    log_file: name of log file, 
            log_file_level: logging level for log file
        OUTPUT: logger object
        '''
        log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_fmt)
        console_handler.setLevel(console_level)

        file_handler = logging.FileHandler(filename=f'./logs/{log_file}.log')
        file_handler.setFormatter(log_fmt)
        file_handler.setLevel(log_file_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger 
    