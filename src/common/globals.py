from src.common.logs import setup_logging
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import os

PROJECT_PATH = Path(__file__).resolve().parents[2]

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

@dataclass
class G:
    ''' Global variables and methods for project
    '''
    name: str 
    growth_tickers = ['AMZN','BABA','GOOGL','MSFT','TSLA']
    value_tikers = ['ABBV','F','INTC','JNJ','VZ']
    all_nasdaq_tickers = pd.read_csv(os.path.join(str(PROJECT_PATH), r'data/00_raw/NASDAQ All Tickers.csv'))['ticker'].tolist()
    raw_daily_full_dir = os.path.join(str(PROJECT_PATH), r'data\00_raw\daily_full')

    @staticmethod
    def get_project_root(as_path=False):
        """Returns project root folder."""
        if as_path:
            return Path(__file__).resolve().parents[2]
        else:
            return str(Path(__file__).resolve().parents[2])