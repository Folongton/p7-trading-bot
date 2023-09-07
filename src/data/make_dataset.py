# -*- coding: utf-8 -*-
import os
import logging

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), verbose=True) # Example:  AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY")

from src.common.globals import G
from src.data.get_data import CSVsLoader

DATA_DIR_RAW = G.daily_full_dir
DATA_DIR_PROCESSED = os.path.join(G.get_project_root(), r'data\03_processed\daily_full')

def main( input_ticker, input_filepath=DATA_DIR_RAW, output_filepath=DATA_DIR_PROCESSED):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = G.setup_logging(logger_name=__name__,
                             console_level=logging.INFO, 
                             log_file=os.path.basename(__file__), 
                             log_file_level=logging.INFO)
    logger.info('making final data set from raw data')
    # ---------------------------DATA WORK---------------------------------------
    # Load data
    df = CSVsLoader(input_ticker)
    # Clean data
    df = df.prep_AV_data()
    # Save data
    df.save_data(output_filepath)
    #------------------------------------------------------------------
    logger.info(f'done final data set from raw data. Saved to: {output_filepath} as {input_ticker}-daily-full.csv')


if __name__ == '__main__':
    main(input_ticker='MSFT')
