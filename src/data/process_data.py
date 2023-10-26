import logging

from src.common.logs import setup_logging
from src.common.globals import G
from src.data.get_data import CSVsLoader

logger = setup_logging(logger_name=__name__,
                                console_level=logging.INFO,
                                log_file_level=logging.INFO)

DATA_DIR_RAW = G.raw_daily_full_dir
DATA_DIR_PROCESSED = G.processed_daily_full_dir
TICKERS = ['VGT']

def main( input_tickers, input_filepath=DATA_DIR_RAW, output_filepath=DATA_DIR_PROCESSED):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    for ticker in input_tickers:
        logger.info('making final data set from raw data')
        # ---------------------------DATA WORK---------------------------------------
        # Load data
        df = CSVsLoader(ticker, logger=logger)
        # Clean data
        df = df.prep_AV_data()
        # Save data
        df.save_data(output_filepath)
        #------------------------------------------------------------------
        logger.info(f'done final data set from raw data. Saved to: {output_filepath} as {ticker}-daily-full.csv')


if __name__ == '__main__':
    main(input_tickers=TICKERS)
